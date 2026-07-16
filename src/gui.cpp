// The embedded web GUI (main menu: neuron --gui)
//
// One httplib server on 127.0.0.1 with an OS-assigned free port (no
// collisions, loopback only). The page (src/gui_page.html, embedded at
// build time) drives the same engine objects the menu driver uses; every
// request is serialized by one mutex because the engine is single-threaded.
// Engine output is captured per-request through util::set_screen() and
// returned to the page as text.

// httplib must be included BEFORE the engine headers: on Windows it pulls
//    in the SDK headers, whose global 'byte' typedef becomes ambiguous with
//    C++17 std::byte once the engine headers' using-directives are in scope
#include "httplib.h"

#include "stdafx.h" // For MSVC, must be first among the engine headers!

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include "gui.h"
#include "dataset.h"
#include "logistic.h"
#include "simpleprop.h"
#include "network.h"
#include "iterative.h"
#include "regressnet.h"
#include "utility.h"
#include "version.h"

#include "gui_page.h" // generated from src/gui_page.html by CMake

using namespace std;

namespace {

// The engine state driven by the page — mirrors the driver's owners
mutex engineMutex;
unique_ptr< DataSet > dataPtr;
unique_ptr< Model > modelPtr;
// The last training run's captured report, downloadable as report.txt
string lastReport;
// The last training run's final error, the baseline for stepwise regression
//    (RegressNet's Wilks GLRT); -1 until a model has been trained
double lastTrainError = -1;

// --- The async training job (ROADMAP 2 Phase 1b) --------------------------
//
// One job at a time. While it runs, job.running gates every handler that
// would touch the engine (HTTP 409, no queueing); only /api/train/status and
// /api/train/stop go through. The worker thread owns the engine for the
// duration and does NOT hold engineMutex while training -- the gate is what
// keeps everyone else out. The worker captures engine output through its own
// thread_local screen (see utility.cpp).
struct TrainJob
{
	thread worker;
	atomic< bool > running{ false }; // toggled under engineMutex
	atomic< bool > cancel{ false };  // the Stop button; observer returns false

	// Everything below is guarded by progressMutex. The error-vs-iteration
	//    series is decimated: capped at MAX_POINTS, halving (and doubling the
	//    keep-every-nth stride) when full, so a million-iteration run still
	//    ships a bounded, evenly thinned curve.
	mutex progressMutex;
	vector< unsigned > iters;
	vector< double > trainErr, testErr; // testErr < 0 = not sampled
	unsigned keepEvery = 1, sampleCounter = 0;
	string result; // the finished run's response JSON, "" while running
};
TrainJob job;

const unsigned JOB_MAX_POINTS = 2000;

// The worker-side observer: samples the error curve on a wall-clock schedule
//    (>= 250 ms apart, so observing costs nothing measurable) and carries the
//    Stop request into the training loop. Test error is sampled over at most
//    ~1000 exemplars per look (sampleTestError's stride).
struct GuiObserver : Iterative::Observer
{
	Network* net;
	unsigned testStride;
	chrono::steady_clock::time_point lastSample;
	bool sampledYet = false;

	GuiObserver( Network* n, unsigned nTest )
		: net( n ),
		testStride( nTest > 1000 ? nTest / 1000 : 1 ),
		lastSample( chrono::steady_clock::now() ) {}

	bool onIteration( unsigned iteration, double setError ) override
	{
		auto now = chrono::steady_clock::now();
		if ( !sampledYet || now - lastSample >= chrono::milliseconds( 250 ) )
		{
			sampledYet = true;
			lastSample = now;

			// Sampled OUTSIDE progressMutex: only the worker touches the engine
			double testError = net ? net->sampleTestError( testStride ) : -1;

			lock_guard< mutex > lock( job.progressMutex );
			if ( ++job.sampleCounter % job.keepEvery == 0 )
			{
				job.iters.push_back( iteration );
				job.trainErr.push_back( setError );
				job.testErr.push_back( testError );
				if ( job.iters.size() >= JOB_MAX_POINTS )
				{
					// Halve the series, double the stride for future samples
					for ( unsigned i = 0, j = 0; j < job.iters.size(); i++, j += 2 )
					{
						job.iters[ i ] = job.iters[ j ];
						job.trainErr[ i ] = job.trainErr[ j ];
						job.testErr[ i ] = job.testErr[ j ];
					}
					job.iters.resize( job.iters.size() / 2 );
					job.trainErr.resize( job.trainErr.size() / 2 );
					job.testErr.resize( job.testErr.size() / 2 );
					job.keepEvery *= 2;
				}
			}
		}
		return !job.cancel.load();
	}
};

// The engine's stop reason as a JSON-friendly name. STOP_OBSERVER reads as
//    "cancelled" here because the GUI's observer only ever stops on the Stop
//    button; a future auto-stop gets its own reason (STOP_PLATEAU, ROADMAP 2).
const char* stopReasonName( Iterative::StopReason r )
{
	switch ( r )
	{
	case Iterative::STOP_MAX_ITERATIONS: return "max_iterations";
	case Iterative::STOP_MIN_ERROR: return "min_error";
	case Iterative::STOP_CHANGE: return "min_change";
	case Iterative::STOP_WINDOW: return "error_window";
	case Iterative::STOP_GRADMAX: return "grad_max";
	case Iterative::STOP_OBSERVER: return "cancelled";
	default: return "none";
	}
}

// --- JSON helpers (responses only; requests arrive form-encoded) ---------

string jsonEscape( const string& s )
{
	ostringstream out;
	for ( unsigned char c : s )
	{
		if ( c == '"' ) out << "\\\"";
		else if ( c == '\\' ) out << "\\\\";
		else if ( c == '\n' ) out << "\\n";
		else if ( c == '\r' ) out << "\\r";
		else if ( c == '\t' ) out << "\\t";
		else if ( c < 0x20 ) { char buf[ 8 ]; snprintf( buf, sizeof buf, "\\u%04x", c ); out << buf; }
		else out << c;
	}
	return out.str();
}

string jsonMsg( bool ok, const string& message )
{
	return string( "{\"ok\":" ) + ( ok ? "true" : "false" )
		+ ",\"message\":\"" + jsonEscape( message ) + "\"}";
}

// A double as a JSON number, or null when not finite -- NaN and infinity are
//    not valid JSON, and degenerate sets (perfectly separated XOR, a singular
//    covariance) do produce them (e.g. a binormal Az of NaN)
string jnum( double v )
{
	if ( !std::isfinite( v ) )
		return "null";
	ostringstream o;
	o.precision( 6 );
	o << v;
	return o.str();
}

string jsonNumbers( const vector< double >& v )
{
	ostringstream out;
	out.precision( 6 );
	out << "[";
	for ( unsigned i = 0; i < v.size(); i++ )
		out << ( i ? "," : "" ) << v[ i ];
	out << "]";
	return out.str();
}

// One ROC series from a TwoSet: recompute the trapezoid area, which also
//    fills the ROCx/ROCy plot vectors (Phase 3.3)
string jsonROCSeries( TwoSet& t )
{
	double area = t.getTrapROCarea(); // also fills the curve points and SE
	double se = t.getTrapSE();        // Hanley-McNeil SE (matches the plotted area)
	double acc = t.getClassAcc();
	// sensitivity/specificity throw DivisionByZero if a class is empty
	double sens = -1, spec = -1;
	try { sens = t.getSens(); } catch ( ... ) { sens = -1; }
	try { spec = t.getSpec(); } catch ( ... ) { spec = -1; }
	ostringstream out;
	out.precision( 6 );
	out << "{\"x\":" << jsonNumbers( t.getROCx() )
		<< ",\"y\":" << jsonNumbers( t.getROCy() )
		<< ",\"area\":" << jnum( area )
		<< ",\"se\":" << jnum( se )
		<< ",\"acc\":" << jnum( acc )
		<< ",\"sens\":" << jnum( sens )
		<< ",\"spec\":" << jnum( spec ) << "}";
	return out.str();
}

// One full statistics block from a TwoSet. Every metric is individually
//    guarded: a value that can't be computed on this set (empty class, a
//    failed binormal fit, degenerate K-S/H-L) serializes as null rather than
//    aborting the rest of the object.
string jsonStatsSet( TwoSet& t )
{
	ostringstream out;
	out.precision( 6 );

	// "name":value, or "name":null if the getter throws or is not finite
	auto emit = [ &out ]( const char* name, function< double() > fn )
	{
		out << "\"" << name << "\":";
		try { out << jnum( fn() ); }
		catch ( ... ) { out << "null"; }
		out << ",";
	};

	out << "{\"n\":" << t.getNumElements() << ",";

	// Confusion counts share the threshold requirement, so guard them together
	out << "\"confusion\":";
	try
	{
		out << "{\"tp\":" << t.getTP() << ",\"tn\":" << t.getTN()
			<< ",\"fp\":" << t.getFP() << ",\"fn\":" << t.getFN() << "}";
	}
	catch ( ... ) { out << "null"; }
	out << ",";

	emit( "acc",  [ &t ] { return t.getClassAcc(); } );
	emit( "sens", [ &t ] { return t.getSens(); } );
	emit( "spec", [ &t ] { return t.getSpec(); } );
	emit( "pvp",  [ &t ] { return t.getPVP(); } );
	emit( "pvn",  [ &t ] { return t.getPVN(); } );

	// Trapezoidal area with its Hanley-McNeil SE (getTrapROCarea also fills
	//    the curve points and the SE input)
	out << "\"trap\":";
	try
	{
		double a = t.getTrapROCarea(), se = t.getTrapSE();
		out << "{\"area\":" << jnum( a ) << ",\"se\":" << jnum( se ) << "}";
	}
	catch ( ... ) { out << "null"; }
	out << ",";

	// The binormal Az the report quotes, with its bootstrap interval. There is
	//    deliberately no SE from the fit itself: the delta method that used to
	//    supply one assumed independent operating points and ran ~5x narrow.
	//    Quoting the engine's fit rather than one of our own is what keeps the
	//    panel and the report telling the same story.
	out << "\"binormal\":";
	try
	{
		TwoSet::ROCfit f = t.getROCfit();
		if ( !f.valid )
			out << "null"; // no fit was possible: say so, do not invent one
		else
		{
			out << "{\"az\":" << jnum( f.az )
				<< ",\"p\":" << jnum( f.p ) << ",\"chi2\":" << jnum( f.chi2 )
				<< ",\"points\":" << f.points << ",\"ci\":";
			TwoSet::CI c = t.getStatCi();
			if ( c.valid )
				out << "{\"lo\":" << jnum( c.lo ) << ",\"hi\":" << jnum( c.hi )
					<< ",\"se\":" << jnum( c.se )
					<< ",\"resamples\":" << c.resamples
					<< ",\"failures\":" << c.failures << "}";
			else
				out << "null";
			out << "}";
		}
	}
	catch ( ... ) { out << "null"; }
	out << ",";

	out << "\"ks\":";
	try
	{
		double d = t.getKSD(), p = t.getKSP();
		out << "{\"d\":" << jnum( d ) << ",\"p\":" << jnum( p ) << "}";
	}
	catch ( ... ) { out << "null"; }
	out << ",";

	emit( "pearsonP", [ &t ] { return t.getPearsonX2(); } );
	// last field: no trailing comma
	out << "\"hlP\":";
	try { out << jnum( t.getHLX2() ); } catch ( ... ) { out << "null"; }
	out << "}";

	return out.str();
}

// The full statistics object for the current trained model: per-set blocks
//    plus, for logistic regression, the Wald table and condition number.
//    Returns "" when there is nothing to report (no model, or a non-discrete
//    / multi-output dataset the classification statistics don't apply to).
string jsonStats()
{
	if ( !modelPtr )
		return "";
	DataSet& md = modelPtr->getDataSet();
	if ( !( md.getDiscrete() && md.getOutput() == 1 && md.trainLoaded() ) )
		return "";

	ostringstream out;
	out.precision( 6 );
	out << "{\"train\":" << jsonStatsSet( md.getTrainTwoSet() );
	if ( md.testLoaded() )
		out << ",\"test\":" << jsonStatsSet( md.getTestTwoSet() );

	Logistic* lg = dynamic_cast< Logistic* >( modelPtr.get() );
	if ( lg )
	{
		out << ",\"logistic\":{\"condNumber\":";
		double cn = lg->getCondNum();
		if ( cn < 0 ) out << "null"; else out << jnum( cn );

		out << ",\"coefficients\":[";
		const vector< double >& b = lg->getBetas();
		const vector< double >& se = lg->getBetaSE();
		const vector< double >& wp = lg->getWaldP();
		unsigned nI = md.getInput();
		for ( unsigned i = 0; i <= nI && i < b.size(); i++ )
		{
			if ( i ) out << ",";
			out << "{\"input\":";
			if ( i < nI ) out << i; else out << "\"intercept\"";
			out << ",\"beta\":" << jnum( b[ i ] ) << ",\"se\":";
			bool haveSE = ( i < se.size() && se[ i ] >= 0 );
			if ( haveSE ) out << jnum( se[ i ] ); else out << "null";
			out << ",\"waldP\":";
			if ( haveSE && i < wp.size() ) out << jnum( wp[ i ] ); else out << "null";
			out << "}";
		}
		out << "]}";
	}

	out << "}";
	return out.str();
}

// --- Small utilities ------------------------------------------------------

bool fileExists( const string& path )
{
	ifstream f( path.c_str() );
	return ( bool ) f;
}

// A request field, whether the POST was form-encoded or multipart
string param( const httplib::Request& req, const string& name )
{
	if ( req.has_param( name ) )
		return req.get_param_value( name );
	if ( req.has_file( name ) )
		return req.get_file_value( name ).content;
	return "";
}

// Just the final component of a picked file's name — no directory parts
string safeBasename( const string& name )
{
	string::size_type slash = name.find_last_of( "/\\" );
	string base = ( slash == string::npos ? name : name.substr( slash + 1 ) );
	return base.empty() ? "upload.txt" : base;
}

// Match the driver: an appropriate number of trapezoid thresholds per set
void tuneThresholds( DataSet& d )
{
	if ( !d.getDiscrete() || d.getOutput() != 1 )
		return;
	if ( d.trainLoaded() && d.getNumTrain() > 1 )
		d.getTrainTwoSet().setTrapThresholds(
			d.getNumTrain() > 100 ? 100 : d.getNumTrain() );
	if ( d.testLoaded() && d.getNumTest() > 1 )
		d.getTestTwoSet().setTrapThresholds(
			d.getNumTest() > 100 ? 100 : d.getNumTest() );
}

// RAII engine-output capture: restores cout even if the engine throws
struct Capture
{
	ostringstream text;
	Capture() { util::set_screen( text ); }
	~Capture() { util::set_screen( cout ); }
};

// Resolve a file argument: prefer an uploaded multipart part (browsers hide
//    paths, so the page sends the file's CONTENT), else a "path"-style field
//    (curl, scripts). An upload is written into the server's directory and its
//    on-disk name recorded in 'saved'. Returns "" (with 'err' set) on failure,
//    or "" (err empty) when the field is simply absent.
static string resolveFile( const httplib::Request& req, const string& fileField,
	const string& pathField, string& saved, string& err )
{
	if ( req.has_file( fileField ) )
	{
		const auto& upload = req.get_file_value( fileField );
		if ( upload.content.empty() ) { err = "the uploaded file is empty"; return ""; }
		string p = safeBasename( upload.filename );
		ofstream out( p.c_str(), ios::out | ios::trunc | ios::binary );
		if ( !out.is_open() ) { err = "can't save the upload as " + p; return ""; }
		out << upload.content;
		out.close();
		saved = p;
		return p;
	}
	return param( req, pathField );
}

// --- Handlers --------------------------------------------------------------

string handleLoad( const httplib::Request& req )
{
	string mode = param( req, "mode" ), savedTrain, savedTest, err;
	unsigned nI = ( unsigned ) atol( param( req, "inputs" ).c_str() ),
		nO = ( unsigned ) atol( param( req, "outputs" ).c_str() );
	double fraction = atof( param( req, "fraction" ).c_str() );

	// neuron is a 1-output system: the raw->train split and every ROC /
	//    classification statistic refuse anything but a single output. So the
	//    output count is fixed at 1 (a caller may still pass it explicitly).
	if ( nO < 1 )
		nO = 1;

	// discrete=0 declares a continuous outcome (regression): the engine then
	//    trains with LMS error and skips the classification statistics. The
	//    default matches the engine's own (discrete), which the page assumes.
	bool discrete = ( param( req, "discrete" ) != "0" );

	string path = resolveFile( req, "file", "path", savedTrain, err );
	if ( !err.empty() )
		return jsonMsg( false, err );

	if ( path.empty() || !fileExists( path ) )
		// Checked here because the engine's missing-file recovery prompts
		//    on stdin, which the GUI must never touch
		return jsonMsg( false, "can't open file: " + path );

	// The data file already carries its own shape: it has exactly
	//    nInput + nOutput columns. With the output count fixed, the input
	//    count is just (columns - nOutput) — derive it rather than make the
	//    user count columns. (A caller may still pass "inputs" to override.)
	if ( nI < 1 )
	{
		Capture probeCap; // keep the probe load's report off the screen
		Matrix< double > probe;
		probe.loadfile( false, path );
		unsigned cols = probe.cols();
		if ( cols <= nO )
			return jsonMsg( false, "the file has too few columns to hold "
				"inputs and an output" );
		nI = cols - nO;
	}

	auto ds = make_unique< DataSet >();
	ds->setInput( nI );
	ds->setOutput( nO );
	if ( !discrete )
		ds->setDiscrete( false );

	Capture cap;
	bool ok = false;

	if ( mode == "raw" )
	{
		ds->loadRaw( path );
		if ( ds->rawLoaded() )
		{
			// The stratified split needs a discrete outcome to stratify on; a
			//    continuous outcome converts whole (the engine's raw2train) --
			//    pre-split regression data loads through mode=train + testfile
			if ( !discrete )
			{
				if ( fraction > 0 )
					return jsonMsg( false, "a continuous outcome cannot be "
						"stratified into a train/test split; load with "
						"fraction=0, or pre-split the data and use mode=train "
						"with a testfile" );
				ok = ds->raw2train();
			}
			else
				ok = ds->randomizeD( fraction );
		}
	}
	else // the file already is a training set
	{
		ds->loadTrain( path );
		ok = ds->trainLoaded();

		// Optional matched, already-split test set (a pre-split pair like
		//    BP40train.txt + BP40test.txt). Only meaningful in "train" mode;
		//    "raw" makes its own split.
		if ( ok )
		{
			string testPath = resolveFile( req, "testfile", "testpath", savedTest, err );
			if ( !err.empty() )
				return jsonMsg( false, err );
			if ( !testPath.empty() )
			{
				if ( !fileExists( testPath ) )
					return jsonMsg( false, "can't open test file: " + testPath );
				ds->loadTest( testPath );
				if ( !ds->testLoaded() )
				{
					string why = cap.text.str();
					return jsonMsg( false, why.empty() ? "test set load failed" : why );
				}
			}
		}
	}

	if ( !ok )
	{
		// The engine's own explanation is the best error message
		string why = cap.text.str();
		return jsonMsg( false, why.empty() ? "load failed" : why );
	}

	tuneThresholds( *ds );

	ostringstream msg;
	msg << ds->getInput() << " inputs, 1 output; "
		<< ds->getNumTrain() << " training exemplars";
	if ( ds->testLoaded() )
		msg << ", " << ds->getNumTest() << " test exemplars";
	msg << " loaded";
	if ( !savedTrain.empty() || !savedTest.empty() )
	{
		msg << " (saved beside the server:";
		if ( !savedTrain.empty() ) msg << " " << savedTrain;
		if ( !savedTest.empty() ) msg << " " << savedTest;
		msg << ")";
	}

	dataPtr = std::move( ds );
	modelPtr.reset(); // a new dataset invalidates any existing model
	lastTrainError = -1; // and any prior training

	return jsonMsg( true, msg.str() );
}

string handleModel( const httplib::Request& req )
{
	string type = param( req, "type" );
	unsigned hidden = ( unsigned ) atol( param( req, "hidden" ).c_str() );

	if ( !dataPtr )
		return jsonMsg( false, "load a dataset first" );
	if ( dataPtr->getOutput() != 1 )
		return jsonMsg( false, "the GUI supports 1-output models" );

	Capture cap;
	lastTrainError = -1; // a fresh model has not been trained yet

	if ( type == "logistic" )
	{
		if ( !dataPtr->getDiscrete() )
			return jsonMsg( false, "logistic regression needs a discrete output" );
		modelPtr = make_unique< Logistic >();
		modelPtr->setDataSet( *dataPtr );
		return jsonMsg( true, "binary logistic regression ready" );
	}

	if ( type == "simpleprop" )
	{
		if ( hidden < 1 )
			return jsonMsg( false, "at least 1 hidden node required" );
		modelPtr = make_unique< SimpleProp >();
		// The dataset must be loaded BEFORE the architecture is specified
		modelPtr->setDataSet( *dataPtr );
		if ( dataPtr->getDiscrete() )
			modelPtr->setXEerror();
		else
			modelPtr->setLMSerror();
		dynamic_cast< SimpleProp* >( modelPtr.get() )->setHidden( hidden );
		ostringstream msg;
		msg << "SimpleProp " << dataPtr->getInput() << "-" << hidden
			<< "-1 network ready";
		return jsonMsg( true, msg.str() );
	}

	return jsonMsg( false, "unknown model type: " + type );
}

string handleRandomize( const httplib::Request& req )
{
	string seed = param( req, "seed" );

	if ( !modelPtr )
		return jsonMsg( false, "create a model first" );
	Network* net = dynamic_cast< Network* >( modelPtr.get() );
	if ( !net )
		return jsonMsg( false, "this model type has no trainable weights" );

	if ( !seed.empty() )
		util::set_seed( ( unsigned ) atol( seed.c_str() ) );
	net->randomize();
	lastTrainError = -1; // fresh weights invalidate any prior training

	return jsonMsg( true, seed.empty()
		? "weights randomized \xe2\x80\x94 the next Train starts fresh"
		: "weights randomized from the seed \xe2\x80\x94 the next Train starts fresh" );
}

// Run modelPtr->train() on the CALLING thread and build the full response
//    JSON (captured report, ROC curves, stats, stop reason). The caller must
//    own the engine: either it holds engineMutex (blocking /api/train) or it
//    is the async worker, with job.running gating every other handler out.
string runTrainingAndBuildResult( bool continued )
{
	Capture cap;
	double finalError;
	try
	{
		finalError = modelPtr->train();
	}
	catch ( const exception& e )
	{
		return string( "{\"ok\":false,\"message\":\"training failed: " )
			+ jsonEscape( e.what() ) + "\",\"output\":\""
			+ jsonEscape( cap.text.str() ) + "\"}";
	}

	// ROC curves come from the model's own DataSet copy — that's where
	//    train() wrote the guesses
	DataSet& md = modelPtr->getDataSet();
	string roc;
	if ( md.getDiscrete() && md.getOutput() == 1 && md.trainLoaded() )
	{
		roc = ",\"roc\":{\"train\":" + jsonROCSeries( md.getTrainTwoSet() );
		if ( md.testLoaded() )
			roc += ",\"test\":" + jsonROCSeries( md.getTestTwoSet() );
		roc += "}";
	}

	// Why the run ended -- "cancelled" is a completed run too (the engine
	//    falls through to its normal epilogue on an observer stop)
	Iterative* iter = dynamic_cast< Iterative* >( modelPtr.get() );
	string stopReason = iter ? stopReasonName( iter->getStopReason() ) : "none";

	ostringstream msg;
	msg.precision( 6 );
	msg << ( continued ? "continued training; final error "
		: "trained; final error " ) << finalError;
	if ( stopReason == "cancelled" )
		msg << " (stopped by request)";

	lastReport = cap.text.str(); // downloadable afterwards as report.txt
	lastTrainError = finalError; // baseline for stepwise regression

	// Full statistics panel (additive; the roc object above is unchanged)
	string stats = jsonStats();
	string statsField = stats.empty() ? "" : ( ",\"stats\":" + stats );

	return string( "{\"ok\":true,\"message\":\"" ) + jsonEscape( msg.str() )
		+ "\",\"stopReason\":\"" + stopReason
		+ "\",\"output\":\"" + jsonEscape( lastReport ) + "\"" + roc
		+ statsField + "}";
}

string handleTrain( const httplib::Request& req )
{
	unsigned algorithm = ( unsigned ) atol( param( req, "algorithm" ).c_str() ),
		maxIter = ( unsigned ) atol( param( req, "maxiter" ).c_str() );
	string seed = param( req, "seed" );
	bool async = ( param( req, "async" ) == "1" );

	if ( !modelPtr )
		return jsonMsg( false, "create a model first" );
	if ( algorithm < 1 || algorithm > 3 )
		return jsonMsg( false, "algorithm must be 1, 2 or 3" );
	if ( maxIter < 1 )
		return jsonMsg( false, "max iterations must be at least 1" );
	if ( !seed.empty() )
		util::set_seed( ( unsigned ) atol( seed.c_str() ) );

	Network* net = dynamic_cast< Network* >( modelPtr.get() );
	Iterative* iter = dynamic_cast< Iterative* >( modelPtr.get() );

	// Training does NOT re-randomize: a second Train continues from the
	//    current weights (pick up where it left off). Only a model that has
	//    never had weights set gets randomized here, so the first Train after
	//    "Create model" just works; use /api/randomize to start over.
	//    "continued" (for the message) means previously *trained*, not merely
	//    randomized -- lastTrainError is -1 after model creation or randomize.
	bool continued = ( lastTrainError >= 0 );
	if ( !net->getWeightsSet() )
		net->randomize();
	iter->setMaxIterations( maxIter );
	net->setTrainingType( algorithm - 1 );

	if ( !async ) // the original blocking contract, unchanged
		return runTrainingAndBuildResult( continued );

	// Async: hand the engine to a worker thread and return at once. The
	//    caller holds engineMutex here; job.running (set before the worker
	//    exists) is what keeps every other handler out from now on.
	if ( job.worker.joinable() )
		job.worker.join(); // reap the previous, finished run's thread

	{
		lock_guard< mutex > lock( job.progressMutex );
		job.iters.clear();
		job.trainErr.clear();
		job.testErr.clear();
		job.keepEvery = 1;
		job.sampleCounter = 0;
		job.result.clear();
	}
	job.cancel = false;
	job.running = true;

	DataSet& md = modelPtr->getDataSet();
	unsigned nTest = md.testLoaded() ? md.getNumTest() : 0;

	job.worker = thread( [ continued, net, iter, nTest ]
	{
		// This thread's screen starts at cout (thread_local); the Capture
		//    inside runTrainingAndBuildResult redirects it for the run
		GuiObserver observer( net, nTest );
		iter->setObserver( &observer );
		string result = runTrainingAndBuildResult( continued );
		iter->setObserver( nullptr );

		{
			lock_guard< mutex > lock( job.progressMutex );
			job.result = result; // publish BEFORE running goes false
		}
		job.running = false;
	} );

	return jsonMsg( true, "training started" );
}

// Progress of the async run: the decimated error series while it runs, plus
//    the finished run's full result (same shape as blocking /api/train) once
//    it completes. The page polls this.
string handleTrainStatus()
{
	lock_guard< mutex > lock( job.progressMutex );

	ostringstream out;
	out.precision( 6 );
	out << "{\"ok\":true,\"running\":" << ( job.running ? "true" : "false" )
		<< ",\"series\":{\"iter\":[";
	for ( unsigned i = 0; i < job.iters.size(); i++ )
		out << ( i ? "," : "" ) << job.iters[ i ];
	out << "],\"train\":[";
	for ( unsigned i = 0; i < job.trainErr.size(); i++ )
		out << ( i ? "," : "" ) << jnum( job.trainErr[ i ] );
	out << "],\"test\":[";
	for ( unsigned i = 0; i < job.testErr.size(); i++ ) // < 0 = not sampled
		out << ( i ? "," : "" )
			<< ( job.testErr[ i ] < 0 ? "null" : jnum( job.testErr[ i ] ) );
	out << "]},\"result\":" << ( job.result.empty() ? "null" : job.result )
		<< "}";
	return out.str();
}

string handleTrainStop()
{
	if ( !job.running )
		return jsonMsg( false, "no training in progress" );
	job.cancel = true;
	return jsonMsg( true, "stopping at the end of this iteration" );
}

// Recompute the full statistics object for the current trained model without
//    retraining (the guesses from the last run are still in the model's DataSet)
string handleStats()
{
	if ( !modelPtr )
		return jsonMsg( false, "create and train a model first" );
	if ( lastTrainError < 0 )
		return jsonMsg( false, "train the model first" );
	string stats = jsonStats();
	if ( stats.empty() )
		return jsonMsg( false, "no classification statistics for this model" );
	return string( "{\"ok\":true,\"stats\":" ) + stats + "}";
}

string handleRegress( const httplib::Request& req )
{
	string structure = param( req, "structure" ),
		direction = param( req, "direction" );
	double threshold = atof( param( req, "threshold" ).c_str() );

	Network* net = modelPtr ? dynamic_cast< Network* >( modelPtr.get() ) : nullptr;
	if ( !net )
		return jsonMsg( false, "train a network or logistic model first" );
	if ( !net->getWeightsSet() )
		return jsonMsg( false, "the network must be trained before regression" );
	if ( lastTrainError < 0 )
		return jsonMsg( false, "train the model before regression" );
	if ( structure.empty() )
		return jsonMsg( false, "specify the input-variable structure "
			"(e.g. 0;1-4;5;6,7;8;9;10)" );
	if ( direction != "reverse" && direction != "forward" )
		return jsonMsg( false, "direction must be reverse or forward" );
	if ( threshold <= 0 || threshold >= 1 )
		return jsonMsg( false, "the p-value threshold must be between 0 and 1" );

	// Parse the variable structure exactly as the CLI does
	vector< vector< unsigned > > variable_defs;
	try
	{
		variable_defs = util::variable_parse( structure );
	}
	catch ( const util::utilErr& e )
	{
		return jsonMsg( false, string( "variable structure: " ) + e.what() );
	}
	if ( variable_defs.empty() )
		return jsonMsg( false, "variable structure parsed to nothing" );

	RegressNet regressionObj;
	regressionObj.setNetwork( net, lastTrainError );
	try
	{
		regressionObj.setInputStructure( variable_defs );
	}
	catch ( const RegressNet::RegressNetErr& e )
	{
		return jsonMsg( false, e.what() );
	}
	regressionObj.setThreshold( threshold ); // no stdin prompt in the GUI

	Capture cap;
	try
	{
		if ( direction == "reverse" )
			regressionObj.reverse_regress();
		else
			regressionObj.forward_regress();
	}
	catch ( const RegressNet::RegressNetErr& e )
	{
		return string( "{\"ok\":false,\"message\":\"" ) + jsonEscape( e.what() )
			+ "\",\"output\":\"" + jsonEscape( cap.text.str() ) + "\"}";
	}

	// Note: the training report (lastReport, the Report download) is left
	//    intact; the regression output is shown on the page and, when history
	//    logging is on, also appended to neuron.log in the workspace.
	return string( "{\"ok\":true,\"message\":\"" )
		+ jsonEscape( direction + " stepwise regression complete" )
		+ "\",\"output\":\"" + jsonEscape( cap.text.str() ) + "\"}";
}

// Produce one session artifact as a file in the workspace (the server's
//    directory), using the same engine save methods the CLI menus call.
//    Returns true and the filename on success; the engine's own message
//    (captured) becomes the error otherwise.
bool saveArtifact( const string& what, string& name, string& err )
{
	static const struct { const char *what, *file; } names[] = {
		{ "network", "network.txt" }, { "scales", "scales.txt" },
		{ "train_set", "train_set.txt" }, { "test_set", "test_set.txt" },
		{ "train_guesses", "train_guesses.txt" },
		{ "test_guesses", "test_guesses.txt" } };
	name.clear();
	for ( const auto& n : names )
		if ( what == n.what )
			name = n.file;
	if ( name.empty() )
	{
		err = "unknown artifact: " + what;
		return false;
	}

	Capture cap;
	bool ok = false;

	if ( what == "network" )
	{
		Network* net = modelPtr ? dynamic_cast< Network* >( modelPtr.get() ) : nullptr;
		if ( !net || !net->getWeightsSet() )
			err = "no trained network to save yet";
		else
			ok = net->save( name );
	}
	else if ( what == "scales" || what == "train_set" || what == "test_set" )
	{
		if ( !dataPtr )
			err = "load a dataset first";
		else if ( what == "scales" )
			ok = dataPtr->saveScales( name );
		else if ( what == "train_set" )
			ok = dataPtr->saveTrain( name );
		else if ( !dataPtr->testLoaded() ) // saveTest doesn't check this itself
			err = "no test set in this session";
		else
			ok = dataPtr->saveTest( name );
	}
	else // guesses live in the model's DataSet copy, written by train()
	{
		if ( !modelPtr )
			err = "train a model first";
		else if ( what == "train_guesses" )
			ok = modelPtr->getDataSet().saveTrainTwoSet( name );
		else
			ok = modelPtr->getDataSet().saveTestTwoSet( name );
	}

	if ( !ok && err.empty() )
	{
		err = cap.text.str();
		if ( err.empty() )
			err = "could not save " + what;
	}
	return ok;
}

// While an async run owns the engine, every other engine-touching request is
//    refused outright with 409 -- no queueing. Returns true if it refused.
//    (Callers hold engineMutex; job.running was set under it too.)
bool busyGate( httplib::Response& res )
{
	if ( !job.running )
		return false;
	res.status = 409;
	res.set_content(
		"{\"ok\":false,\"message\":\"training in progress\",\"busy\":true}",
		"application/json" );
	return true;
}

void openInBrowser( const string& url )
{
#if defined( _WIN32 )
	system( ( "start \"\" " + url ).c_str() );
#elif defined( __APPLE__ )
	system( ( "open " + url ).c_str() );
#else
	system( ( "xdg-open " + url + " >/dev/null 2>&1 &" ).c_str() );
#endif
}

} // namespace

int run_gui( bool openBrowser )
{
	httplib::Server svr;

	svr.Get( "/", []( const httplib::Request&, httplib::Response& res )
	{
		res.set_content( GUI_PAGE, "text/html; charset=utf-8" );
	} );

	svr.Get( "/api/version", []( const httplib::Request&, httplib::Response& res )
	{
		res.set_content( NEURON_PACKAGE_STRING, "text/plain" );
	} );

	svr.Post( "/api/load", []( const httplib::Request& req, httplib::Response& res )
	{
		lock_guard< mutex > lock( engineMutex );
		if ( busyGate( res ) ) return;
		res.set_content( handleLoad( req ), "application/json" );
	} );

	svr.Post( "/api/model", []( const httplib::Request& req, httplib::Response& res )
	{
		lock_guard< mutex > lock( engineMutex );
		if ( busyGate( res ) ) return;
		res.set_content( handleModel( req ), "application/json" );
	} );

	svr.Post( "/api/randomize", []( const httplib::Request& req, httplib::Response& res )
	{
		lock_guard< mutex > lock( engineMutex );
		if ( busyGate( res ) ) return;
		res.set_content( handleRandomize( req ), "application/json" );
	} );

	svr.Post( "/api/train", []( const httplib::Request& req, httplib::Response& res )
	{
		lock_guard< mutex > lock( engineMutex );
		if ( busyGate( res ) ) return;
		res.set_content( handleTrain( req ), "application/json" );
	} );

	// Status and stop are the two doors that stay open while a run owns the
	//    engine; neither touches it (progress buffer and atomics only)
	svr.Get( "/api/train/status", []( const httplib::Request&, httplib::Response& res )
	{
		res.set_content( handleTrainStatus(), "application/json" );
	} );

	svr.Post( "/api/train/stop", []( const httplib::Request&, httplib::Response& res )
	{
		res.set_content( handleTrainStop(), "application/json" );
	} );

	svr.Get( "/api/stats", []( const httplib::Request&, httplib::Response& res )
	{
		lock_guard< mutex > lock( engineMutex );
		if ( busyGate( res ) ) return;
		res.set_content( handleStats(), "application/json" );
	} );

	svr.Post( "/api/regress", []( const httplib::Request& req, httplib::Response& res )
	{
		lock_guard< mutex > lock( engineMutex );
		if ( busyGate( res ) ) return;
		res.set_content( handleRegress( req ), "application/json" );
	} );

	// Session artifacts: written into the workspace by the engine's own
	//    save methods AND sent to the browser as a download
	svr.Get( "/api/save/:what", []( const httplib::Request& req, httplib::Response& res )
	{
		lock_guard< mutex > lock( engineMutex );
		if ( busyGate( res ) ) return;
		string what = req.path_params.at( "what" );

		if ( what == "report" ) // the last training run's captured output
		{
			if ( lastReport.empty() )
			{
				res.status = 400;
				res.set_content( jsonMsg( false, "no training run yet" ),
					"application/json" );
				return;
			}
			res.set_header( "Content-Disposition",
				"attachment; filename=report.txt" );
			res.set_content( lastReport, "text/plain" );
			return;
		}

		string name, err;
		if ( !saveArtifact( what, name, err ) )
		{
			res.status = 400;
			res.set_content( jsonMsg( false, err ), "application/json" );
			return;
		}

		ifstream in( name.c_str(), ios::in | ios::binary );
		stringstream bytes;
		bytes << in.rdbuf();
		res.set_header( "Content-Disposition",
			( "attachment; filename=" + name ).c_str() );
		res.set_content( bytes.str(), "text/plain" );
	} );

	// Port 0: the OS assigns a free port — no collisions, ever
	int port = svr.bind_to_any_port( "127.0.0.1" );
	if ( port <= 0 )
	{
		cerr << "Could not bind a loopback port for the GUI." << endl;
		return 1;
	}

	string url = "http://127.0.0.1:" + to_string( port ) + "/";
	cout << "neuron GUI: " << url << endl;
	cout << "Loopback only; press Ctrl-C to quit." << endl;

	if ( openBrowser )
		openInBrowser( url );

	svr.listen_after_bind();

	// If the server ever exits normally, don't destroy a joinable worker
	if ( job.worker.joinable() )
	{
		job.cancel = true;
		job.worker.join();
	}

	return 0;
}
