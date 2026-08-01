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
#include <ctime>
#include <set>
#include <sstream>
#include <thread>

#include "gui.h"
#include "autoalgo.h"
#include "obd.h"
#include "dataset.h"
#include "logistic.h"
#include "simpleprop.h"
#include "bareprop.h"
#include "backprop.h"
#include "network.h"
#include "modelfactory.h"
#include "ldfa.h"
#include "qdfa.h"
#include "iterative.h"
#include "regressnet.h"
#include "crossval.h"
#include "cvadapters.h"
#include "cvreport.h"
#include "delong.h"
#include "clustered_auc.h"
#include "split.h"
#include "utility.h"
#include "version.h"

#include "gui_page.h" // generated from src/gui_page.html by CMake

using namespace std;

namespace {

// The engine state driven by the page — mirrors the driver's owners
mutex engineMutex;
unique_ptr< DataSet > dataPtr;
unique_ptr< Model > modelPtr;
// The last discriminant function analysis, kept so its guesses stay savable
//    (the CLI offers to save them right after a DFA run); it never becomes
//    the trained model
unique_ptr< Model > dfaPtr;
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
const unsigned JOB_MAX_POINTS = 2000;

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

	// OBD progress for the status poll (empty phase = not an OBD run)
	string obdPhase;
	unsigned obdHidden = 0;

	// Stepwise-regression progress as a ready-made JSON object body (empty =
	//    not a stepwise run). Kept as its OWN field rather than reusing the obd
	//    object: a caller must never have to read "hidden" and guess it means a
	//    candidate variable.
	string stepwise;

	// Cross-validation progress -- its own field for the same reason (2026-07-29).
	//    A four-procedure, five-fold nested-OBD run is the most expensive thing the
	//    GUI does and used to show one unchanging phase word for its whole
	//    duration. The engine fields below are set from crossval::Progress and,
	//    for the inner architecture search, from obd::ProgressFn; handleTrainStatus
	//    renders them. Guarded by progressMutex like everything else here.
	string cvStage;       // "cross-validation" | "locked-test evaluation"; "" = not a CV run
	string cvProcedure;   // the procedure now fitting
	unsigned cvProcIndex = 0, cvProcCount = 0, cvProcDone = 0;
	unsigned cvFold = 0, cvK = 0, cvFoldsDone = 0;
	// The nested OBD search inside the CURRENT fold. Cleared whenever the fold or
	//    procedure changes, so a finished fold's last trial can never be displayed
	//    as the next fold's progress.
	string cvInnerPhase;
	unsigned cvInnerHidden = 0;

	// Reset every CV progress field. One place, so a new field cannot be added
	//    and then left stale by a path that clears the others (the reason
	//    "not copied" is written out rather than omitted -- HISTORY 2026-07-27).
	void clearCvProgress()
	{
		cvStage.clear(); cvProcedure.clear();
		cvProcIndex = cvProcCount = cvProcDone = 0;
		cvFold = cvK = cvFoldsDone = 0;
		cvInnerPhase.clear(); cvInnerHidden = 0;
	}

	// Clear every per-run field before a new job starts, so nothing from the last
	//    run can be read as this one's. Train, OBD, CV and stepwise each used to
	//    do this by hand, which meant a new progress field had to be remembered in
	//    four places -- and the one that forgot would publish stale progress rather
	//    than none. Callers must hold progressMutex (they already did).
	void resetForNewRun()
	{
		iters.clear();
		trainErr.clear();
		testErr.clear();
		keepEvery = 1;
		sampleCounter = 0;
		result.clear();
		obdPhase.clear();
		obdHidden = 0;
		stepwise.clear();
		clearCvProgress();
	}

	// Append one decimated (iteration, train, test) sample. Guards its own
	//    mutex, so callers must NOT already hold progressMutex.
	void pushSample( unsigned iteration, double trainError, double testError )
	{
		lock_guard< mutex > lock( progressMutex );
		if ( ++sampleCounter % keepEvery == 0 )
		{
			iters.push_back( iteration );
			trainErr.push_back( trainError );
			testErr.push_back( testError );
			if ( iters.size() >= JOB_MAX_POINTS )
			{
				// Halve the series, double the stride for future samples
				for ( unsigned i = 0, j = 0; j < iters.size(); i++, j += 2 )
				{
					iters[ i ] = iters[ j ];
					trainErr[ i ] = trainErr[ j ];
					testErr[ i ] = testErr[ j ];
				}
				iters.resize( iters.size() / 2 );
				trainErr.resize( trainErr.size() / 2 );
				testErr.resize( testErr.size() / 2 );
				keepEvery *= 2;
			}
		}
	}
};
TrainJob job;

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
			job.pushSample( iteration, setError, testError );
		}
		return !job.cancel.load();
	}
};

// Carries a Stop request into whichever candidate subnetwork is training right
//    now. It samples nothing: each candidate restarts its own error curve from
//    iteration 0, so a single error-vs-iteration chart across a stepwise run
//    would be a sawtooth of unrelated fits. The progress a user actually needs
//    is the candidate accounting, which the engine pushes separately.
//    whyStopped() is left at the base default (STOP_CANCELLED) -- that IS what
//    happened, and a cancelled fit is not convergence, so it stops the analysis
//    through the same door an unfinished fit does.
struct StepwiseCancelObserver : Iterative::Observer
{
	bool onIteration( unsigned, double ) override { return !job.cancel.load(); }
};

// The engine's stop reason as a JSON-friendly name. These tokens are the
//    machine-readable contract: a caller must be able to tell a converged fit
//    (grad_max / plateau / validation_early_stop) from a run that merely ran
//    out of ceiling (max_iterations) or was stopped from outside (cancelled).
const char* stopReasonName( Iterative::StopReason r )
{
	return Iterative::stopReasonToken( r ); // one spelling, engine-owned
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

	// The Pearson value is the STATISTIC, not a p (no valid p exists at the
	//    individual level -- see TwoSet::PKX2calc); n travels beside it in
	//    this object, and X2 >> n is the overdispersion signal
	emit( "pearsonX2", [ &t ] { return t.getPearsonX2(); } );
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

// Parse a comma-separated list of 1-based input COLUMN numbers into 0-based
//    input-node positions. Returns "" on success, else the error to report.
//    Blank tokens are skipped so a trailing comma is not an error.
//
//    One parser for every endpoint that names columns (/api/load's strata and
//    group, /api/cv's own): the range message and the 1-based -> node
//    conversion are exactly the kind of thing that drifts when each handler
//    writes its own loop, and a column off by one silently stratifies on the
//    wrong variable.
string parseColumnList( const string& text, unsigned nInput, const char* what,
	vector< unsigned >& out )
{
	out.clear();
	stringstream ss( text );
	string tok;
	while ( getline( ss, tok, ',' ) )
	{
		size_t a = tok.find_first_not_of( " \t" );
		if ( a == string::npos ) continue; // skip a blank token
		long v = atol( tok.c_str() + a );
		if ( v < 1 || ( unsigned ) v > nInput )
			return string( what ) + " column out of range (1.."
				+ to_string( nInput ) + ")";
		out.push_back( ( unsigned ) ( v - 1 ) ); // 1-based -> input node
	}
	return "";
}

// Whether the field arrived at all -- req.has_param alone is FALSE for a
//    field sent as a multipart part (the page's file-upload posts), which is
//    exactly when get_file_value holds it instead
bool hasParam( const httplib::Request& req, const string& name )
{
	return req.has_param( name ) || req.has_file( name );
}

// Just the final component of a picked file's name — no directory parts
string safeBasename( const string& name )
{
	string::size_type slash = name.find_last_of( "/\\" );
	string base = ( slash == string::npos ? name : name.substr( slash + 1 ) );
	return base.empty() ? "upload.txt" : base;
}

// RAII engine-output capture: restores cout even if the engine throws
// The per-request engine-output capture. A thin name over util::ScreenCapture, kept
//    because `cap.text.str()` reads through this whole file.
//
//    It used to restore unconditionally to cout, which is wrong twice: a
//    capture inside a capture sent the outer scope's remaining output to the
//    console instead of the outer buffer, and a worker thread's capture reset
//    that thread's stream to cout rather than to whatever it had. util::ScreenCapture
//    restores the stream that was current when it was constructed.
struct Capture
{
	util::ScreenCapture scoped;
	ostringstream& text;
	Capture() : text( scoped.buffer() ) {}
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

// --- Per-action audit log --------------------------------------------------
//    Every GUI user action is appended, with a timestamp and the exact
//    parameter values it carried, to neuron_actions.log in the workspace
//    (beside the data, like neuron.log). This is the audit trail Craig asked
//    for (2026-07-19): "every user action needs to be logged". It complements
//    neuron.log, which records the engine's operations and the fully
//    self-describing training-run header (algorithm, eta, weight decay, and
//    every stopping condition). Its own mutex, so writes never interleave even
//    from the un-serialized /api/train/stop.
static mutex auditMutex;

void logActionLine( const string& line )
{
	lock_guard< mutex > lk( auditMutex );
	string path = util::run_path( "neuron_actions.log" );
	ofstream f( path.c_str(), ios::out | ios::app );
	if ( !f.is_open() ) return;
	time_t now = time( nullptr );
	char stamp[ 32 ];
	strftime( stamp, sizeof stamp, "%Y-%m-%dT%H:%M:%S", localtime( &now ) );
	f << stamp << " " << line << endl;
}

void logAction( const httplib::Request& req, const string& action )
{
	string line = action;
	for ( const auto& p : req.params )
		if ( p.second.size() <= 256 ) // skip an oversized value, keep the key
			line += " " + p.first + "=" + p.second;
	for ( const auto& fp : req.files ) // uploads: the filename, never the content
		line += " " + fp.first + "=@" + fp.second.filename;
	logActionLine( line );
}

// --- Handlers --------------------------------------------------------------

string handleLoad( const httplib::Request& req )
{
	logAction( req, "load" );
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

	// --- Dataset characteristics (CLI dataset menu 11/12) -----------------
	//    Validated and applied BEFORE the load: the variate bounds drive the
	//    scaling, and the threshold classifies every guess from then on. All
	//    read through param() (multipart-safe); an empty field means "keep the
	//    engine's default".
	string thresholdStr = param( req, "threshold" );
	if ( !thresholdStr.empty() )
	{
		double threshold = atof( thresholdStr.c_str() );
		if ( !discrete )
			return jsonMsg( false, "a classification threshold applies only to "
				"a discrete outcome" );
		if ( !( threshold > 0 && threshold < 1 ) )
			return jsonMsg( false, "threshold must be between 0 and 1" );
		ds->setThreshold( threshold );
	}

	string inLoStr = param( req, "in_lower" ), inHiStr = param( req, "in_upper" );
	if ( !inLoStr.empty() || !inHiStr.empty() )
	{
		double lo = inLoStr.empty() ? ds->getInLower() : atof( inLoStr.c_str() );
		double hi = inHiStr.empty() ? ds->getInUpper() : atof( inHiStr.c_str() );
		if ( !( lo < hi ) )
			return jsonMsg( false, "input variate bounds must have lower < upper" );
		ds->setInLower( lo );
		ds->setInUpper( hi );
	}

	string outLoStr = param( req, "out_lower" ), outHiStr = param( req, "out_upper" );
	if ( !outLoStr.empty() || !outHiStr.empty() )
	{
		// The CLI refuses these while the output is discrete (fixed at 0/1)
		if ( discrete )
			return jsonMsg( false, "output variate bounds apply only to a "
				"continuous outcome (discrete=0)" );
		double lo = outLoStr.empty() ? ds->getOutLower() : atof( outLoStr.c_str() );
		double hi = outHiStr.empty() ? ds->getOutUpper() : atof( outHiStr.c_str() );
		if ( !( lo < hi ) )
			return jsonMsg( false, "output variate bounds must have lower < upper" );
		ds->setOutLower( lo );
		ds->setOutUpper( hi );
	}

	if ( !param( req, "history" ).empty() ) // dataset-operations logging
		ds->setHistory( param( req, "history" ) == "1" );

	// --- ROC reporting settings (CLI dataset menu 13) ---------------------
	//    Validated here, applied after the load (the TwoSets must exist);
	//    both sets get the value, which is where the CLI's per-set choice
	//    collapses when everything loads in one request. (The trapezoidal ROC
	//    area is now the exact AUC over every operating point -- there is no
	//    threshold count to set, so that former option is gone.)
	string rocReportStr = param( req, "roc_report" ),
		rocMinStr = param( req, "roc_min" );
	if ( !rocReportStr.empty() || !rocMinStr.empty() )
	{
		if ( !discrete || nO != 1 )
			return jsonMsg( false, "ROC reporting settings need a discrete "
				"1-output dataset" );
		if ( !rocReportStr.empty() && rocReportStr != "both"
			&& rocReportStr != "either" )
			return jsonMsg( false, "roc_report must be both or either" );
		if ( !rocMinStr.empty() && atol( rocMinStr.c_str() ) < 2 )
			return jsonMsg( false, "roc_min must be at least 2" );
	}

	Capture cap;
	bool ok = false;

	// test_n: the CLI's whole-number split form ("place exactly n exemplars
	//    in the test set") -- fraction covers the decimal and n/d forms, but
	//    randomizeD truncates ratio*N, so an exact count needs randomize(n)
	string testNStr = param( req, "test_n" );

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
				if ( fraction > 0 || !testNStr.empty() )
					return jsonMsg( false, "a continuous outcome cannot be "
						"stratified into a train/test split; load with "
						"fraction=0, or pre-split the data and use mode=train "
						"with a testfile" );
				ok = ds->raw2train();
			}
			else
			{
				// Phase 2: optional covariate stratification. The outcome is
				//    always a factor; strata = comma-separated 1-based input
				//    column numbers to also stratify on, strata_bins = the
				//    quantile-bin count for a continuous stratum column.
				string strataStr = param( req, "strata" );
				if ( !strataStr.empty() )
				{
					vector< unsigned > cols;
					string bad = parseColumnList( strataStr, ds->getInput(),
						"strata", cols );
					if ( !bad.empty() ) return jsonMsg( false, bad );
					ds->setStrataColumns( cols );
				}
				string binsStr = param( req, "strata_bins" );
				if ( !binsStr.empty() )
				{
					if ( atol( binsStr.c_str() ) < 2 )
						return jsonMsg( false, "strata_bins must be at least 2" );
					ds->setStrataBins( ( unsigned ) atol( binsStr.c_str() ) );
				}

				// Phase 3: optional group-aware split. group = comma-separated
				//    1-based input columns; rows sharing identical values on all
				//    of them stay together (a harder, unseen-group test). Takes
				//    precedence over covariate strata in the engine.
				string groupStr = param( req, "group" );
				if ( !groupStr.empty() )
				{
					vector< unsigned > cols;
					string bad = parseColumnList( groupStr, ds->getInput(),
						"group", cols );
					if ( !bad.empty() ) return jsonMsg( false, bad );
					ds->setGroupColumns( cols );
				}

				// Phase 4c: a validation fraction/count triggers a THREE-WAY
				//    (train/validation/test) split -- outcome-stratified, so
				//    selection (OBD) monitors the validation set and the test
				//    set stays untouched. Does not compose with covariate strata
				//    / grouping yet.
				string valFracStr = param( req, "val_fraction" );
				string valNStr = param( req, "val_n" );
				if ( !valFracStr.empty() || !valNStr.empty() )
				{
					if ( !ds->getStrataColumns().empty()
						|| !ds->getGroupColumns().empty() )
						return jsonMsg( false, "a validation (three-way) split is "
							"outcome-stratified only; remove strata=/group= or the "
							"validation fraction" );

					if ( !valNStr.empty() ) // exact-count form (paired with test_n)
					{
						if ( testNStr.empty() )
							return jsonMsg( false, "val_n needs test_n (the count form)" );
						if ( atol( valNStr.c_str() ) < 1 )
							return jsonMsg( false, "val_n must be at least 1" );
						ok = ds->randomize3( ( unsigned ) atol( testNStr.c_str() ),
							( unsigned ) atol( valNStr.c_str() ) );
					}
					else // fraction form (paired with fraction)
					{
						double vf = atof( valFracStr.c_str() );
						if ( !( vf > 0 && vf < 1 ) )
							return jsonMsg( false, "val_fraction must be between 0 and 1" );
						ok = ds->randomize3D( fraction, vf );
					}
				}
				else if ( !testNStr.empty() )
					ok = ds->randomize( ( unsigned ) atol( testNStr.c_str() ) );
				else
					ok = ds->randomizeD( fraction );
			}
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

	// Explicit ROC reporting overrides (validated above), exactly where the
	//    CLI lets a user revise them after a load
	if ( !rocReportStr.empty() )
	{
		bool both = ( rocReportStr == "both" );
		if ( ds->trainLoaded() )
			ds->getTrainTwoSet().setROCReportFlag( both );
		if ( ds->testLoaded() )
			ds->getTestTwoSet().setROCReportFlag( both );
	}
	if ( !rocMinStr.empty() )
	{
		if ( ds->trainLoaded() )
			ds->getTrainTwoSet().setROCthresh( ( unsigned ) atol( rocMinStr.c_str() ) );
		if ( ds->testLoaded() )
			ds->getTestTwoSet().setROCthresh( ( unsigned ) atol( rocMinStr.c_str() ) );
	}

	ostringstream msg;
	msg << ds->getInput() << " inputs, " << ds->getOutput() << " output"
		<< ( ds->getOutput() == 1 ? "; " : "s; " )
		<< ds->getNumTrain() << " training exemplars";
	if ( ds->valLoaded() )
		msg << ", " << ds->getNumVal() << " validation exemplars";
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

	// Phase 2/3: when the split was stratified on covariates or grouped, return
	//    the representativeness diagnostic (captured above) so the page shows it.
	if ( !ds->getStrataColumns().empty() || !ds->getGroupColumns().empty() )
	{
		string rpt = cap.text.str();
		size_t at = rpt.find( "Representativeness diagnostic" );
		if ( at != string::npos )
			msg << "\n\n" << rpt.substr( at );
	}

	dataPtr = std::move( ds );
	modelPtr.reset(); // a new dataset invalidates any existing model
	dfaPtr.reset();   // and any prior discriminant analysis
	lastTrainError = -1; // and any prior training

	// Which held-out set a training monitor will watch, from the engine's own rule
	//    (DataSet::monitorSet) -- so the page LABELS its live and OBD charts with
	//    the set that is actually monitored instead of assuming "test". Sent on the
	//    load response because that is when the answer changes.
	return string( "{\"ok\":true,\"message\":\"" ) + jsonEscape( msg.str() )
		+ "\",\"monitor\":\"" + dataPtr->monitorSetName() + "\"}";
}

// Parse a hidden-layer spec: comma-separated positive integers ("5" or "5,3").
//    Returns an empty vector if the spec is empty or holds any non-positive
//    value, so the caller can reject it uniformly.
static vector< unsigned > parseLayers( const string& spec )
{
	vector< unsigned > out;
	stringstream ss( spec );
	string tok;
	while ( getline( ss, tok, ',' ) )
	{
		size_t a = tok.find_first_not_of( " \t" );
		if ( a == string::npos ) continue; // skip a blank token
		long v = atol( tok.c_str() + a );
		if ( v < 1 ) return vector< unsigned >(); // any non-positive invalidates all
		out.push_back( ( unsigned ) v );
	}
	return out;
}

string handleModel( const httplib::Request& req )
{
	logAction( req, "model" );
	string type = param( req, "type" );
	string mode = param( req, "mode" ); // "load" = load a saved network from a file

	if ( !dataPtr )
		return jsonMsg( false, "load a dataset first" );

	Capture cap;
	lastTrainError = -1; // a fresh model has not been trained yet

	// Logging toggles (CLI model menu 6/7) -- default ON, matching the engine,
	//    so a request that omits them logs as before. hasParam, not
	//    req.has_param: the page's load-network post is multipart, where these
	//    arrive as parts and has_param alone would silently keep the default.
	bool logLastop  = !( hasParam( req, "log_lastop" )  && param( req, "log_lastop" )  == "0" );
	bool logHistory = !( hasParam( req, "log_history" ) && param( req, "log_history" ) == "0" );

	// Output error function (CLI model menu 3), shared by the create and load
	//    paths. Default follows the data; an explicit choice overrides, but
	//    X-entropy needs a discrete output. (Logistic is X-entropy by
	//    definition and never consults this.)
	string ef = param( req, "errfunc" );
	if ( ef == "xentropy" && !dataPtr->getDiscrete() )
		return jsonMsg( false, "X-entropy error needs a discrete output" );
	bool xe = ( ef == "xentropy" ) ? true
		: ( ef == "lms" ) ? false
		: dataPtr->getDiscrete(); // default, as the CLI initializes it

	// --- Load a saved network from a file (CLI model menu 4) --------------
	//    The type is the file's first line, exactly as the CLI loader reads it.
	if ( mode == "load" )
	{
		string saved, err;
		string netFile = resolveFile( req, "file", "path", saved, err );
		if ( !err.empty() ) return jsonMsg( false, err );
		if ( netFile.empty() || !fileExists( netFile ) )
			return jsonMsg( false, "can't open network file: " + netFile );

		string line;
		bool backpropBias = true;
		{
			ifstream f( netFile.c_str() );
			getline( f, line );
			if ( !line.empty() && line.back() == '\r' ) line.pop_back(); // CRLF files
			// A BackProp file carries its bias flag on line 2, and load()
			//    reads but never APPLIES it -- setHidden inside load sizes the
			//    weight matrices from biasFlag, so it must be set before load,
			//    exactly as the CLI driver does (neuron.cpp, model menu 4)
			if ( line == "BackProp" )
				f >> backpropBias;
		}

		modelPtr = modelfactory::createByTypeName( line, backpropBias );
		if ( !modelPtr )
			return jsonMsg( false, "unrecognized network type on line 1: '" + line + "'" );

		modelPtr->setDataSet( *dataPtr );
		modelPtr->setLastop( logLastop );
		modelPtr->setHistory( logHistory );
		if ( line != "Binary logistic" ) // X-entropy by definition for logistic
			xe ? modelPtr->setXEerror() : modelPtr->setLMSerror();

		if ( !dynamic_cast< Network* >( modelPtr.get() )->load( netFile ) )
		{
			string why = cap.text.str(); // the engine says what mismatched
			return jsonMsg( false, why.empty()
				? ( "failed to load the network from " + netFile ) : why );
		}
		lastTrainError = 0; // a loaded network has weights -- treat it as trained
		return jsonMsg( true, "loaded " + line + " network from " + netFile );
	}

	if ( type == "logistic" )
	{
		modelfactory::Spec spec;
		spec.logistic = true;
		spec.logLastop = logLastop;
		spec.logHistory = logHistory;
		string err;
		modelPtr = modelfactory::build( spec, *dataPtr, err );
		if ( !modelPtr ) return jsonMsg( false, err );
		return jsonMsg( true, "binary logistic regression ready" );
	}

	if ( type == "simpleprop" )
	{
		// hidden = comma-separated layer sizes; the factory (rule 6) turns the
		//    spec into the right concrete type (SimpleProp / BareProp / BackProp).
		vector< unsigned > layers = parseLayers( param( req, "hidden" ) );
		if ( layers.empty() )
			return jsonMsg( false, "hidden nodes must be one or more positive integers (e.g. 5 or 5,3)" );
		bool bias = !( hasParam( req, "bias" ) && param( req, "bias" ) == "0" );

		modelfactory::Spec spec;
		spec.bias = bias;
		spec.hidden = layers;
		spec.xentropy = xe;
		spec.logLastop = logLastop;
		spec.logHistory = logHistory;
		string err;
		modelPtr = modelfactory::build( spec, *dataPtr, err );
		if ( !modelPtr ) return jsonMsg( false, err );

		// The concrete class the factory chose, for the readback message.
		string kind = dynamic_cast< SimpleProp* >( modelPtr.get() ) ? "SimpleProp"
			: dynamic_cast< BareProp* >( modelPtr.get() ) ? "BareProp" : "BackProp";

		ostringstream msg;
		msg << kind << " " << dataPtr->getInput() << "-";
		for ( size_t i = 0; i < layers.size(); i++ ) msg << ( i ? "," : "" ) << layers[ i ];
		msg << "-" << dataPtr->getOutput() << " network ready ("
			<< ( xe ? "X-entropy" : "LMS" ) << ( bias ? "" : ", no bias" ) << ")";
		return jsonMsg( true, msg.str() );
	}

	return jsonMsg( false, "unknown model type: " + type );
}

// Discriminant function analysis (CLI main menu 4). A standalone analysis on
//    the loaded dataset -- it does NOT replace the trained model in modelPtr.
//    Linear (LDFA) or quadratic (QDFA); the choice is the only parameter, as in
//    the CLI. Returns the captured report plus the ROC curves and stats panel
//    built from the analysis's own TwoSets (same helpers as training).
string handleDFA( const httplib::Request& req )
{
	logAction( req, "dfa" );
	string type = param( req, "type" );
	if ( type != "linear" && type != "quadratic" )
		return jsonMsg( false, "DFA type must be linear or quadratic" );
	if ( !dataPtr )
		return jsonMsg( false, "load a dataset first" );
	if ( !dataPtr->getDiscrete() )
		return jsonMsg( false, "discriminant analysis needs a discrete output" );

	bool logLastop  = !( hasParam( req, "log_lastop" )  && param( req, "log_lastop" )  == "0" );
	bool logHistory = !( hasParam( req, "log_history" ) && param( req, "log_history" ) == "0" );

	unique_ptr< Model > dfa;
	if ( type == "linear" ) dfa = make_unique< LDFA >();
	else dfa = make_unique< QDFA >();

	Capture cap;
	dfa->setDataSet( *dataPtr );
	dfa->setLastop( logLastop );
	dfa->setHistory( logHistory );
	try { dfa->train(); }
	catch ( const exception& e )
	{
		return string( "{\"ok\":false,\"message\":\"DFA failed: " )
			+ jsonEscape( e.what() ) + "\",\"output\":\""
			+ jsonEscape( cap.text.str() ) + "\"}";
	}

	// Keep the finished analysis so its guesses stay savable (the CLI offers
	//    that right after a DFA run); the trained model is untouched
	dfaPtr = std::move( dfa );

	// ROC curves and the stats panel from the analysis's TwoSets (the guesses
	//    reportAccuracy just wrote), via the same helpers training uses.
	//    Multi-output DFA reports accuracy only, as in the CLI -- the guard
	//    below skips the 1-output ROC/statistics machinery for it.
	DataSet& dd = dfaPtr->getDataSet();
	string roc, stats;
	if ( dd.getDiscrete() && dd.getOutput() == 1 && dd.trainLoaded()
		&& dd.getTrainTwoSet().loaded() )
	{
		roc = ",\"roc\":{\"train\":" + jsonROCSeries( dd.getTrainTwoSet() );
		bool haveTest = ( dd.testLoaded() && dd.getTestTwoSet().loaded() );
		if ( haveTest )
			roc += ",\"test\":" + jsonROCSeries( dd.getTestTwoSet() );
		roc += "}";

		ostringstream s;
		s << "{\"train\":" << jsonStatsSet( dd.getTrainTwoSet() );
		if ( haveTest )
			s << ",\"test\":" << jsonStatsSet( dd.getTestTwoSet() );
		s << "}";
		stats = ",\"stats\":" + s.str();
	}

	lastReport = cap.text.str(); // downloadable as report.txt
	string msg = string( type == "linear" ? "linear" : "quadratic" )
		+ " discriminant function analysis complete";
	return string( "{\"ok\":true,\"message\":\"" ) + jsonEscape( msg )
		+ "\",\"output\":\"" + jsonEscape( lastReport ) + "\"" + roc + stats + "}";
}

string handleRandomize( const httplib::Request& req )
{
	logAction( req, "randomize" );
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
//    autoJson (may be empty) is the auto-selection block to embed in the
//    result; preamble (may be empty) is report text captured before this
//    call -- the probe decision summary -- prepended to the report; autoNote
//    (may be empty) names the selection in the user-visible message, because
//    the page shows the message, not the report.
string runTrainingAndBuildResult( bool continued, const string& autoJson,
	const string& preamble, const string& autoNote )
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
			+ jsonEscape( preamble + cap.text.str() ) + "\"}";
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
	msg << autoNote << ( continued ? "continued training; final error "
		: "trained; final error " ) << finalError;
	if ( stopReason == "cancelled" )
		msg << " (stopped by request)";
	// Reaching the iteration ceiling is a FAILURE TO CONVERGE, not a finished
	//    fit. Ordinary training keeps the weights -- the user may continue from
	//    them, which is the whole point of a resumable run -- but the result must
	//    never read as successful completion. The page renders this prominently.
	else if ( stopReason == "max_iterations" )
		msg << " -- WARNING: training did NOT converge. It stopped because it hit "
			"the maximum iteration count, which is a safety limit, not a stopping "
			"condition. The weights are kept so you can continue training; raise "
			"the maximum iterations, or set a stopping condition that can fire.";

	// OPERATION SUCCESS AND FIT VALIDITY ARE DIFFERENT FACTS. The request
	//    succeeded (ok:true) and the weights are resumable, but whether a
	//    stopping rule fired is reported separately so no caller -- the page, a
	//    script, an LLM -- has to infer convergence from ok.
	bool didConverge = iter ? Iterative::converged( iter->getStopReason() ) : false;
	bool hitCeiling = ( iter
		&& iter->getStopReason() == Iterative::STOP_MAX_ITERATIONS );
	string fitFields = string( ",\"converged\":" ) + ( didConverge ? "true" : "false" )
		+ ",\"ceilingExhausted\":" + ( hitCeiling ? "true" : "false" );

	lastReport = preamble + cap.text.str(); // downloadable as report.txt
	lastTrainError = finalError; // baseline for stepwise regression

	// Full statistics panel (additive; the roc object above is unchanged)
	string stats = jsonStats();
	string statsField = stats.empty() ? "" : ( ",\"stats\":" + stats );
	string autoField = autoJson.empty() ? "" : ( ",\"autoAlgo\":" + autoJson );

	return string( "{\"ok\":true,\"message\":\"" ) + jsonEscape( msg.str() )
		+ "\",\"stopReason\":\"" + stopReason + "\"" + fitFields
		// The held-out set the live error chart sampled, named by the engine rule
		//    (DataSet::monitorSet) so the legend matches what was watched.
		+ ",\"monitor\":\"" + dataPtr->monitorSetName() + "\""
		+ ",\"output\":\"" + jsonEscape( lastReport ) + "\"" + roc
		+ statsField + autoField + "}";
}

// The auto-selection result as JSON (the report carries the same story as
//    the printed decision summary; this is the structured form)
string jsonAutoAlgo( const autoalgo::Result& r )
{
	ostringstream out;
	out.precision( 6 );
	out << "{\"selected\":" << r.selected << ",\"selectedName\":\""
		<< jsonEscape( r.selectedName ) << "\",\"cancelled\":"
		<< ( r.cancelled ? "true" : "false" ) << ",\"probes\":[";
	for ( unsigned i = 0; i < r.probes.size(); i++ )
	{
		const autoalgo::Probe& p = r.probes[ i ];
		out << ( i ? "," : "" ) << "{\"algorithm\":" << p.algorithm
			<< ",\"name\":\"" << jsonEscape( p.name )
			<< "\",\"error\":" << ( p.usable ? jnum( p.error ) : "null" )
			<< ",\"iterations\":" << p.iterations
			<< ",\"stopReason\":\"" << stopReasonName( p.stop ) << "\"}";
	}
	out << "]}";
	return out.str();
}

// The whole training job: optional auto algorithm selection (probe all
//    three from identical weights, adopt the winner -- which REPLACES
//    modelPtr, so every pointer is re-derived after it), then the real
//    training run, observed for the GUI's realtime chart when asked. The
//    caller must own the engine (see runTrainingAndBuildResult).
string runTrainJob( bool continued, bool autoSelect, unsigned maxIter,
	bool observed )
{
	string autoJson, preamble, autoNote;

	if ( autoSelect )
	{
		Network* net = dynamic_cast< Network* >( modelPtr.get() );

		// The probe summary is captured separately so it can lead the report
		Capture probeCap;
		autoalgo::Result r = autoalgo::pick( *net, 750,
			observed ? &job.cancel : nullptr );
		preamble = probeCap.text.str();

		autoJson = jsonAutoAlgo( r );
		if ( r.selected ) // name the winner where the user actually looks
			autoNote = "auto selected " + r.selectedName + "; ";

		if ( r.winner ) // adopt the winning clone, probe progress kept
		{
			modelPtr = std::move( r.winner );
			dynamic_cast< Iterative* >( modelPtr.get() )
				->setMaxIterations( maxIter );
		}
	}

	if ( !observed )
		return runTrainingAndBuildResult( continued, autoJson, preamble,
			autoNote );

	// Observed (async) run: wire the realtime-chart observer to the model
	//    that will actually train -- after any adoption above
	Network* net = dynamic_cast< Network* >( modelPtr.get() );
	Iterative* iter = dynamic_cast< Iterative* >( modelPtr.get() );
	DataSet& md = modelPtr->getDataSet();
	GuiObserver observer( net, md.testLoaded() ? md.getNumTest() : 0 );
	iter->setObserver( &observer );
	string result = runTrainingAndBuildResult( continued, autoJson, preamble,
		autoNote );
	iter->setObserver( nullptr );
	return result;
}

string handleTrain( const httplib::Request& req )
{
	logAction( req, "train" );
	string algoStr = param( req, "algorithm" ), seed = param( req, "seed" );
	unsigned maxIter = ( unsigned ) atol( param( req, "maxiter" ).c_str() );
	bool async = ( param( req, "async" ) == "1" );

	// algorithm=auto probes all three optimizers from identical weights and
	//    adopts the winner (autoalgo.h) before the real run
	bool autoSelect = ( algoStr == "auto" );
	unsigned algorithm = autoSelect ? 1
		: ( unsigned ) atol( algoStr.c_str() );

	if ( !modelPtr )
		return jsonMsg( false, "create a model first" );
	if ( algorithm < 1 || algorithm > 3 )
		return jsonMsg( false, "algorithm must be 1, 2, 3 or auto" );
	if ( maxIter < 1 )
		return jsonMsg( false, "max iterations must be at least 1" );

	// The CLI's hard constraint: logistic regression is batch/epoch by
	//    definition (logistic.cpp sets it in the constructor and the menu
	//    refuses to turn it off) -- so does the API
	if ( req.has_param( "batch_epoch" ) && param( req, "batch_epoch" ) != "1"
		&& dynamic_cast< Logistic* >( modelPtr.get() ) )
		return jsonMsg( false, "for logistic regression, batch/epoch must be on" );

	// Plateau auto-stop (default off). tol/window fall back to the engine's
	//    own defaults when the fields are absent; validated here rather than
	//    let setAutoStop's asserts fire (asserts vanish in release builds).
	bool autoStop = ( param( req, "autostop" ) == "1" );
	double autoStopTol = 1e-4;
	unsigned autoStopWin = 100;
	if ( autoStop )
	{
		string t = param( req, "autostop_tol" ), w = param( req, "autostop_window" );
		if ( !t.empty() ) autoStopTol = atof( t.c_str() );
		if ( !w.empty() ) autoStopWin = ( unsigned ) atol( w.c_str() );
		if ( !( autoStopTol > 0 && autoStopTol < 1 ) )
			return jsonMsg( false, "autostop_tol must be between 0 and 1" );
		if ( autoStopWin < 2 )
			return jsonMsg( false, "autostop_window must be at least 2" );
	}

	// --- Additional training controls (GUI/CLI parity, 2026-07-19) --------
	//    Each is applied (below, once net/iter exist) ONLY when its field is
	//    present in the request, so a call that omits it keeps the model's
	//    current setting -- both the existing smoke train and "continue with
	//    new parameters" rely on that. Validate here; setter asserts vanish in
	//    release builds. A present-but-empty stopping-condition field disables
	//    that condition; a present value enables and sets it.
	bool haveEta = req.has_param( "eta" ) && !param( req, "eta" ).empty();
	double etaVal = haveEta ? atof( param( req, "eta" ).c_str() ) : 0;
	if ( haveEta && !( etaVal > 0 && etaVal <= 1 ) )
		return jsonMsg( false, "learning rate must be greater than 0 and at most 1" );

	bool wdOn = req.has_param( "weight_decay" ) && param( req, "weight_decay" ) == "1";
	double decayVal = 0;
	if ( wdOn )
	{
		// A blank lambda falls back to the engine's own default (Network ctor:
		//    5e-5 -- a deliberately weak L2 prior; see the ctor for the sigma
		//    relationship), so turning weight decay on never demands a value the
		//    user may not have a feel for
		string d = param( req, "decay" );
		decayVal = d.empty() ? 5e-5 : atof( d.c_str() );
		if ( !( decayVal >= 0 && decayVal <= 1 ) )
			return jsonMsg( false, "weight decay lambda must be between 0 and 1" );
	}

	if ( req.has_param( "minerr" ) && !param( req, "minerr" ).empty()
		&& !( atof( param( req, "minerr" ).c_str() ) > 0
			&& atof( param( req, "minerr" ).c_str() ) < 1 ) )
		return jsonMsg( false, "lower error limit must be between 0 and 1" );
	if ( req.has_param( "change" ) && !param( req, "change" ).empty()
		&& !( atof( param( req, "change" ).c_str() ) > 0
			&& atof( param( req, "change" ).c_str() ) < 1 ) )
		return jsonMsg( false, "change-in-error limit must be between 0 and 1" );
	if ( req.has_param( "errwindow" ) && !param( req, "errwindow" ).empty()
		&& atol( param( req, "errwindow" ).c_str() ) <= 1 )
		return jsonMsg( false, "error window must be greater than 1" );
	if ( req.has_param( "gradmax" ) && !param( req, "gradmax" ).empty()
		&& atof( param( req, "gradmax" ).c_str() ) <= 0 )
		return jsonMsg( false, "maximum absolute gradient limit must be positive" );
	if ( req.has_param( "printcount" ) && !param( req, "printcount" ).empty()
		&& atol( param( req, "printcount" ).c_str() ) < 1 )
		return jsonMsg( false, "print count must be at least 1" );

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
	iter->setAutoStop( autoStop, autoStopTol, autoStopWin );

	// Apply the parity controls validated above -- present-only, so anything
	//    the request omits keeps the model's current value (persists across
	//    Train calls, which is how "stop, then continue with new parameters"
	//    works). These settings are carried into the autoalgo probe clones and
	//    the adopted winner by Network::copy / Iterative::copy.
	if ( req.has_param( "batch_epoch" ) )
		net->setBatchEpoch( param( req, "batch_epoch" ) == "1" );
	if ( req.has_param( "autostep" ) )
	{
		bool a = ( param( req, "autostep" ) == "1" );
		net->setAutoStepSize( a );
		if ( a ) net->setBatchEpoch( true ); // auto step size requires batch/epoch
	}
	if ( haveEta && !( req.has_param( "autostep" ) && param( req, "autostep" ) == "1" ) )
		net->setEta( etaVal ); // manual rate only when not on automatic
	if ( req.has_param( "weight_decay" ) )
	{
		net->setWeightDecay( wdOn );
		net->setDecay( wdOn ? decayVal : 0 );
	}
	if ( req.has_param( "logprint" ) )
		iter->setLogPrint( param( req, "logprint" ) == "1" );
	// Applied whenever present (it only matters when the counter is linear),
	//    so a curl caller can set it without also sending logprint
	if ( req.has_param( "printcount" ) && !param( req, "printcount" ).empty() )
		iter->setPrintCount( ( unsigned ) atol( param( req, "printcount" ).c_str() ) );
	// Stopping conditions: present+value enables & sets, present+empty disables.
	if ( req.has_param( "minerr" ) )
	{
		string s = param( req, "minerr" );
		if ( s.empty() ) iter->setMinStop( false );
		else { iter->setMinStop( true ); iter->setMinError( atof( s.c_str() ) ); }
	}
	if ( req.has_param( "change" ) )
	{
		string s = param( req, "change" );
		if ( s.empty() ) iter->setChangeStop( false );
		else { iter->setChangeStop( true ); iter->setChange( atof( s.c_str() ) ); }
	}
	if ( req.has_param( "errwindow" ) )
	{
		string s = param( req, "errwindow" );
		if ( s.empty() ) iter->setWindowStop( false );
		else { iter->setWindowStop( true ); iter->setWindow( ( unsigned ) atol( s.c_str() ) ); }
	}
	if ( req.has_param( "gradmax" ) )
	{
		string s = param( req, "gradmax" );
		if ( s.empty() ) iter->setGradStop( false );
		else { iter->setGradStop( true ); iter->setGradMaxLimit( atof( s.c_str() ) ); }
	}

	if ( !autoSelect ) // auto: each probe sets its own; the winner's sticks
		net->setTrainingType( algorithm - 1 );

	if ( !async ) // the original blocking contract, unchanged
		return runTrainJob( continued, autoSelect, maxIter, false );

	// Async: hand the engine to a worker thread and return at once. The
	//    caller holds engineMutex here; job.running (set before the worker
	//    exists) is what keeps every other handler out from now on.
	if ( job.worker.joinable() )
		job.worker.join(); // reap the previous, finished run's thread

	{
		lock_guard< mutex > lock( job.progressMutex );
		job.resetForNewRun(); // a plain train reports no OBD/CV/stepwise progress
	}
	job.cancel = false;
	job.running = true;

	// The worker re-derives every engine pointer itself: auto selection may
	//    REPLACE modelPtr with the winning clone, so nothing dereferencable
	//    is captured here. Its screen starts at cout (thread_local); the
	//    Captures inside runTrainJob redirect it for the run.
	job.worker = thread( [ continued, autoSelect, maxIter ]
	{
		string result = runTrainJob( continued, autoSelect, maxIter, true );

		{
			lock_guard< mutex > lock( job.progressMutex );
			job.result = result; // publish BEFORE running goes false
		}
		job.running = false;
	} );

	return jsonMsg( true, autoSelect
		? "probing algorithms, then training" : "training started" );
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
	out << "]}";
	// OBD runs report which phase (grow/prune) and hidden count they are on;
	//    absent for a plain training run
	if ( !job.obdPhase.empty() )
		out << ",\"obd\":{\"phase\":\"" << jsonEscape( job.obdPhase )
			<< "\",\"hidden\":" << job.obdHidden << "}";
	// Stepwise runs report their candidate accounting; absent for every other
	//    kind of job, so its presence identifies the run
	if ( !job.stepwise.empty() )
		out << ",\"stepwise\":" << job.stepwise;
	// Cross-validation runs report the repetition grid: which stage, which
	//    procedure of how many, which outer fold of k, and -- when a nested
	//    architecture search is running inside the current fold -- its phase and
	//    hidden-node trial. Absent for every other kind of job. "inner" is present
	//    only while a nested search is actually reporting, so its absence means
	//    "no nested search here", never "a search at hidden 0".
	if ( !job.cvStage.empty() )
	{
		out << ",\"cv\":{\"stage\":\"" << jsonEscape( job.cvStage ) << "\""
			<< ",\"procedure\":\"" << jsonEscape( job.cvProcedure ) << "\""
			<< ",\"procedureIndex\":" << job.cvProcIndex
			<< ",\"procedureCount\":" << job.cvProcCount
			<< ",\"completedProcedures\":" << job.cvProcDone
			<< ",\"fold\":" << job.cvFold
			<< ",\"folds\":" << job.cvK
			<< ",\"completedFolds\":" << job.cvFoldsDone;
		if ( !job.cvInnerPhase.empty() )
			out << ",\"inner\":{\"phase\":\"" << jsonEscape( job.cvInnerPhase )
				<< "\",\"hidden\":" << job.cvInnerHidden << "}";
		out << "}";
	}
	out << ",\"result\":" << ( job.result.empty() ? "null" : job.result )
		<< "}";
	return out.str();
}

string handleTrainStop()
{
	logActionLine( "stop" );
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

// --- OBD hidden-layer sizing (ROADMAP 2 Phase 4) --------------------------

string jsonObdHistory( const vector< obd::SizeTrial >& h )
{
	ostringstream out;
	out.precision( 6 );
	out << "[";
	for ( unsigned i = 0; i < h.size(); i++ )
	{
		const obd::SizeTrial& t = h[ i ];
		out << ( i ? "," : "" ) << "{\"hidden\":" << t.hidden
			<< ",\"trainErr\":" << jnum( t.trainErr )
			<< ",\"testErr\":" << jnum( t.testErr )
			<< ",\"trainCA\":" << ( t.trainCA < 0 ? "null" : jnum( t.trainCA ) )
			<< ",\"testCA\":" << ( t.testCA < 0 ? "null" : jnum( t.testCA ) )
			<< ",\"phase\":\"" << ( t.phaseGrow ? "grow" : "prune" )
			<< "\",\"stop\":\"" << stopReasonName( t.stop ) << "\"}";
	}
	out << "]";
	return out.str();
}

// The whole OBD search on the worker thread: run the driver (capturing its
//    size table + final report), adopt the winning net as modelPtr (like the
//    auto-algorithm adoption -- it REPLACES modelPtr, so every pointer is
//    re-derived after), then build the result JSON (history + the winner's ROC
//    and stats). The caller must own the engine (the async worker does).
string runObdJob( const obd::Config& cfg )
{
	Capture cap; // the driver's size table + final report

	// Feed the realtime chart and the status poll's obd phase/hidden. Each
	//    size trial's train() restarts its iteration counter at 0, so the raw
	//    iteration is NON-monotonic across the search -- the chart's x axis
	//    assumes monotonic growth, so plot cumulative iterations instead (a
	//    reset marks a size boundary; lastIter/offset live for the whole
	//    synchronous obd::run call below).
	unsigned lastIter = 0, iterOffset = 0;
	obd::ProgressFn progress = [ &lastIter, &iterOffset ]( const char* phase,
		unsigned hidden, unsigned iteration, double trainErr, double testErr )
	{
		{
			lock_guard< mutex > lock( job.progressMutex );
			job.obdPhase = phase;
			job.obdHidden = hidden;
		}
		// A phase can be announced with NO measurement (the optimizer probe, which
		//    reports before anything has been fitted and passes -1 for both errors).
		//    That is a status, not a sample: plotting it would put a -1 error on the
		//    chart. The phase above is still published, so the status line moves.
		if ( trainErr < 0 )
			return;
		if ( iteration < lastIter ) // a new size's training began
			iterOffset += lastIter;
		lastIter = iteration;
		job.pushSample( iterOffset + iteration, trainErr, testErr );
	};

	obd::Result r = obd::run( *dataPtr, cfg, progress, &job.cancel );

	// Every trial's stop reason and eligibility, machine-readable -- present on
	//    a refusal too, which is when a caller most needs to see WHY nothing was
	//    selected. Emitted from the same records the printed table renders.
	ostringstream trials;
	trials << "[";
	for ( unsigned i = 0; i < r.history.size(); i++ )
	{
		const obd::SizeTrial& t = r.history[ i ];
		trials << ( i ? "," : "" )
			<< "{\"phase\":\"" << ( t.phaseGrow ? "grow" : "prune" )
			<< "\",\"hidden\":" << t.hidden
			<< ",\"iterations\":" << t.iterations
			<< ",\"stopReason\":\"" << obd::stopToken( t.stop, t.eligibility )
			<< "\",\"eligible\":" << ( t.eligibility == obd::ELIGIBLE ? "true" : "false" )
			<< "}";
	}
	trials << "]";

	if ( !r.ok )
		return string( "{\"ok\":false,\"message\":\"" )
			+ jsonEscape( r.message.empty()
				? ( r.cancelled ? "cancelled" : "OBD did not produce a model" )
				: r.message )
			+ "\",\"cancelled\":" + ( r.cancelled ? "true" : "false" )
			// A ceiling refusal is its own outcome: the search ran correctly and
			//    declined to compare unfinished fits. Callers must not read it as
			//    a crash, and must not read it as a selection either.
			+ ",\"ceilingExhausted\":" + ( r.ceilingExhausted ? "true" : "false" )
			+ ",\"trials\":" + trials.str()
			+ ",\"output\":\"" + jsonEscape( cap.text.str() ) + "\"}";

	// Adopt the winner (probe/search progress kept); it is already trained
	modelPtr = std::move( r.winner );
	dfaPtr.reset(); // any prior DFA guesses belong to the old model
	lastTrainError = 0; // the winner has weights -- treat it as trained
	lastReport = cap.text.str();

	DataSet& md = modelPtr->getDataSet();
	string roc;
	if ( md.getDiscrete() && md.getOutput() == 1 && md.trainLoaded() )
	{
		roc = ",\"roc\":{\"train\":" + jsonROCSeries( md.getTrainTwoSet() );
		if ( md.testLoaded() )
			roc += ",\"test\":" + jsonROCSeries( md.getTestTwoSet() );
		roc += "}";
	}
	string stats = jsonStats();
	string statsField = stats.empty() ? "" : ( ",\"stats\":" + stats );

	ostringstream msg;
	msg << "OBD selected " << r.selectedHidden << " hidden node"
		<< ( r.selectedHidden == 1 ? "" : "s" );
	if ( r.cancelled )
		msg << " (stopped by request)";

	return string( "{\"ok\":true,\"message\":\"" ) + jsonEscape( msg.str() )
		+ "\",\"cancelled\":" + ( r.cancelled ? "true" : "false" )
		+ ",\"selectedHidden\":" + to_string( r.selectedHidden )
		+ ",\"optimizer\":\"" + ( r.algorithm == 0 ? "Canonical"
			: r.algorithm == 1 ? "CGD" : r.algorithm == 2 ? "Shanno" : "unknown" )
		+ "\",\"optimizerAuto\":" + ( r.autoSelected ? "true" : "false" )
		// The held-out set the search actually scored on, from the engine's rule
		//    -- the size-search chart's legend is drawn from this, so it can never
		//    call a validation-guided selection a test-set one.
		+ ",\"monitor\":\"" + dataPtr->monitorSetName() + "\""
		+ ",\"obd\":{\"selectedHidden\":" + to_string( r.selectedHidden )
		+ ",\"trials\":" + trials.str()
		+ ",\"history\":" + jsonObdHistory( r.history ) + "}"
		+ ",\"output\":\"" + jsonEscape( lastReport ) + "\"" + roc + statsField + "}";
}

string handleObd( const httplib::Request& req )
{
	logAction( req, "obd" );

	if ( !dataPtr )
		return jsonMsg( false, "load a dataset first" );
	// The driver refuses these too (defence in depth), but a synchronous refusal
	//    is a better experience than one that only surfaces via the status poll
	if ( !( dataPtr->getDiscrete() && dataPtr->getOutput() == 1 ) )
		return jsonMsg( false, "OBD needs a discrete, single-output dataset" );
	if ( !dataPtr->testLoaded() )
		return jsonMsg( false, "OBD needs a held-out test set -- it is the "
			"validation signal early stopping watches" );

	obd::Config cfg;

	// Each field overrides its default only when present; validate here (the
	//    engine's own asserts vanish in release builds)
	auto uintParam = [ & ]( const char* name, unsigned& dst, unsigned lo,
		const char* err, string& bad )
	{
		string s = param( req, name );
		if ( s.empty() ) return;
		long v = atol( s.c_str() );
		if ( v < ( long ) lo ) { bad = err; return; }
		dst = ( unsigned ) v;
	};
	auto fracParam = [ & ]( const char* name, double& dst, const char* err, string& bad )
	{
		string s = param( req, name );
		if ( s.empty() ) return;
		double v = atof( s.c_str() );
		if ( !( v > 0 && v < 1 ) ) { bad = err; return; }
		dst = v;
	};

	string bad;
	uintParam( "hidden_start", cfg.hStart, 1, "hidden_start must be at least 1", bad );
	uintParam( "hidden_max", cfg.hMax, 1, "hidden_max must be at least 1", bad );
	uintParam( "iter_budget", cfg.iterBudget, 1, "iter_budget must be at least 1", bad );
	uintParam( "sample_every", cfg.sampleEvery, 1, "sample_every must be at least 1", bad );
	uintParam( "early_stop_patience", cfg.earlyStopPatience, 1,
		"early_stop_patience must be at least 1", bad );
	uintParam( "grow_patience", cfg.growPatience, 1,
		"grow_patience must be at least 1", bad );
	fracParam( "early_stop_tol", cfg.earlyStopTol,
		"early_stop_tol must be between 0 and 1", bad );
	fracParam( "prune_tol", cfg.pruneTol, "prune_tol must be between 0 and 1", bad );
	// The per-size train-plateau backstop, same names as /api/train's
	fracParam( "autostop_tol", cfg.plateauTol,
		"autostop_tol must be between 0 and 1", bad );
	uintParam( "autostop_window", cfg.plateauWindow, 2,
		"autostop_window must be at least 2", bad );
	if ( !bad.empty() )
		return jsonMsg( false, bad );
	if ( cfg.hMax < cfg.hStart )
		return jsonMsg( false, "hidden_max must be at least hidden_start" );

	// algorithm: 1|2|3 or auto (probe once)
	string algoStr = param( req, "algorithm" );
	if ( algoStr == "auto" ) cfg.algorithm = -1;
	else if ( algoStr.empty() || algoStr == "1" ) cfg.algorithm = 0;
	else if ( algoStr == "2" ) cfg.algorithm = 1;
	else if ( algoStr == "3" ) cfg.algorithm = 2;
	else return jsonMsg( false, "algorithm must be 1, 2, 3 or auto" );

	string seed = param( req, "seed" );
	if ( !seed.empty() )
		util::set_seed( ( unsigned ) atol( seed.c_str() ) );

	// OBD is async-only (it is many training runs): hand it to the worker and
	//    return at once, exactly like async training. The caller holds
	//    engineMutex here; job.running (set before the worker) gates everyone else.
	if ( job.worker.joinable() )
		job.worker.join(); // reap the previous, finished run's thread

	{
		lock_guard< mutex > lock( job.progressMutex );
		job.resetForNewRun();
	}
	job.cancel = false;
	job.running = true;

	job.worker = thread( [ cfg ]
	{
		string result = runObdJob( cfg );
		{
			lock_guard< mutex > lock( job.progressMutex );
			job.result = result; // publish BEFORE running goes false
		}
		job.running = false;
	} );

	return jsonMsg( true, "OBD hidden-layer search started" );
}

// --- Cross-validation model comparison (ROADMAP 4 Phase 4b-CV) ---------------

// Configuration a CV run needs, parsed from the request in handleCv and carried
//    into the worker (so the worker touches only its own copy, not the request).
struct CvConfig
{
	unsigned k = 5, seed = 42, maxIter = 500;
	bool logistic = true, ldfa = false, qdfa = false, neural = true;
	bool neuralObd = true;        // nested OBD per fold, else a fixed hidden count
	unsigned neuralHidden = 5;    // fixed-architecture neural (when !neuralObd)
	double innerVal = 0.25;       // inner validation fraction for nested OBD
	obd::Config obd;              // per-fold OBD search (when neuralObd)

	// The plateau auto-stop applies to the PLAIN logistic / fixed-architecture
	//    neural templates too, not only to the nested-OBD search, when the caller
	//    asks for it. Without this the only stopping rule those procedures have
	//    is the shipped gradient limit, which a slowly-converging problem may
	//    never reach -- and since an unconverged fold is now (correctly) refused,
	//    the caller would have no way to make a fixed-architecture CV succeed.
	//    Default OFF, so a request that does not ask is behaviorally unchanged.
	bool plainAutostop = false;

	// Nested OBD defaults to AUTO: `auto` is a procedure for choosing an
	//    optimizer, run once per fold on that fold's inner training data alone
	//    and once more for the locked-development refit. The engine struct's own
	//    default is canonical (standalone OBD's default), so CV states its
	//    choice here explicitly rather than inheriting a silent one.
	CvConfig() { obd.algorithm = -1; }
	// Locked-test evaluation (ROADMAP 4 Phase 4). lockedN > 0 sets aside an
	//    outcome-stratified ROW holdout held OUT of CV; each procedure is refit on
	//    the development rows by its own rule and scored once on it. Inference is NOT
	//    automatic -- it runs only when `independence` declares a sampling unit the
	//    achieved partition method permits (a row holdout does not by itself
	//    establish independence, and a grouped design does not permit ordinary
	//    DeLong at all).
	unsigned lockedN = 0;         // locked-test size (0 = pure CV, no locked test)
	string primary, reference;    // the prespecified contrast (internal names)
	// The declared sampling unit (DLG-1), as a TYPE rather than a magic string
	//    (DLG-8). Inference is produced ONLY when the declared unit and the achieved
	//    partition method permit it -- evaldesign::chooseInference is the one place
	//    that decides, so a "cluster" declaration can never fall back to ordinary
	//    DeLong by a caller forgetting a branch.
	evaldesign::SamplingUnit unit = evaldesign::SamplingUnit::Unspecified;

	// Fold policy (ROADMAP 4). 0-based input-node positions, parsed from the CV
	//    request's OWN strata= / group= -- a CV request is self-contained and
	//    never inherits /api/load's split configuration, which answers a
	//    different question (one holdout, not a fold plan) and would silently
	//    change what a CV run means when the dataset panel was last used.
	//    Empty = the shipped outcome-stratified behavior, unchanged.
	vector< unsigned > strataColumns;
	unsigned strataBins = 4;
	vector< unsigned > groupColumns;

	// Which planner the two lists select. Derived once in handleCv so the job and
	//    every report read the same answer rather than re-testing emptiness.
	evaldesign::PartitionMethod foldMethod
		= evaldesign::PartitionMethod::OutcomeStratified;

	// The user's 1-based column numbers, for the report -- the form they typed.
	vector< unsigned > userStrata, userGroup;
};

// The whole comparison on the worker thread: build an outcome-stratified k-fold
//    plan over the loaded Raw, assemble the selected procedures, run them over
//    the ONE shared plan (crossval::compare), render the three-tier report, and
//    write the Tier-3 files beside the data. Does NOT touch modelPtr -- CV is a
//    standalone analysis, like DFA. The caller must own the engine.
// Assemble the report's LockedInfo from a locked-test evaluation: the point areas
//    over the procedures that fitted (paired on the same rows), each column's 95% CI
//    from whichever estimator the design permits, and the prespecified contrast
//    AUC(primary) - AUC(reference) when both are present AND fitted. The estimator is
//    chosen ONCE, by evaldesign::chooseInference, from the declared sampling unit and
//    the ACHIEVED partition method -- never by the caller and never by which answer
//    looks better.
static cvreport::LockedInfo buildLockedInfo( const crossval::LockedResult& lr,
	const CvConfig& c, const evaldesign::Partition& split,
	const vector< unsigned >& rowGroup, unsigned nGroups )
{
	cvreport::LockedInfo L;
	L.has = true;
	L.n = ( unsigned ) lr.testRows.size();
	L.events = 0;
	for ( unsigned i = 0; i < lr.outcome.size(); i++ ) L.events += lr.outcome[ i ];
	L.testRows = lr.testRows;
	L.outcome = lr.outcome;
	L.split = split;
	L.unit = c.unit;

	// Cluster identity, gathered by RAW ROW so it can never be reconstructed from
	//    a position after materialization. Empty rowGroup = an ungrouped design,
	//    and the empty cluster vector says exactly that (absence is meaningful).
	//    nClusters counts the groups PRESENT IN THE LOCKED SAMPLE -- the number of
	//    independent units a clustered estimator would actually have.
	if ( !rowGroup.empty() && nGroups )
	{
		L.cluster.resize( lr.testRows.size() );
		set< unsigned > present;
		for ( unsigned i = 0; i < lr.testRows.size(); i++ )
		{
			L.cluster[ i ] = rowGroup[ lr.testRows[ i ] ];
			present.insert( L.cluster[ i ] );
		}
		L.nClusters = ( unsigned ) present.size();
	}

	// Point AUCs (and, if the design permits inference, the covariance) over the
	//    entries that produced predictions -- paired on the same rows.
	vector< vector< double > > preds;
	vector< int > idx( lr.entries.size(), -1 ); // entry -> column in the estimator
	for ( unsigned e = 0; e < lr.entries.size(); e++ )
		if ( lr.entries[ e ].ok )
		{
			idx[ e ] = ( int ) preds.size();
			preds.push_back( lr.entries[ e ].pred );
		}

	// The POINT areas come from the shared placement code, so they exist whenever
	//    both outcome classes are present -- independent of whether any covariance
	//    estimator is permitted or estimable (the DLG-4 principle: a descriptive
	//    number is not withheld because an inferential one is unavailable).
	auccov::Placements place = auccov::compute( lr.outcome, preds );

	// Which estimator the design permits, and whether the data can support it.
	//    Both estimators are asked ONLY when their own unit was declared -- an
	//    undeclared run does no covariance work at all.
	delong::Result dr;
	clustered_auc::Result cr;
	bool estimable = false;
	string estimableReason;
	const bool clusterUnit = ( c.unit == evaldesign::SamplingUnit::Cluster );
	if ( clusterUnit && !L.cluster.empty() )
	{
		cr = clustered_auc::analyze( lr.outcome, L.cluster, preds );
		estimable = cr.ok; estimableReason = cr.reason;
	}
	else if ( c.unit == evaldesign::SamplingUnit::Row )
	{
		dr = delong::analyze( lr.outcome, preds );
		estimable = dr.ok; estimableReason = dr.reason;
	}

	// ONE decision (DLG-8): the declared sampling unit plus the ACHIEVED partition
	//    method decide which estimator may run. A mechanically generated row holdout
	//    is not independent unless the caller says the observations are; a
	//    group-disjoint design forbids ordinary DeLong outright; a clustered unit
	//    needs a group-aware design to have anything to cluster on.
	evaldesign::InferenceChoice choice = evaldesign::chooseInference( c.unit,
		split.method, estimable, estimableReason );
	L.inference = choice.method;
	L.inferenceReason = choice.reason;
	const bool inferenceRan = L.inferenceRan();
	const bool clustered =
		( L.inference == evaldesign::AucInference::ObuchowskiClustered );

	for ( unsigned e = 0; e < lr.entries.size(); e++ )
	{
		cvreport::LockedColumn col;
		col.name = lr.entries[ e ].name;
		col.pred = lr.entries[ e ].pred; // written regardless (DLG-4)
		// The locked refit's own choices: architecture AND the optimizer it ran
		//    on (its own independent Auto probe, when Auto was requested).
		if ( !lr.entries[ e ].selections.empty() )
		{
			const crossval::FoldSelection& s = lr.entries[ e ].selections[ 0 ];
			col.arch = to_string( s.hidden ) + " hidden";
			col.arch += string( ", " ) + ( s.algorithm == 0 ? "Canonical"
				: s.algorithm == 1 ? "CGD" : s.algorithm == 2 ? "Shanno" : "?" );
			if ( s.autoSelected ) col.arch += " (auto)";
		}
		if ( !lr.entries[ e ].ok )
			col.note = "failed: " + lr.entries[ e ].reason;
		else if ( place.ok )
		{
			// The point AUC is available whether or not inference was declared or
			//    estimable; the 95% interval only when an estimator actually ran,
			//    and it comes from THAT estimator (clustered intervals are not
			//    DeLong intervals and are never labelled as such).
			col.hasAuc = true;
			col.auc = place.auc[ idx[ e ] ];
			if ( inferenceRan )
			{
				auccov::Interval iv = clustered
					? clustered_auc::interval( cr, ( unsigned ) idx[ e ] )
					: delong::interval( dr, ( unsigned ) idx[ e ] );
				col.hasCi = true; col.lo = iv.lo; col.hi = iv.hi;
			}
		}
		else col.note = "AUC not computable: " + place.reason;
		L.columns.push_back( col );
	}

	// The prespecified contrast (both procedures must be present AND have fitted).
	if ( !c.primary.empty() && !c.reference.empty() )
	{
		cvreport::LockedContrast& ct = L.contrast;
		ct.primary = c.primary; ct.reference = c.reference;
		int pe = -1, re = -1;
		for ( unsigned e = 0; e < lr.entries.size(); e++ )
		{
			if ( lr.entries[ e ].name == c.primary ) pe = ( int ) e;
			if ( lr.entries[ e ].name == c.reference ) re = ( int ) e;
		}
		if ( pe < 0 || re < 0 )
			ct.note = "a contrast procedure was not evaluated";
		else if ( !place.ok || idx[ pe ] < 0 || idx[ re ] < 0 )
			ct.note = "a contrast procedure has no point AUC on the locked test";
		else
		{
			// The point difference is always available; the p only when the design
			//    permitted an estimator AND that estimator produced a valid contrast.
			ct.hasDelta = true;
			ct.delta = place.auc[ idx[ pe ] ] - place.auc[ idx[ re ] ];
			if ( inferenceRan )
			{
				auccov::Contrast dc = clustered
					? clustered_auc::contrast( cr, ( unsigned ) idx[ pe ],
						( unsigned ) idx[ re ] )
					: delong::contrast( dr, ( unsigned ) idx[ pe ],
						( unsigned ) idx[ re ] );
				if ( !dc.valid )
					ct.note = dc.note.empty()
						? evaldesign::inferenceShortName( L.inference )
							+ " contrast not computable"
						: dc.note;
				else
				{
					ct.hasInference = true;
					ct.delta = dc.delta; ct.p = dc.p;
					ct.degenerate = dc.degenerate; ct.separated = dc.separated;
					ct.significant = ( dc.separated || ( !dc.degenerate && dc.p < 0.05 ) );
				}
			}
			else
				ct.note = choice.reason; // the SAME reason the design layer gave
		}
	}
	return L;
}

// A compact JSON summary of the locked-test inference for machine consumers (the
//    human path reads the rendered tier1/tier2). Numbers use the signed area scale.
static string lockedJson( const cvreport::LockedInfo& L )
{
	ostringstream j;
	j << "{\"n\":" << L.n << ",\"events\":" << L.events
		<< ",\"samplingUnit\":\"" << jsonEscape( L.samplingUnitText() ) << "\""
		<< ",\"independenceStatus\":\"" << jsonEscape( L.independenceText() ) << "\""
		<< ",\"inferenceMethod\":\"" << jsonEscape( L.inferenceText() ) << "\""
		<< ",\"inferenceRan\":" << ( L.inferenceRan() ? "true" : "false" )
		// Why no estimator ran, and how many independent units the design has.
		//    A page can now SAY the reason instead of guessing it from the method
		//    string, and 0 clusters means an ungrouped design, not "unknown".
		<< ",\"inferenceReason\":\"" << jsonEscape( L.inferenceReason ) << "\""
		<< ",\"clusters\":" << L.nClusters
		<< ",\"splitMethod\":\""
		<< jsonEscape( evaldesign::partitionMethodName( L.split.method ) ) << "\""
		<< ",\"splitPlan\":\"" << jsonEscape( L.splitPlanText() ) << "\""
		<< ",\"areas\":[";
	for ( unsigned i = 0; i < L.columns.size(); i++ )
	{
		const cvreport::LockedColumn& col = L.columns[ i ];
		j << ( i ? "," : "" ) << "{\"name\":\"" << jsonEscape( col.name ) << "\"";
		// the point AUC appears whenever it is estimable; the CI only with inference
		if ( col.hasAuc ) j << ",\"auc\":" << col.auc;
		else j << ",\"auc\":null";
		if ( col.hasCi ) j << ",\"lo\":" << col.lo << ",\"hi\":" << col.hi;
		else j << ",\"lo\":null,\"hi\":null";
		if ( !col.note.empty() ) j << ",\"note\":\"" << jsonEscape( col.note ) << "\"";
		j << "}";
	}
	j << "]";
	if ( L.contrast.hasDelta )
		j << ",\"contrast\":{\"primary\":\"" << jsonEscape( L.contrast.primary )
			<< "\",\"reference\":\"" << jsonEscape( L.contrast.reference )
			<< "\",\"delta\":" << L.contrast.delta
			<< ",\"inferenceRan\":" << ( L.contrast.hasInference ? "true" : "false" )
			<< ",\"p\":" << ( ( L.contrast.hasInference && !L.contrast.degenerate )
				? to_string( L.contrast.p ) : string( "null" ) )
			<< ",\"significant\":" << ( ( L.contrast.hasInference && L.contrast.significant )
				? "true" : "false" )
			<< ",\"degenerate\":" << ( L.contrast.degenerate ? "true" : "false" )
			<< ",\"separated\":" << ( L.contrast.separated ? "true" : "false" )
			<< ",\"note\":\"" << jsonEscape( L.contrast.note ) << "\"}";
	else if ( !L.contrast.note.empty() )
		j << ",\"contrast\":{\"note\":\"" << jsonEscape( L.contrast.note ) << "\"}";
	else
		j << ",\"contrast\":null";
	j << "}";
	return j.str();
}

string runCvJob( CvConfig c )
{
	DataSet& data = *dataPtr;

	// Outcome labels + event count over the WHOLE raw dataset.
	Matrix< double >& raw = data.getRawMatrix();
	unsigned n = raw.rows(), outCol = raw.cols() - 1;
	vector< unsigned > label( n );
	unsigned events = 0;
	for ( unsigned r = 0; r < n; r++ )
	{
		label[ r ] = ( raw( r, outCol ) != 0 ) ? 1u : 0u;
		events += label[ r ];
	}

	// Optional locked-test split (ROADMAP 4 Phase 4). When set, an outcome-
	//    stratified ROW holdout is held ENTIRELY out of CV; CV then folds only the
	//    development rows, and each procedure is later refit on the development rows
	//    and scored once on the locked rows (evaluateOnce). DeLong runs only if the
	//    caller declared an independent-row sampling unit. devRows / lockedRows are
	//    raw row indices into the original dataset.
	bool locked = ( c.lockedN > 0 );
	vector< unsigned > devRows, lockedRows;
	DataSet devData;              // built (dev-only Raw) only when locked
	DataSet* cvData = &data;      // the dataset CV folds

	// The structured description of the locked split (DLG-8). It is filled by the
	//    code that PERFORMS the split, so the report cannot describe a design that
	//    was not run; a group-aware locked holdout will set method/groupColumns
	//    here and everything downstream follows without a second decision.
	evaldesign::Partition lockedSplit;
	lockedSplit.method = evaldesign::PartitionMethod::OutcomeStratified;
	lockedSplit.seed = c.seed;

	// Per-RAW-ROW group identity. Built ONCE, over the whole raw dataset, because
	//    the same key must govern both the locked holdout and the outer folds --
	//    two independently built keys would be two different definitions of a
	//    cluster, and the leakage they were introduced to prevent would return.
	//    Empty means an ungrouped request; absence is meaningful downstream.
	const bool grouped =
		( c.foldMethod == evaldesign::PartitionMethod::StratifiedGroup );
	vector< unsigned > rowGroup;
	unsigned nGroups = 0;
	if ( grouped )
	{
		rowGroup = data.groupKey( c.groupColumns );
		set< unsigned > gs( rowGroup.begin(), rowGroup.end() );
		nGroups = ( unsigned ) gs.size();
	}

	if ( locked )
	{
		util::set_seed( c.seed );
		if ( grouped )
		{
			// Whole groups on one side. Groups are indivisible, so the achieved
			//    size only approximates the request -- both are reported.
			nsplit::GroupHoldout h = nsplit::groupHoldout( label, rowGroup, c.lockedN );
			devRows = h.train; lockedRows = h.test;
			lockedSplit.method = evaldesign::PartitionMethod::StratifiedGroup;
			lockedSplit.groupColumns = c.userGroup;
			lockedSplit.nGroups = nGroups;
		}
		else
		{
			nsplit::Holdout h = nsplit::stratifiedHoldout( label, c.lockedN );
			devRows = h.train; lockedRows = h.test;
		}
		lockedSplit.nRequested = c.lockedN;
		lockedSplit.nAchieved = ( unsigned ) lockedRows.size();

		// Independent check that the locked partition is well formed and, when
		//    grouped, that no group straddles it. The planner already guarantees
		//    both; this is the pass that would notice if it stopped.
		string bad = nsplit::partitionError( n, devRows, lockedRows, true );
		if ( !bad.empty() )
			return jsonMsg( false, "the locked-test split is not a valid partition: "
				+ bad );
		if ( grouped )
		{
			vector< char > inDev( nGroups, 0 ), inLocked( nGroups, 0 );
			for ( unsigned i = 0; i < devRows.size(); i++ )
				inDev[ rowGroup[ devRows[ i ] ] ] = 1;
			for ( unsigned i = 0; i < lockedRows.size(); i++ )
				inLocked[ rowGroup[ lockedRows[ i ] ] ] = 1;
			unsigned leaked = 0;
			for ( unsigned g = 0; g < nGroups; g++ ) if ( inDev[ g ] && inLocked[ g ] ) leaked++;
			lockedSplit.leakage = leaked;
			if ( leaked )
				return jsonMsg( false, to_string( leaked ) + " group(s) appear in both "
					"the development set and the locked test; the split is refused" );
		}

		Matrix< double > devRaw = raw.includerows( devRows );
		devData = data;                 // copy config (inputs/outputs/discrete)
		devData.setRawMatrix( devRaw );  // dev-only Raw; CV never sees the locked rows
		cvData = &devData;
	}

	// Fold plan over the CV dataset (development rows only when a locked test was
	//    taken). Which planner runs is the request's fold policy, decided once in
	//    handleCv; the outcome-only path is unchanged and byte-identical.
	Matrix< double >& craw = cvData->getRawMatrix();
	unsigned nCv = craw.rows(), cvOut = craw.cols() - 1;
	vector< unsigned > cvLabel( nCv );
	for ( unsigned r = 0; r < nCv; r++ )
		cvLabel[ r ] = ( craw( r, cvOut ) != 0 ) ? 1u : 0u;

	// The fold plan's structured description, filled by the code that builds it.
	evaldesign::Partition foldDesign;
	foldDesign.method = c.foldMethod;
	foldDesign.seed = c.seed;
	foldDesign.k = c.k;
	foldDesign.developmentOnly = locked;
	foldDesign.strataColumns = c.userStrata;
	foldDesign.strataBins = c.strataColumns.empty() ? 0 : c.strataBins;
	foldDesign.groupColumns = c.userGroup;

	// Group ids for the CV rows, GATHERED by raw row -- never rebuilt on the
	//    development subset, which would be a second definition of a cluster.
	vector< unsigned > cvGroup;
	if ( grouped )
	{
		cvGroup.resize( nCv );
		for ( unsigned r = 0; r < nCv; r++ )
			cvGroup[ r ] = rowGroup[ locked ? devRows[ r ] : r ];
	}

	util::set_seed( c.seed );
	vector< unsigned > foldId;
	if ( grouped )
	{
		nsplit::GroupFoldPlan gp = nsplit::stratifiedGroupKFold( cvLabel, cvGroup, c.k );
		if ( !gp.ok ) return jsonMsg( false, "fold plan: " + gp.reason );
		foldId = gp.foldId;
		foldDesign.nGroups = gp.nGroups;
		foldDesign.leakage = gp.leakageCount;
		foldDesign.warnings = gp.warnings;
		foldDesign.imbalanceScore = gp.imbalanceScore;
		foldDesign.largestGroup = gp.largestGroup;
	}
	else if ( !c.strataColumns.empty() )
	{
		// DataSet owns what a stratum key MEANS; the bins are computed on the CV
		//    rows, so a locked test never influences where a bin boundary falls.
		vector< unsigned > stratum = cvData->strataKey( c.strataColumns, c.strataBins );
		nsplit::FoldPlan fp = nsplit::stratifiedKFold( stratum, c.k );
		if ( !fp.ok ) return jsonMsg( false, "fold plan: " + fp.reason );
		foldId = fp.foldId;
		foldDesign.nStrata = fp.nStrata;
		foldDesign.warnings = fp.warnings;
	}
	else
		foldId = nsplit::kFold( cvLabel, c.k );

	// PER-FOLD achieved counts, recomputed FROM THE ASSIGNMENT rather than taken
	//    from whichever planner ran. Two reasons: every planner then reports the
	//    same way (the outcome-only path returns a bare vector and has no counts
	//    to hand over), and a count derived from the finished plan cannot describe
	//    a plan that was not produced. These are the numbers an external audit
	//    needs -- rows, events and clusters per fold -- and the planners' internal
	//    tallies were being computed and then discarded (Sol, 2026-07-30).
	foldDesign.foldRows.assign( c.k, 0 );
	foldDesign.foldEvents.assign( c.k, 0 );
	{
		vector< set< unsigned > > groupsIn( grouped ? c.k : 0 );
		for ( unsigned r = 0; r < nCv; r++ )
		{
			unsigned f = foldId[ r ];
			if ( f >= c.k ) continue; // cannot happen; a bad plan is refused above
			foldDesign.foldRows[ f ]++;
			foldDesign.foldEvents[ f ] += cvLabel[ r ];
			if ( grouped ) groupsIn[ f ].insert( cvGroup[ r ] );
		}
		if ( grouped )
		{
			foldDesign.foldGroups.assign( c.k, 0 );
			for ( unsigned f = 0; f < c.k; f++ )
				foldDesign.foldGroups[ f ] = ( unsigned ) groupsIn[ f ].size();
		}
	}

	// Templates for the network procedures. They MUST outlive compare()/evaluateOnce
	//    -- trainProcedure captures the template by reference -- so they live here
	//    for the whole synchronous run below. Sized on a train=all fold of the CV
	//    dataset, so the locked rows never enter template construction.
	DataSet full = *cvData;
	vector< unsigned > allRows( nCv ), none;
	for ( unsigned r = 0; r < nCv; r++ ) allRows[ r ] = r;
	full.makeFold( allRows, none );

	unique_ptr< Logistic > lg;
	unique_ptr< SimpleProp > sp;
	if ( c.logistic )
	{
		lg = make_unique< Logistic >();
		lg->setDataSet( full );
		lg->setHistory( false ); lg->setLastop( false ); lg->setLogPrint( false );
		// A reachable stopping rule, when the caller asked for one: the clone
		//    each fold trains carries this through Iterative::copy.
		if ( c.plainAutostop )
			lg->setAutoStop( true, c.obd.plateauTol, c.obd.plateauWindow );
		util::set_seed( c.seed ); lg->randomize();
	}
	if ( c.neural && !c.neuralObd )
	{
		sp = make_unique< SimpleProp >();
		sp->setDataSet( full );
		sp->setHidden( c.neuralHidden );
		sp->setHistory( false ); sp->setLastop( false ); sp->setLogPrint( false );
		if ( c.plainAutostop )
			sp->setAutoStop( true, c.obd.plateauTol, c.obd.plateauWindow );
		util::set_seed( c.seed ); sp->randomize();
	}

	// --- Progress plumbing (2026-07-29) ------------------------------------
	//
	// Two independent callbacks, composed HERE rather than in the engine: CV
	//    reports the repetition grid it owns (stage, procedure, fold), and the
	//    nested architecture search reports the trial it owns (phase, hidden
	//    nodes). Neither layer learns about the other -- crossval never hears the
	//    word "OBD", and OBD never hears the word "fold".
	crossval::ProgressFn cvProgress = []( const crossval::Progress& p )
	{
		lock_guard< mutex > lock( job.progressMutex );
		// A change of fold or procedure retires the inner trial: the search that
		//    was at "prune, 7 hidden" belongs to the fold that just ended, and
		//    showing it against the next one would be a lie about live work.
		if ( job.cvFold != p.fold || job.cvProcedure != p.procedure )
		{
			job.cvInnerPhase.clear();
			job.cvInnerHidden = 0;
		}
		job.cvStage = ( p.stage == crossval::Progress::LOCKED_EVALUATION )
			? "locked-test evaluation" : "cross-validation";
		job.cvProcedure = p.procedure;
		job.cvProcIndex = p.procIndex;
		job.cvProcCount = p.procCount;
		job.cvProcDone = p.completedProcedures;
		job.cvFold = p.fold;
		job.cvK = p.k;
		job.cvFoldsDone = p.completedFolds;
	};
	obd::ProgressFn innerProgress = []( const char* phase, unsigned hidden,
		unsigned, double, double )
	{
		lock_guard< mutex > lock( job.progressMutex );
		job.cvInnerPhase = phase;
		job.cvInnerHidden = hidden;
	};

	// Assemble the procedure specs given an arch sink, so CV and the locked-test
	//    evaluation each get their OWN sink (CV's records a size per fold; the
	//    locked one records the single frozen fit's size).
	//    innerGroup is the per-raw-row group identity for THE DATASET THAT PROCEDURE
	//    WILL RECEIVE -- cvData's rows for the CV pass, the full data's rows for the
	//    locked-development refit. They are different index spaces, so each pass
	//    passes its own; handing the wrong one over would silently mis-group.
	auto assemble = [ & ]( vector< crossval::FoldSelection >* archSink,
		const vector< unsigned >& innerGroup )
	{
		vector< crossval::ProcedureSpec > ps;
		if ( c.logistic )
			ps.push_back( { "Logistic", cvadapters::trainProcedure( *lg, c.maxIter ), nullptr } );
		if ( c.ldfa )
			ps.push_back( { "LDFA", cvadapters::dfaProcedure( false ), nullptr } );
		if ( c.qdfa )
			ps.push_back( { "QDFA", cvadapters::dfaProcedure( true ), nullptr } );
		if ( c.neural )
		{
			if ( c.neuralObd )
				ps.push_back( { "Neural (OBD)",
					cvadapters::nestedObdProcedure( c.obd, c.innerVal, archSink,
						innerProgress, innerGroup ), archSink } );
			else
				ps.push_back( { "Neural", cvadapters::trainProcedure( *sp, c.maxIter ), nullptr } );
		}
		return ps;
	};

	vector< crossval::FoldSelection > cvArchSink;
	vector< crossval::ProcedureSpec > procs = assemble( &cvArchSink, cvGroup );
	if ( procs.empty() )
		return jsonMsg( false, "select at least one procedure to compare" );

	// The coarse phase the page has always shown, kept because it is the one line
	//    that stays true for the whole stage; the structured cv block beside it
	//    carries the detail (fold n/k, procedure, nested trial).
	{
		lock_guard< mutex > lock( job.progressMutex );
		job.obdPhase = "cross-validating";
		job.obdHidden = 0;
		job.cvStage = "cross-validation"; // published before the first fold reports
		job.cvProcCount = ( unsigned ) procs.size();
		job.cvK = c.k;
	}

	util::set_seed( c.seed );
	// Stop propagates INTO the run now: the token reaches each procedure's folds
	//    (the network observer, obd::run) so a long CV cancels promptly (bug B1).
	//    Deterministic per-(procedure,fold) substreams (bug B11): a procedure's CV
	//    result is invariant to which other procedures are compared and their order.
	crossval::Comparison cmp = crossval::compare( *cvData, foldId, procs, &job.cancel,
		true /* substreams */, c.seed, cvProgress );
	if ( cmp.cancelled )
		return string( "{\"ok\":false,\"cancelled\":true,\"message\":\"" )
			+ jsonEscape( cmp.message.empty() ? "cancelled" : cmp.message ) + "\"}";
	if ( !cmp.ok )
		return jsonMsg( false, cmp.message.empty() ? "cross-validation failed" : cmp.message );

	// Locked-test inference: refit each procedure on the development rows by its own
	//    rule, score once on the untouched locked rows, then DeLong on the pairing.
	cvreport::LockedInfo lockedInfo;
	if ( locked )
	{
		{
			lock_guard< mutex > lock( job.progressMutex );
			job.obdPhase = "locked-test evaluation";
			// The stage changed: nothing from the CV grid describes this pass.
			//    evaluateOnce folds nothing, so fold/k go to 0 rather than keeping
			//    the last fold number of the last procedure.
			job.clearCvProgress();
			job.cvStage = "locked-test evaluation";
		}
		vector< crossval::FoldSelection > lockedArchSink;
		// The locked refit runs on the FULL dataset's row space, so it takes the
		//    raw group vector -- not cvGroup, which is indexed by development row.
		vector< crossval::ProcedureSpec > lprocs = assemble( &lockedArchSink, rowGroup );
		util::set_seed( c.seed );
		crossval::LockedResult lr = crossval::evaluateOnce( data, devRows, lockedRows,
			lprocs, &job.cancel, true /* substreams */, c.seed, cvProgress );
		if ( lr.cancelled )
			return string( "{\"ok\":false,\"cancelled\":true,\"message\":\"" )
				+ jsonEscape( lr.message.empty() ? "cancelled" : lr.message ) + "\"}";
		if ( !lr.ok )
			return jsonMsg( false, lr.message.empty()
				? "locked-test evaluation failed" : lr.message );
		lockedInfo = buildLockedInfo( lr, c, lockedSplit, rowGroup, nGroups );
	}

	// Report: Tier 1 + Tier 2 as text, Tier 3 written beside the data.
	cvreport::PlanInfo info;
	info.n = n;
	info.events = events;
	// The fold plan, structured: the report derives its sentence from this rather
	//    than being handed one (DLG-8), so the two can never disagree. It is the
	//    SAME object the planner filled in, not a restatement of the request.
	info.folds = foldDesign;
	// Cluster identity for the folded rows, so cv_predictions.csv can say which
	//    out-of-fold predictions share a sampling unit. Gathered by raw row, like
	//    the locked one -- never rebuilt on the development subset.
	if ( grouped ) info.cluster = cvGroup;
	// The ORIGINAL raw row for each folded row, so cv_predictions.csv and
	//    cv_locked_predictions.csv share one identity space. With a locked test the
	//    Comparison is indexed by development row, and writing that index as the
	//    exemplar id made row 7 mean a different patient in each file.
	if ( locked ) info.rawRow = devRows;
	info.rawRowCount = n; // the ORIGINAL dataset size, so the ids can be range-checked
	if ( !locked && c.neural && c.logistic )
	{
		info.primary = c.neuralObd ? "Neural (OBD)" : "Neural";
		info.reference = "Logistic";
	}

	string tier1 = cvreport::tier1( cmp, info, lockedInfo );
	string tier2 = cvreport::tier2( cmp, info, lockedInfo );

	string dir = util::run_path( "" ); // the run directory (trailing slash, or "")
	if ( dir.empty() ) dir = ".";
	else if ( dir.back() == '/' || dir.back() == '\\' ) dir.pop_back();
	vector< cvreport::ArtifactResult > arts =
		cvreport::writeArtifacts( cmp, info, dir, lockedInfo );

	// files[] lists ONLY the artifacts that fully succeeded (opened, written,
	//    flushed, closed); a failed one goes to warnings[] with its reason, so the
	//    response never claims a file was written unless it truly was (B7). The
	//    in-memory CV results (tier1/tier2) are complete regardless, so ok stays true.
	ostringstream filesJson, warnJson;
	filesJson << "["; warnJson << "[";
	bool firstFile = true, firstWarn = true, anyFailed = false;
	for ( unsigned i = 0; i < arts.size(); i++ )
	{
		if ( arts[ i ].ok )
		{
			filesJson << ( firstFile ? "" : "," ) << "\"" << jsonEscape( arts[ i ].path ) << "\"";
			firstFile = false;
		}
		else
		{
			anyFailed = true;
			warnJson << ( firstWarn ? "" : "," ) << "\"" << jsonEscape(
				"could not write " + arts[ i ].name + ": " + arts[ i ].error ) << "\"";
			firstWarn = false;
		}
	}
	filesJson << "]"; warnJson << "]";

	ostringstream msg;
	msg << "cross-validation complete (" << cmp.entries.size() << " procedure"
		<< ( cmp.entries.size() == 1 ? "" : "s" ) << ", " << c.k << " folds";
	if ( locked ) msg << " + locked test (" << lockedInfo.n << " rows)";
	msg << ")";
	if ( anyFailed )
		msg << " -- WARNING: one or more machine-readable files could not be written";

	string cvBlock = string( "{\"tier1\":\"" ) + jsonEscape( tier1 ) + "\""
		+ ",\"tier2\":\"" + jsonEscape( tier2 ) + "\""
		+ ",\"files\":" + filesJson.str()
		+ ",\"warnings\":" + warnJson.str();
	if ( locked ) cvBlock += ",\"locked\":" + lockedJson( lockedInfo );
	cvBlock += "}";

	return string( "{\"ok\":true,\"message\":\"" ) + jsonEscape( msg.str() )
		+ "\",\"cv\":" + cvBlock + "}";
}

string handleCv( const httplib::Request& req )
{
	logAction( req, "cv" );

	if ( !dataPtr )
		return jsonMsg( false, "load a dataset first" );
	// CV folds the RAW dataset; a pre-split (mode=train) dataset has no Raw (B8)
	if ( !dataPtr->rawLoaded() )
		return jsonMsg( false, "cross-validation needs a raw dataset (load with "
			"'raw data' mode, not a pre-split training set)" );
	if ( !( dataPtr->getDiscrete() && dataPtr->getOutput() == 1 ) )
		return jsonMsg( false, "cross-validation needs a discrete, single-output dataset" );

	CvConfig c;
	string bad;
	auto uintParam = [ & ]( const char* name, unsigned& dst, unsigned lo, const char* err )
	{
		string s = param( req, name );
		if ( s.empty() ) return;
		long v = atol( s.c_str() );
		if ( v < ( long ) lo ) { bad = err; return; }
		dst = ( unsigned ) v;
	};
	uintParam( "folds", c.k, 2, "folds must be at least 2" );
	uintParam( "seed", c.seed, 0, "seed must be a whole number" );
	uintParam( "maxiter", c.maxIter, 1, "maxiter must be at least 1" );
	uintParam( "neural_hidden", c.neuralHidden, 1, "neural_hidden must be at least 1" );
	uintParam( "hidden_max", c.obd.hMax, 1, "hidden_max must be at least 1" );
	uintParam( "iter_budget", c.obd.iterBudget, 1, "iter_budget must be at least 1" );
	// The plateau auto-stop, for the nested-OBD search AND (when asked for) the
	//    plain logistic / fixed-architecture neural templates. Present = the
	//    caller wants a reachable stopping rule on every procedure in this run.
	{
		string t = param( req, "autostop_tol" ), w = param( req, "autostop_window" );
		if ( !t.empty() )
		{
			double v = atof( t.c_str() );
			if ( !( v > 0 && v < 1 ) )
				return jsonMsg( false, "autostop_tol must be between 0 and 1" );
			c.obd.plateauTol = v;
			c.plainAutostop = true;
		}
		if ( !w.empty() )
		{
			long v = atol( w.c_str() );
			if ( v < 2 )
				return jsonMsg( false, "autostop_window must be at least 2" );
			c.obd.plateauWindow = ( unsigned ) v;
			c.plainAutostop = true;
		}
	}
	if ( !bad.empty() ) return jsonMsg( false, bad );
	// The nested-OBD search grows from hStart (2); hidden_max below it is an empty
	//    range that would refuse every fold, so reject it up front (bug B2)
	if ( c.neural && c.neuralObd && c.obd.hMax < c.obd.hStart )
		return jsonMsg( false, "hidden_max must be at least the OBD start size (2)" );

	// Optimizer for the nested-OBD search: 1|2|3 fixed, or auto (probe once per
	//    fold and keep the choice for that fold's whole search). Same encoding and
	//    validation as standalone /api/obd -- one public spelling, not two. Absent
	//    = auto, the documented default (never a silent canonical).
	string cvAlgo = param( req, "algorithm" );
	if ( cvAlgo.empty() || cvAlgo == "auto" ) c.obd.algorithm = -1;
	else if ( cvAlgo == "1" ) c.obd.algorithm = 0;
	else if ( cvAlgo == "2" ) c.obd.algorithm = 1;
	else if ( cvAlgo == "3" ) c.obd.algorithm = 2;
	else return jsonMsg( false, "algorithm must be 1, 2, 3 or auto" );

	// Procedure selection: present = the field decides; absent = the default set.
	auto boolParam = [ & ]( const char* name, bool& dst )
	{
		string s = param( req, name );
		if ( !s.empty() ) dst = ( s == "1" || s == "true" );
	};
	boolParam( "logistic", c.logistic );
	boolParam( "ldfa", c.ldfa );
	boolParam( "qdfa", c.qdfa );
	boolParam( "neural", c.neural );
	boolParam( "neural_obd", c.neuralObd );

	{
		string s = param( req, "inner_val" );
		if ( !s.empty() )
		{
			double v = atof( s.c_str() );
			if ( !( v > 0 && v < 1 ) )
				return jsonMsg( false, "inner_val must be between 0 and 1" );
			c.innerVal = v;
		}
	}

	if ( c.k > dataPtr->getRawMatrix().rows() )
		return jsonMsg( false, "folds cannot exceed the number of rows" );
	if ( !( c.logistic || c.ldfa || c.qdfa || c.neural ) )
		return jsonMsg( false, "select at least one procedure to compare" );

	// Locked-test evaluation (ROADMAP 4 Phase 4): an outcome-stratified ROW holdout
	//    held out of CV. locked_fraction [0,1) or locked_n (a count) sizes it (0 =
	//    none). The two are alternatives -- supplying both is a conflict (DLG-7).
	// The CV request's OWN fold policy (ROADMAP 4). Deliberately re-parsed here
	//    rather than read from the loaded DataSet: /api/load's strata=/group=
	//    configure a train/test HOLDOUT, and silently reusing them would change
	//    what a cross-validation means depending on how the dataset panel was last
	//    driven. Absent = the shipped outcome-stratified fold plan, unchanged.
	{
		string strataStr = param( req, "strata" );
		if ( !strataStr.empty() )
		{
			string bad = parseColumnList( strataStr, dataPtr->getInput(), "strata",
				c.strataColumns );
			if ( !bad.empty() ) return jsonMsg( false, bad );
			string binsStr = param( req, "strata_bins" );
			if ( !binsStr.empty() )
			{
				long b = atol( binsStr.c_str() );
				if ( b < 2 ) return jsonMsg( false, "strata_bins must be at least 2" );
				c.strataBins = ( unsigned ) b;
			}
		}
		string groupStr = param( req, "group" );
		if ( !groupStr.empty() )
		{
			string bad = parseColumnList( groupStr, dataPtr->getInput(), "group",
				c.groupColumns );
			if ( !bad.empty() ) return jsonMsg( false, bad );
		}

		// Stratifying and grouping pull in different directions -- one makes each
		//    fold resemble the sample, the other makes it a set of unseen clusters
		//    -- and there is no tested joint balancing objective yet. Refuse the
		//    combination rather than letting one silently win (which is what
		//    /api/load's engine path does, and is a defect to be fixed there, not
		//    a precedent to copy).
		if ( !c.strataColumns.empty() && !c.groupColumns.empty() )
			return jsonMsg( false, "strata and group cannot be combined for a fold "
				"plan yet: stratifying balances the subgroups your sample represents, "
				"while grouping measures transfer to unseen groups. Choose one." );

		c.userStrata.clear(); c.userGroup.clear();
		for ( unsigned i = 0; i < c.strataColumns.size(); i++ )
			c.userStrata.push_back( c.strataColumns[ i ] + 1 );
		for ( unsigned i = 0; i < c.groupColumns.size(); i++ )
			c.userGroup.push_back( c.groupColumns[ i ] + 1 );

		c.foldMethod = !c.groupColumns.empty()
			? evaldesign::PartitionMethod::StratifiedGroup
			: !c.strataColumns.empty()
				? evaldesign::PartitionMethod::CovariateStratified
				: evaldesign::PartitionMethod::OutcomeStratified;
	}

	// The declared sampling unit. It is parsed HERE, ahead of the locked-test
	//    sizing and its preflight, because that preflight has a CLUSTER-specific
	//    check -- and reading c.unit before it was assigned silently disabled it
	//    (Sol, 2026-07-30: the block compiled, ran, and was unreachable). The
	//    prerequisites that depend on the locked size are checked below, once it
	//    is known; everything checkable now is checked now.
	{
		string ind = param( req, "independence" );
		evaldesign::SamplingUnit u;
		if ( !evaldesign::parseSamplingUnit( ind, u ) )
			return jsonMsg( false, "independence (sampling unit) must be 'rows' "
				"(independent observations), 'cluster', or unset" );
		if ( u == evaldesign::SamplingUnit::Cluster && c.groupColumns.empty() )
			return jsonMsg( false, "clustered inference needs a group= key naming the "
				"clustering columns; without one there are no sampling units to cluster "
				"on. Omit the sampling unit to get predictions and point AUCs without "
				"ordinary DeLong, which would be invalid on clustered data." );
		// Declaring independent ROWS over a group-disjoint design is a contradiction:
		//    grouping stops leakage, it does not make the held-out rows independent.
		//    Refused here rather than described as "descriptive grouping" and then
		//    handed to ordinary DeLong. evaldesign::chooseInference enforces the same
		//    rule downstream; this is the early, specific message.
		if ( u == evaldesign::SamplingUnit::Row && !c.groupColumns.empty() )
			return jsonMsg( false, "independence=rows contradicts a grouped design: "
				"grouping prevents leakage but does not make rows independent, so "
				"ordinary DeLong would not be valid. Omit the sampling unit for point "
				"AUCs without inference, or drop group= if the rows really are "
				"independent." );
		c.unit = u;
	}

	unsigned nRows = dataPtr->getRawMatrix().rows();
	{
		string frac = param( req, "locked_fraction" ), cnt = param( req, "locked_n" );
		if ( !frac.empty() && !cnt.empty() )
			return jsonMsg( false, "specify locked_fraction OR locked_n, not both" );
		if ( !frac.empty() )
		{
			double f = atof( frac.c_str() );
			if ( f < 0 || f >= 1 )
				return jsonMsg( false, "locked_fraction must be in [0, 1) (0 = none)" );
			c.lockedN = ( unsigned )( f * nRows + 0.5 ); // 0 -> none
		}
		else if ( !cnt.empty() )
		{
			long v = atol( cnt.c_str() );
			if ( v < 0 ) return jsonMsg( false, "locked_n cannot be negative" );
			c.lockedN = ( unsigned ) v;
		}
		// Clustered inference is computed ON the locked test, so it needs one.
		//    Checked here, where the size is finally known.
		if ( c.unit == evaldesign::SamplingUnit::Cluster && c.lockedN == 0 )
			return jsonMsg( false, "clustered inference is computed on the locked test "
				"set; set locked_fraction or locked_n. Omit the sampling unit to get "
				"predictions and point AUCs without ordinary DeLong, which would be "
				"invalid on clustered data." );
		if ( c.lockedN > 0 )
		{
			if ( nRows < c.lockedN + c.k )
				return jsonMsg( false, "the locked test leaves too few development "
					"rows for the requested folds" );

			// DLG-3: validate the ACHIEVED per-class counts of the actual seeded
			//    holdout, not just the total. A rare outcome can leave zero or one
			//    event in the locked test (or the development set), which no total-
			//    size check catches; that would accept an expensive job only to
			//    return "AUC not computable". Preview the split (same seed the worker
			//    uses -> identical), require >= 2 of each class on BOTH sides, and
			//    report the achieved counts in the refusal.
			Matrix< double >& rawM = dataPtr->getRawMatrix();
			unsigned oc = rawM.cols() - 1;
			vector< unsigned > lab( nRows );
			for ( unsigned r = 0; r < nRows; r++ )
				lab[ r ] = ( rawM( r, oc ) != 0 ) ? 1u : 0u;
			// Preview the split the WORKER will make -- same seed, same planner.
			//    A grouped request holds out whole groups, so previewing with the
			//    row-wise holdout would validate counts the run never produces.
			util::set_seed( c.seed );
			vector< unsigned > lockedTest, lockedTrain, gkey;
			unsigned gCount = 0;
			if ( !c.groupColumns.empty() )
			{
				gkey = dataPtr->groupKey( c.groupColumns );
				set< unsigned > gs( gkey.begin(), gkey.end() );
				gCount = ( unsigned ) gs.size();
				nsplit::GroupHoldout g = nsplit::groupHoldout( lab, gkey, c.lockedN );
				lockedTest = g.test; lockedTrain = g.train;
			}
			else
			{
				nsplit::Holdout h = nsplit::stratifiedHoldout( lab, c.lockedN );
				lockedTest = h.test; lockedTrain = h.train;
			}
			unsigned lk1 = 0, lk0 = 0, dv1 = 0, dv0 = 0;
			for ( unsigned i = 0; i < lockedTest.size(); i++ )
				( lab[ lockedTest[ i ] ] ? lk1 : lk0 )++;
			for ( unsigned i = 0; i < lockedTrain.size(); i++ )
				( lab[ lockedTrain[ i ] ] ? dv1 : dv0 )++;

			if ( lk0 < 2 || lk1 < 2 )
			{
				ostringstream m;
				m << "the locked test has too few of a class (events=" << lk1
					<< ", non-events=" << lk0 << "; need >= 2 of each). Use a larger "
					"locked size.";
				return jsonMsg( false, m.str() );
			}
			// The development set feeds a k-fold plan, so it needs enough of EACH
			//    class that every outer fold can contain both -- i.e. >= k events and
			//    >= k non-events (stratified k-fold deals each class round-robin, so
			//    fewer than k of a class necessarily leaves some fold without it, and
			//    that fold's ROC area cannot be computed). This is stricter than the
			//    old ">= 2" check, which was independent of k (DLG-3). Nested OBD's
			//    inner validation split is NOT proven feasible by this check -- if an
			//    inner split is degenerate the adapter fails that fold and reports it
			//    (graceful, never silent), which is the intended contract there.
			if ( dv0 < c.k || dv1 < c.k )
			{
				ostringstream m;
				m << "too few of a class remain for " << c.k << "-fold development "
					"(events=" << dv1 << ", non-events=" << dv0 << "; each outer fold "
					"needs both classes, so >= " << c.k << " of each is required). Use a "
					"smaller locked size or fewer folds.";
				return jsonMsg( false, m.str() );
			}
			// A grouped run folds the DEVELOPMENT groups, so there must be at
			//    least k of them -- a check the row counts cannot make, because a
			//    thousand development rows in three clusters still cannot fill
			//    five group-disjoint folds.
			if ( gCount )
			{
				set< unsigned > devGroups;
				for ( unsigned i = 0; i < lockedTrain.size(); i++ )
					devGroups.insert( gkey[ lockedTrain[ i ] ] );
				if ( devGroups.size() < c.k )
				{
					ostringstream m;
					m << "only " << devGroups.size() << " group(s) remain in the "
						"development set for " << c.k << " group-disjoint folds "
						"(the key gives " << gCount << " groups in all). Use fewer "
						"folds, a smaller locked size, or a coarser group key.";
					return jsonMsg( false, m.str() );
				}
				// Clustered inference divides by ( informative clusters - 1 ), so
				//    the LOCKED sample needs at least two clusters carrying each
				//    outcome class. That is the estimator's own requirement, checked
				//    here so an expensive run is not spent to discover it -- and it
				//    is a cluster condition, not a row condition: fifty locked rows
				//    from one cluster carry no between-cluster information at all.
				if ( c.unit == evaldesign::SamplingUnit::Cluster )
				{
					set< unsigned > withEvent, withoutEvent;
					for ( unsigned i = 0; i < lockedTest.size(); i++ )
						( lab[ lockedTest[ i ] ] ? withEvent : withoutEvent )
							.insert( gkey[ lockedTest[ i ] ] );
					if ( withEvent.size() < 2 || withoutEvent.size() < 2 )
					{
						ostringstream m;
						m << "clustered inference needs at least two locked-test "
							"clusters carrying each outcome class (found "
							<< withEvent.size() << " with an event and "
							<< withoutEvent.size() << " without). Use a larger locked "
							"size, or a finer group key.";
						return jsonMsg( false, m.str() );
					}
				}
			}
		}
	}

	// The prespecified DeLong contrast. Tokens (logistic|ldfa|qdfa|neural) resolve to
	//    the SELECTED procedures; an explicit contrast naming an unselected procedure
	//    is a validation error -- we never silently default to a pair that isn't in
	//    the run (Sol's caution). Absent tokens default to neural vs logistic ONLY
	//    when both are selected, else no contrast is drawn.
	{
		string neuralName = c.neuralObd ? "Neural (OBD)" : "Neural";
		auto resolve = [ & ]( const string& tok, string& out ) -> string
		{
			string t = tok;
			for ( unsigned i = 0; i < t.size(); i++ ) t[ i ] = tolower( t[ i ] );
			if ( t == "logistic" ) { if ( !c.logistic ) return "logistic"; out = "Logistic"; }
			else if ( t == "ldfa" ) { if ( !c.ldfa ) return "LDFA"; out = "LDFA"; }
			else if ( t == "qdfa" ) { if ( !c.qdfa ) return "QDFA"; out = "QDFA"; }
			else if ( t == "neural" ) { if ( !c.neural ) return "neural"; out = neuralName; }
			else return "?";
			return ""; // ok
		};
		string pTok = param( req, "primary" ), rTok = param( req, "reference" );
		if ( !pTok.empty() || !rTok.empty() )
		{
			if ( pTok.empty() || rTok.empty() )
				return jsonMsg( false, "specify both primary and reference for a "
					"prespecified contrast, or neither" );
			string badP = resolve( pTok, c.primary ), badR = resolve( rTok, c.reference );
			if ( badP == "?" || badR == "?" )
				return jsonMsg( false, "primary/reference must be one of "
					"logistic, ldfa, qdfa, neural" );
			if ( !badP.empty() )
				return jsonMsg( false, "primary procedure '" + badP
					+ "' is not among the selected procedures" );
			if ( !badR.empty() )
				return jsonMsg( false, "reference procedure '" + badR
					+ "' is not among the selected procedures" );
			if ( c.primary == c.reference )
				return jsonMsg( false, "primary and reference must differ" );
		}
		else if ( c.neural && c.logistic )
		{
			c.primary = neuralName; c.reference = "Logistic"; // default contrast
		}
	}

	// Async-only, like OBD: many training runs on the worker thread.
	if ( job.worker.joinable() )
		job.worker.join();
	{
		lock_guard< mutex > lock( job.progressMutex );
		job.resetForNewRun();
	}
	job.cancel = false;
	job.running = true;

	job.worker = thread( [ c ]
	{
		string result = runCvJob( c );
		{
			lock_guard< mutex > lock( job.progressMutex );
			job.result = result;
		}
		job.running = false;
	} );

	return jsonMsg( true, "cross-validation started" );
}

// The whole stepwise analysis, run identically in blocking and async mode so
//    the two cannot drift: async is the SAME call on a worker thread, with a
//    progress callback and a cancel observer attached. Returns the response
//    JSON. The caller must own the engine.
string runRegressJob( Network* net, const vector< vector< unsigned > >& defs,
	const string& direction, double threshold )
{
	RegressNet regressionObj;
	regressionObj.setNetwork( net, lastTrainError );
	regressionObj.setInputStructure( defs ); // already validated by the caller
	regressionObj.setThreshold( threshold ); // no stdin prompt in the GUI

	// Publish each candidate as the engine reaches it. Structured facts from
	//    the procedure itself -- never scraped from the report's prose.
	StepwiseCancelObserver cancelObserver;
	regressionObj.setObserver( &cancelObserver );

	// The most recently COMPLETED candidate, kept alongside the one currently
	//    training. A finished announcement is followed by the next candidate's
	//    starting announcement within microseconds, so a status poller would
	//    essentially never observe it as a transient phase -- the spec asks for
	//    the last completed candidate's convergence status, which has to
	//    PERSIST while the next one trains. Captured by reference: the callback
	//    is only ever invoked from inside this function's regress call.
	string lastDone;
	regressionObj.setProgress( [ &lastDone ]( const RegressNet::Progress& p )
	{
		ostringstream s;
		s << "{\"direction\":\"" << jsonEscape( p.direction )
			<< "\",\"step\":" << p.step
			<< ",\"candidate\":" << p.candidate
			<< ",\"candidatesThisStep\":" << p.candidatesThisStep
			<< ",\"fitsCompleted\":" << p.fitsCompleted
			<< ",\"variable\":" << p.variable
			<< ",\"inputs\":[";
		for ( unsigned i = 0; i < p.inputs.size(); i++ )
			s << ( i ? "," : "" ) << p.inputs[ i ];
		s << "],\"phase\":\"" << jsonEscape( p.phase ) << "\""
			<< ",\"finished\":" << ( p.finished ? "true" : "false" );

		// A finished announcement updates the sticky record of the last
		//    completed candidate, which then rides along with every subsequent
		//    poll until the next candidate finishes
		if ( p.finished )
		{
			ostringstream d;
			d << "{\"variable\":" << p.variable
				<< ",\"converged\":" << ( p.converged ? "true" : "false" )
				<< ",\"stopReason\":\"" << jsonEscape( p.stopReason ) << "\"}";
			lastDone = d.str();
		}
		if ( !lastDone.empty() )
			s << ",\"lastCompleted\":" << lastDone;
		s << "}";

		lock_guard< mutex > lock( job.progressMutex );
		job.stepwise = s.str();
	} );

	Capture cap;
	bool ok = true;
	string message;
	try
	{
		if ( direction == "reverse" )
			regressionObj.reverse_regress();
		else
			regressionObj.forward_regress();
		message = direction + " stepwise regression complete";
	}
	catch ( const RegressNet::RegressNetErr& e )
	{
		ok = false;
		message = e.what();
	}

	// A cancelled run is reported as cancelled, not as a mysterious failure --
	//    the user asked for it. The candidate that was training ended as
	//    STOP_CANCELLED, which is not convergence, so the analysis stopped
	//    there and no variable was selected from the incomplete pass.
	bool cancelled = job.cancel.load();

	ostringstream out;
	out << "{\"ok\":" << ( ok ? "true" : "false" )
		<< ",\"message\":\"" << jsonEscape( cancelled
			? direction + " stepwise regression cancelled" : message )
		<< "\",\"cancelled\":" << ( cancelled ? "true" : "false" )
		<< ",\"output\":\"" << jsonEscape( cap.text.str() ) << "\"";

	// The structured result, so a consumer never has to parse the prose. The
	//    partial path is included on failure and cancellation too: it is the
	//    audit trail of what the procedure did manage to establish.
	out << ",\"stepwise\":{\"direction\":\"" << jsonEscape( direction )
		<< "\",\"threshold\":" << jnum( threshold )
		<< ",\"fitsCompleted\":" << regressionObj.getFitsCompleted();

	// EVERY candidate considered, in order, with the comparison that was made
	//    and how the fit ended. This is the auditable record: a consumer never
	//    has to parse the prose to reconstruct the analysis, and because it is
	//    built as the run proceeds it survives a failure or a cancel and shows
	//    exactly how far the procedure got.
	out << ",\"candidates\":[";
	const vector< RegressNet::Candidate >& cands = regressionObj.getCandidates();
	for ( unsigned i = 0; i < cands.size(); i++ )
	{
		const RegressNet::Candidate& c = cands[ i ];
		out << ( i ? "," : "" ) << "{\"step\":" << c.step
			<< ",\"candidate\":" << c.candidate
			<< ",\"variable\":" << c.variable
			<< ",\"inputs\":[";
		for ( unsigned k = 0; k < c.inputs.size(); k++ )
			out << ( k ? "," : "" ) << c.inputs[ k ];
		out << "],\"priorError\":" << jnum( c.priorError )
			<< ",\"error\":" << jnum( c.error )
			<< ",\"df\":" << c.df
			<< ",\"g2\":" << jnum( c.G2 )
			<< ",\"p\":" << jnum( c.p )
			<< ",\"converged\":" << ( c.converged ? "true" : "false" )
			<< ",\"stopReason\":\"" << jsonEscape( c.stopReason ) << "\""
			<< ",\"iterations\":" << c.iterations
			<< ",\"selected\":" << ( c.selected ? "true" : "false" ) << "}";
	}
	out << "]";

	out << ",\"path\":[";
	const vector< pair< double, unsigned > >& path
		= regressionObj.getSelectionPath();
	for ( unsigned i = 0; i < path.size(); i++ )
		out << ( i ? "," : "" ) << "{\"variable\":" << path[ i ].second
			<< ",\"p\":" << jnum( path[ i ].first ) << "}";
	out << "]";

	// finalVariables reports what the COMPLETED passes settled, and `complete`
	//    says whether the procedure reached a decision at all. Without the
	//    flag, an empty list on a run that failed in its first pass reads as
	//    "the procedure kept nothing" rather than "the procedure never got
	//    that far" -- two very different claims.
	out << ",\"complete\":"
		<< ( regressionObj.getComplete() ? "true" : "false" )
		<< ",\"finalVariables\":[";
	const vector< unsigned >& fin = regressionObj.getFinalVariables();
	for ( unsigned i = 0; i < fin.size(); i++ )
		out << ( i ? "," : "" ) << fin[ i ];
	out << "]}}";

	// Note: the training report (lastReport, the Report download) is left
	//    intact, and the trained model is untouched -- stepwise is a standalone
	//    analysis over clones. When history logging is on the report is also
	//    appended to neuron.log in the workspace.
	return out.str();
}

string handleRegress( const httplib::Request& req )
{
	logAction( req, "regress" );
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

	// Validate the structure against the network HERE, synchronously, so a bad
	//    request fails in the caller's own response even in async mode -- a job
	//    must never be started only to die of something knowable up front.
	{
		RegressNet validate;
		validate.setNetwork( net, lastTrainError );
		try
		{
			validate.setInputStructure( variable_defs );
		}
		catch ( const RegressNet::RegressNetErr& e )
		{
			return jsonMsg( false, e.what() );
		}
	}

	// Asynchronous mode: validation above has already happened synchronously,
	//    so a bad request still fails fast and in the caller's own response.
	//    Everything past this point is the run itself.
	if ( param( req, "async" ) == "1" )
	{
		if ( job.worker.joinable() )
			job.worker.join(); // reap the previous, finished run's thread

		{
			lock_guard< mutex > lock( job.progressMutex );
			job.resetForNewRun();
		}
		job.cancel = false;
		job.running = true;

		job.worker = thread( [ net, variable_defs, direction, threshold ]
		{
			string result = runRegressJob( net, variable_defs, direction,
				threshold );
			{
				lock_guard< mutex > lock( job.progressMutex );
				job.result = result; // publish BEFORE running goes false
			}
			job.running = false;
		} );

		return jsonMsg( true, direction + " stepwise regression started" );
	}

	// Blocking mode still installs the cancel observer (runRegressJob is ONE
	//    implementation for both modes), and job.cancel is a process-global
	//    latch that only an async START clears. So a Stop from an earlier async
	//    run would still be set here and cancel this run's very first candidate
	//    before it trained an iteration. Clear it: nothing can be cancelling a
	//    run that has not begun.
	job.cancel = false;
	return runRegressJob( net, variable_defs, direction, threshold );
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
		{ "test_guesses", "test_guesses.txt" },
		{ "dfa_train_guesses", "dfa_train_guesses.txt" },
		{ "dfa_test_guesses", "dfa_test_guesses.txt" } };
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
	else if ( what == "dfa_train_guesses" || what == "dfa_test_guesses" )
	{
		// The last discriminant analysis's guesses (kept in dfaPtr; the CLI
		//    offers this save right after a DFA run)
		if ( !dfaPtr )
			err = "run a discriminant function analysis first";
		else if ( what == "dfa_train_guesses" )
			ok = dfaPtr->getDataSet().saveTrainTwoSet( name );
		else
			ok = dfaPtr->getDataSet().saveTestTwoSet( name );
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

	// Bound how long an idle kept-alive connection may park a pool thread.
	//    One second keeps browser keep-alive fully effective (the page polls
	//    every 400 ms) while letting one-shot clients' leftover sockets reap
	//    quickly. NOTE for curl callers (AGENTS.md documents this): a POST
	//    with NO body and no Content-Length header (curl -X POST without -d)
	//    makes the server wait its read timeout (~5 s) for a body that never
	//    comes before dispatching -- always send -d "" on bodyless POSTs.
	//    That stall, measured at exactly 5 s on /api/train/stop, is what broke
	//    the OBD cancel smoke on CI.
	svr.set_keep_alive_timeout( 1 );

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

	svr.Post( "/api/dfa", []( const httplib::Request& req, httplib::Response& res )
	{
		lock_guard< mutex > lock( engineMutex );
		if ( busyGate( res ) ) return;
		res.set_content( handleDFA( req ), "application/json" );
	} );

	// OBD hidden-layer sizing: async-only, shares the train job machinery
	//    (status + stop reach it through the same doors as async training)
	svr.Post( "/api/obd", []( const httplib::Request& req, httplib::Response& res )
	{
		lock_guard< mutex > lock( engineMutex );
		if ( busyGate( res ) ) return;
		res.set_content( handleObd( req ), "application/json" );
	} );

	// Cross-validation model comparison: async-only, shares the train job
	//    machinery (status + stop reach it through the same doors as OBD)
	svr.Post( "/api/cv", []( const httplib::Request& req, httplib::Response& res )
	{
		lock_guard< mutex > lock( engineMutex );
		if ( busyGate( res ) ) return;
		res.set_content( handleCv( req ), "application/json" );
	} );

	// Session artifacts: written into the workspace by the engine's own
	//    save methods AND sent to the browser as a download
	svr.Get( "/api/save/:what", []( const httplib::Request& req, httplib::Response& res )
	{
		lock_guard< mutex > lock( engineMutex );
		if ( busyGate( res ) ) return;
		string what = req.path_params.at( "what" );
		logActionLine( "save " + what );

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
