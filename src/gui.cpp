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

#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>

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
		<< ",\"area\":" << area
		<< ",\"se\":" << se
		<< ",\"acc\":" << acc
		<< ",\"sens\":" << sens
		<< ",\"spec\":" << spec << "}";
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

	Capture cap;
	bool ok = false;

	if ( mode == "raw" )
	{
		ds->loadRaw( path );
		if ( ds->rawLoaded() )
			ok = ds->randomizeD( fraction );
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

string handleTrain( const httplib::Request& req )
{
	unsigned algorithm = ( unsigned ) atol( param( req, "algorithm" ).c_str() ),
		maxIter = ( unsigned ) atol( param( req, "maxiter" ).c_str() );
	string seed = param( req, "seed" );

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

	ostringstream msg;
	msg.precision( 6 );
	msg << ( continued ? "continued training; final error "
		: "trained; final error " ) << finalError;

	lastReport = cap.text.str(); // downloadable afterwards as report.txt
	lastTrainError = finalError; // baseline for stepwise regression

	return string( "{\"ok\":true,\"message\":\"" ) + jsonEscape( msg.str() )
		+ "\",\"output\":\"" + jsonEscape( lastReport ) + "\"" + roc + "}";
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
		res.set_content( handleLoad( req ), "application/json" );
	} );

	svr.Post( "/api/model", []( const httplib::Request& req, httplib::Response& res )
	{
		lock_guard< mutex > lock( engineMutex );
		res.set_content( handleModel( req ), "application/json" );
	} );

	svr.Post( "/api/randomize", []( const httplib::Request& req, httplib::Response& res )
	{
		lock_guard< mutex > lock( engineMutex );
		res.set_content( handleRandomize( req ), "application/json" );
	} );

	svr.Post( "/api/train", []( const httplib::Request& req, httplib::Response& res )
	{
		lock_guard< mutex > lock( engineMutex );
		res.set_content( handleTrain( req ), "application/json" );
	} );

	svr.Post( "/api/regress", []( const httplib::Request& req, httplib::Response& res )
	{
		lock_guard< mutex > lock( engineMutex );
		res.set_content( handleRegress( req ), "application/json" );
	} );

	// Session artifacts: written into the workspace by the engine's own
	//    save methods AND sent to the browser as a download
	svr.Get( "/api/save/:what", []( const httplib::Request& req, httplib::Response& res )
	{
		lock_guard< mutex > lock( engineMutex );
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

	return 0;
}
