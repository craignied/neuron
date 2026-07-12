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
#include "utility.h"
#include "version.h"

#include "gui_page.h" // generated from src/gui_page.html by CMake

using namespace std;

namespace {

// The engine state driven by the page — mirrors the driver's owners
mutex engineMutex;
unique_ptr< DataSet > dataPtr;
unique_ptr< Model > modelPtr;

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
	double area = t.getTrapROCarea();
	ostringstream out;
	out.precision( 6 );
	out << "{\"x\":" << jsonNumbers( t.getROCx() )
		<< ",\"y\":" << jsonNumbers( t.getROCy() )
		<< ",\"area\":" << area << "}";
	return out.str();
}

// --- Small utilities ------------------------------------------------------

bool fileExists( const string& path )
{
	ifstream f( path.c_str() );
	return ( bool ) f;
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

// --- Handlers --------------------------------------------------------------

string handleLoad( const httplib::Request& req )
{
	string mode = req.get_param_value( "mode" ),
		path = req.get_param_value( "path" );
	unsigned nI = ( unsigned ) atol( req.get_param_value( "inputs" ).c_str() ),
		nO = ( unsigned ) atol( req.get_param_value( "outputs" ).c_str() );
	double fraction = atof( req.get_param_value( "fraction" ).c_str() );

	if ( path.empty() || !fileExists( path ) )
		// Checked here because the engine's missing-file recovery prompts
		//    on stdin, which the GUI must never touch
		return jsonMsg( false, "can't open file: " + path );
	if ( nI < 1 || nO < 1 )
		return jsonMsg( false, "inputs and outputs must be at least 1" );

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
	}

	if ( !ok )
	{
		// The engine's own explanation is the best error message
		string why = cap.text.str();
		return jsonMsg( false, why.empty() ? "load failed" : why );
	}

	tuneThresholds( *ds );

	ostringstream msg;
	msg << ds->getNumTrain() << " training exemplars";
	if ( ds->testLoaded() )
		msg << ", " << ds->getNumTest() << " test exemplars";
	msg << " loaded";

	dataPtr = std::move( ds );
	modelPtr.reset(); // a new dataset invalidates any existing model

	return jsonMsg( true, msg.str() );
}

string handleModel( const httplib::Request& req )
{
	string type = req.get_param_value( "type" );
	unsigned hidden = ( unsigned ) atol( req.get_param_value( "hidden" ).c_str() );

	if ( !dataPtr )
		return jsonMsg( false, "load a dataset first" );
	if ( dataPtr->getOutput() != 1 )
		return jsonMsg( false, "the GUI supports 1-output models" );

	Capture cap;

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

string handleTrain( const httplib::Request& req )
{
	unsigned algorithm = ( unsigned ) atol( req.get_param_value( "algorithm" ).c_str() ),
		maxIter = ( unsigned ) atol( req.get_param_value( "maxiter" ).c_str() );
	string seed = req.get_param_value( "seed" );

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
	msg << "trained; final error " << finalError;

	return string( "{\"ok\":true,\"message\":\"" ) + jsonEscape( msg.str() )
		+ "\",\"output\":\"" + jsonEscape( cap.text.str() ) + "\"" + roc + "}";
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

	svr.Post( "/api/train", []( const httplib::Request& req, httplib::Response& res )
	{
		lock_guard< mutex > lock( engineMutex );
		res.set_content( handleTrain( req ), "application/json" );
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
