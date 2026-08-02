// check_dfa.cpp : CHARACTERIZATION of LDFA and QDFA, before they share anything.
//
// refactor_audit.md items 2.4 and 2.5 propose extracting the reporting scaffold
// of DFA::train() and the loop structure of reportAccuracy(). Sol's condition on
// that extraction is explicit, and this file exists to make it checkable:
//
//   * THE PUBLISHED FORMULAE STAY VISIBLE IN LDFA AND QDFA. The linear
//     discriminant is U'S x - K and the larger wins; the quadratic is
//     (x-U)'S(x-U) + K and the SMALLER wins. Those are two published formulae
//     with opposite senses, and they may not become a largerWins() flag, a
//     formula descriptor, or a type switch in a shared layer (standing rule 7).
//     Only orchestration and report mechanics PROVEN identical may be shared.
//
// What this file pins, per Sol's list: the reports, the guesses, what ROC and
// statistics are available for each output arity, save behavior, history and
// last-operation behavior, the singular and non-discrete refusal paths, and
// that a standalone discriminant analysis does not disturb a trained model.
//
// PORTABLE BY CONSTRUCTION, as check_autostep and check_onehidden are: no
// multi-iteration floating-point literals. What is asserted is structure (which
// lines a report contains), integers (row counts, distinct-value counts),
// relations that hold in the same process (this equals that, this is unchanged),
// and inequalities that a sign error would break (AUC above chance).
//
// THE DIRECTION PIN IS THE IMPORTANT ONE. LDFA takes the larger discriminant and
// QDFA the smaller; each writes a graded class-1 score through the sigmoid, so a
// flipped sense would not crash, would not change the number of scores, and
// would leave a perfectly good-looking report -- with the ROC area on the wrong
// side of 0.5. That is exactly what a shared "which one wins" flag would risk,
// and cases 5 and 6 are what would catch it.

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

#include "ldfa.h"
#include "qdfa.h"
#include "simpleprop.h"
#include "dataset.h"
#include "utility.h"

using namespace std;

static int failures = 0;

static void expect( bool ok, const string& what )
{
	if ( ok )
		cout << "ok - " << what << endl;
	else
	{
		cout << "FAIL - " << what << endl;
		failures++;
	}
}

// A private directory under the platform's own temporary root -- /tmp is not a
//    temporary directory on Windows, and a fixed name collides between
//    concurrent runs.
class TempDir {
public:
	TempDir()
	{
		namespace fs = std::filesystem;
		root = fs::temp_directory_path()
			/ ( "neuron_dfa_" + to_string( ( long ) getpid() ) );
		fs::create_directories( root );
	}
	~TempDir()
	{
		std::error_code ec;
		std::filesystem::remove_all( root, ec );
	}
	string file( const char* name ) const
	{
		return ( root / name ).string();
	}
private:
	std::filesystem::path root;
};

static TempDir tempDir;
static string tmp( const char* name ) { return tempDir.file( name ); }

static string slurp( const string& path )
{
	ifstream in( path.c_str() );
	ostringstream all;
	all << in.rdbuf();
	return all.str();
}

static bool has( const string& hay, const string& needle )
{
	return hay.find( needle ) != string::npos;
}

// --- the fixtures ----------------------------------------------------------
//
// Two Gaussian-ish clouds separated along both axes, built without an RNG so
// that every run of every build sees the same numbers. Class 1 is shifted by
// +1.2 and given a wider spread, so LDFA (common covariance) and QDFA (separate
// covariances) both have something to find and neither is degenerate.

static DataSet binaryData( unsigned n = 120 )
{
	Matrix< double > raw( n, 3 );
	for ( unsigned i = 0; i < n; i++ )
	{
		double t = ( ( i * 37 ) % 101 ) / 100.0;   // 0 .. 1, deterministic
		double u = ( ( i * 53 ) % 97 ) / 96.0;
		bool one = ( i % 2 ) == 1;
		raw( i, 0 ) = ( one ? 1.2 : 0.0 ) + ( one ? 1.6 : 1.0 ) * ( t - 0.5 );
		raw( i, 1 ) = ( one ? 1.0 : 0.0 ) + ( one ? 1.4 : 0.9 ) * ( u - 0.5 );
		raw( i, 2 ) = one ? 1 : 0;
	}
	DataSet d;
	d.setInput( 2 );
	d.setOutput( 1 );
	d.setDiscrete( true );
	d.setHistory( false );
	util::ScreenCapture hush;
	d.setRawMatrix( raw );
	d.setTrainMatrix( raw );
	d.setTestMatrix( raw );
	return d;
}

// Three one-hot classes, three separated clouds.
static DataSet multiData( unsigned n = 150 )
{
	Matrix< double > raw( n, 5 );
	for ( unsigned i = 0; i < n; i++ )
	{
		unsigned c = i % 3;
		double t = ( ( i * 41 ) % 89 ) / 88.0, u = ( ( i * 59 ) % 83 ) / 82.0;
		raw( i, 0 ) = c * 1.5 + ( t - 0.5 );
		raw( i, 1 ) = ( c == 1 ? 1.4 : 0.0 ) + ( u - 0.5 );
		raw( i, 2 ) = ( c == 0 ) ? 1 : 0;
		raw( i, 3 ) = ( c == 1 ) ? 1 : 0;
		raw( i, 4 ) = ( c == 2 ) ? 1 : 0;
	}
	DataSet d;
	d.setInput( 2 );
	d.setOutput( 3 );
	d.setDiscrete( true );
	d.setHistory( false );
	util::ScreenCapture hush;
	d.setRawMatrix( raw );
	d.setTrainMatrix( raw );
	d.setTestMatrix( raw );
	return d;
}

// A dataset whose second input is an exact multiple of the first: the covariance
//    matrix is singular, and the inverse must refuse.
static DataSet collinearData( unsigned n = 60 )
{
	Matrix< double > raw( n, 3 );
	for ( unsigned i = 0; i < n; i++ )
	{
		double t = ( ( i * 31 ) % 71 ) / 70.0;
		raw( i, 0 ) = t + ( i % 2 ? 0.8 : 0.0 );
		raw( i, 1 ) = 2.0 * raw( i, 0 ); // exactly collinear
		raw( i, 2 ) = i % 2;
	}
	DataSet d;
	d.setInput( 2 );
	d.setOutput( 1 );
	d.setDiscrete( true );
	d.setHistory( false );
	util::ScreenCapture hush;
	d.setRawMatrix( raw );
	d.setTrainMatrix( raw );
	return d;
}

// Runs a DFA and returns the report it wrote to the screen.
template < class DFAMODEL >
static string runDFA( DFAMODEL& m, DataSet& d )
{
	util::ScreenCapture cap;
	m.setDataSet( d );
	m.setHistory( false );
	m.setLastop( false );
	m.train();
	return cap.text();
}

// --- 1, 2. the report each model writes ------------------------------------

template < class DFAMODEL >
static void test_report( const char* who, const char* banner )
{
	cout << "-- " << who << ": the binary report --" << endl;

	DataSet d = binaryData();
	DFAMODEL m;
	string said = runDFA( m, d );

	expect( !said.empty(), string( who ) + ": the run writes a report" );
	expect( has( said, banner ), string( who ) + ": the report names itself, \""
		+ banner + "\"" );

	// DFA::outputHeader -- the architecture lines, which are the base class's
	expect( has( said, "2 input nodes" ) && has( said, "1 output nodes" ),
		string( who ) + ": the header states the architecture" );
	expect( has( said, "Number of exemplars in training set: 120" ),
		string( who ) + ": the header counts the training exemplars" );
	expect( has( said, "Number of exemplars in test set: 120" ),
		string( who ) + ": the header counts the test exemplars" );

	// A 1-output run goes through DataSet::metricsReport, so it gets the
	//    classification table and the full statistics -- ROC included.
	expect( has( said, "Training set:" ) && has( said, "Test set:" ),
		string( who ) + ": both sets are reported" );
	expect( has( said, "Classification accuracy" ),
		string( who ) + ": the classification table is present" );
	expect( has( said, "ROC" ) || has( said, "roc" ),
		string( who ) + ": ROC statistics are present for 1 output" );
}

// --- 3. the guesses are GRADED, not a hard 0/1 decision --------------------
//
// Both models deliberately store sigmoidal( margin ) rather than the decision
// itself: a hard 0/1 gives the ROC a single operating point, so the trapezoidal
// area collapses to (sens+spec)/2 and no statistical fit is possible. A
// regression to a hard decision would leave at most two distinct scores.
//
// THE RANGE IS CLOSED, [0,1], AND THAT IS NOT A CONCESSION -- it is what the
// two models actually do, measured on this fixture:
//
//     LDFA  min 0.0502  max 0.9666   0 saturated   120 distinct
//     QDFA  min 2.23e-9 max 1.0     20 saturated   101 distinct
//
// The linear margin is a difference of similarities and stays modest; the
// quadratic margin is a difference of Mahalanobis DISTANCES plus log-determinant
// constants, which reaches the tens, and sigmoidal() of that is exactly 1.0 in
// double precision. Seventeen percent of QDFA's scores tie at the top here.
// That costs ROC resolution at the extreme -- it does not change the ordering
// below it, and the area is still well above chance (case 4) -- and it is worth
// knowing before anyone "simplifies" the graded score.
//
// The saturated COUNT is deliberately not asserted: it comes through a matrix
// inverse, so a borderline row could move in the last bits on another
// toolchain. What is asserted is the closed range and the graded-ness.

template < class DFAMODEL >
static void test_graded_guesses( const char* who )
{
	cout << "-- " << who << ": the guesses --" << endl;

	DataSet d = binaryData();
	DFAMODEL m;
	runDFA( m, d );

	TwoSet& ts = m.getDataSet().getTrainTwoSet();
	expect( ts.loaded(), string( who ) + ": the training TwoSet is loaded" );

	set< double > distinct;
	bool inRange = true;
	unsigned n = 0;
	for ( unsigned r = 0; r < 120; r++ )
	{
		double g = ts.test( r );
		distinct.insert( g );
		if ( !( g >= 0.0 && g <= 1.0 ) )
			inRange = false;
		n++;
	}

	expect( n == 120, string( who ) + ": one guess per training exemplar" );
	expect( inRange, string( who ) + ": every guess is a probability in [0,1]" );
	expect( distinct.size() > 2, string( who ) + ": the guesses are graded -- "
		+ to_string( distinct.size() ) + " distinct scores, not a 0/1 decision" );
}

// --- 4. ROC availability, and the DIRECTION of the discriminant ------------
//
// THE SIGN PIN. LDFA predicts class 1 when d1 > d0 and scores sigmoidal(d1-d0);
// QDFA predicts class 1 when d0 >= d1 and scores sigmoidal(d0-d1). The senses
// are opposite because the formulae are: the linear discriminant is a
// similarity and the quadratic is a distance. If a shared layer ever got that
// backwards -- exactly what a largerWins() flag invites -- the report would look
// entirely normal and the area would land BELOW chance.

template < class DFAMODEL >
static void test_direction( const char* who )
{
	cout << "-- " << who << ": the discriminant points the right way --" << endl;

	DataSet d = binaryData();
	DFAMODEL m;
	runDFA( m, d );

	TwoSet& ts = m.getDataSet().getTrainTwoSet();
	double area = 0;
	bool got = true;
	{
		util::ScreenCapture hush;
		try { area = ts.getTrapROCarea(); }
		catch ( ... ) { got = false; }
	}

	expect( got, string( who ) + ": a trapezoidal ROC area is available" );
	expect( got && area > 0.5, string( who )
		+ ": the area is ABOVE chance -- a flipped discriminant sense would "
		"put it below" );

	// And the separation is real on this fixture, not marginal: a sign error
	//    would give 1 - area, so requiring a clear margin makes the assertion
	//    sharp rather than nominal.
	expect( got && area > 0.75, string( who ) + ": the separation is clear ("
		+ to_string( area ) + ")" );
}

// --- 5. multi-output: a different report, and NO ROC ------------------------

template < class DFAMODEL >
static void test_multi_output( const char* who )
{
	cout << "-- " << who << ": three classes --" << endl;

	DataSet d = multiData();
	DFAMODEL m;
	string said = runDFA( m, d );

	expect( has( said, "3 output nodes" ),
		string( who ) + " multi: the header states three outputs" );
	expect( has( said, "Classification accuracy in the training set" ),
		string( who ) + " multi: the training accuracy line is present" );
	expect( has( said, "Classification accuracy in the test set" ),
		string( who ) + " multi: the test accuracy line is present" );

	// THE AVAILABILITY PIN: a multi-output run does NOT go through
	//    DataSet::metricsReport, so there is no classification table and no ROC.
	//    That is a property of the report, not an accident of this fixture.
	expect( !has( said, "Training set:" ),
		string( who ) + " multi: no TwoSet metrics report" );
	expect( !has( said, "Az" ) && !has( said, "area" ),
		string( who ) + " multi: no ROC statistics" );

	// The accuracy must beat chance (1/3) by a wide margin on separated clouds:
	//    a broken class assignment would sit near 33%.
	size_t p = said.find( "Classification accuracy in the training set = " );
	double acc = -1;
	if ( p != string::npos )
		acc = atof( said.c_str() + p
			+ string( "Classification accuracy in the training set = " ).size() );
	expect( acc > 60.0, string( who ) + " multi: training accuracy "
		+ to_string( acc ) + "% beats chance" );
}

// --- 6. saving the guesses --------------------------------------------------

template < class DFAMODEL >
static void test_save_guesses( const char* who, const char* stem )
{
	cout << "-- " << who << ": saving the guesses --" << endl;

	DataSet d = binaryData();
	DFAMODEL m;
	runDFA( m, d );

	string trainPath = tmp( ( string( stem ) + "_train.txt" ).c_str() ),
		testPath = tmp( ( string( stem ) + "_test.txt" ).c_str() );

	bool okTrain, okTest;
	{
		util::ScreenCapture hush;
		okTrain = m.getDataSet().saveTrainTwoSet( trainPath );
		okTest = m.getDataSet().saveTestTwoSet( testPath );
	}

	// EXISTENCE EVIDENCE FIRST. Two failed saves both slurp to "" and every
	//    comparison after that is vacuous.
	expect( okTrain, string( who ) + ": saving the training guesses reports success" );
	expect( okTest, string( who ) + ": saving the test guesses reports success" );

	string trainBytes = slurp( trainPath ), testBytes = slurp( testPath );
	expect( !trainBytes.empty(), string( who ) + ": the training guess file has content" );
	expect( !testBytes.empty(), string( who ) + ": the test guess file has content" );

	unsigned lines = 0;
	for ( size_t i = 0; i < trainBytes.size(); i++ )
		if ( trainBytes[ i ] == '\n' )
			lines++;
	expect( lines == 120, string( who ) + ": one saved line per exemplar (got "
		+ to_string( lines ) + ")" );

	remove( trainPath.c_str() );
	remove( testPath.c_str() );
}

// --- 7. history and the last-operation file ---------------------------------

template < class DFAMODEL >
static void test_history_and_lastop( const char* who, const char* stem )
{
	cout << "-- " << who << ": history and last-operation --" << endl;

	string hist = tmp( ( string( stem ) + "_hist.txt" ).c_str() ),
		lastop = tmp( ( string( stem ) + "_lastop.txt" ).c_str() );

	// Both ON
	{
		DataSet d = binaryData();
		DFAMODEL m;
		util::ScreenCapture hush;
		m.setDataSet( d );
		m.setHistoryFilename( hist );
		m.setHistory( true );
		m.setLastopFilename( lastop );
		m.setLastop( true );
		m.train();
	}

	string h = slurp( hist ), l = slurp( lastop );
	expect( !h.empty(), string( who ) + ": the history file has content" );
	expect( !l.empty(), string( who ) + ": the last-operation file has content" );
	expect( has( l, "Classification accuracy" ), string( who )
		+ ": the last-operation file holds the report, not just a stub" );

	remove( hist.c_str() );
	remove( lastop.c_str() );

	// Both OFF: nothing is written at all. Asserted by ABSENCE of the file,
	//    after deleting it above -- an empty string from a missing file would
	//    otherwise be indistinguishable from a file written empty.
	{
		DataSet d = binaryData();
		DFAMODEL m;
		util::ScreenCapture hush;
		m.setDataSet( d );
		m.setHistoryFilename( hist );
		m.setHistory( false );
		m.setLastopFilename( lastop );
		m.setLastop( false );
		m.train();
	}

	expect( !std::filesystem::exists( hist ), string( who )
		+ ": history off writes no history file" );
	expect( !std::filesystem::exists( lastop ), string( who )
		+ ": last-operation off writes no last-operation file" );
}

// --- 8. the refusal paths ---------------------------------------------------

template < class DFAMODEL >
static void test_singular( const char* who, const char* message )
{
	cout << "-- " << who << ": a singular covariance --" << endl;

	DataSet d = collinearData();
	DFAMODEL m;
	string said = runDFA( m, d );

	expect( has( said, message ), string( who )
		+ ": a singular covariance is refused with \"" + message + "\"" );
	expect( !has( said, "Classification accuracy" ), string( who )
		+ ": and no accuracy is reported from an unfitted model" );
}

template < class DFAMODEL >
static void test_not_discrete( const char* who )
{
	cout << "-- " << who << ": a continuous outcome --" << endl;

	Matrix< double > raw( 40, 3 );
	for ( unsigned i = 0; i < 40; i++ )
	{
		raw( i, 0 ) = i * 0.05;
		raw( i, 1 ) = ( i % 7 ) * 0.1;
		raw( i, 2 ) = i * 0.02; // continuous outcome
	}
	DataSet d;
	d.setInput( 2 );
	d.setOutput( 1 );
	d.setDiscrete( false );
	d.setHistory( false );

	string said;
	{
		util::ScreenCapture cap;
		d.setRawMatrix( raw );
		d.setTrainMatrix( raw );
		DFAMODEL m;
		m.setDataSet( d );
		said = cap.text();
	}

	expect( has( said, "NOT discrete" ), string( who )
		+ ": a non-discrete outcome refuses to construct the model" );
}

// --- 9. a standalone analysis does not disturb a trained model --------------
//
// The GUI keeps dfaPtr separate from modelPtr and says so in the page. The
// ENGINE-level fact underneath that is this: a DFA takes its own copy of the
// DataSet, writes its guesses into that copy, and touches nothing the network
// owns. If DFA ever held a reference instead, a discriminant analysis would
// silently overwrite the trained model's stored guesses.

static void test_standalone()
{
	cout << "-- a standalone DFA leaves a trained network alone --" << endl;

	DataSet d = binaryData();

	SimpleProp net;
	{
		util::ScreenCapture hush;
		net.setDataSet( d );
		net.setHidden( 4 );
		net.setHistory( false );
		net.setLastop( false );
		net.setLogPrint( false );
		net.setQuiet( true );
		net.setXEerror();
		net.setMaxIterations( 40 );
		util::set_seed( 11 );
		net.randomize();
		net.train();
	}

	// The network's own guesses, after training
	TwoSet& netTs = net.getDataSet().getTrainTwoSet();
	expect( netTs.loaded(), "the network has a loaded TwoSet to protect" );
	vector< double > before;
	for ( unsigned r = 0; r < 120; r++ )
		before.push_back( netTs.test( r ) );

	// A standalone LDFA on the NETWORK'S OWN DataSet -- the strong form of the
	//    question, and the one a caller can actually reach: the GUI hands
	//    dfaPtr a DataSet that came from the same place modelPtr's did. What
	//    protects the network is that Model::setDataSet takes a COPY. If DFA
	//    ever held a reference or a pointer instead, this analysis would
	//    overwrite the trained network's stored guesses in place.
	LDFA dfa;
	runDFA( dfa, net.getDataSet() );

	vector< double > after;
	for ( unsigned r = 0; r < 120; r++ )
		after.push_back( net.getDataSet().getTrainTwoSet().test( r ) );

	expect( before == after,
		"the network's guesses are untouched by the analysis" );

	// And the analysis really did produce its own, different guesses -- without
	//    this the comparison above could pass because nothing happened at all.
	vector< double > dfaGuesses;
	for ( unsigned r = 0; r < 120; r++ )
		dfaGuesses.push_back( dfa.getDataSet().getTrainTwoSet().test( r ) );
	expect( dfaGuesses != before,
		"...and the analysis did compute its own, different guesses" );
}

// --- 10. running an analysis twice ------------------------------------------
//
// Characterization, not a claim: what does a second train() on the same object
// do? Recorded because the extraction will move this orchestration, and a
// difference here must be preserved or fixed deliberately rather than by
// accident.

template < class DFAMODEL >
static void test_train_twice( const char* who )
{
	cout << "-- " << who << ": training twice --" << endl;

	DataSet d = binaryData();
	DFAMODEL m;
	string first = runDFA( m, d );

	string second;
	{
		util::ScreenCapture cap;
		m.train();
		second = cap.text();
	}

	expect( first == second, string( who )
		+ ": a second run of the same analysis reports the same thing" );
}

template < class DFAMODEL >
static void test_train_twice_multi( const char* who )
{
	cout << "-- " << who << ": training twice, three classes --" << endl;

	DataSet d = multiData();
	DFAMODEL m;
	string first = runDFA( m, d );

	string second;
	{
		util::ScreenCapture cap;
		m.train();
		second = cap.text();
	}

	expect( first == second, string( who )
		+ " multi: a second run of the same analysis reports the same thing" );
}

int main()
{
	test_report< LDFA >( "LDFA", "I'm running LDFA:" );
	test_report< QDFA >( "QDFA", "I'm running QDFA:" );

	test_graded_guesses< LDFA >( "LDFA" );
	test_graded_guesses< QDFA >( "QDFA" );

	test_direction< LDFA >( "LDFA" );
	test_direction< QDFA >( "QDFA" );

	test_multi_output< LDFA >( "LDFA" );
	test_multi_output< QDFA >( "QDFA" );

	test_save_guesses< LDFA >( "LDFA", "ldfa" );
	test_save_guesses< QDFA >( "QDFA", "qdfa" );

	test_history_and_lastop< LDFA >( "LDFA", "ldfa" );
	test_history_and_lastop< QDFA >( "QDFA", "qdfa" );

	test_singular< LDFA >( "LDFA", "Can't do LDFA" );
	test_singular< QDFA >( "QDFA", "Can't do QDFA" );

	test_not_discrete< LDFA >( "LDFA" );
	test_not_discrete< QDFA >( "QDFA" );

	test_standalone();

	test_train_twice< LDFA >( "LDFA" );
	test_train_twice< QDFA >( "QDFA" );
	test_train_twice_multi< LDFA >( "LDFA" );
	test_train_twice_multi< QDFA >( "QDFA" );

	cout << endl << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
