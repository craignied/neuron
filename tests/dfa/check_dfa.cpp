// check_dfa.cpp : CHARACTERIZATION of LDFA and QDFA, before they share anything.
//
// docs/refactor_audit.md items 2.4 and 2.5 propose extracting the reporting scaffold
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
#include <memory>
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

#include "dfa.h"
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

// A dataset with a ZERO-VARIANCE input column: the covariance matrix has an
//    exactly-zero row, on every platform, so the inverse must refuse.
//
// THIS REPLACES A FIXTURE THAT WAS NOT PORTABLY SINGULAR, and the difference is
// the whole point. The first version made column 1 exactly 2x column 0 --
// singular in exact arithmetic, but with no zero row, so detection depended on
// the elimination arriving at exactly 0. `ludcmp` has two singularity tests
// (matrix.cpp): a structural one, `big == 0`, when a whole row is zero; and an
// arithmetic one, `L( j, j ) == 0`, after eliminating. The collinear fixture
// could only reach the SECOND, and whether the residual cancels to exactly zero
// depends on how the covariance sums were accumulated -- FP contraction into
// FMA, vectorization and reassociation all differ between clang on arm64,
// gcc on x86-64 and MSVC. On macOS it cancelled and LDFA refused; on Ubuntu it
// did not, LDFA inverted a nearly-singular matrix, and the assertion failed. CI
// caught what one machine could not.
//
// A constant column reaches the FIRST test instead. Every deviation from the
// column mean is exactly 0.0, so every covariance product with that column is
// exactly 0.0 and every sum of them is exactly 0.0 -- true under any IEEE
// rounding mode, with or without FMA, in any summation order, because 0*x = 0
// and 0+0 = 0 are exact. The value 0.0 is used rather than any other constant so
// that the column's sum is exactly 0 for ANY row count, making the mean exactly
// 0 without needing an argument about representability. (Measured on this
// fixture: V(0,0) = 0.225009..., V(0,1) = V(1,1) = 0 exactly.)
//
// It is also a real degeneracy rather than a contrivance: a predictor that is
// identically zero in the training data -- a rare indicator with no positive
// rows -- is the same case DataSet::normalize documents as bug B4.
static DataSet singularData( unsigned n = 60 )
{
	Matrix< double > raw( n, 3 );
	for ( unsigned i = 0; i < n; i++ )
	{
		double t = ( ( i * 31 ) % 71 ) / 70.0;
		raw( i, 0 ) = t + ( i % 2 ? 0.8 : 0.0 ); // separates the classes
		raw( i, 1 ) = 0.0;                       // ZERO VARIANCE, exactly
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

// THE FIXTURE PROVES ITSELF, on whatever platform is running. If this ever
// fails, the two refusal cases below are testing nothing and say so here first,
// rather than passing for the wrong reason or failing mysteriously as the
// collinear fixture did on Ubuntu.
static void test_fixture_is_singular()
{
	cout << "-- the singular fixture is exactly singular --" << endl;

	DataSet d = singularData();
	Matrix< double >& T = d.getTrainMatrix();
	Matrix< double > inputs = T.submatrix( 0, T.rows() - 1, 0, 1 );
	Matrix< double > V = inputs.covariance();

	// EXACTLY zero, not nearly: the constant column contributes 0*x to every
	//    product and 0 to every sum, which no rounding mode can disturb.
	expect( V( 1, 0 ) == 0.0 && V( 1, 1 ) == 0.0,
		"the covariance row of the zero-variance column is exactly zero" );
	expect( V( 0, 0 ) > 0.0,
		"...while the other column still carries variance" );

	bool threwSingular = false, threwOther = false;
	try { Matrix< double > I = V.inverse(); ( void ) I; }
	catch ( Matrix< double >::Singular& ) { threwSingular = true; }
	catch ( ... ) { threwOther = true; }

	expect( threwSingular, "inverting it throws Matrix::Singular" );
	expect( !threwOther, "...and not some other exception, which would mean the "
		"refusal below is testing the wrong mechanism" );
}

template < class DFAMODEL >
static void test_singular( const char* who, const char* message )
{
	cout << "-- " << who << ": a singular covariance --" << endl;

	DataSet d = singularData();
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


// --- 11. INTERNAL STATE MUST STAY BOUNDED, and a reload must not be stale ----
//
// The characterization above proves that repeated reports are IDENTICAL. It
// does not prove the object is still correct, and on this engine those are not
// the same thing: DFA::setDataSet() appends to D and U with push_back on every
// call, and QDFA::train() appends to C, S and K on every run. Every reader then
// indexes [ 0 .. nOutput - 1 ], so the leading entries stay the FIRST load's
// class matrices and means. Repeated runs on unchanged data therefore report
// the same numbers -- test_train_twice_multi passes BECAUSE of the defect, not
// despite it -- while the object silently grows and, after a second load of
// different data, fits the wrong dataset entirely.
//
// The probe reaches the protected members through a subclass, which is the
// narrowest way to assert a size that has no accessor. It asserts nothing about
// values, only about COUNTS -- so it is exact on every platform.

template < class DFAMODEL >
class StateProbe : public DFAMODEL {
public:
	size_t classMatrices() const { return this->D.size(); }
	size_t meanVectors() const { return this->U.size(); }
	size_t constants() const { return this->K.size(); }
};

template < class DFAMODEL >
static void test_state_is_bounded( const char* who, unsigned nOutput )
{
	cout << "-- " << who << ": internal state stays bounded --" << endl;

	DataSet d = multiData();
	StateProbe< DFAMODEL > m;
	{
		util::ScreenCapture hush;
		m.setDataSet( d );
		m.setHistory( false );
		m.setLastop( false );
	}

	expect( m.classMatrices() == nOutput && m.meanVectors() == nOutput,
		string( who ) + ": one class matrix and one mean vector per class after "
		"the first load" );

	// A SECOND load of the same data. An object that accumulates has 2n here.
	{
		util::ScreenCapture hush;
		m.setDataSet( d );
	}
	expect( m.classMatrices() == nOutput, string( who )
		+ ": a second load leaves " + to_string( nOutput ) + " class matrices, not "
		+ to_string( m.classMatrices() ) );
	expect( m.meanVectors() == nOutput, string( who )
		+ ": a second load leaves " + to_string( nOutput ) + " mean vectors, not "
		+ to_string( m.meanVectors() ) );

	// Three runs. An object that accumulates has 3n constants here.
	{
		util::ScreenCapture hush;
		m.train(); m.train(); m.train();
	}
	expect( m.constants() == nOutput, string( who )
		+ ": three runs leave " + to_string( nOutput ) + " constants, not "
		+ to_string( m.constants() ) );
}

// A genuinely DIFFERENT three-class dataset. Two things it must satisfy, and
//    the second is one my first attempt missed: the classes must sit ELSEWHERE
//    (so a model fitted on the other dataset scores near chance), AND a correct
//    fit must produce a DIFFERENT REPORT. My first version separated just as
//    cleanly, so both datasets reported 100% / 100% and the whole comparison
//    would have been satisfied by two identical strings. The vacuity guard
//    below caught it. So these classes OVERLAP: a correct fit lands well short
//    of 100%, which is a number dataset A cannot produce.
static DataSet otherMultiData( unsigned n = 150 )
{
	Matrix< double > raw( n, 5 );
	for ( unsigned i = 0; i < n; i++ )
	{
		unsigned c = i % 3;
		double t = ( ( i * 41 ) % 89 ) / 88.0, u = ( ( i * 59 ) % 83 ) / 82.0;
		raw( i, 0 ) = -0.45 * c + 1.3 * ( t - 0.5 );
		raw( i, 1 ) = ( c == 2 ? -0.35 : 0.3 ) + 1.3 * ( u - 0.5 );
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

// A reused object must answer for the data it was LAST given. Compared against
//    a fresh object on the same data -- the only definition of "correct" that
//    does not require pinning a number.
//
// WHAT THIS DOES AND DOES NOT COVER, measured rather than assumed. It catches
//    the D / U accumulation in DFA::setDataSet: removing that reset makes both
//    models report their previous dataset's model and this assertion fails for
//    both. It does NOT catch QDFA's C / S / K accumulation on its own --
//    measured, with only that reset removed the reused object still reports 66%
//    on this fixture, exactly what a fresh object reports, because the stale
//    covariance inverses combine with FRESH means and happen not to change the
//    argmin ordering for these rows. The guard for that half is the bounded
//    state count above, which is an integer and fails deterministically. Two
//    mechanisms, two different assertions; neither is a spare.
template < class DFAMODEL >
static void test_reload_is_not_stale( const char* who )
{
	cout << "-- " << who << ": a reload is not stale --" << endl;

	DataSet a = multiData(), b = otherMultiData();

	DFAMODEL reused;
	string onA, onB;
	{
		util::ScreenCapture cap;
		reused.setDataSet( a );
		reused.setHistory( false );
		reused.setLastop( false );
		reused.train();
		onA = cap.text();
	}
	{
		util::ScreenCapture cap;
		reused.setDataSet( b );
		reused.train();
		onB = cap.text();
	}

	DFAMODEL fresh;
	string freshOnB;
	{
		util::ScreenCapture cap;
		fresh.setDataSet( b );
		fresh.setHistory( false );
		fresh.setLastop( false );
		fresh.train();
		freshOnB = cap.text();
	}

	// EXISTENCE FIRST: three empty reports would compare equal and prove nothing.
	expect( !onA.empty() && !onB.empty() && !freshOnB.empty(),
		string( who ) + ": all three runs produced a report" );

	// The fixtures must actually differ, or the comparison below is vacuous.
	expect( onA != freshOnB, string( who )
		+ ": the two datasets give different reports, so the test can see a difference" );

	expect( onB == freshOnB, string( who )
		+ ": a reused object reports what a fresh object reports on the same data" );
}


// --- 12. QDFA's COVARIANCE state, guarded behaviourally ---------------------
//
// The bounded-state probe above reaches D, U and K, which are DFA's. It cannot
// reach C and S, which are QDFA's own private members -- and the state test
// would therefore stay green if C.clear() or S.clear() were deleted
// individually. This case closes that, without widening any production
// visibility, by choosing a fixture on which the covariance DECIDES the answer.
//
// THE FIXTURE IS THE POINT. Three classes with the SAME mean, separated only by
// the SHAPE of their spread: one wide in x and narrow in y, one the reverse, one
// moderate in both. That is the case quadratic discriminant analysis exists for
// and a linear one cannot touch. If S were stale, every class would share one
// covariance and only the constants would differ, so a single class would win
// every row.
//
// Measured, one reset removed at a time, against a fresh QDFA's 80%:
//
//     only C.clear() removed   reused reports 40.6%
//     only S.clear() removed   reused reports 41.7%
//     only K.clear() removed   reused reports 79.4%
//
// so the single comparison below fails for each of the three independently. The
// C and S margins are enormous (a stale covariance halves the accuracy); K's is
// one row, but K does not depend on this case -- it has the exact integer count
// assertion in test_state_is_bounded, and this is not its guard.
static DataSet covarianceShapeData( unsigned n = 180 )
{
	Matrix< double > raw( n, 5 );
	for ( unsigned i = 0; i < n; i++ )
	{
		unsigned c = i % 3;
		double t = ( ( i * 41 ) % 89 ) / 88.0 - 0.5,
			u = ( ( i * 59 ) % 83 ) / 82.0 - 0.5;
		double sx = ( c == 0 ) ? 1.8 : ( c == 1 ? 0.18 : 0.7 ),
			sy = ( c == 0 ) ? 0.18 : ( c == 1 ? 1.8 : 0.7 );
		raw( i, 0 ) = sx * t * 2.0; // same mean for every class, by construction
		raw( i, 1 ) = sy * u * 2.0;
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

static double trainingAccuracy( const string& report )
{
	const string key = "Classification accuracy in the training set = ";
	size_t p = report.find( key );
	return p == string::npos ? -1.0 : atof( report.c_str() + p + key.size() );
}

static void test_qdfa_covariance_not_stale()
{
	cout << "-- QDFA: a reload does not keep stale covariances --" << endl;

	DataSet meanSeparated = multiData(), shapeSeparated = covarianceShapeData();

	QDFA fresh;
	string freshReport;
	{
		util::ScreenCapture cap;
		fresh.setDataSet( shapeSeparated );
		fresh.setHistory( false );
		fresh.setLastop( false );
		fresh.train();
		freshReport = cap.text();
	}

	QDFA reused;
	string reusedReport;
	{
		util::ScreenCapture cap;
		reused.setDataSet( meanSeparated );
		reused.setHistory( false );
		reused.setLastop( false );
		reused.train();
	}
	{
		util::ScreenCapture cap;
		reused.setDataSet( shapeSeparated );
		reused.train();
		reusedReport = cap.text();
	}

	expect( !freshReport.empty() && !reusedReport.empty(),
		"QDFA covariance: both runs produced a report" );

	// DISTINGUISHABILITY FIRST. The comparison below is only meaningful if a
	//    correct fit can actually solve this fixture -- if it sat at chance,
	//    a stale covariance would sit there too and the assertion would pass
	//    while guarding nothing.
	double freshAcc = trainingAccuracy( freshReport );
	expect( freshAcc > 60.0, "QDFA covariance: a correct fit solves this fixture ("
		+ to_string( freshAcc ) + "%), well above the 33% chance a stale "
		"covariance collapses to" );

	expect( reusedReport == freshReport,
		"QDFA covariance: a reused object reports what a fresh object reports" );
}


// --- 13. THE POLYMORPHIC PATH, which is how cross-validation reaches these ---
//
// Model::train() is pure virtual (model.h), so DFA::train() -- shared or not --
// is an OVERRIDE and stays virtual. That polymorphism is not decorative:
// cvadapters::dfaProcedure holds a unique_ptr< Model > and fitQuietly( Model& )
// calls m.train() through that reference, so every cross-validation fold
// reaches the override virtually. A suite that only ever called train() on a
// concrete type could not tell a working override from a broken one.
//
// Written BEFORE the scaffold extraction and passing against the code that
// preceded it, so it characterizes the contract rather than the change.

template < class DFAMODEL >
static void test_through_base_reference( const char* who )
{
	cout << "-- " << who << ": driven through Model --" << endl;

	// A Model& , which is fitQuietly's parameter shape
	{
		DataSet d = binaryData();
		DFAMODEL concrete;
		Model& m = concrete;
		string said;
		{
			util::ScreenCapture cap;
			m.setDataSet( d );
			m.setHistory( false );
			m.setLastop( false );
			m.train();
			said = cap.text();
		}
		expect( has( said, string( "I'm running " ) + who ),
			string( who ) + ": a Model& reaches the right override" );
		expect( has( said, "Classification accuracy" ),
			string( who ) + ": ...and it runs to a full report" );
	}

	// A unique_ptr< Model >, which is cvadapters' shape, released through the
	//    base pointer -- so the virtual destructor is exercised too
	{
		DataSet d = binaryData();
		unique_ptr< Model > m( new DFAMODEL );
		string said;
		{
			util::ScreenCapture cap;
			m->setDataSet( d );
			m->setHistory( false );
			m->setLastop( false );
			m->train();
			said = cap.text();
		}
		expect( has( said, string( "I'm running " ) + who ),
			string( who ) + ": a unique_ptr< Model > reaches the right override" );
		expect( m->getType() == who,
			string( who ) + ": ...and reports its own type through the base" );
	}
}


// --- 14. THE SCAFFOLD PROBE: order, omission, and dispatch, with no numbers --
//
// DFA::train() is now shared, and what it must do is call the CONCRETE class's
// fitDiscriminant() exactly once and then the CONCRETE class's reportAccuracy()
// exactly once. Three ways that can break -- reversed, one omitted, or the
// override bypassed -- and none of them would be caught by looking at a report.
//
// It is tempting to test the ORDER by fitting nothing and asserting that the
// numbers come out wrong. That is not a test: reading a model that was never
// fitted reads unsized or uninitialised state, so the outcome is a wrong number
// on one platform, an exception on another, and undefined behavior in
// principle. This file already made that mistake once, in the singular fixture
// that passed on macOS and failed on Ubuntu.
//
// So the probe records CALLS, not values. A DFA subclass whose two overrides do
// nothing but append a token, driven through the shared scaffold, must produce
// exactly "fit,report". No floating-point arithmetic is involved anywhere, so
// the assertion is identical on every platform -- and because the tokens are
// appended by the OVERRIDES, a scaffold that bypassed virtual dispatch would
// produce an empty log rather than a wrong number.

class ScaffoldProbe : public DFA {
public:
	string log;
	unsigned fits = 0, reports = 0;

protected:
	void fitDiscriminant() override
	{
		fits++;
		log += log.empty() ? "fit" : ",fit";
	}

public:
	void reportAccuracy( ostream& ) override
	{
		reports++;
		log += log.empty() ? "report" : ",report";
	}
};

static void test_scaffold_calls_in_order()
{
	cout << "-- the scaffold fits once, then reports once --" << endl;

	DataSet d = binaryData();
	ScaffoldProbe p;
	double returned;
	{
		util::ScreenCapture hush;
		p.setDataSet( d );
		p.setHistory( false );
		p.setLastop( false );
		returned = p.train();
	}

	expect( p.log == "fit,report", "the scaffold calls fitDiscriminant then "
		"reportAccuracy -- got \"" + p.log + "\"" );
	expect( p.fits == 1, "fitDiscriminant runs exactly once" );
	expect( p.reports == 1, "reportAccuracy runs exactly once" );
	expect( returned == -1, "the scaffold returns -1, as a discriminant analysis "
		"has no set error" );

	// And through a Model& -- the shape cross-validation uses -- so a scaffold
	//    that somehow lost its virtual reach is caught here too.
	DataSet d2 = binaryData();
	ScaffoldProbe q;
	{
		util::ScreenCapture hush;
		Model& m = q;
		m.setDataSet( d2 );
		m.setHistory( false );
		m.setLastop( false );
		m.train();
	}
	expect( q.log == "fit,report",
		"...and the same through a Model& -- got \"" + q.log + "\"" );
}

int main()
{
	// FIRST, deliberately. It reads no numerical state, so it survives a
	//    scaffold that would make every later case abort -- reversing the two
	//    calls makes the concrete models report a model they have not fitted,
	//    and the run dies. Judged first, the structural break is attributed to
	//    the structure rather than to whatever happens to crash.
	test_scaffold_calls_in_order();

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

	test_fixture_is_singular();
	test_singular< LDFA >( "LDFA", "Can't do LDFA" );
	test_singular< QDFA >( "QDFA", "Can't do QDFA" );

	test_not_discrete< LDFA >( "LDFA" );
	test_not_discrete< QDFA >( "QDFA" );

	test_standalone();

	test_train_twice< LDFA >( "LDFA" );
	test_train_twice< QDFA >( "QDFA" );
	test_train_twice_multi< LDFA >( "LDFA" );
	test_train_twice_multi< QDFA >( "QDFA" );

	test_state_is_bounded< LDFA >( "LDFA", 3 );
	test_state_is_bounded< QDFA >( "QDFA", 3 );

	test_reload_is_not_stale< LDFA >( "LDFA" );
	test_reload_is_not_stale< QDFA >( "QDFA" );

	test_qdfa_covariance_not_stale();

	test_through_base_reference< LDFA >( "LDFA" );
	test_through_base_reference< QDFA >( "QDFA" );

	cout << endl << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
