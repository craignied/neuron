// check_regressnet.cpp : CHARACTERIZATION of stepwise regression, before
// reverse_regress() and forward_regress() share anything.
//
// docs/refactor_audit.md item 1.3 proposes collapsing the two ~200-line selection
// loops into one. Measured (audit section 15.1): 83 of reverse's 128 executable
// lines are byte-identical to forward's. The remaining 45 are not decoration --
// they are the published direction semantics:
//
//   * REVERSE compares the candidate against the network it came FROM. The
//     candidate is nested WITHIN the prior model, so Wilks takes
//     ( N, lastError, subError ) and df is last_df - candidate_df. The variable
//     with the LARGEST p-value -- the least significant -- is removed.
//   * FORWARD compares the network it came from against the candidate. The
//     PRIOR is nested within the candidate, so Wilks takes
//     ( N, fullError, lastError ) and df is candidate_df - last_df. The
//     variable with the SMALLEST p-value -- the most significant -- is added.
//     Its baseline is a TRAINED empty network, not the source model's error.
//
// Those are two published procedures (manifest chapter "Stepwise regression",
// sections revreg and forreg) with opposite senses and opposite nesting. They
// may not become a sign flag, a comparator object, or a generic "direction"
// index in a shared layer (standing rule 7). Only orchestration proven
// identical may be shared. Cases 20-24 are the ones that would catch it: a
// swapped Wilks argument order or a flipped winner sense produces a report that
// still looks entirely reasonable, with the wrong variables selected.
//
// PORTABLE BY CONSTRUCTION, as check_dfa and check_autostep are. No
// multi-iteration floating-point literal is asserted. What is pinned is
// structure (which lines a report contains), integers (candidate counts,
// degrees of freedom, orderings), construction-exact values (the source's own
// error handed straight back as a candidate's prior), same-process relations
// (this run equals that run, this is unchanged, this is larger than that), and
// inequalities a sign error would break.
//
// NOT ASSERTED HERE, DELIBERATELY: reverse_regress() reads an uninitialized
// `largest_var` when every candidate in a pass has p == 0, which crashes the
// process when the threshold is 0. That is a live defect, measured and recorded
// in docs/refactor_audit.md section 15.6, and it is scheduled as its own fail-first
// correctness commit BEFORE the extraction. A characterization suite pins what
// the code does today; it must not pin undefined behavior, and it must not
// quietly assert the fix either.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

#include "dataset.h"
#include "logistic.h"
#include "regressnet.h"
#include "simpleprop.h"
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
//    concurrent runs (the check_onehidden lesson, standing rule 2).
class TempDir {
public:
	TempDir()
	{
		namespace fs = std::filesystem;
		root = fs::temp_directory_path()
			/ ( "neuron_regress_" + to_string( ( long ) getpid() ) );
		fs::create_directories( root );
	}
	~TempDir()
	{
		std::error_code ec;
		std::filesystem::remove_all( root, ec );
	}
	string file( const char* name ) const { return ( root / name ).string(); }
private:
	std::filesystem::path root;
};

static TempDir tempDir;
static string tmp( const char* name ) { return tempDir.file( name ); }

static string slurp( const string& path )
{
	ifstream in( path.c_str(), ios::binary );
	ostringstream all;
	all << in.rdbuf();
	return all.str();
}

static bool exists( const string& path )
{
	return std::filesystem::exists( path );
}

// Strip carriage returns, for the ONE comparison that puts a file written
//    through a text-mode ofstream beside a string held in memory.
//
//    RegressNet::report() opens the history file with ios::out | ios::app and
//    no ios::binary, so on Windows the runtime translates every '\n' it writes
//    into "\r\n" -- which is correct behavior for a text log, and is what a
//    Windows user's editor expects. slurp() reads binary, deliberately, because
//    the model-file comparisons elsewhere in this file are byte-identity claims
//    and must not have any translation applied to them. Those compare two files
//    written the same way, so they need no normalization and get none.
//
//    Case 46e is the odd one out: a file against a std::string. What it claims
//    is that the history file carries the same TEXT as the screen report, and
//    that claim is still exact after this -- any difference in a character, a
//    number or a line survives it. Only the platform's line-ending convention
//    is forgiven. (Measured: this is what failed CI on windows-latest and
//    passed on macOS and Ubuntu, at 85dff61.)
static string sameNewlines( const string& s )
{
	string out;
	out.reserve( s.size() );
	for ( size_t i = 0; i < s.size(); i++ )
		if ( s[ i ] != '\r' )
			out += s[ i ];
	return out;
}

static bool has( const string& hay, const string& needle )
{
	return hay.find( needle ) != string::npos;
}

// --- the fixtures ----------------------------------------------------------
//
// Built without an RNG so every run of every build sees the same numbers.
// Labels are thresholded on a linear score plus a deterministic pseudo-noise
// term: the noise is what keeps the fit NON-SEPARABLE, and a separable fixture
// is not a nuisance to be tuned away -- it drives the logistic weights to
// infinity and the candidate error to NaN, which is a different case entirely
// (case 30 uses exactly that on purpose).

// Four variables: 0 strongly predictive, 1 moderately, 2 and 3 pure noise.
// Measured on this fixture, all candidate p-values are distinct and they
// straddle 0.05 by many orders of magnitude, so both directions have a real
// decision to make. Case 1 asserts that rather than assuming it.
static DataSet mixedData( unsigned n = 800 )
{
	Matrix< double > raw( n, 5 );
	for ( unsigned i = 0; i < n; i++ )
	{
		double a = ( ( i * 37 ) % 101 ) / 100.0 - 0.5;
		double b = ( ( i * 53 ) % 97 ) / 96.0 - 0.5;
		double c = ( ( i * 71 ) % 89 ) / 88.0 - 0.5;
		double d = ( ( i * 43 ) % 83 ) / 82.0 - 0.5;
		double noise = ( ( i * 29 ) % 71 ) / 70.0 - 0.5;
		raw( i, 0 ) = a; raw( i, 1 ) = b; raw( i, 2 ) = c; raw( i, 3 ) = d;
		double z = 2.4 * a + 1.1 * b + 2.0 * noise; // c and d carry NO signal
		raw( i, 4 ) = ( z > 0 ) ? 1 : 0;
	}
	DataSet s;
	s.setInput( 4 ); s.setOutput( 1 ); s.setDiscrete( true ); s.setHistory( false );
	util::ScreenCapture hush;
	s.setRawMatrix( raw );
	s.randomize( 0 );
	return s;
}

// Four variables, EVERY ONE of them strongly predictive, over enough rows that
// every candidate's chi-square underflows its p-value to exactly 0.0.
//
// This is the ordinary case in a well-powered study, not a contrivance:
// stats::pX2 is gammq, whose gcf branch returns exp( -x + a ln x - lngamma a )
// with x = G2/2. Measured on this fixture the weakest candidate has
// G2 = 1557.5, so that exponent is about -776 -- and exp underflows to exactly
// zero below about -745 in IEEE double, on every platform. The margin is ~31 in
// the exponent, i.e. the value would have to move by a factor of 10^13 to stop
// being zero. That is a property of double's range, not of one toolchain's
// arithmetic, which is what separates this fixture from the non-finite spike
// rejected in audit section 15.7.
//
// Case 50 asserts the underflow rather than assuming it: if a platform ever
// stopped producing exact zeros here, every case after it would be testing
// nothing, and it should say so loudly instead.
static DataSet allSignificantData( unsigned n = 6000 )
{
	Matrix< double > raw( n, 5 );
	for ( unsigned i = 0; i < n; i++ )
	{
		double a = ( ( i * 37 ) % 101 ) / 100.0 - 0.5;
		double b = ( ( i * 53 ) % 97 ) / 96.0 - 0.5;
		double c = ( ( i * 71 ) % 89 ) / 88.0 - 0.5;
		double d = ( ( i * 43 ) % 83 ) / 82.0 - 0.5;
		double noise = ( ( i * 29 ) % 71 ) / 70.0 - 0.5;
		raw( i, 0 ) = a; raw( i, 1 ) = b; raw( i, 2 ) = c; raw( i, 3 ) = d;
		double z = 2.2 * a + 2.0 * b + 1.9 * c + 1.8 * d + 1.6 * noise;
		raw( i, 4 ) = ( z > 0 ) ? 1 : 0;
	}
	DataSet s;
	s.setInput( 4 ); s.setOutput( 1 ); s.setDiscrete( true ); s.setHistory( false );
	util::ScreenCapture hush;
	s.setRawMatrix( raw );
	s.randomize( 0 );
	return s;
}

// Three variables, none of which carries any signal at all: the label depends
// only on a term absent from the inputs. A forward run over this must admit
// NOTHING, which is a completed analysis with an empty result -- not a failure.
static DataSet noiseData( unsigned n = 600 )
{
	Matrix< double > raw( n, 4 );
	for ( unsigned i = 0; i < n; i++ )
	{
		raw( i, 0 ) = ( ( i * 37 ) % 101 ) / 100.0 - 0.5;
		raw( i, 1 ) = ( ( i * 53 ) % 97 ) / 96.0 - 0.5;
		raw( i, 2 ) = ( ( i * 71 ) % 89 ) / 88.0 - 0.5;
		double hidden = ( ( i * 29 ) % 71 ) / 70.0 - 0.5; // not an input
		raw( i, 3 ) = ( hidden > 0 ) ? 1 : 0;
	}
	DataSet s;
	s.setInput( 3 ); s.setOutput( 1 ); s.setDiscrete( true ); s.setHistory( false );
	util::ScreenCapture hush;
	s.setRawMatrix( raw );
	s.randomize( 0 );
	return s;
}

// Five input NODES making three conceptual VARIABLES: {0}, {1,2}, {3,4}.
// Nodes 1 and 2 are two halves of one predictor and must move together; a
// procedure that split them would produce a df of 1 where 2 is correct.
static DataSet groupedData( unsigned n = 800 )
{
	Matrix< double > raw( n, 6 );
	for ( unsigned i = 0; i < n; i++ )
	{
		double a = ( ( i * 37 ) % 101 ) / 100.0 - 0.5;
		double b1 = ( ( i * 53 ) % 97 ) / 96.0 - 0.5;
		double b2 = ( ( i * 61 ) % 79 ) / 78.0 - 0.5;
		double c1 = ( ( i * 71 ) % 89 ) / 88.0 - 0.5;
		double c2 = ( ( i * 43 ) % 83 ) / 82.0 - 0.5;
		double noise = ( ( i * 29 ) % 71 ) / 70.0 - 0.5;
		raw( i, 0 ) = a; raw( i, 1 ) = b1; raw( i, 2 ) = b2;
		raw( i, 3 ) = c1; raw( i, 4 ) = c2;
		// variable 1 ( nodes 1 and 2 ) carries signal through BOTH its nodes;
		//    variable 2 ( nodes 3 and 4 ) carries none
		double z = 2.0 * a + 1.3 * b1 + 1.3 * b2 + 2.0 * noise;
		raw( i, 5 ) = ( z > 0 ) ? 1 : 0;
	}
	DataSet s;
	s.setInput( 5 ); s.setOutput( 1 ); s.setDiscrete( true ); s.setHistory( false );
	util::ScreenCapture hush;
	s.setRawMatrix( raw );
	s.randomize( 0 );
	return s;
}

// --- the source models -----------------------------------------------------

// THE SOURCE MODEL IS A PLAIN Logistic, NOT A TEST SUBCLASS. RegressNet clones
//    every candidate through cloneNetwork(), which dispatches on typeid -- a
//    subclass introduced merely to expose weights is an unknown type, every
//    clone comes back null, and the analysis refuses before it starts. So the
//    model's state is read the way the rest of the program reads it: through
//    the file Network::save() writes, which is architecture plus every weight.
static string saveModel( Network& net, const char* name )
{
	string path = tmp( name );
	util::ScreenCapture hush;
	net.save( path );
	return path;
}

// One configuration, used for every source fit, so no case can differ from
//    another by accident. Gradient descent with automatic step size is what the
//    API's algorithm=1 selects.
static double fitSource( Logistic& net, DataSet& d, unsigned maxIterations = 5000 )
{
	net.setDataSet( d );
	net.setHistory( false );
	net.setLastop( false );
	net.setLogPrint( false );
	net.setQuiet( true );
	net.setAutoStepSize( true );
	net.setTrainingType( 0 );
	net.setGradStop( true );
	net.setMaxIterations( maxIterations );
	util::set_seed( 42 );
	net.randomize();
	util::ScreenCapture hush;
	return net.train();
}

static vector< vector< unsigned > > singles( unsigned n )
{
	vector< vector< unsigned > > v;
	for ( unsigned i = 0; i < n; i++ )
		v.push_back( vector< unsigned >( 1, i ) );
	return v;
}

// One stepwise run, with everything a caller can observe collected in one
//    place. `threw` is empty on a completed analysis.
struct Run {
	string report;
	string threw;
	bool complete;
	unsigned fits;
	vector< RegressNet::Candidate > candidates;
	vector< pair< double, unsigned > > path;
	vector< unsigned > finalVariables;
	vector< RegressNet::Progress > progress;
};

static Run regress( Network& net, double e_in,
	const vector< vector< unsigned > >& defs, bool forward, double threshold,
	Iterative::Observer* obs = 0, bool watchProgress = true )
{
	Run r;
	RegressNet analysis;
	analysis.setNetwork( &net, e_in );
	analysis.setInputStructure( defs );
	analysis.setThreshold( threshold );
	if ( obs )
		analysis.setObserver( obs );
	if ( watchProgress )
		analysis.setProgress( [ &r ]( const RegressNet::Progress& p )
			{ r.progress.push_back( p ); } );

	{
		util::ScreenCapture hush;
		try
		{
			if ( forward )
				analysis.forward_regress();
			else
				analysis.reverse_regress();
		}
		catch ( exception& e )
		{
			r.threw = e.what();
		}
		r.report = hush.text();
	}

	r.complete = analysis.getComplete();
	r.fits = analysis.getFitsCompleted();
	r.candidates = analysis.getCandidates();
	r.path = analysis.getSelectionPath();
	r.finalVariables = analysis.getFinalVariables();
	return r;
}

// The variables of a run's selection path, in the order the procedure took it
static vector< unsigned > pathVars( const Run& r )
{
	vector< unsigned > v;
	for ( size_t i = 0; i < r.path.size(); i++ )
		v.push_back( r.path[ i ].second );
	return v;
}

// ---------------------------------------------------------------------------
// 1. THE FIXTURE ITSELF: distinguishable candidates and a real winner.
//
// Everything below compares selections, orderings and winners. If the fixture
// produced candidates that were indistinguishable -- equal p-values, or all of
// them on the same side of the threshold -- then "the right variable won" would
// be satisfied by any implementation at all, and every case after this one
// would be vacuous. This is the guard, and it runs first.
// ---------------------------------------------------------------------------
static void test_fixture_is_discriminating()
{
	DataSet d = mixedData();
	Logistic net;
	double e_in = fitSource( net, d );

	expect( Iterative::converged( net.getStopReason() ),
		"1a: the source fit converges, so the analysis has a valid baseline" );
	expect( std::isfinite( e_in ) && e_in > 0,
		"1b: the source error is a finite positive number" );

	Run r = regress( net, e_in, singles( 4 ), false, 0.05 );
	expect( r.threw.empty(), "1c: the fixture's reverse analysis completes" );
	expect( r.candidates.size() >= 4, "1d: the first pass considered every variable" );

	// The four first-pass p-values must be DISTINCT, or "which one is largest"
	//    is not a question with one answer
	vector< double > firstPass;
	for ( size_t i = 0; i < r.candidates.size(); i++ )
		if ( r.candidates[ i ].step == 0 )
			firstPass.push_back( r.candidates[ i ].p );
	bool allDistinct = firstPass.size() == 4;
	for ( size_t i = 0; i < firstPass.size() && allDistinct; i++ )
		for ( size_t j = i + 1; j < firstPass.size() && allDistinct; j++ )
			if ( firstPass[ i ] == firstPass[ j ] )
				allDistinct = false;
	expect( allDistinct, "1e: every first-pass p-value is distinct" );

	// ... and they must straddle the threshold, or the stopping rule is never
	//     actually exercised by a decision
	unsigned above = 0, below = 0;
	for ( size_t i = 0; i < firstPass.size(); i++ )
		( firstPass[ i ] > 0.05 ? above : below )++;
	expect( above >= 2 && below >= 2,
		"1f: first-pass p-values straddle the threshold in both directions" );

	// Every p-value must be strictly positive. p == 0 is reachable ( pX2 is
	//    gammq, which underflows for a large chi-square ), and a fixture that
	//    produced it would be sitting on the defect recorded in audit 15.6
	//    rather than characterizing the selection rule.
	bool allPositive = true;
	for ( size_t i = 0; i < firstPass.size(); i++ )
		if ( !( firstPass[ i ] > 0 ) )
			allPositive = false;
	expect( allPositive,
		"1g: no p-value underflowed to zero, so a winner is always assigned" );
}

// ---------------------------------------------------------------------------
// 2-9. A COMPLETED REVERSE RUN
// ---------------------------------------------------------------------------
static void test_reverse_completed()
{
	DataSet d = mixedData();
	Logistic net;
	double e_in = fitSource( net, d );
	Run r = regress( net, e_in, singles( 4 ), false, 0.05 );

	expect( r.threw.empty(), "2: a reverse run over converging candidates does not throw" );
	expect( r.complete, "3: a reverse run that reached a decision reports complete" );
	expect( r.fits == r.candidates.size(),
		"4: fitsCompleted equals the number of recorded candidates" );

	// The audit trail is in the order the procedure considered them: steps
	//    non-decreasing, and the candidate index restarting at 1 each step
	bool ordered = true;
	unsigned step = 0, expectCand = 1;
	for ( size_t i = 0; i < r.candidates.size(); i++ )
	{
		const RegressNet::Candidate& c = r.candidates[ i ];
		if ( c.step != step )
		{
			if ( c.step != step + 1 ) ordered = false;
			step = c.step;
			expectCand = 1;
		}
		if ( c.candidate != expectCand ) ordered = false;
		expectCand++;
	}
	expect( ordered, "5: candidates are recorded in exact consideration order" );

	// Within a pass, variables are visited in ascending order, and a variable
	//    already removed never reappears
	bool ascending = true;
	for ( size_t i = 1; i < r.candidates.size(); i++ )
		if ( r.candidates[ i ].step == r.candidates[ i - 1 ].step
			&& r.candidates[ i ].variable <= r.candidates[ i - 1 ].variable )
			ascending = false;
	expect( ascending, "6: within a pass, candidates ascend by variable number" );

	// Every candidate in a completed run was compared, so every field is real
	bool fieldsPopulated = !r.candidates.empty();
	for ( size_t i = 0; i < r.candidates.size(); i++ )
	{
		const RegressNet::Candidate& c = r.candidates[ i ];
		if ( c.inputs.size() != 1 || c.inputs[ 0 ] != c.variable ) fieldsPopulated = false;
		if ( c.df != 1 ) fieldsPopulated = false;          // one node per variable here
		if ( !std::isfinite( c.priorError ) ) fieldsPopulated = false;
		if ( !std::isfinite( c.error ) ) fieldsPopulated = false;
		if ( !std::isfinite( c.G2 ) ) fieldsPopulated = false;
		if ( !std::isfinite( c.p ) ) fieldsPopulated = false;
		if ( !c.converged ) fieldsPopulated = false;
		if ( c.stopReason.empty() ) fieldsPopulated = false;
		if ( c.iterations == 0 ) fieldsPopulated = false;
	}
	expect( fieldsPopulated,
		"7: every compared candidate carries group, errors, df, G2, p, convergence,"
		" stop reason and iteration count" );

	// REVERSE'S NESTING, pinned by a construction-exact value: the first pass
	//    compares against the SOURCE model's own error, handed in through
	//    setNetwork. Not approximately -- the same double.
	expect( !r.candidates.empty() && r.candidates[ 0 ].priorError == e_in,
		"8: reverse's first pass compares against the source model's own error" );

	// Removing a variable can only make the training fit worse, so a reverse
	//    candidate's error exceeds its prior and G2 = 2N( Esub - Efull ) >= 0
	bool nested = true;
	for ( size_t i = 0; i < r.candidates.size(); i++ )
		if ( !( r.candidates[ i ].error >= r.candidates[ i ].priorError )
			|| !( r.candidates[ i ].G2 >= 0 ) )
			nested = false;
	expect( nested,
		"9: a reverse candidate is nested in its prior -- error rises, G2 is non-negative" );
}

// ---------------------------------------------------------------------------
// 10-14. REVERSE'S RESULT: what it removed, in what order, and what it kept
// ---------------------------------------------------------------------------
static void test_reverse_result()
{
	DataSet d = mixedData();
	Logistic net;
	double e_in = fitSource( net, d );
	Run r = regress( net, e_in, singles( 4 ), false, 0.05 );

	// The two noise variables go; the two signal variables stay. This is the
	//    substantive claim -- a flipped winner sense removes 0 and 1 instead.
	expect( r.finalVariables == vector< unsigned >( { 0, 1 } ),
		"10: reverse retains exactly the two variables that carry signal" );
	expect( pathVars( r ) == vector< unsigned >( { 2, 3 } ),
		"11: reverse removes the noise variables, least significant first" );

	// The path is the removal ORDER, which the internal p-sorted table cannot
	//    answer. Its p-values must match the winning candidates', in order.
	vector< unsigned > winners;
	vector< double > winnerP;
	for ( size_t i = 0; i < r.candidates.size(); i++ )
		if ( r.candidates[ i ].selected )
		{
			winners.push_back( r.candidates[ i ].variable );
			winnerP.push_back( r.candidates[ i ].p );
		}
	expect( winners == pathVars( r ),
		"12: the selection path is exactly the candidates flagged as winners, in order" );
	bool pMatches = winnerP.size() == r.path.size();
	for ( size_t i = 0; i < winnerP.size() && pMatches; i++ )
		if ( winnerP[ i ] != r.path[ i ].first ) pMatches = false;
	expect( pMatches, "13: each path entry carries its winning candidate's p-value" );

	// A winner is the largest p in its own pass -- the reverse rule
	bool winnerIsLargest = true;
	for ( size_t i = 0; i < r.candidates.size(); i++ )
		if ( r.candidates[ i ].selected )
			for ( size_t j = 0; j < r.candidates.size(); j++ )
				if ( r.candidates[ j ].step == r.candidates[ i ].step
					&& r.candidates[ j ].p > r.candidates[ i ].p )
					winnerIsLargest = false;
	expect( winnerIsLargest,
		"14: reverse selects the LARGEST p-value in each pass (least significant)" );
}

// ---------------------------------------------------------------------------
// 15-19. A COMPLETED FORWARD RUN, and how it differs from reverse
// ---------------------------------------------------------------------------
static void test_forward_completed()
{
	DataSet d = mixedData();
	Logistic net;
	double e_in = fitSource( net, d );
	Run r = regress( net, e_in, singles( 4 ), true, 0.05 );

	expect( r.threw.empty(), "15: a forward run over converging candidates does not throw" );
	expect( r.complete && r.fits == r.candidates.size(),
		"16: a completed forward run reports complete, and its counts agree" );

	// FORWARD'S BASELINE IS A TRAINED EMPTY NETWORK, not the source model. This
	//    is the structural difference between the directions, and it is what
	//    makes the two fixtures genuinely different conceptual paths rather
	//    than the same walk with the sign changed.
	expect( !r.candidates.empty() && r.candidates[ 0 ].priorError != e_in,
		"17: forward's first pass does NOT compare against the source model's error" );
	// An input-free binary logistic can do no better than predict the base
	//    rate, so its cross-entropy sits at ln 2 for a balanced fixture -- far
	//    above the fitted source error. Asserted as an inequality, not a value.
	expect( !r.candidates.empty() && r.candidates[ 0 ].priorError > e_in,
		"18: forward's baseline is the trained EMPTY network, worse than the source fit" );

	// Adding a variable can only improve the training fit, so a forward
	//    candidate's error falls below its prior -- the opposite of case 9
	bool nested = !r.candidates.empty();
	for ( size_t i = 0; i < r.candidates.size(); i++ )
		if ( !( r.candidates[ i ].error <= r.candidates[ i ].priorError )
			|| !( r.candidates[ i ].G2 >= 0 ) )
			nested = false;
	expect( nested,
		"19: a forward candidate NESTS its prior -- error falls, G2 is non-negative" );
}

// ---------------------------------------------------------------------------
// 20-24. THE DIRECTION SEMANTICS. These are the cases a shared comparator or a
// swapped Wilks argument order would break.
// ---------------------------------------------------------------------------
static void test_direction_semantics()
{
	DataSet d = mixedData();
	Logistic net;
	double e_in = fitSource( net, d );
	Run rev = regress( net, e_in, singles( 4 ), false, 0.05 );
	Run fwd = regress( net, e_in, singles( 4 ), true, 0.05 );

	expect( fwd.finalVariables == vector< unsigned >( { 0, 1 } ),
		"20: forward selects exactly the two variables that carry signal" );
	expect( pathVars( fwd ) == vector< unsigned >( { 0, 1 } ),
		"21: forward admits the MOST significant variable first" );

	// A winner is the smallest p in its own pass -- the forward rule, opposite
	//    to case 14 on the same data
	bool winnerIsSmallest = true;
	for ( size_t i = 0; i < fwd.candidates.size(); i++ )
		if ( fwd.candidates[ i ].selected )
			for ( size_t j = 0; j < fwd.candidates.size(); j++ )
				if ( fwd.candidates[ j ].step == fwd.candidates[ i ].step
					&& fwd.candidates[ j ].p < fwd.candidates[ i ].p )
					winnerIsSmallest = false;
	expect( winnerIsSmallest,
		"22: forward selects the SMALLEST p-value in each pass (most significant)" );

	// The two directions agree on the ANSWER and disagree on the PATH. If a
	//    shared loop collapsed the two senses, the paths would coincide.
	expect( rev.finalVariables == fwd.finalVariables,
		"23: both directions reach the same final variable set on this fixture" );
	expect( pathVars( rev ) != pathVars( fwd ),
		"24: the two directions take genuinely different paths to it" );
}

// ---------------------------------------------------------------------------
// 25-28. THRESHOLD STOPPING, and the reachable "none / one / several" results
// ---------------------------------------------------------------------------
static void test_threshold_stopping()
{
	DataSet d = mixedData();
	Logistic net;
	double e_in = fitSource( net, d );

	// REVERSE, REMOVING NONE. The stop test is largest_p < threshold, so a
	//    threshold above every p-value stops the very first pass with nothing
	//    removed -- a COMPLETED analysis whose answer is "keep everything".
	Run keepAll = regress( net, e_in, singles( 4 ), false, 0.99 );
	expect( keepAll.threw.empty() && keepAll.complete
		&& keepAll.path.empty()
		&& keepAll.finalVariables == vector< unsigned >( { 0, 1, 2, 3 } ),
		"25: reverse at a high threshold completes having removed nothing" );
	expect( has( keepAll.report, "Variables removed, in order: none" )
		&& has( keepAll.report, "Final retained variables: 0, 1, 2, 3" ),
		"26: a reverse run that removed nothing says so in its summary" );

	// FORWARD, ADDING ONE. The stop test is smallest_p > threshold. A threshold
	//    between the two signal variables' p-values admits the first and stops.
	//    ( Measured: p( var 0 ) ~ 1e-62, p( var 1 ) ~ 5e-21. )
	Run addOne = regress( net, e_in, singles( 4 ), true, 1e-40 );
	expect( addOne.threw.empty() && addOne.complete
		&& addOne.finalVariables == vector< unsigned >( { 0 } ),
		"27: forward stops after admitting the one variable that clears the threshold" );

	// FORWARD, ADDING NONE, over a fixture with no signal at all.
	DataSet nd = noiseData();
	Logistic noiseNet;
	double noise_e = fitSource( noiseNet, nd );
	Run addNone = regress( noiseNet, noise_e, singles( 3 ), true, 0.05 );
	expect( addNone.threw.empty() && addNone.complete
		&& addNone.path.empty() && addNone.finalVariables.empty()
		&& has( addNone.report, "Variables added, in order: none" ),
		"28: forward over pure noise completes having admitted nothing" );
}

// ---------------------------------------------------------------------------
// 29-32. THE CONVERGENCE CONTRACT: an unfinished candidate ends the analysis.
//
// ISOLATED, one source model per case: each of these ends in an exception, and
// a single shared model would let the first failure hide the ones after it.
// ---------------------------------------------------------------------------
static void test_ceiling_exhausted_candidate()
{
	DataSet d = mixedData();
	Logistic net;
	// The clone inherits the source's configuration, so a tiny ceiling on the
	//    source is what puts every candidate at that same ceiling
	double e_in = fitSource( net, d, 3 );
	expect( !Iterative::converged( net.getStopReason() ),
		"29a: the tiny-ceiling source fit was supposed NOT to converge" );

	Run r = regress( net, e_in, singles( 4 ), false, 0.05 );

	expect( !r.threw.empty() && has( r.threw, "did not converge" )
		&& has( r.threw, "max_iterations" ),
		"29b: a ceiling-exhausted candidate fails the analysis by name" );
	expect( !r.complete, "29c: the failed analysis does not report complete" );

	// THE RECORD IS WRITTEN BEFORE THE ELIGIBILITY CHECK THAT THROWS. The
	//    candidate a reader most needs is the one that stopped the run.
	expect( !r.candidates.empty(),
		"29d: the candidate that failed is in the audit trail" );
	expect( r.fits == r.candidates.size(),
		"29e: the fit count and the trail still agree after a failure" );

	// A default-constructed stand-in keeps one empty trail from aborting the
	//    whole suite, so the cases after this one are still observed
	RegressNet::Candidate fallback = RegressNet::Candidate();
	const RegressNet::Candidate& bad =
		r.candidates.empty() ? fallback : r.candidates.back();
	expect( !r.candidates.empty() && !bad.converged
		&& bad.stopReason == "max_iterations",
		"29f: the failing candidate records how it actually ended" );
	// It was recorded but never COMPARED, so it contributes no statistic
	expect( !r.candidates.empty() && std::isnan( bad.G2 ) && std::isnan( bad.p ),
		"29g: a recorded-but-never-compared candidate has no G2 and no p" );
	expect( !r.candidates.empty() && std::isfinite( bad.error )
		&& std::isfinite( bad.priorError ),
		"29h: it still records the errors it was going to be compared on" );

	// No pass completed, so there is no result to report
	expect( r.finalVariables.empty() && r.path.empty(),
		"29i: an analysis that reached no decision has no selection to report" );
	expect( has( r.report, "not a finished fit" )
		&& has( r.report, "no variable is selected" )
		&& has( r.report, "Raise the maximum iterations" ),
		"29j: the refusal explains itself and says how to proceed" );
	expect( !has( r.report, "stepwise regression complete." )
		&& !has( r.report, "Final retained variables:" ),
		"29k: a failed analysis prints NO completion summary" );
	// The partial trail reaches the reader before the throw
	expect( has( r.report, "Reverse regressing" ) && has( r.report, "Variable structure:" ),
		"29l: the partial report survives the refusal" );
}

// CASE 30 IS DELIBERATELY ABSENT: the NON-FINITE half of the eligibility rule.
//
// requireConvergedFit refuses when the stop reason is not a convergence reason
// OR when the error is not finite. Case 29 covers the first half and case 31
// covers cancellation. The second half is an INDEPENDENT guard -- a fit can end
// on a genuine convergence reason with a non-finite error, because a saturated
// logistic has gradients that underflow to zero while its cross-entropy
// overflows -- and no robust fixture reaches it through the public API.
//
// Measured, not assumed. A sweep of 270 configurations (three optimizers x ten
// learning rates x three input scales x three ceilings, n = 200) produced ZERO
// cases of converged-and-non-finite. At n = 400 exactly one point hit:
// Shanno, input scale 10, eta 10 -- and its neighbourhood is a spike, not a
// plateau. eta 5 and 20 miss, scale 8 and 15 miss, n 300 and 1200 miss, and
// three of five seeds miss. A fixture sitting on that point would be a bet on
// one toolchain's arithmetic, which is exactly the bet the DFA singular fixture
// lost on Ubuntu (audit section 13, commit ad7ab14).
//
// So this branch is UNCOVERED, and saying so is the point: it is an absent
// guard, not an acceptable narrowness. It is scheduled with the correctness
// phase in audit section 15.6, which already has to open this code to fix the
// uninitialized winner -- a seam introduced deliberately there beats a
// test-only visibility hack or a fixture that is green on one machine.

// An observer that stops the candidate currently training, after letting a
//    fixed amount of work happen first
class Canceller : public Iterative::Observer {
public:
	explicit Canceller( unsigned after ) : budget( after ), seen( 0 ) {}
	bool onIteration( unsigned, double ) override { return ++seen < budget; }
	unsigned seen;
private:
	unsigned budget;
};

static void test_cancelled_candidate()
{
	DataSet d = mixedData();
	Logistic net;
	double e_in = fitSource( net, d );

	Canceller stopper( 40 );
	Run r = regress( net, e_in, singles( 4 ), false, 0.05, &stopper );

	expect( stopper.seen > 0, "31a: the observer actually saw candidate iterations" );
	expect( !r.threw.empty() && has( r.threw, "cancelled" ),
		"31b: a cancelled candidate fails the analysis, by the cancelled reason" );
	expect( !r.complete, "31c: a cancelled analysis does not report complete" );
	expect( !r.candidates.empty() && r.candidates.back().stopReason == "cancelled"
		&& !r.candidates.back().converged,
		"31d: the cancelled candidate is recorded as cancelled, not as converged" );
	expect( !r.candidates.empty() && std::isnan( r.candidates.back().p ),
		"31e: a cancelled candidate is recorded but never compared" );
	expect( has( r.report, "stopped by request" ),
		"31f: the report says the run was stopped by request" );
}

static void test_forward_baseline_refusal()
{
	DataSet d = mixedData();
	Logistic net;
	double e_in = fitSource( net, d, 3 );

	Run r = regress( net, e_in, singles( 4 ), true, 0.05 );

	// Forward's baseline is trained BEFORE any candidate, and it is not a
	//    candidate: if it did not finish, nothing downstream means anything
	expect( !r.threw.empty() && has( r.threw, "baseline network (no variables)" ),
		"32a: forward names the BASELINE when the baseline is what failed" );
	expect( r.candidates.empty() && r.fits == 0,
		"32b: the baseline is not a candidate and is not counted as one" );
	expect( !r.complete, "32c: the analysis did not complete" );
}

// ---------------------------------------------------------------------------
// 33-36. GROUPED VARIABLES AND RANGE SYNTAX
// ---------------------------------------------------------------------------
static void test_grouping()
{
	DataSet d = groupedData();
	Logistic net;
	double e_in = fitSource( net, d );

	vector< vector< unsigned > > defs;
	defs.push_back( { 0 } );
	defs.push_back( { 1, 2 } );
	defs.push_back( { 3, 4 } );

	Run r = regress( net, e_in, defs, false, 0.05 );
	expect( r.threw.empty() && r.complete, "33a: a grouped reverse analysis completes" );

	// A grouped variable is never split: every candidate carries its WHOLE
	//    group, and its degrees of freedom count all of the group's nodes
	bool groupsIntact = !r.candidates.empty();
	for ( size_t i = 0; i < r.candidates.size(); i++ )
	{
		const RegressNet::Candidate& c = r.candidates[ i ];
		if ( c.inputs != defs[ c.variable ] ) groupsIntact = false;
		if ( c.df != defs[ c.variable ].size() ) groupsIntact = false;
	}
	expect( groupsIntact,
		"33b: every candidate carries its full input group, and df counts all its nodes" );

	// The df values must not be uniform, or "df counts the group" is satisfied
	//    by any constant
	bool sawOne = false, sawTwo = false;
	for ( size_t i = 0; i < r.candidates.size(); i++ )
	{
		if ( r.candidates[ i ].df == 1 ) sawOne = true;
		if ( r.candidates[ i ].df == 2 ) sawTwo = true;
	}
	expect( sawOne && sawTwo,
		"34: the analysis produced both 1-node and 2-node comparisons" );

	expect( has( r.report, "node(s) 1 2" ),
		"35: the report names a grouped variable by all of its nodes" );

	// The range syntax "1-2" must mean the same thing as the pair {1,2}
	vector< vector< unsigned > > parsed = util::variable_parse( "0;1-2;3-4" );
	expect( parsed == defs, "36a: variable_parse expands a node range to the same group" );

	Logistic net2;
	double e2 = fitSource( net2, d );
	Run viaRange = regress( net2, e2, parsed, false, 0.05 );
	expect( !viaRange.candidates.empty()
		&& viaRange.finalVariables == r.finalVariables
		&& pathVars( viaRange ) == pathVars( r ),
		"36b: a range-specified structure produces the identical analysis" );
}

// ---------------------------------------------------------------------------
// 37. INVALID GROUPING IS REFUSED
// ---------------------------------------------------------------------------
static void test_invalid_structure()
{
	DataSet d = mixedData();
	Logistic net;
	fitSource( net, d );

	// THE CONTRACT: the groups must be an EXACT PARTITION of the input nodes.
	//    Every node in exactly one group, no group empty, no node outside the
	//    Network, nothing left over. Each of these must produce a RegressNetErr
	//    -- not a crash, and not some unrelated exception, which is why the
	//    helper distinguishes the three outcomes rather than catching (...).
	enum Outcome { REFUSED, ACCEPTED, WRONG_EXCEPTION };
	struct Check {
		static Outcome of( Network& n, const vector< vector< unsigned > >& defs )
		{
			RegressNet r;
			r.setNetwork( &n, 1.0 );
			try { r.setInputStructure( defs ); }
			catch ( RegressNet::RegressNetErr& ) { return REFUSED; }
			catch ( ... ) { return WRONG_EXCEPTION; }
			return ACCEPTED;
		}
	};

	// The control FIRST: a valid partition must be accepted, or every "refused"
	//    below is satisfied by a function that refuses everything.
	expect( Check::of( net, singles( 4 ) ) == ACCEPTED,
		"37a: a valid singleton partition is accepted (control)" );
	vector< vector< unsigned > > grouped;
	grouped.push_back( { 0, 1 } );
	grouped.push_back( { 2 } );
	grouped.push_back( { 3 } );
	expect( Check::of( net, grouped ) == ACCEPTED,
		"37b: a valid grouped partition is accepted (control)" );
	expect( Check::of( net, util::variable_parse( "0-1;2;3" ) ) == ACCEPTED,
		"37c: the same partition written as a range is accepted (control)" );

	// Out of range. This was the ONE malformed case the old maximum-only check
	//    caught, and only because an out-of-range node raises the maximum.
	vector< vector< unsigned > > tooHigh = singles( 4 );
	tooHigh.push_back( { 9 } );
	expect( Check::of( net, tooHigh ) == REFUSED,
		"37d: a structure naming a node the Network does not have is refused" );

	// OMISSION WITH A CORRECT MAXIMUM. `{{3}}` on a four-input Network has
	//    maximum 3, which is exactly what the old check demanded -- and it was
	//    accepted, silently analyzing a model the user never described.
	vector< vector< unsigned > > onlyLast;
	onlyLast.push_back( { 3 } );
	expect( Check::of( net, onlyLast ) == REFUSED,
		"37e: an omission whose maximum still looks valid is refused" );
	vector< vector< unsigned > > twoOfFour;
	twoOfFour.push_back( { 0 } );
	twoOfFour.push_back( { 3 } );
	expect( Check::of( net, twoOfFour ) == REFUSED,
		"37f: a structure covering only some input nodes is refused" );
	expect( Check::of( net, singles( 3 ) ) == REFUSED,
		"37g: a structure whose maximum is too low is refused" );

	// OVERLAP. Node 0 in two variables was accepted and produced a complete
	//    analysis of five variables over four nodes.
	vector< vector< unsigned > > overlapping;
	overlapping.push_back( { 0 } );
	overlapping.push_back( { 0 } );
	overlapping.push_back( { 1 } );
	overlapping.push_back( { 2 } );
	overlapping.push_back( { 3 } );
	expect( Check::of( net, overlapping ) == REFUSED,
		"37h: a node belonging to two variables is refused" );
	vector< vector< unsigned > > repeatedWithin;
	repeatedWithin.push_back( { 0, 0 } );
	repeatedWithin.push_back( { 1 } );
	repeatedWithin.push_back( { 2 } );
	repeatedWithin.push_back( { 3 } );
	expect( Check::of( net, repeatedWithin ) == REFUSED,
		"37i: a node repeated inside one variable is refused" );

	// AN EMPTY GROUP. It removed no nodes, so its chi-square was 0, its p-value
	//    was 1, and a p-value of 1 wins every reverse pass -- it was selected
	//    first on every run, having removed nothing.
	vector< vector< unsigned > > emptyGroup = singles( 4 );
	emptyGroup.push_back( vector< unsigned >() );
	expect( Check::of( net, emptyGroup ) == REFUSED,
		"37j: a variable with no input nodes is refused" );

	// AN EMPTY STRUCTURE. max_element over an empty range dereferenced its end
	//    iterator: SIGSEGV.
	expect( Check::of( net, vector< vector< unsigned > >() ) == REFUSED,
		"37k: an empty input structure is refused" );
	// ... and it must say THAT, rather than reporting the first uncovered node.
	//     An empty structure is caught twice over -- by its own guard and, since
	//     it claims nothing, by the every-node-covered rule -- so without this
	//     the specific guard is unguarded, and a user who supplied no variables
	//     at all would be told "input node 0 belongs to no variable".
	{
		RegressNet r;
		r.setNetwork( &net, 1.0 );
		string message;
		try { r.setInputStructure( vector< vector< unsigned > >() ); }
		catch ( RegressNet::RegressNetErr& e ) { message = e.what(); }
		expect( has( message, "names no variables" ),
			"37k2: ... and the refusal names the real problem" );
	}

	// NO NETWORK AT ALL. netPtr was dereferenced without a null check: SIGSEGV.
	unsigned refusedNoNetwork = 0, wrongException = 0;
	{
		RegressNet r; // setNetwork deliberately NOT called
		try { r.setInputStructure( singles( 4 ) ); }
		catch ( RegressNet::RegressNetErr& ) { refusedNoNetwork = 1; }
		catch ( ... ) { wrongException = 1; }
	}
	expect( refusedNoNetwork == 1 && wrongException == 0,
		"37l: setting a structure before the Network is refused, not a crash" );

	// An analysis with no structure at all is refused too
	unsigned refusedEmpty = 0;
	{
		RegressNet r;
		r.setNetwork( &net, 1.0 );
		util::ScreenCapture hush;
		try { r.reverse_regress(); }
		catch ( RegressNet::RegressNetErr& ) { refusedEmpty = 1; }
	}
	expect( refusedEmpty == 1, "37m: an analysis with no input structure is refused" );

	// ORDER IS PRESERVED, NOT NORMALIZED. Validation must not sort or renumber:
	//    variable numbering in every report and audit row is the caller's own.
	vector< vector< unsigned > > reversedOrder;
	reversedOrder.push_back( { 3, 2 } ); // descending within the group, too
	reversedOrder.push_back( { 1 } );
	reversedOrder.push_back( { 0 } );
	{
		RegressNet r;
		r.setNetwork( &net, 1.0 );
		unsigned accepted = 0;
		try { r.setInputStructure( reversedOrder ); accepted = 1; }
		catch ( RegressNet::RegressNetErr& ) {}
		expect( accepted == 1,
			"37n: a partition given in any order is accepted (control)" );
	}
	{
		DataSet d2 = mixedData();
		Logistic net2;
		double e2 = fitSource( net2, d2 );
		Run r = regress( net2, e2, reversedOrder, false, 0.05 );
		expect( !r.candidates.empty()
			&& r.candidates[ 0 ].inputs == vector< unsigned >( { 3, 2 } ),
			"37o: the caller's group order and within-group order are preserved" );
	}
}

// ---------------------------------------------------------------------------
// 38-40. THE OBJECTIVE MUST BE CROSS-ENTROPY, AND THE MODEL IS NOT OURS
// ---------------------------------------------------------------------------
static void test_lms_refusal_does_not_mutate()
{
	DataSet d = mixedData();

	// SimpleProp trained under least-mean-squares: Wilks compares log
	//    likelihoods, so its errors are in the wrong objective entirely
	SimpleProp net;
	net.setDataSet( d );
	net.setHidden( 3 );
	net.setHistory( false ); net.setLastop( false ); net.setLogPrint( false );
	net.setQuiet( true );
	net.setAutoStepSize( true );
	net.setTrainingType( 0 );
	net.setMaxIterations( 60 );
	util::set_seed( 42 );
	net.randomize();
	double e_in;
	{ util::ScreenCapture hush; e_in = net.train(); }

	expect( !net.getXEerror(), "38a: the source model is a least-mean-squares fit" );

	string before = tmp( "lms_before.txt" ), after = tmp( "lms_after.txt" );
	{ util::ScreenCapture hush; net.save( before ); }

	unsigned refusedReverse = 0, refusedForward = 0;
	string message;
	for ( int forward = 0; forward < 2; forward++ )
	{
		RegressNet r;
		r.setNetwork( &net, e_in );
		r.setInputStructure( singles( 4 ) );
		r.setThreshold( 0.05 );
		util::ScreenCapture hush;
		try
		{
			if ( forward ) r.forward_regress(); else r.reverse_regress();
		}
		catch ( RegressNet::RegressNetErr& e )
		{
			message = e.what();
			( forward ? refusedForward : refusedReverse )++;
		}
	}
	expect( refusedReverse == 1 && refusedForward == 1,
		"38b: BOTH directions refuse a least-mean-squares source fit" );
	expect( has( message, "cross-entropy" ) && has( message, "least-mean-squares" ),
		"38c: the refusal names both objectives" );

	// ... and the user's model is exactly as it was found. The old code called
	//     setXEerror() on it, which mutated a model stepwise must treat as
	//     read-only -- and did not even fix the mismatch, because the baseline
	//     error was captured in the ORIGINAL objective.
	expect( !net.getXEerror(),
		"39a: the refused analysis did not flip the model's error function" );
	{ util::ScreenCapture hush; net.save( after ); }

	// Existence evidence FIRST: two absent or empty files compare equal, and
	//    that comparison would guard nothing (standing rule 2).
	expect( exists( before ) && exists( after ),
		"39b: both model files were actually written" );
	string a = slurp( before ), b = slurp( after );
	expect( !a.empty() && !b.empty() && has( a, "SimpleProp" ),
		"39c: the saved models are non-empty and are the model we saved" );
	expect( a == b, "39d: the model file is byte-identical after a refused analysis" );

	// The model is still usable afterwards
	expect( net.getDataSet().getNumTrain() > 0 && net.getIterations() > 0,
		"40: the model remains trained and usable after a refusal" );
}

// ---------------------------------------------------------------------------
// 41-44. THE SOURCE NETWORK IS NEVER MUTATED BY A SUCCESSFUL ANALYSIS EITHER
// ---------------------------------------------------------------------------
static void test_source_is_immutable()
{
	DataSet d = mixedData();
	Logistic net;
	double e_in = fitSource( net, d );

	// The whole weight vector, as the program itself writes it
	string beforePath = saveModel( net, "immutable_before.txt" );
	string beforeBytes = slurp( beforePath );
	bool xeBefore = net.getXEerror();
	unsigned itersBefore = net.getIterations();
	unsigned inputsBefore = net.getDataSet().getInput();
	unsigned trainBefore = net.getDataSet().getNumTrain();
	Iterative::StopReason stopBefore = net.getStopReason();
	string typeBefore = net.getType();
	bool histBefore = net.getHistory(), lastopBefore = net.getLastop();

	// EXISTENCE EVIDENCE FIRST: a comparison between two failed saves is a
	//    comparison of two empty strings, and guards nothing (standing rule 2).
	expect( exists( beforePath ) && !beforeBytes.empty()
		&& has( beforeBytes, "Binary logistic" ),
		"41a: the model's weights were actually written, and are the model we saved" );

	Run rev = regress( net, e_in, singles( 4 ), false, 0.05 );
	Run fwd = regress( net, e_in, singles( 4 ), true, 0.05 );
	expect( rev.threw.empty() && fwd.threw.empty() && !rev.candidates.empty()
		&& !fwd.candidates.empty(),
		"41b: both analyses actually ran, over real candidates (control)" );

	string afterBytes = slurp( saveModel( net, "immutable_after.txt" ) );
	expect( !afterBytes.empty() && afterBytes == beforeBytes,
		"41c: every weight is unchanged after two stepwise analyses" );
	expect( net.getXEerror() == xeBefore, "42a: the objective is unchanged" );
	expect( net.getIterations() == itersBefore && net.getStopReason() == stopBefore,
		"42b: the model's own training state is unchanged" );
	expect( net.getDataSet().getInput() == inputsBefore
		&& net.getDataSet().getNumTrain() == trainBefore,
		"42c: the model's dataset is unchanged -- no input node was removed from it" );
	expect( net.getType() == typeBefore, "42d: the model's type is unchanged" );
	expect( net.getHistory() == histBefore && net.getLastop() == lastopBefore,
		"42e: the model's reporting flags are unchanged" );

	// CONTINUATION: the model must go on training exactly as it would have if
	//    no analysis had ever run. Compared against an independent model given
	//    the identical treatment minus the regressions.
	Logistic control;
	double controlError = fitSource( control, d );
	expect( controlError == e_in, "43a: the control fit reproduces the source fit exactly" );

	// The gradient stop has already fired on both, so continuing has to be
	//    asked for explicitly -- otherwise train() returns at once, both models
	//    stay exactly where they were, and 44c would compare two files that
	//    never moved. 44b is the guard that caught precisely that.
	net.setGradStop( false );
	control.setGradStop( false );
	net.setMaxIterations( 25 );
	control.setMaxIterations( 25 );
	double afterRegress, afterControl;
	{ util::ScreenCapture hush; afterRegress = net.train(); }
	{ util::ScreenCapture hush; afterControl = control.train(); }
	expect( afterRegress == afterControl,
		"43b: continued training after an analysis matches a model that never had one" );
	string contBytes = slurp( saveModel( net, "cont_regressed.txt" ) ),
		ctrlBytes = slurp( saveModel( control, "cont_control.txt" ) );
	expect( !contBytes.empty() && !ctrlBytes.empty(),
		"44a: both continued models were written (control)" );
	expect( contBytes != beforeBytes,
		"44b: continuing training DID move the weights, so 44c is not vacuous" );
	expect( contBytes == ctrlBytes,
		"44c: and the two continued models are weight-for-weight identical" );
}

// ---------------------------------------------------------------------------
// 45-46. HISTORY: the analysis writes its report where the MODEL says to
// ---------------------------------------------------------------------------
static void test_history()
{
	DataSet d = mixedData();

	// History OFF: nothing is written
	{
		Logistic net;
		double e_in = fitSource( net, d );
		string hf = tmp( "history_off.log" );
		net.setHistoryFilename( hf );
		net.setHistory( false );
		Run r = regress( net, e_in, singles( 4 ), false, 0.05 );
		expect( r.threw.empty() && !r.report.empty(),
			"45a: the analysis ran and produced a report (control)" );
		expect( !exists( hf ),
			"45b: with history off, no history file is written" );
	}

	// History ON: the same report is appended to the model's history file.
	//    RegressNet reads historyFlag at setNetwork() time, so it is set before.
	{
		Logistic net;
		double e_in = fitSource( net, d );
		string hf = tmp( "history_on.log" );
		net.setHistoryFilename( hf );
		net.setHistory( true );
		Run r = regress( net, e_in, singles( 4 ), false, 0.05 );
		expect( r.threw.empty(), "46a: the analysis with history on completed" );
		expect( exists( hf ), "46b: with history on, the history file is written" );
		string logged = slurp( hf );
		expect( !logged.empty() && !r.report.empty(),
			"46c: both the report and the history file are non-empty" );
		expect( has( logged, "Reverse regressing" )
			&& has( logged, "Reverse stepwise regression complete." ),
			"46d: the history file carries the analysis, banner through summary" );
		// The normalization must forgive line endings and NOTHING else, or 46e
		//    is weaker than the assertion it replaced. Proven on this run's own
		//    report: a CRLF copy must compare equal, and a copy differing by a
		//    single character must not.
		string crlf;
		for ( size_t i = 0; i < r.report.size(); i++ )
		{
			if ( r.report[ i ] == '\n' ) crlf += '\r';
			crlf += r.report[ i ];
		}
		expect( crlf != r.report && sameNewlines( crlf ) == sameNewlines( r.report ),
			"46f: the newline normalization forgives CRLF (control)" );
		string tampered = r.report;
		tampered[ tampered.size() / 2 ] = ( tampered[ tampered.size() / 2 ] == 'x' )
			? 'y' : 'x';
		expect( sameNewlines( tampered ) != sameNewlines( r.report ),
			"46g: ... and forgives nothing else -- one changed character still fails" );

		expect( sameNewlines( logged ) == sameNewlines( r.report ),
			"46e: the history file and the screen report are the same text" );
	}
}

// ---------------------------------------------------------------------------
// 47-49. PROGRESS AND OBSERVER ARE OBSERVATION ONLY.
//
// This is the engine-level statement of the blocking/async equivalence the GUI
// smoke test makes over HTTP: async is the same call on a worker thread with a
// progress callback and a cancel observer attached, so ATTACHING them must not
// change one number. Anything else would make the async result a different
// analysis from the blocking one.
// ---------------------------------------------------------------------------
static void test_progress_is_observation_only()
{
	DataSet d = mixedData();
	Logistic bare, watched;
	double e1 = fitSource( bare, d );
	double e2 = fitSource( watched, d );
	expect( e1 == e2, "47a: the two source fits are identical (control)" );

	Run plain = regress( bare, e1, singles( 4 ), false, 0.05, 0, false );

	// An observer that never stops anything -- present, consulted, harmless
	class Watcher : public Iterative::Observer {
	public:
		Watcher() : seen( 0 ) {}
		bool onIteration( unsigned, double ) override { seen++; return true; }
		unsigned seen;
	} watcher;

	Run observed = regress( watched, e2, singles( 4 ), false, 0.05, &watcher, true );

	expect( plain.progress.empty() && !observed.progress.empty(),
		"47b: the callback fired only where one was installed (control)" );
	expect( watcher.seen > 0, "47c: the observer was actually consulted (control)" );
	expect( !plain.candidates.empty(),
		"47d: the unobserved analysis produced candidates (control)" );

	expect( plain.report == observed.report,
		"48a: the report is identical with and without progress reporting" );
	expect( plain.finalVariables == observed.finalVariables
		&& pathVars( plain ) == pathVars( observed )
		&& plain.fits == observed.fits
		&& plain.candidates.size() == observed.candidates.size(),
		"48b: the structured result is identical too" );
	bool sameNumbers = plain.candidates.size() == observed.candidates.size();
	for ( size_t i = 0; i < plain.candidates.size() && sameNumbers; i++ )
		if ( plain.candidates[ i ].error != observed.candidates[ i ].error
			|| plain.candidates[ i ].p != observed.candidates[ i ].p
			|| plain.candidates[ i ].iterations != observed.candidates[ i ].iterations )
			sameNumbers = false;
	expect( sameNumbers,
		"48c: every candidate's error, p-value and iteration count is unchanged" );

	// The progress stream itself is internally consistent
	unsigned finished = 0;
	bool consistent = !observed.progress.empty();
	unsigned lastFits = 0;
	for ( size_t i = 0; i < observed.progress.size(); i++ )
	{
		const RegressNet::Progress& p = observed.progress[ i ];
		if ( p.direction != "reverse" ) consistent = false;
		if ( !( p.candidate >= 1 && p.candidate <= p.candidatesThisStep ) ) consistent = false;
		if ( p.inputs != vector< unsigned >( 1, p.variable ) ) consistent = false;
		if ( p.phase.empty() ) consistent = false;
		if ( p.fitsCompleted < lastFits ) consistent = false;
		lastFits = p.fitsCompleted;
		if ( p.finished )
		{
			finished++;
			if ( p.stopReason.empty() ) consistent = false;
		}
	}
	expect( consistent, "49a: every progress announcement is internally consistent" );
	expect( finished == observed.candidates.size(),
		"49b: exactly one finished announcement per candidate fit" );
}

// ---------------------------------------------------------------------------
// 50-54. THE WINNER OF A PASS IN WHICH EVERY p-VALUE IS EXACTLY ZERO.
//
// FAIL-FIRST. Against the code before this commit, 52 reported a different
// variable on every run and 53 killed the process with SIGBUS: `largest_var`
// was declared without an initializer and assigned only inside
// `if ( p > largest_p )` from largest_p = 0, so a pass in which every candidate
// underflows to exactly 0.0 never assigned it, and the unconditional read at
// the foot of the pass was undefined behavior.
//
// THE POLICY, and it is not the one this file's author first proposed:
// **p == 0 is a p-value, not the absence of one.** It is a candidate that was
// successfully compared and found maximally significant. So the first
// successfully calculated p-value becomes the winner -- zero included -- and
// later candidates replace it only on a strict `p > largest_p`, which is what
// preserves reverse's existing first-candidate tie behavior. The stop test
// stays strictly `largest_p < threshold`, so at threshold 0 a valid p == 0
// remains ELIGIBLE FOR REMOVAL and the pass proceeds. Reading an all-zero pass
// as "remove nothing" would have quietly changed published threshold behavior.
// ---------------------------------------------------------------------------
static void test_all_pvalues_zero()
{
	DataSet d = allSignificantData();
	Logistic net;
	double e_in = fitSource( net, d );

	Run r = regress( net, e_in, singles( 4 ), false, 0.05 );

	// THE FIXTURE'S OWN PRECONDITION. Without exact zeros, 51-54 are testing an
	//    ordinary pass and guard nothing they claim to.
	unsigned zeros = 0, firstPass = 0;
	for ( size_t i = 0; i < r.candidates.size(); i++ )
		if ( r.candidates[ i ].step == 0 )
		{
			firstPass++;
			if ( r.candidates[ i ].p == 0.0 ) zeros++;
		}
	expect( firstPass == 4 && zeros == 4,
		"50: every first-pass p-value underflows to exactly zero (fixture precondition)" );

	// 51 -- it completes, and it does not crash
	expect( r.threw.empty() && r.complete,
		"51a: an all-zero pass completes rather than crashing or refusing" );
	// largest_p is 0, and the stop test is strict, so 0 < 0.05 stops the
	//    procedure with nothing removed. That is the correct reading.
	expect( r.path.empty()
		&& r.finalVariables == vector< unsigned >( { 0, 1, 2, 3 } ),
		"51b: at threshold 0.05 the strict stop test removes nothing" );

	// 52 -- THE WINNER IS DEFINED AND DETERMINISTIC. This is the assertion the
	//    uninitialized read fails: before the fix, two runs of the same binary
	//    printed two different variable numbers here.
	Logistic net2;
	double e2 = fitSource( net2, d );
	Run again = regress( net2, e2, singles( 4 ), false, 0.05 );
	expect( r.report == again.report,
		"52: two runs of an all-zero pass produce identical reports" );

	// 53 -- EXACT-ZERO TIES KEEP THE FIRST ELIGIBLE CANDIDATE. All four are
	//    tied at zero, so the first one considered must be the one named.
	expect( has( r.report, "the largest was variable 0" ),
		"53: on an exact-zero tie the FIRST eligible candidate is the winner" );
}

// 54 -- THRESHOLD 0 KEEPS THE EXISTING STRICT SEMANTICS, and this is the case
// that crashed. `largest_p < threshold` is 0 < 0, which is false, so the pass
// takes the else branch and ADOPTS its winner. Before the fix it adopted an
// indeterminate variable number and indexed variable_defs with it next pass:
// measured exit status 138, SIGBUS.
//
// RUN LAST, and alone: it is the case most likely to abort the process, and an
// abort here must not cost the observation of every case before it.
static void test_threshold_zero_is_still_strict()
{
	DataSet d = allSignificantData();
	Logistic net;
	double e_in = fitSource( net, d );

	Run r = regress( net, e_in, singles( 4 ), false, 0.0 );

	expect( !r.path.empty(),
		"54a: at threshold 0 a valid p == 0 is still eligible, so removal proceeds" );
	expect( !r.path.empty() && r.path[ 0 ].second == 0 && r.path[ 0 ].first == 0.0,
		"54b: the variable removed is the first eligible candidate, at p == 0" );
	// Every variable it names must be a real one -- the indeterminate value was
	//    a number far outside this range
	bool named = true;
	for ( size_t i = 0; i < r.path.size(); i++ )
		if ( r.path[ i ].second > 3 ) named = false;
	for ( size_t i = 0; i < r.candidates.size(); i++ )
		if ( r.candidates[ i ].variable > 3 ) named = false;
	expect( named, "54c: every variable the analysis names actually exists" );
}

// ---------------------------------------------------------------------------
// 55-58. THE ELIGIBILITY POLICY, ALL FOUR COMBINATIONS.
//
// requireConvergedFit refuses on two INDEPENDENT conditions, and case 29 only
// ever exercised the first. The second cannot be reached by training on any
// portable fixture -- measured in audit section 15.7: 270 configurations, zero
// converged-and-non-finite fits, and the single point that did hit moved under
// a change of learning rate, input scale, row count or seed. Asking the
// predicate directly is deterministic on every platform, which a fixture built
// on floating-point overflow is not.
//
// The subclass exposes a protected STATIC PURE PREDICATE. No numerical state
// becomes visible, and nothing mutable is widened.
// ---------------------------------------------------------------------------
class EligibilityProbe : public RegressNet {
public:
	using RegressNet::candidateFitEligible;
};

static void test_candidate_eligibility_policy()
{
	const double finite = 0.25;
	const double inf = numeric_limits< double >::infinity();
	const double nan = numeric_limits< double >::quiet_NaN();

	// The control: the predicate must ACCEPT something, or "refused" below is
	//    satisfied by a function that refuses everything
	expect( EligibilityProbe::candidateFitEligible(
			Iterative::STOP_GRADMAX, finite ),
		"55: converged + finite is eligible" );

	// THE INDEPENDENT HALF. A saturated fit ends on a convergence reason with a
	//    non-finite error; the stop reason alone cannot see the numbers.
	expect( !EligibilityProbe::candidateFitEligible(
			Iterative::STOP_GRADMAX, inf ),
		"56a: converged + infinite is REFUSED" );
	expect( !EligibilityProbe::candidateFitEligible(
			Iterative::STOP_GRADMAX, nan ),
		"56b: converged + not-a-number is REFUSED" );

	expect( !EligibilityProbe::candidateFitEligible(
			Iterative::STOP_MAX_ITERATIONS, finite ),
		"57: unconverged + finite is refused" );
	expect( !EligibilityProbe::candidateFitEligible(
			Iterative::STOP_MAX_ITERATIONS, nan ),
		"58a: unconverged + not-a-number is refused" );
	expect( !EligibilityProbe::candidateFitEligible(
			Iterative::STOP_CANCELLED, inf ),
		"58b: cancelled + infinite is refused" );

	// Every convergence reason the engine recognizes must pass the first half,
	//    so 56 is about the SECOND half rather than about one lucky token
	bool everyConvergedReasonAccepted =
		EligibilityProbe::candidateFitEligible( Iterative::STOP_MIN_ERROR, finite )
		&& EligibilityProbe::candidateFitEligible( Iterative::STOP_CHANGE, finite )
		&& EligibilityProbe::candidateFitEligible( Iterative::STOP_WINDOW, finite )
		&& EligibilityProbe::candidateFitEligible( Iterative::STOP_PLATEAU, finite )
		&& EligibilityProbe::candidateFitEligible( Iterative::STOP_EARLY_STOP, finite );
	expect( everyConvergedReasonAccepted,
		"58c: every convergence reason is eligible when the error is finite" );
}

// ---------------------------------------------------------------------------
// 59. A PASS WITH NOTHING COMPARABLE REFUSES.
//
// Invoked directly rather than reached through a run, and the reason is NOT that
// it is unreachable -- an earlier version of this comment said so and was wrong.
//
// stats::pX2 is gammq. It rejects a non-positive shape parameter or a negative
// argument, neither of which an exact input-node partition can produce -- but it
// ALSO throws out of gcf and gser when the continued fraction or the series
// fails to converge within ITMAX iterations, on entirely valid degrees of
// freedom and chi-square. That is not hypothetical: this repository's own oracle
// reference output records "a too large, ITMAX too small in gcf" defeating a
// Hosmer-Lemeshow p-value (tests/oracle/xor_reference_output.txt).
//
// So a pass in which every candidate's calculation fails numerically is a real
// state, and this refusal is a live defense. What is missing is a PORTABLE
// end-to-end stepwise fixture that drives a whole pass into it -- and a fixture
// tuned to sit on a numerical-convergence boundary would be exactly the bet the
// DFA singular fixture lost on Ubuntu. Invoking the method is deterministic
// everywhere; that is why it is done this way.
// ---------------------------------------------------------------------------
class PassProbe : public RegressNet {
public:
	using RegressNet::requireComparablePass;
};

static void test_no_comparable_candidate_refuses()
{
	PassProbe probe;
	string threw, said;
	{
		util::ScreenCapture hush;
		try { probe.requireComparablePass( 3 ); }
		catch ( RegressNet::RegressNetErr& e ) { threw = e.what(); }
		said = hush.text();
	}

	expect( !threw.empty(), "59a: a pass with nothing comparable refuses" );
	expect( has( threw, "pass 3" ) && has( threw, "no variable was selected" ),
		"59b: the refusal names the pass and says nothing was selected" );
	expect( has( said, "no p-value could be calculated" )
		&& has( said, "none is selected" ),
		"59c: it explains itself in the report before throwing" );
	// It must not present a conclusion, and must not name a winner it has not got
	expect( !has( said, "the largest was variable" )
		&& !has( said, "stepwise regression complete." ),
		"59d: it names no winner and prints no completion conclusion" );
	expect( probe.getSelectionPath().empty() && probe.getFinalVariables().empty()
		&& !probe.getComplete(),
		"59e: no candidate is marked selected and the analysis is not complete" );
}

// ---------------------------------------------------------------------------
// 60. copy() CARRIES e_in.
//
// e_in is the baseline every first-pass reverse comparison is made against. The
// constructor now initializes it, but that default is not observable through the
// public API -- setNetwork() overwrites it on every path that reaches an
// analysis, and no path reaches one without setNetwork(). What IS observable,
// and is the same family of defect, is whether copy() carries it: a RegressNet
// copied before it is run must compare against the same baseline as the
// original. Deleting `e_in = rhs.e_in` from copy() makes this indeterminate.
// ---------------------------------------------------------------------------
static void test_copy_carries_the_baseline()
{
	DataSet d = mixedData();
	Logistic net;
	double e_in = fitSource( net, d );

	RegressNet original;
	original.setNetwork( &net, e_in );
	original.setInputStructure( singles( 4 ) );
	original.setThreshold( 0.05 );

	RegressNet copy( original ); // copy constructor -> copy()

	string threw;
	{
		util::ScreenCapture hush;
		try { copy.reverse_regress(); }
		catch ( exception& e ) { threw = e.what(); }
	}
	expect( threw.empty() && !copy.getCandidates().empty(),
		"60a: the copied analysis ran (control)" );
	expect( !copy.getCandidates().empty()
		&& copy.getCandidates()[ 0 ].priorError == e_in,
		"60b: a copied RegressNet compares against the same baseline error" );
}

// ---------------------------------------------------------------------------
int main()
{
	test_fixture_is_discriminating();
	test_reverse_completed();
	test_reverse_result();
	test_forward_completed();
	test_direction_semantics();
	test_threshold_stopping();
	test_ceiling_exhausted_candidate();
	test_cancelled_candidate();
	test_forward_baseline_refusal();
	test_grouping();
	test_invalid_structure();
	test_lms_refusal_does_not_mutate();
	test_source_is_immutable();
	test_history();
	test_progress_is_observation_only();
	test_candidate_eligibility_policy();
	test_no_comparable_candidate_refuses();
	test_copy_carries_the_baseline();
	test_all_pvalues_zero();
	test_threshold_zero_is_still_strict(); // last: the case that used to abort

	cout << ( failures ? "FAILURES: " : "all stepwise characterization cases pass (" )
		<< failures << ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
