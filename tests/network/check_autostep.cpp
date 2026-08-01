// check_autostep.cpp : the automatic step-size search, in all four models.
//
// SimpleProp, BareProp, BackProp and Logistic each carry their own copy of the
// eta search at the top of trainSet(). The four copies differ in exactly two
// ways, and everything else -- the loop, the lastG/lastF buffering, the
// oldErrorAccumulate transitions, the final real innerTrainSet() -- is the same
// text four times:
//
//   * THE GUARD. Logistic searches whenever automaticStepSizeFlag is set; the
//     three feed-forward nets require batchEpochFlag TOO. That difference is
//     real and must survive any consolidation: a shared implementation that
//     quietly gave Logistic the stricter guard would silently stop searching
//     for it.
//   * THE WEIGHT SNAPSHOT: hW/oW, Weights, or W.
//
// This pins the mechanism BEFORE that consolidation.
//
// WHY THERE ARE NO MULTI-ITERATION FLOATING-POINT LITERALS HERE.
// check_bpoptimizer learned this the expensive way on 2026-08-01: macOS-captured
// bit-exact doubles failed on Ubuntu/GCC and Windows/MSVC while every structural
// assertion passed. What this test pins instead is portable and is exactly what
// the refactor must not change:
//
//   1. THE NUMBER OF innerTrainSet() CALLS per trainSet(). An integer, and the
//      most direct statement of the mechanism there is: the search runs the
//      training set repeatedly and then runs it once more for real.
//   2. THE SELECTED eta. The search starts at 1.0 and multiplies by gamma
//      (0.8) per loop, so the answer is a small power of a fixed constant --
//      the same on every platform, and it moves if the loop runs a different
//      number of times or scales at the wrong point.
//   3. THE GUARD, per model, in all four combinations of batch and auto.
//   4. RESTORATION, without literals: a model that searched and settled on
//      eta* must end up where a model that never searched and was simply GIVEN
//      eta* ends up. If the trial weights were not restored, or lastG/lastF
//      were not restored, the two disagree. This is checked with canonical,
//      CGD and Shanno, because lastG/lastF only matter for the latter two.
//   5. BOTH TERMINATIONS: the loop ceiling (maxLoops) and the error-difference
//      threshold (deltaError). Both are protected members, so a probe can set
//      them and reach each exit deliberately instead of hoping a real fit
//      wanders into one.
//   6. THE FIRST ITERATION IS SPECIAL: at iteration 0 there is no previous
//      error, so the first pass through the loop only primes
//      oldErrorAccumulate. That gives a different selected eta from a later
//      iteration for the same number of calls, which is asserted both ways.
//
// The exact before/after values on the refactoring machine are a separate,
// local artifact (--capture), diffed across the extraction. That is a
// same-machine proof of behavior preservation, which is what it is for; it is
// not a cross-platform contract, which is what it is not.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "simpleprop.h"
#include "bareprop.h"
#include "backprop.h"
#include "logistic.h"
#include "dataset.h"
#include "utility.h"

using namespace std;

int failures = 0;

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

static void expectEq( unsigned got, unsigned want, const string& what )
{
	if ( got != want )
		cout << "         got " << got << ", expected " << want << endl;
	expect( got == want, what );
}

static void expectNear( double got, double want, double tol, const string& what )
{
	bool ok = fabs( got - want ) <= tol;
	if ( !ok )
		printf( "         got %a (%.17g), expected %a (%.17g)\n",
			got, got, want, want );
	expect( ok, what );
}

// Reaches the search's own state. Everything here is protected in Network or
// Iterative; nothing is added to production classes for the test's benefit.
template < class NET >
class Probe : public NET {
public:
	Probe() : calls ( 0 ) { }

	// Every pass through the training set, counted. trainSet() calls this
	//    virtually, so both the trial passes and the final real one land here.
	double innerTrainSet() override { calls++; return NET::innerTrainSet(); }

	unsigned calls;
	void resetCalls() { calls = 0; }

	double etaNow() const { return this->eta; }
	void setIteration( unsigned t ) { this->iteration = t; }
	void setDeltaError( double d ) { this->deltaError = d; }
	void setMaxLoops( unsigned m ) { this->maxLoops = m; }
	double gammaNow() const { return this->gamma; }

	// One number standing for the whole weight state
	double signature()
	{
		double sum = 0;
		Matrix< double >& m = this->theData.getTrainMatrix();
		for ( unsigned r = 0; r < m.rows(); r++ )
		{
			this->forward( m, r );
			sum += this->o;
		}
		return sum;
	}
};

static DataSet makeData( unsigned n )
{
	Matrix< double > raw( n, 3 );
	for ( unsigned i = 0; i < n; i++ )
	{
		double x0 = -1.0 + 2.0 * ( ( i * 37 ) % 100 ) / 99.0;
		double x1 = -1.0 + 2.0 * ( ( i * 53 ) % 100 ) / 99.0;
		raw( i, 0 ) = x0;
		raw( i, 1 ) = x1;
		raw( i, 2 ) = ( x0 + x1 > 0.55 ) ? 1 : 0;
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

// Architecture differs per model; everything else is set identically.
static void arch( Probe< SimpleProp >& p ) { p.setHidden( 3 ); }
static void arch( Probe< BareProp >& p )   { p.setHidden( 3 ); }
static void arch( Probe< Logistic >& p )   { }
static void arch( Probe< BackProp >& p )
{
	vector< unsigned > layers;
	layers.push_back( 4 );
	layers.push_back( 3 ); // TWO hidden layers: a real BackProp
	p.setHidden( layers );
}

template < class NET >
static void configure( Probe< NET >& p, DataSet& d, bool batch, bool autoStep,
	unsigned type )
{
	p.setDataSet( d );
	arch( p );
	p.setHistory( false );
	p.setLastop( false );
	p.setLogPrint( false );
	p.setQuiet( true );
	p.setXEerror();
	p.setWeightDecay( false );
	p.setDecay( 0 );
	p.setBatchEpoch( batch );
	p.setAutoStepSize( autoStep );
	p.setTrainingType( type );
	p.setEta( 0.5 );
	util::set_seed( 7 );
	p.randomize();
	p.resetCalls();
}

// A REAL later iteration. Setting `iteration` by hand is not enough: the loop's
//    first branch compares against oldErrorAccumulate, which only holds a
//    previous error after an actual earlier call. Faking the counter alone
//    leaves it at its constructed value, ErrorDifference comes out negative,
//    and the loop breaks on its first pass -- a different exit than the one
//    under test. So iteration 0 is run for real first.
template < class NET >
static void primeToSecondIteration( Probe< NET >& p )
{
	p.setIteration( 0 );
	{
		util::ScreenCapture hush;
		p.trainSet();
	}
	p.setIteration( 1 );
	p.resetCalls();
}

// --- 1 and 2 and 6. call count and selected eta ----------------------------
//
// gamma is 0.8 and maxLoops is 3, so with the default deltaError (1e-15) the
// loop always runs to its ceiling:
//
//   iteration 0   loop 1 only PRIMES oldErrorAccumulate (no previous error),
//                 loops 2 and 3 each scale eta -> eta = 0.8^2 = 0.64
//   iteration > 0 loops 1, 2 and 3 each scale eta -> eta = 0.8^3 = 0.512
//
// and either way that is 3 trial passes plus 1 real one.

template < class NET >
static void checkSearch( const char* who, bool guardNeedsBatch )
{
	cout << "-- " << who << ": the search --" << endl;

	// Auto OFF: exactly one pass, eta untouched
	{
		DataSet d = makeData( 90 );
		Probe< NET > p;
		configure( p, d, true, false, 0 );
		util::ScreenCapture hush;
		p.trainSet();
		expectEq( p.calls, 1, string( who )
			+ ": with the search off, trainSet is ONE pass" );
		expectNear( p.etaNow(), 0.5, 0.0, string( who )
			+ ": ...and eta is left exactly as set" );
	}

	// Auto ON at iteration 0: three trial passes, then the real one
	{
		DataSet d = makeData( 90 );
		Probe< NET > p;
		configure( p, d, true, true, 0 );
		p.setIteration( 0 );
		util::ScreenCapture hush;
		p.trainSet();
		expectEq( p.calls, 4, string( who )
			+ ": iteration 0 runs 3 trial passes and 1 real one" );
		expectNear( p.etaNow(), 0.8 * 0.8, 0.0, string( who )
			+ ": ...and selects eta = gamma^2 (the first loop only primes)" );
	}

	// Auto ON at a later iteration: the first loop is no longer special
	{
		DataSet d = makeData( 90 );
		Probe< NET > p;
		configure( p, d, true, true, 0 );
		primeToSecondIteration( p );
		util::ScreenCapture hush;
		p.trainSet();
		expectEq( p.calls, 4, string( who )
			+ ": a later iteration also runs 3 trial passes and 1 real one" );
		expectNear( p.etaNow(), 0.8 * 0.8 * 0.8, 0.0, string( who )
			+ ": ...and selects eta = gamma^3" );
	}
}

// --- 5. both terminations --------------------------------------------------

template < class NET >
static void checkTerminations( const char* who )
{
	cout << "-- " << who << ": how the loop ends --" << endl;

	// THE CEILING: a different maxLoops must give a different number of passes
	{
		DataSet d = makeData( 90 );
		Probe< NET > p;
		configure( p, d, true, true, 0 );
		primeToSecondIteration( p );
		p.setMaxLoops( 1 );
		util::ScreenCapture hush;
		p.trainSet();
		expectEq( p.calls, 2, string( who )
			+ ": maxLoops = 1 gives 1 trial pass and 1 real one" );
		expectNear( p.etaNow(), 0.8, 0.0, string( who )
			+ ": ...having scaled eta exactly once" );
	}

	// THE ERROR DIFFERENCE: a threshold nothing can beat breaks immediately,
	//    BEFORE eta is scaled at all
	{
		DataSet d = makeData( 90 );
		Probe< NET > p;
		configure( p, d, true, true, 0 );
		primeToSecondIteration( p );
		p.setDeltaError( 1e9 );
		util::ScreenCapture hush;
		p.trainSet();
		// ErrorDifference is initialised to 1.0 purely to get the while
		//    condition past its first evaluation, so a deltaError above 1.0
		//    fails that condition immediately and NO trial pass runs at all.
		expectEq( p.calls, 1, string( who )
			+ ": an unreachable deltaError runs no trial pass at all" );
		// ...but eta was already reset to 1.0 ABOVE the loop, so the search
		//    still moves it off whatever the caller set. Easy to lose in an
		//    extraction that initialises eta inside the loop instead.
		expectNear( p.etaNow(), 1.0, 0.0, string( who )
			+ ": ...yet eta is still reset to 1.0, which happens before the loop" );
	}
}

// --- 4. restoration, without a single literal ------------------------------
//
// A model that SEARCHED and settled on eta* must land exactly where a model
// that never searched and was simply GIVEN eta* lands. Trial weights left
// installed, or lastG/lastF left holding a trial iteration's history, both
// break this. Run for all three optimizers, because lastG and lastF only carry
// state for CGD and Shanno.

template < class NET >
static void checkRestoration( const char* who )
{
	const char* names[] = { "canonical", "CGD", "Shanno" };

	for ( unsigned type = 0; type < 3; type++ )
	{
		cout << "-- " << who << ": restoration, " << names[ type ] << " --" << endl;

		// The searcher
		DataSet d1 = makeData( 90 );
		Probe< NET > searched;
		configure( searched, d1, true, true, type );
		primeToSecondIteration( searched );
		double chosen;
		{
			util::ScreenCapture hush;
			searched.trainSet();
			chosen = searched.etaNow();
		}

		// The control. It must reach the start of iteration 1 in the SAME
		//    state as the searcher, so it is primed with the search ON --
		//    priming it with the search off would leave it at different
		//    weights and the comparison would be meaningless. Only iteration 1
		//    differs: no search, handed the eta the searcher chose.
		DataSet d2 = makeData( 90 );
		Probe< NET > given;
		configure( given, d2, true, true, type );
		primeToSecondIteration( given );
		given.setAutoStepSize( false );
		given.setEta( chosen );
		{
			util::ScreenCapture hush;
			given.trainSet();
		}

		expectEq( given.calls, 1, string( who ) + " " + names[ type ]
			+ ": the control runs exactly one pass" );
		expectNear( searched.signature(), given.signature(), 1e-9,
			string( who ) + " " + names[ type ]
			+ ": a searched run lands where an eta-given run lands" );
	}
}

// --- 3. THE GUARD, which is NOT the same in all four models ----------------
//
// Logistic searches on automaticStepSizeFlag alone. The three feed-forward nets
// require batchEpochFlag as well. The on-line + auto combination is not
// reachable from the menus or the GUI for Logistic, which is exactly why it
// needs a test: a consolidation that handed Logistic the stricter guard would
// silently stop searching, and nothing else here would notice.

template < class NET >
static void checkGuard( const char* who, bool needsBatch )
{
	cout << "-- " << who << ": the guard --" << endl;

	struct Combo { bool batch; bool autoStep; };
	const Combo combos[] = { { true, true }, { true, false },
		{ false, true }, { false, false } };

	for ( const Combo& c : combos )
	{
		DataSet d = makeData( 90 );
		Probe< NET > p;
		configure( p, d, c.batch, c.autoStep, 0 );
		primeToSecondIteration( p );
		{
			util::ScreenCapture hush;
			p.trainSet();
		}

		bool shouldSearch = c.autoStep && ( !needsBatch || c.batch );
		string what = string( who ) + ": batch="
			+ ( c.batch ? "on" : "off" ) + " auto="
			+ ( c.autoStep ? "on" : "off" ) + " -> "
			+ ( shouldSearch ? "searches" : "does not search" );

		expectEq( p.calls, shouldSearch ? 4u : 1u, what );
	}
}

// --- the local before/after artifact ---------------------------------------
//
// Full precision, for a SAME-MACHINE diff across the extraction. Deliberately
// not turned into assertions: see the header.

static void capture()
{
	struct Row { bool batch; bool autoStep; unsigned type; };
	const Row rows[] = {
		{ true,  false, 0 }, { true,  true,  0 },
		{ true,  true,  1 }, { true,  true,  2 },
		{ false, true,  0 }, { false, false, 0 }
	};

	for ( const Row& r : rows )
	{
#define ARM( NET, NAME ) { \
	DataSet d = makeData( 90 ); \
	Probe< NET > p; \
	configure( p, d, r.batch, r.autoStep, r.type ); \
	primeToSecondIteration( p ); \
	double e; \
	{ util::ScreenCapture hush; e = p.trainSet(); } \
	printf( "%-11s batch=%d auto=%d type=%u  calls=%u eta=%a err=%a sig=%a\n", \
		NAME, r.batch, r.autoStep, r.type, p.calls, p.etaNow(), e, \
		p.signature() ); }
		ARM( SimpleProp, "SimpleProp" )
		ARM( BareProp,   "BareProp" )
		ARM( BackProp,   "BackProp" )
		ARM( Logistic,   "Logistic" )
#undef ARM
	}
}

int main( int argc, char* argv[] )
{
	if ( argc > 1 && string( argv[ 1 ] ) == "--capture" )
	{
		capture();
		return 0;
	}

	checkSearch< SimpleProp >( "SimpleProp", true );
	checkSearch< BareProp >  ( "BareProp",   true );
	checkSearch< BackProp >  ( "BackProp",   true );
	checkSearch< Logistic >  ( "Logistic",   false );

	checkTerminations< SimpleProp >( "SimpleProp" );
	checkTerminations< Logistic >  ( "Logistic" );

	checkRestoration< SimpleProp >( "SimpleProp" );
	checkRestoration< BareProp >  ( "BareProp" );
	checkRestoration< BackProp >  ( "BackProp" );
	checkRestoration< Logistic >  ( "Logistic" );

	checkGuard< SimpleProp >( "SimpleProp", true );
	checkGuard< BareProp >  ( "BareProp",   true );
	checkGuard< BackProp >  ( "BackProp",   true );
	checkGuard< Logistic >  ( "Logistic",   false );

	cout << endl << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
