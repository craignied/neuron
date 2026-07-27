// check_gradcadence.cpp : stopping conditions are evaluated independently of
// reporting cadence.
//
// THE BUG (found 2026-07-26 by Sol, from a live Civic Choice walkthrough report).
// Iterative::train() recalculated the maximum absolute gradient ONLY inside the
// block that prints an iteration row, then compared that cached value against
// gradMaxLimit on every iteration. A presentation setting therefore changed
// optimization:
//
//   * logarithmic printing checks at 1..10, then 20, then 30 ... so a gradient
//     crossing between two print points stayed invisible until the next printed
//     iteration -- thousands of wasted iterations, different final weights;
//   * an iteration ceiling landing between the crossing and the next print
//     reported a FALSE failure to converge, even though the gradient the run
//     was asked to stop at had already been reached;
//   * switching linear/logarithmic printing, or changing printcount, changed
//     the stopping iteration, the final weights, the predictions, and whether
//     OBD/CV would accept the fit at all.
//
// The invariant this program guards: reporting cadence may change output
// VOLUME; it must never change optimization or fit validity.
//
// Three tests, in the order the evidence was built:
//
//   1. SCRIPTED CROSSING between print points -- an Iterative test double whose
//      gradient is above the limit through call 16 and below it from call 17.
//      Linear-every-iteration and logarithmic printing must both stop on call
//      17 with STOP_GRADMAX, in identical model state, having asked for the
//      gradient exactly once per completed iteration (never twice on a printed
//      one).
//   2. CEILING between the crossing and the next print point -- the same script
//      with a budget that runs out at call 18. Must be STOP_GRADMAX on 17, not
//      a false STOP_MAX_ITERATIONS.
//   3. REAL-MODEL cadence invariance -- seeded SimpleProp clones from identical
//      starting weights, trained under logarithmic, linear-every-iteration and
//      coarse-linear printing, for ALL THREE optimizers (canonical / CGD /
//      Shanno -- Network::getGradMax takes a different branch for canonical,
//      which packs stackG, than for the two that maintain currGradMax). Stop
//      reason, stopping iteration, final error and every training-set
//      prediction must be identical.
//
// RED PROOF against the pre-fix engine, 24 failures, all OBSERVED, not assumed
// (standing rule 2; also in docs/HISTORY.md 2026-07-26). The guard assertions
// -- that the fixture really stops on the gradient rule, that the crossing is
// past the dense head of the log schedule, and that every pair of runs began
// from identical weights -- were GREEN pre-fix, as premises must be:
//
//   test 1: every-iteration stopped on call 17 having queried the gradient 17
//           times; logarithmic ran on to call 21 and queried it just 11 times.
//   test 2: logarithmic ended max_iterations after 18 calls -- the gradient it
//           was asked to stop at had been reached on call 17 and never looked at.
//   test 3: stopping iteration, by printing schedule (every / logarithmic /
//           linear-1000), from bit-identical starting weights:
//              canonical  304 / 400 / 1000
//              CGD        308 / 400 / 1000
//              Shanno     119 / 200 / 1000
//           -- three different fitted models per optimizer, chosen by a
//           presentation setting. Shanno's coarse-linear run did 8x the work.

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "iterative.h"
#include "simpleprop.h"
#include "dataset.h"
#include "utility.h"

using namespace std;

int failures = 0;

void expect( bool ok, const string& what )
{
	if ( ok )
		cout << "ok - " << what << endl;
	else
	{
		cout << "FAIL - " << what << endl;
		failures++;
	}
}

// ---------------------------------------------------------------------------
// 1 + 2. A scripted Iterative
// ---------------------------------------------------------------------------

// A minimal concrete Iterative whose training is a script, not a computation:
// trainSet() advances a counter and returns a monotonically shrinking error
// that no other stopping rule can fire on, and getGradMax() reads that same
// counter. Everything else is inert. This isolates the cadence question from
// every numerical property of a real model -- the gradient trace is EXACTLY
// what the test says it is, on both printing schedules.
class ScriptedIterative : public Iterative {
public:
	ScriptedIterative( unsigned crossingCall )
		: crossing ( crossingCall ), trainCalls ( 0 ), gradCalls ( 0 )
	{
		setHistory( false ); // no neuron.log
		setLastop( false ); // no model.txt
		theData.setDiscrete( false ); // no classification-accuracy columns
		// Only the gradient rule is armed: min-error, change, window and
		// plateau stay off (their defaults), so nothing else can stop the run
		// and mask a cadence difference.
		setGradStop( true );
		setGradMaxLimit( 1e-6 );
	}

	// Every completed iteration, in the order they happened. The gradient
	// crosses the limit on call number `crossing` and stays below it.
	double trainSet() override { return 1.0 / ++trainCalls; }

	double getGradMax() override
	{
		gradCalls++;
		return ( trainCalls >= crossing ) ? 1e-9 : 1.0;
	}

	unsigned completedIterations() const { return trainCalls; }
	unsigned gradientQueries() const { return gradCalls; }

	// Inert overrides -- this double reports nothing
	void setDataSet( DataSet& ) override { }
	void outputHeader( ostream& ) override { }
	void reportAccuracy( ostream& ) override { }
	void classAccuracy( ostream& ) override { }

protected:
	void runHeader( ostream& ) override { }

private:
	unsigned crossing, trainCalls, gradCalls;
};

// Run the identical script under one printing schedule. `logarithmic` selects
// the log counter (1..10, 20, 30, ...); otherwise printCount is the linear one.
static void runScript( ScriptedIterative& m, bool logarithmic, unsigned printCount,
	unsigned maxIterations )
{
	m.setLogPrint( logarithmic );
	if ( !logarithmic )
		m.setPrintCount( printCount );
	m.setMaxIterations( maxIterations );
	m.train();
}

// (1) The crossing must be seen on the same iteration under either schedule
static void test_crossing_between_print_points()
{
	// The gradient falls below the limit on completed iteration 17 -- after the
	// logarithmic schedule's print at 10 and before its next one at 20. That
	// gap is the bug's hiding place.
	ScriptedIterative dense( 17 ), sparse( 17 );

	runScript( dense, false, 1, 1000 ); // print every iteration
	runScript( sparse, true, 0, 1000 ); // 1..10, then 20, 30, ...

	expect( dense.getStopReason() == Iterative::STOP_GRADMAX,
		"printing every iteration: stops on the gradient limit" );
	expect( sparse.getStopReason() == Iterative::STOP_GRADMAX,
		"logarithmic printing: stops on the gradient limit" );

	expect( dense.completedIterations() == 17,
		"printing every iteration: stops on the FIRST qualifying iteration (17)" );
	expect( sparse.completedIterations() == 17,
		"logarithmic printing: stops on the FIRST qualifying iteration (17), "
		"not at the next print point" );
	expect( dense.completedIterations() == sparse.completedIterations(),
		"the two schedules agree on the stopping iteration" );

	// The gradient is consulted once per completed iteration -- not zero times
	// on unprinted ones (the bug) and not twice on printed ones (the naive fix
	// that hoists the call but leaves the old one in the print block).
	expect( dense.gradientQueries() == dense.completedIterations(),
		"printing every iteration: one gradient evaluation per iteration" );
	expect( sparse.gradientQueries() == sparse.completedIterations(),
		"logarithmic printing: one gradient evaluation per iteration" );
}

// (2) A ceiling between the crossing and the next print point is the false
//     non-convergence this fix exists to kill
static void test_ceiling_between_print_points()
{
	// Budget runs out on completed iteration 18: past the crossing at 17, still
	// short of the logarithmic schedule's next print at 20.
	ScriptedIterative sparse( 17 );
	runScript( sparse, true, 0, 17 ); // iteration index 0..17 == 18 iterations

	expect( sparse.getStopReason() == Iterative::STOP_GRADMAX,
		"a ceiling just past the crossing still reports the gradient stop, "
		"not a false failure to converge" );
	expect( Iterative::converged( sparse.getStopReason() ),
		"...and the run is therefore CONVERGED" );
	expect( sparse.completedIterations() == 17,
		"...on iteration 17, the first that qualified" );
}

// ---------------------------------------------------------------------------
// 3. A real model
// ---------------------------------------------------------------------------

// Exposes the forward output of a training exemplar: predictions are the
// strongest available statement of "identical model state", stronger than the
// rounded error the report prints.
class ProbeProp : public SimpleProp {
public:
	unsigned trainRows() { return Train.rows(); }
	double trainOutput( unsigned r ) { forward( Train, r ); return o; }
	unsigned stoppedAt() const { return iteration; } // the loop counter train() left
};

// Same learnable fixture the OBD tests use: a linearly separable 2-input
// problem, deterministic and reproducible.
static DataSet makeData( unsigned n, unsigned nTest )
{
	Matrix< double > raw( n, 3 );
	for ( unsigned i = 0; i < n; i++ )
	{
		double x0 = -1.0 + 2.0 * ( ( i * 37 ) % 100 ) / 99.0;
		double x1 = -1.0 + 2.0 * ( ( i * 53 ) % 100 ) / 99.0;
		raw( i, 0 ) = x0;
		raw( i, 1 ) = x1;
		raw( i, 2 ) = ( x0 + x1 > 0 ) ? 1 : 0;
	}

	DataSet d;
	d.setInput( 2 );
	d.setOutput( 1 );
	d.setDiscrete( true );
	d.setHistory( false );
	d.setRawMatrix( raw );
	d.randomize( nTest );
	return d;
}

// One training run, described only by its printing schedule. Reseeding
// immediately before randomize() is what makes the three runs clones: the
// starting weights are drawn from the same stream position, so any difference
// in the result is the cadence and nothing else. The test asserts that
// premise directly rather than trusting it -- startPredictions is the forward
// pass of the UNTRAINED network, so if two runs ever began from different
// weights the comparison says so, and a cadence failure can never be mistaken
// for an RNG divergence.
struct RunOutcome {
	Iterative::StopReason stop;
	unsigned iterations;
	double finalError;
	vector< double > startPredictions;
	vector< double > predictions;
};

static vector< double > sweepPredictions( ProbeProp& sp )
{
	vector< double > p;
	for ( unsigned r = 0; r < sp.trainRows(); r++ )
		p.push_back( sp.trainOutput( r ) );
	return p;
}

static RunOutcome trainWithCadence( DataSet& d, unsigned trainingType,
	bool logarithmic, unsigned printCount, double gradLimit, unsigned maxIterations )
{
	util::set_seed( 424242 );

	ProbeProp sp;
	sp.setDataSet( d );
	sp.setHidden( 3 );
	sp.setHistory( false );
	sp.setLastop( false );
	sp.setTrainingType( trainingType );
	sp.setBatchEpoch( true ); // CGD/Shanno need a true batch gradient
	sp.randomize();

	RunOutcome out;
	out.startPredictions = sweepPredictions( sp ); // the untrained network

	sp.setGradStop( true );
	sp.setGradMaxLimit( gradLimit );
	sp.setLogPrint( logarithmic );
	if ( !logarithmic )
		sp.setPrintCount( printCount );
	sp.setMaxIterations( maxIterations );

	out.finalError = sp.train();
	out.stop = sp.getStopReason();
	out.iterations = sp.stoppedAt();
	out.predictions = sweepPredictions( sp );
	return out;
}

static bool identical( const vector< double >& a, const vector< double >& b )
{
	if ( a.size() != b.size() || a.empty() )
		return false;
	for ( unsigned i = 0; i < a.size(); i++ )
		if ( a[ i ] != b[ i ] )
			return false;
	return true;
}

static void compareCadences( const string& label, const RunOutcome& a,
	const RunOutcome& b )
{
	expect( identical( a.startPredictions, b.startPredictions ),
		label + ": both runs began from identical weights (same RNG state)" );
	expect( a.stop == b.stop, label + ": identical stop reason" );
	expect( a.iterations == b.iterations, label + ": identical stopping iteration" );
	expect( a.finalError == b.finalError, label + ": identical final error" );
	expect( identical( a.predictions, b.predictions ),
		label + ": bit-identical predictions for every training row" );
}

// (3) Printing schedule must not move a real model's endpoint, on any optimizer
static void test_real_model_is_cadence_invariant()
{
	DataSet d = makeData( 150, 45 );

	// Gradient limits MEASURED (2026-07-26, by dumping each optimizer's whole
	// gradient trace) so that the first crossing falls strictly BETWEEN two
	// logarithmic print points -- otherwise the comparison below would hold
	// even with the bug present and would guard nothing. A limit is not a
	// tuning knob here; it is what makes the fixture discriminating.
	//
	//   canonical  1e-2 crosses at 304, next log print point 400
	//   CGD        2e-2 crosses at 308, next log print point 400
	//   Shanno     2e-3 crosses at 119, next log print point 200
	//
	// The caps are far above every crossing, so a run that ends on the ceiling
	// means the fixture drifted -- the first assertion in the loop catches it.
	struct Case { unsigned type; const char* name; double limit; unsigned cap; };
	const Case cases[] = {
		{ 0, "canonical", 1e-2, 5000 },
		{ 1, "CGD", 2e-2, 5000 },
		{ 2, "Shanno", 2e-3, 5000 }
	};

	for ( const Case& c : cases )
	{
		RunOutcome logRun = trainWithCadence( d, c.type, true, 0, c.limit, c.cap );
		RunOutcome everyRun = trainWithCadence( d, c.type, false, 1, c.limit, c.cap );
		RunOutcome coarseRun = trainWithCadence( d, c.type, false, 1000, c.limit, c.cap );

		// The fixture only means something if the run actually stopped on the
		// gradient rule; a ceiling would make all three trivially equal.
		expect( logRun.stop == Iterative::STOP_GRADMAX,
			string( c.name ) + ": the fixture really does stop on the gradient limit" );

		// ...and only if the crossing is NOT on a print point of every schedule,
		// which is what made the pre-fix endpoints differ.
		expect( logRun.iterations > 10,
			string( c.name ) + ": the crossing is past the dense head of the "
			"logarithmic schedule (so the schedules genuinely disagree pre-fix)" );

		compareCadences( string( c.name ) + " log vs every-iteration", logRun, everyRun );
		compareCadences( string( c.name ) + " log vs coarse linear", logRun, coarseRun );
	}
}

int main()
{
	// Engine reports go nowhere: this program asserts on state, not on text.
	ostringstream sink;
	util::set_screen( sink );

	test_crossing_between_print_points();
	test_ceiling_between_print_points();
	test_real_model_is_cadence_invariant();

	util::set_screen( cout );
	cout << ( failures ? "FAILED" : "PASSED" ) << " (" << failures
		<< " failures)" << endl;
	return failures ? 1 : 0;
}
