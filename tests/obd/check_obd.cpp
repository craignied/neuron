// check_obd.cpp : the OBD hidden-layer resizing primitives (ROADMAP 2 Phase 4).
//
// SimpleProp::growHidden / removeHidden / hiddenSaliency are the engine half of
// the grow-then-prune sizing driver (src/obd.{h,cpp}). This program builds a
// small learnable 1-output discrete problem, trains a SimpleProp, and asserts:
//
//   1. growHidden is INVISIBLE -- the forward output of every training exemplar
//      is bit-identical before and after a grow (new units carry zero outgoing
//      weight and the output bias is relocated to the new last slot).
//   2. grow-then-remove ROUND-TRIPS -- removing exactly the grown units returns
//      every output to its pre-grow value, bit for bit.
//   3. saliency SEMANTICS -- a freshly grown (zero-outgoing) unit has saliency
//      exactly 0, every trained unit's saliency is > 0, and the bias is not
//      counted (one value per hidden unit).
//   4. training still WORKS after both ops, for all three optimizers, and a
//      canonical run keeps reducing the error from the grown network.
//
// Each assertion was watched to FAIL against a deliberately sabotaged build
// (standing rule 2): skipping the bias relocation breaks (1) alone; an
// off-by-one keep-list breaks (2); indexing the bias slot in saliency breaks
// (3). See docs/obd_plan.md.

#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include "simpleprop.h"
#include "dataset.h"
#include "utility.h"
#include "obd.h"

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

// A SimpleProp that exposes the forward output of a training exemplar. The
// output o is a protected Network member; reading it directly (rather than
// through the TwoSet guesses reportAccuracy writes) keeps the bit-identity
// check free of any intervening rounding.
class ProbeProp : public SimpleProp {
public:
	unsigned trainRows() { return Train.rows(); }
	double trainOutput( unsigned r ) { forward( Train, r ); return o; }
};

// Build a learnable DataSet: a linearly separable 2-input problem with a little
// noise, split into train/test. randomize() normalizes inputs and stratifies.
static DataSet makeData( unsigned n, unsigned nTest )
{
	Matrix< double > raw( n, 3 ); // 2 inputs + 1 discrete outcome
	// A deterministic, reproducible spread of points around x0 + x1 = 0
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
	d.randomize( nTest ); // stratified train/test split + input normalization
	return d;
}

// Train a ProbeProp to a modest error so its hidden units genuinely differ
static void trainProbe( ProbeProp& sp, DataSet& d, unsigned hidden, unsigned iters )
{
	sp.setDataSet( d );
	sp.setHidden( hidden );
	sp.setHistory( false );
	sp.setLastop( false );
	sp.setLogPrint( false );
	sp.randomize();
	sp.setMaxIterations( iters );
	sp.train();
}

// (1) grow is invisible + (2) grow-then-remove round-trips
static void test_grow_remove()
{
	util::set_seed( 1234 );
	DataSet d = makeData( 150, 45 );

	ProbeProp sp;
	trainProbe( sp, d, 4, 400 );

	unsigned rows = sp.trainRows();
	vector< double > before( rows );
	for ( unsigned r = 0; r < rows; r++ )
		before[ r ] = sp.trainOutput( r );

	// Grow by 2: warm start, so every output must be bit-identical
	sp.growHidden( 2 );
	bool growIdentical = true;
	for ( unsigned r = 0; r < rows; r++ )
		if ( sp.trainOutput( r ) != before[ r ] )
			growIdentical = false;
	expect( growIdentical, "growHidden leaves every training output bit-identical" );

	// Remove exactly the two grown units (indices 4 and 5) -> back to 4 units
	vector< unsigned > drop = { 4, 5 };
	sp.removeHidden( drop );
	bool removeIdentical = true;
	for ( unsigned r = 0; r < rows; r++ )
		if ( sp.trainOutput( r ) != before[ r ] )
			removeIdentical = false;
	expect( removeIdentical,
		"grow-then-remove returns every training output to its exact prior value" );
}

// (3) saliency semantics
static void test_saliency()
{
	util::set_seed( 55 );
	DataSet d = makeData( 150, 45 );

	ProbeProp sp;
	trainProbe( sp, d, 4, 400 );

	vector< double > base = sp.hiddenSaliency();
	bool sizeOk = ( base.size() == 4 ); // one per hidden unit, bias excluded
	bool trainedPositive = true;
	for ( double s : base )
		if ( !( s > 0 ) )
			trainedPositive = false;
	expect( sizeOk && trainedPositive,
		"saliency has one positive value per trained hidden unit (bias excluded)" );

	sp.growHidden( 2 );
	vector< double > grown = sp.hiddenSaliency();
	bool freshZero = ( grown.size() == 6
		&& grown[ 4 ] == 0.0 && grown[ 5 ] == 0.0 );
	expect( freshZero,
		"a freshly grown (zero-outgoing) unit has saliency exactly zero" );
}

// (4) training still works after grow/remove, all three optimizers
static void test_train_after_ops()
{
	bool allFinite = true, canonicalImproved = false;

	for ( unsigned type = 0; type <= 2; type++ )
	{
		util::set_seed( 100 + type );
		DataSet d = makeData( 150, 45 );

		ProbeProp sp;
		trainProbe( sp, d, 3, 200 );
		sp.growHidden( 2 );      // 3 -> 5
		vector< unsigned > one = { 0 };
		sp.removeHidden( one );  // 5 -> 4

		sp.setTrainingType( type );
		sp.setBatchEpoch( true ); // CGD/Shanno need a true batch gradient
		sp.setMaxIterations( 50 );
		double err = sp.train();
		if ( !( isfinite( err ) && err >= 0 ) )
			allFinite = false;

		if ( type == 0 )
		{
			// A longer canonical run from here should keep reducing the error
			sp.setMaxIterations( 800 );
			double err2 = sp.train();
			canonicalImproved = ( isfinite( err2 ) && err2 <= err );
		}
	}

	expect( allFinite,
		"training after grow+remove gives a finite error for every optimizer" );
	expect( canonicalImproved,
		"canonical training keeps reducing the error from the resized network" );
}

// A raw-only DataSet (a training set, no held-out test set) for the refusal test
static DataSet makeTrainOnly( unsigned n )
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
	d.setInput( 2 ); d.setOutput( 1 ); d.setDiscrete( true ); d.setHistory( false );
	d.setRawMatrix( raw );
	d.raw2train(); // convert to a training set with NO test split
	return d;
}

// The grow-then-prune driver: a real end-to-end search on the learnable set.
static void test_driver()
{
	util::set_seed( 7 );
	DataSet d = makeData( 150, 45 );

	obd::Config cfg;
	cfg.hStart = 2; cfg.hMax = 5; cfg.iterBudget = 300; cfg.sampleEvery = 10;
	cfg.earlyStopPatience = 2; cfg.growPatience = 1;

	obd::Result r = obd::run( d, cfg, nullptr, nullptr );

	expect( r.ok && r.winner != nullptr,
		"the search returns ok with a trained winning network" );
	expect( r.selectedHidden >= 1 && r.selectedHidden <= cfg.hMax,
		"the selected hidden count is within the searched range" );

	// The grow-phase sizes are consecutive from hStart -- the warm-start growth
	vector< unsigned > grow;
	for ( const obd::SizeTrial& t : r.history )
		if ( t.phaseGrow )
			grow.push_back( t.hidden );
	bool consecutive = !grow.empty() && grow.front() == cfg.hStart;
	for ( size_t i = 1; i < grow.size(); i++ )
		if ( grow[ i ] != grow[ i - 1 ] + 1 )
			consecutive = false;
	expect( consecutive,
		"grow-phase sizes step up by one from hStart (warm-start growth)" );

	// Every trial carries a per-size classification accuracy in 0..1
	bool caOk = !r.history.empty();
	for ( const obd::SizeTrial& t : r.history )
		if ( !( t.trainCA >= 0 && t.trainCA <= 1 && t.testCA >= 0 && t.testCA <= 1 ) )
			caOk = false;
	expect( caOk, "each size trial records train and test classification accuracy in 0..1" );

	// Refusal: no held-out test set means no validation signal for early stopping
	DataSet trainOnly = makeTrainOnly( 120 );
	obd::Result refused = obd::run( trainOnly, cfg, nullptr, nullptr );
	expect( !refused.ok && refused.winner == nullptr && !refused.message.empty(),
		"the search refuses a dataset with no test set" );

	// Cancel: an already-set flag ends the search with no winner
	atomic< bool > cancel{ true };
	obd::Result cancelled = obd::run( d, cfg, nullptr, &cancel );
	expect( cancelled.cancelled && cancelled.winner == nullptr,
		"a set cancel flag ends the search as cancelled" );
}

// Overtraining MUST early-stop a size, not run to the iteration budget. Built
// deterministically: the test set's labels are INVERTED relative to the rule the
// training set teaches, so the held-out error can only RISE as the net learns --
// its minimum is at the very start, and validation early stopping must fire.
static void test_early_stop_fires()
{
	util::set_seed( 3 );

	// Same inputs, opposite labels in train vs test
	unsigned n = 80;
	Matrix< double > tr( n, 3 ), te( n, 3 );
	for ( unsigned i = 0; i < n; i++ )
	{
		double x0 = -1.0 + 2.0 * ( ( i * 37 ) % 100 ) / 99.0;
		double x1 = -1.0 + 2.0 * ( ( i * 53 ) % 100 ) / 99.0;
		unsigned rule = ( x0 + x1 > 0 ) ? 1 : 0;
		tr( i, 0 ) = x0; tr( i, 1 ) = x1; tr( i, 2 ) = rule;
		te( i, 0 ) = x0; te( i, 1 ) = x1; te( i, 2 ) = 1 - rule; // inverted
	}
	DataSet d;
	d.setInput( 2 ); d.setOutput( 1 ); d.setDiscrete( true ); d.setHistory( false );
	d.setTrainMatrix( tr );
	d.setTestMatrix( te );

	obd::Config cfg;
	cfg.hStart = 4; cfg.hMax = 5; cfg.iterBudget = 2000; cfg.sampleEvery = 5;
	cfg.earlyStopPatience = 2; cfg.growPatience = 1;

	obd::Result r = obd::run( d, cfg, nullptr, nullptr );

	bool anyEarlyStop = false;
	for ( const obd::SizeTrial& t : r.history )
		if ( t.stop == Iterative::STOP_OBSERVER )
			anyEarlyStop = true;
	expect( r.ok && anyEarlyStop && !r.cancelled,
		"a size whose test error only rises is early-stopped, not run to budget" );
}

// The train-plateau backstop must be WIRED into each size's training: a size
// that converges flat never trips the test-error rise, so the plateau detector
// is what saves its remaining budget. Proven with a huge tolerance (0.5): any
// window-over-window relative improvement under 50% strikes, so the plateau
// must fire long before the generous budget -- unless the setAutoStop call is
// missing, in which case every size runs to STOP_MAX_ITERATIONS and this fails.
static void test_plateau_backstop_fires()
{
	util::set_seed( 11 );
	DataSet d = makeData( 150, 45 );

	obd::Config cfg;
	cfg.hStart = 2; cfg.hMax = 3; cfg.iterBudget = 100000; cfg.sampleEvery = 50;
	cfg.growPatience = 1;
	cfg.plateauTol = 0.5; cfg.plateauWindow = 10; // strike on anything but a halving

	obd::Result r = obd::run( d, cfg, nullptr, nullptr );

	bool anyPlateau = false;
	for ( const obd::SizeTrial& t : r.history )
		if ( t.stop == Iterative::STOP_PLATEAU )
			anyPlateau = true;
	expect( r.ok && anyPlateau,
		"the train-plateau backstop ends a flat size before the budget" );
}

int main()
{
	// The engine's training reports go to util::screen(); this test reads none
	//    of them, so route them to a discard stream and keep the output to the
	//    ok/FAIL lines
	ostringstream discard;
	util::set_screen( discard );

	test_grow_remove();
	test_saliency();
	test_train_after_ops();
	test_driver();
	test_early_stop_fires();
	test_plateau_backstop_fires();

	util::set_screen( cout );

	if ( failures == 0 )
	{
		cout << "check_obd: hidden-layer resizing primitives verified" << endl;
		return 0;
	}
	cerr << "check_obd: FAILED" << endl;
	return 1;
}
