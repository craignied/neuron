// check_split.cpp : the stratified index-level splitter (ROADMAP 4 Phase 1).
//
// nsplit::stratifiedHoldout replaced an O(n^2) rejection shuffle + per-row
// addrow accumulation with a partial Fisher-Yates on row indices gathered in
// one pass. This pins the properties a representative holdout must have, each
// chosen so a plausible sabotage of split.cpp breaks a specific assertion
// (standing rule 2):
//   - apportionment/stratification -> the class-count assertions fail if the
//     draw is not stratified (e.g. selecting from a merged pool) or the
//     round-half-up apportionment is dropped;
//   - partition/no-leakage -> fails on any off-by-one that drops or repeats a
//     row between test and train;
//   - reproducibility/divergence -> fails if the split stops riding the seeded
//     RNG stream.
// The extreme-imbalance case is a scale model of the SEER cohort (2.96%
// positives) that this splitter exists to handle.

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <set>

#include "split.h"
#include "utility.h"
#include "matrix.h"
#include "dataset.h"

using namespace std;

int failures = 0;

// Element-wise vector equality. (nvec's own operator== template in
// vector_ops.h -- pulled in transitively by matrix.h -- hardcodes a
// vector<double>::iterator and only compiles for T = double, so it cannot be
// used on these index vectors; std::equal sidesteps it.)
bool sameVec( const vector< unsigned >& a, const vector< unsigned >& b )
{
	return a.size() == b.size() && equal( a.begin(), a.end(), b.begin() );
}

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

// Recompute the apportionment independently of the code under test.
unsigned expectN0Test( unsigned n, unsigned n0, unsigned nTest )
{
	return ( unsigned ) ( ( double ) nTest * ( ( double ) n0 / ( double ) n )
		+ 0.5 );
}

// Count label-value occurrences among a set of row indices.
unsigned countLabel( const vector< unsigned >& idx, const vector< unsigned >& label,
	unsigned value )
{
	unsigned c = 0;
	for ( unsigned i = 0; i < idx.size(); i++ )
		if ( label[ idx[ i ] ] == value ) c++;
	return c;
}

// Assert the invariants that must hold for ANY stratified holdout of `label`.
void checkHoldout( const vector< unsigned >& label, unsigned nTest,
	const string& tag )
{
	unsigned n = ( unsigned ) label.size();
	unsigned n0 = 0;
	for ( unsigned i = 0; i < n; i++ ) if ( label[ i ] == 0 ) n0++;

	nsplit::Holdout h = nsplit::stratifiedHoldout( label, nTest );

	// Sizes
	expect( h.test.size() == nTest, tag + ": test set has exactly nTest rows" );
	expect( h.train.size() == n - nTest, tag + ": train set has the rest" );
	expect( h.n0 == n0 && h.n1 == n - n0, tag + ": class totals are correct" );

	// Apportionment by prevalence (round-half-up on the 0 class)
	unsigned e0 = expectN0Test( n, n0, nTest );
	expect( h.n0Test == e0 && h.n1Test == nTest - e0,
		tag + ": test slots apportioned by base rate" );

	// Stratification: the test set actually holds those class counts
	expect( countLabel( h.test, label, 0 ) == h.n0Test,
		tag + ": test 0-count matches the apportioned n0Test" );
	expect( countLabel( h.test, label, 1 ) == h.n1Test,
		tag + ": test 1-count matches the apportioned n1Test" );

	// Valid partition: every row appears exactly once across test and train
	set< unsigned > seen;
	bool dup = false;
	for ( unsigned i = 0; i < h.test.size(); i++ )
		if ( !seen.insert( h.test[ i ] ).second ) dup = true;
	for ( unsigned i = 0; i < h.train.size(); i++ )
		if ( !seen.insert( h.train[ i ] ).second ) dup = true;
	expect( !dup && seen.size() == n,
		tag + ": test and train partition every row with no leakage" );
}

int main()
{
	// Balanced, easy case
	util::set_seed( 42 );
	{
		vector< unsigned > label( 200 );
		for ( unsigned i = 0; i < 200; i++ ) label[ i ] = i % 2; // 100/100
		checkHoldout( label, 50, "balanced" );
	}

	// Extreme imbalance -- a scale model of SEER (2.96% positive)
	util::set_seed( 7 );
	{
		unsigned n = 10000, nPos = 296;
		vector< unsigned > label( n, 0 );
		for ( unsigned i = 0; i < nPos; i++ ) label[ i * ( n / nPos ) ] = 1;
		// Recount actual positives placed (integer stride may drop the last few)
		unsigned n1 = 0; for ( unsigned i = 0; i < n; i++ ) n1 += label[ i ];
		(void) n1;
		checkHoldout( label, 2500, "rare-event" );
	}

	// Edge cases: no test set, and all-test
	util::set_seed( 1 );
	{
		vector< unsigned > label( 60 );
		for ( unsigned i = 0; i < 60; i++ ) label[ i ] = ( i < 20 ) ? 1 : 0;
		checkHoldout( label, 0, "empty-test" );
		checkHoldout( label, 60, "all-test" );
		checkHoldout( label, 15, "quarter" );
	}

	// Reproducibility: the same seed reproduces the split exactly
	vector< unsigned > label( 500 );
	for ( unsigned i = 0; i < 500; i++ ) label[ i ] = ( i % 7 == 0 ) ? 1 : 0;

	util::set_seed( 123 );
	nsplit::Holdout a = nsplit::stratifiedHoldout( label, 125 );
	util::set_seed( 123 );
	nsplit::Holdout b = nsplit::stratifiedHoldout( label, 125 );
	expect( sameVec( a.test, b.test ) && sameVec( a.train, b.train ),
		"same seed reproduces the identical split" );

	// A different seed moves the membership (sizes stay fixed)
	util::set_seed( 124 );
	nsplit::Holdout c = nsplit::stratifiedHoldout( label, 125 );
	expect( c.test.size() == a.test.size() && !sameVec( c.test, a.test ),
		"a different seed produces a different draw" );

	// ---- DataSet::randomize integration ------------------------------------
	// The index splitter above is exercised in isolation; these cases verify
	// the wiring the unit tests cannot see: that the correct Raw rows are
	// gathered into each set, that BOTH sets are normalized using the TRAINING
	// set's limits, that the loaded-state flags are right, and that the two
	// reachable boundaries (all-test, zero-test) behave.

	// Build a 100-row, 2-input, 1-output raw set. The 4 positives (class 1)
	// carry the global extremes of input column 0 (+/-1000); the 96 negatives
	// sit in the inner range [0, 95]. With nTest = 10 the base-rate
	// apportionment sends round(10 * 96/100) = 10 zeros and 0 ones to test, so
	// EVERY positive -- hence both col-0 extremes -- stays in training. The
	// training column-0 range therefore strictly contains the test range, and
	// if (and only if) the test set is normalized with the training limits its
	// col-0 values land strictly inside the [inLower, inUpper] band. If the
	// test set were self-normalized, its min/max would hit the band exactly.
	Matrix< double > raw( 100, 3 );
	for ( unsigned r = 0; r < 100; r++ )
	{
		raw( r, 1 ) = 0.0; // second input constant -- irrelevant here
		if ( r < 4 ) // the four positives carry the extremes
		{
			double ext[ 4 ] = { -1000.0, 1000.0, -999.0, 999.0 };
			raw( r, 0 ) = ext[ r ];
			raw( r, 2 ) = 1.0; // outcome
		}
		else // negatives sit strictly inside
		{
			raw( r, 0 ) = ( double ) ( r - 4 ); // 0 .. 95
			raw( r, 2 ) = 0.0; // outcome
		}
	}

	DataSet ds;
	ds.setInput( 2 );
	ds.setOutput( 1 );
	ds.setDiscrete( true );
	ds.setHistory( false ); // do not write neuron.log from a test
	ds.setRawMatrix( raw );

	util::set_seed( 99 );
	bool ok = ds.randomize( ( unsigned ) 10 );
	expect( ok, "integration: randomize(10) succeeds" );
	expect( ds.getNumTrain() == 90 && ds.getNumTest() == 10,
		"integration: 90 train / 10 test rows" );
	expect( ds.trainLoaded() && ds.testLoaded(),
		"integration: both loaded-state flags set" );

	// Stratified gather: the outcome column (index 2) rides along, and discrete
	// outputs are NOT normalized, so it is still 0/1. n1Test == 0 here.
	Matrix< double >& tr = ds.getTrainMatrix();
	Matrix< double >& te = ds.getTestMatrix();
	unsigned trPos = 0, tePos = 0;
	for ( unsigned r = 0; r < tr.rows(); r++ ) if ( tr( r, 2 ) == 1.0 ) trPos++;
	for ( unsigned r = 0; r < te.rows(); r++ ) if ( te( r, 2 ) == 1.0 ) tePos++;
	expect( trPos == 4 && tePos == 0,
		"integration: all 4 positives gathered into train, none into test" );

	// Train column 0 spans the band exactly (it holds the global extremes)
	double trMin = tr( 0, 0 ), trMax = tr( 0, 0 );
	for ( unsigned r = 1; r < tr.rows(); r++ )
	{
		if ( tr( r, 0 ) < trMin ) trMin = tr( r, 0 );
		if ( tr( r, 0 ) > trMax ) trMax = tr( r, 0 );
	}
	expect( trMin > ds.getInLower() - 1e-9 && trMin < ds.getInLower() + 1e-9 &&
		trMax > ds.getInUpper() - 1e-9 && trMax < ds.getInUpper() + 1e-9,
		"integration: training column normalized to exactly [inLower, inUpper]" );

	// Test column 0 lands STRICTLY inside the band -- proof it was normalized
	// with the training limits, not its own (self-normalization would touch the
	// endpoints). The inner negatives map to roughly [0.0, 0.086].
	double teMin = te( 0, 0 ), teMax = te( 0, 0 );
	for ( unsigned r = 1; r < te.rows(); r++ )
	{
		if ( te( r, 0 ) < teMin ) teMin = te( r, 0 );
		if ( te( r, 0 ) > teMax ) teMax = te( r, 0 );
	}
	expect( teMin > ds.getInLower() + 0.1 && teMax < ds.getInUpper() - 0.1,
		"integration: test set normalized with TRAINING limits (strictly inside band)" );

	// Boundary: an all-test split must be refused (it would empty the training
	// set and make minimax dereference an empty column).
	DataSet ds2;
	ds2.setInput( 2 ); ds2.setOutput( 1 ); ds2.setDiscrete( true );
	ds2.setHistory( false );
	ds2.setRawMatrix( raw );
	util::set_seed( 5 );
	ostringstream sink;
	util::set_screen( sink ); // swallow the refusal message
	bool allTest = ds2.randomize( ( unsigned ) 100 ); // nTest == rows
	util::set_screen( cout );
	expect( !allTest && !ds2.trainLoaded(),
		"integration: an all-test split is refused, training set not built" );

	// Boundary: a zero-test split is a legitimate all-training request -- it
	// must succeed, leave no test set, and report no nan/inf frequency.
	DataSet ds3;
	ds3.setInput( 2 ); ds3.setOutput( 1 ); ds3.setDiscrete( true );
	ds3.setHistory( false );
	ds3.setRawMatrix( raw );
	util::set_seed( 5 );
	ostringstream report;
	util::set_screen( report );
	bool zeroTest = ds3.randomize( ( unsigned ) 0 );
	util::set_screen( cout );
	string rpt = report.str();
	expect( zeroTest && ds3.getNumTrain() == 100 && !ds3.testLoaded(),
		"integration: a zero-test split loads all rows as training, no test set" );
	expect( rpt.find( "nan" ) == string::npos && rpt.find( "inf" ) == string::npos,
		"integration: the empty-test report has no nan/inf frequency" );

	if ( failures == 0 )
	{
		cout << "check_split: stratified holdout is exact, balanced, and reproducible"
			<< endl;
		return 0;
	}
	cerr << "check_split: FAILED (" << failures << ")" << endl;
	return 1;
}
