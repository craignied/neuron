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

// Assert Hamilton apportionment + a leakage-free, stratum-honoring partition
// for the general splitter. expectCellTest, when non-empty, pins the exact
// per-stratum test counts (the apportionment); otherwise only the invariants
// are checked.
void checkStrata( const vector< unsigned >& stratum, unsigned nTest,
	const vector< unsigned >& expectCellTest, const string& tag )
{
	unsigned n = ( unsigned ) stratum.size();
	nsplit::StratHoldout h = nsplit::holdoutByStrata( stratum, nTest );

	// Apportionment sums to nTest and never exceeds a stratum's size
	unsigned sum = 0;
	bool overCap = false;
	for ( unsigned s = 0; s < h.cellTest.size(); s++ )
	{
		sum += h.cellTest[ s ];
		if ( h.cellTest[ s ] > h.cellTotal[ s ] ) overCap = true;
	}
	expect( sum == nTest && !overCap,
		tag + ": test slots sum to nTest, none over a stratum's size" );

	if ( !expectCellTest.empty() )
		expect( sameVec( h.cellTest, expectCellTest ),
			tag + ": largest-remainder apportionment is exact" );

	// Every test row belongs to the stratum its slot was counted against, and
	// the per-stratum test tally matches cellTest (stratification honored).
	vector< unsigned > tally( h.cellTest.size(), 0 );
	for ( unsigned i = 0; i < h.test.size(); i++ ) tally[ stratum[ h.test[ i ] ] ]++;
	expect( sameVec( tally, h.cellTest ),
		tag + ": each stratum contributes exactly its apportioned test rows" );

	// Valid partition: every row exactly once across test and train
	set< unsigned > seen;
	bool dup = false;
	for ( unsigned i = 0; i < h.test.size(); i++ )
		if ( !seen.insert( h.test[ i ] ).second ) dup = true;
	for ( unsigned i = 0; i < h.train.size(); i++ )
		if ( !seen.insert( h.train[ i ] ).second ) dup = true;
	expect( !dup && seen.size() == n,
		tag + ": strata partition every row with no leakage" );
}

// Assert the group-aware split's core guarantee -- no group straddles the two
// sets (zero leakage) -- plus a valid partition and approximate outcome balance.
void checkGroups( const vector< unsigned >& label, const vector< unsigned >& group,
	unsigned nTest, const string& tag )
{
	unsigned n = ( unsigned ) label.size();
	nsplit::GroupHoldout h = nsplit::groupHoldout( label, group, nTest );

	// ZERO LEAKAGE: a group id never appears on both sides.
	set< unsigned > inTest, inTrain;
	for ( unsigned i = 0; i < h.test.size(); i++ ) inTest.insert( group[ h.test[ i ] ] );
	for ( unsigned i = 0; i < h.train.size(); i++ ) inTrain.insert( group[ h.train[ i ] ] );
	bool straddle = false;
	for ( set< unsigned >::iterator it = inTest.begin(); it != inTest.end(); ++it )
		if ( inTrain.count( *it ) ) straddle = true;
	expect( !straddle, tag + ": no group straddles train and test (zero leakage)" );

	// Valid partition: every row exactly once.
	set< unsigned > seen;
	bool dup = false;
	for ( unsigned i = 0; i < h.test.size(); i++ )
		if ( !seen.insert( h.test[ i ] ).second ) dup = true;
	for ( unsigned i = 0; i < h.train.size(); i++ )
		if ( !seen.insert( h.train[ i ] ).second ) dup = true;
	expect( !dup && seen.size() == n,
		tag + ": groups partition every row with no leakage" );

	// Outcome balance approximately preserved (groups are indivisible, so the
	// tolerance is loose -- this only guards a gross imbalance).
	unsigned nPos = 0, tePos = 0;
	for ( unsigned r = 0; r < n; r++ ) if ( label[ r ] ) nPos++;
	for ( unsigned i = 0; i < h.test.size(); i++ ) if ( label[ h.test[ i ] ] ) tePos++;
	double popRate = ( double ) nPos / n;
	double teRate = h.test.empty() ? 0.0 : ( double ) tePos / h.test.size();
	expect( teRate > popRate - 0.1 && teRate < popRate + 0.1,
		tag + ": test outcome rate near the population rate" );
}

// Assert a stratified k-fold assignment: every row in exactly one fold, folds
// of near-equal size, and -- the point -- each fold's outcome count within 1 of
// the population's per-fold share (the stratification a random deal would miss).
void checkKFold( const vector< unsigned >& label, unsigned k, const string& tag )
{
	unsigned n = ( unsigned ) label.size();
	unsigned n1 = 0;
	for ( unsigned r = 0; r < n; r++ ) if ( label[ r ] ) n1++;

	vector< unsigned > fold = nsplit::kFold( label, k );

	vector< unsigned > size( k, 0 ), pos( k, 0 );
	bool inRange = true;
	for ( unsigned r = 0; r < n; r++ )
	{
		if ( fold[ r ] >= k ) inRange = false;
		else { size[ fold[ r ] ]++; if ( label[ r ] ) pos[ fold[ r ] ]++; }
	}
	expect( inRange, tag + ": every row lands in a valid fold" );

	// Fold sizes within 1 of n/k (a balanced partition)
	bool balanced = true;
	for ( unsigned f = 0; f < k; f++ )
		if ( size[ f ] + 1 < n / k || size[ f ] > n / k + 1 ) balanced = false;
	expect( balanced, tag + ": folds are near-equal in size" );

	// Stratification: each fold's positive count within 1 of n1/k
	bool strat = true;
	for ( unsigned f = 0; f < k; f++ )
	{
		unsigned lo = ( n1 / k > 0 ) ? n1 / k - 0 : 0; // base share
		long diff = ( long ) pos[ f ] - ( long ) ( n1 / k );
		if ( diff < -1 || diff > 1 ) strat = false;
		(void) lo;
	}
	expect( strat, tag + ": each fold's outcome count is within 1 of n1/k" );

	// Partition sums to n
	unsigned tot = 0;
	for ( unsigned f = 0; f < k; f++ ) tot += size[ f ];
	expect( tot == n, tag + ": the folds partition every row" );
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

	// ---- Phase 2: general multi-stratum holdout (Hamilton apportionment) ----
	util::set_seed( 3 );
	{
		// Three strata sized 50/30/20; nTest 10 divides evenly -> {5,3,2}
		vector< unsigned > st;
		for ( unsigned i = 0; i < 50; i++ ) st.push_back( 0 );
		for ( unsigned i = 0; i < 30; i++ ) st.push_back( 1 );
		for ( unsigned i = 0; i < 20; i++ ) st.push_back( 2 );
		checkStrata( st, 10, { 5, 3, 2 }, "hamilton-even" );
	}
	{
		// Sizes 33/33/34, nTest 10: quotas 3.3/3.3/3.4 -> floors 3/3/3, the one
		// leftover goes to the largest fraction (stratum 2) -> {3,3,4}
		vector< unsigned > st;
		for ( unsigned i = 0; i < 33; i++ ) st.push_back( 0 );
		for ( unsigned i = 0; i < 33; i++ ) st.push_back( 1 );
		for ( unsigned i = 0; i < 34; i++ ) st.push_back( 2 );
		checkStrata( st, 10, { 3, 3, 4 }, "hamilton-remainder" );
	}
	{
		// Two equal strata, nTest 5: quotas 2.5/2.5, the leftover breaks the
		// fractional tie to the LOWER stratum id -> {3,2}
		vector< unsigned > st;
		for ( unsigned i = 0; i < 50; i++ ) st.push_back( 0 );
		for ( unsigned i = 0; i < 50; i++ ) st.push_back( 1 );
		checkStrata( st, 5, { 3, 2 }, "hamilton-tie" );
	}

	// ---- Phase 2: DataSet stratification on a covariate ---------------------
	// A covariate (input col 0) is imbalanced with the outcome. Cell sizes are
	// chosen so a 20% split apportions to EXACTLY 8 test rows with col0 == 1 and
	// EXACTLY 4 positives -- outcome AND covariate balance held to the row.
	//   (out0,col0=0)=144  (out1,col0=0)=16  (out0,col0=1)=36  (out1,col0=1)=4
	{
		Matrix< double > sraw( 200, 3 );
		for ( unsigned r = 0; r < 200; r++ ) sraw( r, 1 ) = 0.0;
		for ( unsigned r = 0; r < 200; r++ )
		{
			bool col0one = ( r >= 160 );          // last 40 rows have col0 == 1
			bool pos = ( r >= 144 && r < 160 ) || ( r >= 196 ); // 16 + 4 positives
			sraw( r, 0 ) = col0one ? 1.0 : 0.0;
			sraw( r, 2 ) = pos ? 1.0 : 0.0;
		}

		DataSet sds;
		sds.setInput( 2 ); sds.setOutput( 1 ); sds.setDiscrete( true );
		sds.setHistory( false );
		sds.setRawMatrix( sraw );
		sds.setStrataColumns( vector< unsigned >{ 0 } ); // stratify on col 0 too

		util::set_seed( 11 );
		ostringstream diag;
		util::set_screen( diag );
		bool sok = sds.randomize( ( unsigned ) 40 );
		util::set_screen( cout );

		expect( sok && sds.getNumTest() == 40 && sds.getNumTrain() == 160,
			"strata: 40/160 split on outcome x covariate" );

		// Count col0 == 1 rows in the (normalized) test set: col0 is categorical
		// {0,1}, train spans both, so 1 maps to the upper band (> 0).
		Matrix< double >& ste = sds.getTestMatrix();
		unsigned col0one = 0, pos = 0;
		for ( unsigned r = 0; r < ste.rows(); r++ )
		{
			if ( ste( r, 0 ) > 0.0 ) col0one++;
			if ( ste( r, 2 ) == 1.0 ) pos++;
		}
		expect( col0one == 8,
			"strata: covariate balanced to the row (8 of 40 test rows have col0 == 1)" );
		expect( pos == 4,
			"strata: outcome balanced to the row (4 of 40 test rows are positive)" );

		// The diagnostic names what it stratified on and reports both sets
		string dg = diag.str();
		expect( dg.find( "Representativeness diagnostic" ) != string::npos &&
			dg.find( "Stratified on: outcome, input column 1" ) != string::npos &&
			dg.find( "Outcome-1 rate" ) != string::npos,
			"strata: the representativeness diagnostic is printed" );
	}

	// ---- Phase 3: group-aware (outcome-stratified group) holdout ------------
	util::set_seed( 21 );
	{
		// 20 groups of 5 rows; ~30% positives spread across groups. A 30% split
		// must keep every group intact and roughly preserve the outcome rate.
		vector< unsigned > lab( 100 ), grp( 100 );
		for ( unsigned r = 0; r < 100; r++ )
		{
			grp[ r ] = r / 5;                 // 20 groups of 5
			lab[ r ] = ( r % 10 < 3 ) ? 1 : 0; // 30 positives
		}
		checkGroups( lab, grp, 30, "groups" );
	}

	// Reproducibility of the group split under a fixed seed
	{
		vector< unsigned > lab( 120 ), grp( 120 );
		for ( unsigned r = 0; r < 120; r++ ) { grp[ r ] = r / 4; lab[ r ] = ( r % 5 == 0 ); }
		util::set_seed( 55 );
		nsplit::GroupHoldout g1 = nsplit::groupHoldout( lab, grp, 30 );
		util::set_seed( 55 );
		nsplit::GroupHoldout g2 = nsplit::groupHoldout( lab, grp, 30 );
		expect( sameVec( g1.test, g2.test ) && sameVec( g1.train, g2.train ),
			"groups: same seed reproduces the identical group split" );
	}

	// ---- Phase 3: DataSet group-aware split keeps clusters intact -----------
	// Input col 0 is a group key with 10 distinct values (20 rows each). After a
	// group-aware split no group value may appear in BOTH sets.
	{
		Matrix< double > graw( 200, 3 );
		for ( unsigned r = 0; r < 200; r++ )
		{
			graw( r, 0 ) = ( double ) ( r / 20 ); // 10 groups of 20
			graw( r, 1 ) = 0.0;
			graw( r, 2 ) = ( r % 4 == 0 ) ? 1.0 : 0.0; // 25% positives
		}

		DataSet gds;
		gds.setInput( 2 ); gds.setOutput( 1 ); gds.setDiscrete( true );
		gds.setHistory( false );
		gds.setRawMatrix( graw );
		gds.setGroupColumns( vector< unsigned >{ 0 } ); // group on col 0

		util::set_seed( 9 );
		ostringstream gdiag;
		util::set_screen( gdiag );
		bool gok = gds.randomize( ( unsigned ) 60 );
		util::set_screen( cout );

		expect( gok, "groups: DataSet group-aware split succeeds" );

		// col 0 is a group key; normalization is monotone, so its distinct
		// values are preserved. No group value may be in both sets.
		Matrix< double >& gtr = gds.getTrainMatrix();
		Matrix< double >& gte = gds.getTestMatrix();
		set< double > trVals, teVals;
		for ( unsigned r = 0; r < gtr.rows(); r++ ) trVals.insert( gtr( r, 0 ) );
		for ( unsigned r = 0; r < gte.rows(); r++ ) teVals.insert( gte( r, 0 ) );
		bool overlap = false;
		for ( set< double >::iterator it = teVals.begin(); it != teVals.end(); ++it )
			if ( trVals.count( *it ) ) overlap = true;
		expect( !overlap,
			"groups: no group value appears in both train and test (DataSet path)" );

		string gd = gdiag.str();
		expect( gd.find( "group-aware split" ) != string::npos &&
			gd.find( "leakage = 0 by construction" ) != string::npos,
			"groups: the group-aware diagnostic is printed" );
	}

	// ---- Phase 4: stratified k-fold assignment ------------------------------
	util::set_seed( 31 );
	{
		// 300 rows, 60 positives (20%), 5 folds -> each fold ~60 rows, 12 positives
		vector< unsigned > lab( 300 );
		for ( unsigned r = 0; r < 300; r++ ) lab[ r ] = ( r % 5 == 0 ) ? 1 : 0;
		checkKFold( lab, 5, "kfold-5" );
		checkKFold( lab, 10, "kfold-10" );
	}
	{
		// Rare-event scale model (SEER-like 3%): stratification must still hold
		vector< unsigned > lab( 1000, 0 );
		for ( unsigned i = 0; i < 30; i++ ) lab[ i * 33 ] = 1; // 30 positives
		checkKFold( lab, 5, "kfold-rare" );
	}

	// Reproducibility + divergence
	{
		vector< unsigned > lab( 200 );
		for ( unsigned r = 0; r < 200; r++ ) lab[ r ] = ( r % 4 == 0 );
		util::set_seed( 77 );
		vector< unsigned > f1 = nsplit::kFold( lab, 5 );
		util::set_seed( 77 );
		vector< unsigned > f2 = nsplit::kFold( lab, 5 );
		expect( sameVec( f1, f2 ), "kfold: same seed reproduces the fold assignment" );
		util::set_seed( 78 );
		vector< unsigned > f3 = nsplit::kFold( lab, 5 );
		expect( !sameVec( f1, f3 ), "kfold: a different seed reassigns the folds" );
	}

	if ( failures == 0 )
	{
		cout << "check_split: stratified holdout is exact, balanced, and reproducible"
			<< endl;
		return 0;
	}
	cerr << "check_split: FAILED (" << failures << ")" << endl;
	return 1;
}
