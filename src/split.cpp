/* Representative test-set splitting (ROADMAP 4) -- implementation.
   See split.h and ROADMAP 4 in CLAUDE.md. */

#include "split.h"
#include "utility.h"

#include <algorithm>
#include <cassert>

// Partial Fisher-Yates: move k randomly-chosen elements of idx to its front,
//    consuming k draws from the training RNG stream. idx is modified in place;
//    on return idx[ 0 .. k-1 ] are the selected rows (destined for the test
//    set) and idx[ k .. ] the remainder (the training set). O(k): element i is
//    swapped with a pick from [ i, m ), so every k-subset is equiprobable up to
//    the modulo bias of util::i_random ( rng() % n -- a fraction of a part in
//    2^32, negligible here and deliberately kept for cross-platform
//    determinism, as with util::d_random ). (The legacy random_positions drew a
//    full permutation by rejection sampling, rescanning from 0 on every
//    collision -- O(m^2), dominated by the tail; that is the routine this
//    retires.)
static void selectFront( vector< unsigned >& idx, unsigned k )
{
	unsigned m = ( unsigned ) idx.size();

	for ( unsigned i = 0; i < k; i++ ) // k <= m, so ( m - i ) >= 1 throughout
	{
		unsigned j = i + util::i_random( m - i ); // uniform in [ i, m )
		unsigned t = idx[ i ]; idx[ i ] = idx[ j ]; idx[ j ] = t; // swap
	}
}

nsplit::Holdout nsplit::stratifiedHoldout( const vector< unsigned >& label,
	unsigned nTest )
{
	unsigned n = ( unsigned ) label.size();

	assert( nTest <= n ); // DataSet::randomize bounds-checks before calling

	// Partition the row indices by outcome, preserving original order within
	//    each class -- the shuffle below supplies all the randomness.
	vector< unsigned > zeros, ones;
	zeros.reserve( n );
	ones.reserve( n );

	for ( unsigned r = 0; r < n; r++ )
	{
		assert( label[ r ] == 0 || label[ r ] == 1 ); // binary outcome only
		( label[ r ] == 0 ? zeros : ones ).push_back( r );
	}

	Holdout h;
	h.n0 = ( unsigned ) zeros.size();
	h.n1 = ( unsigned ) ones.size();

	// Apportion test slots by prevalence, round-half-up on the 0 class exactly
	//    as the legacy splitter did, so the test set carries the base rate.
	h.n0Test = ( unsigned ) ( ( double ) nTest
		* ( ( double ) h.n0 / ( double ) n ) + 0.5 );
	h.n1Test = nTest - h.n0Test;

	// Draw each class's test members to the front of its index vector.
	selectFront( zeros, h.n0Test );
	selectFront( ones, h.n1Test );

	// Assemble: test = selected 0s then selected 1s; train = the remainder, 0s
	//    then 1s. Order within a set is immaterial to the fit and the
	//    statistics; this mirrors the legacy assembly for readability.
	h.test.reserve( nTest );
	h.train.reserve( n - nTest );

	for ( unsigned i = 0; i < h.n0Test; i++ ) h.test.push_back( zeros[ i ] );
	for ( unsigned i = 0; i < h.n1Test; i++ ) h.test.push_back( ones[ i ] );
	for ( unsigned i = h.n0Test; i < h.n0; i++ ) h.train.push_back( zeros[ i ] );
	for ( unsigned i = h.n1Test; i < h.n1; i++ ) h.train.push_back( ones[ i ] );

	return h;
}

nsplit::StratHoldout nsplit::holdoutByStrata( const vector< unsigned >& stratum,
	unsigned nTest )
{
	unsigned n = ( unsigned ) stratum.size();

	assert( nTest <= n ); // caller bounds-checks first

	// Number of strata = one past the largest id.
	unsigned S = 0;
	for ( unsigned r = 0; r < n; r++ )
		if ( stratum[ r ] + 1 > S ) S = stratum[ r ] + 1;

	// Group row indices by stratum, preserving original order within each.
	vector< vector< unsigned > > cell( S );
	for ( unsigned r = 0; r < n; r++ )
		cell[ stratum[ r ] ].push_back( r );

	// Largest-remainder (Hamilton) apportionment of nTest across strata: the
	//    floor of each proportional quota, then the leftover slots handed to the
	//    strata with the largest fractional quotas (ties to the lower id, via a
	//    stable sort of the ascending id list).
	vector< unsigned > t( S );
	vector< double > frac( S );
	unsigned assigned = 0;

	for ( unsigned s = 0; s < S; s++ )
	{
		double q = ( double ) nTest * ( double ) cell[ s ].size() / ( double ) n;
		t[ s ] = ( unsigned ) q; // floor
		frac[ s ] = q - ( double ) t[ s ];
		assigned += t[ s ];
	}

	unsigned remainder = nTest - assigned; // in [ 0, S )

	vector< unsigned > order( S );
	for ( unsigned s = 0; s < S; s++ ) order[ s ] = s;
	stable_sort( order.begin(), order.end(),
		[ &frac ]( unsigned a, unsigned b ) { return frac[ a ] > frac[ b ]; } );

	for ( unsigned i = 0; i < remainder; i++ ) t[ order[ i ] ]++;

	// Draw each stratum's test rows to its front and assemble. (Clamp t to the
	//    stratum size for safety -- with nTest < n the Hamilton floor is always
	//    below the size, so this never binds on the reachable path.)
	StratHoldout h;
	h.cellTotal.resize( S );
	h.cellTest.resize( S );
	h.test.reserve( nTest );
	h.train.reserve( n - nTest );

	for ( unsigned s = 0; s < S; s++ )
	{
		if ( t[ s ] > cell[ s ].size() ) t[ s ] = ( unsigned ) cell[ s ].size();
		h.cellTotal[ s ] = ( unsigned ) cell[ s ].size();
		h.cellTest[ s ] = t[ s ];

		selectFront( cell[ s ], t[ s ] );

		for ( unsigned i = 0; i < t[ s ]; i++ ) h.test.push_back( cell[ s ][ i ] );
		for ( unsigned i = t[ s ]; i < cell[ s ].size(); i++ )
			h.train.push_back( cell[ s ][ i ] );
	}

	return h;
}

nsplit::GroupHoldout nsplit::groupHoldout( const vector< unsigned >& label,
	const vector< unsigned >& group, unsigned nTest )
{
	unsigned n = ( unsigned ) label.size();

	assert( nTest <= n && group.size() == n ); // caller bounds-checks first

	// Outcome-proportional test targets (round-half-up on the 0 class, as the
	//    other splitters do), so the group split preserves the base rate as far
	//    as indivisible groups allow.
	unsigned n0 = 0;
	for ( unsigned r = 0; r < n; r++ ) if ( label[ r ] == 0 ) n0++;
	unsigned n1 = n - n0;
	unsigned targetNeg = ( unsigned ) ( ( double ) nTest * ( double ) n0
		/ ( double ) n + 0.5 );
	unsigned targetPos = nTest - targetNeg;

	// Number of groups and their per-class counts.
	unsigned G = 0;
	for ( unsigned r = 0; r < n; r++ ) if ( group[ r ] + 1 > G ) G = group[ r ] + 1;

	vector< unsigned > gPos( G, 0 ), gNeg( G, 0 );
	for ( unsigned r = 0; r < n; r++ )
		( label[ r ] == 0 ? gNeg : gPos )[ group[ r ] ]++;

	// Visit the groups in a seeded-random order (a full Fisher-Yates of the id
	//    list) so the group partition is random yet reproducible.
	vector< unsigned > order( G );
	for ( unsigned g = 0; g < G; g++ ) order[ g ] = g;
	selectFront( order, G );

	// Greedy stratified-group assignment: take a whole group into test while
	//    that keeps both class counts at or below their targets, else train.
	vector< char > inTest( G, 0 );
	unsigned tp = 0, tn = 0;
	for ( unsigned i = 0; i < G; i++ )
	{
		unsigned g = order[ i ];
		if ( tp + gPos[ g ] <= targetPos && tn + gNeg[ g ] <= targetNeg )
		{
			inTest[ g ] = 1;
			tp += gPos[ g ];
			tn += gNeg[ g ];
		}
	}

	// Assemble, keeping every group's rows together on one side.
	GroupHoldout h;
	h.nGroups = G;
	h.groupsInTest = 0;
	for ( unsigned g = 0; g < G; g++ ) if ( inTest[ g ] ) h.groupsInTest++;

	for ( unsigned r = 0; r < n; r++ )
		( inTest[ group[ r ] ] ? h.test : h.train ).push_back( r );

	return h;
}
