/* Representative test-set splitting (ROADMAP 4) -- implementation.
   See split.h and ROADMAP 4 in CLAUDE.md. */

#include "split.h"
#include "utility.h"

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
