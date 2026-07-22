// vector_ops.cpp

// Non template methods go here

#include "stdafx.h" // For MSVC, must be first!

#include <algorithm>
#include <math.h>
#include <time.h>

#include "vector_ops.h"

// Fills vector with (double) random numbers between -n and +n
// Example: random( v, 0.5 );
vector< double >& nvec::random( vector< double >& v, const double n )
{
	// Seed the random number generator
	util::d_random();
	
	// Let's do this through an iterator
	vector< double >::iterator p;
	
	for ( p = v.begin(); p != v.end(); p++ )
		*p = util::d_random( n );
	
	return v; // enables use in vector formulae
}

// (nvec::random_positions was retired 2026-07-22 -- it drew a full permutation
//    by rejection sampling, rescanning from 0 on every collision, which is
//    O(n^2) and catastrophic on the 220k-row SEER negatives. The stratified
//    splitter now uses a partial Fisher-Yates on row indices; see src/split.cpp
//    and ROADMAP 4 in CLAUDE.md.)
