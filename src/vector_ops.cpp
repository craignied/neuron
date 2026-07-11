// vector_ops.cpp

// Non template methods go here

#include "stdafx.h" // For MSVC, must be first!

#include <algorithm> // GCC needs for srand() and rand()
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

// Takes a vector and fills it with unique random integers from 0 to
//    size of vector - 1
// Example: random_positions( v );
vector< unsigned >& nvec::random_positions( vector< unsigned >& v )
{
	// Seed the random number generator
	util::d_random();

	unsigned n = v.size(), // number of elements in vector
		random; // a random integer

	assert( n > 0 ); // vector must have at least 1 element

	v[ 0 ] = rand() % n; // make random 1st element

	// Loop through vector, referencing each element with index i
	for ( unsigned i = 1; i < n; i++ )
	{ 
		random = rand() % n; // make a new random number
		unsigned j = 0; // reset index j for elements prior to index i
		
		// Scan through the vector elements prior to index i to see if
		//    the number is unique, change it until it is 
		while ( j < i )
		{
			if ( v[ j ] == random ) // found previous instance of random number
			{
				random = rand() % n; // make a new random number
				j = 0; // and reset index j for elements prior to index i
			}
			else
				j++; // did not find match, so advance to next instance
		}
		
		v[ i ] = random; // no previous matches, so set element to new random number
	}

	return v; // enables use in vector formulae
}
