/* Vector operations:
Extends C++ vector class for numerical vector-based calculations
for neUROn2++ */

#ifndef VECTOROPS_H
#define VECTOROPS_H

#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <exception>
#include <cassert>

#include "utility.h"

using namespace std;

// For non-template methods in vector.cpp
namespace nvec {
	// To make a vector of random doubles
	vector< double >& random( vector< double >&, const double );

	// (random_positions -- a rejection-sampling permutation shuffle, O(n^2) --
	//    was retired 2026-07-22 when the stratified splitter moved to a partial
	//    Fisher-Yates on row indices; see src/split.cpp and ROADMAP 4.)
}

// Overloaded unary mathematical operators...

// Overloaded +=
template< class T >
vector< T >& operator+=( vector< T >& lhs, const vector< T >& rhs )
{
	// Catch rhs vector smaller than lhs vector
	assert ( lhs.size() <= rhs.size() );
	
	// Uses algorithm::transform and functional::plus
    transform( lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(),
		plus< T >() );

    return lhs; // enables use in vector formulae
}

// Overloaded -=
template< class T >
vector< T >& operator-=( vector< T >& lhs, const vector< T >& rhs )
{
	// Catch rhs vector smaller than lhs vector
	assert ( lhs.size() <= rhs.size() );
	
	// Uses algorithm::transform and functional::minus
    transform( lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(),
		minus< T >() );

    return lhs; // enables use in vector formulae
}

// Overloaded *=
template< class T >
vector< T >& operator*=( vector< T >& lhs, const vector< T >& rhs )
{
	// Catch rhs vector smaller than lhs vector
	assert ( lhs.size() <= rhs.size() );
	
	// Uses algorithm::transform and functional::multiplies
    transform( lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(),
		multiplies< T >() );

    return lhs; // enables use in vector formulae
}

// Overloaded /=
template< class T >
vector< T >& operator/=( vector< T >& lhs, const vector< T >& rhs )
{
	// Catch rhs vector smaller than lhs vector
	assert ( lhs.size() <= rhs.size() );
	
	// Uses algorithm::transform and functional::divides
    transform( lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(),
		divides< T >() );

    return lhs; // enables use in vector formulae
}

// Overloaded binary mathematical operators, which call their unary
// counterparts.  NOTE: lhs is NOT (I repeat NOT) a reference, but
// a local copy.  For efficiency, unary operators are to be preferred
// in algorithms.

// Overloaded +
template< class T >
vector< T > operator+( vector< T > lhs, const vector< T >& rhs )
{
	// Note that lhs is not a reference, but a local copy
	return lhs += rhs; // enables use in vector formulae
}

// Overloaded -
template< class T >
vector< T > operator-( vector< T > lhs, const vector< T >& rhs )
{
	// Note that lhs is not a reference, but a local copy
	return lhs -= rhs; // enables use in vector formulae
}

// Overloaded *
template< class T >
vector< T > operator*( vector< T > lhs, const vector< T >& rhs )
{
	// Note that lhs is not a reference, but a local copy
	return lhs *= rhs; // enables use in vector formulae
}

// Overloaded /
template< class T >
vector< T > operator/( vector< T > lhs, const vector< T >& rhs )
{
	// Note that lhs is not a reference, but a local copy
	return lhs /= rhs; // enables use in vector formulae
}

// Overloaded unary mathematical operators with scalars...

// Overloaded +=
template< class T >
vector< T >& operator+=( vector< T >& lhs, const T rhs )
{
	// Let's do this through an iterator
	typename vector< T >::iterator p;
	
	for ( p = lhs.begin(); p != lhs.end(); p++ )
		*p += rhs;

    return lhs; // enables use in vector formulae
}

// Overloaded -=
template< class T >
vector< T >& operator-=( vector< T >& lhs, const T rhs )
{
	// Let's do this through an iterator
	typename vector< T >::iterator p;
	
	for ( p = lhs.begin(); p != lhs.end(); p++ )
		*p -= rhs;

    return lhs; // enables use in vector formulae
}

// Overloaded *=
template< class T >
vector< T >& operator*=( vector< T >& lhs, const T rhs )
{
	// Let's do this through an iterator
	typename vector< T >::iterator p;
	
	for ( p = lhs.begin(); p != lhs.end(); p++ )
		*p *= rhs;

    return lhs; // enables use in vector formulae
}

// Overloaded /=
template< class T >
vector< T >& operator/=( vector< T >& lhs, const T rhs )
{
	// Let's do this through an iterator
	typename vector< T >::iterator p;
	
	for ( p = lhs.begin(); p != lhs.end(); p++ )
		*p /= rhs;

    return lhs; // enables use in vector formulae
}

// Overloaded binary mathematical operators with scalars, which call
// their unary counterparts.  NOTE: as with the vector binary operators,
// lhs is NOT (I repeat NOT) a reference, but a local copy.  For
// efficiency, unary operators are to be preferred in algorithms.

// Overloaded +
template< class T >
vector< T > operator+( vector< T > vec, const T k )
{
	// Note that vec is not a reference, but a local copy
	return vec += k; // enables use in vector formulae
}

// Overloaded +
template< class T >
vector< T > operator+( const T k, vector< T > vec )
{
	// Note that vec is not a reference, but a local copy
	return vec += k; // enables use in vector formulae
}

// Overloaded -
template< class T >
vector< T > operator-( vector< T > vec, const T k )
{
	// Note that vec is not a reference, but a local copy
	return vec -= k; // enables use in vector formulae
}

// Overloaded *
template< class T >
vector< T > operator*( vector< T > vec, const T k )
{
	// Note that vec is not a reference, but a local copy
	return vec *= k; // enables use in vector formulae
}

// Overloaded *
template< class T >
vector< T > operator*( const T k, vector< T > vec )
{
	// Note that vec is not a reference, but a local copy
	return vec *= k; // enables use in vector formulae
}

// Overloaded /
template< class T >
vector< T > operator/( vector< T > vec, const T k )
{
	// Note that vec is not a reference, but a local copy
	return vec /= k; // enables use in vector formulae
}

// Overloaded << for vector output operations
template< class T >
ostream& operator << ( ostream& output, const vector< T >& rhs )
{
	// Let's do this through an iterator
	typename vector< T >::const_iterator p;
	
	for ( p = rhs.begin(); p != rhs.end(); p++ )
		output << *p << " ";
	
	return output;  // enables cout << A << B;
}

// Overloaded >> for vector input operations
template< class T >
istream& operator >> ( istream& input, vector< T >& rhs )
{
	// Let's do this through an iterator
	typename vector< T >::iterator p;
	
	for ( p = rhs.begin(); p != rhs.end(); p++ )
		input >> *p;
	
	return input;  // enables cin >> A >> B;
}

// Overloaded ==
template< class T >
bool operator == ( const vector< T >& lhs, const vector< T >& rhs )
{
	bool success = true; // the result

	// Catch different size vectors
	if ( lhs.size() != rhs.size() )
		success = false;

	else
	{
		// Iterate through the element type -- NOT a hardcoded double, which
		//    silently limited this comparison to vector< double > for years
		typename vector< T >::const_iterator p1, p2;

		// Compare the vectors element by element
		for ( p1 = lhs.begin(), p2 = rhs.begin(); p1 != lhs.end(); p1++, p2++ )
			if ( *p1 != *p2 )
			{
				success = false; // found an unequal pair
				break;
			}
	}

	return success;
}

// Overloaded !=
template< class T >
bool operator != ( const vector< T >& lhs, const vector< T >& rhs )
{
	// Use previously coded == method
	return ( !( lhs == rhs ) );
}

// Method for computing a dot product between 2 vectors, returns a scalar
// Example: x = dotprod( v1, v2 );
template< class T >
T dotprod( const vector< T >& v1, const vector< T >& v2 )
{
	// Catch different size vectors
	assert ( v1.size() == v2.size() );

	// Use numeric::inner_product
	// Note static_cast, as GCC with double bombs out thinking 0 is *only* int
	return inner_product( v1.begin(), v1.end(), v2.begin(),
		static_cast< T >( 0 ) );
}

// Method for passing a function to a vector, returning another passed vector
// Example: func( v_in, sigmoidal(), v_out );
template < class F, class T >
vector< T >& func( const vector< T >& vec_in, F fx, vector< T >& vec_out )
{
	// Catch different size vectors
	assert ( vec_in.size() == vec_out.size() );
	
	// Uses algorithm::transform
	transform( vec_in.begin(), vec_in.end(), vec_out.begin(), fx );

	return vec_out; // enables use in vector formulae
}

// Method for passing a function to a vector, returning another passed vector
//    but applying the function in the output vector only to elements specified
//    by the range 
// Example: func( v_in, sigmoidal(), v_out, a, b );
//    applies sigmoidal to v_in[ a ] through v_in[ b ] and places the result
//    in v_out[ a ] through v_out[ b ].  Other elements of v_out remain
//    unchanged.  v_out.size() must be > b.
template < class F, class T >
vector< T >& func( const vector< T >& vec_in, F fx, vector< T >& vec_out,
	unsigned a, unsigned b )
{
	// Catch different size vectors, and out of range parameters
	assert ( a <= b && b < vec_out.size() );
	
	// Uses algorithm::transform
	transform( vec_in.begin() + a, vec_in.begin() + b + 1, vec_out.begin() + a, fx );

	return vec_out; // enables use in vector formulae
}

// Method for passing a function to a vector, returns a new vector
// Example: v2 = func( v1, sigmoidal() );
template < class F, class T >
vector< T > func( const vector< T >& vec_in, F fx )
{
	// Construct a new vector of the same size
	vector< T > vec_out( vec_in.size() );
	
	// Use previously coded func
	func( vec_in, fx, vec_out );

	return vec_out; // enables use in vector formulae
}

// Method for passing a function to a vector's range, returns a new vector
// Example: v2 = func( v1, sigmoidal(), a, b );
//    applies sigmoidal to v1[ a ] through v1[ b ], returns a new vector
template < class F, class T >
vector< T > func( const vector< T >& vec_in, F fx, unsigned a, unsigned b )
{
	// Catch out of range parameters
	assert ( a <= b && b < vec_in.size() );

	// Construct a new vector of proper size
	vector< T > vec_out( b - a + 1 );

	// Uses algorithm::transform
	transform( vec_in.begin() + a, vec_in.begin() + b + 1, vec_out.begin(), fx );

	return vec_out; // enables use in vector formulae
}

// Method which "flattens" a container of vector of vector into a vector,
//    takes container as first argument, returns vector as second argument
template < class T >
vector< T >& flatten( const vector< vector< T > >& container, vector< T >& vec )
{
	vec.clear();
	
	// Let's do this through iterators
	typename vector< vector< T > >::const_iterator p_o;
	typename vector< T >::const_iterator p_i;
	
	for ( p_o = container.begin(); p_o != container.end(); p_o++ )
		for ( p_i = p_o->begin(); p_i != p_o->end(); p_i++ )
			vec.push_back( *p_i );

	return vec; // enables use in vector formulae
}

// Method which "flattens" a container of vector of vector into a vector,
//    takes container as argument, returns *new* vector
template < class T >
vector< T > flatten( const vector< vector< T > >& container )
{
	vector< T > vec; // construct new vector

	flatten( container, vec ); // use previously coded method

	return vec; // enables use in vector formulae
}

// Method which computes the sum of squares of a vector
// Example: s = squared( v );
template < class T >
T squared( const vector< T >& v )
{
	// Initialize the sum
	T sum = 0;

	// Let's do this through an iterator
	typename vector< T >::const_iterator p;

	// Iterate through the vector and calculate the sum of squares
	for ( p = v.begin(); p != v.end(); p++ )
		sum += ( *p * *p );

	return sum; // and return the sum of squares
}

// Method which computes the maximum absolute value of a vector
// Example: a = maxabs( v );
template < class T >
T maxabs( const vector< T >& v )
{
	T result = 0, // initialize the result
		absval; // the absolute value

	// Let's do this through an iterator
	typename vector< T >::const_iterator p;

	// Iterate through the vector and calculate the max abs value
	for ( p = v.begin(); p != v.end(); p++ )
	{
		absval = ( *p < 0 ? -*p : *p ); // the abs value
		if ( absval > result ) // the maximum
			result = absval;
	}

	return result; // and return the maximum absolute value;
}

// Method which computes the minimum absolute value of a vector
// Example: a = minabs( v );
template < class T >
T minabs( const vector< T >& v )
{
	// Let's do this through an iterator
	typename vector< T >::const_iterator p;

	// Initialize the result to the absolute first value
	T result = ( *v.begin() < 0 ? -*v.begin() : *v.begin() ),
		absval; // the absolute value	

	if ( v.size() > 1 ) // search if the vector is larger than 1 value
	{
		// Iterate through the vector and calculate the min abs value
		for ( p = v.begin() + 1; p != v.end(); p++ )
		{
			absval = ( *p < 0 ? -*p : *p ); // the abs value
			if ( absval < result ) // the minimum
				result = absval;
		}
	}

	return result; // and return the minimum absolute value;
}

// Partitions a vector into a vector of vector according to specified bin size
// Takes as arguments the incoming vector, the bin size, a flag indicating number
//    in the last bin:
//    false if the last bin contains <= bin size (last bin = remaining elements,
//    true if the last bin contains >= bin size (remaining elements are added
//    to the last bin,
// and as last argument the vector of vector to contain the bins
template < class T >
vector< vector< T > >& bin( const vector< T >& v_in, unsigned b, bool binFlag,
	vector< vector< T > >& v )
{
	assert( b > 0 ); // bin size must be > 0
	assert( v_in.size() > 0 ); // incoming vector must have elements

	typename vector< T >::const_iterator po;

	unsigned i, n = ( v_in.size() / b ); // number of bins

	v.resize( n ); // dimension outgoing vector of vector

	// If the incoming vector is < partition size, just append it
	if ( v_in.size() < b )
		v.push_back( v_in );

	else
	{
		// Iterate across incoming vector by bin size
		for ( po = v_in.begin(), i = 0; i < n; i++, po += b )
		{
			v[ i ].resize( b ); // dimension vector in vector of vector
			copy ( po, po + b, v[ i ].begin() ); // copy elements into it
		}

		unsigned remainder = v_in.size() % b; // calculate leftovers

		if ( remainder ) // there are leftovers
		{
			vector< T > holder( remainder ); // temp vector for leftovers
			copy ( po, v_in.end(), holder.begin() ); // copy leftovers to holder

			if ( !binFlag ) // last bin contains <= bin size
				v.push_back( holder ); // append holder to vector of vector
			else // append holder to last vector using the adaptor back_inserter
				copy( holder.begin(), holder.end(), back_inserter( v[ n - 1 ] ) );
		}
	}

	return v;
}

// Partitions a vector into a vector of vector according to specified bin size
// Takes as arguments the incoming vector, the bin size, a flag indicating number
//    in the last bin:
//    false if the last bin contains <= bin size (last bin = remaining elements,
//    true if the last bin contains >= bin size (remaining elements are added
//    to the last bin,
// and returns *new* vector of vector
template < class T >
vector< vector< T > > bin( const vector< T >& v_in, unsigned b, bool binFlag )
{
	vector< vector< T > > vv; // construct new vector of vector

	bin( v_in, b, binFlag, vv ); // use previously coded method

	return vv;
}

#endif
