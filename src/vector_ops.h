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

#include "utility.h"

using namespace std;

// For non-template methods in vector.cpp
namespace nvec {
	// To make a vector of random doubles
	vector< double >& random( vector< double >&, const double );

	// (random_positions -- a rejection-sampling permutation shuffle, O(n^2) --
	//    was retired 2026-07-22 when the stratified splitter moved to a partial
	//    Fisher-Yates on row indices; see src/split.cpp and ROADMAP 4.)

	// --- The bounds contract of this layer (D5, 2026-08-01) ---------------
	//
	// These operations walk two containers in lockstep, or index into one by
	//    position. Violating that is not a data condition to be handled -- it
	//    is a programming error that reads or writes memory the operation does
	//    not own. They are therefore checked UNCONDITIONALLY.
	//
	// Not by assert. The engine ships with NDEBUG defined (CMakeLists.txt
	//    forces Release when no build type is given, and Release is
	//    `-O3 -DNDEBUG`), so every ctest target already runs with assertions
	//    compiled out: not one assertion in this file has ever executed in the
	//    gate chain. This is the same argument that makes
	//    Matrix< T >::operator() throw rather than assert (standing rule 4),
	//    now applied to the layer beside it.
	//
	// The motivating defect: SimpleProp passed hO (nHidden + 1 elements, the
	//    last one the bias slot) into h_err (nHidden) through the unranged
	//    func(), which writes vec_in.size() elements into vec_out -- one past
	//    the end of h_err in every shipped build. The fix was to make the
	//    caller state its domain, func( hO, d_sigmoidal(), h_err, 0,
	//    nHidden - 1 ); the contract has to keep forcing that.
	//
	// Declared here, in the vector layer. Reusing Matrix's BoundsViolation
	//    would make vector_ops depend UPWARD on a sibling (neither header
	//    includes the other; only matrix.cpp includes this one).
	//
	// Three types, flat, because they answer three different diagnostic
	//    questions. See refactor_audit.md section 11.

	// Parallel containers of incompatible length
	class SizeMismatch : public std::exception
	{
		public:
			const char * what() const noexcept override
				{ return "vector size mismatch"; }
	};

	// A position or range outside the container it indexes
	class RangeViolation : public std::exception
	{
		public:
			const char * what() const noexcept override
				{ return "vector range violation"; }
	};

	// An operation with no value on the empty set (minabs, maxabs). A sum has
	//    an identity and is NOT in this category: squared(), dotprod() and
	//    sumSquaredDifference() all return 0 over empty operands.
	class EmptyVector : public std::exception
	{
		public:
			const char * what() const noexcept override
				{ return "empty vector has no value for this operation"; }
	};
}

// Overloaded unary mathematical operators...
//
// THE PREFIX RULE. These take rhs.size() >= lhs.size(), NOT equality, and the
//    inequality is deliberate: the result has the LHS's size, and the surplus
//    tail of rhs is ignored. The bias-slot arithmetic of both feed-forward
//    networks depends on it -- SimpleProp multiplies h_err (nHidden) by oW
//    (nHidden + 1) and says so at simpleprop.cpp:584; BackProp says the same
//    of its hidden-error chain at backprop.cpp:476. A SHORTER rhs is the error,
//    because transform then reads lhs.size() elements out of it.

// Overloaded +=
template< class T >
vector< T >& operator+=( vector< T >& lhs, const vector< T >& rhs )
{
	// Catch rhs vector smaller than lhs vector (a longer one is the prefix
	//    rule above, and legal)
	if ( lhs.size() > rhs.size() )
		throw nvec::SizeMismatch();

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
	if ( lhs.size() > rhs.size() )
		throw nvec::SizeMismatch();

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
	if ( lhs.size() > rhs.size() )
		throw nvec::SizeMismatch();

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
	if ( lhs.size() > rhs.size() )
		throw nvec::SizeMismatch();

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
	// Catch different size vectors. EQUALITY, not the prefix rule the compound
	//    operators take: a dot product over a prefix is a different scalar, and
	//    returning it silently is exactly what a bounds check exists to stop.
	if ( v1.size() != v2.size() )
		throw nvec::SizeMismatch();

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
	// Catch different size vectors. EQUALITY, deliberately -- transform writes
	//    vec_in.size() elements INTO vec_out, so a longer input runs off the
	//    end of the destination. That was the defect: SimpleProp passed hO
	//    (nHidden + 1, last element the bias slot) into h_err (nHidden). It is
	//    not fixed by accepting a prefix here; a caller that wants part of a
	//    vector says which part, through the ranged overload below.
	if ( vec_in.size() != vec_out.size() )
		throw nvec::SizeMismatch();

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
//    unchanged.  BOTH vectors must be longer than b.
template < class F, class T >
vector< T >& func( const vector< T >& vec_in, F fx, vector< T >& vec_out,
	unsigned a, unsigned b )
{
	// Catch out of range parameters, on BOTH sides. The input bound was never
	//    checked at all -- the assert here tested b against vec_out only, so
	//    reading past a short vec_in was unguarded even in a debug build.
	//    a > b is equally fatal: it forms a reversed iterator range, which
	//    takes a SIGBUS rather than doing nothing.
	if ( a > b || b >= vec_in.size() || b >= vec_out.size() )
		throw nvec::RangeViolation();

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
	// Catch out of range parameters (the destination is allocated below, so
	//    only the input can be overrun here)
	if ( a > b || b >= vec_in.size() )
		throw nvec::RangeViolation();

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

// Sum of squared differences between two vectors: || a - b ||^2, computed
//    without materializing a - b.
// Example: s = sumSquaredDifference( y, o );
//
//    It exists so that a caller can write the least-squares error as the
//    equation it is,
//
//        E = 0.5 * sumSquaredDifference( y, o );
//
//    without the temporary vector that `squared( y - o )` allocates. That
//    matters because the caller is errorFunction, evaluated once per exemplar
//    per iteration on every training path in the engine (standing rule 7: the
//    elementwise loop belongs once, in the numerical layer, not open-coded at
//    the call site).
template < class T >
T sumSquaredDifference( const vector< T >& a, const vector< T >& b )
{
	// Catch different size vectors -- b is walked in lockstep with a, so a
	//    shorter b is read past its end. Equal EMPTY vectors are fine: a sum
	//    over an empty index set is 0, which is the identity, not a refusal.
	if ( a.size() != b.size() )
		throw nvec::SizeMismatch();

	T sum = 0;

	typename vector< T >::const_iterator pa, pb;
	for ( pa = a.begin(), pb = b.begin(); pa != a.end(); ++pa, ++pb )
	{
		T d = *pa - *pb;
		sum += d * d;
	}

	return sum;
}

// Method which computes the maximum absolute value of a vector
// Example: a = maxabs( v );
//
//    Refuses an empty vector. It used to return 0, which is memory-safe and a
//    LIE: the maximum of an empty set has no value, and 0 is the one value a
//    caller acts on. Network::getGradMax() hands this straight to the gradient
//    stopping rule, where "the largest gradient is 0" means CONVERGED -- so a
//    model with no parameters would certify itself as having converged. No
//    live caller passes an empty vector (df() is never 0, and conditionOf()
//    returns before this at dimension 0), so the refusal costs nothing
//    reachable and removes a false convergence signal from anything later.
template < class T >
T maxabs( const vector< T >& v )
{
	if ( v.empty() )
		throw nvec::EmptyVector();

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
	// Refuse an empty vector: the initialization below dereferences begin()
	//    unconditionally, which on an empty vector is an invalid read (a
	//    libc++ hardened build takes a SIGTRAP there).
	if ( v.empty() )
		throw nvec::EmptyVector();

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
	// A bin size of 0 is an extent the container cannot satisfy, and the
	//    division below would be an integer division by zero -- undefined, and
	//    a SIGFPE on both of our targets.
	if ( b == 0 )
		throw nvec::RangeViolation();

	// ( The assert that the incoming vector be non-empty went with it: it was
	//   stricter than the code needs. An empty input takes the short-input
	//   branch and yields one empty bin, which is harmless and stays legal. )

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
