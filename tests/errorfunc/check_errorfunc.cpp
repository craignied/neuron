// check_errorfunc.cpp : the output error function, single and multiple output.
//
// errorFunction is evaluated once per exemplar per iteration, on every training
// path in the engine, so it is both correctness-critical and the hottest small
// object in the code. This pins what it computes, so that the signature and
// allocation work around it can be shown to change nothing a caller can see.
//
// WHAT IS ASSERTED
//   1. Single-output LMS and cross-entropy against hand-computed values.
//   2. The MULTI-output constructor agrees with the single-output one summed
//      over components -- the two must be the same function, and nothing
//      previously checked that they were.
//   3. The cross-entropy boundary approximations (a fitted probability of
//      exactly 0 or 1, where log would be undefined) take the documented
//      linear-in-x form, and set boundsErr().
//   4. Values survive copying.
//
// The multi-output LMS branch is where the allocation work lands: it built a
// whole temporary vector ( y - o ) to feed dotprod. That becomes an
// allocation-free vector_ops primitive, sumSquaredDifference, so the caller
// still reads as the equation
//
//     E = 0.5 * || y - o ||^2
//
// while the elementwise loop lives once, in the numerical layer.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "vector_ops.h"
#include "function_defs.h"

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

static void expectNear( double got, double want, const string& what )
{
	bool ok = ( want == 0 ) ? ( fabs( got ) < 1e-12 )
		: ( fabs( got - want ) / fabs( want ) < 1e-12 );
	if ( !ok )
		cout << "         got " << setprecision( 17 ) << got
			<< ", expected " << want << endl;
	expect( ok, what );
}

// The sigmoid's argument for a given output, so the boundary cases can be built
static double logit( double p ) { return log( p / ( 1 - p ) ); }

static void test_single_output()
{
	cout << "-- single output --" << endl;

	// LMS: E = 0.5 ( y - o )^2
	{
		errorFunction E( 1.0, 0.25, logit( 0.25 ), false );
		expectNear( E.value(), 0.5 * 0.75 * 0.75, "LMS error" );
		expect( !E.boundsErr(), "LMS sets no bounds error" );
	}

	// Cross-entropy: E = -y ln o - ( 1 - y ) ln ( 1 - o )
	{
		errorFunction E( 1.0, 0.25, logit( 0.25 ), true );
		expectNear( E.value(), -log( 0.25 ), "X-entropy error, y = 1" );
	}
	{
		errorFunction E( 0.0, 0.25, logit( 0.25 ), true );
		expectNear( E.value(), -log( 0.75 ), "X-entropy error, y = 0" );
	}

	// The boundary approximations, where log would be undefined
	{
		errorFunction E( 0.0, 1.0, 4.0, true ); // o == 1 -> E = x ( 1 - y )
		expectNear( E.value(), 4.0, "X-entropy boundary at o = 1" );
		expect( E.boundsErr(), "and it reports a bounds error" );
	}
	{
		errorFunction E( 1.0, 0.0, -4.0, true ); // o == 0 -> E = -x y
		expectNear( E.value(), 4.0, "X-entropy boundary at o = 0" );
		expect( E.boundsErr(), "and it reports a bounds error" );
	}
}

// The multi-output form must be the single-output form, summed. Nothing
// asserted this before, and the two constructors are separate code.
static void test_multi_matches_single()
{
	cout << "-- multiple outputs agree with the single-output form --" << endl;

	vector< double > y, o, x;
	y.push_back( 1.0 ); o.push_back( 0.25 );
	y.push_back( 0.0 ); o.push_back( 0.60 );
	y.push_back( 1.0 ); o.push_back( 0.90 );
	y.push_back( 0.0 ); o.push_back( 0.05 );
	for ( unsigned i = 0; i < o.size(); i++ ) x.push_back( logit( o[ i ] ) );

	for ( int xe = 0; xe <= 1; xe++ )
	{
		double summed = 0;
		for ( unsigned i = 0; i < y.size(); i++ )
		{
			errorFunction one( y[ i ], o[ i ], x[ i ], xe != 0 );
			summed += one.value();
		}

		errorFunction many( y, o, x, xe != 0 );
		expectNear( many.value(), summed,
			xe ? "multi-output X-entropy == sum of singles"
			   : "multi-output LMS == sum of singles" );
	}

	// And the LMS value against the closed form, independent of the engine
	{
		double ss = 0;
		for ( unsigned i = 0; i < y.size(); i++ )
			ss += ( y[ i ] - o[ i ] ) * ( y[ i ] - o[ i ] );
		errorFunction many( y, o, x, false );
		expectNear( many.value(), 0.5 * ss, "multi-output LMS is 0.5 ||y - o||^2" );
	}
}

// The primitive the LMS branch now uses, checked on its own
static void test_sum_squared_difference()
{
	cout << "-- sumSquaredDifference --" << endl;

	vector< double > a, b;
	a.push_back( 1.0 ); b.push_back( 0.25 );
	a.push_back( -2.0 ); b.push_back( 0.5 );
	a.push_back( 3.5 ); b.push_back( 3.5 );

	double want = 0.75 * 0.75 + 2.5 * 2.5 + 0.0;
	expectNear( sumSquaredDifference( a, b ), want, "sum of squared differences" );

	vector< double > same( a );
	expectNear( sumSquaredDifference( a, same ), 0.0, "identical vectors give zero" );

	vector< double > empty;
	expectNear( sumSquaredDifference( empty, empty ), 0.0, "empty vectors give zero" );
}

static void test_copy_preserves_value()
{
	cout << "-- copying --" << endl;

	errorFunction E( 1.0, 0.25, logit( 0.25 ), true );
	errorFunction copied( E );
	expectNear( copied.value(), E.value(), "a copy carries the value" );
	expect( copied.boundsErr() == E.boundsErr(), "a copy carries the bounds flag" );

	errorFunction assigned;
	assigned = E;
	expectNear( assigned.value(), E.value(), "assignment carries the value" );
}

int main()
{
	test_single_output();
	test_multi_matches_single();
	test_sum_squared_difference();
	test_copy_preserves_value();

	cout << endl << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
