// check_matrix.cpp : the Matrix row-gather primitive (includerows).
//
// includerows was added 2026-07-16 so a bootstrap resample is one gather
// instead of an element-wise fill loop. Its contract differs from
// includecols on purpose: positions may REPEAT and arrive in ANY ORDER
// (drawing with replacement is the whole point), and a bad row index throws
// BoundsViolation unconditionally -- like operator(), not like an assert --
// because a bad gather in a release build would otherwise read out of
// bounds. Also pins the includecols output dimension, whose assert carried
// excludecols' arithmetic (dormant -- no callers) until the same day.

#include <iostream>
#include <string>
#include <vector>

#include "matrix.h"

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


// Helpers for the assignment tests
static Matrix< double > filledMatrix( unsigned r, unsigned c, double base )
{
	Matrix< double > m( r, c );
	for ( unsigned i = 0; i < r; i++ )
		for ( unsigned j = 0; j < c; j++ )
			m( i, j ) = base + i * 100 + j;
	return m;
}

static bool matricesEqual( Matrix< double >& a, Matrix< double >& b )
{
	if ( a.rows() != b.rows() || a.cols() != b.cols() ) return false;
	for ( unsigned i = 0; i < a.rows(); i++ )
		for ( unsigned j = 0; j < a.cols(); j++ )
			if ( a( i, j ) != b( i, j ) ) return false;
	return true;
}

// --- operator= : resizing, self-assignment, and vector<Matrix> --------------
//
// BackProp's automatic-step-size buffer hand-built its weight buffer: forty
// lines branching on bias, resizing every layer with the boundary cases spelled
// out twice, then copying element by element -- to achieve what
// Matrix::operator= already does, because it resizes when the dimensions
// differ. Before relying on that, assert it: both directions of resize,
// self-assignment, value preservation, and the vector<Matrix> case the buffer
// actually is.
static void test_assignment()
{
	Matrix< double > s = filledMatrix( 3, 4, 1 ), before = s;
	s = s;
	expect( matricesEqual( s, before ), "self-assignment leaves the matrix intact" );

	Matrix< double > small = filledMatrix( 2, 2, 10 ), big = filledMatrix( 5, 7, 20 );
	small = big;
	expect( small.rows() == 5 && small.cols() == 7, "assigning a LARGER matrix resizes" );
	expect( matricesEqual( small, big ), "and copies every value" );

	Matrix< double > big2 = filledMatrix( 6, 6, 30 ), tiny = filledMatrix( 1, 2, 40 );
	big2 = tiny;
	expect( big2.rows() == 1 && big2.cols() == 2, "assigning a SMALLER matrix resizes" );
	expect( matricesEqual( big2, tiny ), "and copies every value" );

	vector< Matrix< double > > src, dst;
	src.push_back( filledMatrix( 2, 3, 70 ) );
	src.push_back( filledMatrix( 4, 1, 80 ) );
	dst.push_back( filledMatrix( 9, 9, 90 ) ); // wrong shape AND wrong count
	dst = src;
	expect( dst.size() == 2, "vector<Matrix> assignment copies the element count" );
	expect( matricesEqual( dst[ 0 ], src[ 0 ] ) && matricesEqual( dst[ 1 ], src[ 1 ] ),
		"and each element resizes and copies" );

	vector< Matrix< double > > orig = src, buf = src;
	src[ 0 ]( 0, 0 ) = -1;
	src = buf;
	expect( matricesEqual( src[ 0 ], orig[ 0 ] ) && matricesEqual( src[ 1 ], orig[ 1 ] ),
		"buffer / restore round trip" );
}

int main()
{
	// A 4x2 Matrix whose values encode their own position: A(r,c) = 10r + c
	Matrix< double > A( 4, 2 );
	for ( unsigned r = 0; r < 4; r++ )
		for ( unsigned c = 0; c < 2; c++ )
			A( r, c ) = 10.0 * r + c;

	// Gather with repeats, out of order -- a resample with replacement
	vector< unsigned > pos = { 2, 0, 2, 3, 0 };
	Matrix< double > G = A.includerows( pos );

	expect( G.rows() == 5 && G.cols() == 2,
		"gather is sized one row per position" );

	bool inOrder = true;
	for ( unsigned i = 0; i < pos.size(); i++ )
		for ( unsigned c = 0; c < 2; c++ )
			if ( G( i, c ) != 10.0 * pos[ i ] + c )
				inOrder = false;
	expect( inOrder, "rows arrive in the requested order, repeats included" );

	// A bad row index must throw -- unconditionally, not via assert
	bool threw = false;
	vector< unsigned > bad = { 1, 4 }; // row 4 does not exist
	try { A.includerows( bad ); }
	catch ( Matrix< double >::BoundsViolation& ) { threw = true; }
	expect( threw, "an out-of-range row index throws BoundsViolation" );

	// includecols: output carries exactly the included columns (its assert
	// said ncols - pos.size() until 2026-07-16; no caller ever fired it)
	vector< unsigned > cols = { 1 };
	Matrix< double > C = A.includecols( cols );
	expect( C.rows() == 4 && C.cols() == 1 && C( 2, 0 ) == 21.0,
		"includecols output is one column per included position" );

	test_assignment();

	if ( failures == 0 )
	{
		cout << "check_matrix: row gather, column include and assignment behave as documented" << endl;
		return 0;
	}
	cerr << "check_matrix: FAILED" << endl;
	return 1;
}
