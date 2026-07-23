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

	// NOTE: the Matrix ctors/resize now VALUE-INITIALIZE (zeros) rather than
	// leaving garbage -- an uninitialized cell read made OBD's architecture
	// selection nondeterministic across processes (2026-07-23; fixed in matrix.h).
	// There is deliberately no unit assertion for it here: proving "was garbage"
	// requires the allocator to hand back a poisoned freed block, which it does
	// not do portably (macOS serves these sizes from freshly zeroed OS pages), so
	// any such test passes vacuously. The fix is verified instead by cross-process
	// determinism (tests/crossval, run repeatedly) and byte-identical goldens.

	if ( failures == 0 )
	{
		cout << "check_matrix: row gather and column include behave as documented" << endl;
		return 0;
	}
	cerr << "check_matrix: FAILED" << endl;
	return 1;
}
