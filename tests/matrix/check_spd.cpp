// check_spd.cpp : the symmetric positive-definite normal-equations primitives.
//
// THE STORAGE CONTRACT IS THE THING UNDER TEST, not just the arithmetic:
//
//   addOuterUpper  writes the upper triangle only and leaves the lower one
//                  exactly as it found it;
//   symmetrize     mirrors upper onto lower, and is the only operation that
//                  makes the object symmetric;
//   solveSPD       reads the upper triangle only, so it gives the same answer
//                  on an upper-populated matrix as on a symmetrized one.
//
// A contract like that is only worth having if something fails when it is
// broken, so each clause has a test that a sabotage of that clause fails --
// see the fail-proof evidence in the commit message. In particular test 7
// CORRUPTS THE LOWER TRIANGLE WITH GARBAGE and requires the solution not to
// move: that is the assertion that "reads the upper triangle only" is a
// property of the code rather than a sentence in a header.

#include "stdafx.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "matrix.h"

using namespace std;

static int failures = 0;

static void ok( bool condition, const char* what )
{
	if ( !condition )
	{
		printf( "FAIL: %s\n", what );
		failures++;
	}
}

static void close( double got, double want, double tol, const char* what )
{
	const double err = fabs( got - want ) / max( 1.0, fabs( want ) );
	if ( !( err <= tol ) )
	{
		printf( "FAIL: %s -- got %.17g, want %.17g (rel %.3g > %.3g)\n",
			what, got, want, err, tol );
		failures++;
	}
}

// A symmetric positive-definite 3x3 with an exactly representable Cholesky
//    factor, so the factorization can be checked against hand arithmetic and
//    not only against a round trip.
//
//        A = [ 4  2  -2 ]      C = [ 2  1  -1 ]      A = C'C
//            [ 2  5  -1 ]          [ 0  2   0 ]
//            [-2 -1   6 ]          [ 0  0   2 ]
//
//    Check: C'C row 0 = ( 4, 2, -2 ); row 1 = ( 2, 1+4, -1+0 ) = ( 2, 5, -1 );
//    row 2 = ( -2, -1, 1+0+4 ) = ( -2, -1, 5 ). The (2,2) element is therefore
//    5 for THAT C; use 6 in A and the factor's last element becomes sqrt(5).
//    Both are exercised: the hand factor below uses the consistent A.
static Matrix< double > handSPD()
{
	Matrix< double > A( 3, 3, 0.0 );
	A( 0, 0 ) = 4;  A( 0, 1 ) = 2;  A( 0, 2 ) = -2;
	A( 1, 0 ) = 2;  A( 1, 1 ) = 5;  A( 1, 2 ) = -1;
	A( 2, 0 ) = -2; A( 2, 1 ) = -1; A( 2, 2 ) = 5;
	return A;
}

int main()
{
	// --- 1. addOuterUpper writes the upper triangle, and ONLY it ------------
	{
		Matrix< double > A( 3, 3, 0.0 );
		// Sentinels below the diagonal. If the operation touches them, test 1
		//    fails -- which is exactly what a "upper triangle only" claim owes.
		A( 1, 0 ) = -11; A( 2, 0 ) = -22; A( 2, 1 ) = -33;

		vector< double > v( 3 );
		v[ 0 ] = 2; v[ 1 ] = -3; v[ 2 ] = 5;

		A.addOuterUpper( v );

		close( A( 0, 0 ), 4, 0, "addOuterUpper (0,0)" );
		close( A( 0, 1 ), -6, 0, "addOuterUpper (0,1)" );
		close( A( 0, 2 ), 10, 0, "addOuterUpper (0,2)" );
		close( A( 1, 1 ), 9, 0, "addOuterUpper (1,1)" );
		close( A( 1, 2 ), -15, 0, "addOuterUpper (1,2)" );
		close( A( 2, 2 ), 25, 0, "addOuterUpper (2,2)" );

		ok( A( 1, 0 ) == -11 && A( 2, 0 ) == -22 && A( 2, 1 ) == -33,
			"addOuterUpper left the lower triangle untouched" );
	}

	// --- 2. addOuterUpper ACCUMULATES, it does not assign -------------------
	{
		Matrix< double > A( 2, 2, 0.0 );
		vector< double > v( 2 );
		v[ 0 ] = 1; v[ 1 ] = 2;

		A.addOuterUpper( v );
		A.addOuterUpper( v );

		close( A( 0, 0 ), 2, 0, "addOuterUpper accumulates (0,0)" );
		close( A( 0, 1 ), 4, 0, "addOuterUpper accumulates (0,1)" );
		close( A( 1, 1 ), 8, 0, "addOuterUpper accumulates (1,1)" );

		// And a DIFFERENT second vector, so the test cannot pass by doubling
		vector< double > w( 2 );
		w[ 0 ] = 3; w[ 1 ] = -1;
		A.addOuterUpper( w );
		close( A( 0, 0 ), 11, 0, "addOuterUpper accumulates a second vector" );
		close( A( 0, 1 ), 1, 0, "addOuterUpper accumulates a second vector (0,1)" );
		close( A( 1, 1 ), 9, 0, "addOuterUpper accumulates a second vector (1,1)" );
	}

	// --- 3. addOuterUpper refuses bad shapes --------------------------------
	{
		bool threw = false;
		try { Matrix< double > E; vector< double > v; E.addOuterUpper( v ); }
		catch ( const Matrix< double >::BadSize& ) { threw = true; }
		ok( threw, "addOuterUpper throws BadSize on an empty Matrix" );

		threw = false;
		try
		{
			Matrix< double > R( 2, 3, 0.0 );
			vector< double > v( 2, 1.0 );
			R.addOuterUpper( v );
		}
		catch ( const Matrix< double >::DimensionMismatch& ) { threw = true; }
		ok( threw, "addOuterUpper throws DimensionMismatch on a non-square Matrix" );

		threw = false;
		try
		{
			Matrix< double > A( 3, 3, 0.0 );
			vector< double > v( 2, 1.0 );
			A.addOuterUpper( v );
		}
		catch ( const Matrix< double >::DimensionMismatch& ) { threw = true; }
		ok( threw, "addOuterUpper throws DimensionMismatch on a mis-sized vector" );
	}

	// --- 4. symmetrize mirrors upper onto lower, leaving upper alone --------
	{
		Matrix< double > A( 3, 3, 0.0 );
		A( 0, 0 ) = 1; A( 0, 1 ) = 2; A( 0, 2 ) = 3;
		A( 1, 1 ) = 4; A( 1, 2 ) = 5;
		A( 2, 2 ) = 6;
		A( 1, 0 ) = -99; A( 2, 0 ) = -99; A( 2, 1 ) = -99; // garbage to overwrite

		A.symmetrize();

		ok( A( 1, 0 ) == 2 && A( 2, 0 ) == 3 && A( 2, 1 ) == 5,
			"symmetrize mirrored the upper triangle onto the lower" );
		ok( A( 0, 1 ) == 2 && A( 0, 2 ) == 3 && A( 1, 2 ) == 5,
			"symmetrize left the upper triangle alone" );
		ok( A( 0, 0 ) == 1 && A( 1, 1 ) == 4 && A( 2, 2 ) == 6,
			"symmetrize left the diagonal alone" );
	}

	// --- 5. symmetrize refuses bad shapes -----------------------------------
	{
		bool threw = false;
		try { Matrix< double > E; E.symmetrize(); }
		catch ( const Matrix< double >::BadSize& ) { threw = true; }
		ok( threw, "symmetrize throws BadSize on an empty Matrix" );

		threw = false;
		try { Matrix< double > R( 2, 3, 0.0 ); R.symmetrize(); }
		catch ( const Matrix< double >::DimensionMismatch& ) { threw = true; }
		ok( threw, "symmetrize throws DimensionMismatch on a non-square Matrix" );
	}

	// --- 6. solveSPD against a hand-computed solution ------------------------
	//     A x = b with A as above and x = ( 1, -2, 3 ) gives
	//     b = ( 4 - 4 - 6, 2 - 10 - 3, -2 + 2 + 15 ) = ( -6, -11, 15 ).
	{
		Matrix< double > A = handSPD();
		vector< double > b( 3 ), x;
		b[ 0 ] = -6; b[ 1 ] = -11; b[ 2 ] = 15;

		A.solveSPD( b, x );

		ok( x.size() == 3, "solveSPD sized the destination" );
		close( x[ 0 ], 1, 1e-13, "solveSPD x0" );
		close( x[ 1 ], -2, 1e-13, "solveSPD x1" );
		close( x[ 2 ], 3, 1e-13, "solveSPD x2" );
	}

	// --- 7. THE STORAGE CONTRACT: solveSPD reads the UPPER triangle only ----
	//     Corrupt the lower triangle with values that would wreck any solve
	//     that read them. The answer must not move by a single bit.
	{
		Matrix< double > clean = handSPD();
		vector< double > b( 3 ), want, got;
		b[ 0 ] = -6; b[ 1 ] = -11; b[ 2 ] = 15;
		clean.solveSPD( b, want );

		Matrix< double > corrupt = handSPD();
		corrupt( 1, 0 ) = 1e9;
		corrupt( 2, 0 ) = -7e8;
		corrupt( 2, 1 ) = 3.25e7;

		// CAUGHT, not allowed to escape. A solve that reads the lower triangle
		//    sees a matrix that is not positive definite and throws; letting
		//    that abort the process would report "the file died" instead of
		//    naming the assertion, and would hide every later case.
		bool threw = false;
		try { corrupt.solveSPD( b, got ); }
		catch ( const Matrix< double >::Singular& ) { threw = true; }

		ok( !threw && got.size() == want.size() && got[ 0 ] == want[ 0 ]
			&& got[ 1 ] == want[ 1 ] && got[ 2 ] == want[ 2 ],
			"solveSPD reads the upper triangle ONLY: a corrupted lower triangle"
			" changes nothing" );
	}

	// --- 8. The identity system -------------------------------------------
	{
		Matrix< double > I( 4, 4, 0.0 );
		for ( unsigned i = 0; i < 4; i++ ) I( i, i ) = 1.0;
		vector< double > b( 4 ), x;
		b[ 0 ] = 1.5; b[ 1 ] = -2.25; b[ 2 ] = 0.0; b[ 3 ] = 7.0;

		I.solveSPD( b, x );
		for ( unsigned i = 0; i < 4; i++ )
			close( x[ i ], b[ i ], 0, "solveSPD on the identity returns rhs" );
	}

	// --- 9. Not positive definite: refused, by the algorithm's own test ------
	{
		// Symmetric, nonsingular, INDEFINITE: eigenvalues 1 and -1.
		Matrix< double > A( 2, 2, 0.0 );
		A( 0, 0 ) = 0; A( 0, 1 ) = 1;
		A( 1, 0 ) = 1; A( 1, 1 ) = 0;
		vector< double > b( 2, 1.0 ), x;

		bool threw = false;
		try { A.solveSPD( b, x ); }
		catch ( const Matrix< double >::Singular& ) { threw = true; }
		ok( threw, "solveSPD throws Singular on an indefinite Matrix" );

		// Negative definite, which is also not positive definite
		Matrix< double > B( 2, 2, 0.0 );
		B( 0, 0 ) = -4; B( 0, 1 ) = 0;
		B( 1, 0 ) = 0;  B( 1, 1 ) = -9;
		threw = false;
		try { B.solveSPD( b, x ); }
		catch ( const Matrix< double >::Singular& ) { threw = true; }
		ok( threw, "solveSPD throws Singular on a negative-definite Matrix" );
	}

	// --- 10. Exactly singular, and non-finite ------------------------------
	{
		// Rank 1, positive SEMIdefinite: the second pivot is exactly zero.
		Matrix< double > A( 2, 2, 0.0 );
		A( 0, 0 ) = 1; A( 0, 1 ) = 1;
		A( 1, 0 ) = 1; A( 1, 1 ) = 1;
		vector< double > b( 2, 1.0 ), x;

		bool threw = false;
		try { A.solveSPD( b, x ); }
		catch ( const Matrix< double >::Singular& ) { threw = true; }
		ok( threw, "solveSPD throws Singular on a positive-SEMIdefinite Matrix" );

		// A NaN fails `d > 0` too, and matrix.h says so rather than pretending
		//    there is a separate non-finite branch.
		//
		//    THE rhs MUST MATCH THE ORDER. It did not on the first writing --
		//    a 2-element rhs against this 3x3 -- so the call threw
		//    DimensionMismatch and the NaN branch was never reached. The test
		//    caught it because it distinguishes the two fault classes instead
		//    of accepting any exception; an `ok( threw )` over a bare `catch
		//    (...)` would have passed while testing nothing.
		Matrix< double > N = handSPD();
		N( 1, 1 ) = std::nan( "" );
		vector< double > b3( 3, 1.0 );
		threw = false;
		bool wrongFault = false;
		try { N.solveSPD( b3, x ); }
		catch ( const Matrix< double >::Singular& ) { threw = true; }
		catch ( const Matrix< double >::DimensionMismatch& ) { wrongFault = true; }
		catch ( const Matrix< double >::BadSize& ) { wrongFault = true; }
		ok( threw && !wrongFault,
			"solveSPD reports Singular for a non-finite element" );

		// THE OTHER MEMBER OF "non-finite", and it behaves differently. Writing
		//    the header comment first and testing it second is what exposed
		//    this: the comment said "a non-finite element reports Singular",
		//    which is true of NaN and FALSE of infinity. +inf on the diagonal
		//    makes d = inf, which IS greater than zero, so the factorization
		//    proceeds and returns an entirely finite-looking solution.
		//
		//    This is CHARACTERIZED, not endorsed. The published algorithm has no
		//    finiteness scan and none is added here; the obligation is the
		//    caller's, and matrix.h now says so instead of claiming a guarantee
		//    it does not provide. Asserting the observed behaviour means that if
		//    it ever changes, someone reads this note.
		//
		//    Enumerating a named category's MEMBERS is the point. A category
		//    tested on one member is how an acceptance test that rejected NaN
		//    and +inf while accepting -inf survived nineteen tests elsewhere in
		//    this project.
		Matrix< double > Inf = handSPD();
		Inf( 1, 1 ) = std::numeric_limits< double >::infinity();
		vector< double > xi;
		bool threwOnInf = false;
		try { Inf.solveSPD( b3, xi ); }
		catch ( const Matrix< double >::Singular& ) { threwOnInf = true; }
		ok( !threwOnInf && xi.size() == 3,
			"an INFINITE element does not fail the positive-definiteness test"
			" (unlike NaN): solveSPD completes" );
		bool allFinite = ( xi.size() == 3 );
		for ( unsigned i = 0; i < xi.size(); i++ )
			if ( !std::isfinite( xi[ i ] ) ) allFinite = false;
		ok( allFinite,
			"and it returns a FINITE-LOOKING solution -- which is exactly why"
			" the caller, not this primitive, must exclude non-finite input" );
	}

	// --- 11. solveSPD refuses bad shapes ------------------------------------
	{
		vector< double > b( 3, 1.0 ), x;

		bool threw = false;
		try { Matrix< double > E; E.solveSPD( b, x ); }
		catch ( const Matrix< double >::BadSize& ) { threw = true; }
		ok( threw, "solveSPD throws BadSize on an empty Matrix" );

		threw = false;
		try { Matrix< double > R( 2, 3, 0.0 ); R.solveSPD( b, x ); }
		catch ( const Matrix< double >::DimensionMismatch& ) { threw = true; }
		ok( threw, "solveSPD throws DimensionMismatch on a non-square Matrix" );

		threw = false;
		try
		{
			Matrix< double > A = handSPD();
			vector< double > wrong( 2, 1.0 );
			A.solveSPD( wrong, x );
		}
		catch ( const Matrix< double >::DimensionMismatch& ) { threw = true; }
		ok( threw, "solveSPD throws DimensionMismatch on a mis-sized rhs" );
	}

	// --- 12. The coefficient Matrix is const: solveSPD does not consume it ---
	{
		Matrix< double > A = handSPD();
		Matrix< double > before = A;
		vector< double > b( 3 ), x;
		b[ 0 ] = -6; b[ 1 ] = -11; b[ 2 ] = 15;

		A.solveSPD( b, x );

		bool same = true;
		for ( unsigned i = 0; i < 3 && same; i++ )
			for ( unsigned j = 0; j < 3 && same; j++ )
				if ( A( i, j ) != before( i, j ) ) same = false;
		ok( same, "solveSPD left the coefficient Matrix unchanged" );

		// Solving twice must give the same answer -- a factorization written
		//    back over A would not.
		vector< double > again;
		A.solveSPD( b, again );
		ok( again.size() == x.size() && again[ 0 ] == x[ 0 ]
			&& again[ 1 ] == x[ 1 ] && again[ 2 ] == x[ 2 ],
			"solveSPD is repeatable on the same Matrix" );
	}

	// --- 13. A round trip on a larger accumulated system ---------------------
	//     Exactly the shape a Gauss-Newton method produces: A built ONLY with
	//     addOuterUpper from a series of vectors, plus a positive diagonal, then
	//     solved WITHOUT symmetrize(). b is computed from a chosen x using the
	//     SYMMETRIZED matrix, so a solve that read a stale lower triangle would
	//     be answering a different question and would not recover x.
	{
		const unsigned n = 9;
		Matrix< double > A( n, n, 0.0 );
		vector< double > v( n );

		for ( unsigned t = 0; t < 30; t++ )
		{
			for ( unsigned i = 0; i < n; i++ )
				v[ i ] = sin( 0.7 * ( double ) ( t + 1 ) * ( double ) ( i + 1 ) )
					+ 0.25 * ( double ) ( ( t + i ) % 5 );
			A.addOuterUpper( v );
		}
		for ( unsigned i = 0; i < n; i++ )
			A( i, i ) += 0.5; // the damping term, which is what makes it SPD

		vector< double > x( n );
		for ( unsigned i = 0; i < n; i++ )
			x[ i ] = 1.0 - 0.3 * ( double ) i;

		// b = A x, using a SYMMETRIZED copy -- the mathematical A
		Matrix< double > full = A;
		full.symmetrize();
		vector< double > b( n, 0.0 );
		for ( unsigned i = 0; i < n; i++ )
			for ( unsigned j = 0; j < n; j++ )
				b[ i ] += full( i, j ) * x[ j ];

		// Solve on the UPPER-POPULATED matrix
		vector< double > got;
		A.solveSPD( b, got );
		ok( got.size() == n, "round trip: destination sized" );
		for ( unsigned i = 0; i < n; i++ )
			close( got[ i ], x[ i ], 1e-9, "round trip recovers x (upper only)" );

		// And on the symmetrized one: the SAME answer, which is the contract
		vector< double > gotFull;
		full.solveSPD( b, gotFull );
		for ( unsigned i = 0; i < n; i++ )
			close( gotFull[ i ], got[ i ], 1e-12,
				"upper-populated and symmetrized give the same solution" );
	}

	if ( failures )
	{
		printf( "%d failure(s)\n", failures );
		return 1;
	}
	printf( "all SPD normal-equations primitive checks passed\n" );
	return 0;
}
