// check_matrix_bounds.cpp : the bounds and dimension contract of the Matrix
// layer, characterized in the build we actually ship (D9).
//
// WHY THIS TARGET DEFINES NDEBUG ITSELF. `CMakeLists.txt` forces Release when
// no build type is given and Release is `-O3 -DNDEBUG`, so every assert in
// matrix.h and matrix.cpp is compiled out of the shipped binary and out of
// every ctest target. A test that passes in a checked build proves nothing
// about what we ship. This file refuses to compile without NDEBUG (below) and
// the target defines it, so the guarantee cannot silently become "we happened
// to build Debug".
//
// WHAT IS BEING CHARACTERIZED. Exactly two Matrix mechanisms enforce a bounds
// or dimension contract in Release -- operator() and includerows, both throwing
// BoundsViolation. Roughly forty-five other public entry points are assert-only,
// so an invalid argument is an out-of-bounds read or WRITE rather than an
// exception. Full inventory: refactor_audit.md section 12.2.
//
// THE TWO HALVES, WHICH MUST NOT BE CONFLATED (section 12.4 step 2):
//
//   Cases marked HOLDS   -- operator(), includerows, BadSize, Singular, and the
//                           deliberate prefix rule. These PASS TODAY and must
//                           keep passing. They are regression cover, never
//                           red-test evidence. If one of these ever goes red,
//                           this phase has broken something.
//   Cases marked ABSENT  -- the Class-A contracts. Each is watched FAILING
//                           against today's matrix.h before any policy exists.
//                           That is what distinguishes an absent contract from
//                           an untested one.
//
// ONE CASE PER PROCESS, AND A CONTROL INSIDE EACH (section 12.5). An absent
// contract here does not return quietly: several of these WRITE outside an
// allocation, so a case can corrupt memory or die. Run sequentially, an early
// crash would hide every case after it -- the vacuous-comparison hole in
// another costume. So:
//
//   * `check_matrix_bounds <n>` runs exactly one case; run_matrix_bounds_demo.sh
//     runs each in its own process and records the exit status, so a crash is a
//     RESULT FOR THAT CASE rather than the end of the run;
//   * every case performs a LEGAL operation of the same family and checks its
//     answer BEFORE making the invalid call. A case that dies before its own
//     control has proved nothing about the contract -- it has proved the
//     harness is broken, and that is a distinct exit status.
//
// Exit status: 0 held, 1 no exception (the Release gap), 3 the control failed,
// 4 threw the wrong type, and anything else is the process dying.

#ifndef NDEBUG
#error "check_matrix_bounds must be built with NDEBUG -- it characterizes the shipped build"
#endif

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "matrix.h"

using namespace std;

typedef Matrix< double > M;

enum Verdict { HELD = 0, NO_THROW = 1, CONTROL_FAILED = 3, WRONG_TYPE = 4 };

// The control runs first, in this process, and must return true. Only then is
//    the invalid call made.
template< class E, class Ctl, class Bad >
static int probe( Ctl control, Bad bad )
{
	try
	{
		if ( !control() )
			return CONTROL_FAILED;
	}
	catch ( ... ) { return CONTROL_FAILED; }

	try { bad(); }
	catch ( const E& ) { return HELD; }
	catch ( ... ) { return WRONG_TYPE; }

	return NO_THROW;
}

// A case with no invalid call: the legal path, asserted for its own sake.
template< class Ctl >
static int control_only( Ctl control )
{
	try { return control() ? HELD : CONTROL_FAILED; }
	catch ( ... ) { return CONTROL_FAILED; }
}

// 1 2 3 / 4 5 6 -- small enough to check by eye, rectangular so that a
//    row/column confusion cannot pass by symmetry.
static M mat23()
{
	M A( 2, 3 );
	A( 0, 0 ) = 1; A( 0, 1 ) = 2; A( 0, 2 ) = 3;
	A( 1, 0 ) = 4; A( 1, 1 ) = 5; A( 1, 2 ) = 6;
	return A;
}

static vector< double > seq( unsigned n, double base = 1 )
{
	vector< double > v;
	for ( unsigned i = 0; i < n; i++ )
		v.push_back( base + i );
	return v;
}

// --- HOLDS: the contracts that already exist and must not regress -----------

// 1. operator() with a row past the end.
static int c01()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); return A( 1, 2 ) == 6; },
		[] { M A = mat23(); double v = A( 2, 0 ); ( void ) v; } );
}

// 2. operator() with a column past the end.
static int c02()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); return A( 0, 0 ) == 1; },
		[] { M A = mat23(); double v = A( 0, 3 ); ( void ) v; } );
}

// 3. The const accessor carries the same contract.
static int c03()
{
	return probe< M::BoundsViolation >(
		[] { const M A = mat23(); return A( 1, 1 ) == 5; },
		[] { const M A = mat23(); double v = A( 9, 9 ); ( void ) v; } );
}

// 4. includerows: a gather position past the end. Checked unconditionally
//    since 2026-07-16, because a bad resample index would read out of bounds.
static int c04()
{
	return probe< M::BoundsViolation >(
		[] {
			M A = mat23();
			vector< unsigned > pos; pos.push_back( 1 ); pos.push_back( 1 );
			M G = A.includerows( pos );
			return G.rows() == 2 && G( 0, 0 ) == 4 && G( 1, 2 ) == 6;
		},
		[] {
			M A = mat23();
			vector< unsigned > pos; pos.push_back( 5 );
			M G = A.includerows( pos ); ( void ) G;
		} );
}

// 5. BadSize on a zero-row construction.
static int c05()
{
	return probe< M::BadSize >(
		[] { M A( 2, 3, 7.0 ); return A( 1, 2 ) == 7; },
		[] { M A( 0, 3, 7.0 ); ( void ) A; } );
}

// 6. Singular, from the inverse family. A numerical refusal, not a bounds one.
static int c06()
{
	return probe< M::Singular >(
		[] {
			M A( 2, 2 );
			A( 0, 0 ) = 2; A( 0, 1 ) = 0; A( 1, 0 ) = 0; A( 1, 1 ) = 4;
			M I = A.inverse();
			return I( 0, 0 ) == 0.5 && I( 1, 1 ) == 0.25;
		},
		[] {
			M A( 2, 2 ); // every row identical -> singular
			A( 0, 0 ) = 1; A( 0, 1 ) = 1; A( 1, 0 ) = 1; A( 1, 1 ) = 1;
			M I = A.inverse(); ( void ) I;
		} );
}

// 7. CONTROL: the deliberate PREFIX rule of dotprod( iVec, oVec ), which takes
//    oVec.size() <= nrows_ on purpose (refactor_audit.md 11.3). A shorter
//    destination computes the first rows only and must KEEP working -- a
//    policy that "fixed" this into an equality would break the bias arithmetic.
static int c07()
{
	return control_only( [] {
		M A = mat23();                 // 2 x 3
		vector< double > in = seq( 3 ); // 1 2 3
		vector< double > out( 1 );     // shorter than nrows_ -- legal
		A.dotprod( in, out );
		return out.size() == 1 && out[ 0 ] == 14; // 1*1 + 2*2 + 3*3
	} );
}

// 8. CONTROL: ordinary arithmetic is unchanged by anything this phase does.
static int c08()
{
	return control_only( [] {
		M A = mat23(), B = mat23();
		A += B;
		M T = B.t();
		vector< double > in = seq( 3 ), out( 2 );
		B.dotprod( in, out );
		return A( 0, 0 ) == 2 && A( 1, 2 ) == 12
			&& T.rows() == 3 && T.cols() == 2 && T( 2, 1 ) == 6
			&& out[ 0 ] == 14 && out[ 1 ] == 32;
	} );
}

// --- ABSENT: every Class-A contract, watched failing --------------------------

// 9. row( r, v ): the row index is past the end. Reads a whole row beyond the
//    allocation. This is the one the assert-enabled build caught in
//    check_onehidden.
static int c09()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); vector< double > v( 3 ); A.row( 1, v );
			return v[ 0 ] == 4 && v[ 2 ] == 6; },
		[] { M A = mat23(); vector< double > v( 3 ); A.row( 7, v ); } );
}

// 10. row( r, v ): the destination is not ncols_ wide.
static int c10()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); vector< double > v( 3 ); A.row( 0, v );
			return v[ 1 ] == 2; },
		[] { M A = mat23(); vector< double > v( 9 ); A.row( 0, v ); } );
}

// 11. row( r ): the returning form allocates correctly, so only the index can
//     be wrong -- and it is unguarded.
static int c11()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); vector< double > v = A.row( 1 ); return v[ 1 ] == 5; },
		[] { M A = mat23(); vector< double > v = A.row( 4 ); ( void ) v; } );
}

// 12. col( c, v ): the column index is past the end -- a strided read.
static int c12()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); vector< double > v( 2 ); A.col( 2, v );
			return v[ 0 ] == 3 && v[ 1 ] == 6; },
		[] { M A = mat23(); vector< double > v( 2 ); A.col( 8, v ); } );
}

// 13. col( c, v ): the destination is not nrows_ tall.
static int c13()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); vector< double > v( 2 ); A.col( 0, v );
			return v[ 1 ] == 4; },
		[] { M A = mat23(); vector< double > v( 7 ); A.col( 0, v ); } );
}

// 14. replacerow: the row index is past the end. This one WRITES.
static int c14()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); A.replacerow( 0, seq( 3, 10 ) ); return A( 0, 0 ) == 10; },
		[] { M A = mat23(); A.replacerow( 6, seq( 3, 10 ) ); } );
}

// 15. replacerow: the source vector is wider than the matrix. WRITES.
static int c15()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); A.replacerow( 1, seq( 3, 10 ) ); return A( 1, 2 ) == 12; },
		[] { M A = mat23(); A.replacerow( 1, seq( 12, 10 ) ); } );
}

// 16. replacecol: the column index is past the end. WRITES.
static int c16()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); A.replacecol( 1, seq( 2, 10 ) ); return A( 0, 1 ) == 10; },
		[] { M A = mat23(); A.replacecol( 9, seq( 2, 10 ) ); } );
}

// 17. replacecol: the source vector is taller than the matrix. WRITES.
static int c17()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); A.replacecol( 2, seq( 2, 10 ) ); return A( 1, 2 ) == 11; },
		[] { M A = mat23(); A.replacecol( 2, seq( 20, 10 ) ); } );
}

// 18. submatrix: the row range runs past the end.
static int c18()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); M S = A.submatrix( 0, 0, 1, 2 );
			return S.rows() == 1 && S.cols() == 2 && S( 0, 0 ) == 2; },
		[] { M A = mat23(); M S = A.submatrix( 0, 5, 0, 2 ); ( void ) S; } );
}

// 19. submatrix: the destination does not match the requested block.
static int c19()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); M S( 1, 2 ); A.submatrix( 1, 1, 0, 1, S );
			return S( 0, 0 ) == 4 && S( 0, 1 ) == 5; },
		[] { M A = mat23(); M S( 5, 5 ); A.submatrix( 1, 1, 0, 1, S ); } );
}

// 20-23. The four compound operators with a differently-shaped right operand.
//     Unlike vector_ops there is NO prefix rule here: these walk both arrays in
//     lockstep by element count.
static int c20()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(), B = mat23(); A += B; return A( 0, 2 ) == 6; },
		[] { M A = mat23(); M B( 4, 4, 1.0 ); A += B; } );
}

static int c21()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(), B = mat23(); A -= B; return A( 0, 2 ) == 0; },
		[] { M A = mat23(); M B( 4, 4, 1.0 ); A -= B; } );
}

static int c22()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(), B = mat23(); A *= B; return A( 1, 0 ) == 16; },
		[] { M A = mat23(); M B( 4, 4, 1.0 ); A *= B; } );
}

static int c23()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(), B = mat23(); A /= B; return A( 1, 1 ) == 1; },
		[] { M A = mat23(); M B( 4, 4, 1.0 ); A /= B; } );
}

// 24. The binary form is the compound one with a copied left operand, so it
//     must inherit the contract rather than restate it.
static int c24()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(), B = mat23(); M C = A + B; return C( 1, 2 ) == 12; },
		[] { M A = mat23(); M B( 4, 4, 1.0 ); M C = A + B; ( void ) C; } );
}

// 25. t( M_in ): the destination is not the transpose's shape. WRITES.
static int c25()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); M T( 3, 2 ); A.t( T ); return T( 2, 1 ) == 6; },
		[] { M A = mat23(); M T( 2, 3 ); A.t( T ); } );
}

// 26. dotprod( iVec, oVec ): the input is not ncols_ long.
static int c26()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); vector< double > in = seq( 3 ), out( 2 );
			A.dotprod( in, out ); return out[ 1 ] == 32; },
		[] { M A = mat23(); vector< double > in = seq( 9 ), out( 2 );
			A.dotprod( in, out ); } );
}

// 27. dotprod( iVec, oVec ): a destination LONGER than nrows_. The prefix rule
//     permits shorter (case 7); longer walks off the data array.
static int c27()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); vector< double > in = seq( 3 ), out( 2 );
			A.dotprod( in, out ); return out[ 0 ] == 14; },
		[] { M A = mat23(); vector< double > in = seq( 3 ), out( 12 );
			A.dotprod( in, out ); } );
}

// 28. Ranged dotprod: the range does not span nrows_ rows.
static int c28()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); vector< double > in = seq( 3 ), out( 4, -1.0 );
			A.dotprod( in, out, 1, 2 );
			return out[ 0 ] == -1 && out[ 1 ] == 14 && out[ 2 ] == 32; },
		[] { M A = mat23(); vector< double > in = seq( 3 ), out( 4 );
			A.dotprod( in, out, 0, 3 ); } );
}

// 29. Ranged dotprod: the range ends outside the destination.
static int c29()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); vector< double > in = seq( 3 ), out( 2 );
			A.dotprod( in, out, 0, 1 ); return out[ 1 ] == 32; },
		[] { M A = mat23(); vector< double > in = seq( 3 ), out( 2 );
			A.dotprod( in, out, 4, 5 ); } );
}

// 30. dotprodt( iVec, oVec ): the input is not nrows_ long.
static int c30()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); vector< double > in = seq( 2 ), out( 3 );
			A.dotprodt( in, out ); return out[ 0 ] == 9; }, // 1*1 + 2*4
		[] { M A = mat23(); vector< double > in = seq( 9 ), out( 3 );
			A.dotprodt( in, out ); } );
}

// 31. Ranged dotprodt: the range does not span ncols_ columns.
static int c31()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); vector< double > in = seq( 2 ), out( 3 );
			A.dotprodt( in, out, 0, 2 ); return out[ 2 ] == 15; },
		[] { M A = mat23(); vector< double > in = seq( 2 ), out( 3 );
			A.dotprodt( in, out, 0, 1 ); } );
}

// 32. dotprod_row( D, r, oVec ): the row index is past the end of D.
static int c32()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(), D = mat23(); vector< double > out( 2 );
			A.dotprod_row( D, 0, out ); return out[ 0 ] == 14; },
		[] { M A = mat23(), D = mat23(); vector< double > out( 2 );
			A.dotprod_row( D, 6, out ); } );
}

// 33. dotprod_row: the two matrices disagree about width.
static int c33()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(), D = mat23(); vector< double > out( 2 );
			A.dotprod_row( D, 1, out ); return out[ 1 ] == 77; },
		[] { M A = mat23(); M D( 2, 7, 1.0 ); vector< double > out( 2 );
			A.dotprod_row( D, 0, out ); } );
}

// 34. dotprod( B, C ): the inner dimensions do not agree.
static int c34()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); M B = A.t(); M C( 2, 2 ); A.dotprod( B, C );
			return C( 0, 0 ) == 14 && C( 1, 1 ) == 77; },
		[] { M A = mat23(); M B( 5, 2, 1.0 ); M C( 2, 2 ); A.dotprod( B, C ); } );
}

// 35. dotprod( B, C ): the destination is the wrong shape. WRITES.
static int c35()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); M B = A.t(); M C( 2, 2 ); A.dotprod( B, C );
			return C( 0, 1 ) == 32; },
		[] { M A = mat23(); M B = A.t(); M C( 6, 6 ); A.dotprod( B, C ); } );
}

// 36. outprod: the left vector is not nrows_ long. WRITES -- and this is on the
//     per-exemplar training path (hWup.outprod( h_err, I )).
static int c36()
{
	return probe< M::BoundsViolation >(
		[] { M A( 2, 3 ); A.outprod( seq( 2 ), seq( 3 ) ); return A( 1, 2 ) == 6; },
		[] { M A( 2, 3 ); A.outprod( seq( 11 ), seq( 3 ) ); } );
}

// 37. outprod: the right vector is not ncols_ long. WRITES.
static int c37()
{
	return probe< M::BoundsViolation >(
		[] { M A( 2, 3 ); A.outprod( seq( 2 ), seq( 3 ) ); return A( 0, 2 ) == 3; },
		[] { M A( 2, 3 ); A.outprod( seq( 2 ), seq( 13 ) ); } );
}

// 38. colsums: the destination is not ncols_ long. WRITES.
static int c38()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); vector< double > s( 3 ); A.colsums( s );
			return s[ 0 ] == 5 && s[ 2 ] == 9; },
		[] { M A = mat23(); vector< double > s( 1 ); A.colsums( s ); } );
}

// 39. rowindex: the destination is not nrows_ long. WRITES.
static int c39()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); vector< unsigned > v( 2 ); A.rowindex( v );
			return v.size() == 2; },
		[] { M A = mat23(); vector< unsigned > v( 1 ); A.rowindex( v ); } );
}

// 40. toVector( v ): the destination is not rows*cols long. WRITES.
static int c40()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); vector< double > v( 6 ); A.toVector( v );
			return v[ 0 ] == 1 && v[ 5 ] == 6; },
		[] { M A = mat23(); vector< double > v( 2 ); A.toVector( v ); } );
}

// 41. toMatrix( M, v ): the source is not rows*cols long.
static int c41()
{
	return probe< M::BoundsViolation >(
		[] { M A( 2, 3 ); vector< double > v = seq( 6 ); toMatrix( A, v );
			return A( 1, 2 ) == 6; },
		[] { M A( 2, 3 ); vector< double > v = seq( 2 ); toMatrix( A, v ); } );
}

// 42. toMatrix( v, r, c ): the free returning form, same contract.
static int c42()
{
	return probe< M::BoundsViolation >(
		[] { vector< double > v = seq( 6 ); M A = toMatrix( v, 2, 3 );
			return A( 0, 1 ) == 2; },
		[] { vector< double > v = seq( 2 ); M A = toMatrix( v, 2, 3 ); ( void ) A; } );
}

// 43. The free func( Mi, fx, Mo ): the two matrices disagree. WRITES.
static int c43()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); M B( 2, 3 ); func( A, []( double x ){ return 2 * x; }, B );
			return B( 1, 2 ) == 12; },
		[] { M A = mat23(); M B( 1, 1 ); func( A, []( double x ){ return 2 * x; }, B ); } );
}

// 44. includecols: a position past the last column. The write is into a
//     vector< bool > sized ncols_, so this one corrupts a different object.
static int c44()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); vector< unsigned > pos; pos.push_back( 0 ); pos.push_back( 2 );
			M C = A.includecols( pos );
			return C.cols() == 2 && C( 0, 1 ) == 3 && C( 1, 0 ) == 4; },
		[] { M A = mat23(); vector< unsigned > pos; pos.push_back( 99 );
			M C = A.includecols( pos ); ( void ) C; } );
}

// 45. includecols with an EMPTY position vector. Recorded because the guard is
//     the pathology: the assert dereferences max_element on an empty range, so
//     a checked build has undefined behavior exactly where Release has no check
//     at all. What Release does with it is what this case reports.
static int c45()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); vector< unsigned > pos; pos.push_back( 1 );
			M C = A.includecols( pos ); return C.cols() == 1 && C( 0, 0 ) == 2; },
		[] { M A = mat23(); vector< unsigned > pos;
			M C = A.includecols( pos ); ( void ) C; } );
}

// 46. excludecols: a position past the last column.
static int c46()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); vector< unsigned > pos; pos.push_back( 1 );
			M C = A.excludecols( pos );
			return C.cols() == 2 && C( 0, 1 ) == 3; },
		[] { M A = mat23(); vector< unsigned > pos; pos.push_back( 42 );
			M C = A.excludecols( pos ); ( void ) C; } );
}

// 47. addrow: the new row is not ncols_ wide.
static int c47()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); M B = A.addrow( seq( 3, 7 ) );
			return B.rows() == 3 && B( 2, 0 ) == 7; },
		[] { M A = mat23(); M B = A.addrow( seq( 9, 7 ) ); ( void ) B; } );
}

// 48. addcol: the new column is not nrows_ tall.
static int c48()
{
	return probe< M::BoundsViolation >(
		[] { M A = mat23(); M B = A.addcol( seq( 2, 7 ) );
			return B.cols() == 4 && B( 0, 3 ) == 7; },
		[] { M A = mat23(); M B = A.addcol( seq( 9, 7 ) ); ( void ) B; } );
}

// 49. covariance( V ): the destination is not ncols_ x ncols_. WRITES.
static int c49()
{
	return probe< M::BoundsViolation >(
		[] { M A( 3, 2 );
			A( 0, 0 ) = 1; A( 0, 1 ) = 2; A( 1, 0 ) = 3; A( 1, 1 ) = 4;
			A( 2, 0 ) = 5; A( 2, 1 ) = 6;
			M V( 2, 2 ); A.covariance( V ); return V( 0, 0 ) == 4; },
		[] { M A( 3, 2 );
			A( 0, 0 ) = 1; A( 0, 1 ) = 2; A( 1, 0 ) = 3; A( 1, 1 ) = 4;
			A( 2, 0 ) = 5; A( 2, 1 ) = 6;
			M V( 5, 5 ); A.covariance( V ); } );
}

// 50. inverse( I ): a non-square matrix has no inverse to compute, and the
//     shape check is an assert while the SINGULAR check is a throw.
static int c50()
{
	return probe< M::BoundsViolation >(
		[] { M A( 2, 2 );
			A( 0, 0 ) = 2; A( 0, 1 ) = 0; A( 1, 0 ) = 0; A( 1, 1 ) = 4;
			M I( 2, 2 ); A.inverse( I ); return I( 0, 0 ) == 0.5; },
		[] { M A = mat23(); M I( 2, 3 ); A.inverse( I ); } );
}

// 51. Matrix( Q, Pt ): an outer product of empty vectors builds a 0 x 0 matrix.
static int c51()
{
	return probe< M::BadSize >(
		[] { M A( seq( 2 ), seq( 3 ) ); return A.rows() == 2 && A.cols() == 3
			&& A( 1, 2 ) == 6; },
		[] { vector< double > e; M A( e, e ); ( void ) A; } );
}

// 52. inverse( I ) on a TALLER-than-wide matrix. Case 50 threw, but not from a
//     precondition: LU decomposition loops over ncols_ and indexes rows by it,
//     so a 2 x 3 matrix runs operator() off the end and that accessor -- the
//     one contract this class does enforce -- refuses it. A 3 x 2 matrix has
//     MORE rows than the loop bound, so every index stays in range and the
//     algorithm runs to completion on a matrix that has no inverse. The pair
//     is what shows the safety is incidental rather than contractual.
static int c52()
{
	return probe< M::BoundsViolation >(
		[] { M A( 2, 2 );
			A( 0, 0 ) = 2; A( 0, 1 ) = 0; A( 1, 0 ) = 0; A( 1, 1 ) = 4;
			M I( 2, 2 ); A.inverse( I ); return I( 1, 1 ) == 0.25; },
		[] { M A( 3, 2 );
			A( 0, 0 ) = 1; A( 0, 1 ) = 2; A( 1, 0 ) = 3; A( 1, 1 ) = 4;
			A( 2, 0 ) = 5; A( 2, 1 ) = 6;
			M I( 3, 2 ); A.inverse( I ); } );
}

// --- the register ----------------------------------------------------------

struct Case { const char* half; const char* name; int ( *run )(); };

static const Case cases[] = {
	{ "HOLDS ", "operator(): row past the end", c01 },
	{ "HOLDS ", "operator(): column past the end", c02 },
	{ "HOLDS ", "operator() const: both past the end", c03 },
	{ "HOLDS ", "includerows: gather position past the end", c04 },
	{ "HOLDS ", "Matrix( 0, c, value ): BadSize", c05 },
	{ "HOLDS ", "inverse: a singular matrix throws Singular", c06 },
	{ "HOLDS ", "CONTROL dotprod: a SHORTER destination is the prefix rule", c07 },
	{ "HOLDS ", "CONTROL: ordinary arithmetic, transpose and dotprod", c08 },
	{ "ABSENT", "row( r, v ): row index past the end", c09 },
	{ "ABSENT", "row( r, v ): destination not ncols_ wide", c10 },
	{ "ABSENT", "row( r ): row index past the end", c11 },
	{ "ABSENT", "col( c, v ): column index past the end", c12 },
	{ "ABSENT", "col( c, v ): destination not nrows_ tall", c13 },
	{ "ABSENT", "replacerow: row index past the end (WRITES)", c14 },
	{ "ABSENT", "replacerow: source wider than the matrix (WRITES)", c15 },
	{ "ABSENT", "replacecol: column index past the end (WRITES)", c16 },
	{ "ABSENT", "replacecol: source taller than the matrix (WRITES)", c17 },
	{ "ABSENT", "submatrix: row range past the end", c18 },
	{ "ABSENT", "submatrix: destination does not match the block", c19 },
	{ "ABSENT", "operator+=: differing dimensions", c20 },
	{ "ABSENT", "operator-=: differing dimensions", c21 },
	{ "ABSENT", "operator*=: differing dimensions", c22 },
	{ "ABSENT", "operator/=: differing dimensions", c23 },
	{ "ABSENT", "operator+: differing dimensions (inherits)", c24 },
	{ "ABSENT", "t( M_in ): destination not the transpose shape (WRITES)", c25 },
	{ "ABSENT", "dotprod( in, out ): input not ncols_ long", c26 },
	{ "ABSENT", "dotprod( in, out ): destination LONGER than nrows_", c27 },
	{ "ABSENT", "dotprod ranged: range does not span nrows_", c28 },
	{ "ABSENT", "dotprod ranged: range ends outside the destination", c29 },
	{ "ABSENT", "dotprodt( in, out ): input not nrows_ long", c30 },
	{ "ABSENT", "dotprodt ranged: range does not span ncols_", c31 },
	{ "ABSENT", "dotprod_row: row index past the end of the dataset", c32 },
	{ "ABSENT", "dotprod_row: the two matrices disagree about width", c33 },
	{ "ABSENT", "dotprod( B, C ): inner dimensions disagree", c34 },
	{ "ABSENT", "dotprod( B, C ): destination wrong shape (WRITES)", c35 },
	{ "ABSENT", "outprod: left vector not nrows_ long (WRITES)", c36 },
	{ "ABSENT", "outprod: right vector not ncols_ long (WRITES)", c37 },
	{ "ABSENT", "colsums: destination not ncols_ long (WRITES)", c38 },
	{ "ABSENT", "rowindex: destination not nrows_ long (WRITES)", c39 },
	{ "ABSENT", "toVector( v ): destination not rows*cols long (WRITES)", c40 },
	{ "ABSENT", "toMatrix( M, v ): source not rows*cols long", c41 },
	{ "ABSENT", "toMatrix( v, r, c ): source not r*c long", c42 },
	{ "ABSENT", "func( Mi, fx, Mo ): the two matrices disagree (WRITES)", c43 },
	{ "ABSENT", "includecols: position past the last column", c44 },
	{ "ABSENT", "includecols: an EMPTY position vector", c45 },
	{ "ABSENT", "excludecols: position past the last column", c46 },
	{ "ABSENT", "addrow: new row not ncols_ wide", c47 },
	{ "ABSENT", "addcol: new column not nrows_ tall", c48 },
	{ "ABSENT", "covariance( V ): destination not ncols_ x ncols_ (WRITES)", c49 },
	{ "ABSENT", "inverse( I ): a non-square matrix", c50 },
	{ "ABSENT", "Matrix( Q, Pt ): empty vectors", c51 },
	{ "ABSENT", "inverse( I ): a TALLER-than-wide matrix (see case 50)", c52 },
};

static const unsigned nCases = sizeof( cases ) / sizeof( cases[ 0 ] );

static const char* verdictWord( int v )
{
	switch ( v )
	{
		case HELD:           return "held";
		case NO_THROW:       return "NO THROW";
		case CONTROL_FAILED: return "CONTROL FAILED";
		case WRONG_TYPE:     return "WRONG TYPE";
		default:             return "?";
	}
}

int main( int argc, char* argv[] )
{
	// The case list, read UP FRONT by the driver so that a case which kills its
	//    own process is still named in the report.
	if ( argc > 1 && strcmp( argv[ 1 ], "-l" ) == 0 )
	{
		for ( unsigned i = 0; i < nCases; i++ )
			cout << ( i + 1 ) << " [" << cases[ i ].half << "] "
				<< cases[ i ].name << endl;
		return 0;
	}

	// Single-case mode: the exit status IS the verdict.
	if ( argc > 1 )
	{
		unsigned n = ( unsigned ) atoi( argv[ 1 ] );
		if ( n < 1 || n > nCases )
		{
			cout << "case out of range 1.." << nCases << endl;
			return 2;
		}
		int v = cases[ n - 1 ].run();
		cout << verdictWord( v ) << " - " << n << ". [" << cases[ n - 1 ].half
			<< "] " << cases[ n - 1 ].name << endl;
		return v;
	}

	// All-cases mode. Until the policy exists this WILL die partway through --
	//    that is the finding, and it is why the driver runs one per process.
	unsigned failures = 0;
	for ( unsigned i = 0; i < nCases; i++ )
	{
		int v = cases[ i ].run();
		if ( v != HELD )
			failures++;
		cout << ( v == HELD ? "ok - " : "FAIL - " ) << ( i + 1 ) << ". ["
			<< cases[ i ].half << "] " << cases[ i ].name
			<< ( v == HELD ? "" : string( " -- " ) + verdictWord( v ) ) << endl;
	}

	cout << endl << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
