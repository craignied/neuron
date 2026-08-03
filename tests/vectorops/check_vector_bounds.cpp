// check_vector_bounds.cpp : the bounds contract of the numerical vector layer.
//
// WHY THIS TARGET DEFINES NDEBUG EXPLICITLY.
//
// Every precondition in vector_ops.h was an assert, and the engine ships with
// NDEBUG defined -- CMakeLists.txt forces Release when no build type is given,
// and Release is `-O3 -DNDEBUG`. So all nineteen ctest targets already ran with
// assertions compiled out: not one assertion in that file had ever executed in
// the gate chain. A test that passes in a checked build proves nothing about
// the binary we ship, so this target sets NDEBUG itself and does not care how
// the build directory was configured.
//
// THE MOTIVATING DEFECT. SimpleProp::trainSet passed hO (nHidden + 1 elements,
// the last one the bias slot) into h_err (nHidden) through the unranged
//
//     func( vec_in, fx, vec_out )
//
// which transforms vec_in.size() elements INTO vec_out -- one past the end of
// h_err, in every shipped build. Case 1 below is that defect exactly.
//
// ONE CASE PER PROCESS. Run with no argument, this runs every case and is an
// ordinary ctest case. Run as `check_vector_bounds <n>` it runs exactly one.
// That mode exists for the demonstration: against the assert-only code these
// violations do not throw, they corrupt memory, and a crash in one case would
// hide the ones after it. run_bounds_demo.sh runs each in its own process.
//
// WHAT IS ASSERTED
//   Cases 1-17  every contract whose violation can read or write outside a
//               container in Release, each expecting its own exception type.
//   Cases 18-27 POSITIVE CONTROLS -- the successful path is unchanged. In
//               particular the deliberate PREFIX semantics of the compound
//               operators (lhs.size() <= rhs.size(), which the bias-slot
//               arithmetic of SimpleProp and BackProp depends on) must keep
//               working, and boundary-valid ranges must not be refused.
//
// The full inventory, the policy, and why `*=` takes a prefix while dotprod
// requires equality are in docs/refactor_audit.md section 11.

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "vector_ops.h"

using namespace std;

// --- expectation helpers ---------------------------------------------------

// Runs `body`; returns true only if it threw exactly E.
template< class E, class Body >
static bool throws( Body body )
{
	try { body(); }
	catch ( const E& ) { return true; }
	catch ( ... ) { return false; } // some other exception is not the contract
	return false; // no throw at all -- the Release gap
}

// Runs `body`; returns true only if it did NOT throw and `body` reported ok.
template< class Body >
static bool succeeds( Body body )
{
	try { return body(); }
	catch ( ... ) { return false; }
}

static double twice( double v ) { return 2 * v; }

static vector< double > seq( unsigned n, double base = 1 )
{
	vector< double > v;
	for ( unsigned i = 0; i < n; i++ )
		v.push_back( base + i );
	return v;
}

// --- the cases -------------------------------------------------------------

// 1. THE PROVEN DEFECT: a source longer than the destination.
static bool c01()
{
	return throws< nvec::SizeMismatch >( []
	{
		vector< double > in = seq( 4 ), out = seq( 3 );
		func( in, twice, out );
	} );
}

// 2. The other direction: a source shorter than the destination.
static bool c02()
{
	return throws< nvec::SizeMismatch >( []
	{
		vector< double > in = seq( 3 ), out = seq( 4 );
		func( in, twice, out );
	} );
}

// 3. Destination-range func with a > b -- a reversed iterator range.
static bool c03()
{
	return throws< nvec::RangeViolation >( []
	{
		vector< double > in = seq( 5 ), out = seq( 5 );
		func( in, twice, out, 3u, 1u );
	} );
}

// 4. Destination-range func with b outside the INPUT. The shipped assert
//    checked b against vec_out only, so this side was never guarded at all --
//    the destination is deliberately large enough here to prove that the input
//    bound is what refuses it.
static bool c04()
{
	return throws< nvec::RangeViolation >( []
	{
		vector< double > in = seq( 3 ), out = seq( 10 );
		func( in, twice, out, 0u, 5u );
	} );
}

// 5. Destination-range func with b outside the DESTINATION.
static bool c05()
{
	return throws< nvec::RangeViolation >( []
	{
		vector< double > in = seq( 10 ), out = seq( 3 );
		func( in, twice, out, 0u, 5u );
	} );
}

// 6. Returning-range func with a > b.
static bool c06()
{
	return throws< nvec::RangeViolation >( []
	{
		vector< double > in = seq( 5 );
		vector< double > got = func( in, twice, 3u, 1u );
		( void ) got;
	} );
}

// 7. Returning-range func with b outside the input.
static bool c07()
{
	return throws< nvec::RangeViolation >( []
	{
		vector< double > in = seq( 3 );
		vector< double > got = func( in, twice, 0u, 5u );
		( void ) got;
	} );
}

// 8. dotprod requires EQUALITY -- a dot product over a prefix is a different
//    scalar, and returning it silently is the failure a check exists to stop.
static bool c08()
{
	return throws< nvec::SizeMismatch >( []
	{
		vector< double > a = seq( 4 ), b = seq( 3 );
		double d = dotprod( a, b );
		( void ) d;
	} );
}

// 9. sumSquaredDifference walks b in lockstep with a, past its end.
static bool c09()
{
	return throws< nvec::SizeMismatch >( []
	{
		vector< double > a = seq( 4 ), b = seq( 3 );
		double d = sumSquaredDifference( a, b );
		( void ) d;
	} );
}

// 10-13. Each compound operator with an UNDERSIZED right operand. A longer
//    rhs is legal (case 18); a shorter one reads past its end.
static bool c10()
{
	return throws< nvec::SizeMismatch >( []
	{
		vector< double > lhs = seq( 4 ), rhs = seq( 3 );
		lhs += rhs;
	} );
}

static bool c11()
{
	return throws< nvec::SizeMismatch >( []
	{
		vector< double > lhs = seq( 4 ), rhs = seq( 3 );
		lhs -= rhs;
	} );
}

static bool c12()
{
	return throws< nvec::SizeMismatch >( []
	{
		vector< double > lhs = seq( 4 ), rhs = seq( 3 );
		lhs *= rhs;
	} );
}

static bool c13()
{
	return throws< nvec::SizeMismatch >( []
	{
		vector< double > lhs = seq( 4 ), rhs = seq( 3 );
		lhs /= rhs;
	} );
}

// 14. The binary operators are the compound ones with a by-value lhs, so they
//     must inherit the contract rather than restate it.
static bool c14()
{
	return throws< nvec::SizeMismatch >( []
	{
		vector< double > a = seq( 4 ), b = seq( 3 );
		vector< double > got = a - b;
		( void ) got;
	} );
}

// 15. minabs dereferences *v.begin() unconditionally.
static bool c15()
{
	return throws< nvec::EmptyVector >( []
	{
		vector< double > empty;
		double m = minabs( empty );
		( void ) m;
	} );
}

// 16. maxabs returns 0 on empty, which is memory-safe and a LIE: the maximum
//     of an empty set has no value, and 0 is the one value a caller acts on --
//     Network::getGradMax feeds it to the gradient stopping rule, where "the
//     largest gradient is 0" means converged. No live caller passes empty
//     (see docs/refactor_audit.md 11.5), so refusing costs nothing and removes a
//     false convergence signal.
static bool c16()
{
	return throws< nvec::EmptyVector >( []
	{
		vector< double > empty;
		double m = maxabs( empty );
		( void ) m;
	} );
}

// 17. bin with a zero bin size: v_in.size() / b is integer division by zero.
//     RangeViolation, not SizeMismatch -- b is not a container length, it is
//     the extent of each output bin, and 0 is outside the extents a container
//     can be partitioned into.
static bool c17()
{
	return throws< nvec::RangeViolation >( []
	{
		vector< double > in = seq( 6 );
		vector< vector< double > > out = bin( in, 0u, false );
		( void ) out;
	} );
}

// --- positive controls: the successful path is unchanged -------------------

// 18. THE PREFIX SEMANTIC, which is deliberate and load-bearing. SimpleProp
//     multiplies h_err (nHidden) by oW (nHidden + 1) and relies on the extra
//     bias element being ignored; BackProp says the same thing at
//     backprop.cpp:476. A longer rhs must keep working.
static bool c18()
{
	return succeeds( []
	{
		vector< double > lhs = seq( 3 ), rhs = seq( 5, 10 ); // 10 11 12 13 14
		lhs *= rhs;
		return lhs.size() == 3 && lhs[ 0 ] == 10 && lhs[ 1 ] == 22
			&& lhs[ 2 ] == 36;
	} );
}

// 19. The same for +=, and the result keeps the LHS's size.
static bool c19()
{
	return succeeds( []
	{
		vector< double > lhs = seq( 2 ), rhs = seq( 6, 100 );
		lhs += rhs;
		return lhs.size() == 2 && lhs[ 0 ] == 101 && lhs[ 1 ] == 103;
	} );
}

// 20. A range covering the whole vector: a = 0, b = size - 1 is VALID and
//     must not be refused by an off-by-one in the new check.
static bool c20()
{
	return succeeds( []
	{
		vector< double > in = seq( 4 ), out( 4, 0.0 );
		func( in, twice, out, 0u, 3u );
		return out[ 0 ] == 2 && out[ 3 ] == 8;
	} );
}

// 21. A single-element range, a == b.
static bool c21()
{
	return succeeds( []
	{
		vector< double > in = seq( 4 ), out( 4, -1.0 );
		func( in, twice, out, 2u, 2u );
		return out[ 2 ] == 6 && out[ 0 ] == -1 && out[ 3 ] == -1;
	} );
}

// 22. The returning range form, single element.
static bool c22()
{
	return succeeds( []
	{
		vector< double > in = seq( 4 );
		vector< double > got = func( in, twice, 2u, 2u );
		return got.size() == 1 && got[ 0 ] == 6;
	} );
}

// 23. A sum over an empty index set is 0 -- an identity, not a refusal. This
//     is why EmptyVector covers minabs/maxabs and not these.
static bool c23()
{
	return succeeds( []
	{
		vector< double > e1, e2;
		return dotprod( e1, e2 ) == 0 && sumSquaredDifference( e1, e2 ) == 0
			&& squared( e1 ) == 0;
	} );
}

// 24. Equal-size operands of every guarded operation, computing the right
//     answers -- the check must be a gate, not a change of arithmetic.
static bool c24()
{
	return succeeds( []
	{
		vector< double > a = seq( 3 ), b = seq( 3, 4 ); // 1 2 3 / 4 5 6
		bool ok = dotprod( a, b ) == 32                  // 4 + 10 + 18
			&& sumSquaredDifference( a, b ) == 27;       // 9 + 9 + 9
		vector< double > out( 3 );
		func( a, twice, out );
		return ok && out[ 0 ] == 2 && out[ 2 ] == 6;
	} );
}

// 25. bin on an empty input is memory-safe today and stays accepted: the
//     short-input branch appends the empty vector. The assert that forbade it
//     was stricter than the code needed.
static bool c25()
{
	return succeeds( []
	{
		vector< double > empty;
		vector< vector< double > > out = bin( empty, 3u, false );
		return out.size() == 1 && out[ 0 ].empty();
	} );
}

// 26. An ordinary bin still partitions as it did.
static bool c26()
{
	return succeeds( []
	{
		vector< double > in = seq( 7 );
		vector< vector< double > > out = bin( in, 3u, false );
		return out.size() == 3 && out[ 0 ].size() == 3 && out[ 1 ].size() == 3
			&& out[ 2 ].size() == 1;
	} );
}

// 27. minabs and maxabs on ordinary data, including negatives.
static bool c27()
{
	return succeeds( []
	{
		vector< double > v;
		v.push_back( -7 ); v.push_back( 3 ); v.push_back( -1 );
		return maxabs( v ) == 7 && minabs( v ) == 1;
	} );
}

// --- the register ----------------------------------------------------------

struct Case { const char* name; bool ( *run )(); };

static const Case cases[] = {
	{ "func( in, f, out ): source LONGER than destination", c01 },
	{ "func( in, f, out ): source shorter than destination", c02 },
	{ "func( in, f, out, a, b ): a > b", c03 },
	{ "func( in, f, out, a, b ): b outside the INPUT", c04 },
	{ "func( in, f, out, a, b ): b outside the DESTINATION", c05 },
	{ "func( in, f, a, b ): a > b", c06 },
	{ "func( in, f, a, b ): b outside the input", c07 },
	{ "dotprod: unequal sizes", c08 },
	{ "sumSquaredDifference: unequal sizes", c09 },
	{ "operator+=: undersized rhs", c10 },
	{ "operator-=: undersized rhs", c11 },
	{ "operator*=: undersized rhs", c12 },
	{ "operator/=: undersized rhs", c13 },
	{ "operator-: undersized rhs (inherits the contract)", c14 },
	{ "minabs: empty vector", c15 },
	{ "maxabs: empty vector", c16 },
	{ "bin: zero bin size", c17 },
	{ "CONTROL *=: a LONGER rhs is a prefix, and legal", c18 },
	{ "CONTROL +=: a longer rhs keeps the lhs size", c19 },
	{ "CONTROL func range: a = 0, b = size - 1 is valid", c20 },
	{ "CONTROL func range: a single element", c21 },
	{ "CONTROL func range returning form: a single element", c22 },
	{ "CONTROL empty operands: a sum over nothing is 0", c23 },
	{ "CONTROL equal sizes: the arithmetic is unchanged", c24 },
	{ "CONTROL bin: an empty input is accepted", c25 },
	{ "CONTROL bin: an ordinary partition", c26 },
	{ "CONTROL minabs / maxabs on ordinary data", c27 },
};

static const unsigned nCases = sizeof( cases ) / sizeof( cases[ 0 ] );

int main( int argc, char* argv[] )
{
	// List the cases, one per line, "<n> <name>". The demonstration driver
	//    reads this UP FRONT, so that a case which kills its process is still
	//    named in the report -- a crashed run prints nothing of its own.
	if ( argc > 1 && strcmp( argv[ 1 ], "-l" ) == 0 )
	{
		for ( unsigned i = 0; i < nCases; i++ )
			cout << ( i + 1 ) << " " << cases[ i ].name << endl;
		return 0;
	}

	// Single-case mode, for the per-process demonstration driver
	if ( argc > 1 )
	{
		unsigned n = ( unsigned ) atoi( argv[ 1 ] );
		if ( n < 1 || n > nCases )
		{
			cout << "case out of range 1.." << nCases << endl;
			return 2;
		}
		bool ok = cases[ n - 1 ].run();
		cout << ( ok ? "ok - " : "FAIL - " ) << n << ". "
			<< cases[ n - 1 ].name << endl;
		return ok ? 0 : 1;
	}

	unsigned failures = 0;
	for ( unsigned i = 0; i < nCases; i++ )
	{
		bool ok = cases[ i ].run();
		if ( !ok )
			failures++;
		cout << ( ok ? "ok - " : "FAIL - " ) << ( i + 1 ) << ". "
			<< cases[ i ].name << endl;
	}

	cout << endl << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
