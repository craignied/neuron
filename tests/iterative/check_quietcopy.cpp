// check_quietcopy.cpp : a cloned model starts AUDIBLE, always.
//
// THE BUG (found 2026-07-27 by Sol, reviewing the stepwise regression work).
// Iterative gained a runtime quietFlag so stepwise regression's candidate
// refits could skip a reporting epilogue nobody consumes (classification
// tables, an ROC fit and a 2000-resample bootstrap over the test set, once per
// candidate). The flag was documented "NOT copied: a clone must not inherit
// silence by accident" -- and then that was implemented by OMISSION.
//
// Omission is not initialisation. Iterative's copy constructor calls copy()
// WITHOUT running the default constructor first, so a member copy() never
// touches holds whatever was on the stack, not false. Every clone in the
// engine goes through that path: autoalgo's optimizer probes, OBD's per-size
// trials, CV's per-fold models, RegressNet's candidates. Any of them could
// have come up quiet and silently discarded its entire report, unpredictably
// and unreproducibly, depending only on memory contents.
//
// Stepwise itself masked this, because it calls setQuiet(true) on every
// candidate explicitly -- so the one caller that cared always got the right
// answer and the defect could only ever have surfaced somewhere else.
//
// This is the same failure as the uninitialised Model::errorType scalar behind
// the nested-OBD nondeterminism (HISTORY 2026-07-23): a heap/stack-layout
// -sensitive read that behaves differently between builds and runs.
//
// The invariant: a copy is audible regardless of the source's setting, and
// regardless of what memory happened to contain.
//
// NOTE ON PRE-FIX BEHAVIOUR. Reading an uninitialised bool is undefined, so a
// pre-fix binary is not guaranteed to FAIL this test -- it is guaranteed to be
// unpredictable, which is the defect. To make the pre-fix failure observable
// rather than a matter of luck, the clones below are constructed over memory
// deliberately dirtied with a nonzero pattern first; on the pre-fix build that
// is what a fresh clone reads.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include "simpleprop.h"

static int failures = 0;

static void check( bool ok, const char* what )
{
	printf( "%-62s %s\n", what, ok ? "ok" : "FAILED" );
	if ( !ok ) failures++;
}

int main()
{
	printf( "quiet flag is not inherited by clones\n\n" );

	// 1. A default-constructed model is audible.
	{
		SimpleProp fresh;
		check( !fresh.getQuiet(), "a fresh model is audible" );
	}

	// 2. A clone of an AUDIBLE model is audible -- over dirtied memory, so a
	//    build that never assigns the flag reads the dirt.
	{
		alignas( SimpleProp ) unsigned char buf[ sizeof( SimpleProp ) ];
		memset( buf, 0xFF, sizeof buf ); // every bool-sized byte reads true
		SimpleProp source;
		SimpleProp* clone = new ( buf ) SimpleProp( source );
		check( !clone->getQuiet(), "a clone of an audible model is audible" );
		clone->~SimpleProp();
	}

	// 3. A clone of a QUIET model is audible too: silence is a per-run decision
	//    its caller makes explicitly, never something a copy picks up.
	{
		alignas( SimpleProp ) unsigned char buf[ sizeof( SimpleProp ) ];
		memset( buf, 0xFF, sizeof buf );
		SimpleProp source;
		source.setQuiet( true );
		SimpleProp* clone = new ( buf ) SimpleProp( source );
		check( !clone->getQuiet(), "a clone of a QUIET model is still audible" );
		// ... and the source is unchanged by being copied
		check( source.getQuiet(), "the quiet source stays quiet" );
		clone->~SimpleProp();
	}

	// 4. Assignment carries the same rule as construction (operator= routes
	//    through the same copy()).
	{
		SimpleProp source;
		source.setQuiet( true );
		SimpleProp target;
		target.setQuiet( true ); // start quiet, so this cannot pass by accident
		target = source;
		check( !target.getQuiet(), "assignment leaves the target audible" );
	}

	printf( "\n%s\n", failures ? "FAILURES" : "all quiet-copy checks passed" );
	return failures ? 1 : 0;
}
