// Verifies util::set_screen(): engine-core output must be capturable into a
// string (the GUI/server contract) and cout must be restorable. Exercises
// both output paths: prints routed internally through util::screen()
// (DataSet's refusal message) and report methods taking an ostream&
// parameter (TwoSet::ClassTable). No GSL needed.

#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <thread>

#include "dataset.h"
#include "twoset.h"
#include "utility.h"

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


// --- ScreenCapture: the SCOPED redirection (2026-08-01) --------------------
//
// Six places in the engine did save/restore by hand around a call that can
// throw (autoalgo's probe, obd's per-size train and optimizer probe,
// cvadapters' quiet fit and nested OBD). On the throw path the restore never
// ran, leaving the engine's stream pointing at an ostringstream about to be
// destroyed -- every later engine print in that process writing through a
// dangling reference. The GUI's own capture restored unconditionally to cout,
// so nesting silently lost the outer scope's output.
//
// SABOTAGE (each watched to fail, then restored):
//   * destructor restoring `cout` instead of `previous`  -> nesting fails;
//   * restore moved out of the destructor into a trailing statement
//     -> exception restoration fails;
//   * constructor saving nothing (previous = &cout)      -> nesting fails.

static void test_scoped_capture()
{
	// (1) It captures, and it restores the PREVIOUS stream -- not cout
	ostringstream outer;
	util::set_screen( outer );
	{
		ScreenCapture inner;
		util::screen() << "inner text";
		expect( inner.text() == "inner text", "ScreenCapture captures" );
	}
	expect( &util::screen() == &outer,
		"ScreenCapture restores the stream that was current, not cout" );

	// (2) NESTING: the outer buffer must still receive outer writes afterwards
	util::screen() << "after";
	expect( outer.str() == "after", "a nested capture does not steal outer output" );

	// (3) EXCEPTION UNWINDING: the restore must happen on the throw path
	try
	{
		ScreenCapture guard;
		util::screen() << "swallowed";
		throw runtime_error( "boom" );
	}
	catch ( const runtime_error& ) { }
	expect( &util::screen() == &outer,
		"ScreenCapture restores while an exception unwinds" );

	// (4) Deep nesting restores in order
	ostringstream a, b;
	{
		ScreenCapture one( a );
		{
			ScreenCapture two( b );
			expect( &util::screen() == &b, "innermost capture is active" );
		}
		expect( &util::screen() == &a, "restores to the enclosing capture" );
	}
	expect( &util::screen() == &outer, "and back out to the original" );

	util::set_screen( cout );
}

// (5) THREAD-LOCAL ISOLATION: a capture on one thread must not redirect
// another. The engine relies on this -- each GUI request thread and the async
// training worker capture independently.
static void test_thread_isolation()
{
	ostringstream mine;
	ScreenCapture capture( mine );
	util::screen() << "main thread";

	bool otherSawCout = false;
	bool otherLeakedIntoMine = false;
	thread t( [ & ]
	{
		// A fresh thread starts at cout, NOT at the capturing thread's stream
		otherSawCout = ( &util::screen() == &cout );
		{
			ostringstream theirs;
			ScreenCapture theirCapture( theirs );
			util::screen() << "worker";
			otherLeakedIntoMine = ( theirs.str() != "worker" );
		}
	} );
	t.join();

	expect( otherSawCout, "a new thread starts at cout, not the capturing stream" );
	expect( !otherLeakedIntoMine, "another thread's capture is its own" );
	expect( mine.str() == "main thread",
		"and the other thread wrote nothing into this one's capture" );
}

int main()
{
	// --- Path 1: engine prints routed through util::screen() -------------
	ostringstream captured;
	util::set_screen( captured );

	// A 5-column Matrix into a DataSet expecting 1 input + 1 output makes
	// the engine print its refusal — through util::screen()
	Matrix< double > wrong( 4, 5 );
	DataSet d;
	d.setRawMatrix( wrong );

	util::set_screen( cout ); // restore before reporting results

	expect( captured.str().find( "I can't load the Matrix" ) != string::npos,
		"engine refusal message was captured, not printed" );

	// --- Path 2: report methods taking an explicit ostream& --------------
	Matrix< double > A( 8, 2 );
	for ( unsigned r = 0; r < 8; r++ )
	{
		A( r, 0 ) = ( r < 4 ? 0 : 1 ); // outcome
		A( r, 1 ) = ( r < 4 ? 0.1 + 0.01 * r : 0.8 + 0.01 * r ); // guess
	}
	TwoSet t;
	t.setMatrix( A );
	t.setThreshold( 0.5 );

	ostringstream table;
	t.ClassTable( table );
	expect( table.str().find( "Predicted" ) != string::npos,
		"ClassTable report captured via ostream& parameter" );

	// --- ROC curve-point capture (Phase 3.3 port from the roc app) -------
	double area = t.getTrapROCarea();
	const vector< double >& rx = t.getROCx();
	const vector< double >& ry = t.getROCy();

	expect( area > 0.99, "cleanly separated data gives ROC area ~1" );
	expect( !rx.empty() && rx.size() == ry.size(),
		"getTrapROCarea captured matching ROC curve point vectors" );

	bool inRange = true;
	for ( unsigned i = 0; i < rx.size(); i++ )
		if ( rx[ i ] < 0 || rx[ i ] > 1 || ry[ i ] < 0 || ry[ i ] > 1 )
			inRange = false;
	expect( inRange, "all captured curve points lie in the unit square" );

	// --- Restoration ------------------------------------------------------
	expect( &util::screen() == static_cast< ostream* >( &cout ),
		"set_screen( cout ) restores the default stream" );

	// --- Per-thread redirection (the async-training contract) -------------
	// Each engine thread must own its screen: a GUI request thread and the
	// training worker redirect independently, so one thread's capture can
	// never swallow another's output. screenPtr is thread_local; before
	// 2026-07-16 it was a process-global and this test failed.
	ostringstream capA, capB;

	thread ta( [ & ]
	{
		util::set_screen( capA );
		util::screen() << "thread A speaking" << endl;
	} );
	ta.join();

	thread tb( [ & ]
	{
		util::set_screen( capB );
		util::screen() << "thread B speaking" << endl;
	} );
	tb.join();

	expect( capA.str().find( "thread A speaking" ) != string::npos
		&& capA.str().find( "thread B" ) == string::npos,
		"thread A captured only its own output" );
	expect( capB.str().find( "thread B speaking" ) != string::npos
		&& capB.str().find( "thread A" ) == string::npos,
		"thread B captured only its own output" );

	// A worker's redirection must not leak into this (main) thread
	expect( &util::screen() == static_cast< ostream* >( &cout ),
		"redirection in other threads left the main thread's screen alone" );

	// And redirecting the main thread must not touch a live worker's screen
	ostringstream mainCap, workerCap;
	string workerSaw;
	util::set_screen( mainCap );
	thread tw( [ & ]
	{
		util::set_screen( workerCap );
		util::screen() << "worker line" << endl;
		workerSaw = workerCap.str();
	} );
	tw.join();
	util::set_screen( cout );

	expect( workerSaw.find( "worker line" ) != string::npos
		&& mainCap.str().empty(),
		"concurrent main-thread capture did not swallow the worker's output" );

	// The SCOPED guard that replaced every manual save/restore pair
	test_scoped_capture();
	test_thread_isolation();

	return failures == 0 ? 0 : 1;
}
