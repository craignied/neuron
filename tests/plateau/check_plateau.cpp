// check_plateau.cpp : the PlateauDetector early-stop signal (ROADMAP 2 Phase 3).
//
// Five synthetic error traces, each chosen to exercise one property of the
// two-window moving-average detector. Per standing rule 2 each assertion was
// watched to FAIL against a deliberately broken detector before being trusted
// (see the header of this file's git commit): a detector that ignores the
// improvement test fires on the two decaying traces; one that lets NaN into the
// windows never fires on divergence.
//
//   1. clean decay              -> must NOT fire (still improving)
//   2. decay then flat          -> must fire (improvement stalls)
//   3. flat sawtooth            -> must fire despite the teeth (windows average
//                                  them out; this is why there are two windows)
//   4. sawtooth on a slow decay -> must NOT fire (a real trend survives teeth)
//   5. divergence to NaN        -> must fire (non-finite errors strike; they are
//                                  kept out of the windows so they can't poison
//                                  the means -- the trap that hung the probes)

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

#include "plateau.h"

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

// Feed a whole trace through a fresh detector; report whether it ever fired.
bool firesOn( const vector< double >& trace, unsigned window = 10,
	double tol = 1e-3, unsigned patience = 3 )
{
	PlateauDetector det( window, tol, patience );
	for ( size_t i = 0; i < trace.size(); i++ )
		if ( det.update( trace[ i ] ) )
			return true;
	return false;
}

int main()
{
	const double nan = numeric_limits< double >::quiet_NaN();

	// 1. Clean exponential decay: window-to-window improvement is a constant
	//    ~18% (scale-invariant), always above tol -> never a strike.
	{
		vector< double > t;
		for ( int i = 0; i < 300; i++ )
			t.push_back( exp( -0.02 * i ) );
		expect( !firesOn( t ), "clean decay does not fire" );
	}

	// 2. Decay for 100 iterations, then pinned at a floor: once both windows
	//    sit on the floor, improvement is ~0 -> strikes -> fire.
	{
		vector< double > t;
		for ( int i = 0; i < 100; i++ )
			t.push_back( 0.05 + exp( -0.05 * i ) );
		for ( int i = 0; i < 100; i++ )
			t.push_back( 0.05 );
		expect( firesOn( t ), "decay then flat fires" );
	}

	// 3. Flat sawtooth: constant base, period-2 teeth. A window of 10 holds
	//    five highs and five lows, so both window means equal the base and the
	//    teeth vanish -> improvement ~0 -> fire. A naive point-to-point detector
	//    would see the teeth as improvement/regression and never settle.
	{
		vector< double > t;
		for ( int i = 0; i < 200; i++ )
			t.push_back( 0.10 + ( ( i % 2 ) ? 0.02 : -0.02 ) );
		expect( firesOn( t ), "flat sawtooth fires despite the teeth" );
	}

	// 4. The same teeth riding a genuine exponential decay: the teeth cancel in
	//    each window mean, leaving the ~18% decay signal -> no strike -> no fire.
	{
		vector< double > t;
		for ( int i = 0; i < 300; i++ )
			t.push_back( exp( -0.02 * i ) + ( ( i % 2 ) ? 0.02 : -0.02 ) );
		expect( !firesOn( t ), "sawtooth on a real decay does not fire" );
	}

	// 5. Divergence: a few finite errors, then NaN forever. Non-finite errors
	//    strike directly and never enter the windows, so patience (3) NaNs fire.
	{
		vector< double > t = { 0.5, 0.4, 0.3 };
		for ( int i = 0; i < 10; i++ )
			t.push_back( nan );
		expect( firesOn( t ), "divergence to NaN fires" );
	}

	// The NaN must not merely be swallowed: with patience 3, exactly the third
	//    consecutive NaN fires (not the first, not never).
	{
		PlateauDetector det( 10, 1e-3, 3 );
		bool f1 = det.update( nan );
		bool f2 = det.update( nan );
		bool f3 = det.update( nan );
		expect( !f1 && !f2 && f3, "three NaNs fire on the third, not before" );
	}

	if ( failures == 0 )
	{
		cout << "check_plateau: the plateau detector behaves as documented" << endl;
		return 0;
	}
	cerr << "check_plateau: FAILED" << endl;
	return 1;
}
