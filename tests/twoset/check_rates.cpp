// check_rates.cpp : the classification rates, and their zero denominators.
//
// TwoSet's four rate accessors computed the ratio and THEN asked whether the
// denominator was zero:
//
//     result = static_cast< double >( tp ) / ( tp + fn );
//     if ( ( tp + fn ) == 0 )   // check division by zero
//         throw DivisionByZero();
//
// The throw is correct and reachable, so nothing observable was wrong -- but
// the division is performed first, on a denominator already known to be zero.
// In IEEE arithmetic that yields an infinity or a NaN rather than a trap, which
// is precisely why it survived: the guard reads as protection while running
// after the thing it protects against.
//
// The order is now denominator-first, through a small checkedRate() helper. The
// four formulae stay written out at their call sites -- sensitivity is
// tp / ( tp + fn ) and must still say so.
//
// WHAT THIS ASSERTS
//   1. Each of the four zero-denominator cases throws DivisionByZero:
//        no real positives   -> sensitivity   ( tp + fn == 0 )
//        no real negatives   -> specificity   ( tn + fp == 0 )
//        no predicted +ves   -> PVP           ( tp + fp == 0 )
//        no predicted -ves   -> PVN           ( tn + fn == 0 )
//   2. Ordinary ratios are exact, against a hand-built confusion matrix.
//   3. The no-threshold contract is unchanged: a message through util::screen()
//      and a return of 0, NOT a throw.
//
// ON PROVING THE ORDER. A regression test cannot prove source ordering here: a
// double divided by zero does not fault, and an optimizer is free to sink the
// division past the throw. The ordering was demonstrated separately with a
// -fsanitize=float-divide-by-zero build, which reports the division on the old
// code and is silent on the new; that proof is recorded in the commit message.
// Manufacturing a permanent assertion out of the floating-point environment
// would be fragile across platforms and would test the compiler, not the engine.

#include <iostream>
#include <string>
#include <vector>

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

// Build a TwoSet from (real, guess) pairs
static TwoSet make( const vector< double >& real, const vector< double >& guess )
{
	Matrix< double > m( ( unsigned ) real.size(), 2 );
	for ( unsigned i = 0; i < real.size(); i++ )
	{
		m( i, 0 ) = real[ i ];
		m( i, 1 ) = guess[ i ];
	}
	TwoSet t;
	t.setMatrix( m );
	t.setThreshold( 0.5 );
	return t;
}

// --- 1. the four zero denominators ----------------------------------------

static void test_zero_denominators()
{
	cout << "-- zero denominators throw --" << endl;

	// No real positives: tp + fn == 0, so sensitivity is undefined
	{
		vector< double > real, guess;
		real.push_back( 0 ); guess.push_back( 0.9 );
		real.push_back( 0 ); guess.push_back( 0.1 );
		TwoSet t = make( real, guess );
		bool threw = false;
		try { t.getSens(); } catch ( TwoSet::DivisionByZero& ) { threw = true; }
		expect( threw, "no real positives: sensitivity throws" );
	}

	// No real negatives: tn + fp == 0, so specificity is undefined
	{
		vector< double > real, guess;
		real.push_back( 1 ); guess.push_back( 0.9 );
		real.push_back( 1 ); guess.push_back( 0.1 );
		TwoSet t = make( real, guess );
		bool threw = false;
		try { t.getSpec(); } catch ( TwoSet::DivisionByZero& ) { threw = true; }
		expect( threw, "no real negatives: specificity throws" );
	}

	// Nothing predicted positive: tp + fp == 0, so PVP is undefined
	{
		vector< double > real, guess;
		real.push_back( 1 ); guess.push_back( 0.1 );
		real.push_back( 0 ); guess.push_back( 0.2 );
		TwoSet t = make( real, guess );
		bool threw = false;
		try { t.getPVP(); } catch ( TwoSet::DivisionByZero& ) { threw = true; }
		expect( threw, "nothing predicted positive: PVP throws" );
	}

	// Nothing predicted negative: tn + fn == 0, so PVN is undefined
	{
		vector< double > real, guess;
		real.push_back( 1 ); guess.push_back( 0.9 );
		real.push_back( 0 ); guess.push_back( 0.8 );
		TwoSet t = make( real, guess );
		bool threw = false;
		try { t.getPVN(); } catch ( TwoSet::DivisionByZero& ) { threw = true; }
		expect( threw, "nothing predicted negative: PVN throws" );
	}
}

// --- 2. ordinary ratios, against a hand-built confusion matrix -------------

static void test_ordinary_rates()
{
	cout << "-- ordinary ratios --" << endl;

	// Constructed to give tp = 3, fn = 1, tn = 4, fp = 2
	vector< double > real, guess;
	for ( int i = 0; i < 3; i++ ) { real.push_back( 1 ); guess.push_back( 0.9 ); } // tp
	real.push_back( 1 ); guess.push_back( 0.1 );                                   // fn
	for ( int i = 0; i < 4; i++ ) { real.push_back( 0 ); guess.push_back( 0.1 ); } // tn
	for ( int i = 0; i < 2; i++ ) { real.push_back( 0 ); guess.push_back( 0.9 ); } // fp

	TwoSet t = make( real, guess );

	expect( t.getTP() == 3 && t.getFN() == 1 && t.getTN() == 4 && t.getFP() == 2,
		"the confusion matrix is as constructed" );
	expect( t.getSens() == 3.0 / 4.0, "sensitivity = tp / ( tp + fn )" );
	expect( t.getSpec() == 4.0 / 6.0, "specificity = tn / ( tn + fp )" );
	expect( t.getPVP() == 3.0 / 5.0, "PVP = tp / ( tp + fp )" );
	expect( t.getPVN() == 4.0 / 5.0, "PVN = tn / ( tn + fn )" );
	expect( t.getClassAcc() == 7.0 / 10.0, "accuracy = ( tp + tn ) / n" );
}

// --- 3. the no-threshold contract is unchanged ----------------------------

static void test_no_threshold_contract()
{
	cout << "-- with no threshold set --" << endl;

	Matrix< double > m( 2, 2 );
	m( 0, 0 ) = 1; m( 0, 1 ) = 0.9;
	m( 1, 0 ) = 0; m( 1, 1 ) = 0.1;
	TwoSet t;
	t.setMatrix( m ); // deliberately NO setThreshold

	string said;
	double sens = 0, spec = 0, pvp = 0, pvn = 0;
	{
		util::ScreenCapture quiet;
		sens = t.getSens();
		spec = t.getSpec();
		pvp = t.getPVP();
		pvn = t.getPVN();
		said = quiet.text();
	}

	expect( sens == 0 && spec == 0 && pvp == 0 && pvn == 0,
		"each rate returns 0 rather than throwing" );
	expect( said.find( "Sensitivity cannot be calculated" ) != string::npos
		&& said.find( "Specificity cannot be calculated" ) != string::npos
		&& said.find( "PVP cannot be calculated" ) != string::npos
		&& said.find( "PVN cannot be calculated" ) != string::npos,
		"and each says so through the engine's screen" );
}

int main()
{
	test_zero_denominators();
	test_ordinary_rates();
	test_no_threshold_contract();

	cout << endl << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
