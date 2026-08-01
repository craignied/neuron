// check_bpoptimizer.cpp : the chosen optimizer must reach BackProp's weights.
//
// LEGACY BUG #12. In batch/epoch mode BackProp::innerTrainSet() computed the
// conjugate direction and then threw it away (backprop.cpp, before 2026-08-01):
//
//     Gradient[ l ] = WeightsAccumulate[ l ] /= nTrain;   // a Matrix COPY
//     engine( trainingType, iteration );                  // writes Gradient
//     Weights[ m ] -= ( WeightsAccumulate[ m ] *= eta );  // reads the OTHER one
//
// Network::CGD and Network::shanno pack() from Gradient, compute the search
// direction, and unpack() back into Gradient. The weight update then read
// WeightsAccumulate, which the optimizer never touches -- so batch BackProp
// under CGD or Shanno was plain gradient descent, wearing the other algorithm's
// name in its own run header. SimpleProp and BareProp update from oG/hG after
// engine() and were always correct; on-line BackProp updates from Gradient and
// was always correct. The defect is exactly
//
//     BackProp  x  batch/epoch  x  ( CGD or Shanno ).
//
// It was legacy (../distro/src/backprop.cpp:630-634) and reachable in
// production: autoalgo::pick forces setBatchEpoch( true ) for CGD and Shanno,
// so algorithm=auto on a BackProp compared three optimizers that were all
// secretly the same one.
//
// WHY NOTHING CAUGHT IT. tests/props' CGD/Shanno case carries expected values
// for SimpleProp and BareProp only, and no golden fixture uses BackProp at all.
// The optimizer tests executed the dispatch, passed, and guarded nothing for the
// one model where it did not work. This file is that guard.
//
// WHAT IS ASSERTED
//   1. INVARIANTS -- captured BEFORE the correction, and they must not move:
//      canonical batch, and all three on-line paths.
//   2. The correction -- batch CGD and Shanno now differ from canonical and
//      from each other, with automatic step size both off and on.
//   3. Exact post-correction values for batch CGD and Shanno.
//   4. The run header names the algorithm that actually ran.
//
// NOT first-iteration divergence: an optimizer's first direction is legitimately
// the raw gradient (Golden's step 1 sets f(0) = -g(0)), so equality there proves
// nothing. Every comparison is taken at iteration 20, where the histories must
// have diverged.
//
// SABOTAGE: restore `Weights[ m ] -= ( WeightsAccumulate[ m ] *= eta );` in the
// batch separate-gradient branch. Group 2 and 3 fail; group 1 still passes,
// which is what makes group 1 an invariant rather than a restatement.
//
// Run with --capture to print the literals in full precision. Re-capturing is
// not a fix for a failure: group 1 moving means canonical or on-line training
// changed, and group 3 moving means the optimizer changed.

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "backprop.h"
#include "dataset.h"
#include "utility.h"

using namespace std;

int failures = 0;

static void expect( bool ok, const string& what )
{
	if ( ok )
		cout << "ok - " << what << endl;
	else
	{
		cout << "FAIL - " << what << endl;
		failures++;
	}
}

// Reach the weights without disturbing them. The signature is the sum of the
// forward outputs over the training set: one number standing for the whole
// weight state, comparable bitwise.
class Probe : public BackProp {
public:
	double signature()
	{
		double sum = 0;
		Matrix< double >& m = theData.getTrainMatrix();
		for ( unsigned r = 0; r < m.rows(); r++ )
		{
			forward( m, r );
			sum += o;
		}
		return sum;
	}
};

static DataSet makeData( unsigned n, unsigned nTest )
{
	Matrix< double > raw( n, 3 );
	for ( unsigned i = 0; i < n; i++ )
	{
		double x0 = -1.0 + 2.0 * ( ( i * 37 ) % 100 ) / 99.0;
		double x1 = -1.0 + 2.0 * ( ( i * 53 ) % 100 ) / 99.0;
		raw( i, 0 ) = x0;
		raw( i, 1 ) = x1;
		raw( i, 2 ) = ( x0 + x1 > 0.55 ) ? 1 : 0;
	}
	DataSet d;
	d.setInput( 2 );
	d.setOutput( 1 );
	d.setDiscrete( true );
	d.setHistory( false );
	util::ScreenCapture hush;
	d.setRawMatrix( raw );
	d.randomize( nTest );
	return d;
}

// One run. Every arm starts from the SAME weights -- randomize() under a fixed
// seed after an identically constructed network -- so a difference in the result
// can only come from the training, not from the initialization.
static double run( bool batch, unsigned type, bool autoStep,
	string* headerOut = 0 )
{
	util::set_seed( 4242 );
	DataSet d = makeData( 120, 36 );

	Probe b;
	b.setDataSet( d );
	vector< unsigned > layers;
	layers.push_back( 4 );
	layers.push_back( 3 );          // TWO hidden layers: a real BackProp
	b.setHidden( layers );

	b.setHistory( false );
	b.setLastop( false );
	b.setLogPrint( false );
	b.setXEerror();
	b.setWeightDecay( false );
	b.setDecay( 0 );
	b.setBatchEpoch( batch );
	b.setAutoStepSize( autoStep );
	// On-line makes one update per exemplar, so a batch-sized step saturates
	//    every unit and the arms all collapse to a signature of 0 -- bitwise
	//    stable, but a number that many kinds of breakage would also produce.
	//    A smaller on-line step keeps the three arms distinct and the
	//    invariants meaningful.
	b.setEta( batch ? 0.5 : 0.05 );
	b.setGradStop( false );         // no early exit: every arm runs the full 20
	b.setTrainingType( type );
	b.setMaxIterations( 20 );

	util::set_seed( 7 );
	b.randomize();

	// The run header is wanted for one case; everything else is silenced.
	b.setQuiet( headerOut == 0 );
	{
		util::ScreenCapture cap;
		b.train();
		if ( headerOut )
			*headerOut = cap.text();
	}
	return b.signature();
}

// --- 1. INVARIANTS: captured before the correction, must not move ----------

// Canonical batch does not go through the separate-gradient branch at all
// (trainingType 0 with gradient stopping off), so the correction cannot touch
// it. On-line BackProp already updated from Gradient, so the correction cannot
// touch that either. If any of these move, the fix reached further than it
// should have.
static const double CANON_BATCH   = 0x1.760ffbe7973c2p+4;
static const double ONLINE_CANON  = 0x1.225f05c606211p+5;
static const double ONLINE_CGD    = 0x1.617604c2720ddp+2;
static const double ONLINE_SHANNO = 0x1.5p+6;

static void test_invariants()
{
	cout << "-- unchanged by the correction --" << endl;

	expect( run( true, 0, false ) == CANON_BATCH,
		"canonical BATCH is bit-identical to before the fix" );
	expect( run( false, 0, false ) == ONLINE_CANON,
		"on-line canonical is bit-identical to before the fix" );
	expect( run( false, 1, false ) == ONLINE_CGD,
		"on-line CGD is bit-identical to before the fix" );
	expect( run( false, 2, false ) == ONLINE_SHANNO,
		"on-line Shanno is bit-identical to before the fix" );
}

// --- 2. The correction: batch optimizers now reach the weights ------------

static void test_batch_optimizers_discriminate()
{
	cout << "-- batch CGD and Shanno reach the weights --" << endl;

	for ( int autoStep = 0; autoStep <= 1; autoStep++ )
	{
		string tag = autoStep ? " (automatic step size ON)"
			: " (automatic step size off)";
		bool a = autoStep != 0;

		double canon  = run( true, 0, a );
		double cgd    = run( true, 1, a );
		double shanno = run( true, 2, a );

		expect( cgd != canon, "batch CGD differs from canonical" + tag );
		expect( shanno != canon, "batch Shanno differs from canonical" + tag );
		expect( cgd != shanno, "batch CGD differs from Shanno" + tag );
	}
}

// --- 3. Exact post-correction values --------------------------------------
//
// HOW THESE WERE OBTAINED. Captured with --capture from the CORRECTED engine on
// 2026-08-01, printed as C99 hexadecimal float so the literal is the double, not
// a decimal rendering of it. They are new expected values for a deliberate
// correctness change -- the arms they describe previously produced the canonical
// numbers, which was the bug. They are NOT re-blessed goldens: no golden
// transcript uses BackProp, and group 1 above pins everything the correction was
// not allowed to move.

static const double BATCH_CGD         = 0x1.50564fab14283p+4;
static const double BATCH_SHANNO      = 0x1.70220b4a8a31ap+4;
static const double BATCH_CGD_AUTO    = 0x1.a5cfd3a6729ffp+4;
static const double BATCH_SHANNO_AUTO = 0x1.6eebfea1aa607p+4;

static void test_exact_values()
{
	cout << "-- exact post-correction values --" << endl;

	expect( run( true, 1, false ) == BATCH_CGD, "batch CGD exact" );
	expect( run( true, 2, false ) == BATCH_SHANNO, "batch Shanno exact" );
	expect( run( true, 1, true ) == BATCH_CGD_AUTO,
		"batch CGD exact, automatic step size" );
	expect( run( true, 2, true ) == BATCH_SHANNO_AUTO,
		"batch Shanno exact, automatic step size" );
}

// --- 4. The name and the computation agree --------------------------------
//
// The defect's signature was a run header announcing an algorithm that had no
// effect on the weights. Asserting the header alone would reproduce exactly the
// mistake that let this ship, so each case asserts the header AND that the
// weights differ from canonical.

static void test_name_matches_computation()
{
	cout << "-- the announced algorithm is the one that ran --" << endl;

	struct Case { unsigned type; const char* says; };
	const Case cases[] = {
		{ 0, "canonical backpropagation" },
		{ 1, "conjugate gradient descent" },
		{ 2, "Shanno" }
	};

	double canon = run( true, 0, false );

	for ( const Case& c : cases )
	{
		string header;
		double got = run( true, c.type, false, &header );

		expect( header.find( c.says ) != string::npos,
			string( "the run header announces " ) + c.says );

		if ( c.type != 0 )
			expect( got != canon,
				string( "...and " ) + c.says
					+ " actually moved the weights differently" );
	}
}

// --- capture mode ----------------------------------------------------------

static void capture()
{
	cout << setprecision( 17 );
	printf( "// invariants (canonical batch, and all three on-line)\n" );
	printf( "static const double CANON_BATCH   = %a;\n", run( true, 0, false ) );
	printf( "static const double ONLINE_CANON  = %a;\n", run( false, 0, false ) );
	printf( "static const double ONLINE_CGD    = %a;\n", run( false, 1, false ) );
	printf( "static const double ONLINE_SHANNO = %a;\n", run( false, 2, false ) );
	printf( "// post-correction batch optimizer values\n" );
	printf( "static const double BATCH_CGD         = %a;\n", run( true, 1, false ) );
	printf( "static const double BATCH_SHANNO      = %a;\n", run( true, 2, false ) );
	printf( "static const double BATCH_CGD_AUTO    = %a;\n", run( true, 1, true ) );
	printf( "static const double BATCH_SHANNO_AUTO = %a;\n", run( true, 2, true ) );
}

int main( int argc, char* argv[] )
{
	if ( argc > 1 && string( argv[ 1 ] ) == "--capture" )
	{
		capture();
		return 0;
	}

	test_invariants();
	test_batch_optimizers_discriminate();
	test_exact_values();
	test_name_matches_computation();

	cout << endl << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
