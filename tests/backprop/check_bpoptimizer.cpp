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
// WHAT IS ASSERTED -- and deliberately NOT asserted
//
// NO BIT-EXACT LITERALS. This test originally pinned macOS-captured hexadecimal
// doubles. They failed on Ubuntu and GCC and on Windows and MSVC while EVERY
// structural assertion passed on all three -- which is the correct result: the
// correction is real everywhere, and twenty iterations of a conjugate method
// are allowed to differ in their low bits between compilers. Pinning one
// machine's bytes as a universal contract was an overclaim, and a per-platform
// table of expected values would be the same overclaim written three times.
// The measured macOS before/after values live in docs/refactor_audit.md as the
// sabotage evidence they are.
//
// What is asserted instead is portable and is about the CODE, not the machine:
//
//   1. THE INVARIANT. Canonical batch training reaches the same weights whether
//      or not gradient stopping is armed -- even though the two run through
//      DIFFERENT branches. With it off, the canonical branch updates from
//      WeightsAccumulate directly; with it on, the separate-gradient branch
//      runs, engine() dispatches to a switch with no case 0 and changes
//      nothing, and the update comes from Gradient. The corrected branch must
//      therefore agree with the canonical one, to within floating-point
//      reassociation: the two orders are ( a * eta ) / n and ( a / n ) * eta.
//      This is a stronger statement than "bit-identical to a capture" because
//      it is a property of the code rather than of one machine, and it fails
//      if the branch drops the average, double-applies eta, or updates the
//      wrong structure.
//
//   2. THE CORRECTION. Batch CGD and Shanno are SEPARATED from canonical and
//      from each other, by a margin far larger than rounding noise, with
//      automatic step size off and on. Under the defect all three were equal.
//
//   3. Every arm is finite, and the on-line arms remain discriminating.
//
//   4. The announced algorithm agrees with the computation -- asserted
//      together, never the name alone, which is the mistake that let the
//      defect ship.
//
// THE SEPARATION THRESHOLD is 0.01, chosen from measurement and not to make CI
// green: the smallest real gap observed is 0.3706 (canonical vs Shanno, batch),
// so the threshold sits ~37x below the true signal, while double rounding noise
// on a sum of magnitude ~23 is around 1e-14. Actual values print in hexadecimal
// on failure, so cross-platform drift stays diagnosable.
//
// NOT first-iteration divergence: an optimizer's first direction is legitimately
// the raw gradient (Golden's step 1 sets f(0) = -g(0)), so equality there proves
// nothing. Every comparison is taken at iteration 20.
//
// ON-LINE CGD AND SHANNO ARE DEGENERATE, and that is a fact about the
// algorithms rather than a flaw in this fixture: conjugate methods assume a
// TRUE BATCH GRADIENT, which is why autoalgo::pick forces batch/epoch for both.
// Driven on-line at any step size tried they either collapse to zero or
// saturate every output. They are smoke here -- they must run and stay finite --
// and only the on-line canonical arm is a quality control.
//
// SABOTAGE: restore `Weights[ m ] -= ( WeightsAccumulate[ m ] *= eta );` in the
// batch separate-gradient branch. The separation assertions fail; the invariant
// and the on-line assertions still pass, which is what makes them invariants.

#include <cmath>
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
	// forward() takes the MODEL's own input matrix -- Model::Train, which each
	//    model built for itself in setDataSet, with a bias column appended where
	//    its architecture has one. It is NOT the DataSet's training matrix: that
	//    one still carries the outcome in its last column, so passing it read the
	//    LABEL into the bias slot of every biased model, and for BareProp -- whose
	//    input vector is narrower still -- it read a short row. Both were silent
	//    until Matrix::row acquired its contract (D9); the same defect was found
	//    and fixed in check_onehidden a day earlier by an assert-enabled build.
	double signature()
	{
		double sum = 0;
		for ( unsigned r = 0; r < Train.rows(); r++ )
		{
			forward( Train, r );
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
static const double ONLINE_ETA = 0.05;

static unsigned trainingRows() { return 120 - 36; }

static double run( bool batch, unsigned type, bool autoStep,
	string* headerOut = 0, bool gradStop = false )
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
	b.setEta( batch ? 0.5 : ONLINE_ETA );
	// Gradient stopping is off for every arm except the invariant's second
	//    run, whose whole point is to reach the separate-gradient branch. The
	//    limit is 0 so the rule can never actually fire and cut a run short --
	//    only the BRANCH changes, never the iteration count.
	b.setGradStop( gradStop );
	b.setGradMaxLimit( 0.0 );
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

// --- the numerical vocabulary of this test --------------------------------

// Comfortably above double rounding noise (~1e-14 at magnitude 23) and far
// below the smallest real separation observed (0.3706). See the header.
static const double SEPARATION = 0.01;

// Reassociation tolerance for the invariant: ( a * eta ) / n versus
// ( a / n ) * eta over 20 iterations. Tight enough that a dropped average or a
// doubled eta could never slip through -- those move the result by whole units.
static const double REASSOC = 1e-9;

static void expectFinite( double v, const string& who )
{
	bool ok = std::isfinite( v );
	if ( !ok )
		cout << "         " << who << " = " << v << endl;
	expect( ok, who + " is finite" );
}

static void expectSeparated( double a, double b, const string& who )
{
	double gap = fabs( a - b );
	bool ok = gap > SEPARATION;
	if ( !ok )
		printf( "         %a vs %a  (gap %a = %.17g, need > %g)\n",
			a, b, gap, gap, SEPARATION );
	expect( ok, who );
}

static void expectAgrees( double a, double b, const string& who )
{
	double scale = fabs( a ) > 1 ? fabs( a ) : 1.0;
	bool ok = fabs( a - b ) / scale < REASSOC;
	if ( !ok )
		printf( "         %a vs %a  (relative %.3g, need < %g)\n",
			a, b, fabs( a - b ) / scale, REASSOC );
	expect( ok, who );
}

// --- 1. THE INVARIANT: the corrected branch agrees with the canonical one ---
//
// Canonical training with gradient stopping ARMED takes the separate-gradient
// branch -- the one that was wrong -- but engine( 0, ... ) dispatches to a
// switch with no case 0, so no optimizer transforms the gradient and the update
// must land in the same place as the canonical branch's own. That is the
// portable form of "the correction did not change what it was not supposed to".

static void test_invariant()
{
	cout << "-- the corrected branch agrees with canonical when no optimizer runs --" << endl;

	double plain = run( true, 0, false );              // canonical branch
	double viaGradientBranch = run( true, 0, false, 0, true ); // separate-gradient branch

	expectFinite( plain, "canonical batch" );
	expectFinite( viaGradientBranch, "canonical batch, gradient stopping armed" );
	expectAgrees( plain, viaGradientBranch,
		"the separate-gradient branch computes the canonical update" );
}

// --- 2. THE CORRECTION: batch optimizers reach the weights -----------------

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

		expectFinite( canon,  "batch canonical" + tag );
		expectFinite( cgd,    "batch CGD" + tag );
		expectFinite( shanno, "batch Shanno" + tag );

		expectSeparated( cgd, canon, "batch CGD differs from canonical" + tag );
		expectSeparated( shanno, canon, "batch Shanno differs from canonical" + tag );
		expectSeparated( cgd, shanno, "batch CGD differs from Shanno" + tag );
	}
}

// --- 3. the on-line arms -----------------------------------------------------
//
// The correction touched only the batch separate-gradient branch, so on-line is
// untouched by construction. These are smoke: they must run and stay finite.
// Only the CANONICAL arm is a quality control -- on-line CGD and Shanno are
// ill-posed (see the header) and legitimately collapse or saturate.

static void test_online_still_runs()
{
	cout << "-- on-line arms --" << endl;

	double canon  = run( false, 0, false );
	double cgd    = run( false, 1, false );
	double shanno = run( false, 2, false );

	expectFinite( canon,  "on-line canonical" );
	expectFinite( cgd,    "on-line CGD" );
	expectFinite( shanno, "on-line Shanno" );

	// The canonical arm is a real training run: it must have moved off the
	// initial state without collapsing to nothing or saturating everything.
	double rows = ( double ) trainingRows();
	expect( canon > SEPARATION && canon < rows - SEPARATION,
		"on-line canonical is a healthy run, neither collapsed nor saturated" );

	// The three remain mutually distinguishable, which is what makes them
	// usable as a control at all.
	expectSeparated( canon, cgd, "on-line canonical and CGD are distinguishable" );
	expectSeparated( canon, shanno, "on-line canonical and Shanno are distinguishable" );
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

// Print every arm in full precision. NOT a source of expected literals any
// more -- these are machine-specific (see the header) and exist so that
// cross-platform drift can be inspected when a separation assertion fails.
static void capture()
{
	printf( "batch    canonical         %a  %.17g\n", run( true, 0, false ), run( true, 0, false ) );
	printf( "batch    canonical+gradstop%a  %.17g\n", run( true, 0, false, 0, true ), run( true, 0, false, 0, true ) );
	printf( "batch    CGD               %a  %.17g\n", run( true, 1, false ), run( true, 1, false ) );
	printf( "batch    Shanno            %a  %.17g\n", run( true, 2, false ), run( true, 2, false ) );
	printf( "batch    CGD    auto       %a  %.17g\n", run( true, 1, true ), run( true, 1, true ) );
	printf( "batch    Shanno auto       %a  %.17g\n", run( true, 2, true ), run( true, 2, true ) );
	printf( "on-line  canonical         %a  %.17g\n", run( false, 0, false ), run( false, 0, false ) );
	printf( "on-line  CGD               %a  %.17g\n", run( false, 1, false ), run( false, 1, false ) );
	printf( "on-line  Shanno            %a  %.17g\n", run( false, 2, false ), run( false, 2, false ) );
}

int main( int argc, char* argv[] )
{
	if ( argc > 1 && string( argv[ 1 ] ) == "--capture" )
	{
		capture();
		return 0;
	}

	test_invariant();
	test_batch_optimizers_discriminate();
	test_online_still_runs();
	test_name_matches_computation();

	cout << endl << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
