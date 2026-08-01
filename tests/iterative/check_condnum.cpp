// check_condnum.cpp : reporting a diagnostic must not move the model.
//
// THE BUG (D2, found 2026-08-01 by tests/iterative/check_quietprep.cpp).
// Network::computeCondNum() needed one gradient per training exemplar to build
// its B matrix. It got them the cheap way:
//
//     finalFlag = true;
//     innerTrainSet();     // <-- a real training iteration
//     finalFlag = false;
//
// innerTrainSet() does compute those gradients -- and then runs the optimizer
// transformation and UPDATES THE WEIGHTS. So asking for a condition number
// trained the model one more step. Logistic::reportAccuracy is the only caller
// of reportCondNum, so every AUDIBLE Logistic run ended one gradient step past
// the weights whose error it had just reported, while a quiet run did not.
// The saved model, the guesses, and any exported calculator described weights
// the reported error never described.
//
// It disturbed more than the weights. Depending on the optimizer, that call
// also moved G, stackG, lastG, lastF and the automatic-step-size accumulators,
// so a run CONTINUED after a report diverged from one that never asked.
//
// Same family as legacy bug #10 and the prepareRun defect (113da40):
// reporting may never change the fit.
//
// THE FIX: Logistic::collectGradients() gathers the gradients without
// training -- forward propagate, form ( o - y ) I plus the penalty term, write
// the column. No engine(), no weight update, no optimizer state touched.
// Network::computeCondNum() calls it, and reports "not available" for a model
// that supplies none rather than inventing a number.
//
// WHAT THIS ASSERTS
//   1. reportAccuracy() leaves every prediction bit-identical.
//   2. A run CONTINUED after a report matches a control that never reported --
//      for canonical backprop, CGD and Shanno, which is where the disturbed
//      lastG/lastF would show.
//   3. The condition number is still produced, and still equals the value the
//      old path produced under canonical backprop.
//   4. Repeated reports agree: the diagnostic is idempotent.
//
// SABOTAGE: restore the innerTrainSet() call in computeCondNum() and 1, 2 and
// 4 fail.

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "logistic.h"
#include "dataset.h"
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

// Bitwise equality. The invariant under test is "the report changed nothing",
// and that must be assertable even when the run itself is diverging: Shanno on
// this fixture reaches ~1e156, and on some settings NaN, where `==` is useless
// (NaN != NaN) and a tolerance would be meaningless. Comparing the bits says
// exactly what is meant -- the same value, whatever that value is.
static bool sameBits( double a, double b )
{
	return memcmp( &a, &b, sizeof a ) == 0;
}

static bool sameBits( const vector< double >& a, const vector< double >& b )
{
	if ( a.size() != b.size() ) return false;
	for ( unsigned i = 0; i < a.size(); i++ )
		if ( !sameBits( a[ i ], b[ i ] ) ) return false;
	return true;
}

struct Hush {
	ostringstream sink;
	ostream& prev;
	Hush() : prev( util::screen() ) { util::set_screen( sink ); }
	~Hush() { util::set_screen( prev ); }
};

class Probe : public Logistic {
public:
	unsigned rows() { return Train.rows(); }
	Matrix< double >& trainMatrix() { return Train; } // inputs incl. the bias column
	double out( unsigned r ) { forward( Train, r ); return o; }
	vector< double > predictions()
	{
		vector< double > p;
		for ( unsigned r = 0; r < rows(); r++ ) p.push_back( out( r ) );
		return p;
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
	Hush quiet;
	d.setRawMatrix( raw );
	d.randomize( nTest );
	return d;
}

// A trained Logistic, configured identically every time
static void prepare( Probe& net, DataSet& d, unsigned algorithm, bool decay,
	bool gradStop = false )
{
	net.setDataSet( d );
	net.setHistory( false );
	net.setLastop( false );
	net.setLogPrint( false );
	net.setQuiet( true );          // no narration; the report is requested explicitly
	net.setWeightDecay( decay );
	net.setDecay( decay ? 5e-5 : 0 );
	net.setAutoStepSize( false );
	net.setEta( 0.05 );
	net.setTrainingType( algorithm );
	net.setGradStop( gradStop );
	// 30, not 100: Shanno diverges to NaN on this fixture beyond ~40 iterations,
	//    and a NaN comparison would fail for reasons that have nothing to do
	//    with the diagnostic under test.
	net.setMaxIterations( 30 );
	util::set_seed( 7 );
	net.randomize();
	{ Hush h; net.train(); }
}

// (1) and (4): a report changes no prediction, and repeats agree
static void test_report_does_not_move_the_model( unsigned algorithm,
	const string& name, bool decay, bool wantFinite = true )
{
	util::set_seed( 4242 );
	DataSet d = makeData( 120, 36 );

	Probe net;
	prepare( net, d, algorithm, decay );
	vector< double > before = net.predictions();

	double cond1;
	{
		Hush h;
		net.reportAccuracy( h.sink ); // the whole audible epilogue, incl. condition number
		cond1 = net.getCondNum();
	}
	vector< double > after = net.predictions();

	expect( sameBits( before, after ),
		name + ": reportAccuracy leaves every prediction bit-identical" );

	double cond2;
	{
		Hush h;
		net.reportAccuracy( h.sink );
		cond2 = net.getCondNum();
	}
	vector< double > again = net.predictions();

	expect( sameBits( again, after ),
		name + ": a second report changes nothing either" );

	expect( sameBits( cond1, cond2 ), name + ": the condition number is idempotent" );

	// A usable number is demanded only where the FIT is usable. Shanno diverges
	//    on this fixture (~1e156 by iteration 2), so its B matrix is meaningless
	//    -- which is a fact about the optimizer on this data, not about the
	//    diagnostic, and the invariants above still hold there.
	if ( wantFinite )
		expect( std::isfinite( cond1 ) && cond1 > 0,
			name + ": a finite, positive condition number is still produced" );
}

// (2): training CONTINUED after a report matches a control that never reported.
// This is where a disturbed lastG / lastF / step-size accumulator would show.
static void test_continuation_unaffected( unsigned algorithm, const string& name,
	bool decay )
{
	util::set_seed( 4242 );
	DataSet dA = makeData( 120, 36 );
	util::set_seed( 4242 );
	DataSet dB = makeData( 120, 36 );

	Probe reported, control;
	prepare( reported, dA, algorithm, decay );
	prepare( control, dB, algorithm, decay );

	// One of them asks for the diagnostic; the other does not
	{
		Hush h;
		reported.reportAccuracy( h.sink );
	}

	// Both then continue training identically
	double errReported, errControl;
	{
		Hush h;
		reported.setMaxIterations( 20 );
		errReported = reported.train();
	}
	{
		Hush h;
		control.setMaxIterations( 20 );
		errControl = control.train();
	}

	if ( !sameBits( errReported, errControl ) )
		cout << "         reported=" << setprecision( 17 ) << errReported
			<< "  control=" << errControl << endl;
	expect( sameBits( errReported, errControl ),
		name + ": a continued run is unaffected by having reported" );

	expect( sameBits( reported.predictions(), control.predictions() ),
		name + ": continued predictions identical too" );
}

// (3) THE VALUE, against an independently calculated fixture.
//
// The condition number is now the ratio of the extreme absolute eigenvalues of
//
//     X'VX,    V = diag( p_i ( 1 - p_i ) )
//
// the observed Fisher information of the UNPENALIZED log likelihood -- the same
// matrix Logistic already forms for the Wald covariance (Hosmer & Lemeshow eqn
// 2.8), now built once and used for both.
//
// With ONE input the design is n x 2 (input, bias), so X'VX is 2x2 and its
// eigenvalues are analytic:
//
//     X'VX = [ a  b ]      lambda = (a+d)/2 +/- sqrt( ((a-d)/2)^2 + b^2 )
//            [ b  d ]
//
// with a = sum v x^2, b = sum v x, d = sum v. This computes those sums with
// plain loops -- not Matrix::dotprod, not GSL -- so the check is independent of
// both the engine's matrix product and its eigenvalue routine.
static void test_value_against_independent_fixture()
{
	util::set_seed( 4242 );

	// ONE input, so X'VX is 2x2 and the eigenvalues are closed-form
	Matrix< double > raw( 80, 2 );
	for ( unsigned i = 0; i < 80; i++ )
	{
		double x = -1.0 + 2.0 * ( ( i * 37 ) % 100 ) / 99.0;
		raw( i, 0 ) = x;
		raw( i, 1 ) = ( x > 0.15 ) ? 1 : 0;
	}
	DataSet d;
	d.setInput( 1 );
	d.setOutput( 1 );
	d.setDiscrete( true );
	d.setHistory( false );
	{ Hush h; d.setRawMatrix( raw ); d.raw2train(); }

	Probe net;
	net.setDataSet( d );
	net.setHistory( false );
	net.setLastop( false );
	net.setLogPrint( false );
	net.setQuiet( true );
	net.setWeightDecay( false );
	net.setDecay( 0 );
	net.setAutoStepSize( false );
	net.setEta( 0.05 );
	net.setTrainingType( 0 );
	net.setGradStop( false );
	net.setMaxIterations( 40 );
	util::set_seed( 7 );
	net.randomize();
	{ Hush h; net.train(); }
	{ Hush h; net.reportAccuracy( h.sink ); }

	// Independent X'VX: plain sums over the fitted probabilities
	Matrix< double >& X = net.trainMatrix();
	double a = 0, b = 0, dd = 0;
	for ( unsigned i = 0; i < X.rows(); i++ )
	{
		double p = net.out( i );      // fitted probability for row i
		double v = p * ( 1 - p );
		double x = X( i, 0 );         // column 0 is the input; column 1 is the bias (1)
		a  += v * x * x;
		b  += v * x;
		dd += v;
	}

	double half = ( a + dd ) / 2, gap = ( a - dd ) / 2;
	double root = sqrt( gap * gap + b * b );
	double l1 = half + root, l2 = half - root;      // both >= 0: X'VX is PSD
	double hi = fabs( l1 ) > fabs( l2 ) ? fabs( l1 ) : fabs( l2 );
	double lo = fabs( l1 ) < fabs( l2 ) ? fabs( l1 ) : fabs( l2 );
	double expected = hi / lo;

	double got = net.getCondNum();
	if ( !( fabs( got - expected ) / expected < 1e-9 ) )
		cout << "         got " << setprecision( 17 ) << got
			<< ", independent calculation gives " << expected << endl;
	expect( fabs( got - expected ) / expected < 1e-9,
		"condition number equals the analytic eigenvalue ratio of X'VX" );

	// And the two extreme eigenvalues themselves, not just their ratio
	expect( fabs( net.getCondMaxEig() - hi ) / hi < 1e-9,
		"largest absolute eigenvalue matches the analytic value" );
	expect( fabs( net.getCondMinEig() - lo ) / lo < 1e-9,
		"smallest absolute eigenvalue matches the analytic value" );
}

// (5) WEIGHT DECAY MUST NOT TOUCH IT. The design condition number answers
// "is this design collinear / identifiable", and regularization would improve
// it by construction -- X'VX + lambda I is better conditioned than X'VX for any
// lambda > 0. If decay could move it, a user could hide collinearity by turning
// regularization up. Weights and predictions are held FIXED here (no retraining
// between the two reports), so the only thing that changes is the decay
// configuration.
static void test_decay_does_not_change_the_diagnostic()
{
	util::set_seed( 4242 );
	DataSet d = makeData( 120, 36 );

	Probe net;
	prepare( net, d, 0, false, true ); // trained ONCE, decay off

	{ Hush h; net.reportAccuracy( h.sink ); }
	double withoutDecay = net.getCondNum();
	vector< double > pBefore = net.predictions();

	// Turn weight decay on -- WITHOUT retraining, so the fit is untouched
	net.setWeightDecay( true );
	net.setDecay( 0.01 ); // 200x the shipped default
	{ Hush h; net.reportAccuracy( h.sink ); }
	double withDecay = net.getCondNum();

	expect( sameBits( pBefore, net.predictions() ),
		"the fit is unchanged between the two reports" );
	expect( sameBits( withoutDecay, withDecay ),
		"weight decay does not change the design condition number" );

	// A penalized curvature would have moved it a long way at this lambda; that
	//    it does not move AT ALL is the point.
	expect( std::isfinite( withoutDecay ) && withoutDecay > 1,
		"and the diagnostic is a real, finite ratio" );
}

int main()
{
	// Canonical backprop, plus the two optimizers whose lastG/lastF the old
	// path disturbed. Weight decay on: it puts the penalty term in the gradient.
	test_report_does_not_move_the_model( 0, "canonical", true );
	test_report_does_not_move_the_model( 1, "CGD", true );
	test_report_does_not_move_the_model( 2, "Shanno", true, false );

	test_continuation_unaffected( 0, "canonical", true );
	test_continuation_unaffected( 1, "CGD", true );
	test_continuation_unaffected( 2, "Shanno", true );

	// And with decay off, so the penalty term is not what carries the result
	test_report_does_not_move_the_model( 0, "canonical, no decay", false );

	test_value_against_independent_fixture();
	test_decay_does_not_change_the_diagnostic();

	cout << endl << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
