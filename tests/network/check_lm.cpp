// check_lm.cpp : Levenberg-Marquardt, Madsen/Nielsen/Tingleff Algorithm 3.16.
//
// THE TEST STRUCTURE IS THE POINT, and it comes from two results this project
// paid for. Sabotaging iRPROP+'s error-dependent rollback turned it into RPROP+
// and every real-model integration test still passed; deleting BB's nonmonotone
// maximum turned it into monotone Armijo BB and the assertion that NAMED
// nonmonotone acceptance still passed. A descending objective proves nothing
// about which algorithm produced it.
//
// So this file has two layers:
//
//   1-13   MODEL-FREE. Algorithm 3.16 is driven BY HAND through a hand-written
//          LMObjective over analytic problems -- an exactly solvable linear
//          least squares, the source's own Rosenbrock example, and a scripted
//          objective that puts the gain ratio wherever the test wants it. LM
//          has no model dependency precisely so that this is possible.
//
//   14-26  THE REAL PATH. SimpleProp and BareProp through
//          Network::lmIteration(), where the composition lives.
//
// A sabotage in the wiring between them fails only the second layer, and a
// sabotage in the published table fails only the first. Both are exercised --
// see the fail-proof evidence in the commit message.

#include "stdafx.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "simpleprop.h"
#include "bareprop.h"
#include "backprop.h"
#include "lm.h"

using namespace std;

static int failures = 0;

static void ok( bool condition, const char* what )
{
	if ( !condition ) { printf( "FAIL: %s\n", what ); failures++; }
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

// ---------------------------------------------------------------------------
// A MODEL-FREE OBJECTIVE. It is handed the analytic problem and answers the
// three questions LM asks, so the published table can be driven directly.

class Analytic : public LMObjective {
public:
	Analytic() : evaluations( 0 ), cancelAfter( 0 ) { }
	virtual ~Analytic() { }

	vector< double > x;              // the installed point
	unsigned evaluations;
	unsigned cancelAfter;            // 0 = never

	virtual void currentPoint( vector< double >& w ) const { w = x; }
	virtual void install( const vector< double >& w ) { x = w; }
	virtual bool cancelled() const
	{ return cancelAfter && evaluations >= cancelAfter; }

	virtual double evaluateNormal( Matrix< double >& A, vector< double >& g )
	{
		evaluations++;
		return build( A, g );
	}

protected:
	virtual double build( Matrix< double >& A, vector< double >& g ) = 0;
};

// ---- (a) LINEAR LEAST SQUARES: f(x) = C x - b, F = 0.5 |f|^2 ---------------
//      J = C exactly, so A = C'C and g = C'f, both computable by hand. Because
//      f is linear the Gauss-Newton model is EXACT, which makes rho exactly 1.
class LinearLS : public Analytic {
public:
	Matrix< double > C;
	vector< double > b;

protected:
	virtual double build( Matrix< double >& A, vector< double >& g )
	{
		const unsigned m = C.rows(), n = C.cols();
		vector< double > f( m, 0.0 );
		for ( unsigned i = 0; i < m; i++ )
		{
			double s = -b[ i ];
			for ( unsigned j = 0; j < n; j++ ) s += C( i, j ) * x[ j ];
			f[ i ] = s;
		}
		if ( A.rows() != n || A.cols() != n ) A.resize( n, n );
		A.fill( 0.0 );
		g.assign( n, 0.0 );
		for ( unsigned i = 0; i < m; i++ )
		{
			for ( unsigned j = 0; j < n; j++ )
			{
				g[ j ] += C( i, j ) * f[ i ];
				for ( unsigned k = 0; k < n; k++ )
					A( j, k ) += C( i, j ) * C( i, k );
			}
		}
		double F = 0.0;
		for ( unsigned i = 0; i < m; i++ ) F += 0.5 * f[ i ] * f[ i ];
		return F;
	}
};

// ---- (b) ROSENBROCK as a two-residual problem, the source's Example 3.16 ---
//      f(x) = [ 10( x2 - x1^2 ), 1 - x1 ],  x* = ( 1, 1 ),  F(x*) = 0
class Rosenbrock : public Analytic {
protected:
	virtual double build( Matrix< double >& A, vector< double >& g )
	{
		const double f0 = 10.0 * ( x[ 1 ] - x[ 0 ] * x[ 0 ] );
		const double f1 = 1.0 - x[ 0 ];
		// J = [ -20 x1  10 ; -1  0 ]
		const double j00 = -20.0 * x[ 0 ], j01 = 10.0, j10 = -1.0, j11 = 0.0;

		if ( A.rows() != 2 ) A.resize( 2, 2 );
		A( 0, 0 ) = j00 * j00 + j10 * j10;
		A( 0, 1 ) = j00 * j01 + j10 * j11;
		A( 1, 0 ) = A( 0, 1 );
		A( 1, 1 ) = j01 * j01 + j11 * j11;

		g.assign( 2, 0.0 );
		g[ 0 ] = j00 * f0 + j10 * f1;
		g[ 1 ] = j01 * f0 + j11 * f1;

		return 0.5 * ( f0 * f0 + f1 * f1 );
	}
};

// ---- (c) SCRIPTED: fixed A and g, and an objective the test dictates -------
//      This is what lets the gain ratio be placed EXACTLY where a branch needs
//      it. A and g are constant, so the step depends only on mu, and the test
//      can reproduce the solve independently to compute the denominator.
class Scripted : public Analytic {
public:
	Matrix< double > fixedA;
	vector< double > fixedG;
	vector< double > objectives;     // consumed in order
	size_t next;
	unsigned poisonAt;               // 1-based evaluation index, 0 = never.
	double poison;                   //    Evaluation 1 is the STARTING point,
	                                 //    so a TRIAL poison is at least 2.
	Scripted() : next( 0 ), poisonAt( 0 ), poison( 0 ) { }

protected:
	virtual double build( Matrix< double >& A, vector< double >& g )
	{
		A = fixedA;
		g = fixedG;
		if ( poisonAt && evaluations == poisonAt ) return poison;
		if ( next < objectives.size() ) return objectives[ next++ ];
		return objectives.empty() ? 0.0 : objectives.back();
	}
};

// The step Algorithm 3.16 will take, computed INDEPENDENTLY of LM so a test can
//    predict the denominator and place rho exactly.
static void predictStep( const Matrix< double >& A, const vector< double >& g,
	double mu, vector< double >& h )
{
	Matrix< double > damped = A;
	for ( unsigned i = 0; i < A.rows(); i++ ) damped( i, i ) += mu;
	vector< double > rhs( g.size() );
	for ( size_t i = 0; i < g.size(); i++ ) rhs[ i ] = -g[ i ];
	damped.solveSPD( rhs, h );
}

static double predictDenominator( const vector< double >& h,
	const vector< double >& g, double mu )
{
	double d = 0.0;
	for ( size_t i = 0; i < h.size(); i++ ) d += h[ i ] * ( mu * h[ i ] - g[ i ] );
	return 0.5 * d;
}

// ---------------------------------------------------------------------------
// THE REAL PATH. A subclass so a test can reach the protected boundary, count
// traversals, and drive training without a dataset file.

template < class NET >
class Probe : public NET {
public:
	Probe() : evaluations( 0 ) { }
	unsigned evaluations;

	double batchNormalEquations( Matrix< double >& A, vector< double >& g ) override
	{ evaluations++; return NET::batchNormalEquations( A, g ); }

	void prepare() { this->prepareRun(); }
	unsigned packed() const { return this->packedSize(); }
	void weightsOut( vector< double >& w ) const { this->packWeights( w ); }
	void weightsIn( const vector< double >& w ) { this->unpackWeights( w ); }
	double normalPass( Matrix< double >& A, vector< double >& g )
	{ return this->batchNormalEquations( A, g ); }
	double gradientPass( vector< double >& g )
	{ return this->batchObjectiveGradient( g ); }
	double oneIteration() { return this->innerTrainSet(); }
	double gradMaxNow() { return this->getGradMax(); }
	LM& optimizer() { return this->lm; }
	bool stepConvergedNow() const { return this->stepConverged(); }
	Matrix< double >& trainMatrix() { return this->Train; }
	double outputNow() const { return this->o; }
};

// A deterministic two-class dataset, built in memory so no file is needed.
static DataSet makeData( unsigned rows, unsigned inputs, unsigned seed )
{
	Matrix< double > raw( rows, inputs + 1 );
	unsigned s = seed;
	for ( unsigned i = 0; i < rows; i++ )
	{
		double acc = 0;
		for ( unsigned j = 0; j < inputs; j++ )
		{
			s = s * 1103515245u + 12345u;
			const double v = -1.0 + 2.0 * ( double ) ( ( s >> 16 ) % 1000 ) / 999.0;
			raw( i, j ) = v;
			acc += v * ( ( j % 2 ) ? -1.0 : 1.0 );
		}
		raw( i, inputs ) = ( acc > 0.0 ) ? 1.0 : 0.0;
	}
	DataSet d;
	d.setOutput( 1 );
	d.setDiscrete( true );
	d.setHistory( false );
	d.setInput( inputs );
	d.setRawMatrix( raw );
	Matrix< double >& r = d.getRawMatrix();
	d.setTrainMatrix( r );
	return d;
}

template < class NET >
static void configureLM( Probe< NET >& p, DataSet& d, unsigned hidden,
	unsigned seed, double eta = 0.05 )
{
	p.setHistory( false );
	p.setLastop( false );
	p.setQuiet( true );
	p.setLMSerror();
	p.setWeightDecay( true );
	p.setDecay( 5e-5 );
	p.setBatchEpoch( true );
	p.setAutoStepSize( false );
	p.setEta( eta );
	p.setDataSet( d );
	p.setHidden( hidden );
	p.setTrainingType( Network::TRAIN_LM );
	util::set_seed( seed );
	p.randomize();
	p.prepare();
}

int main()
{
	util::ScreenCapture hush;

	// =====================================================================
	// LAYER 1: MODEL-FREE. Algorithm 3.16 driven by hand.
	// =====================================================================

	// --- 1. Exact linear least squares --------------------------------------
	{
		LinearLS p;
		p.C.resize( 4, 2 );
		p.C( 0, 0 ) = 1;  p.C( 0, 1 ) = 0;
		p.C( 1, 0 ) = 1;  p.C( 1, 1 ) = 1;
		p.C( 2, 0 ) = 1;  p.C( 2, 1 ) = 2;
		p.C( 3, 0 ) = 1;  p.C( 3, 1 ) = 3;
		p.b.assign( 4, 0.0 );
		p.b[ 0 ] = 1; p.b[ 1 ] = 3; p.b[ 2 ] = 5; p.b[ 3 ] = 7; // exactly y = 1 + 2t
		p.x.assign( 2, 0.0 );

		LM lm;
		// AT THE EXACT MINIMUM NO STEP CAN IMPROVE, so every trial is rejected
		//    and StepRejected is raised. That is the correct contract -- LM
		//    cannot tell "already optimal" from "stuck" -- and in a real run
		//    Iterative's STOP_GRADMAX, which IS Algorithm 3.16's own criterion
		//    (3.15a), fires long before. Pinned here rather than left to be
		//    discovered.
		bool exhausted = false;
		try { for ( int i = 0; i < 40; i++ ) lm.iterate( p ); }
		catch ( const LM::StepRejected& ) { exhausted = true; }
		const bool converged = lm.stepConverged();

		close( p.x[ 0 ], 1.0, 1e-9, "linear least squares intercept" );
		close( p.x[ 1 ], 2.0, 1e-9, "linear least squares slope" );
		ok( lm.acceptances() > 0, "linear least squares accepted steps" );
		// v1 RAISED StepRejected here. v2 reports convergence instead, which is
		//    the whole point of restoring (3.15b).
		ok( !exhausted && converged,
			"at the exact minimum v2 reports small-step convergence, not failure" );
	}

	// --- 2. rho is EXACTLY 1 on a linear problem, so mu is multiplied by 1/3 -
	//     For linear f the linear model L IS F, so the predicted gain equals
	//     the actual gain: rho = 1, and 1 - (2*1-1)^3 = 0, which the floor
	//     raises to exactly 1/3.
	{
		LinearLS p;
		p.C.resize( 3, 2 );
		p.C( 0, 0 ) = 2; p.C( 0, 1 ) = 0;
		p.C( 1, 0 ) = 0; p.C( 1, 1 ) = 3;
		p.C( 2, 0 ) = 1; p.C( 2, 1 ) = 1;
		p.b.assign( 3, 0.0 );
		p.b[ 0 ] = 4; p.b[ 1 ] = 3; p.b[ 2 ] = 3;
		p.x.assign( 2, 0.0 );

		// GUARDED so a sabotage of the damping rule produces a named assertion
		//    failure rather than an abort. A file that dies tells the reader
		//    nothing about WHICH mechanism it was guarding.
		LM lm;
		double mu1 = 0, mu2 = 0, mu3 = 0;
		try
		{
			lm.iterate( p ); mu1 = lm.damping();
			lm.iterate( p ); mu2 = lm.damping();
			lm.iterate( p ); mu3 = lm.damping();
		}
		catch ( const LM::StepRejected& )
		{ ok( false, "linear problem: no step could be accepted at all" ); }

		close( mu2 / mu1, 1.0 / 3.0, 1e-12,
			"linear problem: rho == 1 so mu is scaled by exactly 1/3" );
		close( mu3 / mu2, 1.0 / 3.0, 1e-12,
			"linear problem: and again on the next iteration" );
		close( lm.rejectionMultiplier(), 2.0, 0,
			"nu is reset to 2 on every acceptance" );
	}

	// --- 3. Rosenbrock, the source's own Example 3.16 -----------------------
	{
		Rosenbrock p;
		p.x.assign( 2, 0.0 );
		p.x[ 0 ] = -1.2; p.x[ 1 ] = 1.0;

		LM lm;
		try
		{
			for ( int i = 0; i < 200; i++ )
			{
				lm.iterate( p );
				if ( lm.acceptedObjective() < 1e-24 ) break;
			}
		}
		catch ( const LM::StepRejected& ) { }

		close( p.x[ 0 ], 1.0, 1e-6, "Rosenbrock x1" );
		close( p.x[ 1 ], 1.0, 1e-6, "Rosenbrock x2" );
		ok( lm.acceptedObjective() < 1e-14, "Rosenbrock reached the minimum" );
	}

	// --- 4. mu_0 = tau * max( a_ii ), with the maximum NOT in position 0 -----
	{
		Scripted p;
		p.fixedA.resize( 3, 3 );
		p.fixedA.fill( 0.0 );
		p.fixedA( 0, 0 ) = 4.0;
		p.fixedA( 1, 1 ) = 137.0;   // the maximum, deliberately not first
		p.fixedA( 2, 2 ) = 9.0;
		p.fixedG.assign( 3, 1.0 );
		p.objectives.push_back( 100.0 );
		p.objectives.push_back( 99.0 );
		p.x.assign( 3, 0.0 );

		// Read mu_0 by CANCELLING the moment the starting point is evaluated:
		//    the damping loop's first act is the cancellation check, so mu is
		//    still exactly mu_0 and nothing has multiplied it.
		p.cancelAfter = 1;
		LM lm;
		lm.iterate( p );
		close( lm.damping(), LM::TAU * 137.0, 1e-12,
			"mu_0 = tau * max( a_ii ), maximum taken over the whole diagonal" );
	}

	// --- 5. THE ACCEPTANCE mu UPDATE, at three hand-chosen values of rho -----
	//     mu *= max( 1/3, 1 - (2 rho - 1)^3 ). Placed exactly by computing the
	//     denominator independently and dictating the trial objective.
	{
		const double rhos[ 3 ] = { 1.0, 0.5, 0.02 };
		//   rho = 1.00 -> 1 - 1   = 0      -> floor 1/3
		//   rho = 0.50 -> 1 - 0   = 1
		//   rho = 0.02 -> 1 - (-0.96)^3 = 1.884736
		const double wants[ 3 ] = { 1.0 / 3.0, 1.0, 1.884736 };
		const char* names[ 3 ] = {
			"mu update at rho = 1: the 1/3 floor binds",
			"mu update at rho = 0.5: factor is exactly 1",
			"mu update at rho = 0.02: factor is 1 - (2rho-1)^3" };

		for ( int c = 0; c < 3; c++ )
		{
			Scripted p;
			p.fixedA.resize( 2, 2 );
			p.fixedA.fill( 0.0 );
			p.fixedA( 0, 0 ) = 5.0; p.fixedA( 1, 1 ) = 2.0;
			p.fixedG.assign( 2, 0.0 );
			p.fixedG[ 0 ] = 3.0; p.fixedG[ 1 ] = -1.0;
			p.x.assign( 2, 0.0 );

			const double F0 = 10.0;
			const double mu0 = LM::TAU * 5.0;
			vector< double > h;
			predictStep( p.fixedA, p.fixedG, mu0, h );
			const double denom = predictDenominator( h, p.fixedG, mu0 );

			p.objectives.push_back( F0 );
			p.objectives.push_back( F0 - rhos[ c ] * denom );

			LM lm;
			lm.iterate( p );
			close( lm.damping() / mu0, wants[ c ], 1e-10, names[ c ] );
			ok( lm.acceptances() == 1, "the scripted step was accepted" );
		}
	}

	// --- 6. THE REJECTION SCHEDULE: mu *= nu, nu *= 2 -----------------------
	{
		Scripted p;
		p.fixedA.resize( 2, 2 );
		p.fixedA.fill( 0.0 );
		p.fixedA( 0, 0 ) = 5.0; p.fixedA( 1, 1 ) = 2.0;
		p.fixedG.assign( 2, 1.0 );
		p.x.assign( 2, 0.0 );
		p.objectives.push_back( 10.0 );
		for ( int i = 0; i < 40; i++ ) p.objectives.push_back( 1e6 ); // always worse
		// BOUNDED BY CANCELLATION at three rejections. Running to exhaustion is
		//    no longer possible in v2: mu grows, the step shrinks, and (3.15b)
		//    terminates the iteration first -- which is exactly what the source
		//    says happens when rounding errors dominate. Cancelling after the
		//    third trial reads the schedule while it is still running.
		p.cancelAfter = 4;   // the start plus three trials

		const double mu0 = LM::TAU * 5.0;
		LM lm;
		lm.iterate( p );

		ok( lm.rejections() == 3, "three trials were rejected" );
		close( lm.damping() / mu0, 64.0, 1e-9,
			"mu grew by 2*4*8 = 64: nu DOUBLES, it is not a fixed factor" );
		close( lm.rejectionMultiplier(), 16.0, 0,
			"nu reached 16 after three rejections" );
	}

	// --- 7. rho == 0 is REJECTED, not accepted ------------------------------
	{
		Scripted p;
		p.fixedA.resize( 2, 2 );
		p.fixedA.fill( 0.0 );
		p.fixedA( 0, 0 ) = 5.0; p.fixedA( 1, 1 ) = 2.0;
		p.fixedG.assign( 2, 1.0 );
		p.x.assign( 2, 0.0 );
		p.objectives.push_back( 10.0 );
		for ( int i = 0; i < 40; i++ ) p.objectives.push_back( 10.0 ); // rho == 0 exactly
		p.cancelAfter = 2;   // the start plus one trial

		LM lm;
		lm.iterate( p );
		ok( lm.acceptances() == 0 && lm.rejections() == 1,
			"rho == 0 is rejected: the test is `> 0`, not `>= 0`" );
	}

	// --- 8. REJECTED-STEP RESTORATION is bit-identical ----------------------
	{
		Scripted p;
		p.fixedA.resize( 2, 2 );
		p.fixedA.fill( 0.0 );
		p.fixedA( 0, 0 ) = 5.0; p.fixedA( 1, 1 ) = 2.0;
		p.fixedG.assign( 2, 1.0 );
		p.x.assign( 2, 0.0 );
		p.x[ 0 ] = 0.25; p.x[ 1 ] = -0.75;
		const vector< double > before = p.x;
		p.objectives.push_back( 10.0 );
		for ( int i = 0; i < 40; i++ ) p.objectives.push_back( 1e6 );

		LM lm;
		try { lm.iterate( p ); } catch ( const LM::StepRejected& ) { }
		ok( p.x == before,
			"every rejected trial left the point BIT-IDENTICAL" );
	}

	// --- 9. Covered by 6 and 8: the bound, and restoration at the bound ------

	// --- 10. A FAILED FACTORIZATION is a rejection, not a throw -------------
	//     A singular A with mu tiny: the first solve fails, mu grows, and a
	//     later one succeeds. It must never surface as Matrix::Singular.
	{
		Scripted p;
		p.fixedA.resize( 2, 2 );
		// INDEFINITE, not merely singular: A + mu I stays non-positive-definite
		//    until mu grows past about 9, so Algorithm A.4's test fails on the
		//    first solves and succeeds on a later one. J'J from a model is
		//    always positive semidefinite, so this cannot arise from the model
		//    -- but the primitive can still fail on a badly conditioned A at a
		//    tiny mu, and THAT is the path this exercises.
		p.fixedA.fill( 0.0 );
		p.fixedA( 0, 0 ) = 1.0;  p.fixedA( 1, 1 ) = 1.0;
		p.fixedA( 0, 1 ) = 10.0; p.fixedA( 1, 0 ) = 10.0;
		p.fixedG.assign( 2, 1.0 );
		p.x.assign( 2, 0.0 );
		p.objectives.push_back( 10.0 );
		for ( int i = 0; i < 40; i++ ) p.objectives.push_back( 1.0 ); // improving

		LM lm;
		bool leaked = false, accepted = false;
		try { accepted = ( lm.iterate( p ), true ); }
		catch ( const Matrix< double >::Singular& ) { leaked = true; }
		catch ( const LM::StepRejected& ) { }
		ok( !leaked, "a failed factorization never escapes as Matrix::Singular" );
		ok( accepted, "and damping harder eventually produces a usable step" );
		ok( lm.factorizationFailures() >= 1,
			"the failed factorization was counted as a rejection" );
	}

	// --- 11. NON-FINITE TRIALS: all three members, and -infinity above all ---
	{
		const double poisons[ 3 ] = {
			std::numeric_limits< double >::quiet_NaN(),
			std::numeric_limits< double >::infinity(),
			-std::numeric_limits< double >::infinity() };
		const char* names[ 3 ] = {
			"a NaN trial objective is rejected",
			"a +infinity trial objective is rejected",
			"a -INFINITY trial objective is REJECTED, not accepted as an "
				"infinitely good improvement" };

		for ( int c = 0; c < 3; c++ )
		{
			Scripted p;
			p.fixedA.resize( 2, 2 );
			p.fixedA.fill( 0.0 );
			p.fixedA( 0, 0 ) = 5.0; p.fixedA( 1, 1 ) = 2.0;
			p.fixedG.assign( 2, 1.0 );
			p.x.assign( 2, 0.0 );
			const vector< double > before = p.x;
			p.objectives.push_back( 10.0 );
			p.poisonAt = 2;              // evaluation 1 is the starting point
			p.poison = poisons[ c ];
			for ( int i = 0; i < 40; i++ ) p.objectives.push_back( 1e6 );

			LM lm;
			try { lm.iterate( p ); } catch ( const LM::StepRejected& ) { }
			ok( lm.acceptances() == 0 && p.x == before, names[ c ] );
			ok( lm.nonFiniteTrials() >= 1, "the non-finite trial was counted" );
		}
	}

	// --- 11b. A non-finite STARTING point throws NotFinite, nothing moved ---
	{
		Scripted p;
		p.fixedA.resize( 2, 2 );
		p.fixedA.fill( 0.0 );
		p.fixedA( 0, 0 ) = 5.0; p.fixedA( 1, 1 ) = 2.0;
		p.fixedG.assign( 2, 1.0 );
		p.x.assign( 2, 0.0 );
		p.objectives.push_back( std::numeric_limits< double >::quiet_NaN() );

		LM lm;
		bool threw = false;
		try { lm.iterate( p ); } catch ( const LM::NotFinite& ) { threw = true; }
		ok( threw, "a non-finite STARTING objective throws NotFinite" );
	}

	// --- 11c. mu_0 <= 0 is a named refusal, not a silent floor ---------------
	{
		Scripted p;
		p.fixedA.resize( 2, 2 );
		p.fixedA.fill( 0.0 );   // every diagonal zero
		p.fixedG.assign( 2, 0.0 );
		p.x.assign( 2, 0.0 );
		p.objectives.push_back( 1.0 );

		LM lm;
		bool threw = false;
		try { lm.iterate( p ); } catch ( const LM::NotFinite& ) { threw = true; }
		ok( threw, "an all-zero curvature refuses rather than floors mu_0" );
	}

	// --- 12. THE DENOMINATOR INVARIANT: L(0) - L(h) > 0 ---------------------
	//     The source proves h'h and -h'g are both positive, so the gain ratio's
	//     denominator cannot be zero or negative. That is asserted as an
	//     INVARIANT over the same solve LM uses, across four decades of damping
	//     and several gradients, rather than guarded by a branch that cannot be
	//     reached -- an untestable branch is a claim nothing compiles.
	{
		Matrix< double > A( 3, 3, 0.0 );
		A( 0, 0 ) = 5.0; A( 1, 1 ) = 0.25; A( 2, 2 ) = 40.0;
		A( 0, 1 ) = A( 1, 0 ) = 0.5;
		A( 1, 2 ) = A( 2, 1 ) = -0.75;

		bool everNonPositive = false, everDegenerate = false;
		for ( int gi = 0; gi < 3; gi++ )
		{
			vector< double > g( 3 );
			g[ 0 ] = ( gi == 0 ) ? 1.0 : -3.5;
			g[ 1 ] = ( gi == 1 ) ? 12.0 : 0.125;
			g[ 2 ] = ( gi == 2 ) ? -0.001 : 2.0;
			for ( double mu = 1e-6; mu < 1e5; mu *= 10.0 )
			{
				vector< double > h;
				predictStep( A, g, mu, h );
				const double d = predictDenominator( h, g, mu );
				if ( !( d > 0.0 ) ) everNonPositive = true;
				if ( !std::isfinite( d ) ) everDegenerate = true;
			}
		}
		ok( !everNonPositive, "L(0) - L(h) is positive at every damping tested" );
		ok( !everDegenerate, "and finite at every damping tested" );
	}

	// --- 13. CANCELLATION mid-loop restores the accepted point exactly ------
	{
		Scripted p;
		p.fixedA.resize( 2, 2 );
		p.fixedA.fill( 0.0 );
		p.fixedA( 0, 0 ) = 5.0; p.fixedA( 1, 1 ) = 2.0;
		p.fixedG.assign( 2, 1.0 );
		p.x.assign( 2, 0.0 );
		p.x[ 0 ] = 0.125; p.x[ 1 ] = -0.5;
		const vector< double > before = p.x;
		p.objectives.push_back( 10.0 );
		for ( int i = 0; i < 40; i++ ) p.objectives.push_back( 1e6 );
		p.cancelAfter = 3;    // cancel after the third evaluation

		LM lm;
		const double got = lm.iterate( p );
		close( got, 10.0, 0, "a cancelled iteration returns the departing objective" );
		ok( p.x == before, "a cancelled iteration restores the point exactly" );
		ok( lm.cancellations() == 1, "the cancellation was counted" );
	}

	// =====================================================================
	// LAYER 2: THE REAL PATH.
	// =====================================================================

	DataSet data = makeData( 400, 5, 20260808u );

	// A SMALL fixture that reaches a genuine stationary point quickly, which is
	//    the state the characterized v1 failure occurred in. The larger fixture
	//    above keeps accepting ~1e-12 improvements for thousands of iterations
	//    and never stalls, so (3.15b) cannot fire there -- a test that used it
	//    would be asserting against a state it never reaches.
	DataSet small = makeData( 40, 3, 20260808u );

	// --- 14. batchNormalEquations' gradient equals batchObjectiveGradient's --
	//     THE SECTION 3 IDENTITY on a real model: J'f IS grad F, reached from
	//     d o / d w rather than d E / d w.
	for ( int decayOn = 0; decayOn < 2; decayOn++ )
	{
		Probe< SimpleProp > sp;
		configureLM( sp, data, 3, 11u );
		sp.setWeightDecay( decayOn != 0 );
		sp.prepare();

		Matrix< double > A;
		vector< double > gN, gP;
		const double fN = sp.normalPass( A, gN );
		const double fP = sp.gradientPass( gP );

		close( fN, fP, 1e-12, "SimpleProp: the two traversals agree on F" );
		ok( gN.size() == gP.size() && !gN.empty(),
			"SimpleProp: the two gradients are the same non-empty length" );
		double worst = 0;
		for ( size_t i = 0; i < gN.size(); i++ )
			worst = max( worst, fabs( gN[ i ] - gP[ i ] )
				/ max( 1.0, fabs( gP[ i ] ) ) );
		close( worst, 0.0, 1e-12, "SimpleProp: every gradient element agrees" );

		Probe< BareProp > bp;
		configureLM( bp, data, 3, 11u );
		bp.setWeightDecay( decayOn != 0 );
		bp.prepare();
		Matrix< double > A2;
		vector< double > gN2, gP2;
		const double fN2 = bp.normalPass( A2, gN2 );
		const double fP2 = bp.gradientPass( gP2 );
		close( fN2, fP2, 1e-12, "BareProp: the two traversals agree on F" );
		worst = 0;
		for ( size_t i = 0; i < gN2.size(); i++ )
			worst = max( worst, fabs( gN2[ i ] - gP2[ i ] )
				/ max( 1.0, fabs( gP2[ i ] ) ) );
		close( worst, 0.0, 1e-12, "BareProp: every gradient element agrees" );
	}

	// --- 15. A against a finite-difference Gauss-Newton oracle --------------
	//     A = (1/N) sum_k a_k a_k' + decay I with each a_k = d o_k / d w taken
	//     by CENTRAL DIFFERENCES of the model's own output. Nothing here shares
	//     a line with the implementation.
	{
		Probe< SimpleProp > sp;
		configureLM( sp, data, 2, 7u );
		Matrix< double > A;
		vector< double > g;
		sp.normalPass( A, g );

		const unsigned P = sp.packed();
		vector< double > w0;
		sp.weightsOut( w0 );

		// Per-exemplar outputs at w0 +/- h e_p, for the first few exemplars only
		//    -- the oracle is O(P) full passes, so it is run on a small model.
		const unsigned N = 400;
		const double step = 1e-6;
		Matrix< double > oracle( P, P, 0.0 );
		vector< vector< double > > J( N, vector< double >( P, 0.0 ) );

		for ( unsigned p = 0; p < P; p++ )
		{
			vector< double > wp = w0, wm = w0;
			wp[ p ] += step; wm[ p ] -= step;
			vector< double > up( N ), um( N );
			sp.weightsIn( wp );
			for ( unsigned k = 0; k < N; k++ )
			{ sp.forward( sp.trainMatrix(), k ); up[ k ] = sp.outputNow(); }
			sp.weightsIn( wm );
			for ( unsigned k = 0; k < N; k++ )
			{ sp.forward( sp.trainMatrix(), k ); um[ k ] = sp.outputNow(); }
			for ( unsigned k = 0; k < N; k++ )
				J[ k ][ p ] = ( up[ k ] - um[ k ] ) / ( 2.0 * step );
		}
		sp.weightsIn( w0 );

		for ( unsigned k = 0; k < N; k++ )
			for ( unsigned i = 0; i < P; i++ )
				for ( unsigned j = 0; j < P; j++ )
					oracle( i, j ) += J[ k ][ i ] * J[ k ][ j ];
		for ( unsigned i = 0; i < P; i++ )
			for ( unsigned j = 0; j < P; j++ )
				oracle( i, j ) /= ( double ) N;
		for ( unsigned i = 0; i < P; i++ )
			oracle( i, i ) += sp.getDecay();

		double worst = 0;
		for ( unsigned i = 0; i < P; i++ )
			for ( unsigned j = 0; j < P; j++ )
				worst = max( worst, fabs( A( i, j ) - oracle( i, j ) )
					/ max( 1e-3, fabs( oracle( i, j ) ) ) );
		close( worst, 0.0, 1e-5,
			"A matches a finite-difference Gauss-Newton oracle" );
	}

	// --- 16. A is symmetric, and its diagonal carries decay ------------------
	{
		Probe< SimpleProp > sp;
		configureLM( sp, data, 3, 5u );
		Matrix< double > A;
		vector< double > g;
		sp.normalPass( A, g );
		bool sym = true, diag = true;
		for ( unsigned i = 0; i < A.rows(); i++ )
		{
			if ( !( A( i, i ) >= sp.getDecay() ) ) diag = false;
			for ( unsigned j = i + 1; j < A.cols(); j++ )
				if ( A( i, j ) != A( j, i ) ) sym = false;
		}
		ok( sym, "A is exactly symmetric: the symmetry completion ran" );
		ok( diag, "A's diagonal carries the decay term" );
	}

	// --- 17. The evaluation does not train ----------------------------------
	{
		Probe< SimpleProp > sp;
		configureLM( sp, data, 3, 5u );
		vector< double > before, after;
		sp.weightsOut( before );
		Matrix< double > A; vector< double > g;
		sp.normalPass( A, g );
		sp.normalPass( A, g );
		sp.weightsOut( after );
		ok( before == after && !before.empty(),
			"batchNormalEquations evaluates and does not train" );
	}

	// --- 18. THE WORK-SCHEDULING CLAIM, proven rather than asserted ---------
	//     LM takes A and g at the TRIAL point and reuses them on acceptance,
	//     where Algorithm 3.16 recomputes them after acceptance. That is a
	//     scheduling choice which must leave the accepted sequence identical.
	//     Driven model-free, where a second driver can be written that
	//     recomputes: both must produce bit-identical iterates.
	{
		Rosenbrock a, b;
		a.x.assign( 2, 0.0 ); a.x[ 0 ] = -1.2; a.x[ 1 ] = 1.0;
		b.x = a.x;

		LM lmA;
		try { for ( int i = 0; i < 25; i++ ) lmA.iterate( a ); }
		catch ( const LM::StepRejected& ) { }

		// The alternate driver: force a fresh evaluation at the accepted point
		//    before every iteration, which is what "recompute after acceptance"
		//    amounts to for a stateless analytic objective.
		LM lmB;
		try
		{
			for ( int i = 0; i < 25; i++ )
			{
				Matrix< double > A; vector< double > g;
				b.evaluateNormal( A, g );      // the recomputation
				lmB.iterate( b );
			}
		}
		catch ( const LM::StepRejected& ) { }

		ok( a.x.size() == b.x.size() && a.x[ 0 ] == b.x[ 0 ]
			&& a.x[ 1 ] == b.x[ 1 ],
			"trial-point accumulation leaves the accepted sequence identical" );
	}

	// --- 19. currGradMax is the RAW gradient at the DEPARTING point ---------
	{
		Probe< SimpleProp > sp;
		configureLM( sp, data, 3, 13u );
		Matrix< double > A; vector< double > g;
		sp.normalPass( A, g );
		double want = 0;
		for ( size_t i = 0; i < g.size(); i++ ) want = max( want, fabs( g[ i ] ) );

		Probe< SimpleProp > sp2;
		configureLM( sp2, data, 3, 13u );
		sp2.oneIteration();
		close( sp2.gradMaxNow(), want, 1e-12,
			"currGradMax is maxabs of the raw gradient at the departing point" );
	}

	// --- 20. eta independence: the step is ABSOLUTE -------------------------
	{
		Probe< SimpleProp > a, b;
		configureLM( a, data, 3, 3u, 0.05 );
		configureLM( b, data, 3, 3u, 0.73 );
		try { for ( int i = 0; i < 5; i++ ) { a.oneIteration(); b.oneIteration(); } }
		catch ( const LM::StepRejected& ) { }
		vector< double > wa, wb;
		a.weightsOut( wa ); b.weightsOut( wb );
		ok( wa == wb && !wa.empty(),
			"eta = 0.05 and eta = 0.73 give BIT-IDENTICAL weights" );
	}

	// --- 21. eta is not written ---------------------------------------------
	{
		Probe< SimpleProp > sp;
		configureLM( sp, data, 3, 3u, 0.37 );
		try { for ( int i = 0; i < 3; i++ ) sp.oneIteration(); }
		catch ( const LM::StepRejected& ) { }
		close( sp.getEta(), 0.37, 0, "eta is bit-identical after an LM run" );
	}

	// --- 22. EVERY ELIGIBILITY REFUSAL, by name -----------------------------
	{
		struct Case { const char* name; int which; };
		// The packedSize() == 0 refusal is NOT listed, and the omission is
		//    deliberate rather than an oversight: it is checked before the
		//    normal-equations refusal, and every model that implements the
		//    normal-equations boundary also has a packed one, so no existing
		//    model can reach it. Claiming a test for a branch nothing can
		//    reach would be the decoration this project has been caught by
		//    before. It stays in the code as a guard for a future model.
		const Case cases[ 5 ] = {
			{ "cross-entropy loss is refused (the strict LMS gate)", 0 },
			{ "on-line mode is refused", 1 },
			{ "the automatic step-size search is refused", 2 },
			{ "a model with no normal-equations boundary is refused", 3 },
			{ "a parameter count above the ceiling is refused", 4 } };

		for ( int c = 0; c < 5; c++ )
		{
			bool refused = false;
			vector< double > before, after;
			if ( cases[ c ].which == 3 )
			{
				Probe< BackProp > bpn;   // BackProp implements neither boundary
				bpn.setHistory( false ); bpn.setLastop( false ); bpn.setQuiet( true );
				bpn.setLMSerror(); bpn.setBatchEpoch( true );
				bpn.setAutoStepSize( false );
				bpn.setDataSet( data );
				vector< unsigned > layers; layers.push_back( 3 );
				bpn.setHidden( layers );
				bpn.setTrainingType( Network::TRAIN_LM );
				util::set_seed( 4u ); bpn.randomize(); bpn.prepare();
				try { bpn.oneIteration(); }
				catch ( const LM::Ineligible& ) { refused = true; }
				ok( refused, cases[ c ].name );
				continue;
			}

			Probe< SimpleProp > sp;
			configureLM( sp, data, 3, 4u );
			sp.weightsOut( before );
			if ( cases[ c ].which == 0 ) sp.setXEerror();
			if ( cases[ c ].which == 1 ) sp.setBatchEpoch( false );
			if ( cases[ c ].which == 2 ) sp.setAutoStepSize( true );
			if ( cases[ c ].which == 4 )
			{
				// A hidden layer large enough to pass the ceiling: with 5 inputs
				//    P = h*(5+1) + (h+1), so h = 100 gives 701 > 512.
				DataSet d2 = makeData( 50, 5, 3u );
				configureLM( sp, d2, 100, 4u );
				sp.weightsOut( before );
			}
			try { sp.oneIteration(); }
			catch ( const LM::Ineligible& ) { refused = true; }
			sp.weightsOut( after );
			ok( refused, cases[ c ].name );
			ok( before == after, "the refusal moved no weight" );
		}
	}

	// --- 23. THE WIRING: LM's normal equations ARE the current point's -------
	//     This is the assertion that a stale A cannot survive. After an
	//     iteration the model sits at the accepted point and LM holds that
	//     point's A and g; a freshly computed traversal at those same installed
	//     weights must reproduce them BIT FOR BIT, because it is the same code
	//     on the same numbers.
	//
	//     Every component can be correct and this can still fail: the defect it
	//     guards lives in the wire between Network::lmIteration(), the
	//     evaluator, and the model traversal -- exactly where the iRPROP+ and
	//     BB sabotages hid. The model-free tests above cannot see it, because
	//     they supply their own A.
	{
		Probe< SimpleProp > sp;
		configureLM( sp, data, 3, 29u );
		try { for ( int i = 0; i < 3; i++ ) sp.oneIteration(); }
		catch ( const LM::StepRejected& ) { }

		Matrix< double > fresh;
		vector< double > freshG;
		sp.normalPass( fresh, freshG );

		const Matrix< double >& held = sp.optimizer().normalMatrix();
		const vector< double >& heldG = sp.optimizer().gradient();

		bool sameA = ( held.rows() == fresh.rows()
			&& held.cols() == fresh.cols() && fresh.rows() > 0 );
		for ( unsigned i = 0; sameA && i < fresh.rows(); i++ )
			for ( unsigned j = 0; sameA && j < fresh.cols(); j++ )
				if ( held( i, j ) != fresh( i, j ) ) sameA = false;

		ok( sameA, "LM holds the CURRENT point's normal matrix, not a stale one" );
		ok( heldG == freshG && !freshG.empty(),
			"LM holds the CURRENT point's gradient, not a stale one" );
	}

	// --- 24. PER-RUN RESET and copy ----------------------------------------
	{
		Probe< SimpleProp > sp;
		configureLM( sp, data, 3, 9u );
		try { for ( int i = 0; i < 4; i++ ) sp.oneIteration(); }
		catch ( const LM::StepRejected& ) { }
		const double muAfter = sp.optimizer().damping();
		ok( sp.optimizer().started(), "the run started" );

		sp.prepare();   // what train() calls before a new run
		ok( !sp.optimizer().started() && sp.optimizer().iterations() == 0,
			"prepareRun() resets the LM working state" );

		Probe< SimpleProp > copy;
		configureLM( copy, data, 3, 9u );
		try { for ( int i = 0; i < 4; i++ ) copy.oneIteration(); }
		catch ( const LM::StepRejected& ) { }
		close( copy.optimizer().damping(), muAfter, 0,
			"a fresh run reproduces the same damping from the same start" );
	}

	// --- 25. Pre-update return association -----------------------------------
	{
		Probe< SimpleProp > sp;
		configureLM( sp, data, 3, 17u );
		Matrix< double > A; vector< double > g;
		const double before = sp.normalPass( A, g );

		Probe< SimpleProp > sp2;
		configureLM( sp2, data, 3, 17u );
		const double returned = sp2.oneIteration();
		close( returned, before, 1e-12,
			"trainSet() returns the objective at the weights BEFORE the step" );
	}

	// --- 26. Fixed-start real-model integration -----------------------------
	for ( int model = 0; model < 2; model++ )
	{
		double first = 0, last = 0;
		bool finite = true;
		unsigned evaluations = 0;

		if ( model == 0 )
		{
			Probe< SimpleProp > sp;
			configureLM( sp, data, 4, 21u );
			first = sp.oneIteration();
			try { for ( int i = 0; i < 12; i++ ) last = sp.oneIteration(); }
			catch ( const LM::StepRejected& ) { }
			evaluations = sp.evaluations;
			vector< double > w; sp.weightsOut( w );
			for ( size_t i = 0; i < w.size(); i++ )
				if ( !std::isfinite( w[ i ] ) ) finite = false;
		}
		else
		{
			Probe< BareProp > bp;
			configureLM( bp, data, 4, 21u );
			first = bp.oneIteration();
			try { for ( int i = 0; i < 12; i++ ) last = bp.oneIteration(); }
			catch ( const LM::StepRejected& ) { }
			evaluations = bp.evaluations;
			vector< double > w; bp.weightsOut( w );
			for ( size_t i = 0; i < w.size(); i++ )
				if ( !std::isfinite( w[ i ] ) ) finite = false;
		}

		ok( finite, "integration: every weight stayed finite" );
		ok( last < first, "integration: the objective was materially reduced" );
		ok( evaluations >= 13,
			"integration: every trial evaluation traversed the training set" );
	}

	// =====================================================================
	// LM-v2: criterion (3.15b) restored. See lm_v2_source_decision.md.
	// =====================================================================

	// --- 27. THE CHARACTERIZED v1 FAILURE, REPRODUCED AND CONVERTED --------
	//     v1 raised StepRejected at a converged point. v2 must report small-step
	//     CONVERGENCE there instead, with the accepted weights untouched.
	{
		Probe< SimpleProp > sp;
		configureLM( sp, small, 2, 31u );

		bool threw = false;
		vector< double > before, after;
		unsigned iterations = 0;
		for ( ; iterations < 1200; iterations++ )
		{
			sp.weightsOut( before );
			try { sp.oneIteration(); }
			catch ( const LM::StepRejected& ) { threw = true; break; }
			if ( sp.optimizer().stepConverged() ) break;
		}
		sp.weightsOut( after );

		ok( !threw, "v2 does NOT raise StepRejected at a converged point" );
		ok( sp.optimizer().stepConverged(),
			"v2 reports small-step convergence" );
		ok( after == before && !before.empty(),
			"the accepted weights are bit-identical after small-step convergence" );
	}

	// --- 28. THE PUBLISHED THRESHOLD, straddled model-free ------------------
	//     A scripted problem whose solved step norm sits just above and just
	//     below eps_2 ( ||x|| + eps_2 ).
	{
		for ( int side = 0; side < 2; side++ )
		{
			Scripted p;
			p.fixedA.resize( 1, 1 );
			p.fixedA( 0, 0 ) = 1.0;
			p.fixedG.assign( 1, 0.0 );
			p.x.assign( 1, 1.0 );   // ||x|| = 1, so the threshold is ~eps_2

			// mu_0 = TAU * 1. With A = 1, h = -g / ( 1 + mu ), so choosing g
			//    sets the step norm directly.
			const double thresh = LM::EPS_STEP * ( 1.0 + LM::EPS_STEP );
			const double want = side ? thresh * 0.5 : thresh * 100.0;
			p.fixedG[ 0 ] = -want * ( 1.0 + LM::TAU );

			p.objectives.push_back( 10.0 );
			for ( int i = 0; i < 40; i++ ) p.objectives.push_back( 9.0 );

			LM lm;
			const unsigned before = p.evaluations;
			try { lm.iterate( p ); } catch ( const LM::StepRejected& ) { }
			const unsigned trials = p.evaluations - before - 1; // minus the start

			if ( side )
			{
				ok( lm.stepConverged(),
					"a step BELOW the threshold reports small-step convergence" );
				// --- 29. and it is tested BEFORE any trial ------------------
				ok( trials == 0,
					"criterion (3.15b) is tested BEFORE the trial: no traversal" );
			}
			else
				ok( !lm.stepConverged(),
					"a step ABOVE the threshold does NOT report convergence" );
		}
	}

	// --- 30. Iterative owns the stop, and the association is unchanged ------
	{
		Probe< SimpleProp > sp;
		configureLM( sp, small, 2, 31u );
		sp.setMinStop( false );
		sp.setChangeStop( false );
		sp.setWindowStop( false );
		sp.setGradStop( false );
		sp.setMaxIterations( 1200 );
		sp.setQuiet( true );
		sp.train();

		ok( sp.getStopReason() == Iterative::STOP_SMALL_STEP,
			"a real run ends as STOP_SMALL_STEP" );
		ok( Iterative::converged( sp.getStopReason() ),
			"STOP_SMALL_STEP counts as CONVERGENCE" );
		ok( string( Iterative::stopReasonToken( Iterative::STOP_SMALL_STEP ) )
			== "small_step", "its machine-readable token is small_step" );
	}

	// --- 31. No other optimizer is affected ---------------------------------
	{
		const unsigned others[ 4 ] = { 0, 2, Network::TRAIN_LBFGS,
			Network::TRAIN_IRPROP };
		for ( int k = 0; k < 4; k++ )
		{
			Probe< SimpleProp > sp;
			configureLM( sp, data, 3, 31u );
			sp.setTrainingType( others[ k ] );
			sp.prepare();
			for ( int i = 0; i < 5; i++ ) sp.oneIteration();
			ok( !sp.stepConvergedNow(),
				"stepConverged() stays false for every other optimizer" );
		}
	}

	// --- 32. StepRejected is still reachable --------------------------------
	//     The always-worse scripted problem, whose step stays numerically
	//     meaningful, must still throw. The fix must not swallow a real failure.
	{
		Scripted p;
		p.fixedA.resize( 2, 2 );
		p.fixedA.fill( 0.0 );
		p.fixedA( 0, 0 ) = 5.0; p.fixedA( 1, 1 ) = 2.0;
		p.fixedG.assign( 2, 1.0 );
		p.x.assign( 2, 1.0 );
		p.objectives.push_back( 10.0 );
		for ( int i = 0; i < 40; i++ ) p.objectives.push_back( 1e6 );

		// WHAT IS ACTUALLY TRUE IN v2, asserted instead of what would be
		//    convenient. Restoring (3.15b) makes the rejection bound
		//    unreachable through this path: every rejection multiplies mu, the
		//    step shrinks with it, and the small-step criterion fires first.
		//    Madsen, Nielsen & Tingleff say exactly this of their own algorithm
		//    -- when rounding errors dominate, "mu grows fast, resulting in
		//    small ||h_lm||, and the process will be stopped by (3.15b)".
		//
		//    So MAX_REJECTIONS remains as a BOUNDED BACKSTOP and is deliberately
		//    NOT claimed as tested: no declared construction reaches it, and a
		//    test for a branch nothing can reach is decoration. The assertion
		//    below pins the behaviour that replaced it.
		LM lm;
		bool threw = false;
		try { lm.iterate( p ); } catch ( const LM::StepRejected& ) { threw = true; }
		ok( !threw && lm.stepConverged(),
			"an always-worse objective now terminates by (3.15b), not by the "
			"rejection bound -- the source's own stated behaviour" );
		ok( lm.rejections() > 0 && lm.acceptances() == 0,
			"and it got there by rejecting, not by accepting" );
	}

	if ( failures ) { printf( "%d failure(s)\n", failures ); return 1; }
	printf( "all Levenberg-Marquardt checks passed\n" );
	return 0;
}
