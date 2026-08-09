// LM, the Levenberg-Marquardt optimizer. See lm.h for the sources, the
// ownership boundary and every constant.

#include "stdafx.h" // For MSVC, must be first!

#include <cmath>

#include "lm.h"
#include "vector_ops.h"

// --- The published constants -------------------------------------------------
//
// Madsen, Nielsen & Tingleff (2004), section 3.2 and Algorithm 3.16. Section
// 3.2's footnote 3 gives tau: "use a small value, eg tau = 1e-6 if x_0 is
// believed to be a good approximation to x*. Otherwise, use tau = 1e-3 or even
// tau = 1." Randomized neural starting weights are NOT a good approximation, so
// 1e-3 is the cited default for this case. None of these is configurable.

const double LM::TAU = 1e-3;
const double LM::NU_INIT = 2.0;
const double LM::MU_FLOOR = 1.0 / 3.0;
const double LM::EPS_STEP = 1e-12;

// Declared policy, not published -- see lm.h and the source decision.
const unsigned LM::MAX_PARAMETERS = 512;
const unsigned LM::MAX_REJECTIONS = 20;

LM::LM()
{
	reset();
}

void LM::reset()
{
	accepted.clear();
	trial.clear();
	gradG.clear();
	trialG.clear();
	stepH.clear();
	rhs.clear();
	normalA.clear();
	trialA.clear();
	damped.clear();

	acceptedF = 0.0;
	mu = 0.0;
	nuValue = NU_INIT;
	gradMaxValue = 0.0;
	startedFlag = false;
	smallStep = false;

	iterationCount = accepted_ = rejected_ = factorFail = nonFinite_
		= cancelled_ = 0;
}

// Algorithm 3.16's rejection branch: mu := mu * nu; nu := 2 * nu.
//
//    In one place because THREE conditions reject -- a failed factorization, a
//    non-finite trial, and rho <= 0 -- and a damping schedule that differed
//    between them would be a method nobody named.
void LM::rejectStep()
{
	mu *= nuValue;
	nuValue *= 2.0;
	rejected_++;
}

static bool allFinite( const vector< double >& v )
{
	for ( size_t i = 0; i < v.size(); i++ )
		if ( !std::isfinite( v[ i ] ) )
			return false;
	return true;
}

static bool allFinite( const Matrix< double >& m )
{
	const double* p = m.begin();
	const double* end = m.end();
	for ( ; p != end; ++p )
		if ( !std::isfinite( *p ) )
			return false;
	return true;
}

double LM::iterate( LMObjective& f )
{
	// --- Initialization: Algorithm 3.16's lines above the while loop --------
	//
	//        A := J(x)'J(x);  g := J(x)'f(x)
	//        mu := tau * max{ a_ii };  nu := 2
	//
	//    `found := ( ||g||_inf <= eps_1 )` is deliberately absent: that is
	//    stopping, and Iterative owns it (STOP_GRADMAX is that criterion).
	if ( !startedFlag )
	{
		f.currentPoint( accepted );

		acceptedF = f.evaluateNormal( normalA, gradG );

		// THE STARTING POINT MUST BE FINITE, and this is the one place that
		//    throws for it: nothing has moved yet, so the refusal is clean.
		//    Every LATER non-finite evaluation is a trial, and a trial is
		//    REJECTED rather than fatal (see the trial branch below).
		if ( !std::isfinite( acceptedF ) || !allFinite( gradG )
			|| !allFinite( normalA ) )
			throw NotFinite();

		// mu_0 = tau * max_i { a_ii }, equation (3.14).
		double maxDiagonal = 0.0;
		for ( unsigned i = 0; i < normalA.rows(); i++ )
			if ( normalA( i, i ) > maxDiagonal )
				maxDiagonal = normalA( i, i );

		// A non-positive maximum diagonal means every curvature is zero, which
		//    for a least-squares objective means every sensitivity is zero at
		//    every exemplar. A silent positive floor would be an undeclared
		//    magic constant; a named refusal is testable.
		if ( !( maxDiagonal > 0.0 ) || !std::isfinite( maxDiagonal ) )
			throw NotFinite();

		mu = TAU * maxDiagonal;
		nuValue = NU_INIT;
		startedFlag = true;
	}

	// THE DEPARTING POINT. Both of these describe the point this iteration
	//    steps away from, which is what legacy pre-update reporting means and
	//    what the plan's architecture decision 7 requires of currGradMax. They
	//    are taken BEFORE any trial moves the weights.
	const double departingF = acceptedF;
	gradMaxValue = maxabs( gradG );

	const unsigned n = ( unsigned ) accepted.size();

	// --- The damping loop ---------------------------------------------------
	for ( unsigned attempt = 0; ; attempt++ )
	{
		// Cancellation is checked BETWEEN trials, because one iteration can
		//    traverse the training set several times. The last accepted point
		//    is restored exactly; a cancelled run is not left at a trial.
		if ( f.cancelled() )
		{
			cancelled_++;
			f.install( accepted );
			iterationCount++;
			return departingF;
		}

		// Solve ( A + mu I ) h_lm = -g, equation (3.13). solveSPD reads the
		//    upper triangle only and forms no inverse.
		damped = normalA;
		for ( unsigned i = 0; i < n; i++ )
			damped( i, i ) += mu;

		if ( rhs.size() != n ) rhs.resize( n );
		for ( unsigned i = 0; i < n; i++ )
			rhs[ i ] = -gradG[ i ];

		bool solved = true;
		try
		{
			damped.solveSPD( rhs, stepH );
		}
		catch ( const Matrix< double >::Singular& )
		{
			// Theory says A + mu I is positive definite for mu > 0; floating
			//    point can still fail Algorithm A.4's test when mu is tiny
			//    beside a badly conditioned A. Increasing mu is the directly
			//    analogous remedy to Marquardt's own damping increase, and the
			//    next factorization is strictly better conditioned. Declared
			//    adaptation, not published behavior.
			factorFail++;
			solved = false;
		}

		// --- CRITERION (3.15b), IN ITS PUBLISHED POSITION -------------------
		//
		//        if ||h_lm|| <= eps_2 ( ||x|| + eps_2 )  then  found := true
		//
		//    Algorithm 3.16 tests this immediately after the solve and BEFORE
		//    evaluating any trial, and so does this. It is a CONVERGENCE EXIT,
		//    not a rejection: the step has become numerically negligible against
		//    the parameters, which is what convergence of a damped Gauss-Newton
		//    method looks like from the inside.
		//
		//    Omitting it is what made LM-v1 report StepRejected at a gradient
		//    maximum of 2.2e-10 with bit-identical trial objectives: the run had
		//    converged, and the only exit it owned was a failure.
		//
		//    LM does not stop anything here. It records the fact, returns the
		//    departing objective like any other iteration, and Iterative -- the
		//    one owner of stopping -- reads it and ends the run as
		//    STOP_SMALL_STEP.
		if ( solved )
		{
			double stepNorm = 0.0, pointNorm = 0.0;
			for ( unsigned i = 0; i < n; i++ )
			{
				stepNorm += stepH[ i ] * stepH[ i ];
				pointNorm += accepted[ i ] * accepted[ i ];
			}
			stepNorm = sqrt( stepNorm );
			pointNorm = sqrt( pointNorm );

			if ( stepNorm <= EPS_STEP * ( pointNorm + EPS_STEP ) )
			{
				smallStep = true;
				iterationCount++;
				return departingF; // NO trial is evaluated
			}
		}

		if ( solved && !allFinite( stepH ) )
		{
			// A solve that produced garbage must not move the weights.
			nonFinite_++;
			solved = false;
		}

		if ( solved )
		{
			// x_new := x + h_lm
			if ( trial.size() != n ) trial.resize( n );
			for ( unsigned i = 0; i < n; i++ )
				trial[ i ] = accepted[ i ] + stepH[ i ];

			f.install( trial );
			const double trialF = f.evaluateNormal( trialA, trialG );

			// FINITENESS IS TESTED BEFORE THE GAIN RATIO, and it is tested for
			//    ALL THREE members of "non-finite". A bare `rho > 0` would
			//    ACCEPT -infinity as an infinitely good improvement: that is
			//    precisely the defect a post-screen audit found in the BB
			//    prototype's acceptance comparison, which nineteen tests had
			//    missed. A non-finite trial is a REJECTION, never fatal, and
			//    the weights go back before anything else happens.
			if ( !std::isfinite( trialF ) || !allFinite( trialG )
				|| !allFinite( trialA ) )
			{
				nonFinite_++;
				f.install( accepted );
			}
			else
			{
				// The gain ratio,
				//
				//      rho = ( F(x) - F(x_new) ) / ( L(0) - L(h_lm) )
				//
				//    with the denominator L(0) - L(h_lm) = 0.5 h'( mu h - g ).
				//    The source proves both h'h and -h'g positive, so this is
				//    positive; the tests assert that invariant rather than
				//    guarding a branch that cannot be reached.
				double denominator = 0.0;
				for ( unsigned i = 0; i < n; i++ )
					denominator += stepH[ i ] * ( mu * stepH[ i ] - gradG[ i ] );
				denominator *= 0.5;

				const double rho = ( acceptedF - trialF ) / denominator;

				if ( rho > 0.0 ) // { step acceptable }
				{
					// x := x_new, and A and g are already those of x_new: the
					//    trial evaluation computed them. Algorithm 3.16
					//    recomputes them here instead; taking them at the trial
					//    point is a WORK-SCHEDULING choice that leaves the
					//    accepted sequence identical, because acceptance
					//    depends only on F(x_new) and A, g are taken at x_new
					//    either way. That is a claim about the code, so it is
					//    pinned by a test rather than left as a comment.
					accepted = trial;
					acceptedF = trialF;
					normalA = trialA;
					gradG = trialG;

					// mu := mu * max( 1/3, 1 - (2 rho - 1)^3 );  nu := 2
					//    Written as a product rather than through pow(), which
					//    would introduce an error the published formula does
					//    not have.
					const double t = 2.0 * rho - 1.0;
					double factor = 1.0 - t * t * t;
					if ( factor < MU_FLOOR ) factor = MU_FLOOR;
					mu *= factor;
					nuValue = NU_INIT;

					accepted_++;
					iterationCount++;
					return departingF;
				}

				// rho <= 0: not acceptable. Restore, then damp harder.
				f.install( accepted );
			}
		}

		rejectStep();

		if ( attempt + 1 >= MAX_REJECTIONS )
		{
			// The accepted parameters are already installed by every path that
			//    reaches here, so the model is left exactly where it was.
			f.install( accepted );
			iterationCount++;
			throw StepRejected();
		}
	}
}
