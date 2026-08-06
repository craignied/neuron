// Limited-memory BFGS. See lbfgs.h for the sources, the ownership boundary and
// why the line search is a complete strong-Wolfe search rather than a
// backtracking one.

#include "stdafx.h" // For MSVC, must be first!

#include <cmath>
#include <limits>

#include "lbfgs.h"
#include "vector_ops.h"

// The published constants. Every one of them was fixed in
//    docs/learning_research/lbfgs_source_decision.md before a single
//    measurement was taken.
const double LBFGS::C1 = 1e-4;        // Nocedal & Wright p. 33
const double LBFGS::C2 = 0.9;         // Nocedal & Wright p. 34
const double LBFGS::ALPHA_MAX = 1e10; // declared policy
const double LBFGS::EXPAND = 2.0;     // declared policy
const double LBFGS::CURV_EPS = 1e-10; // declared policy, on top of s'y > 0
const unsigned LBFGS::MAX_EVALS = 20; // declared policy

LBFGS::LBFGS()
{
	memory = 5; // Liu & Nocedal (1989) section 5: 3 <= m <= 7 is adequate
	reset();
}

void LBFGS::setMemory( const unsigned m )
{
	if ( m == 0 )
		throw Ineligible( "L-BFGS memory length must be at least 1" );
	memory = m;
	reset();
}

void LBFGS::reset()
{
	S.assign( memory, vector< double >() );
	Y.assign( memory, vector< double >() );
	Rho.assign( memory, 0.0 );
	next = 0;
	count = 0;

	w.clear();
	g.clear();
	q.clear();
	alphas.clear();
	dir.clear();
	trialW.clear();
	trialG.clear();
	sNew.clear();
	yNew.clear();

	f = 0;
	gMax = 0;
	acceptedF = 0;
	startedFlag = false;

	objEvals = gradEvals = failures = resets = curvSkips = cancels = 0;
	searchStartEvals = 0;
}

void LBFGS::copyConfigurationFrom( const LBFGS& rhs )
{
	memory = rhs.memory;
	reset();
}

void LBFGS::resetHistory()
{
	count = 0;
	next = 0;
	resets++;
}

unsigned LBFGS::slot( const unsigned i ) const
{
	// i = 0 is the OLDEST live pair
	return ( next + memory - count + i ) % memory;
}

double LBFGS::scaling() const
{
	// Nocedal & Wright equation (7.20):
	//    $\gamma_k = \frac{{\bf s}_{k-1}^T {\bf y}_{k-1}}
	//                     {{\bf y}_{k-1}^T {\bf y}_{k-1}}$
	// With no history there is no pair to scale from, and H_k^0 = I.
	if ( count == 0 )
		return 1.0;

	unsigned newest = slot( count - 1 );
	double yy = dotprod( Y[ newest ], Y[ newest ] );
	if ( !( yy > 0 ) || !std::isfinite( yy ) )
		return 1.0;

	double sy = dotprod( S[ newest ], Y[ newest ] );
	double gamma = sy / yy;
	return std::isfinite( gamma ) && gamma > 0 ? gamma : 1.0;
}

bool LBFGS::pushPair( const vector< double >& s, const vector< double >& y )
{
	if ( s.size() != y.size() || s.empty() )
		throw nvec::SizeMismatch();

	double sy = dotprod( s, y );
	double yy = dotprod( y, y );

	// THE CURVATURE CONDITION. The published one is $ {\bf s}^T{\bf y} > 0 $
	//    (Nocedal & Wright equation 6.7), which is what makes the BFGS update
	//    positive definite. The relative floor is a declared numerical
	//    safeguard and not a change to the mathematics: rho = 1/(s'y) has to be
	//    finite and representable, and a pair that clears zero by one ulp makes
	//    the recursion meaningless. A pair that fails is SKIPPED -- Liu &
	//    Nocedal drop pairs, they do not damp them.
	if ( !std::isfinite( sy ) || !std::isfinite( yy )
		|| !( sy > CURV_EPS * yy ) )
	{
		curvSkips++;
		return false;
	}

	S[ next ] = s;
	Y[ next ] = y;
	Rho[ next ] = 1.0 / sy;
	next = ( next + 1 ) % memory;
	if ( count < memory )
		count++;

	return true;
}

// Nocedal & Wright Algorithm 7.4, the two-loop recursion. Writes H_k * g.
//
//    q <- g
//    for i = k-1 ... k-m:      alpha_i <- rho_i s_i' q ;  q <- q - alpha_i y_i
//    r <- gamma_k * q
//    for i = k-m ... k-1:      beta <- rho_i y_i' r ;  r <- r + (alpha_i - beta) s_i
//
// The two loops run in OPPOSITE directions and that is the whole content of the
// recursion: alpha_i computed newest-first is consumed oldest-first. Running
// either one the wrong way gives a vector that is still finite, still a plausible
// descent direction, and not H_k g -- which is why
// tests/network/check_lbfgs.cpp compares this against a dense inverse-BFGS
// matrix built independently.
void LBFGS::applyInverseHessian( const vector< double >& gIn,
	vector< double >& result ) const
{
	q = gIn;
	alphas.assign( count, 0.0 );

	// First loop: NEWEST to OLDEST
	for ( int i = ( int ) count - 1; i >= 0; i-- )
	{
		unsigned k = slot( ( unsigned ) i );
		double alpha = Rho[ k ] * dotprod( S[ k ], q );
		alphas[ i ] = alpha;
		axpy( -alpha, Y[ k ], q );
	}

	// H_k^0 = gamma_k I, applied as a scalar -- never materialized
	result = q;
	double gamma = scaling();
	if ( gamma != 1.0 )
		result *= gamma;

	// Second loop: OLDEST to NEWEST
	for ( unsigned i = 0; i < count; i++ )
	{
		unsigned k = slot( i );
		double beta = Rho[ k ] * dotprod( Y[ k ], result );
		axpy( alphas[ i ] - beta, S[ k ], result );
	}
}

void LBFGS::computeDirection()
{
	if ( count == 0 )
	{
		// Nocedal & Wright Algorithm 7.5 step 2: with no curvature information
		//    the method is steepest descent.
		dir = g;
		dir *= -1.0;
		return;
	}

	applyInverseHessian( g, dir );
	dir *= -1.0;
}

bool LBFGS::isDescent() const
{
	if ( dir.size() != g.size() || dir.empty() )
		return false;

	for ( unsigned i = 0; i < dir.size(); i++ )
		if ( !std::isfinite( dir[ i ] ) )
			return false;

	double dphi0 = dotprod( g, dir );
	return std::isfinite( dphi0 ) && dphi0 < 0;
}

bool LBFGS::evaluateAt( LBFGSObjective& obj, const double a, double& phi,
	double& dphi )
{
	// CANCELLATION IS CHECKED BEFORE EVERY TRIAL EVALUATION. One evaluation is
	//    a whole pass over the training set; a cancelled run must not pay for
	//    the rest of the search.
	if ( obj.cancelled() )
	{
		cancels++;
		return false;
	}

	if ( objEvals - searchStartEvals >= MAX_EVALS )
		return false; // the declared per-search evaluation ceiling

	// The trial point w + a*d, formed without allocating: trialW is sized once
	//    per run and axpy writes into it.
	trialW = w;
	axpy( a, dir, trialW );

	obj.install( trialW );
	phi = obj.evaluate( trialG );
	objEvals++;
	gradEvals++;

	if ( !std::isfinite( phi ) )
	{
		// NOT a stop. Algorithm 3.5's first test is "is phi(a) too large?", and
		//    +infinity answers it: the step was too long, so the search brackets
		//    and shrinks. Treating a non-finite trial as a failure would abandon
		//    a search that is working correctly.
		phi = numeric_limits< double >::infinity();
		dphi = 0;
		return true;
	}

	dphi = dotprod( trialG, dir );
	if ( !std::isfinite( dphi ) )
	{
		phi = numeric_limits< double >::infinity();
		dphi = 0;
	}

	return true;
}

// Nocedal & Wright Algorithm 3.6, "zoom", with BISECTION as its interpolation.
//    Algorithm 3.6 names "quadratic, cubic, or bisection" and requires only
//    that the trial lie between alpha_lo and alpha_hi; bisection is the
//    instantiation with no free parameters and no safeguard of its own to get
//    wrong. With c2 = 0.9 the unit step is usually accepted in the bracketing
//    phase, so this is entered rarely.
bool LBFGS::zoom( LBFGSObjective& obj, double aLo, double fLo,
	double aHi, const double phi0, const double dphi0,
	double& aOut, double& fOut )
{
	for ( ; ; )
	{
		// A bracket this narrow cannot yield a new point in double precision,
		//    and bisecting it until the evaluation ceiling would spend whole
		//    passes over the data learning nothing.
		if ( fabs( aHi - aLo ) <= 1e-16 * max( 1.0, fabs( aLo ) ) )
			return false;

		double a = 0.5 * ( aLo + aHi );

		double phi, dphi;
		if ( !evaluateAt( obj, a, phi, dphi ) )
			return false;

		// Sufficient decrease, Nocedal & Wright condition (3.6a):
		//    $\phi(\alpha) \le \phi(0) + c_1 \alpha \phi'(0)$
		if ( phi > phi0 + C1 * a * dphi0 || phi >= fLo )
			aHi = a;
		else
		{
			// STRONG curvature, condition (3.7b):
			//    $|\phi'(\alpha)| \le c_2 |\phi'(0)|$
			if ( fabs( dphi ) <= -C2 * dphi0 )
			{
				aOut = a;
				fOut = phi;
				return true;
			}

			if ( dphi * ( aHi - aLo ) >= 0 )
				aHi = aLo;

			aLo = a;
			fLo = phi;
		}
	}
}

// Nocedal & Wright Algorithm 3.5, the bracketing phase.
bool LBFGS::lineSearch( LBFGSObjective& obj, const double alpha1,
	const double dphi0 )
{
	const double phi0 = f;
	searchStartEvals = objEvals; // the ceiling is per search

	double aPrev = 0, fPrev = phi0;
	double a = alpha1;

	for ( unsigned i = 1; ; i++ )
	{
		double phi, dphi;
		if ( !evaluateAt( obj, a, phi, dphi ) )
			return false;

		// (3.6a) violated, or no better than the previous trial: the minimizer
		//    is bracketed between the previous step and this one.
		if ( phi > phi0 + C1 * a * dphi0 || ( i > 1 && phi >= fPrev ) )
		{
			double aOut, fOut;
			if ( !zoom( obj, aPrev, fPrev, a, phi0, dphi0, aOut, fOut ) )
				return false;
			acceptedF = fOut;
			return true;
		}

		// (3.7b) satisfied: accept this step.
		if ( fabs( dphi ) <= -C2 * dphi0 )
		{
			acceptedF = phi;
			return true;
		}

		// The slope has turned uphill, so the minimizer is behind this step.
		//    Note the ARGUMENT ORDER: the bracket is (a, aPrev), not
		//    (aPrev, a) -- Algorithm 3.5 is explicit that lo is the end with
		//    the lower objective, and here that is the new point.
		if ( dphi >= 0 )
		{
			double aOut, fOut;
			if ( !zoom( obj, a, phi, aPrev, phi0, dphi0, aOut, fOut ) )
				return false;
			acceptedF = fOut;
			return true;
		}

		aPrev = a;
		fPrev = phi;

		if ( a >= ALPHA_MAX )
			return false; // the declared bracketing ceiling

		a *= EXPAND;
		if ( a > ALPHA_MAX )
			a = ALPHA_MAX;
	}
}

double LBFGS::iterate( LBFGSObjective& obj )
{
	// The first call of a run evaluates the starting point. A direct research
	//    call that bypasses train() therefore initializes here rather than
	//    reading an uninitialized accepted point.
	if ( !startedFlag )
	{
		obj.currentPoint( w );
		if ( w.empty() )
			throw Ineligible( "L-BFGS needs a model with parameters" );
		g.assign( w.size(), 0.0 ); // sized once; evaluate writes into it
		f = obj.evaluate( g );
		objEvals++;
		gradEvals++;
		startedFlag = true;
	}

	// STEP 1, and it happens before anything transforms the gradient: the value
	//    the engine's gradient stopping rule reads is maxabs of the RAW
	//    gradient at the point this iteration departs from.
	gMax = maxabs( g );

	// LEGACY PRE-UPDATE REPORTING: the objective at the point whose gradient
	//    drives this step, matching every other optimizer in the engine.
	const double reported = f;

	computeDirection();

	if ( !isDescent() )
	{
		// A non-finite or uphill direction means the stored curvature is no
		//    longer trustworthy. Drop it and fall back to steepest descent.
		resetHistory();
		computeDirection();
		if ( !isDescent() )
			return reported; // a zero gradient: there is no direction to take
	}

	// THE INITIAL STEP. Unit for a quasi-Newton direction (Nocedal & Wright
	//    p. 142 -- trying anything else first forfeits superlinear
	//    convergence); scaled by the gradient's 1-norm when the direction is
	//    plain steepest descent and therefore not well scaled (N&W p. 59).
	double alpha1 = 1.0;
	if ( count == 0 )
	{
		double norm1 = 0;
		for ( unsigned i = 0; i < g.size(); i++ )
			norm1 += fabs( g[ i ] );
		if ( norm1 > 0 && std::isfinite( norm1 ) )
			alpha1 = min( 1.0, 1.0 / norm1 );
	}

	const double dphi0 = dotprod( g, dir );

	if ( !lineSearch( obj, alpha1, dphi0 ) )
	{
		// EXACT RESTORATION, by installing the stored accepted vector rather
		//    than by undoing arithmetic: a step that was subtracted back off
		//    would not return to the same bits.
		obj.install( w );
		failures++;
		// Drop the history so the next outer iteration retries from steepest
		//    descent. If that fails too the run makes no progress and ends at
		//    the engine's ceiling, which the convergence contract already calls
		//    a failure to converge. No new stop reason is invented here --
		//    Iterative keeps stopping ownership.
		resetHistory();
		return reported;
	}

	// Accepted. trialW / trialG are installed and hold the new point.
	sNew = trialW;
	sNew -= w;
	yNew = trialG;
	yNew -= g;

	pushPair( sNew, yNew ); // skipped silently and counted when curvature fails

	w = trialW;
	g = trialG;
	f = acceptedF;

	return reported;
}
