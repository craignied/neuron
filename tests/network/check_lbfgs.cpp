// check_lbfgs.cpp : the research-only L-BFGS prototype (src/lbfgs.*).
//
// The published pieces are checked against INDEPENDENT constructions, not
// against the same code run twice:
//
//   * the two-loop recursion (Nocedal & Wright Algorithm 7.4) against a DENSE
//     inverse-BFGS matrix built here from the recursion's defining update
//     (N&W equations 7.16-7.19). The dense oracle is written with plain
//     scalar loops on purpose -- it must not share a single line with the code
//     it is judging, and that is worth more here than the numerical
//     vocabulary's readability (standing rule 4 asks that the reason be said
//     beside the scalar code; this is the reason);
//
//   * both strong-Wolfe conditions on every accepted step, computed from
//     OUTSIDE the optimizer. The accepted step p = w_new - w_prev is
//     observable, and alpha cancels out of both conditions when they are
//     written in terms of p:
//         (3.6a)  f(w+p) <= f(w) + c1 * g(w)'p
//         (3.7b)  |g(w+p)'p| <= c2 * |g(w)'p|
//     so nothing has to be exposed for this to be checked, and the check
//     cannot accidentally re-run the search's own arithmetic.
//
// WHAT IS DELIBERATELY NOT CLAIMED. The non-descent fallback has two halves,
// and only one of them is reachable. With the curvature condition enforced,
// the two-loop recursion yields a positive-definite H, so g'(-Hg) < 0 in exact
// arithmetic and the UPHILL half is defensive rather than triggerable. The
// NON-FINITE half is reachable and is tested. Saying that a guard is
// unreachable is a claim about the code, so it is stated here rather than
// asserted by a test that would pass whatever the guard did.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "backprop.h"
#include "bareprop.h"
#include "dataset.h"
#include "lbfgs.h"
#include "logistic.h"
#include "simpleprop.h"
#include "utility.h"

using namespace std;

static int failures = 0;

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

struct Hush {
	ostringstream sink;
	ostream& prev;
	Hush() : prev( util::screen() ) { util::set_screen( sink ); }
	~Hush() { util::set_screen( prev ); }
};

// ===========================================================================
// A small positive-definite quadratic, and the objective wrapper L-BFGS drives.
//
//     f(x) = 0.5 x'A x - b'x        grad f(x) = A x - b
//
// A is symmetric positive definite and deliberately ILL-CONDITIONED, so a
// method that is really using curvature is distinguishable from one that is
// not.

static const unsigned N = 4;

static const double A[ N ][ N ] = {
	{ 4.0, 1.0, 0.0, 0.5 },
	{ 1.0, 3.0, 0.5, 0.0 },
	{ 0.0, 0.5, 20.0, 1.0 },
	{ 0.5, 0.0, 1.0, 0.25 }
};
static const double B[ N ] = { 1.0, -2.0, 0.5, 3.0 };

class Quadratic : public LBFGSObjective {
public:
	Quadratic() : x( N, 0.0 ), evals( 0 ), cancelAfter( 0 ), poison( false ) { }

	virtual void currentPoint( vector< double >& w ) const { w = x; }
	virtual void install( const vector< double >& w ) { x = w; }

	virtual double evaluate( vector< double >& g )
	{
		evals++;
		g.assign( N, 0.0 );
		double f = 0;
		for ( unsigned i = 0; i < N; i++ )
		{
			double ax = 0;
			for ( unsigned j = 0; j < N; j++ )
				ax += A[ i ][ j ] * x[ j ];
			g[ i ] = ax - B[ i ];
			f += 0.5 * x[ i ] * ax - B[ i ] * x[ i ];
		}
		if ( poison )
			g[ 0 ] = numeric_limits< double >::quiet_NaN();

		history.push_back( x );
		values.push_back( f );
		grads.push_back( g );
		return f;
	}

	virtual bool cancelled() const
	{ return cancelAfter && evals >= cancelAfter; }

	vector< double > x;
	unsigned evals;
	unsigned cancelAfter;         // 0 = never
	bool poison;                  // return a non-finite gradient
	vector< vector< double > > history, grads;
	vector< double > values;
};

// An objective with no minimum along any descent direction: f = -|x|^2 is
// unbounded below, so the bracketing phase expands to ALPHA_MAX and the search
// must FAIL rather than run forever.
class Unbounded : public LBFGSObjective {
public:
	Unbounded() : x( N, 1.0 ), evals( 0 ), cancelAfter( 0 ) { }
	virtual void currentPoint( vector< double >& w ) const { w = x; }
	virtual void install( const vector< double >& w ) { x = w; }
	virtual double evaluate( vector< double >& g )
	{
		evals++;
		g.assign( N, 0.0 );
		double f = 0;
		for ( unsigned i = 0; i < N; i++ ) { f -= x[ i ] * x[ i ]; g[ i ] = -2 * x[ i ]; }
		return f;
	}
	virtual bool cancelled() const
	{ return cancelAfter && evals >= cancelAfter; }
	vector< double > x;
	unsigned evals;
	unsigned cancelAfter;
};

// A NEARLY LINEAR objective:  f(x) = 0.5 eps |x|^2 - b'x,  eps tiny.
//
//    THIS FIXTURE EXISTS BECAUSE THE QUADRATIC ABOVE DOES NOT DISCRIMINATE.
//    Measured: removing the curvature test from the bracketing phase entirely
//    -- Armijo-only acceptance -- left every strong-Wolfe assertion on the
//    quadratic passing, because there the unit step happens to satisfy
//    curvature anyway. An assertion that holds whether or not the mechanism
//    exists guards nothing.
//
//    Here the slope barely changes over any short step, so an Armijo-only
//    search accepts the very first trial with |phi'(alpha)| ~ |phi'(0)| --
//    a gross violation of (3.7b) -- while a real strong-Wolfe search must
//    EXPAND the bracket about ten times before curvature is satisfied. The
//    accepted step therefore differs by two orders of magnitude between the
//    two, and the curvature assertion below can tell them apart.
class NearlyLinear : public LBFGSObjective {
public:
	NearlyLinear() : x( N, 0.0 ), evals( 0 ) { }
	virtual void currentPoint( vector< double >& w ) const { w = x; }
	virtual void install( const vector< double >& w ) { x = w; }
	virtual double evaluate( vector< double >& g )
	{
		evals++;
		g.assign( N, 0.0 );
		double f = 0;
		for ( unsigned i = 0; i < N; i++ )
		{
			g[ i ] = EPS * x[ i ] - 1.0;
			f += 0.5 * EPS * x[ i ] * x[ i ] - x[ i ];
		}
		return f;
	}
	virtual bool cancelled() const { return false; }

	static const double EPS;
	vector< double > x;
	unsigned evals;
};
const double NearlyLinear::EPS = 1e-3;

// A probe that reaches LBFGS's published pieces
class OpenLBFGS : public LBFGS {
public:
	using LBFGS::applyInverseHessian;
	using LBFGS::pushPair;
	using LBFGS::resetHistory;
	using LBFGS::scaling;
};

// ===========================================================================
// THE DENSE ORACLE. H is built from the limited-memory BFGS definition,
// N&W equations 7.16-7.19: start from H^0 = gamma I and apply the stored pairs
// OLDEST to NEWEST,
//
//     H <- (I - rho s y') H (I - rho y s') + rho s s'
//
// Plain loops, no shared code with lbfgs.cpp. See the file header.

static void denseInverseHessian( const vector< vector< double > >& S,
	const vector< vector< double > >& Y, double gamma,
	vector< vector< double > >& H )
{
	unsigned n = S.empty() ? N : ( unsigned ) S[ 0 ].size();

	H.assign( n, vector< double >( n, 0.0 ) );
	for ( unsigned i = 0; i < n; i++ )
		H[ i ][ i ] = gamma;

	for ( unsigned k = 0; k < S.size(); k++ )
	{
		const vector< double >& s = S[ k ];
		const vector< double >& y = Y[ k ];

		double sy = 0;
		for ( unsigned i = 0; i < n; i++ ) sy += s[ i ] * y[ i ];
		double rho = 1.0 / sy;

		// V = I - rho y s'  (so that H <- V' H V + rho s s')
		vector< vector< double > > V( n, vector< double >( n, 0.0 ) );
		for ( unsigned i = 0; i < n; i++ )
			for ( unsigned j = 0; j < n; j++ )
				V[ i ][ j ] = ( i == j ? 1.0 : 0.0 ) - rho * y[ i ] * s[ j ];

		// T = V' H
		vector< vector< double > > T( n, vector< double >( n, 0.0 ) );
		for ( unsigned i = 0; i < n; i++ )
			for ( unsigned j = 0; j < n; j++ )
			{
				double acc = 0;
				for ( unsigned p = 0; p < n; p++ )
					acc += V[ p ][ i ] * H[ p ][ j ];
				T[ i ][ j ] = acc;
			}

		// H = T V + rho s s'
		for ( unsigned i = 0; i < n; i++ )
			for ( unsigned j = 0; j < n; j++ )
			{
				double acc = 0;
				for ( unsigned p = 0; p < n; p++ )
					acc += T[ i ][ p ] * V[ p ][ j ];
				H[ i ][ j ] = acc + rho * s[ i ] * s[ j ];
			}
	}
}

static vector< double > applyDense( const vector< vector< double > >& H,
	const vector< double >& g )
{
	vector< double > r( g.size(), 0.0 );
	for ( unsigned i = 0; i < g.size(); i++ )
		for ( unsigned j = 0; j < g.size(); j++ )
			r[ i ] += H[ i ][ j ] * g[ j ];
	return r;
}

static double maxDiff( const vector< double >& a, const vector< double >& b )
{
	if ( a.size() != b.size() ) return 1e300;
	double d = 0;
	for ( unsigned i = 0; i < a.size(); i++ )
		d = max( d, fabs( a[ i ] - b[ i ] ) );
	return d;
}

static double dot( const vector< double >& a, const vector< double >& b )
{
	double s = 0;
	for ( unsigned i = 0; i < a.size(); i++ ) s += a[ i ] * b[ i ];
	return s;
}

// The exact minimizer of the quadratic, by Gaussian elimination with partial
//    pivoting. Independent of everything under test: it is the answer L-BFGS
//    is supposed to find, obtained another way.
static vector< double > solveExactly()
{
	double M[ N ][ N + 1 ];
	for ( unsigned i = 0; i < N; i++ )
	{
		for ( unsigned j = 0; j < N; j++ ) M[ i ][ j ] = A[ i ][ j ];
		M[ i ][ N ] = B[ i ];
	}

	for ( unsigned c = 0; c < N; c++ )
	{
		unsigned pivot = c;
		for ( unsigned r = c + 1; r < N; r++ )
			if ( fabs( M[ r ][ c ] ) > fabs( M[ pivot ][ c ] ) ) pivot = r;
		for ( unsigned j = 0; j <= N; j++ )
			swap( M[ c ][ j ], M[ pivot ][ j ] );

		for ( unsigned r = 0; r < N; r++ )
		{
			if ( r == c ) continue;
			double factor = M[ r ][ c ] / M[ c ][ c ];
			for ( unsigned j = c; j <= N; j++ )
				M[ r ][ j ] -= factor * M[ c ][ j ];
		}
	}

	vector< double > x( N );
	for ( unsigned i = 0; i < N; i++ ) x[ i ] = M[ i ][ N ] / M[ i ][ i ];
	return x;
}

// Deterministic, non-degenerate curvature pairs
static void makePair( unsigned k, vector< double >& s, vector< double >& y )
{
	s.assign( N, 0.0 );
	y.assign( N, 0.0 );
	for ( unsigned i = 0; i < N; i++ )
		s[ i ] = 0.1 * ( 1 + i ) + 0.03 * k * ( i % 3 == 0 ? 1 : -1 );
	// y = A s keeps s'y = s'As > 0 for the positive-definite A above, so the
	//    pairs are legitimate curvature information rather than arbitrary
	//    vectors that happen to pass the test.
	for ( unsigned i = 0; i < N; i++ )
		for ( unsigned j = 0; j < N; j++ )
			y[ i ] += A[ i ][ j ] * s[ j ];
}

// ===========================================================================
// 5 + 6 + 8. Two-loop recursion, history rollover, initial scaling.

static void checkTwoLoop( unsigned memory, unsigned pushed )
{
	ostringstream label;
	label << "two-loop m=" << memory << " with " << pushed << " pairs pushed";

	OpenLBFGS lb;
	lb.setMemory( memory );

	vector< vector< double > > liveS, liveY;
	for ( unsigned k = 0; k < pushed; k++ )
	{
		vector< double > s, y;
		makePair( k, s, y );
		bool stored = lb.pushPair( s, y );
		expect( stored, label.str() + ": a genuine curvature pair is accepted" );
		liveS.push_back( s );
		liveY.push_back( y );
	}

	// The ring buffer keeps the newest min(pushed, m)
	unsigned live = min( pushed, memory );
	expect( lb.pairs() == live, label.str() + ": stores min(pushed, m) pairs" );
	while ( liveS.size() > live )
	{
		liveS.erase( liveS.begin() );
		liveY.erase( liveY.begin() );
	}

	// The equation-7.20 scaling, computed independently from the NEWEST pair
	double gamma = 1.0;
	if ( live )
	{
		const vector< double >& s = liveS.back();
		const vector< double >& y = liveY.back();
		gamma = dot( s, y ) / dot( y, y );
	}
	expect( fabs( lb.scaling() - gamma ) <= 1e-15 * max( 1.0, fabs( gamma ) ),
		label.str() + ": gamma is s'y / y'y from the newest pair (N&W 7.20)" );

	// The recursion against the dense matrix
	vector< vector< double > > H;
	denseInverseHessian( liveS, liveY, gamma, H );

	vector< double > g( N );
	for ( unsigned i = 0; i < N; i++ )
		g[ i ] = 0.7 - 0.3 * i + 0.11 * ( i % 2 );

	vector< double > got;
	lb.applyInverseHessian( g, got );
	vector< double > want = applyDense( H, g );

	// Not vacuous: with any live pair H differs from gamma*I, so the recursion
	//    is being asked for something a scalar multiply could not produce.
	if ( live )
	{
		vector< double > scaled = g;
		for ( unsigned i = 0; i < N; i++ ) scaled[ i ] *= gamma;
		expect( maxDiff( want, scaled ) > 1e-6,
			label.str() + ": the oracle is not merely gamma*I (the test can fail)" );
	}

	double d = maxDiff( got, want );
	if ( d > 1e-10 )
		cout << "         two-loop max difference " << setprecision( 6 ) << d << endl;
	expect( d <= 1e-10,
		label.str() + ": the recursion equals the dense inverse-BFGS matrix" );
}

// ===========================================================================
// 7. Invalid curvature pairs are SKIPPED, not stored and not damped.

static void checkCurvature()
{
	OpenLBFGS lb;
	lb.setMemory( 5 );

	vector< double > s, y;
	makePair( 0, s, y );
	expect( lb.pushPair( s, y ), "a positive-curvature pair is stored" );
	expect( lb.pairs() == 1, "... and the history holds it" );

	// s'y < 0
	vector< double > yNeg = y;
	for ( unsigned i = 0; i < N; i++ ) yNeg[ i ] = -yNeg[ i ];
	expect( !lb.pushPair( s, yNeg ), "a negative-curvature pair is rejected" );
	expect( lb.pairs() == 1, "... and the history is unchanged" );
	expect( lb.curvatureRejections() == 1, "... and the rejection is counted" );

	// s'y == 0 exactly
	vector< double > sOrth( N, 0.0 ), yOrth( N, 0.0 );
	sOrth[ 0 ] = 1.0;
	yOrth[ 1 ] = 1.0;
	expect( !lb.pushPair( sOrth, yOrth ), "an orthogonal pair (s'y = 0) is rejected" );
	expect( lb.pairs() == 1, "... and the history is still unchanged" );

	// s'y > 0 but far below the relative floor: the published condition alone
	//    would accept this, and rho = 1/s'y would be enormous.
	vector< double > sTiny( N, 0.0 ), yTiny( N, 0.0 );
	sTiny[ 0 ] = 1.0;
	yTiny[ 0 ] = 1e-18;
	yTiny[ 1 ] = 1.0;   // makes y'y about 1, so s'y / y'y is about 1e-18
	expect( !lb.pushPair( sTiny, yTiny ),
		"a pair that clears s'y > 0 but not the relative floor is rejected" );

	// resetHistory drops everything and is counted
	lb.resetHistory();
	expect( lb.pairs() == 0, "resetHistory drops the whole history" );
	expect( lb.historyResets() == 1, "... and is counted" );
}

// ===========================================================================
// 9 + 10 + 12 + 13. The line search: both Wolfe conditions on every accepted
// step, the evaluation ceiling, and exact restoration on failure and on
// cancellation.

static void checkLineSearch()
{
	Quadratic q;
	LBFGS lb;
	lb.setMemory( 5 );

	unsigned accepted = 0;
	for ( unsigned it = 0; it < 20; it++ )
	{
		vector< double > wPrev = q.x;
		unsigned before = ( unsigned ) q.history.size();

		lb.iterate( q );

		if ( q.x == wPrev )
			continue;                    // no step taken this iteration
		accepted++;

		// The step actually taken, and the objective/gradient at both ends.
		vector< double > p( N );
		for ( unsigned i = 0; i < N; i++ ) p[ i ] = q.x[ i ] - wPrev[ i ];

		// f and g at w_prev: the evaluation that opened this iteration is the
		//    one before `before`, i.e. the accepted point carried in from the
		//    previous iteration. Recompute independently instead of hunting
		//    for it -- fewer assumptions.
		vector< double > gPrev( N, 0.0 );
		double fPrev = 0;
		for ( unsigned i = 0; i < N; i++ )
		{
			double ax = 0;
			for ( unsigned j = 0; j < N; j++ ) ax += A[ i ][ j ] * wPrev[ j ];
			gPrev[ i ] = ax - B[ i ];
			fPrev += 0.5 * wPrev[ i ] * ax - B[ i ] * wPrev[ i ];
		}
		vector< double > gNew( N, 0.0 );
		double fNew = 0;
		for ( unsigned i = 0; i < N; i++ )
		{
			double ax = 0;
			for ( unsigned j = 0; j < N; j++ ) ax += A[ i ][ j ] * q.x[ j ];
			gNew[ i ] = ax - B[ i ];
			fNew += 0.5 * q.x[ i ] * ax - B[ i ] * q.x[ i ];
		}

		double slope0 = dot( gPrev, p );
		double slope1 = dot( gNew, p );

		expect( slope0 < 0, "the accepted step is a descent direction" );

		// (3.6a) sufficient decrease
		expect( fNew <= fPrev + LBFGS::C1 * slope0 + 1e-12,
			"accepted step satisfies sufficient decrease (N&W 3.6a)" );

		// (3.7b) STRONG curvature -- the two-sided bound a backtracking search
		//    would not give
		expect( fabs( slope1 ) <= LBFGS::C2 * fabs( slope0 ) + 1e-12,
			"accepted step satisfies the strong curvature condition (N&W 3.7b)" );

		// The declared per-search evaluation ceiling
		unsigned spent = ( unsigned ) q.history.size() - before;
		expect( spent <= LBFGS::MAX_EVALS,
			"the line search stayed inside its evaluation ceiling" );
	}
	expect( accepted >= 5, "the quadratic run accepted several steps" );

	// IT SOLVED THE PROBLEM. Not "the gradient got small" -- the iterate is
	//    compared against the exact minimizer, obtained here by Gaussian
	//    elimination with partial pivoting, which shares nothing with the
	//    optimizer.
	vector< double > xStar = solveExactly();
	double scale = 0, err = 0;
	for ( unsigned i = 0; i < N; i++ )
	{
		scale = max( scale, fabs( xStar[ i ] ) );
		err = max( err, fabs( q.x[ i ] - xStar[ i ] ) );
	}
	// The bound is derived, not chosen to pass: for a quadratic,
	//    |x - x*| <= ||A^-1|| |g| = |g| / lambda_min, and this fixture's
	//    smallest eigenvalue is about 0.126, so a gradient at the objective's
	//    floating-point floor leaves a relative error near 1e-8. 1e-6 sits
	//    two orders above that and eight orders below "did not converge".
	if ( !( err <= 1e-6 * scale ) )
		cout << "         iterate differs from the exact minimizer by "
			<< setprecision( 6 ) << err << " (scale " << scale << ")" << endl;
	expect( err <= 1e-6 * scale,
		"L-BFGS reached the exact minimizer of the quadratic" );

	// ... and it reached the exact minimum VALUE. At the solution Ax* = b, so
	//    f* = 0.5 x*'Ax* - b'x* = -0.5 b'x*, with no matrix multiply needed.
	double fStar = -0.5 * dot( vector< double >( B, B + N ), xStar );
	double fFinal = 0;
	for ( unsigned i = 0; i < N; i++ )
	{
		double ax = 0;
		for ( unsigned j = 0; j < N; j++ ) ax += A[ i ][ j ] * q.x[ j ];
		fFinal += 0.5 * q.x[ i ] * ax - B[ i ] * q.x[ i ];
	}
	expect( fabs( fFinal - fStar ) <= 1e-12 * fabs( fStar ),
		"... and the exact minimum value" );

	// AT THE FLOATING-POINT FLOOR THE SEARCH FAILS, AND THAT IS THE CONTRACT.
	//    Once f is at its double-precision floor no step can DEMONSTRATE
	//    sufficient decrease, so the search spends its declared ceiling and
	//    fails. What matters is what happens then: the weights must not move,
	//    and the history must be dropped. Measured on this fixture -- the run
	//    above converges by iteration 16 and every iteration after it is this
	//    case. A real run stops long before, on a rule Iterative owns; this
	//    asserts that the failure path is safe when it does not.
	unsigned long long failuresBefore = lb.lineSearchFailures();
	vector< double > atFloor = q.x;
	lb.iterate( q );
	expect( q.x == atFloor,
		"a search that cannot improve on a converged point moves no weight" );
	expect( lb.lineSearchFailures() > failuresBefore,
		"... and reports the failure rather than claiming a step" );
	expect( lb.pairs() == 0, "... and drops the curvature history" );

	// --- THE CURVATURE CONDITION, on a fixture that can tell ------------
	//    See the NearlyLinear note: an Armijo-only search accepts the first
	//    trial here, and that step violates (3.7b) by a wide margin.
	{
		NearlyLinear nl;
		LBFGS ln;
		vector< double > w0 = nl.x;
		ln.iterate( nl );

		vector< double > p( N );
		bool moved = false;
		for ( unsigned i = 0; i < N; i++ )
		{
			p[ i ] = nl.x[ i ] - w0[ i ];
			if ( p[ i ] != 0 ) moved = true;
		}
		expect( moved, "the nearly linear fixture takes a step" );

		vector< double > g0( N ), g1( N );
		double f0 = 0, f1 = 0;
		for ( unsigned i = 0; i < N; i++ )
		{
			g0[ i ] = NearlyLinear::EPS * w0[ i ] - 1.0;
			f0 += 0.5 * NearlyLinear::EPS * w0[ i ] * w0[ i ] - w0[ i ];
			g1[ i ] = NearlyLinear::EPS * nl.x[ i ] - 1.0;
			f1 += 0.5 * NearlyLinear::EPS * nl.x[ i ] * nl.x[ i ] - nl.x[ i ];
		}
		double slope0 = dot( g0, p ), slope1 = dot( g1, p );

		expect( f1 <= f0 + LBFGS::C1 * slope0 + 1e-12,
			"nearly linear: the accepted step satisfies sufficient decrease" );

		double ratio = fabs( slope1 ) / fabs( slope0 );
		if ( !( ratio <= LBFGS::C2 + 1e-12 ) )
			cout << "         |phi'(alpha)| / |phi'(0)| = " << setprecision( 6 )
				<< ratio << ", c2 = " << LBFGS::C2 << endl;
		expect( ratio <= LBFGS::C2 + 1e-12,
			"nearly linear: the accepted step satisfies the STRONG curvature "
			"condition, which an Armijo-only search would not" );

		// The bracket really did have to expand: a search that accepted its
		//    first trial would have taken a step about 500x shorter.
		expect( nl.evals >= 5,
			"... having expanded the bracket rather than accepting the first trial" );
	}

	// --- failure: no minimum along the direction -------------------------
	Unbounded u;
	LBFGS lf;
	vector< double > w0 = u.x;
	lf.iterate( u );
	expect( u.x == w0,
		"a line search with no acceptable step restores the weights exactly" );
	expect( lf.lineSearchFailures() == 1, "... and the failure is counted" );
	expect( lf.pairs() == 0, "... and the history is dropped" );
	expect( u.evals <= LBFGS::MAX_EVALS + 1,
		"... having spent no more than the declared evaluation ceiling" );

	// --- cancellation, checked BEFORE the first trial of a search ---------
	//    A quasi-Newton step is usually accepted on its first trial, so a test
	//    that waited for a second one would be testing whether the search
	//    happened to be slow. Both entry points are covered instead: here the
	//    very first check, and below a cancellation part-way through a search
	//    that genuinely needs several evaluations.
	Quadratic c;
	LBFGS lc;
	lc.iterate( c );                 // one clean iteration to leave an accepted point
	vector< double > wAccepted = c.x;
	unsigned evalsAtCancel = c.evals;
	c.cancelAfter = c.evals;         // cancelled() is true at the next check
	lc.iterate( c );
	expect( c.x == wAccepted,
		"cancellation before a trial restores the accepted weights exactly" );
	expect( c.evals == evalsAtCancel,
		"... having spent no evaluation after the cancellation" );
	expect( lc.cancellations() >= 1, "... and is counted" );

	// --- cancellation PART WAY THROUGH a multi-evaluation search ----------
	Unbounded u2;
	LBFGS lu2;
	vector< double > wu2 = u2.x;
	u2.cancelAfter = 3;              // the bracketing phase needs many more
	lu2.iterate( u2 );
	expect( u2.x == wu2,
		"cancellation during a trial restores the accepted weights exactly" );
	expect( u2.evals <= 4,
		"... and stopped the search rather than running it to its ceiling" );
	expect( lu2.cancellations() >= 1, "... and is counted" );
}

// ===========================================================================
// 11. The reachable half of the non-descent fallback.

static void checkNonFiniteDirection()
{
	// Poisoned from the start, so it is the ACCEPTED gradient that is
	//    non-finite and therefore the DIRECTION that is. Poisoning later would
	//    only corrupt trial gradients, which is a different path: the accepted
	//    gradient would still be finite and the direction would be fine.
	Quadratic q;
	q.poison = true;
	LBFGS lb;
	vector< double > w = q.x;

	lb.iterate( q );

	expect( lb.historyResets() == 1,
		"a non-finite direction drops the curvature history" );
	expect( q.x == w,
		"... and takes no step, leaving the weights where they were" );
	expect( q.evals == 1,
		"... having evaluated only the starting point, never a trial" );

	// The control: the SAME objective without the poison does take a step, so
	//    "no step" above is the guard firing and not the fixture being inert.
	Quadratic clean;
	LBFGS lc;
	vector< double > w2 = clean.x;
	lc.iterate( clean );
	expect( clean.x != w2 && lc.historyResets() == 0,
		"... while the unpoisoned control takes a step and resets nothing" );
}

// ===========================================================================
// 15. Per-run reset and configuration copy.

static DataSet makeData( unsigned n )
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
	Matrix< double >& r = d.getRawMatrix();
	d.setTrainMatrix( r );
	return d;
}

template < class NET >
class NetProbe : public NET {
public:
	const LBFGS& optimizer() const { return this->lbfgs; }
	LBFGS& optimizer() { return this->lbfgs; }
	double gradMaxNow() { return this->getGradMax(); }
	unsigned packed() const { return this->packedSize(); }
	void packW( vector< double >& w ) const { this->packWeights( w ); }
	double evalAt( vector< double >& g ) { return this->batchObjectiveGradient( g ); }
};

template < class NET >
static void setupNeural( NetProbe< NET >& p, DataSet& d, unsigned seed )
{
	Hush quiet;
	p.setDataSet( d );
	p.setHidden( 3 );
	p.setHistory( false );
	p.setLastop( false );
	p.setLogPrint( false );
	p.setQuiet( true );
	p.setXEerror();
	p.setBatchEpoch( true );
	p.setAutoStepSize( false );
	p.setWeightDecay( true );
	p.setDecay( 5e-5 );
	p.setEta( 0.05 );
	p.setTrainingType( Network::TRAIN_LBFGS );
	p.setMinStop( false );
	p.setChangeStop( false );
	p.setWindowStop( false );
	p.setGradStop( false );
	p.setMaxIterations( 25 );
	util::set_seed( seed );
	p.randomize();
}

static void checkLifecycle()
{
	DataSet d = makeData( 60 );

	NetProbe< SimpleProp > p;
	setupNeural( p, d, 20260806 );
	p.optimizer().setMemory( 7 );

	double f1;
	{ Hush quiet; f1 = p.train(); }
	expect( p.optimizer().objectiveEvaluations() > 0,
		"a run evaluates the objective through the boundary" );
	unsigned long long firstRun = p.optimizer().objectiveEvaluations();
	vector< double > w1;
	p.packW( w1 );

	// THE RESET, asserted exactly. Returning to the identical starting weights
	//    and running again must reproduce the first run in every respect --
	//    same evaluation count, same objective, same end weights. It can only
	//    do that if the curvature history, the accepted point and the counters
	//    were all reset by prepareRun(); a run that inherited any of them would
	//    take different steps from the first iteration onward.
	util::set_seed( 20260806 );
	{ Hush quiet; p.randomize(); }
	double f2;
	{ Hush quiet; f2 = p.train(); }
	vector< double > w2;
	p.packW( w2 );

	expect( p.optimizer().objectiveEvaluations() == firstRun,
		"a second run from the same start spends the identical evaluations" );
	expect( f1 == f2 && w1 == w2,
		"... and reaches the identical objective and end weights" );
	expect( p.optimizer().getMemory() == 7,
		"the configured memory length survives a per-run reset" );

	// A copy carries the CONFIGURATION and not the working history
	NetProbe< SimpleProp > clone;
	clone = p;
	expect( clone.optimizer().getMemory() == 7,
		"a copy carries the configured memory length" );
	expect( clone.optimizer().pairs() == 0
			&& clone.optimizer().objectiveEvaluations() == 0,
		"a copy does not carry the original's mid-run history or counters" );
}

// ===========================================================================
// 14. currGradMax is the RAW gradient at the point the step departed from.

static void checkGradMaxIsRaw()
{
	DataSet d = makeData( 60 );

	NetProbe< SimpleProp > p;
	setupNeural( p, d, 4242 );
	p.setMaxIterations( 0 ); // exactly one iteration
	p.setGradStop( true );
	p.setGradMaxLimit( 0.0 );

	// The raw gradient at the STARTING weights, computed independently through
	//    the boundary before the run touches anything.
	vector< double > g;
	p.evalAt( g );
	double want = 0;
	for ( unsigned i = 0; i < g.size(); i++ ) want = max( want, fabs( g[ i ] ) );

	{ Hush quiet; p.train(); }

	expect( want > 1e-8, "the starting gradient is not zero" );
	expect( fabs( p.gradMaxNow() - want ) <= 1e-15 * max( 1.0, want ),
		"currGradMax is maxabs of the RAW gradient at the departing point" );
}

// ===========================================================================
// 16. Refusals.

static void checkRefusals()
{
	DataSet d = makeData( 60 );

	// On-line mode
	{
		NetProbe< SimpleProp > p;
		setupNeural( p, d, 7 );
		p.setBatchEpoch( false );
		bool threw = false;
		try { Hush quiet; p.train(); }
		catch ( LBFGS::Ineligible& ) { threw = true; }
		expect( threw, "L-BFGS refuses on-line training" );
	}

	// The automatic step-size search
	{
		NetProbe< SimpleProp > p;
		setupNeural( p, d, 7 );
		p.setAutoStepSize( true );
		bool threw = false;
		try { Hush quiet; p.train(); }
		catch ( LBFGS::Ineligible& ) { threw = true; }
		expect( threw, "L-BFGS refuses the automatic step-size search" );
	}

	// A memory length of zero
	{
		LBFGS lb;
		bool threw = false;
		try { lb.setMemory( 0 ); } catch ( LBFGS::Ineligible& ) { threw = true; }
		expect( threw, "L-BFGS refuses a memory length of zero" );
	}
}

// ===========================================================================
// 17. Fixed-start integration for the three neural models.

template < class NET >
static void checkIntegration( const string& label, DataSet& d )
{
	NetProbe< NET > a, b;
	setupNeural( a, d, 31415 );
	setupNeural( b, d, 31415 );

	vector< double > w0a, w0b;
	a.packW( w0a );
	b.packW( w0b );
	expect( w0a == w0b && !w0a.empty(),
		label + ": both arms start from identical weights" );

	vector< double > g0;
	double f0 = a.evalAt( g0 );

	double fa, fb;
	{ Hush quiet; fa = a.train(); }
	{ Hush quiet; fb = b.train(); }

	expect( std::isfinite( fa ), label + ": the run ends with a finite objective" );
	expect( fa < f0, label + ": the objective fell below its starting value" );

	// Determinism: the same start must give the same finish, bit for bit.
	vector< double > w1a, w1b;
	a.packW( w1a );
	b.packW( w1b );
	expect( fa == fb && w1a == w1b,
		label + ": the same fixed start reaches the identical end state" );

	// It really used the line search, i.e. more full passes than iterations
	expect( a.optimizer().objectiveEvaluations() >= a.getIterations(),
		label + ": every outer iteration cost at least one full evaluation" );

	ostringstream s;
	s << label << ": " << a.getIterations() << " iterations, "
		<< a.optimizer().objectiveEvaluations() << " full evaluations, objective "
		<< setprecision( 8 ) << f0 << " -> " << fa;
	cout << "     " << s.str() << endl;
}

// ---------------------------------------------------------------------------

template < class NET >
static void checkBackPropIntegration( const string& label, DataSet& d )
{
	NetProbe< NET > a;
	{
		Hush quiet;
		vector< unsigned > layers( 1, 3 );
		a.setDataSet( d );
		a.setHidden( layers );
		a.setHistory( false );
		a.setLastop( false );
		a.setLogPrint( false );
		a.setQuiet( true );
		a.setXEerror();
		a.setBatchEpoch( true );
		a.setAutoStepSize( false );
		a.setWeightDecay( true );
		a.setDecay( 5e-5 );
		a.setEta( 0.05 );
		a.setTrainingType( Network::TRAIN_LBFGS );
		a.setMinStop( false );
		a.setChangeStop( false );
		a.setWindowStop( false );
		a.setGradStop( false );
		a.setMaxIterations( 25 );
		util::set_seed( 31415 );
		a.randomize();
	}

	vector< double > g0;
	double f0 = a.evalAt( g0 );
	double fa;
	{ Hush quiet; fa = a.train(); }

	expect( std::isfinite( fa ) && fa < f0,
		label + ": the objective fell to a finite value from a fixed start" );
	expect( a.optimizer().objectiveEvaluations() >= a.getIterations(),
		label + ": every outer iteration cost at least one full evaluation" );

	cout << "     " << label << ": " << a.getIterations() << " iterations, "
		<< a.optimizer().objectiveEvaluations() << " full evaluations, objective "
		<< setprecision( 8 ) << f0 << " -> " << fa << endl;
}

int main()
{
	cout << "L-BFGS (research prototype)" << endl;

	// The published core, against an independent dense oracle
	checkTwoLoop( 1, 1 );
	checkTwoLoop( 1, 4 );   // rollover on a one-slot buffer
	checkTwoLoop( 5, 3 );   // partially filled
	checkTwoLoop( 5, 5 );   // exactly full
	checkTwoLoop( 5, 9 );   // rollover
	checkTwoLoop( 20, 7 );  // partially filled, larger memory

	checkCurvature();
	checkLineSearch();
	checkNonFiniteDirection();
	checkLifecycle();
	checkGradMaxIsRaw();
	checkRefusals();

	DataSet d = makeData( 60 );
	checkIntegration< SimpleProp >( "SimpleProp", d );
	checkIntegration< BareProp >( "BareProp", d );
	checkBackPropIntegration< BackProp >( "BackProp", d );

	cout << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}

// ===========================================================================
// SABOTAGE LOG. Each was applied to src/lbfgs.cpp, watched to fail after a
// visible recompilation of that translation unit, then restored and watched to
// pass again.
//
// 1. THE SECOND TWO-LOOP PASS RUN NEWEST-TO-OLDEST instead of oldest-to-newest
//    (Nocedal & Wright Algorithm 7.4's second loop). Failed all four
//    multi-pair two-loop cases -- m=5 with 3, 5 and 9 pairs and m=20 with 7 --
//    at a maximum difference of 3.3e-2 against the dense oracle, and carried
//    through to the quadratic, which no longer reached its minimizer. THE
//    CONTROL: both m=1 cases PASSED, because with a single stored pair the
//    order of the loop cannot matter. A sabotage that failed those too would
//    have meant the test was detecting something other than the order.
//
// 2. THE CURVATURE TEST REMOVED from the bracketing phase, so the search
//    accepts on sufficient decrease alone -- Armijo backtracking wearing
//    Wolfe's name, which is exactly what the plan forbids.
//
//    FIRST ATTEMPT CAUGHT NOTHING IT CLAIMED TO. Every strong-Wolfe assertion
//    on the quadratic fixture PASSED under this sabotage: there the unit step
//    satisfies curvature anyway, so the assertion held whether or not the
//    mechanism existed. The sabotage was caught only indirectly, by the
//    unbounded fixture no longer failing its search. An assertion that passes
//    with the mechanism deleted guards nothing, so the NearlyLinear fixture
//    was added -- a near-linear objective on which an Armijo-only search
//    accepts its first trial while a real strong-Wolfe search must expand the
//    bracket about ten times.
//
//    Re-run against that fixture, the sabotage failed the curvature assertion
//    directly, at |phi'(alpha)|/|phi'(0)| = 0.99975 against c2 = 0.9, and the
//    SUFFICIENT-DECREASE assertion beside it still passed -- the control, since
//    Armijo satisfies that one by construction.
