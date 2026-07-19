// check_wickens.cpp : the literature acceptance test for the binormal ROC.
//
// Wickens, Elementary Signal Detection Theory (Oxford, 2002), Table 5.1 (p. 84)
// is a published rating-scale dataset -- three-level confidence ratings from one
// practiced observer detecting a weak visual pattern (data of L. A. Olzak and
// P. Kramer) -- which Wickens then analyses by hand across Chapter 5. That makes
// it a self-verifying reference for this engine, exactly like the Hosmer-Lemeshow
// low-birth-weight log likelihood of -111.286 is for logistic regression: real
// published data with published answers, so the engine can be checked against the
// literature rather than against itself.
//
// Wickens' answers (Table 5.3 and pp. 89-90):
//   operating points  f = .762 .532 .335 .152 .061
//                     h = .933 .840 .746 .614 .420
//   zROC line (fitted BY EYE)  zH = 0.735 zF + 0.974
//   mu_s = a/b = 1.325,  sigma_s = 1/b = 1.360,  Az = Phi(a/sqrt(1+b^2)) = 0.784
//
// He notes a by-eye fit is nearly as accurate as a computer program (p. 89), and
// least squares on his own z-points agrees: a = 0.970, b = 0.729, Az = 0.7833.
// So the target for Az is 0.783-0.784.
//
// WHY THIS DATA IS THE RIGHT TEST: it is the rating case, which is structurally
// what neuron always has (thresholds swept across one score on one sample --
// see docs/roc_theory.md), and its six categories make it massively tied, which
// is the regime that was silently broken until 2026-07-15.
//
// NOT COVERED HERE, deliberately:
//   - Example 11.8 (p. 214), X^2 = 5.93 unequal vs 35.85 equal variance. That is
//     a multinomial goodness-of-fit over the rating categories, comparing two
//     fitted models. The engine computes no such statistic -- its chi-squared is
//     fitexy's line-fit chi-squared, a different quantity -- so there is nothing
//     to assert against. It would arrive with a maximum-likelihood fit (Phase 3).
//   - Example 11.1 (p. 204), se(Az) = 0.042. That is the analytic single-point
//     (yes/no) standard error, Eq. 11.7, for the case where parameters equal data
//     points. neuron's binormal path is multi-point, and its analytic SE was
//     deliberately removed as mis-specified; the interval is bootstrapped.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "twoset.h"
#include "utility.h"

using namespace std;

// Wickens Table 5.1 (p. 84). Six ordered response categories, N3..Y3.
static const unsigned NOISE[ 6 ]  = { 166, 161, 138, 128, 63, 43 };  // 699 trials
static const unsigned SIGNAL[ 6 ] = { 47, 65, 66, 92, 136, 294 };    // 700 trials

// Wickens' published Az for these data (p. 90)
static const double WICKENS_AZ = 0.784;

// How close the engine must come to it. Phase 2 landed the binomial error bar
// (Wickens Eq. 11.2 + 11.3, p. 202) and stopped binning, and the engine now
// answers 0.7839 against his 0.784 -- so this ratchet is at 0.002, and the
// binning that used to be worth 0.011 of Az is worth nothing at all.
//
// For the record, since it is the whole reason Phase 2 happened: before it, the
// eight binnings spanned 0.7783 to 0.7893 around this same 0.784, with the bin
// count alone deciding which, and both selection criteria choosing badly --
// "best p" took 5 bins (0.7785) and "best AUC" took 7 bins (0.7893, the furthest,
// on a fit whose p was 0.000, since maximising the area selects for a poor fit by
// construction). The tolerance then had to be 0.012 to admit either.
static const double AZ_TOL = 0.002;

static bool nearly( double got, double want, double tol )
{
	return fabs( got - want ) <= tol;
}

// The empirical operating points must be Wickens' -- this is the z-transform and
// the ROC sweep, checked independently of any line fitting. Listed in ascending
// false-alarm rate, the direction getTrapROCarea now walks the curve (origin ->
// (1,1)); Wickens' Table 5.3 tabulates the same points top-down.
static bool check_operating_points( TwoSet& t )
{
	static const double F[ 5 ] = { 0.061, 0.152, 0.335, 0.532, 0.762 };
	static const double H[ 5 ] = { 0.420, 0.614, 0.746, 0.840, 0.933 };

	t.getTrapROCarea(); // fills the curve points
	const vector< double >& x = t.getROCx();
	const vector< double >& y = t.getROCy();

	// Keep the interior points, in the order the sweep found them
	vector< double > f, h;
	for ( size_t i = 0; i < x.size(); i++ )
		if ( x[ i ] > 0 && x[ i ] < 1 && y[ i ] > 0 && y[ i ] < 1 )
		{
			f.push_back( x[ i ] );
			h.push_back( y[ i ] );
		}

	cout << "operating points vs Wickens Table 5.3 (p. 90):" << endl;
	if ( f.size() != 5 )
	{
		cout << "  FAIL: found " << f.size() << " interior points, want 5" << endl;
		return false;
	}

	bool ok = true;
	for ( unsigned i = 0; i < 5; i++ )
	{
		bool good = nearly( f[ i ], F[ i ], 0.001 ) && nearly( h[ i ], H[ i ], 0.001 );
		cout << "  f=" << f[ i ] << " (want " << F[ i ] << ")   h=" << h[ i ]
			<< " (want " << H[ i ] << ")" << ( good ? "  OK" : "  FAIL" ) << endl;
		ok &= good;
	}
	return ok;
}

int main()
{
	util::set_seed( 42 ); // the bootstrap draws; keep the run reproducible

	unsigned n = 0;
	for ( unsigned j = 0; j < 6; j++ )
		n += NOISE[ j ] + SIGNAL[ j ];

	// The rating IS the score: category j (1..6) becomes score j. This is the
	//    rating experiment expressed as the (known, score) pairs the engine takes.
	Matrix< double > m( n, 2 );
	unsigned r = 0;
	for ( unsigned j = 0; j < 6; j++ )
	{
		for ( unsigned i = 0; i < NOISE[ j ]; i++, r++ )
		{
			m( r, 0 ) = 0;
			m( r, 1 ) = j + 1;
		}
		for ( unsigned i = 0; i < SIGNAL[ j ]; i++, r++ )
		{
			m( r, 0 ) = 1;
			m( r, 1 ) = j + 1;
		}
	}

	cout << fixed << setprecision( 4 );
	cout << "Wickens Table 5.1: " << n << " trials (699 noise + 700 signal)" << endl;

	bool ok = true;

	TwoSet points( m );
	ok &= check_operating_points( points );

	// The reported fit must land on Wickens' published area
	TwoSet assay( m );
	assay.setBootstrapResamples( 400 );
	TwoSet::ROCfit f = assay.getROCfit();

	cout << "\nreported fit vs Wickens " << WICKENS_AZ << " (tolerance " << AZ_TOL << "):" << endl;
	bool good = f.valid && nearly( f.az, WICKENS_AZ, AZ_TOL );
	cout << "  Az=" << f.az << " from " << f.points << " operating points"
		<< "  diff=" << showpos << f.az - WICKENS_AZ << noshowpos
		<< "  fit p=" << f.p << ( good ? "  OK" : "  FAIL" ) << endl;
	ok &= good;
	// Wickens reads five criteria off these data (p. 90); so must we
	if ( f.points != 5 )
	{
		cout << "  FAIL: fitted " << f.points << " points, Wickens has 5" << endl;
		ok = false;
	}

	// The bootstrap interval must cover the published value. This is a coverage
	//    check against a real number from the literature, not against ourselves.
	TwoSet::CI ci = assay.getStatCi();
	bool covers = ci.valid && ci.lo <= WICKENS_AZ && WICKENS_AZ <= ci.hi;
	cout << "  95% CI " << ci.lo << " - " << ci.hi << " (" << ci.resamples
		<< " resamples, " << ci.failures << " failed) covers " << WICKENS_AZ
		<< ( covers ? "  OK" : "  FAIL" ) << endl;
	ok &= covers;
	// Tied data is what broke before; no resample may be silently dropped
	if ( ci.failures != 0 )
	{
		cout << "  FAIL: " << ci.failures << " resamples discarded on tied data" << endl;
		ok = false;
	}

	if ( ok )
	{
		cout << "\ncheck_wickens: engine agrees with Wickens' published analysis" << endl;
		return 0;
	}
	cerr << "\ncheck_wickens: FAILED" << endl;
	return 1;
}
