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

// How close the engine must come to it.
//
// **THIS IS A RATCHET.** The binning is arbitrary, and it is worth 0.011 of Az on
// this dataset: the search's eight binnings span 0.7783 to 0.7893 around a truth
// of 0.784, and the bin count alone decides which. Worse, both selection criteria
// choose badly -- "best p" takes 5 bins (Az 0.7785, one of the furthest) and
// "best AUC" takes 7 bins (Az 0.7893, the furthest, on a fit whose p is 0.000,
// since maximising the area selects for a bad fit by construction). So the
// tolerance today can only be loose enough to admit whichever binning is chosen.
//
// ROADMAP 3 Phase 2 replaces the within-bin standard deviations with Wickens' own
// binomial error bar (Eq. 11.2 + 11.3, p. 202) and drops binning entirely, fitting
// the operating points themselves. **When it lands, tighten this to 0.002 and
// delete the paragraph above.** If Phase 2 cannot hold 0.002 on published data
// analysed by hand, it has not worked.
static const double AZ_TOL = 0.012;

static bool nearly( double got, double want, double tol )
{
	return fabs( got - want ) <= tol;
}

// The empirical operating points must be Wickens' -- this is the z-transform and
// the ROC sweep, checked independently of any line fitting.
static bool check_operating_points( TwoSet& t )
{
	static const double F[ 5 ] = { 0.762, 0.532, 0.335, 0.152, 0.061 };
	static const double H[ 5 ] = { 0.933, 0.840, 0.746, 0.614, 0.420 };

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

	// Every binning, printed as a diagnostic: the spread IS the artifact Phase 2
	//    removes, and seeing it beats describing it.
	//
	// These lines are NOT asserted, and they are not stable across builds. The fit
	//    is ill-conditioned on tied data: binning 1399 exemplars that hold only six
	//    distinct scores produces bins of identical z values, whose standard
	//    deviation is legitimately zero, which chixy() weights BIG (1e30). The
	//    objective is then dominated by 1e30 terms and brent balances on the last
	//    bits. Measured: the same source at -O0 and -O3 gives Az 0.7793 vs 0.7789
	//    at 4 bins, and at 10 bins one converges while the other cannot bracket a
	//    root -- floating-point contraction alone. That fragility is another count
	//    against the binning, and Phase 2 removes its cause rather than tuning it:
	//    five real points, five analytic error bars, no zeros, no BIG.
	cout << "\nAz by bin count (Wickens: " << WICKENS_AZ << "):" << endl;
	TwoSet sweep( m );
	for ( unsigned nb = 3; nb <= 10; nb++ )
	{
		sweep.setNbins( nb );
		sweep.setNbinsSetsSize( true );
		sweep.invalidate();
		try
		{
			double az = sweep.getStatROCarea();
			cout << "  " << setw( 2 ) << nb << " bins  Az=" << az
				<< "  diff=" << showpos << az - WICKENS_AZ << noshowpos
				<< "  fit p=" << sweep.getStatP() << endl;
		}
		catch ( exception& e )
		{
			// Non-fatal by design: one binning failing must not cost the others
			cout << "  " << setw( 2 ) << nb << " bins  no fit (" << e.what() << ")" << endl;
		}
	}

	// The reported fits must land near Wickens' published area
	TwoSet assay( m );
	assay.setROCSearchFlag( true );
	assay.setBootstrapResamples( 400 );
	TwoSet::ROCfit bp = assay.getBestPfit(), ba = assay.getBestAUCfit();

	cout << "\nreported fits vs Wickens " << WICKENS_AZ << " (tolerance " << AZ_TOL << "):" << endl;
	for ( int which = 0; which < 2; which++ )
	{
		const TwoSet::ROCfit& f = which ? ba : bp;
		const char* name = which ? "best AUC" : "best p  ";
		bool good = f.valid && nearly( f.az, WICKENS_AZ, AZ_TOL );
		cout << "  " << name << ": Az=" << f.az << " (" << f.nBins << " bins)"
			<< "  diff=" << showpos << f.az - WICKENS_AZ << noshowpos
			<< ( good ? "  OK" : "  FAIL" ) << endl;
		ok &= good;
	}

	// The bootstrap interval must cover the published value. This is a coverage
	//    check against a real number from the literature, not against ourselves.
	TwoSet::CI ci = assay.getBestAUCci();
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
