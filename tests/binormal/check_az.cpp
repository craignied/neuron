// check_az.cpp : verifies the engine's statistical ROC area against signal
// detection theory.
//
// TwoSet::getStatROCarea() implements the binormal method of Wickens,
// Elementary Signal Detection Theory (Oxford, 2002), pp. 60-74: z-transform
// the false-alarm and hit rates, fit the zROC line zH = a + b*zF, and report
// Az = Phi( a / sqrt(1 + b^2) ).
//
// For Gaussian score populations N(mu0, s0) and N(mu1, s1) the theoretical
// area is Az = Phi( (mu1 - mu0) / sqrt(s0^2 + s1^2) ) — algebraically the
// same quantity, since a = (mu1-mu0)/s1 and b = s0/s1. This program draws
// large samples from known Gaussians, runs them through the engine's
// getStatROCarea(), and requires the result to match theory within a
// sampling tolerance. See docs/roc_theory.md.

#include <cmath>
#include <iostream>
#include <limits>
#include <random>

#include "twoset.h"

using namespace std;

static double phi( double z ) // standard normal CDF
{
	return 0.5 * erfc( -z / sqrt( 2.0 ) );
}

// Returns true if the engine's Az for two Gaussian score populations
// matches the theoretical value within tolerance
static bool check( double mu1, double s1, unsigned nPerClass, double tol )
{
	const double mu0 = 0.0, s0 = 1.0;

	mt19937 gen( 12345 );
	normal_distribution< double > negative( mu0, s0 ), positive( mu1, s1 );

	Matrix< double > M( 2 * nPerClass, 2 ); // col 0 = class, col 1 = score
	for ( unsigned i = 0; i < nPerClass; i++ )
	{
		M( 2 * i, 0 ) = 0;
		M( 2 * i, 1 ) = negative( gen );
		M( 2 * i + 1, 0 ) = 1;
		M( 2 * i + 1, 1 ) = positive( gen );
	}

	TwoSet assay( M );
	double az = assay.getStatROCarea();
	double theory = phi( ( mu1 - mu0 ) / sqrt( s0 * s0 + s1 * s1 ) );

	cout << "mu1=" << mu1 << " s1=" << s1
		<< "  engine Az=" << az
		<< "  theory Az=" << theory
		<< "  |diff|=" << fabs( az - theory );

	if ( fabs( az - theory ) <= tol )
	{
		cout << "  OK" << endl;
		return true;
	}
	cout << "  FAIL (tol " << tol << ")" << endl;
	return false;
}

// The variance of a set of identical values is zero. Computed by the textbook
// sumSquares - n*mean^2 form it is instead a roundoff residue of either sign,
// and a negative one made std() return NaN — which reached the zROC fit as a
// NaN error bar (a bin covering a flat run of the empirical ROC holds identical
// z values) and destroyed the area. Asserted here because it is the root of the
// resample discards this suite exists to keep away. See docs/roc_theory.md.
static bool check_var_identical()
{
	bool ok = true;
	// Values chosen to round either way: the first two are z-scores of common
	// hit rates, and before the fix one gave NaN and the other a fake 3e-08 SD.
	for ( double v : { 0.5244005127080409, -1.2815515655446004,
		2.3263478740408408, 0.0, 1.0 } )
	{
		vector< double > bin( 14, v );
		Population p( bin );
		double var = p.var(), sd = p.std();

		cout << "var of 14 identical " << v << " -> var=" << var << " std=" << sd;
		if ( isfinite( var ) && isfinite( sd ) && var >= 0 && sd < 1e-12 )
			cout << "  OK" << endl;
		else
		{
			cout << "  FAIL (want finite, >=0, ~0)" << endl;
			ok = false;
		}
	}
	return ok;
}

// Tied scores make flat runs in the empirical ROC — the regime that used to
// drive the (since-removed) binning degenerate. The fit must still yield a
// finite area and a finite fit p, and the bootstrap must discard no resample.
// Before the 2026-07-15 fixes this data failed 20% of binnings and 27% of
// resamples.
static bool check_ties( unsigned n0, unsigned n1, double mu1 )
{
	mt19937 gen( 7 );
	normal_distribution< double > negative( 0.0, 1.0 ), positive( mu1, 1.0 );

	Matrix< double > M( n0 + n1, 2 );
	unsigned r = 0;
	// Round the scores hard: ties are what a saturating model actually produces
	for ( unsigned i = 0; i < n0; i++, r++ )
	{
		M( r, 0 ) = 0;
		M( r, 1 ) = round( negative( gen ) * 20 ) / 20;
	}
	for ( unsigned i = 0; i < n1; i++, r++ )
	{
		M( r, 0 ) = 1;
		M( r, 1 ) = round( positive( gen ) * 20 ) / 20;
	}

	bool ok = true;

	// The fit itself must survive the ties
	TwoSet fit( M );
	TwoSet::ROCfit f = fit.getROCfit();
	cout << "tied scores, fit: Az=" << f.az << " from " << f.points
		<< " points, fit p=" << f.p;
	if ( f.valid && isfinite( f.az ) && isfinite( f.p ) )
		cout << "  OK" << endl;
	else
	{
		cout << "  FAIL (want a finite area and fit p)" << endl;
		ok = false;
	}

	// The bootstrap must keep every resample, and agree with Hanley-McNeil
	TwoSet assay( M );
	assay.setBootstrapResamples( 400 ); // enough to be a real check, quick in CI
	TwoSet::CI ci = assay.getStatCi();
	double hm = assay.getTrapSE();

	cout << "tied scores, bootstrap: resamples=" << ci.resamples
		<< " failed=" << ci.failures << " SE=" << ci.se
		<< "  (Hanley-McNeil SE=" << hm << ")";
	// Discards here are not random — they track ties — so they bias the
	// interval narrow. Demand none, and demand the SE stay near the
	// independent estimator rather than merely being finite.
	if ( ci.valid && ci.failures == 0 && ci.lo < ci.hi
		&& fabs( ci.se - hm ) / hm < 0.25 )
		cout << "  OK" << endl;
	else
	{
		cout << "  FAIL (want 0 failures, SE within 25% of H-M)" << endl;
		ok = false;
	}

	return ok;
}

// Verifies the bootstrap SE against simulation: the mean reported bootstrap SE
// over replicate samples should match the empirical SD of Az across those same
// replicates. This is the calibration check that matters now that the interval
// is resampled rather than propagated analytically (the delta-method SE this
// once asserted is retired — it tracked the bin count, not the sample size).
static bool check_se( double mu1, double s1, unsigned nPerClass, unsigned reps )
{
	mt19937 gen( 6789 );
	normal_distribution< double > negative( 0.0, 1.0 ), positive( mu1, s1 );

	double sumAz = 0, sumAz2 = 0, sumBoot = 0,
		sumT = 0, sumT2 = 0, sumHM = 0;
	for ( unsigned r = 0; r < reps; r++ )
	{
		Matrix< double > M( 2 * nPerClass, 2 );
		for ( unsigned i = 0; i < nPerClass; i++ )
		{
			M( 2 * i, 0 ) = 0;
			M( 2 * i, 1 ) = negative( gen );
			M( 2 * i + 1, 0 ) = 1;
			M( 2 * i + 1, 1 ) = positive( gen );
		}
		TwoSet assay( M );
		assay.setBootstrapResamples( 200 ); // reps x B — keep CI honest but quick
		double az = assay.getStatROCarea();
		sumAz += az;
		sumAz2 += az * az;
		sumBoot += assay.getStatCi().se;
		double t = assay.getTrapROCarea();
		sumT += t;
		sumT2 += t * t;
		sumHM += assay.getTrapSE();
	}
	double meanAz = sumAz / reps;
	double empSD = sqrt( sumAz2 / reps - meanAz * meanAz );
	double ratio = ( sumBoot / reps ) / empSD;

	double meanT = sumT / reps;
	double empSDT = sqrt( sumT2 / reps - meanT * meanT );
	double ratioHM = ( sumHM / reps ) / empSDT;

	cout << "SE calibration (bootstrap): empirical SD(Az)=" << empSD
		<< "  mean reported SE=" << sumBoot / reps << "  ratio=" << ratio << endl;
	cout << "SE calibration (Hanley-McNeil): empirical SD(trap)=" << empSDT
		<< "  mean reported SE=" << sumHM / reps << "  ratio=" << ratioHM;

	// Both should now be well calibrated; the window allows for the modest
	// number of replicates the empirical SD is itself estimated from.
	if ( ratio > 0.7 && ratio < 1.4 && ratioHM > 0.7 && ratioHM < 1.4 )
	{
		cout << "  OK" << endl;
		return true;
	}
	cout << "  FAIL (want 0.7-1.4 for both)" << endl;
	return false;
}

// A diverged network produces NaN guesses. The sweep must survive them:
// NaN != NaN, so a naive tie-group cumulation never consumes a NaN element
// -- the first diverged training probe found that as an INFINITE loop
// (2026-07-16; this test hangs against that code). The old per-threshold
// recount never counted a NaN row as predicted positive at any finite
// threshold, so the fix excludes NaN scores from the sweep while keeping
// every row in the class denominators.
static bool check_nan_guesses()
{
	mt19937 gen( 99 );
	normal_distribution< double > negative( 0.0, 1.0 ), positive( 1.5, 1.0 );

	Matrix< double > M( 60, 2 );
	for ( unsigned i = 0; i < 60; i++ )
	{
		M( i, 0 ) = ( i % 2 ) ? 1 : 0;
		M( i, 1 ) = ( i % 2 ) ? positive( gen ) : negative( gen );
	}
	// Poison a few guesses in both classes, the way divergence actually does
	M( 10, 1 ) = numeric_limits< double >::quiet_NaN();
	M( 11, 1 ) = numeric_limits< double >::quiet_NaN();
	M( 12, 1 ) = numeric_limits< double >::infinity();

	TwoSet t( M );
	t.setBootstrapResamples( 50 ); // exercise the resample path over NaNs too
	TwoSet::ROCfit f = t.getROCfit();

	cout << "NaN guesses: fit " << ( f.valid ? "valid" : "invalid" )
		<< ", Az = " << f.az << " from " << f.points << " points";
	if ( f.valid && isfinite( f.az ) && f.az > 0.5 && f.points < 60 )
	{
		cout << "  OK" << endl;
		return true;
	}
	cout << "  FAIL (want a finite fit over the finite scores)" << endl;
	return false;
}

int main()
{
	bool ok = true;

	// Equal variance (the classic d' case: Az = Phi(d'/sqrt(2)))
	ok &= check( 1.0, 1.0, 2000, 0.02 );
	// Unequal variance (the case the binormal slope b exists to handle)
	ok &= check( 2.0, 1.5, 2000, 0.02 );
	// Weak separation
	ok &= check( 0.5, 1.0, 2000, 0.02 );
	// A set of identical values has zero variance, not a NaN
	ok &= check_var_identical();
	// Tied scores must not cost binnings or bootstrap resamples
	ok &= check_ties( 100, 42, 0.7 );
	// NaN guesses (a diverged network) must not hang or poison the sweep
	ok &= check_nan_guesses();
	// Bootstrap SE matches sampling variability
	ok &= check_se( 1.0, 1.0, 200, 40 );

	if ( ok )
	{
		cout << "check_az: engine statistical ROC matches signal detection theory" << endl;
		return 0;
	}
	cerr << "check_az: FAILED" << endl;
	return 1;
}
