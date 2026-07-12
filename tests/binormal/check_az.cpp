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

// Verifies the delta-method SE of Az against simulation: the mean reported SE
// over replicate samples should be on the scale of the empirical SD of Az.
// (The covariance term is unavailable on the binned fitexy path, so we accept
// a generous calibration window rather than exact agreement.)
static bool check_se( double mu1, double s1, unsigned nPerClass, unsigned reps )
{
	mt19937 gen( 6789 );
	normal_distribution< double > negative( 0.0, 1.0 ), positive( mu1, s1 );

	double sumAz = 0, sumAz2 = 0, sumSE = 0,
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
		double az = assay.getStatROCarea();
		sumAz += az;
		sumAz2 += az * az;
		sumSE += assay.getStatAzSE();
		double t = assay.getTrapROCarea();
		sumT += t;
		sumT2 += t * t;
		sumHM += assay.getTrapSE();
	}
	double meanAz = sumAz / reps;
	double empSD = sqrt( sumAz2 / reps - meanAz * meanAz );
	double ratio = ( sumSE / reps ) / empSD;

	double meanT = sumT / reps;
	double empSDT = sqrt( sumT2 / reps - meanT * meanT );
	double ratioHM = ( sumHM / reps ) / empSDT;

	cout << "SE calibration (delta): empirical SD(Az)=" << empSD
		<< "  mean reported SE=" << sumSE / reps << "  ratio=" << ratio << endl;
	cout << "SE calibration (Hanley-McNeil): empirical SD(trap)=" << empSDT
		<< "  mean reported SE=" << sumHM / reps << "  ratio=" << ratioHM;

	// Delta method drops the a-b covariance on the binned path — generous
	// window; Hanley-McNeil should be well calibrated — tight window.
	if ( ratio > 0.4 && ratio < 2.5 && ratioHM > 0.7 && ratioHM < 1.4 )
	{
		cout << "  OK" << endl;
		return true;
	}
	cout << "  FAIL (delta want 0.4-2.5, H-M want 0.7-1.4)" << endl;
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
	// Delta-method SE roughly matches sampling variability
	ok &= check_se( 1.0, 1.0, 300, 80 );

	if ( ok )
	{
		cout << "check_az: engine statistical ROC matches signal detection theory" << endl;
		return 0;
	}
	cerr << "check_az: FAILED" << endl;
	return 1;
}
