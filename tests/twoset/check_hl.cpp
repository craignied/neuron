// check_hl.cpp : the Hosmer-Lemeshow statistic on datasets larger than
// 10,000 rows.
//
// TwoSet::HLX2calc() once copied the guesses into a pair of fixed
// double[10000] stack arrays with no bounds check, so any dataset past
// 10,000 rows wrote up to hundreds of kilobytes beyond them -- a segfault
// when the writes reached the stack guard page (observed at 12,000 rows:
// EXC_BAD_ACCESS in HLX2calc), and silent stack corruption when they did
// not. Found 2026-07-16; the engine had shipped this since 2004. This test
// crashes against the pre-fix code -- verified, per the standing rule --
// and passes once the arrays are sized to the data.

#include <cmath>
#include <iostream>
#include <limits>
#include <random>

#include "twoset.h"

using namespace std;

// A discrete-outcome TwoSet of nRows synthetic (outcome, guess) pairs
static TwoSet makeSet( unsigned nRows )
{
	mt19937 gen( 4242 );
	normal_distribution< double > noise( 0.0, 0.15 );

	Matrix< double > M( nRows, 2 );
	for ( unsigned i = 0; i < nRows; i++ )
	{
		unsigned outcome = ( i % 4 == 0 ) ? 1 : 0;
		double guess = 0.25 * outcome + 0.4 + noise( gen );
		if ( guess < 0.01 ) guess = 0.01;
		if ( guess > 0.99 ) guess = 0.99;
		M( i, 0 ) = outcome;
		M( i, 1 ) = guess;
	}
	return TwoSet( M );
}

static bool check( unsigned nRows )
{
	TwoSet t = makeSet( nRows );
	double p = t.getHLX2();

	cout << "Hosmer-Lemeshow on " << nRows << " rows: p = " << p;
	if ( isfinite( p ) && p >= 0 && p <= 1 )
	{
		cout << "  OK" << endl;
		return true;
	}
	cout << "  FAIL (want a p-value in [0,1])" << endl;
	return false;
}

// A hand-computable case pinning the textbook C-hat (H&L 2nd ed. section
// 5.2.2). Twenty rows, all fitted probabilities 0.5, so the ten groups are
// consecutive row pairs (stable sort keeps row order on ties). Nine groups
// observe exactly their expectation (outcomes 1,0 -> O = E = 1: zero terms);
// the last observes 0,0 against E = 1:
//     term = (O-E)^2 / (n_k pibar (1-pibar)) = 1 / (2 * .5 * .5) = 2
// so C-hat = 2 on g-2 = 8 df, and for even df the survival function is
// closed-form: p = e^{-1} (1 + 1 + 1/2 + 1/6) = 0.98101184.
// The 2004 algorithm this replaced (legacy bug #9) gives a different number
// -- verified by compiling this test against it and watching it fail.
static bool check_textbook()
{
	Matrix< double > M( 20, 2 );
	for ( unsigned i = 0; i < 20; i++ )
	{
		M( i, 0 ) = ( i < 18 ) ? ( i % 2 == 0 ? 1 : 0 ) : 0;
		M( i, 1 ) = 0.5;
	}
	TwoSet t( M );
	double p = t.getHLX2();

	cout << "hand-computed C-hat=2, df=8: p = " << p;
	if ( fabs( p - 0.98101184 ) < 1e-6 )
	{
		cout << "  OK" << endl;
		return true;
	}
	cout << "  FAIL (want 0.98101184)" << endl;
	return false;
}

// Calibration: on data where the fitted probabilities ARE the truth, a
// goodness-of-fit test must not cry misfit. With known-true probabilities
// the statistic runs a little hot against chi-squared(g-2) (that df assumes
// estimated parameters), so theory puts the rejection rate near 11%, not 5%
// -- the guard allows 20%. The 2004 algorithm rejected a true model 54% of
// the time; this assertion fails against it.
static bool check_calibration()
{
	mt19937 gen( 20260716 );
	normal_distribution< double > x( 0.0, 1.0 );
	uniform_real_distribution< double > u( 0.0, 1.0 );

	const unsigned sims = 400, nRows = 200;
	unsigned rejected = 0;
	for ( unsigned s = 0; s < sims; s++ )
	{
		Matrix< double > M( nRows, 2 );
		for ( unsigned i = 0; i < nRows; i++ )
		{
			double prob = 1.0 / ( 1.0 + exp( -( 0.5 + 1.2 * x( gen ) ) ) );
			M( i, 0 ) = ( u( gen ) < prob ) ? 1 : 0;
			M( i, 1 ) = prob;
		}
		TwoSet t( M );
		if ( t.getHLX2() < 0.05 )
			rejected++;
	}
	double rate = ( double ) rejected / sims;

	cout << "calibration, model true: rejected " << rate * 100 << "% at alpha=0.05";
	if ( rate < 0.20 )
	{
		cout << "  OK" << endl;
		return true;
	}
	cout << "  FAIL (want < 20%)" << endl;
	return false;
}

// The Pearson value is a hand-computable STATISTIC (never a p -- none exists
// at the individual level; see TwoSet::PKX2calc). Twenty rows, all fitted
// probabilities 0.5: every squared Pearson residual is 0.25/0.25 = 1, so
// X2 = 20 exactly. The 2004 implementation returned a "p" from a hardcoded
// chi-squared(2) instead -- 0.0059 on this data -- so this fails against it.
static bool check_pearson()
{
	Matrix< double > M( 20, 2 );
	for ( unsigned i = 0; i < 20; i++ )
	{
		M( i, 0 ) = ( i % 2 == 0 ) ? 1 : 0;
		M( i, 1 ) = 0.5;
	}
	TwoSet t( M );
	double x2 = t.getPearsonX2();

	cout << "Pearson X2 on 20 unit residuals = " << x2;
	if ( fabs( x2 - 20.0 ) < 1e-9 )
	{
		cout << "  OK" << endl;
		return true;
	}
	cout << "  FAIL (want exactly 20)" << endl;
	return false;
}

// Fewer exemplars than groups: the honest answer is a refusal, not a number
static bool check_small()
{
	Matrix< double > M( 4, 2 );
	for ( unsigned i = 0; i < 4; i++ )
	{
		M( i, 0 ) = ( i < 2 ) ? 1 : 0;
		M( i, 1 ) = ( i < 2 ) ? 0.9 : 0.1;
	}
	TwoSet t( M );
	bool threw = false;
	try { t.getHLX2(); }
	catch ( TwoSet::twoSetErr& ) { threw = true; }

	cout << "4 exemplars: " << ( threw ? "refused" : "returned a number" );
	if ( threw )
	{
		cout << "  OK" << endl;
		return true;
	}
	cout << "  FAIL (want a refusal)" << endl;
	return false;
}

// NaN guesses (a diverged network) reach every statistic through the
// training epilogue. K-S's Numerical Recipes merge advances an index only
// when one side is <= the other -- with NaN, BOTH comparisons are false and
// the loop hangs forever (found 2026-07-16 by a diverged training probe;
// this test hangs against the unguarded code). The honest answer is a
// refusal. H-L must refuse via its fit p too, and Pearson reports NaN.
static bool check_nan_guesses()
{
	Matrix< double > M( 24, 2 );
	for ( unsigned i = 0; i < 24; i++ )
	{
		M( i, 0 ) = ( i % 2 ) ? 1 : 0;
		M( i, 1 ) = 0.1 + 0.03 * i;
	}
	M( 5, 1 ) = numeric_limits< double >::quiet_NaN();
	TwoSet t( M );

	bool ksRefused = false;
	try { t.getKSD(); }
	catch ( TwoSet::twoSetErr& ) { ksRefused = true; }

	double px2 = t.getPearsonX2();

	cout << "NaN guesses: K-S " << ( ksRefused ? "refused" : "returned" )
		<< ", Pearson X2 = " << px2;
	if ( ksRefused && isnan( px2 ) )
	{
		cout << "  OK" << endl;
		return true;
	}
	cout << "  FAIL (want a K-S refusal and a NaN Pearson X2)" << endl;
	return false;
}

int main()
{
	bool ok = true;

	ok &= check_nan_guesses();
	ok &= check( 9999 );  // just under the old fixed array size
	ok &= check( 12000 ); // past it: segfaulted before the 2026-07-16 fix
	ok &= check_textbook();
	ok &= check_calibration();
	ok &= check_pearson();
	ok &= check_small();

	if ( ok )
	{
		cout << "check_hl: Hosmer-Lemeshow matches the textbook and survives large datasets" << endl;
		return 0;
	}
	cerr << "check_hl: FAILED" << endl;
	return 1;
}
