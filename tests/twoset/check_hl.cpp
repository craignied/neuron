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

int main()
{
	bool ok = true;

	ok &= check( 9999 );  // just under the old fixed array size
	ok &= check( 12000 ); // past it: segfaulted before the fix

	if ( ok )
	{
		cout << "check_hl: Hosmer-Lemeshow survives large datasets" << endl;
		return 0;
	}
	cerr << "check_hl: FAILED" << endl;
	return 1;
}
