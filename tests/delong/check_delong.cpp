// check_delong.cpp : DeLong's test for correlated ROC areas (ROADMAP 4 Phase 4,
//   the locked-test inference layer).
//
// Validated three independent ways, and per standing rule 2 each was watched to
// FAIL against a deliberate sabotage of delong.cpp before being trusted:
//
//   1. AUC == the exact Mann-Whitney area. DeLong's area IS the trapezoidal
//      (rank) AUC the engine already computes, so it must equal
//      TwoSet::getTrapROCarea on the same (outcome, score) pairs -- a check the
//      engine's own estimator anchors, independent of the covariance math.
//      (Sabotage: drop the tie half-credit in midranks -> AUC on the tied
//      fixture no longer matches.)
//
//   2. A 12-row, 2-classifier fixture with deliberate ties, pinned to an
//      INDEPENDENT Python DeLong (scratchpad/delong_ref.py) that was itself
//      computed two ways -- a naive O(n0 n1) psi-sum AND the mid-rank formula --
//      which agree to machine precision. Areas, the full covariance, and the
//      difference's se/z/p are pinned to 1e-9. (Sabotage: a wrong covariance
//      divisor (n instead of n-1) -> se/z/p diverge.)
//
//   3. Structural identities that need no reference: identical classifiers give
//      delta == 0, Var(delta) == 0 exactly (cov is singular along that
//      direction), and p == 1; degenerate sets (a class < 2, or a non-finite
//      score) are refused, not fabricated. (Sabotage: compute cov(i,j) from k's
//      components twice instead of k and l -> the off-diagonal is wrong and the
//      identical-classifier Var(delta) is no longer 0.)

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "delong.h"
#include "twoset.h"

using namespace std;

int failures = 0;

void expect( bool ok, const string& what )
{
	cout << ( ok ? "ok - " : "FAIL - " ) << what << endl;
	if ( !ok ) failures++;
}

bool near( double a, double b, double tol = 1e-9 )
{
	return fabs( a - b ) <= tol;
}

// The pinned fixture (scratchpad/delong_ref.py). 6 negatives then 6 positives.
static const vector< unsigned > LABEL =
	{ 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1 };
static const vector< double > PRED_A =
	{ 0.10, 0.20, 0.35, 0.40, 0.55, 0.30, 0.45, 0.60, 0.70, 0.50, 0.65, 0.35 };
static const vector< double > PRED_B =
	{ 0.25, 0.20, 0.30, 0.55, 0.45, 0.40, 0.40, 0.50, 0.35, 0.60, 0.55, 0.30 };

int main()
{
	// ---- 1. AUC == exact Mann-Whitney (the engine's own trapezoidal area) ----
	{
		Matrix< double > D( LABEL.size(), 2 );
		for ( unsigned r = 0; r < LABEL.size(); r++ )
		{
			D( r, 0 ) = LABEL[ r ];      // real outcome
			D( r, 1 ) = PRED_A[ r ];     // score
		}
		TwoSet ts( D );
		double trap = ts.getTrapROCarea();

		delong::Result r = delong::analyze( LABEL, { PRED_A } );
		expect( r.ok, "analyze succeeds on the fixture (predictor A alone)" );
		expect( near( r.auc[ 0 ], trap ),
			"DeLong AUC equals the exact trapezoidal (Mann-Whitney) area" );
		expect( near( r.auc[ 0 ], 0.875 ),
			"DeLong AUC of predictor A is 0.875 (with tie half-credit)" );

		// Both orientations: reverse the score (higher = negative). DeLong must
		//    give 1 - AUC AND still equal the engine's trapezoidal area on the
		//    reversed scores -- proving the two estimators share the "higher score
		//    = positive" convention, so a convention mismatch can't masquerade as
		//    an inference bug (Sol's caution).
		vector< double > revA( PRED_A.size() );
		for ( unsigned r2 = 0; r2 < PRED_A.size(); r2++ ) revA[ r2 ] = 1.0 - PRED_A[ r2 ];
		Matrix< double > Drev( LABEL.size(), 2 );
		for ( unsigned r2 = 0; r2 < LABEL.size(); r2++ )
		{ Drev( r2, 0 ) = LABEL[ r2 ]; Drev( r2, 1 ) = revA[ r2 ]; }
		TwoSet tsr( Drev );
		delong::Result rr = delong::analyze( LABEL, { revA } );
		expect( near( rr.auc[ 0 ], 0.125 ) && near( rr.auc[ 0 ], 1.0 - r.auc[ 0 ] ),
			"reversed score: DeLong AUC == 1 - AUC == 0.125" );
		expect( near( rr.auc[ 0 ], tsr.getTrapROCarea() ),
			"reversed score: DeLong AUC still equals the trapezoidal area" );
	}

	// ---- 2. The pinned fixture: areas, covariance, and the contrast ----
	{
		delong::Result r = delong::analyze( LABEL, { PRED_A, PRED_B } );
		expect( r.ok && r.n1 == 6 && r.n0 == 6, "fixture: two classes of six" );
		expect( near( r.auc[ 0 ], 0.8750000000 ), "area A == 0.875 (Python ref)" );
		expect( near( r.auc[ 1 ], 0.7083333333 ), "area B == 0.7083333 (Python ref)" );

		// Covariance to 1e-9 (Python ref).
		expect( near( r.cov( 0, 0 ), 0.010879629630 ), "cov(A,A) (Python ref)" );
		expect( near( r.cov( 1, 1 ), 0.025694444444 ), "cov(B,B) (Python ref)" );
		expect( near( r.cov( 0, 1 ), 0.008564814815 ), "cov(A,B) (Python ref)" );
		expect( near( r.cov( 0, 1 ), r.cov( 1, 0 ) ), "covariance is symmetric" );

		delong::Contrast c = delong::contrast( r, 0, 1 );
		expect( c.valid, "contrast A vs B is valid" );
		expect( near( c.delta, 0.166666666667 ), "delta == 0.16667 (Python ref)" );
		expect( near( c.seDelta, 0.139443337756 ), "se(delta) (Python ref)" );
		expect( near( c.z, 1.195228609334 ), "z (Python ref)" );
		expect( near( c.p, 0.231997723629 ), "two-sided p (Python ref)" );
		expect( near( c.seI, 0.104305463086 ), "se(A) (Python ref)" );
		expect( near( c.seJ, 0.160294867181 ), "se(B) (Python ref)" );

		delong::Interval iv = delong::interval( r, 0 );
		expect( iv.valid && near( iv.auc, 0.875 ), "interval: area A" );
		expect( iv.lo <= iv.auc && iv.auc <= iv.hi && iv.hi <= 1.0,
			"interval A brackets the area and is clamped to <= 1" );
	}

	// ---- 3. Structural identities (no reference needed) ----
	{
		// Identical classifiers: no difference, and zero variance of the
		//    difference exactly -- the covariance is singular along (A - A).
		delong::Result r = delong::analyze( LABEL, { PRED_A, PRED_A } );
		expect( r.ok, "identical classifiers: analyze ok" );
		expect( near( r.cov( 0, 0 ), r.cov( 1, 1 ) )
			&& near( r.cov( 0, 0 ), r.cov( 0, 1 ) ),
			"identical classifiers: cov(A,A)==cov(A,A')==cov diagonal" );
		delong::Contrast c = delong::contrast( r, 0, 1 );
		expect( near( c.delta, 0.0 ) && near( c.seDelta, 0.0, 1e-12 ),
			"identical classifiers: delta == 0 and Var(delta) == 0 exactly" );
		expect( c.degenerate && near( c.p, 1.0 ),
			"identical classifiers: flagged degenerate (no testable difference), p == 1" );
	}

	// A clearly stronger classifier vs a constant-ish weak one gives a small p.
	{
		vector< unsigned > lab;
		vector< double > strong, weak;
		for ( unsigned i = 0; i < 40; i++ )
		{
			lab.push_back( i % 2 );
			strong.push_back( ( i % 2 ) ? 0.9 - 0.001 * i : 0.1 + 0.001 * i );
			weak.push_back( 0.5 + ( ( i * 7 ) % 11 ) * 0.001 ); // ~noise
		}
		delong::Result r = delong::analyze( lab, { strong, weak } );
		delong::Contrast c = delong::contrast( r, 0, 1 );
		expect( r.auc[ 0 ] > 0.9 && fabs( r.auc[ 1 ] - 0.5 ) < 0.2,
			"separating classifier ~1, noise classifier ~0.5" );
		expect( c.p < 0.05, "strong vs noise: DeLong p < 0.05" );
	}

	// ---- Degeneracy is refused, never fabricated ----
	{
		delong::Result r = delong::analyze( { 0, 0, 0, 1 }, { { .1, .2, .3, .9 } } );
		expect( !r.ok, "one positive: refused (needs >= 2 per class)" );

		double nan = numeric_limits< double >::quiet_NaN();
		delong::Result r2 = delong::analyze( LABEL,
			{ { .1, .2, .3, .4, .5, .6, nan, .8, .9, .5, .6, .7 } } );
		expect( !r2.ok, "non-finite score: refused (a diverged model)" );
	}

	cout << ( failures ? "\nFAILURES: " : "\nAll DeLong checks passed. " )
		<< failures << endl;
	return failures ? 1 : 0;
}
