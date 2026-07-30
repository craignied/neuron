/* DeLong's test for correlated ROC areas -- implementation. See delong.h for the
   method and the independence-scope warning.

   The placements, the tie handling and the contrast algebra live in auccov (they
   are shared with the clustered estimator); what is DeLong-specific, and all that
   remains here, is the combination S10/n1 + S01/n0 -- the step that assumes the
   observations are independent. */

#include "delong.h"

#include <cmath>

delong::Result delong::analyze( const vector< unsigned >& label,
	const vector< vector< double > >& pred )
{
	Result out;

	auccov::Placements p = auccov::compute( label, pred );
	out.n0 = p.n0; out.n1 = p.n1;
	if ( !p.ok ) { out.reason = p.reason; return out; }

	// DeLong's own requirement: the sample covariances below divide by ( n - 1 ),
	//    so each class needs at least two observations. The clustered estimator
	//    counts CLUSTERS instead and states its own minimum -- which is why this
	//    check is here and not in the shared placement code.
	if ( p.n1 < 2 || p.n0 < 2 )
	{
		out.reason = "DeLong needs at least two positive and two negative "
			"test observations";
		return out;
	}

	unsigned m = ( unsigned ) pred.size(), n1 = p.n1, n0 = p.n0;
	out.auc = p.auc;

	// The area covariance: S10/n1 + S01/n0, where S10 (S01) is the sample
	//    covariance of the V10 (V01) components across classifiers (DeLong Eq 4-6).
	//    There is no cross term: under independent observations the positive and
	//    negative placements of the same observation cannot be correlated, because
	//    an observation is only ever one or the other. That missing term is
	//    precisely what the clustered estimator adds (auccov.h / roc_theory.md).
	Matrix< double > S10( m, m ), S01( m, m );
	for ( unsigned k = 0; k < m; k++ )
		for ( unsigned l = 0; l < m; l++ )
		{
			double s10 = 0;
			for ( unsigned a = 0; a < n1; a++ )
				s10 += ( p.v10[ k ][ a ] - out.auc[ k ] ) * ( p.v10[ l ][ a ] - out.auc[ l ] );
			S10( k, l ) = s10 / ( n1 - 1 );

			double s01 = 0;
			for ( unsigned b = 0; b < n0; b++ )
				s01 += ( p.v01[ k ][ b ] - out.auc[ k ] ) * ( p.v01[ l ][ b ] - out.auc[ l ] );
			S01( k, l ) = s01 / ( n0 - 1 );
		}

	out.cov = Matrix< double >( m, m );
	for ( unsigned k = 0; k < m; k++ )
		for ( unsigned l = 0; l < m; l++ )
			out.cov( k, l ) = S10( k, l ) / n1 + S01( k, l ) / n0;

	out.ok = true;
	return out;
}

delong::Contrast delong::contrast( const Result& r, unsigned i, unsigned j )
{
	if ( !r.ok ) return Contrast();
	return auccov::contrast( r.auc, r.cov, i, j, true /* clamp to [0,1] */ );
}

delong::Interval delong::interval( const Result& r, unsigned i )
{
	if ( !r.ok ) return Interval();
	return auccov::interval( r.auc, r.cov, i, true /* clamp to [0,1] */ );
}
