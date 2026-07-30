/* Obuchowski's clustered ROC-area covariance -- implementation. See
   clustered_auc.h for scope and docs/roc_theory.md for the design record. */

#include "clustered_auc.h"

#include <algorithm>
#include <cmath>

clustered_auc::Result clustered_auc::analyze( const vector< unsigned >& label,
	const vector< unsigned >& cluster,
	const vector< vector< double > >& pred )
{
	Result out;
	out.method = "Obuchowski (clustered ROC covariance)";
	out.nRows = ( unsigned ) label.size();

	if ( cluster.size() != label.size() )
	{
		out.reason = "the cluster ids do not pair with the observations";
		return out;
	}

	// The placements, the areas and the tie handling are DeLong's, shared -- a
	//    clustered analysis must not produce a different area from an ordinary one
	//    on the same data (auccov.h).
	auccov::Placements p = auccov::compute( label, pred );
	out.n0 = p.n0; out.n1 = p.n1;
	if ( !p.ok ) { out.reason = p.reason; return out; }

	unsigned m = ( unsigned ) pred.size();
	out.auc = p.auc;

	// Compact the cluster ids. The result must not depend on how the caller
	//    happened to number its sampling units, so nothing below sees the raw ids.
	vector< unsigned > distinct = cluster;
	sort( distinct.begin(), distinct.end() );
	distinct.erase( unique( distinct.begin(), distinct.end() ), distinct.end() );
	unsigned I = ( unsigned ) distinct.size();
	out.nClusters = I;

	vector< unsigned > cid( label.size() );
	for ( unsigned r = 0; r < label.size(); r++ )
		cid[ r ] = ( unsigned ) ( lower_bound( distinct.begin(), distinct.end(),
			cluster[ r ] ) - distinct.begin() );

	// Per-cluster class counts, and how many clusters carry each class. I10 and
	//    I01 -- not I -- are this estimator's effective sample sizes.
	vector< double > mc( I, 0.0 ), nc( I, 0.0 );
	for ( unsigned r = 0; r < label.size(); r++ )
		( label[ r ] ? mc : nc )[ cid[ r ] ] += 1.0;
	for ( unsigned c = 0; c < I; c++ )
	{
		if ( mc[ c ] > 0 ) out.clusters10++;
		if ( nc[ c ] > 0 ) out.clusters01++;
	}

	// The divisors are ( I10 - 1 ) and ( I01 - 1 ): with one informative cluster
	//    there is no spread ACROSS clusters to estimate, however many rows it
	//    holds. That is a different condition from ordinary DeLong's "two of each
	//    class", and it is the one that actually binds on clustered data.
	if ( out.clusters10 < 2 || out.clusters01 < 2 || I < 2 )
	{
		out.reason = "clustered inference needs at least two clusters carrying "
			"each outcome class (found " + to_string( out.clusters10 )
			+ " with an event and " + to_string( out.clusters01 )
			+ " without, over " + to_string( I ) + " clusters); rows within one "
			"cluster carry no information about between-cluster variability";
		return out;
	}

	const double M = ( double ) p.n1, N = ( double ) p.n0;

	// The per-cluster structural sums, centred: X_c = sum of V10 over the
	//    cluster's outcome-1 rows, less m_c * theta; Y_c likewise for V01. One
	//    row of dx / dy per procedure.
	Matrix< double > dx( m, I ), dy( m, I );
	for ( unsigned k = 0; k < m; k++ )
	{
		vector< double > X( I, 0.0 ), Y( I, 0.0 );
		for ( unsigned a = 0; a < p.n1; a++ ) X[ cid[ p.pos[ a ] ] ] += p.v10[ k ][ a ];
		for ( unsigned b = 0; b < p.n0; b++ ) Y[ cid[ p.neg[ b ] ] ] += p.v01[ k ][ b ];
		for ( unsigned c = 0; c < I; c++ )
		{
			dx( k, c ) = X[ c ] - mc[ c ] * out.auc[ k ];
			dy( k, c ) = Y[ c ] - nc[ c ] * out.auc[ k ];
		}
	}

	// Obuchowski (1997) eq. 4-6. S11 carries the covariance between the
	//    outcome-1 and outcome-0 rows OF THE SAME CLUSTER -- the term ordinary
	//    DeLong has no place for, because under independence an observation is
	//    only ever one role or the other and the two cannot correlate. It is not
	//    symmetric in ( r, s ), so both orderings enter the sum below.
	const double f10 = ( double ) out.clusters10
		/ ( ( double ) ( out.clusters10 - 1 ) * M );
	const double f01 = ( double ) out.clusters01
		/ ( ( double ) ( out.clusters01 - 1 ) * N );
	const double f11 = ( double ) I / ( double ) ( I - 1 );

	Matrix< double > S10( m, m ), S01( m, m ), S11( m, m );
	for ( unsigned k = 0; k < m; k++ )
		for ( unsigned l = 0; l < m; l++ )
		{
			double s10 = 0, s01 = 0, s11 = 0;
			for ( unsigned c = 0; c < I; c++ )
			{
				s10 += dx( k, c ) * dx( l, c );
				s01 += dy( k, c ) * dy( l, c );
				s11 += dx( k, c ) * dy( l, c );
			}
			S10( k, l ) = f10 * s10;
			S01( k, l ) = f01 * s01;
			S11( k, l ) = f11 * s11;
		}

	out.cov = Matrix< double >( m, m );
	for ( unsigned k = 0; k < m; k++ )
		for ( unsigned l = 0; l < m; l++ )
			out.cov( k, l ) = S10( k, l ) / M + S01( k, l ) / N
				+ ( S11( k, l ) + S11( l, k ) ) / ( M * N );

	out.ok = true;
	return out;
}

auccov::Contrast clustered_auc::contrast( const Result& r, unsigned i, unsigned j )
{
	if ( !r.ok ) return auccov::Contrast();
	return auccov::contrast( r.auc, r.cov, i, j, false /* do not clamp */ );
}

auccov::Interval clustered_auc::interval( const Result& r, unsigned i )
{
	if ( !r.ok ) return auccov::Interval();
	return auccov::interval( r.auc, r.cov, i, false /* do not clamp */ );
}
