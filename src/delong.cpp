/* DeLong's test for correlated ROC areas -- implementation. See delong.h for the
   method and the independence-scope warning. */

#include "delong.h"

#include <algorithm>
#include <cmath>

#include "stats.h"

namespace {

// 1-based mid-ranks of x (ties share the average of the ranks they span). Order
//    statistics: this is inherently index-based (rule 4), sort once, O(n log n).
vector< double > midranks( const vector< double >& x )
{
	unsigned n = x.size();
	vector< unsigned > order( n );
	for ( unsigned i = 0; i < n; i++ ) order[ i ] = i;
	sort( order.begin(), order.end(),
		[ &x ]( unsigned a, unsigned b ) { return x[ a ] < x[ b ]; } );

	vector< double > r( n );
	unsigned i = 0;
	while ( i < n )
	{
		unsigned j = i;
		while ( j < n && x[ order[ j ] ] == x[ order[ i ] ] ) j++;
		// ranks i+1 .. j (1-based) all tie -> their average
		double avg = ( ( double )( i + 1 ) + ( double ) j ) / 2.0;
		for ( unsigned t = i; t < j; t++ ) r[ order[ t ] ] = avg;
		i = j;
	}
	return r;
}

const double Z975 = 1.959963984540054; // the 97.5th percentile of the standard normal

double clamp01( double v )
{
	if ( v < 0 ) return 0;
	if ( v > 1 ) return 1;
	return v;
}

} // namespace

delong::Result delong::analyze( const vector< unsigned >& label,
	const vector< vector< double > >& pred )
{
	Result out;
	unsigned m = pred.size();

	if ( m == 0 ) { out.reason = "no classifiers to compare"; return out; }

	// Partition observation indices into positives and negatives; a non-finite
	//    score anywhere makes the areas meaningless (a diverged model).
	vector< unsigned > pos, neg;
	for ( unsigned r = 0; r < label.size(); r++ )
	{
		if ( label[ r ] ) pos.push_back( r );
		else neg.push_back( r );
	}
	unsigned n1 = pos.size(), n0 = neg.size();
	out.n0 = n0; out.n1 = n1;

	if ( n1 < 2 || n0 < 2 )
	{
		out.reason = "DeLong needs at least two positive and two negative "
			"test observations";
		return out;
	}
	for ( unsigned k = 0; k < m; k++ )
	{
		if ( pred[ k ].size() != label.size() )
			{ out.reason = "prediction/label length mismatch"; return out; }
		for ( unsigned r = 0; r < pred[ k ].size(); r++ )
			if ( !isfinite( pred[ k ][ r ] ) )
				{ out.reason = "a prediction is not finite (a diverged model?)"; return out; }
	}

	// The structural components (DeLong's V10, V01) via mid-ranks (Sun & Xu):
	//    V10[k][a] = ( 1/n0 ) # negatives a positive beats (half credit for ties);
	//    V01[k][b] = ( 1/n1 ) # positives a negative loses to. AUC = mean( V10 ).
	vector< vector< double > > V10( m, vector< double >( n1 ) );
	vector< vector< double > > V01( m, vector< double >( n0 ) );
	out.auc.assign( m, 0.0 );

	for ( unsigned k = 0; k < m; k++ )
	{
		vector< double > xall( n1 + n0 );
		for ( unsigned a = 0; a < n1; a++ ) xall[ a ] = pred[ k ][ pos[ a ] ];
		for ( unsigned b = 0; b < n0; b++ ) xall[ n1 + b ] = pred[ k ][ neg[ b ] ];

		vector< double > rAll = midranks( xall );
		vector< double > xPos( n1 ), xNeg( n0 );
		for ( unsigned a = 0; a < n1; a++ ) xPos[ a ] = pred[ k ][ pos[ a ] ];
		for ( unsigned b = 0; b < n0; b++ ) xNeg[ b ] = pred[ k ][ neg[ b ] ];
		vector< double > rPos = midranks( xPos );
		vector< double > rNeg = midranks( xNeg );

		double sum = 0;
		for ( unsigned a = 0; a < n1; a++ )
		{
			V10[ k ][ a ] = ( rAll[ a ] - rPos[ a ] ) / n0;
			sum += V10[ k ][ a ];
		}
		out.auc[ k ] = sum / n1;
		for ( unsigned b = 0; b < n0; b++ )
			V01[ k ][ b ] = 1.0 - ( rAll[ n1 + b ] - rNeg[ b ] ) / n1;
	}

	// The area covariance: S10/n1 + S01/n0, where S10 (S01) is the sample
	//    covariance of the V10 (V01) components across classifiers (DeLong Eq 4-6).
	Matrix< double > S10( m, m ), S01( m, m );
	for ( unsigned k = 0; k < m; k++ )
		for ( unsigned l = 0; l < m; l++ )
		{
			double s10 = 0;
			for ( unsigned a = 0; a < n1; a++ )
				s10 += ( V10[ k ][ a ] - out.auc[ k ] ) * ( V10[ l ][ a ] - out.auc[ l ] );
			S10( k, l ) = s10 / ( n1 - 1 );

			double s01 = 0;
			for ( unsigned b = 0; b < n0; b++ )
				s01 += ( V01[ k ][ b ] - out.auc[ k ] ) * ( V01[ l ][ b ] - out.auc[ l ] );
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
	Contrast c;
	if ( !r.ok || i >= r.auc.size() || j >= r.auc.size() ) return c;

	c.aucI = r.auc[ i ];
	c.aucJ = r.auc[ j ];
	c.delta = c.aucI - c.aucJ;

	double vi = r.cov( i, i ), vj = r.cov( j, j ), cij = r.cov( i, j );
	c.seI = vi > 0 ? sqrt( vi ) : 0;
	c.seJ = vj > 0 ? sqrt( vj ) : 0;
	double vd = vi + vj - 2 * cij;
	c.seDelta = vd > 0 ? sqrt( vd ) : 0;

	c.ciLoI = clamp01( c.aucI - Z975 * c.seI );
	c.ciHiI = clamp01( c.aucI + Z975 * c.seI );
	c.ciLoJ = clamp01( c.aucJ - Z975 * c.seJ );
	c.ciHiJ = clamp01( c.aucJ + Z975 * c.seJ );
	c.ciLoDelta = c.delta - Z975 * c.seDelta;
	c.ciHiDelta = c.delta + Z975 * c.seDelta;

	if ( c.seDelta > 0 )
	{
		c.z = c.delta / c.seDelta;
		// two-sided p = P( |Z| > |z| ) = erfc( |z| / sqrt(2) )
		c.p = stats::erfc( fabs( c.z ) / sqrt( 2.0 ) );
	}
	else
	{
		// Zero difference-variance: the two areas cannot be told apart by DeLong
		//    (identical predictions, or a perfectly correlated pair). This is an
		//    explicit degenerate result, not a real z == 0 test -- the caller is
		//    expected to report "no testable difference", not "p = 1, significant".
		c.degenerate = true;
		c.z = 0;
		c.p = 1.0;
	}

	c.valid = true;
	return c;
}

delong::Interval delong::interval( const Result& r, unsigned i )
{
	Interval iv;
	if ( !r.ok || i >= r.auc.size() ) return iv;
	iv.auc = r.auc[ i ];
	double v = r.cov( i, i );
	iv.se = v > 0 ? sqrt( v ) : 0;
	iv.lo = clamp01( iv.auc - Z975 * iv.se );
	iv.hi = clamp01( iv.auc + Z975 * iv.se );
	iv.valid = true;
	return iv;
}
