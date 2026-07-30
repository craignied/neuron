/* Sampling-unit-independent parts of an AUC covariance analysis. See auccov.h.
   Extracted from delong.cpp unchanged (2026-07-30) so the clustered estimator
   shares the tie handling, the point areas, and the contrast case law rather
   than reimplementing them. */

#include "auccov.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "stats.h"

namespace {

const double Z975 = 1.959963984540054; // the 97.5th percentile of the standard normal

double clamp01( double v )
{
	if ( v < 0 ) return 0;
	if ( v > 1 ) return 1;
	return v;
}

} // namespace

// Order statistics: inherently index-based (rule 4), sort once, O(n log n).
vector< double > auccov::midranks( const vector< double >& x )
{
	unsigned n = ( unsigned ) x.size();
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

auccov::Placements auccov::compute( const vector< unsigned >& label,
	const vector< vector< double > >& pred )
{
	Placements out;
	unsigned m = ( unsigned ) pred.size();

	if ( m == 0 ) { out.reason = "no classifiers to compare"; return out; }

	for ( unsigned r = 0; r < label.size(); r++ )
	{
		if ( label[ r ] ) out.pos.push_back( r );
		else out.neg.push_back( r );
	}
	unsigned n1 = ( unsigned ) out.pos.size(), n0 = ( unsigned ) out.neg.size();
	out.n0 = n0; out.n1 = n1;

	if ( n1 == 0 || n0 == 0 )
	{
		out.reason = "an ROC area needs both outcome classes present";
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
	out.v10.assign( m, vector< double >( n1 ) );
	out.v01.assign( m, vector< double >( n0 ) );
	out.auc.assign( m, 0.0 );

	for ( unsigned k = 0; k < m; k++ )
	{
		vector< double > xall( n1 + n0 );
		for ( unsigned a = 0; a < n1; a++ ) xall[ a ] = pred[ k ][ out.pos[ a ] ];
		for ( unsigned b = 0; b < n0; b++ ) xall[ n1 + b ] = pred[ k ][ out.neg[ b ] ];

		vector< double > rAll = midranks( xall );
		vector< double > xPos( n1 ), xNeg( n0 );
		for ( unsigned a = 0; a < n1; a++ ) xPos[ a ] = pred[ k ][ out.pos[ a ] ];
		for ( unsigned b = 0; b < n0; b++ ) xNeg[ b ] = pred[ k ][ out.neg[ b ] ];
		vector< double > rPos = midranks( xPos );
		vector< double > rNeg = midranks( xNeg );

		double sum = 0;
		for ( unsigned a = 0; a < n1; a++ )
		{
			out.v10[ k ][ a ] = ( rAll[ a ] - rPos[ a ] ) / n0;
			sum += out.v10[ k ][ a ];
		}
		out.auc[ k ] = sum / n1;
		for ( unsigned b = 0; b < n0; b++ )
			out.v01[ k ][ b ] = 1.0 - ( rAll[ n1 + b ] - rNeg[ b ] ) / n1;
	}

	out.ok = true;
	return out;
}

auccov::Interval auccov::interval( const vector< double >& auc,
	const Matrix< double >& cov, unsigned i, bool clampToUnit )
{
	Interval iv;
	if ( i >= auc.size() || cov.rows() != auc.size() ) return iv;
	iv.auc = auc[ i ];
	double v = cov( i, i );
	iv.se = v > 0 ? sqrt( v ) : 0;
	iv.lo = iv.auc - Z975 * iv.se;
	iv.hi = iv.auc + Z975 * iv.se;
	if ( clampToUnit ) { iv.lo = clamp01( iv.lo ); iv.hi = clamp01( iv.hi ); }
	iv.valid = true;
	return iv;
}

auccov::Contrast auccov::contrast( const vector< double >& auc,
	const Matrix< double >& cov, unsigned i, unsigned j, bool clampToUnit )
{
	Contrast c;
	if ( i >= auc.size() || j >= auc.size() || cov.rows() != auc.size() ) return c;

	c.aucI = auc[ i ];
	c.aucJ = auc[ j ];
	c.delta = c.aucI - c.aucJ;

	double vi = cov( i, i ), vj = cov( j, j ), cij = cov( i, j );
	c.seI = vi > 0 ? sqrt( vi ) : 0;
	c.seJ = vj > 0 ? sqrt( vj ) : 0;

	// Var(delta) = e^T Cov e with e = (1,-1). Cov is a sum of sample covariance
	//    matrices, so this is >= 0 in exact arithmetic. A tiny negative value is
	//    floating-point cancellation and is clamped to zero; a MATERIALLY negative
	//    value is an invalid covariance and is refused, not silently zeroed.
	double vd = vi + vj - 2 * cij;
	if ( vd < 0 )
	{
		double scale = ( vi + vj ) > 0 ? ( vi + vj ) : 1.0;
		if ( vd < -1e-9 * scale - 1e-15 )
		{
			c.note = "the difference variance is materially negative "
				"(invalid covariance)";
			return c; // valid stays false
		}
		vd = 0; // fp cancellation
	}
	c.seDelta = vd > 0 ? sqrt( vd ) : 0;

	c.ciLoI = c.aucI - Z975 * c.seI;
	c.ciHiI = c.aucI + Z975 * c.seI;
	c.ciLoJ = c.aucJ - Z975 * c.seJ;
	c.ciHiJ = c.aucJ + Z975 * c.seJ;
	if ( clampToUnit )
	{
		c.ciLoI = clamp01( c.ciLoI ); c.ciHiI = clamp01( c.ciHiI );
		c.ciLoJ = clamp01( c.ciLoJ ); c.ciHiJ = clamp01( c.ciHiJ );
	}
	// The DIFFERENCE interval is never clamped -- a difference of areas lives in
	//    [-1,1] and its sign is the point of the contrast.
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
		// seDelta == 0: either genuinely absent (equal areas) or a deterministic
		//    separation. Scale-aware split, never an exact == 0 (see auccov.h).
		double scale = fmax( 1.0, fmax( fabs( c.aucI ), fabs( c.aucJ ) ) );
		double deltaTol = 1024.0 * numeric_limits< double >::epsilon() * scale;
		if ( fabs( c.delta ) <= deltaTol )
		{
			c.degenerate = true;
			c.z = 0;
			c.p = 1.0;
		}
		else
		{
			c.separated = true;
			c.z = 0;
			c.p = 0.0;
		}
	}

	c.valid = true;
	return c;
}
