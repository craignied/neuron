// Statistical global methods, namespace "stats"

#include <stdexcept>
#include <cassert>
#include <vector>
#include <math.h>

using namespace std;

#ifndef STATISTICS_H
#define STATISTICS_H

#ifndef SHFT
#define SHFT( a, b, c, d ) ( a ) = ( b ); ( b ) = ( c ); ( c ) = ( d );
#endif

#ifndef SIGN
#define SIGN( a, b ) ( ( b ) >= 0.0 ? fabs( a ) : -fabs( a ) )
#endif

#ifndef FMAX
static double maxarg1, maxarg2;
#define FMAX( a, b ) ( maxarg1 = ( a ), maxarg2 = ( b ), ( maxarg1 ) > ( maxarg2 ) ? ( maxarg1 ) : ( maxarg2 ) )
#endif

#define ITMAX 100
#define GOLD 1.618034
#define GLIMIT 100.0
#define TINY 1.0e-20
#define CGOLD 0.3819660
#define ZEPS 1.0e-10
#define EPS 3.0e-8

namespace stats {
	// stats exception, extends runtime_error class
	class statsErr : public runtime_error {
	public:
		statsErr( const char* message ) throw() : runtime_error( message ) {}
	};

	// Log gamma function, Numerical Recipes in C p. 214
	double gammln( double );

	// Incomplete gamma function by continued fraction representation
	//    Modification of Numerical Recipes in C p.219
	double gcf( double, double );

	// Incomplete gamma function by series representation
	//    Numerical Recipes in C p.218
	double gser( double, double );

	// Incomplete gamma function P, Numerical Recipes in C p.218
	double gammp( double, double );

	// Incomplete gamma function Q = 1 - P, Numerical Recipes in C p.218
	double gammq( double, double );

	// Beta function, Numerical Recipes in C p.215
	double beta( double, double );

	// Incomplete beta function, Numerical Recipes in C p.226
	double betai( double, double, double );

	// Continued fraction evaluation of incomplete beta function
	//    Numerical Recipes in C p.227
	double betacf( double, double, double );

	// Computes p value from degrees of freedom and Chi-square value
	double pX2( unsigned, double );

	// Error function erff Numerical Recipes in C p.220
	double erff( double );

	// Error function erfc Numerical Recipes in C p.220
	double erfc( double );

	// Error function erfcc Numerical Recipes in C p.221
	double erfcc( double );

	// Computes integral of unit Gaussian distribution (z-score)
	//    "erff" in Numerical Recipes in C p.220 applied to the Gaussian
	//    distribution = 1/2 + 1/2 erf( z / sqrt(2) )
	double Zarea( double );

	// Takuya Ooura's inverse of erfc (c) 1996
	double invErfc( double );

	// Inverse of erff, calls erfc
	double invErff( double );

	// Inverse of Zarea = sqrt(2) * invErff( 2 * A - 1 )
	double invZarea( double );

	// Gaussian distribution functions
	double gaussD( double );
	double gaussD( double, double, double );

	// Kolmogorov-Smirnov probability function Numerical Recipes in C p. 626
	double probks( double );

	// Student's t-test, returns p-value
	double ttest( vector< double >&, vector< double >& ); // equal variance
	double tutest( vector< double >&, vector< double >& ); // unequal variance
	double tptest( vector< double >&, vector< double >& ); // paired

	// F-test, returns p-value
	double ftest( vector< double >&, vector< double >& );

	// Utility functions
	// Returns the square of a value
	inline double sqr( double x ) { return x * x; }
};

// For function manipulation: root finding, etc.
template < class T >
class Funct {
public:
	// Ctor loads function
	Funct( T& i, double ( T::*f )( double ) ) : instance( i ), func( f ) {}
	~Funct(){} // destructor
	// Note no copy ctor or = operator

	// funct exception, extends runtime_error class
	class functErr : public runtime_error {
	public:
		functErr( const char* message ) throw() : runtime_error( message ) {}
	};

	// Just to illustrate how passing functions works here
	double eval( double x ) const { return ( instance.*func )( x ); }

	// Bracketing a function: mnbrak Numerical Recipes in C p.400
	// Set the initial bracketing points ax, bx
	void mnbrak( double, double ); // this will do the actual bracketing
	// Get the new bracketed points, and if desired, evaluated function at them
	double mnbrak_ax(){ return ax; }
	double mnbrak_bx(){ return bx; }
	double mnbrak_cx(){ return cx; }
	double mnbrak_fa(){ return fa; }
	double mnbrak_fb(){ return fb; }
	double mnbrak_fc(){ return fc; }

	// Isolates the minimum of the function using Brent's method, set the
	//    initial bracketing triplet of abscissas ax, bx, cx and tolerance
	//    Numerical Recipes in C p.404
	double brent( double, double, double, double );
	// Get x minimum
	double brent_xmin(){ return xmin; }

	// Root finding using Brent's method, set 2 x bounds x1 and x2, followed
	//    by tolerance tol, Numerical Recipes in C p.359
	double zbrent( double, double, double );

private:
	T& instance;
	double ( T::*func )( double ); // the function

	double ax, bx, cx, // x values for mnbrak
		fa, fb, fc, // function at x values for mnbrak
		xmin; // x minimum for brent
};

// Class for population metrics such as mean, variance, standard deviation
class Population {
public:
	// Only allowed ctor loads vector representing population
	Population( vector< double >& );
	~Population(){} // destructor
	// Note no copy ctor or = operator

	// Population exception, extends runtime_error class
	class PopulationErr : public runtime_error {
	public:
		PopulationErr( const char* message ) throw() : runtime_error( message ) {}
	};

	// Accessors to return Population metrics
	double mean(){ return _mean; } // mean
	double var(); // variance
	double std(); // standard deviation
	double skew(); // accessor returns skew
	double kurtosis(); // accessor returns kurtosis

private:
	vector < double > x; // the dataset
	double _n,     // number of data items
		_mean,    // mean
		_var,     // variance
		_std,     // standard deviation
		sumSquares, // sum of squares
		sumCubes; // sum of cubes

	bool varFlag, // flag to indicate if var calculated
		stddevFlag, // flag indicates if standard deviation calculated
		squaresFlag, // flag to indicate if sum of squares calculated
		cubesFlag; // flag to indicate if sum of cubes calculated

	// Utility method to return sum of the powers of dataset elements
	//    argument is power
	double sumPowers( unsigned );
};

// Class for XY dataset, used for least squares curve fitting, etc.
class XY {
public:
	// Ctors for loading XY datasets
	// Loads an XY dataset without standard deviations
	XY( vector< double >&, vector< double >& );
	// Loads an XY dataset with standard deviations
	XY( vector< double >&, vector< double >&, vector< double >& );
	// Loads an XY dataset with standard deviations in x & y
	XY( vector< double >&, vector< double >&, vector< double >&, vector< double >& );
	~XY(){} // destructor
	// Note no copy ctor or = operator

	// XY exception, extends runtime_error class
	class XYErr : public runtime_error {
	public:
		XYErr( const char* message ) throw() : runtime_error( message ) {}
	};

	// Accessors
	double a(){ return _a; }
	double b(){ return _b; }
	double x0() { return -_a / _b; }
	double siga(){ return _siga; }
	double sigb(){ return _sigb; }
	double chi2(){ return _chi2; }
	double q(){ return _q; }
	double r(){ return _r; }

private:
	double _a, _b, // y = a + bx
		_siga, _sigb, // standard errors on a & b
		_chi2, // minimized chi-squared
		_q, // chi-squared probability (small is a poor fit)
		_r, // correlation coefficient
		aa, offs; // for datasets with errors in both x & y

	vector< double > x, y, // x & y axis values
		sig, // dataset std deviations
		xx, yy, sx, sy, ww; // for datasets with errors in both x & y

	unsigned ndata, // number data points
		nn; // for datasets with errors in both x & y

	bool mwt; // flag indicates if dataset has std deviations

	// Utility functions
	// Returns the square of a value
	inline double sqr( double x ) { return x * x; }
	// Utility function for modeling straight line
	void fit(); // Numerical Recipes in C p.665
	// Captive function of fitexy, see Numerical Recipes in C p.670
	double chixy( double );
};

// Bracketing a function: mnbrak Numerical Recipes in C p.400
//    Set the initial bracketing points ax, bx: this will do the actual bracketing
template< class T >
void Funct< T >::mnbrak( double _ax, double _bx )
{
	// Set the private members in this function
	ax = _ax;
	bx = _bx;

	double ulim, u, r, q, fu, dum;

	fa = ( instance.*func )( ax );
	fb = ( instance.*func )( bx );
	if  ( fb > fa )
	{
		SHFT( dum, ax, bx, dum )
		SHFT( dum, fb, fa, dum )
	}
	cx = bx + GOLD * ( bx - ax );
	fc = ( instance.*func )( cx );
	while  ( fb > fc )
	{
		r = ( bx - ax ) * ( fb - fc );
		q = ( bx - cx ) * ( fb - fa );
		u = bx - ( ( bx - cx ) * q - ( bx - ax ) * r ) /
			( 2.0 * SIGN( FMAX( fabs( q - r ), TINY ), q - r ) );
		ulim = bx + GLIMIT * ( cx - bx );
		if ( ( bx - u ) * ( u - cx ) > 0.0 )
		{
			fu = ( instance.*func )( u );
			if  ( fu < fc )
			{
				ax = bx;
				bx = u;
				fa = fb;
				fb = fu;
				return;
			}
			else if ( fu > fb )
			{
				cx = u;
				fc = fu;
				return;
			}
			u = cx + GOLD * ( cx - bx );
			fu = ( instance.*func )( u );
		}
		else if ( ( cx - u ) * ( u - ulim ) > 0.0 )
		{
			fu = ( instance.*func )( u );
			if ( fu < fc )
			{
				SHFT( bx, cx, u, cx + GOLD * ( cx - bx ) )
				SHFT( fb, fc, fu, ( instance.*func )( u ) )
			}
		}
		else if ( ( u - ulim ) * ( ulim - cx ) >= 0.0 )
		{
			u = ulim;
			fu = ( instance.*func )( u );
		} else
		{
			u = cx + GOLD *  ( cx - bx );
			fu = ( instance.*func )( u );
		}
		SHFT( ax, bx, cx, u )
		SHFT( fa, fb, fc, fu )
	}
}

// Given a bracketing triplet of abscissas ax, bx & cx such that bx is between
//    ax and cx, and f(bx) is less than both f(ax) & f(cx), isolates the minimum
//    to a fractional precision of tol using Brent's method, xmin is the minimum,
//    and the function returns f(xmin) Numerical Recipes in C p.404

template< class T >
double Funct< T >::brent( double ax, double bx, double cx, double tol )
{
	unsigned iter;
	double a, b, etemp, fu, fv, fw, fx, p, q, r, tol1, tol2, u, v, w, x, xm;

	double d = 0.0, e = 0.0;

	a =  ( ax < cx ? ax : cx );
	b =  ( ax > cx ? ax : cx );
	x = w = v = bx;
	fw = fv = fx = ( instance.*func )( x );
	for ( iter = 1; iter <= ITMAX; iter++ )
	{
		xm = 0.5 * ( a + b );
		tol2 = 2.0 * ( tol1 = tol * fabs( x ) + ZEPS );
		if ( fabs( x - xm ) <= ( tol2 - 0.5 * ( b - a ) ) )
		{
			xmin = x;
			return fx;
		}
		if  ( fabs( e ) > tol1 )
		{
			r = ( x - w ) * ( fx - fv );
			q = ( x - v ) * ( fx - fw );
			p = ( x - v ) * q - ( x - w ) * r;
			q = 2.0 * ( q - r );
			if ( q > 0.0 )
				p = -p;
			q = fabs( q );
			etemp = e;
			e = d;
			if ( fabs( p ) >= fabs( 0.5 * q * etemp ) || p <= q * ( a - x )
				|| p >= q * ( b - x ) )
				d = CGOLD * ( e = ( x >= xm ? a - x : b - x ) );
			else
			{
				d = p / q;
				u = x + d;
				if ( u - a < tol2 || b - u < tol2 )
					d = SIGN( tol1, xm - x );
			}
		}
		else
		{
			d = CGOLD * ( e = ( x >= xm ? a - x : b - x ) );
		}
		u = ( fabs( d ) >= tol1 ? x + d : x + SIGN( tol1, d ) );
		fu = ( instance.*func )( u );
		if ( fu <= fx )
		{
			if ( u >= x )
				a = x;
			else
				b = x;
			SHFT ( v, w, x, u )
			SHFT ( fv, fw, fx, fu )
		}
		else
		{
			if ( u < x )
				a = u;
			else
				b = u;
			if ( fu <= fw || w == x )
			{
				v = w;
				w = u;
				fv = fw;
				fw = fu;
			}
			else if ( fu <= fv || v == x || v == w )
			{
				v = u;
				fv = fu;
			}
		}
	}
	throw functErr( "Too many iterations in brent" );
	xmin = x;
	return fx;
}

// Root finding using Brent's method Numerical Recipes in C p.359
//    Takes 2 x bounds x1 and x2, followed by tolerance tol,
//    and returns the nearest root
template< class T >
double Funct< T >::zbrent( double x1, double x2, double tol )
{
	unsigned iter;
	double a = x1, b = x2, c = x2, d = 0.0, e = 0.0, min1, min2,
		fa = ( instance.*func )( a ), fb = ( instance.*func )( b ),
		fc, p, q, r, s, tol1 ,xm;

	if ( ( fa > 0.0 && fb > 0.0 ) || ( fa < 0.0 && fb < 0.0 ) )
		throw functErr( "Root must be bracketed in zbrent" );
	fc = fb;
	for ( iter = 1; iter <= ITMAX; iter++ )
	{
		if ( ( fb > 0.0 && fc > 0.0 ) || ( fb < 0.0 && fc < 0.0 ) )
		{
			c = a;
			fc = fa;
			e = d = b-a;
		}
		if ( fabs( fc ) < fabs( fb ) )
		{
			a = b;
			b = c;
			c = a;
			fa = fb;
			fb = fc;
			fc = fa;
		}
		tol1 = 2.0 * EPS * fabs( b ) + 0.5 * tol;
		xm = 0.5 * ( c - b );
		if ( fabs( xm ) <= tol1 || fb == 0.0 )
			return b;
		if ( fabs( e ) >= tol1 && fabs( fa ) > fabs( fb ) )
		{
			s = fb / fa;
			if ( a == c )
			{
				p = 2.0 * xm * s;
				q = 1.0 - s;
			}
			else
			{
				q = fa / fc;
				r = fb / fc;
				p = s * ( 2.0 * xm * q * ( q - r ) - ( b - a ) * ( r - 1.0 ) );
				q = ( q - 1.0 ) * ( r - 1.0 ) * ( s - 1.0 );
			}
			if ( p > 0.0 )
				q = -q;
			p = fabs( p );
			min1 = 3.0 * xm * q - fabs( tol1 *q );
			min2 = fabs( e * q );
			if ( 2.0 * p < ( min1 < min2 ? min1 : min2 ) )
			{
				e = d;
				d = p / q;
			}
			else
			{
				d = xm;
				e = d;
			}
		}
		else
		{
			d = xm;
			e = d;
		}
		a = b;
		fa = fb;
		if ( fabs( d ) > tol1 )
			b += d;
		else
			b += SIGN( tol1, xm );
		fb = ( instance.*func )( b );
	}
	throw functErr( "Maximum number of iterations exceeded in zbrent" );
	return 0.0;
}

#undef ITMAX
#undef GOLD
#undef GLIMIT
#undef TINY
#undef CGOLD
#undef ZEPS
#undef EPS

#endif
