// stats.cpp

#include "stdafx.h" // For MSVC, must be first!

// #define STATS_DEBUG // For debugging, shows trace of called statistical methods

#ifdef STATS_DEBUG
#include <iostream>
using namespace std;
#endif

#include <stdio.h>

#include <math.h>

#include <limits>

#include "stats.h"

#define ITMAX 100
#define EPS 3e-7
#define FPMIN 1.0e-30
#define POTN 1.571000
#define BIG 1.0e30
#define PI 3.14159265
#define ACC 1.0e-3

// Log gamma function, Numerical Recipes in C p. 214
double stats::gammln( double xx )
{
	double x, y, tmp, ser;
	static double cof[ 6 ] = { 76.18009172947146,
		-86.50532032941677,
		24.01409824083091,
		-1.231739572450155,
		0.1208650973866179e-2,
		-0.5395239384953e-5 };
	unsigned j;

#ifdef STATS_DEBUG
	util::screen() << "In stats::gammln" << endl;
#endif
	
	y = x = xx;
	tmp = x + 5.5;
	tmp -= ( x + 0.5 ) * log( tmp );
	ser = 1.000000000190015;
	for ( j = 0; j <= 5; j++ )
		ser += cof[ j ] / ++y;
	return -tmp + log( 2.5066282746310005 * ser / x );
}


// Chi-squared probability of a fit with ndata points and 2 fitted parameters,
//    or NaN when it cannot be computed. The fit probability is a diagnostic
//    and never a precondition for reporting the fit or an area derived from
//    it: Wickens, Elementary Signal Detection Theory, section 11.5 p. 217,
//    holds that even a model rejected on goodness of fit may still be used to
//    describe performance. gammq does not converge on a non-finite chi-squared
//    (and has no meaning with two or fewer points), so failing here must cost
//    the p value alone.
static double fitProbability( double chi2, unsigned ndata )
{
	if ( ndata <= 2 || !isfinite( chi2 ) || chi2 < 0 )
		return numeric_limits< double >::quiet_NaN();

	try { return stats::gammq( 0.5 * ( ndata - 2 ), 0.5 * chi2 ); }
	catch ( stats::statsErr& ) { return numeric_limits< double >::quiet_NaN(); }
}

// Incomplete gamma function by continued fraction representation
//    Modification of Numerical Recipes in C p.219

double stats::gcf( double a, double x )
{
	double anf, ana, an, g, a1, gold = 0.0, fac = 1.0, a0 = 1.0, b0 = 0.0, b1 = 1.0;
	double gln = gammln( a );

#ifdef STATS_DEBUG
	util::screen() << "In stats::gcf" << endl;
#endif
	
	a1 = x;
	for ( unsigned n = 1; n <= ITMAX; n++ )
	{
		an = ( double ) n;
		ana = an - a;
		a0 = ( a1 + a0 * ana ) * fac;
		b0 = ( b1 + b0 * ana ) * fac;
		anf = an * fac;
		a1 = x * a0 + anf * a1;
		b1 = x * b0 + anf * b1;
		if ( a1 )
		{
			fac = 1.0 / a1;
			g = b1 * fac;
			if ( fabs( ( g - gold ) / g ) < EPS )
				return exp( -x + a * log( x ) - gln ) * g;
			gold = g;
		}
	}
	throw statsErr( "a too large, ITMAX too small in gcf" );
}

// Incomplete gamma function by series representation
//    Numerical Recipes in C p.218
double stats::gser( double a, double x )
{
	unsigned n;
	double sum, del, ap, gln, gamser = 0.0;
	bool success = false;

#ifdef STATS_DEBUG
	util::screen() << "In stats::gser" << endl;
#endif
	
	gln = gammln( a );

	if ( x <= 0.0 )
	{
		if ( x < 0.0 )
		{
			throw statsErr( "x less than 0 in routine gser" );
			gamser = 0.0;
		}
	}
	else
	{
		ap = a;
		del = sum = 1.0 / a;

		for ( n = 1; n <= ITMAX; n++ )
		{
			++ap;
			del *= x / ap;
			sum += del;
			if ( fabs( del ) < fabs( sum ) * EPS )
			{
				gamser = sum * exp( -x + a * log( x ) - ( gln ) );
				success = true;
			}
		}

		if ( !success )
			throw statsErr( "a too large, ITMAX too small in routine gser" );
	}

	return gamser;
}

// Incomplete gamma function P, Numerical Recipes in C p.218
double stats::gammp( double a, double x )
{
	double result;

#ifdef STATS_DEBUG
	util::screen() << "In stats::gammp" << endl;
#endif
	
	if ( x < 0.0 || a <= 0.0 )
		throw statsErr( "Invalid arguments in routine gammp" );

	else if ( x < ( a + 1.0 ) )
		result = gser( a, x );

	else
		result = 1.0 - gcf( a, x );

	return result;
}

// Incomplete gamma function Q = 1 - P, Numerical Recipes in C p.218
double stats::gammq( double a, double x )
{
	double result;

#ifdef STATS_DEBUG
	util::screen() << "In stats::gammq" << endl;
#endif
	
	if ( x < 0.0 || a <= 0.0 )
		throw statsErr( "Invalid arguments in routine gammq" );

	else if ( x < ( a + 1.0 ) )
		result = 1.0 - gser( a, x );

	else
		result = gcf( a, x );

	return result;
}

// Beta function, Numerical Recipes in C p.215
double stats::beta( double z, double w )
{
#ifdef STATS_DEBUG
	util::screen() << "In stats::beta" << endl;
#endif
	return exp( gammln( z ) + gammln( w ) - gammln( z + w ) );
}

// Incomplete beta function, Numerical Recipes in C p.226
double stats::betai( double a, double b, double x )
{
#ifdef STATS_DEBUG
	util::screen() << "In stats::betai" << endl;
#endif
	
	if ( !( x >= 0.0 && x <= 1.0 ) )
		throw statsErr( "Invalid x in routine betai" );

	double bt = 0.0;
	
	if ( ( x != 0.0 ) && ( x != 1.0 ) )
		bt = exp( gammln( a + b ) - gammln( a ) - gammln( b ) +  a * log( x )
			+ b * log( 1.0 - x ) );

	if ( x < ( a + 1.0 ) / ( a + b + 2.0 ) )
		return bt * betacf( a, b, x ) / a;
	else
		return 1.0 - bt * betacf( b, a, 1.0 - x ) / b;
}

// Continued fraction evaluation of incomplete beta function
//    Numerical Recipes in C p.227
double stats::betacf( double a, double b, double x )
{
	int m,m2;
	double aa, c, d, del, h, qab, qam, qap;

#ifdef STATS_DEBUG
	util::screen() << "In stats::betacf" << endl;
#endif
	
	qab = a + b;
	qap = a + 1.0;
	qam = a - 1.0;
	c = 1.0;
	d = 1.0 - qab * x / qap;

	if ( fabs( d ) < FPMIN )
		d = FPMIN;
	
	d= 1.0 / d;
	h = d;
	
	for ( m = 1; m <= ITMAX; m++ )
	{
		m2 = 2 * m;
		aa = m * ( b - m ) * x / ( ( qam + m2 ) * ( a + m2 ) );
		d= 1.0 + aa * d;
		if ( fabs( d ) < FPMIN )
			d = FPMIN;
		c = 1.0 + aa / c;
		if ( fabs( c ) < FPMIN )
			c = FPMIN;
		d = 1.0 / d;
		h *= d * c;
		aa = -( a + m ) * ( qab + m ) * x / ( ( a + m2 ) * ( qap + m2 ) );
		d = 1.0 + aa * d;
		if ( fabs( d ) < FPMIN )
			d = FPMIN;
		c = 1.0 + aa / c;
		if ( fabs( c ) < FPMIN )
			c = FPMIN;
		d = 1.0 / d;
		del = d * c;
		h *= del;
		if ( fabs( del -1.0 ) < EPS )
			break;
	}

	if ( m > ITMAX )
		throw statsErr( "a or b too big, or ITMAX too small in betacf" );

	return h;
}

// Computes p value from degrees of freedom and Chi-square value
double stats::pX2( unsigned df, double X2 )
{
#ifdef STATS_DEBUG
	util::screen() << "In stats::pX2" << endl;
#endif
	return gammq( ( double ) df / 2, X2 / 2 );
}

// Error function erff Numerical Recipes in C p.220
double stats::erff( double x )
{
#ifdef STATS_DEBUG
	util::screen() << "In stats::erff" << endl;
#endif
	return x < 0.0 ? -gammp( 0.5, x * x ) : gammp( 0.5, x * x );
}

// Error function erfc Numerical Recipes in C p.220
double stats::erfc( double x )
{
#ifdef STATS_DEBUG
	util::screen() << "In stats::erfc" << endl;
#endif
	return x < 0.0 ? 1.0 + gammp( 0.5, x * x ) : gammq( 0.5, x * x );
}

// Error function erfcc Numerical Recipes in C p.221
double stats::erfcc( double x )
{
#ifdef STATS_DEBUG
	util::screen() << "In stats::erfcc" << endl;
#endif
	
	double t, z, ans;

	z = fabs( x );
	t = 1.0 / ( 1.0 + 0.5 * z );
	ans = t * exp( -z * z-1.26551223 + t * ( 1.00002368 + 
		t * ( 0.37409196 + t * ( 0.09678418 + t * ( -0.18628806 +
		t * ( 0.27886807 + t * ( -1.13520398 + t * ( 1.48851587 + 
		t * ( -0.82215223 + t * 0.17087277) ) ) ) ) ) ) ) );

	return x >= 0.0 ? ans : 2.0-ans;
}

// Computes integral of unit Gaussian distribution (z-score)
//    "erff" in Numerical Recipes in C p.220 applied to the Gaussian
//    distribution = 1/2 + 1/2 erf( z / sqrt(2) )
double stats::Zarea( double z )
{
#ifdef STATS_DEBUG
	util::screen() << "In stats::Zarea" << endl;
#endif
	return 1.0 - ( 0.5 * erfcc( z / sqrt( 2.0 ) ) ); // note complement
}

// Takuya Ooura's inverse of erfc (c) 1996
double stats::invErfc( double y )
{
	double s, t, u, w, x, z;

#ifdef STATS_DEBUG
	util::screen() << "In stats::invErfc" << endl;
#endif
	
	if ( y <= 0.0 )
		throw statsErr( "Argument <= 0 in routine invErfc" );

	else if ( y >= 2.0 )
		throw statsErr( "Argument >= 2 in routine invErfc" );

	else
	{
		z = y;

		if ( y > 1 )
			z = 2 - y;

		w = 0.916461398268964 - log( z );
		u = sqrt( w );
		s = ( log( u ) + 0.488826640273108 ) / w;
		t = 1 / ( u + 0.231729200323405 );
		x = u * ( 1 - s * ( s * 0.124610454613712 + 0.5 ) ) -
			( ( ( ( -0.0728846765585675 * t + 0.269999308670029 ) * t +
			0.150689047360223 ) * t + 0.116065025341614 ) * t +
			0.499999303439796 ) * t;
		t = 3.97886080735226 / ( x + 3.97886080735226 );
		u = t - 0.5;
		s = ( ( ( ( ( ( ( ( ( 0.00112648096188977922 * u +
			1.05739299623423047e-4 ) * u - 0.00351287146129100025 ) * u -
			7.71708358954120939e-4 ) * u + 0.00685649426074558612 ) * u +
			0.00339721910367775861 ) * u - 0.011274916933250487 ) * u -
			0.0118598117047771104 ) * u + 0.0142961988697898018 ) * u +
			0.0346494207789099922 ) * u + 0.00220995927012179067;
		s = ( ( ( ( ( ( ( ( ( ( ( ( s * u - 0.0743424357241784861 ) * u -
			0.105872177941595488 ) * u + 0.0147297938331485121 ) * u +
			0.316847638520135944 ) * u + 0.713657635868730364 ) * u +
			1.05375024970847138 ) * u + 1.21448730779995237 ) * u +
			1.16374581931560831 ) * u + 0.956464974744799006 ) * u +
			0.686265948274097816 ) * u + 0.434397492331430115 ) * u +
			0.244044510593190935 ) * t -
		z * exp( x * x - 0.120782237635245222 );
		x += s * ( x * s + 1 );

		if ( y > 1 )
			x = -x;
	}

	return x;
}

// Inverse of erff, calls erfc
double stats::invErff( double y )
{
#ifdef STATS_DEBUG
	util::screen() << "In stats::invErff" << endl;
#endif
	
	if ( y <= -1.0 )
		throw statsErr( "Argument <= -1 in routine invErff" );

	else if ( y >= 1.0 )
		throw statsErr( "Argument >= 1 in routine invErff" );

	else
		return stats::invErfc( 1 - y );
}

// Inverse of Zarea = sqrt(2) * invErff( 2 * A - 1 )
double stats::invZarea( double A )
{
#ifdef STATS_DEBUG
	util::screen() << "In stats::invZarea" << endl;
#endif
	
	if ( A <= 0.0 )
		throw statsErr( "Argument <= 0 in routine invZarea" );

	else if ( A >= 1.0 )
		throw statsErr( "Argument >= 1 in routine invZarea" );

	else if ( A == 0.5 )
		return 0;

	else
		return sqrt( 2.0 ) * invErff( 2 * A - 1 );
}

// Gaussian distribution functions
double stats::gaussD( double z )
{
#ifdef STATS_DEBUG
	util::screen() << "In stats::gaussD (1 arg)" << endl;
#endif
	return exp( z * z / - 2.0 ) / sqrt( 2 * 3.14159265359 );
}

double stats::gaussD( double x, double u, double s )
{
#ifdef STATS_DEBUG
	util::screen() << "In stats::gaussD (3 args)" << endl;
#endif
	return exp( ( x - u ) * ( x - u ) / ( - 2.0 * s * s ) )
		/ ( s * sqrt( 2 * 3.14159265359 ) );
}

// Kolmogorov-Smirnov probability function Numerical Recipes in C p. 626
#define EPS1 0.001
#define EPS2 1e-8
double stats::probks( double alam )
{
	double a2, fac = 2.0, sum = 0.0, termbf = 0.0;
	
#ifdef STATS_DEBUG
	util::screen() << "In stats::probks" << endl;
#endif

	a2 = -2.0 * alam * alam;

	for ( unsigned j = 1; j <= 100; j++ )
	{
		double term = fac * exp( a2 * j * j );
		sum += term;
		if ( fabs( term ) <= EPS1 * termbf || fabs( term ) <= EPS2 * sum )
			return sum;
		fac = -fac; // alternating signs in sum
		termbf = fabs( term );
	}

	return 1.0; // get here only by failing to converge
}
#undef EPS1
#undef EPS2

// Student's t-test for equal variance, returns p-value
double stats::ttest( vector< double >& v1, vector< double >& v2 )
{
	Population s1( v1 ), s2( v2 );
	double n1 = ( double ) v1.size();
	double n2 = ( double ) v2.size();

#ifdef STATS_DEBUG
	util::screen() << "In stats::ttest" << endl;
#endif
	
	// Sample sizes must be > 1
	assert( ( n1 > 1 ) && ( n2 > 1 ) );

	double df = n1 + n2 - 2;
	double svar = ( ( n1 - 1 ) * s1.var() + ( n2 - 1 ) * s2.var() ) / df;
	double t = ( s1.mean() - s2.mean() ) / sqrt( svar * ( 1.0 / n1 + 1.0 / n2 ) );
	return betai( 0.5 * df, 0.5, df / ( df + t * t ) );
}

// Student's t-test for unequal variance, returns p-value
double stats::tutest( vector< double >& v1, vector< double >& v2 )
{
	Population s1( v1 ), s2( v2 );
	double n1 = ( double ) v1.size();
	double n2 = ( double ) v2.size();
	double ave1 = s1.mean(), ave2 = s2.mean(), var1 = s1.var(), var2 = s2.var();

#ifdef STATS_DEBUG
	util::screen() << "In stats::tutest" << endl;
#endif
	
	// Sample sizes must be > 1
	assert( ( n1 > 1 ) && ( n2 > 1 ) );

	double t = ( ave1 - ave2 ) / sqrt( var1 / n1 + var2 / n2 );
	double df = sqr( var1 / n1 + var2 / n2 ) / ( sqr( var1 / n1 ) /
		( n1 - 1 ) + sqr( var2 / n2 ) / ( n2 - 1 ) );
	return betai( 0.5 * df, 0.5, df / ( df + sqr( t ) ) );
}

// Paired Student's t-test, returns p-value
double stats::tptest( vector< double >& v1, vector< double >& v2 )
{
	Population s1( v1 ), s2( v2 );
	double n1 = ( double ) v1.size();
	double n2 = ( double ) v2.size();

#ifdef STATS_DEBUG
	util::screen() << "In stats::tptest" << endl;
#endif
	
	// Sample sizes must be > 1 and equal for a paired test
	assert( ( n1 > 1 ) && ( n2 == n1 ) );

	double df = n1 - 1;

	// Calculate covariance 1st
	double cov = 0.0, ave1 = s1.mean(), ave2 = s2.mean();
	vector< double >::iterator p1, p2; // iterate through the samples
	for ( p1 = v1.begin(), p2 = v2.begin(); p1 != v1.end(); p1++, p2++ )
		cov += ( *p1 - ave1 ) * ( *p2 - ave2 );
	cov /= df;

	// Now compute the test
	double sd = sqrt( ( s1.var() + s2.var() - 2.0 * cov ) / n1 );
	double t = ( ave1 - ave2 ) / sd;
	return betai( 0.5 * df, 0.5, df / ( df + t * t ) );
}

// F-test, returns p-value
double stats::ftest( vector< double >& v1, vector< double >& v2 )
{
	Population s1( v1 ), s2( v2 );
	double n1 = ( double ) v1.size();
	double n2 = ( double ) v2.size();

#ifdef STATS_DEBUG
	util::screen() << "In stats::ftest" << endl;
#endif
	
	// Sample sizes must be > 1
	assert( ( n1 > 1 ) && ( n2 > 1 ) );
	
	double f, df1, df2, var1 = s1.var(), var2 = s2.var();

	if ( var1 > var2 )
	{
		f = var1 / var2;
		df1 = ( double ) ( n1 - 1 );
		df2 = ( double ) ( n2 - 1 );
	}
	else
	{
		f = var2 / var1;
		df1 = ( double ) ( n2 - 1 );
		df2 = ( double ) ( n1 - 1 );
	}

	double prob = 2.0 * betai( 0.5 * df2, 0.5 * df1, df2 / ( df2 + df1 * f ) );
	if ( prob > 1.0 )
		prob = 2.0 - prob;

	return prob;
}

// Member functions for Population Class

// Constructor loads Population dataset as vector< double >
Population::Population( vector< double >& v_in ) : varFlag ( false ), stddevFlag ( false ),
	squaresFlag ( false ), cubesFlag ( false )
{
#ifdef STATS_DEBUG
	util::screen() << "In Population:: constructor" << endl;
#endif
	assert ( v_in.size() > 1 ); // datasets must have 2 or more points

	x = v_in; // set the dataset to the incoming vector

	// Calculate mean for every Population loaded
	_mean = 0.0;
	_n = ( double ) x.size();
	vector< double >::iterator p;

	for ( p = x.begin(); p != x.end(); p++ )
		_mean += *p;

	_mean /= _n; // mean
}

// Accessor returns variance
double Population::var()
{
#ifdef STATS_DEBUG
	util::screen() << "In Population::var" << endl;
#endif
	
	if ( !squaresFlag ) // calculate sum of squares if necessary
	{
		sumSquares = sumPowers( 2 );
		squaresFlag = true; // sum of squares now calculated
	}

	// Calculate and return variance, note correction. Summing the squared
	//    deviations from the mean directly is the two-pass form; the textbook
	//    one-pass ( sumSquares - n * mean^2 ) / ( n - 1 ) differences two
	//    nearly equal quantities when the values are nearly equal, losing the
	//    difference to roundoff. It came out with either sign where the true
	//    variance is zero: negative made std() return sqrt of a negative
	//    number -- NaN, which then poisoned everything downstream -- and
	//    positive gave a fake standard deviation around 3e-08. (The zROC
	//    binning reaches exactly this: a bin covering a flat run of the
	//    empirical ROC holds identical z values, so its standard deviation is
	//    legitimately zero. See twoset.cpp.) This form returns that zero
	//    exactly, which fitexy's chixy() already handles. sumSquares stays
	//    calculated above because kurtosis() relies on var() to do it.
	double devSquares = 0;
	for ( vector< double >::iterator p = x.begin(); p != x.end(); p++ )
	{
		double dev = *p - _mean;
		devSquares += dev * dev;
	}

	_var = devSquares / ( _n - 1 );

	varFlag = true; // variance now calculated

	return _var;
}

// Accessor returns standard deviation
double Population::std()
{
#ifdef STATS_DEBUG
	util::screen() << "In Population::std" << endl;
#endif
	
	if ( !varFlag ) // if variance not calculated
		Population::var(); // calculate it

	// Calculate and return standard deviation
	_std = sqrt( _var );

	stddevFlag = true; // standard deviation now calculated

	return _std;
}

// Accessor returns skew
double Population::skew()
{
#ifdef STATS_DEBUG
	util::screen() << "In Population::skew" << endl;
#endif
	
	assert ( _n > 2 ); // must have 3 or more data points to calculate skew

	if ( !stddevFlag ) // if standard deviation not calculated
		Population::std(); // calculate it

	if ( _std == 0.0 ) // can't calculate skew if std dev 0
		throw PopulationErr( "Std dev 0 in routine skew" );
	else
	{
		if ( !cubesFlag ) // calculate sum of cubes if necessary, sum of squares
		                  // will already be calculated by std() by var()
		{
			sumCubes = sumPowers( 3 );
			cubesFlag = true; // sum of cubes now calculated
		}

		// Calculate sum of (x - mean)^3 by expansion
		double S3 = sumCubes - ( 3 * _mean * sumSquares ) + ( 2 * _n * _mean * _mean * _mean );

		// Divide by cube of standard deviation, use correction factors
		return ( _n * S3 ) / ( _var * _std * ( _n - 1 ) * ( _n - 2 ) );
	}
}

// Accessor returns kurtosis
double Population::kurtosis()
{
#ifdef STATS_DEBUG
	util::screen() << "In Population::kurtosis" << endl;
#endif
	
	assert ( _n > 3 ); // must have 4 or more data points to calculate kurtosis

	if ( !stddevFlag ) // if standard deviation not calculated
		Population::std(); // calculate it

	if ( _std == 0.0 ) // can't calculate kurtosis if std dev 0
		throw PopulationErr( "Std dev 0 in routine kurtosis" );
	else
	{
		if ( !cubesFlag ) // calculate sum of cubes if necessary, sum of squares
		                  // will already be calculated by std() by var()
		{
			sumCubes = sumPowers( 3 );
			cubesFlag = true; // sum of cubes now calculated
		}

		// Calculate sum of 4th powers
		double sum4s = sumPowers( 4 );

		double _mean2 = _mean * _mean; // to simplify

		// Calculate sum of (x - mean)^4 by expansion
		double S4 = sum4s - ( 4 * _mean * sumCubes ) + ( 6 * _mean2 * sumSquares )
			- ( 3 * _n * _mean2 * _mean2 );

		// Divide by 4th power of standard deviation
		S4 /= ( _var * _var );

		// Correction factors
		S4 *= _n * ( _n + 1 ) / ( ( _n - 1 ) * ( _n - 2 ) * ( _n - 3 ) );
		S4 -= 3 * ( _n - 1 ) * ( _n - 1 ) / ( ( _n - 2 ) * ( _n - 3 ) );

		return S4;
	}
}

// Utility method to return sum of the powers of dataset elements, argument is power
double Population::sumPowers( unsigned pow )
{
	double sum = 0.0;
	vector< double >::iterator p;

#ifdef STATS_DEBUG
	util::screen() << "In Population::sumPowers" << endl;
#endif

	// Do the sum
	if ( pow == 2 )  // squares
		for ( p = x.begin(); p != x.end(); p++ )
			sum += *p * *p;
	else // powers greater than 2
		for ( p = x.begin(); p != x.end(); p++ )
		{
			double product = 1.0;
			for ( unsigned i = 0; i < pow; i++ ) // inner loop for integral powers
				product *= *p;
			sum += product;
		}

	return sum;
}

// Member functions for XY Class

// Constructor loads XY dataset as 2 vector< double >s, x and y
// without standard deviations
XY::XY( vector< double >& _x, vector< double >& _y )
{
#ifdef STATS_DEBUG
	util::screen() << "In XY:: constructor with X Y vectors" << endl;
#endif
	
	assert ( _x.size() == _y.size() ); // coordinates must match up
	x = _x;
	y = _y;
	ndata = _x.size();
	mwt = false; // no standard deviations
	try { fit(); }
	catch ( XY::XYErr& e ) { throw XYErr( e.what() ); }
}

// Constructor loads XY dataset as 2 vector< double >s, x and y
//    with standard deviations vector< double > sig
XY::XY( vector< double >& _x, vector< double >& _y, vector< double >& _sig )
{
#ifdef STATS_DEBUG
	util::screen() << "In XY:: constructor with X Y and 1 std vectors" << endl;
#endif
	
	// Coordinates must match up
	assert ( ( _x.size() == _y.size() ) && ( _x.size() == _sig.size() ) );
	x = _x;
	y = _y;
	sig = _sig;
	ndata = _x.size();
	mwt = true; // standard deviations present
	try { fit(); }
	catch ( XY::XYErr& e ) { throw XYErr( e.what() ); }
}

// Utility function for modeling straight line, Numerical Recipes in C p.665
void XY::fit()
{
	unsigned i;
	double wt, t, sxoss, sx = 0.0, sy = 0.0, st2 = 0.0, ss, sigdat;
	
#ifdef STATS_DEBUG
	util::screen() << "In XY::fit" << endl;
#endif

	// If standard deviations are passed, make sure they're nonzero
	bool zeroFlag = false;
	for ( i = 0; i < ndata; i++ )
		if ( mwt && ( sig[ i ] == 0 ) )
		{
			zeroFlag = true;
			break;
		}

	if ( zeroFlag )
	{
		throw XYErr( "Zero std dev in XY::fit()" );
	}
	else
	{
		_b = 0.0;
		if ( mwt )
		{
			ss = 0.0;
			for ( i = 0; i < ndata; i++ )
			{
				wt = 1.0 / sqr( sig[ i ] );
				ss += wt;
				sx += x[ i ] * wt;
				sy += y[ i ] * wt;
			}
		}
		else
		{
			for ( i = 0; i < ndata; i++ )
			{
				sx += x[ i ];
				sy += y[ i ];
			}
			ss = ndata;
		}
		sxoss = sx / ss;
		if ( mwt )
		{
			for ( i = 0; i < ndata; i++ )
			{
				t = ( x[ i ] - sxoss ) / sig[ i ];
				st2 += t * t;
				_b += t * y[ i ] / sig[ i ];
			}
		}
		else
		{
			for ( i = 0; i < ndata; i++ )
			{
				t = x[ i ] - sxoss;
				st2 += t * t;
				_b += t * y[ i ];
			}
		}
		_b /= st2;
		_a = ( sy - sx * _b ) / ss;
		_siga = sqrt( ( 1.0 + sx * sx / ( ss * st2 ) ) / ss );
		_sigb = sqrt( 1.0 / st2 );
		_chi2 = 0.0;
		if ( !mwt )
		{
			for ( i = 0; i < ndata; i++ )
				_chi2 += sqr( y[ i ] - _a - _b * x[ i ] );
			_q = 1.0;
			sigdat = sqrt( _chi2 / ( ndata - 2 ) );
			_siga *= sigdat;
			_sigb *= sigdat;
		}
		else
		{
			for ( i = 0; i < ndata; i++ )
				_chi2 += sqr( ( y[ i ] - _a - _b * x[ i ] ) / sig[ i ] );
			_q = fitProbability( _chi2, ndata );
		}
		_r = ( -sx / ( ss * st2 ) ) / ( _siga * _sigb );
	}
}

// Constructor loads XY dataset with errors in both x & y as
//    2 vector< double >s, x and y with standard deviations
//    vector< double >s sigx and sigy, Numerical Recipes in C p.668
XY::XY( vector< double >& __x, vector< double >& __y, vector< double >& sigx,
	   vector< double >& sigy )
{
#ifdef STATS_DEBUG
	util::screen() << "In XY:: constructor with X Y and 2 std vectors" << endl;
#endif
	
	x = __x, y = __y;

	// NR's fitexy does not provide the a-b parameter correlation; set it to 0
	//    so r() never returns uninitialized memory on this path
	_r = 0;

	// Insure correct size vectors
	assert ( ( x.size() == y.size() ) && ( sigx.size() == sigy.size() )
		&& ( x.size() == sigx.size() ) );

	unsigned j;
	double swap, amx, amn, varx, vary, ang[7], ch[7], scale, bmn, bmx, d1, d2, r2;

	// Local copies
	double a, b, siga, sigb, chi2, q;

	unsigned ndat = x.size();

	xx = x;
	yy.resize( ndat );
	sx = sigx;
	sy.resize( ndat );
	ww.resize( ndat );

	Population xPop( x ), yPop( y );
	varx = xPop.var();
	vary = yPop.var();
	scale = sqrt( varx / vary );
	nn = ndat;
	for ( j = 0; j < ndat; j++)
	{
		yy[ j ] = y[ j ] * scale;
		sy[ j ] = sigy[ j ] * scale;
		ww[ j ] = sqrt( sqr( sx[ j ] ) + sqr( sy[ j ] ) );
	}
	sig = ww;
	ndata = ndat;
	mwt = true;

	// The weighted fit here only seeds the minimization below with a starting
	//    slope, but being weighted it divides by each error bar, so a zero one
	//    either trips its own guard or makes the seed NaN. (Zero error bars
	//    reach here from the zROC binning: a bin covering a flat run of the
	//    empirical ROC holds identical z values, so its standard deviation is
	//    legitimately zero. See twoset.cpp.) chixy() below tolerates a zero
	//    error bar by construction -- it weights such a point BIG -- so the
	//    objective is well defined and only the seed is lost. Reseed unweighted:
	//    a starting angle for the same minimization of the same objective
	//    cannot change where a fit that already had a usable seed converges,
	//    which is why this is a fallback and not the default.
	bool seeded;
	try { fit(); seeded = isfinite( _b ); }
	catch ( XY::XYErr& ) { seeded = false; } // a zero error bar; seed unusable

	if ( !seeded )
	{
		mwt = false; // seed from the unweighted fit instead
		try { fit(); }
		catch ( XY::XYErr& e ) { throw XYErr( e.what() ); }
		mwt = true;
	}
	b = _b;

	offs = ang[ 1 ] = 0.0;
	ang[ 2 ] = atan( b );
	ang[ 4 ] = 0.0;
	ang[ 5 ] = ang[ 2 ];
	ang[ 6 ] = POTN;
	for ( j = 4; j <= 6; j++)
		ch[ j ] = chixy( ang[ j ] );
	try
	{
		Funct< XY > f1( *this, &XY::chixy );
		f1.mnbrak( ang[ 1 ], ang[ 2 ] );
		ang[ 1 ] = f1.mnbrak_ax();
		ang[ 2 ] = f1.mnbrak_bx();
		ang[ 3 ] = f1.mnbrak_cx();
		ch[ 1 ] = f1.mnbrak_fa();
		ch[ 2 ] = f1.mnbrak_fb();
		ch[ 3 ] = f1.mnbrak_fc();

		Funct< XY > f2( *this, &XY::chixy );
		f2.brent( ang[ 1 ], ang[ 2 ], ang[ 3 ], ACC );
		b = f2.brent_xmin();
		chi2 = chixy( b );
		a = aa;

		q = fitProbability( chi2, nn );

		for ( r2 = 0.0 ,j = 0; j < nn; j++ )
			r2 += ww[ j ];
		r2 = 1.0 / r2;
		bmx = BIG;
		bmn = BIG;
		offs = chi2 + 1.0;
		for ( j = 1; j <= 6; j++ )
		{
			if ( ch[ j ] > offs)
			{
				d1 = fabs( ang[ j ] - b );
				while ( d1 >= PI )
					d1 -= PI;
				d2 = PI - d1;
				if ( ang[ j ] < b)
				{
					swap = d1;
					d1 = d2;
					d2 = swap;
				}
				if ( d1 < bmx )
					bmx = d1;
				if ( d2 < bmn )
					bmn = d2;
			}
		}
		if ( bmx < BIG )
		{
			Funct< XY > f3( *this, &XY::chixy );
			bmx = f3.zbrent( b, b + bmx, ACC ) - b;
			amx = aa - a;
			bmn = f3.zbrent( b, b - bmn, ACC ) - b;
			amn = aa - a;
			sigb = sqrt( 0.5 * ( bmx * bmx + bmn * bmn ) ) /
				( scale * sqr( cos ( b ) ) );
			siga = sqrt( 0.5 * ( amx * amx + amn * amn ) + r2 ) / scale;
		}
		else
			sigb = siga = BIG;
		a /= scale;
		b = tan( b ) / scale;

		_a = a, _b = b, _siga = siga, _sigb = sigb, _chi2 = chi2, _q = q; // return results
	}
	catch ( Funct< XY >::functErr& e ) { throw XYErr( e.what() ); }
}

// Captive function of fitexy, see Numerical Recipes in C p.670
double XY::chixy( double bang )
{
	unsigned j;
	double ans, avex = 0.0, avey = 0.0, sumw = 0.0, b;

#ifdef STATS_DEBUG
	util::screen() << "In XY::chixy" << endl;
#endif
	
	b = tan( bang );
	for ( j = 0; j < nn; j++ )
	{
		ww[ j ] = sqr( b * sx[ j ] ) + sqr( sy[ j ] );
		sumw += (ww[ j ] = ( ww[ j ] == 0.0 ? BIG : 1.0 / ww[ j ] ) );
		avex += ww[ j ] * xx[ j ];
		avey += ww[ j ] * yy[ j ];
	}
	if ( sumw == 0.0 )
		sumw = BIG;
	avex /= sumw;
	avey /= sumw;
	aa = avey - b * avex;
	for ( ans = -offs, j = 0; j < nn; j++ )
		ans += ww[ j ] * sqr( yy[ j ] - aa - b * xx[ j ] );
	return ans;
}

#undef ITMAX
#undef EPS
#undef FPMIN
#undef POTN
#undef BIG
#undef PI
#undef ACC
#undef BIG
