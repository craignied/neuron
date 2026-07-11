// function_defs.h
// Function definitions for neUROn2++

#ifndef FUNCTIONDEFS_H
#define FUNCTIONDEFS_H

#include <math.h>

// Class functions for passing to vector & matrix using func

// The sigmoidal function
class sigmoidal
{
	public:
		inline double operator()( const double& x )
		{
			return 1.0 / ( 1.0 + exp( -x ) );
		}
};

// The derivative of the sigmoidal function
class d_sigmoidal
{
	public:
		inline double operator()( const double& o )
		{
			return o * ( 1 - o );
		}
};

// Logarithm
class logarithm
{
	public:
		inline double operator()( const double& x )
		{
			return log( x );
		}
};

// Threshold
class threshold
{
	public:
		threshold ( double in_T ) : T ( in_T ) {} // ctor with scalar threshold
		double T; // the threshold
		inline double operator()( const double& x )
		{
			return ( double )( x >= T );
		}
};

// Absolute value
class absval
{
	public:

		inline double operator()( const double& x )
		{
			return fabs( x );
		}
};

// Class to implement error functions
class errorFunction
{
public:
	errorFunction(): defined ( false ), boundsErrorFlag ( false ) {} // default ctor
	// errorFunction( bool errorType, 
	~errorFunction(){} // destructor
	// Copy ctor uses copy utility
	errorFunction( const errorFunction& rhs ) { copy( rhs ); }
	// Overloaded = operator
	errorFunction& operator= ( const errorFunction& rhs )
	{
		if ( &rhs != this ) // check for self-assignment
			copy( rhs ); // use the copy utility
	
		return *this; // enables A = B = C
	}
	// Exception for undefined error function...if I decide to add different kinds
	//    of errors, I'll need to change errorType type bool to uint
	class UndefinedError // : public std::exception
	{
		public:
			const char * what() const { return "Error function not defined"; }
	};
	// Ctor for single sigmoidal output node, takes as arguments known value y,
	//    guess o, argument x in sigmoidal transfer function 1/(1+e^{-x}) and
	//    bool flag errorType (0 = LMS, 1 = X-entropy)
	errorFunction( double y, double o, double x, bool errorType )
	{
		if ( errorType ) // 1 = X-entropy
		{
			if ( o == 1 ) // if output was 1, log( 1 - o ) would error
			{
				boundsErrorFlag = true;
				E = x * ( 1 - y ); // approximate error
			}
			else if ( o == 0 ) // if output was 0, log( o ) would error
			{
				boundsErrorFlag = true;
				E = -1 * x * y; // approximate error
			}
			else // no approximation necessary
			{
				boundsErrorFlag = false;
				E = ( -1 * y * log( o ) ) - ( ( 1 - y ) * log( 1 - o ) );
			}
		}
		else // 0 = LMS
		{
			boundsErrorFlag = false;
			E = 0.5 * ( y - o ) * ( y - o );
		}
		defined = true; // an error has now been calculated
	}
	// Ctor for multiple sigmoidal output nodes, takes as arguments known values
	//    vector y, guess vector o, argument vector x in sigmoidal transfer function
	//    1/(1+e^{-x}) and bool flag errorType (0 = LMS, 1 = X-entropy)
	errorFunction( vector< double > y, vector< double > o, vector< double > x,
		bool errorType )
	{
		if ( errorType ) // 1 = X-entropy
		{
			// All vectors must of course be of equal size
			assert( ( y.size() == o.size() ) && ( o.size() == x.size() ) );

			E = 0; // initialize accumulated error

			// Let's do this through iterators
			vector< double >::iterator py, po, px;
	
			for ( py = y.begin(), po = o.begin(), px = x.begin(); py != y.end(); py++, po++, px++ )
			{
				if ( *po == 1 ) // if output was 1, log( 1 - o ) would error
				{
					boundsErrorFlag = true;
					E += *px * ( 1 - *py ); // approximate error
				}
				else if ( *po == 0 ) // if output was 0, log( o ) would error
				{
					boundsErrorFlag = true;
					E += -1 * *px * *py; // approximate error
				}
				else // no approximation necessary
				{
					boundsErrorFlag = false;
					E += ( -1 * *py * log( *po ) ) - ( ( 1 - *py ) * log( 1 - *po ) );
				}			
			}	
		}
		else // 0 = LMS
		{
			boundsErrorFlag = false;
			vector< double > errors = y - o;
			E = 0.5 * dotprod( errors , errors );
		}

		defined = true; // an error has now been calculated
	}
	double value() // accessor returns error function result
	{
		if ( !defined ) // throw exception if error not yet calculated
			throw UndefinedError();
		else
			return E;
	}
	bool boundsErr() // accessor returns if numerical bounds error (e.g. log(o))
	{
		if ( !defined ) // throw exception if error not yet calculated
			throw UndefinedError();
		else
			return boundsErrorFlag;
	}

private:
	double E; // numerical error result
	bool defined, // flag indicates if an error has been calculated
		multOutput, // false if single output, true if multiple outputs
		boundsErrorFlag; // flag to indicate if log(o) is out of bounds

	void copy( const errorFunction& rhs ) // copy utility
	{
		E = rhs.E;
		defined = rhs.defined;
		multOutput = rhs.multOutput;
		boundsErrorFlag = rhs.boundsErrorFlag;
	}
};

/* / These are *old* functions: NEVER uncomment them.  They are here for
     historical interest and instruction *only*

// LMS error for a single output, takes known value y and model's
//    guess o as arguments, returns LMS error
inline double singleLMSerror( double y, double o )
{
	return 0.5 * ( y - o ) * ( y - o );
}

// X-entropy error for a single output, takes known value y and model's
//    guess o as arguments, returns X-entropy error
inline double singleXEerror( double y, double o )
{
	return ( -1 * y * log( o ) ) - ( ( 1 - y ) * log( 1 - o ) );
}

// X-entropy error for a multi output, takes known vector of outputs y and model's
//    guess output as arguments, returns XE error
inline double multiXEerror( vector< double > y, vector< double > output )
{
	// Construct vectors to contain log( o ), log( 1 - o ) and 1 - y
	vector< double > log_o( output ), log_1_o( output ), one_minus_y( y );

	// Calculate 1 - y, log( o ) and log( 1 - o )
	( one_minus_y *= -1.0 ) += 1.0; // 1 - y
	( log_1_o *= -1.0 ) += 1.0; // 1 - o
	func( log_o, logarithm(), log_o ); // log( o )
	func( log_1_o, logarithm(), log_1_o ); // log ( 1 - o )

	// Calculate & return XE function result
	return ( -1 * dotprod( y , log_o ) ) - ( dotprod( one_minus_y , log_1_o ) );
}

// LMS error for a multiple outputs, takes known vector of outputs y and model's
//    guess output as arguments, returns LMS error
inline double multiLMSerror( vector< double > y, vector< double > output )
{
	vector< double > error = y - output;

	return 0.5 * dotprod( error , error );
} */

#endif
