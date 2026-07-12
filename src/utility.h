// Utility methods header for neUROn2++

// NOTE: Template functions and headers are here, everything else is
//    utility.cpp

#ifndef UTILITY_H
#define UTILITY_H

// Because, apparently, vector< vector< unsigned > > symbol names exceeds 255
//    characters in length, and MSCV will error with that (ugly!)
#ifdef _MSC_VER	// MSVC
#pragma warning (disable: 4786)
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cassert>
#include <stdexcept>

using namespace std;

namespace util {
	// util exception, extends runtime_error class
	class utilErr : public runtime_error {
	public:
		utilErr( const char* message ) throw() : runtime_error( message ) {}
	};

	// Random number methods:
	// Seeds the generator with a user-specified seed for reproducible runs;
	//    overrides the timestamp seeding in d_random()
	void set_seed( unsigned );
	// No arguments seeds generator with timestamp (unless set_seed was called)
	void d_random();
	// 1 argument returns random number between -arg and +arg
	double d_random( double );
	// 2 arguments returns random number between arguments
	double d_random( double, double );
	// Returns a uniform random integer in [0, n-1]
	unsigned i_random( unsigned );

	// Rounds to number of significant digits
	double round( double, unsigned );

	// Ask methods:
	// asks question with a lower bound, returns int
	int askI( const string&, const int );
	// asks question with bounds, returns int
	int askI( const string&, const int, const int );
	// asks question with a lower bound, returns double
	double askD( const string&, const double );
	// asks question with bounds, returns double
	double askD( const string&, const double, const double );
	// asks yes, no question
	bool askYesNo( const string& );

	// Utility function which tests a filename to see if the file exists,
	//    returns the filename if it does, and keeps querying for a good
	//    filename if it doesn't.
	string& getGoodFile( string& );

	// Run directory for engine log files (neuron.log, model.txt): they are
	//    written next to the session's first-loaded dataset file, not the cwd
	void set_run_dir( const string& ); // remember a dataset file's directory
	string run_path( const string& ); // resolve a bare log filename into it

	// Removes carriage return from end of string if exists, returns string
	string& chopEndl( string& );

	// Method which parses a string to determine variable representation from
	//    nodes, and returns a *new* vector of vector of unsigned
	//    ";" delimits variables
	//    "," delimits nodes
	//    "-" specifies a range of nodes
	vector< vector< unsigned > > variable_parse( const string& );
	// Same, passed ifstream holds string
	vector< vector< unsigned > > variable_parse( ifstream& );

	// Method which removes a variable from a vector of vector of unsigned
	//    representation of variables (1st argument), returns a reference to
	//    the now manipulated vector of vector of unsigned, takes position of
	//    variable as 2nd argument, populates vector of vector of unsigned as
	//    3rd argument with result
	vector< vector< unsigned > >& remove_variable( const vector< vector< unsigned > >&,
		const unsigned, vector< vector< unsigned > >& );

	// As above, except returns *new* vector of vector of unsigned (no 3rd argument)
	vector< vector< unsigned > > remove_variable( const vector< vector< unsigned > >&,
		const unsigned );
}

// timestamp class and overloaded << operator allow timestamp method
//    to be inserted in an ostream
// Calling sequence: timestamp ( unsigned x )
//    where x is the number of seconds to print out in HHHH:MM:SS format
// Example:
//    cout << timestamp( 7261 ) << endl;
// prints out 02:01:01 then carriage return
class timestamp
{
	friend ostream& operator << ( ostream& output, const timestamp& t );

	public:
		timestamp ( const unsigned time_value ) { private_time = time_value; }

	private:
		unsigned private_time; // the passed number of seconds
};

#endif
