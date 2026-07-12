// Utility methods for neUROn2++

#include "stdafx.h" // For MSVC, must be first!

// Because, apparently, vector< vector< unsigned > > symbol names exceeds 255
//    characters in length, and MSCV will error with that (ugly!)
#ifdef _MSC_VER	// MSVC
#pragma warning (disable: 4786)
#endif

#include <algorithm>
#include <random>  // std::mt19937
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <time.h>
#include <string>
#include <sstream>
#include <vector>
#include <list>
#include <cstdlib> // for atoi and atof

using namespace std;

#include "utility.h"

// Random number methods

// Mersenne Twister generator (std::mt19937): unlike rand(), the C++ standard
//    specifies its output stream exactly, so a given seed produces identical
//    runs on every platform and compiler (MSVC's rand() has RAND_MAX = 32767).
//    Raw generator output is mapped to doubles here rather than through
//    std::uniform_real_distribution, whose implementation may vary.
static mt19937 rng;

// Set when the user supplies an explicit seed (--seed); suppresses timestamp seeding
static bool userSeededFlag = false;

// Seeds the generator with a user-specified seed for reproducible runs
void util::set_seed( unsigned seed )
{
	rng.seed( seed );
	userSeededFlag = true;
}

// No arguments seeds generator with timestamp (unless set_seed was called)
void util::d_random()
{
	// Insure that this is only done once per program run
	static bool firstFlag = true;

	if ( firstFlag && !userSeededFlag )
		rng.seed( ( unsigned ) time( 0 ) );

	firstFlag = false;
}

// 1 argument returns random number between -arg and +arg
double util::d_random( double limit )
{
	return limit * ( 2 * ( double ) rng() / ( double ) mt19937::max() - 1 );
}

// 2 arguments returns random number between arguments
double util::d_random( double lower, double upper )
{
	return ( upper - lower ) * ( double ) rng() / ( double ) mt19937::max()
		+ lower;
}

// Returns a uniform random integer in [0, n-1] (replaces legacy rand() % n)
unsigned util::i_random( unsigned n )
{
	return ( unsigned )( rng() % n );
}

// Rounds to number of significant digits, first argument is number
//    to be rounded (double), second is number of digits (unsigned)
// Returned is rounded number
double util::round( double x, unsigned d )
{
	assert( d > 0 ); // can't have no significant digits!

	if ( x!= 0 ) // if 0, just return 0
	{
		// Get the exponent base 10
		double n = floor( log10( fabs( x ) ) );

		// Round
		x /= pow( ( double ) 10, ( double )( n - d + 1 ) ); // moves decimal to right place to truncate
		x += ( x > 0 ? 0.5 : -0.5 ); // round up if positive, down if negative
		x = ( x > 0 ? floor( x ) : ceil( x ) ); // truncate after decimal
		x *= pow( ( double ) 10, ( double )( n - d + 1 ) ); // return original decimal
	}

	return x;
}

// overloaded << operator for timestamp class and method
ostream& operator << ( ostream& output, const timestamp& t )
{
	unsigned seconds, minutes, hours;

	// Calculate hours, minutes, seconds
	seconds = t.private_time % 60;
	minutes = t.private_time / 60;
	hours = minutes / 60;
	minutes %= 60;

	// Insert timestamp into the ostream
	output << setw( 4 ) << setfill( '0' ) << hours << ':';
	output << setw( 2 ) << setfill( '0' ) << minutes << ':';
	output << setw( 2 ) << setfill( '0' ) << seconds;

	return output; // enables remainder of the stream
}

// Ask methods

// Guard called after every interactive read: if cin has died (EOF on piped
//    input, or a failed extraction that would otherwise never recover), the
//    reask loops below would spin forever writing prompts — exit instead.
//    (In 2.x a script that ran out of answers looped infinitely.)
static void check_input_ok()
{
	if ( !cin )
	{
		cerr << endl << "Input ended unexpectedly -- exiting." << endl;
		exit( 1 );
	}
}

// askI method, asks integer question with a lower bound
// Arguments: string& question, int lower bound
// Example: number = askI( "What is your number? ", 1 );
int util::askI( const string& question, const int lower )
{
	string inString; // string to hold input number

	int value; // the answer

	// Ask the question, get the answer
	cout << question;
	cin >> inString;
	check_input_ok();

	// Conver the string to integer
	value = atoi( inString.c_str() );

	// Bounds check, reask question if necessary
	while ( value < lower || ( double ) value != atof( inString.c_str() ) )
	{
		cout << "Let's try that again: your answer must be at least ";
		cout << lower << "," << endl;
		cout << "and it must be an integer." << endl;
		cout << question;
		cin >> inString;
		check_input_ok();
		value = atoi( inString.c_str() );
	}

	return value;
}

// askI method, asks integer question with bounds
// Arguments: string& question, int lower bound, int upper bound
// Example: number = askI( "What is your number? ", 1, 10 );
int util::askI( const string& question, const int lower, const int upper )
{
	string inString; // string to hold input number

	int value; // the answer

	// Ask the question, get the answer
	cout << question;
	cin >> inString;
	check_input_ok();

	// Conver the string to integer
	value = atoi( inString.c_str() );

	// Bounds check, reask question if necessary
	while ( value < lower || value > upper ||
		( double ) value != atof( inString.c_str() ) )
	{
		cout << "Let's try that again: your answer must be at least ";
		cout << lower << ", but not greater than " << upper << "," << endl;
		cout << "and it must be an integer." << endl;
		cout << question;
		cin >> inString;
		check_input_ok();
		value = atoi( inString.c_str() );
	}

	return value;
}

// askD method, asks double question with a lower bound
// Arguments: string& question, double lower bound
// Example: number = askD( "What is your number? ", 1 );
double util::askD( const string& question, const double lower )
{
	double value; // the answer

	// Ask the question, get the answer
	cout << question;
	cin >> value;
	check_input_ok();

	// Bounds check, reask question if necessary
	while ( value < lower )
	{
		cout << "Let's try that again: your answer must be at least ";
		cout << lower << "." << endl;
		cout << question;
		cin >> value;
		check_input_ok();
	}

	return value;
}

// askD method, asks double question with bounds
// Arguments: string& question, double lower bound, double upper bound
// Example: number = askI( "What is your number? ", 1, 10 );
double util::askD( const string& question, const double lower, const double upper )
{
	double value; // the answer

	// Ask the question, get the answer
	cout << question;
	cin >> value;
	check_input_ok();

	// Bounds check, reask question if necessary
	while ( value < lower || value > upper )
	{
		cout << "Let's try that again: your answer must be at least ";
		cout << lower << ", but not greater than " << upper << "." << endl;
		cout << question;
		cin >> value;
		check_input_ok();
	}

	return value;
}

// askYesNo method, asks yes or no question
// Arguments: string& question
// Example: answer = askYesNo( "Do you want to start again? " );
bool util::askYesNo( const string& question )
{
	bool answer = false, goodAnswer = false; // returned value, valid flag
	string y1( "yes" ), y2( "y" ), n1( "no" ), n2( "n" ), // possible answers
		s; // user answer

	// Ask the question, get the answer
	cout << question;
	cin >> s;
	check_input_ok();

	// Change to lower case
	int ( *f )( int );
	f = &tolower;
	transform( s.begin(), s.end(), s.begin(), f );

	if ( s == y1 || s == y2 ) // yes answer
	{
		answer = true;
		goodAnswer = true;
	}
	else if ( s == n1 || s == n2 ) // no answer
	{
		answer = false;
		goodAnswer = true;
	}

	while ( !goodAnswer ) // invalid answer
	{
		cout << "Let's try that again: a yes or no answer is required." << endl;
		cout << question;
		cin >> s;
		check_input_ok();

		// Convert to lower case
		int ( *f )( int );
		f = &tolower;
		transform( s.begin(), s.end(), s.begin(), f );

		if ( s == y1 || s == y2 )
		{
			answer = true;
			goodAnswer = true;
		}
		else if ( s == n1 || s == n2 )
		{
			answer = false;
			goodAnswer = true;
		}
	}

	return answer;
}

// Utility function which tests a filename to see if the file exists,
//    returns the filename if it does, and keeps querying for a good
//    filename if it doesn't.
string& util::getGoodFile( string& filename )
{
	bool openfile = false; // file exists flag

	// Try to open the file
	ifstream inputFile( filename.c_str(), ios::in );
	if ( inputFile ) openfile = true; // success
	while ( !openfile ) { // keep asking until a valid name is entered
		cout << "I can't find a file named " << filename << "." << endl;
		cout << "Please enter a valid file name: ";
		cin >> filename;
		check_input_ok();
		ifstream inputFile( filename.c_str(), ios::in ); // test file for input
		if ( inputFile ) openfile = true; // success
	}
	inputFile.close(); // close & clear the file
	inputFile.clear();

	return filename;
}

// Run directory: log files follow the data. The first dataset file loaded
//    in a session fixes the directory; later loads don't move it.
static string runDir;
static bool runDirSet = false;

void util::set_run_dir( const string& fromFile )
{
	if ( runDirSet ) // first dataset load wins
		return;

	string::size_type slash = fromFile.find_last_of( "/\\" );
	runDir = ( slash == string::npos ? "" : fromFile.substr( 0, slash + 1 ) );
	runDirSet = true;
}

string util::run_path( const string& filename )
{
	// Only bare names are redirected; an explicit path is respected
	if ( filename.find_first_of( "/\\" ) != string::npos )
		return filename;

	return runDir + filename;
}

// Engine-core output stream, redirectable for GUI/server capture
static ostream* screenPtr = &cout;

ostream& util::screen()
{
	return *screenPtr;
}

void util::set_screen( ostream& newScreen )
{
	screenPtr = &newScreen;
}

// Utility function which removes carriage return from end of string argument
//    if exists, returns manipulated string
string& util::chopEndl( string& stringArg )
{
	if ( ( int ) stringArg[ stringArg.size() - 1 ] == 13 )
		stringArg.resize( stringArg.size() - 1 );

	return stringArg;
}

// Method which parses a string to determine variable representation from nodes,
//    and returns a *new* vector of vector of unsigned
//    ";" delimits variables
//    "," delimits nodes
//    "-" specifies a range of nodes
// For example: passed string "0-3,5; 4,8; 6,7" returns vector of vectors
//    0 1 2 3 5
//    4 8
//    6 7
vector< vector< unsigned > > util::variable_parse( const string& inString )
{
	vector< vector< unsigned > > result;

	bool errorFlag = false; // indicates a parse error

	unsigned counter; // multipurpose whole integer counter

	string goodChars = "0123456789,;- ", // good characters
		errorString = "util variable_parse error: "; // error message

	stringstream out; // to construct string to return in exception message

	// Search through string to find any bad characters, and report if found
	if ( inString.find_first_not_of( goodChars ) != string::npos )
	{
		out << errorString << "bad character found at string position "
			<< inString.find_first_not_of( goodChars ) << " ( "
			<< inString[ inString.find_first_not_of( goodChars ) ]
			<< " )" << ends;

		// c_str() is added to get c type string
		throw utilErr( out.str().c_str() );

		errorFlag = true;
	}

	else // only good characters were found
	{
		vector< char > vString, // holds the entire string in a vector, because
			                    // MSVC++ doesn't support vector< string > :(
			tmpString; // holds "substrings" parsed by semicolons

		// Copy the incoming string into the character vector
		for ( unsigned c = 0; c < inString.size(); c++ )
			vString.push_back( inString[ c ] );

		vector< vector< char > > varStrings; // holds substrings representing variables

		counter = 0; // character counter in "string"

		// Break string into substrings representing variables, delimited by semicolons
		while ( counter != vString.size() )
		{
			// ignore spaces
			if ( vString[ counter ] == ' ' )
				counter++;

			else // character's not a space
			{
				if ( vString[ counter ] != ';' ) // if not the semicolon delimeter
				{
					tmpString.push_back( vString[ counter ] ); // append character
					counter++;
				}
				else // it's a semicolon delimeter, so break into substring
				{
					// If size of temporary substring nonzero, add to substring vector
					if ( tmpString.size() )
						varStrings.push_back( tmpString );

					tmpString.clear(); // reset temporary substring holder
					counter++;
				}
			}
		}

		// For last substring after semicolon, if it exists, add to substring vector
		if ( tmpString.size() )
			varStrings.push_back( tmpString );

		// Break substrings delimited by semicolons into subsubstrings delimited by
		//    commas
		for ( unsigned varNum = 0; varNum < varStrings.size(); varNum++ )
		{
			if ( errorFlag ) // if error found last time, no sense in continuing
				break;

			vector< unsigned > uVar; // a vector within the returned object
			result.push_back( uVar ); // make room for it in the returned object

			counter = 0; // reset counter

			tmpString.clear(); // clear the temporary "string" holder

			vector< vector< char > > nRanges; // holds numbers and ranges

			// Break semicolon delimited substrings into comma delimited subsubstrings
			while ( counter != varStrings[ varNum ].size() )
			{
				if ( varStrings[ varNum ][ counter ] != ',' ) // if not comma delimeter
				{
					// Append character and advance
					tmpString.push_back( varStrings[ varNum ][ counter ] );
					counter++;
				}
				else // it's a comma delimeter, so break into substring
				{
					// If temporary substring size nonzero, add to subsubstring vector
					if ( tmpString.size() )
						nRanges.push_back( tmpString );

					tmpString.clear(); // reset temporary subsubstring holder
					counter++;
				}
			}

			// If last subsubstring after comma exists, add to subsubstring vector
			if ( tmpString.size() )
				nRanges.push_back( tmpString );

			// Cycle through each number or range of numbers, and interpret
			for ( unsigned rNum = 0; rNum < nRanges.size(); rNum++ )
			{
				if ( errorFlag ) // if error found last time, no sense in continuing
					break;

				counter = 0; // use the counter now to count hyphens
				unsigned hyphens = 0; // number of hyphens
				for ( counter = 0; counter < nRanges[ rNum ].size(); counter++ )
					if ( nRanges[ rNum ][ counter ] == '-' )
						hyphens++;

				if ( hyphens > 1 ) // too many hyphens
				{
					out << errorString << "too many hyphens" << ends;
					errorFlag = true;

					// c_str() added to get c style string
					throw utilErr( out.str().c_str() );
					break;
				}

				else if ( hyphens == 1 ) // one hyphen, so it should be for a range
				{
					// Make sure numbers are on both ends of the hyphen
					if ( nRanges[ rNum ][ 0 ] == '-' ||
						nRanges[ rNum ][ nRanges[ rNum ].size() - 1 ] == '-' )
					{
						out << errorString << "hyphen without a number on both ends"
							<< ends;
						errorFlag = true;
						throw utilErr( out.str().c_str() );
						break;
					}

					else // parse a range
					{
						string n1 = "", n2 = ""; // numbers on each side of the hyphen

						bool hyphenFlag = false; // found a hyphen in the "subsubstring"

						// Cycle through the range "subsubstring"
						for ( unsigned rCount = 0; rCount < nRanges[ rNum ].size(); rCount++ )
						{
							if ( nRanges[ rNum ][ rCount ] == '-' )
								hyphenFlag = true; // found a hyphen
							else // append to the appropriate number
							{
								if ( !hyphenFlag )
									n1 += nRanges[ rNum ][ rCount ];
								else
									n2 += nRanges[ rNum ][ rCount ];
							}
						}

						unsigned num1 = atoi( n1.c_str() ), // convert numbers to integers
							num2 = atoi( n2.c_str() );

						// Error check range
						if ( num1 > num2 ) // 1st number must be less than 2nd
						{
							out << errorString << "1st number in range greater than 2nd";
							errorFlag = true;
							throw utilErr( out.str().c_str() );
							break;
						}

						else if ( num1 == num2 ) // numbers must be different
						{
							out << errorString << "numbers in range are the same";
							errorFlag = true;
							throw utilErr( out.str().c_str() );
							break;
						}

						else // range is good, expand it & put into results object
						{
							for ( unsigned uRange = num1; uRange <= num2; uRange++ )
								result[ varNum ].push_back( uRange );
						}
					}
				}

				else // no hyphens
				{
					string nrString; // for a non-range "subsubstring"

					// Cycle through non-range "subsubstring"
					for ( unsigned nrCount = 0; nrCount < nRanges[ rNum ].size(); nrCount++ )
						nrString += nRanges[ rNum ][ nrCount ];

					unsigned num = atoi( nrString.c_str() ); // convert to integer

					result[ varNum ].push_back( num ); // put into results object
				}
			}
		}
	}

	list< unsigned > populated; // to check if all numbers are represented in result

	// Make a list, "populated", which holds all values
	for ( counter = 0; counter < result.size(); counter++ )
	{
		// Sort the inner vector for prettiness on return
		sort( result[ counter ].begin(), result[ counter ].end() );

		// Append to list
		populated.insert( populated.end(), result[ counter ].begin(),
			result[ counter ].end() );
	}

	populated.sort(); // sort the list
	list< unsigned > uniqueValues = populated; // make a list of unique populated values
	uniqueValues.unique();                     // to check if anything is duplicated

	if ( *( populated.begin() ) != 0 ) // make sure list begins with zero
	{
		out << errorString << "missing zero";
		errorFlag = true;
		throw utilErr( out.str().c_str() );
	}
	// Check for any duplicated values
	else if ( uniqueValues.size() < populated.size() )
	{
		out << errorString << "duplicated positions";
		errorFlag = true;
		throw utilErr( out.str().c_str() );
	}
	// Check for missing values
	else if ( uniqueValues.size() < ( populated.back() + 1 ) )
	{
		out << errorString << "missing positions";
		errorFlag = true;
		throw utilErr( out.str().c_str() );
	}

	// If there was an error, no sense in returning a partial vector of vectors,
	//    so clear it
	if ( errorFlag )
		result.clear();

	return result; // return vector of vectors
}

// Method which parses a string to determine variable representation from nodes,
//    and returns a new vector of vector of unsigned, takes an ifstream file
//    for input, uses variable_parse( const string& inString ) above
vector< vector< unsigned > > util::variable_parse( ifstream& inFile )
{
	string line, // to pull lines from file
		parseline = ""; // the entire line

	// Cycle through file, building line
	while ( !inFile.eof() )
	{
		getline( inFile, line ); // pull a line
		chopEndl( line ); // remove carriage return if it exists
		parseline += line; // build entire line
	}

	inFile.close(); // close & clear the file
	inFile.clear();

	// Use previously encoded variable_parse method
	vector< vector< unsigned > > result = variable_parse( parseline );

	return result;
}

// Method which removes a variable from a vector of vector of unsigned
//    representation of variables (1st argument), returns a reference to
//    the now manipulated vector of vector of unsigned, takes position of
//    variable as 2nd argument, populates vector of vector of unsigned as
//    3rd argument with result
vector< vector< unsigned > >& util::remove_variable( const vector< vector< unsigned > >&
	variables, const unsigned position, vector< vector< unsigned > >& result )
{
	assert ( position < variables.size() ); // range check position

	result = variables; // copy incoming data structure into returned one

	// Cycle through each variable
	for ( unsigned i = 0; i < result.size(); i++ )
	{
		if ( i != position ) // except one to be removed
			for ( unsigned j = 0; j < result[ i ].size(); j++ )
			{
				unsigned counter = 0; // reset counter

				// Count the number of current variable positions greater than
				//    the positions of the variable to be removed
				for ( unsigned k = 0; k < result[ position ].size(); k++ )
					if ( result[ i ][ j ] > result[ position ][ k ] )
						counter++;

				result[ i ][ j ] -= counter; // readjust variable position
			}
	}

	// Now delete the variable itself
	vector< vector< unsigned > >::iterator p = result.begin() + position;
	result.erase( p, p + 1 );

	return result; // return the reference to this data structure
}

// As above, except returns *new* vector of vector of unsigned, and thus has no
//    3rd argument
vector< vector< unsigned > > util::remove_variable( const vector< vector< unsigned > >&
	variables, const unsigned position )
{
	// Create new vector of vector of unsigned, to be returned by method
	vector< vector< unsigned > > result;

	// Call previously coded method
	util::remove_variable( variables, position, result );

	return result;
}

