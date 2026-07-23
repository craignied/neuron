// matrix.h

// Based on Matrix class from http://www.parashift.com/c++-faq-lite/

// Preprocessor directives to prevent multiple header file inclusion
#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <exception>
#include <cassert>

#include "utility.h"

using namespace std;

#define GaussJordan 1 // to specify Gauss Jordan elimination in inverse method

// Tony Makhlouf changes to file 9/11/2005:
// Moved operator <<(ostream&..) and operator >>(istream&..) template declarations
//    to here for for gcc 4.0.1 compiler compliance
//    forward declaration of Matrix since it is referred to in the operator << declaration
template < class T > class Matrix;

// Overloaded << operator for class Matrix
template< class T >
ostream& operator << ( ostream& output, const Matrix< T >& old )
{
	if ( old.outputHeaderFlag ) // output header if specified by flag
	{
		output << old.nrows_ << " rows and ";
		output << old.ncols_ << " columns:" << endl;
	}

	for ( unsigned i = 0; i < old.nrows_; i++ )
	{
		for ( unsigned j = 0; j < old.ncols_; j++ )
			output << old.data_[ i * old.ncols_ + j ] << " ";
		output << endl;
	}
	return output; // enables cout << A << B;
}

// Overloaded >> operator for class Matrix
//    nrows_ and ncols_ MUST have been already set!
template< class T >
istream& operator >> ( istream& input, Matrix< T >& in )
{
	// Make sure nrows_ and ncols_ are both nonzero
	assert ( in.nrows_ != 0 && in.ncols_ != 0 );

	// Then load the Matrix from the istream data
	for ( unsigned i = 0; i < in.nrows_; i++ )
		for ( unsigned j = 0; j < in.ncols_; j++ )
			input >> in.data_[ i * in.ncols_ + j ];

	return input; // enables cin >> A >> B;
}

template< class T >
class Matrix {
// Tony Makhlouf changes to file 9/13/2005:
//    As of the .NET edition, the MS compiler follows the same standard as GCC.
#if( _MSC_VER && _MSC_VER <= 1200 ) // MSVC version 6.0 SP 5.
	friend ostream& operator << ( ostream&, const Matrix< T >& );
	friend istream& operator >> ( istream&, Matrix< T >& );
#else // __GNUC__ or VC .NET ( MSVC 7+ )
	friend ostream& operator << < T >( ostream&, const Matrix< T >& );
	friend istream& operator >> < T >( istream&, Matrix< T >& );
#endif

public:
	// Constructors
	Matrix(); // takes no arguments ( initial sizes are zero )
	Matrix( const unsigned, const unsigned ); // dimensions as arguments
	Matrix( const unsigned, const unsigned, T ); // loads Matrix with 3rd argument
	Matrix( const vector< T >&, const vector< T >& ); // outer product of 2 vectors
	// Loads from file, filename and number columns as arguments
	Matrix( string&, const unsigned ); // filename string may change!
	// Loads from file, flag for report and filename string as arguments
	Matrix( const bool, string&  ); // filename string may change!

	// Throws a BadSize object if either size is zero
	class BadSize /* : public std::exception */
	{
		public:
			const char * what() const { return "Matrix dimension zero"; }
	};

	// Based on the Law Of The Big Three:
	~Matrix();  // destructor
	Matrix( const Matrix< T >& );  // copy constructor
	Matrix< T >& operator= ( const Matrix< T >& );  // = operator

	// Method to resize Matrix: remember the new Matrix is filled with GARBAGE!
	Matrix< T >& resize( const unsigned, const unsigned );

	// Method to clear a Matrix
	Matrix< T >& clear();

	// Method to fill a Matrix with a value
	Matrix< T >& fill( const T );

	// Loads existing Matrix from file, filename & number columns as arguments
	Matrix< T >& loadfile( string&, const unsigned );
	// Generic load from file, doesn't need to know number of columns,
	Matrix< T >& loadfile( const bool, string& ); // bool flag will print report

	// Saves a Matrix to a file, filename (string) is argument
	bool savefile( const string& ); // returns true if file successfully saved

	// Access methods to get the ( i, j ) element:
	T&       operator() ( const unsigned, const unsigned );
	const T& operator() ( const unsigned, const unsigned ) const;
	// These throw a BoundsViolation object if i or j is too big
	class BoundsViolation /* : public std::exception */
	{
		public:
			const char * what() const { return "Matrix bounds violation"; }
	};

	// Accessors for the number of rows, columns
	const unsigned rows() const { return nrows_; } // return number rows
	const unsigned cols() const { return ncols_; } // return number columns

	// Accessors for a submatrix
	Matrix< T >& submatrix( const unsigned, const unsigned,
		const unsigned, const unsigned, Matrix< T >& ) const;
	Matrix< T > submatrix( const unsigned, const unsigned,
		const unsigned, const unsigned ) const; // return new Matrix

	// Methods which add row or column to Matrix
	// Add a row to a Matrix, output Matrix reference passed as argument
	Matrix< T >& addrow( const vector< T >&, Matrix< T >& ) const;
	Matrix< T > addrow( const vector< T >& ) const; // return new Matrix
	// Add a column to a Matrix, output Matrix reference passed as argument
	Matrix< T >& addcol( const vector< T >&, Matrix< T >& ) const;
	Matrix< T > addcol( const vector< T >& ) const; // return new Matrix

	// Methods which replace a row or column in a Matrix
	Matrix< T >& replacerow( const unsigned, const vector< T >& ); // replace row
	Matrix< T >& replacecol( const unsigned, const vector< T >& ); // replace column

	// Accessors to return a vector from a row or column
	vector< T >& row( const unsigned, vector< T >& ) const;
	vector< T > row( const unsigned ) const; // return new vector
	vector< T >& col( const unsigned, vector< T >& ) const;
	vector< T > col( const unsigned ) const; // return new vector

	// Methods which include or exclude columns indicated by the passed vector
	// Include columns, output Matrix reference passed as argument
	Matrix< T >& includecols( Matrix< T >&, const vector< unsigned >& ) const;
	Matrix< T > includecols( const vector< unsigned >& ) const; // returns new Matrix
	// Exclude columns, output Matrix reference passed as argument
	Matrix< T >& excludecols( Matrix< T >&, const vector< unsigned >& ) const;
	Matrix< T > excludecols( const vector< unsigned >& ) const; // returns new Matrix

	// Methods which gather the rows indicated by the passed vector, IN THAT
	//    ORDER. Unlike includecols, positions may repeat and arrive unsorted:
	//    drawing a bootstrap resample with replacement is one gather
	Matrix< T >& includerows( Matrix< T >&, const vector< unsigned >& ) const;
	Matrix< T > includerows( const vector< unsigned >& ) const; // returns new Matrix

	// Simple math unary assignment operators (element by element)
	Matrix< T >& operator+= ( const Matrix< T >& ); // +=
	Matrix< T >& operator-= ( const Matrix< T >& ); // -=
	Matrix< T >& operator*= ( const Matrix< T >& ); // *=
	Matrix< T >& operator/= ( const Matrix< T >& ); // /=

	// Simple math binary operators (element by element)
	Matrix< T > operator+ ( const Matrix< T >& ) const; // +
	Matrix< T > operator- ( const Matrix< T >& ) const; // -
	Matrix< T > operator* ( const Matrix< T >& ) const; // *
	Matrix< T > operator/ ( const Matrix< T >& ) const; // /

	// Simple math unary assignment operators with scalars
	Matrix< T >& operator+= ( const T scalar ); // +=
	Matrix< T >& operator-= ( const T scalar ); // -=
	Matrix< T >& operator*= ( const T scalar ); // *=
	Matrix< T >& operator/= ( const T scalar ); // /=

	// Simple math binary operators with scalars
	Matrix< T > operator+ ( const T scalar ) const; // +
	Matrix< T > operator- ( const T scalar ) const; // -
	Matrix< T > operator* ( const T scalar ) const; // *
	Matrix< T > operator/ ( const T scalar ) const; // /

	// == and !=
	bool operator== ( const Matrix< T >& ); // ==
	bool operator!= ( const Matrix< T >& ); // !=

	// Transpose Matrix
	Matrix< T >& t( Matrix< T >& ) const;
	Matrix< T > t() const; // returns new Matrix

	// Dot product methods
	vector< T > dotprod( const vector< T >& ) const; // returns new vector
	// Output vector reference passed as argument
	vector< T >& dotprod( const vector< T >&, vector< T >& ) const;
	// Output vector reference and range passed as argument
	vector< T >& dotprod( const vector< T >&, vector< T >&, const unsigned,
		const unsigned ) const;
	// Transpose, then dot product
	vector< T > dotprodt( const vector< T >& ) const; // returns new vector
	// Output vector reference passed as argument
	vector< T >& dotprodt( const vector< T >&, vector< T >& ) const;
	// Output vector reference and range passed as argument
	vector< T >& dotprodt( const vector< T >&, vector< T >&, const unsigned,
		const unsigned ) const;
	// Methods which take a single row from a Matrix to use as a vector
	//    for the dot product, first returns a new vector:
	vector< T > dotprod_row( const Matrix< T >&, const unsigned ) const;
	// Output vector reference passed as argument
	vector< T >& dotprod_row( const Matrix< T >&, const unsigned, vector< T >& ) const;
	// Output vector reference and range passed as argument
	vector< T >& dotprod_row( const Matrix< T >&, const unsigned, vector< T >&,
		const unsigned, const unsigned ) const;
	// Method which perform dot product on 2 Matrices
	Matrix< T >& dotprod( const Matrix< T >&, Matrix< T >& ) const;
	Matrix< T > dotprod( const Matrix< T >& ) const; // returns new Matrix

	// Fills Matrix with outer product of 2 vectors
	Matrix< T >& outprod( const vector< T >&, const vector< T >& );

	// Compute sum of squares of Matrix elements
	T squared() const;

	// Compute the maximum absolute value of Matrix elements
	T maxabs() const;

	// Method which returns a vector from a Matrix column sums
	vector< T >& colsums( vector< T >& ) const; // returns result in passed vector
	vector< T > colsums() const; // returns new vector

	// Method which returns a vector from a Matrix row indexes, pass returned vector
	bool rowindex( vector< unsigned >& ) const; // returns true if successful

	// Matrix to vector
	vector< T >& toVector( vector< T >& ) const; // returns result in passed vector
	vector< T > toVector() const; // returns new vector

	// Fills Matrix< double > with random numbers
	Matrix< double >& random( const double );

	// Returns first-order covariance Matrix
	Matrix< double >& covariance( Matrix< double >& ) const; // pass returned Matrix
	Matrix< double > covariance() const; // return new Matrix

	// Matrix inverse and determinant
	Matrix< double >& inverse( Matrix< double >& ) const; // pass returned Matrix
	Matrix< double > inverse() const; // return new Matrix
	// Return determinant in passed argument
	Matrix< double >& inverse( Matrix< double >&, double& ) const; // pass returned Matrix
	Matrix< double > inverse( double& ) const; // return new Matrix
	double determinant() const; // return determinant only
	// Allows user to force type of inverse procedure
	Matrix< double >& inverse( Matrix< double >&, unsigned ) const; // pass return Matrix
	Matrix< double > inverse( unsigned ) const; // returns new Matrix

	// Singular Matrix error
	class Singular
	{
		public:
			const char * what() const { return "singular Matrix"; }
	};

	// Sets the output header flag
	Matrix< T >& setHeader( const bool );

	// DO NOT USE THESE!
	// They give access to Matrix data, and are inherently DANGEROUS.
	// The only reason I made them is that Microsoft Visual C++ does
	// not support member function templates, and I needed a workaround
	// for passing a function to a Matrix.
	// Also note that end() points 1 location beyond the actual end of
	// the data block.  This is necessary for the pointer operation
	// for ( T* p = M.begin(); p != M.end(); ++p )... but it means that
	// *( M.end() ) will return garbage.
	T* begin() { return data_; }
	T* end() { return data_ + ( nrows_ * ncols_ ); }
	T* begin() const { return data_; }
	T* end() const { return data_ + ( nrows_ * ncols_ ); }

private:
	T* data_; // array which houses matrix
	unsigned nrows_, ncols_; // number of rows, columns

	bool outputHeaderFlag; // flag to indicate if header is output in ostream

	// Utility function which returns number of rows in a file
	unsigned filerows( const string&, const unsigned );
	// Utility function which loads a Matrix from a file
	void fileloader( const string& );

	// Matrix inverse utility functions
	// Gauss Jordan inverse
	Matrix< double >& inverseGaussJordan( Matrix< double >& ) const; // pass returned Matrix
	Matrix< double > inverseGaussJordan() const; // return new Matrix
	// Inverse by LU decomposition and backsubstitution,
	Matrix< double >& inverseLU( Matrix< double >& ) const; // pass returned Matrix
	Matrix< double > inverseLU() const; // return new Matrix
	// Returns LU decomposition
	Matrix< double >& ludcmp( Matrix< double >&, double& ) const; // pass returned Matrix
	Matrix< double > ludcmp( double& ) const; // return new Matrix
	Matrix< double >& ludcmp( Matrix< double >&, vector< int >&, double& ); // pass index
	// LU backsubstitution
	Matrix< double >& lubksb( Matrix< double >&, vector< int >&, vector< double >& );
};

// Constructor which takes no arguments, enables Matrix< double > A;
template< class T >
inline Matrix< T >::Matrix() : data_ ( 0 ), nrows_ ( 0 ),
	ncols_ ( 0 ), outputHeaderFlag ( true ) { }

// Constructor with rows, columns as arguments
template< class T >
inline Matrix< T >::Matrix( const unsigned nrows, const unsigned ncols ) :
	data_  ( new T[ nrows * ncols ] ), nrows_ ( nrows ),
	ncols_ ( ncols ), outputHeaderFlag ( true )
{
	assert ( data_ != 0 ); // bomb out if memory can't be allocated
}

// Constructor which loads a matrix with 3rd argument
template< class T >
Matrix< T >::Matrix( const unsigned nrows, const unsigned ncols, const T value ) :
	data_  ( new T[ nrows * ncols ] ), nrows_ ( nrows ),
	ncols_ ( ncols ), outputHeaderFlag ( true )
{
	assert ( data_ != 0 ); // bomb out if memory can't be allocated

	// NOTE: I had to relax these conditions for stepwise regression of a network without
	//    biases, as the weights of the baseline network (that with no input nodes)
	//    will be a Matrix with 0 columns

	// Throwing an exception inside a ctor is a no-no, hence the assert
	// I will fix this someday
	assert ( nrows != 0 /* && ncols != 0 */ );

	// Throw a BadSize exception if either size is zero
	if ( nrows == 0 /* || ncols == 0 */ )
		throw BadSize();

	// Set a pointer ( pData ) to Matrix's data array
	T* pData = data_;

	// Set all elements of data array to passed value
	for ( unsigned i = 0; i < ( nrows_ * ncols_ ); i++, ++pData )
		*pData = value;
}

// Constructor which loads a matrix with the outer product of 2 vectors,
//    by convention if both are vertical, computes Q Pt, if Q is the first
//    vector, and Pt is the transpose of the second vector P.
template < class T >
Matrix< T >::Matrix( const vector< T >& Q, const vector< T >& Pt ) :
	nrows_ ( Q.size() ), ncols_ ( Pt.size() ), outputHeaderFlag ( true )
{
	// Make sure dimensions are nonzero
	assert ( nrows_ != 0 && ncols_ != 0 );

	// Construct the Matrix data array
	data_ = new T[ nrows_ * ncols_ ];
	assert ( data_ != 0 ); // Bomb out if memory can't be allocated

	// Use previously coded outprod method
	this->outprod( Q, Pt );
}

// Constructor which loads from file, takes filename and number columns
// as arguments
template< class T >
Matrix< T >::Matrix( string& filename, const unsigned ncols ) :
	ncols_ ( ncols ), outputHeaderFlag ( true )
{
	// Find the number of rows in the file
	nrows_ = filerows( util::getGoodFile( filename ), ncols );

	// Now that the number of rows is known, construct the Matrix
	//    for the dataset
	assert ( nrows_ != 0 && ncols != 0 );
	data_ = new T[ nrows_ * ncols_ ];
	assert ( data_ != 0 ); // bomb out if memory can't be allocated

	// Load the file into the Matrix data
	fileloader( filename );
}

// Utility function which returns number of rows in a file,
//    takes filename and number of columns as arguments
template< class T >
unsigned Matrix< T >::filerows( const string& filename, const unsigned ncols )
{
	// Open the file as read-only, and associate it with the
	//    'label' inputFile
	ifstream inputFile( filename.c_str(), ios::in );

	unsigned i = 0, // counts the rows in the file
		j = 0; // column counter
	T tmp; // temporary placeholder

	// First, cycle through file and count number of rows
	while ( inputFile >> tmp ) {
		for ( j = 1; j < ncols; j++ )
			inputFile >> tmp;
		i++;
	}
	inputFile.close(); // close & clear the file
	inputFile.clear();

	return i; // number of rows
}

// Utility function which loads a Matrix from a file, takes filename
//    as argument: nrows_ and ncols_ MUST have already been set
template< class T >
void Matrix< T >::fileloader( const string& filename )
{
	// Open the file as read-only, and associate it with the
	//    'label' inputFile
	ifstream inputFile( filename.c_str(), ios::in );

	// Use overloaded >> operator
	inputFile >> *this;

	inputFile.close(); // close & clear the file
	inputFile.clear();
}

// Method which loads existing Matrix from file, takes filename and
//    number of columns as arguments
template< class T >
Matrix< T >& Matrix< T >::loadfile( string& filename, const unsigned ncols )
{
	// Find the number of rows in the file
	unsigned nrows = filerows( util::getGoodFile( filename ), ncols );

	// If file has different dimensions than Matrix, use resize method
	if ( nrows_ != nrows || ncols_ != ncols )
		this->resize( nrows, ncols );

	// Make sure we now have nonzero dimensions
	assert ( nrows_ != 0 && ncols != 0 );

	// Load the file into the Matrix data
	fileloader( filename );

	return *this; // enables use in Matrix formulae
}

// Constructor which loads from file, takes bool as 1st argument to indicate if
//    a report is to be printed out, and string filename as 2nd argument.
//    Programmed by Gaurav Bansal Feb 2001
template< class T >
Matrix< T >::Matrix( const bool report, string& filename ) : data_ ( 0 ),
	nrows_ ( 0 ), ncols_ ( 0 ), outputHeaderFlag ( true ) // first make empty Matrix
{
	loadfile( report, filename ); // Use loadfile method
}

// Method which loads from file, takes bool as 1st argument to indicate if
//    a report is to be printed out, and string filename as 2nd argument.
//    Programmed by Gaurav Bansal Feb 2001
template< class T >
Matrix< T >& Matrix< T >::loadfile( const bool report, string& filename )
{
	// Open the file as read-only, and associate it with the 'label' inputFile
	ifstream inputFile( ( util::getGoodFile( filename ) ).c_str(), ios::in );

	// Unset skip white space
	inputFile.unsetf( ios::skipws );

	vector< char > line; // temporarily store file's data in line vector, can't use
	                     // a string, as MSVC string is limited to 2,048 characters

	int i = 0; // counter to calculate the number of characters read
	char oneChar; // temporary character holder for reading in file

	// Read the complete file in line vector to be used later
	while( inputFile >> oneChar )
	{
		line.push_back( oneChar ); // append character to end of line
		i++; // increment character counter
	}

	// Make sure file ends in carriage return
	//    This is somewhat inefficient, but when I tried to test for cr, e.g.
	//       if ( *( line.end() ) != '\n' )
	//    it would work in MSVC, but not g++ when re-entering the file
	//    ...I will figure that out someday...
		line.push_back( '\n' ); // add a carriage return
		i++; // and increment character counter

	int num = i - 2; // set num to characters actually read

	i = 0; // reset characters read counter
	unsigned rows = 0; // number of matrix rows
	unsigned cols = 0; // number of columns in the present row
	unsigned max_columns = 0; // number of maximum columns found at any time
	unsigned count = 0; // number of characters belonging to one element of matrix
	char* temp; // temporarily store all characters of one element that will be
	            // converted to float
	unsigned j; // j and k are just index variables
	unsigned k;
	double d; // temporary storage for an element
	bool emptyRow = false; // flag to indicate if row has at least 1 element
	bool notPopulated = false; // flag indicates at least 1 row not fully populated

	// This while loop checks for the number of rows having at least one data element
	//    and finds the maximum number of columns. A character by character evaluation
	//    of the string line needs to be done to skip all the unwanted characters
	//    and white spaces
	while ( i <= num )
	{
		// Set all these variables to zero for each iteration of while loop
		cols = 0;
		emptyRow = false;
		count = 0;

		// This while loop checks all characters in a row delimited either by an enter
		//    or as end of the data
		while ( ( i <= num ) && ( line[ i ] != '\n' ) )
		{
			// Check each character, and if the character is not any kind of delimiter
			//    it goes on and increases the value of count which stores the number
			//    of characters in one element
			if ( isdigit ( line[ i ] ) || line[ i ] == '.' || line[ i ] == 'e' ||
				line[ i ] == 'E' || line[ i ] == '+' || line[ i ] == '-' )
				count++;

			// We come here when we hit a delimiter, and if count is > 0 that means
			// that we have an element. Now we fill all the characters in temp array
			// and convert the array of chars to a float
			else if ( count > 0 )
			{
				temp = new char[ count + 1 ];
				k = 0;
				for ( j = ( unsigned ) i - count; j < ( unsigned ) i; j++ )
					temp[ k++ ] = line[ j ];
				temp[ k ] = '\n';
				d = atof ( temp ); // convert to float
				delete temp; // delete temporary storage
				count = 0; // reset the count for next element
				cols++; // we found a column for the current row so we increment
				        // number of columns

				emptyRow = true; // set flag that present row is not empty
			}

			i++; // increment characters read counter

		} // end of while loop that marks the end of a line in data file

		// Same as else-if loop above and is to check the element if the delimiter
		//    itself is return
		if ( count > 0 )
		{
			temp = new char[ count + 1 ];
			k = 0;
			for ( j = ( unsigned ) i - count; j < ( unsigned ) i; j++ )
				temp[ k++ ] = line[ j ];
			temp[ k ] = '\n';
			d = atof ( temp );
			delete temp;
			count = 0;
			cols++;
			emptyRow = true;
		}

		// Check for a non empty row and increment the number of rows
		if ( line[ i ] == '\n' && emptyRow )
			rows++;

		// Check if the current row is not fully populated and if its not,
		//    set the notPopulated flag
		if ( max_columns !=0 && max_columns != cols && cols!= 0 )
			notPopulated = true;

		// Check for the current column to be the maximum read until now, and
		//    update maximum columns counter
		if ( max_columns < cols )
			max_columns = cols;

		i++; // increment characters read counter

	} // end of big while loop marking the end of file

	inputFile.close( ); // Close & clear the file
	inputFile.clear();

	// Now that the number of rows is known, load the Matrix
	//    for the dataset, first make sure dimensions are nonzero
	assert ( ( rows != 0 ) && ( max_columns != 0 ) );

	// If file has different dimensions than Matrix, use resize method
	if ( nrows_ != rows || ncols_ != max_columns )
		this->resize( rows, max_columns );

	unsigned data_index = 0; // number of elements dealt in Matrix before current row
	unsigned data_counter = 0; // number of elements dealt for the present row

	// Reset character, row counters
	rows = 0;
	i = 0;
	count = 0;
	emptyRow = false; // flag to indicate row has at least 1 element

	// If a report is indicated, begin its header
	if ( report )
	{
		util::screen() << "The loaded matrix has " << nrows_ << " rows and ";
		util::screen() << ncols_ << " columns." << endl;
		if ( notPopulated )
			util::screen() << "The following rows were not fully populated in the file: " << endl;
	}

	// This while loop loads the matrix and fills the undefined elements by 0, it
	//    ends when we reach the end of file
	while ( i <= num )
	{
		data_counter = 0;
		emptyRow = false;

		// Inner while loop goes until the end of a line either delimited by return
		//    or end of file
		while ( ( i <= num )  && ( line[ i ] != '\n' ) )
		{
			// If the present character is non of the delimiter we increases the count
			if ( isdigit ( line[ i ] ) || line[ i ] == '.' || line[ i ] == 'e' ||
				line[ i ] == 'E' || line[ i ] == '+' || line[ i ] == '-' )
				count++;

			// We hit a delimiter and that's not end of file or return. Here we fill a
			//    temporary string with all the characters of the element and convert
			//    it to a float
			else if ( count > 0 )
			{
				temp = new char[ count + 1 ];
				k = 0;
				for ( j = ( unsigned ) i - count; j < ( unsigned ) i; j++ )
					temp[ k++ ] = line[ j ];
				temp[ k ] = '\n';
				d = atof ( temp );
				data_[ data_index + data_counter ] = d;
				data_counter++;
				delete temp;
				count = 0;

				// Since we found an element that means the current row is not empty
				emptyRow = true;
			}

			// Increment the index value that we are using to traverse character
			//    by character in the file
			i++;

		} // end of inner while indicates presence of return.

		// Same function as the else-if loop above except that the delimiter is
		//    return here
		if ( count > 0 ) {
			temp = new char[ count + 1 ];
			k = 0;
			for ( j = ( unsigned ) i - count; j < ( unsigned ) i; j++ )
				temp[ k++ ] = line[ j ];
			temp[ k ] = '\n';
			d = atof ( temp );
			data_[ data_index + data_counter ] = d;
			data_counter++;
			delete temp;
			count = 0;
			emptyRow = true;
		}

		// Check for the non empty rows and increment the number of rows added
		//    to Matrix
		if ( ( line[ i ] == '\n' ) && emptyRow )
			rows++;

		// Selectively prints the number of rows and columns for the rows that
		//    are not fully populated
		if ( report && data_counter != max_columns && data_counter != 0 )
			util::screen() << "Row " << rows << " has " << data_counter << " columns" <<
			endl;

		// Fill the not fully populated rows with zeros
		while ( data_counter != max_columns && emptyRow )
		{
			data_[ data_index + data_counter ] = 0;
			data_counter++;
		}

		i++; // increment the character counter

		// If we find a non empty row we increase the data_index, i.e. the number
		//    of elements dealt till we	started dealing with the next row
		if ( emptyRow )
			data_index += max_columns;

	} //end of while to fill unpopulated elements with zero

	// Report if any row not fully populated row
	if ( report && notPopulated )
		util::screen() << "Unpopulated elements were filled with zero." << endl;

	return *this; // enables use in Matrix formulae
}

// Saves a Matrix to a file, filename (string) is argument
//    returns boolean flag to indicate if file succesfully saved
template< class T >
bool Matrix< T >::savefile( const string& filename )
{
	bool success = false; // flag to indicate file successfully saved

	// Open the output file to save Matrix, overwrite file if it exists
	ofstream savefile( filename.c_str(), ios::out | ios::trunc );

	// Test to insure it was opened
	if ( !savefile.is_open() )
		util::screen() << "Error in opening file to save Matrix!" << endl;
	else
	{
		savefile << *this; // output this Matrix to the file

		// Print message to user notifying successful save to file
		util::screen() << "The Matrix was successfully saved to " << filename;
		util::screen() << "." << endl;

		savefile.close(); // close output file

		success = true; // set flag to indicate file successfully saved
	}

	return success; // return flag to indicate if file successfully saved
}

// Matrix destructor
template< class T >
inline Matrix< T >::~Matrix()
{
	delete[] data_;
}

// Copy constructor
template< class T >
Matrix< T >::Matrix( const Matrix< T >& rhs ) :
	nrows_( rhs.nrows_ ), ncols_( rhs.ncols_ ),
	outputHeaderFlag ( rhs.outputHeaderFlag )
{
	data_ = new T[ nrows_ * ncols_ ]; // construct new Matrix space
	assert ( data_ != 0 ); // bomb out if memory can't be allocated

	// Copy old Matrix into new object using algorithm::copy
	copy( rhs.data_, rhs.data_ + ( nrows_ * ncols_ ), data_ );
}

// Overloaded = operator
template< class T >
Matrix< T >& Matrix< T >::operator=( const Matrix< T >& rhs )
{
	if ( &rhs != this ) // check for self-assignment
	{
		outputHeaderFlag = rhs.outputHeaderFlag; // copy output header flag

		// If file has different dimensions than Matrix, use resize method
		if ( nrows_ != rhs.nrows_ || ncols_ != rhs.ncols_ )
			this->resize( rhs.nrows_, rhs.ncols_ );

		// Copy right Matrix into left object using algorithm::copy
		copy( rhs.data_, rhs.data_ + ( nrows_ * ncols_ ), data_ );
	}

	return *this; // enables A = B = C;
}

// Method to resize Matrix: remember the new Matrix is filled with GARBAGE!
template< class T >
Matrix< T >& Matrix< T >::resize( const unsigned nrows, const unsigned ncols )
{
	delete [] data_; // deallocate current Matrix data
	nrows_ = nrows; // reset dimensions
	ncols_ = ncols;
	data_ = new T[ nrows_ * ncols_ ]; // construct new Matrix space
	assert ( data_ != 0 ); // bomb out if memory can't be allocated

	return *this; // enables use in Matrix formulae
}

// Method to clear Matrix
template< class T >
Matrix< T >& Matrix< T >::clear()
{
	delete [] data_; // deallocate current Matrix data
	nrows_ = 0; // reset dimensions
	ncols_ = 0;
	data_ = 0; // cleared Matrix is empty

	return *this; // enables use in Matrix formulae
}

// Method to fill a Matrix with a value
template< class T >
Matrix< T >& Matrix< T >::fill( const T value )
{
	// Set a pointer ( pData ) to Matrix's data array
	T* pData = data_;

	// Fill the data array with the value
	for ( unsigned i = 0; i < nrows_ * ncols_; i++, ++pData )
		*pData = value;

	return *this; // enables use in Matrix formulae
}

// Accessor with overloaded ()
template< class T >
inline T& Matrix< T >::operator() ( const unsigned row, const unsigned col )
{
	// Throw a BoundsViolation exception if i or j is too big
	if ( row >= nrows_ || col >= ncols_ ) throw BoundsViolation();

	// It's so important that we'll use an assert for now
	assert ( row < nrows_ && col < ncols_ );

	return data_[ row * ncols_ + col ];
}

// Const accessor with overloaded ()
template< class T >
inline const T& Matrix< T >::operator() ( const unsigned row, const unsigned col ) const
{
	// Throw a BoundsViolation exception if i or j is too big
	if ( row >= nrows_ || col >= ncols_ ) throw BoundsViolation();

	// It's so important that we'll use an assert for now
	assert ( row < nrows_ && col < ncols_ );

	return data_[ row * ncols_ + col ];
}

// Accessor for a submatrix, takes passed matrix argument to be
//    populated with result
// Example: A.submatrix( r1, rn, c1, cn, B );
//    where r1 is the first row, rn is the last row of the submatrix,
//    c1 is the first row, cn is the last row of the submatrix, and
//    B is the result.  B must have the proper dimensions!
template< class T >
Matrix< T >& Matrix< T >::submatrix( const unsigned r1, const unsigned rN,
	const unsigned c1, const unsigned cN, Matrix< T >& SubM ) const
{
	// Make sure the receiving Matrix has the proper dimensions
	assert ( r1 <= rN && rN <= nrows_ && c1 <= cN && cN <= ncols_
		&& SubM.rows() == ( rN - r1 + 1 ) && SubM.cols() == ( cN - c1 + 1 ) );

	// Set pointers to the starts of the data arrays
	T *pData = data_ + ( ncols_ * r1 + c1 ), *pSubMData = SubM.data_;

	// Cycle through number of submatrix rows
	for ( unsigned i = r1; i <= rN; i++ )
	{
		// Cycle through number of submatrix columns
		for ( unsigned j = 0; j <= ( cN - c1 ); j++, ++pData, ++pSubMData )
			*pSubMData = *pData;

		// Advance origin Matrix pointer to next row
		pData += ( ncols_ - ( cN - c1 + 1 ) );
	}

	return SubM; // enables use in Matrix formulae
}

// Accessor for a submatrix, returns a new Matrix object
// Example: B = A.submatrix( r1, rn, c1, cn );
//    where r1 is the first row, rn is the last row of the submatrix,
//    c1 is the first row, and cn is the last row of the submatrix.
template< class T >
Matrix< T > Matrix< T >::submatrix( const unsigned r1, const unsigned rN,
	const unsigned c1, const unsigned cN ) const
{
	// Construct receiving Matrix
	Matrix< T > result( ( rN - r1 + 1 ), ( cN - c1 + 1 ) );

	// Use previously coded submatrix method
	this->submatrix( r1, rN, c1, cN, result );

	return result; // enables use in Matrix formulae
}

// Accessor to populate a passed vector from a Matrix row
// Example A.row( n, B );
//    would populate vector B with row n of A
template< class T >
vector< T >& Matrix< T >::row( const unsigned r, vector< T >& v ) const
{
	// Number of elements in v must equal number of columns in A
	assert ( v.size() == ncols_ && r < nrows_ );

	// Set an iterator to the beginning of the row
	T* pData = data_ + ( r * ncols_ );

	// Set an iterator ( pv ) to the output vector
	typename vector< T >::iterator pv;

	// Cycle through the row and copy to vector
	for ( pv = v.begin(); pv != v.end(); ++pv, ++pData )
		*pv = *pData;

	return v; // enables use in vector formulae
}

// Accessor to return a new vector object from a Matrix row
// Example A.row( n );
//    would return a new vector containing row n of A
template< class T >
vector< T > Matrix< T >::row( const unsigned r ) const
{
	// Construct receiving vector
	vector< T > result( ncols_ );

	// Use previously coded row method
	this->row( r, result );

	return result; // enables use in vector formulae
}

// Accessor to populate a passed vector from a Matrix column
// Example A.col( n, B );
//    would populate vector B with column n of A
template< class T >
vector< T >& Matrix< T >::col( const unsigned c, vector< T >& v ) const
{
	// Number of elements in v must equal number of columns in A
	assert ( v.size() == nrows_ && c < ncols_ );

	// Set a pointer to the beginning of the column
	T* pData = data_ + c;

	// Set an iterator ( pv ) to the output vector
	typename vector< T >::iterator pv;

	// Cycle through the col and copy to vector
	for ( pv = v.begin(); pv != v.end(); ++pv, pData += ncols_ )
		*pv = *pData;

	return v; // enables use in vector formulae
}

// Accessor to return a new vector object from a Matrix column
// Example A.col( n );
//    would return a new vector containing column n of A
template< class T >
vector< T > Matrix< T >::col( const unsigned c ) const
{
	// Construct receiving vector
	vector< T > result( nrows_ );

	// Use previously coded col method
	this->col( c, result );

	return result; // enables use in vector formulae
}

// Method which replaces a row in a Matrix, takes row number and vector to be
//    placed in that row as arguments, returns a reference to the Matrix object
// Example A.replacerow( n, v );
//    replaces row n in Matrix A with vector v
template< class T >
Matrix< T >& Matrix< T >::replacerow( const unsigned r, const vector< T >& v )
{
	// Number of elements in v must equal number of columns in Matrix
	assert ( v.size() == ncols_ && r < nrows_ );

	// Set a pointer to the beginning of the row
	T* pData = data_ + ( r * ncols_ );

	// Set an iterator ( pv ) to the vector
	typename vector< T >::const_iterator pv;

	// Cycle through the vector and copy to the row
	for ( pv = v.begin(); pv != v.end(); ++pv, ++pData )
		*pData = *pv;

	return *this; // enables use in Matrix formulae
}

// Method which replaces a column in a Matrix, takes row number and vector to be
//    placed in that column as arguments, returns a reference to the Matrix object
// Example A.replacecol( n, v );
//    replaces column n in Matrix A with vector v
template< class T >
Matrix< T >& Matrix< T >::replacecol( const unsigned c, const vector< T >& v )
{
	// Number of elements in v must equal number of columns in A
	assert ( v.size() == nrows_ && c < ncols_ );

	// Set a pointer to the beginning of the column
	T* pData = data_ + c;

	// Set an iterator ( pv ) to the vector
	typename vector< T >::const_iterator pv;

	// Cycle through the vector and copy to the column
	for ( pv = v.begin(); pv != v.end(); ++pv, pData += ncols_ )
		*pData = *pv;

	return *this; // enables use in Matrix formulae
}

// Method which appends a row to a Matrix
// Example A.addrow( v, B );
//    appends vector as a row to A, populating passed Matrix B with the result
template< class T >
Matrix< T >& Matrix< T >::addrow( const vector< T >& v, Matrix< T >& bigM ) const
{
	// Make sure the Matrices have the proper dimensions
	assert ( v.size() == ncols_ && bigM.ncols_ == ncols_
		&& bigM.nrows_ == ( nrows_ + 1 ) );

	unsigned size = nrows_ * ncols_; // size of the sending Matrix data array

	// Copy the sending Matrix data into the receiving Matrix
	copy( data_, data_ + size, bigM.data_ );

	// Copy the vector into the receiving Matrix
	copy( v.begin(), v.end(), bigM.data_ + size );

	return bigM; // enables use in Matrix formulae
}

// Method which appends a row to a Matrix
// Example B = A.addrow( v );
//    appends vector as a row to A, returns new matrix with the result
template< class T >
Matrix< T > Matrix< T >::addrow( const vector< T >& v ) const
{
	// Construct receiving Matrix
	Matrix< T > result( nrows_ + 1, ncols_ );

	// Use previously coded addrow method
	this->addrow( v, result );

	return result; // enables use in Matrix formulae
}

// Method which appends a column to a Matrix
// Example A.addcol( v, B );
//    appends vector as a column to A, populating passed Matrix B with the result
template< class T >
Matrix< T >& Matrix< T >::addcol( const vector< T >& v, Matrix< T >& bigM ) const
{
	// Make sure the Matrices have the proper dimensions
	assert ( v.size() == nrows_ && bigM.nrows_ == nrows_
		&& bigM.ncols_ == ( ncols_ + 1 ) );

	// Set pointers to the beginning of the Matrices data arrays
	T *pData = data_, *pBigData = bigM.data_;

	// Set an iterator to the beginning of the vector
	typename vector< T >::const_iterator pv = v.begin();

	// Cycle through each row of the sending Matrix
	for ( pv = v.begin(); pv != v.end(); ++pv, pData += ncols_,
		pBigData += ( ncols_ + 1 ) )
	{
		// Copy the sending Matrix row into the receiving Matrix
		copy( pData, pData + ncols_, pBigData );
		// Append the vector element
		*( pBigData + ncols_ ) = *pv;
	}

	return bigM; // enables use in Matrix formulae
}

// Method which appends a column to a Matrix
// Example B = A.addcol( v );
//    appends vector as a column to A, returns new matrix with the result
template< class T >
Matrix< T > Matrix< T >::addcol( const vector< T >& v ) const
{
	// Construct receiving Matrix
	Matrix< T > result( nrows_, ncols_ + 1 );

	// Use previously coded addcol method
	this->addcol( v, result );

	return result; // enables use in Matrix formulae
}

// += operator
template< class T >
Matrix< T >& Matrix< T >::operator+= ( const Matrix< T >& rhs )
{
	// Catch different size matrices
	assert ( nrows_ == rhs.nrows_ && ncols_ == rhs.ncols_ );

	// Uses algorithm::transform and functional::plus
	transform( data_, data_ + ( nrows_ * ncols_ ), rhs.data_, data_, plus< T >() );

	return *this; // enables use in Matrix formulae
}

// -= operator
template< class T >
Matrix< T >& Matrix< T >::operator-= ( const Matrix< T >& rhs )
{
	// Catch different size matrices
	assert ( nrows_ == rhs.nrows_ && ncols_ == rhs.ncols_ );

	// Uses algorithm::transform and functional::minus
	transform( data_, data_ + ( nrows_ * ncols_ ), rhs.data_, data_, minus< T >() );

	return *this; // enables use in Matrix formulae
}

// *= operator
template< class T >
Matrix< T >& Matrix< T >::operator*= ( const Matrix< T >& rhs )
{
	// Catch different size matrices
	assert ( nrows_ == rhs.nrows_ && ncols_ == rhs.ncols_ );

	// Uses algorithm::transform and functional::multiplies
	transform( data_, data_ + ( nrows_ * ncols_ ), rhs.data_, data_, multiplies< T >() );

	return *this; // enables use in Matrix formulae
}

// /= operator
template< class T >
Matrix< T >& Matrix< T >::operator/= ( const Matrix< T >& rhs )
{
	// Catch different size matrices
	assert ( nrows_ == rhs.nrows_ && ncols_ == rhs.ncols_ );

	// Uses algorithm::transform and functional::divides
	transform( data_, data_ + ( nrows_ * ncols_ ), rhs.data_, data_, divides< T >() );

	return *this; // enables use in Matrix formulae
}

// For binary arithmetic operators, note that you don't need to reassert
//    to catch different size matrices, as the unary operators that the
//    binary operators call will do this for you

// + operator
template< class T >
Matrix< T > Matrix< T >::operator+ ( const Matrix< T >& rhs ) const
{
	Matrix< T > result( *this ); // initialize result with *this
	result += rhs; // add second operand
	return result; // return result by copy
}

// - operator
template< class T >
Matrix< T > Matrix< T >::operator- ( const Matrix< T >& rhs ) const
{
	Matrix< T > result( *this ); // initialize result with *this
	result -= rhs; // subtract second operand
	return result; // return result by copy
}

// * operator
template< class T >
Matrix< T > Matrix< T >::operator* ( const Matrix< T >& rhs ) const
{
	Matrix< T > result( *this ); // initialize result with *this
	result *= rhs; // multiply second operand
	return result; // return result by copy
}

// / operator
template< class T >
Matrix< T > Matrix< T >::operator/ ( const Matrix< T >& rhs ) const
{
	Matrix< T > result( *this ); // initialize result with *this
	result /= rhs; // divide second operand
	return result; // return result by copy
}

// += operator for scalar
template< class T >
Matrix< T >& Matrix< T >::operator+= ( const T scalar )
{
	// Set a pointer ( pData ) to Matrix's data array
	T* pData = data_;

	// Iterate through data array, adding scalar
	for ( unsigned i = 0; i < ( nrows_ * ncols_ ); i++, pData++ )
		*pData += scalar;

	return *this; // enables use in Matrix formulae
}

// -= operator for scalar
template< class T >
Matrix< T >& Matrix< T >::operator-= ( const T scalar )
{
	// Set a pointer ( pData ) to Matrix's data array
	T* pData = data_;

	// Iterate through data array, subtrating scalar
	for ( unsigned i = 0; i < ( nrows_ * ncols_ ); i++, pData++ )
		*pData -= scalar;

	return *this; // enables use in Matrix formulae
}

// *= operator for scalar
template< class T >
Matrix< T >& Matrix< T >::operator*= ( const T scalar )
{
	// Set a pointer ( pData ) to Matrix's data array
	T* pData = data_;

	// Iterate through data array, multiplying scalar
	for ( unsigned i = 0; i < ( nrows_ * ncols_ ); i++, pData++ )
		*pData *= scalar;

	return *this; // enables use in Matrix formulae
}

// /= operator for scalar
template< class T >
Matrix< T >& Matrix< T >::operator/= ( const T scalar )
{
	// Set a pointer ( pData ) to Matrix's data array
	T* pData = data_;

	// Iterate through data array, dividing scalar
	for ( unsigned i = 0; i < ( nrows_ * ncols_ ); i++, pData++ )
		*pData /= scalar;

	return *this; // enables use in Matrix formulae
}

// + operator for scalar
template< class T >
Matrix< T > Matrix< T >::operator+ ( const T scalar ) const
{
	Matrix< T > result( *this ); // initialize result with *this
	result += scalar; // add scalar using overloaded +=
	return result; // return result by copy
}

// + operator that allows scalar to be on left hand side
template< class T >
Matrix< T > operator+( const T k, Matrix< T > M )
{
	// Note that Matrix is not a reference, but a local copy
	return M += k; // enables use in Matrix formulae
}

// - operator for scalar
template< class T >
Matrix< T > Matrix< T >::operator- ( const T scalar ) const
{
	Matrix< T > result( *this ); // initialize result with *this
	result -= scalar; // subtract scalar using overloaded +=
	return result; // return result by copy
}

// * operator for scalar
template< class T >
Matrix< T > Matrix< T >::operator* ( const T scalar ) const
{
	Matrix< T > result( *this ); // initialize result with *this
	result *= scalar; // multiply by scalar using overloaded +=
	return result; // return result by copy
}

// * operator that allows scalar to be on left hand side
template< class T >
Matrix< T > operator*( const T k, Matrix< T > M )
{
	// Note that Matrix is not a reference, but a local copy
	return M *= k; // enables use in Matrix formulae
}

// / operator for scalar
template< class T >
Matrix< T > Matrix< T >::operator/ ( const T scalar ) const
{
	Matrix< T > result( *this ); // initialize result with *this
	result /= scalar; // divide by scalar using overloaded +=
	return result; // return result by copy
}

// Sets output header flag for header output in ostream
template< class T >
Matrix< T >& Matrix< T >::setHeader( const bool boolFlag )
{
	outputHeaderFlag = boolFlag;

	return *this; // enables use in Matrix formulae
}

// Tony Makhlouf changes to file 9/11/2005:
//    Removed operator << and operator >> template declarations from here and put
//    them before the class declaration for gcc 4.0.1 compiler compliance

// Overloaded == operator
template< class T >
bool Matrix< T >::operator== ( const Matrix< T >& rhs )
{
	bool result = true;

	if ( &rhs != this ) // check for self-assignment
	{
		// If different dimensions, they're not equal
		if ( nrows_ != rhs.nrows_ || ncols_ != rhs.ncols_ )
			result = false;

		else // use algorithm::equal
			result = equal( rhs.data_, rhs.data_ + ( nrows_ * ncols_ ), data_ );
	}

	return result;
}

// Overloaded != operator
template< class T >
bool Matrix< T >::operator!= ( const Matrix< T >& rhs )
{
	// Use previously coded ==
	return !( *this == rhs );
}

// Returns the transpose of a Matrix in the passed Matrix argument
// Calling sequence: A.t( B );
template< class T >
Matrix< T >& Matrix< T >::t( Matrix< T >& M_in ) const
{
	// Dimensions check
	assert( ( M_in.rows() == ncols_ ) && ( M_in.cols() == nrows_ ) );

	for ( unsigned c = 0; c < ncols_; c++ )
		// Use previously coded replacerow method
		M_in.replacerow( c, this->col( c ) );

	return M_in; // enables use in Matrix formulae
}

// Returns the transpose of a Matrix with a new Matrix
// Calling sequence: B = A.t()
template< class T >
Matrix< T > Matrix< T >::t() const
{
	// Construct receiving Matrix
	Matrix< T > result( ncols_, nrows_ );

	// Use previously coded transpose method
	this->t( result );

	return result; // enables use in Matrix formulae
}

// Dot product
// For vectors o and v and Matrix M, where M . v = o
//
//                         [ v1
// [ M11 M21 M31 M41    .    v2  =  [ o1
//   M12 M22 M32 M42 ]       v3       o2 ]
//                           v4 ]
//
// ( Note that the number of columns in M, 4, is the number of rows in v )
//    by convention, ALL vectors are assumed to be vertical
// Calling sequence: M.dotprod( v, o );
template< class T >
vector< T >& Matrix< T >::dotprod( const vector< T >& iVec, vector< T >& oVec ) const
{
	// Number of Matrix columns must equal number of input vector elements
	// Number of Matrix rows must be >= number of output vector elements
	assert ( iVec.size() == ncols_ && oVec.size() <= nrows_);

	// Set a pointer ( pData ) to Matrix's data array
	T* pData = data_;

	// Set a const iterator ( pi ) to the input vector
	typename vector< T >::const_iterator pi;

	// Set an iterator ( po ) to the output vector
	typename vector< T >::iterator po;

	// Clear output vector by setting all elements to zero
	std::fill( oVec.begin(), oVec.end(), 0 );

	// Calculate the dot product using vector iterators
	for ( po = oVec.begin(); po != oVec.end(); ++po )
		// Calculate an output vector element = Matrix row . input vector
		for ( pi = iVec.begin(); pi != iVec.end(); ++pi )
			*po += ( ( *pData++ ) * ( *pi ) );

	return oVec; // enables use in vector formulae
}

// Dot product which returns a new vector object
// Calling sequence: o = M.dotprod( v );
template< class T >
vector< T > Matrix< T >::dotprod( const vector< T >& iVec ) const
{
	// Construct a new vector with number of elements = number Matrix rows
	vector< T > oVec( nrows_ );

	// Use previously coded dot product
	dotprod( iVec, oVec );

	return oVec; // enables use in vector formulae
}

// Dot product which takes output vector and range passed as argument
//    e.g. M.dotprod( v1, v2, n1, n2 ) will populate v2[ n1 ] through
//    v2[ n2 ] with the result of M . v1
template< class T >
vector< T >& Matrix< T >::dotprod( const vector< T >& iVec, vector< T >& oVec,
	const unsigned iBegin, const unsigned iEnd ) const
{
	// Number of Matrix columns must equal number of input vector elements
	//    number of Matrix rows must equal number of output vector elements,
	//    and range must be in bounds
	assert ( iVec.size() == ncols_ && nrows_ == ( iEnd - iBegin + 1 )
		&& iEnd >= iBegin && iEnd < oVec.size() );

	// Set a pointer ( pData ) to Matrix's data array
	T* pData = data_;

	// Set a const iterator ( pi ) to the input vector
	typename vector< T >::const_iterator pi;

	// Set an iterator ( po ) to the output vector
	typename vector< T >::iterator po;

	// Clear output subvector by setting its elements to zero
	std::fill( oVec.begin() + iBegin, oVec.begin() + iEnd + 1, 0 );

	// Calculate the dot product using vector iterators
	for ( po = oVec.begin() + iBegin; po != oVec.begin() + iEnd + 1; ++po )
		// Calculate an output vector element = Matrix row . input vector
		for ( pi = iVec.begin(); pi != iVec.end(); ++pi )
			*po += ( ( *pData++ ) * ( *pi ) );

	return oVec; // enables use in vector formulae
}

// Dot product which takes transpose of matrix prior to dot product with vector
// For vectors o and v and Matrix M, where Mt . v = o
//
// [ M11 M21        [ v1
//   M12 M22   t  .   v2  =  [ o1
//   M13 M23          v3       o2 ]
//   M14 M24 ]        v4 ]
//
// ( Note that the number of rows in M, 4, is the number of rows in v )
//    by convention, ALL vectors are assumed to be vertical
// Calling sequence: M.dotprodt( v, o );
template< class T >
vector< T >& Matrix< T >::dotprodt( const vector< T >& iVec, vector< T >& oVec ) const
{
	// Number of Matrix columns must equal number of input vector elements
	// Number of Matrix rows must be >= of output vector elements
	assert ( iVec.size() == nrows_ && oVec.size() <= ncols_);

	// Set a pointer ( pData ) to Matrix's data array
	T* pData = data_;

	// Set a const iterator ( pi ) to the input vector
	typename vector< T >::const_iterator pi;

	// Set an iterator ( po ) to the output vector
	typename vector< T >::iterator po;

	// Clear output vector by setting all elements to zero
	std::fill( oVec.begin(), oVec.end(), 0 );

	// The amount to reset back to the first row of the data Matrix
	unsigned decrement = ncols_ * nrows_ - 1;

	// Calculate the dot product using vector iterators
	for ( po = oVec.begin(); po != oVec.end(); ++po )
	{
		// Calculate an output vector element = Matrix row . input vector
		for ( pi = iVec.begin(); pi != iVec.end(); ++pi )
		{
			*po += *pData * *pi;
			pData += ncols_; // move "down" the Matrix
		}

		pData -= decrement; // reset back to the first Matrix row
	}

	return oVec; // enables use in vector formulae
}

// Transpose dot product which returns a new vector object
// Calling sequence: o = M.dotprodt( v );
template< class T >
vector< T > Matrix< T >::dotprodt( const vector< T >& iVec ) const
{
	// Construct a new vector with number of elements = number Matrix columns
	vector< T > oVec( ncols_ );

	// Use previously coded transpose dot product
	dotprodt( iVec, oVec );

	return oVec; // enables use in vector formulae
}

// Transpose dot product which takes output vector and range passed as argument
//    e.g. M.dotprodt( v1, v2, n1, n2 ) will populate v2[ n1 ] through
//    v2[ n2 ] with the result of Mt . v1
template< class T >
vector< T >& Matrix< T >::dotprodt( const vector< T >& iVec, vector< T >& oVec,
	const unsigned iBegin, const unsigned iEnd ) const
{
	// Number of Matrix columns must equal number of input vector elements
	//    number of Matrix rows must equal number of output vector elements,
	//    and range must be in bounds
	assert ( iVec.size() == nrows_ && ncols_ == ( iEnd - iBegin + 1 )
		&& iEnd >= iBegin && iEnd < oVec.size() );

	// Set a pointer ( pData ) to Matrix's data array
	T* pData = data_;

	// Set a const iterator ( pi ) to the input vector
	typename vector< T >::const_iterator pi;

	// Set an iterator ( po ) to the output vector
	typename vector< T >::iterator po;

	// Clear output subvector by setting its elements to zero
	std::fill( oVec.begin() + iBegin, oVec.begin() + iEnd + 1, 0 );

	// The amount to reset back to the first row of the data Matrix
	unsigned decrement = ncols_ * nrows_ - 1;

	// Calculate the dot product using vector iterators
	for ( po = oVec.begin() + iBegin; po != oVec.begin() + iEnd + 1; ++po )
	{
		// Calculate an output vector element = Matrix row . input vector
		for ( pi = iVec.begin(); pi != iVec.end(); ++pi )
		{
			*po += *pData * *pi;
			pData += ncols_; // move "down" the Matrix
		}

		pData -= decrement; // reset back to the first Matrix row
	}

	return oVec; // enables use in vector formulae
}

// This dot product takes a single row from a Matrix, and transposes it to be used
//    as the right hand side argument for a dot product
//
// It is useful for extracting a single row from a dataset and passing it to the
//    first layer of a neural network
//
// Calling sequence: M.dotprod_row( D, row, o );
//    where D is the Matrix containing the dataset, row is the row number,
//    and o is the output vector
//
// If the row in D is denoted by the vector d, and its transpose d', then the
//    returned vector o will contain the resulting dot product: M . d' = o
template< class T >
vector< T >& Matrix< T >::dotprod_row( const Matrix< T >& Dataset, const unsigned row,
	vector< T >& oVec ) const
{
	// Number of columns in Matrix must equal number of columns in dataset Matrix
	//    number of Matrix rows must equal number of output vector elements,
	//    and row must be in bounds
	assert ( ncols_ == Dataset.ncols_ && oVec.size() == nrows_
		&& row < Dataset.nrows_ );

	// Set a pointer ( pData ) to Matrix's data array
	T* pData = data_;

	// Set a pointer ( pData ) to 1st element of dataset Matrix's selected row
	unsigned DatasetBegin = row * ncols_;
	T* pDataset = Dataset.data_ + DatasetBegin;

	// Set an iterator ( po ) to the output vector
	typename vector< T >::iterator po;

	// Clear output vector by setting all elements to zero
	std::fill( oVec.begin(), oVec.end(), 0 );

	// Calculate the dot product using array pointers and vector iterator
	for ( po = oVec.begin(); po != oVec.end(); ++po )
	{
		// Calculate an output vector element = Matrix row . dataset row'
		for ( unsigned i = 0; i < ncols_; i++ )
			*po += ( ( *pData++ ) * ( *pDataset++ ) );

		pDataset = Dataset.data_ + DatasetBegin; // reset dataset pointer
	}

	return oVec;  // enables use in vector formulae
}

// dotprod_row (see above) which returns a new vector object
//
// Calling sequence: M.dotprod_row( D, row );
//    where D is the Matrix containing the dataset and row is the row number,
template< class T >
vector< T > Matrix< T >::dotprod_row( const Matrix< T >& Dataset, const unsigned row ) const
{
	// Construct a new vector with number of elements = number Matrix rows
	vector< T > oVec( nrows_ );

	// Use previously coded dotprod_row
	dotprod_row( Dataset, row, oVec );

	return oVec; // enables use in vector formulae
}

// dotprod_row which takes output vector and range passed as argument
//    e.g. M.dotprod_row( D, row, v, n1, n2 ) will populate v[ n1 ] through
//    v[ n2 ] with the result of M . d', if the row in D is denoted by the
//    vector d, and its transpose d'.
template< class T >
vector< T >& Matrix< T >::dotprod_row( const Matrix< T >& Dataset, const unsigned row,
	vector< T >& oVec, const unsigned iBegin, const unsigned iEnd ) const
{
	// Number of columns in Matrix must equal number of columns in dataset Matrix
	//    number of Matrix rows must equal number of output vector elements, and
	//    row and range must be in bounds
	assert ( ncols_ == Dataset.ncols_ && nrows_ == ( iEnd - iBegin + 1 )
		&& iEnd >= iBegin && iEnd < oVec.size() && row < Dataset.nrows_ );

	// Set a pointer ( pData ) to Matrix's data array
	T* pData = data_;

	// Set a pointer ( pDataset ) to 1st element of dataset Matrix's selected row
	unsigned DatasetBegin = row * ncols_;
	T* pDataset = Dataset.data_ + DatasetBegin;

	// Set an iterator ( po ) to the output vector
	typename vector< T >::iterator po;

	// Clear output subvector by setting its elements to zero
	std::fill( oVec.begin() + iBegin, oVec.begin() + iEnd + 1, 0 );


	// Calculate the dot product using array pointers and vector iterator
	for ( po = oVec.begin() + iBegin; po != oVec.begin() + iEnd + 1; ++po )
	{
		// Calculate an output vector element = Matrix row . dataset row'
		for ( unsigned i = 0; i < ncols_; i++ )
			*po += ( ( *pData++ ) * ( *pDataset++ ) );

		pDataset = Dataset.data_ + DatasetBegin; // reset dataset pointer
	}

	return oVec; // enables use in vector formulae
}

// Method which perform dot product on 2 Matrices, 1st argument is the
//    right hand side Matrix, 2nd argument is the passed Matrix to be
//    populated with the result C = A . B
// Calling sequence: A.dotprod( B, C )
template< class T >
Matrix< T >& Matrix< T >::dotprod( const Matrix< T >& B, Matrix< T >& C ) const
{
	// Dimensions check
	assert ( ( C.rows() == nrows_ ) && ( C.cols() == B.cols() ) && ( ncols_ == B.rows() ) );

	for ( unsigned col = 0; col < B.cols(); col++ )
		// Use previously coded vector dotprod and replacecol methods
		C.replacecol( col, this->dotprod( B.col( col ) ) );

	return C; // enables use in Matrix formulae
}

// Method which perform dot product on 2 Matrices, returns new Matrix
// Calling sequence: C = A.dotprod( B )
template< class T >
Matrix< T > Matrix< T >::dotprod( const Matrix< T >& B ) const
{
	// Construct receiving Matrix
	Matrix< T > result( nrows_, B.cols() );

	// Use previously coded dotprod method
	this->dotprod( B, result );

	return result; // enables use in Matrix formulae
}

// Fills a matrix with the outer product of 2 vectors, by convention if
//    both are vertical, computes Q Pt, if Q is the first vector, and Pt is the
//    transpose of the second vector P.
template < class T >
Matrix< T >& Matrix< T >::outprod( const vector< T >& Q, const vector< T >& Pt )
{
	// Make sure Matrix is the right size
	assert ( nrows_ == Q.size() && ncols_ == Pt.size() );

	// Set a pointer ( pData ) to Matrix's data array
	T* pData = data_;

	// Set iterators to the vector's 1st elements
	typename vector< T >::const_iterator pQ, pPt;

	// Using the iterators, calculate the individual vector element products
	//    to construct the outer product
	for ( pQ = Q.begin(); pQ < Q.end(); ++pQ )
		for ( pPt = Pt.begin(); pPt < Pt.end(); ++pPt )
			*pData++ = *pQ * *pPt;

	return *this; // enables use in Matrix formulae
}

// Compute sum of squares of Matrix elements
// Example: s = M.squared();
template < class T >
T Matrix< T >::squared() const
{
	// Set a pointer ( pData ) to Matrix's data array
	T* pData = data_;

	// Initialize the sum of squares
	T sum = 0;

	// Calculate the sum of squares
	for ( unsigned i = 0; i < ( nrows_ * ncols_ ); i++, pData++ )
		sum += ( *pData * *pData );

	return sum; // and return the sum of squares
}

// Compute the maximum absolute value of a Matrix
// Example: a = M.maxabs();
template < class T >
T Matrix< T >::maxabs() const
{
	// Set a pointer ( pData ) to Matrix's data array
	T* pData = data_;

	T result = 0, // initialize the result
		absval; // the absolute value

	// Calculate the max abs value
	for ( unsigned i = 0; i < ( nrows_ * ncols_ ); i++, pData++ )
	{
		absval = ( *pData < 0 ? -*pData : *pData ); // the abs value
		if ( absval > result ) // the maximum
			result = absval;
	}

	return result; // and return the maximum absolute value;
}

// Method for passing a function to a Matrix, returning another passed Matrix
// Example: func( M_in, sigmoidal(), M_out );
// The implementation here which uses pointers to Matrix data SHOULD NOT be used
//    for other methods.  It was necessary because Microsoft Visual C++ does not
//    support member function templates.
template < class T, class F >
Matrix< T >& func( const Matrix< T >& Mi, F fx, Matrix< T >& Mo )
{
	// Catch different size matrices
	assert ( Mo.rows() == Mi.rows() && Mo.cols() == Mi.cols() );

	// Run through data array, and set output Matrix data array to passed function
	for ( T *pMi = Mi.begin(), *pMo = Mo.begin(); pMi != Mi.end(); ++pMi, ++pMo )
		*pMo = fx( *pMi );

	return Mo; // enables use in Matrix formulae
}

// Method for passing a function to a Matrix, returns a new Matrix
// Example: M2 = func( M1, sigmoidal() );
template < class T, class F >
Matrix< T > func( const Matrix< T >& Mi, F fx )
{
	// Construct a new matrix of the same size
	Matrix< T > Mo( Mi.rows(), Mi.cols() );

	// Use previously coded func
	func( Mi, fx, Mo );

	return Mo; // enables use in Matrix formulae
}

// Method which returns a vector from a Matrix column sums, populates passed vector
//    v with result
// Example M.colsums( v )
template < class T >
vector< T >& Matrix< T >::colsums( vector< T >& sums ) const
{
	assert ( ncols_ == sums.size() ); // check vector & Matrix dimensions

	// Clear sums vector by setting all elements to zero
	std::fill( sums.begin(), sums.end(), 0 );

	// Set a pointer ( pData ) to Matrix's data array
	T* pData = data_;

	// Set vector iterator
	typename vector< T >::iterator pv;

	// Using iterators, sum the columns
	for ( unsigned r = 0; r < nrows_; r++ )
		for ( pv = sums.begin(); pv != sums.end(); ++pv, ++pData )
			*pv += *pData;

	return sums; // for use in vector formulae
}

// Method which returns a vector from a Matrix column sums, returns a new vector
// Example v = M.colsums()
template < class T >
vector< T > Matrix< T >::colsums() const
{
	// Construct a new vector for the column sums
	vector< T > sums( ncols_ );

	// Use previously coded colsums( vector< T >& )
	sums = this->colsums( sums );

	return sums; // return the vector of column sums
}

// Method which, given a Matrix which consists of rows that have a single
//    row element with a value of 1, and the remainder with values of 0,
//    returns a vector representing in which column each of the rows was 1,
//    takes passed vector as argument, returns true if successful
template < class T >
bool Matrix< T >::rowindex( vector< unsigned >& v ) const
{
	assert ( v.size() == nrows_ ); // vector size must equal number of rows

	bool success = true; // flag to indicate if successful

	// Set a pointer ( pData ) to Matrix's data array
	T* pData = data_;

	// Set vector iterator
	vector< unsigned >::iterator pv;

	// Using vector iterator, cycle through the rows
	for ( pv = v.begin(); pv != v.end(); ++pv )
	{
		unsigned accumulator = 0;

		for ( unsigned c = 0; c < ncols_; c++, ++pData )
		{
			if ( *pData == 1 )
			{
				*pv = c;
				accumulator += 1;
			}
			else if ( *pData != 0 )
				success = false; // fails if any element not 1 or 0
		}

		if ( accumulator != 1 )
			success = false; // fails if rows contain < or > 1 element of value 1
	}

	return success;
}

// Method which includes only those columns indicated by the passed vector
//    takes Matrix to be populated as first argument, vector< unsigned > as
//    2nd argument, returns reference to returned Matrix
template < class T >
Matrix< T >& Matrix< T >::includecols( Matrix< T >& M, const vector< unsigned >& pos ) const
{
	vector< unsigned > posC( pos ); // copy so positions vector remains untouched

	// Sort incoming positions vector, necessary for coming application of unique
	sort( posC.begin(), posC.end() );

	// The output has ONE column per included position. (This assert read
	//    ncols_ - pos.size() -- excludecols' dimension -- since 2.x; the
	//    method had no callers, so the wrong contract never fired. Fixed
	//    2026-07-16 when includerows was added beside it.)
	assert ( *( max_element( pos.begin(), pos.end() ) ) < ncols_ // bounds check
		&& posC.end() == unique( posC.begin(), posC.end() ) // unique check
		&& M.nrows_ == nrows_ // dimension checks
		&& M.ncols_ == pos.size() );

	vector< bool > vbool( ncols_ ); // indicates if column should be included
	std::fill( vbool.begin(), vbool.end(), false ); // initialize to all false

	for ( unsigned i = 0; i < pos.size(); i++ ) // cycle through passed vector
		vbool[ pos[ i ] ] = true; // turn specified column to true

	// Set pointers to this & incoming Matrices data arrays
	T *pData = data_, *pMData = M.data_;

	// Set vector iterator
	vector< bool >::iterator pv;

	// Using iterators, include the columns as specified
	for ( unsigned r = 0; r < nrows_; r++ )
		for ( pv = vbool.begin(); pv != vbool.end(); ++pv, ++pData )
			if ( *pv ) // if column should be included
			{
				*pMData = *pData; // include it in the incoming Matrix
				pMData++; // and advance the incoming Matrix array pointer
			}

	return M; // for use in Matrix math
}

// Method which includes only those columns indicated by the passed vector
//    takes vector< unsigned > as argument, returns new Matrix
template < class T >
Matrix< T > Matrix< T >::includecols( const vector< unsigned >& pos ) const
{
	// Construct receiving Matrix: one column per included position (sized
	//    ncols_ - pos.size() until 2026-07-16 -- see the note in the
	//    two-argument form)
	Matrix< T > result( nrows_, ( unsigned ) pos.size() );

	// Use previously coded includecols method
	this->includecols( result, pos );

	return result; // for use in Matrix math
}

// Method which gathers the rows indicated by the passed vector, in that
//    order, into the passed Matrix. Positions may repeat and arrive in any
//    order (unlike includecols): a bootstrap resample drawn with replacement
//    is exactly this gather. Row indices are checked unconditionally (a
//    BoundsViolation throw, like operator() -- not an assert), because a bad
//    gather index in a release build would otherwise read out of bounds.
template < class T >
Matrix< T >& Matrix< T >::includerows( Matrix< T >& M, const vector< unsigned >& pos ) const
{
	assert ( M.nrows_ == pos.size() // dimension checks
		&& M.ncols_ == ncols_ );

	T *pMData = M.data_; // pointer to the incoming Matrix data array

	for ( unsigned i = 0; i < pos.size(); i++ ) // cycle through positions
	{
		if ( pos[ i ] >= nrows_ ) // bounds check the gathered row
			throw BoundsViolation();

		// Copy the indicated row into the incoming Matrix
		const T *pData = data_ + pos[ i ] * ncols_;
		for ( unsigned c = 0; c < ncols_; c++, ++pData, ++pMData )
			*pMData = *pData;
	}

	return M; // for use in Matrix math
}

// Method which gathers the rows indicated by the passed vector, in that
//    order, returning a new Matrix
template < class T >
Matrix< T > Matrix< T >::includerows( const vector< unsigned >& pos ) const
{
	// Construct receiving Matrix: one row per gathered position
	Matrix< T > result( ( unsigned ) pos.size(), ncols_ );

	// Use previously coded includerows method
	this->includerows( result, pos );

	return result; // for use in Matrix math
}

// Method which excludes those columns indicated by the passed vector
//    takes Matrix to be populated as first argument, vector< unsigned > as
//    2nd argument, returns reference to returned Matrix
template < class T >
Matrix< T >& Matrix< T >::excludecols( Matrix< T >& M, const vector< unsigned >& pos ) const
{
	vector< unsigned > posC( pos ); // copy so positions vector remains untouched

	// Sort incoming positions vector, necessary for coming application of unique
	sort( posC.begin(), posC.end() );

	assert ( *( max_element( pos.begin(), pos.end() ) ) < ncols_ // bounds check
		&& posC.end() == unique( posC.begin(), posC.end() ) // unique check
		&& M.nrows_ == nrows_ // dimension checks
		&& M.ncols_ == ( ncols_ - pos.size() ) );

	vector< bool > vbool( ncols_ ); // indicates if column should be excluded
	std::fill( vbool.begin(), vbool.end(), true ); // initialize to all true

	for ( unsigned i = 0; i < pos.size(); i++ ) // cycle through passed vector
		vbool[ pos[ i ] ] = false; // turn specified column to false

	// Set pointers to this & incoming Matrices data arrays
	T *pData = data_, *pMData = M.data_;

	// Set vector iterator
	vector< bool >::iterator pv;

	// Using iterators, exclude the columns as specified
	for ( unsigned r = 0; r < nrows_; r++ )
		for ( pv = vbool.begin(); pv != vbool.end(); ++pv, ++pData )
			if ( *pv ) // if column should be excluded
			{
				*pMData = *pData; // exclude it in the incoming Matrix
				pMData++; // and advance the incoming Matrix array pointer
			}

	return M; // for use in Matrix math
}

// Method which excludes those columns indicated by the passed vector
//    takes vector< unsigned > as argument, returns new Matrix
template < class T >
Matrix< T > Matrix< T >::excludecols( const vector< unsigned >& pos ) const
{
	// Construct receiving Matrix
	Matrix< T > result( nrows_, ( ncols_ - pos.size() ) );

	// Use previously coded excludecols method
	this->excludecols( result, pos );

	return result; // for use in Matrix math
}

// Method which converts a Matrix into a vector, populates passed vector
//    with result
// Example M.toVector( v )
template < class T >
vector< T >& Matrix< T >::toVector( vector< T >& v_in ) const
{
	// Passed vector must be same size as Matrix data structure
	assert( v_in.size() == ( nrows_ * ncols_ ) );

	// We'll do this through iterators
	typename vector< T >::iterator pv; // vector iterator
	T *pData; // Matrix data iterator

	// Do the conversion of Matrix data to the incoming vector
	for ( pv = v_in.begin(), pData = data_; pv != v_in.end(); ++pv, ++pData )
		*pv = *pData;

	return v_in; // for use in vector formulae
}

// Method which converts a Matrix into a vector, returns new vector
// Example v = M.toVector()
template < class T >
vector< T > Matrix< T >::toVector() const
{
	// Create new vector with Matrix data (one of those rare cases where the
	//    member function which populates a passed vector is not needed)
	vector< T > v( data_, data_ + ( nrows_ * ncols_ ) );

	return v; // for use in vector formulae
}

// Method which converts a vector into a Matrix, populates passed Matrix
//    with result; incoming Matrix is 1st argument, vector is 2nd argument
// Example toMatrix( M, v )
// The implementation here which uses pointers to Matrix data SHOULD NOT be used
//    for other methods.  It was necessary because Microsoft Visual C++ does not
//    support member function templates.
template < class T >
Matrix< T >& toMatrix( Matrix< T >& M_in, vector< T >& v_in )
{
	// Passed Matrix must be same size as vector
	assert( v_in.size() == ( M_in.rows() * M_in.cols() ) );

	// We'll do this through iterators
	typename vector< T >::iterator pv; // vector iterator
	T *pData; // Matrix data iterator

	// Do the conversion of Matrix data to the incoming vector
	for ( pData = M_in.begin(), pv = v_in.begin(); pData != M_in.end(); ++pv, ++pData )
		*pData = *pv;

	return M_in; // for use in Matrix formulae
}

// Method which converts a vector into a Matrix, returns new Matrix
//    with result; vector to be converted is 1st argument, number rows for
//    new Matrix is 2nd argument, number columns is 3rd argument
// Example M = toMatrix( v, 4, 3 )
template < class T >
Matrix< T > toMatrix( vector< T >& v_in, unsigned nrows, unsigned ncols )
{
	// new Matrix must be same size as vector
	assert( v_in.size() == ( nrows * ncols ) );

	// Construct a new matrix of the same size
	Matrix< T > Mo( nrows, ncols );

	// Use previously coded toMatrix
	toMatrix( Mo, v_in );

	return Mo; // enables use in Matrix formulae
}

#endif
