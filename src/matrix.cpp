// matrix.cpp

// Non template methods go here

#include "stdafx.h" // For MSVC, must be first!

#include <math.h>
#include <algorithm>

#include "matrix.h"
#include "vector_ops.h"

// Tony Makhlouf changes to file 9/11/2005:
//    added the words template <> in front of all template function definitions
//    for gcc 4.0.1 compiler compliance

// Fills Matrix with (double) random numbers between -n and +n
// Example: M.random( 0.5 );

template <> Matrix< double >& Matrix< double >::random( const double n )
{
	// Seed the random number generator
	util::d_random();

	// Set a pointer ( pData ) to Matrix's data array
	double* pData = data_;

	for ( unsigned i = 0; i < nrows_ * ncols_; i++, ++pData )
		*pData = util::d_random( n );

	return *this; // enables use in Matrix formulae
}

// Method which returns first-order covariance Matrix, populates passed Matrix V
//    with result
// Example: M.covariance( V );

template <> Matrix< double >& Matrix< double >::covariance( Matrix< double >& V ) const
{
	// Check dimensions of current and new Matrices
	assert ( V.rows() == ncols_ && V.cols() == ncols_ );

	// Make sure there's at least one row in the data Matrix, and that the
	//    covariance matrix will be at least 2 x 2
	assert ( nrows_ > 0 && ncols_ > 1 );

	V.fill( 0 ); // fill incoming covariance Matrix with zeros

	// A Matrix to hold differences between variates and their means
	Matrix< double > diff( nrows_, 0 );

	vector< double > u( ncols_ ), // the vector of the column means
		v_col; // a vector to hold each column, one at a time

	unsigned r, c, c_i, c_j; // row and column counters

	u = this->colsums(); // calculate column sums
	u /= static_cast< double >( nrows_ ); // calculate column means

	// Iterate through each column, and calculate the differences between
	//    variates and their means
	for ( c = 0; c < ncols_; c++ )
	{
		v_col = this->col( c ); // obtain a column from the data Matrix
		v_col -= u[ c ]; // subtract the mean
		diff = diff.addcol( v_col ); // append the column to the difference Matrix
	}

	// Calculate the covariates "above" the diagonal
	for ( c_i = 0; c_i < ncols_; c_i++ )
		for ( c_j = c_i; c_j < ncols_; c_j++ )
			for ( r = 0; r < nrows_; r++ )
				V( c_i, c_j ) += ( diff( r, c_i ) * diff( r, c_j ) );

	// Copy the covariates "above" the diagonal to "below" the diagonal
	for ( r = 1; r < ncols_; r++ )
		for ( c = 0; c < r; c++ )
			V( r, c ) = V( c, r );

	V /= static_cast< double >( nrows_ - 1 ); // divide the covariance Matrix by n - 1

	return V; // return the new Matrix object
}


// Returns first-order covariance Matrix as a new Matrix object
// Example: A = B.covariance();
template <> Matrix< double > Matrix< double >::covariance() const
{
	Matrix< double > M( ncols_, ncols_ ); // make a new Matrix for result

	return this->covariance( M ); // use previously coded covariance method
}

// Method which returns the inverse of Matrix calculated by Gauss-Jordan
//    elimination, populates passed Matrix I with result
// Example: M.inverse( I );
// Adapted from Numerical Recipes in C, 2nd edition, Cambridge University Press,
//    Cambridge, 1992, pages 36 - 41
// Note that Gauss-Jordan elimination may not be the best way to invert
//    your Matrix!

template <> Matrix< double >& Matrix< double >::inverseGaussJordan( Matrix< double >& I ) const
{
	// Check sizes of Matrices are equal and square
	assert ( I.nrows_ == nrows_ && I.ncols_ == ncols_ && nrows_ == ncols_ );

	I = *this; // copy current Matrix into incoming Matrix

	vector< unsigned > indxc( ncols_ ), indxr( ncols_ ), ipiv( ncols_, 0 );

	unsigned i, j, k, ll, icol = 0, irow = 0;
	int l; // because it counts backwards

	double big, pivinv, dum;

	for ( i = 0; i < ncols_; i++ )
	{
		big = 0;

		for ( j = 0; j < ncols_; j++ )
			if ( ipiv[ j ] != 1 )
				for ( k = 0; k < ncols_; k++ )
				{
					if ( ipiv[ k ] == 0 )
					{
						if ( fabs( I( j, k ) ) >= big )
						{
							big = fabs( I( j, k ) );
							irow = j;
							icol = k;
						}
					}
					else if ( ipiv[ k ] > 1 )
						throw Singular(); // singular Matrix
				}

		++( ipiv[ icol ] );

		if ( irow != icol )
			for ( l = 0; ( unsigned ) l < ncols_; l++ )
				swap( I( irow, l ), I( icol, l ) );

		indxr[ i ] = irow;
		indxc[ i ] = icol;

		if ( I( icol, icol ) == 0 )
			throw Singular(); // singular Matrix

		pivinv = 1 / I( icol, icol );
		I( icol, icol ) = 1;
		for ( l = 0; ( unsigned ) l < ncols_; l++ )
			I( icol, l ) *= pivinv;

		for ( ll = 0; ll < ncols_; ll++ )
			if ( ll != icol )
			{
				dum = I( ll, icol );
				I( ll, icol ) = 0;
				for ( l = 0; ( unsigned ) l < ncols_; l++ )
					I( ll, l ) -= I( icol, l ) * dum;
			}
	}

	for ( l = ( ncols_ - 1 ); l >= 0; l-- )
	{
		if ( indxr[ l ] != indxc[ l ] )
			for ( k = 0; k < ncols_; k++ )
				swap( I( k, indxr[ l ] ), I( k, indxc[ l ] ) );
	}

	return I;
}

// Method which returns the inverse of Matrix calculated by Gauss-Jordan
//    elimination, returns new Matrix with result
// Example: I = M.inverse();

template <> Matrix< double > Matrix< double >::inverseGaussJordan() const
{
	Matrix< double > I = *this; // make a copy of the current Matrix

	return this->inverseGaussJordan( I ); // use previously coded inverse method
}

// Method which returns the LU decomposition of Matrix, populates passed
//    Matrix L with result, returns d with sign
// Example: M.ludcmp( L, d );
// Adapted from Numerical Recipes in C, 2nd edition, Cambridge University Press,
//    Cambridge, 1992, pages 44 - 47

template <> Matrix< double >& Matrix< double >::ludcmp( Matrix< double >& L, double& d ) const
{
	// Check sizes of Matrices are equal and square
	assert ( L.nrows_ == nrows_ && L.ncols_ == ncols_ && nrows_ == ncols_ );

	L = *this; // copy current Matrix into incoming Matrix

	unsigned i, imax = 0, j;
	int k;

	double big, dum, sum, temp;

	vector< double > vv( ncols_ );

	d = 1;

	for ( i = 0; i < ncols_; i++ )
	{
		big = 0;
		for ( j = 0; j < ncols_; j++ )
			if ( ( temp = fabs( L( i, j ) ) ) > big )
				big = temp;

		if ( big == 0 ) throw Singular(); // Singular Matrix

		vv[ i ] = 1 / big;
	}

	for ( j = 0; j < ncols_; j++ )
	{
		for ( i = 0; i < j; i++ )
		{
			sum = L( i, j );
			for ( k = 0; k < ( int ) i ; k++ )
				sum -= L( i, k ) * L( k, j );
			L( i, j ) = sum;
		}
		big = 0;

		for ( i = j; i < ncols_; i++ )
		{
			sum = L( i, j );
			for ( k = 0; k < ( int ) j ; k++ )
				sum -= L( i, k ) * L( k, j );
			L( i, j ) = sum;

			if ( ( dum = vv[ i ] * fabs( sum ) ) >= big )
			{
				big = dum;
				imax = i;
			}
		}

		if ( j != imax )
		{
			for ( k = 0; k < ( int ) ncols_; k++ )
			{
				dum = L( imax, k );
				L( imax, k ) = L( j, k );
				L( j, k ) = dum;
			}
			d = -d;
			vv[ imax ] = vv[ j ];
		}

		if ( j != ncols_ )
		{
			if ( L( j, j ) == 0 ) throw Singular(); // Singular Matrix

			dum = 1 / L( j, j );

			for ( i = ( j + 1 ); i < ncols_; i++ )
				L( i, j ) *= dum;
		}
	}

	return L;
}

// Method which returns the LU decomposition of Matrix, returns new Matrix
//    with result, returns d with sign
// Example: M.ludcmp( d );
// Adapted from Numerical Recipes in C, 2nd edition, Cambridge University Press,
//    Cambridge, 1992, pages 44 - 47

template <> Matrix< double > Matrix< double >::ludcmp( double& d ) const
{
	Matrix< double > L = *this; // make a copy of the current Matrix

	return this->ludcmp( L, d ); // use previously coded ludcmp method
}

// Method which returns the LU decomposition of Matrix, populates passed Matrix
//    with result, returns row permutation index vector indx, and d with sign
// Example: M.ludcmp( L, indx, d );
// Adapted from Numerical Recipes in C, 2nd edition, Cambridge University Press,
//    Cambridge, 1992, pages 44 - 47

template <> Matrix< double >& Matrix< double >::ludcmp( Matrix< double >& L,
	vector< int >& indx, double& d )
{
	// Check sizes of Matrices are equal and square
	assert ( L.nrows_ == nrows_ && L.ncols_ == ncols_ && nrows_ == ncols_ );

	L = *this; // copy current Matrix into incoming Matrix

	unsigned i, imax = 0, j;
	int k;

	double big, dum, sum, temp;

	vector< double > vv( ncols_ );

	d = 1;

	for ( i = 0; i < ncols_; i++ )
	{
		big = 0;
		for ( j = 0; j < ncols_; j++ )
			if ( ( temp = fabs( L( i, j ) ) ) > big )
				big = temp;

		if ( big == 0 ) throw Singular(); // Singular Matrix

		vv[ i ] = 1 / big;
	}

	for ( j = 0; j < ncols_; j++ )
	{
		for ( i = 0; i < j; i++ )
		{
			sum = L( i, j );
			for ( k = 0; k < ( int ) i ; k++ )
				sum -= L( i, k ) * L( k, j );
			L( i, j ) = sum;
		}
		big = 0;

		for ( i = j; i < ncols_; i++ )
		{
			sum = L( i, j );
			for ( k = 0; k < ( int ) j ; k++ )
				sum -= L( i, k ) * L( k, j );
			L( i, j ) = sum;

			if ( ( dum = vv[ i ] * fabs( sum ) ) >= big )
			{
				big = dum;
				imax = i;
			}
		}

		if ( j != imax )
		{
			for ( k = 0; k < ( int ) ncols_; k++ )
			{
				dum = L( imax, k );
				L( imax, k ) = L( j, k );
				L( j, k ) = dum;
			}
			d = -d;
			vv[ imax ] = vv[ j ];
		}

		indx[ j ] = imax;

		if ( j != ncols_ )
		{
			if ( L( j, j ) == 0 ) throw Singular(); // Singular Matrix

			dum = 1 / L( j, j );

			for ( i = ( j + 1 ); i < ncols_; i++ )
				L( i, j ) *= dum;
		}
	}

	return L;
}

// Method which performs LU backsubstitution, takes as arguments row permutation
//    index vector indx and "right hand side" vector B, populates passed Matrix L
//    with result
// Example: M.ludksb( L, indx, B );
// Adapted from Numerical Recipes in C, 2nd edition, Cambridge University Press,
//    Cambridge, 1992, pages 44 - 47

template <> Matrix< double >& Matrix< double >::lubksb( Matrix< double >& L,
	vector< int >& indx, vector< double >& b )
{
	// Check sizes of Matrices are equal and square
	assert ( L.nrows_ == nrows_ && L.ncols_ == ncols_ && nrows_ == ncols_ );

	L = *this; // copy current Matrix into incoming Matrix

	int i, ii = -1, ip, j;
	double sum;

	for ( i = 0; i < ( int ) ncols_; i++ )
	{
		ip = indx[i];
		sum = b[ip];
		b[ip] = b[i];
		if ( ii >= 0 )
			for ( j = ii; j <= i - 1; j++ )
				sum -= L( i, j ) * b[j];
		else if ( sum )
			ii = i;
		b[i] = sum;
	}

	for ( i = ( ( int ) ncols_ - 1 ); i >= 0; i-- )
	{
		sum = b[i];
		for ( j = i + 1; j < ( int ) ncols_; j++ )
			sum -= L( i, j ) * b[j];

		if ( L( i, i ) == 0 ) throw Singular(); // Singular Matrix

		b[i] = sum / L( i, i );
	}

	return L;
}

// Method which computes an inverse by LU decomposition and backsubstitution,
//    populates passed Matrix I with result
// Example: M.inverseLU( I );
// Adapted from Numerical Recipes in C, 2nd edition, Cambridge University Press,
//    Cambridge, 1992, pages 44 - 48

template <> Matrix< double >& Matrix< double >::inverseLU( Matrix< double >& I ) const
{
	// Check sizes of Matrices are equal and square
	assert ( I.nrows_ == nrows_ && I.ncols_ == ncols_ && nrows_ == ncols_ );

	I = *this; // copy current Matrix into incoming Matrix

	unsigned i, j; // counters

	double d; // for sign

	vector< int > indx( ncols_ ); // row permutation index vector
	vector< double > col( ncols_ ); // "right hand side" vector

	// Placeholder Matrices
	Matrix< double > J( ncols_, ncols_ ), K( ncols_, ncols_ ), L( ncols_, ncols_ );

	try
	{
		I.ludcmp( J, indx, d ); // do the LU decomposition, put result in Matrix J

		for ( j = 0; j < ncols_; j++ )
		{
			std::fill( col.begin(), col.end(), 0 );
			col[ j ] = 1;
			J.lubksb( K, indx, col ); // backsubstitute, put result in Matrix K
			for ( i = 0; i < ncols_; i++ ) // find inverse by columns
				L( i, j ) = col[ i ];
		}

		I = L; // copy result into incoming Matrix
	}
	catch ( Singular& ) // catch Singular Matrix exception
	{
		throw; // rethrow the exception
	}

	return I; // for Matrix math
}

// Method which computes an inverse by LU decomposition, returns new Matrix
//    with result
// Example: I = A.inverseLU();
// Adapted from Numerical Recipes in C, 2nd edition, Cambridge University Press,
//    Cambridge, 1992, pages 44 - 48

template <> Matrix< double > Matrix< double >::inverseLU() const
{
	Matrix< double > I = *this; // make a copy of the current Matrix

	return this->inverseLU( I ); // use previously coded inverseLU method
}

// General method to compute inverse, populates passed Matrix with result,
//    returns reference to Matrix inverse, uses LU decomposition by default
// Example: A.inverse( I );

template <> Matrix< double >& Matrix< double >::inverse( Matrix< double >& I ) const
{
	// Check sizes of Matrices are equal and square
	assert ( I.nrows_ == nrows_ && I.ncols_ == ncols_ && nrows_ == ncols_ );

	return this->inverseLU( I ); // use previously coded inverseLU method
}

// General method to compute inverse, returns new Matrix with result,
//    returns reference to Matrix inverse, uses LU decomposition by default
// Example: I = A.inverse();

template <> Matrix< double > Matrix< double >::inverse() const
{
	Matrix< double > I = *this; // make a copy of the current Matrix

	return this->inverseLU( I ); // use previously coded inverseLU method
}

// General method to compute inverse, populates passed Matrix with result,
//    takes reference to determinant as argument, returns reference to
//    Matrix inverse, uses LU decomposition
// Example: A.inverse( I, det );

template <> Matrix< double >& Matrix< double >::inverse( Matrix< double >& I,
	double& det ) const
{
	// Check sizes of Matrices are equal and square
	assert ( I.nrows_ == nrows_ && I.ncols_ == ncols_ && nrows_ == ncols_ );

	I = *this; // copy current Matrix into incoming Matrix

	unsigned i, j; // counters

	double d; // for sign of determinant

	vector< int > indx( ncols_ ); // row permutation index vector
	vector< double > col( ncols_ ); // "right hand side" vector

	// Placeholder Matrices
	Matrix< double > J( ncols_, ncols_ ), K( ncols_, ncols_ ), L( ncols_, ncols_ );

	try
	{
		I.ludcmp( J, indx, d ); // do the LU decomposition, put result in Matrix J

		for ( j = 0; j < ncols_; j++ )
		{
			d *= J( j, j ); // calculate determinant from LU decomposed result

			std::fill( col.begin(), col.end(), 0 );
			col[ j ] = 1;
			J.lubksb( K, indx, col ); // backsubstitute, put result in Matrix K
			for ( i = 0; i < ncols_; i++ ) // find inverse by columns
				L( i, j ) = col[ i ];
		}

		det = d; // return determinant

		I = L; // copy result into incoming Matrix
	}
	catch ( Singular& ) // catch Singular Matrix exception
	{
		throw; // rethrow the exception
	}

	return I; // for Matrix math
}

// General method to compute inverse, returns new Matrix with result,
//    takes reference to determinant as argument, returns reference to
//    Matrix inverse, uses LU decomposition
// Example: I = A.inverse( det );

template <> Matrix< double > Matrix< double >::inverse( double& det ) const
{
	Matrix< double > I = *this; // make a copy of the current Matrix

	return this->inverse( I, det ); // use previously coded inverse method
}

// Return determinant only

template <> double Matrix< double >::determinant() const
{
	double d = 0; // for sign of determinant

	Matrix< double > M( ncols_, ncols_ ); // for decomposed Matrix

	try
	{
		this->ludcmp( M, d ); // do the LU decomposition, put result in Matrix M

		for ( unsigned i = 0; i < ncols_; i++ )
			d *= M( i, i ); // calculate determinant from LU decomposed result
	}
	catch ( Singular& ) // catch Singular Matrix exception
	{
		throw; // rethrow the exception
	}

	return d; // return determinant
}

// General method to compute inverse, populates passed Matrix with result,
//    returns reference to Matrix inverse, allows user to force Gauss Jordan
//    elimination with definition GaussJordan
// Example: A.inverse( I, GaussJordan );

template <> Matrix< double >& Matrix< double >::inverse( Matrix< double >& I,
	unsigned choice ) const
{
	// Check sizes of Matrices are equal and square
	assert ( I.nrows_ == nrows_ && I.ncols_ == ncols_ && nrows_ == ncols_ );

	// User chooses which type of inverse procedure
	if ( choice == GaussJordan )
		this->inverseGaussJordan( I ); // Gauss Jordan elimination
	else
		this->inverseLU( I ); // LU decomposition and backsubstitution is default

	return I; // for use in Matrix math
}

// General method to compute inverse, returns new Matrix with result,
//    returns reference to Matrix inverse, allows user to force Gauss Jordan
//    elimination with definition GaussJordan
// Example: I = A.inverse( GaussJordan );

template <> Matrix< double > Matrix< double >::inverse( unsigned choice ) const
{
	Matrix< double > I = *this; // make a copy of the current Matrix

	return this->inverse( I, choice ); // use previously coded inverse method
}

