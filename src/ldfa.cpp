// Methods for LDFA, linear discriminant function analysis

#include "stdafx.h" // For MSVC, must be first!

#include "ldfa.h"

// Copy constructor
LDFA::LDFA( const LDFA& rhs )
{
	LDFA::copy( rhs ); // use the copy utility
}

// Overloaded = operator
LDFA& LDFA::operator= ( const LDFA& rhs )
{
	if ( &rhs != this ) // check for self-assignment
		LDFA::copy( rhs ); // use the copy utility
	
	return *this; // enables A = B = C
}

// Copy utility
void LDFA::copy( const LDFA& rhs )
{
	DFA::copy( rhs ); // call immediate base object copy
	C = rhs.C;
	S = rhs.S;
}

// Trains the model, returns the final error
double LDFA::train()
{
	// Easier on the eyes
	unsigned nOutput = theData.getOutput(); // number of output nodes
	
	// For reporting to screen and file
	ostringstream screenStream, fileStream;
	screenStream.str( "" ); // just in case, flush the streams
	fileStream.str( "" );

	screenStream << "I'm running LDFA:" << endl;
	outputHeader( screenStream ); // output the header identifying the model object
	
	// Common covariance Matrix
	C = Train.covariance();

	try
	{
		// Inverse of common covariance Matrix
		S = C.inverse();

		if ( nOutput == 1 )
		{
			// Constants for 1 output ( 2 classes )
			K0 = 0.5 * dotprod( U0, S.dotprod( U0 ) ) - log( P0 );
			K1 = 0.5 * dotprod( U1, S.dotprod( U1 ) ) - log( P1 );
			
			reportAccuracy( screenStream ); // report the accuracy
		}
		else
		{
			// Constants for n-outputs
			K.resize( nOutput ); // size constants vector
			for ( unsigned o = 0; o < nOutput; o++ )
				K[ o ] = 0.5 * dotprod( U[ o ], S.dotprod( U[ o ] ) ) - log( P[ o ] );

			reportAccuracy( screenStream ); // report the accuracy
		}
	}
	catch ( Matrix< double >::Singular& e )
	{
		util::screen() << "Can't do LDFA: " << e.what() << endl;
	}

	screenStream << endl; // finish off stream
	fileStream << screenStream.str(); // stream line into file stream
	util::screen() << screenStream.str(); // then print to screen

	addHistory( fileStream ); // append to history file if specified by flag

	if ( lastopFlag ) // make sure flag for last operation output is set
	{
		// Open last operation file for output, overwrite if it exists
		string logPath = util::run_path( lastopFilename );
		ofstream lastopFile( logPath.c_str(), ios::out | ios::trunc );

		if ( !lastopFile.is_open() ) // test to insure it was opened
			util::screen() << "Error in opening " << logPath << "!" << endl;
		else
		{
			lastopFile << fileStream.str(); // write the file stream
			lastopFile.close(); // and close the file
		}
	}

	return -1; // DFA does not return a set error
}

// Outputs to ostream reporting the accuracy of this LDFA Model
void LDFA::reportAccuracy( ostream& outputStream )
{
	// Easier on the eyes
	unsigned nInput = theData.getInput(), // number of input nodes
		nOutput = theData.getOutput(); // number of output nodes
	
	unsigned r; // row counter

	if ( nOutput == 1 ) // for 1-output datasets
	{
		if ( theData.getTrainTwoSet().loaded() ) // training TwoSet must exist
			for ( r = 0; r < Train.rows(); r++ ) // iterate through Train Matrix rows
			{
				X = Train.row( r ); // get input row

				// Calculate discriminant functions
				d0 = dotprod( U0, S.dotprod( X ) ) - K0;
				d1 = dotprod( U1, S.dotprod( X ) ) - K1;
			
				if ( d0 > d1 ) // the *larger* result is the predicted class
					theData.getTrainTwoSet().test( r ) = 0;
				else
					theData.getTrainTwoSet().test( r ) = 1;
			}
				
		if ( theData.getTestTwoSet().loaded() ) // if test TwoSet exists
			for ( r = 0; r < Test.rows(); r++ ) // iterate through the Test Matrix rows
			{
				X = Test.row( r ); // get input row

				// Calculate discriminant functions
				d0 = dotprod( U0, S.dotprod( X ) ) - K0;
				d1 = dotprod( U1, S.dotprod( X ) ) - K1;
			
				if ( d0 > d1 ) // the *larger* result is the predicted class
					theData.getTestTwoSet().test( r ) = 0;
				else
					theData.getTestTwoSet().test( r ) = 1;
			}
			
		theData.metricsReport( outputStream ); // TwoSet metrics report for 1 output
	}

	else // multiple output dataset
	{
		if ( theData.trainLoaded() ) // examine the training set
		{
			double correct = 0; // to accumulate correct guesses

			for ( r = 0; r < Train.rows(); r++ ) // iterate through Train Matrix rows
			{
				X = Train.row( r ); // get input row

				// Discriminant functions for n-outputs
				d.resize( nOutput ); // size discriminant function results vector
				for ( unsigned o = 0; o < nOutput; o++ )
					d[ o ] = dotprod( U[ o ], S.dotprod( X ) ) - K[ o ];

				// The *larger* is the predicted class
				if ( trainClasses[ r ] == ( unsigned )
					( max_element( d.begin(), d.end() ) - d.begin() ) )
					correct++;
			}

			correct /= ( double ) Train.rows(); // calculate training set accuracy

			outputStream << "Classification accuracy in the training set = "
				<< correct * 100 << "%" << endl;
		}

		if ( theData.testLoaded() ) // examine the test set
		{
			// Size the vector of classes for the test set
			testClasses.resize( Test.rows() );
			
			// Get the vector of classes for the test set from its
			//    output columns, test to make sure it's good
			if ( !theData.getTestMatrix().submatrix( 0, Test.rows() - 1,
				nInput, nInput + nOutput - 1 ).rowindex( testClasses ) )
				util::screen() << "Sorry, that test set had bad output columns."
					<< endl;

			else // test set output columns were good
			{
				double correct = 0; // to accumulate correct guesses
				
				for ( r = 0; r < Test.rows(); r++ ) // iterate through Test Matrix rows
				{
					X = Test.row( r ); // get input row
					
					// Discriminant functions for n-outputs
					d.resize( nOutput ); // size discriminant function results vector
					for ( unsigned o = 0; o < nOutput; o++ )
						d[ o ] = dotprod( U[ o ], S.dotprod( X ) ) - K[ o ];
					
					// The *larger* is the predicted class
					if ( testClasses[ r ] == ( unsigned )
						( max_element( d.begin(), d.end() ) - d.begin() ) )
						correct++;
				}
				
				correct /= ( double ) Test.rows(); // calculate test set accuracy
				
				outputStream << "Classification accuracy in the test set = "
					<< correct * 100 << "%" << endl;
			}
		}
	}
}
