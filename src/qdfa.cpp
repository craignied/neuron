// Methods for QDFA, quadratic discriminant function analysis

#include "stdafx.h" // For MSVC, must be first!

#include "qdfa.h"

// Copy constructor
QDFA::QDFA( const QDFA& rhs )
{
	QDFA::copy( rhs ); // use the copy utility
}

// Overloaded = operator
QDFA& QDFA::operator= ( const QDFA& rhs )
{
	if ( &rhs != this ) // check for self-assignment
		QDFA::copy( rhs ); // use the copy utility
	
	return *this; // enables A = B = C
}

// Copy utility
void QDFA::copy( const QDFA& rhs )
{
	DFA::copy( rhs ); // call immediate base object copy
	C0 = rhs.C0;
	C1 = rhs.C1;
	S0 = rhs.S0;
	S1 = rhs.S1;
	C = rhs.C;
	S = rhs.S;
	Det0 = rhs.Det0;
	Det1 = rhs.Det1;
	Det = rhs.Det;
}

// Trains the model, returns the final error
double QDFA::train()
{
	// Easier on the eyes
	unsigned nOutput = theData.getOutput(); // number of output nodes

	// For reporting to screen and file
	ostringstream screenStream, fileStream;
	screenStream.str( "" ); // just in case, flush the streams
	fileStream.str( "" );

	screenStream << "I'm running QDFA:" << endl;
	outputHeader( screenStream ); // output the header identifying the model object

	try
	{
		if ( nOutput == 1 )
		{
			// Covariance Matrices
			C0 = D0.covariance();
			C1 = D1.covariance();

			// Inverse of covariance Matrices, get determinants
			S0 = C0.inverse( Det0 );
			S1 = C1.inverse( Det1 );

			// Constants
			K0 = log( Det0 ) - log( P0 );
			K1 = log( Det1 ) - log( P1 );
			
			reportAccuracy( screenStream ); // report the accuracy
		}
		else // multiple output dataset
		{
			Det.resize( nOutput ); // size the determinant vector

			// Build covariance & inverse Matrices, determinant & constant vectors
			for ( unsigned o = 0; o < nOutput; o++ )
			{
				C.push_back( D[ o ].covariance() ); // covariance Matrix
				S.push_back( C[ o ].inverse( Det[ o ] ) ); // inverse & determinants
				K.push_back( log( Det[ o ] ) - log( P[ o ] ) ); // constant
			}

			reportAccuracy( screenStream ); // report the accuracy
		}
	}
	catch ( Matrix< double >::Singular& e )
	{
		cout << "Can't do QDFA: " << e.what() << endl;
	}
	
	screenStream << endl; // finish off stream
	fileStream << screenStream.str(); // stream line into file stream
	cout << screenStream.str(); // then print to screen

	addHistory( fileStream ); // append to history file if specified by flag

	if ( lastopFlag ) // make sure flag for last operation output is set
	{
		// Open last operation file for output, overwrite if it exists
		string logPath = util::run_path( lastopFilename );
		ofstream lastopFile( logPath.c_str(), ios::out | ios::trunc );

		if ( !lastopFile.is_open() ) // test to insure it was opened
			cout << "Error in opening " << logPath << "!" << endl;
		else
		{
			lastopFile << fileStream.str(); // write the file stream
			lastopFile.close(); // and close the file
		}
	}

	return -1; // DFA does not return a set error
}

// Outputs to ostream reporting the accuracy of this QDFA Model
void QDFA::reportAccuracy( ostream& outputStream )
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
				d0 = dotprod( X - U0, S0.dotprod( X - U0 ) ) + K0;
				d1 = dotprod( X - U1, S1.dotprod( X - U1 ) ) + K1;
				
				if ( d0 < d1 ) // the *smaller* result is the predicted class
					theData.getTrainTwoSet().test( r ) = 0;
				else
					theData.getTrainTwoSet().test( r ) = 1;
			}
				
		if ( theData.getTestTwoSet().loaded() ) // if test TwoSet exists
			for ( r = 0; r < Test.rows(); r++ ) // iterate through the Test Matrix rows
			{
				X = Test.row( r ); // get input row

				// Calculate discriminant functions
				d0 = dotprod( X - U0, S0.dotprod( X - U0 ) ) + K0;
				d1 = dotprod( X - U1, S1.dotprod( X - U1 ) ) + K1;

				if ( d0 < d1 ) // the *smaller* result is the predicted class
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
					d[ o ] = dotprod( X - U[ o ], S[ o ].dotprod( X - U[ o ] ) )
						+ K[ o ];

				// The *smaller* is the predicted class
				if ( trainClasses[ r ] == ( unsigned )
					( min_element( d.begin(), d.end() ) - d.begin() ) )
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
				cout << "Sorry, that test set had bad output columns."
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
						d[ o ] = dotprod( X - U[ o ], S[ o ].dotprod( X - U[ o ] ) )
							+ K[ o ];
					
					// The *smaller* is the predicted class
					if ( testClasses[ r ] == ( unsigned )
						( min_element( d.begin(), d.end() ) - d.begin() ) )
						correct++;
				}
				
				correct /= ( double ) Test.rows(); // calculate test set accuracy
				
				outputStream << "Classification accuracy in the test set = "
					<< correct * 100 << "%" << endl;
			}
		}
	}
}
