// Methods for QDFA, quadratic discriminant function analysis

#include "stdafx.h" // For MSVC, must be first!

#include "qdfa.h"
#include "function_defs.h" // sigmoidal, for the graded discriminant score

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

// The quadratic discriminant fit. The scaffold around it lives once in
//    DFA::train (docs/refactor_audit.md section 13).
void QDFA::fitDiscriminant()
{
	// Easier on the eyes
	unsigned nOutput = theData.getOutput(); // number of output nodes

	if ( nOutput == 1 )
	{
		// Covariance Matrices -- one PER CLASS, which is what makes this
		//    discriminant quadratic
		C0 = D0.covariance();
		C1 = D1.covariance();

		// Inverse of covariance Matrices, get determinants
		S0 = C0.inverse( Det0 );
		S1 = C1.inverse( Det1 );

		// Constants
		K0 = log( Det0 ) - log( P0 );
		K1 = log( Det1 ) - log( P1 );
	}
	else // multiple output dataset
	{
		Det.resize( nOutput ); // size the determinant vector

		// DISCARD THE PREVIOUS RUN'S FIT. The loop below appends with
		//    push_back, so without this a second train() left the first
		//    run's covariances, inverses and constants at 0 .. nOutput-1
		//    and appended the new ones past the end where nothing reads
		//    them: three runs held nine of each, and every run after the
		//    first silently reported the first run's fit. Det escaped it
		//    only because it is resized rather than appended.
		C.clear();
		S.clear();
		K.clear();

		// Build covariance & inverse Matrices, determinant & constant vectors
		for ( unsigned o = 0; o < nOutput; o++ )
		{
			C.push_back( D[ o ].covariance() ); // covariance Matrix
			S.push_back( C[ o ].inverse( Det[ o ] ) ); // inverse & determinants
			K.push_back( log( Det[ o ] ) - log( P[ o ] ) ); // constant
		}
	}
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

				// Store the GRADED class-1 score, not a hard 0/1 decision (see
				//    LDFA). Here the *smaller* discriminant wins, so class 1 is
				//    predicted when d0 >= d1; the margin toward class 1 is
				//    d0 - d1. The sigmoid keeps the 0.5 boundary (>= 0.5), so the
				//    confusion table is unchanged and the ROC gains a real curve.
				theData.getTrainTwoSet().test( r ) = sigmoidal()( d0 - d1 );
			}
				
		if ( theData.getTestTwoSet().loaded() ) // if test TwoSet exists
			for ( r = 0; r < Test.rows(); r++ ) // iterate through the Test Matrix rows
			{
				X = Test.row( r ); // get input row

				// Calculate discriminant functions
				d0 = dotprod( X - U0, S0.dotprod( X - U0 ) ) + K0;
				d1 = dotprod( X - U1, S1.dotprod( X - U1 ) ) + K1;

				// Graded class-1 score (see the train loop above)
				theData.getTestTwoSet().test( r ) = sigmoidal()( d0 - d1 );
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
