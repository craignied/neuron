// Methods for DFA, discriminant function analysis

#include "stdafx.h" // For MSVC, must be first!

#include "dfa.h"

// Default constructor
DFA::DFA() { }

// Default destructor
DFA::~DFA() { }

// Copy constructor
DFA::DFA( const DFA& rhs )
{
	DFA::copy( rhs ); // use the copy utility
}

// Overloaded = operator
DFA& DFA::operator= ( const DFA& rhs )
{
	if ( &rhs != this ) // check for self-assignment
		DFA::copy( rhs ); // use the copy utility
	
	return *this; // enables A = B = C
}

// Copy utility
void DFA::copy( const DFA& rhs )
{
	Model::copy( rhs ); // call immediate base object copy
	X = rhs.X;
	U0 = rhs.U0;
	U1 = rhs.U1;
	P = rhs.P;
	K = rhs.K;
	d = rhs.d;
	U = rhs.U;
	trainClasses = rhs.trainClasses;
	testClasses = rhs.testClasses;
	P0 = rhs.P0;
	P1 = rhs.P1;
	K0 = rhs.K0;
	K1 = rhs.K1;
	d0 = rhs.d0;
	d1 = rhs.d1;
	D0 = rhs.D0;
	D1 = rhs.D1;
	D = rhs.D;
}

// Loads a DataSet object into the DFA Model
void DFA::setDataSet( DataSet& dataObj )
{
	theData = dataObj; // set "theData" to incoming DataSet
	
	// For DFA, DataSet output *must be* discrete
	if ( !theData.getDiscrete() )
		util::screen() << "WARNING: DFA object NOT constructed, output NOT discrete" << endl;
	else
	{
		// Easier on the eyes
		unsigned nInput = theData.getInput(),
			nOutput = theData.getOutput();

		// Size the vector for inputs
		X.resize( nInput ); // holds inputs from one exemplar
		
		// Use Model utility function to extract input Matrices
		Model::extractInputMatrices();

		if ( theData.trainLoaded() ) // a training set was loaded
		{
			if ( nOutput == 1 ) // if 1-output model
			{
				theData.setTrainTwoSet(); // build training TwoSet object
				if ( theData.testLoaded() ) // if test set loaded
					theData.setTestTwoSet(); // build test TwoSet object
				
				// Build Matrices for input columns of classes 0 and 1
				D0.resize( 0, Train.cols() );
				D1.resize( 0, Train.cols() );
				
				unsigned r; // row counter
				double y; // the single output
				
				for ( r = 0; r < Train.rows(); r++ ) // cycle through data input Matrix rows
				{
					y = theData.getTrainMatrix()( r, nInput ); // get known output
					if ( y == 0 ) // if belongs to class 0
						D0 = D0.addrow( Train.row( r ) ); // put in class 0 Matrix
					else // otherwise
						D1 = D1.addrow( Train.row( r ) ); // put in class 1 Matrix
				}
				
				// A priori probabilities
				P0 = ( double ) D0.rows() / ( double ) Train.rows();
				P1 = ( double ) D1.rows() / ( double ) Train.rows();
				
				// Build mean vectors
				U0 = D0.colsums() / ( double ) D0.rows();
				U1 = D1.colsums() / ( double ) D1.rows();
			}

			else // multiple output model
			{
				// Size the vector of classes for the training set
				trainClasses.resize( Train.rows() );

				// Get the vector of classes for the training set from its
				//    output columns, test to make sure it's good
				if ( !theData.getTrainMatrix().submatrix( 0, Train.rows() - 1,
					nInput, nInput + nOutput - 1 ).rowindex( trainClasses ) )
					util::screen() << "Sorry, that training set had bad output columns."
						<< endl;
				else
				{
					unsigned o, // output column counter
						r; // row counter

					// Make initial generic Matrix to hold inputs of each class
					Matrix< double > Class( 0, Train.cols() );
					
					// Build Matrices for input columns of each class
					for ( o = 0; o < nOutput; o++ ) 
						D.push_back( Class );

					// Cycle through training set Matrix
					for ( r = 0; r < Train.rows(); r++ )
						// Add the training set row to its appropriate class
						D[ trainClasses[ r ] ] =
							D[ trainClasses[ r ] ].addrow( Train.row( r ) );

					// A priori probabilities and mean vectors
					P.resize( nOutput ); // size probabilities vector
					for ( o = 0; o < nOutput; o++ )
					{
						// Compute a priori probability for each class
						P[ o ] = ( double ) D[ o ].rows() / ( double ) Train.rows();
					
						// Build mean vectors
						U.push_back( D[ o ].colsums() / ( double ) D[ o ].rows() );
					}
				}
			}
		}
	}
}

// Outputs a header to ostream describing this DFA model architecture
//    and run, which is just the number of input and output nodes, and
//    number of exemplars
void DFA::outputHeader( ostream& outputStream )
{
	// Next line is number of input nodes inherited from model object
	outputStream << theData.getInput() << " input nodes" << endl;

	// Next line is number of output nodes inherited from model object
	outputStream << theData.getOutput() << " output nodes" << endl;

	if ( theData.trainLoaded() ) // output number of exemplars in training set
		outputStream << "Number of exemplars in training set: "
			<< theData.getTrainMatrix().rows() << endl;

	if ( theData.testLoaded() ) // output number of exemplars in test set
		outputStream << "Number of exemplars in test set: "
			<< theData.getTestMatrix().rows() << endl << endl;
}

