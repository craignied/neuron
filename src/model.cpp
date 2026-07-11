// Methods for Model, the most fundamental abstract base class for neUROn2++

#include "stdafx.h" // For MSVC, must be first!

#include "model.h"

// Default constructor
Model::Model() : builtFlag ( false ), historyFlag ( true ), lastopFlag ( true ),
	historyFilename ( "neuron.log" ), lastopFilename ( "model.txt" ),
	errorLabel ( "LMS" ) { }

// Default destructor
Model::~Model() { }

// Copy constructor
Model::Model( const Model& rhs )
{
	Model::copy( rhs ); // use the copy utility
}

// Overloaded = operator
Model& Model::operator= ( const Model& rhs )
{
	if ( &rhs != this ) // check for self-assignment
		Model::copy( rhs ); // use the copy utility
	
	return *this; // enables A = B = C
}

// Copy utility
void Model::copy( const Model& rhs )
{
	theData = rhs.theData;
	Train= rhs.Train;
	Test = rhs.Test;
	TrainOutput = rhs.TrainOutput;
	TestOutput = rhs.TestOutput;
	builtFlag = rhs.builtFlag;
	historyFlag = rhs.historyFlag;
	lastopFlag = rhs.lastopFlag;
	errorType = rhs.errorType;
	objType = rhs.objType;
	historyFilename = rhs.historyFilename;
	lastopFilename = rhs.lastopFilename;
	errorLabel = rhs.errorLabel;
}

// Set output error function to be least mean squared
void Model::setLMSerror()
{
	errorLabel = "LMS"; // string containing name of error function
	errorType = 0;
}

// Set output error function to be x-entropy
void Model::setXEerror()
{
	if ( !theData.getDiscrete() ) // output must be discrete
		cout << "I'm sorry, output must be discrete for x-entropy error.";

	else
	{
		errorLabel = "X-entropy"; // string containing name of error function
		errorType = 1;
	}
}

// Utility function to append ostringstream to history file
//    takes ostringstream as argument, return true if successful
bool Model::addHistory( ostringstream& outputStream )
{
	bool success = false; // flag to indicate if operation successful

	if ( historyFlag ) // only perform if history flag is set
	{
		// Open history file for appended output
		ofstream historyFile( historyFilename.c_str(), ios::out | ios::app );
	
		if ( !historyFile.is_open() ) // test to insure it was opened
			cout << "Error in opening " << historyFilename << "!" << endl;
		else
		{
			historyFile << outputStream.str(); // write the output stream to the file
			historyFile.close(); // and close the file

			success = true; // operation was successful
		}
	}

	return success; // return flag to indicate if operation successful
}

// Utility function to extract inputs submatrices from DataSet Matrices
void Model::extractInputMatrices()
{
	// Easier on the eyes
	unsigned nInput = theData.getInput(); // number of inputs

	// If training set is loaded, extract its input submatrix
	if ( theData.trainLoaded() )
	{
		// Easier on the eyes
		unsigned nTrain = theData.getNumTrain(); // examples in training set
		
		// Build Matrix Train, which is training set columns for inputs only
		if ( nInput == 0 ) // case of no inputs
			Train.resize( nTrain, 0 );
		else if ( nInput == 1 ) // case of 1 input
		{
			Train.resize( nTrain, 0 );
			Train = Train.addcol( theData.getTrainMatrix().col( 0 ) );
		}
		else // multiple inputs
			Train = theData.getTrainMatrix().submatrix( 0, ( nTrain - 1 ),
				0, ( nInput - 1 ) );
	}
	
	// If test set is loaded, extract its input submatrix
	if ( theData.testLoaded() )
	{
		// Easier on the eyes
		unsigned nTest = theData.getNumTest(); // examples in test set

		// Build Matrix Test, which is test set columns for inputs only
		if ( nInput == 0 ) // case of no inputs
			Test.resize( nTest, 0 );
		else if ( nInput == 1 ) // case of 1 input
		{
			Test.resize( nTest, 0 );
			Test = Test.addcol( theData.getTestMatrix().col( 0 ) );
		}
		
		else // multiple inputs
			Test = theData.getTestMatrix().submatrix( 0, ( nTest - 1 ),
				0, ( nInput - 1 ) );
	}
}

// Utility function to extract outputs submatrices from DataSet Matrices
void Model::extractOutputMatrices()
{
	// Easier on the eyes
	unsigned nOutput = theData.getOutput(), // number of outputs
		nInput = theData.getInput(); // number of inputs

	// If training set is loaded, extract its output submatrix
	if ( theData.trainLoaded() )
	{
		// Easier on the eyes
		unsigned nTrain = theData.getNumTrain(); // examples in training set

		// Build Matrix TrainOutput, which is training set columns for outputs only
		if ( nOutput == 1 ) // case of 1 output
		{
			TrainOutput.resize( nTrain, 0 );
			TrainOutput = TrainOutput.addcol( theData.getTrainMatrix().col( nInput ) );

		}
		else // multiple outputs
			TrainOutput = theData.getTrainMatrix().submatrix( 0, ( nTrain - 1 ),
				nInput, ( nInput + nOutput - 1 ) );
	}

	// If test set is loaded, extract its outputs submatrix
	if ( theData.testLoaded() )
	{
		// Easier on the eyes
		unsigned nTest = theData.getNumTest(); // examples in test set

		// Build Test Matrix to hold exemplar outputs for the test set
		if ( nOutput == 1 ) // case of 1 output
		{
			TestOutput.resize( nTest, 0 );
			TestOutput = TestOutput.addcol( theData.getTestMatrix().col( nInput ) );
		}
		
		else // multiple outputs
		TestOutput = theData.getTestMatrix().submatrix( 0, ( nTest - 1 ),
			nInput, ( nInput + nOutput - 1 ) );
	}
}
