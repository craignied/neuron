// Methods for a 1 output node, 1 hidden layer backpropagation network
// without biases

#include "stdafx.h" // For MSVC, must be first!

#include "bareprop.h"

// Default constructor
BareProp::BareProp()
{
	objType = "BareProp";
	biasFlag = false; // BareProp networks don't have bias nodes
}

// Copy constructor
BareProp::BareProp( const BareProp& rhs )
{
	OneHiddenNet::copy( rhs ); // use the copy utility
}

// Overloaded = operator
BareProp& BareProp::operator= ( const BareProp& rhs )
{
	if ( &rhs != this ) // check for self-assignment
		OneHiddenNet::copy( rhs ); // use the copy utility
	
	return *this; // enables A = B = C
}

// Loads a DataSet object into the BareProp Model
void BareProp::setDataSet( DataSet& dataObj )
{
	theData = dataObj; // set "theData" to incoming DataSet

	// Easier on the eyes
	unsigned nInput = theData.getInput();

	// Size the vectors for inputs
	I.resize( nInput ); // holds inputs from one exemplar

	// Use Model utility function to extract input Matrices
	Model::extractInputMatrices();

	// If training set is loaded, size BareProp training Matrices and vectors
	if ( theData.trainLoaded() )
	{
		// Get the output vector y from the training set's last column
		y = theData.getTrainMatrix().col( nInput );

		// Build the DataSet TwoSet object for the training set
		if ( theData.getDiscrete() ) // TwoSet object makes sense only for discrete output
			theData.setTrainTwoSet();
	}

	// If test set is loaded & discrete, build the TwoSet object for the test set
	if ( theData.testLoaded() && theData.getDiscrete() )
		theData.setTestTwoSet();
}

// Set the number of hidden nodes, note that training set must have been loaded!
void BareProp::setHidden( const unsigned n )
{
	// Easier on the eyes
	unsigned nInput = theData.getInput(); // number of input nodes
	
	// Hidden number must be nonzero, and training set must have been loaded
	assert ( n != 0 && theData.trainLoaded() );
	
	nHidden = n;

	// Size the hidden weight, update and gradient Matrices
	hW.resize( nHidden, nInput );
	hWup.resize( nHidden, nInput );
	hG.resize( nHidden, nInput );

	// Size the vectors for hidden outputs and weights
	hO.resize( nHidden ); // hidden node output vector
	oW.resize( nHidden ); // output weight vector
	oG.resize( nHidden ); // output weight gradient vector
	h_err.resize( nHidden ); // error terms for the hidden outputs

	builtFlag = true; // the object has now been formally constructed

	weightsSetFlag = false; // the weights have not yet been set
}

// Returns the degrees of freedom of this Network object
unsigned BareProp::df()
{
	return ( ( theData.getInput() + 1 ) * nHidden );
}

// Outputs a header to ostream describing this BareProp model architecture
void BareProp::outputHeader( ostream& outputStream )
{
	// Easier on the eyes
	unsigned nInput = theData.getInput(); // number of input nodes
	
	outputStream << objType << endl; // type of object
	
	// Remind user biases by definition
	outputStream << "No bias nodes by definition" << endl;

	// Next line is number of input nodes inherited from model object
	outputStream << nInput << " input nodes" << endl;

	// Next line reminds user 1 hidden layer by definition
	outputStream << "1 hidden layer by definition" << endl;

	// Next line is number of hidden nodes
	outputStream << nHidden << " hidden nodes" << endl;

	// Remind user 1 output by definition
	outputStream << "1 output node by definition" << endl;
}

// Trains one iteration through the training set, model dependant
//    returns set error
double BareProp::trainSet()
{
	// The eta search lives once, in Network::searchStepSize. What is written
	//    here is what differs between models: the GUARD, and the model type
	//    from which the search builds its local weight snapshot.
	//    The batchEpochFlag half is real: the search compares the error of a
	//    whole pass against the previous pass, which only means anything when
	//    a pass makes ONE weight update. Logistic omits it -- see logistic.cpp.
	return searchStepSize( *this, batchEpochFlag && automaticStepSizeFlag );
}

// Forward propagates for one input vector in dataset, takes dataset Matrix
//    as first argument, position in dataset as 2nd argument.
//    Reading the exemplar is this model's whole contribution: it has no bias
//    slot to pin. OneHiddenNet::propagate() then carries the equations, which
//    are the same for both architectures.
void BareProp::forward( Matrix< double >& data, unsigned example )
{
	// Get a single row from the input Matrix
	// (Freeman & Skapura p. 102, #1)
	data.row( example, I );

	propagate();
}

// Remove input nodes from this network, takes vector representing
//    which nodes to be removed as argument
void BareProp::removeInputs( const vector< unsigned >& v )
{
	// Check incoming vector elements in bounds	
	assert ( *max_element( v.begin(), v.end() ) < theData.getInput() );

	// Use DataSet::removeInputs to remove inputs in the dataset
	theData.removeInputs( v );

	// Use Model utility function to extract input Matrices
	Model::extractInputMatrices();

	I.resize( theData.getInput() ); // resize the vector for single exemplar inputs

	hW = hW.excludecols( v ); // remove hidden weights corresponding to inputs
	hG = hG.excludecols( v ); // remove hidden gradient columns corresponding to inputs 
	
	hWup.resize( nHidden, theData.getInput() ); // resize hidden weight update Matrix
}

// Convert single vector stackG back to weight gradient structure
void BareProp::unpack()
{
	vector< double > vpack;  // temporary holding vector
	vpack.resize( theData.getInput() * nHidden );
	
	// Copy the part of stackG corresponding to the hidden gradients to
	//    its holding vector, which has been sized in setHidden(...)
	vpack.assign( stackG.begin(), stackG.begin() + ( theData.getInput() * nHidden ) );
	toMatrix( hG, vpack ); // convert the holding vector to hidden gradient Matrix

	// Copy the part of stackG corresponding to the output weights to
	//    the output gradients vector
	oG.assign( stackG.begin() + ( theData.getInput() * nHidden ), stackG.end() );	
}


