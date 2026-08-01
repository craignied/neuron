// Methods for a 1 output node, 1 hidden layer backpropagation network
// with biases

#include "stdafx.h" // For MSVC, must be first!

#include "simpleprop.h"

// Default constructor
SimpleProp::SimpleProp()
{
	objType = "SimpleProp";
	biasFlag = true; // SimpleProp networks have bias nodes
}

// Copy constructor
SimpleProp::SimpleProp( const SimpleProp& rhs )
{
	OneHiddenNet::copy( rhs ); // use the copy utility
}

// Overloaded = operator
SimpleProp& SimpleProp::operator= ( const SimpleProp& rhs )
{
	if ( &rhs != this ) // check for self-assignment
		OneHiddenNet::copy( rhs ); // use the copy utility

	return *this; // enables A = B = C
}

// Loads a DataSet object into the SimpleProp Model
void SimpleProp::setDataSet( DataSet& dataObj )
{
	theData = dataObj; // set "theData" to incoming DataSet

	// Easier on the eyes
	unsigned nInput = theData.getInput(); // number of inputs

	// Size the vectors for inputs
	I.resize( nInput + 1 ); // holds inputs from one exemplar, last is bias

	// Use Model utility function to extract input Matrices
	Model::extractInputMatrices();

	// If training set is loaded, size SimpleProp training Matrices and vectors
	if ( theData.trainLoaded() )
	{
		// Easier on the eyes
		unsigned nTrain = theData.getNumTrain(); // examples in training set
		
		// Add biases to the training set input Matrix
		vector< double > in_bias( nTrain, 1 );
		Train = Train.addcol( in_bias ); // efficiency doesn't matter here

		// Get the output vector y from the training set's last column
		y = theData.getTrainMatrix().col( nInput );

		// Build the DataSet TwoSet object for the training set
		if ( theData.getDiscrete() ) // TwoSet object makes sense only for discrete output
			theData.setTrainTwoSet();
	}

	// If test set is loaded, size SimpleProp test Matrices and vectors
	if ( theData.testLoaded() )
	{
		// Easier on the eyes
		unsigned nTest = theData.getNumTest(); // examples in test set
		
		// Add biases to the test set input Matrix
		vector< double > test_bias( nTest, 1 );
		Test = Test.addcol( test_bias ); // efficiency doesn't matter here either

		if ( theData.getDiscrete() ) // TwoSet object makes sense only for discrete output
			theData.setTestTwoSet(); // build the TwoSet object for the test set
	}

	// Validation set (Phase 4c): prepared exactly like the test set (a bias
	//    column) so the held-out monitor can forward-propagate it. No TwoSet --
	//    the monitor reads the error directly, it never classifies the set.
	if ( theData.valLoaded() )
	{
		vector< double > val_bias( theData.getNumVal(), 1 );
		Validation = Validation.addcol( val_bias );
	}
}

// Set the number of hidden nodes, note that training set must have been loaded!
void SimpleProp::setHidden( const unsigned n )
{
	// Easier on the eyes
	unsigned nInput = theData.getInput(); // number of input nodes
	
	// Hidden number must be nonzero, and training set must have been loaded
	assert ( n != 0 && theData.trainLoaded() );

	nHidden = n;

	// Size the hidden weight, hidden weight update and gradient Matrices
	hW.resize( nHidden, ( nInput + 1 ) );
	hWup.resize( nHidden, ( nInput + 1 ) );
	hG.resize( nHidden, ( nInput + 1 ) );

	// Size the vectors for hidden outputs and weights
	hO.resize( nHidden + 1 ); // hidden node output vector, last is bias
	hO[ nHidden ] = 1; // last hidden output always held to 1 for bias
	oW.resize( nHidden + 1 ); // output weight vector
	oG.resize( nHidden + 1 ); // output gradient vector
	h_err.resize( nHidden ); // error terms for the hidden outputs 

	builtFlag = true; // the object has now been formally constructed

	weightsSetFlag = false; // the weights have not yet been set
}

// Add 'extra' hidden units as a warm start (ROADMAP 2 Phase 4, OBD growth).
//    setHidden resizes the weight structures to GARBAGE (matrix.h resize) and
//    clears weightsSetFlag, so this snapshots the old weights first and copies
//    them back afterward -- nothing may be assumed to survive the resize.
void SimpleProp::growHidden( const unsigned extra )
{
	assert ( builtFlag && weightsSetFlag && extra > 0 );

	unsigned oldH = nHidden, // hidden units before the grow
		nInput = theData.getInput();

	// Snapshot the weights the resize is about to overwrite
	Matrix< double > oldHW = hW; // oldH x ( nInput + 1 )
	vector< double > oldOW = oW; // oldH + 1 ( last element is the output bias )

	// Reuse setHidden for the sizing (it garbage-fills every structure and
	//    clears weightsSetFlag)
	setHidden( oldH + extra );

	// Restore the original hidden rows; the new rows get small random incoming
	//    weights (through the layer, per standing rule 4 -- and random weights
	//    break symmetry when extra > 1)
	for ( unsigned r = 0; r < oldH; r++ )
		hW.replacerow( r, oldHW.row( r ) );
	Matrix< double > newRows( extra, nInput + 1 );
	newRows.random( randomLimit );
	for ( unsigned r = 0; r < extra; r++ )
		hW.replacerow( oldH + r, newRows.row( r ) );

	// Restore the original output weights; the new units get ZERO outgoing
	//    weight (so the forward pass is unchanged), and the output bias moves
	//    from the old last slot to the new last slot
	for ( unsigned j = 0; j < oldH; j++ )
		oW[ j ] = oldOW[ j ];
	for ( unsigned j = oldH; j < nHidden; j++ )
		oW[ j ] = 0;
	oW[ nHidden ] = oldOW[ oldH ]; // the relocated output bias weight

	weightsSetFlag = true;

	// Belt-and-braces (train() reinitializes lastG/lastF at t==0, so this is
	//    not load-bearing): drop any stale packed-gradient / CGD state so a
	//    subsequent run cannot see a vector sized for the old architecture
	stackG.clear();
	lastG.clear();
	lastF.clear();
}

// Remove hidden units by index (ROADMAP 2 Phase 4, OBD pruning).
void SimpleProp::removeHidden( const vector< unsigned >& v )
{
	assert ( builtFlag && weightsSetFlag );
	assert ( !v.empty() && v.size() < nHidden ); // at least one unit survives
	assert ( *max_element( v.begin(), v.end() ) < nHidden );

	unsigned nInput = theData.getInput();

	// Build the keep-list (hidden indices not in v), preserving order
	vector< unsigned > keep;
	for ( unsigned j = 0; j < nHidden; j++ )
		if ( find( v.begin(), v.end(), j ) == v.end() )
			keep.push_back( j );

	unsigned newH = keep.size();

	// Gather the surviving hidden rows via the layer (rule 4); each hidden
	//    unit is a row of hW / hG
	hW = hW.includerows( keep );
	hG = hG.includerows( keep );
	hWup.resize( newH, nInput + 1 ); // update Matrix is scratch; just resize

	// Rebuild the output weights: kept units in order, output bias last
	vector< double > newOW( newH + 1 );
	for ( unsigned j = 0; j < newH; j++ )
		newOW[ j ] = oW[ keep[ j ] ];
	newOW[ newH ] = oW[ nHidden ]; // the output bias weight
	oW = newOW;

	// Resize the remaining per-unit vectors to the new width
	nHidden = newH;
	hO.resize( nHidden + 1 );
	hO[ nHidden ] = 1; // last hidden output is always the bias
	oG.resize( nHidden + 1 );
	h_err.resize( nHidden );

	weightsSetFlag = true;

	// See growHidden: not load-bearing, but keep no stale CGD/packed state
	stackG.clear();
	lastG.clear();
	lastF.clear();
}

// Saliency of each hidden unit for OBD pruning (ROADMAP 2 Phase 4).
vector< double > SimpleProp::hiddenSaliency()
{
	assert ( builtFlag && weightsSetFlag && theData.trainLoaded() );

	unsigned nTrain = theData.getNumTrain();

	// Collect each hidden unit's output over the whole training set. forward()
	//    is called per exemplar and hO read immediately: training clobbers hO
	//    in place, so it must never be trusted after a weight update.
	vector< vector< double > > samples( nHidden );
	for ( unsigned j = 0; j < nHidden; j++ )
		samples[ j ].reserve( nTrain );

	for ( unsigned r = 0; r < nTrain; r++ )
	{
		forward( Train, r );
		for ( unsigned j = 0; j < nHidden; j++ )
			samples[ j ].push_back( hO[ j ] ); // hO[ nHidden ] is the bias, skipped
	}

	// saliency_j = |oW[j]| * std( hidden output j ). Population is the class
	//    layer's moments (rule 4) and its var() is the two-pass form, so a unit
	//    that never varies gives exactly 0, not a NaN (legacy bug #6).
	vector< double > saliency( nHidden );
	for ( unsigned j = 0; j < nHidden; j++ )
	{
		Population p( samples[ j ] );
		saliency[ j ] = fabs( oW[ j ] ) * p.std();
	}

	return saliency;
}

// Returns the degrees of freedom of this Network object
unsigned SimpleProp::df()
{
	return ( ( ( theData.getInput() + 2 ) * nHidden ) + 1 );
}

// Outputs a header to ostream describing this SimpleProp model architecture
void SimpleProp::outputHeader( ostream& outputStream )
{
	// Easier on the eyes
	unsigned nInput = theData.getInput(); // number of input nodes
	
	outputStream << objType << endl; // type of object
	
	// Remind user biases by definition
	outputStream << "Bias nodes on all layers by definition" << endl;

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
double SimpleProp::trainSet()
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
//    The two statements below are this model's whole contribution: read the
//    exemplar, and pin the bias slot. OneHiddenNet::propagate() then carries
//    the equations, which are the same for both architectures.
void SimpleProp::forward( Matrix< double >& data, unsigned example )
{
	// Get a single row from the input Matrix
	// (Freeman & Skapura p. 102, #1)
	data.row( example, I );

	hO[ nHidden ] = 1; // restore bias node

	propagate();
}

// Remove input nodes from this network, takes vector representing
//    which nodes to be removed as argument
void SimpleProp::removeInputs( const vector< unsigned >& v )
{
	// Check incoming vector elements in bounds	
	assert ( *max_element( v.begin(), v.end() ) < theData.getInput() );

	// Use DataSet::removeInputs to remove inputs in the dataset
	theData.removeInputs( v );

	// Use Model utility function to extract input Matrices
	Model::extractInputMatrices();

	// Resize the vector for single exemplar inputs
	I.resize( theData.getInput() + 1 ); // last is bias

	// If training set is loaded, size SimpleProp training Matrices and vectors
	if ( theData.trainLoaded() )
	{
		// Add biases to the training set input Matrix
		vector< double > in_bias( theData.getNumTrain(), 1 );
		Train = Train.addcol( in_bias ); // efficiency doesn't matter here
	}

	// If test set is loaded, size SimpleProp test Matrices and vectors
	if ( theData.testLoaded() )
	{
		// Add biases to the test set input Matrix
		vector< double > test_bias( theData.getNumTest(), 1 );
		Test = Test.addcol( test_bias ); // efficiency doesn't matter here either
	}

	hW = hW.excludecols( v ); // remove hidden weights corresponding to inputs
	hG = hG.excludecols( v ); // remove hidden gradient corresponding to inputs

	hWup.resize( nHidden, theData.getInput() + 1 ); // resize hidden weight update Matrix
}

// Convert single vector stackG back to weight gradient structure
void SimpleProp::unpack()
{
	vector< double > vpack; // temporary holding vector
	vpack.resize( ( theData.getInput() + 1 ) * nHidden );
	
	// Copy the part of stackG corresponding to the hidden gradients to
	//    its holding vector, which has been sized in setHidden(...)
	vpack.assign( stackG.begin(),
		stackG.begin() + ( ( theData.getInput() + 1 ) * nHidden ) );
	toMatrix( hG, vpack ); // convert the holding vector to hidden gradient Matrix

	// Copy the part of stackG corresponding to the output weights to
	//    the output gradients vector
	oG.assign( stackG.begin() + ( ( theData.getInput() + 1 ) * nHidden ),
		stackG.end() );
}

