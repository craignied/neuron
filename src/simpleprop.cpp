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
	SimpleProp::copy( rhs ); // use the copy utility
}

// Overloaded = operator
SimpleProp& SimpleProp::operator= ( const SimpleProp& rhs )
{
	if ( &rhs != this ) // check for self-assignment
		SimpleProp::copy( rhs ); // use the copy utility
	
	return *this; // enables A = B = C
}

// Copy utility
void SimpleProp::copy( const SimpleProp& rhs )
{
	Network::copy( rhs ); // call immediate base object copy
	nHidden = rhs.nHidden;
	nH = rhs.nH;
	hW = rhs.hW;
	hWup = rhs.hWup;
	in_bias = rhs.in_bias;
	y = rhs.y;
	I = rhs.I;
	hO = rhs.hO;
	oW = rhs.oW;
	h_err = rhs.h_err;
	o_err = rhs.o_err;
	hG = rhs.hG;
	oG = rhs.oG;
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
}

// Set the number of hidden nodes, note that training set must have been loaded!
void SimpleProp::setHidden( const unsigned n )
{
	// Easier on the eyes
	unsigned nInput = theData.getInput(); // number of input nodes
	
	// Hidden number must be nonzero, and training set must have been loaded
	assert ( n != 0 && theData.trainLoaded() );

	nHidden = n;
	nH = n - 1; // for ease of indexing with biases, because the last
	            // element of some vectors is a bias, will apply
	            // certain operations only to elements 0 to nHidden - 1

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

// Returns the degrees of freedom of this Network object
unsigned SimpleProp::df()
{
	return ( ( ( theData.getInput() + 2 ) * nHidden ) + 1 );
}

// Randomizes initial weights in this Network object
void SimpleProp::randomize()
{
	assert( builtFlag ); // the object must have been built first
	
	// Randomize the weights
	hW.random( randomLimit ); // randomize hW
	nvec::random( oW, randomLimit ); // randomize oW

	// Log to history file
	if ( historyFlag ) // make sure flag for history is set
	{
		// Construct the message to the output stream
		ostringstream fileStream;
		fileStream << endl << "I've randomized the weights of a " << objType << " object."
			<< endl << endl;
		addHistory( fileStream ); // append to history file
	}

	weightsSetFlag = true; // set flag to indicate weights now set
}

// Save this Network object to a file, takes filename as ( string ) argument
//    returns boolean flag to indicate if file succesfully saved
bool SimpleProp::save( string& filename )
{
	bool success = false; // flag to indicate file successfully saved
	
	// Open the output file to save data, overwrite file if it exists
	ofstream savefile( filename.c_str(), ios::out | ios::trunc );
	
	// Test to insure it was opened
	if ( !savefile.is_open() )
		cout << "Error in opening file to save " << objType << " network!" << endl;
	else
	{
		outputHeader( savefile ); // output the header to the file

		// Next line is hidden layer label, followed by hidden layer matrix
		savefile << "Hidden layer:" << endl;
		savefile << hW.setHeader( false ); // no header is necessary

		// Followed by output layer label, followed by output layer vector
		savefile << "Output layer:" << endl;
		savefile << oW << endl;

		// Print message to user notifying successful save to file
		cout << "The " << objType << " network was successfully saved to " << filename;
		cout << "." << endl;

		savefile.close(); // close output file

		success = true; // set flag to indicate file successfully saved

		// Log to history file
		if ( historyFlag ) // make sure flag for history is set
		{
			// Construct the output stream with object name and header
			ostringstream fileStream;
			fileStream << "I've saved the file " << filename << ":" << endl;
			outputHeader( fileStream ); // output the header to the history file
			fileStream << endl;
			
			addHistory( fileStream ); // append to history file
		}	
	}

	return success; // return flag to indicate if file successfully saved
}

// Load this Network object from a file, takes filename as ( string ) argument
//    returns boolean flag to indicate if file succesfully loaded
bool SimpleProp::load( string& filename )
{
	// Easier on the eyes
	unsigned nInput = theData.getInput(); // number of input nodes
	
	bool success = false; // flag to indicate file successfully loaded
	
	string lineString, // holder for strings pulled from file
		goodFile; // the name of the actual file

	// Open the file as read-only, and associate it with the 'label' loadfile
	goodFile = util::getGoodFile( filename );
	ifstream loadfile( goodFile.c_str(), ios::in );

	getline( loadfile, lineString ); // 1st line should be SimpleProp
	util::chopEndl( lineString ); // remove <cr> from end of string if exists
	assert( lineString == objType );
	
	getline( loadfile, lineString ); // "Bias nodes on all layers by definition"

	unsigned nInputFromFile; // number of input nodes obtained from file
	loadfile >> nInputFromFile; // retrieve number of input nodes
	getline( loadfile, lineString ); // " input nodes"

	// Make sure number of input nodes matches dataset
	if ( nInputFromFile != nInput )
	{
		cout << "I cannot load this file:" << endl;
		cout << "The number of input nodes do not match the dataset ( ";
		cout << nInput << " )" << endl;
	}
	else
	{
		getline( loadfile, lineString ); // "1 hidden layer by definition"

		loadfile >> nHidden; // retrieve number of hidden nodes
		getline( loadfile, lineString ); // " hidden nodes"
		setHidden( nHidden ); // set the architecture of this object

		getline( loadfile, lineString ); // "1 output node by definition"

		getline( loadfile, lineString ); // "Hidden layer:"
		loadfile >> hW; // retrieve hidden layer

		// Pick up the space and carriage return after the Matrix object
		getline( loadfile, lineString ); // THIS IS CRITICAL!

		getline( loadfile, lineString ); // "Output layer:"
		loadfile >> oW; // retrieve output layer

		weightsSetFlag = true; // set flag to indicate weights now set

		outputHeader( cout ); // report to user
		cout << "I've loaded the file." << endl;

		// Log to history file
		if ( historyFlag ) // make sure flag for history is set
		{
			// Construct the output stream with object name and header
			ostringstream fileStream;
			fileStream << "I've loaded the file " << goodFile << ":" << endl;
			outputHeader( fileStream ); // output header to history file
			fileStream << endl;
			
			addHistory( fileStream ); // append to history file
		}

		success = true; // set flag to indicate file successfully loaded
	}

	loadfile.close(); // close input file

	return success; // return flag to indicate if file successfully loaded
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
	// First check if automatic stepsize selection is chosen (which
	//    *requires* offline or batch/epoch training)
	if ( batchEpochFlag && automaticStepSizeFlag )
	{
		// Buffer the weights
		Matrix< double > hWBuffer;
		vector< double > oWBuffer, lastGBuffer, lastFBuffer;
		hWBuffer = hW;
		oWBuffer = oW;
		// Also buffer lastG and lastF for conjugate gradient descent / Shanno's
		lastGBuffer = lastG;
		lastFBuffer = lastF;
	
		// Initialize conditions for automatic stepsize selection
		unsigned loopCounter = 1;
		double newError = 1.0; // current calculated error
		double oldError; // previous calculated error
		double ErrorDifference = 1.0; // initially set to 1 to pass the while condition first time
		eta = 1.0; // learning rate

		// Main loop for automatic stepsize selection
		while( ( loopCounter <= maxLoops ) && ( ErrorDifference > deltaError ) )
		{
			// If we have an old error to compute ErrorDifference
			if( ! ( ( iteration == 0 ) && ( loopCounter == 1 ) ) )
			{
				// Get the previous error
				oldError = oldErrorAccumulate; // from network data member
				// Use current learning rate and compute new error
				innerTrainSet();
				// Get the new error stored in network member
				newError = o_errAccumulate; // get from network member
				// Compute error difference
				ErrorDifference = ( oldError - newError );
				// Set the new error as previous error
				oldErrorAccumulate = o_errAccumulate;

				// Check for loop condition
				if( ( loopCounter > maxLoops ) || ( ErrorDifference < deltaError ) )
					break;
				else
					eta *= gamma; // calculate learning rate
			}
			// For first iteration just iterate through training set and compute error
			else
			{ 
				innerTrainSet();
				oldErrorAccumulate = o_errAccumulate; // set new error as previous error
			}
		
			loopCounter++; // increment loop counter
		}
	
		// We have desired learning rate, lets finish the iteration
		// Retrieve the saved weights
		hW = hWBuffer;
		oW = oWBuffer;
		lastG = lastGBuffer;
		lastF = lastFBuffer;
	}

	// Now actual update of weights takes place
	double returnError;
	returnError = innerTrainSet();

	// Set the new error as previous error
	if ( batchEpochFlag && automaticStepSizeFlag )
		oldErrorAccumulate = o_errAccumulate;

	// Return the set error 
	return returnError;
}

// Inner training set algorithm called for automatic stepsize selection
//    calculates the gradient descent, and returns set error
double SimpleProp::innerTrainSet()
{
	// Easier on the eyes
	unsigned nTrain = theData.getNumTrain(), // examples in training set
		example; // example counter

	double setError = 0; // initialize the set error

	// Commented out statements to demonstrate how to calculate condition
	//    number for a neural network, if this is to be implemented in the
	//    future: initialize the matrix to hold the average gradients
	// if ( finalFlag )
		// storeGrads( df(), nTrain ); // 2 arguments initializes gradient matrix

	// For off-line training: zero'd accumulator containers for hidden
	//    and output weights--would be expensive, but it's *outside* the
	//    main loop, and so doesn't add much inefficiency to the on-line case
	Matrix< double > hWaccumulate( hW.rows(), hW.cols(), 0 );
	vector< double > oWaccumulate( oW.size(), 0 );

	// Reset average output error accumulator for automatic stepsize selection
	o_errAccumulate = 0.0;

	// Loop through all exemplars in the set
	for ( example = 0; example < nTrain; example++ )
	{
		// Begin by forward propagating the exemplar
		forward( Train, example );

		// Calculate error for single output and add to set error
		errorFunction E( y[ example ], o, x, errorType );
		setError += E.value();

		if ( E.boundsErr() ) // check for out of bounds error in log(o)
			boundsErrorFlag = true;

		// Regularization term for error, Manifest Methodology equation 2.1
		//    $\frac{1}{2\sigma_w^2} \sum_{i=1}^m |{\bf y}|^2$ in
		//    $E_k({\bf y}) = e(t^k, {\bf a}^{(m)}_k) + \frac{1}{2\sigma_w^2} \sum_{i=1}^m |{\bf y}|^2$
		if ( weightDecayFlag )
			setError += regularizer * ( hW.squared() + squared( oW ) );

		// Calculate the error term for the output unit
		// (Freeman & Skapura p. 102, #6)
		// $\delta^o_{pk} = (y_{pk} - o_{pk}){f^o_k}'(net^o_{pk})$ for mean squared error
		// Note sign is changed ( o - y ) instead of ( y - o ) to conform to Methodology
		// Methodology equation 2.8 ($t=y$, $a=o$)
		// $\delta^{(m)}_k = -({\bf t}_k - {\bf a}^{(m)})$ for mean squared error,
		// or equation 2.9
		// $\delta^{(m)}_k = -({\bf t}_k - {\bf a}^{(m)}) ./ [{\bf a}^{(m)} .* ({\bf 1} - {\bf a}^{(m)}]$
		//    for x-entropy error,
		// multiplied by the ${\bf Df}^{(j)}_k$ term in equation 2.7
		// Note that if the error is x-entropy, then dE/do = (y-o)/(o(1-o))
		//    and the o(1-o) in the denominator cancels d_sigmoidal
		//   ($={f^o_k}'$, $={\bf Df}^{(j)}_k$) = o(1-o)
		//    hence splitting this code into 2 lines with an if:
		o_err = o - y[ example ];
		if ( errorType == 0 ) // error is LMS
			o_err *= d_sigmoidal()( o ); 

		// If automatic stepsize selection, accumulate average error
		if ( batchEpochFlag && automaticStepSizeFlag )
			o_errAccumulate += o_err;

		// Calculate the error terms for the hidden units
		// (Freeman & Skapura p. 102, #7)
		// $\delta^h_{pj} = {f^h_j}'(net^h_{pj}) \sum_k \delta^o_{pk} w^o_{kj}$
		// Methodology equation 2.7
		// $\delta^{(j-1)}_k =  [{\bf W}^{(j)}]^T {\bf Df}^{(j)}_k \delta^{(j)}_k$
		// Note that using func which takes the output vector as the
		// last argument, and *= instead of *, is the most efficient way
		// Note also that although oW has 1 more element than h_err,
		// its last element will be ignored in *=
		( func( hO, d_sigmoidal(), h_err, 0, nH ) *= o_err ) *= oW;

		// Weight decay for canonical backpropagation
		// $\vec w_{t+1} = (1-2\eta\lambda)\vec w_t - \eta \left. \frac{\partial E}{\partial w} \right|_{w_t}$
		// and where the gradient doesn't need to be separated
		if ( weightDecayFlag && ( trainingType == 0 ) && !gradMaxFlag )
		{
			oW *= decayTerm;
			hW *= decayTerm;
		}

		if ( !batchEpochFlag ) // classic on-line backpropagation
		{
			// Tests of gradient calculation were found to slow training
			//    by more than 50%, hence the if block
			if ( ( trainingType == 0 ) && !gradMaxFlag ) // where gradient doesn't need to be separated
			{
				// Update the output weights (Freeman & Skapura p. 102, #8)
				// $w^o_{kj}(t+1)=w^o_{kj}(t)+\eta\delta^o_{pk}i_{pj}$
				// Note that because the output error term is o-y as in Methodology,
				// it will be a subtraction, as in Methodology equation 2.11
				// ${\bf y}_{t+1} = {\bf y}_t - \eta {\bf g}_k({\bf y}_t)$
				// where ${\bf g}_k = \delta^{(j)}_k{\bf Df}^{(j)}_k[{\bf f}^{(j-1)}_k]^T$
				// (Note that Freeman \& Skapura's $\delta^o_{pk}$ already contains ${\bf Df}^{(j)}_k$
				oW -= ( hO *= ( eta * o_err ) );
				// Update the hidden weights (Freeman & Skapura p. 102, #9)
				// $w^h_{ji}(t+1)=w^h_{ji}(t)+\eta\delta^h_{pj}x_i$
				// Also Methodology equation 2.11 as above (also a subtraction)
				hW -= ( hWup.outprod( h_err, I ) *= eta );
			}

			else // calculate the gradient as a separate structure
			{
				// Calculate the output and hidden gradients, Methodology equation 2.10
				// ${\bf g}_k = \delta^{(j)}_k{\bf Df}^{(j)}_k[{\bf f}^{(j-1)}_k]^T$
				// (Note that Freeman \& Skapura's $\delta$ already contains ${\bf Df}^{(j)}_k$)
				oG = ( hO *= o_err ); // *= for efficiency
				hG.outprod( h_err, I );

				// Weight decay, Manifest Methodology section 2.2.1
				// right hand term $(1/\sigma_w^2)[{\bf W} \;,\; {\bf b}]$
				// in ${\bf G}^{(j)}_k=\delta^{(j)}_k{\bf Df}^{(j)}_k[{\bf f}^{(j-1)}_k]^T+(1/\sigma_w^2)[{\bf W}\;,\;{\bf b}]$
				if ( weightDecayFlag )
				{
					oG += ( oW * decay );
					hG += ( hW * decay );
				}

				// Whatever additional algorithm is chosen
				engine( trainingType, ( iteration * nTrain ) + example );

				// Commented out statements to demonstrate how to calculate condition
				//    number for a neural network, if this is to be implemented in
				//    the future: convert gradient structure to a vector, then store
				//    this exemplar's gradient in the gradient matrix
				// if ( finalFlag )
				// {
					// pack();
					// storeGrads( example );
				// }

				// Update the output and hidden weights
				// Methodology equation 2.11: ${\bf y}_{t+1} = {\bf y}_t - \eta {\bf g}_k({\bf y}_t)$
				oW -= ( oG * eta );
				hW -= ( hG * eta );
			}
		}

		else // off-line or batch/epoch learning
		{
			// Where gradient doesn't need to be separated
			if ( ( trainingType == 0 ) && !gradMaxFlag )
			{
				// Accumulate output weight update, see above note
				oWaccumulate += ( hO *= o_err );
				// Accumulate hidden weight update, see above note
				hWaccumulate += hWup.outprod( h_err, I );
			}

			else // calculate the gradient as a separate structure
			{
				// Calculate the output and hidden gradients, see above note
				oG = ( hO *= o_err );
				hG.outprod( h_err, I );

				// See above note in on-line block
				if ( weightDecayFlag )
				{
					oG += ( oW * decay );
					hG += ( hW * decay );
				}

				// Commented out statements to demonstrate how to calculate condition
				//    number for a neural network, if this is to be implemented in
				//    the future: convert gradient structure to a vector, then store
				//    this exemplar's gradient in the gradient matrix
				// if ( finalFlag )
				// {
					// pack();
					// storeGrads( example );
				// }

				// Update the accumulators
				// Methodology equation 2.14: ${\bf g} = (1/N) \sum_{k=1}^N {\bf g}_k$
				oWaccumulate += oG;
				hWaccumulate += hG;
			}
		}
	} // end of loop for exemplars in training set

	if ( batchEpochFlag ) // off-line or batch/epoch learning
	{
		// Canonical backprop without separate gradient calculation
		if ( ( trainingType == 0 ) && !gradMaxFlag )
		{
			// Batch/epoch updates weights at the end, *now* multiply by eta
			// Methodology equation 2.13: ${\bf y}_{t+1} = {\bf y}_t - \eta {\bf g}({\bf y}_t)$
			// and $1/N$ in Methodology equation 2.14: ${\bf g} = (1/N) \sum_{k=1}^N {\bf g}_k$
			oW -= ( ( oWaccumulate *= eta ) / ( double ) nTrain ); // *=, /= for efficiency
			hW -= ( ( hWaccumulate *= eta ) / ( double ) nTrain );
			
		}
		
		else // routines where gradient is calculated separately
		{
			// Set the gradients to the accumulators so that pack() & unpack() work
			// $1/N$ in Methodology equation 2.14: ${\bf g} = (1/N) \sum_{k=1}^N {\bf g}_k$
			oG = oWaccumulate / ( double ) nTrain;
			hG = hWaccumulate / ( double ) nTrain;

			// Whatever additional algorithm is chosen
			engine( trainingType, iteration );

			// Update the output and hidden weights
			// Methodology equation 2.13: ${\bf y}_{t+1} = {\bf y}_t - \eta {\bf g}({\bf y}_t)$
			oW -= ( oG * eta );
			hW -= ( hG * eta );
		}
	}

	return setError / nTrain; // return the calculated set error
}

// Forward propagates for one input vector in dataset, takes dataset Matrix
//    as first argument, position in dataset as 2nd argument
void SimpleProp::forward( Matrix< double >& data, unsigned example )
{
	// Get a single row from the input Matrix
	// (Freeman & Skapura p. 102, #1)
	data.row( example, I );

	hO[ nHidden ] = 1; // restore bias node

	// Take the dot product of the hidden weights Matrix hW and the
	// transpose of a row of the input Matrix I, and apply the
	// sigmoidal function to the resulting vector
	// Because the last element of hO is the bias, only apply the
	// operations to elements 0 to nHidden - 1 (nH)
	// (Freeman & Skapura p. 102, #2 & #3)
	// 2: $net^h_{pj}=\sum_{i=1}^Nw^h_{ji}x_{pi}+\theta^h_j$
	// 3: $i_{pj}=f_j^h(net^h_{pj})$
	func( hW.dotprod( I, hO, 0, nH ), sigmoidal(), hO, 0, nH );

	// Take the dotproduct of the hidden output vector and the output
	// weights vector, and apply the sigmoidal function to the result
	// (Freeman & Skapura p. 102, #4 & #5)
	// 4: $net^o_{pk}=\sum_{j=1}^Lw^o_{kj}i_{pj}+\theta^o_k$
	// 5: $o_{pk}=f_k^o(net^o_{pk})$
	x = dotprod( hO, oW ); // note that x is inherited from Network
	o = sigmoidal()( x );
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

// Convert weight gradient structure to single vector stackG
void SimpleProp::pack()
{
	// Start by converting hidden gradients Matrix to stackG
	stackG = hG.toVector();

	// Then append output gradients vector to stackG
	std::copy( oG.begin(), oG.end(), back_inserter( stackG ) );
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

