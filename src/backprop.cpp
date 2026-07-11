// Methods for a general backpropagation network

#include "stdafx.h" // For MSVC, must be first!

#include "backprop.h"

// Copy constructor
BackProp::BackProp( const BackProp& rhs )
{
	BackProp::copy( rhs ); // use the copy utility
}

// Overloaded = operator
BackProp& BackProp::operator= ( const BackProp& rhs )
{
	if ( &rhs != this ) // check for self-assignment
		BackProp::copy( rhs ); // use the copy utility

	return *this; // enables A = B = C
}

// Copy utility
void BackProp::copy( const BackProp& rhs )
{
	Network::copy( rhs ); // call immediate base object copy
	nLayer = rhs.nLayer;
	nLayers = rhs.nLayers;
	in_bias = rhs.in_bias;
	I = rhs.I;
	y = rhs.y;
	o_err = rhs.o_err;

	// Resizing Weights, WeightsUp & WeightsAccumulate necessary because they're Vector of Matrix
	Weights.resize( nLayers + 1 );
	WeightsUp.resize( nLayers + 1 );
	WeightsAccumulate.resize( nLayers + 1 );
	Gradient.resize( nLayers + 1 );
	vpack.resize( nLayers + 1 );

	// Here we are explictly calling the operator = for each Matrix
	for ( unsigned i = 0; i < ( nLayers + 1 ); i++ )
	{
		Weights[ i ] = rhs.Weights[ i ];
		WeightsUp[ i ] = rhs.WeightsUp[ i ];
		WeightsAccumulate[ i ] = rhs.WeightsAccumulate[ i ];
		Gradient[ i ] = rhs.Gradient[ i ];
		vpack[ i ] = rhs.vpack[ i ];
	}

	// Resizing the HOutputs and HErrors here is necessary because its a Vector of Vector
	HOutputs.resize( nLayers );
	HErrors.resize( nLayers );

	// Here we are explictly calling the operator = for each Vector
	for ( unsigned j = 0; j < nLayers; j++ )
	{
		HOutputs[ j ] = rhs.HOutputs[ j ];
		HErrors[ j ] = rhs.HErrors[ j ];
	}
}

// Loads a DataSet object into the BackProp Model
void BackProp::setDataSet( DataSet& dataObj )
{
	theData = dataObj; // set "theData" to incoming DataSet

	// Easier on the eyes
	unsigned nInput = theData.getInput(), // number of inputs
		nOutput = theData.getOutput(); // number of outputs

	// Use Model utility functions to extract input & output Matrices
	Model::extractInputMatrices();
	Model::extractOutputMatrices();
		
	// Size vectors & Matrices which are *not* hidden node dependent
	y.resize( nOutput ); // size vector for known single exemplar outputs
	output.resize( nOutput ); // size vector for predicted outputs
	o_err.resize( nOutput ); // size vector for output errors
	xs.resize( nOutput ); // size vector for arguments in sigmoidal transfer function

	if( Network::getBias() ) // true means we have bias nodes
	{
		I.resize( nInput + 1 ); // holds inputs from one exemplar, last is bias

		// If training set is loaded, size BackProp training Matrices and vectors
		if ( theData.trainLoaded() )
		{
			// Easier on the eyes
			unsigned nTrain = theData.getNumTrain(); // examples in training set

			// Add biases to the training set input Matrix
			vector< double > in_bias( nTrain, 1 );

			Train = Train.addcol( in_bias ); // efficiency doesn't matter here
		}

		// If test set is loaded, size BackProp test Matrices and vectors
		if ( theData.testLoaded() )
		{
			// Easier on the eyes
			unsigned nTest = theData.getNumTest(); // examples in test set

			// Add biases to the test set input Matrix
			vector< double > test_bias( nTest, 1 );

			Test = Test.addcol( test_bias ); // efficiency doesn't matter here either
		}
	}

	else // no bias nodes
	{
		I.resize( nInput ); // holds inputs from one exemplar
	}

	// Build TwoSet objects if 1 discrete output
	if ( nOutput == 1 && theData.getDiscrete() )
	{
		// If training set is loaded, build its TwoSet object
		if ( theData.trainLoaded() )
			theData.setTrainTwoSet();

		// If test set is loaded, build its TwoSet object
		if ( theData.testLoaded() )
			theData.setTestTwoSet();
	}
}

// Sets the hidden nodes, training set must be loaded prior to this method
void BackProp::setHidden( const vector< unsigned >& v_in )
{
	// Set member objects for this method
	nLayer = v_in;
	nLayers = nLayer.size();

	// Easier on the eyes
	unsigned nInput = theData.getInput(), // get the number of input nodes
		nOutput = theData.getOutput(); // get number of output nodes

	if ( Network::getBias() ) // true means we have bias nodes
	{
		// vector of Weights matrix is resized to number of hidden layers and output layer
		Weights.resize( nLayers + 1 ); // all hidden layers and output layer

		// Resizing all the matrices 
		// Boundary case, first hidden layer
		Weights[ 0 ].resize( nLayer[ 0 ], nInput + 1 );
		// Boundary case, output layer
		Weights[ nLayers ].resize( nOutput, nLayer[ nLayers - 1 ] + 1 );
		// All the hidden layers between the first hidden layer and output layer
		for ( unsigned l = 0; l < nLayers - 1; l++ )
			Weights[ l + 1 ].resize( nLayer[ l + 1 ], nLayer[ l ] + 1 );
	
		// Vector of WeightsUp matrix is resized to number of hidden layers and output layer
		WeightsUp.resize( nLayers + 1 ); // all hidden layers and output layer
		
		// Resizing all the matrices 
		// Boundary case, first hidden layer
		WeightsUp[ 0 ].resize( nLayer[ 0 ], nInput + 1 );
		// Boundary case, output layer
		WeightsUp[ nLayers ].resize( nOutput, nLayer[ nLayers - 1 ] + 1 );
		// All the hidden layers between the first hidden layer and output layer
		for ( unsigned m = 0; m < nLayers - 1; m++ )
			WeightsUp[ m + 1 ]. resize( nLayer[ m + 1 ], nLayer[ m ] + 1 );

		HOutputs.resize( nLayers ); // sets the number of hidden output layers
		// Set the number of outputs in each hidden layer
		for ( unsigned i = 0; i < nLayers; i++ )
		{
			HOutputs[ i ].resize( nLayer[ i ] + 1 );
			HOutputs[ i ][ HOutputs[ i ].size() - 1 ] = 1; // set biases
		}

		HErrors.resize( nLayers ); // sets size as number of hidden output layers
		// Set the number of outputs in each hidden layer
		for ( unsigned j = 0; j < nLayers; j++ )
			HErrors[ j ].resize( nLayer[ j ] );

		// vector of Gradient matrix is resized to number of hidden layers and output layer
		Gradient.resize( nLayers + 1 ); // all hidden layers and output layer

		// Resizing all the matrices 
		// Boundary case, first hidden layer
		Gradient[ 0 ].resize( nLayer[ 0 ], nInput + 1 );
		// Boundary case, output layer
		Gradient[ nLayers ].resize( nOutput, nLayer[ nLayers - 1 ] + 1 );
		// All the hidden layers between the first hidden layer and output layer
		for ( unsigned n = 0; n < nLayers - 1; n++ )
			Gradient[ n + 1 ].resize( nLayer[ n + 1 ], nLayer[ n ] + 1 );
		
		vpack.resize( nLayers + 1 ); // all hidden layers and output layer
		// Resizing all the matrices 
		// Boundary case, first hidden layer
		vpack[ 0 ].resize( nLayer[ 0 ] * ( nInput + 1 ) );
		// Boundary case, output layer
		vpack[ nLayers ].resize( nOutput * ( nLayer[ nLayers - 1 ] + 1 ) );
		// All the hidden layers between the first hidden layer and output layer
		for ( unsigned p = 0; p < nLayers - 1; p++ )
			vpack[ p + 1 ].resize( nLayer[ p + 1 ] * ( nLayer[ p ] + 1 ) );
	}
	else // no bias
	{
		// Vector of Weights matrix is resized to number of hidden layers and output layer
		Weights.resize( nLayers + 1 ); // all hidden layers and output layer
		
		// Resizing all the matrices 
		// Boundary case, first hidden layer
		Weights[ 0 ].resize( nLayer[ 0 ], nInput );
		// Boundary case, output layer
		Weights[ nLayers ].resize( nOutput, nLayer[ nLayers - 1 ] );
		// All the hidden layers between the first hidden layer and output layer
		for ( unsigned k = 0; k < nLayers - 1; k++ )
			Weights[ k + 1 ]. resize( nLayer[ k + 1 ], nLayer[ k ] );

		// vector of Weights matrix is resized to number of hidden layers and output layer
		WeightsUp.resize( nLayers + 1 ); // all hidden layers and output layer

		// Resizing all the matrices 
		// Boundary case, first hidden layer
		WeightsUp[ 0 ].resize( nLayer[ 0 ], nInput );
		// Boundary case, output layer
		WeightsUp[ nLayers ].resize( nOutput, nLayer[ nLayers - 1 ] );
		// All the hidden layers between the first hidden layer and output layer
		for (unsigned m = 0; m < nLayers - 1; m++ )
			WeightsUp[ m + 1 ]. resize( nLayer[ m + 1 ], nLayer[ m ] );

		HOutputs.resize( nLayers ); // sets the number of hidden output layers
		// Set the number of outputs in each hidden layer
		for ( unsigned i = 0; i < nLayers; i++ )
			HOutputs[ i ].resize( nLayer[ i ] );

		HErrors.resize( nLayers ); // sets size as number of hidden output layers
		// Set the number of outputs in each hidden layer
		for ( unsigned j = 0; j < nLayers; j++ )
			HErrors[ j ].resize( nLayer[ j ] );

		// vector of Gradient matrix is resized to number of hidden layers and output layer
		Gradient.resize( nLayers + 1 ); // all hidden layers and output layer
		
		// Resizing all the matrices 
		// Boundary case, first hidden layer
		Gradient[ 0 ].resize( nLayer[ 0 ], nInput );
		// Boundary case, output layer
		Gradient[ nLayers ].resize( nOutput, nLayer[ nLayers - 1 ] );
		// All the hidden layers between the first hidden layer and output layer
		for ( unsigned p = 0; p < nLayers - 1; p++ )
			Gradient[ p + 1 ]. resize( nLayer[ p + 1 ], nLayer[ p ] );

		// vector vpack of vectors is resized to number of hidden layers and output layer
		vpack.resize( nLayers + 1 ); // all hidden layers and output layer
		// Resizing all the matrices 
		// Boundary case, first hidden layer
		vpack[ 0 ].resize( nLayer[ 0 ] * nInput );
		// Boundary case, output layer
		vpack[ nLayers ].resize( nOutput * nLayer[ nLayers - 1 ] );
		// All the hidden layers between the first hidden layer and output layer
		for ( unsigned q = 0; q < nLayers - 1; q++ )
			vpack[ q + 1 ]. resize( nLayer[ q + 1 ] * nLayer[ q ] );
	}

	// Resizing the vector and each matrix in the vector. See how neat it is to make it 
	//    independent of having bais node or not.
	WeightsAccumulate.resize( Weights.size() );
	for ( unsigned k = 0; k < WeightsAccumulate.size(); k++ )
		WeightsAccumulate[ k ].resize( Weights[ k ].rows(), Weights[ k ].cols() );

	builtFlag = true; // the object has now been formally constructed
	weightsSetFlag = false; // the weights have not yet been set

} // end of setHidden method

// Returns the degrees of freedom of this Network object
unsigned BackProp::df()
{
	unsigned degree = 0;
	if ( Network::getBias() ) // true means we have bias nodes
	{
		degree += ( theData.getInput() + 1 ) * nLayer[ 0 ];
		degree += ( nLayer[ nLayers - 1 ] + 1 ) * theData.getOutput();
		for ( unsigned i = 0; i < nLayers - 1; i++ )
			degree += ( nLayer[ i ] + 1 ) * nLayer[ i + 1 ];	
	}
	else // no bias
	{
		degree += theData.getInput() * nLayer[ 0 ];
		degree += theData.getOutput() * nLayer[ nLayers - 1 ];
		for ( unsigned i = 0; i < nLayers - 1; i++ )
			degree += nLayer[ i ] * nLayer[ i + 1 ];
	}

	return degree;
}

// Randomizes initial weights in this Network object
void BackProp::randomize()
{
	assert( builtFlag ); // the object must have been built first

	// Randomize the weights for all hidden layers and output layer
	for ( unsigned i = 0; i < nLayers + 1; i++ )
	{
		Weights[ i ].random( randomLimit );
	}
	
	// Log to history file
	if ( historyFlag ) // make sure flag for history is set
	{
		// Construct the message to the output stream
		ostringstream fileStream;
		fileStream << endl << "I've randomized the weights of a BackProp object."
			<< endl << endl;
		addHistory( fileStream ); // append to history file
	}

	weightsSetFlag = true; // set flag to indicate weights now set
}

// Trains one iteration through the training set, model dependant
//    returns set error
double BackProp::trainSet()
{
	// First check if automatic stepsize selection is chosen (which
	//    *requires* offline or batch/epoch training)
	if ( batchEpochFlag && automaticStepSizeFlag )
	{
		unsigned nInput = theData.getInput(), // get the number of input nodes
		nOutput = theData.getOutput(); // get number of output nodes
		vector< double > lastGBuffer, lastFBuffer;
		vector< vector< double > > vpackBuffer;
		vector< Matrix< double > > WeightsBuffer;

		// Resize WeightsBuffer
		if ( Network::getBias() ) // true means we have bias nodes
		{
			// vector of WeightsBuffer matrix is resized to number of hidden layers and output layer
			WeightsBuffer.resize( nLayers + 1 ); // all hidden layers and output layer

			// Resizing all the matrices 
			// Boundary case, first hidden layer
			WeightsBuffer[ 0 ].resize( nLayer[ 0 ], nInput + 1 );
			// Boundary case, output layer
			WeightsBuffer[ nLayers ].resize( nOutput, nLayer[ nLayers - 1 ] + 1 );
			// all the hidden layers between the first hidden layer and output layer
			for ( unsigned l = 0; l < nLayers - 1; l++ )
				WeightsBuffer[ l + 1 ].resize( nLayer[ l + 1 ], nLayer[ l ] + 1 ); 
		}
		else // case of no biases
		{
			// vector of Weights matrix is resized to number of hidden layers and output layer
			WeightsBuffer.resize( nLayers + 1 ); // all hidden layers and output layer
		
			// Resizing all the matrices 
			// Boundary case, first hidden layer
			WeightsBuffer[ 0 ].resize( nLayer[ 0 ], nInput );
			// Boundary case, output layer
			WeightsBuffer[ nLayers ].resize( nOutput, nLayer[ nLayers - 1 ] );
			// all the hidden layers between the first hidden layer and output layer
			for ( unsigned k = 0; k < nLayers - 1; k++ )
				WeightsBuffer[ k + 1 ]. resize( nLayer[ k + 1 ], nLayer[ k ] );
		}

		// Buffer the weights
		for ( unsigned z = 0; z < ( nLayers + 1 ); z++ )
			WeightsBuffer[ z ] = Weights[ z ];
			
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
			if ( ! ( ( iteration == 0 ) && ( loopCounter == 1 ) ) )
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
		for ( unsigned x = 0; x < ( nLayers + 1 ); x++ )
			Weights[ x ] = WeightsBuffer[ x ];
				
		lastG = lastGBuffer;
		lastF = lastFBuffer;
		
	} // end of if loop

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
double BackProp::innerTrainSet()
{
	// Easier on the eyes
	unsigned nTrain = theData.getNumTrain(); // examples in training set

	double setError = 0; // initialize the set error

	// For off-line training: accumulator containers for hidden
	//    and output weights--would be expensive, but it's *outside* the
	//    main loop, and so doesn't add much inefficiency to the on-line case
	for ( unsigned k = 0; k < WeightsAccumulate.size(); k++ )
		WeightsAccumulate[ k ].fill( 0.0 );

	// Reset average output error accumulator for automatic stepsize selection
	o_errAccumulate = 0.0;

	// Loop through all exemplars in the set
	for ( unsigned example = 0; example < nTrain; example++ )
	{
		// Get known single exemplar outputs vector y from known outputs Matrix
		TrainOutput.row( example, y );
		
		// Begin by forward propagating the exemplar
		forward( Train, example );

		// Calculate error for single output and add to set error
		errorFunction E( y, output, xs, errorType );
		setError += E.value();

		if ( E.boundsErr() ) // check for out of bounds error in log(o)
			boundsErrorFlag = true;

		// Regularization term for error, Manifest Methodology equation 2.1
		//    $\frac{1}{2\sigma_w^2} \sum_{i=1}^m |{\bf y}|^2$ in
		//    $E_k({\bf y}) = e(t^k, {\bf a}^{(m)}_k) + \frac{1}{2\sigma_w^2} \sum_{i=1}^m |{\bf y}|^2$
		if ( weightDecayFlag )
		{
			for ( unsigned l = 0; l < nLayers + 1; l++ )//adding for all hidden layers and output layer
				setError += regularizer * Weights[ l ].squared();	
		}
		
		// Calculate the error terms for the output units
		// (Freeman & Skapura p. 102, #6)
		// Note that using func which takes the output vector as the
		// last argument, and *= instead of *, is the most efficient way
		// Calculation of o_err is still the same because we still have o_err
		// and for all hidden layers we have HErrors
		// Note also that if the error is x-entropy, then dE/do = (y-o)/(o(1-o))
		// and the o(1-o) in the denominator cancels d_sigmoidal = o(1-o)
		// hence splitting this code into 2 lines with an if:
		// Note that sign changed from y - output to output - y to conform to methodology section.
		o_err =  output - y;
		if ( errorType == 0 ) // error is LMS
			o_err *= func( output, d_sigmoidal() );

		unsigned nOutput = theData.getOutput(); // get number of output nodes

		// If automatic stepsize selection, accumulate average error
		if ( batchEpochFlag && automaticStepSizeFlag )
			for( unsigned y = 0; y < nOutput; y++ )
				o_errAccumulate += o_err[ y ];

		// Calculate the error terms for the hidden units
		// (Freeman & Skapura p. 102, #7)
		// Note the necessity of transposing the output weight matrix Weights[]
		// prior to performing the dot product by using dotprodt, and also
		// in the case of bias, where size HErrors[] < Weights[] cols, that the result
		// of dotprodt will truncate to HErrors[] size, and that the result of
		// func, also > size HErrors[], will truncate to size of HErrors[] because 
		// *= results in the size of LHS
		// Special case of the last hidden layer
		Weights[ nLayers ].dotprodt( o_err, HErrors[ nLayers - 1 ] )
			*= func( HOutputs[ nLayers - 1 ], d_sigmoidal() );
		// For all the hidden layers execept the last layer
		for ( int i = nLayers - 2; i > - 1; i-- )
			Weights[ i + 1 ].dotprodt( HErrors[ i + 1 ], HErrors[ i ] )
				*= func( HOutputs[ i ], d_sigmoidal() );

		// Weight decay
		if ( weightDecayFlag && ( trainingType == 0 ) && !gradMaxFlag )
		{
			for ( unsigned m = 0; m < Weights.size(); m++ )
				Weights[ m ] *= decayTerm;
		}

		if ( !batchEpochFlag ) // classic on-line backpropagation
		{
			// Tests of gradient calculation were found to slow training
			//    by more than 50%, hence the if block
			if ( ( trainingType == 0 ) && !gradMaxFlag ) // canonical backprop
			{
				// Update the output weights, note again the use of *= for efficiency
				// (Freeman & Skapura p. 102, #8)
				Weights[ nLayers ] -= ( WeightsUp[ nLayers ].outprod(
					o_err, HOutputs[ nLayers - 1 ] ) *= eta );

				// Update the hidden weights
				// (Freeman & Skapura p. 102, #9)
				for ( unsigned j = nLayers - 1; j > 0; j-- )
					Weights[ j ] -= ( WeightsUp[ j ].outprod(
						HErrors[ j ], HOutputs[ j - 1 ] ) *= eta );

				// Special case: update of first hidden layer weights
				Weights[ 0 ] -= ( WeightsUp[ 0 ].outprod( HErrors[ 0 ], I ) *= eta );
			}
			else // calculate the gradient as a separate structure
			{
				// Calculate the output and hidden gradients
				Gradient[ nLayers ] = ( WeightsUp[ nLayers ].outprod(
					o_err, HOutputs[ nLayers - 1 ] ) );
				for ( unsigned j = nLayers - 1; j > 0; j-- )
					Gradient[ j ] = ( WeightsUp[ j ].outprod(
						HErrors[ j ], HOutputs[ j - 1 ] ) );
				// Special case: update of first hidden layer weights
				Gradient[ 0 ] = ( WeightsUp[ 0 ].outprod( HErrors[ 0 ], I ) );

				// Weight decay, Manifest Methodology section 2.2.1
				// right hand term $(1/\sigma_w^2)[{\bf W} \;,\; {\bf b}]$
				// in ${\bf G}^{(j)}_k = \delta^{(j)}_k [{\bf f}^{(j-1)}_k]^T + (1/\sigma_w^2)[{\bf W} \;,\; {\bf b}]$
				if( weightDecayFlag )
				{
					Gradient[ nLayers ] += Weights[ nLayers ] * decay;
					for ( unsigned l = nLayers - 1; l > 0; l-- )
						Gradient[ l ] += Weights[ l ] * decay;
					Gradient[ 0 ] += Weights[ 0 ] * decay;
				}

				// Whatever additional algorithm is chosen
				engine( trainingType, ( iteration * nTrain ) + example );

				// Update the output and hidden weights
				Weights[ nLayers ] -= Gradient[ nLayers ] * eta;	
				for ( unsigned k = nLayers - 1; k > 0; k-- )
					Weights[ k ] -= Gradient[ k ] * eta;	
				Weights[ 0 ] -= Gradient[ 0 ] * eta;
			}
		} // end of if ( !batchEpochFlag )

		else // off-line or batch/epoch learning
		{
			if ( ( trainingType == 0 ) && !gradMaxFlag ) // canonical backprop
			{
				// Note that eta has been factored out for the batch update at the end
				WeightsAccumulate[ nLayers ] += ( WeightsUp[ nLayers ].outprod(
					o_err, HOutputs[ nLayers - 1 ] ) );

				for ( unsigned j = nLayers - 1; j > 0; j-- )
					WeightsAccumulate[ j ] += ( WeightsUp[ j ].outprod(
					HErrors[ j ], HOutputs[ j - 1 ] ) );

				WeightsAccumulate[ 0 ] += ( WeightsUp[ 0 ].outprod( HErrors[ 0 ], I ) );
			}
			else // calculate the gradient as a separate structure
			{
				// Calculate the output and hidden gradients
				Gradient[ nLayers ] = ( WeightsUp[ nLayers ].outprod(
					o_err, HOutputs[ nLayers - 1 ] ) );
				for ( unsigned j = nLayers - 1; j > 0; j-- )
					Gradient[ j ] = ( WeightsUp[ j ].outprod(
					HErrors[ j ], HOutputs[ j - 1 ] ) );
				Gradient[ 0 ] = ( WeightsUp[ 0 ].outprod( HErrors[ 0 ], I ) );

				// See above note in on-line block
				if( weightDecayFlag )
				{
					Gradient[ nLayers ] += Weights[ nLayers ] * decay;
					for ( unsigned l = nLayers - 1; l > 0; l-- )
						Gradient[ l ] += Weights[ l ] * decay;
					Gradient[ 0 ] += Weights[ 0 ] * decay;
				}

				// Update the accumulators	
				WeightsAccumulate[ nLayers ] += Gradient[ nLayers ];
				for ( unsigned k = nLayers - 1; k > 0; k-- )
				WeightsAccumulate[ k ] += Gradient[ k ];	
				WeightsAccumulate[ 0 ] += Gradient[ 0 ];
			}
		} // end of offline else
	} // end of for loop for examplars in training set

	if ( batchEpochFlag ) // off-line or batch/epoch learning
	{
		// Canonical backprop without separate gradient calculation
		if ( ( trainingType == 0 ) && !gradMaxFlag )
		{
			// Batch/epoch updates weights at the end, *now* multiply by eta
			for ( unsigned l = 0; l < WeightsAccumulate.size(); l++ )
				Weights[ l ] -= ( ( WeightsAccumulate[ l ] *= eta ) /= ( double ) nTrain );
		}
		else // routines where gradient is calculated separately
		{
			// Set the gradients to the accumulators so that pack() & unpack() work
			for ( unsigned l = 0; l < Gradient.size(); l++ )
				Gradient[ l ] = WeightsAccumulate[ l ] /= ( double ) nTrain;

			// Whatever additional algorithm is chosen
			engine( trainingType, iteration );

			// Update the output and hidden weights
			for ( unsigned m = 0; m < WeightsAccumulate.size(); m++ )
				Weights[ m ] -= ( WeightsAccumulate[ m ] *= eta );
		}
	}

	return setError / nTrain; // return the calculated set error
}

// Forward propagates for one input vector in dataset, takes dataset Matrix
//    as first argument, position in dataset as 2nd argument
void BackProp::forward( Matrix< double >& data, unsigned example )
{
	// Get a single row from the input Matrix
	// (Freeman & Skapura p. 102, #1)
	data.row( example, I );

	if( Network::getBias() ) // bias nodes
	{
		// Take the dot product of the hidden weights Matrix Weights[]  and the
		// transpose of a row of the input Matrix I, and apply the
		// sigmoidal function to the resulting vector

		// Special case of first hidden layer
		// Because the last element of HOutputs[0] is the bias, only apply the
		// operations to elements 0 to nLayer[0]-1 which is one less then the number of 
		// neurons in first hidden layer
		// (Freeman & Skapura p. 102, #2 & #3)
		func( Weights[ 0 ].dotprod( I, HOutputs[ 0 ], 0, nLayer[ 0 ] - 1 ),
			sigmoidal(), HOutputs[ 0 ], 0, nLayer[ 0 ] - 1 );

		// For all other hidden layers except first layer
		for ( unsigned i = 1; i < nLayers; i++ )
			func( Weights[ i ].dotprod( HOutputs[ i - 1 ] , HOutputs[ i ], 0, nLayer[ i ] - 1 ),
				sigmoidal(), HOutputs[ i ], 0, nLayer[ i ] - 1 );
	}

	else // no bias
	{
		// For no bias, apply operations to entire HOutputs[]
		func( Weights[ 0 ].dotprod( I, HOutputs[ 0 ] ), sigmoidal(), HOutputs[ 0 ] );
		for ( unsigned i = 1; i < nLayers; i++ )
			func( Weights[ i ].dotprod( HOutputs[ i - 1 ] , HOutputs[ i ] ),
				sigmoidal(), HOutputs[ i ] );
	}
	
	// Take the dot product of the output weights Matrix Weights[nLayers] and the
	// hidden output vector for last hidden layer HOutputs[ NLayers -1 ], 
	// and apply the sigmoidal function to the resulting vector to get the output vector
	// (Freeman & Skapura p. 102, #4 & #5)
	// xs (inherited from Network) holds arguments in sigmoidal transfer function for
	//    multiple output Networks
	Weights[ nLayers ].dotprod( HOutputs[ nLayers - 1 ], xs );
	func( xs, sigmoidal(), output );

	o = output[ 0 ]; // for classAccuracy(...) for 1-output case 
}

// Remove input nodes from this network, takes vector representing
//    which nodes to be removed as argument
void BackProp::removeInputs( const vector< unsigned >& v )
{
	// Check incoming vector elements in bounds	
	assert ( *max_element( v.begin(), v.end() ) < theData.getInput() );

	// Use DataSet::removeInputs to remove inputs in the dataset
	theData.removeInputs( v );

	// Use Model utility function to extract input Matrices
	Model::extractInputMatrices();

	if( Network::getBias() ) // bias nodes
	{
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
		
		Weights[ 0 ] = Weights[ 0 ].excludecols( v ); // remove hidden weights corresponding to inputs
		Gradient[ 0 ] = Gradient[ 0 ].excludecols( v ); // remove hidden gradient corresponding to inputs
		// Here Weights[ 0 ] represents the matrix of weights between input layer and first hidden layer.
		WeightsUp[ 0 ].resize( nLayer[ 0 ], theData.getInput() + 1 ); // resize the Weight update matrix
		WeightsAccumulate[ 0 ].resize( nLayer[ 0 ], theData.getInput() + 1 ); // resize the accumulator matrix
		vpack[ 0 ].resize( nLayer[ 0 ] * ( theData.getInput() + 1 ) );//resize vpack with new number of inputs
	}
	else
	{
		I.resize( theData.getInput() ); // resize the vector for single exemplar inputs
		Weights[ 0 ] = Weights[ 0 ].excludecols( v ); // remove hidden weights corresponding to inputs
		Gradient[ 0 ] = Gradient[ 0 ].excludecols( v ); // remove hidden gradient corresponding to inputs

		// Here Weights[ 0 ] represents the matrix of weights between input layer and first hidden layer.
		WeightsUp[ 0 ].resize( nLayer[ 0 ], theData.getInput() ); // resize the Weight update matrix
		WeightsAccumulate[ 0 ].resize( nLayer[ 0 ], theData.getInput() ); // resize the accumulator matrix
		vpack[ 0 ].resize( nLayer[ 0 ] * theData.getInput() );//resize vpack with new number of inputs
	}
}

// Save this Network object to a file, takes filename as ( string ) argument
//    returns boolean flag to indicate if file succesfully saved
bool BackProp::save( string& filename )
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

		// Next line is hidden layer label, followed by hidden layer matrices for each hidden layer
		for ( unsigned i = 0; i < nLayers; i++ )
		{
			savefile << "Hidden layer " << i + 1 << " :" << endl;
			savefile << Weights[ i ].setHeader( false ); // no header is necessary
		}

		// Followed by output layer label, followed by output layer vector
		savefile << "Output layer weights:" << endl;
		savefile << Weights[ nLayers ].setHeader( false );

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

// Outputs a header to ostream describing this BackProp model architecture
void BackProp::outputHeader( ostream& outputStream )
{
	// Easier on the eyes
	unsigned nInput = theData.getInput(); // number of input nodes
	
	outputStream << objType << endl; // type of object
	
	if( Network::getBias() ) // bias nodes
		outputStream << "1  bias nodes present " << endl;
	else
		outputStream << "0  bias nodes absent " << endl;

	// Next line is number of input nodes inherited from model object
	outputStream << nInput << " input nodes" << endl;

	// Next line gives the number of hidden layers 
	outputStream << nLayers << " hidden layers " << endl;

	// Next line gives number of hidden nodes for each layer
	for ( unsigned i = 0; i < nLayers; i++ )
		outputStream << nLayer[ i ] << " ";
	outputStream << " are hidden nodes for each hidden layer" << endl;

	// Number of output nodes
	outputStream << Network::output.size() << " output nodes" << endl;
}

// Load this Network object from a file, takes filename as ( string ) argument
//    returns boolean flag to indicate if file succesfully loaded
bool BackProp::load( string& filename )
{
	// Easier on the eyes
	unsigned nInput = theData.getInput(), // number of input nodes
		nOutput = theData.getOutput(); // number of output nodes
	
	bool success = false; // flag to indicate file successfully loaded
	
	string lineString, // holder for strings pulled from file
		goodFile; // the name of the actual file

	// Open the file as read-only, and associate it with the 'label' loadfile
	goodFile = util::getGoodFile( filename );
	ifstream loadfile( goodFile.c_str(), ios::in );

	getline( loadfile, lineString ); // 1st line should be BackProp
	util::chopEndl( lineString ); // remove <cr> from end of string if exists
	assert( lineString == objType );

	unsigned bias; // bias value of 1 or 0 from file, 1 => bias
	loadfile >> bias; // retrieve bias info from file
	getline( loadfile, lineString ); // "  bias node present " or "  bias node absent"

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
	    loadfile >> nLayers; // retrieve the number of hidden layers
		getline( loadfile, lineString ); // " hidden layers"
		util::chopEndl( lineString ); // remove <cr> from end of string if exists
		
		vector< unsigned > nHvecTemp; // vector of number of neurons in each hidden layer
		nHvecTemp.resize( nLayers ); // resize to number of hidden layers

		// Read the values of number of hidden nodes for each hidden layer from file and fill nHvecTemp[]
		for ( unsigned i = 0; i < nLayers; i++ )
			loadfile >> nHvecTemp[i];
		getline( loadfile, lineString ); // " are the hidden nodes for each layer"
		
		setHidden( nHvecTemp ); // set the architecture of this object
		
		unsigned nOutputFromFile; // number of output nodes obtained from file
		loadfile >> nOutputFromFile; // retrieve number of output nodes
		getline( loadfile, lineString ); // " output nodes"

		// Make sure number of output nodes matches dataset
		if ( nOutputFromFile != nOutput )
		{
			cout << "I cannot load this file:" << endl;
			cout << "The number of output nodes do not match the dataset ( ";
			cout << nOutput << " )" << endl;
		}
		else
		{
			// for each hidden layer, get the hidden layer weights from file
			for ( unsigned j = 0; j < nLayers; j++ )
			{
				getline( loadfile, lineString ); // "Hidden layer j:"
				loadfile >> Weights[ j ]; // retrieve hidden layer weights

				// Pick up the space and carriage return after the Matrix object
				getline( loadfile, lineString ); // THIS IS CRITICAL!
			}

			getline( loadfile, lineString ); // "Output layer weights:"
			loadfile >> Weights[ nLayers ]; // retrieve output layer

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
		} // end of inner else
		
	} // end of outer else
	
	loadfile.close(); // close input file

	return success; // return flag to indicate if file successfully loaded
}

// Convert weight gradient structure to single vector stackG
void BackProp::pack()
{
	// Starts by converting first Gradient matrix to stackG
	stackG = Gradient[ 0 ].toVector();

	// Then append all the remaining hidden Gradient matrix and output matrix to stackG
	for( unsigned i = 1; i < nLayers + 1; i++ )
		std::copy( Gradient[ i ].begin(), Gradient[ i ].end(), back_inserter( stackG ) );
}

// Convert single vector stackG back to weight gradient structure
void BackProp::unpack()
{
	vector< double > :: const_iterator start;
	vector< double > :: const_iterator end;

	start = stackG.begin();
	for( unsigned i = 0; i < nLayers + 1; i++ )
	{
		end = start + ( Gradient[ i ].rows() * Gradient[ i ].cols() );
		vpack[ i ].assign( start, end ); // its holding vector, which has been sized in setHidden(...)
		toMatrix( Gradient[ i ], vpack[ i ] ); // convert the holding vector to hidden gradient Matrix
		start = end;
	}
}

