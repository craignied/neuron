// Methods for a binary logistic regression

#include "stdafx.h" // For MSVC, must be first!

#include "logistic.h"

// Default constructor
Logistic::Logistic()
{
	objType = "Binary logistic";
	biasFlag = true; // Binary logistic networks have bias nodes
	setXEerror(); // cross-entropy error by definition
	setBatchEpoch( true ); // batch-epoch by definition

	setWeightDecay( false ); // no weight decay by default
	setDecay( 0 );
	setAutoStepSize( true ); // autostepsize by default
}

// Copy constructor
Logistic::Logistic( const Logistic& rhs )
{
	Logistic::copy( rhs ); // use the copy utility
}

// Overloaded = operator
Logistic& Logistic::operator= ( const Logistic& rhs )
{
	if ( &rhs != this ) // check for self-assignment
		Logistic::copy( rhs ); // use the copy utility
	
	return *this; // enables A = B = C
}

// Copy utility
void Logistic::copy( const Logistic& rhs )
{
	Network::copy( rhs ); // call immediate base object copy
	y = rhs.y;
	I = rhs.I;
	W = rhs.W;
	o_err = rhs.o_err;
	G = rhs.G;
}

// Loads a DataSet object into the binary logistic model
void Logistic::setDataSet( DataSet& dataObj )
{
	theData = dataObj; // set "theData" to incoming DataSet

	if ( !theData.getDiscrete() ) // insure discrete output
		cout << "The output must be discrete for binary logistic regression." << endl;
	else if ( theData.getOutput() != 1 ) // insure only 1 output
		cout << "There can be only 1 output for binary logistic regression." << endl;
	else
	{
		// Easier on the eyes
		unsigned nInput = theData.getInput(); // number of inputs

		// Size the vectors for inputs
		I.resize( nInput + 1 ); // holds inputs from one exemplar, last is bias

		// Use Model utility function to extract input Matrices
		Model::extractInputMatrices();

		// If training set is loaded, size Logistic training vectors
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
			theData.setTrainTwoSet();

			// Size the vector for beta weights
			W.resize( nInput + 1 ); // beta weight vector

			builtFlag = true; // the object has now been formally constructed

			weightsSetFlag = false; // the weights have not yet been set
		}

		// If test set is loaded, size Logistic test vectors
		if ( theData.testLoaded() )
		{
			// Easier on the eyes
			unsigned nTest = theData.getNumTest(); // examples in test set
			
			// Add biases to the test set input Matrix
			vector< double > test_bias( nTest, 1 );
			Test = Test.addcol( test_bias ); // efficiency doesn't matter here either

			theData.setTestTwoSet(); // build the TwoSet object for the test set
		}
	}
}

// Randomizes initial weights in this Network object
void Logistic::randomize()
{
	assert( builtFlag ); // the object must have been built first
	
	// Randomize the beta weights
	nvec::random( W, randomLimit ); // randomize W

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
bool Logistic::save( string& filename )
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

		// Next line is y-intercept label, followed by value
		savefile << "y-intercept:" << endl;
		savefile << W[ W.size() - 1 ] << endl;

		// Next line is beta weights label, followed by values
		savefile << "Beta weights:" << endl;
		vector< double > BW( W ); // vector to hold weights
		BW.pop_back(); // get rid of y-intercept
		savefile << BW << endl;

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
bool Logistic::load( string& filename )
{
	// Easier on the eyes
	unsigned nInput = theData.getInput(); // number of input nodes

	bool success = false; // flag to indicate file successfully loaded

	string lineString, // holder for strings pulled from file
		goodFile; // the name of the actual file

	// Open the file as read-only, and associate it with the 'label' loadfile
	goodFile = util::getGoodFile( filename );
	ifstream loadfile( goodFile.c_str(), ios::in );

	getline( loadfile, lineString ); // 1st line should be Binary logistic
	util::chopEndl( lineString ); // remove <cr> from end of string if exists
	assert( lineString == objType );

	unsigned nInputFromFile; // number of input nodes obtained from file
	loadfile >> nInputFromFile; // retrieve number of input nodes
	getline( loadfile, lineString ); // " inputs"

	// Make sure number of input nodes matches dataset
	if ( nInputFromFile != nInput )
	{
		cout << "I cannot load this file:" << endl;
		cout << "The number of input nodes do not match the dataset ( ";
		cout << nInput << " )" << endl;
	}
	else
	{
		getline( loadfile, lineString ); // "1 output by definition"
		getline( loadfile, lineString ); // "y-intercept:"
		loadfile >> W[ W.size() - 1 ]; // get y-intercept value, put at end of weights vector
		getline( loadfile, lineString ); // carriage return after y-intercept value
		getline( loadfile, lineString ); // "Beta weights:"
		loadfile >> W; // retrieve Beta weights

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

// Outputs a header to ostream describing this Logistic model
void Logistic::outputHeader( ostream& outputStream )
{
	outputStream << objType << endl; // type of object

	// Next line is number of input nodes inherited from model object
	outputStream << theData.getInput() << " inputs" << endl;

	// Remind user 1 output by definition
	outputStream << "1 output by definition" << endl;
}

// Outputs to ostream reporting the accuracy of the Model, adds binary logistic
//    regression specific footer information
void Logistic::reportAccuracy( ostream& outputStream )
{
	Network::reportAccuracy( outputStream ); // call the base method first

	// Easier on the eyes
	unsigned nTrain = theData.getNumTrain(), nInput = theData.getInput();

	// Calculate the variance and errors of beta weights
	// See Hosmer & Lemeshow Applied Logistic Regression, 2nd ed., pp 36-42
	vector< double > Bstderr( nInput + 1 ), // to hold std errors of beta weights
		WaldP( nInput + 1 ); // to hold Wald test p-values (beta weight nonzero)
	std::fill( Bstderr.begin(), Bstderr.end(), -1.0 ); // error value

	// XtV to hold 1st product of X'VX, as V is diagonal (all elements other than diag = 0)
	//    setting up V and doing a Matrix dot product would waste a lot of computations
	Matrix< double > XtV( nInput + 1, nTrain ), Var; // Var to hold covariance matrix

	// Specialized dot product for X' and diagonal Matrix V
	for ( unsigned n = 0; n < nTrain; n++ )
	{
		// Training dataset has already been set up in setDataSet()
		Train.row( n, I );

		// Training set outcomes (pi in Hosmer & Lemeshow) already calc'd in Network::reportAccuracy
		o = theData.getTrainTwoSet().test( n );

		// Diagonal elements of V contain pi(1 - pi)
		for ( unsigned p = 0; p <= nInput; p++ )
			XtV( p, n ) = I[ p ] * o * ( 1 - o );
	}

	try // now do the statistical calculations 
	{
		// Var(B) = inv(X'VX): Hosmer & Lemeshow eqn 2.8, p. 41
		Var = XtV.dotprod( Train ).inverse();

		// Pick up diagonal terms of covariance matrix to calculate std error, Z, p-values
		for ( unsigned i = 0; i <= nInput; i++ )
		{
			Bstderr[ i ] = sqrt( Var( i, i ) ); // calculate std error of beta weights
			WaldP[ i ] = 2 * ( 1 - stats::Zarea( fabs( W[ i ] / Bstderr[ i ] ) ) ); // p-values
		}
	}
	catch ( Matrix< double >::Singular& e )
	{
		outputStream << "Error calculating beta variance: " << e.what() << endl << endl;
	}

	// Print out the table with the results
	outputStream << endl <<
	"   Input:       Beta weight:     Std Err:    Wald p(W!=0):" << endl << 
	"-----------    -------------   ------------  -------------" << endl;

	outputStream << setiosflags( ios::showpoint | ios::right ); // set general io flags
	for ( unsigned i = 0; i < nInput; i++ ) // iterate through the input beta weights
	{
		// Print out the input number and its beta weight
		outputStream << setw( 11 ) << setfill( ' ' ) << i << "  ";
		outputStream << resetiosflags( ios::fixed );
		outputStream << setw( 15 ) << setfill( ' ' ) << setprecision( 6 ) << W[ i ];
		
		if ( Bstderr[ i ] >= 0 ) // print std error & Wald p if could be calculated
		{
			outputStream << setw( 15 ) << setfill( ' ' ) << Bstderr[ i ];
			outputStream << setw( 15 ) << setfill( ' ' ) << setprecision( 2 )
				<< WaldP[ i ] << endl;
		}
		else // error values
		outputStream << "          Error          Error" << endl;
	}

	// Now print out the y-intercept and its beta weight (last in vector)
	outputStream << "y-intercept  ";
	outputStream << resetiosflags( ios::fixed );
	outputStream << setw( 15 ) << setfill( ' ' ) << setprecision( 6 ) << W[ nInput ];

	if ( Bstderr[ nInput ] >= 0 ) // print std error & Wald p if could be calculated
	{
		outputStream << setw( 15 ) << setfill( ' ' ) << Bstderr[ nInput ];
		outputStream << setw( 15 ) << setfill( ' ' ) << setprecision( 2 )
				<< WaldP[ nInput ] << endl;
	}
	else // error values
	outputStream << "          Error          Error" << endl;

	// Report out the condition number
	outputStream << endl;
	reportCondNum( outputStream );
}

// Trains one iteration through the training set, model dependant
//    returns set error
double Logistic::trainSet()
{
	// First check if automatic stepsize selection is chosen
	if ( automaticStepSizeFlag )
	{
		// Buffer the weights
		vector< double > WBuffer = W, lastGBuffer = lastG, lastFBuffer = lastF;
	
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
		W = WBuffer;
		lastG = lastGBuffer;
		lastF = lastFBuffer;
	}

	// Now actual update of weights takes place
	double returnError;
	returnError = innerTrainSet();

	// Set the new error as previous error
	if ( automaticStepSizeFlag )
		oldErrorAccumulate = o_errAccumulate;

	// Return the set error 
	return returnError;
}

// Inner training set algorithm called for automatic stepsize selection
//    calculates the gradient descent, and returns set error
double Logistic::innerTrainSet()
{
	// Easier on the eyes
	unsigned nTrain = theData.getNumTrain(), // examples in training set
		example; // example counter

	double setError = 0; // initialize the set error

	// For calculating the condition number, initialize the matrix to hold
	//    the average gradients
	if ( finalFlag )
		storeGrads( df(), nTrain ); // 2 arguments will initialize the gradient matrix

	// For off-line training: zero'd accumulator containers for beta weights
	vector< double > Waccumulate( W.size(), 0 );

	// Reset average error accumulator for automatic stepsize selection
	o_errAccumulate = 0.0;

	// Loop through all exemplars in the set
	for ( example = 0; example < nTrain; example++ )
	{
		// Begin by forward propagating the exemplar
		forward( Train, example );

		// Calculate error and add to set error
		errorFunction E( y[ example ], o, x, errorType );
		setError += E.value();

		if ( E.boundsErr() ) // check for out of bounds error in log(o)
			boundsErrorFlag = true;

		// Regularization term for error, Manifest Methodology equation 2.1
		//    $\frac{1}{2\sigma_w^2} \sum_{i=1}^m |{\bf y}|^2$ in
		//    $E_k({\bf y}) = e(t^k, {\bf a}^{(m)}_k) + \frac{1}{2\sigma_w^2} \sum_{i=1}^m |{\bf y}|^2$
		if ( weightDecayFlag )
			setError += regularizer * ( squared( W ) );

		// Calculate the error term
		// Note that since the error is x-entropy, then dE/do = (y-o)/(o(1-o))
		//    and the o(1-o) in the denominator cancels d_sigmoidal
		//   ($={f^o_k}'$, $={\bf Df}^{(j)}_k$) = o(1-o)
		o_err = o - y[ example ];

		// If automatic stepsize selection, accumulate average error
		if ( automaticStepSizeFlag )
			o_errAccumulate += o_err;

		// Weight decay for canonical gradient descent
		// $\vec w_{t+1} = (1-2\eta\lambda)\vec w_t - \eta \left. \frac{\partial E}{\partial w} \right|_{w_t}$
		// and where the gradient doesn't need to be separated
		if ( weightDecayFlag && ( trainingType == 0 ) && !gradMaxFlag )
			W *= decayTerm;

		// Where gradient doesn't need to be separated
		if ( ( trainingType == 0 ) && !gradMaxFlag )
			// Accumulate beta weight update, see above note
			Waccumulate += ( I *= o_err );

		else // calculate the gradient as a separate structure
		{
			// Calculate the gradient, see above note
			G = ( I *= o_err );

			// See above note in on-line block
			if ( weightDecayFlag )
				G += ( W * decay );

			// Whatever additional algorithm is chosen
			engine( trainingType, ( iteration * nTrain ) + example );

			// For calculating the condition number, convert gradient structure
			//    to a vector (for a logistic object, it already is), then store
			//    this exemplar's gradient in the gradient matrix
			if ( finalFlag )
			{
				pack();
				storeGrads( example );
			}

			// Update the accumulator
			// Methodology equation 2.14: ${\bf g} = (1/N) \sum_{k=1}^N {\bf g}_k$
			Waccumulate += G;
		}

	} // end of loop for exemplars in training set

	// Canonical backprop without separate gradient calculation
	if ( ( trainingType == 0 ) && !gradMaxFlag )
		// Batch/epoch updates weights at the end, *now* multiply by eta
		// Methodology equation 2.13: ${\bf y}_{t+1} = {\bf y}_t - \eta {\bf g}({\bf y}_t)$
		// and $1/N$ in Methodology equation 2.14: ${\bf g} = (1/N) \sum_{k=1}^N {\bf g}_k$
		W -= ( ( Waccumulate *= eta ) / ( double ) nTrain ); // *=, /= for efficiency
	
	else // routines where gradient is calculated separately
	{
		// Set the gradients to the accumulators so that pack() & unpack() work
		// $1/N$ in Methodology equation 2.14: ${\bf g} = (1/N) \sum_{k=1}^N {\bf g}_k$
		G = Waccumulate / ( double ) nTrain;

		// Whatever additional algorithm is chosen
		engine( trainingType, iteration );

		// Update the beta weights
		// Methodology equation 2.13: ${\bf y}_{t+1} = {\bf y}_t - \eta {\bf g}({\bf y}_t)$
		W -= ( G * eta );
	}

	return setError / nTrain; // return the calculated set error
}

// Forward propagates for one input vector in dataset, takes dataset Matrix
//    as first argument, position in dataset as 2nd argument
void Logistic::forward( Matrix< double >& data, unsigned example )
{
	// Get a single row from the input Matrix
	data.row( example, I );

	// Take the dotproduct of the input vector and the beta
	// weights vector, and apply the sigmoidal function to the result
	x = dotprod( I, W ); // note that x is inherited from Network
	o = sigmoidal()( x );
}

// Remove input nodes from this network, takes vector representing
//    which nodes to be removed as argument
void Logistic::removeInputs( const vector< unsigned >& v_in )
{
	vector< unsigned > v = v_in; // make copy, as incoming is const

	// Sort incoming positions vector, necessary for coming application of unique
	sort( v.begin(), v.end() );
	assert ( v.end() == unique( v.begin(), v.end() ) ); // unique check
	
	// Check incoming vector elements in bounds
	assert ( *max_element( v.begin(), v.end() ) < theData.getInput() );

	// Use DataSet::removeInputs to remove inputs in the dataset
	theData.removeInputs( v );

	// Use Model utility function to extract input Matrices
	Model::extractInputMatrices();

	// Resize the vector for single exemplar inputs
	I.resize( theData.getInput() + 1 ); // last is bias

	// If training set is loaded...
	if ( theData.trainLoaded() )
	{
		// ...add biases to the training set input Matrix
		vector< double > in_bias( theData.getNumTrain(), 1 );
		Train = Train.addcol( in_bias ); // efficiency doesn't matter here
	}

	// If test set is loaded...
	if ( theData.testLoaded() )
	{
		// ...add biases to the test set input Matrix
		vector< double > test_bias( theData.getNumTest(), 1 );
		Test = Test.addcol( test_bias ); // efficiency doesn't matter here either
	}

	// Remove weights from the weight vector
	vector< double > Wnew; // vector to hold feature deficient weights
 	vector< bool > Wbool( W.size() ); // indicates if weight should be included
	std::fill( Wbool.begin(), Wbool.end(), true ); // initialize to all true

	unsigned i; // the usual iterator
	for ( i = 0; i < v.size(); i++ ) // cycle through passed vector
		Wbool[ v[ i ] ] = false; // turn specified weight to false to indicate excluded
	for ( i = 0; i < Wbool.size(); i++ ) // cycle through "inclusion" vector
		if ( Wbool[ i ] ) // if it's to be included...
			Wnew.push_back( W[ i ] ); // ...then include it

	W = Wnew; // copy feature deficient weights into weights vector
}


