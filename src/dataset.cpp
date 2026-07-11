// Methods for DataSet, the object which handles dataset entry and manipulation

#include "stdafx.h" // For MSVC, must be first!

#include "dataset.h"

// Default constructor
DataSet::DataSet() : nInput ( 1 ), nOutput ( 1 ),
	threshold ( 0.5 ), inUpperLimit ( 0.9 ), inLowerLimit ( -0.9 ),
	outUpperLimit ( 0.9 ), outLowerLimit ( 0.1 ), discreteFlag ( true ),
	rawLoadedFlag ( false ), trainLoadedFlag ( false ), testLoadedFlag ( false ),
	trainTwoSetFlag ( false ), testTwoSetFlag ( false ), minimaxFlag( false ),
	historyFlag ( true ), historyFilename ( "neuron.log" ) { }

// Default destructor
DataSet::~DataSet() { }

// Copy constructor
DataSet::DataSet( const DataSet& rhs )
{
	copy( rhs ); // use the copy utility
}

// Overloaded = operator
DataSet& DataSet::operator= ( const DataSet& rhs )
{
	if ( &rhs != this ) // check for self-assignment
		copy( rhs ); // use the copy utility
	
	return *this; // enables A = B = C
}

// Copy utility
void DataSet::copy( const DataSet& rhs )
{
	Raw = rhs.Raw;
	TrainSetData = rhs.TrainSetData;
	TestSetData = rhs.TestSetData;

	TrainTwoSet = rhs.TrainTwoSet;
	TestTwoSet = rhs.TestTwoSet;

	minima = rhs.minima;
	maxima = rhs.maxima;

	nInput = rhs.nInput;
	nOutput = rhs.nOutput;

	threshold = rhs.threshold;
	inUpperLimit = rhs.inUpperLimit;
	inLowerLimit = rhs.inLowerLimit;
	outUpperLimit = rhs.outUpperLimit;
	outLowerLimit = rhs.outLowerLimit;

	discreteFlag = rhs.discreteFlag;
	rawLoadedFlag = rhs.rawLoadedFlag;
	trainLoadedFlag = rhs.trainLoadedFlag;
	testLoadedFlag = rhs.testLoadedFlag;
	trainTwoSetFlag = rhs.trainTwoSetFlag;
	testTwoSetFlag = rhs.testTwoSetFlag;
	minimaxFlag = rhs.minimaxFlag;
	historyFlag = rhs.historyFlag;

	historyFilename = rhs.historyFilename;
}

// Loads a file into the raw dataset Matrix, takes string filename as argument	
void DataSet::loadRaw( string& filename ) // load file into raw dataset
{
	Raw.loadfile( true, filename ); // load the raw dataset Matrix
	
	this->setRawMatrix( Raw ); // use previously coded setRawMatrix method

	// Construct the message to the output stream
	ostringstream fileStream;
	fileStream << "The Matrix was loaded from file " << filename << "."
		<< endl << endl;
	addHistory( fileStream ); // append to history file
}

// Loads a Matrix into the raw dataset Matrix, takes incoming Matrix as argument
void DataSet::setRawMatrix( Matrix< double >& inMatrix )
{
	// Bomb out if either number inputs or number outputs is zero
	assert ( nInput != 0 && nOutput != 0 );
	
	// Bomb out if either dimension of incoming Matrix is zero
	assert ( inMatrix.rows() != 0 && inMatrix.cols() != 0 );

	rawLoadedFlag = false; // initial state of raw data loaded flag

	if ( inMatrix.cols() != ( nInput + nOutput ) ) // check incoming Matrix columns
		// Inform user that dataset will not be loaded
		cout << "I'm sorry, but I can't load the Matrix into a raw dataset:"
			<< endl << "The number of columns in the Matrix doesn't match"
			<< endl << "the number of inputs and outputs." << endl;
	
	// If discrete output set, insure all outputs are 0 or 1
	else if ( discreteFlag && !checkDiscrete( inMatrix ) )
		// Inform user that dataset will not be loaded
		cout << "I'm sorry, but I can't load the Matrix into a raw dataset:"
			<< endl << "Discrete output is specified, but there were outputs that"
			<< endl << "were neither 0 nor 1 in the Matrix." << endl;	
	
	else // Matrix checks out, so complete operation
	{
		// Set the raw dataset Matrix
		Raw = inMatrix;

		rawLoadedFlag = true; // and set the raw data loaded flag
		
		// Loading a raw dataset must clear the training set
		if ( trainLoadedFlag )
		{
			TrainSetData.clear(); // clear the training set Matrix
			trainLoadedFlag = false; // and unset its flag
		}
		
		// Loading a raw dataset must clear the test set
		if ( testLoadedFlag )
		{
			TestSetData.clear(); // clear the test set Matrix
			testLoadedFlag = false; // and unset its flag
		}
		
		// Log to history file
		if ( historyFlag ) // make sure flag for history is set
		{
			// Construct the message to the output stream
			ostringstream fileStream;
			fileStream << "I've loaded a Matrix into a raw dataset."
				<< endl << endl;
			addHistory( fileStream ); // append to history file
		}
	}
}

// Loads a file into the training set Matrix, takes string filename as argument	
void DataSet::loadTrain( string& filename ) // load file into training set
{
	TrainSetData.loadfile( true, filename ); // load the training set Matrix

	this->setTrainMatrix( TrainSetData ); // use previously coded setTrainMatrix

	// Construct the message to the output stream
	ostringstream fileStream;
	fileStream << "The Matrix was loaded from file " << filename << "."
		<< endl << endl;
	addHistory( fileStream ); // append to history file
}

// Loads a Matrix into the training set Matrix, takes incoming Matrix as argument
void DataSet::setTrainMatrix( Matrix< double >& inMatrix )
{
	// Bomb out if either number inputs or number outputs is zero
	assert ( nInput != 0 && nOutput != 0 );
	
	// Bomb out if either dimension of incoming Matrix is zero
	assert ( inMatrix.rows() != 0 && inMatrix.cols() != 0 );
	
	trainLoadedFlag = false; // initial state of training set loaded flag
	
	if ( inMatrix.cols() != ( nInput + nOutput ) ) // check incoming Matrix columns
		// Inform user that dataset will not be loaded
		cout << "I'm sorry, but I can't load the Matrix into a training set:"
		<< endl << "The number of columns in the Matrix doesn't match"
		<< endl << "the number of inputs and outputs." << endl;
	
	// If discrete output set, insure all outputs are 0 or 1
	else if ( discreteFlag && !checkDiscrete( inMatrix ) )
		// Inform user that dataset will not be loaded
		cout << "I'm sorry, but I can't load the Matrix into a training set:"
		<< endl << "Discrete output is specified, but there were outputs that"
		<< endl << "were neither 0 nor 1 in the Matrix." << endl;	
	
	else // Matrix checks out, so complete operation
	{
		// Set the training set Matrix
		TrainSetData = inMatrix;

		trainLoadedFlag = true; // and set the train data loaded flag
		
		// Log to history file
		if ( historyFlag ) // make sure flag for history is set
		{
			// Construct the message to the output stream
			ostringstream fileStream;
			fileStream << "I've loaded a Matrix into a training set."
				<< endl << endl;
			addHistory( fileStream ); // append to history file
		}
	}
}

// Save the training set Matrix to a file, takes filename as ( string ) argument
//    returns boolean flag to indicate if file succesfully saved
bool DataSet::saveTrain( string& filename )
{
	bool success = false; // flag to indicate file successfully saved
	
	// Open the output file to save data, overwrite file if it exists
	ofstream savefile( filename.c_str(), ios::out | ios::trunc );
	
	// Test to insure it was opened
	if ( !savefile.is_open() )
		cout << "Error in opening file to save training set!" << endl;
	else
	{
		// Output the Matrix to the file without a header
		savefile << TrainSetData.setHeader( false );

		// Print message to user notifying successful save to file
		cout << "The training set was successfully saved to the file " << filename
			<< "." << endl;

		savefile.close(); // close output file

		// Log to history file
		if ( historyFlag ) // make sure flag for history is set
		{
			// Construct the message to the output stream
			ostringstream fileStream;
			fileStream << "I've saved a training set to the file " << filename
				<< "." << endl << endl;
			addHistory( fileStream ); // append to history file
		}

		success = true; // set flag to indicate file successfully saved
	}

	return success; // return flag to indicate if file successfully saved
}

// Method to save scaling factors to a file, , takes filename as ( string )
//    argument returns boolean flag to indicate if file succesfully saved
bool DataSet::saveScales( string& filename )
{
	bool success = false; // flag to indicate file successfully saved
	
	// Open the output file to save data, overwrite file if it exists
	ofstream savefile( filename.c_str(), ios::out | ios::trunc );
	
	// Test to insure it was opened
	if ( !savefile.is_open() )
		cout << "Error in opening file to save scaling factors!" << endl;
	else if ( !minimaxFlag ) // minima and maxima vectors must have been computed
		cout << "Error in saving scaling factors: a training set must first be derived from a raw dataset" << endl;
	else
	{
		// Output the explanation of scaling factors to the file
		savefile << "Where x_norm = ( S * ( x - x_min ) ) + lbound," << endl;
		savefile << "lbound = " << inLowerLimit << ", and for each x, S =" << endl;

		// Output the scaling factors
		unsigned i;
		for ( i = 0; i < nInput; i++ )
		{
			double S = ( inUpperLimit - inLowerLimit ) / ( maxima[ i ] - minima[ i ] );
			savefile << S << " ";
		}
		
		// Output the minima
		savefile << endl << "x_min = " << endl;
		for ( i = 0; i < nInput; i++ )
			savefile << minima[ i ] << " ";
		savefile << endl;

		// Print message to user notifying successful save to file
		cout << "Scaling factors were successfully saved to the file " << filename
			<< "." << endl;

		savefile.close(); // close output file

		// Log to history file
		if ( historyFlag ) // make sure flag for history is set
		{
			// Construct the message to the output stream
			ostringstream fileStream;
			fileStream << "I've saved scaling factors to the file " << filename
				<< "." << endl << endl;
			addHistory( fileStream ); // append to history file
		}

		success = true; // set flag to indicate file successfully saved
	}

	return success; // return flag to indicate if file successfully saved
}

// Construct a TwoSet object from training set
bool DataSet::setTrainTwoSet()
{
	trainTwoSetFlag = false; // flag to indicate successful operation
	
	if ( nOutput != 1 ) // must have only 1 output
		cout << "I can't construct a TwoSet object from the DataSet training set:"
			<< endl << "there is more than 1 output." << endl;

	else if ( !trainLoadedFlag ) // the training set must have been loaded
		cout << "I can't construct a TwoSet object from the DataSet training set:"
			<< endl << "the training set is not loaded into the DataSet object."
			<< endl;

	else if ( !discreteFlag ) // the output must be discrete
		cout << "I can't construct a TwoSet object from the DataSet training set:"
			<< endl << "the output must be discrete." << endl;

	else // everything checks out, so construct the TwoSet object
	{
		// Construct a data Matrix for the TwoSet object
		Matrix< double > TwoSetMatrix( TrainSetData.rows(), 2 );

		// Populate the "real" column with the training set Matrix's output column
		// Remember the "test" column is filled with garbage!
		TwoSetMatrix.replacecol( 0, TrainSetData.col( TrainSetData.cols() - 1 ) );

		// Load data Matrix into training set TwoSet object, record and reenter
		//    old number thresholds, as setMatrix will set to number of exemplars
		unsigned n = TrainTwoSet.getTrapThresholds();
		TrainTwoSet.setMatrix( TwoSetMatrix );
		TrainTwoSet.setTrapThresholds( n );

		TrainTwoSet.setThreshold( threshold ); // set the threshold

		trainTwoSetFlag = TrainTwoSet.loaded(); // set the flag to indicate success
	}

	if ( !trainTwoSetFlag ) // if not successful
		TrainTwoSet.getData().clear(); // release the TwoSet object Matrix memory

	return trainTwoSetFlag; // return flag to indicate success
}

// Save the Training TwoSet object to a file, takes string filename as argument
bool DataSet::saveTrainTwoSet( string& filename )
{
	bool success = false; // flag to indicate file successfully saved

	// Use the TwoSet::save() method
	success = this->getTrainTwoSet().save( filename );

	if ( success && historyFlag ) // log to history file if successful
	{
		// Construct the message to the output stream
		ostringstream fileStream;
		fileStream << "I've saved a training TwoSet object to the file " << filename
			<< "." << endl << endl;
		addHistory( fileStream ); // append to history file
	}

	return success; // return flag to indicate if file successfully saved
}

// Loads a file into the test set Matrix, takes string filename as argument	
void DataSet::loadTest( string& filename ) // load file into test set
{
	TestSetData.loadfile( true, filename ); // load the test set Matrix
	
	this->setTestMatrix( TestSetData ); // use previously coded setTestMatrix

	// Construct the message to the output stream
	ostringstream fileStream;
	fileStream << "The Matrix was loaded from file " << filename << "."
		<< endl << endl;
	addHistory( fileStream ); // append to history file
}

// Loads a Matrix into the test set Matrix, takes incoming Matrix as argument
void DataSet::setTestMatrix( Matrix< double >& inMatrix )
{
	// Bomb out if either number inputs or number outputs is zero
	assert ( nInput != 0 && nOutput != 0 );
	
	// Bomb out if either dimension of incoming Matrix is zero
	assert ( inMatrix.rows() != 0 && inMatrix.cols() != 0 );
	
	testLoadedFlag = false; // initial state of test set loaded flag
	
	if ( inMatrix.cols() != ( nInput + nOutput ) ) // check incoming Matrix columns
		// Inform user that dataset will not be loaded
		cout << "I'm sorry, but I can't load the Matrix into a test set:"
		<< endl << "The number of columns in the Matrix doesn't match"
		<< endl << "the number of inputs and outputs." << endl;
	
	// If discrete output set, insure all outputs are 0 or 1
	else if ( discreteFlag && !checkDiscrete( inMatrix ) )
		// Inform user that dataset will not be loaded
		cout << "I'm sorry, but I can't load the Matrix into a test set:"
		<< endl << "Discrete output is specified, but there were outputs that"
		<< endl << "were neither 0 nor 1 in the Matrix." << endl;	
	
	else // Matrix checks out, so complete operation
	{
		// Set the test set Matrix
		TestSetData = inMatrix;
		
		testLoadedFlag = true; // Set the test data loaded flag
		
		// Log to history file
		if ( historyFlag ) // make sure flag for history is set
		{
			// Construct the message to the output stream
			ostringstream fileStream;
			fileStream << "I've loaded a Matrix into a test set."
				<< endl << endl;
			addHistory( fileStream ); // append to history file
		}
	}
}

// Save the test set Matrix to a file, takes filename as ( string ) argument
//    returns boolean flag to indicate if file succesfully saved
bool DataSet::saveTest( string& filename )
{
	bool success = false; // flag to indicate file successfully saved
	
	// Open the output file to save data, overwrite file if it exists
	ofstream savefile( filename.c_str(), ios::out | ios::trunc );
	
	// Test to insure it was opened
	if ( !savefile.is_open() )
		cout << "Error in opening file to save test set!" << endl;
	else
	{
		// Output the Matrix to the file without a header
		savefile << TestSetData.setHeader( false );

		// Print message to user notifying successful save to file
		cout << "The test set was successfully saved to the file " << filename;
		cout << "." << endl;

		savefile.close(); // close output file

		// Log to history file
		if ( historyFlag ) // make sure flag for history is set
		{
			// Construct the message to the output stream
			ostringstream fileStream;
			fileStream << "I've saved a test set to the file " << filename
				<< "." << endl << endl;
			addHistory( fileStream ); // append to history file
		}

		success = true; // set flag to indicate file successfully saved
	}

	return success; // return flag to indicate if file successfully saved
}

// Construct a TwoSet object from test set
bool DataSet::setTestTwoSet()
{
	testTwoSetFlag = false; // flag to indicate successful operation
	
	if ( nOutput != 1 ) // must have only 1 output
		cout << "I can't construct a TwoSet object from the DataSet test set:"
			<< endl << "there is more than 1 output." << endl;

	else if ( !testLoadedFlag ) // the test set must have been loaded
		cout << "I can't construct a TwoSet object from the DataSet test set:"
			<< endl << "the test set is not loaded into the DataSet object."
			<< endl;

	else if ( !discreteFlag ) // the output must be discrete
		cout << "I can't construct a TwoSet object from the DataSet test set:"
			<< endl << "the output must be discrete." << endl;

	else // everything checks out, so construct the TwoSet object
	{
		// Construct a data Matrix for the TwoSet object
		Matrix< double > TwoSetMatrix( TestSetData.rows(), 2 );

		// Populate the "real" column with the test set Matrix's output column
		// Remember the "test" column is filled with garbage!
		TwoSetMatrix.replacecol( 0, TestSetData.col( TestSetData.cols() - 1 ) );

		// Load data Matrix into test set TwoSet object, record and reenter
		//    old number thresholds, as setMatrix will set to number of exemplars
		unsigned n = TestTwoSet.getTrapThresholds();
		TestTwoSet.setMatrix( TwoSetMatrix );
		TestTwoSet.setTrapThresholds( n );

		TestTwoSet.setThreshold( threshold ); // set the threshold

		testTwoSetFlag = TestTwoSet.loaded(); // set the flag to indicate success
	}

	if ( !testTwoSetFlag ) // if not successful
		TestTwoSet.getData().clear(); // release the TwoSet object Matrix memory

	return testTwoSetFlag; // return flag to indicate success
}

// Save the Test TwoSet object to a file, takes string filename as argument
bool DataSet::saveTestTwoSet( string& filename )
{
	bool success = false; // flag to indicate file successfully saved

	// Use the TwoSet::save() method
	success = this->getTestTwoSet().save( filename );

	if ( success && historyFlag ) // log to history file if successful
	{
		// Construct the message to the output stream
		ostringstream fileStream;
		fileStream << "I've saved a test TwoSet object to the file " << filename
			<< "." << endl << endl;
		addHistory( fileStream ); // append to history file
	}

	return success; // return flag to indicate if file successfully saved
}

// Set the number of input nodes
void DataSet::setInput( const unsigned n )
{
	assert ( n != 0 ); // bomb out if number inputs = zero
	nInput = n;

	clear(); // clear all Matrix members and flags
}

// Set the number of output nodes
void DataSet::setOutput( const unsigned n )
{
	assert ( n != 0 ); // bomb out if number inputs = zero
	nOutput = n;

	clear(); // clear all Matrix members and flags
}

// Set discrete output
void DataSet::setDiscrete( const bool flag )
{
	discreteFlag = flag;

	clear(); // clear all Matrix members and flags
}

// Set threshold for discrete output
void DataSet::setThreshold( const double x )
{
	if ( !discreteFlag ) // check to make sure output is discrete
		cout << "I'm sorry, but to set a threshold, the output must be discrete." << endl;

	else
		threshold = x; // set threshold value
}

// Set upper limit to normalize output
void DataSet::setOutUpper( const double x )
{ 
	if ( discreteFlag ) // check to make sure output is nondiscrete
		cout << "I'm sorry, but limits don't apply to a discrete output." << endl;
	
	else	
		outUpperLimit = x;
}

// Set lower limit to normalize output
void DataSet::setOutLower( const double x )
{ 
	if ( discreteFlag ) // check to make sure output is nondiscrete
		cout << "I'm sorry, but limits don't apply to a discrete output." << endl;
	
	else	
		outLowerLimit = x;
}

// Convert the raw dataset into a training set, returns true if successful
bool DataSet::raw2train()
{
	bool success = false; // flag to indicate if operation successful
	
	if ( !rawLoadedFlag ) // check if raw dataset loaded
		cout << "I'm sorry, but I can't convert the raw dataset into a training set:"
		<< endl << "No raw dataset has been loaded." << endl;
	
	else if ( Raw.cols() != ( nInput + nOutput ) ) // check columns in raw dataset
		cout << "I'm sorry, but I can't convert the raw dataset into a training set:"
		<< endl << "The number of columns in the raw dataset doesn't match"
		<< endl << "the number of inputs and outputs." << endl;
	
	// If discrete output set, insure all outputs are 0 or 1
	else if ( discreteFlag && !checkDiscrete( Raw ) )
		cout << "I'm sorry, but I can't convert the raw dataset into a training set:"
		<< endl << "Discrete output is specified, but there were outputs that"
		<< endl << "were neither 0 nor 1 in the raw dataset." << endl;
	
	else // everything checks out, so convert the raw dataset into a training set
	{
		TrainSetData = Raw; // transfer the raw dataset into the training set
		
		minimax( TrainSetData ); // compute the minima and maxima vectors
		
		normalize( TrainSetData ); // normalize the training set
		
		trainLoadedFlag = true; // set the train data loaded flag
		
		// Log to history file
		if ( historyFlag ) // make sure flag for history is set
		{
			// Construct the message to the output stream
			ostringstream fileStream;
			fileStream << "I've converted a raw dataset into a training set."
				<< endl << endl;
			addHistory( fileStream ); // append to history file
		}
		
		success = true; // operation successful
	}
	
	return success; // return flag to indicate if operation successful
}

// Utility method to compute minima and maxima vectors from a data Matrix,
//    takes reference to data Matrix as argument, returns nothing
void DataSet::minimax( const Matrix< double >& dataMatrix )
{
	unsigned c; // column index
	vector< double > column( dataMatrix.rows() ); // column extracted from Matrix

	// Clear the minima and maxima vectors
	minima.clear();
	maxima.clear();

	// Compute minima and maxima vectors from data Matrix
	for ( c = 0; c < dataMatrix.cols(); c++ )
	{
		dataMatrix.col( c, column ); // get a column from the data Matrix
	
		// Append the minimum and maximum values to the respective vectors
		minima.push_back( *min_element( column.begin(), column.end() ) );
		maxima.push_back( *max_element( column.begin(), column.end() ) );
	}

	minimaxFlag = true; // set flag to indicate minima and maxima vectors are set
}

// Utility method to insure that all outputs are 0 or 1, takes reference
//    to data Matrix as argument, returns true if operation was successful
bool DataSet::checkDiscrete( const Matrix< double >& dataMatrix )
{
	bool success = true; // flag to indicate if operation successful
	
	unsigned r, c; // row & column indices
	
	for ( c = nInput; c < ( nInput + nOutput ); c++ ) // iterate through output columns
	{
		for ( r = 0; r < dataMatrix.rows(); r++ ) // iterate through data rows
		{
			// Check if output value 0 or 1
			if ( ( dataMatrix( r, c ) != 0 ) && ( dataMatrix( r, c ) != 1 ) )
			{
				success = false; // if not, set success flag false
				break; // and break out of inner loop
			}
		}
		
		if ( !success ) // if success flag false
			break; // break out of outer loop
	}

	return success; // return if operation successful
}

// Utility method to normalize a data Matrix, takes reference to data Matrix
//    as argument, returns nothing
void DataSet::normalize( Matrix< double >& dataMatrix )
{
	assert( minimaxFlag ); // minima and maxima vectors must have been set
	
	unsigned r, c; // row & column indices
	
	for ( r = 0; r < dataMatrix.rows(); r++ ) // iterate through data rows
	{
		for ( c = 0; c < nInput; c++ ) // normalize input variate columns
			dataMatrix( r, c ) = inLowerLimit + ( ( dataMatrix( r, c ) - minima[ c ] )
			/ ( maxima[ c ] - minima[ c ] ) * ( inUpperLimit - inLowerLimit ) );
		
		if ( !discreteFlag ) // if outputs are specified as nondiscrete
			for ( c = nInput; c < ( nInput + nOutput ); c++ ) // compute output columns
				dataMatrix( r, c ) = outLowerLimit + ( ( dataMatrix( r, c ) - minima[ c ] )
				/ ( maxima[ c ] - minima[ c ] ) * ( outUpperLimit - outLowerLimit ) );
	}
}

// Method to randomize raw dataset into training and test sets, takes the number of
//    examplars to be placed in the test set as argument, returns true if successful
bool DataSet::randomize( const unsigned nTest )
{
	bool success = false; // flag to indicate if operation successful

	if ( !rawLoadedFlag ) // check if raw dataset loaded
		cout << "I'm sorry, but I can't convert the raw dataset into a training set:"
			<< endl << "No raw dataset has been loaded." << endl;

	else if ( Raw.cols() != ( nInput + nOutput ) ) // check columns in raw dataset
		cout << "I'm sorry, but I can't convert the raw dataset into a training set:"
			<< endl << "The number of columns in the raw dataset doesn't match"
			<< endl << "the number of inputs and outputs." << endl;

	else if ( nOutput != 1 ) // check 1 output only
		cout << "I'm sorry, but I can't convert the raw dataset into a training set:"
			<< endl << "the method is only coded for 1 output." << endl;

	else if ( nTest > Raw.rows() ) // bounds check number to place in test set
		cout << "I'm sorry, but I can't convert the raw dataset into a training set:"
			<< endl << "the number to be placed in the test set must be less than"
			<< endl << "the number of examplars in the raw dataset." << endl;

	else if ( !discreteFlag ) // check discrete output
		cout << "I'm sorry, but I can't convert the raw dataset into a training set:"
			<< endl << "the output must be discrete." << endl;
	
	else if ( !checkDiscrete( Raw ) ) // insure all outputs really are 0 or 1
		cout << "I'm sorry, but I can't convert the raw dataset into a training set:"
			<< endl << "Discrete output is specified, but there were outputs that"
			<< endl << "were neither 0 nor 1 in the raw dataset." << endl;

	else // everything checks out, so randomize raw
	{
		unsigned r; // row index

		// Matrices to hold exemplars with only outcome 1, and only outcome 0
		//    note the initial sizes of 0 rows, and number of columns from Raw
		Matrix< double > zeros( 0, Raw.cols() ), ones( 0, Raw.cols() );

		for ( r = 0; r < Raw.rows(); r++ ) // iterate through Raw rows
		{
			if ( Raw( r, ( Raw.cols() - 1 ) ) == 0 ) // if the outcome in Raw is 0
				zeros = zeros.addrow( Raw.row( r ) ); // append the exemplar to zeros
			else // if the outcome in Raw is 1
				ones = ones.addrow( Raw.row( r ) ); // append the exemplar to ones
		}

		// Calculate the number of zeros to be placed in the test set,
		//    round up if necessary
		unsigned nZerosTest = ( unsigned ) ( ( double ) nTest
			* ( ( ( double ) zeros.rows() / ( double ) Raw.rows() ) ) + 0.5 );

		// Calculate the number of ones to be placed in the test set
		unsigned nOnesTest = nTest - nZerosTest;

		vector< unsigned > zerosPos( zeros.rows() ), // random zeros positions vector
			onesPos( ones.rows() ); // random ones positions vector

		// Randomize the positions in the zeros and ones positions vectors
		nvec::random_positions( zerosPos );
		nvec::random_positions( onesPos );

		// Reset the test and training set matrices
		TestSetData.resize( 0, Raw.cols() );
		TrainSetData.resize( 0, Raw.cols() );

		unsigned i; // vector position index

		// Add the random rows from the zeros Matrix to the test set
		for ( i = 0; i < nZerosTest; i++ )
			TestSetData = TestSetData.addrow( zeros.row( zerosPos[ i ] ) );

		// Add the random rows from the zeros Matrix to the training set
		for ( i = nZerosTest; i < zeros.rows(); i++ )
			TrainSetData = TrainSetData.addrow( zeros.row( zerosPos[ i ] ) );

		// Add the random rows from the ones Matrix to the test set
		for ( i = 0; i < nOnesTest; i++ )
			TestSetData = TestSetData.addrow( ones.row( onesPos[ i ] ) );

		// Add the random rows from the ones Matrix to the training set
		for ( i = nOnesTest; i < ones.rows(); i++ )
			TrainSetData = TrainSetData.addrow( ones.row( onesPos[ i ] ) );

		minimax( TrainSetData ); // compute the minima and maxima vectors
		
		normalize( TrainSetData ); // normalize the training set
		
		trainLoadedFlag = true; // set the train data loaded flag

		// Normalize the test set with minima and maxima vectors derived from
		//    the training set
		normalize( TestSetData );

		testLoadedFlag = true; // set the test data loaded flag

		// Easier on the eyes
		unsigned nTrain = TrainSetData.rows(), // number of examples in training set
			nTest = TestSetData.rows(); // number of examples in test set

		// Construct the message to the user and the history file
		ostringstream fileStream;
		
		fileStream << "I've randomized a raw dataset into training and test sets."
			<< endl << "The raw dataset had " << Raw.rows() << " exemplars."
			<< endl << "   The number of 0s in the raw dataset was "
			<< zeros.rows() << "."
			<< endl << "   The frequency of 0s in the raw dataset was "
			<< ( ( double ) zeros.rows() / ( double ) Raw.rows() )
			<< "." << endl << "   The number of 1s in the raw dataset was "
			<< ones.rows() << "."
			<< endl << "   The frequency of 1s in the raw dataset was "
			<< ( ( double ) ones.rows() / ( double ) Raw.rows() ) << "."
			<< endl << "The training set has " << nTrain << " exemplars."
			<< endl << "   The number of 0s in the training set is "
			<< ( zeros.rows() - nZerosTest ) << "."
			<< endl << "   The frequency of 0s in the training set is "
			<< ( ( double ) ( zeros.rows() - nZerosTest ) / ( double ) nTrain )
			<< "." << endl << "   The number of 1s in the training set is "
			<< ( ones.rows() - nOnesTest ) << "."
			<< endl << "   The frequency of 1s in the training set is "
			<< ( ( double ) ( ones.rows() - nOnesTest ) / ( double ) nTrain ) << "."
			<< endl << "The test set has " << nTest << " exemplars."
			<< endl << "   The number of 0s in the test set is " << nZerosTest << "."
			<< endl << "   The frequency of 0s in the test set is "
			<< ( ( double ) nZerosTest / ( double ) nTest ) << "."
			<< endl << "   The number of 1s in the test set is " << nOnesTest << "."
			<< endl << "   The frequency of 1s in the test set is "
			<< ( ( double ) nOnesTest / ( double ) nTest )
			<< "." << endl << endl;

		cout << fileStream.str(); // report to user

		// Log to history file
		if ( historyFlag ) // make sure flag for history is set
			addHistory( fileStream ); // append to history file

		success = true; // operation was a success
	}

	return success; // return if operation successful
}

// Method to randomize raw dataset into training and test sets, takes the numerator
//    and denominator of a fraction representing the portion of examplars to be
//    placed in the test set as arguments, returns true if successful
bool DataSet::randomize( const unsigned numerator, const unsigned denominator )
{
	bool success = false; // flag to indicate if operation successful

	if ( !rawLoadedFlag ) // check if raw dataset loaded
		cout << "I'm sorry, but I can't convert the raw dataset into a training set:"
			<< endl << "No raw dataset has been loaded." << endl;
	
	else
	{
		// Calculate the fraction of elements to be place in the test set
		double ratio = ( double ) numerator / ( double ) denominator;

		success = randomizeD( ratio ); // use previously coded randomizeD method
	}

	return success; // return if operation successful	
}

// Method to randomize raw dataset into training and test sets, takes the fraction
// as a double value from 0 to 1 representing the portion of examplars to be
//    placed in the test set as argument, returns true if successful
bool DataSet::randomizeD( const double ratio )
{
	assert( ratio >= 0 && ratio <= 1 ); // bounds check argument
	
	bool success = false; // flag to indicate if operation successful

	if ( !rawLoadedFlag ) // check if raw dataset loaded
		cout << "I'm sorry, but I can't convert the raw dataset into a training set:"
			<< endl << "No raw dataset has been loaded." << endl;

	else
	{
		// Calculate the number of elements to be placed in the test set
		unsigned nTest = ( unsigned ) ( ratio * ( double ) Raw.rows() );
		
		success = randomize( nTest ); // use previously coded randomize method
	}

	return success; // return if operation successful
}

// Utility function clears DataSet Matrix objects and flags
void DataSet::clear()
{
	Raw.clear(); // clear the raw dataset Matrix
	rawLoadedFlag = false; // and reset its flag

	TrainSetData.clear(); // clear the training set Matrix
	trainLoadedFlag = false; // and reset its flag

	TestSetData.clear(); // clear the test set Matrix
	testLoadedFlag = false; // and reset its flag
}

// Method to append ostringstream to history file, takes ostringstream as argument
//    returns true if successful
bool DataSet::addHistory( ostringstream& outputStream )
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

// Outputs TwoSet metrics for a 1-output model, takes ostream as argument
void DataSet::metricsReport( ostream& outputStream )
{
	if ( !discreteFlag ) // the output must be discrete
		cout << "I'm sorry, but I can't output DataSet metrics for the TwoSet object:"
			<< endl << "The output must be discrete" << endl;

	else if ( nOutput != 1 ) // there must only be one output
		cout << "I'm sorry, but I can't output DataSet metrics for the TwoSet object:"
			<< endl << "There must be only 1 output" << endl;
	
	else
	{
		// DataSet training set TwoSet object must have been loaded
		if ( TrainTwoSet.loaded() )
		{
			// Format the stream for %s
			outputStream << resetiosflags( ios::scientific )
				<< setiosflags( ios::fixed | ios::showpoint ) << setprecision( 1 );

			outputStream << endl << "Training set:" << endl << "-------------" << endl
				<< "Classification accuracy: " << TrainTwoSet.getClassAcc() * 100 << "%" << endl
				<< "Sensitivity: ";
			try // calculate sensitivity, but catch division by zero error
			{ 
				outputStream << TrainTwoSet.getSens() * 100 << "%";
			}
			catch ( TwoSet::DivisionByZero& e )
			{
				outputStream << e.what();
			}
			
			outputStream << endl << "Specificity: ";			
			try // calculate specificity, but catch division by zero error
			{
				outputStream << TrainTwoSet.getSpec() * 100 << "%";
			}
			catch ( TwoSet::DivisionByZero& e )
			{
				outputStream << e.what();
			}
			
			outputStream << endl << "Predictive value positive: ";			
			try // calculate PVP, but catch division by zero error
			{ 
				outputStream << TrainTwoSet.getPVP() * 100 << "%";
			}
			catch ( TwoSet::DivisionByZero& e )
			{
				outputStream << e.what();
			}

			outputStream << endl << "Predictive value negative: ";			
			try // calculate PVN, but catch division by zero error
			{
				outputStream << TrainTwoSet.getPVN() * 100 << "%";
			}
			catch ( TwoSet::DivisionByZero& e )
			{
				outputStream << e.what();
			}

			outputStream << endl;

			// Output classification table for training set
			outputStream << "Classification table for training set:" << endl;
			TrainTwoSet.ClassTable( outputStream );

			// Output training set ROC area
			TrainTwoSet.ROCarea( outputStream );

			// Output training set Kolmogorov-Smirnov test results
			// Craig Niederberger modified 3/12/2009 to catch exceptions
			try
			{
				outputStream << "Kolmogorov-Smirnov goodness of fit D = " <<
					TrainTwoSet.getKSD() << ", p = " << TrainTwoSet.getKSP() << endl;
			}
			catch ( TwoSet::twoSetErr& e )
			{
				outputStream << "Could not calculate Kolmogorov-Smirnov goodness of fit: " << e.what() << endl;
			}	

			// Output training set Pearson's Chi-Square test results Hui Liu added 08/15/2004			
			outputStream << "Pearson's Chi-Square goodness of fit p = " << TrainTwoSet.getPearsonX2() << endl;
		
			// Output training set Hosmer-Lemeshow test results Hui Liu added 08/16/2004
			// Craig Niederberger modified 3/12/2009 to catch exceptions
			try
			{
				outputStream << "Hosmer-Lemeshow goodness of fit p = " << TrainTwoSet.getHLX2() << endl;
			}
			catch ( TwoSet::twoSetErr& e )
			{
				outputStream << "Could not calculate Hosmer-Lemeshow goodness of fit: " << e.what() << endl;
			}			
		}

		// Do the same if the DataSet test set TwoSet object was loaded
		if ( TestTwoSet.loaded() )
		{
			// Format the stream for %s
			outputStream << resetiosflags( ios::scientific )
				<< setiosflags( ios::fixed | ios::showpoint ) << setprecision( 1 );

			outputStream << endl << "Test set:" << endl << "---------" << endl
				<< "Classification accuracy: " << TestTwoSet.getClassAcc() * 100 << "%" << endl
				<< "Sensitivity: ";
			try // calculate sensitivity, but catch division by zero error
			{ 
				outputStream << TestTwoSet.getSens() * 100 << "%";
			}
			catch ( TwoSet::DivisionByZero& e )
			{
				outputStream << e.what();
			}
			
			outputStream << endl << "Specificity: ";			
			try // calculate specificity, but catch division by zero error
			{
				outputStream << TestTwoSet.getSpec() * 100 << "%";
			}
			catch ( TwoSet::DivisionByZero& e )
			{
				outputStream << e.what();
			}
			
			outputStream << endl << "Predictive value positive: ";			
			try // calculate PVP, but catch division by zero error
			{ 
				outputStream << TestTwoSet.getPVP() * 100 << "%";
			}
			catch ( TwoSet::DivisionByZero& e )
			{
				outputStream << e.what();
			}

			outputStream << endl << "Predictive value negative: ";			
			try // calculate PVN, but catch division by zero error
			{
				outputStream << TestTwoSet.getPVN() * 100 << "%";
			}
			catch ( TwoSet::DivisionByZero& e )
			{
				outputStream << e.what();
			}

			outputStream << endl;

			// Output classification table for test set
			outputStream << "Classification table for test set:" << endl;
			TestTwoSet.ClassTable( outputStream );

			// Output test set ROC area
			TestTwoSet.ROCarea( outputStream );

			// Output test set Kolmogorov-Smirnov test results
			// Craig Niederberger modified 3/12/2009 to catch exceptions
			try
			{
				outputStream << "Kolmogorov-Smirnov goodness of fit D = " <<
					TestTwoSet.getKSD() << ", p = " << TestTwoSet.getKSP() << endl;
			}
			catch ( TwoSet::twoSetErr& e )
			{
				outputStream << "Could not calculate Kolmogorov-Smirnov goodness of fit: " << e.what() << endl;
			}
			
		    // Output test set Pearson's Chi-Square test results     Hui Liu added 08/15/2004
			outputStream << "Pearson's Chi-Square goodness of fit p = " << TestTwoSet.getPearsonX2() << endl;
		
			// Output test set Hosmer-Lemeshow test results Hui Liu added 08/16/2004
			// Craig Niederberger modified 3/12/2009 to catch exceptions
			try
			{
				outputStream << "Hosmer-Lemeshow goodness of fit p = " << TestTwoSet.getHLX2() << endl;
			}
			catch ( TwoSet::twoSetErr& e )
			{
				outputStream << "Could not calculate Hosmer-Lemeshow goodness of fit: " << e.what() << endl;
			}	
		}
	}
}

// Method to remove inputs from a dataset
void DataSet::removeInputs( const vector< unsigned >& v )
{
	// Check incoming vector elements in bounds	
	assert ( *max_element( v.begin(), v.end() ) < nInput );

	nInput -= v.size(); // reset number of inputs member

	// Exclude columns from appropriate datasets if loaded
	if ( rawLoadedFlag )
		Raw = Raw.excludecols( v );

	if ( trainLoadedFlag )
		TrainSetData = TrainSetData.excludecols( v );

	if ( testLoadedFlag )
		TestSetData = TestSetData.excludecols( v );
}

