// Methods for DataSet, the object which handles dataset entry and manipulation

#include "stdafx.h" // For MSVC, must be first!

#include "dataset.h"
#include "split.h" // ROADMAP 4: the stratified index-level splitter

#include <algorithm>
#include <map>

// Default constructor
DataSet::DataSet() : nInput ( 1 ), nOutput ( 1 ), strataBins ( 4 ),
	threshold ( 0.5 ), inUpperLimit ( 0.9 ), inLowerLimit ( -0.9 ),
	outUpperLimit ( 0.9 ), outLowerLimit ( 0.1 ), discreteFlag ( true ),
	rawLoadedFlag ( false ), trainLoadedFlag ( false ), testLoadedFlag ( false ),
	valLoadedFlag ( false ),
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
	ValSetData = rhs.ValSetData;

	TrainTwoSet = rhs.TrainTwoSet;
	TestTwoSet = rhs.TestTwoSet;

	minima = rhs.minima;
	maxima = rhs.maxima;

	nInput = rhs.nInput;
	nOutput = rhs.nOutput;

	strataBins = rhs.strataBins;
	strataColumns = rhs.strataColumns;
	groupColumns = rhs.groupColumns;

	threshold = rhs.threshold;
	inUpperLimit = rhs.inUpperLimit;
	inLowerLimit = rhs.inLowerLimit;
	outUpperLimit = rhs.outUpperLimit;
	outLowerLimit = rhs.outLowerLimit;

	discreteFlag = rhs.discreteFlag;
	rawLoadedFlag = rhs.rawLoadedFlag;
	trainLoadedFlag = rhs.trainLoadedFlag;
	testLoadedFlag = rhs.testLoadedFlag;
	valLoadedFlag = rhs.valLoadedFlag;
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

	util::set_run_dir( filename ); // log files follow the data

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
		util::screen() << "I'm sorry, but I can't load the Matrix into a raw dataset:"
			<< endl << "The number of columns in the Matrix doesn't match"
			<< endl << "the number of inputs and outputs." << endl;
	
	// If discrete output set, insure all outputs are 0 or 1
	else if ( discreteFlag && !checkDiscrete( inMatrix ) )
		// Inform user that dataset will not be loaded
		util::screen() << "I'm sorry, but I can't load the Matrix into a raw dataset:"
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

	util::set_run_dir( filename ); // log files follow the data

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
		util::screen() << "I'm sorry, but I can't load the Matrix into a training set:"
		<< endl << "The number of columns in the Matrix doesn't match"
		<< endl << "the number of inputs and outputs." << endl;
	
	// If discrete output set, insure all outputs are 0 or 1
	else if ( discreteFlag && !checkDiscrete( inMatrix ) )
		// Inform user that dataset will not be loaded
		util::screen() << "I'm sorry, but I can't load the Matrix into a training set:"
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
		util::screen() << "Error in opening file to save training set!" << endl;
	else
	{
		// Output the Matrix to the file without a header
		savefile << TrainSetData.setHeader( false );

		// Print message to user notifying successful save to file
		util::screen() << "The training set was successfully saved to the file " << filename
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
		util::screen() << "Error in opening file to save scaling factors!" << endl;
	else if ( !minimaxFlag ) // minima and maxima vectors must have been computed
		util::screen() << "Error in saving scaling factors: a training set must first be derived from a raw dataset" << endl;
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
		util::screen() << "Scaling factors were successfully saved to the file " << filename
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
		util::screen() << "I can't construct a TwoSet object from the DataSet training set:"
			<< endl << "there is more than 1 output." << endl;

	else if ( !trainLoadedFlag ) // the training set must have been loaded
		util::screen() << "I can't construct a TwoSet object from the DataSet training set:"
			<< endl << "the training set is not loaded into the DataSet object."
			<< endl;

	else if ( !discreteFlag ) // the output must be discrete
		util::screen() << "I can't construct a TwoSet object from the DataSet training set:"
			<< endl << "the output must be discrete." << endl;

	else // everything checks out, so construct the TwoSet object
	{
		// Construct a data Matrix for the TwoSet object
		Matrix< double > TwoSetMatrix( TrainSetData.rows(), 2 );

		// Populate the "real" column with the training set Matrix's output column
		// Remember the "test" column is filled with garbage!
		TwoSetMatrix.replacecol( 0, TrainSetData.col( TrainSetData.cols() - 1 ) );

		// Load data Matrix into training set TwoSet object
		TrainTwoSet.setMatrix( TwoSetMatrix );

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

	util::set_run_dir( filename ); // log files follow the data

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
		util::screen() << "I'm sorry, but I can't load the Matrix into a test set:"
		<< endl << "The number of columns in the Matrix doesn't match"
		<< endl << "the number of inputs and outputs." << endl;
	
	// If discrete output set, insure all outputs are 0 or 1
	else if ( discreteFlag && !checkDiscrete( inMatrix ) )
		// Inform user that dataset will not be loaded
		util::screen() << "I'm sorry, but I can't load the Matrix into a test set:"
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
		util::screen() << "Error in opening file to save test set!" << endl;
	else
	{
		// Output the Matrix to the file without a header
		savefile << TestSetData.setHeader( false );

		// Print message to user notifying successful save to file
		util::screen() << "The test set was successfully saved to the file " << filename;
		util::screen() << "." << endl;

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
		util::screen() << "I can't construct a TwoSet object from the DataSet test set:"
			<< endl << "there is more than 1 output." << endl;

	else if ( !testLoadedFlag ) // the test set must have been loaded
		util::screen() << "I can't construct a TwoSet object from the DataSet test set:"
			<< endl << "the test set is not loaded into the DataSet object."
			<< endl;

	else if ( !discreteFlag ) // the output must be discrete
		util::screen() << "I can't construct a TwoSet object from the DataSet test set:"
			<< endl << "the output must be discrete." << endl;

	else // everything checks out, so construct the TwoSet object
	{
		// Construct a data Matrix for the TwoSet object
		Matrix< double > TwoSetMatrix( TestSetData.rows(), 2 );

		// Populate the "real" column with the test set Matrix's output column
		// Remember the "test" column is filled with garbage!
		TwoSetMatrix.replacecol( 0, TestSetData.col( TestSetData.cols() - 1 ) );

		// Load data Matrix into test set TwoSet object
		TestTwoSet.setMatrix( TwoSetMatrix );

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
		util::screen() << "I'm sorry, but to set a threshold, the output must be discrete." << endl;

	else
		threshold = x; // set threshold value
}

// Set upper limit to normalize output
void DataSet::setOutUpper( const double x )
{ 
	if ( discreteFlag ) // check to make sure output is nondiscrete
		util::screen() << "I'm sorry, but limits don't apply to a discrete output." << endl;
	
	else	
		outUpperLimit = x;
}

// Set lower limit to normalize output
void DataSet::setOutLower( const double x )
{ 
	if ( discreteFlag ) // check to make sure output is nondiscrete
		util::screen() << "I'm sorry, but limits don't apply to a discrete output." << endl;
	
	else	
		outLowerLimit = x;
}

// Convert the raw dataset into a training set, returns true if successful
bool DataSet::raw2train()
{
	bool success = false; // flag to indicate if operation successful
	
	if ( !rawLoadedFlag ) // check if raw dataset loaded
		util::screen() << "I'm sorry, but I can't convert the raw dataset into a training set:"
		<< endl << "No raw dataset has been loaded." << endl;
	
	else if ( Raw.cols() != ( nInput + nOutput ) ) // check columns in raw dataset
		util::screen() << "I'm sorry, but I can't convert the raw dataset into a training set:"
		<< endl << "The number of columns in the raw dataset doesn't match"
		<< endl << "the number of inputs and outputs." << endl;
	
	// If discrete output set, insure all outputs are 0 or 1
	else if ( discreteFlag && !checkDiscrete( Raw ) )
		util::screen() << "I'm sorry, but I can't convert the raw dataset into a training set:"
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
		util::screen() << "I'm sorry, but I can't convert the raw dataset into a training set:"
			<< endl << "No raw dataset has been loaded." << endl;

	else if ( Raw.cols() != ( nInput + nOutput ) ) // check columns in raw dataset
		util::screen() << "I'm sorry, but I can't convert the raw dataset into a training set:"
			<< endl << "The number of columns in the raw dataset doesn't match"
			<< endl << "the number of inputs and outputs." << endl;

	else if ( nOutput != 1 ) // check 1 output only
		util::screen() << "I'm sorry, but I can't convert the raw dataset into a training set:"
			<< endl << "the method is only coded for 1 output." << endl;

	// nTest must leave at least one training exemplar: nTest == Raw.rows()
	//    empties the training set, and minimax() would then dereference
	//    min_element() on an empty column (undefined). The legacy test was
	//    ">", which permitted exactly that; the message always said "less
	//    than". ( This also rejects a zero-row raw set -- nTest >= 0 is always
	//    true -- so the splitter never divides by n = 0. )
	else if ( nTest >= Raw.rows() ) // bounds check number to place in test set
		util::screen() << "I'm sorry, but I can't convert the raw dataset into a training set:"
			<< endl << "the number to be placed in the test set must be less than"
			<< endl << "the number of examplars in the raw dataset." << endl;

	else if ( !discreteFlag ) // check discrete output
		util::screen() << "I'm sorry, but I can't convert the raw dataset into a training set:"
			<< endl << "the output must be discrete." << endl;
	
	else if ( !checkDiscrete( Raw ) ) // insure all outputs really are 0 or 1
		util::screen() << "I'm sorry, but I can't convert the raw dataset into a training set:"
			<< endl << "Discrete output is specified, but there were outputs that"
			<< endl << "were neither 0 nor 1 in the raw dataset." << endl;

	else // everything checks out, so randomize raw
	{
		unsigned r; // row index

		// Build the binary outcome-label vector from Raw's last column (the
		//    same "outcome == 0" test the legacy partition used).
		vector< unsigned > label( Raw.rows() );
		for ( r = 0; r < Raw.rows(); r++ )
			label[ r ] = ( Raw( r, ( Raw.cols() - 1 ) ) == 0 ) ? 0u : 1u;

		// The chosen row indices for each set, and the class counts the report
		//    below needs. The index-shuffle-then-gather foundation (ROADMAP 4)
		//    makes the split O(n) rather than the old O(n^2) rejection shuffle
		//    and per-row addrow accumulation.
		vector< unsigned > testRows, trainRows;
		unsigned n0 = 0, n1 = 0, n0Test = 0, n1Test = 0;
		vector< unsigned > cellTotal, cellTest; // per-stratum (Phase 2 path)
		unsigned nGroups = 0, groupsInTest = 0; // group partition (Phase 3 path)

		if ( !groupColumns.empty() ) // Phase 3: keep clusters intact (harder test)
		{
			vector< unsigned > group = buildGroupKey();
			nsplit::GroupHoldout h = nsplit::groupHoldout( label, group, nTest );
			testRows = h.test;
			trainRows = h.train;
			nGroups = h.nGroups;
			groupsInTest = h.groupsInTest;

			// Outcome-level counts from the chosen rows (test size only
			//    approximates nTest -- groups are indivisible).
			for ( r = 0; r < label.size(); r++ ) ( label[ r ] == 0 ? n0 : n1 )++;
			for ( r = 0; r < testRows.size(); r++ )
				( label[ testRows[ r ] ] == 0 ? n0Test : n1Test )++;
		}
		else if ( strataColumns.empty() ) // default: stratify on the outcome only
		{
			nsplit::Holdout h = nsplit::stratifiedHoldout( label, nTest );
			testRows = h.test;
			trainRows = h.train;
			n0 = h.n0; n1 = h.n1; n0Test = h.n0Test; n1Test = h.n1Test;
		}
		else // Phase 2: stratify on the outcome x named covariate cells
		{
			vector< unsigned > stratum = buildStrata();
			nsplit::StratHoldout h = nsplit::holdoutByStrata( stratum, nTest );
			testRows = h.test;
			trainRows = h.train;
			cellTotal = h.cellTotal;
			cellTest = h.cellTest;

			// Recover the outcome-level counts from the chosen rows so the
			//    familiar report below is unchanged in shape.
			for ( r = 0; r < label.size(); r++ ) ( label[ r ] == 0 ? n0 : n1 )++;
			for ( r = 0; r < testRows.size(); r++ )
				( label[ testRows[ r ] ] == 0 ? n0Test : n1Test )++;
		}

		unsigned nZerosTest = n0Test, nOnesTest = n1Test; // legacy report names

		// Gather the chosen rows and normalize both sets from the training
		//    scale -- the one trusted materialization path (also used per fold
		//    by cross-validation, ROADMAP 4 Phase 4).
		makeFold( trainRows, testRows );

		// Easier on the eyes
		unsigned nTrain = TrainSetData.rows(), // number of examples in training set
			nTest = TestSetData.rows(); // number of examples in test set

		// Construct the message to the user and the history file
		ostringstream fileStream;

		fileStream << "I've randomized a raw dataset into training and test sets."
			<< endl << "The raw dataset had " << Raw.rows() << " exemplars."
			<< endl << "   The number of 0s in the raw dataset was "
			<< n0 << "."
			<< endl << "   The frequency of 0s in the raw dataset was "
			<< ( ( double ) n0 / ( double ) Raw.rows() )
			<< "." << endl << "   The number of 1s in the raw dataset was "
			<< n1 << "."
			<< endl << "   The frequency of 1s in the raw dataset was "
			<< ( ( double ) n1 / ( double ) Raw.rows() ) << "."
			<< endl << "The training set has " << nTrain << " exemplars."
			<< endl << "   The number of 0s in the training set is "
			<< ( n0 - nZerosTest ) << "."
			<< endl << "   The frequency of 0s in the training set is "
			<< ( ( double ) ( n0 - nZerosTest ) / ( double ) nTrain )
			<< "." << endl << "   The number of 1s in the training set is "
			<< ( n1 - nOnesTest ) << "."
			<< endl << "   The frequency of 1s in the training set is "
			<< ( ( double ) ( n1 - nOnesTest ) / ( double ) nTrain ) << "."
			<< endl << "The test set has " << nTest << " exemplars."
			<< endl << "   The number of 0s in the test set is " << nZerosTest << "."
			<< endl << "   The frequency of 0s in the test set is "
			// A fraction of 0 is a legitimate all-training request: guard the
			//    empty-test denominator so the report reads 0, not nan.
			<< ( nTest > 0 ? ( double ) nZerosTest / ( double ) nTest : 0.0 ) << "."
			<< endl << "   The number of 1s in the test set is " << nOnesTest << "."
			<< endl << "   The frequency of 1s in the test set is "
			<< ( nTest > 0 ? ( double ) nOnesTest / ( double ) nTest : 0.0 )
			<< "." << endl << endl;

		util::screen() << fileStream.str(); // report to user

		// Log to history file
		if ( historyFlag ) // make sure flag for history is set
			addHistory( fileStream ); // append to history file

		// The representativeness diagnostic: the group version when clusters are
		//    kept intact (Phase 3), the stratum version when covariate strata are
		//    in play (Phase 2); an outcome-only split prints the frequencies
		//    above already.
		if ( !groupColumns.empty() )
			groupDiagnostic( testRows, trainRows, nGroups, groupsInTest, nTest );
		else if ( !strataColumns.empty() )
			splitDiagnostic( testRows, trainRows, cellTotal, cellTest );

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
		util::screen() << "I'm sorry, but I can't convert the raw dataset into a training set:"
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
		util::screen() << "I'm sorry, but I can't convert the raw dataset into a training set:"
			<< endl << "No raw dataset has been loaded." << endl;

	else
	{
		// Calculate the number of elements to be placed in the test set
		unsigned nTest = ( unsigned ) ( ratio * ( double ) Raw.rows() );
		
		success = randomize( nTest ); // use previously coded randomize method
	}

	return success; // return if operation successful
}

// Three-way split into training / validation / test (ROADMAP 4 Phase 4c), each
//    stratified on the outcome. The validation set is what model/architecture
//    selection (OBD) monitors, so the test set is untouched until final
//    evaluation. Implemented as two nested stratified holdouts: peel off the
//    test set, then peel the validation set off the remainder.
bool DataSet::randomize3( const unsigned nTest, const unsigned nVal )
{
	bool success = false;

	if ( !rawLoadedFlag )
		util::screen() << "I'm sorry, but I can't make a three-way split:"
			<< endl << "No raw dataset has been loaded." << endl;

	else if ( Raw.cols() != ( nInput + nOutput ) )
		util::screen() << "I'm sorry, but I can't make a three-way split:"
			<< endl << "The number of columns in the raw dataset doesn't match"
			<< endl << "the number of inputs and outputs." << endl;

	else if ( nOutput != 1 )
		util::screen() << "I'm sorry, but I can't make a three-way split:"
			<< endl << "the method is only coded for 1 output." << endl;

	else if ( ( nTest + nVal ) >= Raw.rows() )
		util::screen() << "I'm sorry, but I can't make a three-way split:"
			<< endl << "the test and validation sets together must leave at least"
			<< endl << "one training exemplar." << endl;

	else if ( !discreteFlag )
		util::screen() << "I'm sorry, but I can't make a three-way split:"
			<< endl << "the output must be discrete." << endl;

	else if ( !checkDiscrete( Raw ) )
		util::screen() << "I'm sorry, but I can't make a three-way split:"
			<< endl << "Discrete output is specified, but there were outputs that"
			<< endl << "were neither 0 nor 1 in the raw dataset." << endl;

	else // everything checks out
	{
		unsigned r;
		vector< unsigned > label( Raw.rows() );
		for ( r = 0; r < Raw.rows(); r++ )
			label[ r ] = ( Raw( r, ( Raw.cols() - 1 ) ) == 0 ) ? 0u : 1u;

		// Peel off the test set, then the validation set from the remainder.
		nsplit::Holdout hTest = nsplit::stratifiedHoldout( label, nTest );

		vector< unsigned > restLabel( hTest.train.size() );
		for ( r = 0; r < hTest.train.size(); r++ )
			restLabel[ r ] = label[ hTest.train[ r ] ];
		nsplit::Holdout hVal = nsplit::stratifiedHoldout( restLabel, nVal );

		// Map the validation/train indices (into the remainder) back to Raw rows.
		vector< unsigned > valRows, trainRows;
		valRows.reserve( hVal.test.size() );
		trainRows.reserve( hVal.train.size() );
		for ( r = 0; r < hVal.test.size(); r++ )
			valRows.push_back( hTest.train[ hVal.test[ r ] ] );
		for ( r = 0; r < hVal.train.size(); r++ )
			trainRows.push_back( hTest.train[ hVal.train[ r ] ] );

		makeFold( trainRows, hTest.test, valRows );

		// Event counts per set for the report.
		unsigned nAll = Raw.rows(), n1All = 0, n1Train = 0, n1Val = 0, n1Test = 0;
		for ( r = 0; r < nAll; r++ ) n1All += label[ r ];
		for ( r = 0; r < trainRows.size(); r++ ) n1Train += label[ trainRows[ r ] ];
		for ( r = 0; r < valRows.size(); r++ ) n1Val += label[ valRows[ r ] ];
		for ( r = 0; r < hTest.test.size(); r++ ) n1Test += label[ hTest.test[ r ] ];

		unsigned nTr = TrainSetData.rows(), nVa = ValSetData.rows(),
			nTe = TestSetData.rows();

		ostringstream fileStream;
		fileStream << "I've split a raw dataset into training, validation, and "
			<< "test sets," << endl
			<< "each stratified on the outcome. Selection (e.g. OBD) monitors the"
			<< endl << "validation set; the test set is held out for final "
			<< "evaluation." << endl
			<< "The raw dataset had " << nAll << " exemplars (" << n1All
			<< " events, rate " << ( ( double ) n1All / nAll ) << ")." << endl
			<< "   Training set:   " << nTr << " exemplars (" << n1Train
			<< " events, rate " << ( nTr ? ( double ) n1Train / nTr : 0.0 ) << ")." << endl
			<< "   Validation set: " << nVa << " exemplars (" << n1Val
			<< " events, rate " << ( nVa ? ( double ) n1Val / nVa : 0.0 ) << ")." << endl
			<< "   Test set:       " << nTe << " exemplars (" << n1Test
			<< " events, rate " << ( nTe ? ( double ) n1Test / nTe : 0.0 ) << ")."
			<< endl << endl;

		util::screen() << fileStream.str();

		if ( historyFlag )
			addHistory( fileStream );

		success = true;
	}

	return success;
}

// Three-way split by decimal fractions of the raw dataset (Phase 4c).
bool DataSet::randomize3D( const double testRatio, const double valRatio )
{
	assert( testRatio >= 0 && valRatio >= 0 && ( testRatio + valRatio ) <= 1 );

	if ( !rawLoadedFlag )
	{
		util::screen() << "I'm sorry, but I can't make a three-way split:"
			<< endl << "No raw dataset has been loaded." << endl;
		return false;
	}

	unsigned nTest = ( unsigned ) ( testRatio * ( double ) Raw.rows() );
	unsigned nVal = ( unsigned ) ( valRatio * ( double ) Raw.rows() );

	return randomize3( nTest, nVal );
}

// Materialize a train/test partition from explicit Raw row indices (ROADMAP 4
//    Phase 4): gather each set with the bounds-checked class-layer primitive
//    (rule 4), derive the scaling from the TRAINING set only, and normalize both
//    with it. This is the single trusted path that randomize() and
//    cross-validation both use to build a fold's sets; it sets the loaded flags
//    (an empty test set is flagged not-loaded, so training never underflows on
//    nTest-1). It does NOT decide which model to train -- that is not DataSet's
//    business (rule 6).
void DataSet::makeFold( const vector< unsigned >& trainRows,
	const vector< unsigned >& testRows,
	const vector< unsigned >& valRows )
{
	TestSetData = Raw.includerows( testRows );
	TrainSetData = Raw.includerows( trainRows );

	minimax( TrainSetData );   // scaling from the training set
	normalize( TrainSetData );
	trainLoadedFlag = true;

	normalize( TestSetData );  // test normalized with the training scale
	testLoadedFlag = ( TestSetData.rows() > 0 );

	// Optional validation set (Phase 4c), also scaled from the training set.
	ValSetData = Raw.includerows( valRows );
	if ( ValSetData.rows() > 0 ) normalize( ValSetData );
	valLoadedFlag = ( ValSetData.rows() > 0 );
}

// Build a per-row stratum id from the outcome and the named stratum columns
//    (ROADMAP 4 Phase 2). Each named input column contributes a level: a column
//    with at most strataBins distinct values contributes one level per value
//    (categorical -- e.g. a 0/1 indicator becomes two levels); a column with
//    more distinct values is cut into strataBins equal-count quantile bins by
//    rank. The binary outcome is always a factor. The (outcome, level, ...)
//    tuples are densified into ids 0 .. S-1 in order of first appearance.
vector< unsigned > DataSet::buildStrata() const
{
	unsigned n = Raw.rows();
	unsigned outCol = Raw.cols() - 1;

	// Every row's key starts with its outcome level (0 or 1).
	vector< vector< unsigned > > key( n, vector< unsigned >( 1 ) );
	for ( unsigned r = 0; r < n; r++ )
		key[ r ][ 0 ] = ( Raw( r, outCol ) == 0 ) ? 0u : 1u;

	// Append a level for each named stratum column.
	for ( unsigned k = 0; k < strataColumns.size(); k++ )
	{
		unsigned c = strataColumns[ k ];

		vector< double > val( n );
		for ( unsigned r = 0; r < n; r++ ) val[ r ] = Raw( r, c );

		vector< double > distinct = val;
		sort( distinct.begin(), distinct.end() );
		distinct.erase( unique( distinct.begin(), distinct.end() ), distinct.end() );

		vector< unsigned > level( n );

		if ( distinct.size() <= strataBins ) // categorical: one level per value
		{
			for ( unsigned r = 0; r < n; r++ )
				level[ r ] = ( unsigned ) ( lower_bound( distinct.begin(),
					distinct.end(), val[ r ] ) - distinct.begin() );
		}
		else // continuous: strataBins equal-count quantile bins, by rank
		{
			vector< unsigned > ord( n );
			for ( unsigned r = 0; r < n; r++ ) ord[ r ] = r;
			stable_sort( ord.begin(), ord.end(),
				[ &val ]( unsigned a, unsigned b ) { return val[ a ] < val[ b ]; } );
			for ( unsigned rank = 0; rank < n; rank++ )
				level[ ord[ rank ] ] =
					( unsigned ) ( ( ( unsigned long ) rank * strataBins ) / n );
		}

		for ( unsigned r = 0; r < n; r++ ) key[ r ].push_back( level[ r ] );
	}

	// Densify the tuple keys into stratum ids (first-appearance order).
	map< vector< unsigned >, unsigned > id;
	vector< unsigned > stratum( n );
	for ( unsigned r = 0; r < n; r++ )
	{
		map< vector< unsigned >, unsigned >::iterator it = id.find( key[ r ] );
		if ( it == id.end() )
		{
			unsigned newId = ( unsigned ) id.size();
			id[ key[ r ] ] = newId;
			stratum[ r ] = newId;
		}
		else
			stratum[ r ] = it->second;
	}

	return stratum;
}

// Print the ROADMAP 4 Phase 2 representativeness diagnostic: what the split was
//    stratified on, how many strata, the outcome balance in each set, and each
//    named column's train-vs-test mean (in natural units, read from the
//    un-normalized Raw matrix) so the balance is inspectable rather than
//    assumed. Emitted through util::screen() so the GUI captures it like every
//    other report.
void DataSet::splitDiagnostic( const vector< unsigned >& testRows,
	const vector< unsigned >& trainRows,
	const vector< unsigned >& cellTotal,
	const vector< unsigned >& cellTest )
{
	unsigned outCol = Raw.cols() - 1;

	ostringstream d;
	d << "Representativeness diagnostic" << endl;

	d << "   Stratified on: outcome";
	for ( unsigned k = 0; k < strataColumns.size(); k++ )
		d << ", input column " << ( strataColumns[ k ] + 1 );
	d << "." << endl;

	d << "   Strata (outcome x covariate cells): " << cellTotal.size() << "." << endl;

	// A stratum too small to receive any test exemplar is unrepresented there.
	unsigned thin = 0;
	for ( unsigned s = 0; s < cellTest.size(); s++ )
		if ( cellTotal[ s ] > 0 && cellTest[ s ] == 0 ) thin++;
	if ( thin )
		d << "   " << thin
			<< " stratum(s) too small to place any test exemplar." << endl;

	// Outcome-1 rate in each set (the base rate the split must preserve).
	unsigned trPos = 0, tePos = 0;
	for ( unsigned i = 0; i < trainRows.size(); i++ )
		if ( Raw( trainRows[ i ], outCol ) != 0 ) trPos++;
	for ( unsigned i = 0; i < testRows.size(); i++ )
		if ( Raw( testRows[ i ], outCol ) != 0 ) tePos++;
	d << "   Outcome-1 rate   train "
		<< ( trainRows.empty() ? 0.0 : ( double ) trPos / trainRows.size() )
		<< "   test "
		<< ( testRows.empty() ? 0.0 : ( double ) tePos / testRows.size() )
		<< "." << endl;

	// Each named column's train-vs-test mean, so any imbalance is visible.
	for ( unsigned k = 0; k < strataColumns.size(); k++ )
	{
		unsigned c = strataColumns[ k ];
		double trSum = 0, teSum = 0;
		for ( unsigned i = 0; i < trainRows.size(); i++ ) trSum += Raw( trainRows[ i ], c );
		for ( unsigned i = 0; i < testRows.size(); i++ ) teSum += Raw( testRows[ i ], c );
		d << "   Column " << ( c + 1 ) << " mean    train "
			<< ( trainRows.empty() ? 0.0 : trSum / trainRows.size() )
			<< "   test "
			<< ( testRows.empty() ? 0.0 : teSum / testRows.size() )
			<< "." << endl;
	}
	d << endl;

	util::screen() << d.str(); // report to user

	if ( historyFlag ) // log to history file
		addHistory( d );
}

// Build a per-row group id from EXACT values on the named group columns
//    (ROADMAP 4 Phase 3). Rows sharing identical values on ALL groupColumns get
//    the same id (no binning -- a cluster is an exact match, e.g. the area-SES
//    tuple that identifies a county). Ids are densified in first-appearance
//    order.
vector< unsigned > DataSet::buildGroupKey() const
{
	unsigned n = Raw.rows();

	map< vector< double >, unsigned > id;
	vector< unsigned > group( n );

	for ( unsigned r = 0; r < n; r++ )
	{
		vector< double > key( groupColumns.size() );
		for ( unsigned k = 0; k < groupColumns.size(); k++ )
			key[ k ] = Raw( r, groupColumns[ k ] );

		map< vector< double >, unsigned >::iterator it = id.find( key );
		if ( it == id.end() )
		{
			unsigned newId = ( unsigned ) id.size();
			id[ key ] = newId;
			group[ r ] = newId;
		}
		else
			group[ r ] = it->second;
	}

	return group;
}

// Print the diagnostic for a group-aware split (ROADMAP 4 Phase 3): what it
//    grouped on, how the groups divided, the zero-leakage guarantee, how close
//    the achieved test size is to the request (groups are indivisible), and the
//    outcome balance. Emitted through util::screen() so the GUI captures it.
void DataSet::groupDiagnostic( const vector< unsigned >& testRows,
	const vector< unsigned >& trainRows,
	unsigned nGroups, unsigned groupsInTest, unsigned nTestRequested )
{
	unsigned outCol = Raw.cols() - 1;

	ostringstream d;
	d << "Representativeness diagnostic (group-aware split)" << endl;

	d << "   Grouped on: input column";
	if ( groupColumns.size() > 1 ) d << "s";
	for ( unsigned k = 0; k < groupColumns.size(); k++ )
		d << " " << ( groupColumns[ k ] + 1 );
	d << " (rows sharing these values stay together)." << endl;

	d << "   Groups: " << nGroups << " total -- " << groupsInTest << " in test, "
		<< ( nGroups - groupsInTest ) << " in train; no group is in both "
		<< "(leakage = 0 by construction)." << endl;

	d << "   Test set: " << testRows.size() << " exemplars (requested "
		<< nTestRequested << "; groups are indivisible, so this only "
		<< "approximates the target)." << endl;

	unsigned trPos = 0, tePos = 0;
	for ( unsigned i = 0; i < trainRows.size(); i++ )
		if ( Raw( trainRows[ i ], outCol ) != 0 ) trPos++;
	for ( unsigned i = 0; i < testRows.size(); i++ )
		if ( Raw( testRows[ i ], outCol ) != 0 ) tePos++;
	d << "   Outcome-1 rate   train "
		<< ( trainRows.empty() ? 0.0 : ( double ) trPos / trainRows.size() )
		<< "   test "
		<< ( testRows.empty() ? 0.0 : ( double ) tePos / testRows.size() )
		<< "." << endl << endl;

	util::screen() << d.str(); // report to user

	if ( historyFlag ) // log to history file
		addHistory( d );
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

	ValSetData.clear(); // clear the validation set Matrix (Phase 4c)
	valLoadedFlag = false; // and reset its flag
}

// Method to append ostringstream to history file, takes ostringstream as argument
//    returns true if successful
bool DataSet::addHistory( ostringstream& outputStream )
{
	bool success = false; // flag to indicate if operation successful

	if ( historyFlag ) // only perform if history flag is set
	{
		// Open history file for appended output
		string logPath = util::run_path( historyFilename );
		ofstream historyFile( logPath.c_str(), ios::out | ios::app );

		if ( !historyFile.is_open() ) // test to insure it was opened
			util::screen() << "Error in opening " << logPath << "!" << endl;
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
		util::screen() << "I'm sorry, but I can't output DataSet metrics for the TwoSet object:"
			<< endl << "The output must be discrete" << endl;

	else if ( nOutput != 1 ) // there must only be one output
		util::screen() << "I'm sorry, but I can't output DataSet metrics for the TwoSet object:"
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

			// The Pearson statistic carries no p on purpose: no valid
			//    chi-squared reference exists at the individual level (see
			//    TwoSet::PKX2calc); under a good fit X2 is about n
			outputStream << "Pearson X2 = " << TrainTwoSet.getPearsonX2()
				<< " (n = " << TrainTwoSet.getNumElements()
				<< "; no valid p at the individual level - see Hosmer-Lemeshow)" << endl;
		
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
			
			// See the training-set Pearson note above: a statistic, never a p
			outputStream << "Pearson X2 = " << TestTwoSet.getPearsonX2()
				<< " (n = " << TestTwoSet.getNumElements()
				<< "; no valid p at the individual level - see Hosmer-Lemeshow)" << endl;
		
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

