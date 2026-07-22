// Header file for DataSet, the object which handles dataset entry and manipulation

#ifndef DATASET_H
#define DATASET_H

#include "matrix.h"
#include "vector_ops.h"
#include "twoset.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

using namespace std;

class DataSet {
public:
	DataSet(); // default constructor
	~DataSet(); // destructor
	DataSet( const DataSet& ); // copy constructor
	DataSet& operator= ( const DataSet& ); // = operator

	// Raw dataset accessors and methods
	void loadRaw( string& ); // load file into raw dataset
	bool rawLoaded() { return rawLoadedFlag; } // return flag if raw dataset loaded
	Matrix< double >& getRawMatrix() { return Raw; } // accesses raw dataset
	void setRawMatrix( Matrix< double >& ); // load Matrix into raw dataset

	// Training set accessors and methods
	void loadTrain( string& ); // load file into training set
	bool trainLoaded() { return trainLoadedFlag; } // return flag if training set loaded
	bool saveTrain( string& ); // save training set to file
	bool saveScales( string& ); // save scaling factors to a file
	Matrix< double >& getTrainMatrix() { return TrainSetData; } // accesses training set
	void setTrainMatrix( Matrix< double >& ); // load Matrix into training set

	// Training set TwoSet object accessors
	// Construct a TwoSet object from training set, return true if successful
	bool setTrainTwoSet(); 
	// Return the TwoSet object from training set
	TwoSet& getTrainTwoSet() { return TrainTwoSet; } 
	// Flag if training set TwoSet object loaded
	bool TrainTwoSetLoaded() { return trainTwoSetFlag; }
	// Save the TwoSet object to a file
	bool saveTrainTwoSet( string& );

	// Test set accessors and methods
	void loadTest( string& ); // load file into test set
	bool testLoaded() { return testLoadedFlag; } // return flag if test set loaded
	bool saveTest( string& ); // save test set to file
	Matrix< double >& getTestMatrix() { return TestSetData; } // accesses test set
	void setTestMatrix( Matrix< double >& ); // load Matrix into test set

	// Test set TwoSet object accessors
	// Construct a TwoSet object from test set, return true if successful
	bool setTestTwoSet();
	// Return the TwoSet object from test set
	TwoSet& getTestTwoSet() { return TestTwoSet; }
	// Flag if test set TwoSet object loaded
	bool TestTwoSetLoaded() { return testTwoSetFlag; }
	// Save the TwoSet object to a file
	bool saveTestTwoSet( string& );

	// Validation set accessors (ROADMAP 4 Phase 4c). A validation set is the
	//    held-out set model/architecture selection (OBD) monitors, so the TEST
	//    set stays untouched until final evaluation -- the no-leakage invariant.
	//    It is optional: when none is loaded, the held-out monitor falls back to
	//    the test set (the pre-4c behavior). No TwoSet -- the monitor reads error
	//    directly, it never classifies the validation set.
	bool valLoaded() { return valLoadedFlag; }
	Matrix< double >& getValMatrix() { return ValSetData; }
	unsigned getNumVal() { return ValSetData.rows(); }

	// Accessors for number input, output nodes
	void setInput( const unsigned ); // set the number of input nodes
	void setOutput( const unsigned ); // set the number of output nodes
	unsigned getInput() { return nInput; } // return number input nodes
	unsigned getOutput() { return nOutput; } // return number output nodes

	// Get the number of examples in the training & test sets
	unsigned getNumTrain() { return TrainSetData.rows(); } // training set
	unsigned getNumTest() { return TestSetData.rows(); } // test set

	// Accessors related to normalizing dataset variates
	// Set, get upper limit of input variates
	void setInUpper( const double x ) { inUpperLimit = x; }
	double getInUpper() { return inUpperLimit; }
	// Set, get lower limit of input variates
	void setInLower( const double x ) { inLowerLimit = x; }
	double getInLower() { return inLowerLimit; }
	// Set, get upper limit of output variates
	void setOutUpper( const double );
	double getOutUpper() { return outUpperLimit; }
	// Set, get lower limit of output variates
	void setOutLower( const double );
	double getOutLower() { return outLowerLimit; }

	// Accessors related to discrete output
	void setDiscrete( const bool ); // set discrete output
	bool getDiscrete() { return discreteFlag; } // get state of discrete output
	void setThreshold( const double ); // set threshold value
	double getThreshold() { return threshold; } // get threshold value

	// Dataset manipulation methods
	bool raw2train(); // converts the raw dataset into a training set
	// Randomizes a raw dataset into training and test sets
	bool randomize( const unsigned );
	bool randomize( const unsigned, const unsigned );
	bool randomizeD( const double );

	// Three-way split into training / validation / test, each stratified on the
	//    outcome (ROADMAP 4 Phase 4c). The validation set is what model/
	//    architecture selection (OBD) monitors, so the test set stays untouched
	//    until final evaluation -- the no-leakage invariant. nTest + nVal must
	//    leave at least one training exemplar.
	bool randomize3( const unsigned nTest, const unsigned nVal );
	bool randomize3D( const double testRatio, const double valRatio );

	// Materialize a train/test (and optionally validation) partition from
	//    explicit Raw row indices, scaling every set from the TRAINING set
	//    (ROADMAP 4 Phase 4). randomize() and cross-validation share this one
	//    path; DataSet owns fold materialization but not the choice of model
	//    (rule 6). An empty valRows leaves no validation set (the pre-4c default).
	void makeFold( const vector< unsigned >& trainRows,
		const vector< unsigned >& testRows,
		const vector< unsigned >& valRows = vector< unsigned >() );

	// Stratification accessors (ROADMAP 4 Phase 2). By default the split is
	//    stratified on the outcome only. Naming input data columns here also
	//    stratifies on those covariates: a column with few distinct values
	//    (<= strataBins) contributes one level per value, a continuous column
	//    is cut into strataBins quantile bins. Column indices are 0-based
	//    input-node positions (0 .. nInput-1). An empty list is the default,
	//    outcome-only, behavior. randomize() then apportions the test set across
	//    the outcome x covariate cells and prints a representativeness diagnostic.
	void setStrataColumns( const vector< unsigned >& c ) { strataColumns = c; }
	const vector< unsigned >& getStrataColumns() const { return strataColumns; }
	void setStrataBins( const unsigned b ) { strataBins = b; }
	unsigned getStrataBins() const { return strataBins; }

	// Group-aware splitting accessors (ROADMAP 4 Phase 3). Naming input data
	//    columns here keeps every cluster of rows that share identical values on
	//    ALL those columns together on one side of the split -- so a hospital or
	//    a county is never split across train and test, giving a harder test of
	//    generalization to unseen groups. Exact-match (no binning), 0-based
	//    input-node positions. Empty is the default (no grouping). When set,
	//    randomize() uses the group-aware path in preference to covariate strata.
	void setGroupColumns( const vector< unsigned >& c ) { groupColumns = c; }
	const vector< unsigned >& getGroupColumns() const { return groupColumns; }

	// Logging to history file accessors
	void setHistory( const bool flag ) { historyFlag = flag; } // set history logging
	bool getHistory() { return historyFlag; } // get history logging
	// Set name of history file
	void setHistoryFilename( const string& filename ) { historyFilename = filename; }

	// Append ostringstream to history file
	bool addHistory( ostringstream& );

	// Output TwoSet metrics for a 1-output model, takes an ostream as argument
	void metricsReport( ostream& );

	// Method to remove inputs from a dataset
	void removeInputs( const vector< unsigned >& );

private:
	Matrix< double > Raw, // the raw dataset
		TrainSetData, // the training set
		TestSetData, // the test set
		ValSetData; // the validation set (optional; Phase 4c)

	TwoSet TrainTwoSet, // the TwoSet object for the training set
		TestTwoSet; // the TwoSet object for the test set

	vector< double > minima, // minimum values in a dataset
		maxima; // maximum values in a dataset
	
	unsigned nInput, // number of input nodes
		nOutput, // number of output nodes
		ROCthresholds, // number of ROC thresholds
		strataBins; // quantile bins for a continuous stratum column (Phase 2)

	vector< unsigned > strataColumns, // input columns to stratify on (Phase 2)
		groupColumns; // input columns whose identical-value rows form a group (Phase 3)

	double threshold, // threshold for discrete output
		inUpperLimit, // upper limit for a normalized input variate
		inLowerLimit, // lower limit for a normalized input variate
		outUpperLimit, // upper limit for a normalized output variate
		outLowerLimit; // lower limit for a normalized output variate

	bool discreteFlag, // flag to indicate if output in dataset is discrete (0,1)
		rawLoadedFlag, // flag to indicate if raw dataset loaded
		trainLoadedFlag, // flag to indicate if training set loaded
		testLoadedFlag, // flag to indicate if test set loaded
		valLoadedFlag, // flag to indicate if validation set loaded (Phase 4c)
		trainTwoSetFlag, // flag to indicate if training set TwoSet object loaded
		testTwoSetFlag, // flag to indicate if test set TwoSet object loaded
		minimaxFlag, // flag to indicate if minima and maxima vectors are set 
		historyFlag; // flag to indicate if logging to history file set

	string historyFilename; // name of history file

	void copy( const DataSet& rhs ); // utility function for copy ctor and =

	// Utility method to compute minima and maxima vectors from a data Matrix
	void minimax( const Matrix< double >& );

	// Utility method to insure outputs are discrete
	bool checkDiscrete( const Matrix< double >& );

	// Utility method to normalize a dataset Matrix
	void normalize( Matrix< double >& );

	// Build a per-row stratum id from the outcome and strataColumns (Phase 2)
	vector< unsigned > buildStrata() const;

	// Print the representativeness diagnostic for a stratified split, given the
	//    test/train row-index sets and the per-stratum counts (Phase 2)
	void splitDiagnostic( const vector< unsigned >& testRows,
		const vector< unsigned >& trainRows,
		const vector< unsigned >& cellTotal,
		const vector< unsigned >& cellTest );

	// Build a per-row group id from exact values on groupColumns (Phase 3)
	vector< unsigned > buildGroupKey() const;

	// Print the diagnostic for a group-aware split (Phase 3)
	void groupDiagnostic( const vector< unsigned >& testRows,
		const vector< unsigned >& trainRows,
		unsigned nGroups, unsigned groupsInTest, unsigned nTestRequested );

	// Utility method to clear dataset Matrix members and flags
	void clear();
};

#endif
