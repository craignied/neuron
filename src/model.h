// Header file for Model, the most fundamental abstract base class for neUROn2++

#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include "matrix.h"
#include "dataset.h"
#include "function_defs.h"

class Model {
public:
	Model(); // default constructor
	virtual ~Model(); // destructor

	// Copy constructor
	Model( const Model& rhs );

	// Overloaded = operator
	Model& operator= ( const Model& rhs );

	// Type accessor
	string getType() { return objType; }

	// DataSet accessors
	// Load DataSet object into Model
	virtual void setDataSet( DataSet& ) = 0;  // pure virtual
	DataSet& getDataSet() { return theData; } // retrieve from Model

	// Logging to file accessors
	void setHistory( const bool flag ) { historyFlag = flag; } // set history logging
	bool getHistory() { return historyFlag; } // get history logging
	// Set name of history file
	void setHistoryFilename( const string& filename ) { historyFilename = filename; }
	string getHistoryFilename() { return historyFilename; } // get history filename
	void setLastop( const bool flag ) { lastopFlag = flag; } // set last operation logging
	bool getLastop() { return lastopFlag; } // get last operation logging
	// Set name of last operation logging file
	void setLastopFilename( const string& filename ) { lastopFilename = filename; }
	string getLastopFilename() { return lastopFilename; } // get last operation file name

	// Set output error function
	void setLMSerror(); // least mean squared
	void setXEerror(); // X-entropy

	// Outputs a header to ostream describing the architecture of the model
	virtual void outputHeader( ostream& ) = 0; // pure virtual

	// Trains the Model, returns the final error
	virtual double train() = 0; // pure virtual

	// Outputs to ostream reporting the accuracy of the Model
	virtual void reportAccuracy( ostream& ) = 0; // pure virtual

protected:
	DataSet theData; // the primary dataset to be modeled

	Matrix< double > Train, // to hold DataSet training set inputs submatrix
		Test,               // to hold DataSet test set inputs submatrix
		Validation, // to hold DataSet validation set inputs submatrix (Phase 4c)
		TrainOutput, // to hold DataSet training set outputs submatrix
		TestOutput; // to hold DataSet test set outputs submatrix

	bool builtFlag, // flag to indicate if Model formally constructed
		historyFlag, // flag to indicate if operations logged to history file
		lastopFlag, // flag to indicate if last operation logged to file
		errorType; // type of error function: 0 = LMS, 1 = X-entropy

	string objType, // type of model
		historyFilename, // name of history file
		lastopFilename, // name of log file
		errorLabel; // name of the error function

	// Copy utility
	void copy( const Model& rhs );

	// Utility function to extract inputs submatrices from DataSet Matrices
	void extractInputMatrices();

	// Utility function to extract outputs submatrices from DataSet Matrices
	void extractOutputMatrices();

	bool addHistory( ostringstream& ); // utility function to append to history file
};

#endif
