// Header file for Network, the abstract base class for neUROn2++ neural networks
//    which use some form of iterative training method.
//    All Networks have weights in some form which may be randomized,
//    a feed forward method, a training method, and a learning rate

#ifndef NETWORK_H
#define NETWORK_H

#include <string>
#include <iostream>
#include <iomanip>

#include "iterative.h"

class Network : public Iterative {
public:
	Network(); // default constructor
	virtual ~Network(); // destructor

	// Copy constructor
	Network( const Network& rhs );

	// Overloaded = operator
	Network& operator= ( const Network& rhs );

	// Load DataSet object into Network Model
	virtual void setDataSet( DataSet& ) = 0; // pure virtual

	void setEta( const double ); // sets the learning rate
	double getEta() { return eta; } // returns the learning rate

	// Sets if bias nodes in network
	void setBias( const bool flag ) { biasFlag = flag; }
	bool getBias() { return biasFlag; } // returns if bias nodes in network

	// Sets if Batch/Epoch training used
	void setBatchEpoch( const bool flag ) { batchEpochFlag = flag; }
	bool getBatchEpoch() { return batchEpochFlag; } // returns if Batch/Epoch used

	// Sets if weight decay used
	void setWeightDecay( const bool flag ) { weightDecayFlag = flag; }
	bool getWeightDecay() { return weightDecayFlag; } // returns if weight decay used

	void setDecay( const double ); // sets weight decay value
	double getDecay() { return decay; } // returns weight decay value

	// Set training type
	void setTrainingType( unsigned typeValue ) { trainingType = typeValue; }
	unsigned getTrainingType() { return trainingType; }

	// Sets/gets if automatic stepsize selection is used in network
	void setAutoStepSize( const bool flag ) { automaticStepSizeFlag = flag; }
	bool getAutoStepSize() { return automaticStepSizeFlag; }

	// Degrees of freedom of a network
	virtual unsigned df() = 0; // pure virtual

	// Outputs a header to ostream describing the architecture of the model
	virtual void outputHeader( ostream& ) = 0; // pure virtual

	// Sets the limit for the random weights -L to +L
	void setRandomLimit( const double L ) { randomLimit = L; } 

	// Randomizes initial weights in a Network object
	virtual void randomize() = 0; // pure virtual
	bool getWeightsSet() { return weightsSetFlag; } // returns if weights set

	// Saves Network object architecture and weights to a file,
	//    takes filename as ( string ) argument
	virtual bool save( string& ) = 0; // pure virtual

	// Loads Network object architecture and weights from a file,
	//    takes filename as ( string ) argument
	virtual bool load( string& ) = 0; // pure virtual

	// Trains one iteration through the training set, Model type dependant
	//    returns set error
	virtual double trainSet() = 0; // pure virtual

	// Inner training set algorithm for automatic stepsize selection
	virtual double innerTrainSet() = 0; // pure virtual

	// Forward propagates for one input vector in dataset
	// Takes dataset Matrix as first argument, position in dataset as 2nd argument
	virtual void forward( Matrix< double >&, const unsigned ) = 0; // pure virtual

	// Outputs to ostream reporting the accuracy of the Model
	virtual void reportAccuracy( ostream& );

	// Outputs classification accuracies to ostream for Iterative table
	virtual void classAccuracy( ostream& );

	// Remove input nodes from this network
	virtual void removeInputs( const vector< unsigned >& ) = 0; // pure virtual

protected:
	vector< double > output, // outputs vector for a multiple output Network
		xs, // arguments in sigmoidal transfer function for multiple output Network
		stackG, // single vector which holds packed gradient
		lastG, // $G_{t-1}$ for shanno/CGD
		lastF; // $F_{t-1}$ for shanno/CGD

	Matrix< double > grads; // to hold gradient matrix (each column is an exemplar
							//    gradient) to be used to compute the B matrix for
							//    calculating the condition number

	double o,               // the output for a Network with only 1 output
		x,                  // argument in sigmoidal transfer function for 1 output Network
		randomLimit,        // the limit for the random weights
		eta,                // learning rate
		decay,              // decay value 2 * lambda for weight decay
		regularizer,        // the regularization term lambda, calculated in runHeader method
		decayTerm,	        // 1 - ( eta * decay ), also calculated in runHeader method
		o_errAccumulate,    // stores average output error accumulation over the examplars
					        //    used in innerTrainSet() method
		oldErrorAccumulate, // stores average output error accumulation for previous run
					        //    over the examplars, used in innerTrainSet() method
		deltaError,         // deltaError constant for automatic stepsize selection
		currGradMax,		// current absolute maximum gradient
		gamma;              // gamma constant for automatic stepsize selection	

	bool weightsSetFlag,       // flag to indicate weights set
		biasFlag,              // flag indicates if bias nodes in network
		batchEpochFlag,        // flag indicates if batch/epoch training
		weightDecayFlag,       // flag indicates if weight decay is used
		automaticStepSizeFlag, // flag indicates if automatic step size algorithm is used
		finalFlag;             // flag identifies a final innerTrainSet()

	unsigned trainingType, // training method ( canonical backprop = 0,
		                   // conjugate gradient descent = 1,
		                   // Shanno's algorithm = 2 )
		maxLoops;          // maxLoops constant for automatic stepsize selection

	// Copy utility
	void copy( const Network& rhs );

	// Utility to output Network specific parameters prior to an Iterative run
	virtual void runHeader( ostream& );

	// Convert weight gradient structure to single vector
	virtual void pack() = 0; // pure virtual

	// Convert single vector back to weight gradient structure
	virtual void unpack() = 0; // pure virtual

	// Returns maximum gradient of a Network object
	virtual double getGradMax();

	// The master training engine
	void engine( unsigned type, unsigned t );

	// Conjugate gradient descent, Golden pp. 221-222
	void CGD( unsigned t );

	// Shanno algorithm, Golden pp. 217-218
	void shanno( unsigned t );

	// Initialize grads matrix, first argument is dimension (size of packed
	//    gradient), second argument is the number of exemplars
	void storeGrads( unsigned, unsigned );

	// Accumulate grads matrix to compute condition number, passed argument
	//    is the exemplar number
	void storeGrads( unsigned );

	// Utility to report out the condition number
	void reportCondNum( ostream& );
};

#endif
