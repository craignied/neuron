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

	// Condition-number diagnostics from the last reportAccuracy() (via
	//    reportCondNum/computeCondNum); -1 until computed for this model
	double getCondNum() const { return condNum; }
	double getCondMaxEig() const { return condMaxEig; }
	double getCondMinEig() const { return condMinEig; }

	// Mean error over every stride-th test exemplar -- the cheap mid-training
	//    test-error sample for the GUI's realtime graph (mirrors the 1-output
	//    test loop of reportAccuracy, minus the TwoSet writes: sampling must
	//    never invalidate cached statistics). Returns -1 when there is no
	//    1-output test set to sample. stride 1 is the full test error.
	double sampleTestError( unsigned stride );

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
		decay,              // weight-decay strength = 2*lambda = 1/sigma^2 (see the Network ctor)
		regularizer,        // the regularization term lambda, calculated in runHeader method
		decayTerm,	        // 1 - ( eta * decay ), also calculated in runHeader method
		o_errAccumulate,    // stores average output error accumulation over the examplars
					        //    used in innerTrainSet() method
		oldErrorAccumulate, // stores average output error accumulation for previous run
					        //    over the examplars, used in innerTrainSet() method
		deltaError,         // deltaError constant for automatic stepsize selection
		currGradMax,		// current absolute maximum gradient
		gamma,              // gamma constant for automatic stepsize selection
		condNum,            // condition number of the B matrix (-1 until computed)
		condMaxEig,         // maximum eigenvalue of the B matrix (-1 until computed)
		condMinEig;         // minimum eigenvalue of the B matrix (-1 until computed)

	bool weightsSetFlag,       // flag to indicate weights set
		biasFlag,              // flag indicates if bias nodes in network
		batchEpochFlag,        // flag indicates if batch/epoch training
		weightDecayFlag,       // flag indicates if weight decay is used
		automaticStepSizeFlag, // flag indicates if automatic step size algorithm is used
		finalFlag;             // RETIRED 2026-08-01: nothing sets it. It marked
		                       //    the gradient-harvesting call to
		                       //    innerTrainSet() that computeCondNum() used
		                       //    to make; collectGradients() replaced that.
		                       //    Kept only because Network::copy carries it
		                       //    and removing a member is a separate change.

	unsigned trainingType, // training method ( canonical backprop = 0,
		                   // conjugate gradient descent = 1,
		                   // Shanno's algorithm = 2 )
		maxLoops;          // maxLoops constant for automatic stepsize selection

	// Copy utility
	void copy( const Network& rhs );

	// Derive the per-run weight-decay constants (regularizer, decayTerm) from
	//    the CURRENT eta and decay. Training reads both, so this must run for
	//    every run, quiet or not -- see Iterative::prepareRun. It stays a
	//    per-run calculation rather than a one-time constructor assignment
	//    because eta and decay may change between runs.
	virtual void prepareRun();

	// Utility to output Network specific parameters prior to an Iterative run.
	//    REPORTING ONLY: it must not compute anything the fit depends on.
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

	// Fill the grads Matrix with one column per training exemplar, each column
	//    that exemplar's gradient of the objective with respect to the model's
	//    parameters -- the raw material for the B matrix below.
	//
	//    IT MUST NOT TRAIN. Until 2026-08-01 computeCondNum() obtained these by
	//    setting finalFlag and calling innerTrainSet(), which forward
	//    propagates and computes gradients -- and then also runs the optimizer
	//    transformation and UPDATES THE WEIGHTS. Reporting a diagnostic
	//    therefore moved the model: an audible Logistic run ended one gradient
	//    step past the weights whose error it had just reported, while a quiet
	//    run did not. It also disturbed G, stackG, lastG/lastF and the
	//    step-size accumulators, so a continued run diverged from a control.
	//
	//    Returns false when this model does not supply gradients for a
	//    condition number, which is the default: only Logistic reports one, and
	//    a premature generic interface would oblige every network to implement
	//    something nothing calls. computeCondNum() reports "not available"
	//    rather than inventing a number.
	virtual bool collectGradients() { return false; }

	// Compute the condition number of the B matrix from collectGradients()'s
	//    columns, storing condNum/condMaxEig/condMinEig; reportCondNum prints
	//    those stored values.
	//
	//    It preserves the model's FITTED and OPTIMIZER state -- the weights, and
	//    G / stackG / lastG / lastF / the step-size accumulators -- which is the
	//    property that was broken. It does not leave the whole object untouched,
	//    and is not meant to: it deliberately refreshes the diagnostic state
	//    (grads, condNum, condMaxEig, condMinEig), and forward propagation
	//    updates the per-exemplar scratch members (I, o, x) as it always does.
	void computeCondNum();

	// Utility to report out the condition number
	void reportCondNum( ostream& );
};

#endif
