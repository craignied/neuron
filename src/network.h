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
		condNum,            // condition number of X'VX (-1 until computed)
		condMaxEig,         // largest absolute eigenvalue of X'VX (-1 until computed)
		condMinEig;         // smallest absolute eigenvalue of X'VX (-1 until computed)

	bool weightsSetFlag,       // flag to indicate weights set
		biasFlag,              // flag indicates if bias nodes in network
		batchEpochFlag,        // flag indicates if batch/epoch training
		weightDecayFlag,       // flag indicates if weight decay is used
		automaticStepSizeFlag; // flag indicates if automatic step size algorithm is used

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

	// Store the eigenvalue diagnostics of a SYMMETRIC matrix: its largest and
	//    smallest absolute eigenvalues and their ratio, the condition number.
	//    The caller supplies the matrix, because what "the condition number"
	//    means is the caller's statistical decision, not this class's.
	//
	//    Logistic passes the observed Fisher information X'VX -- the curvature
	//    of the UNPENALIZED log likelihood, which is what a design-conditioning
	//    / collinearity diagnostic must be measured on. Until 2026-08-01 this
	//    class instead built an outer product of per-exemplar gradients, and
	//    obtained them by running a training iteration; the gradient harvest is
	//    gone with it. See docs/tex and docs/refactor_audit.md.
	void conditionOf( const Matrix< double >& symmetric );

	// THE AUTOMATIC STEP-SIZE SEARCH -- the one implementation, for every model
	//    that has one. It was the same forty lines at the top of
	//    SimpleProp::trainSet(), BareProp::trainSet(), BackProp::trainSet() and
	//    Logistic::trainSet(), differing in exactly two things:
	//
	//    THE GUARD, which the CALLER passes, because it is genuinely not the
	//       same. Logistic searches whenever automaticStepSizeFlag is set; the
	//       three feed-forward nets require batchEpochFlag as well. That
	//       asymmetry stays visible at each call site rather than hiding behind
	//       a virtual policy method -- a shared guard would silently stop
	//       Logistic searching, and no interface can reach that combination to
	//       notice (tests/network/check_autostep.cpp is the only thing that
	//       would).
	//    THE WEIGHTS, which the model owns: hW/oW, Weights, or W. Each model
	//       declares a private WeightSnapshot, and this constructs one as a
	//       LOCAL of the search block. So no model carries a second copy of its
	//       weights between calls -- which for BackProp would be a whole extra
	//       vector< Matrix > -- and with the search off nothing is copied at
	//       all: trainSet() is then exactly one innerTrainSet().
	//
	//    Network keeps lastG and lastF, which are ITS members and are what CGD
	//    and Shanno carry between iterations. Nothing else is restored: eta,
	//    the gradients, the accumulators and currGradMax are all left as the
	//    trial passes left them, exactly as before.
	//
	//    A template, not std::function or a virtual hook: this is called once
	//    per training iteration and must not add allocation or indirect calls
	//    to it. innerTrainSet() below is still a virtual call on this, so a
	//    subclass that counts passes counts the production dispatch.
	template < class MODEL >
	double searchStepSize( MODEL& model, const bool search );

	// Utility to report out the condition number
	void reportCondNum( ostream& );
};

// The search itself. Read it against the manual's step-size section: the shape
//    is a snapshot, a loop that scales eta by gamma while the error keeps
//    improving, a restoration, and then one real pass.
template < class MODEL >
double Network::searchStepSize( MODEL& model, const bool search )
{
	if ( search )
	{
		// Buffer the weights -- a local, alive only for this search
		typename MODEL::WeightSnapshot weights( model );

		// Also buffer lastG and lastF for conjugate gradient descent / Shanno's
		vector< double > lastGBuffer = lastG, lastFBuffer = lastF;

		// Initialize conditions for automatic stepsize selection
		unsigned loopCounter = 1;
		double newError = 1.0; // current calculated error
		double oldError; // previous calculated error
		double ErrorDifference = 1.0; // initially set to 1 to pass the while condition first time
		eta = 1.0; // learning rate
		//    Note that eta is reset ABOVE the loop, so it moves off whatever the
		//    caller set even when the condition below is false and no trial pass
		//    runs at all (a deltaError above 1.0 does exactly that).

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
		// Retrieve the saved weights, and the optimizer's history with them.
		//    Restoration is EXPLICIT and not a destructor: an exception out of
		//    innerTrainSet() left trial weights installed before this extraction
		//    and still does, so the behavior is unchanged. Making it automatic
		//    would be a correctness change and belongs in its own commit.
		weights.restore( model );
		lastG = lastGBuffer;
		lastF = lastFBuffer;
	}

	// Now actual update of weights takes place
	double returnError;
	returnError = innerTrainSet();

	// Set the new error as previous error
	if ( search )
		oldErrorAccumulate = o_errAccumulate;

	// Return the set error
	return returnError;
}

#endif
