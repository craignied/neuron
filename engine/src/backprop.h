// Header for a general backpropagation network

#ifndef BACKPROP_H
#define BACKPROP_H

#include "network.h"
#include "matrix.h"
#include "vector_ops.h"
#include "function_defs.h"

class BackProp : public Network {
public:
	BackProp() { objType = "BackProp"; } // default constructor
	virtual ~BackProp() { } // destructor

	// Copy constructor
	BackProp( const BackProp& rhs );

	// Overloaded = operator
	BackProp& operator= ( const BackProp& rhs );

	// Load DataSet object into BackProp Model
	virtual void setDataSet( DataSet& );

	// Sets hidden layers
	void setHidden( const vector< unsigned >& );

	// Degrees of freedom of this network
	virtual unsigned df();

	// Outputs a header to ostream describing this BackProp model architecture
	virtual void outputHeader( ostream& );

	// Randomizes initial weights in this BackProp object
	virtual void randomize();

	// Saves BackProp architecture and weights to a file,
	//    takes filename as ( string ) argument
	virtual bool save( string& );

	// Loads BackProp architecture and weights from a file,
	//    takes filename as ( string ) argument
	virtual bool load( string& );

	// Trains one iteration through the training set
	virtual double trainSet(); // returns set error

	// Inner training set algorithm for automatic stepsize selection
	virtual double innerTrainSet(); // returns set error

	// Forward propagates for one input vector in dataset
	// Takes dataset Matrix as first argument, position in dataset as 2nd argument
	virtual void forward( Matrix< double >&, const unsigned );

	// Remove input nodes from this network
	virtual void removeInputs( const vector< unsigned >& );

private:
	vector< unsigned > nLayer; // number of nodes on each hidden layer
		
	unsigned nLayers; // number of hidden node layers

	vector< double > in_bias, // input bias vector to add to input Matrix
		I,                    // vector for single exemplar inputs
		y,                    // vector for single exemplar outputs
		o_err;                // output error vector
	
	vector< Matrix< double > > Weights, // vector of weight matrix for all hidden layers and output layer
		WeightsUp, // vector of matrices to hold updated weights one for each weights matrix
		WeightsAccumulate, // accumulator container for hidden and output weights
		Gradient;//Weight gradient matrix

	vector< vector< double > > HOutputs, // vector containing a vector of outputs for each hidden layer
		HErrors, // vector containing a vector of errors for each hidden layer
		vpack; // vectors containing hidden gradients for (un)pack

	// Copy utility
	void copy( const BackProp& rhs );

	// Convert weight gradient structure to single vector
	virtual void pack();

	// Convert single vector back to weight gradient structure
	virtual void unpack();
};

#endif
