// Header for a 1 output node, 1 hidden layer backpropagation network
// with biases

#ifndef SIMPLEPROP_H
#define SIMPLEPROP_H

#include "network.h"
#include "matrix.h"
#include "vector_ops.h"
#include "function_defs.h"

class SimpleProp : public Network {
public:
	SimpleProp(); // default constructor
	virtual ~SimpleProp() { } // destructor

	// Copy constructor
	SimpleProp( const SimpleProp& rhs );

	// Overloaded = operator
	SimpleProp& operator= ( const SimpleProp& rhs );

	// Load DataSet object into SimpleProp Model
	virtual void setDataSet( DataSet& );

	// Sets number of hidden nodes, which is all that is required to
	//    specify the architecture for a SimpleProp object
	void setHidden( const unsigned );

	// Degrees of freedom of this network
	virtual unsigned df(); 

	// Outputs a header to ostream describing this SimpleProp model architecture
	virtual void outputHeader( ostream& );

	// Randomizes initial weights in this SimpleProp object
	virtual void randomize();

	// Saves SimpleProp architecture and weights to a file,
	//    takes filename as ( string ) argument
	virtual bool save( string& );

	// Loads SimpleProp architecture and weights from a file,
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
	unsigned nHidden, // number of hidden nodes
		nH; // will be defined as nHidden - 1 for ease of indexing with biases,
		    // because the last element of some vectors is a bias, will apply
		    // certain operations only to elements 0 to nHidden - 1

	Matrix< double > hW, // hidden weight Matrix
		hWup,            // hidden weight update Matrix
		hG;                 // hidden weight gradient Matrix

	vector< double > in_bias, // input bias vector to add to input Matrix
		y,                    // vector containing dataset outputs
		I,                    // vector for single exemplar inputs
		hO,                   // hidden output vector
		oW,                   // output weight vector
		h_err,                // hidden error vector
		oG;                   // output weight gradient vector

	double o_err; // output error term

	// Copy utility
	void copy( const SimpleProp& rhs );

	// Convert weight gradient structure to single vector
	virtual void pack();

	// Convert single vector back to weight gradient structure
	virtual void unpack();
};

#endif
