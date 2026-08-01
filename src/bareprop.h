// Header for a 1 output node, 1 hidden layer backpropagation network
// without biases

#ifndef BAREPROP_H
#define BAREPROP_H

#include "network.h"
#include "matrix.h"
#include "vector_ops.h"
#include "function_defs.h"

class BareProp : public Network {
public:
	BareProp(); // default constructor
	virtual ~BareProp() { } // destructor

	// Copy constructor
	BareProp( const BareProp& rhs );

	// Overloaded = operator
	BareProp& operator= ( const BareProp& rhs );

	// Load DataSet object into BareProp Model
	virtual void setDataSet( DataSet& ); // pure virtual

	// Sets number of hidden nodes, which is all that is required to
	//    specify the architecture for a BareProp object
	void setHidden( const unsigned );

	// Degrees of freedom of this network
	virtual unsigned df(); 

	// Outputs a header to ostream describing this BareProp model architecture
	virtual void outputHeader( ostream& );

	// Randomizes initial weights in this BareProp object
	virtual void randomize();

	// Saves BareProp architecture and weights to a file,
	//    takes filename as ( string ) argument
	virtual bool save( string& );

	// Loads BareProp architecture and weights from a file,
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
	unsigned nHidden; // number of hidden nodes

	Matrix< double > hW,    // hidden weight Matrix
		hWup,               // hidden weight update Matrix
		hG;                 // hidden weight gradient Matrix

	vector< double > y,       // vector containing dataset outputs
		I,                    // vector for single exemplar inputs
		hO,                   // hidden output vector
		oW,                   // output weight vector
		h_err,                // hidden error vector
		oG;                   // output weight gradient vector

	double o_err; // output error term

	// The weights Network::searchStepSize puts back after its trial passes.
	//    Constructed as a LOCAL of the search, so this model never carries a
	//    second copy of its weights between calls, and nothing is copied at all
	//    when the search is off. Restoration is explicit rather than a
	//    destructor: automatic rollback would change what happens when
	//    innerTrainSet() throws, which is a separate question.
	struct WeightSnapshot {
		Matrix< double > hW;
		vector< double > oW;
		explicit WeightSnapshot( const BareProp& n ) : hW ( n.hW ), oW ( n.oW ) { }
		void restore( BareProp& n ) const { n.hW = hW; n.oW = oW; }
	};
	friend class Network; // reaches WeightSnapshot, and nothing else

	// Copy utility
	void copy( const BareProp& rhs );

	// Convert weight gradient structure to single vector
	virtual void pack();

	// Convert single vector back to weight gradient structure
	virtual void unpack();
};

#endif
