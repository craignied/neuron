// Header for a 1 output node, 1 hidden layer backpropagation network
// without biases

#ifndef BAREPROP_H
#define BAREPROP_H

#include "onehidden.h"
#include "matrix.h"
#include "vector_ops.h"
#include "function_defs.h"

// OneHiddenNet holds what this class and SimpleProp carry identically -- the
//    hidden-layer state, randomize(), save(), load(), pack(), the copy utility
//    and the step-size weight snapshot. Everything below is either this model's
//    own mathematics or a width that the ABSENCE of a bias column and bias slot
//    decides; see onehidden.h.
class BareProp : public OneHiddenNet {
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
	//    specify the architecture for a BareProp object. Every width here is
	//    the unbiased one, which is why it is this class's and not
	//    OneHiddenNet's.
	virtual void setHidden( const unsigned );

	// Degrees of freedom of this network
	virtual unsigned df(); 

	// Outputs a header to ostream describing this BareProp model architecture.
	//    Its text is a MODEL FILE FORMAT line that OneHiddenNet::load reads
	//    back, so it is frozen and stays here.
	virtual void outputHeader( ostream& );

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
	// nHidden, hW, hWup, hG, y, I, hO, oW, h_err, oG and o_err are
	//    OneHiddenNet's -- this model and SimpleProp carried them identically.

	// Convert single vector back to weight gradient structure. The offset into
	//    the packed vector is nInput * nHidden -- no bias column -- so this is
	//    not pack()'s shared counterpart.
	virtual void unpack();
};

#endif
