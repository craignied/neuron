// Header for a 1 output node, 1 hidden layer backpropagation network
// with biases

#ifndef SIMPLEPROP_H
#define SIMPLEPROP_H

#include "onehidden.h"
#include "matrix.h"
#include "vector_ops.h"
#include "function_defs.h"

// OneHiddenNet holds what this class and BareProp carry identically -- the
//    hidden-layer state, randomize(), save(), load(), pack(), the copy utility
//    and the step-size weight snapshot. Everything below is either this model's
//    own mathematics or a width that the bias column and the pinned bias slot
//    decide; see onehidden.h.
class SimpleProp : public OneHiddenNet {
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
	//    specify the architecture for a SimpleProp object. Every width here
	//    carries the bias column and the pinned bias slot, which is why it is
	//    this class's and not OneHiddenNet's.
	virtual void setHidden( const unsigned );

	// --- Hidden-layer resizing for OBD sizing (ROADMAP 2 Phase 4) ---------
	//    These operate on a network whose weights are already set. See
	//    src/obd.{h,cpp} for the grow-then-prune driver that uses them.

	// Add 'extra' hidden units as a WARM START: existing units keep their
	//    weights, the new units get small random incoming weights and ZERO
	//    outgoing weights, so forward() is bit-identical immediately after
	//    (the new units contribute exactly 0 to the output) but gradients flow
	//    into them from the first training step. Requires set weights.
	void growHidden( const unsigned extra );

	// Remove the hidden units at the given indices (0..nHidden-1; the bias
	//    pseudo-unit is not addressable). Kept units keep their weights and
	//    order; at least one unit must survive. Removing a unit whose outgoing
	//    weight is zero leaves forward() bit-identical. Requires set weights.
	void removeHidden( const vector< unsigned >& );

	// Saliency of each hidden unit j = |oW[j]| * std(hidden output j over the
	//    training set): a unit is prunable when its output barely varies or
	//    barely reaches the output. Returns one value per hidden unit (the bias
	//    is excluded). Requires a loaded training set and set weights.
	vector< double > hiddenSaliency();

	// Degrees of freedom of this network
	virtual unsigned df(); 

	// Outputs a header to ostream describing this SimpleProp model architecture.
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
	//    OneHiddenNet's -- this model and BareProp carried them identically.
	// ( nH, a stored copy of nHidden - 1, was removed 2026-08-01. It was a
	//   derived value that had to be kept in sync by setHidden and
	//   removeHidden, and copied by copy(), to say what "nHidden - 1" already
	//   says. The two ranged calls that used it now write the domain out:
	//   elements 0 .. nHidden - 1, i.e. every hidden unit except the bias slot. )

	// Convert single vector back to weight gradient structure. The offset into
	//    the packed vector is ( nInput + 1 ) * nHidden -- the bias column --
	//    so this is not pack()'s shared counterpart.
	virtual void unpack();
};

#endif
