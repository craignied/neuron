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
	unsigned nHidden; // number of hidden nodes
	// ( nH, a stored copy of nHidden - 1, was removed 2026-08-01. It was a
	//   derived value that had to be kept in sync by setHidden and
	//   removeHidden, and copied by copy(), to say what "nHidden - 1" already
	//   says. The two ranged calls that used it now write the domain out:
	//   elements 0 .. nHidden - 1, i.e. every hidden unit except the bias slot. )

	Matrix< double > hW, // hidden weight Matrix
		hWup,            // hidden weight update Matrix
		hG;                 // hidden weight gradient Matrix

	vector< double > y,   // vector containing dataset outputs
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
		explicit WeightSnapshot( const SimpleProp& n ) : hW ( n.hW ), oW ( n.oW ) { }
		void restore( SimpleProp& n ) const { n.hW = hW; n.oW = oW; }
	};
	friend class Network; // reaches WeightSnapshot, and nothing else

	// Copy utility
	void copy( const SimpleProp& rhs );

	// Convert weight gradient structure to single vector
	virtual void pack();

	// Convert single vector back to weight gradient structure
	virtual void unpack();
};

#endif
