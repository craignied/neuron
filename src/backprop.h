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

protected:
	// READ-ONLY OBSERVATION OF THE FITTED PARAMETERS, and nothing else.
	//
	//    Every other Network exposes its weights to a subclass already:
	//    OneHiddenNet keeps hW and oW protected, and Logistic publishes W
	//    through getBetas(). BackProp was the one model whose parameters no
	//    derived class could read, so a benchmark subclass could not compute a
	//    parameter-state identity for it -- and a comparison whose arms cannot
	//    be shown to start from the same weights is not a comparison
	//    (tests/optimizer/README.md).
	//
	//    DELIBERATELY NARROW. It returns the AUTHORITATIVE weights by const
	//    reference and reaches nothing else: WeightsUp, WeightsAccumulate,
	//    Gradient and vpack are training workspace, not parameters, and stay
	//    private. Widening the whole private section instead would have exposed
	//    all four as mutable state to every future subclass, which is a larger
	//    promise than anything here needs.
	//
	//    Costs nothing and changes nothing: const, non-virtual, inline, no
	//    allocation, no copy, and no production caller. It cannot appear in a
	//    hot loop because nothing in src/ calls it (rule 7).
	const vector< Matrix< double > >& weightMatrices() const { return Weights; }

	// --- The packed parameter boundary (see network.h) --------------------
	//
	//    This model's layout is its vector-of-Matrix: Weights[0] flattened row
	//    by row, then Weights[1], and so on through the output layer -- the
	//    same order pack() uses for Gradient, so weights and gradient are one
	//    layout.
	virtual unsigned packedSize() const;
	virtual void packWeights( vector< double >& destination ) const;
	virtual void unpackWeights( const vector< double >& source );
	virtual double batchObjectiveGradient( vector< double >& packedRawGradient );

	// THE AUTHORITATIVE BATCH PASS: one traversal at the currently installed
	//    weights, returning the mean objective and leaving Gradient holding the
	//    RAW mean gradient. Non-virtual, so the legacy path pays no indirect
	//    call and no packing (rule 7). See onehidden.cpp for the shape.
	double batchGradient();

	// The per-exemplar prologue shared by every path here. setError is added to
	//    in place so the arithmetic is not reassociated -- see onehidden.h.
	void exemplarErrorTerms( const unsigned example, double& setError );

private:
	vector< unsigned > nLayer; // number of nodes on each hidden layer
		
	unsigned nLayers; // number of hidden node layers

	vector< double > I,   // vector for single exemplar inputs
		y,                    // vector for single exemplar outputs
		o_err;                // output error vector
	
	vector< Matrix< double > > Weights, // vector of weight matrix for all hidden layers and output layer
		WeightsUp, // vector of matrices to hold updated weights one for each weights matrix
		WeightsAccumulate, // accumulator container for hidden and output weights
		Gradient;//Weight gradient matrix

	vector< vector< double > > HOutputs, // vector containing a vector of outputs for each hidden layer
		HErrors, // vector containing a vector of errors for each hidden layer
		vpack; // vectors containing hidden gradients for (un)pack

	// The weights Network::searchStepSize puts back after its trial passes.
	//    Constructed as a LOCAL of the search, so this model never carries a
	//    second copy of its weights between calls, and nothing is copied at all
	//    when the search is off. Restoration is explicit rather than a
	//    destructor: automatic rollback would change what happens when
	//    innerTrainSet() throws, which is a separate question.
	struct WeightSnapshot {
		vector< Matrix< double > > Weights;
		explicit WeightSnapshot( const BackProp& n ) : Weights ( n.Weights ) { }
		void restore( BackProp& n ) const { n.Weights = Weights; }
	};
	friend class Network; // reaches WeightSnapshot, and nothing else

	// Copy utility
	void copy( const BackProp& rhs );

	// Convert weight gradient structure to single vector
	virtual void pack();

	// Convert single vector back to weight gradient structure
	virtual void unpack();
};

#endif
