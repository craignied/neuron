// Header for OneHiddenNet, the shared implementation of a 1 output node,
// 1 hidden layer backpropagation network.
//
// SimpleProp (bias nodes on every layer) and BareProp (no bias nodes) carry
// byte-identical state and a handful of methods whose bodies differed by
// nothing but the class name in the signature -- docs/refactor_audit.md section 8.2,
// category A. Those live here once: randomize(), save(), load(), pack(), the
// copy utility, and the automatic step-size weight snapshot.
//
// The training loop and the forward equations live here too (category B, one
// mechanism with bias-dependent dimensions). They are shared with ZERO
// parameterization, not with a flag: innerTrainSet() was ~185 lines written
// out twice whose executable difference was a single line, and that line is
// the same formula once its domain is written as a range -- elements
// 0 .. nHidden - 1 are the hidden units, which is every element of an
// unbiased hO and all but the pinned bias slot of a biased one.
//
// WHAT DELIBERATELY DOES NOT LIVE HERE:
//
//   * SimpleProp and BareProp REMAIN TWO CONCRETE TYPES. typeid dispatch in
//     netclone.cpp, modelfactory::createByTypeName, the first line of every
//     saved model file and the golden transcripts all depend on that. Sharing
//     their bodies is not a format change; merging the classes would be.
//   * df() and outputHeader() stay per-class. They are two published parameter
//     counts and two model-file format lines that load() reads back -- genuine
//     per-class mathematics and frozen bytes, not duplication (rule 7).
//   * THE BIAS ARCHITECTURE. Every width that depends on the bias column or the
//     pinned bias slot -- setHidden(), setDataSet(), unpack(), removeInputs(),
//     and the one statement in each forward() that reads the exemplar and pins
//     the bias slot -- stays in the concrete class, where the biased and
//     unbiased architectures remain legible side by side. Nothing here reads
//     Network::biasFlag: that flag is publicly settable through setBias(), so a
//     shared implementation that consulted it could be made to reshape a live
//     model or change what it writes to disk. The architecture is fixed by the
//     concrete type at construction, and tests/onehidden/check_onehidden.cpp
//     flips the flag on both types and requires nothing observable to move.
//
// load() sizes the object through setHidden(), which is a narrow pure virtual
// below rather than a switch on the type or on the bias flag.

#ifndef ONEHIDDEN_H
#define ONEHIDDEN_H

#include "network.h"
#include "matrix.h"
#include "vector_ops.h"
#include "function_defs.h"

class OneHiddenNet : public Network {
public:
	virtual ~OneHiddenNet() { } // destructor

	// Sets number of hidden nodes, which is all that is required to specify
	//    the architecture of a one-hidden-layer network. Each concrete class
	//    sizes its own structures, because the widths are the bias
	//    architecture (see the note above); load() below calls this.
	virtual void setHidden( const unsigned ) = 0;

	// Randomizes initial weights in this object
	virtual void randomize();

	// Saves architecture and weights to a file,
	//    takes filename as ( string ) argument
	virtual bool save( string& );

	// Loads architecture and weights from a file,
	//    takes filename as ( string ) argument
	virtual bool load( string& );

	// Inner training set algorithm for automatic stepsize selection.
	//    One implementation for both architectures -- the hidden error term is
	//    the same formula once its domain is written as a range. See the note
	//    at that line in onehidden.cpp.
	virtual double innerTrainSet(); // returns set error

protected:
	// Forward propagation from the exemplar ALREADY IN I, with the hidden bias
	//    slot ALREADY PINNED if this model has one. Each concrete forward()
	//    does those two things and calls this. Non-virtual and called directly:
	//    this is a per-exemplar path, which may not acquire indirect calls to
	//    remove source duplication (rule 7).
	void propagate();

	unsigned nHidden; // number of hidden nodes

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
		explicit WeightSnapshot( const OneHiddenNet& n ) : hW ( n.hW ), oW ( n.oW ) { }
		void restore( OneHiddenNet& n ) const { n.hW = hW; n.oW = oW; }
	};
	friend class Network; // reaches WeightSnapshot, and nothing else

	// Copy utility. Every member above is copied; nothing is left to the
	//    default constructor, which a copy constructor delegating to a copy()
	//    utility never runs (see Iterative::copy).
	void copy( const OneHiddenNet& rhs );

	// Convert weight gradient structure to single vector
	virtual void pack();
};

#endif
