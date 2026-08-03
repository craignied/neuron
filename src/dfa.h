// Header file for DFA, discriminant function analysis

#ifndef DFA_H
#define DFA_H

#include "model.h"

// `override` is used consistently throughout this class. It arrived with
//    train() below, where `override final` states that the shared scaffold may
//    not be replaced; clang's -Winconsistent-missing-override then requires the
//    rest, and a zero-warning build is a gate here.
class DFA : public Model {
public:
	DFA(); // default constructor
	virtual ~DFA(); // destructor

	// Copy constructor
	DFA( const DFA& rhs );

	// Overloaded = operator
	DFA& operator= ( const DFA& rhs );

	// Load DataSet object into DFA Model
	void setDataSet( DataSet& ) override;

	// Outputs a header to ostream describing the architecture of the DFA model
	void outputHeader( ostream& ) override;

	// THE SHARED SCAFFOLD of a discriminant run: the streams, the banner, the
	//    header, the Singular refusal, the single reportAccuracy call, the
	//    history and last-operation writes, and the -1. Everything here runs
	//    ONCE per run and was written out twice, differing only in the model's
	//    own name (docs/refactor_audit.md section 13).
	//
	//    It is still a VIRTUAL FUNCTION: Model::train() is pure virtual, so this
	//    overrides it. `final` says the intent -- no subclass may replace the
	//    scaffold -- and `override` says what it is. The polymorphism is live:
	//    cvadapters::dfaProcedure holds a unique_ptr< Model > and calls train()
	//    through it for every cross-validation fold.
	double train() override final;

	// Outputs to ostream reporting the accuracy of the DFA Model. Still PURE
	//    VIRTUAL and still written out in full by each model: its per-exemplar
	//    loops carry the discriminant formulae and the opposite senses (larger
	//    wins for the linear, smaller for the quadratic), and sharing them
	//    would need either per-exemplar dispatch or a comparator parameter.
	//    Neither is permitted; see docs/refactor_audit.md section 13.5.
	void reportAccuracy( ostream& ) override = 0;

protected:
	// THE FIT, and only the fit: covariances, inverses, constants. Runs ONCE
	//    per train(), from inside the scaffold's try block. Each model writes
	//    its own published formulae here -- there is no sign, comparator,
	//    winner flag or formula descriptor anywhere in this class, and the
	//    discriminant mathematics stays where it is read.
	virtual void fitDiscriminant() = 0;

	vector< double > X, // to hold inputs for 1 exemplar
		U0, U1,         // mean vectors for 1-output datasets
		P,              // a priori probabilities for n-output datasets
		K,              // constants for n-output datasets
		d;              // discriminant function results for n-output datasets

	vector< vector< double > > U; // mean vectors for n-output datasets

	vector< unsigned > trainClasses, // outputs for multiple output training sets
		testClasses;                 // outputs for multiple output test sets

	double P0, P1, // a priori probabilities for 1-output datasets
		K0, K1,    // constants for 1-output datasets
		d0, d1;    // discriminant function results for 1-output datasets

	Matrix< double > D0, D1; // Matrices for inputs of 1-output datasets

	vector< Matrix< double > > D; // Matrices for inputs of n-output datasets

	// Copy utility
	void copy( const DFA& rhs );
};

#endif
