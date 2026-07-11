// Header file for DFA, discriminant function analysis

#ifndef DFA_H
#define DFA_H

#include "model.h"

class DFA : public Model {
public:
	DFA(); // default constructor
	virtual ~DFA(); // destructor

	// Copy constructor
	DFA( const DFA& rhs );

	// Overloaded = operator
	DFA& operator= ( const DFA& rhs );

	// Load DataSet object into DFA Model
	virtual void setDataSet( DataSet& );

	// Outputs a header to ostream describing the architecture of the DFA model
	virtual void outputHeader( ostream& );

	// Trains the DFA Model, returns the final error
	virtual double train() = 0; // pure virtual

	// Outputs to ostream reporting the accuracy of the DFA Model
	virtual void reportAccuracy( ostream& ) = 0; // pure virtual

protected:
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
