// Header file for QDFA, quadratic discriminant function analysis

#ifndef QDFA_H
#define QDFA_H

#include "dfa.h"

class QDFA : public DFA {
public:
	QDFA() { objType = "QDFA"; } // default constructor
	virtual ~QDFA() { } // destructor

	// Copy constructor
	QDFA( const QDFA& rhs );

	// Overloaded = operator
	QDFA& operator= ( const QDFA& rhs );

	// Trains the QDFA Model
	virtual double train();

	// Outputs to ostream reporting the accuracy of the QDFA Model
	virtual void reportAccuracy( ostream& );

private:
	Matrix< double > C0, C1, // covariance Matrices for 1-output datasets
		S0, S1;              // covariance Matrix inverses for 1-output datasets

	vector< Matrix< double > > C, // covariance Matrices for n-output datasets
		S;                        // covariance Matrices for n-output datasets

	double Det0, Det1; // covariance Matrix determinants for 1-output datasets

	vector< double > Det; // covariance Matrix determinants for n-output datasets

	// Copy utility
	void copy( const QDFA& rhs );
};

#endif
