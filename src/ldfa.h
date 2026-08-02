// Header file for LDFA, linear discriminant function analysis

#ifndef LDFA_H
#define LDFA_H

#include "dfa.h"

class LDFA : public DFA {
public:
	LDFA() { objType = "LDFA"; } // default constructor
	virtual ~LDFA() { } // destructor

	// Copy constructor
	LDFA( const LDFA& rhs );

	// Overloaded = operator
	LDFA& operator= ( const LDFA& rhs );

	// Outputs to ostream reporting the accuracy of the LDFA Model
	virtual void reportAccuracy( ostream& );

protected:
	// The LINEAR fit: one POOLED covariance across all classes, its inverse,
	//    and the constants. DFA::train calls this once per run.
	virtual void fitDiscriminant();

private:
	Matrix< double > C, // common covariance Matrix
		S;              // inverse of common covariance Matrix

	// Copy utility
	void copy( const LDFA& rhs );
};

#endif
