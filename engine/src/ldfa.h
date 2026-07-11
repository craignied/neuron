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

	// Trains the LDFA Model
	virtual double train();

	// Outputs to ostream reporting the accuracy of the LDFA Model
	virtual void reportAccuracy( ostream& );

private:
	Matrix< double > C, // common covariance Matrix
		S;              // inverse of common covariance Matrix

	// Copy utility
	void copy( const LDFA& rhs );
};

#endif
