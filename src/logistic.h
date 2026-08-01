// Header for a 1 output node, 1 hidden layer backpropagation network
// with biases

#ifndef LOGISTIC_H
#define LOGISTIC_H

#include "network.h"
#include "matrix.h"
#include "vector_ops.h"
#include "function_defs.h"

class Logistic : public Network {
public:
	Logistic(); // default constructor
	virtual ~Logistic() { } // destructor

	// Copy constructor
	Logistic( const Logistic& rhs );

	// Overloaded = operator
	Logistic& operator= ( const Logistic& rhs );

	// Load DataSet object into Logistic Model
	virtual void setDataSet( DataSet& );

	// Degrees of freedom of this network
	virtual unsigned df(){ return theData.getInput() + 1; } 

	// Outputs a header to ostream describing this Logistic model architecture
	virtual void outputHeader( ostream& );

	// Randomizes initial weights in this Logistic object
	virtual void randomize();

	// Outputs to ostream reporting the accuracy of the Model, adds binary logistic
	//    regression specific footer information
	virtual void reportAccuracy( ostream& );

	// Saves Logistic architecture and weights to a file,
	//    takes filename as ( string ) argument
	virtual bool save( string& );

	// Loads Logistic architecture and weights from a file,
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

	// Wald-test results filled by the last reportAccuracy() call; index nInput
	//    is the y-intercept. getBetaSE returns -1 for any weight whose standard
	//    error was not computable (singular covariance)
	const vector< double >& getBetas() const { return W; }
	const vector< double >& getBetaSE() const { return Bstderr; }
	const vector< double >& getWaldP() const { return WaldP; }

private:

	vector< double > y,       // vector containing dataset outputs
		I,                    // vector for single exemplar inputs
		W,                    // beta weight vector
		G,                    // gradient vector
		Bstderr,              // std errors of beta weights (last reportAccuracy)
		WaldP;                // Wald-test p-values (last reportAccuracy)

	double o_err; // output error term

	// Copy utility
	void copy( const Logistic& rhs );

	// Convert weight gradient structure (vector here) to single vector
	virtual void pack(){ stackG = G; }

	// Convert single vector back to weight gradient structure (vector here)
	virtual void unpack(){ G = stackG; }
};

#endif
