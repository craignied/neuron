// Header file for RegressNet, an object which performs stepwise regression on neural networks

#ifndef REGRESSNET_H
#define REGRESSNET_H

// Because, apparently, vector< vector< unsigned > > symbol names exceeds 255
//    characters in length, and MSCV will error with that (ugly!)
#ifdef _MSC_VER	// MSVC
#pragma warning (disable: 4786)
#endif

#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <memory>
#include <stdexcept>

using namespace std;

#include "network.h"
#include "utility.h"

class RegressNet {
public:
	RegressNet(); // default constructor
	~RegressNet() { } // destructor

	// Copy constructor
	RegressNet( const RegressNet& rhs );

	// Overloaded = operator
	RegressNet& operator= ( const RegressNet& rhs );

	// Accessor for data structure representing input variables,
	//    takes vector of vector of unsigned as argument
	void setInputStructure( const vector< vector< unsigned > >& );

	// Accessor for Network, takes Network & last error as arguments
	void setNetwork( Network*, const double );

	// Stepwise regression methods
	void reverse_regress();	// stepwise reverse regression
	void forward_regress(); // stepwise forward regression

	// RegressNet exception, extends runtime_error class
	class RegressNetErr : public runtime_error {
	public:
		RegressNetErr( const char* message ) throw() : runtime_error( message ) {}
	};

private:
	Network *netPtr; // incoming Network (not owned)
	unique_ptr< Network > netCopyPtr; // owned copy of incoming Network to make subnetworks

	bool historyFlag; // indicates logging to history file

	double e_in; // incoming Network's error

	// Data structure representing input variables
	vector< vector< unsigned > > variable_defs;

	// Define stepwise regression table type, p-value first, followed by variable number
	typedef multimap< double, unsigned, less< double > > ptable;
	
	ostringstream out; // for writing to screen & file
	stringstream errorOut; // for exceptions
	
	string errorString; 

	// Utility function which outputs an ostringstream to screen and history file,
	//    takes ostringstream as argument
	void report( ostringstream& );

	// Utility function prints input variable structure to screen and history
	//    file, takes data structure representing input variables as argument
	void print_input_structure( const vector< vector< unsigned > >& );

	// Utility function which prints out stepwise regression table, takes regression
	//    table multimap type as argument
	void print_regression_table( const ptable& );

	// Utility function which makes a copy of the incoming Network
	bool copy_network();

	// Utility function which returns the name of the type of Network
	string network_name() const;

	// Utility function which calculates chi-square using Wilk's GLRT
	inline double Wilks( const double, const double, const double ) const;

	// Copy utility
	void copy( const RegressNet& rhs );
};

#endif
