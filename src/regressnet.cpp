// Methods for RegressNet, the object that performs stepwise regression on neural networks

#include "stdafx.h" // For MSVC, must be first!

#include "regressnet.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <typeinfo>

#include "simpleprop.h" // SimpleProp class
#include "bareprop.h"   // BareProp class
#include "backprop.h"   // BackProp class
#include "logistic.h"   // Logistic class
#include "stats.h"

// Default constructor
RegressNet::RegressNet() : netPtr ( 0 ),
	historyFlag ( false ), errorString ( "RegressNet error: " ) { }

// Copy constructor
RegressNet::RegressNet( const RegressNet& rhs )
{
	RegressNet::copy( rhs ); // use the copy utility
}

// Overloaded = operator
RegressNet& RegressNet::operator= ( const RegressNet& rhs )
{
	if ( &rhs != this ) // check for self-assignment
		RegressNet::copy( rhs ); // use the copy utility
	
	return *this; // enables A = B = C
}

// Copy utility -- note netCopyPtr & ptables are *not* copied
void RegressNet::copy( const RegressNet& rhs )
{
	historyFlag = rhs.historyFlag;
	netPtr = rhs.netPtr;
	e_in = rhs.e_in;
	variable_defs = rhs.variable_defs;
}

// Accessor for data structure representing input variables,
//    takes vector of vector of unsigned as argument
//    Note: setNetwork must be called first
void RegressNet::setInputStructure( const vector< vector< unsigned > >& v_in )
{ 
	// Find maximum element of vector structure
	vector< unsigned > v_max = *max_element( v_in.begin(),
		v_in.end() );
	unsigned max = *max_element( v_max.begin(), v_max.end() ),
		// What the max should be
		correct = netPtr->getDataSet().getInput() - 1;
				
	// Check to make sure maximum element is correct
	if ( max != correct )
	{
		errorOut << errorString << "maximum node value not " << correct;
		throw RegressNetErr( errorOut.str().c_str() );
	}
	else
		variable_defs = v_in;
}

// Accessor for Network, takes Network & last error as arguments
void RegressNet::setNetwork( Network* n_p, const double x )
{
	netPtr = n_p; // sets the pointer to the incoming Network
	e_in = x; // sets the Network's last error
	historyFlag = n_p->getHistory(); // sets the state of history logging
}

// Method which outputs an ostringstream to screen and to history file if specified,
//    takes ostringstream as argument
void RegressNet::report( ostringstream& out )
{
	cout << out.str(); // write to screen
	if ( historyFlag ) // and to history file if specified
	{
		ofstream hFile( netPtr->getHistoryFilename().c_str(), ios::out | ios::app );
		hFile << out.str();
		hFile.close();
	}
	out.str( "" ); // and clear ostringstream object
}

// Method prints input variable structure to screen, and if specified, file, takes
//    data structure representing input variables as argument
void RegressNet::print_input_structure( const vector< vector< unsigned > >& variable_defs )
{
	out.str( "" ); // clear ostringstream object
	out << "Variable structure:" << endl; // header

	// Iterate through elements of input variable data structure
	for ( unsigned i = 0; i < variable_defs.size(); i++ )
		out << "variable " << i << " is represented by input node(s) "
			<< variable_defs[ i ] << endl;

	report( out ); // output to screen & history file
}

// Method which prints out stepwise regression table, takes regression table multimap
//    type as argument
void RegressNet::print_regression_table( const ptable& p_values )
{
	out.str( "" ); // clear ostringstream object
	out << "Variable:     p-value:" << endl; // header

	// Iterate through multimap, and output formatted by element
	for ( ptable::const_iterator pi = p_values.begin();
		pi != p_values.end(); ++pi )
	{
		out << setiosflags( ios::fixed ) << setw( 9 )
			<< setfill( ' ' ) << pi->second << "  ";
		out << resetiosflags( ios::fixed )
			<< setiosflags( ios::scientific );
		out << setw( 12 ) << setfill( ' ' ) << pi->first << endl;
	}
	report( out ); // use previously coded report() method
}

// Makes a copy of the incoming Network (add here if you add a new Network object)
bool RegressNet::copy_network()
{
	bool success = true;

	if ( typeid( *netPtr ) == typeid( BareProp ) ) // it's a BareProp network
		// Create new BareProp object with copy ctor & set pointer to the copy
		netCopyPtr = make_unique< BareProp >( *dynamic_cast< BareProp* >( netPtr ) );
	else if ( typeid( *netPtr ) == typeid( SimpleProp ) ) // it's a SimpleProp network
		// Create new SimpleProp object with copy ctor & set pointer to the copy
		netCopyPtr = make_unique< SimpleProp >( *dynamic_cast< SimpleProp* >( netPtr ) );
	else if ( typeid( *netPtr ) == typeid( BackProp ) ) // it's a BackProp network
		// Create new BackProp object with copy ctor & set pointer to the copy
		netCopyPtr = make_unique< BackProp >( *dynamic_cast< BackProp* >( netPtr ) );
	else if ( typeid( *netPtr ) == typeid( Logistic ) ) // it's a Binary logistic network
		// Create new Logistic object with copy ctor & set pointer to the copy
		netCopyPtr = make_unique< Logistic >( *dynamic_cast< Logistic* >( netPtr ) );
	else // no identifiable Network type was found
	{
		errorOut << errorString << "couldn't copy Network, unknown type";
		throw RegressNetErr( errorOut.str().c_str() );
		success = false;
	}

	return success;
}

// Returns the name of the type of Network
string RegressNet::network_name() const
{
	string network_type;

	if ( typeid( *netPtr ) == typeid( BareProp ) ) // it's a BareProp network
		network_type = "BareProp";
	else if ( typeid( *netPtr ) == typeid( SimpleProp ) ) // it's a SimpleProp network
		network_type = "SimpleProp";
	else if ( typeid( *netPtr ) == typeid( BackProp ) ) // it's a BackProp network
		network_type = "BackProp";
	else if ( typeid( *netPtr ) == typeid( Logistic ) ) // it's a Binary logistic network
		network_type = "Binary logistic";
	else
		network_type = "unknown";

	return network_type; // return string now containing name
}

// Stepwise reverse regression
void RegressNet::reverse_regress()
{
	netPtr->setXEerror(); // cross-entropy is required for Wilk's GLRT

	if ( variable_defs.empty() ) // input structure specified requires Network specified
	{
		errorOut << errorString << "stepwise regression not fully specified";
		throw RegressNetErr( errorOut.str().c_str() );
	}

	else if ( copy_network() ) // test copying the Network for a subnetwork
	{
		double lastError = e_in; // prior Network's error

		vector< unsigned > removed; // holds removed variables

		ptable p_values; // multimap holds final table p-values of added variables

		// Get the threshold
		string question = "\nWhat is the smallest p-value at which variables should\n";
		question += "stop being removed in the stepwise regression? ";
		double threshold = util::askD( question, 0, 1 ); 
		
		// Print message reporting what kind of Network is being regressed
		out << endl << "Reverse regressing a " << network_name() << " Network:" << endl;
		report( out ); // output to screen & history file
		print_input_structure( variable_defs ); // print input variable structure
		
		// Prior Network's degrees of freedom
		unsigned last_df = netPtr->df();

		// Outer loop: number of sets of comparisons = number of variables
		for ( unsigned i = 0; i < variable_defs.size(); i++ )
		{
			double largest_p = 0, // largest p-value found
				holdError = 0; // holds error of least significant variable

			unsigned largest_var, // variable with largest p-value
				hold_df = last_df; // holds df of least significant variable

			ptable inner_ps; // holds p-values of removed variables within a single pass

			// Inner loop: iterate through remaining variables, find largest p-value
			for ( unsigned j = 0; j < variable_defs.size(); j++ )
			{
				// Only examine the variable if it hasn't already been removed
				if ( find( removed.begin(), removed.end(), j ) == removed.end() )
				{
					removed.push_back( j ); // add the variable to those removed

					// Build the structure containing removed variables
					vector< vector< unsigned > > sub_variables;
					for ( unsigned k = 0; k < removed.size(); k++ )
						sub_variables.push_back( variable_defs[ removed[ k ] ] );
					
					netCopyPtr.reset(); // delete copy if it exists
					copy_network(); // copy the incoming Network to create a subnetwork

					// Print message reporting which variables & nodes are being removed
					out << endl << "Removing variable(s) " << removed << " = node(s) "
						<< flatten( sub_variables ) << ":" << endl;
					report( out );

					// Remove the input nodes from the Network object copy
					netCopyPtr->removeInputs( flatten( sub_variables ) );

					double subError = netCopyPtr->train(); // train the subnetwork

					// Calculate chi-square from Wilk's GLRT and print results
					double G2 = Wilks( netPtr->getDataSet().getNumTrain(), lastError, subError );
					unsigned df = last_df - netCopyPtr->df();
					out << "Error prior network = " << lastError << endl
						<< "Error this subnetwork = " << subError << endl
						<< "Chi-square = " << G2 << ", degrees of freedom = " << df << endl;
					try // print p-value for this comparison
					{
						double p;
						if ( G2 <= 0 ) // subnetwork's error is less!
							p = 1;
						else // calculate p-value from chi-square
							p = stats::pX2( df, G2 );
						inner_ps.insert( ptable::value_type( p, j ) ); // make table
						out << "p = " << p << endl;
						if ( p > largest_p ) // if this p is largest found
						{
							holdError = subError; // record this subnetwork's error & df
							hold_df = netCopyPtr->df();
							largest_var = j; // remember which variable it is
							largest_p = p; // record largest p for the final table
						}
					}
					catch ( stats::statsErr& e ) // error in chi-square calculation
					{
						out << e.what() << endl;
					}
					report( out ); // output to screen & history file

					removed.pop_back(); // put the variable back for next round
				}
			}

			// Print results of this set of subnetwork comparisons
			out << endl << "p-value of each removed variable in pass " << i << ":" << endl;
			report( out );
			print_regression_table( inner_ps );
			out << "the largest was variable " << largest_var << endl;
			report( out );

			// Stop if all subnetwork comparisons are less than the threshold
			bool stop = false;
			if ( largest_p < threshold ) 
			{
				out << "All values were less than the threshold p = " << threshold << endl;
				report( out );
				stop = true;
			}
			else // copy temporary placeholders (error, df) for next round
			{
				lastError = holdError;
				last_df = hold_df;
				removed.push_back( largest_var ); // add variable to those removed
				// Insert into p-value final table
				p_values.insert( ptable::value_type( largest_p, largest_var ) );
			}

			// Print p-values of removed variables
			if ( !p_values.empty() ) // only print the table if it has something in it
			{
				out << endl << "p-values of all removed variables:" << endl;
				report( out );
				print_regression_table( p_values );
			}

			// Stop after printing if indicated
			if ( stop ) 
				break;
		}
	}

	netCopyPtr.reset(); // get rid of copy of Network object for subnetworks
}

// Stepwise forward regression
void RegressNet::forward_regress()
{
	netPtr->setXEerror(); // cross-entropy is required for Wilk's GLRT

	if ( variable_defs.empty() ) // input structure specified requires Network specified
	{
		errorOut << errorString << "stepwise regression not fully specified";
		throw RegressNetErr( errorOut.str().c_str() );
	}

	if ( copy_network() ) // test copying the Network for a subnetwork
	{
		vector< unsigned > added; // holds added variables

		ptable p_values; // multimap holds final table p-values of added variables

		// Get the threshold
		string question = "\nWhat is the largest p-value at which variables should\n";
		question += "stop being added in the stepwise regression? ";
		double threshold = util::askD( question, 0, 1 ); 
		
		// Print message reporting what kind of Network is being regressed
		out << endl << "Forward regressing a " << network_name() << " Network:" << endl;
		report( out ); // output to screen & history file
		print_input_structure( variable_defs ); // print input variable structure
		
		unsigned i; // the usual iterator

		// Create a vector of all positions from which to derive complements
		vector< unsigned > all;
		for ( i = 0; i < netPtr->getDataSet().getInput(); i++ )
			all.push_back( i );

		// Baseline network to start has all nodes removed
		netCopyPtr->removeInputs( all );
		double lastError = netCopyPtr->train(); // train the baseline network
		unsigned last_df = netCopyPtr->df(); // & get its df

		// Outer loop: number of sets of comparisons = number of variables
		for ( i = 0; i < variable_defs.size(); i++ )
		{
			double smallest_p = 1, // smallest p-value found
				holdError = 0; // holds error of most significant variable

			unsigned smallest_var = 0, // variable with smallest p-value
				hold_df = last_df; // holds df of most significant variable

			ptable inner_ps; // holds p-values of added variables within a single pass

			// Inner loop: iterate through remaining variables, find largest p-value
			for ( unsigned j = 0; j < variable_defs.size(); j++ )
			{
				// Only examine the variable if it hasn't already been added
				if ( find( added.begin(), added.end(), j ) == added.end() )
				{
					added.push_back( j ); // add the variable to those added

					// Build the structure containing added variables
					vector< vector< unsigned > > sub_variables;
					for ( unsigned k = 0; k < added.size(); k++ )
						sub_variables.push_back( variable_defs[ added[ k ] ] );

					netCopyPtr.reset(); // delete copy if it exists
					copy_network(); // copy the incoming Network to create new network

					// Print message reporting which variables & nodes are being added
					out << endl << "Adding variable(s) " << added << " = node(s) "
						<< flatten( sub_variables ) << ":" << endl;
					report( out );

					// Adding variables is accomplished by removing their complements
					//    from the full Network
					vector< unsigned > flattened = flatten( sub_variables ), complement;
					back_insert_iterator< vector< unsigned > > complement_i( complement );
					sort( flattened.begin(), flattened.end() ); // must be sorted for set_difference
					set_difference( all.begin(), all.end(), flattened.begin(), flattened.end(),
						complement_i );

					// Remove the complementary input nodes from the Network object copy
					if ( !complement.empty() ) // only remove them if there are nodes to remove
						netCopyPtr->removeInputs( complement );

					// Train this new network, note that it is considered the "full" network
					double fullError = netCopyPtr->train();

					// Calculate chi-square from Wilk's GLRT and print results
					double G2 = Wilks( netPtr->getDataSet().getNumTrain(), fullError, lastError );
					unsigned df = netCopyPtr->df() - last_df;
					out << "Error prior network = " << lastError << endl
						<< "Error this subnetwork = " << fullError << endl
						<< "Chi-square = " << G2 << ", degrees of freedom = " << df << endl;
					try // print p-value for this comparison
					{
						double p;
						if ( G2 <= 0 ) // last network's error is less!
							p = 1;
						else // calculate p-value from chi-square
							p = stats::pX2( df, G2 );
						inner_ps.insert( ptable::value_type( p, j ) ); // make table
						out << "p = " << p << endl;
						if ( p <= smallest_p ) // if this p is smallest found
						{
							holdError = fullError; // record this network's error & df
							hold_df = netCopyPtr->df();
							smallest_var = j; // remember which variable it is
							smallest_p = p; // record smallest p for the final table
						}
					}
					catch ( stats::statsErr& e ) // error in chi-square calculation
					{
						out << e.what() << endl;
					}
					report( out ); // output to screen & history file

					added.pop_back(); // put the variable back for next round
				}
			}

			// Print results of this set of this network comparisons
			out << endl << "p-value of each added variable in pass " << i << ":" << endl;
			report( out );
			print_regression_table( inner_ps );
			out << "the smallest was variable " << smallest_var << endl;
			report( out );

			// Stop if all network comparisons are greater than the threshold
			bool stop = false;
			if ( smallest_p > threshold )
			{
				out << "The smallest was greater than the threshold p = " << threshold << endl;
				report( out );
				stop = true;
			}
			else // copy temporary placeholders (error, df) for next round 
			{
				lastError = holdError;
				last_df = hold_df;
				added.push_back( smallest_var ); // add variable to those added
				// Insert into p-value final table
				p_values.insert( ptable::value_type( smallest_p, smallest_var ) );
			}

			// Print p-values of added variables
			if ( !p_values.empty() ) // only print the table if it has something in it
			{
				out << endl << "p-values of all added variables:" << endl << endl;
				report( out );
				print_regression_table( p_values );
			}

			// Stop after printing if indicated
			if ( stop )
				break;
		}
	}

	netCopyPtr.reset(); // get rid of copy of Network object for subnetworks
}

// Calculate chi-square using Wilk's GLRT
inline double RegressNet::Wilks( const double N, const double Efull, const double Esub ) const
{
	return 2.0 * N * ( Esub - Efull );
}
