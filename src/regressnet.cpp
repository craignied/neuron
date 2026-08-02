// Methods for RegressNet, the object that performs stepwise regression on neural networks

#include "stdafx.h" // For MSVC, must be first!

#include "regressnet.h"

#include <cmath>    // isfinite: a non-finite candidate error is not a fit either
#include <limits>   // quiet_NaN for a p-value that could not be calculated
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <typeinfo>

#include "netclone.h"   // cloneNetwork, the generalized network copy
#include "simpleprop.h" // SimpleProp class
#include "bareprop.h"   // BareProp class
#include "backprop.h"   // BackProp class
#include "logistic.h"   // Logistic class
#include "stats.h"

// Default constructor
//    e_in is initialized here for the same reason Iterative::copy writes
//    "not copied" out explicitly: a member left out of this list holds
//    indeterminate memory, not a zero. setNetwork() overwrites it on every
//    path that reaches the analysis today, which is exactly the argument that
//    was made about quietFlag and about Model::errorType before each of them
//    became reachable.
RegressNet::RegressNet() : netPtr ( 0 ),
	historyFlag ( false ), e_in ( 0 ),
	regressThreshold ( 0 ), thresholdSet ( false ),
	errorString ( "RegressNet error: " ), candidateObserver ( 0 ),
	fitsCompleted ( 0 ), analysisComplete ( false ) { }

// Push one progress fact to whoever is listening. A run with no callback
//    installed (every CLI run) does nothing here.
void RegressNet::announce( const string& direction, unsigned step,
	unsigned candidate, unsigned candidatesThisStep, unsigned variable,
	const string& phase, bool finished )
{
	if ( !progressFn )
		return;

	Progress p;
	p.direction = direction;
	p.step = step;
	p.candidate = candidate;
	p.candidatesThisStep = candidatesThisStep;
	p.fitsCompleted = fitsCompleted;
	p.variable = variable;
	p.inputs = variable_defs[ variable ]; // the WHOLE group, never split
	p.phase = phase;
	p.finished = finished;

	// How the fit ended is knowable only after it returns, and only from the
	//    clone that ran it
	p.converged = false;
	if ( finished && netCopyPtr )
	{
		p.converged = Iterative::converged( netCopyPtr->getStopReason() );
		p.stopReason = Iterative::stopReasonToken( netCopyPtr->getStopReason() );
	}

	progressFn( p );
}

// Record one considered candidate in the audit trail
unsigned RegressNet::recordCandidate( unsigned step, unsigned candidate,
	unsigned variable, double priorError, double error, unsigned df,
	double G2, double p )
{
	Candidate c;
	c.step = step;
	c.candidate = candidate;
	c.variable = variable;
	c.inputs = variable_defs[ variable ];
	c.priorError = priorError;
	c.error = error;
	c.df = df;
	c.G2 = G2;
	c.p = p;
	c.converged = Iterative::converged( netCopyPtr->getStopReason() );
	c.stopReason = Iterative::stopReasonToken( netCopyPtr->getStopReason() );
	c.iterations = netCopyPtr->getIterations();
	c.selected = false; // the pass marks its winner afterwards
	candidates.push_back( c );
	return ( unsigned ) candidates.size() - 1;
}

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
	regressThreshold = rhs.regressThreshold;
	thresholdSet = rhs.thresholdSet;

	// A copy is a fresh analysis, and "not copied" is WRITTEN here rather than
	//    left out -- the copy constructor delegates straight to this function
	//    without running the default constructor, so anything omitted holds
	//    indeterminate memory rather than a zero (the quietFlag lesson, and
	//    before it the uninitialised Model::errorType). The scalars below are
	//    the ones that would be indeterminate; the containers and the
	//    std::function default-construct safely on construction but would
	//    carry stale state through assignment, so they are reset too.
	candidateObserver = 0;
	progressFn = ProgressFn();
	fitsCompleted = 0;
	analysisComplete = false;
	candidates.clear();
	selectionPath.clear();
	finalVariables.clear();
}

// Supply the p-value threshold up front, bypassing the interactive prompt
void RegressNet::setThreshold( double t )
{
	regressThreshold = t;
	thresholdSet = true;
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
	util::screen() << out.str(); // write to screen
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

// Makes a copy of the incoming Network via cloneNetwork(), the generalized
//    typeid dispatch that once lived here (netclone.h -- add new Network
//    types THERE; auto algorithm selection clones through the same path)
bool RegressNet::copy_network()
{
	bool success = true;

	netCopyPtr = cloneNetwork( *netPtr );

	if ( !netCopyPtr ) // no identifiable Network type was found
	{
		errorOut << errorString << "couldn't copy Network, unknown type";
		throw RegressNetErr( errorOut.str().c_str() );
		success = false;
	}

	// Iterative::copy deliberately nulls a clone's observer (a working copy must
	//    not drive the original's buffers), so it is re-installed here, at the
	//    ONE place candidates are created: a Stop must reach whichever
	//    subnetwork is training now, not be noticed after every refit finished.
	if ( candidateObserver )
		netCopyPtr->setObserver( candidateObserver );

	// Candidate subnetworks are QUIET. Stepwise consumes exactly two things
	//    from a candidate fit -- the training error and the stop reason -- and
	//    train()'s epilogue was additionally re-deriving, for EVERY candidate,
	//    the classification tables, the ROC fit and a 2000-resample bootstrap
	//    over the TEST set. None of it reaches the Wilks comparison, which
	//    reads the training likelihood. Beyond the cost, it meant model
	//    selection repeatedly evaluated a set that is supposed to stay
	//    untouched until a procedure has been chosen.
	//    This changes what the run PRINTS, never what it computes: the weights,
	//    the stopping iteration, the returned error and the RNG streams are
	//    identical either way (proven against the pre-change binary).
	netCopyPtr->setQuiet( true );

	return success;
}

// Returns the name of the type of Network -- which the Model already knows.
//    This was seventeen lines of typeid dispatch returning exactly the objType
//    strings the four constructors set, with "unknown" for anything else. Model
//    carries that string, saves it as the first line of every model file, and
//    reads it back on load; getType() is therefore the authoritative answer and
//    a new Network type now reports its real name instead of "unknown".
string RegressNet::network_name() const
{
	return netPtr->getType();
}

// Stepwise reverse regression
void RegressNet::reverse_regress()
{
	requireCrossEntropySource(); // Wilks' GLRT compares log likelihoods
	analysisComplete = false; // set only when the outer loop finishes

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
		double threshold = thresholdSet ? regressThreshold
			: util::askD( question, 0, 1 );
		
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

			// THE FIRST SUCCESSFULLY CALCULATED p-VALUE IS THE WINNER, and
			//    later candidates replace it only on a strict p > largest_p --
			//    which is what keeps the FIRST of several tied candidates.
			//
			//    This flag exists because largest_p starts at 0 and p == 0 is a
			//    perfectly good p-value: it means maximally significant, and
			//    stats::pX2 is gammq, which underflows to exactly 0 for a large
			//    chi-square. A pass in which every candidate is strongly
			//    significant therefore never satisfied `p > largest_p` at all,
			//    and largest_var -- which had no initializer -- was read
			//    unconditionally at the foot of the pass. Measured: the report
			//    named "variable 83256288", a different number on every run,
			//    0xAAAAAAAA under -ftrivial-auto-var-init=pattern, and at
			//    threshold 0 the value was pushed into `removed` and used to
			//    index variable_defs on the next pass -- SIGBUS.
			bool haveWinner = false;

			unsigned largest_var = 0, // variable with largest p-value
				hold_df = last_df; // holds df of least significant variable

			ptable inner_ps; // holds p-values of removed variables within a single pass

			// Exactly knowable now: this pass considers one candidate for every
			//    variable still in the model. (What LATER passes will consider
			//    is not knowable -- the threshold may stop the procedure first.)
			unsigned candidateNo = 0,
				candidatesThisStep = variable_defs.size() - removed.size();

			// variable -> its row in the audit trail, so the pass can mark its
			//    winner once the comparison is settled
			map< unsigned, unsigned > passIndex;

			// Inner loop: iterate through remaining variables, find largest p-value
			for ( unsigned j = 0; j < variable_defs.size(); j++ )
			{
				// Only examine the variable if it hasn't already been removed
				if ( find( removed.begin(), removed.end(), j ) == removed.end() )
				{
					removed.push_back( j ); // add the variable to those removed
					candidateNo++;

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

					announce( "reverse", i, candidateNo, candidatesThisStep, j,
						"training candidate" );

					double subError = netCopyPtr->train(); // train the subnetwork
					fitsCompleted++;

					// The fit is finished: say so, with how it ended, BEFORE
					//    the eligibility check below can throw. Otherwise the
					//    last thing a watcher ever saw was "training candidate"
					//    and a fit count one behind reality.
					announce( "reverse", i, candidateNo, candidatesThisStep, j,
						"candidate complete", true );

					// The audit trail is written BEFORE the eligibility
					//    check, because that check THROWS -- and the candidate
					//    that fails is precisely the one a reader needs
					//    recorded. Its Wilks statistic and p-value stay
					//    not-a-number until a comparison is actually permitted;
					//    a rejected fit contributes no statistic, but it is
					//    still a candidate the procedure considered.
					passIndex[ j ] = recordCandidate( i, candidateNo, j,
						lastError, subError, last_df - netCopyPtr->df(),
						numeric_limits< double >::quiet_NaN(), numeric_limits< double >::quiet_NaN() );

					// An unfinished fit may not be compared (see the method)
					{
						ostringstream who;
						who << "variable " << j << " = node(s) " << variable_defs[ j ];
						requireConvergedFit( who.str(), subError );
					}

					// Calculate chi-square from Wilk's GLRT and print results
					double G2 = Wilks( netPtr->getDataSet().getNumTrain(), lastError, subError );
					unsigned df = last_df - netCopyPtr->df();
					out << "Error prior network = " << lastError << endl
						<< "Error this subnetwork = " << subError << endl
						<< "Chi-square = " << G2 << ", degrees of freedom = " << df << endl;
					double pThis = numeric_limits< double >::quiet_NaN();
					try // print p-value for this comparison
					{
						double p;
						if ( G2 <= 0 ) // subnetwork's error is less!
							p = 1;
						else // calculate p-value from chi-square
							p = stats::pX2( df, G2 );
						pThis = p;
						inner_ps.insert( ptable::value_type( p, j ) ); // make table
						out << "p = " << p << endl;
						// The first calculated p-value takes the pass; after
						//    that, strictly larger wins, so ties keep the first
						if ( !haveWinner || p > largest_p )
						{
							holdError = subError; // record this subnetwork's error & df
							hold_df = netCopyPtr->df();
							largest_var = j; // remember which variable it is
							largest_p = p; // record largest p for the final table
							haveWinner = true;
						}
					}
					catch ( stats::statsErr& e ) // error in chi-square calculation
					{
						out << e.what() << endl;
					}
					// The comparison now exists: complete the record made
					//    before the eligibility check above
					candidates[ passIndex[ j ] ].df = df;
					candidates[ passIndex[ j ] ].G2 = G2;
					candidates[ passIndex[ j ] ].p = pThis;
					report( out ); // output to screen & history file

					removed.pop_back(); // put the variable back for next round
				}
			}

			// Print results of this set of subnetwork comparisons
			out << endl << "p-value of each removed variable in pass " << i << ":" << endl;
			report( out );
			print_regression_table( inner_ps );

			// Nothing in this pass could be compared, so there is no least
			//    significant variable to name (see the method)
			if ( !haveWinner )
				requireComparablePass( i );

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
				// ... and record it in REMOVAL ORDER: the table above is sorted
				//     by p-value, so it cannot say what happened first
				selectionPath.push_back( make_pair( largest_p, largest_var ) );
				if ( passIndex.count( largest_var ) )
					candidates[ passIndex[ largest_var ] ].selected = true;
			}

			// Print p-values of removed variables
			if ( !p_values.empty() ) // only print the table if it has something in it
			{
				out << endl << "p-values of all removed variables:" << endl;
				report( out );
				print_regression_table( p_values );
			}

			// The variables still in the model AFTER THIS COMPLETED PASS.
			//    Maintained per pass, not once at the end: an exception (an
			//    unconverged candidate, a cancel) leaves the loop early, and
			//    reporting an empty list would claim the procedure retained
			//    nothing when in fact every earlier pass had settled.
			finalVariables.clear();
			for ( unsigned v = 0; v < variable_defs.size(); v++ )
				if ( find( removed.begin(), removed.end(), v ) == removed.end() )
					finalVariables.push_back( v );

			// Stop after printing if indicated
			if ( stop )
				break;
		}

		analysisComplete = true; // the loop ran to a decision, not an exception
		report_summary( "Reverse", "removed" );
	}

	netCopyPtr.reset(); // get rid of copy of Network object for subnetworks
}

// Stepwise forward regression
void RegressNet::forward_regress()
{
	requireCrossEntropySource(); // Wilks' GLRT compares log likelihoods
	analysisComplete = false; // set only when the outer loop finishes

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
		double threshold = thresholdSet ? regressThreshold
			: util::askD( question, 0, 1 );
		
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

		// The baseline is not a candidate, but every comparison in the first
		//    pass is made against it: if it did not finish, nothing downstream
		//    means anything
		requireConvergedFit( "the baseline network (no variables)", lastError );

		unsigned last_df = netCopyPtr->df(); // & get its df

		// Outer loop: number of sets of comparisons = number of variables
		for ( i = 0; i < variable_defs.size(); i++ )
		{
			double smallest_p = 1, // smallest p-value found
				holdError = 0; // holds error of most significant variable

			// The same flag reverse carries, for the same reason -- a pass in
			//    which every p-value calculation was refused has no winner and
			//    must not name one. Forward never had reverse's uninitialized
			//    read (smallest_var is initialized, and p <= smallest_p from 1
			//    always takes the first candidate), so this changes no
			//    selection; it only lets the no-comparable-candidate case be
			//    refused rather than reported as "the smallest was variable 0".
			//
			//    THE TIE SENSE IS DELIBERATELY NOT NORMALIZED against reverse's.
			//    Reverse's strict `>` keeps the FIRST maximal candidate;
			//    forward's `<=` keeps the LAST minimal one. That asymmetry is
			//    preserved here on purpose -- changing it would change which
			//    variable a tie selects, which is a selection change, not a
			//    correctness fix, and does not belong in this commit.
			bool haveWinner = false;

			unsigned smallest_var = 0, // variable with smallest p-value
				hold_df = last_df; // holds df of most significant variable

			ptable inner_ps; // holds p-values of added variables within a single pass

			// Exactly knowable now: one candidate per variable not yet admitted
			unsigned candidateNo = 0,
				candidatesThisStep = variable_defs.size() - added.size();

			// variable -> its row in the audit trail (see reverse_regress)
			map< unsigned, unsigned > passIndex;

			// Inner loop: iterate through remaining variables, find largest p-value
			for ( unsigned j = 0; j < variable_defs.size(); j++ )
			{
				// Only examine the variable if it hasn't already been added
				if ( find( added.begin(), added.end(), j ) == added.end() )
				{
					added.push_back( j ); // add the variable to those added
					candidateNo++;

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

					announce( "forward", i, candidateNo, candidatesThisStep, j,
						"training candidate" );

					// Train this new network, note that it is considered the "full" network
					double fullError = netCopyPtr->train();
					fitsCompleted++;

					// The fit is finished: say so, with how it ended, BEFORE
					//    the eligibility check below can throw
					announce( "forward", i, candidateNo, candidatesThisStep, j,
						"candidate complete", true );

					// The audit trail is written BEFORE the eligibility
					//    check, because that check THROWS -- and the candidate
					//    that fails is precisely the one a reader needs
					//    recorded. Its Wilks statistic and p-value stay
					//    not-a-number until a comparison is actually permitted;
					//    a rejected fit contributes no statistic, but it is
					//    still a candidate the procedure considered.
					passIndex[ j ] = recordCandidate( i, candidateNo, j,
						lastError, fullError, netCopyPtr->df() - last_df,
						numeric_limits< double >::quiet_NaN(), numeric_limits< double >::quiet_NaN() );

					// An unfinished fit may not be compared (see the method)
					{
						ostringstream who;
						who << "variable " << j << " = node(s) " << variable_defs[ j ];
						requireConvergedFit( who.str(), fullError );
					}

					// Calculate chi-square from Wilk's GLRT and print results
					double G2 = Wilks( netPtr->getDataSet().getNumTrain(), fullError, lastError );
					unsigned df = netCopyPtr->df() - last_df;
					out << "Error prior network = " << lastError << endl
						<< "Error this subnetwork = " << fullError << endl
						<< "Chi-square = " << G2 << ", degrees of freedom = " << df << endl;
					double pThis = numeric_limits< double >::quiet_NaN();
					try // print p-value for this comparison
					{
						double p;
						if ( G2 <= 0 ) // last network's error is less!
							p = 1;
						else // calculate p-value from chi-square
							p = stats::pX2( df, G2 );
						pThis = p;
						inner_ps.insert( ptable::value_type( p, j ) ); // make table
						out << "p = " << p << endl;
						// `<=`, not `<`: forward keeps the LAST of tied minima.
						//    See the haveWinner comment above -- preserved.
						if ( !haveWinner || p <= smallest_p )
						{
							holdError = fullError; // record this network's error & df
							hold_df = netCopyPtr->df();
							smallest_var = j; // remember which variable it is
							smallest_p = p; // record smallest p for the final table
							haveWinner = true;
						}
					}
					catch ( stats::statsErr& e ) // error in chi-square calculation
					{
						out << e.what() << endl;
					}
					// The comparison now exists: complete the record made
					//    before the eligibility check above
					candidates[ passIndex[ j ] ].df = df;
					candidates[ passIndex[ j ] ].G2 = G2;
					candidates[ passIndex[ j ] ].p = pThis;
					report( out ); // output to screen & history file

					added.pop_back(); // put the variable back for next round
				}
			}

			// Print results of this set of this network comparisons
			out << endl << "p-value of each added variable in pass " << i << ":" << endl;
			report( out );
			print_regression_table( inner_ps );

			// Nothing in this pass could be compared (see reverse_regress)
			if ( !haveWinner )
				requireComparablePass( i );

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
				// ... and record it in ADDITION ORDER (see reverse_regress)
				selectionPath.push_back( make_pair( smallest_p, smallest_var ) );
				if ( passIndex.count( smallest_var ) )
					candidates[ passIndex[ smallest_var ] ].selected = true;
			}

			// Print p-values of added variables
			if ( !p_values.empty() ) // only print the table if it has something in it
			{
				out << endl << "p-values of all added variables:" << endl << endl;
				report( out );
				print_regression_table( p_values );
			}

			// The variables admitted AFTER THIS COMPLETED PASS (see
			//    reverse_regress: maintained per pass so an early exit still
			//    reports what the completed passes established)
			finalVariables = added;

			// Stop after printing if indicated
			if ( stop )
				break;
		}

		analysisComplete = true; // the loop ran to a decision, not an exception
		report_summary( "Forward", "added" );
	}

	netCopyPtr.reset(); // get rid of copy of Network object for subnetworks
}

// Wilks' generalized likelihood ratio test compares LOG LIKELIHOODS, so every
//    error in the comparison must come from a cross-entropy fit.
//
//    This used to call netPtr->setXEerror(), which was wrong twice over. It
//    MUTATED the user's trained model -- stepwise is a standalone analysis over
//    clones and must leave the original exactly as it found it. And it did not
//    achieve what it looked like it achieved: e_in, the prior error every first
//    -pass comparison is made against, was captured from the ORIGINAL fit and
//    is in that fit's objective. Flipping the model to cross-entropy afterwards
//    left candidates training under cross-entropy while their baseline error
//    was still least-mean-squares, so G2 = 2N(Esub - Efull) subtracted two
//    different objectives and reported the result as a chi-square. Logistic
//    models are cross-entropy by construction, which is why the walkthrough
//    never saw it; an LMS-trained network is the case that breaks.
//
//    Refusing is the honest option: the fix is to retrain the source model
//    under cross-entropy, which is the user's decision, not ours to make
//    silently on their model.
void RegressNet::requireCrossEntropySource()
{
	if ( netPtr->getXEerror() ) // errorType 1 = cross-entropy
		return;

	errorOut.str( "" );
	errorOut << errorString << "stepwise regression needs a cross-entropy fit: "
		<< "Wilks' generalized likelihood ratio compares log likelihoods, and "
		<< "this model was trained with least-mean-squares error. Retrain it "
		<< "with the cross-entropy output error function, then regress.";
	throw RegressNetErr( errorOut.str().c_str() );
}

// The closing summary of a completed analysis. The selection result was
//    previously reachable only from the API's structured `finalVariables`, or
//    by reading the p-value table and working out which variables were absent
//    from it -- the report answered "what happened" but never "what did I end
//    up with". This states it, in the ENGINE report, so the GUI results pane,
//    the CLI transcript and neuron.log all carry the same sentence rather than
//    each interface having to compose its own.
//
//    Only a COMPLETED analysis reaches here: both callers invoke this after
//    their selection loop reaches a decision, and every failure path throws out
//    of the loop first. A run that stopped early therefore prints no final
//    variable set, which is the point -- it does not have one, and an empty
//    list presented as a conclusion would claim the procedure kept nothing.
void RegressNet::report_summary( const string& direction, const string& verb )
{
	out << endl << direction << " stepwise regression complete." << endl;

	// The path, in the order the procedure actually took it
	out << "Variables " << verb << ", in order: ";
	if ( selectionPath.empty() )
		out << "none";
	else
		for ( unsigned i = 0; i < selectionPath.size(); i++ )
			out << ( i ? ", " : "" ) << selectionPath[ i ].second;
	out << endl;

	// ... and what the model ended with. "retained" and "selected" are not
	//     interchangeable: reverse KEEPS what it never removed, forward ADMITS
	//     what it added, and a reader must not have to infer which run this was.
	out << "Final " << ( direction == "Reverse" ? "retained" : "selected" )
		<< " variables: ";
	if ( finalVariables.empty() )
		out << "none";
	else
		for ( unsigned i = 0; i < finalVariables.size(); i++ )
			out << ( i ? ", " : "" ) << finalVariables[ i ];
	out << endl;

	report( out ); // output to screen & history file
}

// A candidate subnetwork's error may enter the Wilks comparison only if the fit
//    actually finished. Iterative::converged is THE engine-wide predicate (the
//    training report, OBD eligibility and the CV adapters all ask it); stepwise
//    consumed train()'s return value directly and asked nothing, so a fit that
//    merely ran out of iteration ceiling produced a chi-square and a p-value
//    indistinguishable from a converged one.
//
//    Why this FAILS the analysis instead of skipping the candidate: a pass
//    selects the variable with the largest (reverse) or smallest (forward)
//    p-value among ALL candidates in that pass. Silently dropping one does not
//    leave a gap -- it changes which variable wins, biasing selection toward
//    whichever variables happened to fit. A refusal naming the candidate is
//    recoverable; a quietly reordered selection is not. The partial audit trail
//    is reported first, so the user can see every candidate that did complete
//    and exactly where the procedure stopped.
// The eligibility policy itself, separated from the reporting so it can be
//    asked all four questions directly. See the header for why.
bool RegressNet::candidateFitEligible( Iterative::StopReason reason, double error )
{
	return Iterative::converged( reason ) && std::isfinite( error );
}

void RegressNet::requireConvergedFit( const string& what, double error )
{
	Iterative::StopReason why = netCopyPtr->getStopReason();

	if ( candidateFitEligible( why, error ) )
		return; // a finished fit: it may be compared

	out << endl << "STOPPING: the fit for " << what
		<< " is not a finished fit." << endl;

	if ( !std::isfinite( error ) )
		out << "Its training error is not a finite number." << endl;
	else
		out << "It ended as '" << Iterative::stopReasonToken( why ) << "' after "
			<< netCopyPtr->getIterations()
			<< " iterations, which is not a stopping condition." << endl;

	out << "No chi-square or p-value is calculated for it, and no variable is "
		<< "selected from an incomplete pass." << endl;

	if ( why == Iterative::STOP_MAX_ITERATIONS )
		out << "Raise the maximum iterations, or set a stopping condition that "
			<< "can fire, then run the regression again." << endl;
	else if ( why == Iterative::STOP_CANCELLED )
		out << "The regression was stopped by request." << endl;

	report( out ); // the partial trail reaches the screen/history before the throw

	errorOut.str( "" ); // one-shot, but never inherit a previous message
	errorOut << errorString << what << " did not converge ("
		<< Iterative::stopReasonToken( why ) << "); no variable was selected";
	throw RegressNetErr( errorOut.str().c_str() );
}

// A pass with nothing to select from. See the header for the distinction this
//    turns on: an all-ZERO pass is not this case -- zero is a p-value, and the
//    first candidate to produce one wins the pass.
void RegressNet::requireComparablePass( unsigned step )
{
	out << endl << "STOPPING: no p-value could be calculated for any candidate "
		<< "in pass " << step << "." << endl
		<< "Every chi-square comparison in the pass was refused, so no variable "
		<< "is least significant and none is selected." << endl;
	report( out ); // the partial trail reaches the reader before the throw

	errorOut.str( "" ); // one-shot, but never inherit a previous message
	errorOut << errorString << "no candidate in pass " << step
		<< " could be compared; no variable was selected";
	throw RegressNetErr( errorOut.str().c_str() );
}

// Calculate chi-square using Wilk's GLRT
inline double RegressNet::Wilks( const double N, const double Efull, const double Esub ) const
{
	return 2.0 * N * ( Esub - Efull );
}
