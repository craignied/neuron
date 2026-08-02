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
#include <functional> // the progress callback
#include <utility>    // the selection path's (p-value, variable) pairs

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

	// Supply the p-value threshold up front (non-interactive callers such as
	//    the GUI). When set, reverse_regress/forward_regress use it instead of
	//    prompting on stdin. Unset (the default), the CLI prompt is unchanged.
	void setThreshold( double t );

	// Live progress from a running stepwise analysis. The engine PUSHES these
	//    facts through a callback, so a caller can report what the procedure is
	//    doing without scraping the prose it writes to util::screen(). Every
	//    field is something the procedure already knows; nothing here is parsed
	//    back out of a report.
	struct Progress {
		string direction;             // "reverse" or "forward"
		unsigned step;                // outer selection pass, 0-based
		unsigned candidate;           // candidate within this pass, 1-based
		unsigned candidatesThisStep;  // exactly knowable at the pass's start
		unsigned fitsCompleted;       // candidate fits finished, all passes
		unsigned variable;            // the conceptual variable under test
		vector< unsigned > inputs;    // its FULL input-node group, never split
		string phase;                 // what is happening to it right now

		// Set only on the announcement made AFTER a candidate's fit returns,
		//    so a caller can report how the last completed candidate ended
		//    rather than leaving it labelled "training" until the whole run
		//    finishes. finished=false on the announcement made before a fit.
		bool finished;
		bool converged;               // meaningful when finished
		string stopReason;            // engine token, when finished
	};
	typedef function< void( const Progress& ) > ProgressFn;
	void setProgress( ProgressFn f ) { progressFn = f; }

	// An observer installed on every candidate clone, so a Stop request reaches
	//    the subnetwork training NOW rather than being noticed after all the
	//    refits have finished. NOT owned. A cancelled candidate ends as
	//    STOP_CANCELLED, which is not convergence, so it ends the analysis
	//    through the same path an unfinished fit does.
	void setObserver( Iterative::Observer* obs ) { candidateObserver = obs; }

	// Stepwise regression methods
	void reverse_regress();	// stepwise reverse regression
	void forward_regress(); // stepwise forward regression

	// EVERY candidate the procedure considered, in the order it considered
	//    them -- the audit trail. One entry per subnetwork trained, carrying
	//    the comparison that was made and how the fit ended, so a consumer
	//    never has to parse the prose report to reconstruct the analysis.
	//    Populated as the run proceeds, so it survives a failure or a cancel
	//    and describes exactly how far the procedure got.
	struct Candidate {
		unsigned step;                // outer selection pass, 0-based
		unsigned candidate;           // candidate within that pass, 1-based
		unsigned variable;            // conceptual variable
		vector< unsigned > inputs;    // its full input-node group
		double priorError;            // error of the network compared against
		double error;                 // this subnetwork's error
		unsigned df;                  // degrees of freedom of the comparison
		double G2;                    // Wilks statistic
		double p;                     // its p-value (NaN if not calculable)
		bool converged;               // did a stopping rule fire?
		string stopReason;            // engine token
		unsigned iterations;          // how long the fit ran
		bool selected;                // did this candidate win its pass?
	};
	const vector< Candidate >& getCandidates() const { return candidates; }

	// The selection path in the order variables were actually removed (reverse)
	//    or added (forward), each with the p-value that selected it. Available
	//    after a run. The internal ptable is sorted BY p-value, so it cannot
	//    answer "what happened first"; this can.
	const vector< pair< double, unsigned > >& getSelectionPath() const
		{ return selectionPath; }

	// The variables the procedure ended with: those still in the model after a
	//    reverse run, those admitted by a forward run.
	const vector< unsigned >& getFinalVariables() const { return finalVariables; }

	// How many candidate subnetworks were trained in total
	unsigned getFitsCompleted() const { return fitsCompleted; }

	// Did the selection loop run to a decision? False when an unconverged
	//    candidate or a cancellation ended the analysis early. It qualifies
	//    getFinalVariables(), which reports what the COMPLETED passes settled:
	//    without this flag a caller cannot tell "the procedure kept these
	//    variables" from "the procedure got this far".
	bool getComplete() const { return analysisComplete; }

	// RegressNet exception, extends runtime_error class
	class RegressNetErr : public runtime_error {
	public:
		RegressNetErr( const char* message ) throw() : runtime_error( message ) {}
	};

protected:
	// May a candidate fit's error enter a Wilks comparison? TWO INDEPENDENT
	//    conditions, and the second is not implied by the first: a stopping
	//    RULE must have ended the run (Iterative::converged -- the engine-wide
	//    predicate), AND the error it returned must be a real number. A
	//    saturated fit can satisfy the first and fail the second, because its
	//    gradients underflow to zero while its cross-entropy overflows.
	//
	//    Pure and static so that the policy can be pinned directly. Reaching
	//    converged-and-non-finite through training is a knife edge -- measured,
	//    270 configurations gave zero portable hits and the one that hit moved
	//    under a change of learning rate, input scale, row count or seed -- so a
	//    test that had to arrive here by training would be asserting one
	//    toolchain's arithmetic. A predicate can be asked all four questions
	//    directly. requireConvergedFit() is its only production caller.
	static bool candidateFitEligible( Iterative::StopReason reason, double error );

	// A pass in which NO candidate's p-value could be calculated has nothing to
	//    select from, and must say so rather than name a variable it never
	//    compared. Reports, then throws.
	//
	//    This is NOT the all-p-values-are-zero case. Zero is a p-value: it means
	//    maximally significant, the candidate WAS compared, and it wins its pass
	//    if it is the first one. This is the case where stats::pX2 itself
	//    refused every time.
	//
	//    AN INVARIANT DEFENSE, NOT A PUBLIC-INPUT BRANCH. Since the input
	//    structure must be an exact partition (see setInputStructure), no
	//    caller-supplied structure can drive a whole pass into it: pX2 is
	//    gammq, which refuses only on a non-positive shape parameter or a
	//    negative argument -- df == 0 or G2 < 0 -- and a partition gives every
	//    variable at least one node, while the call site already handles
	//    G2 <= 0 without calling pX2 at all. What remains is an internal or
	//    statistical failure, and this refuses it rather than naming a winner
	//    that does not exist.
	//
	//    So its call sites are unreachable by construction and no behavioral
	//    test can turn a sabotage of them red; the BODY is characterized
	//    directly instead (check_regressnet case 59). Do not add a production
	//    hook whose only purpose is to make the call site reachable.
	void requireComparablePass( unsigned step );

private:
	Network *netPtr; // incoming Network (not owned)
	unique_ptr< Network > netCopyPtr; // owned copy of incoming Network to make subnetworks

	bool historyFlag; // indicates logging to history file

	double e_in; // incoming Network's error

	double regressThreshold; // p-value threshold supplied via setThreshold()
	bool thresholdSet; // true once setThreshold() has been called

	// Data structure representing input variables
	vector< vector< unsigned > > variable_defs;

	// Progress reporting and cancellation. Neither is copied (see copy()): a
	//    working copy must never drive the original's GUI buffers, exactly as
	//    Iterative::copy nulls a clone's observer.
	ProgressFn progressFn;
	Iterative::Observer* candidateObserver;

	// The run's structured result, in the order things actually happened
	vector< Candidate > candidates;
	vector< pair< double, unsigned > > selectionPath;
	vector< unsigned > finalVariables;
	unsigned fitsCompleted; // candidate subnetworks trained
	bool analysisComplete;  // did the selection loop reach a decision?

	// Push one progress fact to the callback, if a caller installed one. Costs
	//    nothing when nobody is listening, so the CLI path is unchanged.
	//    finished=false announces a candidate about to train; finished=true
	//    announces the same candidate's completed fit and how it ended.
	void announce( const string& direction, unsigned step, unsigned candidate,
		unsigned candidatesThisStep, unsigned variable, const string& phase,
		bool finished = false );

	// Record one considered candidate in the audit trail, returning its index
	//    so the pass can mark the winner afterwards.
	unsigned recordCandidate( unsigned step, unsigned candidate,
		unsigned variable, double priorError, double error, unsigned df,
		double G2, double p );

	// Wilks' GLRT compares log likelihoods, so the SOURCE fit must be
	//    cross-entropy. Throws otherwise; never mutates the original.
	void requireCrossEntropySource();

	// The closing summary of a COMPLETED analysis: direction, that it
	//    completed, the selection path in order, and the variables the
	//    procedure ended with. Called only after the selection loop reaches a
	//    decision, so an analysis that failed or was cancelled never prints a
	//    final variable set -- it has none to report.
	void report_summary( const string& direction, const string& verb );

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

	// A candidate subnetwork's error may enter a Wilks comparison only if the
	//    fit actually FINISHED -- the engine-wide convergence rule, applied at
	//    the one place stepwise consumes a training result. Writes the failure
	//    into the audit trail, then throws. Takes a label naming the fit (the
	//    conceptual variable and its input nodes, or the baseline network) and
	//    the error train() returned.
	void requireConvergedFit( const string& what, double error );

	// Copy utility
	void copy( const RegressNet& rhs );
};

#endif
