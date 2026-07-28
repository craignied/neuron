// Header file for Iterative, the abstract base class for neUROn2++ Models
//    that require iterative algorithms for training

#ifndef ITERATIVE_H
#define ITERATIVE_H

#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <time.h>

using namespace std;

#include "model.h"
#include "utility.h"
#include "plateau.h"

class Iterative : public Model {
public:
	// Why the last train() ended. Set at every exit from the training loop.
	//
	// ELIGIBILITY: only STOP_MIN_ERROR, STOP_CHANGE, STOP_WINDOW, STOP_GRADMAX,
	//    STOP_PLATEAU, and STOP_EARLY_STOP mean a stopping RULE fired -- the fit
	//    reached a point it was asked to stop at. STOP_MAX_ITERATIONS is a safety
	//    ceiling, not a stopping rule: reaching it is a failure to converge, and
	//    no caller may treat those weights as a finished fit (see obd::eligible).
	//    STOP_CANCELLED and STOP_PROBE_BUDGET are likewise not convergence.
	enum StopReason {
		STOP_NONE, // train() has not run
		STOP_MAX_ITERATIONS, // the iteration budget ran out: NOT convergence
		STOP_MIN_ERROR, // error fell below the minimum
		STOP_CHANGE, // change in error over 1 iteration fell below delta
		STOP_WINDOW, // error increased over the window
		STOP_GRADMAX, // maximum absolute gradient fell below the limit
		STOP_PLATEAU, // the error stopped improving (auto-stop, ROADMAP 2 Ph3)
		STOP_CANCELLED, // an outside request stopped the run (GUI/CV cancel)
		STOP_EARLY_STOP, // held-out error deteriorated (OBD validation early stop)
		STOP_PROBE_BUDGET // an autoalgo probe's time budget expired: a bounded
		                  //    experiment, never a claim of convergence
	};

	// Observer of a training run. onIteration() is called at the BOTTOM of
	//    every training iteration, after all the existing stop checks, so a
	//    run with no observer is bit-identical to one that never had the
	//    hook (the goldens' guarantee). Return false to stop training: the
	//    run falls through to the normal epilogue (report, logs, final
	//    error), so a stopped run is a completed run.
	class Observer {
	public:
		virtual ~Observer() {}
		virtual bool onIteration( unsigned iteration, double setError ) = 0;

		// Why this observer just asked training to stop. Called by the train
		//    loop ONLY after onIteration() returned false, and recorded as the
		//    run's StopReason. The observer is the only thing that knows: the
		//    loop cannot tell a user cancel from held-out deterioration from an
		//    expired probe window, and collapsing them once made a cancelled
		//    trial indistinguishable from a successful early stop.
		//    Default = STOP_CANCELLED, the conservative reading (an outside
		//    request). An observer that stops for its own reason overrides this
		//    and MUST NOT report a convergence reason it did not observe.
		virtual StopReason whyStopped() const { return STOP_CANCELLED; }
	};

	// THE machine-readable name of a stop reason -- one spelling, used by every
	//    report, JSON payload, and artifact so a caller never meets two names for
	//    the same outcome (rule 6).
	static const char* stopReasonToken( StopReason r );

	// Did the run end because a stopping RULE fired? This is THE convergence
	//    predicate for the whole engine: the training report, OBD's trial
	//    eligibility, and the CV adapters all ask it, so "converged" cannot come
	//    to mean different things in different layers.
	//    True for MIN_ERROR / CHANGE / WINDOW / GRADMAX / PLATEAU / EARLY_STOP.
	//    False for MAX_ITERATIONS (a safety ceiling, not a stopping rule -- the
	//    weights are wherever the run happened to be), CANCELLED, PROBE_BUDGET
	//    and NONE. A caller that must also reject a non-finite loss checks that
	//    separately; a reason alone cannot see the numbers.
	static bool converged( StopReason r );

	Iterative(); // default constructor
	virtual ~Iterative(); // destructor

	// Copy constructor
	Iterative( const Iterative& rhs );

	// Overloaded = operator
	Iterative& operator= ( const Iterative& rhs );

	// Gets & sets maximum number of iterations through training set
	void setMaxIterations( const unsigned );
	// Gets maximum number of iterations through training set
	unsigned getMaxIterations() { return maxIterations; }

	// Gets & sets if minimum error will be used, and value
	void setMinStop( const bool flag ) { minStopFlag = flag; } // sets if will be used
	bool getMinStop() { return minStopFlag; } // returns flag indicating if will be used
	void setMinError( const double ); // sets value
	double getMinError() { return minError; } // returns value
	
	// Gets & sets if training stops after error increases over an iteration
	void setChangeStop( const bool flag ) { changeStopFlag = flag; } // sets if will be used
	bool getChangeStop() { return changeStopFlag; } // returns flag indicating if will be used
	void setChange( const double ); // sets change value
	double getChange() { return delta; } // returns change value

	// Gets & sets if training stops when grad max decreases below threshold
	void setGradStop( const bool flag ) { gradMaxFlag = flag; } // sets if will be used
	bool getGradStop() { return gradMaxFlag; } // returns flag indicating if will be used
	void setGradMaxLimit( const double ); // sets grad max limit
	double getGradMaxLimit() { return gradMaxLimit; } // returns grad max limit

	// Gets & sets if training stops after error increases over a window
	void setWindowStop( const bool flag ) { windowStopFlag = flag; } // sets if will be used
	bool getWindowStop() { return windowStopFlag; } // returns flag indicating if will be used
	void setWindow( const unsigned ); // sets window width
	unsigned getWindow() { return window; } // returns window width

	// Gets & sets auto-stop: stop when the training error plateaus (stops
	//    improving) over a two-window moving average. Default OFF, so a run
	//    without it is bit-identical to one before the feature (goldens' rule).
	//    See src/plateau.h for the detector. tol is the relative-improvement
	//    threshold; win is each averaging window's width.
	void setAutoStop( const bool flag, const double tol, const unsigned win );
	bool getAutoStop() { return autoStopFlag; }
	double getAutoStopTol() { return autoStopTol; }
	unsigned getAutoStopWindow() { return autoStopWindow; }

	// The observer is NOT owned and NOT copied: a clone (RegressNet's working
	//    copies) must never drive its original's GUI buffers, so copy() nulls it
	void setObserver( Observer* obs ) { observerPtr = obs; }
	Observer* getObserver() { return observerPtr; }
	StopReason getStopReason() { return stopReason; }
	// How many iterations the last train() ran. A caller that has to explain
	//    WHY it is rejecting a fit needs the count alongside the reason ("ended
	//    as max_iterations after 2000 iterations"); the number was previously
	//    reachable only by parsing the printed report.
	unsigned getIterations() { return iteration; }

	// A QUIET run produces no narration: no run header, no per-iteration rows,
	//    and no epilogue (elapsed time, convergence warning, accuracy report,
	//    last-operation file, history append). It exists for fits whose OUTPUT
	//    nobody consumes -- stepwise regression's candidate subnetworks, where
	//    only the training error and the stop reason feed the Wilks comparison,
	//    while the epilogue was re-deriving classification tables, ROC fits and
	//    a 2000-resample bootstrap on the TEST set for every candidate. That is
	//    both a large avoidable cost and a repeated look at a set that is
	//    supposed to be untouched during model selection.
	//
	//    QUIET AFFECTS OUTPUT ONLY. It must never guard a calculation a
	//    stopping rule depends on -- that was legacy bug #10, where the maximum
	//    gradient was computed inside the block that PRINTS a row, so the print
	//    schedule chose the model. The gradient calculation deliberately sits
	//    outside every reporting guard for exactly this reason; keep it there.
	//    Default OFF, so an ordinary run is bit-identical to one before this
	//    existed (the goldens' rule). NOT copied: a clone must not inherit
	//    silence by accident -- callers that want a quiet candidate say so.
	void setQuiet( const bool flag ) { quietFlag = flag; }
	bool getQuiet() { return quietFlag; }

	// Specifies the type of print counter, and if applicable, its value
	void setLogPrint( const bool flag ) { logPrintFlag = flag; } // sets if log printing used
	bool getLogPrint() { return logPrintFlag; } // returns flag indicating if log printing used
	// If log printing is *not* used, these specify the linear print count
	unsigned getPrintCount() { return printCount; } // get the linear print count
	void setPrintCount( const unsigned ); // set the linear print count

	// Load DataSet object into Iterative Model
	virtual void setDataSet( DataSet& ) = 0; // pure virtual

	// Outputs a header to ostream describing the architecture of the model
	virtual void outputHeader( ostream& ) = 0; // pure virtual

	// Trains an Iterative Model, returns the final error
	virtual double train();

	// Trains one iteration through the training set, Model type dependant
	//    returns set error
	virtual double trainSet() = 0; // pure virtual

	// Outputs to ostream reporting the accuracy of the Model
	virtual void reportAccuracy( ostream& ) = 0; // pure virtual

	// Outputs classification accuracies to ostream for Iterative table
	virtual void classAccuracy( ostream& ) = 0; // pure virtual

protected:
	unsigned iteration, // the iteration counter
		maxIterations, // maximum number of iterations through training set
		printCount, // linear print counter
		window; // window width over which error is detected to increase

	bool minStopFlag, // flag indicating if minimum error used
		windowStopFlag, // flag indicating if training stops if error increases over window
		changeStopFlag, // flag indicating if change over 1 iteration used to stop training
		gradMaxFlag, // flag indicating if training stops with a maximum gradient
		autoStopFlag, // flag indicating if plateau auto-stop used
		logPrintFlag, // flag indicating if log printing used
		boundsErrorFlag; // flag indicating if out of bounds error encountered

	double minError, // minimum error value
		delta, // change in error value
		gradMaxLimit, // limit of maximum gradient
		autoStopTol; // relative-improvement threshold for plateau auto-stop

	unsigned autoStopWindow; // averaging-window width for plateau auto-stop
	static const unsigned autoStopPatience = 3; // consecutive strikes to stop

	Observer* observerPtr; // non-owning; see setObserver
	bool quietFlag; // suppress ALL narration; see setQuiet. Output only.
	StopReason stopReason; // why the last train() ended

	// Copy utility
	void copy( const Iterative& rhs );

	// Utility to output parameters specific to objects derived from Iterative
	//    preceeding an Iterative run
	virtual void runHeader( ostream& ) = 0; // pure virtual

	// Returns maximum gradient of derived object
	virtual double getGradMax() = 0; // pure virtual
};

#endif
