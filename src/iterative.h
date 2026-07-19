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
	};

	// Why the last train() ended. Set at every exit from the training loop;
	//    printed output is unchanged (each reason's message predates this).
	enum StopReason {
		STOP_NONE, // train() has not run
		STOP_MAX_ITERATIONS, // the iteration budget ran out
		STOP_MIN_ERROR, // error fell below the minimum
		STOP_CHANGE, // change in error over 1 iteration fell below delta
		STOP_WINDOW, // error increased over the window
		STOP_GRADMAX, // maximum absolute gradient fell below the limit
		STOP_PLATEAU, // the error stopped improving (auto-stop, ROADMAP 2 Ph3)
		STOP_OBSERVER // the observer said stop (GUI cancel)
	};

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
