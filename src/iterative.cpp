// Methods for Iterative, the abstract base class for neUROn2++ Models
//    that require iterative algorithms for training

#include "stdafx.h" // For MSVC, must be first!

#include "iterative.h"

#include <deque> // for error window

// #define REGRESS_DEBUG // turns a lot of the printing off for stepwise regression debugging

// Default constructor with initial conditions
Iterative::Iterative() : maxIterations ( 1000000 ), printCount ( 1000 ),
	window ( 1000 ), minStopFlag ( false ), windowStopFlag ( false ),
	changeStopFlag ( false ), gradMaxFlag ( true ), autoStopFlag ( false ),
	logPrintFlag ( true ), boundsErrorFlag ( false ), minError ( 1e-30 ),
	delta ( 1e-16 ), gradMaxLimit ( 1e-6 ), autoStopTol ( 1e-4 ),
	autoStopWindow ( 100 ), observerPtr ( nullptr ), quietFlag ( false ),
	stopReason ( STOP_NONE ) { }

// Publish a stop and record its reason.
//
//    The CALLER has already composed its own message into screenStream: each
//    stopping condition states itself, in its own words and its own stream
//    formatting, and this must never become a table of messages. Nor does this
//    decide whether a condition fired -- the formulae and their order stay
//    written out in train(), where they can be read against the manual.
//
//    What repeated six times, and is here instead, is only the publication: a
//    quiet run says nothing at all -- not even why it stopped, its caller
//    reporting the reason itself from getStopReason() -- while an audible run
//    writes the same text to the history file and the screen. Either way the
//    reason is recorded.
//
//    The streams are train()'s locals, so they are passed rather than reached
//    for. Guarded by tests/iterative/check_announcestop.cpp, which pins the
//    reason, its token, converged(), the exact text and the quiet contract for
//    every exit.
void Iterative::announceStop( StopReason why, ostringstream& screenStream,
	ostringstream& fileStream )
{
	if ( !quietFlag )
	{
		fileStream << screenStream.str(); // stream line into file stream
		util::screen() << screenStream.str(); // then print to screen
	}

	stopReason = why;
}

const char* Iterative::stopReasonToken( StopReason r )
{
	switch ( r )
	{
	case STOP_MAX_ITERATIONS: return "max_iterations"; // NOT convergence
	case STOP_MIN_ERROR: return "min_error";
	case STOP_CHANGE: return "min_change";
	case STOP_WINDOW: return "error_window";
	case STOP_GRADMAX: return "grad_max";
	case STOP_PLATEAU: return "plateau";
	case STOP_CANCELLED: return "cancelled";
	case STOP_EARLY_STOP: return "validation_early_stop";
	case STOP_PROBE_BUDGET: return "probe_budget";
	default: return "none";
	}
}

bool Iterative::converged( StopReason r )
{
	switch ( r )
	{
	case STOP_MIN_ERROR:
	case STOP_CHANGE:
	case STOP_WINDOW:
	case STOP_GRADMAX:
	case STOP_PLATEAU:
	case STOP_EARLY_STOP:
		return true;
	default: // MAX_ITERATIONS, CANCELLED, PROBE_BUDGET, NONE
		return false;
	}
}

// Default destructor
Iterative::~Iterative() { }

// Copy constructor
Iterative::Iterative( const Iterative& rhs )
{
	Iterative::copy( rhs ); // use the copy utility
}

// Overloaded = operator
Iterative& Iterative::operator= ( const Iterative& rhs )
{
	if ( &rhs != this ) // check for self-assignment
		Iterative::copy( rhs ); // use the copy utility
	
	return *this; // enables A = B = C
}

// Copy utility
void Iterative::copy( const Iterative& rhs )
{
	Model::copy( rhs ); // call immediate base object copy
	iteration = rhs.iteration;
	maxIterations = rhs.maxIterations;
	printCount = rhs.printCount;
	window = rhs.window;
	minStopFlag = rhs.minStopFlag;
	windowStopFlag = rhs.windowStopFlag;
	changeStopFlag = rhs.changeStopFlag;
	gradMaxFlag = rhs.gradMaxFlag;
	autoStopFlag = rhs.autoStopFlag;
	logPrintFlag = rhs.logPrintFlag;
	boundsErrorFlag = rhs.boundsErrorFlag;
	minError = rhs.minError;
	delta = rhs.delta;
	gradMaxLimit = rhs.gradMaxLimit;
	autoStopTol = rhs.autoStopTol;
	autoStopWindow = rhs.autoStopWindow;
	stopReason = rhs.stopReason;
	// Deliberately NOT copied: a clone (RegressNet's working copies, the
	//    coming autoalgo probes) must never drive its original's observer
	observerPtr = nullptr;
	// Likewise not copied -- and "not copied" has to be WRITTEN, not omitted.
	//    The copy constructor calls this without running the default
	//    constructor first, so a member left untouched here holds an
	//    indeterminate value, not false. Omitting it would let an autoalgo
	//    probe, an OBD trial or a CV clone silently suppress its entire
	//    reporting epilogue depending on what was on the stack. That is the
	//    same failure as the uninitialised Model::errorType scalar behind the
	//    nested-OBD flake; a clone starts audible and says so explicitly.
	quietFlag = false;
}

// Sets the maximum number of iterations through training set
void Iterative::setMaxIterations( const unsigned n )
{
	assert ( n > 0 ); // make sure iterations nonzero
	maxIterations = n; // set the private member maxIterations
}

// Sets the linear print counter
void Iterative::setPrintCount( const unsigned n )
{
	assert ( n > 0 ); // make sure linear print counter nonzero
	printCount = n; // set the private member printCount
}

// Sets the minimum error
void Iterative::setMinError( const double e )
{
	assert ( e > 0 && e < 1 ); // bounds check incoming error
	minError = e; // set the private member minError
}

// Sets the change in error over 1 iteration
void Iterative::setChange( const double d )
{
	assert ( d > 0 && d < 1 ); // bounds check incoming error
	delta = d; // set the private member minError
}

// Sets maximum gradient
void Iterative::setGradMaxLimit( const double x )
{
	assert ( x > 0 ); // bounds check incoming error
	gradMaxLimit = x; // set the private member gradMax
}

// Sets window width over which error is detected to increase
void Iterative::setWindow( const unsigned n )
{
	assert ( n > 1 ); // window must be at least 1 iteration
	window = n; // set the private member window
}

// Sets plateau auto-stop: whether it is used, the relative-improvement
//    tolerance, and the averaging-window width (see src/plateau.h)
void Iterative::setAutoStop( const bool flag, const double tol,
	const unsigned win )
{
	assert ( tol > 0 && tol < 1 ); // relative improvement is a fraction
	assert ( win >= 2 ); // each averaging window needs at least 2 points
	autoStopFlag = flag;
	autoStopTol = tol;
	autoStopWindow = win;
}

// Trains the model, returns the final error
double Iterative::train()
{
	unsigned start = time( 0 ), // set start time
		logCounter = 1, // counter for logarithmic printing
		logMultiplier = 1, // multiplier for logarithmic printing
		logIncrement = 1; // incrementer for logarithmic printing

	double setError = -1, // error through training set, if method returns -1,
	                      //    something is wrong
		lastError = setError, // placeholder for last iteration's error value
		gradMaxValue = 1e10; // maximum absolute gradient of the iteration just
		                     //    finished; refreshed every iteration when the
		                     //    rule is armed (see below). The huge initial
		                     //    value is only a belt-and-braces guard against
		                     //    stopping before a single iteration has run.

	boundsErrorFlag = false; // new training resets bounds error flag

	// If the loop below runs out, that IS the stop reason; every earlier
	//    exit overwrites this at its break
	stopReason = STOP_MAX_ITERATIONS;

	// PREPARE the run before anything else, and OUTSIDE every reporting guard
	//    (the quiet branch below, and the REGRESS_DEBUG switch). Network derives
	//    its weight-decay constants here; those are training inputs, and until
	//    2026-08-01 they were computed inside the quiet branch, which meant a
	//    quiet run trained on uninitialised memory. See Iterative::prepareRun.
	prepareRun();

	deque< double > errorsWindow; // window of error values, use deque for speed

	// Plateau auto-stop detector (inert unless autoStopFlag; constructed from
	//    the copied config so a clone trains with the same settings). Local,
	//    like errorsWindow -- its state is per-run, so it stays off the class.
	PlateauDetector plateau( autoStopWindow, autoStopTol, autoStopPatience );

	// For reporting to screen and file
	ostringstream screenStream, fileStream;

#ifndef REGRESS_DEBUG
	fileStream.str( "" ); // just in case, reset file stream

	// A quiet run (a stepwise candidate refit) skips the whole run header.
	//    Note what is NOT skipped: everything below the reporting blocks --
	//    the training loop, every stopping rule, and the gradient calculation
	//    the gradient rule reads. Quiet changes what a run SAYS, never what it
	//    computes or where it stops.
	if ( !quietFlag )
	{
	screenStream << "I'm running an iterative model:" <<'\r'<< endl;
	outputHeader( screenStream ); // output the header identifying the model object

	if ( theData.trainLoaded() ) // output number of exemplars in training set
		screenStream << "Number of exemplars in training set: "
			<< theData.getTrainMatrix().rows() << endl;

	if ( theData.testLoaded() ) // output number of exemplars in test set
		screenStream << "Number of exemplars in test set: "
			<< theData.getTestMatrix().rows() << endl;

	// Output parameters specific to the object derived from Iterative
	runHeader( screenStream );

	// Output error checking routines in effect
	screenStream << "Error checking routines in effect:" << endl;
	screenStream << "Maximum number of iterations: " << maxIterations << endl;

	if ( minStopFlag ) // error less than minimum check
		screenStream << "Stop if error becomes less than " << minError << endl;

	if ( changeStopFlag ) // change in error check
		screenStream << "Stop if change in error over 1 iteration becomes less than "
			<< delta << endl;

	if ( windowStopFlag ) // error increases over a window
		screenStream << "Stop if error increases over a window of " << window
			<< " iterations" << endl;

	if ( gradMaxFlag ) // error increases over a window
		screenStream << "Stop if maximum absolute gradient decreases below "
			<< gradMaxLimit << endl;

	if ( autoStopFlag ) // error plateaus (auto-stop)
		screenStream << "Stop if the error plateaus over a window of "
			<< autoStopWindow << " iterations" << endl;

	// Output header for printout to follow, then finish off screen stream for now
	screenStream << endl << "    Iteration:";
	// Because g++ does not handle setiosflags::right for strings :(
	for ( unsigned spaces = 0; spaces < ( 10 - errorLabel.size() ); spaces++ )
		screenStream << " ";
	screenStream << errorLabel + " error:";

	// If grad max is to be output
	if ( gradMaxFlag )
		screenStream << "  Max abs grad:";

	// If outputs are discrete, add header for classification accuracy table entries
	if ( theData.getDiscrete() )
	{
		if ( theData.trainLoaded() )
			screenStream << "  CA Train %";
		
		if ( theData.testLoaded() )
			screenStream << "   CA Test %";
	}

	screenStream << endl; // finish off stream for now

	fileStream << screenStream.str(); // stream line into file stream
	util::screen() << screenStream.str(); // then print to screen

	// Format the ostream
	screenStream << setiosflags( ios::showpoint | ios::right );
	} // end of the run header (skipped entirely by a quiet run)
#endif

	// Iterate to maximum number of iterations
	for ( iteration = 0; iteration <= maxIterations; iteration++ )
	{
		setError = trainSet(); // train once through training set, return set error

		// THE CURRENT maximum absolute gradient, of the iteration just finished.
		//    It is calculated HERE, once, on every iteration -- not inside the
		//    printing block below, where it lived until 2026-07-26.
		//
		//    STOPPING CONDITIONS ARE EVALUATED INDEPENDENTLY OF REPORTING
		//    CADENCE. Reporting may change how much output a run produces; it
		//    must never change the optimization or the validity of the fit.
		//    With the calculation inside the print block, the check below
		//    compared a value cached at the last PRINTED iteration: logarithmic
		//    printing checks at 1..10, then 20, then 30, so a crossing in
		//    between stayed invisible until the next printed iteration, and a
		//    ceiling landing in that gap reported a false failure to converge.
		//    Switching linear/logarithmic printing, or changing printcount,
		//    moved the stopping iteration, the final weights, the predictions,
		//    and whether OBD/CV would accept the fit (tests/iterative).
		//
		//    Guarded by gradMaxFlag, which arms both the rule and the printed
		//    column: a run with gradient stopping off does no extra work and is
		//    bit-identical to one before this existed (the goldens' rule). It
		//    sits OUTSIDE the REGRESS_DEBUG guard because a stopping rule is not
		//    debug output -- under that switch the value was never refreshed at
		//    all, so the rule could not fire.
		if ( gradMaxFlag )
			gradMaxValue = getGradMax();

#ifndef REGRESS_DEBUG
		// Print if print counter reached, first condition tests linear print counter,
		//    second condition tests logarithmic print counter
		//    A quiet run prints no rows at all. This is a REPORTING condition
		//    only: the gradient above is already calculated, and every stop
		//    check below runs regardless (legacy bug #10).
		if ( !quietFlag && ( ( !logPrintFlag && iteration % printCount == 0 )
			|| ( logPrintFlag && ( iteration == logCounter ) ) ) )
		{
			screenStream.str( "" ); // reset screen stream

			// Prepare line for printing to screen, first the iteration
			screenStream << setw( 14 ) << setfill( ' ' ) << iteration << "  ";
			// Then the training set error
			screenStream << resetiosflags( ios::fixed )
				<< setiosflags( ios::scientific );
			screenStream << setw( 15 ) << setfill( ' ' ) << setprecision( 6 )
				<< setError;

			// If grad max is to be output -- the SAME value the stop check
			//    below uses, calculated once above. Never recalculate it here:
			//    a printed row that disagrees with the decision it accompanies
			//    is a report of a run that did not happen.
			if ( gradMaxFlag )
				screenStream << setw( 15 ) << setfill( ' ' ) << setprecision( 6 )
					<< gradMaxValue;

			if ( theData.getDiscrete() ) // if outputs are discrete
				classAccuracy( screenStream ); // add entries for class accuracy

			screenStream << endl; // finish off stream for now

			fileStream << screenStream.str(); // stream line into file stream
			util::screen() << screenStream.str(); // then print to screen

			// If logarithmic counting requested, and logarithmic counter reached,
			if ( logPrintFlag && ( iteration == logCounter ) )
			{
				logIncrement++; // increment the logarithmic incrementer
				logCounter = logIncrement * logMultiplier; // calculate the counter
				if ( logIncrement == 10 ) // if the incrementer is 10
				{
					logIncrement = 1; // reset the incrementer
					logMultiplier *= 10; // and advance the multiplier
				}
			}
		}
#endif

		// Exit if error less than minimum
		if ( minStopFlag )
			if ( setError < minError )
			{
				screenStream.str( "" ); // reset screen stream

				// Prepare line for printing to screen
				screenStream << "The error became lower than " << minError
					<< "." << endl;

				announceStop( STOP_MIN_ERROR, screenStream, fileStream );
				break;
			}

		// Exit if change in error less than specified delta value
		if ( changeStopFlag )
		{
			if ( ( setError < lastError ) && ( ( lastError - setError ) < delta ) )
			{
				screenStream.str( "" ); // reset screen stream
				
				// Prepare line for printing to screen
				screenStream << "The change in error became lower than " << delta
					<< "." << endl;

				announceStop( STOP_CHANGE, screenStream, fileStream );
				break;
			}
			else
				lastError = setError;
		}

		// Exit if error increases over window
		if ( windowStopFlag )
		{
			if ( iteration <= window ) // if window not yet populated,
				errorsWindow.push_back( setError ); // then append new set error
			else
			{
				if ( setError > *( errorsWindow.begin() ) ) // set error increased
				{
					screenStream.str( "" ); // reset screen stream
					
					// Prepare line for printing to screen
					screenStream << "The error increased over the window width of "
						<< window << "." << endl;

					announceStop( STOP_WINDOW, screenStream, fileStream );
					break;
				}
				else // set error did not increase, so advance window
				{
					errorsWindow.pop_front(); // remove 1st set error
					errorsWindow.push_back( setError ); // append new set error
				}
			}
		}

		// Exit if maximum absolute gradient decreases below limit
		if ( gradMaxFlag )
		{
			if ( gradMaxValue < gradMaxLimit )
			{
				screenStream.str( "" ); // reset screen stream

				// Prepare line for printing to screen
				screenStream << "The maximum absolute gradient became lower than "
					<< resetiosflags( ios::fixed ) << setiosflags( ios::scientific )
					<< gradMaxLimit << "." << endl;

				announceStop( STOP_GRADMAX, screenStream, fileStream );
				break;
			}
		}

		// Exit if the error has plateaued (stopped improving). The detector
		//    must see every iteration to accumulate, so update() is called
		//    whenever the flag is on -- but only then, so a run with auto-stop
		//    off never touches it and stays bit-identical (goldens' rule).
		if ( autoStopFlag && plateau.update( setError ) )
		{
			screenStream.str( "" ); // reset screen stream

			// Prepare line for printing to screen
			screenStream << "The error plateaued over a window of "
				<< autoStopWindow << " iterations." << endl;

			announceStop( STOP_PLATEAU, screenStream, fileStream );
			break;
		}

		// Give the observer its look at the finished iteration -- at the
		//    BOTTOM, after every stop check above, so a run with no observer
		//    is bit-identical to one before the hook existed
		if ( observerPtr && !observerPtr->onIteration( iteration, setError ) )
		{
			screenStream.str( "" ); // reset screen stream

			// The observer names its own reason -- a cancel, a validation early
			//    stop, and an expired probe window are different facts and must
			//    not be reported alike. Only observer-driven runs can print any
			//    of these, so transcripts without one are unchanged.
			StopReason why = observerPtr->whyStopped();

			if ( why == STOP_EARLY_STOP )
				screenStream << "Held-out error deteriorated: stopped early." << endl;
			else if ( why == STOP_PROBE_BUDGET )
				screenStream << "The probe's time budget expired." << endl;
			else
				screenStream << "Training was stopped by request." << endl;

			announceStop( why, screenStream, fileStream );
			break;
		}
	}

	screenStream.str( "" ); // reset screen stream

#ifndef REGRESS_DEBUG
	// The whole epilogue is reporting, and a quiet run skips all of it. This is
	//    where the cost lives: reportAccuracy() below re-derives the
	//    classification tables, the ROC fit and a 2000-resample bootstrap over
	//    the TEST set. Stepwise regression called it once per candidate
	//    subnetwork and used none of it -- Wilks selection reads the TRAINING
	//    error, which train() has already returned. Skipping it cannot change
	//    the fit: the weights are final, and the bootstrap draws from the
	//    reserved i_resample stream, so it can never perturb weight init or
	//    splits (a settled decision that this relies on).
	if ( quietFlag )
		return setError;

	// Print number of iterations and elapsed time
	unsigned elapsed_time = time( 0 ) - start;
	screenStream << endl << "Total iterations = " << iteration << endl <<
		"That took " << timestamp( elapsed_time ) << endl << endl;

	fileStream << screenStream.str(); // stream line into file stream
	util::screen() << screenStream.str(); // then print to screen

	// Reaching the iteration ceiling is a FAILURE TO CONVERGE, not a stopping
	//    condition: the weights are wherever the run happened to be. Say so in
	//    the authoritative report, so the CLI transcript, neuron.log, a captured
	//    report and the GUI all carry the same plain statement -- a caller must
	//    not have to infer it from a JSON field. Runs that end on a real stopping
	//    rule print nothing here, so a converged transcript is unchanged.
	if ( !converged( stopReason ) )
	{
		screenStream.str( "" );
		if ( stopReason == STOP_MAX_ITERATIONS )
			screenStream << "WARNING: training did NOT converge. It stopped at the "
				<< "maximum of " << maxIterations << " iterations, which is a safety "
				<< "limit, not a stopping condition." << endl
				<< "         The weights are kept, so training can be continued from "
				<< "here, but this is not a fitted model:" << endl
				<< "         raise the maximum iterations, or set a stopping "
				<< "condition that can fire." << endl << endl;
		else
			// Cancellation and an expired probe window already printed WHY the
			//    loop ended, at the break above. What they have not said is what
			//    it means: no stopping rule fired, so this is not a fitted model.
			screenStream << "WARNING: training did NOT converge -- it ended before "
				<< "any stopping condition fired (" << stopReasonToken( stopReason )
				<< ")." << endl
				<< "         The weights are kept and can be trained further, but "
				<< "this is not a fitted model." << endl << endl;

		fileStream << screenStream.str();
		util::screen() << screenStream.str();
	}

	// Print warning message if out of bounds error
	if ( boundsErrorFlag )
	{
		screenStream.str( "" ); // reset screen stream
		screenStream << "WARNING: Numerical out of bounds encountered when calculating error"
			<< endl;
		fileStream << screenStream.str(); // stream line into file stream
		util::screen() << screenStream.str(); // then print to screen
	}

	// Print accuracy report for trained Iterative Model
	screenStream.str( "" ); // reset screen stream
	reportAccuracy( screenStream ); // report the accuracy
	screenStream << endl; // end the stream
	fileStream << screenStream.str(); // stream line into file stream
	util::screen() << screenStream.str(); // then print to screen

	writeLastop( fileStream.str() );

	addHistory( fileStream ); // append to history file if specified by flag
#endif

	return setError; // return the final set error
}
