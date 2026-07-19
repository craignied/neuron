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
	autoStopWindow ( 100 ), observerPtr ( nullptr ), stopReason ( STOP_NONE ) { }

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
		gradMaxValue = 1e10; // maximum absolute gradient value (initialized to huge)

	boundsErrorFlag = false; // new training resets bounds error flag

	// If the loop below runs out, that IS the stop reason; every earlier
	//    exit overwrites this at its break
	stopReason = STOP_MAX_ITERATIONS;

	deque< double > errorsWindow; // window of error values, use deque for speed

	// Plateau auto-stop detector (inert unless autoStopFlag; constructed from
	//    the copied config so a clone trains with the same settings). Local,
	//    like errorsWindow -- its state is per-run, so it stays off the class.
	PlateauDetector plateau( autoStopWindow, autoStopTol, autoStopPatience );

	// For reporting to screen and file
	ostringstream screenStream, fileStream;

#ifndef REGRESS_DEBUG
	fileStream.str( "" ); // just in case, reset file stream

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
#endif

	// Iterate to maximum number of iterations
	for ( iteration = 0; iteration <= maxIterations; iteration++ )
	{
		setError = trainSet(); // train once through training set, return set error

#ifndef REGRESS_DEBUG
		// Print if print counter reached, first condition tests linear print counter,
		//    second condition tests logarithmic print counter
		if ( ( !logPrintFlag && iteration % printCount == 0 ) || ( logPrintFlag
			&& ( iteration == logCounter ) ) )
		{
			screenStream.str( "" ); // reset screen stream

			// Prepare line for printing to screen, first the iteration
			screenStream << setw( 14 ) << setfill( ' ' ) << iteration << "  ";
			// Then the training set error
			screenStream << resetiosflags( ios::fixed )
				<< setiosflags( ios::scientific );
			screenStream << setw( 15 ) << setfill( ' ' ) << setprecision( 6 )
				<< setError;

			// If grad max is to be output
			if ( gradMaxFlag )
			{
				gradMaxValue = getGradMax();
				screenStream << setw( 15 ) << setfill( ' ' ) << setprecision( 6 )
					<< gradMaxValue;
			}

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

				fileStream << screenStream.str(); // stream line into file stream
				util::screen() << screenStream.str(); // then print to screen

				stopReason = STOP_MIN_ERROR;
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

				fileStream << screenStream.str(); // stream line into file stream
				util::screen() << screenStream.str(); // then print to screen

				stopReason = STOP_CHANGE;
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

					fileStream << screenStream.str(); // stream line into file stream
					util::screen() << screenStream.str(); // then print to screen

					stopReason = STOP_WINDOW;
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

				fileStream << screenStream.str(); // stream line into file stream
				util::screen() << screenStream.str(); // then print to screen

				stopReason = STOP_GRADMAX;
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

			fileStream << screenStream.str(); // stream line into file stream
			util::screen() << screenStream.str(); // then print to screen

			stopReason = STOP_PLATEAU;
			break;
		}

		// Give the observer its look at the finished iteration -- at the
		//    BOTTOM, after every stop check above, so a run with no observer
		//    is bit-identical to one before the hook existed
		if ( observerPtr && !observerPtr->onIteration( iteration, setError ) )
		{
			screenStream.str( "" ); // reset screen stream

			// Prepare line for printing to screen (only observer-driven runs
			//    can ever print this, so transcripts without one are unchanged)
			screenStream << "Training was stopped by request." << endl;

			fileStream << screenStream.str(); // stream line into file stream
			util::screen() << screenStream.str(); // then print to screen

			stopReason = STOP_OBSERVER;
			break;
		}
	}

	screenStream.str( "" ); // reset screen stream

#ifndef REGRESS_DEBUG
	// Print number of iterations and elapsed time
	unsigned elapsed_time = time( 0 ) - start;
	screenStream << endl << "Total iterations = " << iteration << endl <<
		"That took " << timestamp( elapsed_time ) << endl << endl;

	fileStream << screenStream.str(); // stream line into file stream
	util::screen() << screenStream.str(); // then print to screen

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

	if ( lastopFlag ) // make sure flag for last operation output is set
	{
		// Open last operation file for output, overwrite if it exists
		string logPath = util::run_path( lastopFilename );
		ofstream lastopFile( logPath.c_str(), ios::out | ios::trunc );

		if ( !lastopFile.is_open() ) // test to insure it was opened
			util::screen() << "Error in opening " << logPath << "!" << endl;
		else
		{
			lastopFile << fileStream.str(); // write the file stream to the file 
			lastopFile.close(); // and close the file
		}
	}

	addHistory( fileStream ); // append to history file if specified by flag
#endif

	return setError; // return the final set error
}
