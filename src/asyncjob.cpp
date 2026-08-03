#include "asyncjob.h"

using std::function;
using std::lock_guard;
using std::mutex;
using std::string;
using std::thread;

bool AsyncJob::start( function< string() > body )
{
	// Reap the previous, finished run's thread. Assigning a std::thread over a
	//    joinable one calls std::terminate, so this is not tidiness -- it is
	//    the difference between a second run and a dead server.
	//
	// It happens BEFORE progressMutex is taken and never while holding it: the
	//    outgoing worker's last act before returning is to take that same
	//    mutex to publish (see finish), so joining under it would deadlock.
	if ( worker.joinable() )
		worker.join();

	{
		lock_guard< mutex > lock( progressMutex );
		resetForNewRun();
	}

	// Armed before the worker exists, so a caller that has returned from
	//    start() is guaranteed to find the engine already owned. The GUI's
	//    busy gate reads `running` under the engine lock this caller holds,
	//    which is what keeps every other handler out from here on.
	cancel = false;
	running = true;

	try
	{
		worker = thread( [ this, body ] { runBody( body ); } );
	}
	catch ( ... )
	{
		// `running` was set above and nothing else will ever clear it: without
		//    this the engine stays owned forever, every request is refused as
		//    busy, and Stop answers "stopping" while nothing is running. Leave
		//    a coherent terminal state instead, through the SAME
		//    publish-then-clear every finished run uses.
		//
		// The trigger -- std::thread's constructor failing -- cannot be forced
		//    portably, so this catch's wiring is inspected rather than tested;
		//    what finish() does is exercised by every job that ever runs.
		finish( renderFailure( "the run could not be started" ) );
		return false;
	}

	return true;
}

void AsyncJob::runBody( const function< string() >& body )
{
	string published;

	try
	{
		published = body();
	}
	catch ( const std::exception& e )
	{
		// The renderer escapes its own message -- do not escape it twice
		published = renderFailure( string( "the run failed: " ) + e.what() );
	}
	catch ( ... )
	{
		published = renderFailure( "the run failed with an unrecognized error" );
	}

	finish( published );
}

void AsyncJob::finish( const string& published )
{
	{
		lock_guard< mutex > lock( progressMutex );
		result = published; // publish BEFORE running goes false
	}
	running = false;
}

bool AsyncJob::requestStop()
{
	if ( !running )
		return false;
	cancel = true;
	return true;
}

void AsyncJob::joinForShutdown()
{
	if ( worker.joinable() )
	{
		cancel = true;
		worker.join();
	}
}

void AsyncJob::clearCvProgress()
{
	cvStage.clear(); cvProcedure.clear();
	cvProcIndex = cvProcCount = cvProcDone = 0;
	cvFold = cvK = cvFoldsDone = 0;
	cvInnerPhase.clear(); cvInnerHidden = 0;
}

void AsyncJob::resetForNewRun()
{
	iters.clear();
	trainErr.clear();
	testErr.clear();
	keepEvery = 1;
	sampleCounter = 0;
	result.clear();
	obdPhase.clear();
	obdHidden = 0;
	stepwise.clear();
	clearCvProgress();
}

void AsyncJob::pushSample( unsigned iteration, double trainError,
	double testError )
{
	lock_guard< mutex > lock( progressMutex );
	if ( ++sampleCounter % keepEvery == 0 )
	{
		iters.push_back( iteration );
		trainErr.push_back( trainError );
		testErr.push_back( testError );
		if ( iters.size() >= MAX_POINTS )
		{
			// Halve the series, double the stride for future samples
			for ( unsigned i = 0, j = 0; j < iters.size(); i++, j += 2 )
			{
				iters[ i ] = iters[ j ];
				trainErr[ i ] = trainErr[ j ];
				testErr[ i ] = testErr[ j ];
			}
			iters.resize( iters.size() / 2 );
			trainErr.resize( trainErr.size() / 2 );
			testErr.resize( testErr.size() / 2 );
			keepEvery *= 2;
		}
	}
}
