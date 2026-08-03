// check_asyncjob.cpp : the asynchronous-job lifecycle and the process
// boundary, as deterministic in-process contracts.
//
// WHY THIS FILE EXISTS. tests/gui/asyncjob.sh characterizes the same machinery
// through the live server, and it cannot see the two invariants that matter
// most. Measured, not assumed (refactor_audit.md section 20.4): inverting
// publish-then-clear passes that suite 3/3 with one observer and 3/3 with
// eight, and setting the running flag inside the worker passes it 3/3. A
// status request holds the progress mutex for microseconds out of a round trip
// of hundreds, so mutex OCCUPANCY -- not observer count -- is the ceiling, and
// a thread starts in microseconds where the next HTTP call takes hundreds.
//
// In this process both windows are reachable, because there is no network in
// between: a reader can hold the mutex a large fraction of the time, and the
// check after start() returns happens nanoseconds after the thread was created
// rather than after a round trip. That is the whole reason the lifecycle was
// moved into a linkable unit.
//
// NO TEST-ONLY PRODUCTION API. Every case drives the real AsyncJob through its
// public contract, supplying its own body -- which is what a launcher takes
// anyway -- and reads state exactly the way the status handler does: under
// progressMutex, both fields in one critical section. procguard::run likewise
// takes a body and a stream, so a test can hand it a throwing body without the
// shipped binary gaining any way to inject one.
//
// ONE CASE PER PROCESS WHEN A CASE CAN KILL IT. Cases 9 and 10 exist because a
// missing handler in the worker boundary calls std::terminate. In all-cases
// mode a death there would hide every later case, so the driver also takes a
// case number and reports its verdict as its exit status -- the
// check_matrix_bounds precedent.
//
// NO TIMING ASSERTIONS. Deadlines appear only so that a broken contract fails
// instead of hanging CI; nothing is asserted about how long anything took.

#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "asyncjob.h"
#include "matrix.h"
#include "procguard.h"

using namespace std;

static unsigned checks = 0;
static unsigned failures = 0;

static void check( bool cond, const string& what )
{
	checks++;
	if ( !cond )
	{
		failures++;
		cout << "  FAIL: " << what << endl;
	}
}

// The renderer under test is the identity, so a case can assert the MESSAGE the
//    boundary composed rather than whatever dressing a caller puts on it. The
//    GUI's renderer wraps the same message in jsonMsg( false, ... ), which every
//    other endpoint already exercises.
static AsyncJob::FailureRenderer identityRenderer( unsigned* calls )
{
	return [ calls ]( const string& message )
	{
		if ( calls )
			( *calls )++;
		return message;
	};
}

// Liveness only: wait for the job to reach its terminal state, or give up so a
//    broken contract fails rather than hangs.
static bool waitIdle( AsyncJob& job, unsigned ms = 10000 )
{
	chrono::steady_clock::time_point end =
		chrono::steady_clock::now() + chrono::milliseconds( ms );
	while ( chrono::steady_clock::now() < end )
	{
		if ( !job.isRunning() )
			return true;
		this_thread::sleep_for( chrono::milliseconds( 1 ) );
	}
	return false;
}

// Read the two fields the ordering contract binds together, the way the status
//    handler reads them: one critical section, both values.
static void readTogether( AsyncJob& job, bool& running, bool& hasResult )
{
	lock_guard< mutex > lock( job.progressMutex );
	running = job.isRunning();
	hasResult = !job.result.empty();
}

// --------------------------------------------------------------- the cases

static int caseResultPublishedOnce()
{
	AsyncJob job( identityRenderer( 0 ) );
	atomic< int > bodyRuns( 0 );

	check( job.start( [ &bodyRuns ]
		{ bodyRuns++; return string( "the body's own answer" ); } ),
		"start() must report that the worker began" );
	check( waitIdle( job ), "the job must reach a terminal state" );

	check( bodyRuns.load() == 1, "the body must run exactly once" );
	bool running = true, hasResult = false;
	readTogether( job, running, hasResult );
	check( !running, "the job must be idle when it has finished" );
	check( hasResult, "a finished job must have published a result" );
	check( job.result == "the body's own answer",
		"the body's result must be published verbatim, got '"
			+ job.result + "'" );
	return failures ? 1 : 0;
}

static int caseReap()
{
	// Assigning a std::thread over a joinable one calls std::terminate, so a
	//    missing reap does not fail an assertion here -- it kills the process,
	//    and the case never reports. Surviving three consecutive jobs IS the
	//    assertion.
	AsyncJob job( identityRenderer( 0 ) );
	atomic< int > runs( 0 );

	for ( int i = 0; i < 3; i++ )
	{
		check( job.start( [ &runs ] { runs++; return string( "done" ); } ),
			"consecutive job must start" );
		check( waitIdle( job ), "consecutive job must finish" );
	}
	check( runs.load() == 3, "all three bodies must have run, got "
		+ to_string( runs.load() ) );
	return failures ? 1 : 0;
}

static int caseResetClearsEveryField()
{
	AsyncJob job( identityRenderer( 0 ) );

	// Dirty every per-run field the way a real job would, then start another
	//    and require all of them clear. A field added to the class and left out
	//    of resetForNewRun fails here.
	job.pushSample( 7, 0.5, 0.25 );
	{
		lock_guard< mutex > lock( job.progressMutex );
		job.result = "a previous run's result";
		job.obdPhase = "grow";
		job.obdHidden = 9;
		job.stepwise = "{\"pass\":2}";
		job.cvStage = "cross-validation";
		job.cvProcedure = "Logistic";
		job.cvProcIndex = 1; job.cvProcCount = 4; job.cvProcDone = 1;
		job.cvFold = 3; job.cvK = 5; job.cvFoldsDone = 2;
		job.cvInnerPhase = "grow";
		job.cvInnerHidden = 6;
	}
	// The control: the fields really were set, so their later absence is the
	//    reset and not a field that was never populated.
	{
		lock_guard< mutex > lock( job.progressMutex );
		check( !job.iters.empty() && !job.obdPhase.empty()
			&& !job.stepwise.empty() && !job.cvStage.empty()
			&& job.cvFold == 3,
			"control: the progress fields must be dirty before the reset" );
	}

	promise< void > release;
	shared_future< void > gate = release.get_future().share();
	check( job.start( [ gate ] { gate.wait(); return string( "next" ); } ),
		"the next job must start" );

	// Inspected while the new job is running: the reset happens at the START
	//    of a run, so this is the state a status poll would see.
	{
		lock_guard< mutex > lock( job.progressMutex );
		check( job.iters.empty() && job.trainErr.empty() && job.testErr.empty(),
			"the series must be cleared for a new run" );
		check( job.keepEvery == 1 && job.sampleCounter == 0,
			"the decimation state must be cleared for a new run" );
		check( job.result.empty(),
			"the previous run's result must not survive into this one" );
		check( job.obdPhase.empty() && job.obdHidden == 0,
			"OBD progress must not survive into another job" );
		check( job.stepwise.empty(),
			"stepwise progress must not survive into another job" );
		check( job.cvStage.empty() && job.cvProcedure.empty()
			&& job.cvProcIndex == 0 && job.cvProcCount == 0
			&& job.cvProcDone == 0 && job.cvFold == 0 && job.cvK == 0
			&& job.cvFoldsDone == 0 && job.cvInnerPhase.empty()
			&& job.cvInnerHidden == 0,
			"every CV progress field must be cleared for a new run" );
	}

	release.set_value();
	check( waitIdle( job ), "the job must finish" );
	return failures ? 1 : 0;
}

static int caseRunningMarkedBeforeStartReturns()
{
	// THE CONTRACT THE HTTP SUITE COULD NOT SEE. A caller that has returned
	//    from start() must find the engine owned -- that is what the GUI's busy
	//    gate reads, under the engine lock the launching handler still holds.
	//
	// The check happens nanoseconds after the thread was created, so a worker
	//    that marks the job itself has almost certainly not run yet. Repeated,
	//    because the assertion is SOUND on correct code (the flag is set before
	//    the thread exists, so it cannot be observed false) while an
	//    implementation that defers it is caught by the first observation that
	//    beats the scheduler.
	AsyncJob job( identityRenderer( 0 ) );
	const int rounds = 200;
	int notMarked = 0, stopNotCleared = 0;

	for ( int i = 0; i < rounds; i++ )
	{
		promise< void > release;
		shared_future< void > gate = release.get_future().share();
		job.start( [ gate ] { gate.wait(); return string( "done" ); } );

		if ( !job.isRunning() )
			notMarked++;
		if ( job.isStopRequested() )
			stopNotCleared++;

		release.set_value();
	}
	check( waitIdle( job ), "the last job must finish" );

	check( notMarked == 0,
		"start() must mark the job running before it returns; it had not on "
			+ to_string( notMarked ) + " of " + to_string( rounds )
			+ " starts" );
	check( stopNotCleared == 0,
		"start() must clear the stop latch before it returns; it had not on "
			+ to_string( stopNotCleared ) + " of " + to_string( rounds )
			+ " starts" );
	return failures ? 1 : 0;
}

static int casePublishBeforeClear()
{
	// THE OTHER CONTRACT THE HTTP SUITE COULD NOT SEE. An observer that finds
	//    the job idle must already be able to see its result.
	//
	// The reader spins with no sleep, so it holds progressMutex a large
	//    fraction of the time. Inverting the order in AsyncJob::finish leaves a
	//    window between clearing the flag and taking the mutex; a reader that
	//    occupies the mutex lands in it. Across many cycles that is no longer a
	//    lottery, which it was through HTTP.
	AsyncJob job( identityRenderer( 0 ) );
	const int cycles = 200;
	atomic< int > violations( 0 );
	atomic< int > sawRunning( 0 );

	for ( int i = 0; i < cycles; i++ )
	{
		promise< void > release;
		shared_future< void > gate = release.get_future().share();
		job.start( [ gate ] { gate.wait(); return string( "published" ); } );

		// The reader starts only once the job is running, so it can never see
		//    the legitimate idle window inside start() -- the reset happens
		//    before the job is marked running -- and report it as a violation.
		//
		// AND THE GATE IS NOT RELEASED UNTIL THE READER HAS ACTUALLY READ.
		//    Constructing the thread does not mean it has been scheduled: on
		//    Windows this case failed its own control because the body ran to
		//    completion first, so the reader's first read found the job already
		//    finished and it exited having overlapped nothing. Waiting for the
		//    reader's first observation makes the overlap structural instead of
		//    a race the fast platforms happened to win. It cannot deadlock: the
		//    body is blocked on `gate`, which only the line after this wait
		//    releases, so the job cannot finish before the reader observes it.
		promise< void > observed;
		shared_future< void > firstRead = observed.get_future().share();
		atomic< bool > announced( false );
		thread reader( [ &job, &violations, &sawRunning, &observed, &announced ]
		{
			for ( ;; )
			{
				bool running = false, hasResult = false;
				readTogether( job, running, hasResult );
				if ( running )
				{
					sawRunning++;
					if ( !announced.exchange( true ) )
						observed.set_value();
				}
				else
				{
					if ( !hasResult )
						violations++;
					return;
				}
			}
		} );

		firstRead.wait();
		release.set_value();
		reader.join();
	}
	check( waitIdle( job ), "the last cycle must finish" );

	check( sawRunning.load() > 0,
		"control: the reader must have observed the job running at least once"
		" (otherwise it never overlapped a worker and proved nothing)" );
	check( violations.load() == 0,
		"an observer that sees the job idle must already see its result; it "
		"did not on " + to_string( violations.load() ) + " of "
			+ to_string( cycles ) + " cycles" );
	return failures ? 1 : 0;
}

static int caseStopRequest()
{
	AsyncJob job( identityRenderer( 0 ) );

	check( !job.requestStop(),
		"Stop must refuse when nothing is running" );
	check( !job.isStopRequested(),
		"a refused Stop must not latch cancellation" );

	promise< void > release;
	shared_future< void > gate = release.get_future().share();
	atomic< bool > bodySawStop( false );
	promise< void > running;
	shared_future< void > isRunning = running.get_future().share();

	job.start( [ &job, gate, &bodySawStop, &running ]
	{
		running.set_value();
		gate.wait();
		bodySawStop = job.isStopRequested();
		return string( "stopped" );
	} );

	isRunning.wait();
	check( job.requestStop(), "Stop must be accepted while a job runs" );
	check( job.isStopRequested(), "Stop must latch while a job runs" );
	release.set_value();
	check( waitIdle( job ), "the stopped job must finish" );
	check( bodySawStop.load(),
		"the running body must be able to see the stop request" );

	// The latch survives the run -- only a start clears it. That is the
	//    behaviour the blocking stepwise path compensates for with
	//    clearStopRequest(), and changing it would change that path too.
	check( job.isStopRequested(),
		"the latch stays set after the run: only a start clears it" );
	job.clearStopRequest();
	check( !job.isStopRequested(), "clearStopRequest() must clear the latch" );
	return failures ? 1 : 0;
}

static int caseStaleStopNotInherited()
{
	AsyncJob job( identityRenderer( 0 ) );
	atomic< bool > secondSawStop( true );

	promise< void > r1;
	shared_future< void > g1 = r1.get_future().share();
	promise< void > up1;
	shared_future< void > u1 = up1.get_future().share();
	job.start( [ g1, &up1 ] { up1.set_value(); g1.wait();
		return string( "first" ); } );
	u1.wait();
	check( job.requestStop(), "control: the first job must accept Stop" );
	r1.set_value();
	check( waitIdle( job ), "the first job must finish" );
	check( job.isStopRequested(),
		"control: the latch must still be set going into the next start" );

	job.start( [ &job, &secondSawStop ]
	{
		secondSawStop = job.isStopRequested();
		return string( "second" );
	} );
	check( waitIdle( job ), "the second job must finish" );
	check( !secondSawStop.load(),
		"a new job must not inherit the previous job's stop request" );
	return failures ? 1 : 0;
}

static int caseKnownExceptionBecomesAResult()
{
	unsigned rendererCalls = 0;
	AsyncJob job( identityRenderer( &rendererCalls ) );

	check( job.start( []() -> string
		{ throw runtime_error( "boom" ); } ),
		"a job whose body throws must still start" );
	check( waitIdle( job ),
		"a throwing body must still reach a terminal state" );

	check( rendererCalls == 1,
		"the boundary must render exactly one failure" );
	check( job.result == "the run failed: boom",
		"a std::exception must become its own message, got '"
			+ job.result + "'" );
	check( !job.isRunning(),
		"the engine must be released even when the body throws" );

	// Reaching this line at all is the assertion that nothing escaped the
	//    worker: an escape calls std::terminate and this process would be gone.
	return failures ? 1 : 0;
}

static int caseUnknownExceptionBecomesAResult()
{
	unsigned rendererCalls = 0;
	AsyncJob job( identityRenderer( &rendererCalls ) );

	check( job.start( []() -> string { throw 42; } ),
		"a job whose body throws a non-exception must still start" );
	check( waitIdle( job ),
		"a body throwing a non-exception must still reach a terminal state" );

	check( rendererCalls == 1,
		"the boundary must render exactly one failure" );
	check( job.result == "the run failed with an unrecognized error",
		"an unknown exception must become the generic message, got '"
			+ job.result + "'" );
	check( !job.isRunning(),
		"the engine must be released even when the body throws an int" );
	return failures ? 1 : 0;
}

static int caseShutdownJoin()
{
	{
		AsyncJob idle( identityRenderer( 0 ) );
		idle.joinForShutdown(); // no worker: must be safe
		check( !idle.isRunning(), "shutdown with no worker must be safe" );
	}

	AsyncJob job( identityRenderer( 0 ) );
	atomic< bool > bodySawStop( false );
	promise< void > up;
	shared_future< void > running = up.get_future().share();

	job.start( [ &job, &up, &bodySawStop ]
	{
		up.set_value();
		while ( !job.isStopRequested() )
			this_thread::sleep_for( chrono::milliseconds( 1 ) );
		bodySawStop = true;
		return string( "shut down" );
	} );
	running.wait();

	job.joinForShutdown(); // must cancel AND join, not detach or destroy
	check( bodySawStop.load(),
		"shutdown must request cancellation, not just wait" );
	check( !job.isRunning(), "shutdown must leave the job idle" );
	check( job.result == "shut down",
		"a joined worker must have published before shutdown returned" );
	return failures ? 1 : 0;
}

// ------------------------------------------- the CLI process boundary

static int caseGuardNormalReturn()
{
	ostringstream err;
	int status = procguard::run( [] { return 7; }, err, "neuron" );
	check( status == 7, "a body that returns must keep its exit status" );
	check( err.str().empty(), "a body that returns must print nothing" );
	return failures ? 1 : 0;
}

static int caseGuardKnownException()
{
	ostringstream err;
	int status = procguard::run(
		[]() -> int { throw runtime_error( "the engine gave up" ); },
		err, "neuron" );
	check( status == 1, "a std::exception must exit 1, got "
		+ to_string( status ) );
	check( err.str() == "neuron: fatal: the engine gave up\n",
		"the diagnostic must name the program and the reason, got '"
			+ err.str() + "'" );
	return failures ? 1 : 0;
}

static int caseGuardContractException()
{
	// The exception this boundary was built for (D9): a Matrix contract
	//    failure used to terminate the CLI with no message at all.
	ostringstream err;
	int status = procguard::run( []() -> int
	{
		Matrix< double > m( 2, 2 );
		return ( int ) m( 5, 5 ); // BoundsViolation, thrown in Release
	}, err, "neuron" );

	check( status == 1, "a Matrix contract failure must exit 1" );
	check( err.str().find( "neuron: fatal: " ) == 0,
		"a contract failure must be reported as fatal, got '"
			+ err.str() + "'" );
	check( err.str().find( "bounds" ) != string::npos,
		"the contract's own message must survive to the boundary, got '"
			+ err.str() + "'" );
	return failures ? 1 : 0;
}

static int caseGuardUnknownException()
{
	ostringstream err;
	int status = procguard::run( []() -> int { throw 42; }, err, "neuron" );
	check( status == 1, "an unknown exception must exit 1" );
	check( err.str() == "neuron: fatal: unrecognized error\n",
		"an unknown exception must get the generic diagnostic, got '"
			+ err.str() + "'" );
	return failures ? 1 : 0;
}

// ------------------------------------------------------------- the driver

struct Case { const char* name; int ( *run )(); };

static const Case cases[] = {
	{ "a body's result is published verbatim, once", caseResultPublishedOnce },
	{ "a finished worker is reaped before the next start", caseReap },
	{ "a new run starts with every progress field clear",
		caseResetClearsEveryField },
	{ "start() marks the job running before it returns",
		caseRunningMarkedBeforeStartReturns },
	{ "the result is published before the job goes idle",
		casePublishBeforeClear },
	{ "Stop is refused when idle and latches when running", caseStopRequest },
	{ "a new job does not inherit a stale stop request",
		caseStaleStopNotInherited },
	{ "a std::exception becomes a published failure",
		caseKnownExceptionBecomesAResult },
	{ "an unknown exception becomes a published failure",
		caseUnknownExceptionBecomesAResult },
	{ "shutdown cancels and joins a live worker", caseShutdownJoin },
	{ "the CLI boundary passes a normal exit status through",
		caseGuardNormalReturn },
	{ "the CLI boundary reports a std::exception and exits 1",
		caseGuardKnownException },
	{ "the CLI boundary reports a Matrix contract failure and exits 1",
		caseGuardContractException },
	{ "the CLI boundary reports an unknown exception and exits 1",
		caseGuardUnknownException },
};
static const unsigned nCases = sizeof cases / sizeof cases[ 0 ];

int main( int argc, char* argv[] )
{
	// The case list, read UP FRONT by a driver script, so that a case which
	//    kills its own process is still named in the report.
	if ( argc > 1 && strcmp( argv[ 1 ], "-l" ) == 0 )
	{
		for ( unsigned i = 0; i < nCases; i++ )
			cout << ( i + 1 ) << " " << cases[ i ].name << endl;
		return 0;
	}

	// Single-case mode: for sabotage evidence, where a missing handler in the
	//    worker boundary terminates the process and would hide every case
	//    after it.
	if ( argc > 1 )
	{
		unsigned n = ( unsigned ) atoi( argv[ 1 ] );
		if ( n < 1 || n > nCases )
		{
			cout << "case out of range 1.." << nCases << endl;
			return 2;
		}
		int rc = cases[ n - 1 ].run();
		cout << ( rc ? "FAIL - " : "ok - " ) << n << ". "
			<< cases[ n - 1 ].name << endl;
		return rc;
	}

	for ( unsigned i = 0; i < nCases; i++ )
	{
		unsigned before = failures;
		cases[ i ].run();
		cout << ( failures == before ? "ok - " : "FAIL - " ) << ( i + 1 )
			<< ". " << cases[ i ].name << endl;
	}

	cout << endl << checks << " checks, " << failures << " failures" << endl;
	return failures ? 1 : 0;
}
