#ifndef ASYNCJOB_H
#define ASYNCJOB_H

// The asynchronous-job lifecycle shared by every long GUI operation:
// training, hidden-layer sizing, cross-validation and stepwise regression.
//
// WHY THIS IS A CLASS AND NOT FOUR COPIES. The launch prelude -- reap the
//    finished worker, clear every per-run field, arm the cancellation latch,
//    mark the engine owned, start the thread -- was written out at four call
//    sites and was character-identical at all four. The single executable
//    difference between them was the payload the thread carried. That is one
//    mechanism, so it gets one implementation (standing rule 6), and the
//    ordering it maintains gets one place to be tested.
//
// WHAT IT DOES NOT DO. It is not a task framework: it runs a job that returns
//    the response body for the operation that ran, and it knows nothing about
//    training, folds, saliency or candidate variables. No caller's result is
//    normalized here -- a cancelled OBD search, cross-validation, training run
//    and stepwise analysis publish four different shapes, and this class hands
//    each of them through untouched.
//
// THE ORDERING CONTRACT, which is the whole reason the class exists.
//
//    The worker stores `result` under `progressMutex` and clears `running`
//    only after releasing it. A reader must therefore hold `progressMutex`
//    across BOTH reads -- the running flag and the result -- and then its
//    critical section totally orders against the worker's publication: it is
//    either entirely before (and sees running true, because the clear comes
//    after the publish) or entirely after (and sees the result). Read one of
//    them outside the lock and a finished run can be reported as an idle one
//    with nothing to show.
//
//    That constraint binds the CALLER, not just this class, so it is stated
//    here rather than in the status handler that happens to satisfy it.
//
// LOCK ORDER. `progressMutex` is never held across a join, a thread
//    construction, or any engine call. The join in particular must happen
//    before the mutex is taken: the outgoing worker's last act before
//    returning is to take that same mutex to publish, so joining while
//    holding it would deadlock.
//
// LIFETIME OF A BODY'S CAPTURES. A body captures by VALUE. It may capture a
//    pointer only where the caller's own exclusion guarantees the pointee
//    cannot be replaced for the run's duration -- in the GUI that is the busy
//    gate, which refuses every engine-touching request while `running` is
//    set. Request objects are never captured; each call site copies what it
//    needs before starting. The training launch goes further and captures
//    nothing dereferenceable at all, because automatic optimizer selection may
//    REPLACE the model, so the worker re-derives every pointer itself.

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class AsyncJob
{
public:
	// How a failure the boundary caught becomes a published result. Injected
	//    because the wording of a response body belongs to whoever speaks that
	//    protocol -- the GUI renders JSON; this class must not learn to. The
	//    MESSAGE is this class's own (it is the boundary that knows what
	//    happened); only its dressing comes from outside.
	typedef std::function< std::string( const std::string& ) > FailureRenderer;

	explicit AsyncJob( FailureRenderer renderer )
		: renderFailure( renderer ) {}

	// Destroying a std::thread that is still joinable calls std::terminate,
	//    and a worker stays joinable after it finishes until something reaps
	//    it -- so a job that ran even once and was then destroyed would kill
	//    the process. The GUI's instance is a global that the running server
	//    never destroys, which is why this was never seen; it is a property of
	//    the type, not of that one instance, and the type is now something
	//    anyone can hold.
	~AsyncJob() { joinForShutdown(); }

	// --- lifecycle ------------------------------------------------------

	// Reap, reset, arm, mark running, start -- in that order, which is the
	//    order the busy gate and the status endpoint both depend on. The
	//    caller owns whatever lock protects the engine. Returns false only if
	//    the worker could not be started, in which case nothing is left
	//    running and a failure result has already been published.
	bool start( std::function< std::string() > body );

	// The Stop button. Returns false when nothing is running, so the caller
	//    can say so; the latch is set only when a job is actually there to
	//    receive it.
	bool requestStop();

	// Cancel and join a worker that outlived the server loop. Safe when there
	//    is none.
	void joinForShutdown();

	// Clear the latch without starting anything. The one caller is a BLOCKING
	//    operation that shares its implementation with the async form and so
	//    still installs a cancel observer: the latch is process-global and only
	//    a start clears it, so a Stop from an earlier async run would otherwise
	//    cancel this one's first iteration. Nothing can be cancelling a run
	//    that has not begun.
	void clearStopRequest() { cancel = false; }

	bool isRunning() const { return running.load(); }
	bool isStopRequested() const { return cancel.load(); }

	// The address of the latch, for engine code that polls it directly
	//    (observers, the optimizer probe). Handing out the atomic itself keeps
	//    one latch rather than a copy that can go stale.
	std::atomic< bool >* cancelLatch() { return &cancel; }

	// --- progress, all of it guarded by progressMutex --------------------
	//
	// A reader takes progressMutex and keeps it for every field it reads
	//    together (see THE ORDERING CONTRACT above).

	std::mutex progressMutex;

	// The error-vs-iteration series is decimated: capped at MAX_POINTS,
	//    halving (and doubling the keep-every-nth stride) when full, so a
	//    million-iteration run still ships a bounded, evenly thinned curve.
	static const unsigned MAX_POINTS = 2000;
	std::vector< unsigned > iters;
	std::vector< double > trainErr, testErr; // testErr < 0 = not sampled
	unsigned keepEvery = 1, sampleCounter = 0;
	std::string result; // the finished run's response body, "" while running

	// OBD progress for the status poll (empty phase = not an OBD run)
	std::string obdPhase;
	unsigned obdHidden = 0;

	// Stepwise-regression progress as a ready-made object body (empty = not a
	//    stepwise run). Kept as its OWN field rather than reusing the obd
	//    object: a caller must never have to read "hidden" and guess it means
	//    a candidate variable.
	std::string stepwise;

	// Cross-validation progress -- its own field for the same reason
	//    (2026-07-29). A four-procedure, five-fold nested-OBD run is the most
	//    expensive thing the GUI does and used to show one unchanging phase
	//    word for its whole duration.
	std::string cvStage;     // "cross-validation" | "locked-test evaluation"
	std::string cvProcedure; // the procedure now fitting
	unsigned cvProcIndex = 0, cvProcCount = 0, cvProcDone = 0;
	unsigned cvFold = 0, cvK = 0, cvFoldsDone = 0;
	// The nested search inside the CURRENT fold. Cleared whenever the fold or
	//    procedure changes, so a finished fold's last trial can never be
	//    displayed as the next fold's progress.
	std::string cvInnerPhase;
	unsigned cvInnerHidden = 0;

	// Reset every CV progress field. One place, so a new field cannot be added
	//    and then left stale by a path that clears the others (the reason
	//    "not copied" is written out rather than omitted -- HISTORY
	//    2026-07-27).
	void clearCvProgress();

	// Clear every per-run field before a new job starts, so nothing from the
	//    last run can be read as this one's. Train, OBD, CV and stepwise each
	//    used to do this by hand, which meant a new progress field had to be
	//    remembered in four places -- and the one that forgot would publish
	//    stale progress rather than none. Callers must hold progressMutex.
	void resetForNewRun();

	// Append one decimated (iteration, train, test) sample. Guards its own
	//    mutex, so callers must NOT already hold progressMutex.
	void pushSample( unsigned iteration, double trainError, double testError );

private:
	// The worker boundary. An exception that escapes a thread's function calls
	//    std::terminate: the server would die while the page is still polling
	//    for status. Every escape becomes a published failure instead, and the
	//    publish-then-clear ordering holds on every path including the
	//    throwing ones.
	void runBody( const std::function< std::string() >& body );

	// Publish a result and release the engine, in that order. The last act of
	//    every finished run AND the whole of the failed-start rollback, so the
	//    two cannot drift apart.
	void finish( const std::string& published );

	FailureRenderer renderFailure;
	std::thread worker;
	std::atomic< bool > running{ false };
	std::atomic< bool > cancel{ false }; // the Stop button
};

#endif
