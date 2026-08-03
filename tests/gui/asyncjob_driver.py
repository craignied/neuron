#!/usr/bin/env python3
"""Characterization of the GUI async-job lifecycle. Driven by asyncjob.sh.

Stdlib only (the project's Python rule). Every wait has a deadline, so a broken
server fails this file instead of hanging CI, and no assertion is about elapsed
time -- a sleep here only paces a poll loop.

The mechanisms characterized, all of them in src/gui.cpp today:

  * the busy gate           one job owns the engine; everyone else gets 409
  * the two open doors      /api/train/status and /api/train/stop stay reachable
  * publish-then-clear      job.result is stored under progressMutex BEFORE
                            job.running goes false, so an observer that sees
                            running:false must already see the result
  * the reap                a finished worker is joined before it is replaced
  * resetForNewRun          no job's progress survives into the next job's status
  * the cancellation latch  cleared by a genuine async start, and by the
                            blocking stepwise path, so a stale Stop cannot
                            cancel work that had not begun
"""

import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

URL = sys.argv[ 1 ].rstrip( "/" )

checks = 0
failures = []


def need( cond, why ):
    global checks
    checks += 1
    if not cond:
        failures.append( why )
        print( "  FAIL: " + why )


def http( path, data = None, timeout = 60 ):
    """(status, text). A 409 is a response to be read, not an exception.

    data=None is a GET; data={} is a POST with a zero-length body (which the
    server needs -- a bodyless POST with no Content-Length makes it wait for
    its read timeout, see AGENTS.md).
    """
    body = None if data is None else urllib.parse.urlencode( data ).encode()
    req = urllib.request.Request( URL + path, data = body,
        method = "GET" if data is None else "POST" )
    try:
        with urllib.request.urlopen( req, timeout = timeout ) as r:
            return r.status, r.read().decode( "utf-8", "replace" )
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode( "utf-8", "replace" )
    except Exception as e:  # noqa: BLE001
        # The server stopped answering. That is THE failure this file exists to
        #    notice -- a worker thread that terminated the process takes the
        #    whole GUI with it -- so report it as one rather than as a traceback
        #    from whichever request happened to be in flight.
        print( "\nFAIL: the server stopped answering %s (%s)" % ( path, e ) )
        print( "      a terminated worker thread kills the server: this is the "
            "boundary failing, not the test" )
        sys.exit( 1 )


def jhttp( path, data = None, timeout = 60 ):
    """(status, parsed-or-None). A body that will not parse is itself a failure:
    the status endpoint assembles its JSON by hand under a mutex, and a torn
    read would show up right here."""
    status, text = http( path, data, timeout )
    try:
        return status, json.loads( text )
    except Exception as e:  # noqa: BLE001 -- any parse failure is the finding
        need( False, "%s did not return JSON (%s): %.200s" % ( path, e, text ) )
        return status, None


class Poller( threading.Thread ):
    """Polls /api/train/status until it observes running:false.

    Records every sample. The deadline is a liveness bound, never an assertion:
    a job that never finishes fails `finished`, it does not hang the suite.
    """

    def __init__( self, deadline = 180 ):
        threading.Thread.__init__( self )
        self.deadline = deadline
        self.samples = []
        self.torn = []
        self.finished = False

    def run( self ):
        end = time.monotonic() + self.deadline
        while time.monotonic() < end:
            status, text = http( "/api/train/status", timeout = 15 )
            try:
                sample = json.loads( text )
            except Exception:  # noqa: BLE001
                self.torn.append( ( status, text ) )
                return
            self.samples.append( sample )
            if sample.get( "running" ) is False:
                self.finished = True
                return
            time.sleep( 0.01 )

    # --- what a finished poll proves ------------------------------------

    def saw_running( self ):
        return any( s.get( "running" ) is True for s in self.samples )

    def first_idle( self ):
        for s in self.samples:
            if s.get( "running" ) is False:
                return s
        return None

    def any_key( self, key ):
        return any( key in s for s in self.samples )

    def running_samples( self ):
        return [ s for s in self.samples if s.get( "running" ) is True ]


def wait_for_overlap( p, timeout = 60 ):
    """Block until the poller has actually seen the job running.

    This is the synchronization that makes the cancellation cases deterministic:
    Stop is sent only once an observer has recorded running:true, so the test
    can never be stopping a job that already finished. The timeout only bounds
    failure -- it is not what the case asserts.
    """
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if p.saw_running():
            return True
        if p.finished:
            return False
        time.sleep( 0.005 )
    return False


def drain( p, label, deadline = 180, expect_overlap = True ):
    """Join a running poller and check the invariants every job shares.

    Returns (poller, first idle sample). `expect_overlap` is the control that
    the observer really did overlap the worker: without it, "no stale progress
    appeared" could be true because nothing was ever observed at all.

    It is required only where a case asserts the ABSENCE of something during a
    run. A case that asserts a terminal fact, or that a progress field was
    published at all, does not need it -- the progress fields are not cleared
    when a job ends, only when the next one starts, so the terminal sample
    still carries them. Where overlap IS required, the caller must use a job
    that cannot finish on its own (see start_long_train), because "make the
    fixture bigger" is a bet on the slowest machine in CI, and that bet was
    lost on Windows at dfffde8.
    """
    p.join( deadline + 30 )
    need( not p.is_alive(), "%s: the status poller did not return" % label )
    need( p.torn == [], "%s: /api/train/status returned unparseable JSON" % label )
    need( p.finished, "%s: the job never reported running:false" % label )
    if expect_overlap:
        need( p.saw_running(),
            "%s: no sample observed the job running (the observer never "
            "overlapped the worker, so its other assertions are vacuous)"
            % label )
    idle = p.first_idle()
    need( idle is not None, "%s: no idle sample to inspect" % label )
    if idle is not None:
        # THE ORDERING CONTRACT, as far as an outside observer can see it. The
        #    worker stores job.result under progressMutex and only then clears
        #    job.running, and handleTrainStatus holds that same mutex across
        #    BOTH reads -- so an observer that sees running:false cannot be
        #    looking at a run whose result has not been published.
        #
        #    The assertion is sound but nearly powerless from out here: see the
        #    measurement at case 10. It stays because a job that publishes NO
        #    result at all -- a body whose return value is dropped, a launch
        #    that clears running without ever running -- fails it every time.
        need( idle.get( "result" ) is not None,
            "%s: the FIRST status with running:false carried result:null -- "
            "publish-then-clear was violated" % label )
    return p, idle


def poll_to_completion( label, deadline = 180, expect_overlap = True ):
    """Start an observer now and run it to the job's end."""
    p = Poller( deadline )
    p.start()
    return drain( p, label, deadline, expect_overlap )


def busy( path, data = None ):
    status, body = jhttp( path, data )
    return status == 409 and body is not None and body.get( "busy" ) is True


def start_long_train():
    """An async training run that cannot finish on its own: XOR with every
    stopping condition switched off and a ceiling far beyond the test's life.
    Anything asserted while it runs is asserted against a genuinely running
    job, not a race with a fast one."""
    status, j = jhttp( "/api/train", {
        "algorithm": "1", "maxiter": "20000000", "seed": "42", "async": "1",
        "gradmax": "", "change": "", "errwindow": "", "minerr": "",
        "autostop": "0" } )
    return status, j


def stop_and_drain( label ):
    """Observe the job running, THEN stop it, then drain to the end.

    The order matters and is the reason this helper exists: sending Stop first
    and looking afterwards would let a job that had already finished pass as a
    cancelled one.
    """
    p = Poller( 180 )
    p.start()
    overlapped = wait_for_overlap( p )
    need( overlapped,
        "%s: the job was never observed running, so Stop had nothing to reach"
        % label )
    status, j = jhttp( "/api/train/stop", {} )
    need( status == 200 and j is not None and j.get( "ok" ) is True,
        "%s: Stop was refused (%s)" % ( label, j ) )
    return drain( p, label )


print( "GUI async-job characterization at " + URL )

# ---------------------------------------------------------------- controls
# Nothing below means anything if the server is not answering and the engine
# cannot train at all, so establish both first.

status, version = http( "/api/version" )
need( status == 200 and version.strip() != "",
    "control: /api/version must answer with a version string" )

status, j = jhttp( "/api/load", { "mode": "train", "path": "xor_discrete.set" } )
need( j is not None and j.get( "ok" ) is True, "control: load the XOR set" )

status, j = jhttp( "/api/model", { "type": "simpleprop", "hidden": "3" } )
need( j is not None and j.get( "ok" ) is True, "control: create a SimpleProp" )

# --------------------------------------------------- 1. the blocking path
# A blocking run must not go through the job channel at all: it publishes no
# result, no series and no progress there. This is the baseline the extraction
# must not disturb (blocking training and blocking stepwise stay separate).

status, blocking = jhttp( "/api/train",
    { "algorithm": "1", "maxiter": "200", "seed": "42" } )
need( blocking is not None and blocking.get( "ok" ) is True,
    "control: a blocking training run must succeed" )
need( blocking is not None and blocking.get( "output", "" ).strip() != "",
    "control: the blocking run must return its report inline" )

status, st = jhttp( "/api/train/status" )
need( st is not None and st.get( "running" ) is False,
    "a blocking run must never set job.running" )
need( st is not None and st.get( "result" ) is None,
    "a blocking run must publish nothing to the async job channel" )
need( st is not None and st.get( "series", {} ).get( "iter" ) == [],
    "a blocking run must push no samples to the async series" )

# ------------------------------------- 2. the busy gate and the two doors
# The launch response returns only after job.running is set, so the very next
# engine-touching request must already be refused.
#
# MEASURED LIMIT, so that this case is not read as more than it is: moving
# `job.running = true` into the worker -- which breaks exactly the ordering the
# first assertion names -- passes this file 3 times out of 3. A thread starts in
# microseconds and an HTTP round trip takes hundreds, so no external caller can
# observe that window. What the assertion still catches is running never being
# set at all. The ordering itself needs an in-process test.

status, j = start_long_train()
need( status == 200 and j is not None and j.get( "ok" ) is True,
    "an async training run must start" )

need( busy( "/api/model", { "type": "logistic" } ),
    "job.running must be set BEFORE the launch returns: the next /api/model "
    "must already be refused with 409 busy" )
need( busy( "/api/load", { "mode": "train", "path": "xor_discrete.set" } ),
    "/api/load must be refused while a job owns the engine" )
need( busy( "/api/stats" ), "/api/stats must be refused while a job runs" )
need( busy( "/api/regress", { "structure": "0;1", "direction": "reverse" } ),
    "/api/regress must be refused while a job runs" )
need( busy( "/api/obd", {} ), "/api/obd must be refused while a job runs" )
need( busy( "/api/cv", {} ), "/api/cv must be refused while a job runs" )
need( busy( "/api/train", { "algorithm": "1", "maxiter": "10", "async": "1" } ),
    "a SECOND async job must be refused -- one job owns the engine" )

status, st = jhttp( "/api/train/status" )
need( status == 200 and st is not None and st.get( "running" ) is True,
    "/api/train/status must stay open while a job runs" )

# ------------------------ 3. cancellation, publish-then-clear, torn reads
poller, idle = stop_and_drain( "cancelled training" )
need( len( poller.samples ) > 1,
    "control: the poller must have taken more than one sample" )
if idle is not None and idle.get( "result" ):
    r = idle[ "result" ]
    need( r.get( "ok" ) is True,
        "a cancelled run is a successful operation: ok:true" )
    need( r.get( "stopReason" ) == "cancelled",
        "a cancelled training run must report stopReason cancelled, got %s"
        % r.get( "stopReason" ) )
need( all( "series" in s for s in poller.samples ),
    "every status sample must carry the series object" )

# ------------------------------- 4. the engine is released, the reap works
# The 409s above were the gate and not a broken endpoint -- prove it by making
# the same call succeed now.
status, j = jhttp( "/api/model", { "type": "simpleprop", "hidden": "3" } )
need( j is not None and j.get( "ok" ) is True,
    "control: /api/model must succeed once the job has finished (so the 409s "
    "above were the busy gate, not a broken endpoint)" )

# Launching again is itself the assertion that the finished worker was joined:
#    assigning a std::thread over a joinable one calls std::terminate, so a
#    missing reap does not fail an assertion here, it kills the server -- which
#    asyncjob.sh checks for explicitly after the driver returns.
status, j = jhttp( "/api/train", { "algorithm": "1", "maxiter": "3000",
    "seed": "42", "async": "1", "gradmax": "", "change": "", "errwindow": "",
    "minerr": "", "autostop": "0" } )
need( j is not None and j.get( "ok" ) is True,
    "a second async run must start after the first was reaped" )

# No overlap control here: what this case asserts is the TERMINAL stop reason,
#    and a run that ended before the first poll answers it just as well.
poller, idle = poll_to_completion( "run after a cancelled run",
    expect_overlap = False )
if idle is not None and idle.get( "result" ):
    r = idle[ "result" ]
    need( r.get( "ok" ) is True, "the run after a cancelled one must succeed" )
    # The cancellation latch is cleared by a genuine async start. Without that,
    #    this run would end at its first iteration wearing the previous run's
    #    Stop.
    need( r.get( "stopReason" ) != "cancelled",
        "a new async job must not inherit the previous job's cancellation" )
    need( r.get( "stopReason" ) == "max_iterations",
        "the second run must end at its own ceiling, got %s"
        % r.get( "stopReason" ) )

# ------------------------------------ 5. no job's progress reaches the next
# Each long job publishes its OWN progress object, and resetForNewRun clears
# every one of them. Each pair below runs its control first -- the field must
# have been published by its own job -- because a later absence proves nothing
# otherwise.
#
# The ABSENCE half is asserted against a job that cannot finish on its own, and
# is then stopped. That is not caution about speed: a bounded fixture that
# happens to end before the first poll leaves exactly one terminal sample, and
# "no obd key in any sample" would then pass without a single running sample
# behind it. Windows proved that at dfffde8, where a 3000-iteration XOR run
# finished inside the first HTTP round trip.

status, j = jhttp( "/api/load", { "mode": "raw",
    "path": "lowbwt2-2train.txt", "fraction": "0.25", "seed": "1" } )
need( j is not None and j.get( "ok" ) is True,
    "control: load the low-birth-weight split (OBD needs a held-out set)" )

status, j = jhttp( "/api/obd", { "hidden_start": "2", "hidden_max": "3",
    "iter_budget": "150", "seed": "42", "algorithm": "1" } )
need( j is not None and j.get( "ok" ) is True, "an OBD search must start" )

poller, idle = poll_to_completion( "OBD", expect_overlap = False )
need( poller.any_key( "obd" ),
    "control: an OBD search must publish its obd progress object" )
need( idle is not None and idle.get( "result" ) is not None,
    "OBD must publish a result (a refusal is a result too)" )

status, j = jhttp( "/api/model", { "type": "simpleprop", "hidden": "3" } )
need( j is not None and j.get( "ok" ) is True, "control: model after OBD" )
status, j = start_long_train()
need( j is not None and j.get( "ok" ) is True, "train after OBD must start" )
poller, idle = stop_and_drain( "training after OBD" )
need( not poller.any_key( "obd" ),
    "a plain training run must not report the previous OBD run's progress" )
need( not poller.any_key( "stepwise" ),
    "a plain training run must not report stepwise progress" )
need( not poller.any_key( "cv" ),
    "a plain training run must not report CV progress" )

# ------------------------------------------------- 6. stepwise, then train
status, j = jhttp( "/api/model", { "type": "logistic" } )
need( j is not None and j.get( "ok" ) is True, "control: a logistic model" )
status, j = jhttp( "/api/train",
    { "algorithm": "1", "maxiter": "2000", "seed": "42" } )
need( j is not None and j.get( "stopReason" ) == "grad_max",
    "control: the stepwise fixture's own fit must converge before regression" )

status, j = jhttp( "/api/regress", { "structure": "0;1;2;3;4",
    "direction": "reverse", "threshold": "0.05", "async": "1" } )
need( j is not None and j.get( "ok" ) is True,
    "an async stepwise regression must start" )
poller, idle = poll_to_completion( "stepwise", expect_overlap = False )
need( poller.any_key( "stepwise" ),
    "control: a stepwise analysis must publish its candidate accounting" )
if idle is not None and idle.get( "result" ):
    need( idle[ "result" ].get( "ok" ) is True,
        "the stepwise analysis must complete" )
    need( idle[ "result" ].get( "output", "" ).strip() != "",
        "the stepwise result must carry its report" )

status, j = jhttp( "/api/model", { "type": "simpleprop", "hidden": "3" } )
need( j is not None and j.get( "ok" ) is True, "control: model after stepwise" )
status, j = start_long_train()
need( j is not None and j.get( "ok" ) is True, "train after stepwise" )
poller, idle = stop_and_drain( "training after stepwise" )
need( not poller.any_key( "stepwise" ),
    "a plain training run must not report the previous stepwise run's "
    "progress" )

# ------------------------------------------------------- 7. CV, then train
status, j = jhttp( "/api/cv", { "folds": "3", "seed": "42", "maxiter": "300",
    "logistic": "1", "ldfa": "1", "neural": "0" } )
need( j is not None and j.get( "ok" ) is True, "a CV comparison must start" )
poller, idle = poll_to_completion( "cross-validation", expect_overlap = False )
need( poller.any_key( "cv" ),
    "control: a CV run must publish its repetition grid" )
need( idle is not None and idle.get( "result" ) is not None,
    "CV must publish a result" )

status, j = jhttp( "/api/model", { "type": "simpleprop", "hidden": "3" } )
need( j is not None and j.get( "ok" ) is True, "control: model after CV" )
status, j = start_long_train()
need( j is not None and j.get( "ok" ) is True, "train after CV" )
poller, idle = stop_and_drain( "training after CV" )
need( not poller.any_key( "cv" ),
    "a plain training run must not report the previous CV run's progress" )
need( not poller.any_key( "obd" ),
    "a plain training run must not report the coarse obd phase CV also sets" )

# -------------------------------- 8. cancellation reaches every job kind
# Each one keeps its OWN result shape through cancellation -- the shared
# lifecycle must not normalize them into one generic answer.

status, j = jhttp( "/api/model", { "type": "simpleprop", "hidden": "3" } )
need( j is not None and j.get( "ok" ) is True, "control: model before OBD stop" )
status, j = jhttp( "/api/obd", { "hidden_start": "2", "hidden_max": "8",
    "iter_budget": "4000", "seed": "42", "algorithm": "1" } )
need( j is not None and j.get( "ok" ) is True, "OBD must start (for Stop)" )
poller, idle = stop_and_drain( "cancelled OBD" )
need( idle is not None and idle.get( "result" ) is not None,
    "a cancelled OBD must still publish a result" )
if idle is not None and idle.get( "result" ):
    need( "trials" in idle[ "result" ] or "message" in idle[ "result" ],
        "a cancelled OBD keeps OBD's own result shape" )

status, j = jhttp( "/api/cv", { "folds": "5", "seed": "42", "maxiter": "4000",
    "logistic": "1", "ldfa": "1", "qdfa": "1", "neural": "0" } )
need( j is not None and j.get( "ok" ) is True, "CV must start (for Stop)" )
poller, idle = stop_and_drain( "cancelled CV" )
need( idle is not None and idle.get( "result" ) is not None,
    "a cancelled CV must still publish a result" )

status, j = jhttp( "/api/model", { "type": "logistic" } )
need( j is not None and j.get( "ok" ) is True, "control: logistic again" )
status, j = jhttp( "/api/train",
    { "algorithm": "1", "maxiter": "2000", "seed": "42" } )
need( j is not None and j.get( "ok" ) is True, "control: refit before stepwise" )
status, j = jhttp( "/api/regress", { "structure": "0;1;2;3;4",
    "direction": "reverse", "threshold": "0.05", "async": "1" } )
need( j is not None and j.get( "ok" ) is True, "stepwise must start (for Stop)" )
poller, idle = stop_and_drain( "cancelled stepwise" )
need( idle is not None and idle.get( "result" ) is not None,
    "a cancelled stepwise must still publish a result" )
if idle is not None and idle.get( "result" ):
    need( idle[ "result" ].get( "cancelled" ) is True
        or "cancel" in json.dumps( idle[ "result" ] ).lower(),
        "a cancelled stepwise says so in its own result" )

# ---------------- 9. a stale Stop must not cancel later BLOCKING stepwise
# job.cancel is a process-global latch that only an async START clears, so the
# Stop above is still set right now. handleRegress's blocking path clears it
# for exactly this reason (src/gui.cpp). Without that line this analysis is
# cancelled before its first candidate has trained an iteration.
status, blocking_regress = jhttp( "/api/regress", { "structure": "0;1;2;3;4",
    "direction": "reverse", "threshold": "0.05" } )
need( blocking_regress is not None and blocking_regress.get( "ok" ) is True,
    "a BLOCKING stepwise run after a Stop must not inherit the cancellation" )
if blocking_regress is not None:
    need( blocking_regress.get( "cancelled" ) is not True,
        "a blocking stepwise run must not report itself cancelled" )
    need( "complete" in blocking_regress.get( "message", "" ),
        "a blocking stepwise run after a Stop must run to completion, got '%s'"
        % blocking_regress.get( "message", "" ) )

# ------------------- 10. concurrent observers see a consistent terminal state
# What this case DOES prove: many simultaneous readers of a hand-assembled JSON
# document never see a torn or unparseable one, never hang, and every one of
# them reaches a terminal state carrying the published result.
#
# What it does NOT prove, measured rather than assumed: it does not catch an
# inverted publish-then-clear. Inverting runOnWorker so that job.running is
# cleared BEFORE job.result is stored passes this file 3/3 with one observer and
# 3/3 with eight. The window is only visible to a reader holding progressMutex
# at that exact instant, and a status request holds it for a few microseconds
# out of a round trip of hundreds -- so occupancy, not observer count, is the
# ceiling, and no amount of external polling raises it. The ordering invariant
# that the whole launcher exists to maintain is unreachable from outside the
# process. That measurement is the argument for making the launcher linkable;
# see docs/refactor_audit.md item 13.

status, j = jhttp( "/api/load", { "mode": "train", "path": "xor_discrete.set" } )
need( j is not None and j.get( "ok" ) is True, "control: reload XOR" )
status, j = jhttp( "/api/model", { "type": "simpleprop", "hidden": "3" } )
need( j is not None and j.get( "ok" ) is True, "control: model for the storm" )

STORM = 8
storm_failures = []
for round_no in range( 3 ):
    status, j = start_long_train()
    need( j is not None and j.get( "ok" ) is True,
        "storm round %d: the job must start" % round_no )
    pollers = [ Poller( 120 ) for _ in range( STORM ) ]
    for p in pollers:
        p.start()
    need( wait_for_overlap( pollers[ 0 ] ),
        "storm round %d: control -- the crowd must observe the job running"
        % round_no )
    jhttp( "/api/train/stop", {} )
    for p in pollers:
        p.join( 150 )
    for n, p in enumerate( pollers ):
        if p.is_alive():
            storm_failures.append( "round %d poller %d never returned"
                % ( round_no, n ) )
            continue
        if p.torn:
            storm_failures.append(
                "round %d poller %d read torn JSON: %.120s"
                % ( round_no, n, p.torn[ 0 ][ 1 ] ) )
        idle = p.first_idle()
        if idle is None:
            storm_failures.append( "round %d poller %d never saw the job end"
                % ( round_no, n ) )
        elif idle.get( "result" ) is None:
            storm_failures.append(
                "round %d poller %d saw running:false with result:null"
                % ( round_no, n ) )

need( storm_failures == [],
    "under %d concurrent observers, every one that saw running:false must "
    "already have seen the published result: %s"
    % ( STORM, "; ".join( storm_failures[ :4 ] ) ) )

# ------------------------------------------------- 11. the doors when idle
status, j = jhttp( "/api/train/stop", {} )
need( j is not None and j.get( "ok" ) is False,
    "Stop with nothing running must refuse rather than latch cancel" )

status, st = jhttp( "/api/train/status" )
need( st is not None and st.get( "running" ) is False,
    "status must report idle when nothing is running" )

print( "\n%d checks, %d failures" % ( checks, len( failures ) ) )
if failures:
    for f in failures:
        print( "  - " + f )
    sys.exit( 1 )
print( "asyncjob characterization OK" )
