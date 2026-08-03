#!/usr/bin/env python3
"""Characterization of how every GUI/HTTP handler reads its request fields.

ROADMAP 4 item B9. Each row below is a MEASUREMENT of a running server, not a
reading of gui.cpp, and each carries one of two dispositions:

    PIN     the behaviour is the contract and must survive B9 unchanged
    DEFECT  the behaviour is a misparse; B9 turns it into a field-specific
            refusal, and this row is flipped in the same commit that does so

The PIN rows are the positive controls. They exist so that a strict parser
cannot be declared a success while it has quietly broken the omission rules,
the present-but-empty rules, or the values a real request carries -- which is
the failure mode that matters, because every DEFECT row can be satisfied by a
parser that simply refuses more.

Run through tests/gui/strictparse.sh, which owns the server.
"""

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = sys.argv[ 1 ]

failures = []
checks = 0


# --- transport ---------------------------------------------------------------

def post( path, fields ):
    body = urllib.parse.urlencode( fields ).encode()
    req = urllib.request.Request( BASE + path, data = body )
    try:
        with urllib.request.urlopen( req, timeout = 180 ) as r:
            return json.loads( r.read().decode() )
    except urllib.error.HTTPError as e:
        return { "ok": None, "message": "HTTP %d" % e.code }
    except Exception as e:                                 # noqa: BLE001
        return { "ok": None, "message": "TRANSPORT: %s" % e }


def running():
    with urllib.request.urlopen( BASE + "/api/train/status", timeout = 30 ) as r:
        return json.loads( r.read().decode() )[ "running" ]


def drain():
    """Stop any async job and wait for the worker to be reaped.

    Deadline only -- never a correctness assertion. A job that will not stop
    is reported as such rather than silently letting the next case run into a
    busy gate and read the refusal as its own answer.
    """
    post( "/api/train/stop", { "": "" } )
    for _ in range( 1200 ):
        if not running():
            return
        time.sleep( 0.05 )
    failures.append( "drain: a job never stopped" )


# --- assertions --------------------------------------------------------------

def check( label, cond, detail = "" ):
    global checks
    checks += 1
    if not cond:
        failures.append( "%s%s" % ( label, ( " -- " + detail ) if detail else "" ) )


def accepted( label, r, contains = None ):
    """The request was accepted, and optionally said something in particular."""
    check( label + " [accepted]", r.get( "ok" ) is True,
        "got ok=%s: %s" % ( r.get( "ok" ), str( r.get( "message" ) )[ :160 ] ) )
    if contains is not None:
        check( label + " [message]", contains in str( r.get( "message", "" ) ),
            "expected %r in %r" % ( contains, str( r.get( "message" ) )[ :160 ] ) )


def refused( label, r, contains ):
    """The request was refused, with a message that says which field and why.

    `contains` is asserted so that a refusal for an unrelated reason -- a busy
    gate, a missing dataset, an engine failure downstream -- cannot be counted
    as the refusal under test. That distinction is the whole point: "wrong
    error" and "right error" are different results.
    """
    check( label + " [refused]", r.get( "ok" ) is False,
        "got ok=%s: %s" % ( r.get( "ok" ), str( r.get( "message" ) )[ :160 ] ) )
    check( label + " [reason]", contains in str( r.get( "message", "" ) ),
        "expected %r in %r" % ( contains, str( r.get( "message" ) )[ :160 ] ) )


# --- fixtures ----------------------------------------------------------------

LOWBWT = "lowbwt2-2train.txt"     # 189 rows, 5 inputs, discrete outcome
XOR = "xor_discrete.set"          # 4 rows, 2 inputs -- training is instant


def load( **kw ):
    f = { "mode": "raw", "path": LOWBWT, "fraction": "0.3" }
    f.update( kw )
    return post( "/api/load", f )


def load_xor( **kw ):
    f = { "mode": "train", "path": XOR, "inputs": "2", "outputs": "1" }
    f.update( kw )
    return post( "/api/load", f )


def xor_net():
    load_xor()
    post( "/api/model", { "type": "simpleprop", "hidden": "2" } )


def train( **kw ):
    f = { "maxiter": "1", "algorithm": "1" }
    f.update( kw )
    return post( "/api/train", f )


# =============================================================================
# 1. /api/load
# =============================================================================

def group_load():
    # --- PIN: the values a real request carries -------------------------------
    accepted( "load: clean split", load(), "56 test exemplars" )
    accepted( "load: inputs derived when omitted", load( inputs = "" ),
        "5 inputs" )
    accepted( "load: explicit inputs", load( inputs = "5" ), "5 inputs" )
    accepted( "load: exact test count", load( test_n = "20", fraction = "" ),
        "20 test exemplars" )
    accepted( "load: threshold in range", load( threshold = "0.4" ) )
    accepted( "load: variate bounds", load( in_lower = "-1", in_upper = "1" ) )
    accepted( "load: roc_min", load( roc_min = "2" ) )
    accepted( "load: three-way split",
        load( val_fraction = "0.2" ), "validation exemplars" )
    accepted( "load: strata + bins",
        load( strata = "1", strata_bins = "3" ) )

    # PIN: surrounding whitespace is tolerated today and must stay tolerated
    accepted( "load: leading/trailing spaces on a double",
        load( fraction = " 0.3 " ), "56 test exemplars" )
    accepted( "load: leading/trailing spaces on an integer",
        load( test_n = " 20 ", fraction = "" ), "20 test exemplars" )

    # PIN: the domain refusals that already exist
    refused( "load: threshold out of range", load( threshold = "1.5" ),
        "threshold must be between 0 and 1" )
    refused( "load: strata_bins below 2",
        load( strata = "1", strata_bins = "1" ), "strata_bins must be at least 2" )
    refused( "load: inverted variate bounds",
        load( in_lower = "1", in_upper = "-1" ), "lower < upper" )
    refused( "load: strata column out of range", load( strata = "99" ),
        "strata column out of range" )
    refused( "load: roc_report token", load( roc_report = "neither" ),
        "roc_report must be both or either" )

    # --- DEFECT: trailing junk is silently truncated --------------------------
    accepted( "DEFECT load: inputs=5junk is five inputs",
        load( inputs = "5junk" ), "5 inputs" )
    accepted( "DEFECT load: test_n=20junk is twenty",
        load( test_n = "20junk", fraction = "" ), "20 test exemplars" )
    accepted( "DEFECT load: threshold=0.4junk is accepted",
        load( threshold = "0.4junk" ) )
    accepted( "DEFECT load: val_fraction=0.2junk is accepted",
        load( val_fraction = "0.2junk" ), "validation exemplars" )
    accepted( "DEFECT load: roc_min=2junk is accepted", load( roc_min = "2junk" ) )

    # DEFECT: an unreadable fraction silently produces NO test set at all
    accepted( "DEFECT load: fraction=abc silently loads without a test set",
        load( fraction = "abc" ), "189 training exemplars" )
    check( "DEFECT load: fraction=abc really has no test set",
        "test exemplars" not in str( load( fraction = "abc" ).get( "message" ) ) )

    # DEFECT: a syntax fault reported as a domain fault
    refused( "DEFECT load: strata_bins=abc blamed on the domain",
        load( strata = "1", strata_bins = "abc" ),
        "strata_bins must be at least 2" )

    # DEFECT: a negative count becomes an enormous unsigned one
    r = load( test_n = "-5", fraction = "" )
    check( "DEFECT load: test_n=-5 is not refused by name",
        r.get( "ok" ) is False and "test_n" not in str( r.get( "message" ) ),
        str( r.get( "message" ) )[ :160 ] )

    # DEFECT: infinities pass as variate bounds
    accepted( "DEFECT load: infinite variate bounds accepted",
        load( in_lower = "-inf", in_upper = "inf" ) )

    # DEFECT: any token but "0" means true, so "false" means true
    accepted( "DEFECT load: discrete=false means discrete",
        load( discrete = "false" ), "56 test exemplars" )
    accepted( "DEFECT load: history=true means false", load( history = "true" ) )


# =============================================================================
# 2. /api/model
# =============================================================================

def group_model():
    load()
    accepted( "model: one hidden layer",
        post( "/api/model", { "type": "simpleprop", "hidden": "3" } ),
        "SimpleProp 5-3-1" )
    accepted( "model: two hidden layers",
        post( "/api/model", { "type": "simpleprop", "hidden": "3,2" } ),
        "BackProp 5-3,2-1" )
    accepted( "model: spaces around layer sizes",
        post( "/api/model", { "type": "simpleprop", "hidden": " 3 , 2 " } ),
        "BackProp 5-3,2-1" )
    accepted( "model: bias off",
        post( "/api/model",
            { "type": "simpleprop", "hidden": "3", "bias": "0" } ), "no bias" )
    accepted( "model: logistic", post( "/api/model", { "type": "logistic" } ) )
    refused( "model: unknown type", post( "/api/model", { "type": "sideways" } ),
        "unknown model type" )
    refused( "model: zero hidden nodes",
        post( "/api/model", { "type": "simpleprop", "hidden": "0" } ),
        "positive integers" )

    accepted( "DEFECT model: hidden=3junk is three nodes",
        post( "/api/model", { "type": "simpleprop", "hidden": "3junk" } ),
        "SimpleProp 5-3-1" )
    accepted( "DEFECT model: bias=false leaves the bias ON",
        post( "/api/model",
            { "type": "simpleprop", "hidden": "3", "bias": "false" } ),
        "SimpleProp 5-3-1 network ready (X-entropy)" )
    accepted( "DEFECT model: log_lastop=no leaves logging ON",
        post( "/api/model",
            { "type": "simpleprop", "hidden": "3", "log_lastop": "no" } ) )


# =============================================================================
# 3. /api/randomize and /api/train
# =============================================================================

def group_train():
    xor_net()
    accepted( "randomize: seeded", post( "/api/randomize", { "seed": "42" } ) )
    accepted( "randomize: unseeded", post( "/api/randomize", {} ) )
    accepted( "DEFECT randomize: seed=99junk is accepted",
        post( "/api/randomize", { "seed": "99junk" } ) )
    accepted( "DEFECT randomize: seed=-1 wraps silently",
        post( "/api/randomize", { "seed": "-1" } ) )

    # PIN: the parity controls, each in its ordinary form
    accepted( "train: plain", train() )
    accepted( "train: eta", train( eta = "0.5" ) )
    accepted( "train: weight decay", train( weight_decay = "1", decay = "0.001" ) )
    accepted( "train: weight decay with a blank lambda",
        train( weight_decay = "1", decay = "" ) )
    accepted( "train: autostop with explicit tol/window",
        train( autostop = "1", autostop_tol = "0.001",
            autostop_window = "5" ) )
    accepted( "train: autostop with blank tol/window",
        train( autostop = "1", autostop_tol = "", autostop_window = "" ) )
    accepted( "train: printcount", train( printcount = "3", logprint = "0" ) )
    accepted( "train: batch/epoch and autostep",
        train( batch_epoch = "1", autostep = "1" ) )

    # PIN: present-but-empty disables a stopping condition, and that is a
    #    documented contract -- an empty value must not become a global refusal
    accepted( "train: empty minerr disables it", train( minerr = "" ) )
    accepted( "train: empty change disables it", train( change = "" ) )
    accepted( "train: empty errwindow disables it", train( errwindow = "" ) )
    accepted( "train: empty gradmax disables it", train( gradmax = "" ) )
    accepted( "train: empty printcount is a no-op", train( printcount = "" ) )

    # PIN: stopping conditions with values
    accepted( "train: minerr", train( minerr = "0.001" ) )
    accepted( "train: change", train( change = "0.001" ) )
    accepted( "train: errwindow", train( errwindow = "5" ) )
    accepted( "train: gradmax", train( gradmax = "0.01" ) )

    # PIN: the domain refusals that already exist
    refused( "train: maxiter below 1", train( maxiter = "0" ),
        "max iterations must be at least 1" )
    refused( "train: algorithm out of range", train( algorithm = "4" ),
        "algorithm must be 1, 2, 3 or auto" )
    refused( "train: eta above 1", train( eta = "2" ),
        "learning rate must be greater than 0 and at most 1" )
    refused( "train: errwindow of 1", train( errwindow = "1" ),
        "error window must be greater than 1" )
    refused( "train: autostop_tol out of range",
        train( autostop = "1", autostop_tol = "5" ),
        "autostop_tol must be between 0 and 1" )
    refused( "train: autostop_window below 2",
        train( autostop = "1", autostop_window = "1" ),
        "autostop_window must be at least 2" )
    refused( "train: eta of nan", train( eta = "nan" ),
        "learning rate must be greater than 0 and at most 1" )

    # --- DEFECT ---------------------------------------------------------------
    accepted( "DEFECT train: maxiter=1junk is one iteration",
        train( maxiter = "1junk" ) )
    accepted( "DEFECT train: algorithm=1x is algorithm 1",
        train( algorithm = "1x" ) )
    accepted( "DEFECT train: eta=0.5junk is 0.5", train( eta = "0.5junk" ) )
    accepted( "DEFECT train: printcount=3junk is 3",
        train( printcount = "3junk" ) )
    accepted( "DEFECT train: errwindow=5junk is 5", train( errwindow = "5junk" ) )
    accepted( "DEFECT train: gradmax=inf is a stop that cannot fire",
        train( gradmax = "inf" ) )
    accepted( "DEFECT train: autostop_tol=0.5junk is 0.5",
        train( autostop = "1", autostop_tol = "0.5junk" ) )

    # DEFECT: maxiter wraps modulo 2^32. 2^32 lands exactly on zero, which the
    #    existing domain check then reports as "at least 1" -- an unambiguous
    #    demonstration of the wrap that needs no timing and no iteration count.
    #    2^32+1 lands on one, and the run completes instead of asking for four
    #    billion iterations.
    refused( "DEFECT train: maxiter=2^32 wraps to zero",
        train( maxiter = "4294967296" ), "max iterations must be at least 1" )
    accepted( "DEFECT train: maxiter=2^32+1 wraps to one and trains",
        train( maxiter = "4294967297" ) )

    # DEFECT: async=true runs synchronously -- the response already carries the
    #    finished run, which an async start never does
    r = train( maxiter = "1", **{ "async": "true" } )
    accepted( "DEFECT train: async=true runs blocking", r )
    check( "DEFECT train: async=true really did block",
        "training started" not in str( r.get( "message" ) ),
        str( r.get( "message" ) )[ :160 ] )

    # PIN: async=1 really is asynchronous
    r = train( maxiter = "20000", **{ "async": "1" } )
    accepted( "train: async=1 starts a job", r, "training started" )
    drain()


# =============================================================================
# 4. /api/regress
# =============================================================================

def group_regress():
    load()
    post( "/api/model", { "type": "logistic" } )
    post( "/api/train", { "maxiter": "200", "algorithm": "1" } )
    S = { "structure": "0;1;2;3;4", "direction": "reverse" }

    refused( "regress: threshold of 0", dict( post( "/api/regress",
        dict( S, threshold = "0" ) ) ), "p-value threshold must be between 0 and 1" )
    refused( "regress: threshold of 1", post( "/api/regress",
        dict( S, threshold = "1" ) ), "p-value threshold must be between 0 and 1" )
    refused( "regress: threshold of inf", post( "/api/regress",
        dict( S, threshold = "inf" ) ), "p-value threshold must be between 0 and 1" )
    refused( "regress: direction token", post( "/api/regress",
        dict( S, direction = "sideways", threshold = "0.05" ) ),
        "direction must be reverse or forward" )

    # DEFECT: every comparison with NaN is false, so both guards pass and the
    #    analysis runs with a NaN threshold
    r = post( "/api/regress", dict( S, threshold = "nan" ) )
    check( "DEFECT regress: threshold=nan passes the guards",
        "p-value threshold" not in str( r.get( "message" ) ),
        str( r.get( "message" ) )[ :160 ] )
    drain()

    # DEFECT: trailing junk truncates
    r = post( "/api/regress", dict( S, threshold = "0.05junk" ) )
    check( "DEFECT regress: threshold=0.05junk is not refused",
        "p-value threshold" not in str( r.get( "message" ) ),
        str( r.get( "message" ) )[ :160 ] )
    drain()


# =============================================================================
# 5. /api/obd and /api/cv
# =============================================================================

def group_obd_cv():
    load()
    post( "/api/model", { "type": "simpleprop", "hidden": "2" } )

    refused( "obd: iter_budget below 1",
        post( "/api/obd", { "hidden_max": "3", "iter_budget": "-1" } ),
        "iter_budget must be at least 1" )
    refused( "obd: early_stop_tol of inf",
        post( "/api/obd",
            { "hidden_max": "3", "iter_budget": "10", "early_stop_tol": "inf" } ),
        "early_stop_tol must be between 0 and 1" )
    refused( "obd: algorithm token",
        post( "/api/obd", { "hidden_max": "3", "iter_budget": "10",
            "algorithm": "9" } ), "algorithm must be 1, 2, 3 or auto" )

    accepted( "obd: a clean search starts",
        post( "/api/obd", { "hidden_max": "3", "iter_budget": "10" } ),
        "OBD hidden-layer search started" )
    drain()

    accepted( "DEFECT obd: hidden_max=4junk starts a search",
        post( "/api/obd", { "hidden_max": "4junk", "iter_budget": "10" } ) )
    drain()
    accepted( "DEFECT obd: sample_every=2junk starts a search",
        post( "/api/obd", { "hidden_max": "3", "iter_budget": "10",
            "sample_every": "2junk" } ) )
    drain()

    cv = { "maxiter": "10", "neural": "0", "ldfa": "0", "qdfa": "0" }
    refused( "cv: folds below 2", post( "/api/cv", dict( cv, folds = "1" ) ),
        "folds must be at least 2" )
    refused( "cv: negative seed", post( "/api/cv", dict( cv, folds = "2",
        seed = "-1" ) ), "seed must be a whole number" )
    refused( "cv: negative locked_n", post( "/api/cv", dict( cv, folds = "2",
        locked_n = "-1" ) ), "locked_n cannot be negative" )
    refused( "cv: no procedure selected",
        post( "/api/cv", dict( cv, folds = "2", logistic = "0" ) ),
        "select at least one procedure" )

    accepted( "cv: a clean run starts", post( "/api/cv", dict( cv, folds = "2" ) ),
        "cross-validation started" )
    drain()

    accepted( "DEFECT cv: folds=5junk starts a 5-fold run",
        post( "/api/cv", dict( cv, folds = "5junk" ) ) )
    drain()
    accepted( "DEFECT cv: inner_val=0.25junk is accepted",
        post( "/api/cv", dict( cv, folds = "2", inner_val = "0.25junk" ) ) )
    drain()
    accepted( "DEFECT cv: maxiter=10x is accepted",
        post( "/api/cv", dict( cv, folds = "2", maxiter = "10x" ) ) )
    drain()

    # DEFECT: boolParam accepts "1" or lowercase "true"; every other token,
    #    including "TRUE" and "yes", silently means false
    accepted( "DEFECT cv: logistic=true is accepted",
        post( "/api/cv", dict( cv, folds = "2", logistic = "true" ) ) )
    drain()
    refused( "DEFECT cv: logistic=TRUE silently means false",
        post( "/api/cv", dict( cv, folds = "2", logistic = "TRUE" ) ),
        "select at least one procedure" )
    refused( "DEFECT cv: logistic=yes silently means false",
        post( "/api/cv", dict( cv, folds = "2", logistic = "yes" ) ),
        "select at least one procedure" )


# =============================================================================

def main():
    group_load()
    group_model()
    group_train()
    group_regress()
    group_obd_cv()

    print( "strict-parse characterization: %d checks" % checks )
    if failures:
        print( "FAILURES (%d):" % len( failures ) )
        for f in failures:
            print( "   ", f )
        return 1
    # A driver that silently skipped its work must not look like a pass.
    if checks < 120:
        print( "FAIL: only %d checks ran; the table did not execute" % checks )
        return 1
    print( "OK" )
    return 0


sys.exit( main() )
