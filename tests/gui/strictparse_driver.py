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

    # --- was DEFECT, now refused by name --------------------------------------
    #    Every row below silently misread its value before B9. The assertion is
    #    not merely "refused" but refused with THIS field's name in the message,
    #    because a handler that refuses everything for the wrong reason is the
    #    other way to make a strict-parsing test pass.
    refused( "load: inputs=5junk", load( inputs = "5junk" ),
        "inputs: '5junk' is not a whole number" )
    refused( "load: test_n=20junk", load( test_n = "20junk", fraction = "" ),
        "test_n: '20junk' is not a whole number" )
    refused( "load: threshold=0.4junk", load( threshold = "0.4junk" ),
        "threshold: '0.4junk' is not a number" )
    refused( "load: val_fraction=0.2junk", load( val_fraction = "0.2junk" ),
        "val_fraction: '0.2junk' is not a number" )
    refused( "load: roc_min=2junk", load( roc_min = "2junk" ),
        "roc_min: '2junk' is not a whole number" )

    # The one that silently produced no test set at all
    refused( "load: fraction=abc", load( fraction = "abc" ),
        "fraction: 'abc' is not a number" )
    # ... and the exponent that parsed as 1.0 and split the whole dataset off
    refused( "load: fraction=1e", load( fraction = "1e" ),
        "fraction: '1e' is not a number" )

    # A syntax fault no longer borrows the domain's sentence
    r = load( strata = "1", strata_bins = "abc" )
    refused( "load: strata_bins=abc", r,
        "strata_bins: 'abc' is not a whole number" )
    check( "load: strata_bins=abc is not blamed on the domain",
        "at least 2" not in str( r.get( "message" ) ),
        str( r.get( "message" ) )[ :160 ] )

    # A negative count is a sign fault, named, instead of an enormous unsigned
    refused( "load: test_n=-5", load( test_n = "-5", fraction = "" ),
        "test_n: '-5' cannot be negative" )

    # Infinities are not variate bounds
    refused( "load: in_lower=-inf", load( in_lower = "-inf", in_upper = "1" ),
        "in_lower: '-inf' is not a finite number" )
    refused( "load: in_upper=inf", load( in_lower = "-1", in_upper = "inf" ),
        "in_upper: 'inf' is not a finite number" )

    # Booleans take 1 and 0 only, so a token that used to mean its opposite
    #    now says so
    refused( "load: discrete=false", load( discrete = "false" ),
        "discrete: 'false' must be 1 or 0" )
    refused( "load: history=true", load( history = "true" ),
        "history: 'true' must be 1 or 0" )
    refused( "load: discrete= (empty)", load( discrete = "" ),
        "discrete is empty" )


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

    refused( "model: hidden=3junk",
        post( "/api/model", { "type": "simpleprop", "hidden": "3junk" } ),
        "hidden: '3junk' is not a whole number" )
    refused( "model: hidden=3,junk names the bad token",
        post( "/api/model", { "type": "simpleprop", "hidden": "3,junk" } ),
        "hidden: 'junk' is not a whole number" )
    refused( "model: bias=false",
        post( "/api/model",
            { "type": "simpleprop", "hidden": "3", "bias": "false" } ),
        "bias: 'false' must be 1 or 0" )
    refused( "model: log_lastop=no",
        post( "/api/model",
            { "type": "simpleprop", "hidden": "3", "log_lastop": "no" } ),
        "log_lastop: 'no' must be 1 or 0" )


# =============================================================================
# 3. /api/randomize and /api/train
# =============================================================================

def group_train():
    xor_net()
    accepted( "randomize: seeded", post( "/api/randomize", { "seed": "42" } ) )
    accepted( "randomize: unseeded", post( "/api/randomize", {} ) )
    refused( "randomize: seed=99junk", post( "/api/randomize", { "seed": "99junk" } ),
        "seed: '99junk' is not a whole number" )
    refused( "randomize: seed=-1", post( "/api/randomize", { "seed": "-1" } ),
        "seed: '-1' cannot be negative" )
    # PIN: the page sends seed= on every call, and an empty one still means
    #    "do not seed" -- a strict parser must not turn that into a refusal
    accepted( "randomize: empty seed is still unseeded",
        post( "/api/randomize", { "seed": "" } ),
        "weights randomized — the next Train" )

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
    accepted( "train: L-BFGS with explicit memory", train( algorithm = "4",
        lbfgs_memory = "10", batch_epoch = "1", autostep = "0" ) )

    # PIN: the domain refusals that already exist
    refused( "train: maxiter below 1", train( maxiter = "0" ),
        "max iterations must be at least 1" )
    refused( "train: algorithm out of range", train( algorithm = "5" ),
        "algorithm must be 1, 2, 3, 4 or auto" )
    refused( "train: L-BFGS memory below 1", train( algorithm = "4",
        lbfgs_memory = "0", batch_epoch = "1", autostep = "0" ),
        "lbfgs_memory must be at least 1" )
    refused( "train: L-BFGS memory on another algorithm", train( algorithm = "1",
        lbfgs_memory = "10" ), "lbfgs_memory is only valid with algorithm=4" )
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
    # nan and inf used to be caught, when they were caught at all, by a domain
    #    comparison that happened to be false for them. They are now refused as
    #    what they are, one layer earlier.
    refused( "train: eta of nan", train( eta = "nan" ),
        "eta: 'nan' is not a finite number" )

    # --- was DEFECT, now refused by name --------------------------------------
    refused( "train: maxiter=1junk", train( maxiter = "1junk" ),
        "maxiter: '1junk' is not a whole number" )
    refused( "train: algorithm=1x", train( algorithm = "1x" ),
        "algorithm: '1x' is not a whole number" )
    refused( "train: lbfgs_memory=10x", train( algorithm = "4",
        lbfgs_memory = "10x", batch_epoch = "1", autostep = "0" ),
        "lbfgs_memory: '10x' is not a whole number" )
    refused( "train: eta=0.5junk", train( eta = "0.5junk" ),
        "eta: '0.5junk' is not a number" )
    refused( "train: printcount=3junk", train( printcount = "3junk" ),
        "printcount: '3junk' is not a whole number" )
    refused( "train: errwindow=5junk", train( errwindow = "5junk" ),
        "errwindow: '5junk' is not a whole number" )
    refused( "train: gradmax=inf", train( gradmax = "inf" ),
        "gradmax: 'inf' is not a finite number" )
    refused( "train: minerr=nan", train( minerr = "nan" ),
        "minerr: 'nan' is not a finite number" )
    refused( "train: autostop_tol=0.5junk",
        train( autostop = "1", autostop_tol = "0.5junk" ),
        "autostop_tol: '0.5junk' is not a number" )
    refused( "train: decay=0.1junk",
        train( weight_decay = "1", decay = "0.1junk" ),
        "decay: '0.1junk' is not a number" )
    refused( "train: seed=-1", train( seed = "-1" ),
        "seed: '-1' cannot be negative" )

    # An out-of-range count is asserted on /api/cv (group_obd_cv), where the
    #    value is refused before anything runs. Asserting it on maxiter would
    #    mean asking the engine to obey whatever the number turned into.
    refused( "train: maxiter=99999999999999999999",
        train( maxiter = "99999999999999999999" ),
        "maxiter: '99999999999999999999' is out of range" )

    # async=true used to run blocking
    refused( "train: async=true", train( maxiter = "1", **{ "async": "true" } ),
        "async: 'true' must be 1 or 0" )

    # A malformed LATER field must not leave the model half-configured. The
    #    engine's run header prints the learning rate it is actually using, so
    #    this is observable rather than argued: set 0.25, have a request
    #    carrying 0.75 refused for a bad print count, then look at what the
    #    next clean run reports.
    r = train( eta = "0.25", autostep = "0" )
    accepted( "train: eta 0.25 applied", r )
    check( "train: the header shows the rate that was applied",
        "Learning rate eta: 0.25" in str( r.get( "output", "" ) ),
        "header did not name eta 0.25" )
    refused( "train: a bad later field refuses the whole request",
        train( eta = "0.75", autostep = "0", printcount = "3junk" ),
        "printcount: '3junk' is not a whole number" )
    r = train( autostep = "0" )
    accepted( "train: after the syntax refusal", r )
    check( "train: a refused SYNTAX fault applied nothing",
        "Learning rate eta: 0.25" in str( r.get( "output", "" ) )
            and "Learning rate eta: 0.75" not in str( r.get( "output", "" ) ),
        "eta from a refused request survived into the model" )

    # And the same for a DOMAIN fault in a later field, which is a different
    #    path: a syntax fault is refused by the reader at the top of the
    #    handler whatever the ordering, so only this case can tell whether the
    #    domain checks really run before the first setter. (Measured: without
    #    it, moving one domain check below the apply is not caught at all.)
    refused( "train: a later DOMAIN fault refuses the whole request",
        train( eta = "0.6", autostep = "0", printcount = "0" ),
        "print count must be at least 1" )
    r = train( autostep = "0" )
    accepted( "train: after the domain refusal", r )
    check( "train: a refused DOMAIN fault applied nothing",
        "Learning rate eta: 0.25" in str( r.get( "output", "" ) )
            and "Learning rate eta: 0.6" not in str( r.get( "output", "" ) ),
        "eta from a domain-refused request survived into the model" )

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
        dict( S, threshold = "inf" ) ), "threshold: 'inf' is not a finite number" )
    refused( "regress: direction token", post( "/api/regress",
        dict( S, direction = "sideways", threshold = "0.05" ) ),
        "direction must be reverse or forward" )

    # Every comparison with NaN is false, so both of the handler's own guards
    #    passed it through. The parser refuses it before they are reached.
    refused( "regress: threshold=nan",
        post( "/api/regress", dict( S, threshold = "nan" ) ),
        "threshold: 'nan' is not a finite number" )
    refused( "regress: threshold=0.05junk",
        post( "/api/regress", dict( S, threshold = "0.05junk" ) ),
        "threshold: '0.05junk' is not a number" )
    refused( "regress: async=true",
        post( "/api/regress",
            dict( S, threshold = "0.05", **{ "async": "true" } ) ),
        "async: 'true' must be 1 or 0" )


# =============================================================================
# 5. /api/obd and /api/cv
# =============================================================================

def group_obd_cv():
    load()
    post( "/api/model", { "type": "simpleprop", "hidden": "2" } )

    refused( "obd: iter_budget of 0 is a domain refusal",
        post( "/api/obd", { "hidden_max": "3", "iter_budget": "0" } ),
        "iter_budget must be at least 1" )
    refused( "obd: iter_budget of -1 is a sign refusal",
        post( "/api/obd", { "hidden_max": "3", "iter_budget": "-1" } ),
        "iter_budget: '-1' cannot be negative" )
    refused( "obd: early_stop_tol of inf",
        post( "/api/obd",
            { "hidden_max": "3", "iter_budget": "10", "early_stop_tol": "inf" } ),
        "early_stop_tol: 'inf' is not a finite number" )
    refused( "obd: early_stop_tol of 5 is still a domain refusal",
        post( "/api/obd",
            { "hidden_max": "3", "iter_budget": "10", "early_stop_tol": "5" } ),
        "early_stop_tol must be between 0 and 1" )
    refused( "obd: algorithm token",
        post( "/api/obd", { "hidden_max": "3", "iter_budget": "10",
            "algorithm": "9" } ), "algorithm must be 1, 2, 3 or auto" )

    accepted( "obd: a clean search starts",
        post( "/api/obd", { "hidden_max": "3", "iter_budget": "10" } ),
        "OBD hidden-layer search started" )
    drain()

    refused( "obd: hidden_max=4junk",
        post( "/api/obd", { "hidden_max": "4junk", "iter_budget": "10" } ),
        "hidden_max: '4junk' is not a whole number" )
    refused( "obd: sample_every=2junk",
        post( "/api/obd", { "hidden_max": "3", "iter_budget": "10",
            "sample_every": "2junk" } ),
        "sample_every: '2junk' is not a whole number" )
    refused( "obd: prune_tol=nan",
        post( "/api/obd", { "hidden_max": "3", "iter_budget": "10",
            "prune_tol": "nan" } ),
        "prune_tol: 'nan' is not a finite number" )
    refused( "obd: seed=-1",
        post( "/api/obd", { "hidden_max": "3", "iter_budget": "10",
            "seed": "-1" } ), "seed: '-1' cannot be negative" )

    cv = { "maxiter": "10", "neural": "0", "ldfa": "0", "qdfa": "0" }
    refused( "cv: folds below 2", post( "/api/cv", dict( cv, folds = "1" ) ),
        "folds must be at least 2" )
    # These two were already refusals; B9 changes only WHY they read as they
    #    do -- a sign fault named by the parser rather than a domain sentence
    #    reached after the value had been reinterpreted as a huge unsigned one
    refused( "cv: negative seed", post( "/api/cv", dict( cv, folds = "2",
        seed = "-1" ) ), "seed: '-1' cannot be negative" )
    refused( "cv: no procedure selected",
        post( "/api/cv", dict( cv, folds = "2", logistic = "0" ) ),
        "select at least one procedure" )

    accepted( "cv: a clean run starts", post( "/api/cv", dict( cv, folds = "2" ) ),
        "cross-validation started" )
    drain()

    refused( "cv: folds=5junk", post( "/api/cv", dict( cv, folds = "5junk" ) ),
        "folds: '5junk' is not a whole number" )
    refused( "cv: inner_val=0.25junk",
        post( "/api/cv", dict( cv, folds = "2", inner_val = "0.25junk" ) ),
        "inner_val: '0.25junk' is not a number" )
    refused( "cv: maxiter=10x",
        post( "/api/cv", dict( cv, folds = "2", maxiter = "10x" ) ),
        "maxiter: '10x' is not a whole number" )
    refused( "cv: locked_fraction=nan",
        post( "/api/cv", dict( cv, folds = "2", locked_fraction = "nan" ) ),
        "locked_fraction: 'nan' is not a finite number" )
    refused( "cv: locked_n=-1 is a sign fault",
        post( "/api/cv", dict( cv, folds = "2", locked_n = "-1" ) ),
        "locked_n: '-1' cannot be negative" )

    # A count above the width of the conversion used to be reinterpreted as
    #    whatever the overflow produced -- zero on a 64-bit long, LONG_MAX on
    #    MSVC's 32-bit one -- and then refused, if at all, by a domain check
    #    that was handed a number the caller never wrote. Now it is refused for
    #    being out of range, identically on both.
    refused( "cv: folds=2^64", post( "/api/cv",
        dict( cv, folds = "18446744073709551616" ) ),
        "folds: '18446744073709551616' is out of range" )

    # The five procedure flags were the ONE place on this API that accepted
    #    lowercase "true". They now take the same tokens as every other flag,
    #    which narrows that one spelling and makes the two that silently meant
    #    false say so instead.
    refused( "cv: logistic=true (the narrowed spelling)",
        post( "/api/cv", dict( cv, folds = "2", logistic = "true" ) ),
        "logistic: 'true' must be 1 or 0" )
    refused( "cv: logistic=TRUE",
        post( "/api/cv", dict( cv, folds = "2", logistic = "TRUE" ) ),
        "logistic: 'TRUE' must be 1 or 0" )
    refused( "cv: logistic=yes",
        post( "/api/cv", dict( cv, folds = "2", logistic = "yes" ) ),
        "logistic: 'yes' must be 1 or 0" )
    refused( "cv: neural_obd=false",
        post( "/api/cv", dict( cv, folds = "2", neural_obd = "false" ) ),
        "neural_obd: 'false' must be 1 or 0" )



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
