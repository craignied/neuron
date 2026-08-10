# The GUI page's own BEHAVIOUR, clicked in a real browser.
#
# WHY THIS FILE EXISTS, AND WHY IT IS NOT IN THE GATE.
#
# smoke.sh characterizes what every endpoint RETURNS and greps the served page
# for the controls it must carry.  strictparse.sh characterizes the request
# boundary.  Neither can see the page RUN: the served HTML can contain every
# control, every id and every handler and still compose a request the user
# cannot reach, because what a control does depends on what the other controls
# are doing at the time.
#
# That is not a hypothetical.  The algorithm-checkbox panel shipped for review
# with a dead end that neither curl nor a hand-stubbed DOM could produce: the
# page asked the eligibility endpoint about the LIVE batch/epoch and step-search
# values, so turning on "Automatic learning rate" greyed out L-BFGS, iRPROP+ and
# Levenberg-Marquardt -- and the control the user would have had to change first
# was the one the tick was going to change for them.  Every server-side test
# passed, because the server was right; the page was asking the wrong question.
# It took two clicks in Chrome to find.
#
# It is NOT part of the release gate because it needs Playwright and a real
# Chrome, which the engine deliberately does not depend on (the shipped Python
# tools are stdlib-only).  Run it by hand whenever gui_page.html's JavaScript
# changes.  When it finds something, pin the finding in smoke.sh as an assertion
# against the served page and prove that assertion fails -- this file is a probe
# that finds defects, smoke.sh is the gate that keeps them fixed.
#
# A MISSING DEPENDENCY IS A FAILURE HERE, NOT A SKIP.  A harness that reports
# success when it did not run is worse than one that is never run at all, and if
# anyone ever does wire this into a gate it must fail loudly rather than pass
# vacuously.
#
# Usage:  python3 browser_driver.py <url> [--headed]
#         (tests/gui/browser.sh starts the server and passes the url)

import json
import sys
import time

try:
    from playwright.sync_api import sync_playwright
except ImportError:                                     # deliberate hard failure
    sys.stderr.write(
        "FAIL: this harness needs Playwright, which the engine does not depend on.\n"
        "      python3 -m pip install playwright\n"
        "      (it drives the Chrome already installed; no extra browser download\n"
        "       is needed because it launches with channel='chrome')\n" )
    sys.exit( 2 )

failures = []


def expect( ok, what ):
    print( ( "ok   - " if ok else "FAIL - " ) + what )
    if not ok:
        failures.append( what )


# --- Reading the page ------------------------------------------------------

def boxes( page ):
    """Every algorithm checkbox, as plain data."""
    return page.evaluate(
        """() => Array.from(document.querySelectorAll('#algoset .algo'))
             .map(b => ({v: b.value, checked: b.checked, disabled: b.disabled}))""" )


def boxOf( page, value ):
    return page.locator( '#algoset input.algo[value="%s"]' % value )


def report( page ):
    """THE TRAINING REPORT IS NOT ON THE PAGE.  It is captured server-side and
       offered as a download, so there is no element to read it from -- and a
       negative assertion ("the report does not mention a competition") made
       against the page's own text would pass no matter what.  Ask the server
       for the same text the Report button downloads, and let the caller assert
       a positive control on it first."""
    return page.evaluate( """async () => (await fetch("/api/save/report")).text()""" )


def trainAndWait( page, timeout = 120000 ):
    """Click Train and wait for THIS run.

       The status line still holds the PREVIOUS run's "final error" when the
       button is clicked, so waiting for that text returns immediately, and
       every later step then runs against an engine that is still busy (each
       one answering 409).  Clearing the line first is what makes the wait mean
       what it says; the button returning to "Train" is what makes it complete."""
    page.evaluate( """() => { document.getElementById('tstatus').textContent = ''; }""" )
    page.click( "#trainbtn" )
    page.wait_for_function(
        "() => /final error|tick at least one/.test("
        "document.getElementById('tstatus').textContent)", timeout = timeout )
    page.wait_for_function(
        "() => document.getElementById('trainbtn').textContent === 'Train'",
        timeout = timeout )


def loadDataset( page, name, fraction = "0.25", seed = "1" ):
    """The Data file control opens a NATIVE file dialog, which cannot be driven.
       The server resolves a path against its OWN working directory, so posting
       the load directly is equivalent for everything downstream -- model,
       training, charts -- and every later click still works because the state
       lives on the server.  Only the Dataset panel's own status line is left
       un-updated, which is not what this file is testing."""
    page.evaluate(
        """async ([name, fraction, seed]) => { await fetch("/api/load", {
               method: "POST", body: new URLSearchParams(
                   {mode: "raw", path: name, fraction: fraction, seed: seed}) }); }""",
        [ name, fraction, seed ] )


def makeModel( page, errfunc, hidden = "3" ):
    page.select_option( "#mtype", "simpleprop" )
    page.fill( "#hidden", hidden )
    page.select_option( "#errfunc", errfunc )
    page.click( "text=Create model" )
    page.wait_for_timeout( 500 )
    return page.inner_text( "#mstatus" )


# --- The walkthrough -------------------------------------------------------

def walkAlgorithmPanel( page ):
    """The Train panel's algorithm checkboxes: what the set of ticked methods
       does to the other controls, to the run, and to what the user is told."""

    # 1. A METHOD THAT CANNOT RUN ON THIS MODEL GREYS ITSELF OUT, and says why.
    #    The tick is applied BEFORE the model exists, so this also proves the
    #    panel reacts to the model rather than only to its own clicks.
    boxOf( page, "6" ).check()
    status = makeModel( page, "xentropy" )
    expect( "X-entropy" in status, "CONTROL: the cross-entropy model was created (%r)"
        % status[ :50 ] )
    state = { b[ "v" ]: b for b in boxes( page ) }
    expect( state[ "6" ][ "disabled" ] and not state[ "6" ][ "checked" ],
        "cross-entropy: the Levenberg-Marquardt box greys out and its tick does "
        "not survive -- a ticked box the run would ignore is a lie" )
    expect( all( not state[ v ][ "disabled" ] for v in "12345" ),
        "CONTROL: the other five stay available, so this is the RULE and not a "
        "panel that disables everything" )
    reason = page.inner_text( "#algostatus" )
    expect( "least-squares" in reason,
        "and the server's own sentence is shown, not a sentence the page wrote: "
        "%r" % reason[ :80 ] )
    expect( page.locator( '#algoset label.unavailable' ).count() == 1,
        "and the LABEL is dimmed too -- a disabled checkbox alone leaves the "
        "text full black, so only the tiny box carries the news" )

    # 2. THE STEP CONTROLS LOCK FROM THE TICKED SET, and in that direction only.
    #    Automatic learning rate is turned ON first, so the assertion is about
    #    the tick FORCING it off rather than about it happening to be off.
    expect( not page.is_disabled( "#autostep" ) and not page.is_disabled( "#batch_epoch" ),
        "CONTROL: with only canonical ticked, both step controls are free" )
    page.check( "#autostep" )
    boxOf( page, "4" ).check()                          # L-BFGS
    page.wait_for_timeout( 400 )
    expect( page.is_disabled( "#autostep" ) and page.is_disabled( "#batch_epoch" ),
        "ticking L-BFGS LOCKS Automatic learning rate and Batch/epoch" )
    expect( not page.is_checked( "#autostep" ) and page.is_checked( "#batch_epoch" ),
        "and FORCES them to the only values it can run under" )

    # 2b. AND THE ELIGIBILITY ANSWER DOES NOT DEPEND ON THOSE TWO CONTROLS.
    #     This is the regression guard, asserted directly rather than through
    #     whatever happens to trigger a refresh: the query is re-run BY HAND
    #     with Automatic learning rate on, and the methods that own their step
    #     must still be available.  Asked about the live values the server would
    #     rightly answer "requires autostep=0" and the panel would grey out the
    #     three boxes whose tick was going to turn that control off -- a dead
    #     end, and the defect a real browser found on the second click.
    #
    #     Two separate things prevent it (the query is built from the forced
    #     values, and nothing re-runs it when those controls move), so sabotage
    #     one and this assertion is what fails; the earlier lock assertions
    #     cannot see either.  The evidence, including the first sabotage that
    #     knocked out BOTH and was therefore not caught, is entries H and I of
    #     the sabotage log at the bottom of tests/network/check_autoalgo.cpp.
    boxOf( page, "4" ).uncheck()
    page.wait_for_timeout( 300 )
    page.check( "#autostep" )
    page.evaluate( """async () => { await refreshEligibility(); }""" )
    page.wait_for_timeout( 300 )
    state = { b[ "v" ]: b for b in boxes( page ) }
    expect( not state[ "4" ][ "disabled" ] and not state[ "5" ][ "disabled" ],
        "with Automatic learning rate ON, the methods that own their own step "
        "are STILL available -- the panel must never grey out a method because "
        "of a control the tick itself would set" )
    # THE DISCRIMINATING CONTROL, and what keeps the assertion above from being
    #    satisfied by a panel that has simply stopped greying anything out: on
    #    this cross-entropy model LM must still be greyed, for a reason the tick
    #    cannot fix.  Both facts come from the same refresh.
    expect( state[ "6" ][ "disabled" ]
        and "least-squares" in page.inner_text( "#algostatus" ),
        "CONTROL: the refresh really ran, and Levenberg-Marquardt is still out "
        "for the one reason a tick cannot satisfy" )
    page.uncheck( "#autostep" )
    boxOf( page, "4" ).check()
    page.wait_for_timeout( 300 )
    boxOf( page, "4" ).uncheck()
    page.wait_for_timeout( 400 )
    expect( not page.is_disabled( "#autostep" ) and not page.is_disabled( "#batch_epoch" ),
        "and unticking it frees them again" )

    # 3. ONE BOX TICKED IS NOT A COMPETITION.
    page.fill( "#maxiter", "300" )
    page.fill( "#seed", "42" )
    page.click( "text=Randomize weights" )
    page.wait_for_timeout( 400 )
    trainAndWait( page )
    text = report( page )
    expect( "Training algorithm is canonical backpropagation" in text,
        "CONTROL: the report reader really read THIS run's report (%d chars) -- "
        "the negative assertion below is worthless without it" % len( text ) )
    expect( "Auto algorithm selection" not in text,
        "one box ticked: no competition summary in the report" )
    expect( "auto selected" not in page.inner_text( "#tstatus" ),
        "and the status line does not claim a selection" )

    # 4. TWO ELIGIBLE BOXES COMPETE, for the same fixed total.
    boxOf( page, "5" ).check()                          # canonical + iRPROP+
    page.wait_for_timeout( 300 )
    page.click( "text=Randomize weights" )
    page.wait_for_timeout( 400 )
    started = time.time()
    trainAndWait( page )
    elapsed = time.time() - started
    text = report( page )
    expect( "Auto algorithm selection (2250 ms total, 1125 ms each across 2 "
        "eligible candidates)" in text,
        "two boxes ticked: they competed, and a narrower set bought each of "
        "them a LARGER share of the same total rather than a shorter run" )
    expect( "Selected: " in text, "and the report names the winner" )
    message = page.inner_text( "#tstatus" )
    expect( "auto selected" in message,
        "and so does the status line, which is where the user actually looks: "
        "%r" % message[ :70 ] )
    expect( elapsed > 2.0,
        "and the budget was really spent (%.1f s) -- a label alone would not "
        "prove a probe ran" % elapsed )

    # 5. THE PANEL FOLLOWS THE MODEL, unprompted.
    status = makeModel( page, "lms" )
    expect( "busy" not in status,
        "CONTROL: the engine was idle, so the model really was recreated (%r)"
        % status[ :50 ] )
    state = { b[ "v" ]: b for b in boxes( page ) }
    expect( not state[ "6" ][ "disabled" ],
        "switching to LMS re-enables the Levenberg-Marquardt box with no click" )
    expect( page.inner_text( "#algostatus" ) == "", "and the reason line clears" )
    expect( page.locator( '#algoset label.unavailable' ).count() == 0,
        "and no label is left dimmed" )

    # 6. TICKED BUT INELIGIBLE: the survivor trains, and the user is told why
    #    the other one did not.  The tick is forced back on from script, which
    #    is exactly the state a page left open across a model change would be
    #    in -- and the case that proves the SERVER decides, not the page.
    makeModel( page, "xentropy" )
    page.evaluate(
        """() => { document.querySelectorAll('#algoset .algo').forEach(b => {
               b.checked = (b.value === "1" || b.value === "6");
               if (b.value === "6") b.disabled = false; }); }""" )
    page.click( "text=Randomize weights" )
    page.wait_for_timeout( 400 )
    trainAndWait( page )
    text = report( page )
    expect( "no competition was run" in text,
        "canonical + LM on cross-entropy: ONE eligible method, so no "
        "competition and no budget spent" )
    expect( "Levenberg-Marquardt: not eligible here" in text
        and "least-squares" in text,
        "and the report names the method that could not run, with its reason -- "
        "a method missing from a list with no explanation reads as one that lost" )
    expect( "Training algorithm is canonical backpropagation" in text,
        "and the ENGINE's own run header confirms it trained on the survivor, "
        "which is the fact a label echoed by the handler cannot establish" )

    # 7. NOTHING TICKED IS REFUSED BY THE PAGE, before any request is sent.
    page.evaluate(
        """() => document.querySelectorAll('#algoset .algo')
                   .forEach(b => { b.checked = false; })""" )
    page.evaluate( """() => { document.getElementById('tstatus').textContent = ''; }""" )
    page.click( "#trainbtn" )
    page.wait_for_timeout( 700 )
    expect( "tick at least one algorithm" in page.inner_text( "#tstatus" ),
        "no boxes ticked: refused in the page, with nothing sent" )


def main():
    if len( sys.argv ) < 2:
        sys.stderr.write( "usage: browser_driver.py <url> [--headed]\n" )
        return 2
    url = sys.argv[ 1 ].rstrip( "/" ) + "/"
    headed = "--headed" in sys.argv

    print( "GUI page behaviour, driven in Chrome: %s" % url )
    with sync_playwright() as play:
        # THE CHROME THAT IS ALREADY INSTALLED (channel="chrome"), not a
        #    downloaded bundle: this harness must not require a separate
        #    `playwright install` step, and the page ships to a real browser.
        browser = play.chromium.launch( channel = "chrome", headless = not headed )
        page = browser.new_page( viewport = { "width": 1280, "height": 1000 } )

        # A page error is a failure of the page even when every assertion below
        #    passes.  The extension chatter Chrome itself emits is not.
        errors = []
        page.on( "pageerror", lambda e: errors.append( str( e ) ) )
        page.goto( url )
        loadDataset( page, "lowbwt2-2train.txt" )

        walkAlgorithmPanel( page )

        real = [ e for e in errors if "message channel closed" not in e
            and "Receiving end does not exist" not in e ]
        expect( not real, "no page JavaScript errors during the walkthrough: %s"
            % ( real if real else "none" ) )
        browser.close()

    print( ( "FAILURES: %d" % len( failures ) ) if failures else "OK: page behaviour" )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit( main() )
