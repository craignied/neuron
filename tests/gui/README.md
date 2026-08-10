# GUI tests

Four harnesses, each characterizing a different boundary. The first three are
part of the release gate; the fourth is manual.

| Script | What it characterizes | In the gate |
|---|---|---|
| `smoke.sh` | What every endpoint **returns**, plus a grep of the served page for the controls it must carry | yes |
| `strictparse.sh` | The request **boundary**: omission rules, present-but-empty rules, domain refusals, malformed fields | yes |
| `asyncjob.sh` | The machinery under the four long jobs: start, progress, cancel, busy | yes |
| `browser.sh` | The page's **behaviour**, clicked in a real Chrome | no — needs Playwright |

## Driving the page in a browser

**Use `./tests/gui/browser.sh`.** It starts a server in its own run directory,
drives Chrome with Playwright, and tears the server down. `--headed` lets you
watch it click.

```bash
./tests/gui/browser.sh              # headless
./tests/gui/browser.sh --headed     # watch it
```

Playwright launches the Chrome that is **already installed**
(`launch(channel="chrome")`), so there is no separate browser download and no
extension, bridge, or connector involved. This matters because the "Claude in
Chrome" extension bridge is **not available in every environment** — in the
VSCode extension environment the command that connects it does not exist at
all, so no browser tools ever appear and there is nothing to connect. Playwright
works everywhere. If it is not installed:

```bash
python3 -m pip install playwright
```

A missing Playwright is a **hard failure**, not a skip. A harness that reports
success when it did not run is worse than one nobody runs.

## Why a browser harness exists at all

The served HTML can contain every control, every id and every handler and still
compose a request the user cannot reach, because what a control does depends on
what the other controls are doing at the time. `smoke.sh` greps the page; it
cannot see it run.

The algorithm-checkbox panel reached review with exactly that defect. The page
asked `/api/algorithms` about the **live** batch/epoch and step-search values,
so turning on *Automatic learning rate* greyed out L-BFGS, iRPROP+ and
Levenberg–Marquardt — and the control the user would have had to change first
was the one the tick was going to change for them. Every server-side test
passed, because the server was right; the page was asking the wrong question.
Two clicks in Chrome found it. A hand-stubbed DOM had passed 9/9 beforehand,
because the canned server response was one we wrote ourselves — **a stub cannot
produce an interaction between two real controls.**

## When it finds something

`browser.sh` is a probe that finds defects; the gate is what keeps them fixed.
So:

1. Fix the page.
2. Add a **behavioural** assertion here, stated as the invariant rather than as
   whatever happens to trigger it (the regression guard above re-runs the
   eligibility query by hand rather than relying on a listener, and carries a
   discriminating control so it cannot be satisfied by a panel that has stopped
   greying anything out).
3. Pin what can be pinned **statically** in `smoke.sh`, against the served page,
   so the gate catches a revert without a browser.
4. Prove both new assertions fail: sabotage **one** mechanism at a time. Two
   independent things prevented this defect, and a sabotage of both at once
   silently proved nothing — that null result, and the sabotage that finally
   did fail, are entries **H and I** of the sabotage log at the bottom of
   `tests/network/check_autoalgo.cpp`, which is where this change's evidence
   lives whichever boundary each sabotage crossed.

## Notes that cost a re-run

- **The training report is not on the page.** It is captured server-side and
  offered as a download; there is no element holding it. Read it with
  `fetch("/api/save/report")`, and assert a positive control that the reader
  returned this run's report before trusting any "the report does not contain X".
- **`#tstatus` keeps the previous run's `final error`**, so waiting for that text
  returns instantly and every later step runs against a still-busy engine, each
  answering 409. Clear it before clicking Train and wait for the button to read
  `Train` again — `trainAndWait()` in `browser_driver.py` does both.
- **The Data file control opens a native dialog** that cannot be driven. Post
  the load directly (`loadDataset()`); the server resolves the path against its
  own working directory and every later click still works, because the state
  lives on the server. Only the Dataset panel's status line is left un-updated.
- Console noise about "message channel closed" / "Receiving end does not exist"
  comes from Chrome extensions, not the page. `browser_driver.py` filters it.
