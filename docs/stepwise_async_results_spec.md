# Stepwise regression: visible results, asynchronous progress, and fit validity

> **Historical implementation spec.** This work shipped on 2026-07-27 (commit
> `79014e7`).  The current contract is in the Design Manifest, `docs/gui_cli_parity.md`
> and source; the record of what was found and fixed is in `docs/HISTORY.md`.  Load
> this file only for the measured pre-fix behavior, the required user experience,
> and the red tests the fix was held to.

## Why this work exists

Craig encountered this during the live Civic Choice GUI walkthrough on
2026-07-27. Reverse grouped stepwise regression on a converged binary logistic
model displayed only:

```text
regressing…
```

for the entire operation, then:

```text
reverse stepwise regression complete
```

There was no progress, no Stop control, and no result below the panel. This was
already opaque for logistic regression; neural subnetwork refits can take much
longer.

This is a general product defect, not walkthrough-specific polish. A
model-selection procedure that performs many training runs must expose what it
is doing, and an analysis result must be visible where the user invoked it.

## Measured current behavior

Read the implementation before changing it:

- `src/gui_page.html::regress()` sends blocking `POST /api/regress`, awaits the
  whole operation, then renders only `j.message`.
- `src/gui.cpp::handleRegress()` captures the complete stepwise report and
  already returns it as JSON field `output`.
- The server comment immediately above that return says the regression output
  is “shown on the page,” but the page discards `j.output`. The comment describes
  intended behavior, not actual behavior.
- When history logging is enabled, `RegressNet::report()` also appends the text
  incrementally to `neuron.log`. A user should not have to discover or inspect
  that file to learn the analysis result.
- `RegressNet::reverse_regress()` and `forward_regress()` call
  `netCopyPtr->train()` for candidate subnetworks and immediately use the
  returned error in Wilks comparisons. They do not check `StopReason`,
  convergence, cancellation, or a non-finite error. After the engine-wide
  convergence rule adopted 2026-07-25, this is another validity gap:
  ceiling-exhausted fits must not produce p-values or influence selection.
- The recovered Civic Choice `neuron.log` shows that every candidate
  `netCopyPtr->train()` also emits the complete training **and test** statistics,
  including a 2,000-resample ROC bootstrap. Stepwise Wilks selection uses the
  training likelihood, not those test statistics. Recomputing them for every
  candidate is a major avoidable cost and repeatedly exposes the nominally
  untouched test set during model selection.

Verify every statement above against the current source; do not treat this
handoff as a substitute for inspection.

## Required user experience

### While running

The Stepwise regression panel must show an honest, advancing status such as:

```text
Reverse stepwise · step 2 · candidate 4 of 7
Testing removal of variable 6 (inputs 6–8)
11 candidate fits completed
```

Expose at least:

- direction (`forward` or `reverse`);
- current outer selection step;
- variables currently selected/removed;
- candidate conceptual variable and its input-node group;
- candidate number within the current step;
- completed candidate fits;
- the exact number of candidates remaining when knowable;
- convergence/failure status of the most recently completed candidate; and
- elapsed time.

Do not invent a wall-clock ETA. Candidate subnetworks can have very different
training times. A remaining-fit count is defensible; a confident countdown is
not.

The Run button becomes **Stop** while active. Stop must propagate into the
currently training candidate, finish its current iteration gracefully, and end
the stepwise job promptly. It must not merely relabel work that continued to
completion.

### On completion

The page must render a persistent Stepwise results section containing:

- direction and p-value threshold;
- the supplied variable grouping;
- the elimination/addition path in order;
- every pass and candidate considered;
- conceptual variable number and input nodes;
- prior/subnetwork errors, degrees of freedom, Wilks statistic, and p-value;
- candidate convergence/failure status and stopping reason;
- the variable selected at each pass;
- the final retained-variable set (reverse) or selected-variable set (forward);
  and
- a clear completion/cancellation/failure summary.

At minimum, render the complete existing `output` report in a readable `<pre>`
or report pane immediately. Prefer also returning structured JSON so the GUI
does not have to parse prose and future consumers can build tables reliably.

The results must remain visible after completion; replacing them with a short
green status is not sufficient. The existing training report in `lastReport`
must remain intact.

If practical within this focused work, add a **Stepwise report** Session-files
download rather than making users extract the analysis from `neuron.log`.
Whether or not that artifact is added, visible in-page results are mandatory.

### Do not repeatedly evaluate the test set

Candidate refits must compute only what the stepwise procedure needs:

- training error/log likelihood;
- degrees of freedom;
- convergence and stopping metadata; and
- the resulting Wilks statistic and p-value.

They must not run classification tables, ROC fitting, or 2,000 bootstrap
resamples on the test set for every candidate. The test set is for evaluating a
procedure after selection, not for producing dozens of tempting intermediate
reports. Removing this unnecessary work should materially improve runtime
without changing candidate order or p-values.

Measure where the cost occurs and implement a fit-only/quiet candidate path
without creating a second training algorithm. Preserve one authoritative
optimization implementation; suppressing the expensive reporting epilogue must
not change weights, errors, RNG consumption, stopping iteration, or selection.
Consider CLI compatibility explicitly: do not maintain statistically
questionable repeated test evaluation merely because the legacy transcript was
verbose, but document and prove any intentional transcript/golden change.

## Fit-validity rule

Every candidate training run must satisfy the same engine-wide rule as ordinary
training, OBD, and CV:

- `Iterative::converged(stopReason)` must be true;
- the final error must be finite;
- cancellation, probe expiration, and maximum-iteration exhaustion are not
  convergence; and
- an ineligible candidate contributes no Wilks statistic or p-value.

Because stepwise selection compares all eligible candidates within a pass,
silently omitting one failed candidate can bias which variable is chosen.
Conservative required behavior: if any candidate needed for the current pass
does not converge, stop and fail the stepwise analysis with the variable/group,
stopping reason, iterations, and remediation. Preserve the partial audit trail.
Do not continue as though the comparison set were complete.

Do not loosen tolerances, raise ceilings, or enable plateau stopping merely to
make fixtures pass. Candidate clones already inherit the trained model's
configuration through the established copy path; keep that one authoritative
configuration path.

## API contract

Preserve blocking `POST /api/regress` for compatibility unless measurement
shows it is unused and Craig explicitly approves removal.

Add:

```text
POST /api/regress ...&async=1
```

which validates synchronously, starts the job, and returns immediately.

Use the established asynchronous job doors:

- `GET /api/train/status` for status/result, unless a general analysis-status
  endpoint is introduced coherently without breaking training/OBD/CV;
- `POST /api/train/stop` with an explicit empty body for cancellation; and
- HTTP 409 with `"busy":true` from all other engine-touching endpoints while
  stepwise owns the engine.

Do not overload the existing `obd` status object with misleading terminology.
Expose a clearly named object, for example:

```json
{
  "running": true,
  "stepwise": {
    "direction": "reverse",
    "step": 2,
    "candidate": 4,
    "candidatesThisStep": 7,
    "fitsCompleted": 11,
    "fitsRemaining": 24,
    "variable": 6,
    "inputs": [6, 7, 8],
    "phase": "training candidate"
  }
}
```

The exact names may follow existing conventions, but they must be documented
and stable. Once complete, `result` must include the same `ok`, `message`, and
`output` fields as blocking mode plus a structured `stepwise` result. Include
`cancelled` and partial history where applicable.

Every action remains recorded in `neuron_actions.log` with exact parameters.

## Engine design constraints

- Progress must come from structured callbacks/observers in the stepwise engine,
  not by scraping strings written to `util::screen()`.
- Cancellation must reach the `Iterative::Observer` of the current candidate
  clone.
- Do not mutate or replace the user's trained original network. Stepwise remains
  a standalone analysis over clones.
- Clean up the current candidate and worker safely on success, failure, and
  cancellation.
- Preserve deterministic seeded behavior. Progress observation must not consume
  RNG values or change the fitted result.
- Reporting cadence remains presentation-only and must not affect selection.
- Forward and reverse directions require the same progress, cancellation,
  convergence, and result contracts.
- Candidate fitting must not calculate held-out ROC/bootstrap statistics that
  the Wilks selection rule never consumes.

## Exact remaining-work accounting

At the beginning of a pass with `r` eligible conceptual variables:

- reverse considers `r` removal candidates;
- forward considers the variables not yet added; and
- after choosing one variable, the next pass has one fewer candidate.

The job can report the exact candidates in the current pass. It may report a
maximum remaining count across all possible later passes, but label it
accurately because the p-value threshold may stop the procedure early. Do not
present a maximum as a promise that every fit will run.

## Required red tests

Every new behavioral assertion must be watched to fail against the pre-fix
implementation for the stated reason.

### 1. Results are actually visible

Run a small deterministic stepwise analysis whose output contains known
variable/pass lines.

Assert:

- `/api/regress` returns nonempty `output` (control: this is already true);
- the GUI renders that output in a persistent Stepwise results pane; and
- the pane includes the final retained/selected variables, not only
  “regression complete.”

The page-render assertion must fail pre-fix because `regress()` discards
`j.output`. A source grep alone is weak; use the live page/browser where
possible, supplemented by smoke/API assertions.

### 2. Async returns before completion and progress advances

Use a deterministic long-enough neural stepwise fixture:

- `async=1` returns immediately;
- status reports `running:true` and a stepwise phase;
- at least two distinct progress states are observed;
- completed-fit count is monotonic;
- candidate/group identifiers agree with the supplied structure; and
- the final result matches blocking mode for the same seed/configuration.

### 3. Stop is real

Start a deliberately long neural stepwise run, wait until a candidate is
training, then send:

```bash
curl -X POST -d "" .../api/train/stop
```

Assert:

- Stop is acknowledged;
- the job reaches `running:false` promptly;
- the current candidate reports cancellation;
- no later candidate starts;
- the result says cancelled/not complete;
- partial history is retained; and
- the original trained model is unchanged and remains usable.

Prove this test would fail if Stop only changed a flag after all refits.

### 4. Busy contract

While stepwise is running, representative engine-touching endpoints must return
HTTP 409 with `"busy":true`. Read-only status remains available. A second
stepwise start must also be refused.

### 5. Unconverged candidate cannot enter selection

Construct a pass whose candidate hits a deliberately tiny iteration ceiling.

Assert:

- the analysis fails at that candidate;
- its reason is `max_iterations`/ceiling exhaustion;
- no Wilks statistic or p-value is emitted for that candidate;
- no variable is selected from the incomplete pass; and
- the visible/API result explains how to proceed.

Prove the test fails pre-fix because `RegressNet` currently consumes the
returned error and continues.

Repeat or parameterize for a non-finite error. Cover cancellation separately as
above.

### 6. Direction and grouping

Exercise both forward and reverse selection with grouped variables. Assert that
progress and final history always identify conceptual variables and their full
input-node groups; no one-hot group may be split.

### 7. Reproducibility and observer neutrality

For the same seed and configuration, blocking and async-with-progress runs must
produce identical candidate order, errors, statistics, p-values, and final
selection. Merely observing progress must not change RNG state or training.

### 8. Candidate fitting does not touch test statistics

Instrument a stepwise fixture with a real test set and bootstrap counter/call
guard. Assert that candidate refits:

- produce the same converged training errors, Wilks statistics, p-values, and
  selection as the current algorithm;
- perform no test ROC fit or bootstrap resampling; and
- do not write candidate training/test accuracy reports into the stepwise
  output.

Prove this fails pre-fix: the current Civic Choice log itself shows `Training
set`, `Test set`, and `2000 bootstrap resamples` for each candidate. Include a
measured before/after runtime on this 6,000-row walkthrough dataset, separating
the reporting savings from any optimization changes.

## Documentation and parity

Update in the same commit:

- `docs/gui_cli_parity.md`;
- `AGENTS.md` API and GUI operating instructions;
- `docs/HISTORY.md` with the measured live-walkthrough finding and red proofs;
  and
- any API/report specification that becomes authoritative for the structured
  stepwise result.

The CLI remains frozen. This work does not add a CLI feature; it makes the GUI
equivalent usable and keeps the existing blocking API compatible.

## Gates

Run:

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./tests/tools/run_tools.sh
./tests/golden/run_golden.sh
./tests/gui/smoke.sh
./tests/oracle/verify_oracle.sh
```

Perform a live GUI click-through for:

- visible progress;
- Stop;
- completed reverse results;
- completed forward results; and
- persistence of the original trained model.

If established goldens change, explain and prove why. Stepwise convergence
enforcement may legitimately expose a fixture that previously compared an
unfinished candidate; do not re-bless that output without measuring the cause.

## Acceptance criteria

Complete means:

- the GUI displays the report it currently discards;
- stepwise runs asynchronously with structured, advancing progress;
- graceful Stop interrupts the current candidate and prevents later work;
- busy behavior matches other long engine jobs;
- every candidate fit is converged and finite before entering Wilks selection;
- candidate fits do not repeatedly evaluate/bootstrap the test set;
- incomplete comparison passes cannot select a variable;
- final and partial results are auditable in the API and visible in the page;
- blocking and async results are reproducibly identical;
- forward and reverse grouped selection are covered;
- red proofs fail against the pre-fix implementation for the measured reasons;
- required documentation is current; and
- all gates and live GUI checks are green.
