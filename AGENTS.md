# Operating neuron — a guide for AI assistants

You are working in **neuron 3.0**, a neural computational modeling
environment: feed-forward neural networks and binary logistic regression
with serious statistics (ROC areas with 95% confidence intervals, Wald
tests, goodness-of-fit). A user will ask things like *"I have a dataset
here"*, *"train a model on this"*, or *"make a deployed model"*. This file
tells you exactly how to do each of those with what's in this repository.
Follow the recipes; they are verified. When you finish a task, tell the
user what the numbers mean, not just what they are.

Ground rules:

- **GUI/CLI parity is a hard contract (standing rule 5).** Every capability in
  the CLI menu interface must have a GUI equivalent — a page control *and* an
  HTTP API parameter. The GUI is the primary human interface; the CLI menus are
  frozen but authoritative. A CLI option with no GUI equivalent is a bug. If you
  touch the menus or the GUI, update **`docs/gui_cli_parity.md`** (the menu ↔
  control ↔ API matrix) in the same commit.
- **If you are about to write engine code, read standing rules 4, 6 and 7 in
  `CLAUDE.md` first.** In short: the class layer (`Matrix`, `vector_ops`,
  `Population`) is the numerical vocabulary and code must read against the
  equations it came from (rule 4); each mechanism has one authoritative
  implementation in the class that owns it (rule 6); and **speed is an
  architectural requirement** (rule 7) — prefer the destination-taking overload
  and the compound operators in loops, pass read-only containers by `const&`,
  keep `std::function`/virtual dispatch/allocation out of hot loops, measure
  before optimizing, and never hide a published formula to save lines.
- The engine is `build/neuron`, an interactive menu program. **Never drive
  it by typing interactively** — write the menu answers to a file, one per
  line, and run `./build/neuron --seed 42 < session.in > session.out`.
  Always pass `--seed N` so the run is reproducible; always read
  `session.out` afterward to harvest results (and check it doesn't end in
  an error or an unanswered prompt).
- The Python tools in `tools/` run on bare `python3` — never pip-install
  anything for them.
- The engine writes its logs (`model.txt`, `neuron.log`) into the
  directory of the first dataset file the session loads (falling back to
  the current directory if none is ever loaded) — so the run's artifacts
  end up next to the user's data, where they expect them.
- Costs: grooming is instant; the trainings below take seconds. If the
  *iteration table* is still printing after ~2 minutes, something is wrong —
  kill it and re-check the recipe (usually a missing iteration cap or
  missing `--refcat`). The statistical report AFTER training (including its
  2,000-resample ROC bootstrap per set) adds a few seconds — ~2 s on the
  3,391-row bank walkthrough, ~3 s at 12,000 rows — and scales gently
  (nothing in the report is quadratic any more). A short pause after
  "Total iterations" is the report working, not a hang.
- The train/test split (menu 5 / `/api/load` with a fraction) is stratified on
  the outcome and scales to large data — a 226,000-row raw load plus split is
  sub-second (it was O(n²) and would have taken minutes until ROADMAP 4).

## 0. Build the engine (once)

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/neuron --version        # → neuron 3.0.0-dev
```

Needs CMake and GSL (`brew install gsl` / `apt install libgsl-dev` /
`vcpkg install gsl`). If the build directory was copied from elsewhere,
`rm -rf build` first.

## 1. "I have a dataset" — grooming a CSV

Goal: a numeric, comma-separated file with **the outcome in the last
column** (binary outcomes must be 0/1), plus a key file naming the columns.

Look at the raw file first: delimiter? header line? text columns? what is
the outcome and is it last? (If the outcome isn't last, reorder columns
first with a small Python script.) Then:

```
python3 tools/mkdataset.py [--delimiter ';'] [--onehot [--refcat]] \
    --key key.txt --inputs inputs.txt -o data.txt raw.csv
```

Decision rules:

- Non-comma delimiter → `--delimiter ';'` (or `'\t'`).
- Any text columns (categories, yes/no outcome) → `--onehot`
  (auto-detects them; one-hot encodes inputs, maps a two-valued text
  outcome to 0/1).
- **Will any model with an intercept be trained — logistic regression
  especially? Add `--refcat`.** Without it, one-hot groups are perfectly
  collinear with the intercept: coefficients are not identifiable and
  training can grind forever. For neural networks either coding works;
  when in doubt, use `--refcat` so one groomed file serves both.
- Rows with blank outcomes are dropped automatically; columns with blanks
  become "indicator pairs" (2 columns). This is normal — the key file
  explains every output column.

Read the tool's output: it ends with **"Number of columns was X, which
would be X−1 input nodes"** — remember that input count, every later step
needs it. Show the user the one-hot report and the key file so they can
sanity-check the encoding.

Sample data to demo with: `docs/datasets/` (prostate-biopsy and
low-birth-weight are neuron-ready; bank-marketing is raw and its
`WALKTHROUGH.md` is the worked example of this whole pipeline).

## 2. "Train a model" — scripted sessions

Two templates. Both assume a groomed `data.txt` with **NI** inputs (from
step 1) in the current directory, hold out 25% as a test set, and finish
in seconds. Menu lines are answered one per line — comments here are for
you, do not put them in the file.

**Logistic regression** (interpretable: Wald tests per coefficient,
condition number; needs `--refcat` grooming):

```
1               # specify dataset
1               # number of input nodes...
NI              #   ...= your input count
2               # number of output nodes...
1               #   ...= 1
3               # load raw dataset
data.txt
5               # randomize into train/test
3               #   as a decimal fraction
0.25            #   25% to test
no              #   don't save training set
no              #   don't save test set
8               # save scaling factors  ← REQUIRED for deployment
scales.txt
14              # back to main menu
2               # specify model
5               # binary logistic → returns to main menu
3               # use model
1               # randomize weights
3               # stopping conditions
1               #   max iterations...
20000           #   ...capped (default 1M is far too many)
6               #   back
7               # train
1               # canonical backpropagation ← NOT Shanno/CGD (they can hang)
yes             # save the model  ← REQUIRED for deployment
network.txt
no              # don't save train guesses
no              # don't save test guesses
10              # back to main menu
9               # quit
yes
```

**Neural network** (SimpleProp, 5 hidden nodes): same file with the model
section (`2` through `5` above) replaced by `2` / `2` (hidden layers) /
`1` (one layer) / `5` (nodes) / `8` (return), and the iteration cap set to
`2000` instead of `20000`. Everything else identical.

Run: `./build/neuron --seed 42 < session.in > session.out`

Harvest from `session.out` and report to the user:

- Convergence: the iteration table's last lines — error flat and max
  gradient small (logistic should reach ~1e-6; that's the converged MLE).
- The **Test set** block (not just training): classification accuracy,
  sensitivity/specificity, and **ROC area with its 95% CI**.
  - **Report A_z as the area and quote its bootstrap 95% CI.** Wickens holds the
    trapezoidal area negatively biased and A_z the primary measure (pp. 70–72);
    the trapezoidal area and its Hanley–McNeil interval are a useful independent
    cross-check, and the two intervals should roughly agree — say so if they do.
    (Builds before 2026-07-15 printed a delta-method interval that was
    mis-specified and ~5× too narrow. If the CI line does not say "bootstrap
    resamples", you are on an old binary: quote the trapezoidal interval instead.)
  - **Quote the resample count and any failures** shown on the CI line. Failures
    track ties rather than occurring at random, so a nonzero count means a
    slightly narrow interval. On current builds this is normally "2000 bootstrap
    resamples" with no failures.
  - **Quote "Operating points fitted" with the area** — an A_z from five points and one
    from five hundred are different claims. (Advice to quote a *bin count* is obsolete:
    since 2026-07-15 there is no binning, and the "best p"/"best AUC" pair the report
    used to print is gone with it. There is one A_z. A build that still prints two, or
    prints "Bin size = ... Number bins = ...", predates the fix and its A_z depends on
    an arbitrary bin count — worth ~0.011 on Wickens' own data.)
  - `ITMAX too small in gcf` should no longer appear (its cause was fixed
    2026-07-15). If you see it on a current build, that is a real finding worth
    reporting, not a quirk to shrug at. `p = not available` is benign by design:
    goodness of fit never gates the area (Wickens §11.5, p. 217).
  - **Do not over-read the fit p on continuous data.** With a swept continuous score
    it rounds to 1.000 because the operating points are cumulated and so scatter far
    less than their own error bars — not because the fit is perfect. It is a real
    diagnostic only for rating-style data with few distinct scores.
  - Background for all of this: `docs/roc_theory.md`, which is the authority.
- For logistic: notable Wald rows (smallest p-values) — translate the
  input numbers to variable names via the key file — and the condition
  number (≫1e5 suggests collinearity; did they forget `--refcat`?).
  The condition number is the ratio of the extreme absolute eigenvalues of
  the **information matrix** `X'VX` — the same matrix the Wald standard
  errors invert. It measures the *design*, so it is deliberately computed
  **without** the weight-decay penalty: regularization improves conditioning
  by construction, and a penalized number could hide the collinearity this
  exists to reveal. Turning weight decay up will not move it.
- Warn if training accuracy ≫ test accuracy (overfitting), and if any
  input is something the user couldn't know at prediction time (like call
  `duration` in the bank demo — see WALKTHROUGH.md §8).

If a class is rare (say 10%), accuracy alone is misleading — the model can
score ~90% by always predicting "no". Judge by the ROC area and its CI.

## 3. "Make a deployed model" — web calculator

Needs the three files from step 2's session: `network.txt`, `scales.txt`
(both saves are marked REQUIRED in the template — if the user's earlier
session skipped them, re-run it with the saves), and a **label spec** you
write for the user. **Caveat — pre-normalized data:** if the dataset was
already scaled to roughly [−0.9, 0.9] before you got it (like
`docs/datasets/prostate-biopsy`), saving scaling factors from a training
run is useless — that run only sees normalized numbers and records a
near-identity mapping, so the calculator would demand normalized inputs. The
real natural→normalized scaling lives wherever the data was first normalized;
find it and hand-write a scales file in the engine's format (worked example:
`docs/datasets/prostate-biopsy/README.md`, "Deploying a trained model"). Full spec reference: `docs/deploy.md`. Short form —
one line per variable **in groomed-column order** (copy from the key
file), then outcome labels:

```
N %% Age %% years                                   ← numeric + units
C %% Education %% *primary %% secondary %% tertiary ← dropdown; * = the
                                                      refcat reference level
B %% Housing loan %% yes %% no                      ← binary, true first
K %% N %% PSA %% ng/mL                              ← indicator pair
O %% No cancer %% Cancer present                    ← 0-label, 1-label
R %% Odds of cancer                                 ← odds line (optional)
```

Ask the user for human-friendly names and units; don't ship raw column
names. A complete worked spec: `docs/datasets/bank-marketing/bank_spec.txt`.

```
python3 tools/neuron2web.py --network network.txt --scales scales.txt \
    --spec spec.txt -o calculator.html
```

The tool validates the spec against the network's input count and tells
you exactly what mismatches. Verify before handing over: run
`--eval "v1, v2, ..."` with a data row whose outcome is known (natural
units, no outcome column) and check the probability is sensible.

`calculator.html` is fully self-contained (all JavaScript inline): the
user can open it from disk, email it, or upload it to any static web host
— it runs entirely in the browser. `--serve` previews it locally on an
OS-assigned port.

## 3b. The GUI (`neuron --gui`)

For a human who prefers pointing and clicking: `./build/neuron --gui`
starts a local web page (127.0.0.1, OS-assigned port, URL printed; the
browser opens automatically — `--no-browser` suppresses that) with the
same load → model → train flow and a live ROC plot. The dataset panel
loads either a raw file (split by test fraction) or an **already-split
training set** — and in the latter mode a second picker takes an optional
**matched test set**, so a pre-split pair like
`docs/datasets/prostate-biopsy/BP40train.txt` + `BP40test.txt` loads in one
step and trains with a proper held-out test evaluation. The input count is
read from the file (its columns minus the output count — one output unless
the Outputs field says otherwise; multi-output datasets train as BackProp
with accuracy-only reports, since every ROC/classification statistic is
single-output). **Train continues from the
current weights**, so clicking Train again runs more iterations from where it
left off; **Randomize weights** (honoring the seed) resets to a fresh random
start. After each run the key results — ROC area with its 95% CI, accuracy,
sensitivity/specificity for train and test — appear beside the ROC plot, with
the full statistical report below. **As an agent, keep
using the scripted sessions of §2** — they are reproducible, capturable,
and testable; suggest `--gui` to the user when they want to explore
interactively. The GUI's file picker uploads the chosen file to the local
server, which saves a copy — and all logs and session files — in the
directory where `neuron --gui` was launched, so one directory per
experiment keeps everything together. After training, the "Session files"
buttons save/download the network + scaling factors (deployment trio
minus the spec) and the train/test sets, guesses, and report.

**Training is asynchronous in the GUI** (since 2026-07-16): the page shows a
realtime error-vs-iteration chart while a run is live, the Train button
becomes Stop, and stopping is graceful — the engine finishes the iteration
and produces its full report, so a stopped run is a completed run. For
scripts and agents driving the HTTP API with curl, the same machinery is:
`POST /api/train` with `async=1` returns immediately; poll
`GET /api/train/status` (running flag, decimated iter/train/test error
series, and — once finished — the same result JSON a blocking train returns,
including a `stopReason` of max_iterations / grad_max / cancelled / …);
`POST /api/train/stop` cancels — **send it as `curl -X POST -d "" …`, never a
bare `-X POST`**: a POST with no body and no Content-Length makes the server
wait its ~5 s read timeout for a body that never comes before dispatching, so
a bare-POST stop arrives seconds late (late enough to miss a short run
entirely; this bit on CI 2026-07-21). The same applies to any bodyless POST.
While a run is live every other
engine-touching endpoint returns **HTTP 409** with `"busy":true` — retry
after the run. Without `async=1`, `POST /api/train` blocks exactly as
before. `algorithm` accepts `1`–`3` or **`auto`** (since 2026-07-16): auto
probes all three optimizers on clones of the same starting weights for
750 ms each (batch/epoch forced for CGD and Shanno, as they require),
adopts the lowest-error probe — its progress kept — and continues training
it to `maxiter`. The result JSON gains
`autoAlgo{selected, selectedName, probes[{algorithm, name, error,
iterations, stopReason}]}` and the report opens with the decision summary;
a probe whose `stopReason` is `grad_max` **converged inside its budget**,
which usually settles the choice outright. `POST /api/train` also accepts
**`autostop=1`** (since 2026-07-19, off by default): the run stops once the
training error stops improving — a two-window moving average of the error,
compared window-over-window, that strikes when relative improvement falls
below `autostop_tol` (default `1e-4`) and stops after three consecutive
strikes; each window is `autostop_window` iterations wide (default `100`).
A rising error strikes too, so it doubles as crude overlearning protection,
and a diverged (NaN) run stops rather than running to `maxiter`. The result's
`stopReason` reads `plateau`; the page exposes it as an "Auto-stop on plateau"
checkbox. It is a genuine early stop, not a relabeling: the same run without
it runs on to `grad_max` or `max_iterations`. `POST /api/load` also accepts `discrete=0` for a continuous
(regression) outcome — with `mode=raw` that requires `fraction=0` (the
stratified split needs classes), so pre-split regression data loads via
`mode=train` plus `testfile`/`testpath`.

Since the 2026-07-19 parity work the API covers the **entire CLI menu surface**
(the authoritative option-by-option map is `docs/gui_cli_parity.md`). The
highlights for scripted use: `POST /api/model` takes `hidden=` (comma list —
several layers make it a BackProp), `bias=0` (→ BareProp),
`errfunc=auto|xentropy|lms`, `log_lastop=`/`log_history=`, and `mode=load`
(with the saved network as a `file` upload or `path=`) to resume a network
whose type is read from the file's first line. `POST /api/train` additionally
accepts `eta=`, `autostep=`, `batch_epoch=` (must stay on for logistic — the
API refuses otherwise, as the CLI does), `weight_decay=` + `decay=`,
`logprint=`/`printcount=`, and the stopping conditions `minerr=`, `change=`,
`errwindow=`, `gradmax=` — each applied **only when present** (omitting a
field keeps the model's current value; a present-but-empty stopping condition
disables it), which is how "stop, then continue from the held weights with new
parameters" works. **`logprint`/`printcount` change only how much output you
get back — never the fit.** Stopping conditions are evaluated every iteration,
independently of the printing schedule, so the same seed and parameters give
the same stopping iteration, weights and predictions whether you print every
iteration or logarithmically. (That was a real defect until 2026-07-26: the
gradient was recalculated only when a row was printed, so switching the print
counter moved the endpoint. `tests/iterative/check_gradcadence.cpp` guards it.) `POST /api/dfa` `type=linear|quadratic` runs a standalone
discriminant analysis — captured report plus, on 1-output data, ROC curves and
the statistics object; it does not disturb the trained model — and its guesses
save via `GET /api/save/dfa_train_guesses` / `dfa_test_guesses`.
`POST /api/load` further accepts `outputs=` (more than one → a multi-output
BackProp with accuracy-only reports, exactly the CLI's behavior), `test_n=`
(an exact test-set count instead of a fraction), `threshold=`,
`in_lower=`/`in_upper=`, `out_lower=`/`out_upper=` (continuous outcomes only),
`history=0` (dataset-operations logging off), and the ROC reporting settings
`roc_report=both|either`, `roc_min=`. (There is no `trap_thresholds` any more:
the trapezoidal ROC area is now the exact non-parametric AUC over every
operating point, so there is no threshold count to set.)

For a raw split the outcome is always balanced; to also balance covariates
(ROADMAP 4 Phase 2, a GUI-beyond-CLI feature) pass `strata=` a comma-separated
list of **1-based input column numbers** — e.g. `strata=10` stratifies on the
outcome × that column's cells, so a rare high-risk subgroup lands in the test
set proportionally rather than by luck. A column with few distinct values
becomes one level per value; a continuous column is cut into `strata_bins=`
quantile bins (default 4). The load response then carries a **representativeness
diagnostic** (strata count, train-vs-test outcome rate and per-column means) so
the balance is inspectable. This is the tool for a large, imbalanced, clumped
cohort (the SEER prostate-cancer set: `strata=10` matches M-stage prevalence
train-vs-test to four decimals).

**When to use `strata=` (and when NOT to).** The outcome is ALWAYS balanced —
that part is free and you should never turn it off. Covariate stratification
(`strata=`) is a deliberate choice, not a default. Reach for it only when:
- you will **report performance within a subgroup** (e.g. metastatic patients)
  and need that subgroup adequately populated in the held-out set; or
- the dataset is **small** and a random draw could imbalance a strong predictor; or
- you are doing **k-fold** and each fold is small enough that stratifying cuts
  fold-to-fold variance.

Do NOT add `strata=` when:
- you want the holdout to **detect covariate drift** — matching the test set to
  the training mix hides the very shift a held-out set exists to catch;
- you want an **independent** estimate rather than a balanced one (matching makes
  train and test more alike than two fresh samples, narrowing apparent variance);
- **n is large** — with tens of thousands of rows the outcome-only split already
  reproduces covariate marginals to a few parts per thousand, so covariate strata
  buy a rounding-level guarantee, not a real precision gain;
- it would **fragment the data**: every extra column (and every quantile bin)
  multiplies the cells, and thin cells round to zero test rows. Watch the
  diagnostic's "stratum(s) too small to place any test exemplar" line — if it
  fires, use fewer columns or fewer `strata_bins`.

Rule of thumb: **outcome stratification ≈ always; covariate stratification = a
deliberate choice for subgroup coverage on smaller data, not something to sprinkle
on by reflex.** If unsure, leave `strata=` off and read the outcome-1 rate in the
report — on a large set it is already balanced.

**Group-aware split (`group=`) — the opposite tool.** Where stratification makes
the test set RESEMBLE the sample, grouping makes it HARDER. Pass `group=` a
comma-separated list of 1-based input columns; rows sharing identical values on
ALL of them form a cluster (a hospital, a county) that is kept **entirely on one
side** of the split — no cluster straddles train and test. Use it when records
cluster and you want to estimate generalization to groups the model **never
trained on** (e.g. new counties), which is a more honest, usually lower number.
For the SEER cohort the four area-SES columns identify the county, so
`group=19,20,21,22` keeps each county intact (612 counties, none split, base rate
still balanced to five decimals). Because clusters are indivisible the achieved
test size only approximates the target and the outcome balance is preserved only
as far as whole clusters allow — the diagnostic (also `group=`-triggered) reports
what was achieved and states the zero-leakage guarantee. `group=` takes precedence
over `strata=`. Do NOT use grouping when you actually want performance on new
patients from KNOWN sites — that is the plain (optionally stratified) split.

**Three-way split (`val_fraction=`) — hold out a validation set (ROADMAP 4
Phase 4c).** By default the split is two-way (train/test) and automatic model
selection (OBD hidden-layer sizing) early-stops on the TEST set — which means the
net's reported test performance was tuned on the very rows it is scored on, an
optimistic **leak** when you compare it against logistic/DFA that did not tune
there. To avoid that, pass `val_fraction=` (a decimal, or `val_n=` with `test_n=`
for exact counts): the split becomes **train / validation / test**, all
outcome-stratified; OBD then monitors the **validation** set and the **test set
stays untouched** until final evaluation. Use this whenever you run OBD and intend
to report a fair comparison. It is outcome-stratified only — it does not compose
with `strata=`/`group=` yet (the request is refused if you combine them).

Every GUI/API user action is also appended — timestamped, with its exact
parameter values — to **`neuron_actions.log`** in the run directory beside the
data.

The GUI also runs **stepwise regression** on a trained network (the
"Stepwise regression" panel): give it the input-variable structure — the
same grouping string `mkdataset.py --inputs` writes and `psa_defs.txt`
holds, e.g. `0;1-4;5;6,7;8;9;10` (semicolons separate variables, commas
join nodes, hyphens give ranges) — a direction (reverse drops the least
significant variable at a time, forward adds the most significant), and a
p-value threshold. It refits subnetworks with each variable removed/added
and reports the Wilks-GLRT p-values, the engine's classic input-selection
tool. The network must be trained first. The full selection report is
rendered in a **persistent results pane below the panel** and, with history
logging on, is also appended to `neuron.log`. The trained model is not
disturbed — stepwise is a standalone analysis over clones.

Like OBD and CV it runs **asynchronously**: `POST /api/regress` with
`async=1` validates, starts the job and returns at once, and progress +
result come back through the same doors (`GET /api/train/status`,
`POST /api/train/stop` — send an explicit empty body). The status carries a
`stepwise` object naming the direction, the pass (`step`), `candidate` of
`candidatesThisStep`, `fitsCompleted`, and the conceptual `variable` with
its full `inputs` node group. Only the candidate count in the CURRENT pass
is reported: the threshold may end the procedure before later passes run, so
there is no wall-clock estimate. **Stop** reaches the candidate training at
that moment; the run then reports `cancelled` and keeps its partial audit
trail. Blocking mode (no `async`) is unchanged and returns the identical
result. Both modes also return a structured `stepwise` result: `threshold`,
`fitsCompleted`, the `path` of (variable, p) in selection order, and
`finalVariables`.

The status also carries `lastCompleted` — the variable, `converged` and
`stopReason` of the most recently finished candidate. It is sticky, persisting
while the next candidate trains, because a finished candidate is superseded
within microseconds and a poller would otherwise never see it. Both modes
return a `candidates` array in the structured result: every candidate
considered, in order, with its input group, prior and subnetwork errors,
degrees of freedom, Wilks statistic, p-value, convergence, stop reason,
iterations, and whether it won its pass. It is built as the run proceeds and
each candidate is recorded **before** its eligibility is judged, so a failed or
cancelled analysis includes the very candidate that ended it — with its stop
reason, and with `g2`/`p` null, because a rejected fit is recorded but never
compared. `fitsCompleted` always equals the length of `candidates`.

The report itself ends with a completion summary — direction, the selection
path in order, and **"Final retained variables"** (reverse) or **"Final
selected variables"** (forward), or `none`. It is written by the engine, so the
GUI results pane, the CLI transcript and `neuron.log` all carry the same
sentence. Only a completed analysis prints it: a run that failed or was
cancelled has no final set, and an empty list shown as a conclusion would claim
a result the procedure never produced.

The result also carries **`complete`**: false when an unconverged candidate or
a cancellation ended the analysis early. `finalVariables` reports what the
*completed* passes settled, maintained per pass rather than only at the end, so
`complete:false` with a non-empty list means "this is how far it got" while
`complete:true` means "this is the procedure's result". Check the flag before
reading the list as a conclusion.

**Stepwise requires a cross-entropy fit and never mutates the model.** Wilks'
GLRT compares log likelihoods, so `/api/regress` refuses an LMS-trained network
rather than converting it. (It used to call `setXEerror()` on the user's model:
that mutated a model stepwise treats as read-only, and it did not even work —
the baseline error every first-pass comparison uses was captured from the
original fit and stayed least-mean-squares, so `G2 = 2N(Esub - Efull)`
subtracted two different objectives. Logistic is cross-entropy by construction,
which is why this was invisible until an LMS network was tried.)

Two rules govern candidate fits. **Every candidate must converge**
(`Iterative::converged`, finite error) before its error may enter a Wilks
comparison; if one does not, the analysis FAILS at that candidate naming it,
its stop reason and its iterations, rather than skipping it — a pass selects
by comparing all its candidates, so silently dropping one biases which
variable wins. And **candidate refits are quiet**: they produce no training
narration and, crucially, no classification tables, ROC fit or test-set
bootstrap. Wilks selection reads the training likelihood; re-deriving
held-out statistics for every candidate was both a large avoidable cost and
a repeated look at a set that must stay untouched during model selection
(measured on the 189-row low-birth-weight split: 1.01 s → 0.15 s, 24
2000-resample bootstraps → 0, identical p-values and selection).

The GUI also sizes the hidden layer automatically — **OBD** (`POST /api/obd`,
the "OBD hidden-layer sizing" panel). It grows a SimpleProp's hidden layer while
a larger net lowers the held-out error, detecting overtraining by **validation
early stopping** (it samples the test error during each size's training and stops
that size the moment the error turns back up — no size trains to completion), then
prunes back by saliency. It needs a training/test split, and the winning sized,
trained network REPLACES the current model. Params: `hidden_start` (2),
`hidden_max` (30), `iter_budget` (per-size ceiling, 2000), `sample_every` (20),
`early_stop_tol` (0.02), `early_stop_patience` (3), `grow_patience` (2),
`prune_tol` (0.02), `autostop_tol` (1e-4) and `autostop_window` (100) for the
per-size train-plateau backstop (a size that converges flat never trips the
test-error rise; the plateau detector saves its remaining budget), `algorithm`
(1|2|3|auto — `auto` probes once and keeps the choice), `seed`. It is **async-only**: the call returns at once, and progress +
completion arrive through the same `GET /api/train/status` (a running OBD adds an
`obd:{phase,hidden}` field) and `POST /api/train/stop` doors as async training.
The phase is `"probing optimizers"` (algorithm `auto` only, announced once before
anything is fitted), then `"grow"` / `"prune"` / `"final"`. **The probe phase is
reported because it is wall-clock budgeted per candidate optimizer and can be the
largest single part of a search's elapsed time** — on a five-fold nested run over
189 rows it was ~11 s of 13 s, and reporting nothing there made a working search
look hung. It carries no measurement (iteration 0, both errors `-1`): it is a
status, never a chart sample. The success result also carries
`monitor:"validation"|"test"|"none"` — the held-out set the search actually scored
on, from the engine's own rule, so a caller labels its chart instead of assuming.
**Cost:** an OBD run is many training runs — roughly `iter_budget` × (sizes grown
+ pruned) iterations, though early stopping usually ends each size well under the
budget; keep `iter_budget` modest (hundreds to low thousands) and `hidden_max`
sane. See `docs/obd_plan.md`.

**The ceiling is a safety limit, not a stopping condition — everywhere, not
just in OBD.** Ordinary training keeps its weights when it hits `maxiter` (so you
can continue from them) but reports `converged:false` and `ceilingExhausted:true`
alongside `ok:true` — **operation success and fit validity are different facts** —
and the ENGINE report itself (CLI transcript, `neuron.log`, any captured report)
states plainly that training did not converge. A CV fold or locked-test refit
whose training ends at the ceiling, is cancelled, or produces a non-finite error
FAILS: it contributes no prediction and cannot reach a pooled AUC or a DeLong
contrast. `/api/cv` `autostop_tol`/`autostop_window` apply to the plain logistic
and fixed-architecture neural procedures as well as the nested-OBD search, which
is how you give them a stopping rule they can actually reach.

**In OBD specifically:** a trial that ends by reaching `iter_budget` did not
converge, so OBD will NOT compare its loss, select it, or prune from it. The search stops at the first such trial and refuses with
`ok:false`, `ceilingExhausted:true`, and a message naming the size — the whole
trial table is still printed, and `trials[]` gives each trial's `stopReason`,
`iterations`, and `eligible` flag. **This is the correct outcome, not a bug:** the
remedy is a bigger `iter_budget`, a different `algorithm`, or a stopping condition
that can actually fire. Only `grad_max`, `plateau`, `validation_early_stop` and the
configured error conditions count as convergence; `max_iterations`, `cancelled`
and a non-finite loss do not.

**Practical warning, measured:** with the shipped defaults (`autostop_tol=1e-4`,
gradient limit 1e-6) a slowly-converging problem may reach NO stopping rule within
any practical ceiling, so OBD refuses. On the 6,000-row Civic Choice fixture,
canonical and CGD refuse at 1,500 / 3,000 / 6,000 iterations, while **Shanno**
converges at 6,000 (plateau + validation early stop) and finds 5 hidden units with
test AUC 0.817. `algorithm=auto` picks Shanno there, so **Auto's advantage is not
only speed — it is often the only optimizer that reaches a real stopping condition
on a given dataset.** Raising `autostop_tol` to make a run "finish" is usually the
wrong move: at `0.01` the same fixture stops far too early and gives 1 hidden unit
with AUC 0.53 (canonical) — a worse model than refusing would have told you about.
Prefer a larger ceiling and Auto.

The GUI also runs **honest cross-validation model comparison** (`POST /api/cv`,
the "Cross-validation" panel). It scores several procedures — **logistic**, **LDFA**,
**QDFA**, and a **neural network** (with **nested OBD**: the whole architecture search
reruns inside each fold, so nothing leaks) — over ONE shared, outcome-stratified
k-fold plan, so their out-of-fold predictions are paired patient-for-patient. This
is the standard "I always run logistic + DFA alongside my chosen net" comparison,
done without leakage. Params: `folds` (k, ≥2, default 5), `seed` (42), `maxiter`
(per-fold cap for logistic / fixed-architecture neural, 500), the procedure flags
`logistic`/`ldfa`/`qdfa`/`neural` (1/0; default logistic+neural), `neural_obd`
(1 = nested OBD per fold, 0 = a fixed hidden count), `neural_hidden` (the fixed
count when `neural_obd=0`), `hidden_max`/`iter_budget` (the per-fold OBD search),
`algorithm` (1|2|3|auto for that search — **default `auto`**, probed independently
inside each fold and again for the locked refit; same encoding as `/api/obd`),
`autostop_tol`/`autostop_window` (the per-size plateau backstop),
`inner_val` (share of each fold's training rows held out as the inner validation
set OBD monitors, 0.25). A fold whose OBD search cannot finish a trial inside its
ceiling is a **failed fold**: it contributes no prediction, no architecture and no
optimizer metadata, its reason is printed per fold, and the pooled AUC reads `n/a`
rather than an average over whatever happened to fit. It is **async-only** and a **standalone analysis** — it does
NOT touch the current model — and reaches completion through the same
`GET /api/train/status` and `POST /api/train/stop` doors.

**A running CV publishes structured progress** in a top-level `cv` object (added
2026-07-29; the coarse `obd:{phase:"cross-validating"}` field is still sent
alongside it). It is assembled from two independent engine callbacks and is never
scraped from report text:

```
cv: { stage: "cross-validation" | "locked-test evaluation",
      procedure, procedureIndex, procedureCount, completedProcedures,
      fold, folds, completedFolds,
      inner: { phase, hidden } }      // only while a nested search is running
```

`crossval::ProgressFn` supplies the repetition grid (the runner fills fold/k, the
coordinator stamps the procedure identity onto it) and `obd::ProgressFn` supplies
the architecture trial inside the current fold; the GUI composes them. **CV never
learns what OBD is and OBD never learns what a fold is.** Read the fields
literally: `inner` ABSENT means no nested search is running (never "a search at
zero nodes"), `fold: 0` means no fold is running — the locked-test stage folds
nothing, and the comparison-complete event names neither fold nor procedure.
`inner` is stale-proof: it is cleared whenever the fold or procedure changes.

The result carries a **three-tier report**: `cv.tier1`
(a one-screen headline table — AUC-per-fold mean ± sd, which is DESCRIPTIVE spread
across dependent folds, **not** a confidence interval — plus a prespecified
descriptive contrast and the OBD architecture-selection frequency), `cv.tier2`
(per-fold AUC/sens/spec detail — **its fold-plan header's `n` and `events` are the
rows the fold plan COVERED**, which with a locked test are the development rows,
not the dataset totals Tier 1 reports; the same two numbers are `cv_run.json`'s
`n` and `events`), and `cv.files` (the paths of `cv_predictions.csv`
/ `cv_metrics.csv` / `cv_run.json`, written beside your data — the paired
out-of-fold substrate, never printed). `cv.files` lists ONLY files that were fully
written; if any could not be (unwritable dir, full disk) the run still succeeds with
the in-memory results and names the failure in `cv.warnings[]` — it never claims a
file was written unless it truly was. **Fold policy** (ROADMAP 4): `strata=` (comma-separated 1-based input columns) and
`strata_bins=` give outcome × covariate-stratified folds; `group=` gives
group-disjoint folds where rows sharing identical values on all named columns are
one indivisible cluster, never split across folds — **and the same key governs the
locked holdout**, so a cluster cannot straddle development and locked either. These
are the CV request's OWN parameters and are never inherited from `/api/load`'s
`strata=`/`group=`, which size a train/test holdout and answer a different question.
Absent both, the fold plan is the shipped outcome-stratified k-fold, unchanged. The
two **cannot be combined** yet and the combination is refused, rather than one
silently winning. Stratifying balances the subgroups your sample already represents;
grouping measures transfer to groups the model never saw. **Neither makes rows
independent** — see `independence=` below, which is a separate axis. A grouped run is
additionally preflighted for ≥ k development *groups* (a thousand rows in three
counties still cannot fill five group-disjoint folds). **Cost:** CV is k trainings per procedure,
and the nested-OBD procedure is k full OBD searches, so it is the most expensive
run in the GUI — start with small `folds`/`hidden_max`/`iter_budget` on large data.
No formal cross-validation inference is reported (the fold results are dependent);
the inferential comparison is a **locked untouched test set** — supported in the same
call (see next paragraph).

**Locked-test inference.** Add `locked_fraction` (0–1, `0` = none) or
`locked_n` (a count — supply one, not both) to set aside a locked test held
ENTIRELY out of CV. It is outcome-stratified by default and group-disjoint when
`group=` is supplied. Then CV folds only the development rows,
each procedure is **refit on the development rows by its own prespecified rule**
(logistic/DFA on all dev rows; nested OBD carves its own inner validation split of
the dev rows — no forced full-dev refit), and scored **once** on the untouched
locked test. Under a grouped fold plan that inner validation split is
**group-disjoint too**: group-disjoint outer folds alone would still let the
architecture be chosen using rows from clusters in its own inner training set, so
the "generalizes to unseen groups" claim would cover the score but not the
selection. A fold whose training rows turn out to be a single group has no such
split and **fails that fold with the reason** — there is no fallback to a row-wise
inner validation. `cv.locked` carries per-procedure **point AUCs** and predictions;
`cv_locked_predictions.csv` (raw row id, outcome, one column per procedure — the
auditable pairing, written whenever predictions exist; a `cluster` column follows
the row id **only** for a grouped design) joins the Tier-3 files.
**Inference is OPT-IN (this is the guardrail).** Ordinary DeLong assumes
*independent* test observations, which a mechanical row holdout does not establish
— so the AUC 95% CIs and the contrast *p* are produced **only when you declare the
sampling unit**: `independence=rows` (independent observations). Without it you
still get predictions + point AUCs, but ordinary DeLong is **withheld** with an
explanation (never a silently invalid *p*, e.g. on clustered SEER county data).
`independence=cluster` routes to **Obuchowski's clustered ROC covariance**
(*Biometrics* 1997;53:567–578) — the cluster, not the row, is the independent unit.
It requires a `group=` key (there is nothing to cluster on without one) and a locked
test, and the refusal names whichever is missing; it **never** falls back to ordinary
DeLong, and a clustered *p* is never labelled "DeLong". The locked sample is
preflighted for **≥ 2 clusters carrying each outcome class** (the estimator's own
divisor is `informative clusters − 1`, so this is a cluster condition, not a row one:
fifty locked rows from one cluster carry no between-cluster information). Tier 1
states how many independent clusters the interval and *p* rest on, because a *p* from
four clusters and one from four hundred otherwise read identically. Clustered area
intervals are deliberately **not** clamped to [0,1] (the reference implementation's
behaviour, and an interval past the boundary is information — the ordinary path still
clamps). `independence=rows` **together with** `group=` is refused: a
group-disjoint design prevents leakage but does not make the held-out rows
independent, so ordinary DeLong would not be valid, and calling the grouping
"descriptive" would not repair the *p*. When inference runs, Tier 1 gains the
`AUC (test) [95% CI]` column and the prespecified-contrast verdict (ΔAUC plus the
two-sided *p* from the estimator named in the report; a *deterministic separation* of areas
reports p≈0, and *equal areas* report "no testable difference"); `cv.locked`
adds the CIs and the signed contrast (`delta` = AUC(primary) − AUC(reference)) plus
`samplingUnit`/`independenceStatus`/`inferenceMethod`/`inferenceRan` metadata. When
inference is withheld, `inferenceReason` states **why** — an undeclared sampling
unit, a locked test too sparse to estimate a covariance, and a design that forbids
the estimator are three different facts with different remedies, and the report
names the one that applied. `cv.locked` also carries `splitMethod`/`splitPlan` and
`clusters` (0 = an ungrouped design, not "unknown"); `cv_run.json` carries the same
design as **structure** in `foldDesign` and `lockedTest.splitDesign` (method, seed,
`k`, the 1-based strata/group columns, group count, achieved-vs-requested size,
leakage, warnings, refusal) rather than only as the prose beside it. The
contrast is set with `primary`/`reference` tokens (`logistic`|`ldfa`|`qdfa`|
`neural`) which **must name selected procedures** (an unselected one is a
validation error, never a silent default); absent, it defaults to neural vs
logistic only when both are selected. The locked size is validated for **achieved
per-class counts** (not just the total): the locked test needs ≥ 2 events and ≥ 2
non-events (DeLong's minimum), and the **development set needs ≥ k of each class**
so every outer fold can contain both — a rare-event request that would leave some
fold without an event is refused up front with the counts and `k`.

**Reproducibility:** for a given `seed`, each procedure runs on its own deterministic
RNG substream keyed by its NAME and fold, so a procedure's fitted CV predictions are
invariant to which OTHER procedures you compare and in what order (adding logistic to
the run does not change the neural net's numbers). See
`docs/evaluation_report_spec.md` and `docs/cross_validation.md`.

## 4. Verifying the installation

- Quick: `./tests/tools/run_tools.sh` (Python tools vs committed outputs,
  including the deployment forward-pass check against the engine) and
  `./tests/golden/run_golden.sh` (three seeded engine sessions -- `xor_seed42`,
  `regress_seed42`, `binormal_seed42` -- must match
  committed transcripts byte-for-byte).
- The low-birth-weight dataset is a self-verifying reference: follow
  `docs/datasets/low-birth-weight/README.md` and the engine should report
  log likelihood −111.2865 on the committed betas.

## 5. Repository map (for when you need more)

| Where | What |
|---|---|
| `src/`, `build/neuron` | The C++ engine (menus: dataset → model → train → stats) |
| `tools/mkdataset.py` | CSV → neuron-ready dataset (this file, §1) |
| `tools/neuron2web.py` | trained model → standalone HTML calculator (§3) |
| `docs/datasets/bank-marketing/WALKTHROUGH.md` | The full pipeline, every step verified with real output |
| `docs/deploy.md` | Label-spec reference for deployment |
| `docs/roc_theory.md` | What the ROC statistics mean and how to cite them |
| `docs/manifest.pdf` | The complete manual (menu-by-menu) |
| `CLAUDE.md` | Standing rules, current state, settled decisions, roadmap (for working **on** neuron rather than **with** it) |
| `docs/HISTORY.md` | The dated development record — read on demand for the reasoning behind a decision |

Maintainers: keep this file in sync when tools or menus change — it is the
operating manual for AI assistants, and its recipes are promised to work.

**If you are changing the engine rather than using it**, read the standing rules
at the top of `CLAUDE.md` first. The short version, learned expensively: a green
test suite is not evidence until you know it executes what you changed — run a
new test against the old binary and watch it fail before you trust it — and a
doc that names a mechanism (including one in this repo) is a hypothesis, not a
finding. Measure it.
