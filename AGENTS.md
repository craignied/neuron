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
`POST /api/train/stop` cancels. While a run is live every other
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
parameters" works. `POST /api/dfa` `type=linear|quadratic` runs a standalone
discriminant analysis — captured report plus, on 1-output data, ROC curves and
the statistics object; it does not disturb the trained model — and its guesses
save via `GET /api/save/dfa_train_guesses` / `dfa_test_guesses`.
`POST /api/load` further accepts `outputs=` (more than one → a multi-output
BackProp with accuracy-only reports, exactly the CLI's behavior), `test_n=`
(an exact test-set count instead of a fraction), `threshold=`,
`in_lower=`/`in_upper=`, `out_lower=`/`out_upper=` (continuous outcomes only),
`history=0` (dataset-operations logging off), and the ROC reporting settings
`trap_thresholds=`, `roc_report=both|either`, `roc_min=`. Every GUI/API user
action is also appended — timestamped, with its exact parameter values — to
**`neuron_actions.log`** in the run directory beside the data.

The GUI also runs **stepwise regression** on a trained network (the
"Stepwise regression" panel): give it the input-variable structure — the
same grouping string `mkdataset.py --inputs` writes and `psa_defs.txt`
holds, e.g. `0;1-4;5;6,7;8;9;10` (semicolons separate variables, commas
join nodes, hyphens give ranges) — a direction (reverse drops the least
significant variable at a time, forward adds the most significant), and a
p-value threshold. It refits subnetworks with each variable removed/added
and reports the Wilks-GLRT p-values, the engine's classic input-selection
tool. The network must be trained first; output appears in the report pane
and, with history logging on, is appended to `neuron.log`.

## 4. Verifying the installation

- Quick: `./tests/tools/run_tools.sh` (Python tools vs committed outputs,
  including the deployment forward-pass check against the engine) and
  `./tests/golden/run_golden.sh` (two seeded engine sessions must match
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
| `CLAUDE.md` | Development log and roadmap (for working **on** neuron rather than **with** it) |

Maintainers: keep this file in sync when tools or menus change — it is the
operating manual for AI assistants, and its recipes are promised to work.

**If you are changing the engine rather than using it**, read the standing rules
at the top of `CLAUDE.md` first. The short version, learned expensively: a green
test suite is not evidence until you know it executes what you changed — run a
new test against the old binary and watch it fail before you trust it — and a
doc that names a mechanism (including one in this repo) is a hypothesis, not a
finding. Measure it.
