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
  a few seconds on datasets with thousands of rows — and scales gently
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
low-birth-weight are neuron-ready; civic-choice is synthetic and has the
maintained illustrated walkthrough).

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
- Warn if training accuracy ≫ test accuracy (overfitting), and if any input is
  something the user could not know at prediction time.

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
names. Maintained worked specs live beside their datasets under `docs/datasets/`.

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

## 3b. The GUI and loopback API (`neuron --gui`)

`./build/neuron --gui` starts the browser interface on loopback; use
`--no-browser` for scripted smoke work. The GUI exposes the frozen CLI surface
through page controls and HTTP fields, plus documented evaluation extensions.

Do not load the entire API contract for an unrelated training task. For GUI or
HTTP work, read these focused authorities:

- `docs/gui_cli_parity.md`: menu ↔ page control ↔ API field contract.
- Manifest REST chapter: endpoint fields, defaults, strict parsing, response and
  busy/error behavior.
- `docs/cross_validation.md` and `docs/evaluation_report_spec.md`: CV, nested
  selection, locked-test inference and artifacts.
- `docs/obd_plan.md`: historical OBD implementation rationale; current public
  contract is the Manifest and source.
- `docs/b9_strict_parsing.md`: historical B9 evidence; current accepted field
  tokens are the Manifest REST tables.

Operational invariants that belong in every GUI session:

- Long training, OBD, CV and stepwise operations are asynchronous when requested.
  Poll `GET /api/train/status`; cancel with `POST /api/train/stop`.
- Send bodyless POSTs with an explicit empty body, for example
  `curl -X POST -d "" ...`; otherwise the HTTP library waits for its read timeout.
- While a long job owns the engine, other engine-touching endpoints return 409
  with `busy:true`.
- Training continues from current weights. Randomize explicitly for a fresh start.
- Present boolean fields use exactly `1` and `0`. Unknown, trailing, overflowing,
  or non-finite numeric text is a field-specific error; omission and an empty
  field retain the endpoint-specific defaults documented in the Manifest.
- The GUI and API use one-based input-column numbers; C++ engine methods use
  zero-based positions.
- GUI actions are appended to `neuron_actions.log` in the run directory.
- If a menu or GUI/API capability changes, update `docs/gui_cli_parity.md` in the
  same commit and exercise both blocking and asynchronous forms where applicable.

For agents, prefer the scripted CLI sessions in §2 unless the task specifically
concerns GUI/API behavior. The verified end-to-end API gate is
`tests/gui/smoke.sh`; focused GUI characterization lives under `tests/gui/`.

## 4. Verifying the installation

- Quick: `./tests/tools/run_tools.sh` (Python tools vs committed outputs,
  including the deployment forward-pass check against the engine) and
  `./tests/golden/run_golden.sh` (three seeded engine sessions -- `xor_seed42`,
  `regress_seed42`, `binormal_seed42` -- must match
  committed transcripts byte-for-byte).
- GUI: `./tests/gui/smoke.sh` (every endpoint and the page's own controls) and
  `./tests/gui/asyncjob.sh` (the async-job lifecycle underneath all four long
  jobs — the busy gate, the two open doors, cancellation, and the reset that
  keeps one job's progress out of the next one's status). Both start a real
  server on an OS-assigned port; the second is separate so that a hang or a
  terminated server there cannot take the endpoint coverage with it.
  The same lifecycle's *ordering* rules — a result is published before the job
  reports itself idle, and the job is marked running before `start()` returns —
  are pinned in-process by the `asyncjob_lifecycle` ctest case, because neither
  is observable through HTTP (`docs/refactor_audit.md` §20.4).
- Request fields: `./tests/gui/strictparse.sh` (every handler's omission rules,
  present-but-empty rules, domain refusals, and the malformed values each field
  must refuse by name). Third server-starting script, separate for the same
  reason as the second. The contract it pins is `docs/b9_strict_parsing.md`.

### How a request field is read (since 2026-08-03)

Every field on every endpoint goes through the same rules, so a malformed
request is refused by name instead of being silently reinterpreted:

- **Whitespace** around a value is trimmed, and the rest must be consumed
  entirely. `fraction=0.3` and `fraction=" 0.3 "` are the same; `fraction=0.3junk`
  is refused. There is no partial reading.
- **Whole numbers**: optional `+`, then digits. A negative is a sign error, and
  a value wider than the type is a range error, not a wrap.
- **Numbers**: anything `strtod` accepts, consumed entirely, and **finite** —
  `nan` and `inf` are refused on every field.
- **Flags**: exactly `1` or `0`, case-sensitively, everywhere. `true`, `yes`
  and `TRUE` are refused; before this they silently meant false.
- **Omitted** keeps the current value. **Present-but-empty** does too for
  numbers and counts — the page sends `fraction`, `seed` and `decay` on every
  request whether or not they are filled in — with two exceptions: on
  `/api/train` an empty `minerr`, `change`, `errwindow` or `gradmax` *disables*
  that stopping condition, and an empty **flag** is refused.
- **Refusals name the field**: `folds: '5junk' is not a whole number`,
  `gradmax: 'inf' is not a finite number`, `async: 'true' must be 1 or 0`. A
  syntax fault and a domain fault read differently on purpose.
- **A refusal applies nothing.** Each handler reads all of its fields before it
  changes anything, so a request refused for its last parameter has not applied
  its first.

Every curl recipe in this file already uses `=1`/`=0` and unambiguous numbers,
so none of them changes. Writing a new endpoint or parameter means using
`readUnsigned` / `readDouble` / `readBool` in `src/gui.cpp`;
`tools/check_strict_parsing.py` fails the build otherwise.
- The low-birth-weight dataset is a self-verifying reference: follow
  `docs/datasets/low-birth-weight/README.md` and the engine should report
  log likelihood −111.2865 on the committed betas.

## 5. Repository map (for when you need more)

| Where | What |
|---|---|
| `src/`, `build/neuron` | The C++ engine (menus: dataset → model → train → stats) |
| `tools/mkdataset.py` | CSV → neuron-ready dataset (this file, §1) |
| `tools/neuron2web.py` | trained model → standalone HTML calculator (§3) |
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
