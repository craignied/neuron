# GUI / CLI parity matrix

**This file is a contract, not a report.** Standing rule 5 (CLAUDE.md): every
capability in the CLI menu interface MUST have a GUI equivalent — a page control
*and* an HTTP API parameter. A CLI menu option with no GUI equivalent is a bug.
Any change to the menus (`src/neuron.cpp`) or the GUI (`src/gui.cpp`,
`src/gui_page.html`) updates the matching row here in the same commit.

The CLI is the authoritative feature list because its menus are frozen; the GUI
is the primary human interface and must be a superset. "API param" is the field
`POST` accepts; "GUI control" is the page element that sends it.

Status: ✅ present · 🔲 gap (must be closed before the change lands) · — n/a.
As of 2026-07-19 (second audit — the first missed the dataset characteristics,
ROC-reporting, and DFA-guesses rows below, all closed the same day) there are
**no gaps**. A new 🔲 row appears only when a menu/GUI change opens a gap, and
must be closed (turned ✅) in the same change.

## Main menu

| CLI menu option | GUI control | API | Status |
|---|---|---|---|
| 1 Specify dataset | § Dataset panel | `POST /api/load` | ✅ |
| 2 Specify model | § Model panel | `POST /api/model` | ✅ |
| 3 Use model (train) | § Train panel | `POST /api/train` | ✅ |
| 4 Discriminant function analysis | § DFA panel | `POST /api/dfa` | ✅ |
| 9 Quit | — (server lifecycle) | — | — |

## Dataset submenu

| CLI menu option | GUI control | API | Status |
|---|---|---|---|
| 1 Number of input nodes | derived from the file's columns | `POST /api/load` `inputs=` (override) | ✅ |
| 2 Number of output nodes (incl. >1 → BackProp) | Outputs field | `POST /api/load` `outputs=` | ✅ |
| 3+5 Load raw + randomize into train/test | mode=raw + Test fraction / exact n | `POST /api/load` `mode=raw&fraction=` or `test_n=` | ✅ |
| 4 Convert raw → training set (scale, no split) | mode=raw + fraction=0 | `POST /api/load` `fraction=0` | ✅ |
| 6+9 Load training set / test set | mode=train + Test set upload | `POST /api/load` `mode=train`,`testfile` | ✅ |
| 7+10 Save training set / test set | § Session files buttons | `GET /api/save/{train_set,test_set}` | ✅ |
| 8 Save scaling factors | § Session files → Scaling factors | `GET /api/save/scales` | ✅ |
| 11 Log dataset operations to history file | Log-dataset-ops toggle | `POST /api/load` `history=` | ✅ |
| 12 Output discrete / continuous | Outcome select | `POST /api/load` `discrete=` | ✅ |
| 12 Threshold for discrete output | Threshold field | `POST /api/load` `threshold=` | ✅ |
| 12 Input variate lower/upper limits | Scaling-bounds inputs fields | `POST /api/load` `in_lower=`,`in_upper=` | ✅ |
| 12 Output variate lower/upper limits (continuous) | Scaling-bounds outputs fields | `POST /api/load` `out_lower=`,`out_upper=` | ✅ |
| 13 Statistical/trapezoidal both-or-either | ROC report select | `POST /api/load` `roc_report=both|either` | ✅ |
| 13 Minimum data for statistical ROC | ROC statistical-min field | `POST /api/load` `roc_min=` | ✅ |

Menu 13's per-set (train/test/both) selector collapses in the GUI: both sets
arrive in one `/api/load`, and an explicit setting is applied to both. Menu 5's
three entry forms map to two params: `fraction` covers the decimal and
numerator/denominator forms (the CLI's own `randomize(n,d)` delegates to
`randomizeD`), and `test_n` is the whole-number form (`randomize(n)` exactly —
`randomizeD` truncates `ratio·N`, so a fraction cannot promise an exact count).

Menu 13's former "trapezoidal thresholds" option was **removed from both
interfaces** (2026-07-19), so parity holds: the trapezoidal ROC area is now the
exact non-parametric AUC integrated over every operating point (the same
`operatingPoints()` sweep the statistical method and bootstrap use), so there is
no threshold count to configure. The CLI submenu renumbered accordingly (1
statistical/trapezoidal, 2 minimum data, 3 return); the GUI dropped the
`trap_thresholds` field and param.

## Model submenu

| CLI menu option | GUI control | API | Status |
|---|---|---|---|
| 1 Bias nodes on/off | Bias-nodes toggle | `POST /api/model` `bias=` (off → BareProp) | ✅ |
| 2 Hidden layers / nodes | Hidden nodes field | `POST /api/model` `hidden=` (comma list → BackProp) | ✅ |
| (factory) Multi-output dataset → BackProp | Outputs field (Dataset panel) | `POST /api/model` (automatic, as the CLI factory) | ✅ |
| 3 Output error function (LMS / X-entropy) | Error-function select | `POST /api/model` `errfunc=` (create AND load) | ✅ |
| 4 Load network from a file (incl. BackProp bias from line 2) | Load-network upload | `POST /api/model` `mode=load` | ✅ |
| 5 Binary logistic regression | Model-type select | `POST /api/model` `type=logistic` | ✅ |
| 6 Log last operation to file | Log-last-op toggle | `POST /api/model` `log_lastop=` | ✅ |
| 7 Log to history file | Log-history toggle | `POST /api/model` `log_history=` | ✅ |

## Use-model / Train submenu

| CLI menu option | GUI control | API | Status |
|---|---|---|---|
| 1 Randomize weights | Randomize weights button | `POST /api/randomize` | ✅ |
| 2 Learning rate (auto / manual η) | Auto-step toggle + η field | `POST /api/train` `eta=`,`autostep=` | ✅ |
| 3 Stopping: max iterations | Max iterations field | `POST /api/train` `maxiter=` | ✅ |
| 3 Stopping: lower error limit | Min-error field | `POST /api/train` `minerr=` | ✅ |
| 3 Stopping: change-in-error limit | Change field | `POST /api/train` `change=` | ✅ |
| 3 Stopping: error-window increase | Window field | `POST /api/train` `errwindow=` | ✅ |
| 3 Stopping: max absolute gradient | Grad-max field | `POST /api/train` `gradmax=` | ✅ |
| (new) Plateau auto-stop tol / window | Auto-stop checkbox + tol/window fields | `POST /api/train` `autostop`,`autostop_tol`,`autostop_window` | ✅ |
| 4 Batch/epoch on/off (forced ON for logistic, as the CLI forces it) | Batch/epoch toggle | `POST /api/train` `batch_epoch=` | ✅ |
| 5 Weight decay (on/off + λ) | Weight-decay toggle + λ field | `POST /api/train` `weight_decay=`,`decay=` | ✅ |
| 6 Print counter (log / linear) | Print-counter select + count | `POST /api/train` `logprint=`,`printcount=` | ✅ |
| (train-time) Algorithm (GD/CGD/Shanno/auto) | Algorithm select | `POST /api/train` `algorithm=` | ✅ |
| 7 Train model | Train button | `POST /api/train` | ✅ |
| 7/8 Save network + guesses after training | § Session files → Network / guesses | `GET /api/save/{network,train_guesses,test_guesses}` | ✅ |
| 9 Stepwise regression | § Stepwise regression panel | `POST /api/regress` | ✅ |

## DFA submenu

| CLI menu option | GUI control | API | Status |
|---|---|---|---|
| 1 Linear DFA (incl. multi-output accuracy report) | DFA panel → Linear + Run | `POST /api/dfa` `type=linear` | ✅ |
| 2 Quadratic DFA (incl. multi-output) | DFA panel → Quadratic + Run | `POST /api/dfa` `type=quadratic` | ✅ |
| 3+4 Logging toggles | Model panel's log toggles (sent along) | `POST /api/dfa` `log_lastop=`,`log_history=` | ✅ |
| (after run) Save the DFA's guesses | DFA train/test guesses buttons | `GET /api/save/{dfa_train_guesses,dfa_test_guesses}` | ✅ |

## GUI-beyond-CLI features

The contract is GUI ⊇ CLI: the GUI must cover every CLI menu option, but it may
also do MORE. The CLI menus are frozen (no new features, 2026-07-14), so these
have **no CLI equivalent by design** — that is not a parity gap.

| Feature | GUI control | API | CLI |
|---|---|---|---|
| Automatic training-algorithm selection | Algorithm → "Auto" | `POST /api/train` `algorithm=auto` | — n/a (menus frozen) |
| Plateau auto-stop | "Auto-stop on plateau" + tol/window | `POST /api/train` `autostop=` | — n/a (menus frozen) |
| Realtime error-vs-iteration chart | Training-error chart | `GET /api/train/status` series | — n/a (menus frozen) |
| DFA graded ROC AUC | DFA ROC/stats panels | `POST /api/dfa` (ROC in the response) | — n/a (menus frozen) |
| OBD hidden-layer sizing (grow-then-prune, validation early stopping) | OBD panel + size-vs-error chart | `POST /api/obd` (async; `GET /api/train/status` `obd{phase,hidden}`) | — n/a (menus frozen) |
| Covariate stratification of the raw split + representativeness diagnostic (ROADMAP 4 Phase 2) | Dataset panel "Stratify on" columns + bins | `POST /api/load` `strata=` (1-based cols) `strata_bins=` | — n/a (the CLI already stratifies on the outcome; covariate strata are new, menus frozen) |
| Group-aware split — keep clusters intact for a harder unseen-group test (ROADMAP 4 Phase 3) | Dataset panel "Group on" columns | `POST /api/load` `group=` (1-based cols; rows with identical values stay together) | — n/a (new capability, menus frozen) |
| Three-way split — train/validation/test so selection (OBD) monitors validation and the test set stays untouched (ROADMAP 4 Phase 4c) | Dataset panel "Validation fraction" | `POST /api/load` `val_fraction=` (or `val_n=` with `test_n=`) | — n/a (new capability, menus frozen) |
| Cross-validation model comparison — logistic / LDFA / QDFA / neural (nested OBD) over ONE shared outcome-stratified k-fold plan, three-tier report (ROADMAP 4 Phase 4) | Dataset-independent "Cross-validation" panel + pinned Tier-1 headline table | `POST /api/cv` (async; three-tier report in the result) | — n/a (new capability, menus frozen) |

`POST /api/obd` params: `hidden_start`, `hidden_max`, `iter_budget`,
`sample_every`, `early_stop_tol`, `early_stop_patience`, `grow_patience`,
`prune_tol`, `autostop_tol`/`autostop_window` (the per-size train-plateau
backstop), `algorithm` (1|2|3|auto), `seed`. It is async-only and shares the
training job (status + stop reach it through the same doors); the winning sized
network replaces the current model.

`POST /api/cv` params: `folds` (k, ≥2), `seed`, `maxiter` (per-fold cap for
logistic / fixed-architecture neural), procedure flags `logistic`/`ldfa`/`qdfa`/
`neural`, `neural_obd` (nested OBD per fold vs a fixed count), `neural_hidden`
(the fixed count), `hidden_max`/`iter_budget` (the per-fold OBD search),
`inner_val` (share of each fold's training rows held out as the inner validation
set OBD monitors). Async-only, shares the training job (status + stop). It is a
standalone analysis — it does NOT touch the current model. The result carries the
report as text (`cv.tier1`/`cv.tier2`) and the paths of the Tier-3 files
(`cv_predictions.csv` / `cv_metrics.csv` / `cv_run.json`, written beside the data).
The fold plan is outcome-stratified k-fold; composing CV with the covariate-strata
/ group-aware split modes is a later extension.

## Logging (cross-cutting)

Standing requirement (Craig, 2026-07-19): **every user action is logged.**

| Behavior | Mechanism | Status |
|---|---|---|
| Engine operations (dataset split, randomize, each training run incl. stopped) | `neuron.log` via `addHistory` | ✅ (pre-existing) |
| Per-action audit trail (load/model/dfa/train/obd/randomize/stop/save) with timestamp + values | `neuron_actions.log` via `logAction` | ✅ |
| Each training run's header records ALL params in effect (η, decay, stopping conditions, plateau) | run header (self-describing) | ✅ (already: `Network::runHeader` + `Iterative::train`) |
