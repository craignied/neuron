# GUI / REST contract and legacy CLI coverage

**This file is a contract, not a new-work checklist for the CLI.** Standing rule
5 freezes the legacy interactive menus permanently: do not add menu entries,
prompts, options, or equivalents for new features. The menu remains only for
compatibility, scripted operational use, oracle comparison, and regression
testing.

The REST API is authoritative for every new interactive capability. The GUI is
its primary human client and receives a visible control when one is appropriate.
A GUI/REST surface change updates this file in the same commit. The tables below
retain the completed menu-to-GUI mapping as historical compatibility evidence;
they do not authorize or require further work in `src/neuron.cpp`.

**How every API parameter below is read** is one contract, not per-row detail:
whitespace-trimmed and fully consumed; whole numbers with no wrap and no
silent truncation; numbers finite, so `nan` and `inf` are refused everywhere;
flags exactly `1` or `0`, case-sensitively, on every endpoint; an absent field
keeps the current value; a present-but-empty *number* does too, except that
`/api/train`'s `minerr`, `change`, `errwindow` and `gradmax` disable their
stopping condition and `printcount` is ignored; a present-but-empty *flag* is
refused. Refusals name the field and quote the text. → `docs/b9_strict_parsing.md`
and the Manifest's REST chapter. **A new parameter is read through
`util::parseUnsigned` / `parseDouble` / `parseBool` via `gui.cpp`'s
`readUnsigned` / `readDouble` / `readBool`; `tools/check_strict_parsing.py`
fails the build if it is not.**

Status: ✅ present · 🔲 gap (must be closed before the change lands) · — n/a.
As of 2026-07-19 (second audit — the first missed the dataset characteristics,
ROC-reporting, and DFA-guesses rows below, all closed the same day) there are
**no legacy-coverage gaps**. New capabilities belong in the REST/GUI section and
have no CLI equivalent by design.

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
| 6 Print counter (log / linear) | Print-counter select + count | `POST /api/train` `logprint=`,`printcount=` | ✅ (presentation only — see note) |
| (train-time) Algorithm (GD/CGD/Shanno plus REST-era L-BFGS/iRPROP+/auto) | Algorithm select | `POST /api/train` `algorithm=` | ✅ |
| 7 Train model | Train button | `POST /api/train` | ✅ |
| 7/8 Save network + guesses after training | § Session files → Network / guesses | `GET /api/save/{network,train_guesses,test_guesses}` | ✅ |
| 9 Stepwise regression | § Stepwise regression panel (+ persistent results pane, live progress, Stop) | `POST /api/regress` (+ `async=1`, `GET /api/train/status` → `stepwise`, `POST /api/train/stop`) | ✅ (GUI beyond CLI: async + progress + Stop) |

**Note on the print counter (2026-07-26).** It is a *presentation* control in
both interfaces and nothing more: stopping conditions are evaluated every
iteration, independently of the printing schedule, so the same seed and
parameters give the same stopping iteration, weights and predictions under
logarithmic or linear printing. That had to be made true — until 2026-07-26 the
maximum absolute gradient was recalculated only inside the block that printed a
row, so this control silently chose the fit. `tests/gui/smoke.sh` asserts the
invariant on `/api/train`; `tests/iterative/check_gradcadence.cpp` pins the
engine mechanism. Anything added to either interface that reads a value only
when it is about to be displayed is the same bug.

## DFA submenu

| CLI menu option | GUI control | API | Status |
|---|---|---|---|
| 1 Linear DFA (incl. multi-output accuracy report) | DFA panel → Linear + Run | `POST /api/dfa` `type=linear` | ✅ |
| 2 Quadratic DFA (incl. multi-output) | DFA panel → Quadratic + Run | `POST /api/dfa` `type=quadratic` | ✅ |
| 3+4 Logging toggles | Model panel's log toggles (sent along) | `POST /api/dfa` `log_lastop=`,`log_history=` | ✅ |
| (after run) Save the DFA's guesses | DFA train/test guesses buttons | `GET /api/save/{dfa_train_guesses,dfa_test_guesses}` | ✅ |

## REST/GUI features added after the CLI freeze

The CLI menus are frozen permanently. These capabilities are REST-first and have
**no CLI equivalent by design**. All future interactive capabilities go here,
with a GUI control where a human-facing control is appropriate.

| Feature | GUI control | API | CLI |
|---|---|---|---|
| Automatic training-algorithm selection | Algorithm → "Auto" | `POST /api/train` `algorithm=auto` | — n/a (menus frozen) |
| Direct retained neural optimizers | Algorithm → "L-BFGS" or "iRPROP+"; either selection forces and locks Batch/epoch on and Automatic learning rate off | `POST /api/train` `algorithm=4` for L-BFGS (optional `lbfgs_memory=`) or `algorithm=5` for iRPROP+; both neural, batch-only, `autostep=0` | — n/a (menus retired) |
| Plateau auto-stop | "Auto-stop on plateau" + tol/window | `POST /api/train` `autostop=` | — n/a (menus frozen) |
| Realtime error-vs-iteration chart | Training-error chart | `GET /api/train/status` series | — n/a (menus frozen) |
| DFA graded ROC AUC | DFA ROC/stats panels | `POST /api/dfa` (ROC in the response) | — n/a (menus frozen) |
| OBD hidden-layer sizing (grow-then-prune, validation early stopping) — sizes the NODE COUNT of ONE hidden layer; the controls are "Starting hidden nodes" / "Maximum hidden nodes" and the panel says so, because "Start hidden 2" was read as two hidden *layers* (2026-07-29) | OBD panel + size-vs-error chart | `POST /api/obd` (async; `GET /api/train/status` `obd{phase,hidden}`; phase `"probing optimizers"` \| `"grow"` \| `"prune"` \| `"final"`) | — n/a (menus frozen) |
| Covariate stratification of the raw split + representativeness diagnostic (ROADMAP 4 Phase 2) | Dataset panel "Stratify on" columns + bins | `POST /api/load` `strata=` (1-based cols) `strata_bins=` | — n/a (the CLI already stratifies on the outcome; covariate strata are new, menus frozen) |
| Group-aware split — keep clusters intact for a harder unseen-group test (ROADMAP 4 Phase 3) | Dataset panel "Group on" columns | `POST /api/load` `group=` (1-based cols; rows with identical values stay together) | — n/a (new capability, menus frozen) |
| Three-way split — train/validation/test so selection (OBD) monitors validation and the test set stays untouched (ROADMAP 4 Phase 4c) | Dataset panel "Validation fraction" | `POST /api/load` `val_fraction=` (or `val_n=` with `test_n=`) | — n/a (new capability, menus frozen) |
| Cross-validation model comparison — logistic / LDFA / QDFA / neural (nested OBD) over ONE shared outcome-, covariate-, or group-stratified fold plan, three-tier report (ROADMAP 4 Phase 4) | Dataset-independent "Cross-validation" panel + pinned Tier-1 headline table | `POST /api/cv` (async; three-tier report in the result) | — n/a (new capability, menus frozen) |
| Nested-OBD optimizer rule — canonical / CGD / Shanno / Auto for the architecture search inside each CV fold, selected independently per fold and for the locked refit | CV panel "OBD optimizer" select (same four choices and wording as the standalone OBD panel) | `POST /api/cv` `algorithm=1\|2\|3\|auto` (default `auto`) | — n/a (new capability, menus frozen) |
| Clustered locked-test inference (Obuchowski 1997) — the cluster, not the row, is the independent unit; needs a group key + a locked test, preflighted for ≥ 2 clusters per outcome class, and never falls back to DeLong (ROADMAP 4) | CV panel "Sampling unit" → *clustered (Obuchowski)* | `POST /api/cv` `independence=cluster` (with `group=` and `locked_fraction=`/`locked_n=`) | — n/a (new capability, menus frozen) |
| Locked-test evaluation + opt-in inference — set aside an untouched row-wise or group-disjoint holdout; refit each procedure on development, score once for point AUCs; use DeLong for declared independent rows or Obuchowski for declared clusters | CV panel "Locked-test fraction" + primary/reference contrast selects + "Sampling unit" (not declared / independent rows / clustered) | `POST /api/cv` `locked_fraction=` (or `locked_n=`), `primary`/`reference`, `independence=rows|cluster`; clustered mode also requires `group=` | — n/a (new capability, menus frozen) |
| Cross-validation fold policy — covariate-stratified or group-aware (cluster-disjoint) fold plans, chosen per CV request and never inherited from the Dataset panel (ROADMAP 4) | CV panel "Stratify folds on columns" + "bins", "Group by columns (never split)" | `POST /api/cv` `strata=` (1-based cols) `strata_bins=`, `group=` (1-based cols) | — n/a (new capability, menus frozen) |
| Group-disjoint inner validation for nested OBD — under a grouped fold plan the architecture search's own train/validation split is group-disjoint as well; an infeasible one fails that fold with a reason and never falls back to row-wise (ROADMAP 4) | (no separate control — follows the CV panel's "Group by columns") | `POST /api/cv` `group=` (the same key governs outer folds, the locked holdout, and the inner split) | — n/a (new capability, menus frozen) |
| Structured cross-validation progress — stage, procedure n of m, outer fold n of k, completed folds/procedures, and the nested architecture trial inside the current fold (2026-07-29) | CV panel status line beside Run/Stop | `GET /api/train/status` → top-level `cv{stage,procedure,procedureIndex,procedureCount,completedProcedures,fold,folds,completedFolds,inner{phase,hidden}}` | — n/a (new capability, menus frozen) |
| Monitored held-out set, named by the engine — which set training and the OBD score actually watch (validation when loaded, else test) so a chart legend cannot contradict it (2026-07-29) | Live-error and OBD-search chart legends | `POST /api/load`, `POST /api/train`, `POST /api/obd` → `monitor:"validation"\|"test"\|"none"` | — n/a (new capability, menus frozen) |
| Result provenance — the ROC/Statistics panels name the operation that produced them, and a standalone analysis (DFA, CV) says it did not become the current model (2026-07-29) | Provenance banner above the ROC panel + Statistics heading + the CV report's closing scope note | (presentation; derived from which endpoint produced the panels) | — n/a (new capability, menus frozen) |

`POST /api/obd` params: `hidden_start`, `hidden_max`, `iter_budget`,
`sample_every`, `early_stop_tol`, `early_stop_patience`, `grow_patience`,
`prune_tol`, `autostop_tol`/`autostop_window` (the per-size train-plateau
backstop), `algorithm` (1|2|3|auto), `seed`. It is async-only and shares the
training job (status + stop reach it through the same doors); the winning sized
network replaces the current model.

`POST /api/train` reports `converged` and `ceilingExhausted` beside `ok`: a run
that hits the iteration ceiling is a successful *operation* with resumable weights
but an invalid *fit*, and the page renders it as a warning state rather than the
green "done". The same statement appears in the engine's own training report, so
CLI, logs and captured reports carry it too.

`POST /api/cv` params: `folds` (k, ≥2), `seed`, `maxiter` (per-fold cap for
logistic / fixed-architecture neural), procedure flags `logistic`/`ldfa`/`qdfa`/
`neural`, `neural_obd` (nested OBD per fold vs a fixed count), `neural_hidden`
(the fixed count), `hidden_max`/`iter_budget` (the per-fold OBD search),
`algorithm` (1|2|3|auto — the optimizer for the nested-OBD search; **default
auto**, same encoding and validation as `/api/obd`), `autostop_tol`/
`autostop_window` (the per-size train-plateau backstop), `inner_val` (share of
each fold's training rows held out as the inner validation set OBD monitors).
`auto` is a procedure for CHOOSING an optimizer, not an optimizer: it probes once
inside each fold on that fold's inner training data, keeps the choice for that
fold's whole grow-and-prune search, and probes independently again for the
locked-development refit — no fold reuses another fold's choice, the standalone
panel's, or a previous run's. Async-only, shares the training job (status + stop). It is a
standalone analysis — it does NOT touch the current model. The result carries the
report as text (`cv.tier1`/`cv.tier2`) and the paths of the Tier-3 files
(`cv_predictions.csv` / `cv_metrics.csv` / `cv_run.json`, written beside the data).
**Fold policy** is the CV request's own, parsed from `strata=` / `strata_bins=` /
`group=` on this call and **never** read from `/api/load`'s split configuration —
those size a train/test holdout and answer a different question, so inheriting them
would change what a cross-validation means depending on how the Dataset panel was
last driven. Absent both, the fold plan is the shipped outcome-stratified k-fold,
byte-identical. `strata=` gives outcome × covariate-stratified folds (a column with
few distinct values contributes one level per value; a continuous column is cut into
`strata_bins` quantile bins, computed on the *development* rows so the locked test
never influences a bin boundary). `group=` gives group-disjoint folds: rows sharing
identical values on all named columns are one indivisible group, never split across
folds — **and the same key governs the locked holdout**, so a cluster cannot straddle
development and locked either. The two **cannot be combined** yet (there is no tested
joint balancing objective) and the combination is refused rather than letting one
silently win. Stratifying balances the subgroups the sample already represents;
grouping measures transfer to groups the model never saw. Neither makes rows
independent — that is what `independence=` declares, and it is a separate axis.

Locked-test params on the same call: `locked_fraction` (0–1, `0` = none) OR
`locked_n` (a count — not both) set aside an untouched holdout: outcome-stratified
and row-wise by default, or group-disjoint when `group=` is present. Each procedure
is refit on the development rows by its own rule and
scored once for point AUCs. **Inference is opt-in:** DeLong CIs and the contrast *p*
are produced ONLY with `independence=rows` (declaring independent observations);
without it, predictions + point AUCs are returned and ordinary DeLong is withheld
(never a silently invalid *p*). `independence=cluster` routes to Obuchowski's clustered ROC covariance (needs `group=`
and a locked test; refused naming whichever is missing, and never downgraded to DeLong).
`primary`/`reference` (tokens `logistic`|`ldfa`|`qdfa`|`neural`) name the
prespecified contrast and **must be selected procedures** (else a validation error);
absent, it defaults to neural vs logistic when both are present. When inference
runs, Tier 1 gains an `AUC (test) [95% CI]` column plus a contrast verdict naming
the estimator that supplied its *p*; `cv.locked` carries the areas/CIs, the signed contrast, and
`samplingUnit`/`independenceStatus`/`inferenceMethod`/`inferenceRan`/`inferenceReason`/
`clusters`/`splitMethod`/`splitPlan` metadata, and `cv_locked_predictions.csv`
(row id, a `cluster` column for a grouped design, outcome, one column per procedure)
is written. `cv_predictions.csv` gains the same `cluster` column when the fold plan is
group-aware; ids are **global** across both files, so they join. The locked size is
validated for achieved per-class counts (≥ 2 of each on both sides) **using the
planner the run will actually use** — a grouped request previews a group holdout — and
a grouped run additionally needs ≥ k development *groups*, which no row count can
establish. **Ordinary DeLong assumes independent test rows** — not valid when rows share a sampling
unit. For that, declare `independence=cluster`, which runs **Obuchowski's clustered ROC
covariance** over the `group=` key (requires a locked test; the refusal names whichever
prerequisite is missing) and is never labelled "DeLong". Its locked sample is preflighted
for ≥ 2 clusters carrying each outcome class — a cluster condition, not a row one.
`independence=rows` with `group=` is refused outright rather than described away: a
group-disjoint design does not make rows independent. Note the row-based SE is **not**
reliably conservative or anti-conservative on clustered data — it can be too small or too
large depending on the within-cluster covariance structure (`docs/roc_theory.md`).

## Logging (cross-cutting)

Standing requirement (Craig, 2026-07-19): **every user action is logged.**

| Behavior | Mechanism | Status |
|---|---|---|
| Engine operations (dataset split, randomize, each training run incl. stopped) | `neuron.log` via `addHistory` | ✅ (pre-existing) |
| Per-action audit trail (load/model/dfa/train/obd/cv/regress/randomize/stop/save) with timestamp + values | `neuron_actions.log` via `logAction` | ✅ |
| Each training run's header records ALL params in effect (η, decay, stopping conditions, plateau) | run header (self-describing) | ✅ (already: `Network::runHeader` + `Iterative::train`) |
