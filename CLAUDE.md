Read and follow the instructions in /Users/craign/code/locker/CLAUDE.md before proceeding.

# neuron-3.0

Reanimation of **neUROn2++** — Craig's neural computational modeling environment — as a
modern project. This directory is the new home; the legacy code is read-only reference.

## Lineage

- **neUROn** (1992) — original, begun by Craig Niederberger
- **neUROn2++** (1996 → C++, 2000 → full OO redesign, 2007 → GNU autotools distribution)
- Final release: Sep 2016 (`../distro/neuron-2.64.tar.gz` — the only place the old
  version number matters; inside this repo the legacy build is just called **the oracle**).
- Substantial contributors over the years: Vinod Kutty, Yuan Qin, Joe Jovero, Young Hong,
  Kai Liu, Richard Schoor, Gaurav Bansal, Ashay Kparker, Hui Liu, Tony Makhlouf.
- **neuron-3.0** (2026) — this reanimation.

## Legacy codebase (`../distro/`)

Interactive menu-driven CLI (`neuron`): specify dataset → specify model → use model →
discriminant function analysis. C++ with GNU autotools, depends on **GSL** (tested against
GSL 1.9–1.10; modern GSL is 2.x — untested).

### Source map (`../distro/src/`)

| Files | What |
|---|---|
| `neuron.cpp` | Main driver: menus, program flow |
| `model.{h,cpp}`, `network.{h,cpp}`, `iterative.{h,cpp}` | Abstract model / network / iterative-training base classes |
| `simpleprop`, `bareprop`, `backprop` | Feed-forward neural nets trained by backpropagation (three implementations of increasing sophistication) |
| `logistic.{h,cpp}` | Binary logistic regression — weight decay, gradient methods, conjugate gradient descent, Shanno's algorithm, Wald tests, condition number |
| `ldfa`, `qdfa`, `dfa` | Linear & quadratic discriminant function analysis |
| `regressnet.{h,cpp}` | Stepwise regression over network inputs |
| `dataset.{h,cpp}`, `twoset.{h,cpp}` | Dataset loading, train/test randomization, ROC reporting, classification tables |
| `stats.{h,cpp}` | Statistics: Kolmogorov–Smirnov, Pearson chi-square, Hosmer–Lemeshow, error functions |
| `matrix.{h,cpp}`, `vector_ops.{h,cpp}` | Templated matrix/vector math (matrix.h is 67 KB of templates) |
| `utility.{h,cpp}`, `function_defs.h` | Console I/O helpers, activation function definitions |

### Other legacy assets

- `../distro/doc/manifest.pdf` — the full manual (manifest.tex source). `spin.html` is a tutorial.
  **Copied into `docs/` here** (see below).
- `../distro/scripts/` — model exporters: `neuron2html.pl` (model → standalone HTML calculator),
  `neuron2palm.pl`, `neuron2iphone.rb`, plus `mkdataset.pl`. Documented in manifest.pdf ch. 11.
  The HTML exporter idea is reborn as the planned `tools/neuron2web.py` (see roadmap);
  its inputs are the same trio (weights file, scaling-factors file, label spec), and
  `perl-html_creator_spec.txt` in that directory defines the label-spec tag format we
  carry forward. Palm/iPhone exporters stay dead.
- `../distro/data/` — sample datasets: PSA (`psa_defs.txt`, `ordata`), low birthweight
  (`lowbwt2-2*`), XOR, BP40 train/test.
- `../distro/*.tar.gz` — release tarballs; the newest one feeds `tests/oracle/build_oracle.sh`.

### The oracle (verified 2026-07-11)

- **The legacy source builds clean on Apple Silicon** — zero changes, modern clang++
  (C++14 default) + Homebrew GSL 2.8. Only warnings: `std::unary_function` deprecation
  in `function_defs.h` (would break under `-std=c++17`; fixed in our `src/`).
- Oracle build + scripted XOR smoke test live in `tests/oracle/` (see its README).
  XOR trains to 100% CA with full stats output; weight randomization is unseeded, so
  runs are nondeterministic — compare endpoint statistics, not traces
  (`verify_oracle.sh` avoids this by forward-passing identical saved weights).
- The old `../distro/src/neuron` binary is x86_64; ignore it, rebuild via
  `tests/oracle/build_oracle.sh` (→ `tests/oracle/bin/oracle`).
- Program quirks: the oracle's welcome banner shows a stale older version string
  (pre-generated configure predates the final bump — cosmetic); the program writes
  `model.txt` + `neuron.log` logs into its cwd.

## neuron-3.0 direction — DECIDED (2026-07-11)

1. **Language/stack: hybrid.** C++ for the engine — speed matters. Python for data
   grooming and everything around the engine.
2. **Interface: KISS.** Keep it simple and portable — porting to other systems is a
   priority, so no web UI, no heavy frameworks. Simple CLI.
3. **Scope: full engine port.** The statistical parts (goodness-of-fit, ROC, Wald,
   condition numbers, DFA, logistic regression, stepwise) are the novel and useful pieces —
   they all come along, together with the neural nets.
4. **Exporters: dropped** *(superseded 2026-07-13)*. Originally: not carried forward.
   Craig revived the HTML-calculator idea as a first-class deployment path — see the
   roadmap's **deployment phase** (`tools/neuron2web.py`). Palm/iPhone stay dead.

## Docs (`docs/`)

Legacy documentation copied from `../distro/doc/` (2026-07-11):

- `docs/manifest.pdf` — the full neUROn2++ manual (ch. 11 covers the dropped exporters)
- `docs/spin.html` — tutorial (self-contained, no external assets)
- `docs/maths.pdf`, `bareprop.pdf`, `network.pdf`, `driver.pdf` — algorithm/design docs
- `docs/tex/` — LaTeX sources (grep-able; `manifest.tex` is the searchable form of the
  manual) + `figures/` with small PDF versions of the manifest figures. The original
  `manifest.tex` references multi-MB `.bmp` scans that were deliberately NOT copied;
  a rebuild would need `\includegraphics` switched to the PDF figures.

## Standing rules

1. **Whenever tools, menus, or recipes change, update `AGENTS.md` in the same commit** —
   its recipes are promised to work.

2. **A test must be proven to fail.** Before you believe a test guards anything, run it
   against the code it is supposed to catch — `git stash push -- src/`, rebuild, confirm
   it fails, `git stash pop`. A test that has never failed is a hypothesis, not a guard.
   *Why this is a rule and not advice:* on 2026-07-15 the entire ROC confidence interval
   was replaced — delta method out, bootstrap in — with **all five invariants green**,
   because none of them executed the code. `xor_seed42` and `regress_seed42` have four
   exemplars and print "Cannot calculate ROC statistically"; the oracle likewise. Zero
   "By statistical method" lines between them. The same day, `smoke.sh` was found to be
   *asserting a fabricated Az of 0* — the test was pinning the bug in place. Both were
   caught only by running the new tests against the old binary, and each caught something
   real. `tests/golden/run_golden.sh` now asserts its own coverage (see the list at its
   foot) so that this particular hole cannot silently reopen; nothing equivalent guards
   the other suites, so the practice is the guard.

3. **A doc that names a mechanism is a hypothesis, not a finding — measure before acting
   on it, including docs written here.** Also 2026-07-15: this file's own ROADMAP said
   "make the χ² p-value non-fatal, do this first", from a causal story ("zero-SD bins →
   degenerate fitexy weights → non-finite χ²") that was wrong in **every link** — there
   were zero zero-SD bins, `chixy` guards zero weights, and the NaN came from
   `Population::var()`. Doing what the plan said would have shipped silent NaN areas;
   `gammq` was not the disease, it was the immune system. A 20-line probe settled in
   minutes what a session of reasoning had inverted. Three further plan steps
   ("goldens re-bless", "the oracle needs an exclusion", "Metz corners are needed") were
   likewise hollow, all written from *"the invariants cover this"*.

4. **Engine code lives in the class layer.** The computational core was built on
   `Matrix` / `vector_ops` / `Population` so that (a) the code reads like the matrix
   notation in the paper it came from — you can go from the page to the code and back —
   and (b) every element access is bounds-checked (`Matrix::operator()` throws
   `BoundsViolation` even in release builds, where asserts vanish). New engine or
   statistics code uses those classes; when the layer lacks a primitive, **extend the
   layer** (as `includerows` was added 2026-07-16 for the bootstrap resample) rather
   than dropping to raw arrays or hand-rolled recounts. Scalar code is sometimes
   genuinely the right tool (order statistics, Numerical Recipes routines) — say so in
   a comment where you do it. *Why this is a rule:* legacy bug #8 was `HLX2calc`
   copying data OUT of the bounds-checked Matrix into raw `double[10000]` C arrays —
   the layer would have thrown; the raw arrays scribbled the stack for twenty years.
   The same day, the ROC threshold sweep's hand-rolled per-threshold recount (O(n²),
   multiplied by 2000 bootstrap resamples) turned out to be the entire bootstrap scale
   cliff; reformulated as one sort plus cumulation — which is also how Wickens' own
   Table 5.2 is built — the 12,000-row report went from minutes to seconds,
   byte-identical. Leaving the layer cost correctness once and performance twice.

5. **GUI/CLI parity is a hard contract: every capability in the CLI menu interface MUST
   have a GUI equivalent (a page control AND an HTTP API parameter).** The GUI is the
   primary human interface; the CLI menus are frozen but remain the authoritative feature
   list. A CLI menu option with no GUI equivalent is a **bug**, not a backlog item. Any
   change that touches the menus or the GUI updates **`docs/gui_cli_parity.md`** — the
   matrix of every menu option ↔ GUI control ↔ API param — in the same commit; that
   matrix is the enforcement artifact (full automation of the mapping is brittle, so the
   checklist is the guard). *Why this is a rule:* the parity was stated aloud when the GUI
   began and then went unenforced — by 2026-07-19 the GUI exposed roughly a *third* of the
   CLI surface (no learning rate, weight decay, most stopping conditions, batch/epoch,
   print counter, output error function, load-network-from-file, or DFA), and Craig, who
   had deliberately declined to also add new GUI features to the CLI ("the GUI is the
   primary interface from now on"), had no way to reach half the engine. A rule left as
   advice decays (cf. rule 2); this one already did.

6. **Engine refactors conform to the layer-ownership constitution in
   `docs/cv_refactoring_architecture.md`** (adopted 2026-07-22; written for the Phase 4 CV
   refactor but permanent and repo-wide). The manifest (`docs/manifest.pdf`) is the
   architectural constitution; the ownership doc is its operating interpretation of DRY.
   The core: **one authoritative implementation of each mechanism, in the class that
   conceptually owns it** — not the fewest lines, and never a generic god object that
   collapses distinct responsibilities. Dependencies point downward (GUI/API → CV/OBD
   orchestration → Model/DataSet interfaces → Iterative/Network/concrete models → Matrix);
   a lower layer never learns about a higher concern (`Matrix` gathers rows but must not
   know what a fold is; `DataSet` materializes a fold but must not decide which model to
   train; a model fits itself but must not run the CV loop). Extract duplicated code to the
   **lowest class-layer boundary that naturally owns it** (the GUI may call it; CV/OBD may
   call it; none may copy it), preserve the manifest's construction order (`setDataSet`
   before architecture-dependent sizing), and distinguish genuine duplication from
   intentional polymorphism (per-class model math is NOT a consolidation target). **Refactor
   before adding behavior:** a behavior-preserving extraction, *proven* to leave seeded
   outputs / reports / model files / tests unchanged, comes first; new behavior goes through
   the extracted interfaces after. The Phase 4 ownership rule — *CV owns repetition not
   training; OBD owns selection not evaluation; models own fitting not fold management;
   DataSet owns fold materialization not modeling policy* — is the governing interpretation.

## Housekeeping

- **GitHub:** https://github.com/craignied/neuron (HTTPS remote per the locker's
  Mac Studio SSH-push note). Created 2026-07-11; push freely on session handoff.
- **Memory:** enrolled in `~/code/claude-memory/` as `neuron-3.0` (2026-07-11).

## Where things stand (2026-07-24)

Present tense, rewritten each session — **not** a log. The dated record is
`docs/HISTORY.md`.

**The engine.** The legacy neUROn2++ C++ source carried forward and modernized in place
(C++17, zero-warning Release build, `unique_ptr` ownership, seedable `std::mt19937` behind
`util::d_random()`). Models: SimpleProp / BareProp / BackProp feed-forward nets, binary
logistic, LDFA / QDFA, RegressNet stepwise. Statistics: exact non-parametric trapezoidal
AUC and binormal Az (least-squares z-ROC fit via fitexy over distinct operating points),
stratified bootstrap CI plus Hanley-McNeil, Kolmogorov-Smirnov, Pearson X² (statistic
only — see settled decisions), Hosmer-Lemeshow Ĉ on 10 fixed deciles, Wald tests,
condition number. **Ten legacy bugs** were found and fixed during the reanimation; each
is written up in HISTORY with the measurement that proved it.

**Interfaces.** The CLI menus are **frozen** but fully working — they remain the
authoritative feature list rule 5 measures the GUI against. `neuron --gui` (embedded
cpp-httplib server, 127.0.0.1, OS-assigned port) is the primary human interface; the HTTP
API is the permanent headless/LLM path. `tools/` holds stdlib-only Python (`mkdataset.py`
grooming, `neuron2web.py` deployment to a self-contained HTML calculator).

**Training automation.** `algorithm=auto` (wall-clock-budgeted probes of GD/CGD/Shanno,
winner adopted), plateau auto-stop, and OBD hidden-layer sizing (grow-then-prune, each
size early-stopped at its held-out minimum — no size trains to completion). **Reaching
the iteration ceiling is a failure to converge, never a stopping condition:** a trial that
ends at `max_iterations` is ineligible, so OBD refuses loudly instead of comparing
unfinished fits, and ordinary training warns. Nested OBD inside CV takes its optimizer
from `/api/cv algorithm=` (default auto, probed independently per fold). **Stopping
conditions are evaluated every iteration, independently of reporting cadence** — the
print counter is presentation only (legacy bug #10, 2026-07-26).

**Splitting and evaluation.** `src/split.{h,cpp}` owns the index-level cube: outcome-
stratified holdout, outcome × covariate strata (quantile-binned, Hamilton apportionment),
group-aware (indivisible clusters), stratified k-fold, and three-way train/validation/test.
`src/crossval.{h,cpp}` runs procedures over one shared fold plan (logistic / LDFA / QDFA /
neural with nested OBD inside every fold); `src/cvreport.{h,cpp}` renders the three-tier
report (Tier 1 headline table, Tier 2 detail, Tier 3 machine-readable files). The
locked-test layer (`src/delong.{h,cpp}`) scores procedures once on rows held entirely out
of CV and compares a prespecified contrast — **inference is opt-in** (see settled
decisions).

**The gates, run at the end of every piece of work** (all currently green): zero-warning
Release build → `tests/golden/run_golden.sh` byte-identical (3 transcripts: `xor_seed42`,
`regress_seed42`, `binormal_seed42`) → `ctest` (11 tests) → `tests/gui/smoke.sh` →
`tests/oracle/verify_oracle.sh` numerically identical → live `neuron --gui` click-through
for anything that adds a control → the SEER acceptance run for splitter work. CI runs the
build, goldens, ctest, smoke, and the Python tools on macOS/Linux/Windows.

**Open work** is the three remaining ROADMAP 4 items under "What remains" below. There is
no known defect outstanding.

## Settled decisions — do not reopen

Each of these was decided against a measured alternative and cost real time to establish.
Re-proposing one is rework, not initiative. Full reasoning at the cited HISTORY entry.

- **Binormal CI: do NOT "recover the a–b covariance."** The delta method assumed
  independent z-ROC points; they are cumulated from one sample and are not (Wickens
  pp. 87-88), so no cross term rescues it. The delta method is gone; the bootstrap
  replaced it. → HISTORY 2026-07-15 (later); `docs/roc_theory.md`.
- **`Matrix` value-initialization was a MISDIAGNOSIS and was reverted.** The nested-OBD
  flake was an uninitialized `Model::errorType` scalar. A fix that only *reduces* a
  heap-layout-sensitive flake is a suspect, not a cure. → HISTORY 2026-07-23;
  `docs/nested_obd_nondeterminism_resolution_report.md`.
- **Do not replace the RNG's raw-output-to-double mapping with
  `std::uniform_real_distribution`** — its implementation varies between stdlibs and would
  break cross-platform seeded reproducibility. → HISTORY 2026-07-12 (later).
- **Any new resampling uses `util::i_resample()`, never `i_random`/`d_random`** — so that
  computing an interval can never perturb weight init or splits. → HISTORY 2026-07-15.
- **`i_random`'s modulo bias is deliberately NOT fixed** (a fraction of a part in 2³²);
  rewriting it re-blesses every seeded stream in the repo. → HISTORY 2026-07-22.
- **The CLI menus are frozen.** New capability goes to the GUI as a documented
  GUI-beyond-CLI feature (the OBD precedent). This does not weaken rule 5: everything the
  CLI *has* must still be reachable in the GUI. → `docs/gui_cli_parity.md`.
- **Hosmer-Lemeshow: g = 10 deciles, fixed.** Scanning group counts for a favorable p was
  part of legacy bug #9, not a feature. → HISTORY 2026-07-16 (evening).
- **Pearson X² prints as a statistic with n and NO p-value.** Individual-level X² has no
  valid χ² reference with continuous covariates — that is the problem H-L exists to solve.
  → HISTORY 2026-07-16 (evening).
- **Metz/LABROC corner categorisation is not needed** for a least-squares fit over distinct
  operating points; it returns only if Dorfman-Alf ML is ever built. → HISTORY 2026-07-15.
- **Cross-validation carries no formal inference.** Fold mean ± sd is descriptive spread
  across dependent folds, never a CI; classical DeLong is invalid on pooled out-of-fold
  predictions. → HISTORY 2026-07-22; `docs/cross_validation.md`.
- **Locked-test inference is opt-in and independent-rows-only.** DeLong runs only when the
  caller declares `independence=rows`; `independence=cluster` is refused and must never
  silently fall back to ordinary DeLong. Craig's words: *"Metadata cannot repair an invalid
  p-value after it has been presented."* → HISTORY 2026-07-24.
- **No `TrainingConfig` extraction.** `Network::copy`/`Iterative::copy` already carry the
  training config (autoalgo depends on it), so CV clones a configured model per fold; a
  second config-application path would be against DRY. → HISTORY 2026-07-22.
- **The oracle keeps its legacy bugs on purpose.** `verify_oracle.sh` excludes the known
  divergent lines (K-S, Pearson, H-L, 95% CI, "Number thresholds") and asserts 3.0's
  known-correct values instead. Do not "fix" the oracle. → `tests/oracle/README`.
- **Do NOT loosen `autostop_tol` to make an OBD run "finish".** Measured on Civic Choice:
  at the 1e-4 default a sufficient ceiling gives 5 hidden units and test AUC 0.817, while
  at 0.01 the same search stops far too early and gives 1 unit at AUC 0.53. A refusal is
  more informative than a model produced by a tolerance chosen to silence it. Raise the
  ceiling or use `algorithm=auto`. → HISTORY 2026-07-25.
- **Reporting cadence may change output VOLUME; it may never change optimization or fit
  validity.** The print counter is presentation only. Do not "solve" a stopping problem by
  choosing a denser print schedule, and never compute a quantity a stopping rule depends
  on inside a block that runs only when something is displayed — that was legacy bug #10,
  where logarithmic vs linear printing chose the fit (canonical stopped at 304 / 400 /
  1000 on one fixture) and a ceiling between print points reported a false failure to
  converge. → HISTORY 2026-07-26; `tests/iterative/check_gradcadence.cpp`.
- **Palm/iPhone exporters stay dead**; the HTML calculator lives on as
  `tools/neuron2web.py`. Python tooling is **stdlib-only** — no pip, no venv, enforced by
  CI on all three OSes.

## ROADMAP 4 (agreed with Craig 2026-07-22) — a general representative test-set splitter

### Why (rationale)

**The problem.** `DataSet::randomize` (`dataset.cpp:690`) is the engine's only train/test
splitter. It does one thing — an **outcome-stratified single holdout** — with two
independent O(n²) hot spots:
1. `nvec::random_positions` (`vector_ops.cpp:32`) is a *rejection* shuffle that rescans
   from index 0 on every collision; on a class of m rows it is O(m²), dominated by the
   tail (the last elements collide against a nearly-full set).
2. The zeros/ones partition and the train/test sets are built by repeated
   `Matrix::addrow` (`dataset.cpp:732–734`, `760–772`), each append reallocating and
   copying a growing matrix — O(n²) total (~10^10 element copies at SEER scale).

Neither bit at the repo's historical sizes (hundreds to a few thousand rows). **The next
dataset is 226,679 rows** (SEER prostate-cancer 5-year mortality), where both are
catastrophic. The splitter must be rebuilt.

**Why *general*, not a SEER fix.** Craig's call (2026-07-22): the rebuild targets a
splitter general over prevalence, feature types, and data structure, **with SEER as the
acceptance test.** SEER earns that role by stressing every axis at once:
- **Rare events** — 2.96% prevalence (6,705 / 226,679). Accuracy is useless (always-negative
  = 97%); the split must protect the positives.
- **Clumped positives** — events are not spread across covariate space: **M1 (metastatic)
  disease is 2.2% of the cohort but 40% of all deaths** (54% event rate); Gleason 8–10 is
  14.5% of the cohort but 68% of deaths. Outcome-stratification alone does not *guarantee*
  a rare decisive subgroup is proportionally represented.
- **Clustering** — the four socioeconomic inputs are *area-level* (shared within a county):
  ~612 distinct areas, mean 370 patients, one with 20,364. Patients within an area are not
  independent, which raises whether the same area may appear in both train and test.
- **Scale** — 226k rows (the O(n²) killer above).
- **Mixed types** — 7 continuous inputs + 15 binary indicators; a stratum built from a
  continuous column must be quantile-binned.

**If the general splitter handles SEER, it handles almost anything** — that is the design
discipline; SEER is the standing acceptance test at every phase.

**What "representative" means, precisely.** A test set is representative when the held-out
estimate is a low-bias, low-variance estimate of population performance. The binding
constraint here is NOT row count — with 6,705 positives even a 10% holdout gives a
Hanley-McNeil AUC SE ≈ 0.013, and 25% gives ≈ 0.006–0.009, so the headline AUC is precise
at any sensible fraction. The real questions are (a) do rare decisive subgroups land
proportionally, (b) can a single draw be trusted, (c) which population — new patients from
*known* areas (standard split) or *unseen* areas (grouped split). Those three map exactly
onto the three design axes below.

**The design — a small principled family, not one method** (sklearn's `model_selection`
is the reference taxonomy; we own the useful subset in the class layer, zero
dependencies). The mechanism is **parameterized; the policy is the user's** — we do NOT
bake in "stratify on M-stage"; M-stage is one instance of "stratify on outcome × a named
covariate," and area is one instance of "a group key." Three axes:
- **Stratify axis:** none → outcome → outcome × named strata (continuous columns quantile-binned).
- **Group axis:** none → group-aware (a cluster key that may not straddle the split).
- **Estimator axis:** single holdout → three-way (train/val/test) → k-fold → repeated k-fold.

**The two places generality is actually won or lost** (both forced by SEER):
1. **Stratify × group do not compose exactly.** Once a whole county lands in test its
   outcome mix is fixed — you cannot both keep groups intact and perfectly balance
   outcome. This is **stratified-group k-fold**, a greedy bin-packing approximation
   (assign each group to the fold currently most under-quota for its outcome mass;
   Sechidis, Tsoumakas & Vlahavas 2011). It is the one approximate part, and SEER (rare
   events *and* clustering) is exactly the case that needs it.
2. **Degeneracy is first-class.** A general tool routinely meets a class smaller than k,
   an empty outcome×stratum cell, a group larger than a fold, a requested test count
   above a class size. SEER produces these the moment you cross outcome × M1 × a rare
   race category. One documented ladder — refuse / warn-and-collapse-the-stratum / clamp —
   applied uniformly, replacing the scattered ad-hoc refusals in today's `randomize`.

**The common foundation** (identical for every cell of the cube, and the fix for the
O(n²) code): per-stratum **index vectors → partial Fisher-Yates on indices via
`util::i_random`** (the same `rng` stream splits already ride — NOT the reserved
`i_resample` bootstrap stream; uniform up to `i_random`'s negligible modulo bias;
O(m)) **→ one `Matrix::includerows` gather**
(already in the class layer, rule 4; O(n), one allocation per output set). k-fold folds
fall out of the same shuffled indices for free.

**One deliverable elevated to first-class: a split-diagnostic report** — per output set /
fold: n, outcome counts + rate; per named stratum: counts; group-leakage count (must be 0
when grouping); continuous-covariate means train-vs-test. Printed via `util::screen()`
(capturable → GUI). This is how representativeness becomes *verifiable* on any dataset
rather than trusted — on SEER it is how you confirm the M1 positives actually landed
proportionally.

**Scope boundary.** neuron is a discrete-outcome engine, so "general" means general over
datasets with a discrete (or quantile-binnable) outcome and arbitrary inputs — not over
arbitrary ML tasks. SEER sits comfortably inside that scope. Bonus: the continuous-column
binning that strata need also yields continuous-*outcome* stratification for the
regression path, a corner today's splitter refuses outright.

### What remains

Phases 1–3 (the efficient index-gather rewrite, generalized stratification + the
diagnostic, group-aware splitting) and Phase 4 (cross-validation, the three-tier report,
`/api/cv`, and the locked-test DeLong layer) are **DONE and shipped**. The phase-by-phase
plan as agreed and executed is in `docs/HISTORY.md` → "Completed roadmaps → ROADMAP 4";
read it before extending any of that machinery, because it records which parts of the
design were corrected mid-build and why.

Three items remain, in Craig's priority order:

1. **Group-aware / covariate-stratified CV folds.** Fold plans are outcome-stratified
   k-fold today; the splitter already owns group and covariate-strata modes, so this is
   composing existing mechanisms in `crossval::run`'s plan construction, plus the GUI
   params and the diagnostic. `nsplit::kFold` is the primitive to extend.
2. **County-cluster-aware (non-IID) locked-test inference.** The successor to ordinary
   DeLong for clustered data (Obuchowski 1997). Group-aware splitting stops leakage but
   does not make rows independent, which is exactly why `independence=cluster` is
   currently *refused* rather than approximated. Plan: `group-aware_plan.md` (Sol).
   Prerequisite: DLG-8, structured cluster/sampling-unit metadata in the Tier-3 artifacts.
3. **B9 — the GUI-wide strict-parsing pass.** Shared strict integer / floating-point /
   boolean parsers with full-string consumption, range and overflow checks, and
   field-specific errors, migrated across **every** handler (`atol`/`atof` today accept
   `folds=5junk` → 5, and any non-`"1"`/`"true"` boolean → false). Do NOT broaden accepted
   boolean spellings unless that becomes explicit API policy.

### Verification (end of every phase)
Zero-warning Release build → `tests/golden/run_golden.sh` (byte-identical; the Phase-1
`binormal_seed42` re-bless is done and no further move is expected — read any diff) →
`tests/gui/smoke.sh` (extended per phase) → `ctest` (new `split_*` cases) →
`tests/oracle/verify_oracle.sh` (numerically identical — the splitter path is not on the
oracle) → **the SEER acceptance run** (timed + diagnostic inspected) → live `neuron --gui`
click-through for any phase that adds a control. AGENTS.md, `docs/gui_cli_parity.md`, this
file, and the session entry in `docs/HISTORY.md` all land in the same commits.

## Backlog (unordered, nothing forcing them)

- **Dorfman-Alf maximum-likelihood binormal fit.** A real improvement with no live defect
  forcing it — which is the definition of backlog. The surviving reasons: degeneracy /
  the "proper" binormal (Metz & Pan 1999 — our least-squares curve is improper for b ≠ 1
  and has no guard, though the crossing sits beyond F ≈ 0.9999 on well-behaved data),
  efficiency, a fit χ² that would actually mean something (ours is uninterpretable on
  continuous data: χ² ≪ df, p → 1.000), and the medical-publication standard. Full
  rationale: `docs/HISTORY.md` → ROADMAP 3 → Phase 3; Methods language ready in
  `docs/roc_theory.md`.
- **New training algorithms from the literature** — Rprop (ideal full-batch), Adam,
  L-BFGS (`gsl_multimin_fdfminimizer_vector_bfgs2`), optional Levenberg-Marquardt; Muon as
  an experimental candidate. Slot in as trainingType 3+ in `Network::engine` dispatch;
  `autoalgo::pick` iterates the extended list automatically; batch-only methods force
  `batchEpochFlag` like the CGD/Shanno probes. → HISTORY, ROADMAP 2 Phase 5.
- **Automated neural-network restarts for initialization-sensitive local minima.**
  Craig's pre-3.0 manual practice was to rerandomize a network with two or more hidden
  nodes when a long run appeared trapped in a poor local error minimum. Automatic
  stopping answers only whether ONE run finished; a small gradient can certify a
  converged but suboptimal basin. A future multi-start procedure could train deterministic
  random restarts, reject every unconverged start, select only by validation error (never
  test error), and report stability across starts. This is distinct from `algorithm=auto`,
  which compares optimizers from a common starting point. **Do not choose restart counts,
  improvement tolerances, or stopping policy yet:** we have not measured how often
  materially different basins occur, and nested OBD/CV would multiply the cost by folds ×
  architectures × restarts. First measure incidence and cost on representative small and
  large datasets; then consider an adaptive minimum/maximum restart policy with
  deterministic RNG substreams keyed by procedure/fold/architecture/restart.
- **OBD metric: stop on loss, select on AUC.** OBD early-stops and scores on held-out
  *error* today. At low prevalence both accuracy and loss are majority-dominated while AUC
  is prevalence-robust, so keep loss as the cheap smooth trigger but score each size by
  held-out AUC at its stopping point. Caveat: AUC ignores calibration — surface loss, AUC,
  and a calibration number together. A product decision, not a defect.
- **Tier-2 calibration numbers, per-fold timing, and Tier-3 download buttons** in the CV
  report (files are written to disk and their paths reported; only the buttons are
  missing). → `docs/evaluation_report_spec.md`.
- `getGoodData()` port from the legacy roc app, if something ever needs it.
- More grooming tools (dataset describe/summary; train/test split outside the engine).

## History

The full dated development record — every session entry from 2026-07-11 onward, the nine
legacy bugs with the measurements that proved them, the ROC/Wickens reconciliation, the
Hosmer-Lemeshow and Pearson forensics, the nested-OBD nondeterminism hunt, and ROADMAPs
1–3 as agreed and executed — is **`docs/HISTORY.md`**.

It is deliberately **not** imported (no `@docs/HISTORY.md`): at ~160 KB it would be pasted
into every prompt, which is what this split exists to prevent. Read or grep it on demand.

**Where new work gets written down.** Append the session entry to `docs/HISTORY.md`, in
the house style — findings first, measurements over stories. Then *rewrite* "Where things
stand" above in place, add any new prohibition to "Settled decisions", and tick the
backlog. Do not grow this file with narrative; that is how it reached 179 KB.
