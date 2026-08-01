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

The user-facing entry point is **`docs/datasets/civic-choice/WALKTHROUGH.md`** — the
illustrated GUI walkthrough (published 2026-07-29), with the fictional synthetic dataset,
its generator, and its data notes beside it in the same directory.

Legacy documentation copied from `../distro/doc/` (2026-07-11):

- `docs/manifest.pdf` — the design manifest, updated for neuron 3.0 architecture,
  build, CLI, Python tools, and loopback REST API. Obsolete MSVC project instructions,
  Perl/Palm/iPhone exporters, and the pre-menu file/header configuration are omitted
  from the published PDF; the retained mathematical and class chapters are explicitly
  labeled as the foundational design record.
- `docs/spin.html` — tutorial (self-contained, no external assets)
- `docs/maths.pdf`, `bareprop.pdf`, `network.pdf`, `driver.pdf` — algorithm/design docs
- `docs/tex/` — buildable LaTeX sources (grep-able; `manifest.tex` is the root) +
  `figures/` with PDF versions of the manifest figures. Build the manual from that
  directory with `latexmk -pdf manifest.tex`.
- `docs/manifest_maintenance.md` — source organization, authority map, writing rules,
  build/visual-QA workflow, and checklist for every future manifest addition.

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

   **Artifact comparisons require existence evidence.** Before comparing two generated
   files, reports, arrays, or parsed results, assert *independently* that each production
   operation succeeded and that each artifact exists and is non-empty where non-empty
   output is required. **Equality between two absent, empty, default, null, or failed
   results is vacuous and guards nothing** — it is the same hole as a test that executes a
   mechanism and asserts something the mechanism cannot affect. This applies to files, to
   empty JSON results, to empty prediction arrays, and to default numerical values alike.
   Test paths must also be portable and isolated: use the platform temporary-directory API
   rather than a hardcoded `/tmp`, and avoid shared fixed filenames that collide between
   concurrent runs. *Why this is a rule:* on 2026-08-01 `check_onehidden` hardcoded `/tmp`,
   which is not a temporary directory on Windows, so every save silently failed there — and
   **three of its four save-comparison assertions passed on two empty strings**. Only the
   one assertion that inspected *content* failed. The portability slip was a repeat; the
   vacuous comparison is the defect worth the rule.

   **Memory-safety instrumentation is a targeted diagnostic, not a full-suite gate.**
   ASan is currently unusable in this macOS agent environment: on 2026-08-01 three
   separately built ASan executables, including a trivial `printf( "hello" )` control,
   hung during startup even with leak detection disabled. This proves an execution-
   environment limitation, not a neuron defect or necessarily a machine-wide defect.
   **Do not retry ASan, increase its timeout, or launch broader ASan builds here unless
   the agent environment changes and a trivial control first completes immediately.**
   Terminate a timed-out diagnostic and verify that it did not remain in the background.

   For vector/container bounds questions, a tiny libc++ hardened-mode harness is the
   working local substitute (it ran immediately and independently caught D5's invalid
   dereference as `SIGTRAP`). Run only the smallest relevant library and deterministic
   cases, isolate expected-abort cases so one cannot hide the rest, and first prove a
   valid control succeeds. Do not run all of `ctest`, GUI smoke, goldens, bootstrap-heavy
   reports, or the oracle under ASan, UBSan, or hardened mode unless the task specifically
   requires it. Permanent coverage belongs in ordinary Release/`NDEBUG` tests of the
   public contract; instrumentation is supporting sabotage evidence. Verify that the
   instrumented or hardened source was actually rebuilt before trusting any result, and
   remove temporary harnesses and binaries afterward.

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
   and (b) **both** classes reject bounds and dimension violations in Release, where
   asserts vanish: `Matrix::operator()` throws `BoundsViolation`, and every
   `vector_ops` operation that walks two containers in lockstep or indexes one by
   position throws `nvec::SizeMismatch` / `RangeViolation` / `EmptyVector`. This
   is why staying inside the layer is worth it. It was half true until 2026-08-01:
   `vector_ops` guarded its preconditions with `assert` alone, and because this
   project builds Release by default, **not one of those assertions had ever
   executed in the gate chain** — seventeen contracts, measured, unenforced in the
   shipped binary (D5; `refactor_audit.md` §11). New engine or
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

7. **Speed is an architectural requirement, and the mathematics must stay visible. Neither
   may be traded for the other, and neither may be traded for fewer lines.** The engine is
   compiled C++ because neural computation is compute-bound; the class layer exists so that
   code can be read against the equations it came from. Rule 4 states the readability half.
   This is the other half, and the idioms that implement it — which lived only in scattered
   implementation comments until 2026-07-31:

   - **`Matrix`, `vector_ops` and `Population` are the numerical vocabulary.** Extend them;
     do not work around them (rule 4).
   - **Prefer the destination-taking overload when a buffer can be reused.** Every
     `Matrix`/`vector_ops` operation comes in two forms — one that fills an object you pass,
     one that returns a new one. The pair is deliberate and is NOT duplication: the returning
     form is convenience, the destination form is what belongs in a loop.
   - **Prefer the compound operators (`*=`, `+=`, `-=`) where they avoid a temporary
     container** and do not obscure the equation.
   - **Use the range overloads** (`func( v, f, out, a, b )`, `dotprod( v, out, a, b )`) where
     they express the mathematical domain directly — e.g. "all hidden units except the bias
     slot" — rather than computing the whole vector and fixing it afterward.
   - **Do not introduce `std::function`, virtual dispatch, allocation or copying into a hot
     loop to remove source duplication.** A hot loop is anything inside `trainSet` /
     `innerTrainSet` / `forward` / a per-exemplar scoring pass. In cold paths (reporting,
     file writing, one-shot setup) those tools are fine.
   - **Pass read-only containers by `const&`.** By-value parameters on a per-exemplar path
     are heap traffic per exemplar per iteration.
   - **Measure before you change an algorithm for speed.** Rule 3 applies to performance
     claims, including ones written here. `./build/scale_probe` and a timed run are the
     evidence; a reading of the source is a hypothesis.
   - **An abstraction that hides the equation has failed even if it removes lines.** The
     biased and unbiased forms of a gradient are two published formulae; keeping them
     separately legible is worth the duplication that a bias flag would remove. DRY means one
     authoritative mechanism at the correct layer (rule 6), never minimum line count.

   *Why this is a rule and not advice:* on 2026-07-31 a full-repo DRY audit found genuine
   duplication (SimpleProp/BareProp, four copies of the auto-stepsize loop) alongside
   proposals that would have collapsed the deliberate destination/value overload pairs and
   replaced visible per-class formulae with boolean switches — because the efficiency and
   legibility constraints existed in Craig's head and in code comments, not in this file.
   → `refactor_audit.md`, and Sol's review of it.

## Housekeeping

- **GitHub:** https://github.com/craignied/neuron (HTTPS remote per the locker's
  Mac Studio SSH-push note). Created 2026-07-11; push freely on session handoff.
- **Memory:** enrolled in `~/code/claude-memory/` as `neuron-3.0` (2026-07-11).

## Where things stand (2026-07-29)

Present tense, rewritten each session — **not** a log. The dated record is
`docs/HISTORY.md`.

**The engine.** The legacy neUROn2++ C++ source carried forward and modernized in place
(C++17, zero-warning Release build, `unique_ptr` ownership, seedable `std::mt19937` behind
`util::d_random()`). Models: SimpleProp / BareProp / BackProp feed-forward nets, binary
logistic, LDFA / QDFA, RegressNet stepwise. Statistics: exact non-parametric trapezoidal
AUC and binormal Az (least-squares z-ROC fit via fitexy over distinct operating points),
stratified bootstrap CI plus Hanley-McNeil, Kolmogorov-Smirnov, Pearson X² (statistic
only — see settled decisions), Hosmer-Lemeshow Ĉ on 10 fixed deciles, Wald tests,
condition number. **Twelve numbered legacy bugs** were found and fixed during the
reanimation; each is written up in HISTORY with the measurement that proved it. The
twelfth (2026-08-01) is the one to read if you are about to trust a passing test:
batch `BackProp` computed the CGD/Shanno search direction and then updated its weights
from the raw accumulator instead, so **selecting an optimizer was nominal rather than
computational** for twenty years — while the optimizer tests executed the dispatch and
passed, because they carry expected values for SimpleProp and BareProp only and no
golden fixture uses `BackProp` at all.

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

**Long runs report what they are doing, structurally.** Training, OBD, stepwise and CV all
run async through the same `/api/train/status` + `/api/train/stop` doors, each publishing
its own progress object — never a phase word standing in for an hour of work. CV's
(2026-07-29) is composed from two independent engine callbacks: `crossval::ProgressFn`
carries the repetition grid (runner fills fold/k, coordinator stamps procedure identity)
and `obd::ProgressFn` carries the architecture trial inside the current fold. **CV never
learns what OBD is and OBD never learns what a fold is** — the GUI composes them. Absence
is meaningful: no `inner` means no nested search, `fold: 0` means no fold is running.
OBD announces its `"probing optimizers"` phase because `algorithm=auto` spends most of a
nested run there (measured: ~11 s of 13 s, under 1% of samples carrying detail before,
97% after).

**The GUI says what it is showing.** Every panel's primary action is its last row, after
all its choices, with the status beside it (`smoke.sh` asserts this for every fieldset).
OBD's controls are named in **nodes** and the panel states it sizes one hidden layer.
Chart legends print `DataSet::monitorSet()`'s answer — validation when one is loaded, else
test — rather than re-deriving the rule in JavaScript, which is how they came to claim a
three-way split was tuning on the untouched test set. The ROC/Statistics panels name the
operation that produced them, and a standalone analysis (DFA, CV) says it did not become
the current model. The CV contrast keeps `Primary (X) − Reference (Y)` and its sign
convention visible after selection.

**Splitting and evaluation.** `src/split.{h,cpp}` owns the index-level cube: outcome-
stratified holdout, outcome × covariate strata (quantile-binned, Hamilton apportionment),
group-aware (indivisible clusters), general stratified k-fold over arbitrary stratum ids,
outcome-balanced **group** k-fold (whole-cluster bin packing), and three-way
train/validation/test. `src/crossval.{h,cpp}` runs procedures over one shared fold plan
(logistic / LDFA / QDFA / neural with nested OBD inside every fold); `src/cvreport.{h,cpp}`
renders the three-tier report. The locked-test layer scores procedures once on rows held
entirely out of CV and compares a prespecified contrast — **inference is opt-in** (see
settled decisions).

**The evaluation design is a type, not prose** (`src/evaldesign.{h,cpp}`, 2026-07-30):
`SamplingUnit` / `AucInference` / `PartitionMethod` / `Partition`, with every displayed
string and every JSON field derived from them, and **one** function —
`evaldesign::chooseInference` — mapping (declared unit × achieved partition method ×
estimability) to a permitted estimator plus a reason when the answer is none. A report can
no longer describe a design that was not run.

**Fold policy and cluster-aware inference.** `/api/cv` takes its OWN `strata=` /
`strata_bins=` / `group=` (never inherited from `/api/load`, which sizes a holdout and
answers a different question). One group key governs the locked holdout, the outer folds,
**and** the nested search's inner validation split — group-disjoint outer folds alone would
still let the architecture be chosen on rows from clusters in its own inner training set.
`independence=cluster` routes to **Obuchowski's clustered ROC covariance**
(`src/clustered_auc.{h,cpp}`), validated against the published worked example through all
**nine** of its intermediate structural components; the shared statistics layer
(`src/auccov.{h,cpp}`) holds the placements, tie handling and contrast algebra, so the two
estimators cannot disagree about an area. A cluster is any sampling unit — a clinic, a
household, a school, a repeated subject. **Grouping prevents leakage; it does not make rows
independent**, so `independence=rows` over a grouped design is refused, and Tier 1 states
how many independent clusters an interval and a *p* rest on.

**The gates, run at the end of every piece of work**: zero-warning
Release build → `tests/golden/run_golden.sh` byte-identical (3 transcripts: `xor_seed42`,
`regress_seed42`, `binormal_seed42`) → `ctest` (13 tests) → `tests/gui/smoke.sh` →
`tests/oracle/verify_oracle.sh` numerically identical → live `neuron --gui` click-through
for anything that adds a control. CI runs the
build, goldens, ctest, smoke, and the Python tools on macOS/Linux/Windows.
The automated gates are currently green, and **no GUI click-through is outstanding** —
the last one, the group-aware CV controls, was run and approved by Craig on 2026-07-31. The
live walkthrough that was running through late July is **complete and published** —
`docs/datasets/civic-choice/WALKTHROUGH.md` (2026-07-29, commit `13bc12c`), sixteen real
screenshots of the session it documents, linked from the README as the project's entry
point. It drove the whole workflow in a browser on the `52bd30f` build: the three-way
split, the logistic baseline, grouped stepwise (reverse progress/Stop/cancellation, an
immediate rerun, the forward selection summary, and the original model retraining normally
afterward), standalone LDFA/QDFA, validation-guided OBD, and the four-procedure nested CV
with its locked test — which is also the live review of the 2026-07-29 GUI batch (panel
action placement, OBD node-count labels, validation-aware legends, CV progress, contrast
direction, result provenance), every item of it visible in the published screenshots.

**Stepwise regression** (2026-07-27) runs like the other long jobs: `/api/regress&async=1` with
structured progress, a Stop that reaches the candidate training at that moment, and a persistent
results pane (the report was previously computed, returned, and discarded by the page). Candidate
refits obey the convergence contract and are **quiet** — no per-candidate classification tables,
ROC fit or test-set bootstrap, which was both a large cost and a repeated look at a set that must
stay untouched during selection. **Legacy bug #11** fixed with it: a null dereference in
`Network::computeCondNum()` on a degenerate B matrix segfaulted the process on every forward
stepwise run (and on any reverse pass that shrank the model to one input).

**The condition number** now uses every eigenvalue. It had been computed over `dimension - 1` of
`dimension` — a half-open range ending at the last one — which collapsed a 2-parameter model to a
condition number of exactly 1 and was silently correct elsewhere only when the discarded eigenvalue
happened not to be extremal. Fixed in its own commit, with no golden re-bless required. The leaked
`gsl_vector` went with it. → HISTORY 2026-07-27.

**Open work** is the one remaining ROADMAP 4 item under "What remains" below (B9, the
GUI-wide strict-parsing pass). **No GUI click-through is outstanding: Craig ran the live
GUI on the group-aware CV batch and approved it (2026-07-31).** The CV panel's three new
controls — fold stratification columns/bins, group columns, and the clustered sampling
unit — are visible, selectable and submitted by the page, after the 2026-07-30 help-text
corrections. Sol had driven the same controls on 2026-07-30; Craig's run is the approval
the walkthrough protocol asks for.

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
- **Locked-test inference is opt-in, and the DESIGN chooses the estimator.** DeLong runs
  only when the caller declares `independence=rows` over a row-wise design;
  `independence=cluster` over a group-aware design runs Obuchowski's clustered covariance;
  neither ever falls back to the other, and every refusal states its own reason. Craig's
  words: *"Metadata cannot repair an invalid p-value after it has been presented."*
  → HISTORY 2026-07-24, 2026-07-30.
- **Group-aware splitting and clustered inference are different mechanisms and are NOT
  substitutes.** Grouping decides which rows may share a partition — it prevents leakage
  and changes the estimand to "a group the model never saw". Clustering decides the
  variance. Keeping a cluster whole does not make its rows independent, so
  `independence=rows` over a grouped design is **refused**, not relabelled "descriptive
  grouping". → HISTORY 2026-07-30; `docs/roc_theory.md`.
- **The clustered estimand is the ordinary one; only the uncertainty changes.** The point
  area stays the exact patient-ROW Mann-Whitney probability over all pairs, computed by the
  SAME shared placement code as DeLong. Do NOT cluster-average the area — that estimates a
  different quantity. → `docs/roc_theory.md`.
- **The design effect runs BOTH ways — do not assert that clustering always widens the
  interval.** Measured: a cluster offset that moves both classes together makes
  within-cluster pairs more concordant and gives a *smaller* clustered SE (0.0164 vs
  DeLong's 0.0188), and the published reference example has `S11` positive for one reader
  and negative for the other. The general truth is that the clustered SE tracks a
  whole-cluster bootstrap and the row-based one does not. → HISTORY 2026-07-30.
- **Clustered refusals count CLUSTERS, not rows.** The divisors are (informative
  clusters − 1), so fifty locked rows from one cluster carry no between-cluster
  information. → HISTORY 2026-07-30.
- **The clustered area interval is deliberately NOT clamped to [0,1]**; the ordinary Wald
  DeLong interval still is (DLG-9). Matching the reference implementation's unclamped
  interval is part of the acceptance test, and an interval past the boundary tells the
  reader the normal approximation is straining. → `docs/roc_theory.md`.
- **A group tie-break must be a property of the DATA, never a group id.** Both the group
  fold planner and the nested inner split were renumbering-dependent until their final
  tie-break became the group's first row: relabelling the same clustering produced a
  different partition of the same data. → HISTORY 2026-07-30.
- **`strata=` and `group=` do not compose** into a fold plan yet (no tested joint balancing
  objective) and the combination is **refused** — never resolved by letting one silently
  win. → HISTORY 2026-07-30.
- **A large external cohort is an optional scale confirmation, never an implementation
  gate.** Acceptance is the generic criteria (arbitrary/relabelled group ids, multi-column
  keys, unequal and one-class and oversized groups, renumbering invariance, leakage,
  infeasible partitions), exercised on synthetic fixtures and repository datasets.
  Repository datasets are **integration fixtures** — they establish the endpoint and the
  workflow, not performance at scale; `./build/scale_probe` is the reproducible scale
  measurement, on generated data. → Sol, 2026-07-30.
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
- **An audit trail records a candidate BEFORE judging it, and a partial result says it is
  partial.** The entry that matters most is the one that failed, so anything written after an
  eligibility check that throws is missing exactly when it is needed; and a result assembled only
  at the end of a loop reports "nothing" instead of "incomplete" when an exception exits early.
  Record first with the statistic null, maintain per-pass state as it settles, and ship an explicit
  completeness flag. → HISTORY 2026-07-27 (Sol review).
- **"Not copied" must be WRITTEN in `copy()`, never left out.** A copy constructor that delegates
  to a `copy()` utility does NOT run the default constructor first, so any member `copy()` omits
  holds indeterminate memory. `Iterative::copy` assigns `observerPtr = nullptr` and
  `quietFlag = false` explicitly for this reason; a new non-copied member does the same.
  → HISTORY 2026-07-27 (Sol review); `tests/iterative/check_quietcopy.cpp`.
- **Stepwise regression requires a cross-entropy source fit and never mutates the model.** Wilks'
  GLRT compares log likelihoods; the baseline error comes from the ORIGINAL fit, so converting the
  model afterwards mixes objectives instead of fixing them. Refuse an LMS fit; do not call
  `setXEerror()` on a user's model. → HISTORY 2026-07-27 (Sol review).
- **A quiet fit is a silent fit, never a different one.** `Iterative::setQuiet()` suppresses the
  run header, the per-iteration rows and the whole epilogue for fits whose output nobody consumes
  (stepwise candidate refits). It is OUTPUT ONLY: it may never guard a calculation a stopping rule
  reads — the gradient calculation stays outside every reporting guard (legacy bug #10's sibling).
  Default OFF, so an ordinary run is bit-identical to one before it existed. → HISTORY 2026-07-27.
- **Stepwise candidate fits obey the convergence contract, and an unconverged candidate FAILS the
  analysis rather than being skipped.** A pass selects by comparing all its candidates, so dropping
  one silently changes which variable wins. Do not "fix" a refusal by loosening tolerances or
  raising ceilings to make a fixture pass. → HISTORY 2026-07-27.
- **Reporting cadence may change output VOLUME; it may never change optimization or fit
  validity.** The print counter is presentation only. Do not "solve" a stopping problem by
  choosing a denser print schedule, and never compute a quantity a stopping rule depends
  on inside a block that runs only when something is displayed — that was legacy bug #10,
  where logarithmic vs linear printing chose the fit (canonical stopped at 304 / 400 /
  1000 on one fixture) and a ceiling between print points reported a false failure to
  converge. → HISTORY 2026-07-26; `tests/iterative/check_gradcadence.cpp`.
- **A displayed fact is derived from the object that owns it, never copied by a caller.**
  Tier 2's fold-plan header printed `(development rows only) ... n = 6000` from
  `PlanInfo`, whose n is the whole dataset, and `cv_run.json` shipped `"n": 142` beside
  `"events": 59` — a row count and an event count describing different sets of rows. The
  fold plan's counts now come from the `Comparison`, which knows which rows it folded. Do
  NOT add a "development n" field for a caller to fill in; that is the same defect with a
  longer name. → HISTORY 2026-07-29.
- **A rule with two kinds of consumer is stated ONCE, in the class layer.** Which held-out
  set a monitor watches is `DataSet::monitorSet()`: samplers act on it and displays name
  it. The chart legends were a second implementation in JavaScript, hard-coded to "test
  set", and so announced that a three-way split was tuning on the untouched test set while
  the engine correctly watched validation. Do not re-derive it in a caller. → HISTORY
  2026-07-29.
- **Progress reporting composes callbacks; it never teaches one layer about another.** CV
  publishes fold/procedure through `crossval::ProgressFn`, the nested search publishes its
  trial through `obd::ProgressFn`, and the GUI joins them. A "nested phase" field on
  `crossval::Progress` would put model knowledge in the runner. Absence stays meaningful:
  no `inner` means no nested search, `fold: 0` means no fold is running — never a
  fabricated zero. → HISTORY 2026-07-29.
- **A phase that costs wall-clock time must report itself, even when it has nothing to
  plot.** `algorithm=auto`'s optimizer probe was ~11 s of a 13 s nested run and reported
  nothing, so under 1% of status samples carried any detail; announcing a
  `"probing optimizers"` phase took it to 97%. The probe carries no measurement
  (iteration 0, errors -1) and must NOT be pushed to the chart — it is a status, not a
  sample. Polling harder is not the fix for a silent phase. → HISTORY 2026-07-29.
- **A panel that renders someone else's numbers says whose they are.** The ROC/Statistics
  panels name the operation that produced them; DFA and CV declare themselves standalone
  because neither becomes `modelPtr`. Never label a standalone analysis "current model",
  and never let CV's report sit above an unlabelled panel from a different fit. → HISTORY
  2026-07-29.
- **Palm/iPhone exporters stay dead**; the HTML calculator lives on as
  `tools/neuron2web.py`. Python tooling is **stdlib-only** — no pip, no venv, enforced by
  CI on all three OSes.

## ROADMAP 4 (agreed with Craig 2026-07-22) — a general representative test-set splitter

### Why (rationale)

**Moved to `docs/HISTORY.md` → "ROADMAP 4 — the plan as executed" → "Why (the rationale)"
on 2026-07-28**, once all four phases had shipped. It is not retired — it still governs
splitter design, and you read it before extending the splitter, the fold plans, or the
diagnostic. It left the constitution because this file carries live work, and the argument
for finished work belongs with the record of finishing it. The three items still open are
below.

<!-- The rationale that used to sit here covered: the two O(n²) hot spots in the old
     `DataSet::randomize`; why the rebuild targets a GENERAL splitter with SEER as the
     acceptance test rather than a SEER fix; what "representative" means precisely; the
     three design axes (stratify × group × estimator); the two places generality is won
     or lost (stratified-group k-fold as a greedy approximation, per Sechidis, Tsoumakas
     & Vlahavas 2011; degeneracy as first-class); the common index-gather foundation; the
     split-diagnostic report; and the discrete-outcome scope boundary. -->

### What remains

Phases 1–3 (the efficient index-gather rewrite, generalized stratification + the
diagnostic, group-aware splitting) and Phase 4 (cross-validation, the three-tier report,
`/api/cv`, and the locked-test DeLong layer) are **DONE and shipped**. The phase-by-phase
plan as agreed and executed is in `docs/HISTORY.md` → "Completed roadmaps → ROADMAP 4";
read it before extending any of that machinery, because it records which parts of the
design were corrected mid-build and why.

Items 1 and 2 — **group-aware / covariate-stratified CV folds** and **cluster-aware
(non-IID) locked-test inference** — are **DONE and shipped** (2026-07-30). Their
implementation record and settled decisions are in `docs/HISTORY.md`. One item remains:

1. **B9 — the GUI-wide strict-parsing pass.** Shared strict integer / floating-point /
   boolean parsers with full-string consumption, range and overflow checks, and
   field-specific errors, migrated across **every** handler (`atol`/`atof` today accept
   `folds=5junk` → 5, and any non-`"1"`/`"true"` boolean → false). Do NOT broaden accepted
   boolean spellings unless that becomes explicit API policy.

### Verification (end of every phase)
Zero-warning Release build → `tests/golden/run_golden.sh` (byte-identical; the Phase-1
`binormal_seed42` re-bless is done and no further move is expected — read any diff) →
`tests/gui/smoke.sh` (extended per phase) → `ctest` → `tests/oracle/verify_oracle.sh`
(numerically identical — the splitter path is not on the oracle) → live `neuron --gui`
click-through for any phase that adds a control. AGENTS.md, `docs/gui_cli_parity.md`, this
file, and the session entry in `docs/HISTORY.md` all land in the same commits.

**Acceptance is the generic criteria, not a dataset.** What the partition and inference
layers must satisfy is exercised on synthetic fixtures and repository datasets: arbitrary
(non-dense, relabelled) group identities, multiple group-key columns, unequal group sizes,
one-class groups, an oversized group, renumbering invariance, zero leakage, and infeasible
partitions refused with a reason. `./build/scale_probe [rows] [clusters]` is a reproducible
scale measurement on generated data (built by CI, deliberately not a ctest case — timings
are machine-dependent). A large external cohort is an **optional** confirmation of scale
and never an implementation gate; the repository datasets are integration fixtures — they
establish the endpoint and the workflow, not performance or memory at scale.

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
