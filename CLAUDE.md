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

## Housekeeping

- **GitHub:** https://github.com/craignied/neuron (HTTPS remote per the locker's
  Mac Studio SSH-push note). Created 2026-07-11; push freely on session handoff.
- **Memory:** enrolled in `~/code/claude-memory/` as `neuron-3.0` (2026-07-11).

## Status

- **2026-07-11** — Project reanimated: explored `../distro`, wrote CLAUDE.md + README.md,
  initialized git (branch `main`). Direction decided (see above); legacy docs pulled into
  `docs/`.
- **2026-07-11 (later)** — Legacy binary verified working as the numerical oracle: built
  arm64 against GSL 2.8, ran a scripted XOR session end-to-end (100% CA, ROC 1.0, K-S /
  Pearson / H-L stats all exercised). Harness committed in `tests/oracle/` (originally `baseline/`, reorganized same day).
- **2026-07-11 (evening)** — **neuron 3.0.0-dev is alive.** Strategy: carry the legacy
  engine forward as the 3.0 base and modernize in place (not a ground-up rewrite).
  Done: legacy source copied to `src/` (originally `engine/src/`), made C++17-clean (dropped 5
  `unary_function` inheritances in function_defs.h, stripped 6 `throw(DivisionByZero)`
  specs in twoset.{h,cpp}), `config.h` → `version.h` (NEURON_PACKAGE_STRING),
  top-level CMakeLists.txt (C++17, `find_package(GSL)`), builds with **zero warnings**.
  Verified two ways: (1) XOR training run — same endpoint stats as legacy;
  (2) `tests/oracle/verify_oracle.sh` — legacy trains & saves a network, both binaries load
  the same weights and do an eta-0 forward pass, outputs **numerically identical**
  (diff-clean). Note: `distro/data/Run9` (11-input SimpleProp) has no matching dataset
  in distro/data (BP40 files are 13-col), so the oracle check generates its own network.
  Next: modernization passes on the engine (memory safety — raw new/delete →
  smart pointers; `matrix.h` templates; seedable RNG for reproducible training),
  then start `tools/` (Python data grooming, replacing mkdataset.pl).
- **2026-07-11 (night)** — Version-reference cleanup + **first real bug fix**. The
  legacy build is now just "the oracle" everywhere (`tests/oracle/bin/oracle`); the
  old version number appears only in the tarball path inside `build_oracle.sh`.
  While re-verifying, the harness caught a genuine ~20-year-old bug: `TwoSet::KScalc()`
  kept Numerical Recipes' 1-based indexing on 0-based vectors → skipped element 0 and
  read past the end of both arrays (heap-garbage K-S values; oracle showed D=0.5 vs
  correct D=1 on identical weights). Fixed in `src/twoset.cpp`; oracle keeps the bug,
  so `verify_oracle.sh` excludes the K-S line from the diff and asserts 3.0's K-S
  against the known-correct value. Verify is now fully deterministic — it loads the
  committed reference network `tests/oracle/xor_net.txt` instead of retraining.
  Audited `stats.cpp` for sibling 1-based NR bugs: none found (`fit()` is 0-based;
  the `1..ITMAX` loops are iteration counters).
- **2026-07-12** — Pushed to GitHub (`craignied/neuron`), memory enrolled. **Seedable
  RNG landed:** `util::set_seed()` + `neuron --seed N` (and `--version`). One seeding
  point controls everything (weights + train/test splits all flow through
  `util::d_random()`/`rand()`); unseeded behavior unchanged (clock). Verified: two
  seed-42 XOR runs byte-identical except the wall-clock "That took" line; different
  seed diverges; verify_oracle.sh still passes. Note: `rand()` streams are
  libc-specific, so cross-platform runs with the same seed may differ — same-machine
  reproducibility is the contract.
- **2026-07-12 (later)** — **Smart-pointer pass + EOF guard.** Driver ownership is now
  `unique_ptr`: `specify_data()`/`specify_model()` return `unique_ptr<DataSet/Model>`,
  main() holds them (manual delete/double-free risk gone); RegressNet's `netCopyPtr`
  is `unique_ptr<Network>` (empty dtor used to leak the working copy on exceptions).
  Also fixed en passant: 2.x's `specify_data()` failure path nulled the raw pointer
  without deleting (leak), and all `util::ask*` functions looped forever on dead cin —
  a desynced scripted run wrote a 9 GB prompt file before being caught. New
  `check_input_ok()` guard exits cleanly on EOF/failed extraction ("Input ended
  unexpectedly"). Verified: builds zero-warning; verify_oracle.sh passes; seed-42 XOR
  transcript byte-identical to pre-refactor; stepwise reverse regression (RegressNet's
  copy machinery) smoke-tested end-to-end (needs a p-value threshold answer after
  choosing reverse/forward — scripts beware). Non-owning params (use_model, dfa, etc.)
  stay raw pointers by design.
- **2026-07-12 (later)** — **RNG switched to std::mt19937** (in utility.cpp behind the
  unchanged `util::d_random()` interface; new `util::i_random(n)` replaces the raw
  `rand() % n` calls in vector_ops.cpp). Seeded streams are now specified by the C++
  standard → reproducible across platforms/compilers (old `rand()` was libc-specific;
  MSVC's RAND_MAX is 32767). Doubles are mapped from raw generator output, NOT
  std::uniform_real_distribution (whose implementation varies between stdlibs — don't
  "modernize" that away). Seed streams differ from the rand() era (seed-42 XOR now
  converges at 90k iters vs 300k). Verified: zero warnings, verify_oracle.sh passes,
  seed-42 reproducible, stratified train/test randomize path exercised. This unlocks a
  cross-platform golden-transcript test (not yet built).
- **2026-07-12 (night)** — **Golden-transcript test + CI matrix.** `tests/golden/`:
  two seeded sessions (xor_seed42: training+stats; regress_seed42: + stepwise reverse
  regression) must reproduce committed transcripts byte-for-byte (only "That took"
  excluded); `run_golden.sh --bless` regenerates after intentional output changes.
  `.github/workflows/ci.yml`: macOS/Linux/Windows matrix — GSL via apt/brew/vcpkg
  (vcpkg cached), CMake build, golden test on all three (mt19937 makes transcripts
  platform-portable). `.gitattributes` forces LF so Windows checkouts compare clean;
  run_golden.sh also strips CR and finds build/Release/neuron.exe (MSVC multi-config).
  Oracle check stays local-only (needs ../distro tarball).
- **2026-07-12 (CI day-one catch)** — First matrix run: xor golden passed on ALL
  three OSes (mt19937 portability proven); regress golden failed on Linux/Windows
  with different garbage → **third legacy bug found: stale/uninitialized TwoSet
  statistics.** Guesses are written via `test(r) = o` (network.cpp) which never
  invalidated cached stats: 2.x reprinted the FIRST run's K-S/Pearson/H-L on every
  retrain, and TwoSet::copy() copied calc flags but not values (regression printed
  uninitialized memory; mac's zero pages hid it). Fix: `TwoSet::invalidate()` called
  from the non-const `test()` accessor; copy() now copies KSD/KSP/PKX2P/HLX2P;
  initialize() zeroes all stat values. Goldens re-blessed — null-model lines now
  show computed values (D=0 **p=1**, Pearson p=0.318907) instead of lucky zeros.
- **2026-07-12 (tools begin)** — `tools/` scaffolded with the **stdlib-only rule**
  (bare python3, no pip/venv — Craig's #1 distribution pain; CI runs the tools with
  system python3 on all 3 OSes to enforce it). First tool: `mkdataset.py`, faithful
  port of mkdataset.pl (blank-outcome row elimination, non-numeric row warnings,
  missing-value indicator pairs "0,0"/"1,value", --key and --inputs files; csv-module
  parsing handles quoted Excel fields, an improvement over Perl's split). Fixture
  test `tests/tools/run_tools.sh` (--bless pattern) diffs 4 outputs against committed
  expected; output verified loadable by the engine (3x5 raw load). Windows note:
  scripts locate python3 OR python; output files written with newline='\n'.
- **2026-07-13** — **Wickens ROC verified + documented.** Investigated
  `code/C++/msvc/roc` (MFC app, 2001/2005): it IS the Wickens binormal Az method —
  and the engine already contains it (`TwoSet::getStatROCarea`, cites Wickens
  pp. 60-74; merged pre-2.6, engine copy is newer than the app's). New unit test
  `tests/binormal/check_az.cpp` (ctest, no GSL): Gaussian populations with known
  mu/sigma → engine Az matches Phi(dmu/sqrt(s0²+s1²)) to ~4 decimals incl. unequal
  variance; runs in CI. Craig's theory notes adapted into `docs/roc_theory.md` with
  an implementation-mapping + provenance section. Still to port from the roc app
  when the GUI lands: ROCx/ROCy curve-point capture (for plotting). GUI decisions
  so far: embedded server in binary (`neuron --gui`), bind port 0 (OS-assigned),
  loopback only.
- **2026-07-13 (ROC CIs)** — **95% confidence intervals on every ROC area.** Binormal
  Az: **delta method** — propagates the z-ROC fit's siga/sigb through Az=Φ(a/√(1+b²))
  via the analytic gradient (`TwoSet::azSE`). Trapezoidal: **Hanley-McNeil** closed-form
  variance from the Mann-Whitney U equivalence (`TwoSet::hmSE`). NEITHER is a bootstrap —
  both analytic/parametric. Printed in `statReport` and after the trapezoid line; new
  public getters `getStatAzSE()`/`getTrapSE()`. `check_az` now also validates SE
  calibration by simulation. Oracle diff excludes "95% CI" (new in 3.0). Fixed a latent
  bug found on the way: `XY` fitexy ctor left `_r` uninitialized.
  - **Caveat for publication:** the binormal delta-method interval has a known
    approximation here — on the binned fitexy path the routine doesn't expose the a–b
    covariance, so that cross term is set to zero. In calibration testing the delta SE
    ran a bit narrow (reported SE ≈ 0.56× empirical SD — anti-conservative); Hanley-McNeil
    calibrated cleanly (ratio ≈ 1.09). The trapezoidal CI is the more trustworthy of the
    two as it stands; recover the covariance term or validate the binormal CI further
    before leaning on it in print. Methods-section language + this caveat live in
    `docs/roc_theory.md` and README.

- **2026-07-13 (Phase 1 complete: bank walkthrough)** — `mkdataset.py` gained
  `--delimiter`, `--onehot` (auto-detect or explicit columns; two-valued text outcome
  → single 0/1 column; blank categorical = all zeros), and `--refcat` (k−1 reference-
  category coding; first bank logistic run proved why: full one-hot + intercept =
  perfect collinearity, Shanno ground 15+ CPU-min without converging). Fixtures now
  cover all three invocations; original expected files untouched (default path is
  byte-identical). `docs/datasets/bank-marketing/WALKTHROUGH.md` written from real
  verified transcripts: groom (42 inputs) → seed-42 stratified 75/25 split → logistic
  (canonical GD, 20k-iter cap, 12 s, converged MLE, test ROC 0.874 both methods) and
  SimpleProp 42-5-1 (canonical, 2k cap, 4 s, test ROC 0.871). **Training-recipe
  lesson baked into the doc:** default stop conditions (1M iters) are wrong for
  thousands of rows — uncapped CGD ran 80 min and ended worse than useless (test ROC
  0.57, auto-step instability + "Numerical out of bounds"); cap iterations, prefer
  canonical GD; Shanno/CGD line searches are expensive here. Known quirk surfaced:
  binormal search can fail with "a too large, ITMAX too small in gcf" (legacy message,
  in spin.html too) — trapezoid+H-M CI is the fallback. Also imported
  `docs/datasets/low-birth-weight/` (H&L classic): committed betas are the converged
  MLE (max grad 3.4e-07 on load) and LL = −111.2865 matches H&L's published −111.286
  — a self-verifying logistic reference check. CI/README builds now use `--parallel`.

- **2026-07-13 (Phase 2 complete: deployment + AGENTS.md)** — `tools/neuron2web.py`
  built and verified: parses saved SimpleProp/Binary-logistic networks + scaling
  factors + legacy tag spec (with the `*`-reference-level extension for --refcat),
  emits ONE self-contained HTML calculator (all inline JS; post on any static host —
  Craig's Dreamhost question, answered by design), `--serve` on port 0, `--eval` for
  spot checks. Forward pass verified against engine guesses on the committed XOR
  network (max |diff| = 0, fixture-tested in run_tools.sh; reference files
  tests/tools/xor_{scales,guesses,spec}.txt generated by the engine itself).
  Bank walkthrough gained §9: retrained logistic saving network+scales (seed-42
  identical), deployed with committed `bank_spec.txt` (42 inputs validated).
  `docs/deploy.md` = full spec reference. **`AGENTS.md` added at repo base** —
  operating manual for AI assistants ("I have a dataset" → groom → train → deploy,
  with verified session templates and decision rules). STANDING RULE: whenever
  tools, menus, or recipes change, update AGENTS.md in the same commit — its
  recipes are promised to work.

- **2026-07-13 (logs follow the data)** — Craig: run files should land where the
  input files are. `util::set_run_dir()`/`util::run_path()`: the first dataset file
  loaded in a session (loadRaw/loadTrain/loadTest — getGoodFile writes the resolved
  path back into the caller's string) fixes the run directory; all five log-open
  sites (dataset/model addHistory, iterative/ldfa/qdfa lastop) resolve neuron.log /
  model.txt through it. Bare names only — explicit paths set via the API are
  respected; no dataset loaded → cwd as before. stdout unchanged → goldens/oracle
  byte-identical. neuron.log + model.txt now gitignored; spin.html + AGENTS.md
  updated. Behavior-verified: launch dir stays empty, log lands next to data.

- **2026-07-13 (Phase 3 complete: engine/UI decoupling)** — Inventory: neuron.cpp
  (189 couts) + util::ask* (23) are the interactive driver, stay cout/cin; 13 engine
  .cpp files + matrix.h load/save reports (155 sites total) now print via
  **util::screen()** (settable ostream, default cout; util::set_screen() redirects —
  the GUI/server capture contract). Conversion was mechanical (regex on code, comments
  skipped); goldens/oracle byte-identical (default unchanged). New ctest
  `screen_capture` (tests/capture/) proves both capture paths: internally-routed
  prints (DataSet refusal message) and ostream&-parameter reports (ClassTable).
  **ROCx/ROCy curve capture ported** from `code/C++/msvc/roc/roc 2005/twoset.cpp`:
  getTrapROCarea() fills the vectors (x = 1-spec, y = sens), getROCx()/getROCy()
  const-ref getters, copy() copies them; asserted in screen_capture (area ~1 on
  separated data, points in unit square). GUI (Phase 4) now has both prerequisites:
  capturable reports + plottable curve points.

- **2026-07-13 (Phase 4 complete: `neuron --gui`)** — Embedded web GUI shipped.
  cpp-httplib v0.18.3 vendored (third_party/, MIT + license file); `src/gui.{h,cpp}`
  + `src/gui_page.html` (embedded via CMake configure_file → raw-string header,
  CMAKE_CONFIGURE_DEPENDS so page edits re-embed). `--gui [--no-browser]`: binds
  127.0.0.1 **port 0** (OS-assigned), prints URL, auto-opens browser (open/start/
  xdg-open). Endpoints (form-encoded POST → JSON): /api/load (raw+randomizeD or
  training-set file; **pre-checks file existence because getGoodFile would prompt
  on stdin**), /api/model (logistic | simpleprop+hidden; setDataSet BEFORE
  setHidden), /api/train (algorithm 1-3, maxiter, optional seed; RAII Capture via
  util::set_screen returns the full engine report as text; ROC curves from the
  MODEL's DataSet copy — that's where train() writes guesses — via getTrapROCarea()
  + getROCx/y from Phase 3). One mutex serializes all engine access. Page: single
  self-contained HTML/JS, canvas ROC plot (train/test + areas). CMake: Threads;
  ws2_32 on Windows. Verified: full curl session (XOR → SimpleProp → train → JSON
  with report + curve); `tests/gui/smoke.sh` (bash+curl, in CI on all 3 OSes).
  In-browser click-through not yet done (Chrome extension disconnected) — page JS
  is fetch/canvas, verified only by inspection; if the page misbehaves, that's
  where to look. Parked idea remains: wasm build on GitHub Pages.

- **2026-07-13 (GUI session files + legacy bug #5)** — Craig: the GUI must output
  everything the CLI could (deployment trio + paper trail), and asked about binary
  relocatability (yes: page embedded, GSL by absolute path — same-machine relocatable;
  outputs land in the CWD it's run from, so one dir per experiment). GUI gained
  `/api/save/:what` (network, scales, train_set, test_set, train_guesses,
  test_guesses, report): the engine's own save methods write into the workspace AND
  the browser downloads a copy (page section 4). test_set guarded in the GUI —
  DataSet::saveTest doesn't check testLoadedFlag itself. File picker (same day):
  uploads content (browsers hide paths), server saves beside itself. **Legacy bug
  #5 found by the fraction=0 path: DataSet::randomize flagged an EMPTY test set as
  loaded → Model::extractInputMatrices underflowed (nTest-1) → crash.** Fixed:
  testLoadedFlag = (rows > 0). Verified end-to-end: bank 42-input logistic through
  the GUI (upload → split → train → all saves) → downloaded network+scales deploy
  cleanly via neuron2web.py (0.118488 for the known-no first client). smoke.sh
  covers saves + the no-test-set refusal.

- **2026-07-14 (ROADMAP 2 planned)** — Session spent in plan mode; nothing implemented
  yet. Craig set five priorities: (1) GUI overhaul — drop the log pane, surface ALL stats
  beside the ROC plot, realtime error-vs-iteration chart (dashboard-card style per his
  DQN dashboard screenshot); (2) auto algorithm selection (probe GD/CGD/Shanno, pick,
  report); (3) plateau auto-stop; (4) automated OBD hidden-layer sizing (hybrid
  grow→prune, Craig's choice); (5) new optimizers from the literature (Rprop/Adam/
  L-BFGS/LM; Muon experimental). Key decision: **CLI menus sunset as the human driver**
  — GUI for humans, HTTP API for LLMs/scripts (curl); menus frozen but working, goldens
  stay byte-identical. Full approved plan → ROADMAP 2 below. Exploration verified:
  train() has no hook, screenPtr is a process-global (thread_local fix planned),
  Wald/condNum/confusion counts are print-only today, setHidden is destructive.
  Phase 1a is on the session task list, not started.

- **2026-07-15 (Phase 1a verified in a real browser + binormal reconciled)** — First
  actual in-browser click-through of `gui_page.html` (Chrome extension connected; the
  page had only ever been checked by curl + inspection). Drove the H-L low-birth-weight
  set (189 rows → 142/47) through upload → split → logistic → train: ROC plot, full
  stats panel, confusion tables, and the coefficient table all render; **zero page JS
  errors**. Two findings, both about the binormal Az:
  1. **The delta-method SE tracks the BIN COUNT, not the exemplar count.** n=142 and
     n=47 gave near-identical SEs (0.0102 vs 0.0103) while Hanley-McNeil scaled properly
     (0.049 vs 0.091 ≈ √(142/47)). Within one set, the 9-bin fit gave SE 0.014 and the
     5-bin fit 0.034 — same data, 2.4× apart. This *sharpens* the known caveat in
     `docs/roc_theory.md` (SE ≈ 0.56× empirical): the delta method propagates the z-ROC
     line fit over a handful of binned points, so it never sees n. **Still open** —
     the backlog's "recover the a-b covariance" item is the fix. Hanley-McNeil remains
     the trustworthy interval and is what the page's headline "95% CI" uses.
  2. **Panel/report mismatch — FIXED.** `ROCarea` searches nBins 3..10 and prints the
     best-p and best-AUC fits, then restores `nBins=10`; the GUI afterwards called
     `getStatROCarea()`, which re-binned at 10 and produced a *third* Az/SE appearing
     nowhere in the report. Fix: the search is extracted into `TwoSet::searchROC(ostream&)`
     caching two `TwoSet::ROCfit`s (`az,se,chi2,p,nBins`), exposed via `getBestPfit()`/
     `getBestAUCfit()`/`getROCsearchFailed()`; `ROCarea` calls it (printed bytes
     unchanged — goldens byte-identical), and the panel now quotes those same two fits
     **labeled with their bin counts**. Getters search on demand only when the cache is
     stale, so asking never changes what a later report prints; `invalidate()` clears
     the cache and `copy()` copies it (the 2026-07-12 stale-stats bug class).
     `countGoodData()` extracted as a shared helper. Verified: panel == report field for
     field, incl. `SE = nan` → `n/a` on a degenerate 3-bin test fit.
  Gates: zero-warning build, goldens byte-identical, smoke (+3 binormal assertions),
  ctest, verify_oracle, live click-through. **Lesson: a binormal Az is meaningless
  without the nBins that produced it — never quote one without the other.**

## ROADMAP 2 (agreed with Craig 2026-07-14) — training automation & GUI overhaul

Plan-mode approved 2026-07-14. Work the phases in order; each lands independently with
tests green. **Task #1 (Phase 1a) is already on the session task list — start there.**

**Decisions made with Craig:**
- **OBD strategy: hybrid** — grow until overlearning (test diverging from train), prune
  back by saliency, confirm.
- **CLI menus are sunset as the human driver.** The GUI is the human interface; the
  **HTTP API is the permanent headless/LLM access path** (curl, documented in AGENTS.md).
  Menus keep working unchanged for now (goldens byte-identical) but get no new features.
- UI visual reference: Craig's DQN dashboard (screenshot 2026-07-14) — white cards with
  titled live line charts, big current-value readout, big-number stat tiles, status chip.

**Standing constraints:** engine features UI-agnostic (engine → HTTP API → page); zero new
dependencies; goldens/oracle/smoke green each phase; AGENTS.md updated in the same commit
as any API change. Invoke the `dataviz` skill before writing chart code (Phase 1b).

### Architecture decisions (apply across phases)

- **Async training** (enabler for everything realtime): add `Iterative::Observer`
  (`onIteration(iteration, setError) → bool`, false = stop) called at the *bottom* of the
  train loop after all existing stop checks — no-observer runs bit-identical (goldens
  safe). `StopReason` enum + `getStopReason()`, set at each existing `break` site (zero
  output change). **`Iterative::copy()` must null the observer** (clones must not drive
  GUI buffers). `Network::sampleTestError(stride)` mirrors reportAccuracy's test loop;
  GUI observer samples on a wall-clock schedule (≥250 ms apart, ≤~1000 exemplars/sample).
  gui.cpp `TrainJob`: worker `std::thread`, `atomic<bool> cancel`, `progressMutex`-guarded
  decimated series (cap 2000 points, halve+double-stride when full). Worker does NOT hold
  `engineMutex` while training; `job.running` (toggled under `engineMutex`) gates every
  mutating handler → HTTP 409 `{"ok":false,...,"busy":true}`, no queueing.
  `POST /api/train` stays **blocking by default** (smoke.sh contract intact); page passes
  `async=1` → returns immediately. New: `GET /api/train/status` (poll 400 ms),
  `POST /api/train/stop`. Cancellation falls through to the normal epilogue
  (reportAccuracy, logs, lastTrainError) → a cancelled run is a completed run,
  `stopReason:"cancelled"`. **Race fix: make `screenPtr` `thread_local`**
  (utility.cpp:355) — CLI unchanged; each request thread + worker get independent
  Captures; extend tests/capture with a two-thread assertion. Rule: any new engine
  thread must set its own screen redirection first.
- **Stats surfacing** (print-only → getters; printed bytes identical): TwoSet
  `getTP/getTN/getFP/getFN`; Logistic `Bstderr`/`WaldP` promoted to members (+ copy()) with
  getters; Network `reportCondNum` split into `computeCondNum()` + members/getters +
  identical prints. gui.cpp `jsonStatsSeries(TwoSet&)` — **every stat individually
  try/caught → null** (binormal/K-S/H-L/sens/spec throw on degenerate sets). `stats`
  object added to train JSON (additive; `roc` kept verbatim) + new `GET /api/stats`
  (recompute w/o retraining; 409 while training). Shape: per-set `{n, confusion{tp,tn,
  fp,fn}, acc, sens, spec, pvp, pvn, trap{area,se}, binormal{bestP{az,se,p,chi2,nBins},
  bestAUC{...}}, ks{d,p}, pearsonP, hlP}` + `logistic{condNumber,
  coefficients[{input,beta,se,waldP}]}`. **The binormal block quotes the report's two
  searched fits, not a fit of its own** — see the 2026-07-15 status entry for why the
  bin count travels with each Az.
- **Auto algorithm** (engine-side): new `src/netclone.{h,cpp}` `cloneNetwork()` — the
  RegressNet::copy_network typeid dispatch generalized (refactor RegressNet to use it;
  pure refactor, goldens hold). New `src/autoalgo.{h,cpp}` `pick(start, budgetMs,
  cancel)`: clone ×3 from identical weights, set trainingType per clone AFTER copying,
  force `setBatchEpoch(true)` for CGD/Shanno probes, **equal wall-clock budget (default
  750 ms/probe)** via a small Observer; probes train into a discard stream, then one
  compact decision summary via `util::screen()`. Winner = lowest final training error
  (NaN/diverged last; tie → simpler algorithm). **Adopt the winning clone** (swap into
  modelPtr; probe progress kept) and continue training to the user's maxiter. API:
  `algorithm=auto`; JSON `autoAlgo{selected, selectedName, probes[]}`.
- **Plateau auto-stop**: new header-only `src/plateau.h` `PlateauDetector(window, tol,
  patience)` — two-window moving-average comparison; relative improvement < tol = strike
  (flat OR rising both count); patience consecutive strikes → stop. Sawtooth robustness:
  `W_eff = max(window, 2·plateauPeriodHint())`, Network overrides hint to `df()`
  (CGD/Shanno restart period). Defaults 100 / 1e-4 / 3. New flag family in Iterative,
  **default OFF** → goldens untouched; distinct "The error plateaued…" line;
  `STOP_PLATEAU`. API: `autostop=1` (+ `autostop_tol`, `autostop_window`); JSON
  `stopReason` string on every result.
- **OBD hybrid driver**: SimpleProp `growHidden(extra)` (new units: small random hidden
  weights, **zero outgoing weights** → function unchanged at growth = warm start),
  `shrinkHidden(keep)`, `hiddenSaliency()` = `|oW[j]|·std(hO_j over train set)`. Clear
  `stackG/lastG/lastF` on resize; `weightsSetFlag` stays true. New `src/obd.{h,cpp}`
  `ObdDriver` (engine-side, RegressNet-style; **requires a test set**, refuse otherwise):
  optional autoalgo probe once → GROW from h=2 by 1 (each size: plateau-stop forced on +
  iterBudget cap — features 3+4 compose), overlearning = test error above
  `bestTestErr·(1+riseTol)` for 2 consecutive sizes while train error non-increasing →
  PRUNE from best net: remove argmin saliency, brief confirm retrain, accept within
  pruneTol else restore and stop. Emits size-vs-metrics table into the report. API:
  `POST /api/obd` (async-only, shared job machinery; result `selectedHidden`, `history[]`
  + normal output/roc/stats).
- **Page overhaul**: report `<pre>` **deleted** (report stays on disk +
  `/api/save/report`); full stats panel (tiles + 2×2 confusion tables + coefficient
  table) beside the ROC canvas; realtime error-vs-iteration canvas (log-y, train +
  sampled test); Train button becomes Start/Stop toggle; OBD button + size-vs-error chart.

### Phases
- **1a — stats getters + full stats panel** (training still blocking). Files: twoset,
  logistic, network, gui.cpp, gui_page.html. Smoke asserts `stats`/`confusion`/`waldP`/
  `condNumber`; `/api/stats` round-trip; goldens byte-identical through print refactors.
- **1b — async training + realtime graph**. Files: utility.cpp (thread_local),
  iterative (Observer/StopReason), network (sampleTestError), gui.cpp (TrainJob, 409,
  async=1, status/stop), gui_page.html. Smoke: existing lines unchanged + async
  poll-to-done, 409 busy, stop → `"cancelled"`; capture ctest two-thread assertion.
- **2 — auto algorithm**. New netclone + autoalgo; regressnet refactor. Smoke:
  `algorithm=auto` → `autoAlgo` w/ 3 probes + `Selected` in report; goldens hold.
- **3 — plateau auto-stop**. New plateau.h + tests/plateau ctest (4 synthetic traces:
  improving decay no-trigger, decay-then-flat trigger, flat sawtooth w/ window ≥2p
  trigger, sawtooth+slow-decay no-trigger). Smoke `autostop=1` → `stopReason`. Also
  document (AGENTS.md) the pre-existing quirk: gradMax stop only refreshes on print
  iterations — leave as-is.
- **4 — OBD**. simpleprop grow/shrink/saliency + obd driver + `/api/obd` +
  tests/obd ctest (grow-with-zero-oW → `forward()` bit-identical; grow-then-shrink-back
  restores outputs; zeroed unit saliency 0). Smoke: raw load fraction=0.25 → obd →
  `selectedHidden`+`history`; refusal without test set.
- **5 (later) — new training algorithms from the literature**: Rprop (ideal full-batch),
  Adam, L-BFGS (`gsl_multimin_fdfminimizer_vector_bfgs2`), optional Levenberg–Marquardt;
  Muon = experimental novel candidate (2026 tabular-MLP benchmark, arXiv 2604.15297).
  Slot in as trainingType 3+ in `Network::engine` dispatch; runHeader naming extended;
  autoalgo::pick iterates the extended list automatically; batch-only methods force
  batchEpochFlag like CGD/Shanno probes.

### Verification (end of every phase)
Zero-warning build → `tests/golden/run_golden.sh` byte-identical → `tests/gui/smoke.sh`
(extended per phase) → `ctest` → `tests/oracle/verify_oracle.sh` (local) → live
`neuron --gui` click-through on the bank dataset. AGENTS.md + this file updated in the
same commits.

## ROADMAP 1 (agreed with Craig 2026-07-13) — ALL FOUR PHASES DONE, kept for reference

Work these in order; each phase lands independently with tests + CI green.

### Phase 1 — Bank-data grooming walkthrough (do first: self-contained, no engine risk)
1. Extend `tools/mkdataset.py` (stdlib-only rule holds): `--delimiter` option
   (bank.csv is semicolon-delimited, quoted fields — csv module handles quotes),
   **categorical one-hot encoding** (auto-detect non-numeric columns, or explicit
   column list), and binary text outcome mapping ("yes"/"no" → 1/0). Must still emit
   `--key` and `--inputs` (variable structure groups one-hot columns, e.g. "1-4" style
   like psa_defs.txt).
2. Extend `tests/tools/` fixtures for the new paths (same --bless pattern).
3. Write `docs/datasets/bank-marketing/WALKTHROUGH.md`: bank.csv → mkdataset.py
   command → load in neuron (N inputs/1 output) → train (logistic AND SimpleProp,
   --seed for reproducibility) → read the ROC report incl. 95% CI. Verify every step
   by actually running it; put the real commands + real output excerpts in the doc.
4. Consider importing `distro/data/lowbwt2-2*` as a third sample dataset
   (Hosmer-Lemeshow low-birth-weight classic; the betas file is a reference-coefficient
   check for logistic regression). Optional, cheap.

### Phase 2 — Model deployment: `tools/neuron2web.py` — **DONE 2026-07-13** (see status)
The neuron2html.pl idea reborn: pick up a trained network and deploy it as a
calculator anyone can open in a browser. Lives in `tools/` (Craig's call, and right:
it's tooling around the engine, stdlib-only Python, no engine changes).
1. **Inputs (the same trio as neuron2html.pl):**
   - the saved network file (menu "Save the network"; SimpleProp/BareProp/BackProp/
     Logistic — parse the type from line 1 exactly like the engine's loader);
   - the scaling-factors file (menu "Save scaling factors"; format:
     `x_norm = S*(x - x_min) + lbound`, one S row + one x_min row) — required
     because deployed inputs arrive in natural units;
   - a **label spec** telling the page how to present inputs/output. Format = the
     legacy tag spec (`distro/scripts/perl-html_creator_spec.txt`, manifest ch. 11):
     `N %% name %% units`, `C %% name %% choice1 %% choice2...`, `B %% name %%
     true-label %% false-label`, `K %%` prefix for missing-indicator pairs,
     `O %% 0-label %% 1-label`, `R %% odds label`. Extension for --refcat grooming:
     a `*`-marked choice is the reference level (no column). The tool validates that
     the spec's expanded column count equals the network's input count and says
     exactly what mismatches if not.
2. **Output: one self-contained HTML file** (inline CSS/JS, zero external requests):
   form generated from the spec (numeric fields with units, dropdowns, yes/no,
   "unknown" toggles for K pairs), JS applies the scaling factors and runs the
   forward pass (activation functions must match function_defs.h exactly), displays
   probability + odds and the outcome labels. Works from file:// — deployment can be
   "email the file".
3. **`--serve` flag**: stdlib `http.server` on 127.0.0.1, **port 0** (OS-assigned,
   same decision as the GUI), print the URL, open the browser via `webbrowser`.
   No non-stdlib anything.
4. **Verification**: the tool re-implements the forward pass in Python; a fixture
   test deploys a committed network (e.g. tests/oracle/xor_net.txt + a tiny spec)
   and asserts the Python-computed outputs against engine eta-0 output; the
   generated HTML is byte-compared (--bless pattern). The JS and Python paths are
   generated from the same coefficients, so testing Python + embedding verified
   constants covers the page without needing node in CI.
5. **Documentation**: `docs/deploy.md` — spec-format reference (adapted from the
   perl spec + manifest ch. 11), end-to-end example, and a WALKTHROUGH.md addendum
   deploying the trained bank model (groom → train → save → deploy, the full arc).

### Phase 3 — Engine/UI decoupling survey — **DONE 2026-07-13** (see status)
1. Inventory where the engine prints directly to cout vs. takes an ostream&
   (dataset.cpp report methods already take ostream&; Iterative/train() progress
   printing is the main cout offender) and where it reads via util::ask*.
2. Route engine-core output through ostream& parameters (or a settable stream) so the
   server can capture reports into strings. Menus/driver keep cout. Golden transcripts
   must stay byte-identical — they are the refactor's safety net.
3. Port ROCx/ROCy curve-point capture from `code/C++/msvc/roc/roc 2005/twoset.cpp`
   (~10 lines in the trapezoid walk) + expose a getter. Output-neutral (goldens/oracle
   unaffected). Needed for GUI ROC plots.

### Phase 4 — Embedded web GUI (`neuron --gui`) — **DONE 2026-07-13** (see status)
Decisions already made by Craig: embedded HTTP server IN the binary (no Python
server); bind 127.0.0.1 with **port 0 = OS-assigned free port** (no collisions, no
PORTS.md entry needed); loopback only (LAN exposure only ever as explicit opt-in
flag); browser is the display layer; CLI stays fully functional alongside.
1. Vendor cpp-httplib (single MIT header) under `third_party/`; CMake option.
2. `--gui` flag: start server on port 0, print the URL (http://127.0.0.1:<port>/),
   auto-open browser (open / xdg-open / start per platform).
3. Serve a single self-contained HTML/JS page embedded in the binary (CMake
   string-embed step). JSON endpoints: load dataset, configure model, train
   (serialized by one mutex — engine is single-threaded), fetch report + ROC curve
   (ROCx/ROCy from Phase 3).
4. Golden transcripts + oracle keep guarding the CLI; add a curl-based smoke test for
   the endpoints if practical in CI.
Longer-term idea (parked): compile engine to WebAssembly → GUI as a static page on
GitHub Pages, no install at all (the neuron2html philosophy at full fidelity).

### Backlog (unordered)
- Binormal CI: recover the a-b covariance on the fitexy path (removes the
  anti-conservative caveat in docs/roc_theory.md) or validate further by simulation.
- `getGoodData()` port from the roc app if something needs it.
- More grooming tools (dataset describe/summary; train/test split outside engine).
