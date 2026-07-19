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
  - **Caveat for publication — SUPERSEDED 2026-07-15, see the entry below and
    `docs/roc_theory.md`.** *(Kept for the record; do not act on it.)* It read: the
    delta interval omits the a–b covariance on the binned fitexy path and runs
    ≈0.56× narrow, so prefer the trapezoidal CI and recover the covariance term.
    **Both prescriptions were wrong.** The covariance term is not the defect — the
    delta method assumes independent z-ROC points, and neuron's are cumulated from
    one sample (Wickens pp. 87-88); the SE is ~5× narrow, not 0.56×, and tracks bin
    count rather than n. And "prefer the trapezoidal" conflates interval with
    estimate: its CI is sound, but Wickens holds the trapezoidal *area* negatively
    biased and A_z primary (pp. 70-72).

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

- **2026-07-15 (later) — ROC inference audited against Wickens; delta method retired.**
  Craig supplied photographs of Wickens ch. 3, 4, 5, and 11; every ROC interval was
  traced to the page. **The full account + Methods-section language + a page-level
  citation table now live in `docs/roc_theory.md` — read that before touching ROC
  code or quoting an interval.** Headline findings:
  1. **neuron's data is the RATING case, not the bias case.** Wickens gets several
     operating points either from independent bias conditions (Fig 4.1 p. 61 — each
     point its own sample, points independent) or from one condition with graded
     responses (§5.2 pp. 85–87 — points cumulated from the same observations, **not**
     independent). A continuous classifier score swept over thresholds is the latter.
     He ships two programs for this reason (FitRoc vs FitRating) and says to use the
     right one (§5.3 p. 88). neuron computed rating data with a bias-style fit.
  2. **The point estimate is faithful — keep it.** Wickens fits the same line to the
     same cumulated z-points by eye (Ex 5.1 p. 89: zH = 0.735zF + 0.974 → μ̂ₛ=1.325,
     σ̂ₛ=1.360, Az=0.784 p. 90) and carries those estimates into his own GOF test
     (Ex 11.8 p. 214). `getStatROCarea()` is his Eq 4.7 (p. 68).
  3. **The delta-method CI was mis-specified, not merely imprecise.** It assumed
     independent points (contradicted, pp. 87–88); it was symmetric when Wickens says
     estimate±z·se is only valid for constant SE and Az's SE varies (§11.4 pp. 206–207);
     and analytic/multinomial intervals run short under overdispersion anyway (§11.2
     p. 201, p. 210). **Wickens offers no analytic SE for a least-squares z-ROC line** —
     his closed forms stop where parameters = data points (p. 205) and he routes the
     rest to ML, which supplies the SEs (§3.6 p. 57; Dorfman & Alf, ref. notes p. 58).
     Measured: delta SE ≈5× too narrow, and it tracked **bin count** not n.
  4. **Bootstrap is right, and was Craig's original intent** (Fable built delta+H-M
     instead; commit d9a3e73 says so in its own body: "Both analytic, not bootstrap").
     Resampling cases reproduces the cumulation dependence — exactly the accommodation
     §5.3 p. 88 demands — and the percentile interval is asymmetric by construction,
     delivering free what Wickens' endpoint iteration (p. 207) approximates.
  5. **The within-bin SD error bars have NO source in Wickens.** Table 5.3 (p. 90) has
     no error-bar column; Ex 5.1 fits by eye. They were invented to feed fitexy, and
     because bins are chunks of a sorted monotone sequence they measure *bin width* —
     hence SE∝bin count, and zero-SD bins on flat ROC runs. *(This entry originally
     continued "→ degenerate fitexy weights, NaN χ², `gammq` throws". **That chain was
     wrong** — see the evening entry above: `chixy` guards zero weights, and the NaN came
     from `Population::var()`. The zero-SD bins are real; what they broke was not.)*
     Wickens' own binomial z error bar (Eq 11.2+11.3 p. 202: σ²_z ≈ p(1−p)/N ÷ φ²(z))
     is the fix, and it makes binning unnecessary.
  6. **fitexy (2-axis) is Wickens' own** (p. 56: y-only regression is unsatisfactory;
     regression-to-mean flattens b, overestimating σ̂ₛ). Keep it.
  7. **MLE demoted from "destination" to backlog.** Wickens routes to ML *because ML
     gives the SEs* — a need the bootstrap already meets. ML stays better (efficiency,
     no arbitrary categorisation, publication standard) but is no longer urgent.
  Wickens' supplementary programs are still live at `twickens.bol.ucla.edu/sdt.htm`
  (HTTP only — WebFetch forces HTTPS and fails; use curl. Last modified 2002-04-11).

- **2026-07-15 (evening) — ROADMAP 3 Phase 1 finished; legacy bugs #6 and #7 found.**
  The plan said "make the χ² p-value non-fatal, do this first." **Measurement said the
  plan was wrong**, and following it literally would have shipped NaN areas. Instrumenting
  before editing is the only reason it didn't. What was actually true:
  1. **Legacy bug #6 — `Population::var()` returned NaN for identical values.** It used
     the one-pass `(Σx² − n·x̄²)/(n−1)`; for identical values those terms agree to within
     roundoff and the result lands at ±1e-16 with a **random sign**. Negative → `std()`
     = `sqrt(negative)` = NaN → poisons the zROC fit → the `gammq` throw everyone had
     been shrugging at for ~20 years. **Positive → a fake error bar of ~3e-08**, i.e. a
     point weighted ~1e15 in a fit that reported no error at all — so *passing* fits were
     silently distorted too. Reached from the binning: a bin covering a flat run of the
     empirical ROC holds identical z values. Fixed with the two-pass form (returns exact
     zero, which `chixy` already handles via `ww==0 → BIG`). Measured: 800/4000 binnings
     failed → **0**; 63% of resamples discarded → **0**; on real low-birth-weight data
     `1448 resamples, 552 failed` → **2000, 0 failed**. **`gammq` was never the disease
     — it was the immune system.** It threw on a NaN that was already there, and it was
     the only thing preventing a NaN Az from being printed. Suppressing it, as planned,
     would have been strictly worse than the bug.
  2. **Legacy bug #7 — the GUI fabricated an Az of 0.** On data too small for a binormal
     fit (`goodData < calcThresh`), `searchROC`'s loop never executes, and the panel
     published the zero-initialized `ROCfit`: `{"az":0,...,"nBins":0}` — "perfectly
     anti-predictive" — **while the report beside it correctly said "Cannot calculate ROC
     statistically."** The panel/report divergence the 2026-07-15 reconciliation was
     supposed to end, in a second place. **smoke.sh was asserting the fabrication's
     presence** (`grep '"bestP":'` passed on the zeros). Now `null`, and smoke asserts
     that, plus real fits on a set big enough to have them.
  3. Also fixed: one failing binning used to set `searchErrorFlag` and thereby suppress
     the **entire** report (including the seven binnings that worked) and discard the whole
     bootstrap resample. `getROCsearchFailed()` now means "no binning yielded an area".
     `ROCfit.valid` added; a NaN p can't win "best p"; best-p and best-AUC intervals carry
     separate failure counts.
  4. The delta method is **gone** (`azSE`, `statAzSE`, `getStatAzSE`, `ROCfit.se`) — it
     was still being displayed in the GUI panel long after being removed from the report.
  **Lesson, recorded because it generalizes:** the roadmap item was written from a
  plausible causal story ("zero-SD bins → degenerate fitexy weights → non-finite χ²") that
  was wrong in every link — there were **zero** zero-SD bins, `chixy` *guards* zero
  weights, and the NaN came from the variance. A doc that names a mechanism is still a
  hypothesis. Measure first; a 20-line probe settled in minutes what a session of reasoning
  had gotten backwards.

- **2026-07-15 (close of day) — ROADMAP 3 is DONE bar the backlog; the ROC work is
  finished.** Phases 1 and 2 both landed today (see the entries above and the roadmap).
  Where it ends up: every ROC number the engine prints traces to a page in Wickens, the
  binormal Az is validated against his own published worked example to **±0.0001**
  (`check_wickens`, Table 5.1 → 0.7839 vs his 0.784), the interval is a stratified
  bootstrap calibrated two independent ways (Hanley-McNeil to ~3%, simulation to ~1%),
  and the arbitrary binning that used to move Az by 0.011 is gone along with the whole
  search/best-p/best-AUC apparatus. **Seven legacy bugs** found and fixed across the
  reanimation, two of them today (#6 `Population::var()` NaN, #7 the GUI's fabricated
  Az of 0). What remains is Phase 3 (Dorfman-Alf ML) — genuinely backlog, with its
  rationale rewritten after Craig asked whether it was now dead; the surviving reasons
  (degeneracy/PROPROC, efficiency, a meaningful GOF, the publication standard) were
  never written down until he asked.
  **The methodological residue, which outlives the ROC work:** three plan steps and one
  causal chain in this file proved hollow today, all written from *"the invariants cover
  this"*, and a rewritten confidence interval once landed with all five invariants green
  and none of them executing the code. That is now **rules + a mechanism**, not a story
  (see "Standing rules" at the top, and the coverage list at the foot of
  `run_golden.sh`) — because the stories written here decayed within hours, which is
  precisely the evidence for not trusting stories. **Next thread: ROADMAP 2 Phase 1b**
  (async training + realtime error graph; starts with `screenPtr` → `thread_local`). Its
  plan makes concrete claims — `train()` has no hook, `screenPtr` is a process-global,
  `setHidden` is destructive — that were verified by exploration on 2026-07-14 and are
  probably sound. Rule 3 applies anyway: measure the specific claim before building on it.

- **2026-07-16 (morning) — post-ROC audit: docs destaled, legacy bug #8 found and
  fixed.** Craig asked for a documentation-consistency pass and a bug pass over
  yesterday's work before starting ROADMAP 2 Phase 1b. Results:
  1. **Legacy bug #8 — `TwoSet::HLX2calc()` segfaulted on any dataset over 10,000
     rows.** It copied the guesses into fixed `double[10000]` stack arrays with no
     bounds check (since 2004). Proven by measurement, not inspection: a synthetic
     12,000-row logistic session crashed with `EXC_BAD_ACCESS` in `HLX2calc` (lldb
     backtrace on the store into `aHL`); under 10,000 rows it worked. The walkthrough's
     bank data never tripped it (3,391 training rows — under the line). Fixed by sizing
     the arrays to the data (`vector<double>`), byte-identical below the line. New
     ctest `hosmer_lemeshow_large` (`tests/twoset/check_hl.cpp`), **verified to
     segfault against the pre-fix code** per standing rule 2. (Noted while in there,
     NOT changed because the oracle pins the behavior: HLX2calc's χ² accumulates
     across the NG=10..4 group-count loop without resetting, recomputes the last group
     inside the group loop, and takes the LARGEST p over group counts — statistically
     odd, but legacy-frozen; and its selection sort is O(n²). Craig should rule on
     whether the H-L statistic itself deserves a ROADMAP item.)
  2. **The bootstrap has a scale cliff, and B is not user-settable.** Measured: the
     post-training report costs ~20 s on the 3,391-row bank set (two 2,000-resample
     bootstraps), minutes by 12,000 rows — cost grows ~quadratically (per-resample
     sweep is O(distinct·n)). `setBootstrapResamples` exists in the engine but no CLI
     menu or GUI parameter reaches it. Not changed (any default change moves goldens);
     AGENTS.md now warns agents that the post-training pause is the bootstrap, not a
     hang. **Open question for Craig:** expose B (menu 13 / GUI field), scale it with
     n, or leave it.
  3. **Docs destaled** (everything the ROADMAP 3 cleanup overtook): `roc_theory.md`
     (GOF-gating section no longer describes the removed bin search; "until Phase 2
     lands" disclosure bullet; "what is tested" section; Metz reference row; ROCx/ROCy
     provenance), bank `WALKTHROUGH.md` **re-run for real** (§4/§5 transcripts now show
     the bootstrap CI and the statistical fit that the old `ITMAX too small in gcf`
     failure used to suppress — the fixed estimator now succeeds on the SimpleProp run
     where 2026-07-13's binned fit failed; Az values: logistic test 0.873922, SimpleProp
     test 0.871037), §6 rewritten around the single fit + bootstrap, ROADMAP 2's stats
     JSON shape corrected to the shipped single-`binormal` object, stale comments in
     twoset.{h,cpp}/check_az.cpp. AGENTS.md cost guidance updated (see 2).
  4. Gates all green after the fix: zero-warning build, goldens byte-identical, 4/4
     ctest (incl. the new one), smoke, verify_oracle numerically identical.
  Phase 1b remains the next thread and its plan claims remain unverified — measure
  `train()`/`screenPtr`/`setHidden` before building, per rule 3.

- **2026-07-16 (afternoon) — ROADMAP 2 Phase 1b DONE: async training + realtime error
  graph.** The plan's claims were measured first, per rule 3: `train()` had no hook
  (four break sites, all print-then-break — verified by reading iterative.cpp:99),
  `screenPtr` was a process-global (utility.cpp:376, not the plan's :355 — drifted but
  true), `setHidden` deferred to Phase 4. What landed, in prove-fail order:
  1. **`screenPtr` → `thread_local`.** The two-thread assertions were added to
     tests/capture FIRST and **watched fail against the global** ("redirection in other
     threads left the main thread's screen alone"), then the one-word fix made them
     pass. Rule now in the code comment: any new engine thread must set its own screen
     redirection first — it starts at cout, not at the spawner's stream.
  2. **`Iterative::Observer` + `StopReason`.** onIteration(iteration, setError) called
     at the BOTTOM of the train loop after every existing stop check; false = stop,
     falling through to the normal epilogue (a cancelled run is a completed run).
     `copy()` nulls the observer (clones must not drive their original's GUI buffers).
     StopReason set at every loop exit; printed output unchanged — goldens
     byte-identical throughout. New "Training was stopped by request." line prints
     only on observer-driven runs.
  3. **`Network::sampleTestError(stride)`** — mirrors reportAccuracy's 1-output test
     loop but deliberately never writes the TwoSet guesses (a mid-run sample must not
     invalidate cached statistics).
  4. **gui.cpp `TrainJob`**: worker thread owns the engine, `job.running` gates every
     other engine-touching endpoint with **HTTP 409 + `"busy":true`** (no queueing);
     `POST /api/train` blocking by default, `async=1` returns at once;
     `GET /api/train/status` (decimated series: ≥250 ms wall-clock samples, cap 2000
     points, halve+double-stride); `POST /api/train/stop`. `stopReason` string on every
     train result (blocking too — additive). **En passant: `/api/load` gained
     `discrete=0`** (continuous outcomes) because the async smoke needed a regression
     set and the GUI turned out unable to load one at all; raw+continuous requires
     fraction=0 (stratification needs classes), pre-split pairs load via mode=train.
  5. **Page**: realtime log-y error-vs-iteration canvas (train + sampled test, dataviz
     skill palette — slots 1/2 validated, ROC plot repainted to match so color follows
     the entity), Train↔Stop toggle, 400 ms poll, crosshair tooltip, and an honest
     "computing the statistical report… (ROC bootstrap)" status when iterations stop
     advancing but the run is in its epilogue — on bank-size data that phase is ~25 s
     of otherwise-suspicious silence.
  6. **Verified**: smoke extended (async start → running:true, 409 busy mid-run,
     stop → `stopReason:"cancelled"` with full stats, second async run to natural
     completion) and green on the new binary; **live click-through in Chrome on the
     bank data** — chart streams during a 20k-iteration logistic run, Stop cancels
     within one iteration (series froze at 23,150), the cancelled run still delivered
     the full ROC/stats panels, zero page JS errors. All gates: zero-warning build,
     goldens byte-identical, 4/4 ctest, smoke, verify_oracle identical.
  Next in ROADMAP 2: Phase 2 (auto algorithm selection).

- **2026-07-16 (late day) — back into the class layer; the bootstrap scale cliff is
  GONE.** Craig asked whether new engine code was using his Matrix/vector_ops/Population
  layer as designed (matrix notation ↔ the page, bounds safety) or layering scalar code
  on top. Honest audit: storage/moments yes, algorithms no — and the inherited scalar
  idiom was exactly where the morning's performance problem lived. What landed:
  1. **The ROC sweep is now sort-once + cumulate** (`TwoSet::operatingPoints()`, shared
     by `getStatROCarea` and `countGoodData`): every operating point is a cumulative
     count over the score-sorted exemplars — which is literally how Wickens builds his
     Table 5.2 — replacing the per-threshold O(n) recount that made each curve O(n²)
     and the bootstrap 2000× that. **Equivalence proven at the bit level**: identical
     F/H arithmetic (same integer divisions), goldens byte-identical including
     `binormal_seed42`, check_wickens still 0.7839, bank transcript diff-clean.
     **Measured**: bank logistic session 33.2 s → 14.0 s (report ~20 s → ~2 s); the
     12,000-row synthetic session that spent minutes in the bootstrap this morning
     (and then segfaulted at H-L) now runs end-to-end in **5.1 s**. The morning entry's
     open question about exposing/scaling bootstrap B is largely **mooted** — the cost
     was never inherent to B=2000, it was the sweep.
  2. **`Matrix::includerows`** — the row-gather primitive (repeats allowed, any order,
     unconditional `BoundsViolation` throw like `operator()` since release builds strip
     asserts). A bootstrap resample is now one expression, `A.includerows( pick )`, with
     the same RNG consumption order (goldens prove it). New ctest `matrix_gather`,
     **proven non-vacuous by sabotage** (gather ignoring positions → test fails).
     Found en passant: **`includecols` had carried `excludecols`' dimension contract
     since 2.x** (asserted/allocated ncols−pos.size() columns while copying pos.size())
     — dormant, zero callers, fixed and pinned by the new test rather than numbered as
     a live bug.
  3. **Standing rule 4 added** (engine code lives in the class layer; extend the layer
     when it lacks a primitive; stated reason required where scalar code is right).
     AGENTS.md + walkthrough cost notes re-measured and updated in the same commit.
  4. **HLX2calc layer-lift (Craig's item 3) assessed and DEFERRED.** Measured in
     isolation: the selection sort costs 17 ms at bank size, 0.6 s at 34k rows, 3.8 s
     at 100k — now the engine's last quadratic hot spot, but sub-second at every
     dataset the repo has ever seen. Memory safety is already fixed; a cosmetic lift
     would risk tie-order changes against the oracle-pinned p-value for no statistical
     gain; and the function's real defects are statistical (χ² accumulates across the
     NG=10..4 loop without reset, recomputes the last group inside the group loop,
     reports the LARGEST p over group counts) — polishing it before Craig rules on the
     statistic means doing the work twice. **Recommendation: fold the lift into an
     "H-L statistic review" roadmap item, where outputs are expected to change anyway.**

- **2026-07-16 (evening) — legacy bug #9: the Hosmer-Lemeshow test was not the
  Hosmer-Lemeshow test.** Craig flagged the χ² accumulation and largest-p selection;
  forensics (a Python reimplementation validated against the engine's printed output to
  five decimals on three datasets — `scratchpad hl_forensics.py`, results recorded here)
  found FIVE compounding defects in the 2004 implementation: a tail-complement
  pseudo-group term added at every group; the χ² never reset across a group-count scan
  from 10 down to 4; observed- instead of expected-based variance denominators (with
  divide-by-0.5 fallbacks); the most favorable p over the seven scans reported; and
  df = NG−2 applied to a sum with ~2× that many terms. **Measured: it rejected a TRUE
  model 54-56% of the time at α = 0.05** (median p 0.035, 2000 sims, n = 500 — confirmed
  independently at 55.75% by the C++ test against the old code), and on the repo's own
  bank walkthrough it **flipped a conclusion**: test-set H-L p = 0.0014 where the true
  statistic gives 0.18 (no evidence of misfit). It hid on well-fitted training data
  because MLE residuals sum to zero, which kills the junk tail terms — benign exactly
  where the reference checks live. **Fix: the textbook Ĉ** (H&L 2nd ed. §5.2.2), g = 10
  deciles of risk FIXED by Craig's decision (the canonical choice, the one every
  published reference value uses; scanning group counts for a favorable p was part of
  the disease), E-based denominators, χ²(8), stable_sort so tie order is deterministic
  cross-platform, honest refusal when n < g. In the class layer per rule 4 — which also
  removed the O(n²) selection sort en passant (getHLX2 at 100k rows: 3.8 s → 11 ms;
  nothing in the report is quadratic any more, and yesterday's deferred "H-L layer
  lift" item is mooted). Verified: `check_hl` gained a hand-computed Ĉ case (closed-form
  p = 0.98101184) and a true-model calibration guard (reject < 20%; new code 9.5%,
  theory ~11% for known-true probabilities) — **both watched fail against the old
  code**; engine H-L on the H&L reference fit now matches the independent Python
  textbook value (0.997925 vs 0.9979); goldens re-blessed (the diff is ONLY the seven
  H-L lines — lbw train 0.6392→0.7468, test 0.4209→0.7332, and the small-set refusals,
  where the old code was already failing via the gcf throw); oracle H-L line excluded
  with 3.0's refusal asserted (the KScalc pattern).
  **Pearson forensics (same session):** `PKX2calc` was unsalvageable as a p-value —
  denominator g(1−g/n) ≈ g instead of g(1−g), so the statistic's expectation under a
  TRUE model was ≈ n − Σĝ (measured: lbw 127.9 vs predicted 130.0), referred to a
  hardcoded χ²(2) → **p → 0 as n grows regardless of fit** (true-model median p: 0.14 at
  n=10, 8e-5 at n=50, 1e-44 at n=500). Deeper: even computed correctly, individual-level
  Pearson X² has no valid χ² reference with continuous covariates (cells of size 1, df
  needs the model's parameter count, which TwoSet doesn't know) — the problem the H-L
  test was invented to solve. **Craig's decision: print the correct X² with n and NO p**
  ("Pearson X2 = ... (n = ...; no valid p at the individual level - see
  Hosmer-Lemeshow)"), the explanation kept in the docs (walkthrough §6 reading guide +
  the PKX2calc/getPearsonX2 comments carry the full H&L citation and the E[X²] ≈ n
  reading rule). Landed 2026-07-16 evening: correct squared-Pearson-residual sum,
  saturated-model guard (a contradicted certain prediction → X² = inf), GUI JSON
  `pearsonP` → `pearsonX2` with the page labeling it as a statistic. `check_hl` gained a
  hand-exact case (20 unit residuals → X² = 20 exactly — cannot pass against the old
  getter, which returned a p ∈ [0,1]). Goldens re-blessed (Pearson lines only), and the
  diff **corroborates the implementation on real data**: lbw train X² = 142.6 on
  n = 142, test 45.5 on n = 47 — the E[X²] ≈ n signature of a well-fitted model — and
  the XOR null-model case gives X² = 4 on n = 4 exactly. Oracle: Pearson excluded, 3.0's
  line pinned (X² = 0.0103 on the near-perfectly-fitted XOR net — near-zero residuals,
  the known-correct answer).

- **2026-07-16 (night) — ROADMAP 2 Phase 2 DONE: `algorithm=auto`. Its first diverged
  probe found a bug I had shipped 24 hours earlier.** What landed:
  1. **`src/netclone.{h,cpp}`** — RegressNet's typeid clone dispatch generalized to
     `cloneNetwork()`; RegressNet refactored onto it (pure refactor, regress golden
     byte-identical).
  2. **`src/autoalgo.{h,cpp}` `pick()`** — three clones from identical weights, 750 ms
     wall-clock each via a `BudgetObserver` (Iterative::copy nulls observers, so clones
     never drive the GUI's), batch/epoch forced for CGD/Shanno, probes leave no trace
     (setLastop/setHistory off, **bootstrap disabled on probe clones' TwoSets** — else
     each probe's discarded report would cost more than its budget), probe reports
     discarded, one decision summary + per-probe stopReason printed to the caller's
     screen. Winner = lowest finite error, strict comparison keeps the simpler
     algorithm on ties; a diverged probe is a result, not a failure; the winning CLONE
     is adopted (probe progress kept) and trains on to the user's maxiter.
  3. **GUI**: `algorithm=auto` on `/api/train` (blocking and async), `autoAlgo` JSON
     block, page dropdown option. The async worker was restructured to re-derive every
     engine pointer inside the job — adoption REPLACES modelPtr, so the 1b-era lambda
     captures would have dangled; smoke covers async+auto specifically for this.
  4. **The regression the probes caught: yesterday's `operatingPoints()` looped forever
     on NaN guesses.** A diverged Shanno probe (error 1e63 → NaN guesses) hung its own
     epilogue: NaN != NaN, so the tie-group cumulation never consumed a NaN element —
     `i` never advanced, push_back forever (lldb sample: ROCarea → operatingPoints →
     memmove). The wall-clock budget couldn't save it (the hang is inside one call).
     Fix faithful to the old recount's semantics: non-finite scores never enter the
     sweep (a NaN guess compares false at every finite threshold) but stay in the class
     denominators. `check_az` gained a NaN-guesses case **verified to hang against
     yesterday's committed code** (12 s timeout kill) and pass now; HLX2calc's
     stable_sort comparator got the same NaN guard (NaNs last — comparing NaN in
     std::sort is UB, and a NaN group poisons E so the fit p honestly fails).
     Observed working after the fix: on lbw, canonical CONVERGES inside its 750 ms
     probe (grad_max, error at the MLE) while CGD/Shanno diverge to 1e5/1e63 — the
     selection picks canonical, exactly the bank-walkthrough lesson, automated.
  Gates: zero-warning build, goldens byte-identical, 5/5 ctest, smoke (incl.
  auto blocking + async), verify_oracle identical. **Next: ROADMAP 2 Phase 3**
  (plateau auto-stop).
  **Post-landing (same night):** Craig asked whether the selection is REPORTED —
  it was in the report and the JSON, but the page shows only the message, so the
  message now opens with "auto selected <name>; " (smoke asserts it). And the probes'
  chaos monkey kept paying: a **second NaN infinite loop**, nondeterministic this time
  (wall-clock probe boundaries → different divergence trajectories → hung 1 run in ~2,
  then not for 25). Sampled at `KScalc()`: the NR merge advances an index only on a
  `<=` comparison, and NaN fails BOTH directions — neither index moves. K-S now refuses
  honestly on NaN guesses ("guesses contain NaN (a diverged model?)" → the existing
  "Could not calculate" catch); `check_hl` gained the case, **verified to hang against
  the unguarded code**; the hang-hunt script then ran 25 clean attempts. Lesson worth
  keeping: the engine's statistics were written for guesses a converged model produces;
  the probes deliberately manufacture diverged ones, so every statistic they reach is
  getting its first NaN audit — two infinite loops so far (operatingPoints, KScalc),
  found by sampling actual hangs, not by reading for them.

- **2026-07-19 — ROADMAP 2 Phase 3 DONE: plateau auto-stop. The plan's central
  mechanism was built on a false premise, caught by measuring before building (rule 3).**
  The handoff sketch (and the roadmap) called for a `plateauPeriodHint()=df()` /
  `W_eff = max(window, 2·period)` scheme to average out the "period-df() sawtooth" that
  CGD/Shanno restarts supposedly produce. **Reading network.cpp first killed it:** the
  restart test is `( t == 0 ) || ( t == df() )` — a *one-time* equality, not
  `t % df() == 0`. In batch/epoch mode `engine()` gets `t = iteration`, so conjugacy
  restarts exactly once (at `iteration == df()`) and then accumulates unbroken. There is
  no period-df() sawtooth to average out, and auto-step-size (the other oscillation
  source) runs an inner η search *every* iteration, so it is aperiodic too. **Dropped the
  hint/`W_eff` machinery entirely** — the detector is a plain two-window moving-average
  comparison (`src/plateau.h`, header-only, engine-free), which smooths oscillation up to
  ~window-scale on its own without any period assumption. Simpler, and not fiction.
  1. **`src/plateau.h` `PlateauDetector(window, tol, patience)`** — holds the last
     2·window errors as two adjacent windows, strikes when window-over-window relative
     improvement `< tol` (rising counts too → crude overlearning guard), `patience`
     strikes fire. **NaN decided at design time (the other rule-3 item):** a non-finite
     error is a strike *outright* and is kept OUT of the windows so it can't poison the
     means — the exact trap that hung the autoalgo probes twice. So a diverged run
     plateau-stops instead of sailing to maxiter.
  2. **`tests/plateau/check_plateau.cpp`** (ctest `plateau_detector`) — five synthetic
     traces, **both directions proven non-vacuous by sabotage** per rule 2: forcing
     "never strike on a finite plateau" fails exactly the two fire-detection assertions
     (decay-then-flat, flat sawtooth); making non-finite a reset instead of a strike
     fails exactly the two NaN assertions. Neither sabotage touches the other's tests.
  3. **Engine wiring in `Iterative`** — `setAutoStop(flag,tol,win)` + getters,
     `STOP_PLATEAU`, checked in the train loop AFTER gradMax (true convergence wins) and
     before the observer. **Default OFF**, and `update()` is called only when the flag is
     on, so a run without auto-stop is bit-identical — goldens byte-identical, oracle
     numerically identical, both verified. `copy()` carries the config so the autoalgo
     winner clone keeps it.
  4. **GUI**: `POST /api/train` gains `autostop=1` (+ `autostop_tol` default 1e-4,
     `autostop_window` default 100), validated in the handler (asserts vanish in release);
     `stopReason:"plateau"` via a new `stopReasonName` case; page "Auto-stop on plateau"
     checkbox + an honest "Stopped early — the training error plateaued." chart note.
     Smoke asserts the fire, a **control** (same seed, no autostop → `grad_max`, proving
     it's a real stop not a relabel), and both validation rejections.
  5. **Confirmed-and-left**: the pre-existing gradMax-staleness quirk (`getGradMax()` only
     refreshes inside the print block, so grad_max fires *on* a print iteration) is real
     and oracle/golden-pinned; the plateau detector does NOT share it (it sees the fresh
     `setError` every iteration). Not documented in AGENTS.md — it changes no recipe.
  Gates: zero-warning build, goldens byte-identical, 6/6 ctest (new plateau_detector),
  smoke (+ plateau block), verify_oracle identical, live curl drive of the plateau path
  incl. report text and the no-autostop control. **Not yet run: in-browser click of the
  checkbox** (needs the Chrome extension connected) — the endpoint it drives is fully
  verified and the JS change is one param. **Next: ROADMAP 2 Phase 4** (OBD hidden-layer
  sizing) — its plan claims (`setHidden` destructive, grow-with-zero-outgoing-weights =
  bit-identical forward pass, OBD needs a test set) are unverified; measure first.

- **2026-07-19 (later) — full GUI/CLI parity + standing rule 5. The GUI was a third of
  the CLI; now it is a superset, and a hard rule guards it.** Craig re-stated a rule that
  was spoken when the GUI began and never enforced: **everything in the CLI menu interface
  must also be in the GUI** (he had deliberately declined to add new GUI features back to
  the frozen CLI — "the GUI is the primary interface from now on" — which makes GUI≥CLI
  parity *more* important). A menu option with no GUI equivalent is a bug. Landed as
  **standing rule 5** + `docs/gui_cli_parity.md` (the enforcement artifact: every menu
  option ↔ GUI control ↔ API param, updated in the same commit as any menu/GUI change) +
  an AGENTS.md banner. Then closed every gap the audit found, engine→API→page→smoke, one
  commit per area, goldens byte-identical and oracle identical throughout (the GUI never
  touches the CLI transcripts):
  1. **Train panel** (994b25a): learning rate (manual η or automatic step size), weight
     decay + λ, batch/epoch, print counter, the four stopping conditions (lower error,
     change, error window, max gradient — joining max iterations), and the plateau
     tol/window promoted from hardcoded defaults into fields. Each param is applied ONLY
     when present in the request, so a bare train keeps the model's current value — which
     is exactly how "stop, then continue from the held weights with **new** parameters"
     works (Craig's core ask), and keeps every existing bare-train call byte-identical.
     Settings ride into the autoalgo probe clones via `Network::copy`/`Iterative::copy`.
  2. **Model panel** (24caafd): bias toggle (off → BareProp), multi-hidden-layer (comma
     list → BackProp), output error function (X-entropy refuses continuous data), the two
     logging toggles, and **load a saved network from a file** (upload → type read from
     line 1 → weights loaded → treated as trained).
  3. **DFA** (86c2261): `POST /api/dfa` type=linear|quadratic — LDFA/QDFA as a standalone
     analysis (does NOT replace `modelPtr`), returning the report + ROC + stats via the
     same `jsonROCSeries`/`jsonStatsSet` helpers. The last CLI main-menu branch.
  4. **Audit log** (fb266d2): every user action (load/model/dfa/train/randomize/regress/
     stop/save) appended to **`neuron_actions.log`** with a timestamp and its exact param
     values, via `util::run_path` so it sits in the run directory beside the data,
     `neuron.log`, and `model.txt` (Craig: store it locally with the run files). The other
     half — a self-describing run header — was **already** true (`Network::runHeader`
     prints algorithm/eta/weight-decay/batch, `Iterative::train` prints every stopping
     condition), so `neuron.log` alone already records what a run used.
  Verified: the "stop → continue from held weights with new params" and "randomize →
  restart" flows measured through the API; a live in-browser click-through of the plateau
  checkbox (from the Phase 3 thread) confirmed the page path. Gates green every commit.
  **The parity matrix now has zero gaps.** Rule 5 is the guard going forward.
  **Browser-verified (2026-07-19, later):** clicked through the new Model panel (bias off →
  BareProp, hidden `4,2` + LMS → BackProp), the Train panel (Change-limit field → the run
  came back `min_change` with the header recording the condition), DFA (Run DFA → report
  renders), and load-network (file injected → "loaded Binary logistic network"), zero page
  JS errors. Reflow note: switching model type hides the NN options and shifts fixed
  coordinates — use `find`/ref-based clicks (scroll- and reflow-proof), per
  [[neuron-gui-browser-testing]]. **Next: ROADMAP 2 Phase 4** (OBD) as before.

- **2026-07-19 (later) — DFA now has a real ROC AUC (graded discriminant score).** Craig
  asked whether DFA had a ROC AUC; it did not, in either interface. `LDFA`/`QDFA`
  `reportAccuracy` stored the **hard 0/1 class decision** (`d0 > d1` for LDFA, `d0 < d1`
  for QDFA) into the TwoSet, so the ROC had a single operating point → trapezoid area =
  (sens+spec)/2 and `binormal:null` ("too few distinct operating points"). Fix: store the
  **graded class-1 score** `sigmoidal()( margin )` — margin `d1−d0` for LDFA, `d0−d1` for
  QDFA (opposite convention) — which the engine's own `sigmoidal` functor maps to (0,1)
  with the 0.5 boundary `calculate()` already uses (`A(i,1) >= T`), so the classification
  is preserved exactly while the ROC sweep sees a real curve. Az is invariant to the
  monotonic map. **Proven by a controlled old-vs-new run on a fixed (unsplit) training
  set** (the seedless GUI split had masked it as noise): confusion **byte-identical**
  (4/125/5/55, sens 0.0678, spec 0.9615) while binormal went **null → Az 0.6530 (169
  points)** and trap 0.5147 → 0.6552. DFA is in **neither the goldens nor the oracle**, so
  no re-bless or exclusion was needed (a worry I'd raised that measurement dissolved).
  smoke asserts DFA's binormal is now a fittable `az` and **not** null (verified to fail
  against the pre-graded code). Gates green. Not a legacy *bug* — the classification was
  always right; DFA just never exposed a graded score for ROC.

## ROADMAP 3 (agreed with Craig 2026-07-15) — ROC inference

Rationale, citations, and Methods language: **`docs/roc_theory.md`**. Work in order.

### Phase 1 — land the interval (**DONE** 2026-07-15; two commits, see status entries)

**DONE (in `main`):**
- **Bootstrap CI** replaces the delta method in `statReport`: `TwoSet::bootstrapROC()`
  + `CI` struct + `getBestPci()`/`getBestAUCci()`/`getStatCi()`. Stratified within
  class (preserves n0/n1, so every resample has an ROC), B=2000 (`setBootstrapResamples`,
  0 disables), percentile interval, bootstrap SD reported as SE, **the whole procedure
  re-run per resample including the bin search** (Craig's call: choosing the binning is
  part of the procedure, so it belongs in the spread).
- **`util::i_resample()`** — an RNG stream *independent* of `d_random()`, so computing
  an interval can never perturb weight init or train/test splits. `set_seed()` seeds
  both (`seed ^ 0x9E3779B9`); clock-seeding likewise. **Rule: any new resampling must
  use this stream, never `i_random`.**
- Validated: bootstrap SE vs Hanley-McNeil on low-birth-weight = 0.0497/0.0522 (n=142)
  and 0.0844/0.0870 (n=47) — two methods, different assumptions, agreeing; and it scales
  with n, which the delta SE never did. Cost ≈1 s for 2×2000 resamples on 142 rows.

**ALSO DONE (second commit, 2026-07-15 evening — the four remaining items):**
1. **The 27% resample discard is gone: 2000/2000 on low-birth-weight.** Root cause was
   NOT the p-value — see the status entry below. `Population::var()` now uses the
   two-pass form; the χ² p-value is non-fatal anyway (§11.5 p. 217) as a guard;
   a failing binning no longer poisons the search, the report, or the resample.
2. **GUI**: `binormal` JSON carries each fit's bootstrap `ci` (lo/hi/se/resamples/
   failures); the panel shows it. The delta SE it used to display is **deleted**.
3. **`check_az` rewritten**: delta-SE calibration replaced by bootstrap calibration
   (ratio 1.00 vs the delta method's 0.4–2.5 window), plus the identical-values
   variance assertion and a tied-scores regression test. `azSE`/`statAzSE`/
   `getStatAzSE` **removed** — the delta method has no traces left in the engine.
4. **Coverage added where the hole was**: `check_az` now drives the binormal path incl.
   ties/degenerate binnings; `smoke.sh` asserts `binormal:null` on 4 exemplars and real
   fits+CI on low-birth-weight. **Both were verified to FAIL against the pre-fix
   binary** — they are not vacuous.
5. **THE GOLDEN HOLE IS CLOSED** — `tests/golden/binormal_seed42` (added before starting
   Phase 2, deliberately: see below). Low-birth-weight 189 rows → 142/47, seeded
   logistic, 0.9 s. Freezes the entire printed ROC report byte-for-byte — both searched
   fits + bin counts, χ², fit p, both bootstrap CIs, the H-M CI, and *both* the binned
   (train) and unbinned (test) paths. Resolution = the report's 6 decimals: a **0.01%
   change in Az fails it while xor_seed42 and regress_seed42 both still pass** —
   demonstrated by perturbing Az, not assumed. Cross-platform stability (fitexy/brent/
   gammq/erfc are more libm-dependent than anything the old goldens touch) is proven by
   CI, not by argument — if it ever flaps on one OS, that is the thing to reconsider.
- **Az did not move**: oracle numerically identical, goldens byte-identical, and
  low-birth-weight Az 0.618420/0.620688 before and after. Only the *interval* widened
  (0.513–0.711 → 0.513–0.721), which was the point.

### Phase 2 — fix the error bars (**DONE** 2026-07-15)
- **DONE — Replaced within-bin SD with σ²_z ≈ p(1−p)/N ÷ φ²(z)** (Wickens Eq 11.2+11.3
  p. 202), and stopped binning: `getStatROCarea()` now sweeps **distinct** thresholds
  (a tied score repeated per exemplar is the same operating point) and fits those points
  directly with fitexy. **Result on Wickens' own published data: Az = 0.7839 vs his
  0.784** — was 0.7785/0.7893 depending on the binning. The bin count is now **inert**
  (asserted as a null test in check_wickens), `AZ_TOL` ratcheted 0.012 → **0.002**, and
  the zero-SD/`BIG`/FMA fragility is gone with it (no error bar is zero any more).
  Report line "Bin size = X Number bins = Y" / "WARNING: DATA WAS NOT BINNED" →
  **"Operating points fitted = N"** (new `statPoints`), which is the number a reader
  actually needs. Goldens re-blessed: lbw train Az 0.618420/0.620688 → **0.620527**
  (127 points), test 0.675294 → 0.669981 (37 points); 18 lines moved, rest identical.
- **The oracle needed NO exclusion** — the plan predicted one ("Az changes → oracle needs
  a documented exclusion, same pattern as the KScalc fix"). **Wrong, for the same reason
  the "goldens re-bless" step was a no-op**: the oracle check has *0* "By statistical
  method" lines (its xor_net has 4 exemplars → "Cannot calculate ROC statistically"). It
  never tested this path. Only `binormal_seed42` — added the day before for exactly this
  — actually saw the change. Third time a plan step written from "the invariants cover
  this" has proved hollow; check what a test executes before believing it guards you.
- **Metz/LABROC ROC-corner categorisation turned out NOT to be needed** and is not
  implemented (the plan assumed it would be). Its purpose is to categorise for a
  *maximum-likelihood* fit; a least-squares line over the distinct operating points needs
  no categories at all. It returns with ML (Phase 3) or not at all.
- **Known and documented, not a regression:** the fit χ² is a weak diagnostic on
  continuous data — cumulated points scatter far less than their own marginal error
  bars, so χ² ≪ df and p rounds to 1.000 (lbw train: χ²=15.1, df=125). Wickens says the
  dependence invalidates it (p. 212). It *is* meaningful in the rating regime: Wickens'
  Table 5.1 gives χ²=1.93 on 3 df, p=0.587. The area never depends on it (§11.5 p. 217).
- **DONE — the dead machinery is gone** (cleanup commit). Removed: `searchROC` and the
  3..10 search (it was running 8 identical fits, and the bootstrap did that *per
  resample*), the best-p/best-AUC pair → **one `ROCfit` / one `CI`**, `nBins`/`binSize`/
  `binThresh`/`nBinFlag`/`binnedFlag`/`minBins`/`maxBins` and all their accessors,
  `searchFlag`, `searchErrorMsg`. New `getROCfit()` returns `valid=false` instead of
  throwing, so a caller that cannot have an answer says so rather than inventing one
  (that is legacy bug #7's class, closed structurally). GUI JSON: `binormal` is now one
  object (`az`/`p`/`chi2`/`points`/`ci`), not a `bestP`/`bestAUC` pair. **CLI submenu 13
  loses options 4/5/6** (search, min bins, max bins) — quit moves 7 → 4. ROADMAP 2 froze
  the menus, but "frozen" meant no *new* features; a control that silently does nothing
  is worse than a removed one.
- **Golden diff was report shape only, no numbers moving** — exactly as predicted.
  binormal_seed42: the duplicated "Searching for best p:"/"Searching for best AUC:"
  blocks collapse to one `By statistical method, ROC area = ...`; Az/CI/χ²/points
  byte-identical. xor + regress each lost 2 lines: the `WARNING: Maximum number of bins
  to be searched (10) exceeds data (0)` pair, a warning from a search that no longer
  exists.
- **The oracle DID need an exclusion after all — but not for the predicted reason.** The
  plan said "Az changes → oracle needs a documented exclusion". Az never reaches the
  oracle (0 statistical-method lines). It needs one because the *oracle* still prints
  that two-line maxBins warning and 3.0 no longer does. Right conclusion, wrong reason —
  documented in `verify_oracle.sh` alongside the KScalc and 95%-CI exclusions.
- **Drop fixed-count binning** — unnecessary once error bars are analytic. Categorise by
  the **corners of the empirical ROC** (Metz/LABROC: truth-state runs are the natural
  categorisation; flat runs carry no information). This removes the 3..10 search,
  `nBins`, and the best-p/best-AUC pair **together** — and obsoletes the two-fit stats
  panel added 2026-07-15 (that reconciliation was correct for the old design; it becomes
  moot, not wasted). Note the zero-SD/NaN failure class is **already gone** (fixed at
  `Population::var()`, evening of 2026-07-15) — Phase 2 no longer has to carry it, and
  the arbitrary binning, not a crash, is what remains to justify this work.
- **The literature acceptance test is BUILT** — `tests/binormal/check_wickens.cpp`
  (ctest `binormal_wickens`), landed 2026-07-15 before any Phase 2 code, and it already
  earns its keep. Wickens' Table 5.1 (p. 84; Olzak & Kramer rating data, 699 noise + 700
  signal, six categories) fed in as (known, rating-as-score) pairs. **What it found,
  running against the CURRENT estimator:**
  - **The operating points are EXACTLY Wickens' Table 5.3** (f = .762 .532 .335 .152
    .061, h = .933 .840 .746 .614 .420, all to 3 dp). The z-transform and ROC sweep are
    provably right — the defect is entirely downstream, in the fit.
  - **The binning is worth 0.011 of Az**: the 8 binnings span 0.7783–0.7893 around a
    published truth of 0.784. That is the size of the whole bootstrap SE (0.0142) —
    the arbitrary choice costs as much as the sampling noise. **This is Phase 2's target,
    now quantified against the literature instead of asserted.**
  - **BOTH selection criteria pick badly.** "Best p" → 5 bins → Az 0.7785 (−0.0055, one
    of the furthest); "best AUC" → 7 bins → Az 0.7893 (+0.0053, THE furthest) **on a fit
    whose p is 0.000** — maximising area selects for a bad fit by construction. The
    near-exact binning (8 bins, −0.0012) is one *neither* criterion chooses. So the
    best-p/best-AUC pair isn't merely redundant after Phase 2, it is actively harmful now.
  - **The fit is FMA-sensitive on tied data** (new, 2026-07-15): same source at -O0 vs
    -O3 gives Az 0.7789 vs 0.7793 at 4 bins, and at 10 bins one converges while the other
    throws `Root must be bracketed in zbrent`. Cause: binning 1399 exemplars holding 6
    distinct scores makes bins of identical z values → SD legitimately 0 → `chixy` weights
    them `BIG`=1e30 → the objective is dominated by 1e30 terms and brent balances on the
    last bits. (Returning exact 0 from `var()` was still right — NaN was worse — but it
    routes into a pathological weighting.) **Caveat this puts on `binormal_seed42`:** it
    is byte-stable on all 3 OSes today, but this path is more libm/FMA-exposed than
    anything the older goldens touch. If it ever flaps on a compiler bump, this is why.
  - Also surfaced: `zbrent` bracketing failure is a **third** distinct failure mode
    (after gammq and "Zero std dev"). Non-fatal now — the searchROC fix skips it and the
    other binnings still report; before 2026-07-15 it would have killed the whole report.
  - **AZ_TOL is a RATCHET**: 0.012 today (loose enough to admit whichever binning is
    chosen). **Phase 2 must tighten it to 0.002** — if it cannot hold that on published
    data Wickens analysed by hand, it has not worked.
  - **NOT covered, deliberately** (the plan over-specified this): Ex 11.8 X²=5.93/35.85
    (p. 214) is a *multinomial* GOF over rating categories comparing two fitted models —
    the engine computes no such statistic (its χ² is fitexy's line-fit χ², a different
    quantity); it arrives with ML (Phase 3). Ex 11.1 se(Âz)=0.042 (p. 204) is the analytic
    *single-point* SE (Eq 11.7, parameters = data points) — a path neuron doesn't have,
    and whose analytic SE we deliberately removed. Both are documented in the test.
- Az changes → **oracle needs a documented exclusion** (same pattern as the KScalc fix)
  and **`binormal_seed42` re-bless — read that diff, it is the whole record of what
  moved**. *(This line used to say just "goldens re-bless", which was a **no-op**: no
  golden printed a single ROC line. The re-bless step looked like a safety check and was
  not one. `binormal_seed42` was added 2026-07-15, before starting Phase 2, precisely so
  that this step means something — a golden blessed AFTER a change that moves Az would
  only enshrine whatever Phase 2 produced, bugs included. It was frozen while the current
  Az is independently corroborated from two directions: the oracle says it matches the
  legacy engine bit-for-bit, and check_az says it matches SDT theory.)*

### Phase 3 (backlog, not scheduled) — Dorfman–Alf ML

**Rationale rewritten 2026-07-15 — Craig asked whether Phase 3 was now irrelevant and
should be deleted. As it was written, yes: both its stated reasons were dead.** It said
ML "dissolves the categorisation question entirely" (Phase 2 just did that *without* ML)
and that it was deferred because "its stated advantage — supplying the SEs — is already
met by the bootstrap" (still true). Neither is a reason to keep an item. The reasons that
actually survive were never written down, so here they are:

1. **Degeneracy / the "proper" binormal** (Metz & Pan 1999) — the real one, and entirely
   unaddressed. A binormal curve with slope b ≠ 1 is *improper*: it crosses the chance
   line and hooks. Ours does. Measured on Wickens' Table 5.1 fit (a=0.970, b=0.729): H
   drops below F beyond **F ≈ 0.9999** — real, but far outside any observed point (his
   largest f is 0.762), so it does not bite on well-behaved data. It bites when b is far
   from 1, or the data are sparse/lopsided. Least squares has **no guard whatsoever**;
   PROPROC exists for this.
2. **Efficiency** — ML is the efficient estimator. Least squares with marginal-variance
   weights on cumulated (correlated) points is *consistent* but not efficient. The
   bootstrap gives us honest intervals around a slightly noisier point estimate.
3. **It would make the fit χ² mean something.** Ours is uninterpretable on continuous
   data (χ² ≪ df, p → 1.000; see the Phase 2 notes) — and the GUI panel prints
   "fit p 1.000" with no caveat, which reads as "perfect fit" to anyone who has not read
   `docs/roc_theory.md`. ML with a proper categorisation gives Wickens' own multinomial
   GOF — **Example 11.8, X² = 5.93 unequal vs 35.85 equal (p. 214)** — which is exactly
   the acceptance-test item `check_wickens` had to document as un-implementable.
4. **Publication standard.** Metz's suite (ROCFIT/LABROC4/PROPROC/ROCKIT) is what medical
   ROC reviewers know; Swets (1996) surveys the implementations. `docs/roc_theory.md`
   already ships the Methods substitution paragraph for when this lands.

**Still not scheduled**, and that is deliberate: the current estimator is validated
against published data to ±0.0001, its interval is bootstrap-validated two ways, and the
degeneracy is confined to a region no dataset reaches. Phase 3 is a real improvement with
no live defect forcing it — which is the definition of backlog, not of dead.

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
  fp,fn}, acc, sens, spec, pvp, pvn, trap{area,se}, binormal{az,p,chi2,points,ci{lo,hi,
  se,resamples,failures}}|null, ks{d,p}, pearsonP, hlP}` + `logistic{condNumber,
  coefficients[{input,beta,se,waldP}]}`. *(This paragraph originally specified a
  `bestP`/`bestAUC` fit pair with bin counts; ROADMAP 3 Phase 2 removed the bin search
  before 1b started, so the shipped Phase 1a shape is the single fit above.)* **The
  binormal block quotes the report's own fit (`getROCfit()`), never one of its own,
  and is `null` when no fit is possible** — legacy bug #7's class, closed structurally.
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
- **1b — async training + realtime graph** (**DONE** 2026-07-16, see status entry).
  Files: utility.cpp (thread_local), iterative (Observer/StopReason), network
  (sampleTestError), gui.cpp (TrainJob, 409, async=1, status/stop), gui_page.html.
  Smoke: existing lines unchanged + async poll-to-done, 409 busy, stop →
  `"cancelled"`; capture ctest two-thread assertion (verified to fail pre-fix).
- **2 — auto algorithm** (**DONE** 2026-07-16, see status entry). New netclone +
  autoalgo; regressnet refactor (goldens held). Smoke: `algorithm=auto` → `autoAlgo`
  w/ 3 probes + `Selected` in report, blocking AND async (adoption replaces modelPtr
  mid-job — the worker re-derives its pointers). Probes carry per-probe stopReason;
  bootstrap disabled on probe clones; diverged probes are results, not failures.
- **3 — plateau auto-stop** (**DONE** 2026-07-19, see status entry). New plateau.h +
  tests/plateau ctest (five synthetic traces: clean decay no-fire, decay-then-flat
  fire, flat sawtooth fire, sawtooth+slow-decay no-fire, NaN divergence fire).
  Smoke `autostop=1` → `stopReason:"plateau"` + control run + validation. **The plan's
  `plateauPeriodHint()=df()`/`W_eff` sawtooth-averaging mechanism was DROPPED** — a
  Rule-3 measurement showed its premise false (see status entry). The pre-existing
  gradMax-staleness quirk (`getGradMax()` only refreshes on print iterations, so
  grad_max fires *on* a print iteration) was confirmed real and left as-is — it is
  oracle/golden-pinned and the plateau detector does NOT share it (`update()` sees the
  fresh `setError` every iteration).
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
- ~~Binormal CI: recover the a-b covariance on the fitexy path~~ — **REJECTED
  2026-07-15, do not attempt.** The delta method assumed the z-ROC points were
  independent; they are cumulated from one sample and are not (Wickens pp. 87-88).
  A missing covariance term is not the defect — the correlation structure is absent
  from the model entirely, so no cross term rescues it. Superseded by ROADMAP 3
  (bootstrap). Reasoning: `docs/roc_theory.md`.
- `getGoodData()` port from the roc app if something needs it.
- More grooming tools (dataset describe/summary; train/test split outside engine).
