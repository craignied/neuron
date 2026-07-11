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
  **Not carried forward** (see decisions).
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
4. **Exporters: dropped.** The HTML/Palm/iPhone exporters are not carried forward — in the
   current era an LLM can generate a calculator from a saved model directly.

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
