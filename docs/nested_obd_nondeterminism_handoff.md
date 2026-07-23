# Handoff: residual cross-process nondeterminism in nested-OBD cross-validation

**Repo:** `/Users/craign/code/neUROn2++/neuron-3.0` (branch `main`)
**Context doc:** `CLAUDE.md` → ROADMAP 4 Phase 4 (cross-validation). Read the standing rules
at the top of `CLAUDE.md` first — especially **rule 2** (a test must be proven to fail for
the measured reason), **rule 3** (measure before believing a mechanism), **rule 4** (engine
code lives in the bounds-checked `Matrix`/`vector_ops` class layer). Goldens and the oracle
are the hard invariants; do not move them.

## TL;DR

`build/check_crossval` (a ctest) has an assertion that two identical-seed nested-OBD
cross-validation runs produce byte-identical out-of-fold predictions. It **fails ~2–4% of
the time, but only across separate process invocations** — never within a single process.
This is the signature of an **uninitialized-memory read**. Two real bugs were found and
fixed (reducing the flake from ~20% to ~3%), but a **rare residual remains and has not been
localized**. macOS clang lacks MemorySanitizer, which is the tool that would name the exact
read. The ask: **localize and eliminate the residual** (a Linux MSan/valgrind run is the
obvious lever), OR conclude it is acceptable and make the test reliable.

## What the code does (just enough)

neuron-3.0 is a C++17 neural/statistical modeling engine (fork of neUROn2++, GSL dep,
embedded web GUI). ROADMAP 4 Phase 4 adds honest cross-validation model comparison.

The relevant path, **nested OBD cross-validation** — for each outer CV fold, the ENTIRE
architecture search runs *inside* that fold so the held-out rows never influence selection
(the no-leakage invariant):

- `src/crossval.{h,cpp}` — the generic CV runner (`crossval::run`) and coordinator
  (`crossval::compare`). Owns repetition only; a `Procedure` callback is the model adapter.
- `src/cvadapters.cpp` → `nestedObdProcedure(cfg, innerValFraction, selectedHidden*)`:
  given a fold, it (1) carves an inner train/validation split from the fold's TRAINING rows
  via `nsplit::stratifiedHoldout`, (2) `foldData.makeFold(innerTrain, testRows, innerVal)`
  materializes the fold three ways (train=inner-train, **validation**=inner-val, test=outer
  held-out), (3) runs `obd::run(foldData, ...)`, (4) reads the winner's held-out predictions.
- `src/obd.cpp` — the OBD grow-then-prune hidden-layer search. Early stops each size via a
  `ValidationObserver` that samples `Network::sampleTestError()`. Post-Phase-4c,
  `sampleTestError` reroutes to the VALIDATION set when one is loaded (so architecture
  selection never touches the test set). Grow phase warm-starts via `SimpleProp::growHidden`;
  prune phase clones the best net and removes units via `SimpleProp::removeHidden`.
- `src/simpleprop.cpp` — `growHidden` (add hidden units, warm start: restore old weights,
  new units get small random incoming + zero outgoing weights, bias slot relocates),
  `removeHidden` (prune), `forward`, and the training step.
- `src/network.cpp` → `sampleTestError` (the validation/test reroute).
- `tests/crossval/check_crossval.cpp` — the test. The flaky line is the assertion
  *"the same seed reproduces the nested-OBD out-of-fold predictions and sizes"*.

## The symptom, precisely characterized

- **Cross-process, not in-process.** A standalone program that loops `set_seed(7); run(...)`
  30× *in one process* never diverges. The ctest runs a lot of other work before the two
  nested-OBD runs (`ro`, `ro2`), so the heap/allocator state differs between `ro` and `ro2`
  — and *that* is what makes them occasionally differ. Classic uninitialized-read behavior:
  the read value tracks prior heap contents, which vary across processes / ASLR.
- **When it diverges: the selected architecture SIZES are identical, but a few PREDICTIONS
  differ** by a real amount (observed e.g. `0.526029` vs `0.492744` at row 2, fold 0). So
  OBD's *selection* is deterministic; the *winner's trained weights* occasionally end at a
  slightly different point — consistent with a tiny nondeterministic perturbation during
  training that rarely flips the early-stop iteration.
- **Confirmed uninitialized memory (two independent ways):**
  - Making `Matrix::resize` / ctors value-initialize (`new T[n]()`) → flake gone.
  - Poisoning freshly-allocated Matrix memory with NaN → determinism restored (both runs
    read the same NaN pattern). Both zero-init and a fixed poison remove the cross-process
    variation, which is exactly what "uninitialized read" predicts.

## What was already fixed (do NOT regress these)

Both are committed-intent (currently uncommitted in the working tree, pending this decision).

1. **`src/model.cpp` `Model::copy` did not copy the `Validation` submatrix** (a Phase-4c
   omission). A cloned network's held-out monitor (`sampleTestError`) therefore read an
   EMPTY validation set → **OBD's prune phase silently ignored validation early-stopping**.
   Fix: add `Validation = rhs.Validation;` to `Model::copy`. Guarded by a new assertion in
   `tests/obd/check_obd.cpp` (*"a copied network carries its validation set"*), **proven to
   fail** against the old code. This is a correctness bug independent of the flake, and it
   *amplified* the flake (once validation is actually sampled on clones, the nondeterminism
   surfaces more often: ~1/40 → ~15/40 before the Matrix fix).

2. **`src/matrix.h` left all `Matrix` allocations as garbage** (`new T[n]`). Value-initialized
   every allocation site (ctors at ~327/372/391/760, `resize` at ~793–794) to `new T[n]()`.
   This is the bulk fix (rule 4: the bounds-safe layer is now init-safe). **Verification:**
   `tests/golden/run_golden.sh` byte-identical and `tests/oracle/verify_oracle.sh`
   numerically identical (nothing correct depended on the garbage); **reverting the value-init
   reproduces the flake at 17/40**, which pins causation. (The plain `Matrix(rows,cols)` ctor
   at line ~327 is the arithmetic-temporary path and was the biggest contributor.)

Measured flake rates through the session (Release build, `build/check_crossval`, N processes):
- With only the `Model::copy` fix (Matrix still garbage): **15/40, 17/40**.
- No-growth probe (`hStart=hMax=2`, so no `growHidden`): **1/40** — i.e. growth is the
  dominant trigger, and a small residual exists even without growth.
- All Matrix allocations value-initialized: **0/40 and 0/60** in early batches, then **1/40**
  and **2/55** in later batches → residual **~2–4%**.

## What has been ruled out for the residual

- **Matrix allocations** — all value-initialized now; goldens/oracle byte-identical; residual
  persists, so it is a **non-Matrix** uninitialized read.
- **RNG** — `util::set_seed()` reseeds both the training stream (`d_random`/`i_random`) and the
  resample stream; the in-process 30× loop is bit-stable, so it is not an RNG-ordering issue.
- **Raw heap allocations in the path** — `grep -n "new \|malloc\|\[[0-9]" ` across
  `simpleprop.cpp network.cpp iterative.cpp twoset.cpp vector_ops.h plateau.h autoalgo.cpp
  obd.cpp` finds nothing (the legacy `double[10000]` H-L stack arrays were already replaced by
  `std::vector`). So it is not an obvious raw buffer.
- **`growHidden` weight matrices** — `hW`/`oW` are fully restored; the scratch matrices
  `hWup`/`hG` are overwritten by `Matrix::outprod` (which writes every cell) before use; for
  `trainingType==0` (the config the test uses) `hG` is not even read.
- Note: `std::vector` members (`hO`, `oW`, `oG`, `h_err`, `I`, `stackG`, …) are value-initialized
  by the standard library, so those are not garbage sources.

## Prime suspects (unverified — for Fable to measure, not assume; rule 3)

- A **scalar or vector element read before it is written** somewhere in the grow → clone →
  prune → validation-early-stop path, whose value only perturbs the winner's training endpoint
  rarely (near an early-stop tie). The divergence being "same sizes, slightly different weights"
  points at the *training endpoint*, i.e. the early-stop iteration for the selected size.
- The **prune clone** path specifically (`removeHidden` on a `cloneNetwork` copy, then a short
  retrain that now samples validation). This is what the `Model::copy` fix newly activated, and
  the growth+validation combination is where the flake concentrates.
- Anything in `Iterative`/`SimpleProp` state that is **copied by the copy-constructor chain**
  (`SimpleProp::copy → Network::copy → Iterative::copy → Model::copy`) but is a scalar/flag that
  is read before being set on the clone. Worth diffing each `copy()` against the class's member
  list to find a member that is *not* copied (that is exactly how the `Validation` bug looked).
  **Concretely: audit `Iterative::copy` and `Network::copy` and `SimpleProp::copy` for any member
  omitted from the copy** — the `Validation` omission in `Model::copy` was found this way and may
  have a sibling.

## Exact reproduction

```bash
cd /Users/craign/code/neUROn2++/neuron-3.0
cmake -B build -DCMAKE_BUILD_TYPE=Release        # Release matters; -O0 hides timing-y effects
cmake --build build --parallel
# Run the test many times across SEPARATE processes and count failures:
fail=0; for i in $(seq 1 60); do build/check_crossval >/dev/null 2>&1 || fail=$((fail+1)); done
echo "$fail / 60"        # expect ~1-3 failures; the failing line is the nested-OBD reproduce assertion
```

The failing assertion prints:
`FAIL - the same seed reproduces the nested-OBD out-of-fold predictions and sizes`

To see *what* diverges (sizes vs predictions, which row/fold), the two vectors compared are
`ro.oofPrediction` vs `ro2.oofPrediction` and `pickedHidden` vs `pickedHidden2` near the end of
`tests/crossval/check_crossval.cpp`. A post-run diff printer (does not perturb the runs, since it
runs after both) can dump the first differing row/fold and the two values.

## The right tool

macOS clang has **no MemorySanitizer** (MSan), and Instruments/leaks/guard-malloc do not flag
uninitialized *reads*. The decisive move is a **Linux build under MSan** (or valgrind
`--track-origins=yes`), which names the exact file:line of the uninitialized read and its
allocation origin. The CI matrix already has an ubuntu job (`.github/workflows/ci.yml`); a
temporary MSan build of `check_crossval` there (or in any Linux container) should pinpoint it in
one run. Note MSan wants an instrumented libc++ for zero false positives, but even an
uninstrumented run with `--track-origins` under valgrind on Linux will very likely flag it.

## Constraints on any fix

- `tests/golden/run_golden.sh` must stay **byte-identical** and `tests/oracle/verify_oracle.sh`
  **numerically identical** (these guard the whole CLI/engine; the CV path is not in them, but a
  fix in shared engine code must not disturb them).
- Keep the two existing fixes (`Model::copy` Validation; Matrix value-init).
- 9 ctests must stay green; `tests/gui/smoke.sh` green.
- Per rule 2, any new guard must be **proven to fail** against the specific bug — and note that
  cross-process determinism is hard to unit-test in one process (that is *why* this bug shipped
  in commit 3b unnoticed). A reliable guard may need MSan in CI rather than an in-process assert.

## Current git state

- Committed: `6a17463` (CV step 3b: the nested-OBD adapter) — this is where the flake was
  introduced/shipped.
- Uncommitted working tree (this session): `src/cvreport.{h,cpp}` (the CV report, done and
  tested), `src/crossval.{h,cpp}` (coordinator gains timing/arch/foldId + shared `metricsFor`),
  `src/matrix.h` (value-init), `src/model.cpp` (Validation copy), `tests/{crossval,matrix,obd}`,
  `CMakeLists.txt`. All gates green **except** the ~3% cross-process flake on the one assertion.
- `NOTES.md` is Craig's untracked scratch — never commit it.
