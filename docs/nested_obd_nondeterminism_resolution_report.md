# Nested-OBD nondeterminism resolution report

## Problem

Nested-OBD cross-validation occasionally produced different out-of-fold
predictions across separate process invocations despite using the same seed.
The selected hidden-layer sizes remained identical, but some trained
predictions differed.

Earlier work had already fixed `Model::copy()` failing to copy the validation
matrix (a real correctness bug — kept). It had ALSO value-initialized `Matrix`
allocations, believing that the fix; a residual ~2–4% failure remained.

**That Matrix change was later REVERTED.** The decisive measurement (below): the
`errorType` fix eliminates the flake with Matrix allocations back to `new T[n]`
(garbage), so the Matrix value-init only perturbed the heap layout and never
cured anything. Current `src/matrix.h` allocates with `new T[n]`, and
`src/simpleprop.cpp` again describes the grow/prune scratch as garbage-filled.

## Root cause

`Model::Model()` initialized the displayed error label to `"LMS"` but did not
initialize the corresponding `errorType` boolean.

Normal menu and API workflows explicitly select an error function, which
masked the defect. OBD constructs `SimpleProp` directly and does not pass
through those setters. Consequently, OBD sometimes trained with LMS and
sometimes with cross-entropy according to the indeterminate value stored in
`errorType`.

This explains the observed behavior:

- Variation occurred across processes because initial memory contents varied.
- Same-process reruns were stable because allocator state was repeatable.
- Architecture selection often stayed unchanged while trained endpoints and
  predictions moved.
- The remaining defect was a non-Matrix scalar read, as suspected.

## Fixes

### 1. Initialize the default error type

In `src/model.cpp`, `Model::Model()` now initializes:

```cpp
errorType(false)
```

This makes the implementation agree with its existing `"LMS"` default label.

### 2. Fix a sibling copy-state omission

The copy-chain audit also found that `Network::currGradMax` was neither
initialized nor copied.

In `src/network.cpp`:

- The constructor initializes `currGradMax` to zero.
- `Network::copy()` copies it from the source network.

This was not the canonical-backprop OBD trigger because canonical
`getGradMax()` repacks and measures the gradient. It was nevertheless a
genuine uninitialized/copy-state defect affecting CGD and Shanno state.

### 3. Add deterministic regression tests

`tests/obd/check_obd.cpp` now contains two guards:

- It placement-constructs a `ProbeProp` in storage pre-filled with `0xff` and
  inspects the raw `errorType` byte. This reliably detects a constructor that
  fails to initialize the LMS default.
- It seeds `currGradMax` with a known value, copy-constructs the network, and
  verifies that the copied value matches.

The tests avoid relying on the original low-probability cross-process flake.

### 4. Update documentation

`docs/nested_obd_nondeterminism_handoff.md` now begins with a resolution
section documenting the measured cause, the fixes, and the regression-test
proof.

(The `SimpleProp::growHidden()` comment was briefly changed to “zero-initializes”
to match the Matrix value-init, then changed back to “garbage-fills” when that
Matrix change was reverted — the scratch structures are again garbage on resize,
which is fine because they are overwritten before use.)

## Required red-test proof

The new scalar fixes were temporarily removed, `check_obd` was rebuilt, and
the test was run.

Both new assertions failed:

```text
FAIL - a directly constructed network defaults to LMS error
FAIL - a copied network carries its current maximum gradient
```

The executable exited with status 1. The fixes were then restored and the
test rebuilt.

This satisfies standing rule 2: the regression tests were demonstrated to
fail against the bugs they guard.

## Verification results

All checks passed after restoring the fixes:

- CMake build: passed.
- CTest: 9/9 passed.
- Cross-process stress test: 0 failures in 120 separate invocations of
  `build/check_crossval`.
- Golden transcripts: byte-identical.
- Oracle comparison: numerically identical.
- Python/data/deployment tool outputs: matched committed outputs.
- Full GUI smoke test: passed.
- `git diff --check`: passed.

The first GUI smoke attempt failed because the sandbox could not bind a
loopback port. Re-running with loopback binding permitted passed the complete
GUI endpoint suite.

## What landed in the final commit (`a8d7c7e`)

The real fixes and the Matrix REVERT shipped together:

- `src/model.cpp` — `errorType( false )` in the constructor (the fix).
- `src/network.cpp` — `currGradMax` initialized in the constructor and copied in
  `Network::copy` (the sibling fix).
- `tests/obd/check_obd.cpp` — the two scalar guards (LMS default via placement
  into poisoned storage; the copied-gradient check).
- `src/matrix.h` — REVERTED to `new T[n]` (garbage allocation); the value-init was
  a heap-layout red herring, not a cure.
- `src/simpleprop.cpp`, `tests/matrix/check_matrix.cpp` — the value-init-tied
  comment and the vacuous Matrix-init note removed with the revert.
- `CLAUDE.md` — the CV-Step-3c entry corrected to credit `errorType` and record
  the Matrix misdiagnosis (rule 3: a fix that only reduces a heap-sensitive flake
  is a suspect, not a cure).

`NOTES.md` was left untouched. `AGENTS.md` and `docs/gui_cli_parity.md` needed no
change (no menu/GUI behavior moved).
