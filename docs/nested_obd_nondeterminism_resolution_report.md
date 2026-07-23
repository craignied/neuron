# Nested-OBD nondeterminism resolution report

## Problem

Nested-OBD cross-validation occasionally produced different out-of-fold
predictions across separate process invocations despite using the same seed.
The selected hidden-layer sizes remained identical, but some trained
predictions differed.

Earlier work had already fixed:

- `Model::copy()` failing to copy the validation matrix.
- Uninitialized allocations in `Matrix`.

A residual failure remained at roughly 2–4%.

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

An outdated `SimpleProp::growHidden()` comment was also changed from
“garbage-fills” to “zero-initializes,” reflecting the existing Matrix
initialization fix.

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

## Files changed

- `src/model.cpp`
- `src/network.cpp`
- `src/simpleprop.cpp`
- `tests/obd/check_obd.cpp`
- `docs/nested_obd_nondeterminism_handoff.md`
- `docs/nested_obd_nondeterminism_resolution_report.md`

`NOTES.md` was left untouched.

No menus or GUI behavior changed, so `AGENTS.md` and
`docs/gui_cli_parity.md` did not require updates.
