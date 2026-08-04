# The optimizer benchmark harness

Phase 0, Step 0A of `docs/learning_research/optimizer_implementation_plan.md`.

This directory measures the optimizers neuron **already has** — canonical
gradient descent, conjugate gradient descent, and Shanno — from identical
starting states to a fixed objective. It contains no new optimizer, no packed
weight API, and no pure objective/gradient boundary. Those belong to the L-BFGS
phase, shaped by their first real consumer.

## What is here

| File | Role |
|---|---|
| `harness.h` | shared mechanics: cases, fixtures, identities, the timed run, the row schema |
| `optimizer_probe.cpp` | the **measuring** binary. Built, never a ctest case |
| `check_optimizer_harness.cpp` | the **mechanics**, asserted deterministically. ctest case `optimizer_harness` |
| `run_probe.py` | orchestration: interleaving, row validation, comparison-group enforcement, summaries. ctest case `optimizer_runner_selftest` |

Two binaries over one header, because the binary that measures and the binary
that proves the measurement trustworthy must run the same code or the proof is
about something else. The orchestration layer is part of the evidence system, so
it is tested too.

## Why `optimizer_probe` is not a CTest timing gate

Wall time is machine-dependent. A timing assertion in the suite is a flake
generator, and the project has already paid for the related lesson once: a test
that pinned one machine's bit-exact doubles failed on two other platforms while
every structural assertion passed (`tests/backprop/check_bpoptimizer.cpp`).
`tests/clustered/scale_probe.cpp` set the precedent — the target is **built** so
it cannot rot, and **running** it is a deliberate act.

Everything the measurement depends on *is* gated, with no timing assertion
anywhere: `optimizer_harness` (161 deterministic C++ checks) and
`optimizer_runner_selftest` (the Python validation logic).

## What is timed, and what is excluded

The timed region is **exactly `train()`**, bracketed by
`std::chrono::steady_clock` (monotonic):

```
    ... dataset, architecture, weights, weight identity, function fingerprint ...
    t0 = steady_clock::now();
    achieved = p.train();
    t1 = steady_clock::now();
    ... end identities, memory reading, row serialization ...
```

Excluded by being outside the bracket: dataset and split construction, model
construction, `randomize()`, all four identities, row serialization, and summary
statistics. Excluded by `setQuiet(true)`: the reporting epilogue — accuracy
report, classification tables, the ROC fit and its 2000-resample bootstrap, the
single largest thing that would otherwise contaminate an optimizer timing.
`setHistory(false)` and `setLastop(false)` keep `neuron.log` and `model.txt` off
the disk entirely.

Quiet mode changes what a run **says**, never what it computes — the settled
`setQuiet` contract, and why the gradient calculation the stopping rules read
sits outside every reporting guard (legacy bug #10).

## Identity: three questions, three answers

Arms are built by **deterministic reconstruction** — same seed, same
construction sequence, then `randomize()`. Cloning is unavailable: `cloneNetwork`
dispatches on `typeid` and the counting subclass is a distinct type.
Reconstruction is *verified*, never assumed.

| field | question | how |
|---|---|---|
| `split` | **which data** | full-precision content hash of the training matrix and its shape |
| `weight_start_id` / `weight_end_id` | **which parameter state** | full-precision hash of the model's **actual weight structures**, including dimensions, ordering and the concrete model tag |
| `function_start_id` / `function_end_id` | **which function** | hash of `forward()` outputs over the training inputs — **secondary, diagnostic only** |

**An arm is compared on its parameter state.** The function fingerprint cannot
serve that role and is not asked to: distinct weight vectors can agree on every
training row — a hidden-unit permutation is exactly such a collision — and two
fixtures sharing input columns produce the same value from the same weights.
That last property is *pinned as a test*, so nobody later mistakes it for a
defect or for a parameter identity.

`save()` was rejected as a serialization source: `Matrix::operator<<` writes at
the stream's default precision, six significant digits, so a serialization hash
would silently equate starts differing below 1e-6 — and it would put files on
disk needing cleanup.

Non-vacuity is gated: an untouched FNV offset basis, a zero-row traversal, or an
empty weight container is **refused**, never hashed into a comparison.

### The BackProp blocker

| model | weights | reached by |
|---|---|---|
| `Logistic` | `W` | public `getBetas()` |
| `SimpleProp`, `BareProp` | `hW`, `oW` | `protected` in `OneHiddenNet` |
| `BackProp` | `Weights` | `protected BackProp::weightMatrices()` |

**Every model now has a parameter-state identity.** BackProp was the sole
exception — its `Weights` were `private` with no accessor, so no subclass could
read them and `run_probe.py` refused to certify a BackProp comparison group at
all.

`BackProp::weightMatrices()` (added 2026-08-04) closes that with the narrowest
possible seam: a `protected`, `const`, non-virtual, inline accessor returning the
**authoritative** weights by const reference. It exposes `Weights` and nothing
else — `WeightsUp`, `WeightsAccumulate`, `Gradient` and `vpack` are training
*workspace*, not parameters, and stay private. Widening the whole private section
instead would have promised every future subclass mutable access to all four,
which is a far larger commitment than observation requires. No production code
calls it, so it cannot enter a hot loop (rule 7). Documented in the Manifest's
BackProp section under *Protected methods*, with an index entry.

`weight_id_available` remains, and remains checked: an unavailable or empty
identity is **refused**, never quietly replaced by the weaker function
fingerprint. A weaker identity wearing the stronger one's name is the defect this
whole mechanism exists to prevent.

## Comparison groups

Every case declares a `comparison_group` and a `group_axis`. Arms in one group
must agree on **everything that defines the work**; only the declared axis may
vary:

```
split, fixture, model, arch, loss, rows, inputs, params, weight_seed,
weight_start_id, mode, eta, auto_step, decay_on, decay, grad_stop,
target, ceiling
```

`run_probe.py` **refuses** a result set whose groups violate this — it does not
warn. A group whose arms describe different work cannot support a timing
conclusion, and a warning nobody reads is not a control.

### Gradient stopping is part of the group, not a free choice

`grad_stop` does two things at once: it arms `STOP_GRADMAX`, and it selects which
production branch canonical training takes. With it off the canonical
accumulator branch runs; with it on the separate-gradient branch runs and
`engine()` dispatches through a switch with no `case 0`. **CGD and Shanno require
the separate-gradient branch**, so a canonical arm compared against them must run
it too — otherwise the group is timing two different code paths under one label.

Every optimizer group therefore uses `grad_stop:true` with the limit at 0, so the
branch changes and the rule can never fire. The fast accumulator branch is
retained as its own group, `canonical-branch`, with axis `grad_stop branch` —
honestly labelled rather than silently timed against optimizers.

## Iteration semantics

`Iterative::getIterations()` **means two different things depending on how a run
ended**: on a stopping-rule break it is the zero-based index of the iteration
that just finished, while on ceiling exhaustion the loop leaves it at
`maxIterations+1`, which is a count. Measured, and asserted both ways:

| case | `iteration_index` | `iterations_completed` |
|---|---:|---:|
| `logistic-canonical` (breaks on `min_error`) | 28 | **29** |
| `impossible-target` (ceiling 50) | 51 | **51** |

The row therefore carries both. `iterations_completed` counts `trainSet()` calls
directly and is correct on every exit path; it is what pass ratios and summaries
use. `iteration_index` is retained only for correlating with engine reports.
Production `Iterative` semantics were not touched.

## Full-pass counting

`Probe<NET>` overrides `innerTrainSet()` and `trainSet()`, increments a counter,
and calls the production implementation — the idiom
`tests/network/check_autostep.cpp` established and CI runs. Both are called once
per pass and once per iteration, never per exemplar, so the counters sit outside
every hot loop and rule 7 is untouched. No production counter was added.

| case | iterations completed | full passes |
|---|---:|---:|
| `passcount-nosearch` | 21 | 21 |
| `passcount-search` | 21 | **84** |

Both run to the same fixed ceiling on an unreachable target, so the iteration
counts are equal by construction and the pass count is the only free variable.
The ratio is asserted exactly (`maxLoops` trial passes plus one real pass), not
merely as "more".

## Binary identity

A benchmark row must identify the code that produced it, and a revision alone
cannot: the harness is developed uncommitted, so every row would claim a commit
whose tree does not contain the binary's source. Three facts are compiled in:

| field | meaning |
|---|---|
| `source_id` | **THE AUTHORITY.** SHA-256 over every source input that can change what is measured: the three harness files, **all 75 `src/*.h` and `src/*.cpp`**, and `CMakeLists.txt`. Each file's path is hashed with its content, so a rename moves it too |
| `source_files` | how many files that identity covers — a changed source *set* is a different fact from a changed source *file* |
| `rev` | the committed revision, for orientation |
| `dirty` | whether the tree differed from that revision **at configure time**: a hint, not the authority |

An earlier version hashed only the three harness files. That was wrong: a dirty
edit to `src/network.cpp` or `src/backprop.cpp` could change the measured
optimizer while the id stood still, so it could not honestly be called the
authority. It now covers the engine the harness links.

Every hashed file is a `CMAKE_CONFIGURE_DEPENDS`, so editing any of them re-runs
cmake, refreshes the id, and rebuilds the targets together. `.git/HEAD` and
`.git/index` are dependencies too, so **committing** — which changes `rev` and
clears `dirty` without touching a source file — also refreshes them.

What is *not* claimed: editing a file outside that list (a document, say) can
leave `dirty` reading stale until the next configure. That cannot change what is
measured, which is precisely why `source_id` and not `dirty` is the authority.

`run_probe.py` asks the binary what it is (`--identity`) and **refuses** any row
disagreeing on any identity field. It does **not** overwrite `rev` with the
working tree's HEAD — that is how a stale binary comes to claim source it does
not contain. `--rev` annotates only.

### The runner is identified too

The orchestrator materially controls a campaign: it selects the arms, orders
them, validates every row and decides what is summarized. A result identifying
only the measuring binary cannot answer *"what selected and filtered these
rows?"* So an `--out` file begins with a **campaign record** —
`{"record":"campaign", ...}` on line 1, arm rows after it — carrying a runtime
SHA-256 of `run_probe.py` itself alongside the probe's full identity, the
orchestration seed, repetitions, timeout and case list. Arm rows carry `schema`
and no `record`, so a reader tells them apart without positional assumptions.

## Output schema (version 2)

One JSON Lines row per arm on stdout; diagnostics on stderr. Schema 2 because
field *meanings* changed: the start identity is a parameter state, the iteration
count is a completed count, and `data_seed` is gone.

`schema`, `rev`, `dirty`, `source_id`, `source_files`, `build`, `case`,
`comparison_group`,
`group_axis`, `fixture`, `split`, `model`, `arch`, `loss`, `rows`, `inputs`,
`params`, `weight_seed`, `weight_id_available`, `weight_start_id`,
`weight_end_id`, `weight_elements`, `weight_id_note`, `function_start_id`,
`function_end_id`, `optimizer`, `optimizer_name`, `mode`, `eta`, `auto_step`,
`decay_on`, `decay`, `grad_stop`, `target`, `achieved`, `ceiling`,
`iteration_index`, `iterations_completed`, `full_passes`, `elapsed_ns`,
`peak_rss_kb`, `stop_reason`, `converged`, `target_reached`, `finite`, `usable`,
`failure_stage`, `error`.

**`data_seed` was removed**: `fixtureMatrix()` never used it, so a row reporting
it was false provenance. Step 0B introduces a real split/data seed with real
holdout splits.

A row is usable **only** if it is finite, reached its target, converged, stopped
on `min_error`, did not end at the ceiling / cancellation / probe budget,
completed a positive number of iterations and passes, and its **end state** is
sound — final outputs finite and the whole training set traversed. A finite
returned objective sitting on non-finite final outputs is not a completed fit
however fast it was.

Non-finite numbers are emitted as JSON `null`, never a bare `nan`.
`peak_rss_kb` is **process-cumulative** and Unix-only, which is why the runner
runs one arm per process.

### Failure rows

**Every runtime failure still emits exactly one row.** Exceptions out of setup or
out of `train()` are caught, staged (`failure_stage` = `refused` | `setup` |
`training`), and returned with the exception's message. The process does not
terminate and the campaign does not lose the arm — which matters immediately for
Step 0B's singular, separated and non-finite cases, where throwing is the engine
behaving correctly. On a training fault the start identity is retained and **no
end identity is taken**, because a model left mid-update is not a state to
fingerprint.

The runner aborts only for a malformed, missing or inconsistent row — never
because a valid row is `usable:false`.

## Choosing targets

The matched endpoint uses `STOP_MIN_ERROR`; every other rule is off, with a high
finite ceiling. **A target comes from a canonical control, never from the arm
being timed**, and `--characterize` now enforces that rather than merely claiming
it:

```
$ ./build/optimizer_probe --characterize --case simpleprop-shanno
refused -- characterize: 'simpleprop-shanno' is not a canonical reference
(optimizer=shanno, auto_step=off). A matched target must come from a canonical
control, not from the arm being timed.
            characterize 'simpleprop-canonical' instead.
```

Refusal is preferred over silently substituting the group's canonical case: the
caller asked for a specific arm, and quietly running a different one is how a
result comes to describe a run nobody requested. An automatic-step arm is
likewise not a canonical reference.

Pilot targets remain **pilot** values. Step 0B replaces them with a committed
table measured per model.

## Reproducing

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release      # prints the benchmark identity
cmake --build build --parallel

ctest --test-dir build -R 'optimizer'          # both deterministic gates
./build/optimizer_probe --identity
./build/optimizer_probe --list
./build/optimizer_probe --characterize --case simpleprop-canonical

./tests/optimizer/run_probe.py --pilot
./tests/optimizer/run_probe.py --reps 15 --seed 20260804 --out results.jsonl
./tests/optimizer/run_probe.py --self-test
```

The runner warms each case once and discards it, then shuffles arm order
independently within each repetition from a recorded seed. A per-arm timeout
(default 120 s, matching the project's "a training table still running after two
minutes is suspect" rule) terminates a hung arm, names it, fabricates no row, and
exits nonzero. Raise `--timeout` deliberately for a long Step 0B workload.

### Measured (Apple Silicon, Release/NDEBUG, 5 interleaved repetitions, seed 20260804)

Pilot numbers, recorded as evidence that the harness discriminates — not as a
baseline claim:

| case | median ms | MAD ms | passes |
|---|---:|---:|---:|
| `simpleprop-shanno` | 0.212 | 0.003 | 10 |
| `logistic-canonical` | 0.218 | 0.004 | 29 |
| `bareprop-shanno` | 0.240 | 0.006 | 12 |
| `bareprop-canonical` | 1.892 | 0.023 | 99 |
| `simpleprop-canonical-accumulator` | 2.344 | 0.016 | 120 |
| `simpleprop-canonical` | 2.516 | 0.060 | 120 |
| `simpleprop-canonical-separate` | 2.586 | 0.051 | 120 |
| `simpleprop-cgd` | 4.958 | 0.076 | 229 |

`logistic-shanno`, `passcount-*`, `impossible-target` are unusable and reported
as such. `logistic-shanno` failing to reach the target is a **finding**, not a
harness fault: it is recorded, never averaged.

### The harness detects a known overhead

Phase 0's exit criterion. A deterministic per-pass cost injected inside the timed
region, measured against the same seed and workloads, then removed:

| case | before ms | after ms | delta ms | combined MAD | passes | delta/pass ms |
|---|---:|---:|---:|---:|---:|---:|
| `logistic-canonical` | 0.225 | 5.611 | +5.386 | 0.011 | 29 | 0.1857 |
| `simpleprop-canonical` | 2.520 | 24.672 | +22.151 | 0.379 | 120 | 0.1846 |
| `simpleprop-shanno` | 0.212 | 2.071 | +1.859 | 0.074 | 10 | 0.1859 |

Every delta has the same sign — the signature of a true common per-iteration
overhead — and each exceeds its combined spread. The stronger result is the last
column: the harness recovered **the same per-pass cost to three significant
figures from workloads spanning a 12x range in pass count**, so it did not merely
detect the overhead, it attributed it to the per-pass mechanism. The injected
code is not retained and no timing assertion was created.

## Step 0B

**`docs/datasets/civic-choice` is the primary application benchmark.** Its role
is to answer whether an optimizer accelerates a representative *complete
analysis*, not to be a mathematical correctness fixture. Groom and load it
through the maintained dataset recipe, preserve one committed split/seed and one
serialized initial weight state per model arm, and benchmark at least:

- one Logistic fit;
- forward and reverse `RegressNet` procedures;
- repeated cross-validation and the locked refit;
- one eligible neural fit where its architecture and objective make the
  comparison meaningful.

Report the single-fit result and the complete stepwise/CV elapsed-time
multiplier separately. Optimizer-only timing continues to exclude the final
statistical reports and the ROC bootstrap.

Other reference roles, unchanged from the plan: **low-birth-weight** is the
logistic endpoint reference including its known log likelihood; **XOR and the
existing goldens** are the neural correctness references; **small analytic
quadratics** test optimizer formulas exactly; **deterministically generated
fixtures** cover scaling, correlation, separation, non-finite behavior, row
sweeps and parameter sweeps.

## Limitations carried into Step 0B

1. Two fixtures only (`linear2`, `xor2`), generated in the probe. The correlated,
   separated, poorly scaled, row-sweep and parameter-sweep fixtures are 0B's.
3. No holdout split yet, so `split` equals the fixture identity and no real data
   seed exists.
4. Pilot targets are pilot values, not a committed table.
5. `peak_rss_kb` is process-cumulative and Unix-only.
6. `logistic-shanno` does not reach the shared `logistic-opt` target within its
   ceiling. Left visible rather than tuned away: choosing a target that flatters
   an arm is exactly what the canonical-control rule forbids.
6. Four pilot cases are deliberately unusable (`passcount-nosearch`,
   `passcount-search`, `impossible-target`, and currently `logistic-shanno`);
   the runner refuses to average them.
