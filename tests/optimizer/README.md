# The optimizer benchmark harness

Phase 0, Step 0A of `docs/learning_research/optimizer_implementation_plan.md`.

This directory measures neuron's optimizers from identical starting states to a
fixed objective: canonical gradient descent, conjugate gradient descent, Shanno,
the L-BFGS implementation retained in Phase 3 (`src/lbfgs.*`,
`Network::TRAIN_LBFGS`), and — since Phase 4 — the research-only iRPROP+
prototype (`src/irprop.*`, `Network::TRAIN_IRPROP`). The packed
weight/objective/gradient boundary both of the last two use lives in
`src/network.h`, shaped by the first of them as its one real consumer.

**Phase 3 results: `docs/learning_research/lbfgs_screen_results.md`.** The short
version: L-BFGS reaches the committed practical endpoint 8.8x to 13.0x faster
than Shanno across three row counts and four weight seeds, in 14-20 full passes
against 123-210.

**Phase 4 results: `docs/learning_research/irprop_screen_results.md`.** Measured
against the standing portfolio panel rather than against a single winner — see
the plan's Phase 5 portfolio policy, which is explicit that benchmark results
rank recommendations and do not hide a correct, stable, eligible algorithm.

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

### What `full_passes` means once one iteration can traverse the data many times

`innerTrainSet()` counted a traversal for every method that makes exactly one
per call, which was all of them until L-BFGS. A Wolfe line search makes
**several per outer iteration**, one per trial point, and they all go through
`Network::batchObjectiveGradient()`. `Probe` overrides that too, in the same
idiom, and `full_passes` reports **training-set traversals**:

- an arm that made evaluation calls reports those — every trial point included;
- every other arm reports its `innerTrainSet()` count, unchanged.

The test is what the run *did*, not which model it is. Counting only outer
iterations is precisely how an optimizer comes to look like a winner because its
extra passes were invisible, and the plan is explicit that a lower
outer-iteration count is not a win. Canonical, CGD and Shanno never enter
`batchObjectiveGradient()` — their batch path calls the non-virtual
`batchGradient()` directly — so their pass counts are bit-identical to Step 0B's,
and Shanno's 194 passes on Civic Choice 6K reproduced exactly.

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

## Step 0B: what is measured, and what is not

Step 0B answers one question: **does an optimizer neuron already ships make the
intended large-data workload tractable?**  Canonical training stays the
numerical reference that defines the endpoints; it is not assumed to be the
operational speed baseline.

`docs/datasets/civic-choice` is the primary application benchmark, groomed and
loaded through the maintained recipe (`tools/mkdataset.py --onehot --refcat`),
not through a second encoder written here.  `prepare_data.py` materializes the
row-count series and **refuses** unless every size grooms to a byte-identical
column key — otherwise the "same problem at several sizes" claim would be false
and a scaling comparison would be comparing two design matrices.

| what | why it is here |
|---|---|
| Civic Choice logistic, 6k–400k rows | the application benchmark, and cheap enough to run whole |
| Civic Choice neural at the walkthrough's own 4 hidden units, 6k–100k | the architecture a reader of the walkthrough would actually fit |
| hidden = 2 / 4 / 8 / 16 at 25,000 rows | parameter-count scaling over sizes relevant to intended use |
| repeated 5-fold CV x2 plus the locked refit | the operation the intended workflow spends its time in |

**Deliberately absent**: correlated, separated, poorly scaled and non-finite
fixtures.  Those are candidate-specific *correctness* questions and belong to
the phase that has a candidate to discriminate.  Building the taxonomy now would
be work no pending decision depends on -- and the plan's scope governor is
explicit that timing minutiae must not displace candidate investigation.  Step
0A's failure and refusal fixtures are retained.

**A baseline is not a verdict.**  Step 0B establishes what the shipped methods
cost so a candidate has something honest to beat.  It does not decide that a
candidate is unnecessary: the governor states that neither a fast pilot nor
merely tractable training ends the search for material 10x-100x gains, so
`optimizer_baseline_results.md` reports the best speedup observed, which
operation dominates what remains, and **which credible candidates are still
untested**.

### Model configuration is each model's own defaults

A benchmark of a configuration nobody runs answers a question nobody asked, so
the Civic Choice arms use what a user gets from the constructor, written down
explicitly so the row reports it and the group invariant checks it:

| | eta | loss | decay | step-size search |
|---|---|---|---|---|
| `Network` (`network.cpp:20-46`, `model.cpp:9`) | 0.05 | LMS | on, 5e-5 | off |
| `Logistic` (`logistic.cpp:8-18`) | 0.05 | cross-entropy | off | **on** |

The step-size search being Logistic's default is itself worth knowing, and it is
why `canonical` and `canonical-autostep` are counted as two **methods** rather
than one setting: the search costs `maxLoops` extra full passes per iteration.

## The two predeclared endpoints

Both are training-objective values, both come from a **canonical** control run,
and neither may be tuned for a candidate.  They answer different questions and
live in different comparison groups, so no arm can be timed against the other's
target.

| endpoint | definition | what it detects |
|---|---|---|
| `practical` | the training objective canonical had reached when the **held-out** error stopped improving | when the usable model arrives — the tractability question |
| `strict` | the training objective canonical reached when it **converged** | late-stage optimizer failure |

### Neither definition was chosen by this harness

`strict` uses **the engine's own convergence rule**: `Iterative` constructs with
`gradMaxFlag` on and `gradMaxLimit` 1e-6 (`iterative.cpp:33-35`), so "converged"
already had a shipped meaning and the strict endpoint uses it.  `practical` uses
**the engine's own plateau detector**, `PlateauDetector` (`src/plateau.h`) at its
own default window, tolerance and patience — the same detector `setAutoStop`
uses.  A second definition of "has this stopped improving" would be a second
implementation of a decision the project already made (rule 6).

### The practical endpoint must not move when you watch longer

The first version of this rule took the **best** held-out error over the whole
characterization and asked when the series first came within 1% of it.  That
makes the endpoint a function of how long you looked: a longer run finds a
better best, moves the band, and moves the endpoint.  Measured on the Civic
Choice neural workload, the identical configuration reported its practical
endpoint at **iteration 11,299 under a 20,000 ceiling and 78,764 under a 100,000
one**.  The plateau detector is local — it fires where the series flattens and
cannot see what follows — and the same workload now reports **15,984 under
both**.

`practicalEndpoint()` is a free function over two vectors precisely so this is
testable: `optimizer_harness` hands it one synthetic trace truncated at two
horizons ten-fold apart, and **carries its own control** — the discarded
global-best rule is computed on the same trace and must be shown to move on it.
Without that control the test could pass on a trace no rule could have moved.

### A ceiling is not a floor, and a missing endpoint is a result

A characterization that exhausts its iteration ceiling **did not converge**, so
the objective it stopped on is where an unfinished fit happened to be.  No
strict endpoint is published from it.  In the committed table a strict value of
0 therefore means *this workload has no strict endpoint*, `addMethods()` declares
no strict arms for it, and `optimizer_harness` refuses a case whose target is 0
so nothing can run against an uncharacterized endpoint by neglect.

**This is not hypothetical.**  Canonical gradient descent does not converge on
the Civic Choice neural workload at the engine's own criterion: its maximum
gradient settles around 5e-4 and does not approach 1e-6 — 4.5e-4 at 20,000
iterations, and 6.2e-4 at 100,000, having *risen* in between.  The neural
workloads therefore carry a practical endpoint only, and that absence is one of
Step 0B's findings rather than a gap in it.  See
`docs/learning_research/optimizer_baseline_results.md`.

### Watching the held-out set must not change the fit

Characterization samples the held-out set every iteration, which calls
`forward()` on held-out rows and writes the network's scratch output.  That is
legacy bug #10's exact shape — a *reporting* action choosing the model.  So
every characterization runs the same short window twice, watched and unwatched,
and **refuses** unless the two objective trajectories are bit-identical.  What is
not claimed: that they were compared out to the strict endpoint.  The guard is
short on purpose — a mechanism that perturbed the fit would perturb it on the
first iteration, not the hundred-thousandth.

## Two identities, because they answer two questions

Step 0A established `source_id`: a hash over everything that can change what is
measured — the harness, **all** of `src/`, and the build file.  That is the right
authority for an **arm row**.

It is the wrong one for a **target**.  The committed endpoint table lives in
`harness.h`, so writing a measured target into it changes `source_id` by
construction — "the targets and the arms share a source identity" is a condition
that can never hold, and demanding it would mean re-characterizing forever.
What a target actually depends on is `src/`.

| field | covers | moves when |
|---|---|---|
| `source_id` | harness + `src/` + `CMakeLists.txt` | anything that can change a measurement |
| `engine_id` | `src/` alone | the engine changes — **not** when the target table is rewritten |

Every characterization record carries `engine_id`, `fill_targets.py` **refuses**
to merge characterizations from two different engines into one table, and the
table's generated header records the engine it describes.  An edit to
`src/network.cpp` invalidates the table loudly; an edit to this README does not.

## Two scopes, never one table

| scope | the clock covers |
|---|---|
| `optimizer` | exactly `train()` on one model, epilogue suppressed |
| `workflow` | every fold's fit **and** its scoring epilogue, plus the locked refit |

A whole-workflow number is the one the user actually waits for, and it is not an
optimizer timing.  `timing_scope` is a **group invariant**, a summary heading and
a column, and `validate()` refuses a cv arm that claims optimizer scope outright
— a cv clock necessarily covers each fold's scoring epilogue, because that is
where a fold's held-out predictions come from.

The repeated-fit arm runs the **maintained** policy, not a benchmark invention:
a stratified locked holdout (`nsplit::stratifiedHoldout`), cross-validation over
the development rows only, `nsplit::kFold`, and `crossval::evaluateOnce` for the
locked refit — the sequence `/api/cv` performs.  It inherits the convergence
contract rather than restating it: `cvadapters::trainProcedure` already fails any
fold whose fit ended at its ceiling, so an unreachable endpoint makes the arm
**fail** instead of returning a fast time for models nobody would use.

A cv row carries **no** iteration or pass count.  It ran `k*repeats+1` fits and
has no single one; a summed count would invite a per-pass ratio describing no
training run that happened.

## Repetitions are scaled to run cost, and the reason is recorded

Fifteen interleaved repetitions is right for a millisecond cell and wasteful for
a four-minute one.  Each case's count comes from its **own warm-up duration** — a
number the campaign already pays for and discarded:

| warm-up | repetitions |
|---|---:|
| under 1 s | 15 |
| under 2 min | 5 |
| over 2 min | 3 |

The count, its reason, and the observed warm-up time all go into the campaign
record and the summary.  A campaign that ran one arm three times and another
fifteen looks identical from the counts alone; what separates a legitimate one
from an illegitimate one is whether a declared policy chose from an observation,
so that is what is recorded.  Where the reduced count leaves a group's two
fastest arms with overlapping p10–p90 intervals, the summary says **ORDERING NOT
ESTABLISHED** rather than printing a decisive-looking table.

## Reproducing Step 0B

```bash
python3 tests/optimizer/prepare_data.py        # groom 6k/25k/100k/400k
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build -R optimizer --output-on-failure

./build/optimizer_probe --characterize --case <a canonical reference>
./tests/optimizer/run_probe.py --step0b --timeout 3600 --out step0b.jsonl
```

## Reproducing the Phase 3 screen

```bash
./tests/optimizer/run_probe.py --screen --seed 20260806 --timeout 600 --out screen.jsonl
```

`--screen` is the candidate comparison and nothing else: L-BFGS against the
Shanno incumbent at 6,000/25,000/100,000 rows, the predeclared `m` in {5,10,20}
sweep in its own group on the `lbfgs_memory` axis, three further predeclared
weight seeds each as its own matched group, and two late-stage arms that run to
the engine's plateau rule and therefore declare `endpoint: none`. Canonical and
CGD are deliberately absent — their standing is settled, and the plan forbids
scaling every historical arm.

The runner narrows to one half of the table with `--step0b` (the workload
matrix) or `--pilot` (the Step 0A mechanics, at 3 repetitions — plumbing, not
evidence). Membership is the probe’s to decide: the runner asks it rather than
matching a name prefix, so a renamed case cannot silently fall out of a subset.

## Reproducing the Phase 4 screen

```bash
./tests/optimizer/run_probe.py --irprop --seed 20260807 --timeout 3600 --out irprop.jsonl
```

`--irprop` is the candidate comparison against the **standing portfolio panel**,
which is four arms with four roles and not a race with one winner: L-BFGS as the
current speed leader, Shanno as the legacy quasi-Newton control, canonical as
the behavioral and matched-objective reference, and iRPROP+ as the candidate.
Canonical is present here although the Phase 3 screen omitted it — it is the
source of the endpoint every arm races to, and at 6,000 rows one run is ~18 s.
CGD is absent: its standing is settled and the plan forbids scaling every
historical arm.

Beyond the base panel it declares the three predeclared weight seeds (each its
own group), the late-stage plateau arms (`endpoint: none`), the 25,000- and
100,000-row scaling groups, and **the conditioning pair** below.

### The conditioning pair, and the fixture that nearly did not work

`well4` and `poor4` are one problem at two conditionings: same rows, same
outcome rule, same architecture, inputs on scales spanning 1000x. This is the
poorly scaled fixture Step 0B deliberately deferred to "the phase that has a
candidate to discriminate", and iRPROP+ is that candidate — a poorly scaled
objective is exactly where a per-parameter adaptive method is hypothesized to
earn its place.

**The obvious construction is a no-op, and the harness now gates against it.**
Multiplying each input column by a different constant produces a design matrix
*bit-identical* to the well-scaled one: `DataSet::normalize`
(`src/dataset.cpp:692`) min-max normalizes every input column onto
`[inLowerLimit, inUpperLimit]`, so any per-column **linear** rescale is exactly
cancelled before training sees it. That arm would have reported "no difference on
poorly scaled data" from a fixture that was not poorly scaled.

What survives normalization is *where the bulk of a column sits inside its own
range*. So each column anchors its min and max at ∓1 and compresses its
remaining values by `10^-j`: after normalization column 0 spans the full
`[-0.9, 0.9]` and column 3 occupies about ±0.0009 of it, so input 3's weight
must be ~1000x input 0's to contribute equally. `optimizer_harness` pins the
property the whole comparison rests on — that the two fixtures reach training
with **different split identities from identical starting weights**.

Canonical's own plateau is **21x worse** on the ill-conditioned twin (0.0762
against 0.00366), which is what establishes the fixture as genuinely harder
rather than merely different. Neither plateaued at a 40,000-iteration ceiling and
neither published an endpoint there — the ceiling-is-not-a-floor rule working;
both were characterized at 400,000.

`tests/optimizer/data/` is generated and not committed; a row identifies its data
by content (`data_id`), so provenance does not depend on that directory
surviving.

## Results

`docs/learning_research/optimizer_baseline_results.md`.  The short version:
**Shanno reaches the neural endpoint 81x faster than canonical** on real data
from an identical start, with a better held-out error -- and **the same method
cannot fit Logistic at all**, at any size tested.  CGD is 2x slower than
canonical.  Optimizer advantage is model-family-specific.

## Limitations carried out of Step 0B

0. **The campaign is partial.** The neural row and parameter series and the
   repeated-fit consumer are declared, tested and characterized, but not timed.
   The Logistic arms consumed the budget: a method that cannot reach the endpoint
   burns its whole ceiling every run, and re-establishing that failure at the
   larger sizes cost hours on the model family this program is not aimed at.
1. **The 400,000-row neural workload is projected, not measured.**
   Characterizing it alone costs hours before a single arm is timed. The
   logistic series runs the whole way because its canonical reference converges
   in ~17,000 iterations at every size.
2. **The neural workloads have no strict endpoint**, because canonical does not
   converge on them (above). Late-stage failure on those workloads is therefore
   not tested by a matched strict target.
3. **`simpleprop-25000-2` has no practical endpoint either**: its held-out error
   had not plateaued at the characterization ceiling, so the smallest point of
   the parameter sweep is absent rather than estimated.
4. Correlated, separated, poorly scaled and non-finite fixtures are not built.
5. `peak_rss_kb` is process-cumulative and Unix-only.
6. Step 0A's `logistic-shanno` still does not reach the shared `logistic-opt`
   target within its ceiling. Left visible rather than tuned away.
7. Four pilot cases are deliberately unusable (`passcount-nosearch`,
   `passcount-search`, `impossible-target`, `logistic-shanno`); the runner
   refuses to average them.
