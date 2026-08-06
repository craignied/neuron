# Optimizer implementation and measurement plan

Status: **authoritative current-work plan**

Date: 2026-08-03

Intended reader: a fresh implementation agent (likely Claude Opus) working in this
repository. This document is designed to be referenced directly from `CLAUDE.md` while
the work is active.

## Mission

Determine, by implementation-quality prototypes and reproducible wall-clock evidence,
which new learning algorithms materially reduce neuron's training time while reaching a
comparable numerical endpoint. Integrate every correct, stable, eligible retained
algorithm; use measurements to set recommendations and defaults, not to hide valid
choices. Preserve existing
statistical results, stopping semantics, goldens, oracle behavior, REST/GUI contracts, and the
project's numerical vocabulary.

The execution order is:

1. Characterize existing neural optimizers only far enough to establish the incumbent.
2. Prototype L-BFGS against that incumbent on a cheap representative neural workload.
3. Reject it early or scale the winner/incumbent pair; do not scale every historical arm.
4. Prototype iRPROP+ and apply the same staged gate.
5. Use safeguarded BB as a low-cost additional candidate/control when it advances the
   neural decision; investigate other neural candidates when evidence justifies them.
6. Decide retain/reject per candidate from matched-endpoint and held-out evidence.
7. Integrate every correct, stable, eligible retained implementation into the
   REST API and GUI, with benchmarks guiding defaults and bounded automatic
   selection rather than hiding algorithms. Do not extend the legacy CLI menus.
8. Consider LM, Adam/AMSGrad/Nesterov, or SVRG when scaled neural evidence identifies an
   workload that justifies them.

This is an implementation-and-research program, not authorization to ship every named
algorithm.

## Large-workload speed scope governor

This section governs every phase and every implementation brief derived from this plan.
Its purpose is to keep the program aimed at the user's real requirement: making
hours-scale training on large datasets dramatically faster. Tractability is a minimum
bar, not a reason to stop investigating a credible additional 10x or 100x improvement.
It overrides a mechanically exhaustive reading of later measurement lists when those
details do not help choose, validate, or reject an algorithm.

**Current program boundary.** This effort is specifically about novel training
algorithms for neuron's neural models and engines. Logistic regression, IRLS, DFA, and
general statistical-model timing are out of scope unless a neural candidate directly
depends on a shared primitive whose correctness must be isolated there. Cross-validation,
stepwise selection, and complete-workflow timing are secondary multipliers, not research
targets: measure them only after a neural optimizer wins the single-fit gate and only when
needed to quantify the user's actual end-to-end saving.

1. **Measure the best existing method first.** Canonical training may define an initial
   practical endpoint, but it is not automatically the operational or strict numerical
   reference. The fastest reliable existing optimizer that reaches the predeclared useful
   endpoint becomes the incumbent a new candidate must beat; late-stage comparisons may
   use a separately predeclared best-known incumbent endpoint when canonical stalls.
2. **The incumbent is established.** Step 0A found Shanno roughly ten times faster across
   small neural fixtures; Step 0B measured it 81 times faster than canonical on Civic
   Choice at the same practical objective. Shanno is now the neural incumbent. Missing
   scale points move into candidate experiments and do not block prototype work.
3. **Investigate algorithms, not timing minutiae.** Prioritize implementation-quality
   prototypes of credible high-impact candidates and compare them on Civic Choice,
   representative large-data scaling, and the repeated-fit consumers that dominate
   intended use. Add timing decomposition or broader synthetic matrices only when they
   help choose, validate, diagnose, or reject a candidate. Do not spend the research
   budget exhaustively characterizing small baseline effects while major candidates
   remain untested.
4. **Use two predeclared endpoints when useful.** A practical endpoint represents a
   model whose further training no longer materially improves its intended use; a strict
   or late-stage endpoint tests continued numerical progress. Derive both independently
   before comparing arms, never tune either for a candidate, and report time and
   reliability separately. If canonical stalls, do not declare strict comparison
   impossible by definition: characterize a best-known incumbent endpoint explicitly.
5. **Scale repetition effort to the decision.** Retain at least 15 randomized/interleaved
   repetitions for short cells. For genuinely long cells, at least five repetitions may
   be used for seconds-to-minutes runs and at least three for sufficiently expensive runs,
   with the reason recorded and more runs required when spread leaves the ordering
   ambiguous. Prefer testing another credible algorithm or scale point over refining the
   last decimal of an already decisive timing ratio.
6. **Keep high-upside neural candidates in scope.** An existing method making a workload
   tractable does not by itself justify stopping: a credible further 10x or 100x gain
   remains valuable. L-BFGS and iRPROP+ should be investigated unless staged evidence or
   an implementation-quality prototype rejects their applicability. BB remains a low-cost
   harness/control experiment and a possible useful method, not a mandatory production
   feature. Candidate priority may change when measurements expose a larger opportunity.
7. **Prefer measured time or bounded scaling evidence to toy speedups.** Millisecond
   microbenchmarks validate mechanics and isolate overhead, but they do not by themselves
   justify production work intended to save hours. When a full-scale characterization
   would itself be wasteful, use a smaller scaling series and label any extrapolation
   explicitly.
8. **Stop a candidate when its decision is made, not the whole search prematurely.** Every
   phase report must state whether the workload is tractable, the best speedup observed,
   which operation dominates remaining time, which credible candidates remain untested,
   and the resulting candidate order. Reject or defer candidates on evidence, but do not
   equate “good enough” with “no worthwhile improvement remains.”
9. **Benchmark inside the candidate loop.** Phase 0 establishes only enough baseline to
   name a trustworthy incumbent. Do not finish every row sweep, parameter sweep, CV cell,
   or timing decomposition before implementing candidates. First compare a candidate with
   the incumbent on a cheap representative neural workload; reject obvious losers early,
   then spend large-dataset time only on survivors. Use three interleaved repetitions for
   a decisive first screen, expand repetitions only when noise could change the decision,
   and scale the winner/incumbent pair rather than every historical method.

## Authorities to read before coding

Read these in this order at the start of the implementation session:

1. Repository `CLAUDE.md` and the universal file it names.
2. `AGENTS.md`.
3. `docs/development_rules.md`, especially rules 2, 3, 4, 5, 6, 7, 8, and 9.
4. `docs/optimizer_research.md` in full.
5. This plan.
6. `docs/learning_research/research_comparison.md` and
   `research_comparison_fable_comment.md` for the decisions behind this plan.
7. The relevant current Manifest sections before each phase:
   - Chapter 2, Parameter Estimation Formulas and Example Learning Algorithms;
   - Chapter 5, `vector_ops` and `Matrix`;
   - Chapter 12 sections for `Iterative`, `Network`, `OneHiddenNet`, `SimpleProp`,
     `BareProp`, `BackProp`, and `Logistic`.
8. Before any Manifest edit, read `docs/manifest_maintenance.md` in full.
9. Before any GUI/API edit, read `docs/gui_cli_parity.md` and the Manifest REST chapter.

Do not load `docs/HISTORY.md` or `docs/refactor_audit.md` wholesale. Search them only for
a named historical question encountered during implementation.

## Accepted conclusions from the research review

The following are settled for this program unless new measurements disprove them.

### Ranking hypotheses

The ranges below are prioritization hypotheses, not acceptance thresholds or claims about
neuron:

| Candidate | Hypothesized opportunity | Intended workload |
|---|---:|---|
| Corrected IRLS/Newton | order-of-magnitude to potentially 100x+ | Logistic, especially repeated stepwise/CV fits |
| L-BFGS | approximately 5–20x | Smooth full-batch neural and logistic objectives |
| iRPROP+ | approximately 5–10x | Full-batch neural objectives, especially poorly scaled problems |
| Safeguarded BB | approximately 2–10x over fixed/expensive-step GD | Full-batch control and low-overhead adaptive step |
| LM | potentially very large, narrow eligibility | Small least-squares networks only |
| Adam/AMSGrad/Nesterov | uncertain, likely online-specific | Online/noisy gradients |
| SVRG | uncertain, row-count-dependent | Very large finite datasets |

These estimates decide research order only. A candidate is accepted from measured time to
a matched endpoint, not because it falls inside its hypothesized range.

### Architecture decisions

1. `Iterative` continues to own the training loop, stopping, cancellation, progress,
   validation monitoring, and stop reasons.
2. `Network::engine()` remains a direction/step dispatch point for algorithms that can
   honestly fit that contract. It is not a universal optimizer framework.
3. IRLS belongs to `Logistic`, not `Network::engine()`.
4. A genuine Wolfe line search cannot use `innerTrainSet()` as its evaluator because
   `innerTrainSet()` updates weights. L-BFGS requires explicit packed weights plus a
   trial-point objective/raw-gradient evaluation boundary.
5. Do not build that boundary in Phase 0. BB and IRLS do not need it. Add it in the
   L-BFGS phase, shaped by the first real consumer.
6. New optimizer working state is **per `train()` run**. Size/reset it in
   `Network::prepareRun()` or the owning concrete model's override. Do not save it in
   network files. Copy persistent configuration explicitly; the next `train()` resets
   working history.
7. `currGradMax` always represents the raw objective gradient at the point used for the
   iteration, never a transformed direction or absolute step.
8. A step-owning optimizer must have an explicit absolute-step application path. Never
   implement it by dividing by eta or by mutating the user's eta to one.
9. Public auto-selection will not automatically probe every new algorithm. Its candidate
   set must be curated and model-family-aware from benchmark evidence.

### Measurement decisions

1. Speed-to-endpoint claims are workload-scoped and predeclared.
2. Per-iteration overhead claims use a sign test across applicable workloads: a true
   common overhead should have a consistent direction; mixed signs inside observed spread
   are noise.
3. Use at least 15 randomized/interleaved repetitions per benchmark cell after warm-up.
4. Report median, MAD, p10/p90, endpoint objective, iterations, full-pass/evaluation
   counts, exemplar counts where available, stop reason, failures, and peak memory.
5. Exclude final reports and ROC bootstrap from optimizer timing.

## Branch, commit, and evidence discipline

Work on a dedicated branch such as `research/optimizer-speed`; do not develop the full
program as an unreviewable change on `main`.

Each phase has three possible endings:

- **Retain:** measurements justify the next phase or production integration.
- **Research-only:** useful evidence, but no public option; preserve only deterministic
  harness/oracle code that is worth maintaining.
- **Reject:** record the method, fixture, configuration, endpoint, timings/spread, and
  failure reason in `docs/HISTORY.md`, then remove disposable implementation code.

Commit boundaries must separate:

1. characterization/evidence harness;
2. numerical primitives;
3. behavior-preserving extraction;
4. each candidate implementation;
5. each candidate's measurements/decision;
6. public-surface integration;
7. Manifest and operational documentation where practical, while keeping required
   same-commit parity/Manifest synchronization with the public code it describes.

Never combine a newly discovered correctness defect with a refactor or optimizer feature.
Characterize and land the correctness fix first.

For every new test that claims to guard a mechanism:

1. Run the restored implementation and show the test passes.
2. Sabotage the exact term/branch the test claims to guard.
3. Rebuild and retain build output showing every affected translation unit recompiled.
4. Run the test and show the intended failure (not an earlier/wrong fault).
5. Restore the implementation.
6. Rebuild and again retain output showing affected translation units recompiled.
7. Run the test and show it passes.
8. Remove temporary sabotage and scratch output from the final tree; summarize the
   evidence in the commit message or retained research record.

## Phase 0 — baseline characterization and benchmark harness

### Objective

Measure what currently costs time and create a deterministic, non-CTest performance tool
without changing production optimizer behavior or creating a speculative optimizer
abstraction.

### Files to inspect

- `src/iterative.*`
- `src/network.*`
- `src/autoalgo.*`
- `src/onehidden.*`
- `src/backprop.*`
- `src/logistic.*`
- `tests/props/check_props.cpp`
- `tests/backprop/check_bpoptimizer.cpp`
- `tests/network/check_autostep.cpp`
- `tests/clustered/scale_probe.cpp` as the pattern for a built-but-not-CTest probe
- relevant CMake target definitions

### Deliverables

1. `tests/optimizer/optimizer_probe.cpp` (or an equally clear name).
2. A CMake executable target `optimizer_probe`, built in Release but not registered with
   `add_test` because timing assertions would be flaky.
3. A deterministic runner in `tests/optimizer/` using bare `python3` only if orchestration
   is clearer outside C++; it must not require pip packages.
4. A compact fixture/configuration manifest in `tests/optimizer/README.md` describing
   datasets, architectures, seeds, target objectives, and invocation.
5. A baseline evidence table under `docs/learning_research/`, for example
   `optimizer_baseline_results.md`. Do not put machine-dependent raw timing literals in
   correctness tests.

### Harness design

The probe must support at least:

- model: Logistic, SimpleProp, BareProp, BackProp;
- optimizer: canonical, CGD, Shanno;
- mode: batch and online where valid;
- fixed dataset/fixture name;
- seed and/or saved initial-weight fixture;
- eta, automatic step-size flag, decay, and stopping configuration;
- target objective;
- repetition identifier;
- quiet training;
- machine-readable one-row CSV or JSON output.

Each row must include:

- git revision and Release/debug identity;
- model, architecture, row count, input count, parameter count;
- dataset/fixture and split identity;
- seed and initial-weight hash;
- optimizer/configuration;
- target objective and achieved objective;
- iteration count;
- `innerTrainSet()`/full-pass count where measurable without production changes;
- elapsed monotonic-clock time around quiet training only;
- stop reason and `Iterative::converged()`;
- finite/nonfinite status and failure message;
- peak RSS if portable measurement is available without contaminating the timed region.

Use built-in `STOP_MIN_ERROR` to stop at the matched target during timing, with all other
convergence conditions disabled except a very high safety ceiling. Quiet mode excludes
the report and ROC bootstrap. Obtain target objectives in a separate characterization
stage from a converged canonical control, record them in the fixture manifest, and do not
recompute/tune the target separately for each timed arm.

Where full-pass counts are needed, prefer benchmark subclasses that override
`innerTrainSet()` to count calls and then invoke the production implementation. Do not add
hot-path production counters merely for the benchmark unless the later accepted optimizer
needs those counters as public progress.

### Workload matrix

**Primary application benchmark: Civic Choice.** Use the maintained synthetic
`docs/datasets/civic-choice` dataset as the principal realistic end-to-end workload. Its
role is to answer whether an optimizer accelerates a representative complete analysis,
not to serve as the only mathematical correctness fixture. Groom/load it through the
maintained dataset recipe, preserve one committed split/seed and one serialized initial
weight state per model arm, and benchmark at least:

- one Logistic fit;
- forward and reverse `RegressNet` procedures;
- repeated cross-validation and the locked refit;
- one eligible neural fit when its architecture and objective make a meaningful
  optimizer comparison.

Report both the single-fit result and the complete stepwise/CV elapsed-time multiplier.
Do not include final statistical reports or the ROC bootstrap in optimizer-only timing;
time them separately only when reporting complete user-workflow latency.

The rest of the committed deterministic matrix separates correctness, robustness, and
scaling questions:

1. Logistic, well-scaled, small `p`.
2. Logistic, larger `p`.
3. Logistic with correlated inputs.
4. Logistic with complete or near separation, treated as a failure/robustness case rather
   than a timing win.
5. Biased SimpleProp, one hidden layer.
6. Unbiased BareProp with the same dimensional scale.
7. Multi-layer BackProp.
8. At least one poorly scaled neural fixture.
9. Row-count sweep with fixed parameter count.
10. Parameter-count sweep with fixed row count.

Use the low-birth-weight reference for logistic endpoint correctness (including its known
log likelihood), and the existing XOR/golden neural fixtures where they discriminate.
Use small analytic quadratics for exact optimizer-formula tests. Generate the correlated,
separated, poorly scaled, nonfinite, row-sweep, and parameter-sweep fixtures
deterministically in the probe when doing so keeps the repository smaller.

### Trial protocol

1. Warm each executable path once.
2. Restore identical starting weights for every optimizer arm.
3. Randomize arm order within each repetition using a separately recorded orchestration
   seed.
4. Run at least 15 repetitions per cell.
5. Report median, MAD, p10, p90, failures, and full-pass counts.
6. Run on an otherwise idle machine; record CPU/OS/compiler/build type.
7. Keep raw result files out of git unless they are intentionally chosen compact evidence.

### Baseline questions that must be answered

- What fraction of canonical training time is forward/backprop work versus automatic
  step-size trial passes?
- Does fixed eta or automatic step size win by workload?
- Are CGD/Shanno iteration reductions real wall-clock reductions?
- Which real consumers dominate cumulative training time: a single fit, stepwise, CV,
  OBD, or auto-selection?
- At what row/parameter scales do full passes dominate fixed overhead?

### Phase 0 tests and gates

- The probe must verify nonempty finite results before reporting timings.
- Two arms initialized from the same snapshot must report the same initial-weight hash.
- A deliberately different snapshot must change the hash (non-vacuity guard).
- A deliberately impossible target must end at the ceiling and be reported as failure,
  not a speed result.
- Run existing focused tests and `git diff --check`.

### Exit criterion

Proceed only when the harness reproduces stable baseline medians/spread and can distinguish
an intentionally added known overhead. Use the per-iteration sign test for that overhead
check. Do not add packed-weight or pure-evaluation interfaces in this phase.

## Phase 1 — research-only safeguarded Barzilai-Borwein prototype

### Objective

Validate the harness and estimate the value of replacing fixed/legacy eta selection with
a spectral step at negligible per-iteration cost. BB is implemented first because it is
small, not because it is predicted to be the universal fastest method.

### Source contract

Before coding, obtain and cite:

- Barzilai and Borwein (1988), the BB1/BB2 formulas;
- Raydan (1997), the exact selected globalization/safeguard policy.

Choose one exact production candidate before measurement. BB1 is the initial control:

`s_t = w_t - w_(t-1)`

`y_t = g_t - g_(t-1)`

`alpha_t = (s_t' s_t) / (s_t' y_t)`.

Prototype BB2 only as a separately named arm, not a hidden flag in a claimed BB1 method.

### Minimal implementation shape

Do not add generic weight packing. In the batch separate-gradient path, reconstruct the
previous parameter displacement from the update already applied:

`s_t = -alpha_(t-1) d_(t-1)`.

Store explicitly:

- previous raw packed gradient;
- previous applied packed direction/step information sufficient to reconstruct `s`;
- previous accepted alpha;
- initialization/reset flag.

At the optimizer call:

1. `pack()` the current raw gradient.
2. Set `currGradMax = maxabs(rawGradient)` before transforming anything.
3. On the first iteration, use the configured initial eta.
4. Thereafter compute `s`, `y`, numerator, and denominator through `vector_ops`.
5. Apply the exact cited positivity, finite, lower-bound, upper-bound, and fallback policy.
6. Leave the raw gradient as direction and expose the chosen alpha to the existing batch
   update without turning automatic step-size search on.
7. Record research counters for fallback, clamp, and accepted spectral steps outside hot
   element loops.

Research-only dispatch may temporarily reserve an internal `trainingType` value on the
feature branch. Do not expose it through CLI/GUI/API or document it as a public capability.
If BB is rejected, remove the dispatch and state. If retained, assign its final public
token during Phase 5.

BB is batch-only. Refuse or make the research harness reject online use; never silently
reinterpret per-exemplar differences as the published batch method.

### DRY and speed rules

- Reuse `stackG`, `lastG` only when their meanings remain legible; add one clearly named
  BB state vector rather than overloading `lastF` with a different formula.
- Add a destination-taking vector primitive only if the published equation cannot be
  expressed without allocation using existing operations.
- Reset all BB working state in `prepareRun()`.
- Do not mutate user eta as a side effect of selecting or leaving BB. BB may update the
  effective per-iteration alpha, but persistent configured eta remains the initial/fallback
  configuration and must be restored/unchanged for a later canonical run.

### Formula and integration tests

Add `tests/optimizer/check_bb.cpp` or equivalent:

1. Hand-computed two-dimensional positive-definite quadratic, exact BB1 alpha.
2. First-step initial eta.
3. Nonpositive `s'y` fallback.
4. Near-zero denominator fallback.
5. NaN/Inf refusal/fallback according to the cited contract.
6. Lower and upper clamps.
7. `currGradMax` equals the raw-gradient maximum, not the scaled update.
8. Per-run reset: two separate `train()` calls start with clean BB history.
9. Quiet/audible equivalence if an audible research path exists.
10. Fixed-start real Logistic, SimpleProp, BareProp, and BackProp batch runs reach the
    matched target or report a clear failure.

Prove the formula test fails by sabotaging the BB numerator or denominator, with fresh
recompilation evidence both ways.

### Benchmark comparison

Compare BB1 against:

- canonical fixed eta;
- canonical automatic step size;
- CGD;
- Shanno.

Report time-to-target, epochs, full-pass counts, clamp/fallback frequency, and failures.
Use the sign test only for the common per-iteration overhead question; allow
workload-specific time-to-target conclusions.

### Exit decision

- Retain BB as a candidate only if its wall-clock gain exceeds spread on a predeclared
  useful workload and it reaches the matched endpoint reliably.
- Even if BB is not the final public winner, the harness is validated if it correctly
  measures BB's extra dot-product overhead and step-size-search pass savings.
- Record the decision before Phase 2. Remove rejected disposable code after evidence is
  preserved.

## Phase 2 — corrected IRLS/Newton for Logistic

### Objective

Measure the largest plausible family-specific speed opportunity while preserving exactly
the objective currently optimized by `Logistic::innerTrainSet()`.

### Mathematical contract

neuron's penalized logistic objective is:

`mean cross-entropy + (decay/2) * |W|^2`.

The intercept is the last element of `W` and is currently decayed. Preserve that behavior.
For `N` training rows, define:

- `p = sigmoid(X W)`;
- minimization gradient sum `gSum = X' (p-y)`;
- unpenalized information sum `Hsum = X' V X`, with
  `V_ii = p_i (1-p_i)`.

The Newton system matching the current mean objective is:

`[Hsum/N + decay I] delta = -[gSum/N + decay W]`.

For better arithmetic and fewer divisions, the equivalent system is:

`[Hsum + N*decay I] delta = -[gSum + N*decay W]`.

Then `W_new = W + delta`.

When weight decay is off (the Logistic default), all penalty terms vanish. Never use
`Hsum + decay I` with an unnormalized score; that changes regularization by a factor of N.

### Numerical primitive prerequisite: direct solve

Add an authoritative `Matrix<double>::solve` operation rather than training through an
explicit matrix inverse.

Recommended interface, subject to consistency with existing Matrix conventions:

`vector<double>& Matrix<double>::solve(const vector<double>& rhs,
                                       vector<double>& solution) const`.

Requirements:

- square nonempty coefficient matrix;
- rhs and destination dimension checks at entry;
- destination-taking implementation;
- one LU decomposition and backsubstitution using the existing Matrix numerical
  vocabulary or GSL behind `Matrix`, not from `Logistic`;
- `DimensionMismatch`, `BadSize`, or `Singular` according to existing contracts;
- no explicit inverse construction;
- documentation and index entry in the Matrix Manifest section.

Test known systems, identity, dimension failures, singular systems, const input
preservation, and destination correctness. Prove the solve test fails by sabotaging a
backsubstitution term and visibly rebuilding/restoring `matrix.cpp`.

Land this primitive separately before IRLS.

### One authoritative logistic derivative calculation

Extract the current information-matrix construction from `Logistic::reportAccuracy()`
into one Logistic-owned method that computes the unpenalized likelihood quantities.

Recommended concrete contract:

`double Logistic::likelihoodDerivatives(Matrix<double>& informationSum,
                                       vector<double>& gradientSum)`

where the return value is the **mean unpenalized cross-entropy at current W**,
`informationSum = X'VX`, and `gradientSum = X'(p-y)`.

The exact signature may change if current class conventions demand it, but retain these
properties:

- one owner for `X'VX`;
- one forward loop over the training rows per derivative evaluation;
- no optimizer update or state mutation;
- no statistical report writes;
- no use of the test or validation set;
- unpenalized information remains available to Wald covariance and condition number;
- penalty is added only by the training caller;
- raw gradient `G` is populated from `gradientSum/N + decay W` for `getGradMax()`.

`Logistic::reportAccuracy()` continues to invert the **unpenalized sum** for Wald
covariance and passes that same unpenalized matrix to `conditionOf`. Do not let penalized
training curvature improve or replace the design-conditioning diagnostic.

### IRLS training path

Add a Logistic-owned one-step method, for example `double Logistic::irlsStep()`, and a
research-only dispatch in `Logistic::trainSet()`.

Each iteration:

1. Compute current mean data error, `Hsum`, and `gSum` once.
2. Add the current penalty to the returned pre-update objective.
3. Form the penalized system using `N*decay` scaling.
4. Solve for `delta`; never form the inverse.
5. Validate every matrix/vector/step value is finite.
6. Apply exactly the published IRLS/Newton step policy selected before coding.
7. Populate `G` with the raw current penalized mean gradient.
8. Set `currGradMax` from that raw gradient or allow `getGradMax()` to pack it without
   recomputation.
9. Update `W` once.
10. Return the pre-update objective, preserving existing `Iterative::train()` cadence.

Before coding step damping/halving, obtain and cite the exact primary/authoritative rule.
Do not invent an uncited “safeguarded IRLS” hybrid. Start the research arm with the exact
full Newton/IRLS update if that is the selected source; add a cited globalization variant
as a separate arm only if characterization demonstrates a need.

Automatic step-size search and eta are inapplicable to IRLS. The research harness must
disable them; a later public option must refuse incompatible configuration or clearly
disable the irrelevant controls in both CLI and GUI.

### Failure contract

Do not assume complete separation always throws `Matrix::Singular`. Characterize and test
separately:

- exact singular information/system matrix;
- ill-conditioned but solvable system;
- nonfinite probability, objective, gradient, matrix, or step;
- coefficient growth under complete separation;
- iteration ceiling without convergence;
- step policy exhaustion, if a safeguarded variant is used;
- cancellation between iterations.

Each fault fixture must be proven to reach the intended branch. Never silently fall back
to gradient descent; optimizer choice is part of the result.

### IRLS tests

1. One Newton step against an independent high-precision calculation.
2. Unpenalized endpoint agreement with long-budget canonical logistic on a stable fixture.
3. Penalized endpoint agreement with canonical logistic, specifically guarding the factor
   of N and decayed intercept.
4. Finite-difference agreement for mean objective/gradient.
5. `X'VX` equality between training derivative method and Wald/condition report path.
6. Condition number remains unpenalized and does not change when decay changes at fixed W.
7. `currGradMax` is the current raw penalized mean-gradient maximum.
8. Exact singular, ill-conditioned, separated, and nonfinite contracts.
9. Per-run state/reset (IRLS should be effectively stateless beyond scratch).
10. Stepwise selection agrees with converged canonical logistic on a decisive fixture.
11. CV predictions/statistics agree within predeclared numerical tolerance at comparable
    fitted coefficients.
12. Quiet/audible equivalence.

Sabotage the `N*decay` term and require the penalized endpoint/formula guard to fail.

### IRLS benchmarks

Measure:

- single fits across `N` and `p` sweeps;
- unpenalized and penalized cases;
- well-conditioned and correlated designs;
- complete/near separation robustness;
- full `RegressNet` forward and reverse procedures;
- repeated CV with locked refit;
- auto-selection only later, after any public integration.

Compare time to canonical's matched objective and, for logistic, coefficients/log
likelihood. Report solve time separately if it becomes measurable.

### Exit decision

IRLS advances only if it reaches the same objective/coefficients within declared tolerance,
handles failures explicitly, and materially reduces end-to-end logistic workload time.
Do not publicize it based on iteration count alone.

## Phase 3 — L-BFGS and the packed objective/gradient boundary

### Objective

Introduce the minimum common evaluation mechanism required by L-BFGS, prove that the
extraction preserves existing behavior, then measure a complete limited-memory BFGS method
with one exact cited line search.

### Pre-code source decision

Obtain the exact sources and pin the implementation in this plan or an adjacent research
note before writing code:

- Nocedal (1980) and Liu & Nocedal (1989) for limited-memory updates;
- Nocedal & Wright for the two-loop recursion and initial scaling;
- Wolfe and the selected complete line-search algorithm.

Choose exactly one of:

- a complete weak-Wolfe search; or
- a complete strong-Wolfe search with its bracketing/zoom/interpolation rules.

Do not implement Armijo-only backtracking and call it Wolfe. Record `c1`, `c2`, initial
step, maximum evaluations, bracketing limits, interpolation safeguards, curvature-pair
threshold, and failure/restoration behavior.

### Characterization before extraction

Add or extend tests that pass on the old engine and pin:

- batch objective and raw gradient for every eligible model;
- one batch canonical update;
- one CGD and Shanno update;
- weight-decay objective/gradient;
- batch versus online separation;
- automatic step-size trial restoration;
- BackProp's post-optimizer gradient consumption;
- quiet/audible and report/no-report continuation behavior.

These are behavior characterizations, not new-mechanism tests. Confirm they pass before
the extraction.

### Required model boundary

Add protected Network contracts with explicit meanings, subject to naming review:

`virtual void packWeights(vector<double>& destination) const = 0;`

`virtual void unpackWeights(const vector<double>& source) = 0;`

`virtual double batchObjectiveGradient(vector<double>& packedRawGradient) = 0;`

Properties:

- packed weight layout exactly matches packed gradient layout;
- destination size is validated once;
- `batchObjectiveGradient` evaluates current installed weights, returns the current mean
  objective including current decay policy, and writes the raw mean gradient;
- it does not update weights, optimizer state, iteration state, reports, TwoSet caches,
  test/validation data, or stopping state;
- one virtual call per full trial evaluation is acceptable; there is no virtual dispatch
  inside exemplar or element loops;
- all hot scratch containers are pre-sized outside repeated trial loops where practical;
- no `std::function`, generic descriptor, or type switch enters the hot path.

Ownership:

- `OneHiddenNet` implements the shared SimpleProp/BareProp weight packing and batch
  objective/raw-gradient equations where their mathematics is identical.
- Concrete SimpleProp/BareProp retain bias-architecture-specific packing fragments only
  if their weight layouts truly differ.
- `BackProp` implements its vector-of-Matrix layout.
- `Logistic` implements its vector layout, reusing `likelihoodDerivatives`.

Refactor existing batch separate-gradient `innerTrainSet()` paths to consume the same
authoritative batch objective/raw-gradient implementation. Keep online paths separate.
Do not duplicate all model equations once for L-BFGS and once for legacy training.

Land and verify this extraction as a behavior-preserving commit before adding L-BFGS.
Goldens and oracle comparisons remain byte-identical.

### L-BFGS ownership

Add `src/lbfgs.h/.cpp` (or a comparably named service) as the one owner of:

- history length `m` configuration;
- `s`, `y`, and `rho` ring buffers;
- previous accepted parameters and raw gradient;
- two-loop scratch vectors;
- line-search counters/state;
- curvature-pair acceptance/reset;
- direction calculation;
- trial-step selection and restoration policy.

`Network` owns or composes one `LBFGSState` per run and supplies model evaluation through
the boundary above. `Iterative` still owns cancellation/stopping. Check cancellation
between every trial evaluation.

### L-BFGS step

At an accepted point `(w_k, f_k, g_k)`:

1. Set `currGradMax = maxabs(g_k)` before direction transformation.
2. If no history, use steepest descent.
3. Otherwise execute the published two-loop recursion newest-to-oldest then
   oldest-to-newest.
4. Use `gamma_k = (s'y)/(y'y)` initial inverse-Hessian scaling when valid.
5. Require a descent direction; if not finite/descent, reset history and retry steepest
   descent according to the cited contract.
6. Run the selected Wolfe search using installed trial weights and pure
   `batchObjectiveGradient` evaluations.
7. On acceptance, leave accepted weights installed and retain the accepted objective/raw
   gradient for the next outer iteration.
8. On failure or cancellation, restore the last accepted parameters exactly.
9. Add `(s,y,rho)` only when the cited curvature condition is satisfied.
10. Count objective and gradient evaluations separately from outer iterations.

Preserve existing training semantics: the value returned to `Iterative::train()` must have
a documented point association. Prefer returning the objective at the point whose raw
gradient drives that outer step, matching legacy pre-update reporting. If caching an
accepted post-step evaluation avoids an extra pass, make the iteration timeline explicit
and test it; do not silently move stop semantics.

### L-BFGS state lifecycle

- Reset working history and counters in `prepareRun()`.
- Persistent configuration (history length, line-search constants if configurable) is
  copied explicitly.
- Copying a model does not promise mid-run continuation because `train()` begins a new
  run and resets working history.
- Randomize/load/structural changes need no special optimizer cleanup beyond the next
  per-run reset, but direct research calls that bypass `train()` must be refused or
  explicitly initialize state.
- Saved networks contain weights/architecture only.

### L-BFGS tests

1. Weight pack/unpack round trip for every model, including nonempty/hash guards.
2. Packed weight layout matches gradient layout dimension.
3. Pure objective/gradient call leaves weights and optimizer state unchanged.
4. Finite-difference gradient agreement for every model/loss/decay combination.
5. Two-loop recursion against an independent dense inverse-BFGS oracle on a small
   positive-definite quadratic.
6. History rollover for `m=1`, `m=5`, and a partially filled buffer.
7. Invalid curvature-pair skip/reset.
8. Initial scaling formula.
9. Both line-search conditions on accepted steps.
10. Bracketing/zoom/evaluation ceiling.
11. Non-descent fallback.
12. Trial rejection and exact restoration.
13. Cancellation during a trial and exact restoration.
14. Raw-gradient `currGradMax`.
15. Per-run reset and configuration copy.
16. Batch-only refusal.
17. Fixed-start integration for all four model families.

Sabotage the second loop or a Wolfe curvature check and prove the targeted test fails after
fresh recompilation, then restore and prove it passes.

### L-BFGS benchmarks

Compare history lengths 5, 10, and 20 only; do not launch an unbounded tuning exercise.
Compare against canonical fixed/automatic eta, CGD, Shanno, retained BB if applicable,
and IRLS on Logistic. Count every full trial evaluation. Stratify results by parameter
count, row count, conditioning, loss, and model family.

### Exit decision

Retain L-BFGS only when extra line-search passes still yield a material wall-clock win to
the matched objective with acceptable robustness and memory. A lower outer-iteration count
alone is insufficient.

## Phase 4 — iRPROP+ using an explicit absolute-step contract

### Objective

Measure the lowest-overhead per-parameter batch adaptation without corrupting eta or
hiding the published absolute step.

### Published contract

Transcribe the exact iRPROP+ table from Igel & Hüsken (2003), with:

- initial `Delta = 0.1`;
- `etaPlus = 1.2`;
- `etaMinus = 0.5`;
- `DeltaMin = 1e-6`;
- `DeltaMax = 50`;
- the exact error-dependent rollback branch;
- current-gradient zeroing on a sign flip as specified.

Do not combine RPROP-, RPROP+, iRPROP-, and iRPROP+ behind variant flags. Implement one
named algorithm first.

### Step ownership

Add an explicit result/dispatch distinction between:

- a **direction** that the model updates as `weights -= eta * direction`; and
- an **absolute step** that the model applies as `weights += step` or
  `weights -= step`, according to one documented sign convention.

The distinction may be an enum returned once per epoch or separate named methods; it must
not be a generic hot-loop descriptor. The absolute-step branch is outside exemplar loops
and shared by the batch packed update path.

Never:

- divide the iRPROP+ step by eta;
- set eta to one;
- leave automatic step-size search active;
- silently run it online.

### State and implementation

Own in one `IRpropState`:

- previous raw gradient;
- per-parameter `Delta`;
- previous applied step;
- previous objective;
- initialized flag.

Reset all working state in `prepareRun()`. At each current point, use the pure batch
objective/raw-gradient boundary introduced for L-BFGS. Compute sign products and step
changes with destination-taking `vector_ops` primitives; keep the paper's branch structure
visible. Apply the absolute step through the shared packed-weight boundary.

Weight decay remains part of both objective and raw gradient before sign comparison.

### iRPROP+ tests

1. A hand-computed multi-coordinate sequence covering positive, negative, and zero sign
   products.
2. Growth/shrink clamps.
3. Error-worsened rollback and the no-rollback branch.
4. Current-gradient zeroing after sign flip.
5. Zero raw gradient.
6. Nonfinite objective/gradient/Delta refusal.
7. Absolute step unaffected by configured eta.
8. Selecting iRPROP+ and then canonical leaves the configured canonical eta unchanged.
9. Automatic step-size incompatibility.
10. Batch-only refusal.
11. Raw-gradient `currGradMax`.
12. Per-run reset/configuration copy.
13. Fixed-start real-model integration.

Sabotage rollback or sign-flip zeroing and prove the precise mechanism guard fails and then
passes after restoration/recompilation.

### iRPROP+ benchmarks

Apply the standing comparison panel below. Include poorly scaled fixtures and strict
late-stage objectives; report early speed separately from time to the matched final
target.

### Exit decision

Retain if it is correct and stable, reaches comparable endpoints, and is either reasonably
competitive with the panel or adds a meaningfully different robustness/workload profile.
It need not beat L-BFGS on the synthetic benchmark. Reject it only for a clear performance,
stability, endpoint, or redundancy failure under the portfolio policy below.

## Phase 5 — candidate decision and public integration

### Standing comparison and portfolio policy

Do not turn the latest benchmark winner into the sole gate for the next candidate. Every
neural candidate is measured against a stable reference panel with distinct roles:

- **L-BFGS is the current speed leader.** It is the primary modern wall-clock and
  full-pass reference, not a requirement that every retained method must beat.
- **Shanno is the established legacy quasi-Newton control.** It preserves continuity
  with the historical optimizer evidence and exposes regressions specific to the newer
  implementation path.
- **Canonical training is the behavioral and matched-objective reference.** It defines
  legacy training semantics and a numerical endpoint where it can reach that endpoint;
  it is not presumed to be the speed competitor.
- **Eventually, all retained algorithms enter the bounded limited-run selector.** Their
  performance on the user's actual dataset, under identical starts and declared budgets,
  guides the full-training choice.

Benchmark results rank recommendations and defaults; they do not hide a correct, stable,
eligible algorithm merely because another method is faster on the synthetic dataset.
Retain a candidate that is within a defensible performance band or has complementary
behavior across datasets, seeds, scaling, conditioning, architectures, losses, endpoint
depth, or failure modes. Reject methods that are prohibitively slow, unstable, fail the
comparable endpoint, or are demonstrably redundant without a distinct operational value.
Record both the portfolio-retention decision and the separate default/recommendation
ranking.

### Decision meeting artifact

Before changing any public surface, create
`docs/learning_research/optimizer_candidate_results.md` containing one table per candidate:

- exact source/equation and adaptation;
- workload claim;
- configuration;
- endpoint target and achieved endpoint;
- median/MAD/p10/p90;
- outer iterations, full passes/evaluations, failures, memory;
- robustness across seeds/scales;
- retain, research-only, or reject decision;
- reason the benefit does or does not justify permanent public complexity.

Do not require a candidate to beat the most recent winner. Define whether it is reasonably
competitive or whether its eligible workloads, convergence behavior, or failure profile
are distinct enough to add portfolio value.

### Public algorithm identifiers

Preserve existing meanings:

- `1` = canonical;
- `2` = CGD;
- `3` = Shanno;
- `auto` = automatic selection.

Append new stable tokens only for retained algorithms. Do not renumber existing choices.
Centralize cold-path token/name/eligibility mapping in the lowest appropriate service so
REST, GUI, auto-selection, OBD, CV, and reports cannot drift, while keeping hot optimizer
dispatch direct and visible.

### Family eligibility

Define and test explicitly:

- IRLS: Logistic only, batch by definition, eta/autostep irrelevant.
- L-BFGS: eligible smooth full-batch model families measured successfully.
- iRPROP+: full-batch eligible models measured successfully.
- BB: full-batch eligible models if retained.
- LM: LMS-only and size-limited if ever retained.
- stochastic methods: only modes actually measured.

Ineligible selections are refused by name before changing model configuration. Nothing
silently falls back or coerces mode.

### Automatic selection policy

Benchmark and choose one explicit policy:

1. probe every eligible retained optimizer;
2. probe a curated global subset; or
3. probe a family-aware subset.

Family-aware is the default recommendation: IRLS is relevant only to Logistic, while LM
would be LMS/small-network only. Keep the total auto budget bounded; adding algorithms
must not linearly make every auto run unacceptably slower without documentation.

Update `autoalgo::Result` and reporting so the exact candidate set and winner remain
machine-readable. Test identical starts, clean per-run state, equal declared budgets,
cancellation, divergence, tie policy, and the fact that auto is a selection procedure—not
an optimizer.

### REST and GUI contract

In the same commit as each public option:

- do not change or extend the legacy CLI menus;
- update the GUI visible algorithm selector and disable/refuse incompatible controls;
- update `/api/train` strict parsing and field-specific refusal messages;
- update optimizer names in JSON, progress, reports, OBD, CV, and locked refit artifacts;
- update standalone OBD and nested-CV optimizer choices only for eligible neural methods;
- update async and blocking forms;
- update action logging;
- update `docs/gui_cli_parity.md`;
- update `AGENTS.md` training recipes and interpretation where behavior changes.

Exercise bodyless POST rules, busy responses, cancellation, and strict present/empty token
behavior. Existing numeric tokens remain accepted exactly.

### Clone, save, continuation, and structural behavior

- Copy persistent optimizer configuration explicitly in `Network::copy` or the owning
  concrete class.
- Working state resets at the next `train()`.
- Saved network formats remain weights/architecture only; no optimizer history/version
  change.
- Continued training begins a new optimizer run, consistent with existing CGD/Shanno
  rebase behavior; document it.
- Input removal and hidden grow/prune are safe because the next run resizes/reset state;
  direct calls that bypass `train()` are not supported unless explicitly initialized.

### Manifest work

Follow `docs/manifest_maintenance.md` in full. For each retained method add:

1. Chapter 2 published derivation, exact citation, assumptions, defaults, and symbol map.
2. A clear distinction between published mathematics and neuron-specific policy.
3. Chapter 12 ownership, complete signatures, state/configuration, mutation, return and
   failure behavior, stopping/cancellation, reset/copy semantics, and example.
4. Matrix/vector primitive entries for every new public numerical operation.
5. CLI and REST tables, eligibility, defaults, strict parsing, async behavior, and errors.
6. Performance evidence and the workload scope of the claim.
7. Method-level index entries and `tools/check_manifest_index.py` updates.
8. Object figure updates only if public ownership/classes changed.

Rebuild the PDF and inspect every affected page as an image.

### Public integration tests

At minimum update/add:

- optimizer unit/formula tests;
- `tests/props` and `tests/backprop/check_bpoptimizer.cpp`;
- gradient cadence and stop-reason tests;
- quiet/copy/prepare tests;
- auto-step incompatibility tests;
- autoalgo selection tests;
- OBD and cross-validation tests;
- stepwise Logistic tests for IRLS;
- clone/configuration tests;
- GUI smoke, async lifecycle where touched, and strict parsing;
- golden sessions only for intentionally changed menu text, never to mask numerical drift;
- oracle verification for behavior expected to remain unchanged.

New tests must prove the actual selected optimizer reaches the weights. A dispatch label or
finite result is not enough; `check_bpoptimizer.cpp` documents the prior false-positive
failure mode.

### Phase 5 complete gates

Run proportional focused tests during development, then before the integration commit:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
tests/golden/run_golden.sh
tests/gui/smoke.sh
tests/gui/asyncjob.sh
tests/gui/strictparse.sh
tests/oracle/verify_oracle.sh
tests/tools/run_tools.sh
python3 tools/check_manifest_index.py
cd docs/tex
dot -Tpdf figures/objects.dot -o figures/objects.pdf
latexmk -pdf manifest.tex
git diff --check
```

Inspect affected Manifest PDF pages as rendered images. Verify the tests executed rebuilt
objects, not a stale binary.

## Phase 6 — conditional later candidates

Do not start this phase merely because earlier phases are complete. Start only from a
measured remaining bottleneck.

### Levenberg-Marquardt

Trigger: small LMS networks remain a major wall-clock consumer after accepted batch
optimizers.

Scope: `OneHiddenNet` first, strict LMS eligibility, measured parameter/Jacobian-memory
ceiling, direct solve through Matrix, no cross-entropy analogy. Benchmark row and parameter
crossover. Preserve per-exemplar equations in the model layer.

### Adam versus AMSGrad versus Nesterov

Trigger: online training is a real important workload and canonical online eta is the
measured bottleneck.

Prototype original Adam and AMSGrad as separate exact algorithms; include Nesterov as the
cheap control. Do not select AMSGrad solely from theory or Adam solely from popularity.
Use explicit absolute-step semantics, exact bias-correction/denominator rules, and per-run
moment reset.

### SVRG

Trigger: row-count profiling shows repeated full gradients dominate and online/noisy
variance prevents ordinary SGD from reaching the desired endpoint efficiently.

Requires a seeded sampling-plan service and model-owned no-allocation exemplar gradient.
This is a separate architecture project; do not retrofit it into a generic callback loop.

## Global acceptance criteria

A candidate may become public only if all are true:

1. It implements one exact cited algorithm; no neighboring variant is hidden under its
   name.
2. It reaches a comparable objective and model endpoint.
3. Its wall-clock improvement exceeds observed run-to-run spread on a predeclared useful
   workload.
4. The result is robust across relevant seeds, scaling, and model sizes.
5. Added memory and per-pass/evaluation costs are measured.
6. Numerical failures and eligibility are explicit and tested.
7. Raw-gradient stopping semantics remain correct.
8. Cancellation, ceiling, quiet mode, cloning/config copy, per-run reset, reports, OBD,
   CV, stepwise, and auto-selection remain defined.
9. The public benefit justifies permanent REST/GUI, documentation, and test burden.
10. Manifest, index, REST/GUI contract, and operational docs are synchronized.
11. New guards have fail-proven sabotage and fresh recompilation evidence.
12. Relevant full gates pass in Release.

## Definition of done for the program

The program is complete when:

- baseline evidence is reproducible;
- BB, corrected IRLS, L-BFGS, and iRPROP+ each have an explicit retain/research-only/reject
  decision supported by matched-endpoint timing distributions;
- disposable rejected implementations are removed;
- measured winners, if any, are fully integrated with bounded family-aware auto-selection;
- existing defaults and numeric optimizer tokens remain compatible unless a separately
  authorized change says otherwise;
- all required tests, parity docs, Manifest sections, index gates, and rendered PDF checks
  pass;
- `docs/HISTORY.md` records negative as well as positive findings;
- the working tree contains no temporary data, benchmark dumps, sabotage, generated logs,
  or stale binaries presented as evidence.

## Suggested `CLAUDE.md` pointer while active

When this becomes the active implementation program, add a concise current-state pointer
to the repository `CLAUDE.md`, for example:

> Current optimizer work follows `docs/learning_research/optimizer_implementation_plan.md`.
> Resume at the first incomplete phase and do not expose a candidate publicly before its
> matched-endpoint acceptance gate.

Update that pointer with the current phase and last completed evidence commit as work
progresses; do not copy the entire plan into `CLAUDE.md`.
