# Comparison of the Sol and Fable learning-algorithm research

Date: 2026-08-03

Documents compared:

- `research_sol.md` (Sol)
- `research_fable.md` (Fable)

## Executive finding

The documents agree on the central engineering conclusion: neuron's most promising
general-purpose batch-training experiments are L-BFGS, iRPROP+, and Barzilai-Borwein
(BB), and none should become a permanent public option before a fixed-start,
matched-objective wall-clock benchmark demonstrates a real advantage.

Their most important difference is scope. Fable includes model-specific IRLS/Newton
training for logistic regression and ranks it first. Sol deliberately treats
logistic-specific Newton/IRLS as a separate future solver study and ranks L-BFGS first
among broadly applicable optimizers. Fable's inclusion is valuable: IRLS is plausibly the
largest opportunity for repeated logistic fits in stepwise regression and cross-validation.
However, Fable's claimed 100–1000x expected improvement is not yet measured, and its
penalized update as written does not preserve neuron's current mean-loss normalization.

The best combined plan is therefore:

1. Build the shared deterministic benchmark and pure packed objective/gradient boundary.
2. Use BB as the smallest harness-validation prototype.
3. Prototype corrected, safeguarded IRLS specifically in `Logistic` and measure complete
   stepwise/CV workloads.
4. Prototype L-BFGS and iRPROP+ for full-batch neural and logistic workloads, retaining
   only measured winners.
5. Consider LM only for measured small-LMS workloads, and stochastic methods only if
   large-row or online profiling shows that they address a real bottleneck.

This combines Fable's workload-aware IRLS insight and source specificity with Sol's safer
optimizer boundary, stronger benchmark design, and more cautious performance claims.

## Side-by-side scope and ranking

| Rank | Sol | Fable | Finding |
|---:|---|---|---|
| 1 | L-BFGS | IRLS / penalized Newton for Logistic | Different objective: broad optimizer versus family-specific solver. IRLS should be added to the research program, but not treated as evidence-backed fastest until measured. |
| 2 | iRPROP+ | L-BFGS | Strong agreement that both deserve an early batch prototype. |
| 3 | Safeguarded BB | iRPROP+ | Same top general-purpose trio, slightly different order. |
| 4 | LM | BB | Sol ranks LM by possible niche speed; Fable ranks BB higher by implementation simplicity and breadth. BB is the better first harness prototype; LM may still be faster within its eligible niche. |
| 5 | Nesterov momentum | Adam | Both target online/stochastic training later. Fable chooses original Adam; Sol avoids it in favor of AMSGrad because of Adam's known convergence counterexample. |
| 6 | SVRG | Momentum/Nesterov | Sol uniquely covers a variance-reduced large-data method; Fable uniquely treats classical heavy-ball momentum explicitly. |
| 7 | AMSGrad | LM | Both retain LM as specialized/research-only in practice despite the numerical rank difference. |

Fable also separates **expected-speed ranking** from **implementation order**: it proposes
building the harness with BB before implementing its first-ranked IRLS. That is sensible.
The comparison should not confuse “most likely to be fastest” with “best first disposable
prototype.”

## Major agreements

### 1. Measurement, not iteration count, decides

Both require fixed seeds, identical starting weights, comparable endpoint objectives,
stop reasons, failure counts, and elapsed time. Both correctly reject an apparent speedup
that merely stops at a worse objective. Both also require negative results to be retained
as project evidence rather than quietly discarded.

### 2. The initial broad batch shortlist is stable

Both independently place L-BFGS, iRPROP+, and BB near the top. This convergence is more
important than their exact order:

- L-BFGS offers limited-memory curvature information and likely reduces expensive full
  passes, at the cost of line-search evaluations.
- iRPROP+ has exceptionally cheap per-epoch arithmetic and removes global-eta sensitivity.
- BB is the smallest spectral-step experiment and a useful control for determining how
  much of the current cost comes from step-size selection.

### 3. LM is specialized

Both restrict Levenberg-Marquardt to small least-squares networks because its Jacobian,
normal matrix, and solve costs grow rapidly. Neither recommends applying it to neuron's
cross-entropy logistic path by analogy.

### 4. Public integration has a large parity bill

Both identify the CLI menu, GUI algorithm control, `/api/train`, strict parsing,
`autoalgo`, OBD, CV, run headers, cloning, tests, `AGENTS.md`, and
`docs/gui_cli_parity.md`. Both recognize that shipping several near-duplicate optimizers
would create permanent documentation and testing cost.

### 5. Published equations must remain visible

Both reject a generic callback/descriptor framework that makes unlike algorithms appear
identical. Both route missing reusable arithmetic through destination-taking
`vector_ops`/`Matrix` primitives and keep allocation and indirect dispatch out of exemplar
loops.

## Where Fable is stronger

### Workload-aware IRLS proposal

Fable notices that a model-specific solver can matter more than a general optimizer when
the model is repeatedly refit. Stepwise regression and cross-validation multiply logistic
training cost, so measuring IRLS end-to-end there is a strong recommendation. It also
correctly assigns IRLS to `Logistic::trainSet()` rather than forcing a matrix solve through
`Network::engine()`.

### Concrete current-code audit

Fable names current members, switch sites, GUI mappings, raw-gradient reporting, clone
state, and the need for every new direction method to set `currGradMax` from the raw
gradient. Its warning about preserving `getGradMax()` semantics is particularly useful:
stopping must not begin testing a transformed search direction.

### Bibliographic completeness

Fable supplies a consolidated bibliography for the existing Golden methods and adds
important sources that Sol's per-algorithm citations omit:

- Nocedal (1980), Nocedal and Wright, and Wolfe for the L-BFGS recursion and line search.
- Raydan (1997) for BB globalization/safeguards.
- Levenberg (1944) and Marquardt (1963), not only the neural application paper.
- Polyak (1964) and Nesterov (1983), not only the later neural momentum application.

Those sources should be retained in any final research specification.

### Mechanism-focused test details

Fable explicitly calls for clone-mid-training equivalence, quiet/audible invariance,
raw-gradient maximum checks, and sabotage of specific branches such as iRPROP+ rollback.
Sol covers most of the same territory at a broader level, but Fable's examples are easier
to turn directly into regression tests.

## Where Sol is stronger

### Correct L-BFGS service boundary

Sol recognizes that L-BFGS is not merely another direction transformation. A Wolfe line
search needs repeated objective and gradient evaluations at candidate parameters, step
acceptance/rejection, restoration, and evaluation counts. The current
`Network::engine()` is called only after a concrete model has formed a gradient, and the
concrete model applies the weight update afterward. It does not own a pure trial-point
evaluation contract.

Sol's proposed model-owned `packWeights`/`unpackWeights` and
`evaluateObjectiveAndGradient` boundary is therefore the safer basis for L-BFGS. Fable's
proposal to call a `wolfeStep` from `engine()` through `innerTrainSet()` has two problems:

1. `innerTrainSet()` performs weight updates; it is not a pure function/gradient evaluator.
2. Fable describes backtracking Armijo while naming the operation Wolfe. Armijo decrease
   alone does not satisfy the Wolfe curvature condition, which requires a directional
   derivative at the trial point.

The common evaluation boundary should be introduced only after a prototype proves value,
but the production L-BFGS design needs it.

### Stronger benchmark statistics

Sol proposes at least 15 interleaved repetitions, median, MAD, p10/p90, peak memory, and
separate objective/gradient evaluation counts. Fable asks for at least five repetitions
and a same-sign-across-all-workloads rule. Five may be too few for a speed-critical
decision, while requiring the same sign on every workload is too strict: an optimizer can
be legitimately valuable for a clearly defined family or scale and slower elsewhere.

The right acceptance rule is a predeclared workload claim with improvement larger than
the observed spread, a comparable endpoint, and no material reliability regression.

### More cautious performance language

Sol consistently labels the ranking provisional and generally avoids numerical speedup
multipliers. Fable labels its numbers hypotheses, but headings such as “expected
100–1000x” and “expected 5–20x” look much more precise than the cited evidence or current
measurements justify. Those ranges should not guide acceptance thresholds.

### Safer stochastic candidate

Sol chooses AMSGrad instead of ordinary Adam and cites Reddi, Kale, and Kumar's Adam
nonconvergence result. Fable's Adam proposal is useful as an online baseline, but a new
permanent implementation should compare Adam and AMSGrad in a disposable oracle/prototype
before selecting either. It should not assume popularity implies the best time to a
reliable endpoint.

### Large-data coverage

Sol includes SVRG and correctly makes it conditional on row-count profiling. Fable does
not cover a variance-reduced finite-sum method. SVRG should remain deferred, but it is the
more relevant later candidate if large full-gradient passes are measured as the dominant
cost.

## Technical issues requiring correction or resolution

### 1. Fable's penalized IRLS normalization does not yet match neuron

`Logistic::innerTrainSet()` minimizes a **mean** data loss plus the weight penalty:
the accumulated gradient is divided by `nTrain`, while `decay * W` remains once in the
resulting mean gradient. Therefore an equivalent penalized Newton system is, using the
minimization sign convention,

`[(X' V X) / N + decay I] delta = -[(X' (p-y)) / N + decay W]`.

Equivalently, after multiplying the whole system by `N`:

`[X' V X + N decay I] delta = X' (y-p) - N decay W`.

Fable currently writes `X' V X + decay I` with an unnormalized data score. That changes
the relative regularization strength by a factor of `N`, so it would not reach the same
penalized optimum as existing training. The intercept is currently part of `W` and is
also decayed; IRLS must preserve that existing behavior unless a separate correctness
change is authorized.

IRLS should solve the linear system directly through an authoritative `Matrix`/GSL
factorization rather than compute an explicit inverse in the hot training path. The
unpenalized `X'VX` used for Wald covariance and condition-number reporting remains a
different statistical object; sharing its construction is good, but the penalized
training curvature must not leak into the reported design diagnostic.

Finally, complete separation is not guaranteed to throw `Matrix::Singular` on a chosen
iteration. It may instead drive coefficients toward infinity while curvature becomes
ill-conditioned. Tests and failure contracts need nonfinite checks, conditioning/solve
failure, step safeguarding, iteration ceiling behavior, and endpoint comparison—not a
single assumption that separation always produces `Singular`.

### 2. L-BFGS line-search details must be one coherent published method

Sol requests a strong-Wolfe search but cites only the L-BFGS paper in that section; Fable
adds the needed Wolfe and Nocedal/Wright sources but describes weak Wolfe, then a
backtracking Armijo implementation. The final plan must select one exact line-search
algorithm, cite it, transcribe all conditions and defaults, and test both sufficient
decrease and curvature where applicable. It must not call Armijo-only search “Wolfe.”

### 3. Fable's direction-only `engine()` integration is not universal

The existing `engine()` is appropriate for methods that transform a raw gradient into a
direction. BB can also use it to update a scalar eta. iRPROP+ and Adam/AMSGrad own the
absolute step magnitude, however, while concrete models currently multiply the returned
structure by `eta`.

Fable proposes pinning eta to one for iRPROP+. That makes the current statement produce
the desired magnitude, but it mutates user configuration and creates subtle interactions
with `prepareRun()`'s `decayTerm`, run reporting, optimizer changes, and continued
training. Sol's separate state/service direction is cleaner, though potentially more
invasive. The disposable prototype should compare two explicit contracts:

- `engine()` returns a direction and the model applies eta; or
- a step-owning algorithm returns an absolute packed step and the model applies it without
  eta.

The chosen contract must be visible in types/names, not encoded by dividing by eta or
silently setting eta to one.

### 4. Weight packing is a real missing boundary

Existing `pack()`/`unpack()` adapt **gradient** structures, not model weights. L-BFGS and
BB require parameter differences `s`, and line searches require candidate parameters.
Fable refers to previous packed weights without specifying a common way to obtain them.
Sol explicitly proposes `packWeights`/`unpackWeights`; this is needed unless each
algorithm remains duplicated inside every concrete weight structure, which would violate
single ownership.

### 5. `autoalgo` is useful but is not the benchmark harness

Fable says `autoalgo::pick` is already an interleaved benchmark harness. In current code
it runs each of three optimizers once, sequentially, under one equal wall-clock budget and
selects the lowest resulting training error. It provides identical cloned starts and is a
valuable integration workload, but it does not randomize/interleave repeated trial order,
report run-to-run spread, match a predeclared endpoint, or separate evaluation counts.

Sol's dedicated built-but-not-CTest probe is the better performance-evidence tool.
`autoalgo` should be tested as a consumer after an optimizer succeeds, not used as the
sole benchmark.

### 6. BB safeguards need their own source and exact policy

Sol correctly says raw BB needs globalization but leaves the nonmonotone Armijo source to
be added later. Fable supplies Raydan (1997), which should be used to define the exact
safeguard rather than referring vaguely to “published clamps.” The prototype may start
with raw BB plus explicit alpha bounds, but production acceptance requires a cited and
tested failure/globalization policy.

### 7. State reset versus copy needs a lifecycle decision

Fable generally copies state and proposes clone-mid-training trajectory equality. Sol
allows either copying state for true continuation or explicitly resetting it when the
contract changes. The final rule should distinguish operations:

- Copy/clone for a continuation-equivalent model should copy optimizer state.
- Randomize, load prediction weights, input removal, and hidden grow/prune should reset
  incompatible state.
- Auto-selection probes need identical starts and clean optimizer-specific histories.
- Saved model files currently contain prediction state only; adding optimizer state would
  require explicit versioning and is not recommended initially.

## Source and citation comparison

Both documents cite primary publications for their named methods. Sol has convenient
clickable publisher/proceedings links for every enumerated candidate. Fable's bibliography
is stronger as a scholarly implementation checklist because it distinguishes original
methods, globalization rules, practical neural adaptations, and the existing Golden
baseline.

The consolidated specification should use Fable's fuller references while retaining
Sol's verified clickable URLs. Before implementation, the exact algorithm table—not a
secondary summary—must be obtained for:

- the chosen strong/weak Wolfe line search and interpolation/zoom rules;
- iRPROP+ rollback and defaults;
- Raydan's chosen BB safeguard/globalization variant;
- the chosen Adam-family algorithm (Adam versus AMSGrad);
- LM damping acceptance and adjustment schedule; and
- corrected penalized IRLS with neuron's mean-loss convention.

## Testing comparison and recommended combined suite

Fable's strongest test contributions are mechanism-specific sabotage, raw-gradient
maximum, clone continuation, and quiet/audible invariance. Sol's strongest contributions
are independent dense/quadratic or high-precision oracles, finite-difference gradients,
broader public-surface integration, nonfinite/denominator failures, and trial restoration.

The combined suite should contain:

1. A hand/independent one-step formula oracle for every candidate.
2. A finite-difference objective/gradient agreement test for every eligible model and
   decay setting.
3. Mechanism-specific sabotage with visible recompilation before and after restoration.
4. Raw-gradient `currGradMax` checks after direction/step transformation.
5. Trial rejection/restoration, cancellation during trials, ceiling exhaustion, and
   nonfinite/degenerate arithmetic.
6. Copy-continuation equivalence plus explicit reset tests for randomize/load/structural
   change.
7. Fixed-start model integration for Logistic, SimpleProp, BareProp, and BackProp where
   eligible, in batch and online modes where defined.
8. Auto-selection, OBD, stepwise, CV, blocking/async API, strict parsing, busy gate, GUI,
   and parity tests only after a candidate is approved for public integration.

## Benchmark comparison and recommended protocol

Use Sol's more statistically informative harness with Fable's workload emphasis:

- Release build; training only, excluding reports and ROC bootstrap.
- One serialized/randomized start restored identically to every arm.
- Randomized interleaving of at least 15 repetitions per cell after warm-up.
- Median, MAD, p10/p90, failures, peak memory, iterations, full objective/gradient
  evaluations, exemplar evaluations, and stop reason.
- Primary endpoint: time to a predeclared matched objective; secondary endpoints include
  predictions/weights where appropriate and time to existing stopping rules.
- Workloads: logistic dimension/conditioning sweeps; biased SimpleProp; unbiased BareProp;
  multi-layer BackProp; online and batch when eligible; row-count and parameter-count
  sweeps; poor scaling, collinearity, saturation, separation, and nonlinear fixtures.
- End-to-end multipliers: stepwise logistic regression, repeated CV, OBD, and automatic
  algorithm selection.
- Accept workload-specific wins when the claim is defined in advance and the improvement
  exceeds observed spread. Do not demand universal improvement, and do not promote a
  candidate based on a tuned best run.

## Recommended consolidated research order

### Phase 0: characterize and build the evidence harness

Measure canonical, CGD, Shanno, and legacy automatic step-size costs. Add the packed
parameter and pure objective/gradient research boundary needed to count and compare
identical work. Keep it disposable until a candidate proves the abstraction useful.

### Phase 1: BB harness validation

Implement a non-public BB prototype because it has minimal state and makes a good test of
fixed-start restoration, raw-gradient stopping, pass counting, and safeguard reporting.
Its numerical rank is lower than IRLS/L-BFGS; its implementation order is earlier because
it validates the experiment cheaply.

### Phase 2: corrected logistic IRLS

Prototype IRLS in `Logistic`, using the normalized penalized system above, a direct solve,
step/finiteness safeguards, and a distinct unpenalized information matrix for inference.
Benchmark single fits and complete stepwise/CV procedures. This is the largest plausible
model-specific opportunity and the largest omission in Sol.

### Phase 3: L-BFGS and iRPROP+

Prototype both against the same broad full-batch grid. L-BFGS uses the pure
objective/gradient evaluation boundary and one exact cited line search. iRPROP+ uses an
explicit step-owning contract. Decide whether either or both justify permanent public
options; do not assume they both should ship.

### Phase 4: specialized and online gaps

Prototype LM only if small LMS networks remain an important measured cost. Profile online
and large-row training before choosing Nesterov, Adam, AMSGrad, or SVRG. If noisy online
training matters, compare original Adam and AMSGrad rather than selecting by reputation.
If repeated full gradients dominate very large finite datasets, SVRG is the more targeted
experiment.

### Phase 5: production integration

For measured winners only, complete CLI/GUI/API parity, strict parsing, auto-selection,
OBD/CV/stepwise behavior, cloning/reset semantics, Manifest Chapter 2 mathematics and
Chapter 12 ownership contracts, index gates, PDF inspection, fail-proven tests, and
proportional full verification.

## Final assessment

Neither document should replace the other unchanged.

- Fable contributes the most important additional candidate (IRLS), the best consolidated
  bibliography, exact current-code touchpoints, and useful mechanism-specific guards.
- Sol contributes the sounder L-BFGS architecture, safer treatment of optimizer state and
  stochastic convergence, a more rigorous performance protocol, and more appropriately
  cautious claims.

The consolidated order is not simply either ranking. BB should validate the harness;
corrected IRLS should test the largest model-specific opportunity; L-BFGS and iRPROP+
should then compete for broad batch workloads. Only measured winners should alter neuron's
public optimizer surface.
