# iRPROP+: the pre-code source decision

Phase 4 of `docs/learning_research/optimizer_implementation_plan.md`. Written
**before** any implementation, as that plan's "Published contract" step requires.
Nothing here may be revised after a benchmark result is seen; a constant changed
to improve a measured number would make the measurement a description of the
tuning, not of the algorithm.

This candidate is measured against the plan's **standing portfolio panel**, not
against a single winner. L-BFGS is the current speed leader and a reference; it
is not a winner-takes-all retention gate. The retention rule is recorded in
section 8 below, before any arm was run.

## 1. The authoritative sources

| What | Source |
|---|---|
| iRPROP+ itself: the error-dependent backtracking and the sign-flip gradient zeroing | Igel, C. & Hüsken, M. (2000), "Improving the Rprop learning algorithm," in *Proceedings of the Second International ICSC Symposium on Neural Computation (NC 2000)*, ICSC Academic Press, pp. 115–121 |
| The empirical evaluation that fixes iRPROP+ as the recommended variant, and its pseudocode as transcribed below | Igel, C. & Hüsken, M. (2003), "Empirical evaluation of the improved Rprop learning algorithms," *Neurocomputing* **50**:105–123 |
| The original sign-based resilient-propagation update, and the constants `eta+`, `eta-`, `Delta_0`, `Delta_min`, `Delta_max` | Riedmiller, M. & Braun, H. (1993), "A direct adaptive method for faster backpropagation learning: The RPROP algorithm," *Proceedings of the IEEE International Conference on Neural Networks (ICNN)*, pp. 586–591 |

**One named algorithm, no variant flags.** RPROP-, RPROP+, iRPROP- and iRPROP+
are four different published methods. Only **iRPROP+** is implemented. There is
no flag, no mode and no configuration value that turns this code into any of the
other three, because a variant switch is how a benchmark comes to describe a
method nobody named.

## 2. The transcribed table

Igel & Hüsken (2003), the iRPROP+ pseudocode. `E` is the objective, `t` the
iteration, and every branch is per parameter `w_ij`:

```
for each w_ij:

    if  dE^(t-1)/dw_ij * dE^(t)/dw_ij  >  0 :
            Delta_ij^(t)  = min( Delta_ij^(t-1) * etaPlus , DeltaMax )
            dw_ij^(t)     = -sign( dE^(t)/dw_ij ) * Delta_ij^(t)
            w_ij^(t+1)    = w_ij^(t) + dw_ij^(t)

    else if  dE^(t-1)/dw_ij * dE^(t)/dw_ij  <  0 :
            Delta_ij^(t)  = max( Delta_ij^(t-1) * etaMinus , DeltaMin )
            if  E^(t) > E^(t-1) :
                    w_ij^(t+1) = w_ij^(t) - dw_ij^(t-1)
            dE^(t)/dw_ij  = 0

    else if  dE^(t-1)/dw_ij * dE^(t)/dw_ij  =  0 :
            dw_ij^(t)     = -sign( dE^(t)/dw_ij ) * Delta_ij^(t)
            w_ij^(t+1)    = w_ij^(t) + dw_ij^(t)
```

Three properties of that table are the whole reason iRPROP+ is a distinct
method, and each is a separate test in section 7:

1. **The rollback is error-dependent.** RPROP+ reverts the previous step on
   every sign flip; iRPROP+ reverts it **only when the objective got worse**.
   When the objective improved despite the flip, the coordinate takes **no step
   at all** this iteration — it does not revert and it does not advance.
2. **The current gradient is zeroed after a flip**, so the following iteration
   necessarily enters the `= 0` branch and steps by the freshly shrunk
   `Delta` along the *new* gradient's sign. This is the algorithm's memory, not
   a cleanup.
3. **`Delta` is never a learning rate.** It is an absolute step magnitude in
   parameter units. Nothing multiplies it by `eta`.

### Every constant, fixed in advance

| Symbol | Value | Where it comes from |
|---|---|---|
| `Delta_0` (initial `Delta`, every parameter) | `0.1` | Riedmiller & Braun (1993); carried unchanged by Igel & Hüsken (2003) |
| `etaPlus` | `1.2` | Riedmiller & Braun (1993) §3; Igel & Hüsken (2003) use the same |
| `etaMinus` | `0.5` | as above |
| `DeltaMin` | `1e-6` | as above |
| `DeltaMax` | `50` | as above |

None of these is configurable. `Delta_0` in particular is **not** exposed as a
knob: a tunable initial step would make the screen a tuning exercise, and the
plan forbids that. If a later phase wants a sweep it is declared in advance and
run as its own comparison group, exactly as the L-BFGS `m in {5,10,20}` sweep
was.

### What the table leaves undefined, and how that is resolved

- **`dw_ij^(t-1)` after a no-rollback sign flip.** The `< 0` branch assigns no
  `dw_ij^(t)` when the objective improved. This implementation stores **the step
  it actually applied**, which is `-dw_ij^(t-1)` when it rolled back and `0` when
  it did not. That value is never read: the same branch zeroes the gradient, so
  the next iteration's sign product is `0` and takes the `= 0` branch, which does
  not read the previous step. The `< 0` branch cannot fire again until at least
  two iterations later, by which time the `= 0` branch has overwritten it. That
  is a claim about the code, so it is **pinned by a test** (section 7, test 6)
  rather than left as a comment.
- **Iteration 1.** There is no `dE^(t-1)` and no `E^(t-1)`. The previous gradient
  is initialized to zero, which makes every sign product `0` and sends every
  coordinate through the `= 0` branch at `Delta_0` — which is precisely what
  `Delta_ij^(0) = Delta_0` means. The previous objective is therefore unread on
  the first iteration; it is initialized to `+infinity` so that no rollback can
  be expressed even if the branch were somehow reached. Pinned by test 1.
- **`sign(0) = 0`**, so a coordinate with a zero gradient takes no step in either
  stepping branch and its `Delta` is unchanged. Pinned by test 7.

## 3. Declared neuron policy, labelled as such

These are **not** in the papers. They are this engine's decisions, recorded here
so no later reader mistakes one for published mathematics.

| Policy | Decision | Why |
|---|---|---|
| Objective and gradient source | `Network::batchObjectiveGradient()` — the packed boundary Phase 3 built, evaluating the currently installed weights in one traversal | rule 6: one authoritative implementation of the model equations. iRPROP+ adds no second copy |
| Weight decay | included in **both** the objective and the raw gradient, before any sign comparison | the plan says so explicitly; the sign product must be the sign product of the objective actually being minimized |
| Non-finite refusal | a non-finite objective, or any non-finite raw-gradient element, **throws** `IRpropState::NotFinite` before any state is modified and before any weight moves | never silently fall back to gradient descent; optimizer choice is part of the result. The harness already turns an exception out of `train()` into one failure row |
| `Delta` finiteness | there is **no** separate `Delta` refusal branch. `Delta` starts finite, is only ever multiplied by a finite constant and immediately clamped into `[1e-6, 50]`, so it cannot leave that range. The invariant is asserted after every step of every test sequence (test 9) instead of guarding a branch that cannot be reached | an untestable branch is a claim nothing compiles. The invariant is checkable; the branch is not |
| Returned objective | the objective at the point the step **departed from**, `E^(t)` — legacy pre-update reporting, matching canonical, CGD, Shanno and L-BFGS | moving it would silently move what every stopping rule in `Iterative` compares |
| `currGradMax` | `maxabs` of the **raw** gradient as evaluated, taken **before** the table runs and therefore before the sign-flip zeroing | the plan's architecture decision 7. Reading it after the table would let a coordinate zeroed by a sign flip shrink the reported maximum and fake a convergence. Pinned by test 15 |
| Cancellation | one evaluation per iteration, so `Iterative`'s existing between-iteration check is the whole story; no per-trial check is added | there are no trial points. L-BFGS needed one because a line search spends up to twenty full passes inside a single iteration |
| Exception type | `IRpropState::Ineligible`, distinct from `LBFGS::Ineligible` | each optimizer states its own eligibility. Unifying them is a public-surface question for Phase 5, not a research-phase refactor |

## 4. The absolute-step contract

The plan's architecture decision 8 requires an explicit absolute-step path, and
forbids three specific ways of faking one. This implementation adds a **named
method**, not a hot-loop descriptor or an enum consumed inside a loop:

| | contract | applied as | who |
|---|---|---|---|
| a **direction** | what `engine()` produces: a transformed gradient the model then scales | `w -= g * eta` | canonical, CGD, Shanno |
| an **absolute step** | a displacement already in parameter units, sign included | `w += step` | iRPROP+ |

One sign convention: **`w += step`**, because the paper's `dw_ij` already carries
the leading minus sign of `-sign(g) * Delta`. The step is applied once per epoch
through the shared packed-weight boundary, outside every exemplar loop.

Never, and each is separately tested:

- the step is **not** divided by `eta` (test 10);
- `eta` is **not** set to one, and is not written at all (tests 10, 11);
- the automatic step-size search is **refused**, not left running (test 12);
- online mode is **refused**, not silently reinterpreted (test 13).

## 5. Every published symbol mapped to neuron-owned state

The plan forbids coding before this mapping exists.

| Published symbol | Meaning | neuron state |
|---|---|---|
| `w_ij^(t)` | the parameter vector | the packed weights — `hW` row-major then `oW` for `OneHiddenNet`, `Weights[0..n]` flattened for `BackProp`. Reached through the existing `Network::packWeights` / `unpackWeights`; **no new layout is introduced** |
| `E^(t)` | the objective | the mean training-set error plus the current weight-decay penalty, returned by `Network::batchObjectiveGradient()` — the same value `innerTrainSet()` returns, from the same code |
| `dE^(t)/dw_ij` | the gradient | the **raw** mean batch gradient including the decay term, in the same packed layout. Raw: it is the gradient, never a search direction, and it never passes through `engine()`, `stackG`, `lastG` or `lastF` |
| `dE^(t-1)/dw_ij` | the previous gradient | `IRpropState::prevG` — stored **after** the sign-flip zeroing, because that zeroing *is* the algorithm's memory |
| `Delta_ij^(t)` | the per-parameter update value | `IRpropState::delta`, one double per packed parameter |
| `dw_ij^(t)` | the applied step | `IRpropState::prevStep`, and the same vector handed to `Network::applyAbsoluteStep()` |
| `E^(t-1)` | the previous objective | `IRpropState::prevF` |
| `etaPlus`, `etaMinus`, `DeltaMin`, `DeltaMax`, `Delta_0` | the constants | `IRpropState::ETA_PLUS`, `ETA_MINUS`, `DELTA_MIN`, `DELTA_MAX`, `DELTA_INIT` — `static const`, section 2 |
| (initialization) | whether this run has a previous point | `IRpropState::startedFlag` |

**What is deliberately not mapped.** `eta`, `stackG`, `lastG`, `lastF`,
`deltaError`, `gamma` and `maxLoops` belong to other algorithms. iRPROP+ reads
and writes none of them. Sharing one buffer between two published algorithms is
how two methods come to share a defect.

## 6. Ownership

| Owner | Owns |
|---|---|
| `src/irprop.*`, `IRpropState` | the published table, all five state vectors/scalars, the clamps, the rollback decision, the gradient zeroing, the per-run reset, and the research counters. It has **no** model dependency: it is handed an objective and a raw gradient and produces an absolute step, which is exactly why it can be driven by hand in a test |
| `Network::irpropIteration()` | composition only — the eligibility refusals, one `batchObjectiveGradient()` call, `currGradMax` from that raw gradient, one call into `IRpropState`, and one `applyAbsoluteStep()` |
| `Network::applyAbsoluteStep()` | the one absolute-step application path, shared by any later step-owning optimizer |
| `Iterative` | stopping, cancellation, the iteration counter, reporting — unchanged |

`IRpropState` does not go through `engine()`, for the same reason L-BFGS does
not: `engine()` is a direction-transform dispatch point whose caller applies
`w -= g * eta`, and an absolute step cannot honestly fit that contract.

Working state resets in `Network::prepareRun()`, beside the existing
`lbfgs.reset()`. There is **no persistent iRPROP+ configuration** — every
constant is published and fixed — so there is nothing for `Network::copy` to
carry, and a copy therefore starts a clean run. Both facts are tested (test 16).

Research-only: `Network::TRAIN_IRPROP = 4` is an internal `trainingType` value.
**No REST field, no GUI control, no menu token, no automatic-selection entry and
no saved-network field is added in this phase.** The public identifier, if the
method is retained, is assigned in Phase 5.

## 7. The deterministic tests, declared before the code

`tests/network/check_irprop.cpp`, ctest case `network_irprop`. Numbers 1–13 are
the plan's own Phase 4 list; 14–18 are what this contract adds.

| # | Test | What it would catch |
|---:|---|---|
| 1 | Hand-computed multi-coordinate sequence: three coordinates driven through `> 0`, `< 0` and `= 0` products in the same iteration, with `Delta`, applied step and installed weights asserted exactly at each of several iterations. Iteration 1 asserted to take the `= 0` branch at `Delta_0` | any branch transcribed wrongly |
| 2 | Growth clamp: repeated same-sign products drive `Delta` to exactly `50` and hold it | a missing `min` |
| 3 | Shrink clamp: driven to exactly `1e-6` and held. Also asserts that shrinking takes **two** iterations per flip, because the zeroed gradient sends the intervening one through the `= 0` branch | a missing `max`; a zeroing that does not happen |
| 4 | Rollback: sign flip with `E^(t) > E^(t-1)` applies exactly `-dw^(t-1)` | the error test deleted or inverted |
| 5 | No-rollback: sign flip with `E^(t) <= E^(t-1)` applies exactly `0` to that coordinate and leaves its weight bit-identical, while `Delta` still shrinks | RPROP+ smuggled in under the iRPROP+ name — the single most likely defect |
| 6 | Sign-flip gradient zeroing: `prevGradient(i) == 0` after the flip, **and** the next iteration provably takes the `= 0` branch (its step magnitude is the shrunk `Delta`, not a further-shrunk one and not a rollback) | the zeroing removed; the "unreachable `dw^(t-1)`" claim in section 2 |
| 7 | Zero raw gradient: `sign(0) = 0` ⇒ no step, `Delta` unchanged; an all-zero gradient leaves every weight bit-identical | a `sign` that returns `+1` at zero |
| 8 | Non-finite refusal: NaN objective, NaN gradient element and Inf gradient element each throw `NotFinite` with the state and the weights untouched | a diverged run stepping on garbage |
| 9 | `Delta` invariant: after every step of every sequence above, every `Delta` is finite and within `[1e-6, 50]` | the branch section 3 declines to write |
| 10 | eta independence: two identical models from identical starting weights, `eta = 0.05` and `eta = 7.3`, produce **bit-identical** weights after the same number of iRPROP+ iterations | a step divided or multiplied by `eta` anywhere |
| 11 | Canonical eta preservation: iRPROP+ run, then `getEta()` is bit-identical to what was configured, and a subsequent canonical run behaves as it would have without the iRPROP+ run | `eta` mutated to express an absolute step |
| 12 | Automatic step-size incompatibility: refused by name, before any weight moves | two step rules owning one iteration |
| 13 | Batch-only refusal: online mode refused by name | a per-exemplar update wearing a batch method's name |
| 14 | No packed boundary: a model reporting `packedSize() == 0` is refused | a zero-length parameter vector reported as convergence |
| 15 | Raw-gradient `currGradMax`: on an iteration whose table zeroes coordinates, `getGradMax()` equals `maxabs` of the **raw** gradient computed independently, not of the zeroed one | a stopping rule that fakes convergence after a sign flip |
| 16 | Per-run reset and copy: a second `train()` starts at `Delta_0` with a zero previous gradient; so does a copied network | a run inheriting the previous run's `Delta` |
| 17 | Pre-update return association: the value `trainSet()` returns equals `batchObjectiveGradient()` evaluated at the weights **before** the iteration | the reporting cadence moved |
| 18 | Fixed-start real-model integration: SimpleProp, BareProp and BackProp each run from a fixed seed to a declared objective or report a clear failure, with every weight finite and the objective materially reduced | a method that is correct on hand vectors and inert on a model |

**Fail-proof, per the plan's eight-step discipline.** Two sabotages, run
separately, each with build output retained showing the affected translation
units recompiling both on injection and on restoration:

- **(a)** delete the `E^(t) > E^(t-1)` test so the `< 0` branch always reverts —
  turning iRPROP+ into RPROP+. Test 5 must fail, and test 5 is the assertion that
  names the mechanism.
- **(b)** delete the current-gradient zeroing. Tests 3 and 6 must fail, and the
  named assertion inside test 6 must be among the failing lines.

A sabotage that fails a file for some other reason has proven nothing. The
failing assertion is read, not merely the file's exit status.

## 8. The screen, and the retention rule — both declared before any arm was run

### The cheap representative workload

The identical workload, split and endpoint the standing panel already lives on:
**Civic Choice, 6,000 rows, SimpleProp at the walkthrough's own four hidden
units, the committed 25% stratified holdout, the committed practical objective,
identical starting weights.** One comparison group, axis `optimizer`:

| arm | role in the panel |
|---|---|
| iRPROP+ | the candidate |
| L-BFGS | the current speed leader |
| Shanno | the legacy quasi-Newton control |
| canonical | the behavioral and matched-objective reference |

Canonical is included here although the Phase 3 screen omitted it: the portfolio
policy names it a standing panel member, it is the source of the matched
endpoint, and at 6,000 rows one run costs ~19 s, so the whole arm is a few
minutes. CGD is not a panel member and is not run.

### What is reported per arm

Elapsed median / MAD / p10 / p90; **full training-set traversals** (not outer
iterations — iRPROP+ makes exactly one traversal per iteration, and the harness
counts what the run did rather than what the method is said to do); achieved
objective; held-out error and ROC at the endpoint; stop reason; `converged`;
failures; peak RSS. **Early-stage speed is reported separately from time to the
matched endpoint**, as the plan requires.

### Predeclared additional arms

1. **A poorly scaled neural fixture**, which is where iRPROP+'s hypothesized
   advantage lives and which Step 0B deliberately deferred to "the phase that has
   a candidate to discriminate". Its own comparison group and its own endpoint,
   characterized from a canonical control before any candidate arm runs.

   **Corrected before it was built, and the correction matters.** The first
   construction declared here was "a fixed per-column scale factor applied to the
   groomed file". That fixture would have been **bit-identical to the plain one**:
   `DataSet::normalize` (`src/dataset.cpp:692`) min-max normalizes every input
   column onto `[inLowerLimit, inUpperLimit]`, so any per-column *linear* rescale
   of the raw data is exactly cancelled before training sees it. A benchmark arm
   built that way would have reported "no difference on poorly scaled data" from a
   fixture that was not poorly scaled — a vacuous result wearing an informative
   name.

   What survives min-max normalization is a change to the *distribution* within a
   column, not its units. The fixture is therefore a fixed monotone per-column
   transform applied before grooming — `x -> exp(k_j * x)` with `k_j` cycling over
   a declared set — which leaves the rows and the outcome column untouched, keeps
   the problem learnable because the transform is monotone, and leaves columns
   with genuinely different effective spreads after normalization. The
   construction is verified non-vacuous by requiring the fixture's `split`
   identity to **differ** from the plain fixture's before any arm is timed on it.
2. **Weight-seed robustness** at the three seeds Phase 3 predeclared —
   101, 202, 303 — each its own comparison group, panel iRPROP+ / L-BFGS /
   Shanno.
3. **The late-stage question**, asked the only way this workload permits: the
   neural workload has no strict endpoint because canonical does not converge on
   it, so all three run to the engine's own plateau rule with `endpoint: none`,
   and **where each lands is read alongside how fast it got there**. A method
   that stops earlier at a worse objective has not won.
4. **Scaling to 25,000 and 100,000 rows only if iRPROP+ is still a plausible
   portfolio candidate after the 6,000-row screen.** The plan's staged gate: an
   obvious loser is rejected cheaply, not scaled.

### The retention rule, fixed now

Retain iRPROP+ if it is **correct** (the mechanism tests pass and are
fail-proven), **stable** (no failures across the predeclared seeds), **reaches
comparable endpoints**, and is **either** reasonably competitive with the panel
**or** adds a meaningfully different robustness or workload profile — for
instance holding up on the poorly scaled fixture where a quasi-Newton method
degrades, being more seed-robust, or landing at a better late-stage objective.

**It does not have to beat L-BFGS.** Reject it only for a clear performance,
stability, endpoint or redundancy failure: prohibitively slow, unstable,
unable to reach the matched endpoint, or demonstrably redundant with an existing
panel member and adding no distinct operational value.

Retention and default-ranking are recorded as **two separate decisions**, per the
portfolio policy.
