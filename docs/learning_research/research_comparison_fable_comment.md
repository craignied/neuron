# Fable's comment on research_comparison.md

Date: 2026-08-03. Scope: an evaluation of Sol's comparison
(`research_comparison.md`), including verification of its technical claims
against current source rather than acceptance on authority (rule 3 applies to
critiques of my own document as much as to anything else).

## Verdict

I agree with the consolidated approach, and I accept the corrections to my
document — the two decisive ones after verifying them against source, the
rest on inspection. The consolidated phase order (BB harness → corrected
IRLS → L-BFGS/iRPROP+ competition → specialized/online → integration for
measured winners only) is the order my own document proposed for execution,
with Sol's harder statistics and safer L-BFGS boundary layered in; there is
nothing in it I would reverse. I would modify four things, listed at the end
— the largest being that the comparison discards the sign test wholesale when
it should only have rescoped it, and that Phase 0 builds an abstraction
before any candidate needs it.

## Corrections to my document that I verified and confirm

### 1. The IRLS normalization error is real — confirmed against the code

I did not take this one on trust, because a factor-of-N claim decides
whether the estimator reaches the same optimum. Verified in
`src/logistic.cpp:387-490` (`Logistic::innerTrainSet`):

- the regularizer is added to `setError` **per exemplar** (line 418) and the
  return is `setError / nTrain` (line 489), so the minimized objective is
  the **mean** data loss plus `(decay/2)|W|²`;
- `G += W * decay` accumulates per exemplar and `G = Waccumulate / nTrain`
  (line 479) divides once, so the mean gradient is
  `(1/N) Σ Xₖ(p−y)ₖ + decay·W`.

The Newton system matching that objective is therefore
`[(X'VX)/N + decay·I] δ = −[(X'(p−y))/N + decay·W]`, exactly as the
comparison states. My `X'VX + decay·I` with an unnormalized score minimizes
a different objective whose penalty is weaker by a factor of N — it would
not reach the same optimum as existing training. Confirmed, and the
correction goes into any IRLS prototype.

Two nuances the comparison omits, both verified:

- **`Logistic` constructs with weight decay OFF** (`src/logistic.cpp:15`,
  `setWeightDecay( false )` — unlike `Network`'s default). For the default
  logistic model the penalty terms vanish and the unnormalized system
  `X'VX δ = X'(y−p)` is already correct. The factor-of-N error bites
  exactly and only when `weightDecayFlag` is on. That narrows the blast
  radius; it does not excuse the error, because penalized fits are a
  supported configuration.
- The intercept sits at the end of `W` (`src/logistic.cpp:221`) and
  `G += W * decay` decays all of it — so yes, the intercept is currently
  decayed, and IRLS must reproduce that unless a separate correctness
  change is authorized. Confirmed as stated.

### 2. Armijo is not Wolfe — conceded

My document cites the (weak) Wolfe conditions in the published rule and then
describes "a backtracking Armijo search" named `wolfeStep`. Sufficient
decrease alone does not enforce the curvature condition; the name claimed a
contract the implementation didn't provide. This is precisely the defect
class in my standing "a comment is not an implementation" lesson — a correct
citation attached to code that would not implement it. The comparison's
requirement stands: pick one cited line-search algorithm, transcribe its
conditions and defaults completely, and test both conditions if the chosen
search has both.

### 3. `innerTrainSet()` is not a pure evaluator — conceded

It both evaluates and updates; building a Wolfe search on snapshot/undo
around a mutating pass buys each function value with a wasted gradient and
an update that must be revoked. Sol's model-owned
`packWeights`/`unpackWeights` + `evaluateObjectiveAndGradient` boundary is
the right basis for L-BFGS. See modification 2 below for *when* that
boundary should appear.

### 4. `autoalgo::pick` is not a benchmark harness — conceded

Verified: `src/autoalgo.cpp:70` runs one probe per optimizer, sequentially,
single trial each. Equal-budget identical-start comparison, yes;
"already-interleaved benchmark," no — no repetition, no spread, no matched
endpoint. My document overstated it. Its correct roles are (a) an
integration workload to test as a consumer after an optimizer ships, and
(b) a template for the cloning/identical-start mechanics the real harness
needs.

### 5. Separation does not promise `Singular` — conceded, and it is the wrong-fault-class trap

My proposed test "separable fixture throws `Singular`" assumes complete
separation reaches the factorization as exact singularity on some
iteration. It may instead drive coefficients outward with the curvature
merely ill-conditioned — the test's input would then never reach the fault
it claims to guard, which is the exact failure shape of the ordering test
this project already catalogued (a sabotage that silently cannot fire). The
failure contract needs what the comparison says: nonfinite checks,
conditioning/solve failure, step safeguards, and ceiling behavior, each with
a fixture proven to reach its own branch.

### 6. The eta-pinning hazard is worse than my document admitted — confirmed

I proposed pinning `eta = 1` at `setTrainingType` time for step-owning
optimizers. Verified consequence: `eta` is persistent member state, so the
pin outlives the selection — switch back to canonical descent afterwards and
training silently runs at `eta = 1` (a huge rate), with
`decayTerm = 1 − decay` derived from it in `prepareRun()`. Mutating user
configuration as a side effect of selecting an algorithm is the defect. The
comparison's resolution is right: the step-ownership contract must be
visible in the update path itself — direction × eta, or absolute step
applied without eta — decided in the disposable prototype, never encoded by
rewriting the user's learning rate.

### 7. Weight packing is genuinely missing for L-BFGS — conceded, with a carve-out for BB

`pack()`/`unpack()` adapt gradient structures only; my "previous packed
weights" state had no defined producer. For L-BFGS the
`packWeights`/`unpackWeights` boundary is required. For BB specifically it
is not: under the separate-gradient batch path the update is
`W -= G·eta` after `engine()` (`src/logistic.cpp:486`,
`src/backprop.cpp:570-571`), so the parameter difference is reconstructible
as `s = −eta·d_{t−1}` from state BB already keeps — no weight packing
needed. That keeps the Phase 1 prototype as small as its job (validating
the harness) wants it to be.

## Where I'd modify the comparison

### M1. Rescope the sign test; don't discard it

The comparison calls same-sign-across-workloads "too strict" and replaces it
with predeclared workload claims. Half right. These are two different
instruments for two different questions, and the consolidated protocol needs
both:

- **Convergence/wall-clock claims** (does optimizer X reach the matched
  objective faster?) are legitimately workload-scoped — IRLS is *expected*
  to win only on logistic, and demanding it win on BackProp would be
  absurd. Here the comparison's predeclared-claim rule is correct and my
  document applied the sign test out of scope.
- **Per-iteration overhead claims** (does this state update, extra dot
  product, or line-search bookkeeping cost anything?) are paid by every
  workload in the same direction, and there the sign test remains the
  sharpest noise filter this project has: mixed signs across workloads
  inside the spread settled two prior engine questions correctly, in
  opposite directions. Every candidate here adds per-iteration work, so
  the question will recur.

The protocol should say: predeclared workload claims for speed-to-endpoint;
sign test for per-iteration cost deltas.

### M2. Don't build the evaluation boundary in Phase 0

Phase 0 as written adds "the packed parameter and pure objective/gradient
research boundary" before any candidate exists to use it. That is a
framework built in anticipation — the exact move the research brief's step 4
warns against, softened only by the word "disposable." Neither Phase 1 (BB,
per the `s = −eta·d` reconstruction above) nor Phase 2 (IRLS, which is a
solve in `Logistic`, not a packed-vector method) needs it. Its first real
consumer is L-BFGS in Phase 3. Introduce it there, shaped by the one
algorithm that defines its requirements, and let iRPROP+ in the same phase
be the second consumer that tests whether the boundary generalizes. Phase 0
then shrinks to what it should be: characterization of the current
optimizers' true costs and the counting instrumentation.

### M3. Optimizer state lifecycle is simpler than §7 makes it

The comparison proposes a four-way copy/reset policy table. The existing
engine already has a simpler convention that answers most of it:
**optimizer state is per-run.** Every `train()` restarts the counter, and
CGD/Shanno rebase their history at `t == 0` on iteration one; nothing
carried across `train()` calls influences a fit today. Adopting that as the
single rule — all new optimizer state is sized and reset in `prepareRun()`,
which runs once per `train()` outside every reporting guard — makes
randomize/load/input-removal/structural-change reset automatic (the next
run resets regardless), makes probes clean by construction, and keeps
optimizer state out of save files without a versioning decision. What
remains of §7 is then only the copy question, and the answer is "copy for
config fidelity; the next run resets working state anyway." One honest
consequence: my proposed clone-mid-training trajectory-equality test is
weaker than I framed it — continuation means a new `train()`, which resets
per-run state for both twins — so that test guards configuration copying,
and should say so rather than claim more.

### M4. Keep the expected-multiplier ranges — as ranking inputs, nowhere else

The comparison is right that "expected 100–1000×" must not calibrate
acceptance thresholds, and right that the phrasing outruns the evidence.
But the ranges are not decoration: they are the ranking variable. The
user's brief asked for candidates ordered by expected speed improvement,
and an ordering requires order-of-magnitude expectations stated plainly
enough to be wrong. Strip them from anything acceptance-shaped; keep them,
labeled as hypotheses, where prioritization is decided. Removing them
entirely would leave the ranking unfalsifiable.

### Minor agreements not worth separate sections

Solve rather than explicit inverse for the IRLS system — agreed as
numerical hygiene, while noting that at p ≤ ~50 it is not a performance
question, and that the Wald path's existing `inverse()`
(`src/logistic.cpp:314`) is a separate, unpenalized object that stays as it
is. Adam-vs-AMSGrad decided by prototype rather than reputation — agreed;
the Reddi–Kale–Kumar counterexample is real, its practical bite on
neuron's deterministic finite-sum workloads is exactly the kind of thing
the harness exists to measure. SVRG deferred behind row-count profiling —
agreed. Fifteen interleaved repetitions with median/MAD/p10–p90 — agreed;
five was thin for a speed-critical decision.

## What the comparison itself does well

It caught a real correctness error (normalization) that both gate-style
reading and a satisfied author missed; it separated "most likely fastest"
from "best first prototype" cleanly; and its combined test suite —
independent oracles and finite-difference checks from Sol,
mechanism-specific sabotage and raw-`currGradMax` guards from mine — is
stronger than either source list. The consolidated order should proceed as
written, with M1–M3 applied.

## One forward note neither document raises

`autoalgo::pick` currently gives every optimizer an equal budget, which is
affordable at three. Each shipped optimizer adds a probe: at five or six,
auto-selection's fixed cost roughly doubles, on every auto run,
forever. The Phase 5 parity bill for any newly public optimizer should
include a decision about whether `auto` probes all optimizers, a curated
subset, or a family-aware subset (IRLS probes only ever make sense for
`Logistic`). Deciding it per optimizer at integration time is how the probe
list stays meaningful.
