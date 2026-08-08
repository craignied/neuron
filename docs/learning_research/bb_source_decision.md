# Safeguarded Barzilai-Borwein: the pre-code source decision

Phase 1 of `docs/learning_research/optimizer_implementation_plan.md`, taken out
of order: it is the plan's smallest candidate, and it runs now against a
**four-arm standing panel** (canonical, Shanno, L-BFGS, iRPROP+) that did not
exist when the plan was written.

Everything below was fixed **before any engine code was written and before any
arm was run**, for the reason `docs/optimizer_research.md` gives: a constant
chosen after seeing a benchmark makes the benchmark a description of the tuning
rather than of the algorithm.

## 1. The authoritative sources

- **Barzilai, J. & Borwein, J. M. (1988)**, "Two-point step size gradient
  methods", *IMA Journal of Numerical Analysis* 8(1):141-148 — BB1 and BB2.
- **Raydan, M. (1997)**, "The Barzilai and Borwein gradient method for the large
  scale unconstrained minimization problem", *SIAM Journal on Optimization*
  7(1):26-33 — the globalization, published as **Algorithm GBB**.
- **Grippo, L., Lampariello, F. & Lucidi, S. (1986)**, "A nonmonotone line search
  technique for Newton's method", *SIAM Journal on Numerical Analysis*
  23(4):707-716 — the nonmonotone acceptance rule GBB is built on.

### The access problem, stated rather than papered over

**Raydan (1997) is closed access and no lawful full text could be obtained.**
Verified, not assumed: OpenAlex reports `is_oa: false`, `oa_status: closed`,
`any_repository_has_fulltext: false`; the Semantic Scholar graph reports
`openAccessPdf.status: CLOSED`; and the author's own ResearchGate copy returns
HTTP 403 to automated retrieval.

`docs/optimizer_research.md` step 1 asks for "the primary paper **or
authoritative specification**". The specification below is transcribed from two
obtainable authoritative restatements, and this document says so rather than
implying the 1997 PDF was read:

- **Sun, W. & Yuan, Y.-X. (2006)**, *Optimization Theory and Methods: Nonlinear
  Programming*, Springer, §3.1.3, **Algorithm 3.1.9** ("The Barzilai-Borwein
  gradient algorithm with nonmonotone linesearch") — the GBB skeleton, step by
  step, with every free parameter named. This is the **structure**.
- **Birgin, E. G., Martínez, J. M. & Raydan, M.**, "Spectral Projected Gradient
  Methods: Review and Perspectives", *Journal of Statistical Software*
  (`https://www.ime.usp.br/~egbirgin/publications/bmr5.pdf`), **Algorithms 2.1
  and 2.2** and the "In practice, it is usual to set ..." paragraph on page 6 —
  the explicit interpolation formula and **every numerical value**. Raydan is an
  author of it; it states that SPG "was born from the marriage of the global
  Barzilai-Borwein nonmonotone scheme (Raydan 1997) with the classical projected
  gradient method", and with the feasible set taken as all of R^n the projection
  P_Omega is the identity, so Algorithms 2.1/2.2 reduce to the unconstrained
  method they describe as GBB's descendant.

BB1 itself is additionally confirmed verbatim, in a third independent place, as
equation (1.6) of **Dai, Hager, Schittkowski & Zhang (2006)**, "The cyclic
Barzilai-Borwein method for unconstrained optimization", *IMA J. Numer. Anal.*
(`https://www.math.lsu.edu/~hozhang/papers/cbb.pdf`), and as Sun & Yuan (3.1.30).

The three renderings agree. Where they differ in presentation — Raydan bounds
`alpha` and resets it to `delta`, while SPG clamps `lambda` and has a separate
`s'y <= 0` branch — **this implementation follows Raydan's**, because Raydan's is
the unconstrained method and is the one being cited. Section 3 shows the two are
the same safeguard.

## 2. The transcribed algorithm

Sun & Yuan Algorithm 3.1.9, in their notation, with `g_k = grad f(x_k)`:

```
Step 0.  Given x_0, 0 < eps << 1, integer M >= 0, rho in (0,1), delta > 0,
         0 < sigma1 < sigma2 < 1, alpha_l, alpha_u.  Set k = 0.

Step 1.  If ||g_k|| <= eps, stop.

Step 2.  If alpha_k <= alpha_l or alpha_k >= alpha_u, then set alpha_k = delta.

Step 3.  Set lambda = 1 / alpha_k.

Step 4.  (nonmonotone line search)  If

             f(x_k - lambda*g_k)  <=   max        f(x_{k-j})  -  rho*lambda*g_k'g_k
                                    0<=j<=min(k,M)

         then set lambda_k = lambda, x_{k+1} = x_k - lambda_k*g_k, go to Step 6.

Step 5.  Choose sigma in [sigma1, sigma2], set lambda = sigma*lambda, go to Step 4.

Step 6.  Set alpha_{k+1} = -(g_k'y_k) / (lambda_k * g_k'g_k),
         k := k+1, return to Step 1.
```

### Why this is BB1 and not "gradient descent with a line search"

Step 6 does not look like the published BB formula. It is it. Since the accepted
step is `s_k = x_{k+1} - x_k = -lambda_k*g_k`,

```
alpha_{k+1} = -(g_k'y_k)/(lambda_k*g_k'g_k)
            = (-lambda_k*g_k)'y_k / (lambda_k^2*g_k'g_k)
            = (s_k'y_k)/(s_k's_k)
```

so the step length actually used at the next iteration is

```
lambda_{k+1} = 1/alpha_{k+1} = (s_k's_k)/(s_k'y_k)          <-- BB1
```

which is Barzilai & Borwein (1988) exactly, and Sun & Yuan (3.1.30), and Dai et
al. (1.6), and Birgin-Martínez-Raydan (2), all with `s = x_k - x_{k-1}` and
`y = g_k - g_{k-1}`.

**This identity is the implementation shape.** `s` is reconstructed from the step
that was applied, so no previous weight vector is stored — which is precisely the
"minimal implementation shape" the plan asks for (`s_t = -alpha_{t-1} d_{t-1}`),
and it is published, not an optimization of it.

### Step 5's `sigma`, made concrete

Raydan's Step 5 leaves `sigma in [sigma1, sigma2]` free and chooses it by
quadratic interpolation. Birgin-Martínez-Raydan Algorithm 2.2 Step 2 writes that
interpolation out. Specialized to `d = -g` (so `g'd = -g'g`):

```
lambda_tmp = 0.5 * lambda^2 * g'g / [ f(x_k - lambda*g) - f(x_k) + lambda*g'g ]

if lambda_tmp in [sigma1*lambda, sigma2*lambda]   then lambda <- lambda_tmp
else                                                   lambda <- lambda/2
```

### Every constant, fixed in advance

From the Birgin-Martínez-Raydan "in practice" paragraph, which is the only
published source of numbers found:

| symbol | value | role |
|---|---|---|
| `rho` (their `gamma`) | `1e-4` | sufficient-decrease parameter |
| `M` | `10` | nonmonotone window |
| `sigma1` | `0.1` | interpolation lower safeguard |
| `sigma2` | `0.9` | interpolation upper safeguard |
| `alpha_l` | `1e-30` | their `1/lambda_max` |
| `alpha_u` | `1e30` | their `1/lambda_min` |

None is configurable. `M = 10` in particular is not a knob: the same paragraph
notes values from 2 to 100 have been reported and that the best is
problem-dependent, which is exactly the invitation to tune that a screen must
refuse.

### What the algorithm leaves undefined, and how that is resolved

Three things are genuinely free in the published statement. Each is resolved
below as **declared neuron policy** and labelled as such, never as a citation.

1. **`delta`** — Raydan requires `delta > 0` and publishes no value.
2. **`alpha_0`** — "given", with no rule.
3. **The Step 4/5 loop has no evaluation bound.** It terminates in theory; an
   implementation that pays one full training-set traversal per trial cannot
   spend an unbounded number of them.

## 3. Declared neuron policy, labelled as such

- **`delta = 1/eta`**, i.e. the fallback *step length* is the user's configured
  learning rate. `delta` is the one free parameter with no published value, and
  the plan's own DRY rule already fixes this role: "persistent configured eta
  remains the initial/fallback configuration". The fallback is therefore a step
  neuron already considers reasonable for the model, not a number invented here.
- **`alpha_0 = 1/eta`**, the same value — the plan: "On the first iteration, use
  the configured initial eta." The first iteration runs the line search from
  there; it does **not** apply BB1, because there is no `s` yet.
- **A per-search evaluation ceiling of 20**, matching `LBFGS::MAX_EVALS` so the
  two searches in this engine cost the same worst case. Reaching it **throws**
  `GBB::LineSearchFailed` after restoring the parameters exactly. Continuing from
  an unaccepted trial point would make the method something other than the one
  named, which is the defect legacy bug #12 was; and silently retrying forever
  would burn the iteration ceiling, which the convergence contract already treats
  as a failure to converge. Note this is a loud path, not an expected one: as
  `lambda -> 0`, `f(x_k - lambda*g) = f(x_k) - lambda*g'g + O(lambda^2)` and
  `rho = 1e-4 < 1`, so acceptance is reached for small enough `lambda` whenever
  the arithmetic is finite and `g'g > 0`.
- **`g'g == 0` exactly**: no step, no state change, return the objective. A zero
  gradient is a stationary point and stopping is `Iterative`'s (rule 6).
- **Non-finite refusal at the accepted point**: a non-finite objective or any
  non-finite gradient element throws `GBB::NotFinite` with no state modified and
  no weight moved — the same contract `IRpropState` has. A non-finite objective
  at a **trial** point is *not* a refusal: it fails the acceptance test, the
  interpolation is skipped (its denominator is not finite), and the search
  halves. That is the line search doing its job, and it is a declared test.
- **`eta` is read, never written.** The two uses above are the only ones. The
  plan's architecture decision 8 forbids expressing a step by mutating the user's
  learning rate, and nothing here does.

## 4. Every published symbol mapped to neuron-owned state

| paper | neuron | owner |
|---|---|---|
| `x_k` | packed weights, `packWeights` / `unpackWeights` | `Network` (the packed boundary Phase 3 built) |
| `f(x)`, `g(x)` | `batchObjectiveGradient()` | `Network` |
| `alpha_k` | `alpha`, the stored reciprocal step | `GBB` |
| `lambda`, `lambda_k` | the trial and accepted step length | `GBB` |
| `s_k`, `y_k` | never both stored: `s` is `-lambda_k*g_k`, `y` is `g_{k+1} - g_k` | `GBB` |
| the `M`-window | ring buffer of the last `M+1` accepted objectives | `GBB` |
| `delta`, `alpha_0` | `1/eta`, passed in per iteration | `Network` supplies, `GBB` uses |
| `eps` / Step 1 | **not implemented here** — `Iterative` owns stopping | `Iterative` |

Step 1 is deliberately absent from `GBB`. The engine already has a gradient
stopping rule, a convergence predicate and a ceiling, and a second one inside an
optimizer is a second definition of "done" (rule 6, and the convergence
contract).

## 5. Ownership

`class GBB` in `src/gbb.*` **owns** `alpha`; the nonmonotone objective window;
the line search including its interpolation, safeguards and counters; the
accepted point, objective and raw gradient; trial installation, acceptance,
failure and exact restoration.

It does **not** own the model, stopping, cancellation, the iteration counter,
reporting, `eta`, or `lastG`/`lastF` (those are CGD's and Shanno's).

`Network::gbbIteration()` composes only: the three eligibility refusals, the
evaluator, `currGradMax` from the **raw** gradient at the accepted point before
anything transforms it, and the pre-update objective every other optimizer in
this engine returns.

Internal `Network::TRAIN_GBB = 5`. **Research only**: no REST token, no GUI
control, no menu entry, no automatic-selection membership, no saved-network
field, no Manifest capability claim. Its public token, if it earns one, is
assigned at integration.

### One recorded piece of debt

`GBBObjective` in `gbb.h` is deliberately identical to `LBFGSObjective` in
`lbfgs.h`: four methods, same signatures, same meaning. Two optimizers now need
the same packed-model port. The right end state is **one** port both consume, but
renaming a public class is a Manifest change and this phase is research-only, so
the duplication is declared here as debt to be closed at integration rather than
hidden. If GBB is rejected, the debt disappears with it.

## 6. The prediction, and what would falsify it

**Mechanism.** BB1 buys a curvature-scaled step for two dot products. Against
canonical fixed-eta descent that should be a large reduction in iterations.
Against Shanno and L-BFGS it should lose on iterations and may win on cost per
iteration, since it stores nothing and computes no direction.

**The prediction most worth testing, stated before measuring.** The plan
describes BB as costing "negligible per-iteration cost" and its minimal shape
adds no line search at all. *Safeguarded* BB is not that method: every trial
point in Step 4/5 is a full training-set traversal. So the honest prediction is
that **BB's advantage will be markedly smaller in traversals than in
iterations**, and if it looks good on iterations and bad on traversals that is
the headline, not a footnote. This is the same accounting that made L-BFGS's
result trustworthy, and the harness already counts every trial evaluation.

**Falsification, fixed now.** BB is **rejected** if either holds:

- it does not beat canonical fixed-eta on time-to-endpoint by more than the
  observed spread on the cheap representative workload; or
- it fails to reach the matched endpoint on panel workloads where the three
  retained methods reach it.

**Retention is the portfolio question, not a race.** Per the plan's Phase 5
policy and the Phase 4 precedent, BB is retained only if it is correct, stable,
reaches the endpoint reliably, **and** is complementary: fastest on some declared
workload family by more than spread, or well-behaved in a declared regime where
the leaders degrade. It does not have to beat L-BFGS or iRPROP+. Retention,
default ranking and automatic-selection membership remain three separate
decisions.

## 7. The deterministic tests, declared before the code

`tests/optimizer/check_bb.cpp`, ctest case `optimizer_bb`. Most are driven **by
hand** against a model-free objective, for the Phase 4 reason: an integration
test can watch a descending objective and never see which algorithm produced it.

1. BB1 on a hand-computed two-dimensional positive-definite quadratic — exact
   `lambda` from hand-computed `s` and `y`.
2. First iteration uses `alpha_0 = 1/eta`; BB1 is not applied when there is no
   `s`.
3. `s'y <= 0` triggers Step 2 and `alpha = delta`, counted as a fallback.
4. `alpha` below `alpha_l` and above `alpha_u` each trigger Step 2 — two cases,
   both bounds.
5. Near-zero `s'y` reaches the `alpha_l` bound rather than producing a huge step.
6. **Nonmonotone acceptance**: a trial that *increases* the objective but stays
   within `f_max - rho*lambda*g'g` is ACCEPTED. This is what separates GLL from
   monotone Armijo. **Sabotage target 1.**
7. The window is exactly `min(k,M)+1` values: the same increase is REJECTED when
   the worse objective has aged out and ACCEPTED when it has not.
8. Quadratic interpolation: hand-computed `lambda_tmp` is used when it lands in
   `[sigma1*lambda, sigma2*lambda]`.
9. Interpolation safeguard: `lambda_tmp` outside that bracket gives exactly
   `lambda/2`. **Sabotage target 2.**
10. A non-finite trial objective is a rejection, not a refusal: interpolation is
    skipped and the search halves.
11. A non-finite objective or gradient at the accepted point throws `NotFinite`
    with no weight moved.
12. `g'g == 0` takes no step and changes no state.
13. The evaluation ceiling throws `LineSearchFailed` and the restored parameters
    are **bit-identical** to the entry point.
14. `eta` independence: `GBB` never writes it, and two runs differing only in
    `eta` differ only through `delta` and `alpha_0`.
15. Refusals: on-line mode, the automatic step-size search, and a model with no
    packed boundary.
16. `currGradMax` is the raw-gradient maximum at the accepted point — not a trial
    point, not a direction.
17. Per-run reset: two `train()` calls start with clean `alpha`, window and
    counters.
18. Every trial evaluation is counted as a full traversal by the harness.
19. Fixed-start real-model integration: SimpleProp, BareProp and BackProp batch
    runs reach the matched target or report a clear failure.

### The two sabotages, declared with what each would turn GBB into

- **Delete the nonmonotone maximum** so the window collapses to `f(x_k)`. That is
  monotone Armijo BB — a different method, and exactly the Phase 4 shape where
  iRPROP+ silently became RPROP+. The assertion that *names* nonmonotone
  acceptance must fail.
- **Remove the interpolation bracket** so `lambda_tmp` is accepted
  unconditionally. Test 9 must fail by name.

Both with visible recompilation of `src/gbb.cpp` in each direction, and both
restored.

## 8. The screen, declared before any arm was run

The cheap representative workload, unchanged from Phases 3 and 4 so the results
are comparable: Civic Choice `simpleprop-6000-4`, the committed **practical**
endpoint, identical starting weights, `run_probe.py --bb`.

The panel is **five arms with five roles**: canonical (the behavioral and
matched-objective reference, and the source of the endpoint), Shanno (the legacy
quasi-Newton control), L-BFGS (leader on the generated fixtures and late-stage),
iRPROP+ (leader on the application benchmark), and BB (the candidate).

Beyond the base panel, predeclared: the three additional weight seeds each as its
own matched group; the 25,000- and 100,000-row scaling groups; the `well4`/`poor4`
conditioning pair; and late-stage arms run to the engine's own plateau rule
(`endpoint: none`).

Per arm: time-to-endpoint distribution, **full training-set traversals**,
iterations completed, fallback and interpolation counts, line-search failures,
held-out error at the endpoint, and every failure row. Traversals and iterations
are reported side by side because the difference between them is the prediction
in section 6.

## 9. Durable prototype archive and post-screen audit

The rejected prototype is absent from active source, tests and build files. Its
exact pre-removal working tree is preserved by the annotated Git tag
`research/bb-prototype-2026-08-08`, based on production commit `7f17e5a`. Inspect
or reconstruct it without changing `main`:

```sh
git diff 7f17e5a research/bb-prototype-2026-08-08
git archive research/bb-prototype-2026-08-08 | tar -x -C /path/to/scratch
```

The archive is historical experimental evidence, not a supported branch or a
public BB capability. It preserves the implementation, model-free tests,
Network/CMake composition and benchmark arms that produced the screen.

**Audit correction.** The archived `accepts()` comparison rejects NaN and
positive infinity, but negative infinity satisfies its ordinary `<=` test. The
declared policy above requires every non-finite trial objective to be rejected.
The test covered NaN and positive infinity, not negative infinity. No measured
row produced a non-finite objective or gradient, so this defect does not change
the screen or rejection; any reconstruction intended for new measurement must
add an explicit `isfinite(trialF)` refusal and fail-prove all three cases.
