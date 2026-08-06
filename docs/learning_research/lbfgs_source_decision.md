# L-BFGS: the pre-code source decision

Phase 3 of `docs/learning_research/optimizer_implementation_plan.md`. Written
**before** any implementation, as that plan's "Pre-code source decision" step
requires. Nothing here may be revised after a benchmark result is seen; a
constant changed to improve a measured number would make the measurement a
description of the tuning, not of the algorithm.

## 1. The authoritative sources

| What | Source |
|---|---|
| Limited-memory update, and why the discarded pairs may be dropped | Nocedal, J. (1980), "Updating Quasi-Newton Matrices with Limited Storage," *Mathematics of Computation* **35**(151):773–782 |
| The method as an optimization algorithm, and the choice of `m` | Liu, D. C. & Nocedal, J. (1989), "On the limited memory BFGS method for large scale optimization," *Mathematical Programming* **45**:503–528 |
| Two-loop recursion; initial inverse-Hessian scaling; strong-Wolfe line search with bracketing and zoom | Nocedal, J. & Wright, S. J., *Numerical Optimization*, 2nd ed., Springer, 2006 |
| The Wolfe conditions themselves | Wolfe, P. (1969), "Convergence Conditions for Ascent Methods," *SIAM Review* **11**(2):226–235 |

Specific locations used, all in Nocedal & Wright 2nd ed.:

- **Algorithm 7.4**, "L-BFGS two-loop recursion", §7.2 p. 178.
- **Equation (7.20)**, the scaling `gamma_k = (s_{k-1}' y_{k-1}) / (y_{k-1}' y_{k-1})`, §7.2 p. 178.
- **Algorithm 7.5**, "L-BFGS", §7.2 p. 179 — the outer loop this implementation follows.
- **Conditions (3.6a)/(3.7b)**, sufficient decrease and the *strong* curvature
  condition, §3.1 pp. 33–34.
- **Algorithm 3.5**, "Line Search Algorithm", §3.5 p. 60 — the bracketing phase.
- **Algorithm 3.6**, "zoom", §3.5 p. 61 — the refinement phase.
- **§3.5 p. 59**, initial step length for a method whose direction is not
  well scaled; **§6.1 p. 142**, why `alpha = 1` must be the first trial step of a
  quasi-Newton method.
- **Equation (6.7)**, the curvature condition `s' y > 0` that makes the BFGS
  update positive definite, §6.1 p. 137.

## 2. The one selected line search

**A complete strong-Wolfe line search: Nocedal & Wright Algorithm 3.5 with
Algorithm 3.6 (`zoom`) as its refinement phase.**

This is chosen over a weak-Wolfe search because the strong condition bounds
`|phi'(alpha)|`, and it is that two-sided bound which makes `s' y > 0` hold
automatically (N&W §3.1) rather than by a separate safeguard.

**Armijo-only backtracking is not implemented and is not called Wolfe.** A
backtracking search satisfies (3.6a) alone, never tests curvature, and cannot
guarantee the pair acceptance condition. It is excluded by name here so that no
later reading of the code can mistake one for the other.

### Every constant, fixed in advance

| Symbol | Value | Where it comes from |
|---|---|---|
| `c1` (sufficient decrease) | `1e-4` | N&W §3.1 p. 33, the recommended value for Newton and quasi-Newton methods |
| `c2` (curvature) | `0.9` | N&W §3.1 p. 34, the recommended value for Newton and quasi-Newton methods |
| Initial step, with history | `1.0` | N&W §6.1 p. 142: a quasi-Newton method must try the unit step first, or it forfeits superlinear convergence |
| Initial step, empty history (steepest descent) | `min( 1.0, 1.0 / sum_i \|g_i\| )` | N&W §3.5 p. 59: use the available problem information when the direction is not well scaled |
| Bracketing expansion factor | `2.0` | N&W Algorithm 3.5 requires `alpha_{i+1} in (alpha_i, alpha_max)` and leaves the rule to the implementer; doubling is declared here |
| `alpha_max` | `1e10` | declared neuron policy: a bracket wider than this on a bounded sigmoidal objective is a failed search, not a long one |
| Evaluations per line search | `20` | declared neuron policy. Each evaluation is one full pass over the training set, so this is a cost ceiling with teeth; MINPACK's `maxfev` default is the same number |
| `zoom` interpolation | **bisection**, `alpha = (alpha_lo + alpha_hi) / 2` | N&W Algorithm 3.6 names "quadratic, cubic, or bisection" and requires only that the trial lie between `alpha_lo` and `alpha_hi`. Bisection is the instantiation with no free parameters and no safeguard of its own to get wrong; with `c2 = 0.9` the unit step is usually accepted in the bracketing phase, so `zoom` is entered rarely and its extra evaluations are not the dominant cost |
| `zoom` degenerate-interval refusal | `\|alpha_hi - alpha_lo\| <= 1e-16 * max( 1, \|alpha_lo\| )` | declared neuron policy: bisection on an interval this narrow cannot produce a new point in double precision, and looping until the evaluation ceiling would spend full passes learning nothing |
| Curvature-pair acceptance | `s'y > 1e-10 * (y'y)` | the published condition is `s'y > 0` (N&W eq. 6.7). The relative floor is a declared neuron safeguard: `rho = 1/(s'y)` must be finite and representable, and a pair that passes `> 0` by one ulp makes the two-loop recursion meaningless. A rejected pair is **skipped**, not damped — Liu & Nocedal's method drops pairs, it does not modify them |
| Memory length `m` | `5` by default | Liu & Nocedal (1989) §5 report `3 <= m <= 7` as generally adequate; the predeclared comparison set if the candidate survives the first screen is `m in {5, 10, 20}`, and nothing outside that set will be tried |

### Failure, cancellation and restoration

- **Non-descent or non-finite direction.** If `g' d >= 0` or any element of `d`
  is not finite, the history is cleared and `d = -g` is used (N&W §7.2: the
  method restarts from steepest descent). If `g` itself is zero the iteration
  returns with the point unchanged; there is no direction to take.
- **Line-search failure** (evaluation ceiling, degenerate interval, or a bracket
  that reaches `alpha_max`): the last accepted weights are reinstalled
  **exactly**, by installing the stored packed vector rather than by undoing
  arithmetic. The history is then cleared, so the next outer iteration retries
  from steepest descent. If the search fails again the iteration reports the
  unchanged accepted objective; the run makes no progress and ends at the
  engine's ceiling, which the convergence contract already calls a failure to
  converge. **No new stop reason is introduced** — `Iterative` keeps stopping
  ownership.
- **Cancellation** is checked *before every trial evaluation*. On cancellation
  the last accepted weights are reinstalled exactly and the iteration returns
  the accepted objective; the run then ends through the existing observer path.

## 3. Every published symbol mapped to neuron-owned state

The plan forbids coding before this mapping exists.

| Published symbol | Meaning | neuron state |
|---|---|---|
| `x_k` | the parameter vector | the packed weights: `hW` row-major then `oW` for `OneHiddenNet`; `Weights[0..nLayers]` flattened in order for `BackProp`. Reached through `Network::packWeights` / `unpackWeights`, and owned by `LBFGS::accepted` between iterations |
| `f(x)` | the objective | the **mean** training-set error plus the current weight-decay penalty — exactly what `innerTrainSet()` already returns, computed by the same authoritative code (Manifest Methodology eq. 2.1) |
| `∇f(x)` | the gradient | the **raw mean** batch gradient, Methodology eq. 2.14, including the decay term `decay * w` (Methodology §2.2.1). Raw: it is the gradient, never a search direction. `LBFGS` never sees `engine()`'s transformed `stackG` |
| `p_k` | the search direction | `LBFGS::dir`, produced by the two-loop recursion. Never written into `hG`/`oG`/`Gradient`, so it cannot be confused with a gradient or consumed by the legacy `w -= g * eta` update |
| `alpha_k` | the step length | `LBFGS`'s own line-search variable. **It is not `eta`.** `eta` is untouched by an L-BFGS run — the plan's architecture decision 8, which forbids implementing an absolute step by mutating the user's learning rate |
| `s_k = x_{k+1} - x_k` | the parameter difference | `LBFGS::S`, a ring buffer of `m` packed vectors |
| `y_k = ∇f_{k+1} - ∇f_k` | the gradient difference | `LBFGS::Y`, same shape |
| `rho_k = 1/(y_k' s_k)` | the pair scaling | `LBFGS::Rho` |
| `gamma_k` | initial inverse-Hessian scaling, eq. (7.20) | computed each iteration from the newest accepted pair; `1.0` when the history is empty |
| `alpha_i`, `beta` | two-loop recursion scalars | `LBFGS::alphaScratch` (sized to `m`, allocated once) and a local |
| `H_k^0 = gamma_k I` | the initial inverse Hessian | never materialized; applied as a scalar multiply, which is the whole point of the recursion |
| `m` | history length | `LBFGS::memory` |

**What is deliberately *not* mapped.** `currGradMax` is set from
`maxabs(g_k)` — the raw gradient at the accepted point — *before* the direction
is computed, per the plan's architecture decision 7. `lastF`/`lastG` belong to
CGD and Shanno and are not touched by L-BFGS: sharing them would make two
published algorithms share one buffer with two meanings.

## 4. What the boundary must therefore provide

Fixed by the mapping above, and by nothing else:

- `packWeights(dest) const` and `unpackWeights(src)` — `x_k` in and out, one
  layout, validated once at entry.
- `batchObjectiveGradient(packedRawGradient)` — `f` and `∇f` at the **currently
  installed** weights, in one traversal, mutating no weight, no optimizer
  history, no `Iterative` state, no report and no cached statistic.
- `packedSize()` — so a model that does not implement the boundary can be
  refused before a run rather than during one.

A universal callback framework is **not** built. There is one consumer, and the
boundary is the minimum that consumer needs.

## 5. Eligibility, declared in advance

L-BFGS **refuses**, rather than silently doing something else:

- **online mode** (`batchEpochFlag` false) — the method needs a deterministic
  objective, and a per-exemplar update has none;
- **the automatic step-size search** (`automaticStepSizeFlag`) — the search
  exists to choose `eta` for a fixed-direction update, and L-BFGS chooses its own
  step. Running both would mean two step rules fighting over one iteration;
- **a model with no packed boundary** (`packedSize() == 0`).
