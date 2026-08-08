# Levenberg-Marquardt: the pre-code source decision

Phase 6 of `docs/learning_research/optimizer_implementation_plan.md`, taken up as
the next neural candidate after Phase 1 rejected safeguarded BB. Written
**before** any implementation, as the plan's "Pre-code source decision" step
requires. Nothing here may be revised after a benchmark result is seen; a
constant changed to improve a measured number would make the measurement a
description of the tuning, not of the algorithm.

This candidate is measured against the plan's **standing portfolio panel**, not
against a single winner. The panel now has four members — canonical, Shanno,
L-BFGS and iRPROP+ — and neither retained method dominates the other: the
ranking flips between the Civic Choice application benchmark and the generated
conditioning fixtures. The retention rule is recorded in section 12, before any
arm is run.

**Status.** Reviewed and accepted. **Step L0 has been run and passed** (section
0), so the implementation phase is authorized: the `Matrix` primitives as their
own commit, then Algorithm 3.16 exactly as declared here, then the tests and
sabotages of section 10, then the screen of section 12. **No constant in this
document may be changed now that a measurement has been seen.** No public
surface — REST field, GUI control, optimizer token, automatic-selection entry or
menu change — is added by the research phase; that is Phase 5's question and only
after retention.

---

## 0. The cost model, and the measurement that contradicted it

### 0.1 The pre-measurement hypothesis, as it was written

*Recorded before Step L0 ran, and retained verbatim in substance because a
falsified pre-declared prediction is evidence. Read this subsection as the
hypothesis that was tested, never as a finding.*

LM's per-iteration cost was expected to be structurally different from every
method already in the panel, by a factor of **P/4**, where `P` is the parameter
count:

| method | work per full training-set traversal |
|---|---|
| canonical, Shanno, L-BFGS, iRPROP+ | `O(N·P)` — one forward and one backward pass per exemplar |
| **LM** | `O(N·P²)` — the same passes, **plus** a rank-one accumulation `J'J += a_k a_k'` per exemplar |

The `O(P³/3)` solve is negligible beside that: at `P = 65` it is ~91,000 flops
per iteration against ~13 million for the accumulation.

Applying that factor to the panel's measured traversal counts
(`docs/learning_research/irprop_screen_results.md`) gave:

| workload | `P` | fastest panel arm | its traversals | **hypothesized** LM budget |
|---|---:|---|---:|---:|
| Civic Choice 6,000 | 65 | iRPROP+ | 10 | ≈ 0.6 |
| Civic Choice 6,000 | 65 | L-BFGS | 20 | ≈ 1.2 |
| `well4` | 25 | L-BFGS | 26 | ≈ 4.3 |
| `poor4` | 25 | L-BFGS | 32 | ≈ 5.3 |

Since an optimizer cannot take fewer than one traversal per iteration, the
hypothesis implied LM had no win region on the application benchmark and only a
narrow one on the conditioning pair.

### 0.2 The measurement contradicted it

**Step L0 has been run: `docs/learning_research/lm_step0_results.md`.**

| workload | `P` | hypothesized ratio | **measured ratio** |
|---|---:|---:|---:|
| Civic Choice 6,000 | 65 | 16.2x | **1.79, 1.84, 1.84, 1.86** |
| `well4` | 25 | 6.2x | **0.88, 0.90, 0.86, 0.86** |
| `poor4` | 25 | 6.2x | **0.88, 0.85, 0.86, 0.87** |

Wrong ninefold at `P = 65`, sevenfold at `P = 25`, and at `P = 25` **wrong in
sign**: the proposed traversal is *cheaper* than the production one it replaces.
Four replicates across two orchestration seeds, agreeing to within 3%.

**The measured budgets, on the authoritative decay-on workload:**

| workload | reference arm | its traversals | **measured LM budget** |
|---|---|---:|---:|
| Civic Choice 6,000 | iRPROP+ | 10 | **5.5** |
| Civic Choice 6,000 | L-BFGS | 20 | 11.0 |
| `well4` | L-BFGS | 26 | **29.9** |
| `poor4` | L-BFGS | 32 | **36.8** |

**Civic Choice is not foreclosed.** LM has about 5.5 traversals there against the
fastest panel arm, and roughly 30 to 37 on the conditioning pair. Those budgets
are credible enough to implement Algorithm 3.16, and the user authorized that on
this evidence.

### 0.3 Why the hypothesis was wrong

It counted multiply-adds and ignored the **transcendental cost of the forward
pass**, which both paths pay identically and which dominates at these sizes.
Every exemplar evaluates `nHidden + 1` sigmoids — five per exemplar here, 22,500
per traversal — and `exp()` costs far more per call than a fused multiply-add.
The `P(P+1)/2` accumulation LM adds is real, but it is added on top of a large
shared floor rather than to a small one.

The general form, and it is the phase's most transferable result: **a cost model
that omits a term both sides pay overstates the ratio, because that term is the
denominator.** `O()` and flop counts are hypotheses that choose what to measure.
They never decide.

### 0.4 The trigger, still unconfirmed

The plan's own trigger for this phase is *"small LMS networks remain a major
wall-clock consumer after accepted batch optimizers."* **That trigger has not
been shown to fire.** No measurement in this program identifies small LMS
networks as a remaining bottleneck; Phases 3 and 4 made the 6,000-row fit take
11 ms. The commissioning instruction directs this phase anyway, and this document
delivers the complete contract on that instruction. Recorded once, here.

---

## 1. The authoritative sources

| What | Source | Access |
|---|---|---|
| The damped Gauss-Newton idea and the original damping argument | Levenberg, K. (1944), "A method for the solution of certain non-linear problems in least squares," *Quarterly of Applied Mathematics* **2**(2):164–168 | primary, cited |
| The algorithm that carries the name, and the `diag(J'J)` damping variant | Marquardt, D.W. (1963), "An algorithm for least-squares estimation of nonlinear parameters," *SIAM Journal on Applied Mathematics* **11**(2):431–441 | primary, cited |
| **The exact variant implemented here**: equations (3.13), (3.14), the gain ratio, Algorithm 3.16, and the appendix Cholesky factorization A.4 | Madsen, K., Nielsen, H.B. & Tingleff, O. (2004), *Methods for Non-Linear Least Squares Problems*, 2nd ed., Informatics and Mathematical Modelling, Technical University of Denmark, Lyngby | **obtained and read in full**, `http://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf` |
| The damping-update rule Algorithm 3.16 uses | Nielsen, H.B. (1999), *Damping Parameter in Marquardt's Method*, IMM, DTU, Report IMM-REP-1999-05 | cited by the above as its source for the update; not read directly |
| Precedent for LM on feedforward-network training | Hagan, M.T. & Menhaj, M.B. (1994), "Training feedforward networks with the Marquardt algorithm," *IEEE Transactions on Neural Networks* **5**(6):989–993, DOI 10.1109/72.329697 | **closed access; not read** |

**The Hagan & Menhaj substitution, disclosed.** The primary neural-LM paper is
behind IEEE's paywall. Rather than substitute a restatement, **no equation, no
constant and no control-flow decision in this document is taken from it.** It is
cited for one thing only: that applying LM to feedforward-network training by
backpropagating output sensitivities is a published, refereed idea and not this
project's invention. Its bibliographic record was verified through three
independent indexes (PubMed 18267874, the DOI, Semantic Scholar). Its reported
scope — that the method is efficient "when the network contains no more than a
few hundred weights" — is the published basis for the parameter ceiling in
section 9, and is the only substantive claim carried from it.

The per-exemplar Jacobian row in section 5 is **derived here from neuron's own
forward equations** (Manifest Methodology equations 2.2–2.5 as transcribed in
`src/onehidden.cpp:455-476`), not copied from any secondary restatement. That is
stronger than citing someone else's version of the same elementary chain rule.

**One named algorithm, no variant flags.** Levenberg's `μI`, Marquardt's
`μ·diag(J'J)`, Moré's trust-region scaling, and the dog-leg method are four
different published methods. Only **Algorithm 3.16 — Levenberg-Marquardt with
`μI` damping and Nielsen's `ν`-based update** — is implemented. There is no
flag, no mode and no configuration value that turns this code into any of the
others, because a variant switch is how a benchmark comes to describe a method
nobody named. If `μ·diag(J'J)` is ever wanted it is a separately declared arm
with its own source decision, not a parameter.

---

## 2. The transcribed algorithm

Madsen, Nielsen & Tingleff (2004), §3.2. Their `x` is the parameter vector,
`f(x)` the residual vector, `F(x) = ½ f(x)ᵀf(x)`, `J = J(x) = ∂f/∂x`.

### The damped normal equations, equation (3.13)

```
    (Jᵀ J + μ I) h_lm  =  -g        with  g = Jᵀ f   and  μ ≥ 0
```

The notes state three properties of `μ` that this implementation relies on and
does not re-derive:

- **(a)** for all `μ > 0` the coefficient matrix is positive definite, which
  ensures `h_lm` is a descent direction;
- **(b)** for large `μ`, `h_lm ≈ -(1/μ) g` — a short step in the steepest
  descent direction, which is right when far from the solution;
- **(c)** for small `μ`, `h_lm ≈ h_gn`, the Gauss-Newton step, giving near
  quadratic final convergence when `F(x*)` is small.

### The initial damping, equation (3.14)

```
    μ_0  =  τ · max_i { a_ii }        where  A_0 = J(x_0)ᵀ J(x_0)
```

### The gain ratio and its denominator

```
                F(x) - F(x + h_lm)
    ϱ  =  ---------------------------
                L(0) - L(h_lm)

    L(0) - L(h_lm)  =  ½ h_lmᵀ ( μ h_lm - g )
```

The notes prove both `h_lmᵀ h_lm` and `-h_lmᵀ g` are positive, so the
denominator is guaranteed positive. That is an invariant this implementation
asserts (test 12) rather than a branch it guards.

### Algorithm 3.16, verbatim

```
begin
    k := 0;   ν := 2;   x := x_0
    A := J(x)ᵀ J(x);   g := J(x)ᵀ f(x)
    found := ( ||g||_inf <= ε_1 );   μ := τ * max{a_ii}
    while (not found) and (k < k_max)
        k := k+1;   Solve (A + μI) h_lm = -g
        if ||h_lm|| <= ε_2 ( ||x|| + ε_2 )
            found := true
        else
            x_new := x + h_lm
            ϱ := ( F(x) - F(x_new) ) / ( L(0) - L(h_lm) )
            if ϱ > 0                                  { step acceptable }
                x := x_new
                A := J(x)ᵀ J(x);   g := J(x)ᵀ f(x)
                found := ( ||g||_inf <= ε_1 )
                μ := μ * max{ 1/3 , 1 - (2ϱ - 1)³ };   ν := 2
            else
                μ := μ * ν;   ν := 2 * ν
end
```

### Every constant, fixed in advance

| Symbol | Value | Where it comes from |
|---|---|---|
| `ν_0` (initial rejection multiplier) | `2` | Algorithm 3.16, initialization line |
| `ν` update on rejection | `ν := 2ν` | Algorithm 3.16, `else` branch |
| `ν` reset on acceptance | `ν := 2` | Algorithm 3.16, acceptance branch |
| `μ` update on acceptance | `μ := μ · max(1/3, 1 - (2ϱ-1)³)` | Algorithm 3.16, acceptance branch (Nielsen 1999) |
| `μ` update on rejection | `μ := μ · ν` | Algorithm 3.16, `else` branch |
| `τ` (initial damping scale) | `1e-3` | §3.2 footnote 3: "use a small value, eg `τ = 1e-6` if `x_0` is believed to be a good approximation to `x*`. Otherwise, use `τ = 1e-3` or even `τ = 1`." Randomized neural starting weights are **not** a good approximation, so `1e-3` is the cited default for this case |
| acceptance test | `ϱ > 0` | Algorithm 3.16. Strictly greater: `ϱ = 0` is not an improvement |

None of these is configurable, and `τ` in particular is not exposed as a knob: a
tunable initial damping would make the screen a tuning exercise, and the plan
forbids that. If a `τ` sweep is ever wanted it is declared in advance and run as
its own comparison group, exactly as the L-BFGS `m ∈ {5,10,20}` sweep was.

### What the algorithm leaves to the caller, and how that is resolved

- **`ε_1`, `ε_2`, `k_max` are "chosen by the user"** (equations 3.15a–c). In
  neuron they are **not chosen by LM at all** — `Iterative` owns stopping,
  cancellation and the ceiling for every optimizer in the engine, and an
  optimizer that carried its own stopping rules would be a second owner of a
  mechanism rule 6 gives to one. See section 7 for the exact mapping, including
  the observation that (3.15a) *already exists* as neuron's `STOP_GRADMAX`.
- **The linear solve is unspecified.** Section 4.
- **A failed factorization is unspecified.** Section 8.
- **`μ_0 = 0` is unspecified.** It arises when every diagonal element of `A_0` is
  zero. Section 8.
- **There is no bound on consecutive rejections.** `k_max` is the algorithm's
  only bound and it counts both accepted and rejected steps. neuron's iteration
  counter counts `trainSet()` calls, so an unbounded inner rejection loop would
  be invisible to it. Section 8.

---

## 3. Mapping neuron's objective onto `F(x) = ½ fᵀf`

This is the one place where a mistake would silently change what is being
optimized, so it is written out completely.

### neuron's objective

For a one-hidden-layer network under **LMS** loss (`errorType == 0`), the
objective `innerTrainSet()` minimizes and `batchObjectiveGradient()` returns is,
reading `src/onehidden.cpp:181-294`:

```
    F(w)  =  (1/N) Σ_{k=1..N} ½ ( o_k - y_k )²  +  (decay/2) ||w||²
```

The `½` is `errorFunction`'s LMS branch (`E = 0.5*(y-o)*(y-o)`,
`src/function_defs.h:114`). The penalty is `regularizer * (hW.squared() +
squared(oW))` added **once per exemplar** and then divided by `N`, with
`regularizer = decay/2` (`src/network.cpp:468`) — so it contributes
`(decay/2)||w||²` exactly once, not `N` times.

Its gradient, as `batchGradient()` computes it:

```
    ∇F(w)  =  (1/N) Σ_k r_k a_k  +  decay · w        with  r_k = o_k - y_k
```

where `a_k = ∂o_k/∂w` is derived in section 5.

### The augmented residual vector

Write `F` in the published form by folding the `1/N` and the penalty into the
residual. Define `f ∈ R^(N+P)`:

```
    f_k      =  r_k / sqrt(N)          k = 1 .. N          (the data rows)
    f_(N+p)  =  sqrt(decay) · w_p      p = 1 .. P          (the penalty rows)
```

Then

```
    ½ fᵀf  =  (1/2N) Σ_k r_k²  +  (decay/2) ||w||²  =  F(w)      exactly.
```

The Jacobian of that augmented residual is `(N+P) × P`:

```
    J[k, :]      =  (1/sqrt(N)) · a_kᵀ           k = 1 .. N
    J[N+p, :]    =  sqrt(decay) · e_pᵀ           p = 1 .. P
```

and therefore

```
    A  =  Jᵀ J  =  (1/N) Σ_k a_k a_kᵀ  +  decay · I               (P × P)

    g  =  Jᵀ f  =  (1/N) Σ_k r_k a_k   +  decay · w   =  ∇F(w)    (P)
```

**Three consequences, each of which is a test in section 10:**

1. `g` computed this way is **∇F**, the same quantity
   `batchObjectiveGradient()` returns. Not bit-identical — the accumulation
   order differs, and the existing path adds `decay·w` per exemplar before
   dividing by `N` while this adds it once after — but agreeing to within
   floating-point tolerance. This is a strong cross-check that costs nothing
   (test 15), and the tolerance is declared, not discovered.
2. `currGradMax = maxabs(g)` is the raw objective gradient at the point used for
   the iteration, satisfying the plan's architecture decision 7 without any
   special handling.
3. The penalty rows are **linear** in `w`, so their second derivative is exactly
   zero. The Gauss-Newton approximation `∇²F ≈ JᵀJ` therefore **drops nothing at
   all from the penalty term** — the `decay·I` block of `A` is the exact Hessian
   of the ridge penalty, and the only approximation is in the data rows. This is
   worth stating because "LM approximates the Hessian" invites the reading that
   the regularization is approximated too, and it is not.

### The `sqrt(N)` and `sqrt(decay)` never appear in the code

They are a derivation device. The implementation accumulates `Σ_k a_k a_kᵀ` and
`Σ_k r_k a_k`, divides both by `N`, and adds `decay` to the diagonal and
`decay·w` to the gradient. No square root is taken, and no `(N+P)`-element
residual vector is ever formed. That is the point of section 6.

### Why not simply use `Σ r_k²`, as the notes do

Because `F` would then differ from the objective every other neuron optimizer
returns, every stopping rule in `Iterative` compares, and every benchmark row
records — by a factor of `2N` plus a missing penalty. The gain ratio `ϱ` is a
ratio and is invariant to a positive scale factor, so nothing in Algorithm 3.16
changes; but `μ` and `μ_0` are *not* scale-invariant, so the scale must be fixed
once and stated, and the one that must be fixed is neuron's.

---

## 4. The direct solve — a decision that needs review

`(A + μI) h = -g` where `A + μI` is symmetric and, for `μ > 0`, positive
definite. The right factorization is **Cholesky**, and the notes give it as
Algorithm A.4 with its positive-definiteness test built in:

```
Algorithm A.4. Cholesky Factorization
begin
    k := 0;  posdef := true
    while posdef and k < n
        k := k+1;   d := a_kk - Σ_{i=1..k-1} c_ik²
        if d > 0
            c_kk := sqrt(d)
            for j := k+1, ..., n
                c_kj := ( a_kj - Σ_{i=1..k-1} c_ik c_ij ) / c_kk
        else
            posdef := false
end
```

with the solve then `Cᵀ z = b`, `C x = z`.

### What neuron has

`Matrix<double>` has `inverse()` in two forms (Gauss-Jordan and LU), a `Singular`
exception, `transpose`, `outprod`, `dotprod`. **It has no solve and no
Cholesky.** It also has no destination-taking symmetric rank-one accumulate:
`outprod` *sets* a matrix, so `A += scratch.outprod(v, v)` writes `P²` elements
and then adds `P²` elements per exemplar, wasting half the work on the symmetric
half and doubling the memory traffic in what is the hottest loop this candidate
has.

### The decision, as ruled

Two numerical-vocabulary extensions are needed:

1. **A symmetric positive-definite solve.** Cholesky per Algorithm A.4,
   destination-taking, no explicit inverse, reporting the positive-definiteness
   failure the algorithm's own test detects.
2. **A destination-taking symmetric rank-one accumulate**, `A += v vᵀ`, with no
   allocation.

**They do not enter `Matrix` yet.** The user's ruling, and the staging it fixes:

- implement both **only in the disposable research working tree**, and only to
  the extent Step L0 and a later prototype actually need. Step L0 needs the
  accumulate; it does not need the solve, and so does not build one;
- **if LM fails Step L0**, remove them with the prototype. Nothing durable was
  added to the public numerical vocabulary for a candidate that was rejected;
- **if LM survives**, promote both into `Matrix` as a **separate, fully tested,
  Manifest-synchronized commit landed before Algorithm 3.16 is implemented** —
  which is where rule 4, rule 6, and the plan's own "land this primitive
  separately before IRLS" all point, and where the plan already named
  `Matrix<double>::solve` as a prerequisite primitive.

This keeps the research prototype entirely non-public, as the commissioning
instruction requires, without conceding that a private Cholesky beside a public
`inverse()` is an acceptable end state.

### Storage of the symmetric accumulator, stated explicitly

A general dense `Matrix` **may not leave one triangle stale while calling the
result symmetric.** A caller that reads `A(i,j)` for `i > j` would silently read
zero, and nothing in the type says so. Two representations are admissible and
the choice is declared, not left to the implementation:

- **(chosen)** accumulate into the **upper triangle only inside the exemplar
  loop** — that is where the `P(P+1)/2`-versus-`P²` saving actually lives — and
  **mirror once per traversal, after the loop**, so every `Matrix` handed to any
  caller is fully populated and genuinely symmetric. The mirror is `O(P²)` once
  per traversal against `O(N·P²)` inside it, so it costs nothing measurable at
  `N ≫ P`, and **it is inside Step L0's timed region** (section 12);
- **(rejected for now)** an explicitly triangular representation whose Cholesky
  reads only the populated triangle. It is the faster end state and remains open
  if LM survives, but it is a second matrix type in the numerical vocabulary,
  and introducing one to serve a candidate that has not passed its first gate is
  the wrong order.

Test 16 asserts full symmetry of every `A` an evaluation produces, which is the
assertion that would fail if the mirror were ever dropped.

---

## 5. The per-exemplar Jacobian row, derived

The forward equations of `OneHiddenNet::propagate()` (`src/onehidden.cpp:455`),
with `σ` the logistic sigmoid, `I` the exemplar input vector (bias input pinned
for `SimpleProp`), `hO` the hidden output vector (bias slot pinned to 1 for
`SimpleProp`):

```
    net_j  =  Σ_i hW(j,i) · I_i             j = 0 .. nHidden-1
    hO_j   =  σ( net_j )
    x      =  Σ_j oW_j · hO_j               j over every element of hO
    o      =  σ( x )
```

Differentiating `o` — **not** the error — with respect to each parameter, and
using `d_sigmoidal()(u) = u(1-u)` applied to the sigmoid's *output*, exactly as
the engine does:

```
    s       :=  σ'(x)  =  o (1 - o)                             the output sensitivity
    hs_j    :=  s · oW_j · hO_j (1 - hO_j)      j = 0 .. nHidden-1

    ∂o/∂oW_j     =  s · hO_j                    every element of oW
    ∂o/∂hW(j,i)  =  hs_j · I_i                  j = 0 .. nHidden-1
```

So the Jacobian row, **in the packed layout `packPair()` already defines** — `hW`
row by row, then `oW`, so index `j·(nInput+1) + i` for `hW(j,i)` — is

```
    a_k  =  [ hs ⊗ I ,  s · hO ]
```

### The consistency identity that makes this checkable

The engine's existing per-exemplar terms are, from
`OneHiddenNet::exemplarErrorTerms()`:

```
    o_err  =  ( o - y ) · σ'(x)                      =  r · s
    h_err  =  ( σ'(hO) · o_err ) · oW                =  r · hs        (elementwise, j = 0..nHidden-1)
```

and its per-exemplar gradient is `oG = hO · o_err`, `hG = h_err ⊗ I`. Therefore

```
    g_k  =  r_k · a_k                                exactly, term for term.
```

That identity is not a comment. It is test 14: on a real model, at real weights,
for a real exemplar, the accumulated `Σ r_k a_k / N + decay·w` must equal
`batchObjectiveGradient()`'s vector within the declared tolerance.

### The duplication this creates, and how it is ruled

`a_k` is a **different quantity** from `g_k` — the sensitivity of the *output*,
not of the *error* — and it cannot be obtained from the existing terms without
dividing by `r_k`, which is unacceptable as `r_k → 0`. So the LM traversal
computes its own sensitivities.

That means the backward chain rule for a one-hidden-layer network appears in
`OneHiddenNet` **twice**: once in `exemplarErrorTerms()` for `∂E/∂w`, once in the
new normal-equations pass for `∂o/∂w`. This is a real cost against rule 6 and it
is recorded here rather than discovered later.

**The ruling: preserve the separate calculation.** The existing gradient path is
not refactored, and **its accumulation order is not altered**, merely to share
code with the new one. Two consequences follow and both are deliberate:

- the alternative — expressing `batchGradient()` as `r_k · a_k` so the chain rule
  exists once — **changes the last bits of every legacy neural result**, because
  it reassociates the arithmetic. Rule 8 forbids folding that into an optimizer
  change, and the goldens and oracle would move. It is not part of this phase and
  is not scheduled by it;
- the duplication is therefore guarded rather than removed. **Test 14** (the two
  paths' gradients agree within a declared tolerance) and **test 15** (`A` against
  a finite-difference Gauss-Newton oracle) are what stop the two calculations
  drifting apart. That is the whole reason both tests are declared, and it is why
  test 14 runs on both concrete models, with decay on and off.

What is genuinely shared — `forward()`, `propagate()`, the sigmoid derivative,
the packed layout — remains shared, and the new pass calls the same `forward()`
the rest of the engine does.

---

## 6. Can `JᵀJ` and `Jᵀr` be accumulated per exemplar? — **Yes**

This is the brief's question 5, and the answer is unqualified.

```
    A  =  (1/N) Σ_k a_k a_kᵀ  +  decay · I
    g  =  (1/N) Σ_k r_k a_k   +  decay · w
```

Both are sums over exemplars of quantities that depend on exemplar `k` alone.
One traversal computes, per exemplar: `forward()`, then `s`, then `hs`, then the
rank-one accumulation into the upper triangle of `A` and the `axpy` into `g`.
The `N × P` Jacobian is **never materialized**, and nothing in the loop depends
on `N`.

### Exact storage, independent of `N`

Per run, owned by the optimizer:

| buffer | size | at `P = 65` | at `P = 512` (the ceiling) |
|---|---|---:|---:|
| `A` at the accepted point | `P²` doubles | 34 KB | 2.10 MB |
| `A` accumulated at the trial point | `P²` | 34 KB | 2.10 MB |
| damped working copy for the factorization | `P²` | 34 KB | 2.10 MB |
| `g`, `h`, accepted weights, trial weights, scratch | `≈ 5P` | 2.6 KB | 20 KB |
| **total** | `3P² + O(P)` | **≈ 104 KB** | **≈ 6.3 MB** |

The trial copy is required because a rejected step must leave `A` and `g` at the
accepted point untouched (section 8). Acceptance is a **swap**, not a copy, so no
`P × P` copy happens per iteration.

### Keeping allocation and dispatch out of the exemplar loop

- every buffer above is sized once per run, in the reset that
  `Network::prepareRun()` calls, and reused for every iteration and every
  exemplar;
- the accumulation is one destination-taking rank-one primitive call per
  exemplar (section 4), touching the upper triangle only, not
  `A += scratch.outprod(v,v)`, which would allocate or double-traverse. The
  symmetry completion is **once per traversal, outside the exemplar loop**, and
  is `O(P²)` against the loop's `O(N·P²)`;
- `forward()` remains the one virtual call in the loop, exactly as it is today
  in `batchGradient()`. No `std::function`, no descriptor, no type switch is
  added (rule 7);
- the `decay` diagonal and `decay·w` are added **once, outside the loop** — they
  are not per-exemplar terms in this formulation, unlike the legacy path where
  they are added per exemplar and divided by `N`. This difference is the reason
  section 3's cross-check is a tolerance test, not a bit-identity test.

### The published equations stay visible

`A`, `g`, `ϱ`, `L(0) - L(h)` and the `μ` update are written as section 2's
formulas, in `src/lm.cpp`, with the equation numbers in the comments — the same
treatment `src/irprop.cpp` gives the iRPROP+ table.

---

## 7. Every published symbol mapped to neuron-owned state

The plan forbids coding before this mapping exists.

| Published symbol | Meaning | neuron state |
|---|---|---|
| `x` | the parameter vector | the packed weights — `hW` row-major then `oW`. Reached through the existing `Network::packWeights` / `unpackWeights`; **no new layout is introduced** |
| `f(x)` | the residual vector | never formed. Its two blocks enter only through `A` and `g` (section 3) |
| `F(x)` | the objective | neuron's mean LMS error plus the current weight-decay penalty — the same value `innerTrainSet()` and `batchObjectiveGradient()` return, from the same forward equations |
| `J(x)` | the Jacobian | never formed. Its rows `a_k` are transient per-exemplar vectors |
| `A = JᵀJ` | the Gauss-Newton matrix | `LM::normal`, one `P × P` `Matrix<double>` |
| `g = Jᵀf` | the gradient | `LM::gradient`, and it **is** `∇F` — see section 3 |
| `μ` | the damping parameter | `LM::damping` |
| `ν` | the rejection multiplier | `LM::nu` |
| `h_lm` | the step | `LM::step`, applied through the existing `Network::applyAbsoluteStep()` |
| `ϱ` | the gain ratio | a local of one iteration; never stored |
| `τ`, `ν_0`, the `1/3` and the cubic | the constants | `LM::TAU`, `LM::NU_INIT`, `LM::MU_FLOOR_FACTOR` — `static const`, section 2 |
| `ε_1` | the gradient stopping tolerance | **`Iterative`'s existing `gradMaxLimit`**, tested against `currGradMax` by the existing `STOP_GRADMAX` rule. Criterion (3.15a) already exists in this engine and LM does not reimplement it |
| `ε_2` | the step-size stopping tolerance | **not implemented.** `Iterative` has `STOP_CHANGE` on the objective, which is a different rule; adding a step-norm rule would be a new stopping condition, and stopping belongs to `Iterative` (rule 6). Recorded as a deliberate omission, not an oversight |
| `k_max` | the ceiling | `Iterative::maxIterations`, which is **a failure to converge, never a stopping condition** (the settled convergence contract) |
| `k` | the iteration counter | `Iterative::iteration` |
| (initialization) | whether this run has a current point | `LM::startedFlag` |

**What is deliberately not mapped.** `eta`, `stackG`, `lastG`, `lastF`,
`deltaError`, `gamma`, `maxLoops` and the iRPROP+ and L-BFGS state belong to other
algorithms. LM reads and writes none of them. Sharing one buffer between two
published algorithms is how two methods come to share a defect.

### The step contract

LM produces an **absolute step**, not a direction: `h_lm` is already in parameter
units and carries its own sign, and Algorithm 3.16 applies it as `x := x + h_lm`.
So it uses `Network::applyAbsoluteStep()` — the path Phase 4 built for exactly
this, under the one sign convention `w += step` — and never `engine()`, never
`w -= g·eta`. The plan's architecture decision 8 forbids the three fakes, and each
is separately tested (tests 20, 21): the step is not divided by `eta`, `eta` is
not written at all, and the automatic step-size search is refused rather than
left running.

---

## 8. Failure, cancellation and per-run lifecycle — the declared policy

These are **not** in the sources. They are this engine's decisions, recorded here
so no later reader mistakes one for published mathematics.

| Situation | Decision | Why |
|---|---|---|
| Objective, `A`, or `g` non-finite **at an accepted point** | throw `LM::NotFinite`, before any state is modified and before any weight moves | never silently fall back to gradient descent; the optimizer that ran is part of the result. The harness turns an exception out of `train()` into one failure row |
| **Trial objective** `F(x_new)` is NaN, `+∞`, **or `-∞`** | a **rejected** step: `μ := μν`, `ν := 2ν`, no weight change. Never accepted | `ϱ > 0` alone would **accept `-∞`** as an infinitely good improvement. That is precisely the defect the post-screen audit found in BB's `accepts()`, which nineteen tests missed. The finiteness test runs **first**, and test 11 enumerates all three members of "non-finite" rather than the two that are easy to think of |
| The step `h_lm` has any non-finite element | a rejected step, same handling | a solve that produced garbage must not move the weights |
| Cholesky positive-definiteness test fails | a rejected step, same handling | theory says `A + μI` is positive definite for `μ > 0`; floating point can still fail the test when `μ` is tiny beside a badly conditioned `A`. Increasing `μ` is the *directly analogous* remedy to Marquardt's own damping increase, and the next factorization is strictly better conditioned. It is an adaptation and is labelled as one |
| `μ_0 ≤ 0` or non-finite | throw `LM::NotFinite` | `μ_0 = τ·max(a_ii)` and `a_ii ≥ decay ≥ 0`. It is zero only when every output sensitivity is zero at every exemplar and decay is off — a point at which `g = 0` too. A silent positive floor would be an undeclared magic constant; a named refusal is testable |
| Consecutive rejections within one `trainSet()` call | bounded at **`MAX_REJECTIONS = 20`**. On exhaustion: restore the last accepted weights exactly and throw `LM::StepRejected` | Algorithm 3.16's only bound is `k_max`, which counts accepted and rejected steps together; neuron's iteration counter counts `trainSet()` calls, so an unbounded inner loop would be **invisible to `Iterative`** and a stalled run would look like a slow one. After 20 rejections `μ` has grown by `2^210`, so the step is numerically indistinguishable from zero; and `20` is far short of the ~45 rejections at which `μ` overflows a double. Returning quietly instead of throwing would let `STOP_CHANGE` or the plateau rule call a total failure "converged", which the convergence contract forbids |
| Cancellation | checked **between trial evaluations**, through the existing `Iterative::Observer::cancelled()`; on cancel, restore the last accepted weights exactly and return the cached accepted objective | LM's iteration can make several full traversals, exactly like a Wolfe line search. That is the case `Observer::cancelled()` was added for, and L-BFGS is the precedent |
| Ceiling exhaustion | `Iterative` owns it; `STOP_MAX_ITERATIONS` is **not** convergence | the settled convergence contract |
| Restoration on any rejection | the accepted weights, `A`, `g` and `F` are never overwritten until `ϱ > 0` **and** the trial objective is finite | a rejected trial that left the model at the trial point would silently make LM a random-walk method |

### Per-run state and reset

Every buffer in section 6, plus `μ`, `ν`, `startedFlag` and the research
counters, is **working state**, reset in `Network::prepareRun()` beside
`lbfgs.reset()` and `irprop.reset()` (the plan's architecture decision 6). It is
never saved in a network file.

There is **no persistent LM configuration** — `τ`, `ν_0`, the update constants
and `MAX_REJECTIONS` are all published or declared and fixed — so there is
nothing for `Network::copy` to carry, and a copied network starts a clean run.
Both facts are tested (test 24).

### Research counters

`iterations`, `acceptances`, `rejections`, `factorizationFailures`,
`nonFiniteTrials`, `dampingIncreases`, `dampingDecreases`. All live in the
per-iteration path, never in an exemplar loop (rule 7). They exist because
section 0's prediction is about *where the time goes*, and Phase 1 taught that
the predicted cost driver can be the wrong one — BB's line search backtracked
fifteen times in eleven thousand iterations.

---

## 9. Eligibility — strict, refused by name, before anything moves

`Network::lmIteration()` states every refusal before doing any work. Each is a
configuration under which this method would have to become a different method to
run.

| # | Refusal | Reason |
|---|---|---|
| 1 | **`getXEerror()` is true** (cross-entropy loss) | **the strict LMS gate.** LM minimizes a sum of squares. Cross-entropy is not one, and there is no residual vector whose `½fᵀf` is that objective. The plan says "no cross-entropy analogy," and this is the line that enforces it |
| 2 | on-line mode (`!batchEpochFlag`) | `A` and `g` are full-batch quantities; a per-exemplar update has no `F(x)` to compare against `F(x + h)` |
| 3 | the automatic step-size search is on | LM owns its own step; two step rules cannot own one iteration |
| 4 | `packedSize() == 0` | no packed boundary. A zero-length parameter vector reported as convergence is the false-convergence shape `maxabs()` was changed to refuse |
| 5 | the model does not implement the normal-equations boundary | **`OneHiddenNet` only in this phase** — `SimpleProp` and `BareProp`. `BackProp` and `Logistic` are deliberately out of scope, per the commissioning instruction and the plan's "`OneHiddenNet` first". An unimplemented boundary refuses; it does not return something plausible |
| 6 | `packedSize() > LM_MAX_PARAMETERS = 512` | the parameter ceiling. Hagan & Menhaj's published scope is "no more than a few hundred weights"; `512` is that rounded to a power of two. At the ceiling, storage is bounded at 6.3 MB **independent of `N`**, and the solve is ~4.5e7 flops per iteration. Above it the `O(N·P²)` accumulation is hopeless against the panel's `O(N·P)` — section 0 |

There is **no row-count limit**, and that is a positive statement about the
design rather than an omission: section 6's accumulation means `N` never enters
storage.

Ineligible selections are refused **by name, before any model configuration
changes**. Nothing silently falls back or coerces a mode.

Exception type: `LM::Ineligible`, distinct from `LBFGS::Ineligible` and
`IRpropState::Ineligible`. Each optimizer states its own eligibility; unifying
them is a public-surface question for Phase 5, not a research-phase refactor.

---

## 10. Ownership, and the two-layer test structure that follows from it

| Owner | Owns |
|---|---|
| `src/lm.*`, `class LM` | **Algorithm 3.16's entire control flow** — `μ_0`, the solve, the gain ratio, the acceptance test, both `μ` updates, the `ν` schedule, the rejection loop and its bound, restoration, the finiteness policy, and the per-run reset. It has **no model dependency**: it is handed a way to evaluate `F`, `A` and `g` at a point, and it installs points through that same interface |
| `class LMObjective` | the abstract evaluation interface, exactly parallel to `LBFGSObjective`: `currentPoint`, `install`, `evaluateNormal(A, g)`, `cancelled` |
| `Network::lmIteration()` | composition only — the six eligibility refusals, `currGradMax` from the raw gradient, one `LM::iterate()` call |
| `Network::Evaluator` | extended to implement `LMObjective` as well, reaching the protected boundary as it already does for L-BFGS |
| `OneHiddenNet::batchNormalEquations()` | **one traversal** producing `F`, `A` and `g` at the currently installed weights, from the derivation in section 5 |
| `Network::applyAbsoluteStep()` | the existing one absolute-step path, unchanged |
| `Iterative` | stopping, cancellation, the iteration counter, reporting — unchanged |

`LM` does not go through `engine()`, for the same reason L-BFGS and iRPROP+ do
not: `engine()` is a direction-transform dispatch point whose caller applies
`w -= g·eta`, and an absolute step chosen by a damped solve cannot honestly fit
that contract.

Research-only: `Network::TRAIN_LM = 5` is an internal `trainingType` value.
**No REST field, no GUI control, no menu token, no automatic-selection entry and
no saved-network field is added in this phase.** The public identifier, if the
method is retained, is assigned in Phase 5.

### The Manifest gate, faced prospectively for the second time

Adding `src/lm.h` with top-level `class LM` and `class LMObjective` turns
`tools/check_manifest_index.py` **red** on two new public roots, exactly as
`src/gbb.h` did in Phase 1. The rule the last two phases established is that
there is no third state:

- **retained** ⇒ the Manifest owes it a Chapter 12 section, index entries, a
  figure update and a Chapter 3 published derivation, in the same commit as the
  public code;
- **rejected** ⇒ the prototype is removed and the gate returns green on its own;
- **never** committed with the gate red.

And the correction Phase 1 added: **tag the rejected prototype before deleting
it.** If LM is rejected, its complete pre-removal tree is preserved under an
annotated `research/lm-prototype-<date>` tag, with base commit, known defects and
reconstruction command recorded in the evidence, *before* `git rm`.

### The deterministic tests, declared before the code

`tests/network/check_lm.cpp`, ctest case `network_lm`. Numbers 1–13 drive `LM`
through a **hand-written, model-free `LMObjective`**; 14–26 exercise the real
composition path.

| # | Test | What it would catch |
|---:|---|---|
| 1 | **Exact linear least squares.** A model-free `LMObjective` for `f(x) = b - Cx` with `C` `4×2` of full rank: `A = CᵀC` and `g = -Cᵀ(b - Cx)` are exact and hand-computable, and the least-squares solution is known in closed form. LM must reach it | the solve, the sign of the step, the whole assembly |
| 2 | **`ϱ = 1` exactly on a linear problem.** For linear `f`, the linear model `L` is `F`, so the gain ratio is exactly `1` at every step, and `μ` is therefore multiplied by exactly `max(1/3, 1-1³) = 1/3` every iteration | the gain-ratio denominator `½hᵀ(μh - g)` transcribed wrongly; a sign error in `L(0)-L(h)` |
| 3 | **Rosenbrock as a two-residual problem**, the notes' own Example 3.16: `f(x) = [10(x₂ - x₁²), 1 - x₁]`, `x_0 = [-1.2, 1]`, `τ = 1e-3`. `J` is analytic. LM must reach `x* = [1,1]` | a nonlinear case the linear tests cannot discriminate |
| 4 | `μ_0 = τ · max(a_ii)` exactly, on a hand-built `A_0` whose diagonal maximum is not its first element | `max` over the wrong axis; `τ` applied to the wrong quantity |
| 5 | **The acceptance `μ` update**, driven at three hand-chosen values of `ϱ`: one where `1-(2ϱ-1)³ > 1/3`, one where it is `< 1/3` so the floor binds, and one where `ϱ` is just above `0`. Each asserts the resulting `μ` to full precision, and asserts `ν` is reset to `2` | the cubic transcribed wrongly; the `1/3` floor dropped; `ν` not reset — each of which is a **different published method** |
| 6 | **The rejection schedule**: forced rejections must produce `μ` multiplied by `2, 4, 8, 16` and `ν` doubling, in that order | `ν` held constant, which is Marquardt's fixed-factor rule and not Nielsen's |
| 7 | `ϱ = 0` is **rejected**, not accepted | `>=` where the algorithm says `>` |
| 8 | **Rejected-step restoration**: after a rejection, the installed point, `A`, `g` and `F` are **bit-identical** to what they were before the trial | a rejected trial left installed — LM as a random walk |
| 9 | **Rejection bound**: an objective that always worsens must throw `LM::StepRejected` after exactly `MAX_REJECTIONS` trials, with the accepted point restored bit-identically | an unbounded inner loop invisible to `Iterative` |
| 10 | **Cholesky failure is a rejection, not a throw**: an `A` that fails the positive-definiteness test at small `μ` must increase `μ` and succeed | a numerically hard problem reported as a crash |
| 11 | **Non-finite trials, all three members**: a trial objective of NaN, of `+∞`, and of **`-∞`** each produce a rejection with the point restored. Separately: a non-finite `g` and a non-finite `A` at an accepted point each throw `LM::NotFinite` | **the BB `accepts()` defect**, reproduced one phase later. `ϱ > 0` accepts `-∞` |
| 12 | **The denominator invariant**: after every step of every sequence above, `L(0) - L(h) > 0` | the notes' guarantee turned into a comment nothing compiles |
| 13 | **Cancellation mid-rejection-loop**: an objective that cancels on its third trial leaves the accepted point installed bit-identically and returns the cached accepted objective | a cancelled run left at a trial point |
| 14 | **`batchNormalEquations()`'s gradient equals `batchObjectiveGradient()`'s**, on `SimpleProp` and `BareProp`, at randomized weights, decay on and off, to a declared relative tolerance of `1e-12` | the section 5 derivation wrong; a transposed index in the packed layout; a missing `decay` term |
| 15 | **`A` against a finite-difference Gauss-Newton oracle**: `A` recomputed in the test as `(1/N)Σ a_k a_kᵀ + decay·I` with each `a_k` obtained by central differences of `o_k` against each weight | a chain-rule slip that the gradient identity in 14 happens to be insensitive to |
| 16 | **`A` is symmetric and its diagonal is `≥ decay`** at every evaluation | the upper-triangle accumulation not mirrored |
| 17 | **`batchNormalEquations()` does not train**: weights, `lastG`, `lastF`, the iteration counter and the TwoSet caches are unchanged across a call | an evaluator that updates |
| 18 | **The trial-point accumulation does not change the accepted sequence.** Two runs from identical starts, one accumulating `A` at trial points and reusing on acceptance, one recomputing `A` after acceptance, produce **bit-identical** weight sequences. Built as a test-only alternate driver, removed with the sabotage harness | section 11's work-scheduling adaptation asserted rather than proven — the "correct code with a wrong rationale" failure mode |
| 19 | **`currGradMax` is `maxabs` of the raw `g`** at the point the step departed from, computed independently in the test | a stopping rule reading a transformed quantity |
| 20 | **eta independence**: identical starts, `eta = 0.05` and `eta = 7.3`, produce **bit-identical** weights after the same number of LM iterations | a step divided or multiplied by `eta` anywhere |
| 21 | **Canonical eta preservation**: after an LM run `getEta()` is bit-identical to what was configured, and a following canonical run behaves as it would have | `eta` mutated to express an absolute step |
| 22 | **Every eligibility refusal by name**, one process per case: cross-entropy, on-line, auto-step, no packed boundary, no normal-equations boundary, `P > 512` | a method quietly becoming a different method |
| 23 | **`μ_0 ≤ 0` refusal**, on a hand-built all-zero `A_0` | an undeclared silent floor |
| 24 | **Per-run reset and copy**: a second `train()` starts at `μ_0` recomputed from its own `A_0` with `ν = 2`; so does a copied network | a run inheriting the previous run's damping |
| 25 | **Pre-update return association**: the value `trainSet()` returns equals `F` evaluated at the weights **before** the iteration | the reporting cadence moved — legacy bug #10's shape |
| 26 | **Fixed-start real-model integration**: `SimpleProp` and `BareProp` each run from a fixed seed to a declared objective or report a clear failure, with every weight finite and the objective materially reduced | a method correct on hand vectors and inert on a model |

### The predeclared sabotages

The plan's eight-step discipline, with build output retained showing the affected
translation units recompiling **both** on injection and on restoration. Three
sabotages, run separately, chosen so that **at least one crosses the production
wiring rather than a component** — the failure shape now measured in two
consecutive phases.

- **(a) `src/lm.cpp`, the component.** Delete the `1/3` floor from the acceptance
  update, making it `μ := μ·(1 - (2ϱ-1)³)` — a **different damping rule**, and one
  that drives `μ` to zero and turns LM into undamped Gauss-Newton whenever
  `ϱ → 1`. **Test 5 must fail, and the assertion that names the floor must be
  among the failing lines.** Control: tests 1 and 3 must be checked — if they
  *also* fail, the sabotage is too coarse to be evidence about the floor.

- **(b) `src/lm.cpp`, the finiteness policy.** Delete the trial-objective
  finiteness test, leaving bare `ϱ > 0`. **Test 11's `-∞` case must fail.** The
  control that makes this informative: **tests 1, 3 and 26 must all still pass**,
  because every trial in them is finite. That control is the finding — it is what
  demonstrates that no real-model integration test can see this defect, which is
  exactly what happened to BB.

- **(c) `src/network.cpp` / `src/onehidden.cpp`, the wiring.** In
  `Network::lmIteration()`, evaluate the normal equations at the **accepted**
  point and hand LM a stale `A` for the trial — every component still correct,
  every published formula intact, the objective still descending, and the method
  no longer Levenberg-Marquardt. **Tests 3 and 18 must fail.** Control: tests 1,
  2 and 5 must still pass, because a hand-driven `LMObjective` supplies its own
  `A` and never crosses that wire. **If the model-free tests fail too, the
  sabotage has proven nothing about the wiring.**

A sabotage that fails a file for some other reason has proven nothing. The
failing **assertions** are read, not merely the file's exit status.

---

## 11. Neuron-specific adaptations, separated from the published mathematics

Everything in section 2 is Madsen, Nielsen & Tingleff (2004). Everything below is
this engine's decision.

| # | Adaptation | Published behavior it departs from | Justification |
|---:|---|---|---|
| 1 | `F(x)` is neuron's mean-half-squared error plus the ridge penalty, expressed as `½fᵀf` through the augmented residual of section 3 | the notes leave `f` to the application | a re-parameterization, not a change: `A`, `g` and `ϱ` are exactly the notes' quantities for that `f`. Fixes the scale `μ` is measured in, which is not scale-invariant |
| 2 | `A` and `g` are accumulated per exemplar; `J` and `f` are never formed | the notes form `J` | `O(P²)` storage instead of `O(NP)`, exactly equal quantities. Section 6 |
| 3 | Stopping criteria (3.15a–c) are **not implemented in `LM`** | Algorithm 3.16 owns `found` | `Iterative` owns stopping for every optimizer in this engine (rule 6). (3.15a) already exists as `STOP_GRADMAX`; (3.15c) as `maxIterations`; (3.15b) is deliberately absent |
| 4 | The trial point is evaluated with the **full** normal-equations accumulation, and the result is reused when the step is accepted | Algorithm 3.16 evaluates only `F(x_new)` at a trial, and recomputes `A`, `g` after acceptance | pure work scheduling: acceptance depends only on `F(x_new)`, and `A`, `g` are taken at `x_new` either way, so **the accepted sequence is identical**. It makes the common case **one traversal per iteration**, directly comparable to iRPROP+. It costs the wasted accumulation on a rejection, which the counters measure. **Proven by test 18, not asserted** |
| 5 | A failed Cholesky positive-definiteness test is a **rejected step** | unspecified | section 8 |
| 6 | A non-finite trial objective — NaN, `+∞` or `-∞` — is a **rejected step**, tested before `ϱ` | unspecified; `ϱ > 0` alone accepts `-∞` | section 8, and the BB audit |
| 7 | Consecutive rejections bounded at `20`, then `LM::StepRejected` with exact restoration | unbounded within `k_max` | section 8 |
| 8 | `μ_0 ≤ 0` or non-finite is a named refusal | unspecified | section 8 |
| 9 | The returned objective is the **pre-update** `F`, matching canonical, CGD, Shanno, L-BFGS and iRPROP+ | the notes return nothing per iteration | moving it would silently move what every stopping rule in `Iterative` compares |
| 10 | `P ≤ 512` | no limit stated | section 9 |
| 11 | Cancellation checked between trial evaluations | not addressed | the L-BFGS precedent; `Observer::cancelled()` exists for exactly this |

---

## 12. The screen, and the retention rule — both declared before any arm is run

### Step L0 — the measured cost-ratio gate, run first and cheaply

**This is the only authorized step.** It measures what section 0 only estimates,
and it needs no optimizer: the two traversals are model-layer passes, so Step L0
is buildable and runnable before Algorithm 3.16 exists.

**What is timed, on each side, from identical installed weights.** Both must be
the *complete* traversal, not a fragment, or the ratio describes nothing:

| the proposed LM traversal | the production reference |
|---|---|
| forward propagation of every training exemplar | forward propagation of every training exemplar |
| the output sensitivity `s` and the hidden sensitivities `hs` | the error terms `o_err` and `h_err` |
| construction of the packed Jacobian row `a_k` | construction of the packed gradient contribution |
| `JᵀJ += a_k a_kᵀ` (upper triangle) | — |
| `Jᵀr += r_k a_k` | gradient accumulation |
| objective accumulation | objective accumulation |
| the `1/N` division, the `decay` diagonal and `decay·w`, and the penalty term | the `1/N` division and the per-exemplar decay terms |
| **the symmetry completion** (section 4) | — |
| | the packing `packPair()` performs for its caller |

The production side is the real `Network::batchObjectiveGradient()` — the same
virtual the panel's L-BFGS and iRPROP+ arms call, not a reimplementation of it.

**Protocol.**

- workloads: `P = 65` (Civic Choice, 6,000 rows, `SimpleProp`, four hidden units,
  the committed 25% stratified holdout) and `P = 25` (`well4`, 6,000 rows, four
  inputs, four hidden units) — the two parameter counts the panel already
  measures, so the ratio can be applied to real traversal counts;
- identical starting weights per workload, from the panel's own weight seed, with
  **the same installed weights for both arms** — the two paths are timed at the
  same point, not at points one of them moved to;
- **15 randomized, interleaved repetitions** per workload, arm order drawn per
  repetition from a recorded orchestration seed, after a discarded warm-up;
- reported per arm: **median, MAD, p10, p90**, and the **measured ratio** of
  medians with its own p10/p90 band;
- **validity gates, checked before any timing is reported**: both paths return a
  finite objective and a non-empty finite gradient of the expected length; their
  objectives agree to a relative tolerance of `1e-12`; their gradients agree
  elementwise to a relative tolerance of `1e-12`; and `A` is symmetric with
  diagonal `≥ decay`. A ratio between two paths that do not compute the same
  thing is not evidence, and equality between two absent or default results is
  vacuous.

**The pre-declared gate.** Combine the measured ratio `R` with the panel's
measured traversal counts: LM must finish in fewer than
`(fastest panel traversals) / R` traversals to win on wall-clock.

- **if the break-even is below two LM traversals on every eligible workload**,
  LM is **rejected immediately at Step L0**. The negative result is recorded and
  every disposable source change is removed;
- **if either `P = 65` or `P = 25` leaves a credible two-or-more-traversal win
  region**, work **stops and the evidence is reported** before Algorithm 3.16 is
  implemented.

Recording this now is what makes an early rejection a *predeclared* result rather
than a retreat. It is the plan's staged gate applied one stage earlier than usual,
warranted because the cost asymmetry is structural in form even though its size
is unmeasured.

### The cheap representative workload

If Step L0 leaves a plausible win region, the screen is the identical workload,
split and endpoint the standing panel already lives on: **Civic Choice, 6,000
rows, `SimpleProp` at the walkthrough's own four hidden units, the committed 25%
stratified holdout, the committed practical objective, identical starting
weights.** One comparison group, axis `optimizer`:

| arm | role in the panel |
|---|---|
| LM | the candidate |
| iRPROP+ | the fastest arm on this workload family |
| L-BFGS | the fastest arm on the generated fixtures and late-stage |
| Shanno | the legacy quasi-Newton control |
| canonical | the behavioral and matched-objective reference, and the source of the endpoint |

Nothing is re-characterized and no endpoint is invented for the candidate.

### The two first-class acceptance axes, run **in** the screen, not after it

Phase 1's finding was explicit: the seed panel is what rejected BB, it costs
almost nothing, and the conditioning pair should be run early rather than last.
Both are therefore part of the initial screen, not a follow-up conditional on it.

1. **The four-weight seed panel** — the base seed plus 101, 202, 303, each its own
   comparison group (two arms from different starting weights are not racing the
   same race). Panel: LM / iRPROP+ / L-BFGS / Shanno. **A traversal-count spread
   across seeds is a first-class result, not a robustness footnote.**
2. **The conditioning pair `well4` / `poor4`** — one problem at two conditionings,
   `P = 25`. This is **the only workload where section 0's arithmetic leaves LM a
   plausible win**, and it is also where a damped Gauss-Newton method has a real
   theoretical claim: `μ` adapts the step to curvature directly, which is what
   ill-conditioning attacks. Both twins, full panel, read **across** the two
   groups.

### Predeclared additional arms

3. **The late-stage question**, asked the only way this workload permits: the
   neural workloads have no strict endpoint because canonical does not converge on
   them, so every arm runs to the engine's own plateau rule with
   `endpoint: none`, and **where each lands is read alongside how fast it got
   there**. A method that stops earlier at a worse objective has not won. This is
   the arm where LM has its second theoretical claim: near a solution with small
   residual, `μ → 0` and the method approaches Gauss-Newton, whose convergence is
   near quadratic. Both retained methods land *worse* than Shanno here.
4. **Scaling to 25,000 and 100,000 rows only through the gate below.** Section 0
   predicts scaling cannot change LM's standing, since both costs are linear in
   `N`; running it anyway is how that prediction is tested rather than assumed,
   but only if the gate opens.

### What is reported per arm

Elapsed median / MAD / p10 / p90; **full training-set traversals** (the harness
counts what the run did, not what the method is said to do — see the harness
change below); **the solve's share of elapsed time, measured separately**;
outer iterations; acceptances and rejections; factorization failures; non-finite
trials; achieved objective; held-out error and ROC at the endpoint; stop reason;
`converged`; failures; peak RSS. Early-stage speed is reported separately from
time to the matched endpoint.

### The harness change this requires, declared now

`tests/optimizer/harness.h`'s `Probe` counts a traversal by overriding
`batchObjectiveGradient()`. **LM's traversals do not go through that method** —
they go through `batchNormalEquations()`. Left alone, LM would report its passes
as zero and **appear to win because its extra passes were invisible**, which is
precisely the failure the override exists to prevent and which the plan names
explicitly.

So `Probe` gains a `batchNormalEquations()` override incrementing the **same**
`evaluationCalls` counter: a traversal is a traversal regardless of which
boundary method made it. Plus a `--lm` subset mirroring `--irprop`, and an `lm`
entry in `optimizerName()`. Declared here, before any number is produced.

### The scale gate

LM advances to 25,000 and 100,000 rows **only** if all four hold at 6,000:

1. **correct** — the mechanism tests pass and all three sabotages are fail-proven,
   with the controls holding;
2. **stable** — no failures across the four predeclared weight seeds and both
   conditioning twins; no non-finite row; no `StepRejected`;
3. **reaches the matched endpoint** — on every arm, on every repetition, with
   `converged: true` and the stop reason `min_error`;
4. **adds a defensible workload advantage** — it is the fastest arm on at least
   one workload family, **or** it is within a defensible band while being
   materially more seed-stable or materially less conditioning-sensitive than
   both retained methods, **or** it lands at a materially better late-stage
   objective.

An obvious loser is rejected cheaply, not scaled. At 25,000 and 100,000 the
measured quantities are the same list above, with **memory** read explicitly
against section 6's table to confirm it is flat in `N`.

### The retention rule, fixed now

Retain LM if it is **correct**, **stable**, **reaches comparable endpoints**, and
is **either** reasonably competitive with the panel **or** adds a meaningfully
different robustness or workload profile — for instance being the fastest method
on the conditioning pair where both retained methods are slow, being materially
more seed-robust than either, or landing at a better late-stage objective.

**It does not have to beat iRPROP+ or L-BFGS.** Reject it only for a clear
performance, stability, endpoint or redundancy failure: prohibitively slow,
unstable, unable to reach the matched endpoint, or demonstrably redundant with an
existing panel member and adding no distinct operational value.

**The portfolio admission rule, which predates this phase, applies unchanged**:
it is what retained iRPROP+ despite losing two of three workload families, and
what rejected BB despite its always converging. There must be a workload a user
should be told to pick this method for.

Retention and default-ranking are recorded as **two separate decisions**.

### Two predictions recorded now, so a falsification is legible later

Phase 1 established that a pre-declared prediction falsified *in the candidate's
favour* is worth as much as one falsified against it, and must be recorded as
such.

1. **LM will lose on wall-clock on Civic Choice at every row count**, because its
   break-even is below one traversal (section 0). If it wins there, section 0's
   cost model is wrong and that is the finding.
2. **If LM wins anywhere, it will be `poor4`**, and the margin over `well4` will
   be *smaller* than every other method's — that is, LM's conditioning penalty
   will be below iRPROP+'s 1.14x and L-BFGS's 1.23x. If instead LM degrades
   *more* between the twins, the damping is not doing what the theory says and
   that is a stronger result than a speed number.

---

## 13. What is authorized, and what is settled

**Step L0 has run and passed** (section 0; evidence in
`lm_step0_results.md`). The authorized sequence is now, in order: the `Matrix`
primitives as their own tested and Manifest-synchronized commit; Algorithm 3.16
exactly as declared here; the section 10 tests and all three sabotages; the
harness traversal accounting; then the section 12 screen — stopping to report the
complete screen before any retention decision or public integration. **No
constant may be changed now that a measurement has been seen.** The research
phase still adds no public surface: no REST field, no GUI control, no optimizer
token, no automatic-selection entry, no menu change.

Settled by the review, and no longer open:

1. **Section 0.** `P/4` is a **cost-model estimate**, never a proof. Constant
   factors, cache behaviour, symmetry handling, vectorization and traversal
   fusion are all implementation-dependent and unmeasured. Step L0 decides.
2. **Section 4.** The SPD solve and the symmetric rank-one accumulate live in the
   disposable tree for now, are removed with the prototype if LM fails Step L0,
   and are promoted into `Matrix` as a separate fully tested,
   Manifest-synchronized commit — before Algorithm 3.16 — only if LM survives.
3. **Section 4.** The accumulator's storage is explicit: upper triangle inside
   the exemplar loop, **mirrored once per traversal**, so no `Matrix` is ever
   handed out half-populated while being called symmetric. The triangular
   representation is deferred, not adopted.
4. **Section 5.** The separate `OneHiddenNet` chain-rule calculation is
   **preserved**. Neither the existing gradient path nor its accumulation order
   is altered to share it; tests 14 and 15 guard the duplication instead.
5. **Section 12.** Step L0 times the *complete* traversal on both sides, from
   identical installed weights, 15 randomized interleaved repetitions at `P = 65`
   and `P = 25`, with the agreement and non-vacuity gates checked before any
   timing is reported.

Still open, and answerable only after Step L0:

6. **Section 10.** Naming: `LM` / `LMObjective` for parallelism with `LBFGS` /
   `LBFGSObjective`. `LM` is a short global identifier and is worth a second
   opinion — but only if there is going to be an `LM`.
7. **Section 12.** If LM proceeds, the harness must count its traversals through
   the new boundary, or LM will look faster than it is.
