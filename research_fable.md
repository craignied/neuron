# Gradient-descent / learning-algorithm research — candidates ranked by expected wall-clock gain

Status: **research plan** under `docs/optimizer_research.md`. Every "expected"
number below is a hypothesis to be measured (rule 3), not a claim. Ranking is
by *expected wall-clock improvement on neuron's real workloads*, i.e.
(expected iteration reduction) × (share of actual engine time the affected
path represents). Nothing here authorizes changing a default path: all
candidates are **new, opt-in `trainingType` options**, so goldens and the
oracle stay byte-identical (rule 8).

## 0. The baseline being improved

- Three optimizers exist: canonical gradient descent (`trainingType 0`),
  conjugate gradient descent (`1`), Shanno's memoryless quasi-Newton (`2`)
  — `src/network.cpp:518` (`engine`), `:532` (`CGD`), `:573` (`shanno`),
  both from Golden pp. 217–222.
- The optimizer contract: each model computes its raw gradient structure,
  `engine()` `pack()`s it into `stackG`, transforms it into a search
  direction, `unpack()`s it back, and the model applies
  `Weights -= Gradient * eta` (batch: `src/backprop.cpp:560-571`; on-line:
  `:469-475`). `lastG`/`lastF` (`src/network.h:113-114`) are the only
  persistent optimizer state, copied by `Network::copy`.
- Step size: fixed `eta`, or the legacy geometric search
  `Network::searchStepSize` (`src/network.h:229`) — up to `maxLoops` extra
  full epoch passes per iteration.
- The loop, stopping rules, and the convergence contract live in
  `Iterative::train` (`src/iterative.cpp:328`). **Invariants no candidate may
  disturb:** `STOP_MAX_ITERATIONS` is failure, not convergence;
  `Iterative::converged()` is the one predicate; reporting cadence never
  changes the fit; shipped tolerances are never loosened to make a run finish.
- `autoalgo::pick` probes every optimizer from identical weights on equal
  wall-clock budgets and adopts the winner — the built-in, already-interleaved
  benchmark harness.
- Typical workloads: small dense nets (logistic with p ≤ ~50 inputs;
  one-hidden and multi-layer nets with df in the tens to low thousands),
  full-batch or on-line, weight decay on, seeded mt19937, one output.
  **The heavy consumers are stepwise regression (RegressNet trains one quiet
  candidate fit per input per step) and cross-validation (folds × repetitions
  × families)** — an optimizer gain multiplies by those counts.

## Shared integration contract (applies to every candidate; written once)

*Where in code.* A new optimizer is a new `trainingType` value
(`src/network.h:140`), a new case in `Network::engine()`'s switch, a named
method beside `CGD`/`shanno` in `network.cpp` with the published equations
visible in comments (rule 7), and its persistent state as `Network` members
beside `lastG`/`lastF`. Per-run sizing/reset of that state goes in
`Network::prepareRun()` (`src/network.cpp:459`) — once per run, outside every
reporting guard and outside the exemplar loop. **Every new `engine()` case
must set `currGradMax = maxabs( stackG )` from the RAW gradient right after
`pack()`**, exactly as CGD/Shanno do, or `getGradMax()`
(`src/network.cpp:500`) reports a direction, not a gradient, and the gradient
stopping rule silently changes meaning.

*DRY.* Reuse, never duplicate: `pack()`/`unpack()` (the model-structure
adapters already written per model), `stackG`/`lastG`, `vector_ops` primitives
(`dotprod`, `squared`, `maxabs`, compound `-=`/`*=`), and `WeightSnapshot`
where a trial step must be revocable. Missing elementwise primitives (sign,
sqrt, multiply/divide of vectors) are added to `vector_ops` with
destination-taking overloads and the standard `SizeMismatch` contracts
(rule 4) — never open-coded loops in the optimizer. Per the research brief:
do **not** invent a generic "Optimizer" descriptor to make unlike algorithms
look alike; the switch in `engine()` is the intended shape, and IRLS (below)
deliberately does not go through `engine()` at all.

*Other code that must change for any selectable optimizer* (the parity bill,
rule 5 — one commit):
- `src/neuron.cpp:1114-1131` — menu list, `askI` upper bound, and the
  batch-recommended warning where applicable.
- `src/gui.cpp:1349-1377` and `:1819-1821` — `algorithm` parameter range and
  its refusal message ("algorithm must be 1, 2, 3 or auto"), `:1541`
  `setTrainingType`, and the optimizer-name maps at `:1759-1760`, `:2010`.
- `src/gui_page.html` — the algorithm `<select>`.
- `src/autoalgo.cpp` — the probe list (comments say "all three"; they and the
  loop generalize), so auto-selection keeps its meaning.
- `Network::copy` (`src/network.cpp:88`) — copy every new state member;
  "not copied" must be written, not omitted (clones: RegressNet, autoalgo
  probes, CV, OBD).
- `Network::runHeader` switch (`src/network.cpp:470`) — the run announces its
  algorithm.
- `docs/gui_cli_parity.md` and `AGENTS.md` in the same commit.
- Untouched by design: `save`/`load` (weights only — a loaded model retrains
  from step 1, state resets in `prepareRun`), `modelfactory`/`netclone`
  (state rides `Network::copy`), stop-reason semantics, plateau, observer.

*Where in manifest* (rule 9, `docs/manifest_maintenance.md` in full):
- **Methodology chapter** (`docs/tex/manifest.tex`, beside "Error Gradient
  Calculation Algorithm" at line 213 / "Parameter Estimation Formulas" at
  405): the published update rule, citation, symbol-to-member mapping, and
  neuron's adaptations, distinguishing published math from neuron policy.
- **Network section** (line 6854): the new method beside CGD/Shanno —
  purpose, ownership, signature, state members, mutation, failure behavior,
  example.
- IRLS goes in the **Logistic section** (line 7422) instead, since Logistic
  owns it.
- Method-level index entries + `tools/check_manifest_index.py` extension +
  PDF rebuild with visual page inspection, same commit. `figures/objects.dot`
  only if a new class appears (none of these plans adds one).

*How tested* (rule 2, every candidate):
1. **Formula guard**: a tiny fixture (2–3 weights, 2–4 exemplars, hand-computed
   arithmetic) asserting one full update step numerically — the *mechanism*,
   not a label or a printed name. Proven able to fail by sabotaging one term
   of the update, with fresh-compilation evidence in the build log both ways.
2. **Characterization**: seeded deterministic run must reach `converged()`
   with an eligible stop reason on a fixture the baseline also converges on;
   assert error monotonicity/final-objective relations, not bit-exact floats
   (portable-expectations rule: pin same-process relations and integers).
3. **State-across-clone**: clone mid-training, continue both, assert identical
   trajectories in-process (guards `Network::copy` of new state).
4. **Contract edges**: cancellation mid-run, ceiling exhaustion reporting
   `max_iterations`, `getGradMax` returning the raw-gradient max, quiet runs
   bit-identical in weights to audible ones. Expected-throw cases (e.g. IRLS
   `Singular`) isolated one per process.
5. New tests live in the matching `tests/` family (`tests/network/`,
   `tests/backprop/` pattern of `check_bpoptimizer.cpp`, `tests/props/`,
   or a new `tests/optimizer/` registered in CTest).

*How benchmarked* (every candidate; brief §5 + the sign-test rule):
- Fixed seeds, **identical starting weights** (randomize once, snapshot,
  restore into every arm), identical stopping configuration.
- Workload grid from the brief §3: logistic (small p, larger p), one biased
  net, one unbiased net, batch and on-line where applicable — at least one
  real dataset (the civic-choice fixtures) and one synthetic separable/
  ill-conditioned pair.
- Interleave candidate and control trials; ≥ 5 repetitions; report median
  elapsed, spread, endpoint objective, iterations, stop reason, failures in
  the brief's table format.
- **Comparable objective or it doesn't count**: a run that stops earlier at a
  worse objective is not a speedup (acceptance question 2).
- **Sign test**: a real per-iteration gain has the same sign on every
  workload; mixed signs inside the spread are noise and the decision is
  "change nothing," recorded in HISTORY with the evidence.
- `autoalgo::pick` doubles as an in-engine harness: extend the probe list and
  the equal-budget, identical-start comparison is already interleaved.

---

## 1. IRLS / penalized Newton–Raphson for Logistic — expected 10²–10³× on the logistic family

**Published rule.** Iteratively reweighted least squares for logistic
regression: β ← β + (XᵀVX)⁻¹ Xᵀ(y − p), V = diag(pᵢ(1−pᵢ)) — McCullagh &
Nelder, *Generalized Linear Models* (2nd ed., §2.5); Hosmer & Lemeshow
(the codebase's existing Wald citation, eqn 2.8 p. 41). With neuron's Gaussian
prior (decay = 2λ): penalized Newton, gradient Xᵀ(y−p) − decay·β, Hessian
XᵀVX + decay·I. Converges quadratically near the optimum — typically 5–15
iterations where first-order methods take 10³–10⁶.

**Why first.** Largest expected ratio, on the family that dominates real
wall-clock: every stepwise-regression candidate fit and every logistic CV
fold. It is also the smallest risk: the objective and optimum are unchanged
(same penalized likelihood), only the path to it.

**Where in code.** `Logistic` owns it (rule 6 — Logistic is the only model
whose Hessian is this cheap and already computed). A new `trainingType`
value dispatched in `Logistic::trainSet()`: when selected, `trainSet()`
performs one Newton step instead of calling `searchStepSize`. It does **not**
go through `Network::engine()` — forcing a weight *solve* through a
gradient-direction pipe would hide the published equation (brief step 4).
`Iterative::train()` still drives the loop, so every stopping rule, print
cadence, observer, and the convergence contract apply unchanged.

**Implementation.**
1. Extract the XᵀVX construction now inlined in `Logistic::reportAccuracy`
   (`src/logistic.cpp:278-314`) into one private
   `Logistic::fisherInformation( Matrix<double>& XtVX, vector<double>& score )`
   (one forward pass over `Train`, V diagonal exploited exactly as the
   existing code does). `reportAccuracy` and the IRLS step both call it —
   one owner for the formula.
2. Per iteration: build score g = Xᵀ(y−p) (− decay·W when `weightDecayFlag`),
   H = XᵀVX (+ decay·I on the diagonal); step `W += H.inverse() * g` — or
   better, a `Matrix::solve` if profiling says inversion cost matters (p is
   small; it won't at p ≤ 50).
3. Return the same set error `innerTrainSet` returns (one more forward
   accumulation, or fold it into the same pass), so stopping rules and the
   printed rows read identically.
4. Set `G` (the gradient member) from the score so `pack()`/`getGradMax()`
   and the `STOP_GRADMAX` rule stay meaningful.
5. `Singular` from `inverse()` (separated or collinear data) propagates as a
   refusal with the model's failure contract — never a silent fallback to
   gradient descent (the estimator-choice rule: nothing ever falls back
   silently).

**DRY.** `fisherInformation` shared with the Wald report; `Matrix::inverse`,
`dotprod`, existing forward pass; no new state vectors at all (stateless per
iteration). No change to `engine()`, `pack`/`unpack`, or any other model.

**Other updates.** The shared parity bill. Menu/GUI: option is only offered
for Logistic (the GUI already knows the model family; the CLI menu branch at
`neuron.cpp:1114` is inside the network menu and needs a family guard).
`batchEpochFlag` is irrelevant to IRLS — the run header should say so rather
than print a misleading on-line/off-line line. `autoalgo`: include the IRLS
probe only when the network is Logistic. `searchStepSize` guard: never active
for IRLS (no eta).

**Manifest.** Logistic section (line 7422): full Chapter-7-pattern entry for
the IRLS trainer and `fisherInformation` (citations above, penalized-Newton
adaptation, `Singular` failure contract, example); a Methodology subsection
under Parameter Estimation Formulas for the penalized update; index entries
(`Logistic!IRLS`, `Logistic!fisherInformation`).

**Tests.** Shared recipe, plus: (a) endpoint agreement — IRLS and canonical
GD (long budget) reach the same β within a measured tolerance on a fixed
fixture, same process; (b) the Wald report after IRLS equals the Wald report
after GD training to the same β (the shared `fisherInformation` guard);
(c) separable-data fixture throws `Singular`, one case per process;
(d) stepwise regression over IRLS-trained candidates selects the same inputs
as over GD-trained candidates on a fixture where selection is decisive.

**Benchmark.** Shared protocol; the decisive workloads are (i) one stepwise
regression run end-to-end on civic-choice-sized data, (ii) a k-fold CV batch,
each IRLS vs canonical/CGD/Shanno — these measure the multiplier, not just
one fit.

---

## 2. L-BFGS with line search — expected 5–20× on batch neural nets

**Published rule.** Liu & Nocedal (1989), *On the limited memory BFGS method
for large scale optimization*, Math. Prog. 45; two-loop recursion (Nocedal &
Wright, *Numerical Optimization*, Alg. 7.4-7.5) over the last m pairs
sₖ = xₖ₊₁−xₖ, yₖ = gₖ₊₁−gₖ, with a line search satisfying the (weak) Wolfe
conditions. Shanno's memoryless method already in the engine is essentially
m = 1 without a real line search — L-BFGS with m ≈ 5–10 is its published
stronger sibling and the standard fast batch optimizer for smooth objectives
of this size.

**Where in code.** A `Network::lbfgs( unsigned t )` case in `engine()`,
beside `shanno`. State on `Network`: two ring buffers
`vector< vector<double> > memS, memY`, `vector<double> memRho`, an insertion
index, and the previous packed weights — sized in `prepareRun()` from `df()`,
copied in `Network::copy`. The two-loop recursion transforms `stackG` in
place; `unpack()` returns the direction.

**Implementation.**
1. `pack()`; `currGradMax = maxabs( stackG )`.
2. Update the (s, y) pair from the previous iteration's stored weights and
   gradient; skip the pair when sᵀy ≤ ε (curvature guard, published
   safeguard — cite it) and restart the memory exactly as CGD/Shanno restart
   at `t == 0 || t == df()`.
3. Two-loop recursion with `dotprod` and compound `vector_ops`; initial
   H₀ = (sᵀy / yᵀy)·I scaling (Nocedal & Wright eqn 7.20).
4. Line search: a backtracking Armijo search implemented as a *named*
   `Network::wolfeStep( ... )` reusing the model's `WeightSnapshot` for
   revocable trials and `innerTrainSet()` for function values — structurally
   the shape `searchStepSize` already has (snapshot → trials → restore →
   one real pass), and templated the same way to avoid hot-loop
   `std::function` (rule 7). Not a replacement of `searchStepSize`: the two
   coexist; the legacy search stays untouched for the legacy optimizers.
5. Batch-only: the pair (s, y) is meaningless across on-line per-exemplar
   updates. Menu/GUI enforce or warn exactly as the existing CGD/Shanno
   warning does (`neuron.cpp:1122-1126`).

**DRY.** Ring-buffer vectors reuse plain `vector<double>` + `vector_ops`;
snapshot machinery reused; restart policy mirrors the existing optimizers'
`t == df()` convention so the manifest can describe one restart rule.

**Other updates.** Shared parity bill; `maxLoops`-style cap for the line
search as a named constant with the published default; run header prints m.

**Manifest.** Methodology: the two-loop recursion and Wolfe conditions with
citations, symbol table mapping s/y/ρ to members. Network section: the
method entry beside Shanno; index `Network!lbfgs`.

**Tests.** Shared recipe, plus: (a) two-loop recursion guard — on a fixed
quadratic (expressible with `Matrix`), one L-BFGS step with m = 1 and exact
line search must equal the hand-derived step; (b) curvature-skip fixture
(engineered sᵀy < 0) proving the safeguard branch fires — assert the
*mechanism* (memory not updated), not absence of a crash; (c) restart at
`t == df()` actually clears memory (wrong-fault-class trap: build the fixture
so the restart is reachable).

**Benchmark.** Shared protocol vs CGD and Shanno on the neural-net grid;
per-iteration cost includes the line search's extra epoch passes — count
*epoch-equivalent passes*, not iterations, so the comparison is honest.

---

## 3. iRprop+ — expected 5–10× on batch nets, cheapest per iteration

**Published rule.** Riedmiller & Braun (1993), *A direct adaptive method for
faster backpropagation learning: the RPROP algorithm*, IEEE ICNN; improved
variant iRprop+ in Igel & Hüsken (2000), *Improving the Rprop learning
algorithm*. Per-weight step sizes Δᵢⱼ adapted by gradient-sign agreement:
multiply by η⁺ = 1.2 on same sign, by η⁻ = 0.5 on sign flip (with the
iRprop+ error-dependent revert), step = −sign(g)·Δ. No line search, no global
learning rate, famously robust defaults, batch-only.

**Why third.** On small/medium MLPs Rprop's wall-clock is competitive with
quasi-Newton at a fraction of the per-iteration cost (one pass, two extra
vectors, no trial passes), and it removes eta sensitivity — which also makes
`autoalgo` probes fairer. Ranked below L-BFGS only because its gain
concentrates where canonical GD is currently used with a poor eta.

**Where in code.** `Network::rprop()` case in `engine()`. State:
`vector<double> rpropDelta, rpropLastStep` (+ previous error for iRprop+),
sized/reset in `prepareRun()` (Δ₀ = 0.1, Δmin = 1e-6, Δmax = 50 — the
published defaults as named constants), copied in `Network::copy`.

**Implementation — the one honest wrinkle.** Rprop's update is an absolute
per-weight step; the models apply `Weights -= Gradient * eta`. Writing
`stackG = sign(g)·Δ / eta` would hide the published formula behind a scale
(rule 7 forbids exactly this). The plan: `engine()` writes the *step itself*
into `stackG`/`Gradient`, and each model's separate-gradient update branch
applies it through the existing statement with **eta pinned to 1 for
step-owning optimizers at `setTrainingType` time, announced in the run
header** ("step size owned by iRprop+; eta not used"). That is one visible,
documented policy — no per-model update duplication, no hidden scale. This
integration decision is exactly the kind step 4 of the brief says to settle
in the disposable prototype before committing.

**DRY.** Needs `vector_ops` elementwise sign/multiply primitives (shared
with Adam below — add once). Everything else is `pack`/`unpack` reuse.

**Other updates.** Shared parity bill; batch-only warning as for CGD/Shanno;
`searchStepSize` guard never active (it would fight the adaptation).

**Manifest.** Methodology: the sign-adaptation rule and the iRprop+ revert,
both citations, defaults table. Network section entry; index
`Network!rprop`.

**Tests.** Shared recipe, plus a three-iteration hand-computed fixture
asserting: same-sign growth by η⁺, flip shrink by η⁻ *and* the iRprop+
revert actually restoring the previous step (sabotage the revert; prove the
test fails).

**Benchmark.** Shared protocol; the informative comparison is against
canonical GD with `searchStepSize` on (its real competitor) and against
L-BFGS (per-iteration cost vs iteration count trade).

---

## 4. Barzilai–Borwein step size — expected 2–10× over fixed-eta canonical GD, smallest change

**Published rule.** Barzilai & Borwein (1988), *Two-point step size gradient
methods*, IMA J. Numer. Anal. 8: η_t = sᵀs / sᵀy (BB1) with s = x_t−x_{t−1},
y = g_t−g_{t−1}. Two dot products per iteration turn plain gradient descent
into a method that often tracks CGD on quadratics — with essentially zero
per-iteration overhead and no state beyond what `lastG` already provides.

**Where in code.** A `Network::bbStep()` case in `engine()` that leaves the
direction as the raw gradient and *sets the member `eta`* for this iteration
(clamped to published safeguards η ∈ [ηmin, ηmax]; fall back to the previous
eta when sᵀy ≤ 0 — cite the standard safeguard). State: previous packed
weights vector (new), `lastG` (existing). Because it monkeys with `eta`, the
run header states "step size: Barzilai–Borwein," and `searchStepSize` must
be inert for this type.

**Implementation.** `pack()`; `currGradMax`; compute s and y with
`vector_ops`; η_t = `dotprod(s,s)/dotprod(s,y)`; store packed weights;
`unpack()` unchanged gradient. First iteration uses the user's eta.
Batch-only (per-exemplar BB is not the published method).

**DRY.** No new primitives, no new structures beyond one vector. The
smallest possible prototype — a good *first experiment* to shake out the
harness even though it ranks fourth on expected gain, and the natural
control arm for every later benchmark.

**Other updates / manifest / tests.** Shared bills; Methodology subsection
(published formula + safeguards); formula-guard test on a 2-weight quadratic
where η_t is hand-computable; characterization vs fixed-eta GD.

**Benchmark.** Shared protocol vs canonical GD (fixed eta and
`searchStepSize`) — also quantifies how much of `searchStepSize`'s cost
(maxLoops extra epochs) BB eliminates.

---

## 5. Adam — expected 2–5×, mainly for on-line mode

**Published rule.** Kingma & Ba (2015), *Adam: A Method for Stochastic
Optimization*, ICLR: m_t, v_t exponential moments (β₁ = 0.9, β₂ = 0.999,
ε = 1e-8), bias-corrected, step α·m̂/(√v̂+ε). Its natural home is exactly the
regime neuron's other candidates ignore: **on-line (per-exemplar) training**,
where CGD/Shanno are inappropriate (the menu already warns) and canonical GD
is eta-fragile.

**Where in code.** `Network::adam()` case in `engine()` — which is already
called per exemplar in on-line mode (`backprop.cpp:469`) and per epoch in
batch, so both modes come for free. State: `vector<double> adamM, adamV` and
an `unsigned adamT` **owned by the optimizer, incremented per update** — not
derived from the `t` argument, whose `iteration*nTrain+example` composition
and `t == df()` restart convention encode CGD/Shanno policy, not Adam's.
Sized/reset in `prepareRun()`; copied in `Network::copy`.

**Implementation.** Published α maps directly onto the existing `eta`
member (α := eta — a faithful mapping, no hidden scale, unlike Rprop), so
the models' `Weights -= Gradient * eta` applies the published step when
`engine()` writes m̂/(√v̂+ε) into `stackG`. Needs elementwise
multiply/divide/sqrt `vector_ops` primitives (shared with iRprop+).
Defaults as named constants; optionally surfaced later, but the first
version ships the published defaults only.

**Other updates / manifest / tests.** Shared bills; Methodology subsection
with the bias-correction derivation; formula guard hand-computing two Adam
steps including bias correction (sabotage: drop the correction; the test
must fail); on-line characterization fixture.

**Benchmark.** Shared protocol with the **on-line workloads primary**; in
batch mode expect it to lose to L-BFGS/Rprop — record that honestly rather
than promoting it.

---

## 6. Classical momentum / Nesterov — expected ≤ 2–5×, cheap control arm

**Published rule.** Polyak (1964) heavy ball: v_t = μv_{t−1} − ηg;
Nesterov (1983) accelerated gradient (Sutskever et al. 2013 formulation for
nets). One extra state vector, two `vector_ops` lines in an `engine()` case.

**Assessment.** For batch training its acceleration overlaps what CGD
already provides, and on-line it is dominated by Adam's expected gain. Its
value is as a *control arm* and as the cheapest on-line improvement if Adam
measures poorly. Implement only if the Adam/Rprop measurements leave a gap;
otherwise record the negative decision. All shared bills apply as above;
Methodology subsection beside the others; formula guard trivial.

---

## 7. Levenberg–Marquardt — research-only candidate, small LMS nets

**Published rule.** Levenberg (1944), Marquardt (1963); for MLP training
Hagan & Menhaj (1994), *Training feedforward networks with the Marquardt
algorithm*, IEEE Trans. NN 5(6): (JᵀJ + μI)δ = Jᵀe with adaptive μ.
Classically the fastest trainer for small LMS-error networks — often 10–100×
fewer iterations — but per-iteration cost O(N·df²) + O(df³) and O(df²)
memory, and it is tied to sum-of-squares error (neuron's cross-entropy path
would need the Gauss-Newton-for-GLM generalization — new math, new risk).

**Assessment.** Expected total gain is real but confined to LMS one-hidden
nets with small df; the Jacobian harvest requires new per-exemplar
machinery (a per-exemplar gradient collection much like the one deliberately
removed from `conditionOf` in 2026-08-01). Prototype at the concrete layer
(OneHidden) per brief step 4 **only after** 1–3 land and only if the
benchmark grid shows LMS nets still dominating some real workload's
wall-clock. Otherwise record as evaluated-and-rejected in HISTORY — a
negative result is complete work.

---

## Recommended execution order

1. **Harness + BB (candidate 4)** — one shared, deterministic benchmark
   harness (identical-start snapshots, interleaving, medians/spread, the
   brief's table), validated on the smallest prototype.
2. **IRLS (1)** — the big multiplier, measured end-to-end on stepwise and CV.
3. **L-BFGS (2)**, then **iRprop+ (3)** — both batch; decide between them on
   the same grid before making *either* a permanent public option
   (acceptance question 8: each public optimizer is a forever documentation,
   parity, autoalgo and testing burden — don't ship both if one dominates).
4. **Adam (5)** for the on-line regime; **momentum (6)** only as fallback.
5. **LM (7)** only on measured evidence of a remaining LMS bottleneck.

Each candidate ends with an explicit decision — integrate, retain as
research-only, or reject — recorded with its evidence table in
`docs/HISTORY.md`.

## References

Full citations for every algorithm named above, so the primary source can be
obtained without a literature search (brief step 1 requires the primary paper
in hand before coding). DOIs are the stable locators; prefer them over
publisher page URLs.

### Baseline (the optimizers already in the engine)

- Golden, R. M. (1996). *Mathematical Methods for Neural Network Analysis
  and Design*. MIT Press. ISBN 978-0-262-07174-1. — The codebase's cited
  source for conjugate gradient descent (pp. 221–222) and Shanno's
  memoryless quasi-Newton method (pp. 217–218), `src/network.cpp:532,573`.

### 1. IRLS / penalized Newton–Raphson for logistic regression

- Nelder, J. A. & Wedderburn, R. W. M. (1972). "Generalized Linear Models."
  *Journal of the Royal Statistical Society, Series A* 135(3): 370–384.
  doi:10.2307/2344614 — the original GLM/IRLS paper.
- McCullagh, P. & Nelder, J. A. (1989). *Generalized Linear Models*, 2nd ed.
  Chapman & Hall/CRC. ISBN 978-0-412-31760-6. §2.5 (iterative weighted
  least squares fitting).
- Hosmer, D. W. & Lemeshow, S. (2000). *Applied Logistic Regression*,
  2nd ed. Wiley. ISBN 978-0-471-35632-5. Eqn 2.8, p. 41 — already the
  codebase's citation for Var(β) = (XᵀVX)⁻¹ at `src/logistic.cpp:313`.

### 2. L-BFGS with Wolfe line search

- Nocedal, J. (1980). "Updating quasi-Newton matrices with limited storage."
  *Mathematics of Computation* 35(151): 773–782.
  doi:10.1090/S0025-5718-1980-0572855-7 — the original limited-memory
  update and two-loop recursion.
- Liu, D. C. & Nocedal, J. (1989). "On the limited memory BFGS method for
  large scale optimization." *Mathematical Programming* 45(1–3): 503–528.
  doi:10.1007/BF01589116.
- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed.
  Springer. doi:10.1007/978-0-387-40065-5. Algorithms 7.4–7.5 (two-loop
  recursion), eqn 7.20 (H₀ scaling), §3.1 (Wolfe conditions).
- Wolfe, P. (1969). "Convergence conditions for ascent methods." *SIAM
  Review* 11(2): 226–235. doi:10.1137/1011036 — the line-search conditions
  themselves.

### 3. Rprop / iRprop+

- Riedmiller, M. & Braun, H. (1993). "A direct adaptive method for faster
  backpropagation learning: The RPROP algorithm." *Proceedings of the IEEE
  International Conference on Neural Networks (ICNN)*, San Francisco,
  pp. 586–591. doi:10.1109/ICNN.1993.298623.
- Igel, C. & Hüsken, M. (2003). "Empirical evaluation of the improved Rprop
  learning algorithms." *Neurocomputing* 50: 105–123.
  doi:10.1016/S0925-2312(01)00700-7 — the journal version defining iRprop+
  and iRprop−; easier to obtain than the NC'2000 proceedings paper
  (Igel & Hüsken, 2000, *Proc. Second International Symposium on Neural
  Computation*, pp. 115–121, ICSC Academic Press).

### 4. Barzilai–Borwein step size

- Barzilai, J. & Borwein, J. M. (1988). "Two-point step size gradient
  methods." *IMA Journal of Numerical Analysis* 8(1): 141–148.
  doi:10.1093/imanum/8.1.141.
- Raydan, M. (1997). "The Barzilai and Borwein gradient method for the
  large scale unconstrained minimization problem." *SIAM Journal on
  Optimization* 7(1): 26–33. doi:10.1137/S1052623494266365 — the
  globalization/safeguard scheme the implementation's clamps come from.

### 5. Adam

- Kingma, D. P. & Ba, J. (2015). "Adam: A Method for Stochastic
  Optimization." *3rd International Conference on Learning Representations
  (ICLR 2015)*. arXiv:1412.6980. https://arxiv.org/abs/1412.6980 —
  Algorithm 1 is the exact update rule; §2 gives the defaults
  (α = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸).

### 6. Classical momentum / Nesterov

- Polyak, B. T. (1964). "Some methods of speeding up the convergence of
  iteration methods." *USSR Computational Mathematics and Mathematical
  Physics* 4(5): 1–17. doi:10.1016/0041-5553(64)90137-5 — the heavy-ball
  method.
- Nesterov, Y. (1983). "A method for solving the convex programming problem
  with convergence rate O(1/k²)." *Doklady Akademii Nauk SSSR* 269(3):
  543–547; English translation in *Soviet Mathematics Doklady* 27(2):
  372–376.
- Sutskever, I., Martens, J., Dahl, G. & Hinton, G. (2013). "On the
  importance of initialization and momentum in deep learning."
  *Proceedings of the 30th International Conference on Machine Learning
  (ICML 2013)*, PMLR 28(3): 1139–1147.
  https://proceedings.mlr.press/v28/sutskever13.html — the practical
  Nesterov-for-networks formulation the implementation would follow.

### 7. Levenberg–Marquardt

- Levenberg, K. (1944). "A method for the solution of certain non-linear
  problems in least squares." *Quarterly of Applied Mathematics* 2(2):
  164–168. doi:10.1090/qam/10666.
- Marquardt, D. W. (1963). "An algorithm for least-squares estimation of
  nonlinear parameters." *Journal of the Society for Industrial and Applied
  Mathematics* 11(2): 431–441. doi:10.1137/0111030.
- Hagan, M. T. & Menhaj, M. B. (1994). "Training feedforward networks with
  the Marquardt algorithm." *IEEE Transactions on Neural Networks* 5(6):
  989–993. doi:10.1109/72.329697 — the MLP-training formulation
  ((JᵀJ + μI)δ = Jᵀe with adaptive μ) a prototype would implement.
