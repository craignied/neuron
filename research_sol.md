# Gradient-descent and learning-algorithm research for neuron 3.0

Date: 2026-08-03

## Executive recommendation

The order below is a **provisional implementation order ranked by expected reduction in
wall-clock time to a comparable objective on neuron's present workloads**, not by novelty,
popularity, or asymptotic iteration count.  It is based on the published evidence and on
the current engine architecture; it is not a performance claim about neuron until the
benchmarks specified here have been run.

1. **Limited-memory BFGS (L-BFGS)** — best first broad trial for deterministic,
   full-batch logistic and neural training.
2. **Improved resilient propagation with weight backtracking (iRPROP+)** — very low
   per-epoch overhead and a particularly natural fit to full-batch neural training.
3. **Barzilai-Borwein spectral gradient with a safeguarded nonmonotone line search** —
   smallest implementation and memory cost among serious adaptive full-batch methods.
4. **Levenberg-Marquardt (LM)** — potentially the fastest option for small
   least-mean-squares networks, but deliberately model- and size-limited.
5. **Nesterov accelerated gradient / scheduled momentum** — cheap and suitable for both
   online and batch training, but more tuning-sensitive than the first three here.
6. **Stochastic variance-reduced gradient (SVRG)** — potentially important for very large
   datasets, but it requires a real stochastic-training service that neuron does not yet
   have.
7. **AMSGrad (the convergent Adam-family candidate)** — useful as a robust stochastic
   baseline, but unlikely to beat good full-batch methods on neuron's current small and
   medium deterministic problems.

The first production experiment should compare **L-BFGS, iRPROP+, and safeguarded BB**.
LM should be prototyped in parallel only as a constrained LMS specialization.  SVRG and
AMSGrad should wait until measurements show that full passes over large datasets, rather
than poor conditioning, dominate elapsed time.

“Fastest” cannot honestly be a single universal ordering.  LM may be first on a network
with fewer than a few hundred weights and squared-error loss; SVRG may rise to first on a
dataset so large that repeated full gradients dominate.  The ranking above instead asks:
which implementation is most likely to improve the largest share of neuron's existing
logistic, `SimpleProp`, `BareProp`, and `BackProp` fits per unit of added complexity?

## Current architecture and constraints

The relevant production path is:

`Iterative::train()` -> concrete `trainSet()` -> concrete `innerTrainSet()` ->
`Network::engine()` -> concrete weight update.

- `Iterative` owns run lifecycle, iteration count, stopping reasons, cancellation,
  progress, validation monitoring, and the training loop (`src/iterative.*`).
- `Network` owns optimizer selection and the existing packed-vector state `stackG`,
  `lastG`, and `lastF`; `engine()` currently transforms gradients for conjugate-gradient
  descent and Shanno (`src/network.*`).
- `SimpleProp`, `BareProp`, `BackProp`, and `Logistic` own forward propagation,
  gradient formation, weight structures, and `pack()`/`unpack()`
  (`src/simpleprop.*`, `bareprop.*`, `backprop.*`, and `logistic.*`).
- `Matrix`, `vector_ops`, and `Population` are the required numerical vocabulary.
  Optimizer work must add destination-taking primitives there when needed rather than
  introducing raw arrays.
- The current `engine()` contract is enough for a direction-only method.  It is **not**
  enough for L-BFGS or LM, which must evaluate trial points and accept/reject a step.
  Those methods need a once-per-iteration optimizer service; they must not smuggle a line
  search through a per-exemplar callback.
- CGD and Shanno currently assume a true batch gradient.  The same restriction applies
  to L-BFGS, iRPROP+, BB, and LM.  Nesterov, SVRG, and AMSGrad can define online behavior.
- Adding any public algorithm changes the frozen CLI choice list and therefore requires
  a GUI control, HTTP token, `docs/gui_cli_parity.md`, strict parsing, auto-selection,
  OBD, cross-validation, reporting, cloning, and save/continued-training decisions.

### Shared integration shape (DRY without hiding equations)

Before adding an algorithm, introduce the smallest common cold-path interface that the
first measured winner actually needs:

1. A packed parameter vector distinct from `stackG`, with model-owned
   `packWeights(destination)` and `unpackWeights(source)` implementations.  Packing
   layout must exactly match the existing gradient `pack()` layout.
2. One full-batch `evaluateObjectiveAndGradient(parameters, objective, gradient)` entry
   point owned by each concrete model or by the lowest existing model base with identical
   equations.  It calls each model's existing forward/backprop implementation; it does
   not duplicate those formulas in an optimizer.
3. Destination-taking vector primitives for `axpy`, elementwise sign/product,
   elementwise square/root/max, and fused scaled addition if measurement shows they are
   missing.  Size checks occur once at entry; no allocation occurs in element loops.
4. Algorithm-specific state classes (for example `LBFGSState` and `IRpropState`) owned
   by `Network` or an optimizer service.  Do not make a generic map of named arrays, a
   sign flag, or an `std::function` update rule merely to share loops.
5. Reset optimizer history when weights are randomized, loaded, structurally changed,
   or a materially incompatible optimizer configuration changes.  Copy/clone either
   copies state for true continued training or explicitly resets it; tests must pin the
   chosen contract.  Saved network files should remain prediction-only unless resumable
   optimizer state is made an explicit versioned feature.

This shared layer should be implemented only after a disposable prototype demonstrates
a worthwhile speedup.  Otherwise it is speculative architecture.

## Common characterization, testing, and benchmark protocol

All candidates use the same evidence protocol so “fewer iterations” cannot masquerade as
“faster.”

### Characterization before implementation

- Record current canonical, CGD, and Shanno results for logistic, biased `SimpleProp`,
  unbiased `BareProp`, and multi-layer `BackProp`; include batch and online modes where
  the algorithm supports them.
- Use fixed datasets, fixed splits, fixed seeds, and serialized identical starting
  weights.  Include well-scaled, poorly scaled, mildly collinear, saturated, and
  non-separable/nonlinear fixtures.
- Record objective including the same weight-decay convention, maximum raw gradient,
  predictions, final weights, iterations/passes, exemplar evaluations, stop reason,
  validation endpoint, numerical failures, and elapsed time.
- Existing behavior characterization must pass before implementation.  Every new guard
  must then be proven by sabotage, with affected translation units visibly recompiled
  both after sabotage and after restoration, as required by `docs/development_rules.md`.

### Correctness tests for every retained optimizer

- A tiny quadratic with a known minimizer tests the published update independently of a
  network.
- A one-iteration test pins every state vector and coefficient against a hand or
  independent high-precision calculation.
- Finite-difference gradient checks cover every target model, loss, bias mode, and weight
  decay; the analytic objective and gradient must refer to the same point.
- Fixed-seed integration tests require nonempty finite results, a decreasing comparable
  objective, correct stop/cancel/ceiling behavior, and valid continued-training/reset
  semantics.
- Clone, OBD, auto-algorithm, stepwise, CV, blocking API, async API, busy gate, strict
  parser, GUI control, save/load prediction, input removal, and hidden-node grow/prune
  tests are updated if the algorithm is public.
- Release/`NDEBUG` is tested.  Degenerate denominators, NaN/Inf gradients, zero gradients,
  overflow, and rejected trial steps get explicit failure contracts.
- Goldens remain byte-identical until an explicitly authorized public-surface change;
  no golden is re-blessed to conceal a behavior change.

### Benchmark design

- Add a built-but-not-CTest `optimizer_probe` executable and a deterministic driver that
  emits CSV/JSON: dataset/model, dimensions, seed, initial-weight hash, optimizer and
  configuration, endpoint objective, passes, gradient/objective evaluations, stop
  reason, elapsed nanoseconds, and failure.
- Benchmark Release builds on one otherwise idle machine.  Warm up once, then interleave
  randomized optimizer order for at least 15 measured repetitions per cell.  Report
  median, MAD, p10/p90, and peak resident memory; do not compare isolated best runs.
- Primary endpoint: wall time to reach the **same predeclared objective tolerance** as a
  converged control.  Secondary endpoints: time to a matched prediction tolerance,
  time to the existing stop rule, passes, evaluations, and robustness across seeds.
- Include tiny problems (fixed overhead), medium neural/logistic problems, large-row
  problems, and parameter-count sweeps.  Separate training time from the statistical
  report and ROC bootstrap.
- A candidate advances only if it improves median elapsed time beyond run-to-run spread,
  reaches a comparable endpoint, and does not trade speed for materially more failures.
  Publish negative measurements in `docs/HISTORY.md` and remove disposable code.

## 1. Limited-memory BFGS (L-BFGS)

**Why it is first.** neuron presently computes deterministic full-batch gradients and
already pays for many passes during convergence.  L-BFGS obtains curvature-scaled search
directions using only a small history of parameter and gradient differences.  Liu and
Nocedal reported that it was faster than the compared limited-memory/conjugate method and
could exploit additional storage for acceleration.  This is the broadest strong fit for
both logistic and neural objectives.

**Verified primary source.** Dong C. Liu and Jorge Nocedal, “On the limited memory BFGS
method for large scale optimization,” *Mathematical Programming* 45, 503–528 (1989),
[doi:10.1007/BF01589116](https://doi.org/10.1007/BF01589116).  The publisher record and
bibliographic details are available at
[Springer](https://link.springer.com/article/10.1007/BF01589116).

**Where it goes.** Add `src/lbfgs.h/.cpp` as the authoritative two-loop-recursion and
line-search service.  `Network` owns an `LBFGSState` because parameters and packed
gradients are common to all network models; concrete models retain objective/gradient
evaluation and weight packing.  `Iterative` continues to own stopping and cancellation.

**Specific implementation.** For history length `m` (prototype 5, 10, and 20), retain
accepted pairs `s_k = w_{k+1}-w_k`, `y_k = g_{k+1}-g_k`, and
`rho_k = 1/(y_k^T s_k)`.  Compute the direction with Algorithm 7.4's standard two-loop
recursion: walk history newest-to-oldest to form `q`, scale by
`gamma_k=(s^T y)/(y^T y)`, then walk oldest-to-newest to form `r`; use `p=-r`.
Reject or reset history when curvature is nonpositive or nonfinite.  Use a safeguarded
strong-Wolfe line search with explicit evaluation and iteration caps, cancellation checks
between trial evaluations, and restoration of the last accepted weights on failure.
Define whether the reported “iteration” is an accepted outer step; additionally count
objective/gradient evaluations.  Weight decay must be present identically in objective
and gradient.  Online mode is refused, not silently coerced.

**DRY plan.** Reuse packed weights/gradients, `dotprod`, `maxabs`, and new destination
`axpy`; keep the two-loop equation once in `LBFGSState::direction`.  All four models call
the same service but evaluate their own equations.  Do not reuse Shanno through flags:
full BFGS-like and limited-memory recursions have different state and acceptance rules.

**Other updates.** `network.*`, each model's packing/evaluation boundary,
`vector_ops.*`, `modelfactory`/`netclone`, `autoalgo`, `obd`, CV adapters/reporting,
CLI menu, GUI algorithm select and `/api/train`, strict parsing, run headers, action log,
`docs/gui_cli_parity.md`, `AGENTS.md`, CMake, tests, and possibly async progress fields for
evaluation count/line-search phase.  Decide explicitly whether auto-selection may compare
methods with multiple trial evaluations under the same wall-clock budget.

**Manifest placement.** Add the published derivation and neuron's line-search adaptation
under Chapter 2, `Parameter Estimation Formulas` -> `Example learning algorithms`.
Document `LBFGSState` and its public service in Chapter 12 Object Design after `Network`,
then update `Network`, each concrete model's evaluation method, `Iterative` progress and
failure semantics, vector primitives in Chapter 5, CLI/REST tables, object figure, index,
and `tools/check_manifest_index.py`.

**Testing.** In addition to the common suite, test the two-loop recursion against an
explicit dense inverse-BFGS calculation on small positive-definite quadratics; pin strong
Wolfe inequalities, rejected curvature pairs, history rollover, line-search exhaustion,
restoration, and cancellation during a trial evaluation.

**Benchmarking.** Sweep history 5/10/20 and line-search caps against canonical, CGD, and
Shanno.  Report both wall time and full objective/gradient evaluations because a single
L-BFGS iteration can cost several passes.  Stratify by parameter count and conditioning.

## 2. Improved resilient propagation with weight backtracking (iRPROP+)

**Why it is second.** iRPROP+ uses only the sign pattern of successive full-batch
gradients and one step size per parameter.  It avoids a line search and is insensitive to
gradient magnitude, making its per-epoch overhead tiny and attractive on poorly scaled
neural objectives.  The 2003 comparison reports improved RPROP outperforming the tested
RPROP variants, conjugate gradient, Quickprop, and BFGS over its neural benchmarks, with
BFGS better only in some late-stage cases.

**Verified primary sources.** Martin Riedmiller and Heinrich Braun, “A Direct Adaptive
Method for Faster Backpropagation Learning: The RPROP Algorithm,” IEEE ICNN 1993,
586–591, [doi:10.1109/ICNN.1993.298623](https://doi.org/10.1109/ICNN.1993.298623),
[paper PDF](https://www.cs.cmu.edu/afs/cs/user/bhiksha/WWW/courses/deeplearning/Fall.2016/pdfs/Rprop.pdf).
Christian Igel and Michael Hüsken, “Empirical evaluation of the improved Rprop learning
algorithms,” *Neurocomputing* 50, 105–123 (2003),
[doi:10.1016/S0925-2312(01)00700-7](https://doi.org/10.1016/S0925-2312(01)00700-7),
[paper PDF](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2003-Neuro-Igel-IRprop%2B.pdf).

**Where it goes.** `src/irprop.h/.cpp` owns per-parameter previous gradients, update
values, last weight changes, and previous objective.  `Network` owns the state and invokes
it once after a full-batch raw gradient is packed.  It is refused for online mode.

**Specific implementation.** Initialize every `Delta_i` (paper default 0.1).  With
`etaPlus=1.2`, `etaMinus=0.5`, `DeltaMin=1e-6`, and `DeltaMax=50` as the
Igel/Hüsken experimental settings:
if `g_i(t-1)g_i(t)>0`, enlarge `Delta_i` and step opposite `sign(g_i)`; if the product is
negative, shrink `Delta_i`, set the current stored gradient to zero, and—when the current
objective is worse than the preceding objective—undo the previous weight change; if zero,
step with unchanged `Delta_i`.  Confirm the exact iRPROP+ branch from the Igel/Hüsken
algorithm table during prototype implementation.  Apply weight decay by adding its
derivative to the batch gradient and its penalty to the objective before sign comparison.

**DRY plan.** One packed-vector state update serves all models; elementwise sign/product
and clamping belong in `vector_ops` only if destination-taking forms are generally useful.
Do not combine iRPROP+, RPROP-, and other variants behind flags until measurements justify
more than one public variant.  Publish one named equation and implement it once.

**Other updates.** The same public-surface and selection files listed for L-BFGS; add
objective history to copied optimizer state and reset it on structural changes.  Automatic
step-size selection and `eta` are not meaningful for iRPROP+ and must be disabled/refused
with a clear CLI/API contract rather than ignored.

**Manifest placement.** Chapter 2 learning algorithms for the sign/product cases and
defaults; Chapter 12 after `Network` for `IRpropState`; update `Network`, concrete batch
gradient providers, run configuration, CLI/REST, failures, index, and object figure.

**Testing.** Hand-calculate a multi-coordinate sequence that exercises positive, negative,
and zero sign products plus objective-worsened rollback.  Test each clamp, zero gradient,
nonfinite objective, reset/copy, refusal in online mode, and exact inclusion of decay.

**Benchmarking.** Compare time-to-objective across fixed initial step sizes and defaults;
include badly scaled inputs specifically.  Count one objective/gradient pass per epoch and
report late-stage accuracy because RPROP can reach moderate error quickly yet lose to
quasi-Newton methods near a strict optimum.

## 3. Safeguarded Barzilai-Borwein (BB) spectral gradient

**Why it is third.** BB approximates secant curvature with one scalar step using only the
last parameter and gradient differences.  It adds two vectors and a few dot products to a
normal batch epoch—less state and computation than L-BFGS—and is therefore an excellent
low-risk speed experiment.  Its raw form is nonmonotone and needs safeguards on general
nonconvex neural objectives.

**Verified primary source.** Jonathan Barzilai and Jonathan M. Borwein, “Two-Point Step
Size Gradient Methods,” *IMA Journal of Numerical Analysis* 8(1), 141–148 (1988),
[doi:10.1093/imanum/8.1.141](https://doi.org/10.1093/imanum/8.1.141); the author's
[publication index](https://www.jonborwein.org/jmbpapers/) verifies the citation.

**Where it goes.** `src/spectralgd.h/.cpp` owns `wPrevious`, `gPrevious`, accepted
spectral step, and line-search window.  It consumes packed full-batch parameters and raw
gradients once per epoch through `Network`; concrete models remain unchanged beyond the
common packed evaluation boundary.

**Specific implementation.** Let `s=w_k-w_{k-1}` and `y=g_k-g_{k-1}`.  Prototype both
published steps `alpha_BB1=(s^T s)/(s^T y)` and
`alpha_BB2=(s^T y)/(y^T y)`, then predeclare selection (alternating is a separate
adaptation).  Clamp alpha to measured safe bounds; if a denominator is nonpositive,
near-zero, or nonfinite, fall back to a known safe eta and reset history.  Because raw BB
is not monotone, add a clearly cited/derived nonmonotone Armijo safeguard before production
or retain the raw method as research-only.  Trial points must be restored on rejection.

**DRY plan.** Reuse the same packed-weight/evaluation boundary as L-BFGS and shared dot/
axpy primitives.  Keep BB's scalar secant formula in its own service; do not make L-BFGS
a generic “number of history vectors” implementation with `m=1`, because its direction
and acceptance rule are not BB.

**Other updates.** Same surface, clone, selection, reporting, documentation, and build
updates as L-BFGS.  Define interaction with manual eta (initial/fallback only), automatic
step search (mutually exclusive), online mode (refused), and progress evaluation counts.

**Manifest placement.** Chapter 2 learning algorithms for both BB equations and neuron's
safeguards; Chapter 12 after `Network` for `SpectralGradientState`; vector operations,
configuration, REST/CLI, failure semantics, index, and figure as applicable.

**Testing.** Verify BB1/BB2 exactly on diagonal quadratics; test denominator and clamp
boundaries, nonmonotone acceptance window, fallback/reset, trial restoration, and decay.

**Benchmarking.** Sweep BB1, BB2, and only a predeclared small set of safeguards on the
same workloads as L-BFGS.  BB wins only if its cheaper epochs outweigh extra epochs; report
both.  Do not tune per dataset and then report the tuned winner as a default result.

## 4. Levenberg-Marquardt (LM), restricted to least-squares networks

**Why it is fourth.** Hagan and Menhaj found the Marquardt method much more efficient than
the compared methods for feed-forward networks with no more than a few hundred weights.
It can therefore be the absolute wall-clock winner in its niche.  Its Jacobian storage and
normal-equation solve make it unsuitable as neuron's general default, especially for large
models or cross-entropy logistic regression.

**Verified primary source.** Martin T. Hagan and Mohammad B. Menhaj, “Training Feedforward
Networks with the Marquardt Algorithm,” *IEEE Transactions on Neural Networks* 5(6),
989–993 (1994), [doi:10.1109/72.329697](https://doi.org/10.1109/72.329697),
[PubMed record](https://pubmed.ncbi.nlm.nih.gov/18267874/).

**Where it goes.** `src/levenberg.h/.cpp` owns damping, trial acceptance, and the linear
solve.  A model-level Jacobian service belongs at the lowest common class that owns the
identical derivative equations—likely `OneHiddenNet` for `SimpleProp`/`BareProp`, with a
separate `BackProp` implementation.  Do not put Jacobian formation in `Network`.

**Specific implementation.** For residual vector `e` and Jacobian `J`, solve
`(J^T J + mu I) delta = -J^T e`, evaluate `w+delta`, decrease `mu` after improvement and
increase it after rejection according to the cited schedule, restoring weights when
rejected.  Use a stable factorization from the project's numerical layer/GSL rather than
forming an inverse.  Include weight decay consistently (diagonal/rhs contribution and
objective).  Enforce LMS loss, batch mode, a configurable hard parameter/Jacobian-memory
limit, finite values, and a trial cap.  Prefer blocked accumulation of `J^T J` and `J^T e`
if measurement shows full `N x P` storage is prohibitive, while keeping the published
formula visible.

**DRY plan.** Share the damping/acceptance/solve service and Matrix factorization; reuse
each model's forward activations and derivative terms.  Do not force cross-entropy into a
least-squares residual or duplicate backprop equations in the optimizer.

**Other updates.** Add Matrix/GSL factorization support if absent, memory estimates and
refusals, model/loss eligibility in CLI/GUI/API and auto-selection, progress trial count,
clone/reset rules, CMake, tests, parity, and operational documentation.

**Manifest placement.** Chapter 2 derives the Gauss-Newton/Marquardt approximation and
states the LMS-only adaptation; Chapter 5 documents the solve primitive; Chapter 12
documents `LevenbergMarquardt`, Jacobian methods in eligible concrete models, limits,
failures, and lifecycle.  Update CLI/REST, figure, and index.

**Testing.** Compare one step with an independently calculated tiny Jacobian and normal
equation; test accepted and rejected damping changes, singular/ill-conditioned solve,
memory/parameter limit, loss and mode refusals, decay, restoration, and cancellation.

**Benchmarking.** Sweep rows and parameters independently; record Jacobian/normal-matrix
memory and factorization time.  Compare only within LMS-eligible models and identify the
measured crossover where LM becomes slower or too memory-intensive.

## 5. Nesterov accelerated gradient with scheduled momentum

**Why it is fifth.** Momentum costs one velocity vector and a fused update per step and can
reduce oscillation in narrow valleys.  It works in online and batch modes, but published
neural success depends strongly on initialization and momentum scheduling, so expected
out-of-box wall-clock improvement is less certain than the first three candidates.

**Verified primary/application source.** Ilya Sutskever, James Martens, George Dahl, and
Geoffrey Hinton, “On the importance of initialization and momentum in deep learning,”
ICML 2013, 1139–1147, [official PMLR paper](https://proceedings.mlr.press/v28/sutskever13.html)
and [PDF](https://proceedings.mlr.press/v28/sutskever13.pdf).  The paper specifies the
look-ahead formulation and a slowly increasing momentum schedule used for neural training.

**Where it goes.** `src/momentum.h/.cpp` owns velocity and schedule state at `Network`
scope.  Look-ahead objective/gradient evaluation must occur at model weights temporarily
set to the look-ahead point; the concrete model still computes the gradient.

**Specific implementation.** Implement the paper's Nesterov form explicitly: construct
the look-ahead parameters from current parameters and velocity, compute the gradient at
that exact point, then update velocity and parameters using the published schedule.
Prototype fixed momentum and the cited schedule, but expose only the measured winner.
Clarify whether one “iteration” means an online exemplar step or batch epoch.  Include
decay in the look-ahead objective gradient, avoid per-step allocations, and reset velocity
on randomize/load/structural change.

**DRY plan.** One packed velocity update is shared.  Online models may need direct
structured destination updates to avoid pack/unpack per exemplar; implement those in the
lowest common concrete base only if profiling proves packing expensive.  Do not merge
classical and Nesterov momentum through an opaque sign/ordering flag.

**Other updates.** All public selection/parity files; initialization policy deserves a
separate measured decision but should not be silently changed with optimizer selection.
Define interactions with eta, autostep, batch/online, decay, cloning, and continuation.

**Manifest placement.** Chapter 2 learning algorithms for look-ahead equations and
schedule; Chapter 12 after `Network` for state/lifecycle and concrete evaluation point;
configuration, CLI/REST, failures, index, and object figure.

**Testing.** A two-step hand calculation must distinguish Nesterov from classical
momentum; verify the gradient is evaluated at look-ahead weights, schedule boundaries,
batch/online cadence, reset/copy, decay, cancellation, and no allocations in the measured
inner path.

**Benchmarking.** Predeclare a small schedule grid and compare matched eta budgets across
conditioned and ill-conditioned problems.  Include pack/unpack cost and both online and
batch cases; report sensitivity across seeds rather than only the best momentum value.

## 6. Stochastic variance-reduced gradient (SVRG)

**Why it is sixth.** SVRG targets the large-data regime: it reduces stochastic-gradient
variance without storing a gradient for every observation.  neuron currently favors full
passes and has no mini-batch scheduler, so this is a larger architectural change whose
benefit will appear only when dataset size makes full-gradient methods expensive.

**Verified primary source.** Rie Johnson and Tong Zhang, “Accelerating Stochastic Gradient
Descent using Predictive Variance Reduction,” NeurIPS 2013,
[official proceedings page](https://papers.nips.cc/paper_files/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html)
and [paper PDF](https://papers.nips.cc/paper_files/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf).

**Where it goes.** `src/svrg.h/.cpp` owns snapshot parameters, full snapshot gradient,
inner-loop counters, and seeded sampling.  Concrete models need a no-allocation
`exemplarObjectiveGradient(row, destination)` service.  Dataset sampling/order belongs in
a data/training-plan service, not in model formulas.

**Specific implementation.** At snapshot `wTilde`, compute full gradient `mu`.  For each
seeded sampled exemplar `i`, compute
`v = grad f_i(w) - grad f_i(wTilde) + mu`, then `w -= eta*v`; after `m` inner updates,
choose the next snapshot exactly according to the selected paper option.  Cache or
recompute snapshot activations only after measuring the memory/time tradeoff.  Define
weight decay once at objective level so it is not accidentally counted in both stochastic
terms.  Sampling must be reproducible, cancellation-aware, and free of allocation and
virtual dispatch inside the exemplar loop.

**DRY plan.** Share only seeded index planning and packed vector arithmetic.  Each model
owns its exemplar gradient.  Do not turn `innerTrainSet()` into a generic callback loop;
that would add hot-loop dispatch and obscure model equations.

**Other updates.** New batch-size/inner-length/sampling configuration, seed propagation,
progress semantics, API strict parsing, GUI controls, CLI parity, auto-selection budget,
CV leakage review, cloning, OBD, save/continuation, and dataset-plan tests.  This is large
enough to require a separately reviewed architecture proposal after baseline measurement.

**Manifest placement.** Chapter 2 for the estimator and assumptions; Chapter 12 for SVRG
state and the sampling-plan service; concrete exemplar-gradient contracts, Dataset policy,
Iterative progress/stops, CLI/REST, failures, figure, and index.

**Testing.** Verify the estimator against hand gradients and that its expectation matches
the full gradient over all indices; pin seeded sequences, snapshot cadence, decay counted
once, online stopping, clone/reset, and leakage-free fold-local sampling.

**Benchmarking.** Use row-count sweeps into hundreds of thousands with fixed parameter
counts.  Count exemplar gradient evaluations—not epochs alone—and compare time-to-matched
objective with L-BFGS, BB, canonical online SGD, and full-batch controls.  Determine the
dataset-size crossover before considering a public option.

## 7. AMSGrad (Adam-family stochastic baseline)

**Why it is seventh.** Adam is inexpensive, popular, and designed for noisy/sparse
stochastic gradients, but those are not the dominant characteristics of current neuron
training.  In addition, the original Adam convergence argument has known problems.
AMSGrad adds long-term second-moment memory to address a demonstrated nonconvergence mode;
it is the safer Adam-family research candidate, though later proof discussion means tests
and empirical acceptance remain essential.

**Verified primary sources.** Diederik P. Kingma and Jimmy Ba, “Adam: A Method for
Stochastic Optimization,” ICLR 2015, [arXiv:1412.6980](https://arxiv.org/abs/1412.6980).
Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar, “On the Convergence of Adam and Beyond,”
ICLR 2018, [official ICLR page](https://iclr.cc/virtual/2018/poster/78) and
[paper PDF](https://www.satyenkale.com/papers/amsgrad.pdf).

**Where it goes.** `src/amsgrad.h/.cpp` owns first moment, second moment, maximum second
moment, and time step in `Network` for batch mode; a later stochastic service can apply it
per exemplar/mini-batch.  Concrete models provide raw gradients.

**Specific implementation.** Transcribe Reddi et al.'s AMSGrad algorithm table exactly:
compute `m_t=beta1*m_(t-1)+(1-beta1)g_t`,
`v_t=beta2*v_(t-1)+(1-beta2)g_t^2`, and
`vhat_t=max(vhat_(t-1),v_t)` elementwise, then update with the paper's `alpha_t`, projection
(the identity for neuron's unconstrained weights), and denominator.  Do not import Adam's
bias-correction or epsilon convention into AMSGrad unless a separately named adaptation is
specified and tested; do not label ordinary Adam as AMSGrad or silently switch equations.
Decide coupled L2 penalty versus decoupled weight decay explicitly;
the current neuron penalty is part of the objective, so coupled gradient treatment is the
behavior-preserving starting point.  Reject nonfinite moments and prevent time-step
overflow.

**DRY plan.** Reuse destination elementwise square/max/root and packed gradients; keep the
AMSGrad equation in one state class.  Adam and RMSProp should not be added as flag variants
unless separate measurements justify their permanent surface burden.

**Other updates.** Beta/epsilon configuration and parsing, UI/API/CLI parity, run header,
auto-selection, OBD/CV/report naming, clone/reset/continued training, step counting,
CMake, tests, and docs.  If mini-batches are added, share the SVRG sampling-plan service
without sharing their different estimators.

**Manifest placement.** Chapter 2 for the exact moment equations, step schedule,
denominator safeguards, and decay policy; Chapter 12 for `AMSGradState`; vector primitives, configuration,
CLI/REST, lifecycle/failures, figure, and index.

**Testing.** Reproduce several steps from an independent scalar/vector calculation;
exercise the step schedule, elementwise maximum, denominator safeguard, zero/sparse gradients,
nonfinite state, overflow, reset/copy, decay, and batch/online cadence.

**Benchmarking.** Compare fixed published defaults and a predeclared small learning-rate
grid, never a large tuned grid unavailable to controls.  Focus on large/noisy/poorly scaled
and online cases.  Report time-to-matched objective, variance across seeds, and final
statistical endpoints; Adam-family rapid early loss reduction is not enough if the fitted
objective or coefficients remain inferior.

## Recommended execution sequence and decision gates

1. **Baseline first:** build `optimizer_probe` around existing canonical, CGD, and Shanno
   without changing production behavior.  This identifies whether gradient formation,
   packing, step-size search, or reporting actually dominates.
2. **Disposable prototypes:** implement L-BFGS, iRPROP+, and BB one at a time at the
   concrete/optimizer layer.  Do not expose menus or GUI fields yet.
3. **First gate:** retain only candidates with a reproducible wall-clock win to a matched
   endpoint across more than one model family.  A useful target for accepting permanent
   public complexity is at least a 20% median improvement beyond observed spread, or a
   much larger win in a clearly documented important workload.
4. **Specialized prototype:** test LM on LMS networks below measured memory/parameter
   thresholds.  It may be retained as an explicitly eligible specialized solver even if
   it is not a general optimizer.
5. **Second gate:** if large-row profiling shows full gradients dominate, prototype SVRG;
   otherwise defer both SVRG and AMSGrad.  Prototype Nesterov earlier only if existing
   online training is an important measured workload.
6. **Production integration:** only after a winner is selected, add the public CLI + GUI
   + API option, auto-selection policy, Manifest contract, fail-proven tests, and complete
   proportional gates.  This prevents seven speculative algorithms from permanently
   expanding the public surface.

## What is deliberately not recommended first

- **Plain Adam or RMSProp:** optimized for noisy stochastic settings and weaker than
  AMSGrad as a convergence-conscious research baseline; adding all variants would create
  surface area without evidence.
- **Full BFGS/Newton:** quadratic parameter memory, and potentially cubic factorization,
  are unjustified while L-BFGS offers a broad curvature-aware trial.  Logistic-specific
  Newton/IRLS could be a separate future statistical-solver study, not mislabeled as a
  general network learning algorithm.
- **Quickprop:** iRPROP+ has more directly favorable comparative neural evidence and a
  cleaner sign-based full-batch update.
- **Hessian-free/truncated Newton:** valuable for much larger/deeper networks than neuron
  normally fits, but substantially more complex than L-BFGS and LM.  Reconsider only if
  parameter sweeps show L-BFGS history is insufficient and LM memory is prohibitive.
- **A generic external optimizer wrapper:** it would encourage callbacks, allocation,
  opaque equations, and inconsistent ownership.  An external implementation may be used
  as an oracle in tests, but production equations must remain visible in neuron's class
  and numerical vocabulary.

## Final conclusion

The strongest first bet is **L-BFGS**, followed closely by **iRPROP+** and a
**safeguarded Barzilai-Borwein** method.  They match neuron's deterministic full-batch
architecture, require only `O(P)` or `O(mP)` state, and have credible prospects of reducing
expensive dataset passes without introducing per-exemplar dispatch.  **LM** deserves a
separate, sharply bounded LMS experiment because it may be exceptionally fast on small
networks.  **Nesterov, SVRG, and AMSGrad** are later candidates whose value depends on
online or truly large-data workloads that must first be measured.

No ordering in this document substitutes for the required benchmark.  The production
decision is the measured wall-clock time to a comparable numerical and statistical result,
with identical starting state and reported variance.
