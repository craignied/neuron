# Optimizer and learning-algorithm research brief

Load this document only for work on a new gradient-descent or learning
algorithm. It is a research protocol, not a list of preselected algorithms.

## Objective

Evaluate whether a published optimizer materially improves convergence,
robustness, or wall-clock cost for neuron's existing model families without
obscuring their equations, changing statistical endpoints, or adding hot-loop
dispatch/allocation. A novel name is not a reason to implement it.

## Read first

1. `docs/development_rules.md`, especially rules 2, 3, 4, 6, and 7.
2. Manifest sections for `Iterative`, `Network`, the target concrete model,
   matrix/vector operations, and training-control services.
3. `src/iterative.*`, `network.*`, and the target model's `innerTrainSet`,
   `pack`/`unpack`, and gradient ownership.
4. Existing optimizer paths: canonical descent, conjugate-gradient descent,
   Shanno, step-size search, `autoalgo`, and plateau/stopping semantics.

Search `docs/HISTORY.md` or `refactor_audit.md` only for a named historical
question. They are not prerequisite reading.

## Required sequence

1. State the published algorithm and question it is expected to answer. Obtain
   the primary paper or authoritative specification and cite the exact update
   rule, defaults, and convergence assumptions.
2. Map every symbol in the paper to neuron's existing state. Identify whether
   the method needs gradients, batches, moments, line search, curvature, extra
   persistent state, or stochastic sampling. Do not code until ownership is clear.
3. Characterize current behavior on deterministic fixtures spanning at least
   logistic, one biased neural net, one unbiased net where applicable, on-line
   and batch modes, and existing stopping reasons.
4. Build a disposable prototype at the concrete/optimizer layer. Keep published
   equations visible. Do not add a generic flag/comparator/descriptor to make
   unlike algorithms look alike.
5. Measure identical work with fixed seeds and starting weights. Record final
   objective, predictions/weights as appropriate, iterations, stop reason,
   wall-clock distribution, and numerical failures. Interleave candidates and
   controls; report run-to-run spread.
6. Decide explicitly: reject, retain as research-only, or propose production
   integration. A negative result is complete work and belongs in HISTORY.
   Remove a rejected prototype from active source and tests. When the conclusion
   depends on the exact tested implementation, preserve that complete working
   tree under a documented annotated `research/` Git tag before removal; record
   its base commit, known defects and reconstruction command in the evidence.
   An archival tree is not a supported capability, is never merged into
   `main`, and does not excuse a red gate on `main`.
7. Before integration, write characterization that passes on the old engine and
   a new guard proven by fresh-compilation sabotage. Preserve CLI/GUI parity if
   the optimizer becomes selectable.

   **Sabotage must cross the production wiring, not only the components.**
   Testing a mechanism's component operations is insufficient: sabotage the code
   path that *composes* them, and carry a control proving the behavioral
   distinction the sabotage removes. Measured twice, in two different
   algorithms. Deleting iRPROP+'s error-dependent rollback turned it into RPROP+
   and every real-model integration test still passed. Deleting the nonmonotone
   maximum from BB's line search turned it into monotone Armijo BB, and the
   assertion that named nonmonotone acceptance *still passed* — it drove
   `accepts()` and the window directly, and the sabotage sat between them. A
   descending objective proves nothing about which algorithm produced it, and an
   assertion that names a mechanism is not automatically a test of it. When a
   sabotage fails a file, read WHICH assertions failed; if the one naming the
   sabotaged mechanism is not among them, it is decoration.
8. Update the Manifest with the published derivation and citation, neuron's
   adaptation, configuration/state, failure and stopping semantics, performance
   evidence, example, index entries, and object figure if ownership changes.

## Acceptance questions

- What workload improves, by how much, and relative to what variance?
- Does the method reach a comparable objective or merely stop earlier?
- Are results stable across model families, scaling, batch mode, and seeds?
- What memory and per-exemplar costs were added?
- Which class owns the new state and why?
- Can existing `Matrix`/`vector_ops` primitives express the paper directly?
- Does cancellation, ceiling exhaustion, convergence, auto-selection, cloning,
  save/load, and continued training remain well-defined?
- Is the benefit large enough to justify a permanent public option and its
  documentation/testing burden?

## Evidence format

Keep a compact table of dataset/model, seed, starting-state identity, optimizer,
configuration, endpoint objective, iterations, stop reason, elapsed median,
spread, and failures. Preserve scripts only when they are deterministic and
useful as a regression or benchmark; otherwise record the method and result in
HISTORY and remove scratch artifacts.
