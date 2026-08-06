# Neural optimizer research handoff

Date: 2026-08-06

## Read first after restart

1. `CLAUDE.md`
2. `docs/learning_research/optimizer_implementation_plan.md`, especially the
   **Large-workload speed scope governor**
3. `docs/learning_research/optimizer_baseline_results.md`
4. `tests/optimizer/README.md`

## Program boundary

This program investigates novel training algorithms for neuron's **neural** models and
engines so large-dataset training can be made dramatically faster and its runtime can be
estimated. Logistic regression, IRLS, DFA, exhaustive baseline characterization,
stepwise timing, and cross-validation timing are out of scope unless a winning neural
candidate later requires one narrowly relevant end-to-end measurement.

Do not resume the stopped Step 0B campaign. Benchmarking belongs inside each candidate's
staged accept/reject loop. Prefer testing another credible neural algorithm over refining
timing minutiae.

## Established incumbent

On Civic Choice with 6,000 rows, a 25% fixed holdout, 14 inputs, four hidden units, and
identical starting weights, three interleaved runs to the same practical training
objective measured:

| method | median | full passes | relative to canonical |
|---|---:|---:|---:|
| Shanno | 238.7 ms | 194 | 81.4x faster |
| canonical autostep | 2,817 ms | 2,322 | 6.9x faster |
| canonical | 19,423 ms | 15,971 | reference |
| CGD | 38,959 ms | 31,924 | 2x slower |

Shanno is therefore the incumbent neural speed baseline. This is a strong result for one
configuration, not a large-data scaling law. The held-out result supports “no observed
degradation,” not a general claim that Shanno predicts better. Repetition from one fixed
start proves deterministic timing stability, not reliability across initializations.

Canonical stalls above its configured neural gradient threshold. That means canonical
cannot define the late-stage neural endpoint; it does not mean no late-stage endpoint is
possible. When needed, precharacterize a best-known incumbent/Shanno endpoint separately
and do not tune it after seeing candidate results.

## Current repository state

- Step 0A was committed as `91ef241 Add optimizer benchmark harness`.
- Step 0B is deliberately partial. Its retained result is the neural incumbent above.
- The uncommitted Step 0B tree contains no production optimizer changes; it extends the
  research harness, preparation/orchestration tools, tests, and evidence documentation.
- `CLAUDE.md` and the implementation plan contain the corrected neural-only scope.
- At the last report, all discovered CTest cases and goldens passed and
  `git diff --check` was clean. Re-run the focused gates before committing rather than
  relying on this handoff.

## Exact next research step

Prototype **L-BFGS** as a research-only neural optimizer and compare it directly with
Shanno. Do not begin with another baseline campaign.

Use staged gates:

1. Implement the minimum correct packed-weight and pure objective/raw-gradient boundary
   required by the real L-BFGS consumer; do not create a universal callback framework.
2. Prove packing order, round-trip state, analytic/raw-gradient correctness, trial-point
   nonmutation, line-search acceptance/failure, history reset, and the published L-BFGS
   recurrence with deterministic tests and focused sabotage.
3. Screen L-BFGS versus Shanno on Civic Choice 6K/h4 from identical starts to a
   predeclared practical endpoint, with three interleaved repetitions.
4. Reject an obvious loser immediately and record why. Do not compensate with extensive
   tuning after seeing the comparison.
5. If competitive, compare a small predeclared set of defensible memory/line-search
   configurations, then scale only L-BFGS and Shanno to 25K and 100K.
6. Only after it survives scaled timing, test a small predeclared set of weight seeds and
   a late-stage incumbent endpoint. Then decide retain/reject.

No public CLI, GUI, API, Manifest capability, or automatic-selection change belongs in
the research prototype. Public integration occurs only after a candidate wins its full
acceptance gate.

## Subsequent candidate order

1. L-BFGS
2. iRPROP+
3. Safeguarded BB as a low-cost candidate/control
4. Other neural candidates such as LM or online/noisy-gradient methods only when their
   eligibility matches a measured neural workload

Each candidate gets the same cheap-screen-first treatment. A tractable Shanno result does
not end the search for a credible further 10x or 100x improvement.
