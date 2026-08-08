# Neural optimizer handoff

Date: 2026-08-08 (Phase 1 complete: BB rejected)

## Snapshot

- Repository: `/Users/craign/code/neUROn2++/neuron-3.0`
- Branch: `main`; inspect `git status -sb` rather than relying on this snapshot.
- Last substantive commit before this phase: `7f17e5a`. Phase 1 (BB) then landed
  as a documentation-only commit: the candidate was screened and rejected, so its
  prototype was removed and no engine source changed.
- L-BFGS and iRPROP+ implementation, measurement, retain decisions, public REST
  integration, GUI controls, Manifest work, and the portfolio policy are
  committed and pushed. The Phase 4 tests, harness arms and evidence are included.
  Nothing from either phase remains only in the build directory or an untracked
  file.
- `tests/optimizer/data/` is generated and not committed; regenerate it with
  `python3 tests/optimizer/prepare_data.py` before running a campaign.
- **CI history, corrected.** Actions did not dispatch for the L-BFGS push during
  GitHub's 2026-08-06 outage, but it recovered: run 31131633513 on `63c0a4e`
  passed on all three platforms. The first Phase 4 push (`0388d01`) then FAILED
  on all three — build and all 38 tests passed everywhere, and the only failing
  step was `run_tools.sh` hitting the Manifest index check. That is what the
  Manifest work below closes. If a future push produces no run at all, retrigger
  with an empty commit; a no-op `git push` does not reliably create a push
  event.

Immediately after a restart, run:

```sh
cd /Users/craign/code/neUROn2++/neuron-3.0
git status -sb
git log -5 --oneline --decorate
```

Expect a clean `main` **one commit ahead of `origin/main`** until the Phase 1
commit is pushed, and in sync once it has been; either way its CI result must be
confirmed. Do not discard unexpected changes; identify their owner before
proceeding. Reconfigure or rebuild only if the existing `build/` directory is
unavailable or invalid.

## Read first after restart

1. `CLAUDE.md`
2. `docs/learning_research/optimizer_implementation_plan.md`, especially the
   **Large-workload speed scope governor** and the Phase 5 **portfolio policy**
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

## Phase 3 is complete: L-BFGS is the new bar

Read `docs/learning_research/lbfgs_screen_results.md` for the evidence and
`lbfgs_source_decision.md` for the algorithm, its sources and every constant.

On the committed practical endpoint, from identical starting weights, the
L-BFGS implementation beat Shanno by **8.8x to 13.0x** at 6,000,
25,000 and 100,000 rows and across four weight seeds, taking **14-20 full
training-set traversals against Shanno's 123-210**. Every trial evaluation the
line search makes is counted. Decision: **RETAIN** — the measurements justify
the next phase. Its public REST integration is also complete: `POST /api/train`
selects it with `algorithm=4`, with optional positive `lbfgs_memory`. It is
neural, batch-only, and requires `autostep=0`. The retired CLI menus deliberately
have no L-BFGS entry. The GUI includes it; automatic selection remains deferred
until the researched optimizer set is complete.

So the speed bar moved off Shanno: roughly 15 full passes on a 4,500-row problem.
It is **not** a winner-takes-all retention rule, and Phase 4 is the concrete
demonstration — see below, where iRPROP+ beats L-BFGS on the application
benchmark and loses to it on the generated fixtures and late-stage.

Two results that must travel with the headline:
- the held-out differences are **noise** (five of six matched groups favour
  L-BFGS, one favours Shanno by the largest margin of the six), so this
  supports *no degradation*, never *better*;
- **late-stage it is not strictly dominant**: run to the engine's own plateau
  rule it is 22x faster but lands slightly *worse* on both training objective
  and held-out error.

## Established incumbent (superseded as the bar; still the control)

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
- Step 0B was committed as `a9525b0 Characterize neural optimizer baseline`. It is
  deliberately partial and CLOSED; its retained result is the incumbent above.
- Phase 3 landed as the behavior-preserving packed-boundary extraction, the
  L-BFGS implementation, its measurements and retain decision, and its public
  REST integration. The relevant commits are `6bc343b`, `4ced9c2`, `b132922`,
  and `71b9b34`.
- The restart/status correction is `6ac2ad7`; the standing portfolio policy is
  `e7a8188`.
- Phase 4 landed as three commits. `0388d01` is the prototype: `src/irprop.*`, the
  `Network` composition and absolute-step path,
  `tests/network/check_irprop.cpp` (ctest `network_irprop`), the harness's
  iRPROP+ arms and conditioning fixtures, and the evidence documents. The commit
  after it synchronizes the Manifest, Figure 12.1 and the index guard. The
  subsequent `d712d30` integration exposes the retained method through REST and
  the GUI as `algorithm=5`, without changing the retired menus, and closes the
  Chapter 4 optimizer-selection and maintenance contracts.
- Phase 1 (safeguarded BB) landed as a single documentation-only commit. Its
  prototype (`src/gbb.*`, `Network::TRAIN_GBB`, `tests/network/check_bb.cpp` and
  the harness BB arms) existed only within that phase's working tree and was
  removed on rejection, so no engine source differs from `7f17e5a`.
- `CLAUDE.md` and the implementation plan contain the corrected neural-only scope.
- Re-run the focused gates before committing rather than relying on this handoff;
  never copy a remembered CTest count into a status report.

## Phase 4 is complete: iRPROP+ is retained and directly selectable

Read `docs/learning_research/irprop_screen_results.md` for the evidence and
`irprop_source_decision.md` for the algorithm, its sources and every constant.

The implementation is `src/irprop.*` (`IRpropState`), composed by
`Network::irpropIteration()` through the packed boundary Phase 3 built, applied
by the new `Network::applyAbsoluteStep()` — the explicit absolute-step path that
architecture decision 8 requires. `Network::TRAIN_IRPROP = 4` maps to public
`POST /api/train` `algorithm=5`; the GUI exposes the same choice and locks batch
training on and automatic step-size search off. No menu token, automatic-selection
entry or saved-network field produces it.

Decision: **RETAIN under the portfolio policy**, not as a new sole bar. On Civic
Choice it is the fastest of the panel at 6,000/25,000/100,000 rows and four
weight seeds — 1.28x to 2.08x faster than L-BFGS, 11x to 27x faster than Shanno,
1,607x faster than canonical at 6,000 rows — with zero failures in 346 rows.

Three results that must travel with the headline:

- **the ranking flips by workload family.** On the generated 4-input conditioning
  fixtures L-BFGS is ~2x faster than iRPROP+. iRPROP+ is not the panel's fastest
  method in general; it is the fastest on the application benchmark. Neither
  method makes the other redundant, which is exactly the case the portfolio
  policy exists for;
- **the poorly-scaled hypothesis failed.** The plan predicted iRPROP+ would excel
  on poorly scaled objectives. Measured directly against a well-scaled twin, the
  conditioning penalty is 1.14x for iRPROP+ and 1.23x for L-BFGS — both almost
  unaffected, and L-BFGS ahead on both. Whatever separates the two workload
  families, it is not conditioning;
- **late-stage it is not dominant.** Run to the engine's own plateau rule it is
  1.85x slower than L-BFGS and lands worse on both training objective and
  held-out error.

Held-out differences at the matched endpoint are **noise** (seven of eight
matched groups favour iRPROP+, one favours L-BFGS by a margin comparable to the
largest of the seven), so this supports *no degradation*, never *better*.

### The Manifest is synchronized — and deferring it was a mistake worth recording

`IRpropState` is documented in Chapter 12 of the Manifest (§12.7.4), indexed,
added to Figure 12.1 as a `Network API` composition beside `LBFGS`, and covered
by `tools/check_manifest_index.py`. Its retained direct-training integration is
REST/GUI-only as `algorithm=5`; the retired menus remain unchanged and automatic
selection remains deferred until the researched optimizer set is complete.

**Why this is called out.** The prototype was first committed with that gate
deliberately failing, on the reasoning that Phase 3 had made the same trade —
`4ced9c2` landed `LBFGS` with the index gate red and `71b9b34` documented it
during integration. That reasoning was wrong in a way worth remembering:
**Phase 3's version of the trade was never tested, because Actions did not
dispatch during the 2026-08-06 outage.** When Phase 4 was pushed, CI went red on
all three platforms — build and all 38 tests passed everywhere, and the sole
failure on every platform was `run_tools.sh` hitting the index check. A red
`main` also makes every later commit's CI uninformative, which is exactly when a
real regression needs to be visible. `CLAUDE.md` already required this: the
Manifest is normative and stays synchronized with every public class.

The general rule: **a gate deferred by precedent is still a gate, and "the last
phase did it too" is not evidence that it was acceptable — check whether that
precedent was ever actually exercised.**

## Phase 1 is complete: safeguarded BB was screened and REJECTED

Read `docs/learning_research/bb_screen_results.md` for the evidence and
`bb_source_decision.md` for the algorithm, its citations and every constant.

The candidate was **Raydan's Algorithm GBB** -- BB1 plus the
Grippo-Lampariello-Lucidi nonmonotone line search with safeguarded quadratic
interpolation -- declared in full before any code was written. It was
implemented, both of its declared sabotages were fail-proven, and it was screened
against the five-arm panel on Civic Choice 6,000 rows at the committed practical
endpoint.

Decision: **REJECT**, and the prototype and its public roots have been removed.
`src/gbb.*`, `Network::TRAIN_GBB`, the dispatch, `tests/network/check_bb.cpp` and
the harness's BB arms are all gone; `tools/check_manifest_index.py` is green and
no Manifest entry was ever written, because BB never had a public token.

**Lead with the stability finding, not a speed ratio.** BB reached every
endpoint, on every arm, on every repetition, and never produced a non-finite or
failed row. But on ONE workload from four predeclared starting weight vectors it
took 25, 17, **11,398** and **4,690** full traversals -- a 670x spread, where
L-BFGS spans 1.43x, iRPROP+ 1.43x and Shanno 1.33x. And between the `well4` and
`poor4` conditioning twins it degrades **153x**, against L-BFGS's 1.23x and
iRPROP+'s 1.11x.

The long runs were tested for exactly the two things that would have made them an
artifact, and are neither: across interleaved repetitions every arm produced a
**bit-identical** traversal count, iteration count, achieved objective and end
weight fingerprint, with wall-clock p10-p90 inside 3% of the median; and every
matched-endpoint row stopped on `min_error` having passed its target, with
`converged: true`. So the long runs are reproducible.

Scope, stated exactly: this is the behaviour of **the declared neuron GBB
implementation under the declared safeguarding policy**, on these workloads. It
is not a result about Barzilai-Borwein methods generally, and it does not carry
to BB2, cyclic or adaptive BB, or a different globalization.

Three further results that must travel with it:

- **it is fastest on nothing.** Third of five at the base seed, third of four at
  seed 101, second on `well4`. Its best placing anywhere is second;
- **late-stage it stops earliest of the three fast methods and lands worst**, on
  both training objective and held-out error;
- **a pre-declared prediction was falsified in BB's favour and is recorded as
  such**: the source decision predicted the line search's trial points would
  dominate the cost. Measured, the search backtracked fifteen times in 11,383
  iterations. The cost is the spectral step itself.

**Neither pre-declared falsification criterion fired, and the rejection is
still pre-declared.** The two candidate-specific criteria were early rejection
gates, not sufficient retention conditions; BB passed them -- it beats canonical
640x at the base seed and reaches every endpoint -- and then failed the
**pre-existing portfolio admission rule** that predates this phase and governed
Phase 4's retention of iRPROP+. It added no winning workload and introduced
extreme, unexplained screening-cost variance. No standard was written or altered
after results became visible. Rejected under the portfolio policy, then: there is no workload a user should be told to pick it for, and
**no predictor was established from the measured workload and
starting-point diagnostics** for which starting weights cost 17 traversals and
which cost 11,398. Four starts justify the rejection; they do not prove that
prediction is impossible. That is the policy doing real work in the opposite direction from
Phase 4, where it retained a method that loses two of three workload families.

**The sabotage result is the one to carry forward.** Deleting the nonmonotone
maximum turned GBB into monotone Armijo BB -- a different published method -- and
`checkNonmonotone()`, the test that asserts nonmonotone acceptance BY NAME, still
passed, because it drove the pieces directly and the sabotage was in the wiring
between them. So did every real-model integration test on all three
architectures. This is Phase 4's iRPROP+/RPROP+ result reproduced one phase
later in a different algorithm: **an assertion that names a mechanism is not
automatically a test of it, and a descending objective proves nothing about which
algorithm produced it.**

## Exact next research step

The panel stands at four arms: canonical, Shanno, L-BFGS and iRPROP+. Phases 0,
1, 3 and 4 are complete; Phase 2 (corrected IRLS/Newton for Logistic) is out of
scope under the program boundary above, which is neural-only.

So the next step is a **new neural candidate**, chosen and declared under the
same staged protocol: pin the published contract and every constant before
coding, fail-prove the mechanism guards by focused sabotage against a
hand-driven, model-free port, then screen on the cheap representative workload
against the four-arm panel. Levenberg-Marquardt and the online/noisy-gradient
methods are the plan's remaining named options, each admissible only when its
eligibility matches a measured neural workload.

Two things Phase 1 established that should shape that choice:

- **the seed panel is a first-class acceptance axis, not a robustness footnote.**
  It is what rejected BB, and it costs almost nothing to run;
- **a method with no per-parameter scaling should be expected to fail the
  conditioning pair.** That pair now has a demonstrated dynamic range of two
  orders of magnitude, so it is worth running early rather than last.

Do not reopen the completed L-BFGS or iRPROP+ phases, and do not begin another
baseline campaign.

No public REST, GUI, Manifest capability, or automatic-selection change belongs in
a research prototype. The retired CLI menus are never extended. Public REST/GUI
integration occurs only after the research acceptance gate; automatic selection
remains deferred until the researched optimizer set is complete.
Eventually every retained eligible algorithm belongs in the bounded limited-run
selector so the user's actual dataset, rather than the synthetic benchmark alone,
guides the full-training choice -- and the Phase 4 ranking flip is the concrete
argument for that selector, since the best method here is workload-dependent.

## Subsequent candidate order

1. ~~L-BFGS~~ — done, retained, the speed leader on the small generated fixtures
   and late-stage
2. ~~iRPROP+~~ — done, retained, the speed leader on the application benchmark
3. ~~Safeguarded BB~~ -- done, REJECTED and removed: correct and always
   convergent, but 670x seed-dependent and 153x conditioning-sensitive, and
   fastest on nothing
4. Other neural candidates such as LM or online/noisy-gradient methods only when their
   eligibility matches a measured neural workload

Each candidate gets the same cheap-screen-first treatment. A tractable Shanno result does
not end the search for a credible further 10x or 100x improvement.

## Last completed verification

These are the Phase 1 gates, run before its commit. Counts were **discovered from
CTest at the time**, not remembered.

- Release build and **all 38 discovered CTest cases** -- back to the Phase 4
  count, because `network_bb` was added and then removed with its prototype;
- golden transcripts -- byte-identical;
- oracle verification -- identical;
- `tests/tools/run_tools.sh` in full;
- `python3 tools/check_manifest_index.py`: **green**, 45 public roots. This was
  RED while the prototype existed (`GBB`, `GBBObjective`) and was deliberately
  not committed in that state -- the Phase 4 lesson below, applied;
- `git diff --check`.

`tests/gui/*` were not re-run: Phase 1 changed no REST, GUI or request-parsing
surface, and its final tree changes no engine source at all.

**Fail-proof evidence for the BB prototype**, recorded here because the code it
guarded no longer exists. Both sabotages applied to `src/gbb.cpp`, rebuilt with
visible recompilation in each direction, and restored:

- **the nonmonotone maximum deleted**, making it monotone Armijo BB. Two
  assertions failed, both naming the wiring. Two controls held, and they are the
  finding: the assertion that names nonmonotone acceptance still passed, and so
  did every real-model integration test on all three architectures;
- **the interpolation bracket removed.** Five assertions failed, three naming the
  safeguard directly.

### The Manifest gate, second time asked and answered correctly

Phase 4 recorded the rule: *a gate deferred by precedent is still a gate, and
"the last phase did it too" is not evidence that it was acceptable.* Phase 1 is
the first phase to face that rule prospectively. Adding `src/gbb.h` turned
`check_manifest_index.py` red on two new public roots, and the phase was **not
committed in that state**. The gate was resolved by the retain/reject decision
itself rather than by documenting a method the project had decided against, or by
exempting it: rejection removed the public roots, and the gate returned green on
its own.

What that establishes for the next candidate: **a research prototype that adds a
class to `src/` owes the Manifest an entry the moment it is retained, and owes
the tree its removal the moment it is rejected.** There is no third state in
which the gate may sit red.

Treat all of this as provenance, not permission to skip new verification. The
next candidate must add and fail-prove its own mechanism tests, then rerun the
relevant Release gates before any commit.
