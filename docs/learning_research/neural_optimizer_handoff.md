# Neural optimizer reboot handoff

Date: 2026-08-06 (ready for a system reboot)

## Reboot snapshot

- Repository: `/Users/craign/code/neUROn2++/neuron-3.0`
- Branch: `main`, synchronized with `origin/main` when this handoff was written.
- Last substantive commit: `e7a8188 Adopt a portfolio policy for optimizers`.
- The working tree was clean before this handoff-only update.
- L-BFGS implementation, measurement, REST integration, Manifest work, and the
  portfolio policy are committed and pushed. Nothing from those tasks remains
  only in the build directory or an untracked file.
- GitHub Actions did not dispatch for the L-BFGS push during GitHub's 2026-08-06
  Actions outage. Full local Release verification passed, but three-platform CI
  must still be confirmed after GitHub recovers. Check the Actions page first;
  if no run exists for current `main`, retrigger it with an empty commit. A no-op
  `git push` alone does not reliably create a push event.

Immediately after reboot, run:

```sh
cd /Users/craign/code/neUROn2++/neuron-3.0
git status -sb
git log -5 --oneline --decorate
```

Expect a clean `main` synchronized with `origin/main`. Do not discard unexpected
changes; identify their owner before proceeding. Reconfigure or rebuild only if
the existing `build/` directory is unavailable or invalid.

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
have no L-BFGS entry. GUI selection and automatic selection remain deferred until
the researched optimizer set is complete.

So the speed bar is now L-BFGS, not Shanno: roughly 15 full passes on a
4,500-row problem. The next candidate is **iRPROP+**. Start with the cheap
representative workload and scale only a plausible portfolio candidate, but do
not use “faster than L-BFGS” as a winner-takes-all retention rule.

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
- `CLAUDE.md` and the implementation plan contain the corrected neural-only scope.
- Re-run the focused gates before committing rather than relying on this handoff;
  never copy a remembered CTest count into a status report.

## Exact next research step

Prototype **iRPROP+** as a research-only neural optimizer and compare it directly with
the plan's standing reference panel: **L-BFGS as the current speed leader, Shanno as the
established legacy quasi-Newton control, and canonical training as the behavioral and
matched-objective reference**. L-BFGS is a reference, not a winner-takes-all gate:
iRPROP+ may be retained without beating it when iRPROP+ is correct, stable, reasonably
competitive, and adds complementary behavior across workloads or failure modes. Do not
reopen the completed L-BFGS phase or begin another baseline campaign. Follow Phase 4 of
the implementation plan: pin the published absolute-step contract before coding, prove
the sign-product and rollback branches with deterministic tests and focused sabotage,
then screen on the cheap representative workload before scaling a plausible portfolio
candidate.

No public REST, GUI, Manifest capability, or automatic-selection change belongs in the
iRPROP+ research prototype. The retired CLI menus are never extended. Public REST
integration occurs only after the research acceptance gate; GUI and automatic selection
remain deferred until the researched optimizer set is complete. Eventually every retained
eligible algorithm belongs in the bounded limited-run selector so the user's actual
dataset, rather than the synthetic benchmark alone, guides the full-training choice.

## Subsequent candidate order

1. ~~L-BFGS~~ — done, retained, now the bar
2. iRPROP+
3. Safeguarded BB as a low-cost candidate/control
4. Other neural candidates such as LM or online/noisy-gradient methods only when their
   eligibility matches a measured neural workload

Each candidate gets the same cheap-screen-first treatment. A tractable Shanno result does
not end the search for a credible further 10x or 100x improvement.

## Last completed verification

Before the REST/Manifest integration commit, the completed gates were:

- Release build and all 37 discovered CTest cases;
- `tests/gui/strictparse.sh` (222 checks);
- `tests/gui/smoke.sh`;
- golden transcripts, oracle verification, and Python tools;
- Manifest index coverage and PDF rebuild/visual inspection;
- `git diff --check`.

Treat these as provenance, not permission to skip new verification. The iRPROP+
phase must add and fail-prove its own mechanism tests, then rerun the relevant
Release gates before any commit.
