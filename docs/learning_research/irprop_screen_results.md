# iRPROP+: the Phase 4 screen

Phase 4 of `docs/learning_research/optimizer_implementation_plan.md`. The
pre-code contract — sources, constants, symbol map, eligibility, tests and the
retention rule — is `irprop_source_decision.md`, written before any code and not
revised after a benchmark was seen.

Decision: **RETAIN**, under the plan's portfolio policy. The default ranking is
recorded separately in section 7, because it is a different question.

## 1. What was run

One campaign, 30 arms, 346 rows, orchestration seed 20260807, Apple Silicon,
Release/NDEBUG, one arm per process, arm order shuffled independently within each
repetition. Reproduce with:

```bash
./tests/optimizer/run_probe.py --irprop --seed 20260807 --timeout 3600 --out irprop.jsonl
```

Every arm is measured against the **standing portfolio panel** — L-BFGS as the
current speed leader, Shanno as the legacy quasi-Newton control, canonical as the
behavioral and matched-objective reference — from identical starting weights, on
one committed split, to one committed objective per group. `run_probe.py` refuses
a group whose members disagree on anything defining the work.

`full_passes` counts **training-set traversals**, not outer iterations. iRPROP+
makes exactly one per iteration, and that is not assumed: `network_irprop`
asserts the table is applied exactly once per traversal on all three neural
models.

## 2. The application benchmark: iRPROP+ is fastest of the panel

Civic Choice, SimpleProp at the walkthrough's four hidden units, the committed
25% stratified holdout and the committed practical endpoint. 15 repetitions per
cell except canonical (5, chosen from its own warm-up by the declared policy).

| group | iRPROP+ | L-BFGS | Shanno | canonical |
|---|---:|---:|---:|---:|
| 6,000 rows, seed 7 | **11.37 ms / 10** | 23.20 / 20 | 222.90 / 194 | 18,262.78 / 15,971 |
| 6,000, seed 101 | **7.77 / 7** | 16.16 / 14 | 209.30 / 181 | — |
| 6,000, seed 202 | **11.35 / 10** | 17.20 / 15 | 179.63 / 158 | — |
| 6,000, seed 303 | **10.34 / 9** | 19.72 / 17 | 241.68 / 210 | — |
| 25,000 | **52.15 / 11** | 66.99 / 14 | 588.57 / 123 | — |
| 100,000 | **209.05 / 11** | 268.19 / 14 | 3,055.03 / 157 | — |

*median ms / full training-set traversals.* MAD is 0.10–1.17 ms on every
15-repetition cell and 11.6 ms on the 100,000-row Shanno cell, so every gap in
the table is many times its own spread.

Against the panel on this benchmark:

- **vs L-BFGS: 1.28x to 2.08x faster**, in six matched groups, all six with the
  same sign. The pass ratios match the time ratios to two figures
  (2.00, 2.00, 1.50, 1.89, 1.27, 1.27), so the win is fewer traversals and not a
  cheaper traversal.
- **vs Shanno: 11.3x to 27.0x faster.**
- **vs canonical: 1,607x faster** at 6,000 rows, the one size where canonical was
  re-run.

**The margin over L-BFGS shrinks as the workload grows** — 2.0x at 6,000 rows,
1.27x at both 25,000 and 100,000. iRPROP+'s traversal count is nearly flat in row
count (10, 11, 11) and so is L-BFGS's (20, 14, 14); the ratio narrows because
L-BFGS improves with size and iRPROP+ does not.

## 3. The conditioning pair: the hypothesis is NOT supported

The plan hypothesized iRPROP+ at "approximately 5–10x, especially poorly scaled
problems". That is the claim this fixture pair exists to test, and it did not
survive.

`well4` and `poor4` are one problem at two conditionings — same rows, same
outcome rule, same architecture, inputs on scales spanning 1000x. The pair is
gated for non-vacuity (`optimizer_harness`), and canonical's own plateau is 21x
worse on the ill-conditioned twin, so the fixture is genuinely harder rather than
merely different.

| fixture | iRPROP+ | L-BFGS | Shanno | canonical |
|---|---:|---:|---:|---:|
| `well4` | 47.18 ms / 53 | **23.67 / 26** | 2,982.05 / 3,328 | 130,129.90 / 145,700 |
| `poor4` | 53.85 / 59 | **29.04 / 32** | 5,583.67 / 6,279 | 72,516.58 / 79,991 |

Two things follow, and the second matters more than the first:

1. **The ranking flips.** L-BFGS is ~2x faster than iRPROP+ on both fixtures —
   the opposite of every Civic Choice group. iRPROP+ is not the panel's fastest
   method in general; it is the fastest on the application benchmark.
2. **Ill-conditioning does not favour iRPROP+.** Going from `well4` to `poor4`
   costs iRPROP+ 1.14x and L-BFGS 1.23x. Both are almost unaffected, and L-BFGS
   remains ahead on the poorly scaled fixture. The hypothesized advantage on
   poorly scaled objectives **did not appear at all**, and the difference between
   the two workload families is therefore *not* conditioning.

What the two families do differ in is size and origin: Civic Choice is 14 inputs
and 65 parameters of real groomed data, the generated pair is 4 inputs and 25
parameters. **Which of those drives the flip is not established here**, and this
screen does not claim it. It is the obvious next question if the ranking ever
needs to be predicted rather than measured.

## 4. Late-stage: iRPROP+ is not dominant

The neural workloads have no strict endpoint, because canonical does not converge
on them at the engine's own 1e-6 rule. So the late-stage question is asked the
only way this workload permits: all three methods run to **the engine's own
plateau rule**, identically configured, and the row declares `endpoint: none`
rather than borrowing a name it has not earned. This is not a matched-endpoint
race, and *where each lands* is the reading, not only how fast it got there.

| method | elapsed | traversals | training objective | held-out |
|---|---:|---:|---:|---:|
| Shanno | 10,492.9 ms | 9,163 | **0.0914732** | **0.0870610** |
| L-BFGS | 468.5 | 416 | 0.0916086 | 0.0874695 |
| iRPROP+ | 865.6 | 749 | 0.0940683 | 0.0884944 |

iRPROP+ is 12.1x faster than Shanno here but **1.85x slower than L-BFGS and lands
worse than both**, on the training objective and on held-out error. A method that
stops earlier at a worse objective has not won. This is the same shape Phase 3
recorded for L-BFGS against Shanno, more pronounced.

These are single observations (n=1): the arms stop on a plateau rather than a
target, so the runner records them once and refuses to average them. The landing
points are deterministic given a fixed start; only the millisecond timings are
unreplicated.

## 5. Held-out behavior: no degradation, and nothing stronger

At the matched endpoint, iRPROP+ minus L-BFGS held-out error across all eight
matched groups:

| group | delta | favours |
|---|---:|---|
| 6k seed 7 | −2.7e−05 | iRPROP+ |
| 6k seed 101 | −4.7e−04 | iRPROP+ |
| 6k seed 202 | **+5.5e−04** | **L-BFGS** |
| 6k seed 303 | −5.1e−04 | iRPROP+ |
| 25,000 | −9.3e−04 | iRPROP+ |
| 100,000 | −1.1e−03 | iRPROP+ |
| `well4` | −1.3e−05 | iRPROP+ |
| `poor4` | −2.3e−04 | iRPROP+ |

Seven of eight favour iRPROP+ and one favours L-BFGS **by a margin comparable to
the largest of the seven**. Mixed signs inside that spread are the signature of
noise, not of an effect. This supports **no observed degradation**. It is not
evidence that iRPROP+ generalizes better, and is not reported as such.

## 6. Stability, failures and memory

- **Zero failures.** 346 arm rows, none non-finite, none with a failure stage.
  The only `usable:false` rows are the three late-stage plateau arms, which are
  `endpoint: none` by construction.
- **Every arm reached its matched target** and stopped on `min_error` with
  `converged` true.
- **Peak RSS is indistinguishable** across the panel (7.46–7.52 MB at 6,000
  rows): at these sizes the dataset dominates and the optimizer's own state is
  not measurable. Structurally iRPROP+ holds three vectors of length *p*
  (previous gradient, Δ, previous step) against L-BFGS's 2*m* history vectors
  plus scratch, so it is the lighter method — but that is a count, not a
  measurement, and no memory claim is made from this campaign.

## 7. Decision

**RETAIN**, against the retention rule fixed in the source decision before any
arm was run:

| criterion | result |
|---|---|
| implements one exact cited algorithm | yes — Igel & Hüsken (2003) iRPROP+, no variant reachable; the error-dependent rollback and the sign-flip zeroing are both fail-proven by sabotage |
| correct | `network_irprop`, driven by hand against the published table |
| stable | zero failures in 346 rows across 4 weight seeds, 3 row counts and 2 conditionings |
| reaches comparable endpoints | every arm reached the matched target, at or below it |
| competitive | fastest of the panel on the application benchmark at every size; ~2x off the leader on the small generated fixtures |
| complementary | **yes, and this is the decisive one:** the ranking between iRPROP+ and L-BFGS *flips by workload family* |

The portfolio policy is explicit that a candidate need not beat the current speed
leader, and that benchmark results rank recommendations rather than hide a
correct, stable, eligible algorithm. iRPROP+ is faster than L-BFGS on the
workload this program is aimed at and slower on another, which is the precise
situation a portfolio exists to hold. Neither method makes the other redundant.

**Reject** was not reached on any of its grounds: it is not prohibitively slow,
not unstable, does not fail the comparable endpoint, and is not redundant.

### The default ranking, which is a separate decision

- **Civic-Choice-like workloads (real groomed data, tens of parameters, thousands
  to hundreds of thousands of rows): iRPROP+ first**, L-BFGS second.
- **Small generated fixtures (4 inputs, 25 parameters), either conditioning:
  L-BFGS first**, iRPROP+ second.
- **Late-stage, run to a plateau rather than a matched target: L-BFGS**, which
  lands better and faster than iRPROP+.
- Shanno remains the legacy control and canonical the behavioral reference;
  neither is a default.

This is exactly why the plan ends in a **bounded limited-run selector**: the
ranking above is workload-dependent, so the user's own dataset, under identical
starts and a declared budget, should choose — not this table.

## 8. What this screen does not claim

- **Not** that iRPROP+ is the fastest neural optimizer in neuron. It is the
  fastest of the panel on Civic Choice at three sizes and four starts, and the
  second fastest on the generated pair.
- **Not** that it helps on poorly scaled problems. That hypothesis was tested
  directly and failed; the conditioning penalty is ~1.1–1.2x for both adaptive
  methods.
- **Not** that it generalizes better. The held-out differences are mixed in sign.
- **Not** a claim about Logistic, BackProp or online training. Logistic has no
  packed boundary and is refused by name; BackProp and BareProp are covered by
  the correctness tests but were not benchmarked, and online mode is refused.
- **Not** a scaling law. 400,000 rows was not run.

## 9. Two verification notes worth keeping

**The engine is unchanged by this work.** Goldens are byte-identical, oracle
verification is identical, and Shanno reproduced Step 0B's 194 traversals on the
6,000-row group exactly. Independently, the parent commit `63c0a4e` was built in
a separate worktree and re-characterized: it returns the identical practical
objective (0.11812297413633177), the identical plateau iteration (15,984) and the
identical held-out value. Adding iRPROP+ moved canonical not at all.

**The committed endpoint table is current, and an apparent 1.2e-6 drift was an
error of reading.** A fresh characterization reports `practical_objective`
0.11812297413633177 while the table holds 0.118124155. Those are the same
measurement: `fill_targets.py` applies a documented `HEADROOM` of 1e-5, and
0.11812297413633177 × 1.00001 = 0.118124155 to all nine figures. No target was
changed — and none could honourably have been, since the candidate's results were
already visible.
