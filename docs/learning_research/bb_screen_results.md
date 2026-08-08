# Safeguarded Barzilai-Borwein: the screen, and the decision

Phase 1 of `docs/learning_research/optimizer_implementation_plan.md`, run against
the standing portfolio panel after Phases 3 and 4.

The algorithm, its sources, every constant, the declared neuron policy, the
ownership map, the tests and the retention rule were all fixed in
`bb_source_decision.md` **before any code was written and before any arm was
run**. Nothing below was changed after results became visible.

## Decision: REJECT

**BB is rejected as a portfolio candidate.** It is correct, it always converges,
and it never fails — and it has no bounded, predictable role, because its cost on
one fixed workload varies by **670x** across four starting weight vectors.

The decision is led by that stability finding, not by a speed ratio. BB's
base-seed result is genuinely good (639x faster than canonical descent, third of
five arms); it does not survive contact with the other three seeds.

## 1. The stability finding, which is the decision

One workload — Civic Choice, 6,000 rows, SimpleProp, 4 hidden units — raced to
one committed practical endpoint from four predeclared starting weight vectors.
Full training-set traversals:

| method | base seed | seed 101 | seed 202 | seed 303 | **spread** |
|---|---:|---:|---:|---:|---:|
| iRPROP+ | 10 | 7 | 10 | 9 | **1.43x** |
| L-BFGS | 20 | 14 | 15 | 17 | **1.43x** |
| Shanno | 194 | 181 | 158 | 210 | **1.33x** |
| **BB** | **25** | **17** | **11,398** | **4,690** | **670x** |

Wall time, median milliseconds, tells the same story because the traversal is
what costs:

| method | base seed | seed 101 | seed 202 | seed 303 | spread |
|---|---:|---:|---:|---:|---:|
| iRPROP+ | 11.03 | 7.69 | 11.09 | 10.18 | 1.44x |
| L-BFGS | 22.47 | 15.88 | 16.75 | 19.09 | 1.41x |
| Shanno | 220.49 | 205.97 | 176.20 | 237.43 | 1.35x |
| **BB** | **28.11** | **19.20** | **12,708.15** | **5,299.88** | **662x** |

Every other arm in the panel is flat across the seed panel to within a factor of
1.5. BB is not.

### The long runs are reproducible, not noise and not a bad fit

This was tested directly rather than assumed, because a multi-second tail could
be a scheduling artifact or a fit that never really landed.

**Not harness noise.** Across interleaved repetitions every BB arm produced a
*bit-identical* traversal count, iteration count, achieved objective and **end
weight fingerprint** — one distinct `weight_end_id` per arm, over 5 or 15
repetitions each. Seed 202 ran 11,398 traversals to 0.1169484817 every single
time. The only thing that varied was the clock: p10-p90 of 12,601-12,986 ms
around a 12,708 ms median, about 3% of the median, and 5,243-5,351 ms for seed
303, about 2%. A 670x effect is not hiding inside a 3% band.

**Not an invalid fit.** All 61 BB rows are finite with no failure stage, all 61
report `converged: true`, and 60 of 61 are `usable` — the single exception is the
late-stage arm, which declares `endpoint: none` and therefore cannot set
`target_reached` by construction. Every matched-endpoint BB arm stopped on
`min_error` having *passed* its target, not on a ceiling and not on a plateau.
Seed 202 landed at 0.11695 against a target of 0.11812, which is a better
objective than L-BFGS's 0.11795 at the same endpoint. The slow runs are slow
runs, not failures wearing a time.

**What this establishes, at its true scope.** These measurements describe **the
behaviour of the declared neuron GBB implementation under the declared
safeguarding policy** — Raydan's Algorithm GBB with the constants and the three
declared-policy values in `bb_source_decision.md`, on these workloads. They are
not a statement about Barzilai-Borwein methods in general, and they do not carry
to BB2, to cyclic or adaptive BB, or to a different globalization.

### What actually costs the time, and it is not the line search

Traversals against iterations, which the harness counts separately precisely so
this question can be asked:

| arm | traversals | iterations | trial evaluations beyond one per iteration |
|---|---:|---:|---:|
| base seed | 25 | 18 | 7 |
| seed 101 | 17 | 12 | 5 |
| seed 202 | 11,398 | 11,383 | **15** |
| seed 303 | 4,690 | 4,670 | 20 |
| well4 | 39 | 36 | 3 |
| poor4 | 5,985 | 5,947 | 38 |

**The nonmonotone line search accepts its first trial almost every time.** Over
11,383 iterations at seed 202 it backtracked fifteen times in total. So the cost
is not globalization overhead; it is that the BB1 spectral step itself needs
eleven thousand iterations to reach an endpoint the same method reaches in
eighteen from a different start.

**This falsifies the prediction recorded in the source decision.** Section 6 of
`bb_source_decision.md` predicted that "BB's advantage will be markedly smaller
in traversals than in iterations", because every trial point is a full pass. That
prediction was wrong, and it was wrong in BB's favour: measured, traversals
exceed iterations by 3 to 38 across every arm. It is recorded here as a failed
prediction rather than quietly dropped, and it means the plan's original concern
about the line search's cost does not apply to this workload.

## 2. Conditioning, where the panel's other prediction finally landed

`well4` and `poor4` are one problem at two conditionings (see
`tests/optimizer/README.md`). Each is its own comparison group with its own
characterized endpoint, so the reading is each method's own ratio between them —
the same statistic Phase 4 reported.

| method | well4 traversals | poor4 traversals | conditioning penalty |
|---|---:|---:|---:|
| iRPROP+ | 53 | 59 | 1.11x |
| L-BFGS | 26 | 32 | 1.23x |
| Shanno | 3,328 | 6,279 | 1.89x |
| **BB** | **39** | **5,985** | **153x** |

Phase 4 measured 1.14x for iRPROP+ and 1.23x for L-BFGS on this same pair; the
1.11x and 1.23x here reproduce that, which is a useful check that the fixture and
the endpoints did not move.

**The fixture that found nothing in Phase 4 is decisive here.** Phase 4 built
`well4`/`poor4` to test a predicted iRPROP+ advantage on poorly scaled
objectives, and that prediction failed — both retained methods were almost
unaffected. The same pair now separates BB from the panel by two orders of
magnitude. That is the difference between a method that scales each parameter
individually and one that applies a single scalar step to every parameter at
once: BB has no per-parameter scaling at all, and this is where that shows.

On the well-conditioned twin BB is genuinely good — 39 traversals, second of five
and ahead of iRPROP+'s 53. The ill-conditioned twin is the same problem.

## 3. Speed, kept subordinate, and reported in full

Base workload, median ms, against the canonical control that defines the
endpoint:

| method | median ms | traversals | vs canonical |
|---|---:|---:|---:|
| iRPROP+ | 11.03 | 10 | 1,630x |
| L-BFGS | 22.47 | 20 | 801x |
| **BB** | **28.11** | **25** | **640x** |
| Shanno | 220.49 | 194 | 82x |
| canonical | 17,986.56 | 15,971 | 1.00x |

BB is third of five here and third of four at seed 101 (19.20 ms against
iRPROP+'s 7.69 and L-BFGS's 15.88). **It is not the fastest arm on any declared
workload family in the panel** — its best placing anywhere is second, on `well4`.

### Late stage

Run to the engine's own plateau rule, all four arms identically configured,
declaring `endpoint: none`:

| method | traversals | training objective | held-out error |
|---|---:|---:|---:|
| Shanno | 9,163 | 0.09147 | 0.08706 |
| L-BFGS | 416 | 0.09161 | 0.08747 |
| iRPROP+ | 749 | 0.09407 | 0.08849 |
| **BB** | **524** | **0.10039** | **0.09743** |

BB stops earliest of the three fast methods *and lands worst*, on both the
training objective and the held-out error. A method that stops sooner at a worse
point has not won anything; this is the same reading Phase 3 applied to L-BFGS
and Phase 4 to iRPROP+, and BB is the clearest instance of it so far.

### Held-out error at the matched endpoint is noise

At the matched endpoint BB's held-out errors are 0.115538, 0.115455, 0.115012 and
0.114966 across the four seeds, against iRPROP+'s 0.115241, 0.115422, 0.114917
and 0.114981. The differences are in the fourth decimal and change sign between
groups. This supports *no degradation* and nothing stronger. It is not a reason
to retain.

## 4. Against the retention rule fixed before the run

The rule in `bb_source_decision.md` section 6, applied as written:

- **Falsification criterion 1** — "does not beat canonical fixed-eta by more than
  the observed spread": **not triggered.** BB beats canonical by 640x at the base
  seed and by 13x even on `poor4`.
- **Falsification criterion 2** — "fails to reach the matched endpoint where the
  three retained methods reach it": **not triggered.** BB reached every endpoint,
  on every arm, on every repetition.
- **The portfolio test** — "retained only if correct, stable, reaches the endpoint
  reliably, **and** complementary: fastest on some declared workload family by
  more than spread, or well-behaved in a declared regime where the leaders
  degrade": **fails on both halves of the conjunction.** BB is fastest on nothing.
  And the regime where it is distinctive is the one where it degrades and the
  leaders do not.

**This is a pre-declared rejection, not a post hoc one, and the distinction
rests on what the two criteria were for.** The two candidate-specific
falsification criteria were *early rejection gates* — cheap ways to stop a phase
before it earned a full screen — and never sufficient conditions for retention.
BB passed both and then failed the **pre-existing portfolio admission rule**,
which predates this phase, governed Phase 4's retention of iRPROP+, and is quoted
verbatim in section 6 of `bb_source_decision.md`: it added no winning workload and
introduced extreme, unexplained screening-cost variance. Nothing in the standard
was written or altered after results became visible.

That distinction is worth keeping for a second reason: this is not a broken
implementation. It is a correct implementation of a published method that this
engine's workloads have no use for.

**The portfolio policy is what makes this a real decision rather than a
formality.** Phase 4 retained iRPROP+ although it loses to L-BFGS on two of the
three workload families, because it wins decisively on the third and its failure
profile is bounded. The same policy rejects BB, because there is no workload in
the panel a user should be told to pick it for, and **no predictor was
established from the measured workload and starting-point diagnostics** for which
starting weights cost 17 traversals and which cost 11,398. Four starts justify
the rejection; they cannot show that no predictor exists.

## 5. What was removed, and what is kept

Per the plan's Phase 1 exit decision — "remove rejected disposable code after
evidence is preserved" — the production prototype and its public roots are
removed: `src/gbb.*`, `Network::TRAIN_GBB` and its composition, the model
dispatch, `tests/network/check_bb.cpp`, and the harness's BB arms and `--bb`
subset. `tools/check_manifest_index.py` returns green, because no BB public root
remains for the Manifest to be out of sync with.

**No Manifest entry was ever written for it.** It stayed research-only for its
whole life: no REST token, no GUI control, no legacy-menu entry, no
saved-network field, no `auto` membership, and no Chapter 12 object contract.

Kept as the record: this document and `bb_source_decision.md`. Between them they
specify the algorithm, every constant, the neuron policy for the three values the
paper leaves free, the ownership map, the nineteen declared tests and both
sabotage targets — enough to rebuild the prototype exactly if a future workload
makes the question live again.

## 6. Correctness evidence, preserved because it outlives the code

`tests/network/check_bb.cpp` drove Raydan's table by hand against a model-free
port, and both declared sabotages were fail-proven with visible recompilation of
`src/gbb.cpp` in each direction:

- **the nonmonotone maximum deleted**, turning GBB into monotone Armijo BB — a
  different published method. Two assertions failed, both naming the wiring.
- **the interpolation bracket removed**, so the quadratic candidate was trusted
  unconditionally. Five assertions failed, three naming the safeguard.

**One result from that work matters beyond this phase and is recorded in full.**
Under the first sabotage, `checkNonmonotone()` — the test that asserts nonmonotone
acceptance *by name*, and which was written as the load-bearing test — **still
passed**, because it drives `accepts()` and `windowMax()` directly and the
sabotage was in neither. So did every real-model integration test on all three
architectures: the objective still fell, every run was still deterministic, every
number was still finite. The sabotage was caught only by a separate test written
specifically to pin the *wiring* — one that forces `iterate()` into a situation
only the window maximum can resolve, with a control showing the same trial is
rejected without it.

That is Phase 4's iRPROP+/RPROP+ result reproduced exactly, one phase later, in a
different algorithm: **an assertion that names a mechanism is not automatically a
test of that mechanism, and a descending objective proves nothing about which
algorithm produced it.**

## 7. Limitations

1. **Four starting weight vectors is not a characterization of the failure.** Two
   were fast and two were slow; nothing here identifies what distinguishes them,
   so no user-facing rule of the form "BB is safe when..." can be offered. That
   absence is part of the rejection: an unpredictable cost is worse than a
   predictable bad one.
2. **The screen stopped at 6,000 rows.** The 25,000- and 100,000-row scaling arms
   were declared but not run, under the plan's staged gate: scaling is spent only
   on a candidate still plausible after the cheap screen, and BB was not.
3. **One model family, one architecture.** SimpleProp at four hidden units, plus
   the two generated conditioning fixtures. BareProp and BackProp were exercised
   only by the correctness tests, not timed.
4. **`poor4` and `well4` race different endpoints**, so the per-method penalty
   ratio is the honest statistic and the raw traversal counts across the two
   groups are not comparable. Canonical needs *fewer* traversals on `poor4`
   because its endpoint there is much looser.
