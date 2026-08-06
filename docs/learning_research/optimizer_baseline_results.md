# Optimizer baseline results (Phase 0, Step 0B)

**Question.** On Civic Choice and at genuinely larger sizes, which training
method that neuron *already has* reaches a usable matched endpoint fastest — and
does the fastest one make the intended workload tractable without compromising
the result?

**Standing.** Canonical training remains the numerical reference that *defines*
the endpoints. It is not assumed to be the operational speed baseline. The
fastest reliable existing method becomes the baseline any new candidate must
beat.

This document records measurements. The harness that produced them, and the
reasoning behind every rule it enforces, is `tests/optimizer/README.md`; this
file does not repeat it.

---

## 1. What was measured

**Status: Step 0B is PARTIAL, and closed.** The campaign was stopped
deliberately, before completion, and is not to be resumed. Logistic, DFA, cross-validation and
workflow timing are out of scope for the rest of this program; the remaining
scale points move into candidate experiments rather than into more baseline work.

**The incumbent is Shanno at 194 full passes / 238.7 ms on Civic Choice 6K.**
That pair of numbers is what a novel neural candidate must beat.

What is complete and trustworthy:

- **All ten workloads characterized** against one engine, both endpoints where
  they exist, with the derivation defended by tests (section 4).
- **The neural single-fit comparison on Civic Choice**, four methods, three
  interleaved repetitions, identical starts, identical endpoint (section 5).
- **Logistic feasibility at four row counts**, from warm-up observations
  (section 5b).

What is built and tested but **not measured**: the neural row-count series at
25,000/100,000, the parameter-count sweep, and the repeated-fit consumer.

**Why it stopped.** The Logistic arms consumed the budget for very little. CGD
and Shanno cannot fit Logistic at all, and a failing arm burns its entire
iteration ceiling on every run — 39 s at 6,000 rows, 725 s at 100,000. Hours
went into re-demonstrating a failure already established six times, on a model
family that is not the target of this program. That is precisely the "timing
minutiae displacing candidate investigation" the plan's scope governor warns
about, and stopping was the right call.

### The three results that matter

1. **Shanno is 81x faster than canonical on the neural model** — on real data, to
   an identical endpoint from an identical start, with no held-out degradation.
   Step 0A's pilot said ~10x. The pilot understated it eightfold. This is one
   workload at one size, not a scaling law.
2. **The same method cannot fit Logistic at all**, at any size tested. Optimizer
   advantage is model-family-specific; a neural result is not a Logistic result.
3. **Canonical gradient descent does not converge on the neural model** at the
   engine's own 1e-6 gradient criterion — its gradient settles near 5e-4 and
   does not fall. Canonical therefore cannot define a strict neural endpoint; a
   precharacterized best-known Shanno endpoint can take that role later.

### What this does *not* settle

The governor is explicit that neither a fast pilot nor tractable training ends
the search for material gains, so: **Shanno at 81x is the new baseline, not a
verdict.** It leaves 194 full passes on a 4,500-row problem, and no
neural-specific candidate has been tested against it. Section 9 states which
candidates remain untested and in what order they should be.

---

## 2. Configuration

### Data

`docs/datasets/civic-choice`, groomed through the maintained recipe
(`tools/mkdataset.py --onehot --refcat`) to 14 input nodes and one binary
outcome — the same encoding the published walkthrough uses. The row-count series
is the same generator at one seed drawing more rows;
`tests/optimizer/prepare_data.py` refuses to prepare it unless every size grooms
to a byte-identical column key, so "the same problem at several sizes" is checked
rather than asserted.

Prepared by `tests/optimizer/prepare_data.py`, which verified that all four
sizes groom to a byte-identical column key:

| file | rows | training rows after the 25% holdout | SHA-256 |
|---|---:|---:|---|
| `civic_6000.txt` | 6,000 | 4,500 | `d6458e1740a410b3…` |
| `civic_25000.txt` | 25,000 | 18,750 | `0a09adc857215bd4…` |
| `civic_100000.txt` | 100,000 | 75,000 | `79674ca02ada6685…` |
| `civic_400000.txt` | 400,000 | 300,000 | `fcd56c62c771bbc0…` |

All 14 input nodes, one binary outcome, identical encoding at every size.

**One provenance note, found while preparing the data and left alone.** The
committed `docs/datasets/civic-choice/civic_choice.csv` is stored with CRLF line
endings, so its raw SHA-256 is `6bfbac6c…` while the digest its own README
publishes, `412f58f9…`, is the LF form the generator writes. The observations
are byte-identical once line endings are normalized — which is what
`prepare_data.py` checks, and what it confirmed. This is an end-of-line artifact,
not a data divergence, but the README currently attributes that digest to
`generate.py` (whose own digest is `68c8c520…`) rather than to the CSV.

### Models

Each model's own constructed defaults, so the benchmark describes a
configuration a user actually gets:

| | eta | loss | weight decay | step-size search |
|---|---|---|---|---|
| `SimpleProp` (via `Network`) | 0.05 | LMS | on, 5e-5 | off |
| `Logistic` | 0.05 | cross-entropy | off | **on by default** |

The neural architecture is **one hidden layer of four units** — what the
published walkthrough's own OBD search selected on this dataset, and what
`modelfactory` builds for a one-output, one-hidden-layer, bias network.

### Split

A stratified 25% holdout at seed 20260804. Every arm of a comparison group
trains on the identical rows from the identical initial weight state; both facts
are hashed into every row (`split`, `weight_start_id`) and the runner **refuses**
a group whose arms disagree on either.

### Methods compared

`canonical` (fixed eta), `canonical-autostep` (the engine's step-size search),
`cgd`, `shanno`. All four run the separate-gradient branch, because CGD and
Shanno require it and a canonical arm compared against them must run the same
production path.

---

## 3. The two endpoints

| endpoint | definition | derived from |
|---|---|---|
| `practical` | the training objective canonical had reached when the **held-out** error stopped improving | `PlateauDetector` at the engine's own defaults |
| `strict` | the training objective canonical reached when it **converged** | the engine's own `gradMaxLimit` of 1e-6 |

Neither threshold was chosen here: both are the engine's own shipped values.
Both endpoints were derived from a canonical control **before** any arm was
timed, and no target was adjusted afterwards.

### A defect found and fixed during derivation

The first practical-endpoint rule took the **best** held-out error over the whole
characterization and asked when the series first came within 1% of it. That makes
the endpoint a function of how long you look. Measured on the Civic Choice neural
workload, one configuration reported its practical endpoint at:

| characterization ceiling | practical iteration | practical objective |
|---:|---:|---:|
| 20,000 | 11,299 | 0.11862794 |
| 100,000 | 78,764 | 0.10858250 |

Two endpoints for one workload. Replacing the rule with the engine's local
plateau detector gives **15,984 under both ceilings**, bit-identical. The rule now
lives in one free function, `practicalEndpoint()`, and `optimizer_harness` tests
it against a synthetic trace at two horizons ten-fold apart — carrying its own
control, since the discarded global-best rule must be shown to move on that same
trace or the test would be asserting nothing.

### Watching the held-out set does not change the fit

Every characterization runs its opening window twice, watched and unwatched, and
refuses unless the two objective trajectories are bit-identical. This is legacy
bug #10's shape in a new place — a reporting action choosing the model. It passed
on every workload below. Not claimed: that the trajectories were compared out to
the strict endpoint.

---

## 4. Characterization

Every endpoint below was measured by `optimizer_probe --characterize` against
**`engine_id 20233b71ed257605`** (71 files in `src/`), Release/NDEBUG, Apple
Silicon. `fill_targets.py` refuses to merge characterizations from two engines
into one table, and the generated table header records which engine it describes.

The held-out-sampling guard passed on every workload: watching the held-out set
left the objective trajectory bit-identical.

| workload | train rows | params | practical iter | practical objective | strict | canonical gradient reached | char cost |
|---|---:|---:|---:|---:|---|---:|---:|
| `logistic-6000` | 4,500 | 15 | 865 | 0.665215 | 0.664933 | 1.00e-06 | 5 s |
| `logistic-25000` | 18,750 | 15 | 482 | 0.662449 | 0.661937 | 1.00e-06 | 19 s |
| `logistic-100000` | 75,000 | 15 | 694 | 0.665036 | 0.664742 | 1.00e-06 | 75 s |
| `logistic-400000` | 300,000 | 15 | 660 | 0.663604 | 0.663440 | 1.00e-06 | 282 s |
| `simpleprop-6000-h4` | 4,500 | 65 | 15,984 | 0.118123 | **none** | 7.52e-04 | 51 s |
| `simpleprop-25000-h4` | 18,750 | 65 | 13,655 | 0.117688 | **none** | 4.29e-04 | 215 s |
| `simpleprop-100000-h4` | 75,000 | 65 | 15,380 | 0.118292 | **none** | 6.11e-04 | 880 s |
| `simpleprop-25000-h2` | 18,750 | 33 | 48,542 | 0.107596 | **none** | 4.26e-05 | 803 s |
| `simpleprop-25000-h8` | 18,750 | 129 | 45,170 | 0.106751 | **none** | 3.75e-04 | 1,605 s |
| `simpleprop-25000-h16` | 18,750 | 257 | 8,933 | 0.118095 | **none** | 1.67e-03 | 554 s |

### Finding C1 — canonical cannot define a strict neural endpoint

On every Logistic workload canonical reaches the engine's own 1e-6 gradient rule,
in **15,800–17,200 iterations regardless of row count** — batch gradient descent's
iteration count is set by conditioning, not by *n*, and the row count buys only
linear wall-clock cost.

On every neural workload it does not. Its maximum gradient settles between
4e-5 and 2e-3 and does not approach 1e-6, and it does not fall monotonically:
on `simpleprop-6000-h4` it was 4.5e-4 at 20,000 iterations and **6.2e-4 at
100,000**, having risen in between.

This is a result about canonical gradient descent, not a gap in the measurement —
and it is a statement about the *reference*, not about endpoints in general. A
late-stage neural endpoint remains perfectly possible; it simply has to be
precharacterized from the incumbent (Shanno) rather than from canonical.

### Finding C2 — the useful model arrives long before the converged one

For Logistic the held-out error plateaus at iteration 482–865 while convergence
takes ~16,000: **a 20–35× separation between the model worth having and the model
the gradient rule certifies.** The objectives differ in the fourth decimal
(0.665215 vs 0.664933 at 6,000 rows). Whatever those extra ~15,000 iterations
buy, the held-out data cannot see it.

This is the single most consequential number for the tractability question, and
it is why both endpoints are measured rather than one.

### Finding C3 — characterization cost is not the same shape as fit cost

Two neural workloads needed a 200,000-iteration characterization budget (their
held-out error had not plateaued at 40,000): `h2` plateaus at 48,542 and `h8` at
45,170, while `h4` plateaus at 13,655 and `h16` at 8,933 on the same data. The
plateau iteration is **not monotone in parameter count**. Because the practical
endpoint is horizon-independent, mixing characterization budgets across workloads
is sound — the endpoint is where the series flattens, not where the budget ended.

---

## 5. Civic Choice, single fit

### Finding N1 — Shanno is 81x faster than canonical on the neural model

**The headline result.** Civic Choice, 6,000 rows, one hidden layer of four
units, all four methods from the **identical** initial weight state
(`0xd9e6037efb…`) on the **identical** split (`0x44f8e21a45ec8acd`), racing to the
**identical** practical endpoint (0.118124155). Three repetitions each,
interleaved, seed 20260805; every arm usable.

| method | median ms | MAD ms | full passes | vs canonical |
|---|---:|---:|---:|---:|
| **`shanno`** | **238.7** | 2.2 | 194 | **81.4x** |
| `canonical-autostep` | 2,817.3 | 0.4 | 2,322 | 6.9x |
| `canonical` | 19,423.4 | 20.1 | 15,971 | 1.00x |
| `cgd` | 38,959.1 | 18.5 | 31,924 | **0.50x** — twice as *slow* |

The spread is negligible against the gaps (MAD 20 ms on 19,423 ms), so the
ordering is not in question. Every arm was bit-reproducible across its three
repetitions: identical achieved objective and identical end-weight identity every
time.

### Finding N2 — no degradation, on this one split

| method | achieved objective | held-out error | end weight state |
|---|---:|---:|---|
| `shanno` | 0.11812109 | **0.11551582** | `0x690cd42e4054` |
| `canonical` | 0.11812411 | 0.11602230 | `0x80be68425938` |
| `canonical-autostep` | 0.11812388 | 0.11601917 | `0x5d9ad5fd499f` |
| `cgd` | 0.11812414 | 0.11602251 | `0xf6e24192208b` |

All four land on the same training objective to seven figures — that is the
matched endpoint working — and arrive at different points in weight space, as
they must.

**The defensible reading is that Shanno showed no degradation here, not that it
is better.** Its held-out error is the lowest of the four, but the gap is 0.0005
on one split at one seed. That is far too small a difference, on far too little
data, to support a quality claim in either direction. What it does rule out is
the obvious worry — that the 81x was bought by landing somewhere materially
worse. It was not.

### Finding N3 — the pilot understated it, and CGD is a liability

Step 0A's pilot signal was "Shanno reaches the matched target in roughly a tenth
of canonical's wall clock." On the real application workload it is **an
eightieth**. The pilot was measured on a 240-row synthetic fixture at eta = 0.5;
on real data at the engine's default eta = 0.05 the advantage is nearly an order
of magnitude larger than the pilot suggested. Pilot ratios do not transfer.

CGD moves the other way: 2x *slower* than canonical, consuming twice the passes
to reach the same place. On this evidence CGD is not a candidate for anything.

---

## 5b. Civic Choice, Logistic

### Finding L1 — CGD and Shanno cannot fit Logistic at all

Six independent demonstrations, at three row counts and both endpoints. Every one
exhausted its iteration ceiling without reaching the target canonical reached in
under a second:

| workload | canonical | canonical-autostep | CGD | Shanno |
|---|---:|---:|---|---|
| 6,000 practical | 0.2 s | 0.04 s | **FAILED** (39.7 s) | **FAILED** (38.9 s) |
| 6,000 strict | 1.1 s | 0.12 s | **FAILED** (39.6 s) | **FAILED** (38.3 s) |
| 25,000 practical | 0.5 s | 0.11 s | **FAILED** (166 s) | **FAILED** (180 s) |
| 100,000 practical | 2.8 s | 0.5 s | **FAILED** (632 s) | **FAILED** (725 s) |
| 100,000 strict | 16.8 s | 1.9 s | not re-run | not re-run |
| 400,000 practical | 10.5 s | 1.8 s | not re-run | not re-run |

This confirms Step 0A's `logistic-shanno` pilot failure and shows it is not a
fixture artifact: **the quasi-Newton methods are unusable on Logistic on real
data at every size tested.** The "not re-run" cells are a deliberate budget
decision recorded in `harness.h`: a failing arm burns its whole ceiling on every
run, ~11 minutes at 100,000 rows and ~45 at 400,000, and the failure is already
established six times over.

### Finding L2 — the step-size search is the fastest Logistic method, by 5–10×

`canonical-autostep` beats fixed-eta canonical at every size and both endpoints —
**5.6× at 6,000 practical, 8.8× at 100,000 strict, 5.8× at 400,000** — despite
paying `maxLoops` extra full passes per iteration. It is also the Logistic
constructor's own default, so this is what a user already gets.

**The operational Logistic baseline is `canonical-autostep`.** Not Shanno, which
cannot finish.

### Caveat on the numbers in L1/L2

These are **warm-up observations**, one run per arm, taken from a campaign that
was stopped before its measured rounds completed. They are single cold runs with
no repetitions, no median and no spread. The failures are categorical and not in
doubt — a ceiling exhausted is not a timing question — but the *ratios* in L2
should be treated as indicative until a completed campaign replaces them.

---

## 6. Row-count scaling

Canonical Logistic, wall clock to the practical endpoint, one warm-up run each:

| rows | training rows | canonical | canonical-autostep | iterations to converge |
|---:|---:|---:|---:|---:|
| 6,000 | 4,500 | 0.2 s | 0.04 s | 17,223 |
| 25,000 | 18,750 | 0.5 s | 0.11 s | 16,635 |
| 100,000 | 75,000 | 2.8 s | 0.5 s | 16,920 |
| 400,000 | 300,000 | 10.5 s | 1.8 s | 15,803 |

### Finding R1 — scaling is linear in rows and flat in iterations

From 6,000 to 400,000 rows — a **66×** increase — canonical's time to the
practical endpoint rose 0.2 s → 10.5 s (**53×**) and the iteration count to full
convergence *fell slightly*, 17,223 → 15,803. Batch gradient descent's iteration
count is set by the problem's conditioning, not by *n*; row count buys linear
wall-clock cost and nothing else.

**Consequence for the program:** on Logistic there is no large-data wall to hit.
A 400,000-row fit to the practical endpoint takes under two seconds with the
method that is already the default. Logistic is not where the hours go.

---

## 7. Parameter-count scaling

**Not measured.** The endpoints were characterized (section 4, all four of
hidden = 2 / 4 / 8 / 16 at 25,000 rows) but the timed arms were not run before
the campaign was stopped.

The characterization alone carries one finding worth keeping:

### Finding P1 — plateau iteration is not monotone in parameter count

| hidden units | parameters | canonical iterations to the held-out plateau |
|---:|---:|---:|
| 2 | 33 | 48,542 |
| 4 | 65 | 13,655 |
| 8 | 129 | 45,170 |
| 16 | 257 | 8,933 |

A 33-parameter network takes **5.4× more canonical iterations** to reach its
useful model than a 257-parameter one on the same data. Whatever governs
time-to-useful-model here, it is not model size — which is a caution against
reading any single architecture's speedup as a size-scaling law.

---

## 8. The repeated-fit consumer

**Not measured.** The mechanism is built, tested and smoke-run; the timed arms
were not reached before the campaign was stopped.

What the smoke runs established, which is a design finding rather than a timing:

### Finding W1 — a matched objective endpoint is the wrong instrument for CV

The endpoint characterized on the full development set was **unreachable on 4 of
10 folds**, because a fold trains on 80% of those rows and its achievable
objective is its own. The first CV smoke run produced 6 usable folds out of 10
under canonical and **0 out of 10** under Shanno.

Racing methods to an objective that 40% of the work cannot reach yields no
timing at all. The cv arms therefore stop on the engine's own plateau rule
(`setAutoStop`), identical across every arm of a group, and declare
`endpoint: none` rather than borrowing a name they have not earned. With that
change all 10 folds plus the locked refit completed: canonical logistic at 6,000
rows, **19.0 s**, pooled out-of-fold AUC 0.6005, locked-test AUC 0.6409.

That single number is the only workflow timing in this document. A 5-fold ×2
comparison plus the locked refit is **11 fits**, so a per-fit saving is
multiplied by 11 before a user sees it — which is why the repeated-fit consumer
was on the list, and why it should stay on it.

---

## 9. Answers to the eleven questions

1. **Does Shanno's pilot advantage survive Civic Choice?** Yes, and it grows:
   **81x**, against the pilot's ~10x. Measured, 3 interleaved repetitions,
   negligible spread.
2. **Does it survive as row count increases?** **Not measured.** The endpoints
   for 25,000 and 100,000 are characterized and the arms are declared; only the
   timed runs are missing.
3. **Does it survive as parameter count increases?** **Not measured**, same
   position. Characterization did show plateau iteration is *not* monotone in
   parameter count (P1), so this must be measured rather than reasoned about.
4. **Is its endpoint numerically comparable to canonical?** Yes. Same training
   objective to seven figures — that is what the matched endpoint enforces — and
   a *lower* held-out error (N2).
5. **Is it reliable across identical repeated starts?** The three repetitions
   were bit-identical in achieved objective and end-weight identity, which
   establishes **timing stability and reproducibility** — not reliability. Every
   repetition used the same deterministic start, so nothing here speaks to
   behavior across *different* initializations. That is a separate experiment,
   and it belongs after a candidate survives a speed screen.
6. **Does the step-size search pay for its extra passes?** Emphatically. On the
   neural model **6.9x** faster than fixed-eta canonical despite `maxLoops` extra
   passes per iteration; on Logistic 5–10x, where it is already the default.
7. **Is CGD competitive anywhere?** No. **2x slower than canonical** on the
   neural model, and it fails outright on Logistic. Nowhere in this evidence.
8. **Which operation dominates the intended workflow?** Not established. A 5-fold
   x2 CV plus locked refit is 11 fits, so single-fit cost is multiplied by 11 —
   but the measurement was not reached.
9. **Projected wall time on the large dataset?** For Logistic, settled: 400,000
   rows to the practical endpoint in **1.8 s** with the default method. There is
   no large-data wall on Logistic. For the neural model, **not projected** — one
   size measured is not a scaling law, and the P1 finding is a direct warning
   against extrapolating from it.
10. **After the best existing optimizer, is a new one still necessary?** **This
    evidence cannot close the question, and should not be read as closing it.**
    Shanno still spends 194 full passes on a 4,500-row problem; whether a
    neural-specific method beats that is untested. What has changed is the bar: a
    candidate must now beat Shanno, not canonical, and the honest comparison is
    against a baseline 81x above where Step 0A left it.
11. **Which candidate addresses the remaining bottleneck?** Revised order, on
    this evidence:

    | rank | candidate | standing after Step 0B |
    |---|---|---|
    | 1 | **L-BFGS** | The direct competitor to Shanno on the family where the wins are. Shanno is already a quasi-Newton method; the question is whether limited-memory BFGS with a proper line search beats it. **Untested and now the highest-value experiment.** |
    | 2 | **iRPROP+** | Sign-based, no line search, strong on batch neural fits. Cheap to implement, and it attacks the same 194 passes. **Untested.** |
    | 3 | **IRLS** | Still the right answer for repeated Logistic fits, but demoted: Logistic already fits 400,000 rows in 1.8 s, so it saves seconds, not hours. |
    | 4 | **BB** | A low-cost harness/control experiment, as the plan says. Not a production candidate on this evidence. |
    | — | **CGD** | Measured 2x *slower* than canonical and failing on Logistic. Deprioritize. |

    The program should aim at **neural** computation. Logistic and DFA are not
    where the hours are — Step 0B establishes that positively rather than
    assuming it.

---

## 10. Limitations

1. **The campaign is partial.** The neural row-count series, the parameter sweep
   and the repeated-fit consumer are built, tested and declared but not timed.
2. **The Logistic numbers in 5b are warm-up observations** — one cold run per
   arm, no repetitions, no spread. The *failures* are categorical; the *ratios*
   are indicative only.
3. **The neural result is one workload.** 6,000 rows, four hidden units, one
   split seed, one weight seed. It is a strong result at that point and not a
   scaling law — and finding P1 is a specific warning against treating it as one.
4. **Canonical cannot define a strict endpoint for any neural workload**, because
   it does not converge on them. This does not mean no meaningful late-stage
   endpoint exists: a future candidate can be compared against a separately
   precharacterized best-known *Shanno* endpoint. What is missing is the
   reference, not the possibility.
5. **CGD and Shanno were not re-run** on the 100,000-row strict and 400,000-row
   Logistic groups. Deliberate, recorded in `harness.h`, on the strength of six
   prior demonstrations of the same failure.
6. **One seed throughout.** Split seed 20260804, weight seed 7. Nothing here
   speaks to variability across initializations.
7. `peak_rss_kb` is process-cumulative and Unix-only; no memory claim is made.
8. Correlated, separated, poorly scaled and non-finite fixtures are not built —
   deliberately, as candidate-specific correctness work.
