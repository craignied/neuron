# L-BFGS against the incumbent: the Phase 3 screen

**Question.** Shanno is neuron's neural speed incumbent. Does a limited-memory
BFGS method with a genuine strong-Wolfe line search beat it to the same
endpoint, counting every full pass the line search costs?

**Answer.** Yes, decisively, at every size and every starting point tested:
**8.8x to 13.0x faster** to the committed practical endpoint, taking **14 to 20
full traversals of the training set against Shanno's 123 to 210**.

The measuring apparatus, and the reasoning behind every rule it enforces, is
`tests/optimizer/README.md`. The algorithm, its sources and every constant were
fixed before implementation in `lbfgs_source_decision.md` and were not revised
after any measurement here.

---

## 1. What was measured

Civic Choice, groomed through the maintained recipe to 14 input nodes and one
binary outcome; one hidden layer of four units — the architecture the published
walkthrough's own OBD search selected; a stratified 25% holdout at seed
20260804; each model's own constructed defaults (eta 0.05, LMS, weight decay on
at 5e-5, no step-size search).

Every comparison is a **matched endpoint**: both arms race to the objective
value canonical had reached when the held-out error plateaued, committed in
`harness.h` and measured in Step 0B. Both arms in a group start from the
**identical** weights and train on the **identical** rows, and `run_probe.py`
refuses a group whose arms disagree on either.

Repetitions were chosen by the harness's own declared policy from each case's
warm-up duration: 15 interleaved repetitions for every arm except Shanno at
100,000 rows, which took 3.2 s per run and got 5. Orchestration seed 20260806.
Apple Silicon, Release/NDEBUG.

**Every full pass is counted.** This is the trap the plan warns about: one
L-BFGS iteration performs several traversals of the training set, one per trial
point of the line search, and a method that wins on outer iterations while
losing on passes has won nothing. `Probe` therefore counts
`batchObjectiveGradient()` calls — every traversal L-BFGS makes, trial points
included — and the reported `full_passes` is traversals, not iterations. For
canonical, CGD and Shanno nothing changed: their batch path never enters that
method, and their pass counts are bit-identical to Step 0B's.

---

## 2. The result

### Finding L1 — L-BFGS beats the incumbent by roughly 10x, and the win is in passes

| workload | weight seed | Shanno median | L-BFGS median | speed-up | Shanno passes | L-BFGS passes | pass ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6,000 rows, h4 | 7 | 223.45 ms | **22.97 ms** | **9.73x** | 194 | 20 | 9.70x |
| 25,000 rows, h4 | 7 | 591.09 ms | **67.19 ms** | **8.80x** | 123 | 14 | 8.79x |
| 100,000 rows, h4 | 7 | 3,015.42 ms | **265.65 ms** | **11.35x** | 157 | 14 | 11.21x |
| 6,000 rows, h4 | 101 | 210.09 ms | **16.17 ms** | **12.99x** | 181 | 14 | 12.93x |
| 6,000 rows, h4 | 202 | 179.40 ms | **17.06 ms** | **10.52x** | 158 | 15 | 10.53x |
| 6,000 rows, h4 | 303 | 242.27 ms | **19.32 ms** | **12.54x** | 210 | 17 | 12.35x |

Six independent matched comparisons, all in the same direction, none close.
Spread is negligible against the gaps — at 6,000 rows the L-BFGS p10–p90 band is
22.9–23.5 ms against Shanno's 222.4–225.7 ms, so the intervals are nearly ten
times their own width apart.

**The pass ratio tracks the time ratio to two figures in every row.** That is
the load-bearing observation: the advantage is not a cheaper pass, it is an
order of magnitude fewer of them. A per-pass implementation difference would
show up as a divergence between those two columns, and there is none.

**The incumbent was re-measured, not quoted.** Step 0B published 238.7 ms / 194
passes for Shanno on this workload. Here it is 223.45 ms / 194 passes — the same
pass count, the same achieved objective to all 17 digits, and the same
end-weight identity `0x690cd42e4054c91a`. The wall-clock difference is machine
state between campaigns; nothing about the fit moved. That reproduction is also
independent evidence that the packed-boundary extraction preserved behavior on
the benchmark workload, not only on the goldens.

### Finding L2 — no degradation, and the held-out differences are noise

| workload | Shanno held-out | L-BFGS held-out | difference |
|---|---:|---:|---:|
| 6,000 rows, seed 7 | 0.11551582 | 0.11526821 | −0.00024761 |
| 25,000 rows, seed 7 | 0.11745951 | 0.11728605 | −0.00017346 |
| 100,000 rows, seed 7 | 0.11707790 | 0.11689592 | −0.00018197 |
| 6,000 rows, seed 101 | 0.11510196 | 0.11589085 | **+0.00078889** |
| 6,000 rows, seed 202 | 0.11467278 | 0.11436467 | −0.00030812 |
| 6,000 rows, seed 303 | 0.11572069 | 0.11549308 | −0.00022761 |

Five of six favour L-BFGS and **one, seed 101, favours Shanno — by the largest
margin of the six.** A real effect is paid in the same direction by every
workload; a mixed sign inside this spread is the signature of noise, which is
how this project has settled such questions before.

**The defensible reading is that L-BFGS showed no degradation, not that it
predicts better.** What the table rules out is the obvious worry — that a 10x
speed-up was bought by landing somewhere materially worse. It was not. It
supports no quality claim in either direction.

### Finding L3 — m = 10 and m = 20 are indistinguishable, and both beat m = 5

The predeclared comparison, `m` in {5, 10, 20}, its own comparison group with
`lbfgs_memory` as the declared axis. 6,000 rows, 15 repetitions each.

| m | median | full passes | outer iterations | achieved objective |
|---:|---:|---:|---:|---:|
| 5 | 22.99 ms | 20 | 15 | 0.118015589 |
| 10 | **18.32 ms** | 16 | 13 | 0.118008355 |
| 20 | **18.34 ms** | 16 | 13 | 0.118008451 |

The runner reports **ORDERING NOT ESTABLISHED** between m = 10 and m = 20 —
their p10–p90 intervals overlap — and that is the honest result rather than a
20 µs preference. Both are about 20% faster than m = 5.

There is a structural reason not to read m = 20 as "as good as m = 10 for free":
the run takes 13 outer iterations, so a 20-slot history **never fills and never
evicts**. At this problem size m = 20 is not being tested as a limited memory at
all. No configuration beyond the three predeclared values was tried.

### Finding L4 — late-stage, L-BFGS is 22x faster and lands very slightly worse

Canonical does not converge on this workload, so no canonical strict endpoint
exists and none was invented. What can be asked is where each method stops when
run to **the engine's own plateau rule** (`setAutoStop` at `Iterative`'s shipped
tolerance and window, identical for both arms). This is not a matched-endpoint
race — each stops at its own plateau — so where it lands matters as much as when.

| method | median | full passes | outer iterations | training objective | held-out |
|---|---:|---:|---:|---:|---:|
| Shanno | 10,597.7 ms | 9,163 | 9,163 | 0.091473191 | 0.08706100 |
| L-BFGS | **480.5 ms** | 416 | 372 | 0.091608646 | 0.08746945 |

**22.1x faster to its plateau — and it lands slightly worse on both**: +0.000135
in training objective and +0.00041 in held-out error. L-BFGS is not strictly
dominant late-stage. Its per-iteration progress is larger, so the plateau
detector fires while a little improvement remains. Whether that matters depends
on what the extra 10 seconds are worth; it is recorded here rather than smoothed
over.

### Finding L5 — memory cost is not distinguishable

Peak RSS is process-cumulative and Unix-only, so no optimizer memory claim is
made from it. What it does show is that no arm stands out: at 6,000 rows every
arm peaked between 7.2 and 8.4 MB, at 25,000 rows both arms spanned the
identical 25.8–29.1 MB, and at 100,000 rows both spanned 105–109 MB. The
measurement is dominated by the dataset. By construction L-BFGS's own history is
`2mn` doubles — about 5 KB at m = 5 with 65 parameters — which is why it does
not appear.

---

## 3. Robustness

- **Every one of the 15 arms was bit-reproducible across its repetitions**:
  identical achieved objective and identical end-weight identity every time.
  From a fixed start that establishes deterministic reproducibility, **not**
  reliability — which is why the four weight seeds above exist.
- **Every arm reached its target and stopped on `min_error`**, converged and
  usable, with no failure, no ceiling exhaustion and no non-finite value.
- L-BFGS's achieved objective is *below* Shanno's in every matched group, so it
  is not being flattered by stopping at a worse point that merely clears the
  threshold.
- Deterministic correctness — the two-loop recursion against an independent
  dense inverse-BFGS matrix, both Wolfe conditions on every accepted step, exact
  restoration on failure and cancellation, curvature rejection, the refusals —
  is in `tests/network/check_lbfgs.cpp`, with its sabotage log.

---

## 4. Limitations

1. **One dataset.** Civic Choice at three sizes. Nothing here speaks to a
   different problem, a different loss, or a poorly scaled or separated design;
   those fixtures are deliberately unbuilt.
2. **One architecture.** One hidden layer of four units throughout. Step 0B's
   finding P1 — that plateau iteration is *not* monotone in parameter count — is
   a specific warning against reading a single architecture's speed-up as a
   size-scaling law, and it applies to this table as much as to that one.
3. **Four weight seeds, one split seed.** Enough to show the result is not an
   artifact of one initialization; not enough to characterize a distribution.
4. **SimpleProp only in the benchmark.** BareProp and BackProp are covered by
   fixed-start integration tests, not by timed arms.
5. **The endpoint table was measured on the previous engine id.** The extraction
   changed `engine_id` by adding files to `src/`. The endpoint is a property of
   the objective, the extraction is proven behavior-preserving, and Shanno
   reproduced its Step 0B pass count, objective and end weights exactly — so the
   committed targets remain valid and were not re-characterized.
6. **The late-stage comparison is a plateau comparison, not a matched endpoint**,
   and is labelled `endpoint: none` for that reason.
7. **Logistic is untouched**, by design. It does not implement the packed
   boundary, and L-BFGS refuses it.

---

## 5. Decision

**RETAIN**, in the plan's sense: the measurements justify the next phase. This
is a research-only prototype and **no public integration has been performed** —
no CLI option, GUI control, HTTP field, automatic-selection rule, save-file
field or user-visible capability. Public integration is Phase 5's decision, on
the full acceptance gate, not this document's.

What has changed for the program: **the bar is now L-BFGS, not Shanno.** A
candidate that wants to displace it must beat roughly 15 full passes on a
4,500-row problem, where Step 0B left the bar at 194 and Step 0A's canonical
reference at 15,971.

### What remains untested, and in what order

| rank | candidate | standing |
|---|---|---|
| 1 | **iRPROP+** | Sign-based, no line search, strong on batch neural fits and cheap to implement. Untested, and now the highest-value remaining experiment. |
| 2 | **Safeguarded BB** | The low-cost control the plan describes. Untested. |
| 3 | **LM / online methods** | Only when a measured workload matches their eligibility. |
| — | **IRLS, CGD** | Out of scope and deprioritized respectively; nothing here changes either. |

A 10x gain over the incumbent does not end the search. It moves it: the
remaining question is whether a further order of magnitude exists on top of ~15
passes, and only a candidate measured against *this* baseline can answer it.

---

## 6. Reproducing

```bash
python3 tests/optimizer/prepare_data.py        # groom 6k/25k/100k
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build -R 'optimizer|lbfgs|packed|batch_gradient' --output-on-failure

./tests/optimizer/run_probe.py --screen --seed 20260806 --timeout 600 --out screen.jsonl
```

The late-stage arms carry no objective target, so the summary reports them as
unusable by the fit criteria; read them from the raw rows
(`civic-latestage-simpleprop-r6000-h4-*`).
