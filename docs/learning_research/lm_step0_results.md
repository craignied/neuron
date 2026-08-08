# LM Step L0: the measured per-traversal cost ratio

Date: 2026-08-08

The gate declared in `docs/learning_research/lm_source_decision.md` section 12,
run before Algorithm 3.16 was written, exactly as predeclared.

**Verdict: a two-or-more-traversal win region exists on every eligible workload,
by a wide margin. Per the predeclared gate, work stopped here and reported;
the user then authorized the implementation phase on this evidence.**

**The headline is a falsified prediction, and it is my own.** The source
decision's cost model estimated the LM traversal at `P/4` times the production
traversal — 16.2x at `P = 65` and 6.2x at `P = 25` — and concluded that LM was
"arithmetically foreclosed" on the application benchmark, needing to finish in
less than a single pass. Measured, the ratio is **1.8x** and **0.87x**. The
estimate was wrong by roughly nine-fold and seven-fold, in the candidate's
favour, and at `P = 25` it has the sign wrong: **the proposed LM traversal is
cheaper than the production one it replaces.**

The user's correction — that `P/4` is a cost model and Step L0, not asymptotic
notation, makes the decision — is what this measurement vindicates.

---

## 1. What was measured

Two complete traversals of the same training set, at **identical installed
weights**, with the weights asserted bit-identical afterwards so neither path
trained:

| the proposed LM traversal | the production reference |
|---|---|
| forward propagation of every exemplar | forward propagation of every exemplar |
| output sensitivity `s`, hidden sensitivities `hs` | error terms `o_err`, `h_err` |
| the packed Jacobian row `a_k` | the packed gradient contribution |
| `JᵀJ += a_k a_kᵀ`, upper triangle | — |
| `Jᵀr += r_k a_k` | gradient accumulation |
| objective accumulation | objective accumulation |
| `1/N`, the `decay` diagonal, `decay·w`, the penalty | `1/N` and the per-exemplar decay terms |
| the symmetry completion | — |
| — | `packPair()`'s packing |

The production side is the real `Network::batchObjectiveGradient()` — the same
virtual the panel's L-BFGS and iRPROP+ arms call — not a reimplementation.

Machine: Apple M2 Ultra, macOS 26.6.1, Apple clang 17.0.0, Release/`NDEBUG`,
engine id `8df15cbbeb7b9038`. 15 randomized, interleaved repetitions per arm
after a discarded warm-up; arm order drawn per repetition from a recorded
orchestration seed.

### The validity gates, passed before any timing was reported

Both paths must compute the same thing, or a ratio between them means nothing:

| gate | civic-6000 | well4 | poor4 |
|---|---:|---:|---:|
| objectives agree (relative) | 9.13e-15 | 5.05e-15 | 4.39e-15 |
| every gradient element agrees (worst relative) | 1.08e-16 | 1.04e-15 | 1.34e-15 |

against a predeclared tolerance of `1e-12`. Also asserted before timing: both
objectives finite; both gradients non-empty, of the packed length, and finite;
the LM gradient not identically zero; `A` exactly symmetric after the mirror;
`A`'s diagonal `≥ decay`; and the packed weights **bit-identical** before and
after each evaluation.

That gradient agreement is the source decision's section 3 identity confirmed on
real models: `Jᵀf` really is `∇F`, computed by a different route from a different
quantity (`∂o/∂w` rather than `∂E/∂w`), and it agrees to within one part in
`1e15`.

---

## 2. The measurement

Two orchestration seeds, and each seed measures every workload twice (the gate
run and the attribution run below), so each ratio has four independent replicates.

| workload | `P` | rows | production median | LM median | **ratio** | cost model said |
|---|---:|---:|---:|---:|---:|---:|
| civic-6000-h4 | 65 | 4,500 | 1.08–1.41 ms | 1.98–2.53 ms | **1.79, 1.84, 1.84, 1.86** | 16.2x |
| `well4` | 25 | 4,500 | 0.84–0.88 ms | 0.72–0.79 ms | **0.88, 0.90, 0.86, 0.86** | 6.2x |
| `poor4` | 25 | 4,500 | 0.82–0.85 ms | 0.70–0.72 ms | **0.88, 0.85, 0.86, 0.87** | 6.2x |

MAD is 0.002–0.098 ms on every cell and the p10–p90 ratio bands are narrow
(civic 1.78x–1.94x, `well4` 0.83x–0.91x on the tightest seed). The two
orchestration seeds agree to within 3%, so the ordering is established and the
ratios are not a seed artifact.

Absolute medians drift a few percent between runs on a shared machine; the
**ratio** is what is stable, which is why the arms are interleaved.

### Why the cost model was so wrong

It counted multiply-adds and ignored the **transcendental cost of the forward
pass**, which both paths pay identically and which dominates at these sizes.
Every exemplar evaluates `nHidden + 1` sigmoids — five per exemplar here, 22,500
per traversal — and `exp()` is far more expensive per call than a fused
multiply-add. The `P(P+1)/2` accumulation LM adds is real, but it is added on top
of a large shared floor rather than to a small one. `P/4` is the ratio of the
*extra* work to the *gradient* work, and the gradient work is not what the
traversal spends its time on.

This is the general form of the error: **a cost model that omits a term both
sides share will overstate a ratio, because the shared term is the denominator.**

---

## 3. The attribution: how much of the ratio is LM, and how much is the engine

A ratio has two possible causes — the candidate's extra work, and the baseline's
overhead — and reporting one number without separating them would leave the
reader unable to tell which. So the same measurement was repeated with **weight
decay off**, which removes an asymmetry between the two paths:

`OneHiddenNet::batchGradient()` adds its decay term **per exemplar**, as
`hG += ( hW * decay )` and `oG += ( oW * decay )`. Both operators return **by
value** (`Matrix<T> operator*(const T) const`, `vector<T> operator*(vector<T>, T)`),
so **each exemplar heap-allocates one `Matrix` and one `vector`** — 9,000
allocations per traversal at 4,500 rows. The proposed LM traversal adds decay
once, outside the loop.

| workload | production, decay **on** | production, decay **off** | ratio with decay on | ratio with decay off |
|---|---:|---:|---:|---:|
| civic-6000-h4 | 1.08–1.41 ms | **0.51 ms** | 1.79–1.86x | **3.85x, 4.07x** |
| `well4` | 0.84–0.88 ms | **0.40 ms** | 0.86–0.90x | **1.77x, 1.77x** |
| `poor4` | 0.82–0.85 ms | **0.39 ms** | 0.85–0.88x | **1.80x, 1.79x** |

So roughly **55–64% of the production traversal's elapsed time, with decay on, is
the per-exemplar temporaries** — not gradient arithmetic.

### What the decay-off run is, and what it is not

**Decay-on is the authoritative comparison and the only basis for the gate.** It
is the shipped neural configuration (`Network` constructs with weight decay on at
5e-5, `network.cpp:20-46`), it is the configuration the committed panel's
objective and every committed traversal count were produced under, and it is what
a user actually runs.

**The decay-off run is attribution evidence, not a replacement baseline, and it
is not merely "the same workload, faster."** Turning weight decay off changes the
objective being minimized, hence the gradients, hence the optimum, hence how many
traversals any method needs to reach any endpoint. Its 3.9x / 1.78x ratios
therefore may **not** be combined with the panel's decay-on traversal counts to
produce a break-even: the two halves of that quotient would come from two
different problems. It is reported for one purpose only — to separate how much of
the decay-on ratio is LM's extra work from how much is the existing path's
per-exemplar allocation — and the answer is that the allocation accounts for
roughly 55–64% of the production traversal.

The decay-off ratios do bound LM's *algorithmic* overhead against a gradient
traversal that allocates nothing: 3.9x at `P = 65` and 1.78x at `P = 25`, still
four times below the hypothesis. That is a statement about the two traversals, not
a second gate.

---

## 4. The gate

LM must reach the endpoint in fewer than `(panel traversals) / R` traversals to
win on wall-clock. The panel traversal counts are the measured ones from
`docs/learning_research/irprop_screen_results.md`, not remembered.

**Under the engine as it is (`R` = 1.8 at `P` = 65, 0.87 at `P` = 25):**

| workload | reference arm | its traversals | **LM traversals to break even** |
|---|---|---:|---:|
| civic-6000-h4 | iRPROP+ | 10 | **5.5** |
| civic-6000-h4 | L-BFGS | 20 | **11.0** |
| `well4` | L-BFGS | 26 | **29.9** |
| `well4` | iRPROP+ | 53 | **60.9** |
| `poor4` | L-BFGS | 32 | **36.8** |
| `poor4` | iRPROP+ | 59 | **67.8** |

**Every cell is well above two traversals.** The predeclared gate — reject if
break-even is below two on *every* eligible workload — is not met, and its
complement is: work stops, reports, and the implementation phase is authorized on
this evidence.

No second table is computed from the decay-off ratios, for the reason in section
3: those ratios belong to a different objective, and dividing this workload's
traversal counts by that workload's ratio would be a quotient of two different
problems.

The tightest budget is **5.5 traversals**, Civic Choice against iRPROP+. That is
the number to keep in view: on the application benchmark LM must reach the matched
endpoint in five iterations or fewer to win, which is a real requirement and not a
comfortable one. The conditioning pair is where the margin is wide — roughly 30 to
37 traversals against L-BFGS's 26 to 32 — and it is also where a damped
Gauss-Newton method has its strongest theoretical claim.

---

## 5. What this does and does not establish

**Establishes:**

- the proposed traversal computes the same objective and the same gradient as the
  production path, on both concrete workloads, to one part in `1e15`;
- `JᵀJ` and `Jᵀr` accumulate per exemplar with no allocation in the loop and
  storage flat in `N`, as section 6 of the source decision claims;
- the per-traversal cost ratio is **1.8x** at `P = 65` and **0.87x** at `P = 25`
  on the authoritative decay-on workload — the shipped neural configuration and
  the one every committed panel number was taken under. The decay-off attribution
  run gives 3.9x / 1.78x, but on a *different objective*, so it bounds LM's
  algorithmic overhead and may not be divided into this workload's traversal
  counts;
- the `P/4` cost model is wrong by roughly a factor of nine, and the reason is
  identified and measured.

**Does not establish:**

- **anything about how many traversals LM actually takes.** Break-even is a
  budget, not a result. Every number in section 4 is an *allowance*; whether LM
  converges inside it is the question Algorithm 3.16 exists to answer, and it is
  unanswered;
- anything about correctness, stability, seed spread, conditioning behaviour or
  endpoint quality. Step L0 timed two evaluations; it ran no optimizer;
- anything about `P` beyond 65. The ratio grows with `P`, and the parameter
  ceiling of 512 remains untested;
- that the ratio is machine-independent. One machine, one compiler.

---

## 6. A separate finding, deliberately not acted on

The per-exemplar `Matrix` and `vector` temporaries in `OneHiddenNet::batchGradient()`
cost **55–64% of every batch neural traversal in the engine** when weight decay is
on. That is measured here, on two workloads, with a control.

It is **not** being fixed now, and the reason matters more than the finding:

- weight decay is on by default for every `Network` (`network.cpp:20-46`), so this
  is on the hot path of canonical, CGD, Shanno, L-BFGS and iRPROP+ alike;
- removing it would make **every arm in the standing panel faster by a different
  amount**, which would invalidate comparability with every committed measurement
  in `optimizer_baseline_results.md`, `lbfgs_screen_results.md`,
  `irprop_screen_results.md` and `bb_screen_results.md`;
- rule 8 requires it to land separately from any optimizer work, with its own
  characterization, and rule 3 requires the panel to be re-measured afterwards
  rather than assumed to shift uniformly.

It is recorded here as a measured opportunity for its own future decision, not as
a task this phase may absorb.

---

## 7. Reproducing

```bash
python3 tests/optimizer/prepare_data.py     # if tests/optimizer/data/ is absent
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target lm_step0 --parallel
./build/lm_step0                            # or --reps N --seed S
```

`tests/optimizer/lm_step0.cpp` and its CMake target are **research evidence**:
committed so the measurement can be reproduced, but not registered with `add_test`, adding no REST field, GUI
control, optimizer token, automatic-selection entry or menu change, and touching
no file under `src/`. The proposed traversal lives in a `SimpleProp` subclass
rather than in `OneHiddenNet` precisely so that deciding against it costs a
deletion rather than a revert.

The SPD Cholesky solve is **not** built: Step L0 does not solve anything, and the
user's ruling is to implement the primitives only to the extent a step actually
needs them.
