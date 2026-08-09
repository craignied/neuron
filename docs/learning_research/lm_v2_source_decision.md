# Levenberg-Marquardt v2: the corrected contract, and the v1 record

Date: 2026-08-08

Two documents in one file, in the order they must be read.

**Part A** records LM-v1 as a **rejected neuron adaptation** — not a rejection of
Levenberg-Marquardt. **Part B** is the corrected v2 contract, declared **before
any code is modified**, exactly as the plan requires of a candidate.

---

# Part A — LM-v1: a rejected adaptation, with its defect named

## A.1 What v1 was, and what it got right

LM-v1 is the implementation screened on 2026-08-08: Madsen, Nielsen & Tingleff
(2004) Algorithm 3.16 with `mu*I` damping and Nielsen's `nu`-based update,
composed by `Network::lmIteration()` over a new normal-equations model boundary.
Its evidence is `lm_screen_results` below and `lm_step0_results.md`.

It was **correct in every respect the screen tested**:

- the mechanism tests pass, and three sabotages were fail-proven — including one
  crossing the real `Network::lmIteration()` wiring that neither the model-free
  layer nor the real-model integration tests could see;
- **seed-stable**: 6–9 traversals across the four predeclared weight seeds;
- **reached every practical matched endpoint**, `converged: true`, with
  **bit-identical end-weight fingerprints across all 15 repetitions** at every
  size and every workload;
- **uniquely wins `poor4`** — 11.0 ms / 15 traversals against L-BFGS's 27.2 / 32
  and iRPROP+'s 50.3 / 59 — and is the only panel member whose cost *falls* on
  the ill-conditioned twin (`poor4`/`well4` cost ratio 0.12x, against L-BFGS
  1.26x and iRPROP+ 1.13x);
- takes the **fewest traversals of any panel method at every Civic Choice size**
  (8 / 7 / 7 against L-BFGS's 20 / 14 / 14).

## A.2 The defect that makes it unretainable

**LM-v1 reports failure after numerical convergence.** On the late-stage arm it
raised `LM::StepRejected` — a named failure, `converged: false`,
`failure_stage: training` — at a point that had converged:

| quantity at the failing iteration | value |
|---|---|
| gradient maximum | **2.22e-10** (the engine's own rule fires below 1e-6) |
| `\|\|g\|\|` | 5.76e-10 |
| step norm at refusal | **6.45e-72** |
| trials 1–3 objective | 19, 6, 2 ULPs above the accepted one — last-bit noise |
| **trials 4–20 objective** | **bit-identical to the accepted one, 0 ULPs** |
| numerator `F(x) - F(x_new)` | **exactly 0** from trial 4 onward |
| factorization failures / non-finite trials | **0 / 0** |
| restoration after refusal | bit-identical, correct |

So `F` could not be improved **because there was nothing left to improve in
double precision**, and the acceptance rule `rho > 0` cannot accept a zero
improvement. The optimizer was finished and had no way to say so.

**The cause is a neuron adaptation, not the published algorithm.** v1's declared
adaptation 3 gave *all* stopping to `Iterative` and therefore omitted Algorithm
3.16's criterion (3.15b),

```
    if ||h_lm|| <= eps_2 ( ||x|| + eps_2 )  then  found := true
```

which is a **convergence exit**, not a rejection path — it is tested immediately
after the solve, before any trial is evaluated. With `||h|| = 6.45e-72` against
`||x||` of order 1, **any `eps_2` the source suggests (1e-8 to 1e-15) would have
fired on the first trial and declared success.**

## A.3 The decision, stated precisely

**LM-v1 is research-only and rejected as an adaptation.** An optimizer that
reports failure after numerical convergence is operationally unsafe: a caller
cannot distinguish it from a genuine divergence, and the convergence contract
makes `converged: false` a fact every downstream consumer acts on.

**This is not a rejection of Levenberg-Marquardt.** The defect is in one declared
adaptation, the evidence identifies it exactly, and every other axis the screen
measured came out in the candidate's favour. LM earns a corrected second stage.

**A correction the record must keep.** I first reported the failure as "no
damping improves `F`". That promoted a bounded observation — twenty declared
trials failed one acceptance rule — into a universal claim. The characterization
then showed the truth was different again: `F` was *already at its
double-precision floor*. Both the overclaim and its correction stay visible.

---

# Part B — LM-v2: the corrected contract, declared before any code changes

Everything in `lm_source_decision.md` sections 1-11 carries forward **unchanged**
unless restated here. No published constant is altered. `tau = 1e-3`,
`nu_0 = 2`, the `1/3` floor, `MAX_REJECTIONS = 20`, `MAX_PARAMETERS = 512` and
the `mu*I` damping are all as declared and measured.

## B.1 The restored criterion, and the one constant it needs

Algorithm 3.16's (3.15b) is **restored verbatim**, in its published position —
immediately after the solve, before any trial is evaluated:

```
    Solve ( A + mu I ) h_lm = -g
    if ||h_lm|| <= eps_2 ( ||x|| + eps_2 )
        found := true
    else
        ... the trial, the gain ratio, the mu update ...
```

**`eps_2 = 1e-12`, fixed now, before v2 is rerun.**

Its justification is the source, not the measurement: Madsen, Nielsen & Tingleff
(2004) §3.2 leaves `eps_2` to the user and their worked examples use `1e-8`
(Example 3.7), `1e-12` (Example 3.16, the Rosenbrock problem this project already
uses as a test oracle) and `1e-15` (Examples 3.8, 3.17). **`1e-12` is the middle
of the source's own range and the value of the example v2's tests already drive.**

It is **not configurable**: no REST field, no GUI control, no setter, no
constructor argument. A tunable convergence tolerance would let a later
measurement be a description of the tuning.

*Consequence, stated as a consequence and not as the reason for the choice:* on
the characterized v1 failure the first trial's step norm was ~1.4e-14, so
`1e-12` fires there at the first opportunity. Any value in the source's range
would also have fired; `1e-12` was not selected to make that case pass.

## B.2 Stopping stays `Iterative`'s, and LM reports rather than terminates

The v1 defect was **not** that LM lacked the criterion — it was that LM had no
way to tell anyone it had converged, so the only exit it owned was a failure.
v2 fixes the reporting channel, not the ownership:

| | v1 | **v2** |
|---|---|---|
| who decides the run stops | `Iterative` | `Iterative`, **unchanged** |
| what LM does at a small step | nothing — falls through to rejection, then throws | **sets a fact and returns normally** |
| how `Iterative` learns of it | it cannot | a **virtual predicate**, exactly as `getGradMax()` already works |

Concretely, and this is the whole mechanism:

- `LM` gains `bool stepConverged() const` — a **fact**, not an instruction. It
  becomes true when (3.15b) is satisfied, and `reset()` clears it.
- `Iterative` gains `virtual bool stepConverged() const { return false; }`.
  **Default false, so no existing run changes by a single bit** — the goldens'
  rule. `Network` overrides it to report `lm.stepConverged()`.
- `Iterative::train()` checks it **beside the existing stop checks**, and records
  a new reason. `Iterative` decides; LM only answers a question.

This is deliberately the same shape as `STOP_GRADMAX`, which is Algorithm 3.16's
*other* criterion (3.15a) and has always lived in `Iterative`. v2 makes the two
published criteria symmetric instead of one being owned and the other missing.

## B.3 The new stop reason and its semantics

- **Token**: `STOP_SMALL_STEP`, machine-readable name `small_step`.
- **Convergence semantics**: `Iterative::converged()` returns **true** for it. It
  is a stopping *rule* that fired, in exactly the sense `STOP_MIN_ERROR`,
  `STOP_CHANGE`, `STOP_WINDOW`, `STOP_GRADMAX`, `STOP_PLATEAU` and
  `STOP_EARLY_STOP` are — the fit reached a point it was asked to stop at. It is
  **not** a ceiling, a cancellation or a budget.
- **Progress/result**: a run ending this way is a **completed fit**. Its weights
  are usable, `obd::eligible` accepts it, and the CV adapters treat it as they
  treat any converged fit. Nothing downstream needs a new case.
- **Pre-update objective association**: **unchanged**. The iteration returns the
  objective at the point the step departed from, and `currGradMax` is that same
  point's raw gradient maximum. v2 adds no new association and moves no cadence.
- **What it means**: the damped Gauss-Newton step has become numerically
  negligible relative to the parameters. That is convergence, and it is reported
  as convergence.

`StepRejected` **remains** and keeps its meaning: no damping in the declared
schedule produced an acceptable step *while the step was still numerically
meaningful*. v2 narrows what can reach it; it does not delete it.

## B.4 What v2 does not add

- **no tunable public field** — `eps_2` is a fixed published constant;
- no REST field, GUI control, menu token, automatic-selection entry, or saved
  network field for LM. The research boundary is unchanged;
- no change to any other optimizer's behaviour. `stepConverged()` is false for
  canonical, CGD, Shanno, L-BFGS and iRPROP+, so their runs are bit-identical to
  before — goldens and oracle included.

## B.5 The declared tests, before the code

Added to `tests/network/check_lm.cpp`:

| # | Test | What it would catch |
|---:|---|---|
| 27 | **The characterized v1 failure, reproduced.** Drive LM to the late-stage stationary point and assert v2 reports **small-step convergence** — `stepConverged()` true, no exception — with the accepted weights **bit-identical** to the last accepted point | the defect this whole stage exists to fix |
| 28 | **(3.15b) at its published threshold**, driven model-free: a scripted problem whose solved step norm straddles `eps_2(\|\|x\|\|+eps_2)`. Just above ⇒ the iteration proceeds to a trial; just below ⇒ small-step convergence, no trial evaluated | the inequality inverted, or `eps_2` applied to the wrong quantity |
| 29 | **The criterion is tested BEFORE the trial**, in its published position: on a small-step iteration the evaluation count does **not** increase | a "convergence" that still burns a traversal, i.e. the criterion moved out of place |
| 30 | **`Iterative` owns the stop**: a real run ending this way reports `small_step`, `converged()` is **true**, and the returned objective is the pre-update one | a model privately terminating with an invisible state |
| 31 | **No other optimizer is affected**: `stepConverged()` is false for canonical, CGD, Shanno, L-BFGS and iRPROP+ after real runs | a default that leaked into every model |
| 32 | **`StepRejected` still reachable**: the scripted always-worse problem, whose step stays numerically meaningful, still throws | the fix swallowing a genuine failure |

**Predeclared sabotage (d), crossing the production wiring**: suppress the
small-step signal at the boundary — `Network::stepConverged()` returns false
regardless of what LM reports. **The characterized late-stage case must revert to
`StepRejected`** (test 27 fails), while **every matched-endpoint control still
passes**, because none of them reaches a numerically negligible step. Both
directions rebuilt with the affected translation units visibly recompiling.

## B.6 Re-measurement: v2 is a new candidate

**No v1 outcome is reused as v2 evidence, even where results prove bit-identical.**
The candidate changed, so the screen is rerun in full:

1. the complete cheap screen — the five-arm panel at 6,000 rows, the four-seed
   panel, and the `well4`/`poor4` conditioning pair;
2. the late-stage arm, which is the one expected to *change*: from a failure row
   to a converged row with a stop reason;
3. the 25,000- and 100,000-row comparison against iRPROP+ and L-BFGS, if the
   cheap screen leaves v2 eligible under the declared scale gate.

## B.7 The bimodal timing, bounded in advance

v1 showed a persistent **1.35x–1.52x p10–p90 wall-clock spread with bit-identical
work** — same traversal count, same end-weight fingerprint — at every size, where
L-BFGS and iRPROP+ span 1.03x–1.07x.

The investigation is **bounded to one question**: do the p10–p90 intervals change
either operational conclusion?

- **LM loses Civic Choice.** Does LM's p10 reach below iRPROP+'s or L-BFGS's p90?
- **LM wins `poor4`.** Does LM's p90 reach above L-BFGS's p10?

Reported: association with **run order** (is the fast mode early or late in the
15 repetitions?), with **warm-up**, and with **measured work** (traversals and
end-weight identity, which are already known constant). If the ordering is
decisive at the interval level, **the investigation stops there** — no
open-ended profiling campaign. If it is not decisive, the summary says
**ORDERING NOT ESTABLISHED** rather than printing a decisive-looking table.

## B.8 Where this stops

At the completed v2 evidence. **No retention decision, no Manifest documentation
of LM, and no public integration** until that evidence is reviewed. The Manifest
index gate stays red while `src/lm.h` exists undocumented, and the prototype
stays uncommitted, exactly as it did for v1.
