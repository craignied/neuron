# Levenberg-Marquardt: the screen, and the retention decision

Date: 2026-08-08

**Decision: RETAIN LM-v2 under the portfolio policy.** Not as a new sole bar, and
not as the Civic Choice default. Read `lm_source_decision.md` for the algorithm
and every constant, `lm_v2_source_decision.md` for the v1 defect and the v2
correction, and `lm_step0_results.md` for the per-traversal cost measurement.

## 1. The decision, stated precisely

- **LM is the recommended method for the measured severely ill-conditioned
  small-LMS workload.** On `poor4` it reaches the matched endpoint in 10.8 ms /
  15 traversals against L-BFGS's 26.8 / 32 and iRPROP+'s 49.9 / 59, with
  non-overlapping p10-p90 intervals in two independent campaigns.
- **iRPROP+ remains the application-benchmark leader.** It beats LM on every
  Civic Choice arm at every size, decisively.
- **LM versus L-BFGS on Civic Choice is ORDERING NOT ESTABLISHED**, and it does
  not need to be resolved for LM to be retained. See section 4.
- **LM is worst on `well4`** (90.6 ms / 129 traversals). Conditioning behaviour
  must not be generalized beyond the measured pair.
- **Held-out differences are descriptive**, never evidence of predictive
  superiority.

## 2. The measured screen (v2, 15 interleaved repetitions, medians in ms)

| group | canonical | Shanno | L-BFGS | iRPROP+ | **LM** |
|---|---:|---:|---:|---:|---:|
| civic 6,000 base | 17,402.3 | 214.1 | 22.1 | **10.9** | 16.3 |
| civic seed 101 | — | 201.2 | 15.5 | **7.5** | 12.2 |
| civic seed 202 | — | 172.0 | **16.2** | **10.9** | 18.3 |
| civic seed 303 | — | 231.3 | 18.6 | **9.9** | 18.3 |
| civic 25,000 | — | — | 64.0 | **50.0** | 59.0 |
| civic 100,000 | — | — | 257.6 | **199.2** | 237.3 |
| `well4` | 120,798.3 | 2,760.9 | **21.8** | 43.7 | 90.6 |
| `poor4` | 67,156.8 | 5,153.5 | 26.8 | 49.9 | **10.8** |

LM traversals: 8 / 6 / 9 / 9 / 7 / 7 / 129 / 15. MAD 0.05-0.39 ms on every
15-repetition cell. **Every arm produced bit-identical end-weight fingerprints
and identical traversal counts across all repetitions**, so the runs are
reproducible before any timing is read.

**LM takes the fewest traversals of any panel method on every Civic Choice arm**
(8/7/7 against L-BFGS's 20/14/14) and still loses to iRPROP+ on elapsed time,
because its traversal costs more -- exactly what Step L0 measured in advance.

## 3. Late-stage: the v1 defect and its correction

| | v1 | **v2** |
|---|---|---|
| outcome | `StepRejected` exception | **`small_step`** |
| `converged` | **false** | **true** |
| traversals | 110 | 80 |
| training objective | — | 0.0932574 |
| held-out | — | 0.0890267 |

v1 reported a **false failure** at a gradient maximum of 2.22e-10, with trial
objectives bit-identical to the accepted one and a step norm of 6.45e-72. The
cause was a neuron adaptation that omitted Algorithm 3.16's published criterion
(3.15b); v2 restores it in its published position and reports convergence to
`Iterative`, which still owns stopping. Full characterization in
`lm_v2_source_decision.md` Part A.

At the plateau, LM lands third on training objective (0.09326 against Shanno's
0.09147 and L-BFGS's 0.09161) and **worst on held-out** (0.08903), in 80
traversals against 416, 749 and 9,163. Fast, and not the best endpoint.

## 4. The v1/v2 timing shift, and why an ordering is left unresolved

Between the two campaigns LM's Civic medians moved substantially (6,000 rows:
24.7 -> 16.3 ms) **with identical traversal counts and identical end weights**.
Restoring (3.15b) only adds a norm computation, so v2 cannot be genuinely faster.

What changed is the spread: v1's LM arms showed a persistent 1.35x-1.52x
p10-p90 bimodality that the other methods did not, and v2's sit entirely in v1's
fast mode with MAD 0.053. **The cause is unknown at campaign level.** Identical
traversal counts show it was *not* a changed optimization trajectory; they do
**not** establish whether machine state, binary layout, or something else was
responsible.

The consequence is recorded rather than resolved: **LM versus L-BFGS on Civic
Choice is ORDERING NOT ESTABLISHED.** LM leads at 6,000 base, seed 101, 25,000
and 100,000; ties at seed 303; trails at seed 202. That ordering does not affect
retention or the recommendation, so no further timing campaign was run to settle
it, and the Manifest labels it unresolved rather than manufacturing precision.

Two orderings **are** decisive, with non-overlapping intervals in both campaigns:
iRPROP+ beats LM on every Civic arm, and LM wins `poor4`.

## 5. Why retention, under the portfolio policy

Retain a candidate that is correct, stable, reaches comparable endpoints, and is
either reasonably competitive or adds a meaningfully different workload profile.
LM is all of these:

- **correct** -- mechanism tests pass; five sabotages fail-proven, two of them
  crossing production wiring that neither the model-free layer nor the real-model
  integration tests could see;
- **stable** -- 6-9 traversals across four weight seeds; no failure on any
  matched-endpoint arm at any size; bit-identical repetitions throughout;
- **reaches every practical matched endpoint**, `converged: true`;
- **adds a decisive workload** -- the clear winner on the severely
  ill-conditioned twin, and the only panel member whose cost *falls* there
  (`poor4`/`well4` ratio 0.12x against L-BFGS 1.26x and iRPROP+ 1.13x).

It is **not** the recommended Civic Choice default: iRPROP+ beats it there
consistently. Retention and default ranking are two separate decisions, as the
portfolio policy requires.

## 6. Public integration after retention

The retained method is now selectable through `/api/train` and the GUI as
`algorithm=6`, without a legacy-menu entry. Direct requests are refused before
mutation unless the model is neural, LMS, batch/epoch, `autostep=0`, implements
the packed and normal-equations boundaries, and has at most 512 packed
parameters. Validation and weight decay are allowed.

The bounded automatic selector considers LM together with canonical, CGD,
Shanno, L-BFGS, and iRPROP+ whenever those same restrictions pass. Eligible
methods share one fixed 2250 ms total budget equally; omitted methods and reasons
are reported. OBD applies the parameter ceiling to its planned maximum
architecture rather than only its starting size. The normative public contract
is Manifest Chapter 4 and the current `autoalgo`/`Network` specifications in
Chapter 12; the source-decision documents remain the pre-code historical record.
