# Neural optimizer handoff

Date: 2026-08-09 (LM-v2 retained and publicly integrated; Manifest synchronized)

This is the live optimizer-program handoff. The implementation plan preserves the
declarations under which each phase was run; this file owns current status and the exact
next boundary.

## Restart check

Run before acting:

```sh
cd /Users/craign/code/neUROn2++/neuron-3.0
git status -sb
git log -6 --oneline --decorate
```

At this handoff, the LM public-surface integration follows `a4e65e1`; inspect the current
HEAD and synchronization state rather than relying on a copied hash. Confirm rather than
assuming the tree is clean. Do not discard unexpected changes.

The relevant completed commits are:

- `06ff93c` — LM pre-code contract and Step L0 evidence;
- `bc19dec` — `Matrix::addOuterUpper`, `Matrix::symmetrize`, and
  `Matrix::solveSPD`, documented and Manifest-synchronized before LM;
- `6352979` — retained LM-v2 engine implementation, tests, harness, and evidence;
- `7e6ee88` — complete optimizer Manifest audit, consolidated comparison table,
  maintenance rule, Figure 12.1, and index synchronization.
- the commit containing this handoff — REST/GUI `algorithm=6`, family-aware bounded
  automatic selection, full public contracts, and the rule that production retention is
  incomplete until those surfaces and the Manifest close.

Discover the current test count from CTest. Do not copy a remembered count into a status
report.

## Read first

1. `CLAUDE.md`
2. `docs/optimizer_research.md`
3. `docs/learning_research/optimizer_implementation_plan.md`
4. This handoff
5. For LM details, in order:
   - `lm_source_decision.md`
   - `lm_v2_source_decision.md`
   - `lm_step0_results.md`
   - `lm_screen_results.md`
6. Before public REST/GUI work, `docs/gui_cli_parity.md`, Manifest Chapter 4, and
   `docs/manifest_maintenance.md`

## Program boundary

The program remains neural-only and aims at large-dataset training speed. Logistic,
IRLS, DFA, exhaustive baseline timing, stepwise timing, and CV timing are out of scope
unless an accepted neural optimizer requires one narrowly relevant end-to-end check.

Do not reopen completed L-BFGS, iRPROP+, BB, LM-v1, or LM-v2 research. Do not resume the
closed Step 0B campaign. BB's exact rejected tree is the annotated
`research/bb-prototype-2026-08-08` tag; it is evidence, not supported source.

The seed panel and conditioning pair remain first-class acceptance axes for any future
candidate. Benchmark inside the candidate loop, interleave close comparisons, and lead
with the portfolio policy rather than one speed number.

## Retained portfolio

### L-BFGS

Retained and public as REST/GUI `algorithm=4`. It is neural, full-batch, requires
`autostep=0`, and has an optional positive `lbfgs_memory`. It remains the leader on the
generated well-conditioned fixture and a principal modern control.

### iRPROP+

Retained and public as REST/GUI `algorithm=5`. It is neural, full-batch, requires
`autostep=0`, and remains the application-benchmark recommendation: it beats LM on every
measured Civic Choice arm at 6,000, 25,000, and 100,000 rows.

### Levenberg--Marquardt (LM-v2)

Retained in `6352979` and now public as REST/GUI `algorithm=6`. It also participates in
`algorithm=auto` whenever eligible. There is deliberately no legacy-menu entry or saved
optimizer state.

LM implements Madsen, Nielsen & Tingleff (2004) Algorithm 3.16 with `mu*I` damping,
Nielsen's `nu` update, `tau=1e-3`, and restored small-step criterion (3.15b) at
`eps_2=1e-12`. `Iterative` remains the stopping owner and reports that criterion as
`STOP_SMALL_STEP` / `small_step`, with `converged() == true`.

Eligibility is deliberately narrow:

- neural `OneHiddenNet` models only;
- LMS only; cross-entropy is refused;
- batch mode only;
- automatic step-size search off;
- at most 512 packed parameters.

Validation/early stopping and weight decay are explicitly allowed: neither changes LM's
training objective or its published iteration, and `Iterative` continues to own stopping.

The implementation accumulates `J'J` and `J'r` per exemplar; it never forms an `N x P`
Jacobian. Storage is flat in row count and bounded by the parameter ceiling.

## LM evidence and decision

Step L0 disproved the initial arithmetic cost model. The proposed normal-equations
traversal measured about 1.8 times the production gradient traversal at `P=65`, but about
0.86--0.90 times it at `P=25`; forward-pass transcendental and allocation costs dominated
the multiply-add estimate. That falsification is retained rather than rewritten.

LM-v1 then exposed a real adaptation defect: at a numerically converged late-stage point,
the omitted Algorithm 3.16 small-step criterion left it able to report only
`StepRejected`. The measured point had gradient maximum `2.22e-10`, bit-identical trial
objectives from trial four onward, and a step norm collapsing to `6.45e-72`. LM-v2 restored
criterion (3.15b), reports it through `Iterative`, and was screened again from scratch.

The retained workload-scoped conclusions are:

| workload/result | conclusion |
|---|---|
| Civic Choice, every measured size and seed | iRPROP+ beats LM decisively |
| Civic Choice, LM versus L-BFGS | **ORDERING NOT ESTABLISHED**; campaign-level timing shifted while traversals and endpoints stayed identical |
| `poor4`, severely ill-conditioned small LMS | **LM recommended:** 10.8 ms / 15 traversals versus L-BFGS 26.8 / 32 and iRPROP+ 49.9 / 59 |
| `well4` twin | LM worst: 90.6 ms / 129 traversals |
| late stage | LM-v2 converges by `small_step` in 80 traversals, but its held-out result is worst of the compared arms |

Do not generalize from the conditioning pair: LM is recommended for the measured
`poor4` workload, not for “ill-conditioned problems” as a class. Held-out differences
are descriptive and do not establish predictive superiority.

The v1/v2 Civic timing shift remains unexplained. Identical traversal counts establish
that the trajectory did not change; they do not identify machine state, binary layout,
or another cause. Re-run any close wall-clock ordering interleaved before quoting it.

## BB remains closed and rejected

Safeguarded Raydan GBB was rejected under the portfolio policy: it won no workload,
varied 670-fold across four starting seeds, and degraded 153-fold across the conditioning
pair. The archived implementation's acceptance comparison rejects NaN and positive
infinity but would accept negative infinity; every measured row was finite, so the audit
defect does not change the screen or rejection. Do not reopen or repair it on `main`.

## Manifest state

The Manifest and its maintenance authority are current through LM:

- Chapter 12 explains LM/LMObjective ownership, complete public methods, state,
  equations, eligibility, failure and stopping semantics, tests, and citations;
- the Matrix section explains `addOuterUpper`, `symmetrize`, and `solveSPD`, including
  storage and failure contracts;
- Figure 12.1 includes the new major objects and their composition;
- the index and `tools/check_manifest_index.py` cover LM, LMObjective, the Matrix
  primitives, and the rejected BB research reference;
- the consolidated optimizer comparison table includes canonical, Shanno, L-BFGS,
  iRPROP+, rejected BB, and LM;
- `docs/manifest_maintenance.md` requires every future optimizer, retained or rejected,
  to update that table in the same decision commit;
- Chapter 4 records public LM token 6, its exact refusals, verified blocking/asynchronous
  request examples, and its automatic-selection participation.

`AGENTS.md` and `docs/gui_cli_parity.md` record the same public surface and the permanently
frozen legacy menus.

## Exact next boundary

The optimizer research and LM public-surface phase are complete. Do not reopen them and
do not start another candidate. The user intends to discuss the next broader GUI update;
await that design direction. The LM choice already present in the training selector is
the synchronized client of its completed REST contract, not a legacy-menu change.

Automatic selection now considers canonical, CGD, Shanno, L-BFGS, iRPROP+, and LM when
eligible. The eligible list is settled before probing, all candidates share one fixed
2250 ms total budget equally, and every omission is returned with its reason. OBD judges
LM's 512-parameter ceiling against its planned maximum architecture, so it cannot select
LM at a small starting size and fail after growing past the ceiling. Fixed OBD/CV tokens
remain `1|2|3|auto`; an `auto` result may resolve to any eligible retained optimizer.

## Verification discipline

For future changes, follow the proportional focused gates during development and the full
chain in `CLAUDE.md` before a cross-cutting integration commit. New mechanism tests require
fresh-compilation sabotage evidence through production wiring. Manifest edits require the
index gate, PDF rebuild, and visual inspection. Never extend the retired CLI menus.
