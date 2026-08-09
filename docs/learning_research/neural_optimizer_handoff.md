# Neural optimizer handoff

Date: 2026-08-09 (LM-v2 retained; Manifest synchronized; public LM surface pending)

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

At this handoff, `main` is clean and synchronized with `origin/main` at `7e6ee88`.
Confirm rather than assuming that remains true. Do not discard unexpected changes.

The relevant completed commits are:

- `06ff93c` — LM pre-code contract and Step L0 evidence;
- `bc19dec` — `Matrix::addOuterUpper`, `Matrix::symmetrize`, and
  `Matrix::solveSPD`, documented and Manifest-synchronized before LM;
- `6352979` — retained LM-v2 engine implementation, tests, harness, and evidence;
- `7e6ee88` — complete optimizer Manifest audit, consolidated comparison table,
  maintenance rule, Figure 12.1, and index synchronization.

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

Retained as engine capability in `6352979`, but **not public yet**. There is no accepted
REST `algorithm=6`, GUI choice, menu entry, automatic-selection entry, or public recipe.
Chapter 4 states this explicitly. Do not describe engine retention as public access.

LM implements Madsen, Nielsen & Tingleff (2004) Algorithm 3.16 with `mu*I` damping,
Nielsen's `nu` update, `tau=1e-3`, and restored small-step criterion (3.15b) at
`eps_2=1e-12`. `Iterative` remains the stopping owner and reports that criterion as
`STOP_SMALL_STEP` / `small_step`, with `converged() == true`.

Eligibility is deliberately narrow:

- neural `OneHiddenNet` models only;
- LMS only; cross-entropy is refused;
- batch mode only;
- automatic step-size search off;
- no validation/early-stopping split;
- at most 512 packed parameters.

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
- Chapter 4 accurately records the current public REST algorithms and explicitly says
  that retained C++ LM has no REST token yet.

`AGENTS.md` and `docs/gui_cli_parity.md` correctly remain unchanged for LM while it has no
public surface. Update both in the same commit that adds REST/GUI access.

## Exact next boundary

The next task is **not another candidate** and is **not automatic selection**. First make
the public-surface decision for retained LM, REST-first:

1. append `/api/train` `algorithm=6` without renumbering existing tokens;
2. enforce the six LM eligibility refusals by field/name before applying any request;
3. synchronize optimizer naming, blocking and asynchronous training, action logging,
   cloning/continuation, OBD/CV eligibility, and strict parsing;
4. only then add the matching GUI choice and lock incompatible controls;
5. update `docs/gui_cli_parity.md`, `AGENTS.md`, and Manifest Chapter 4 in the same commit;
6. run focused tests plus GUI smoke, async lifecycle, strict parsing, the full Release
   gate, index gate, and rendered Manifest inspection.

The user has explicitly paused before GUI work to discuss preparatory work. Therefore do
not start these changes merely because they are listed here. Await the public-surface
design decision. Automatic selection stays deferred until separately authorized.

## Verification discipline

For future changes, follow the proportional focused gates during development and the full
chain in `CLAUDE.md` before a cross-cutting integration commit. New mechanism tests require
fresh-compilation sabotage evidence through production wiring. Manifest edits require the
index gate, PDF rebuild, and visual inspection. Never extend the retired CLI menus.
