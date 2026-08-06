Read and follow the instructions in /Users/craign/code/locker/CLAUDE.md before proceeding.

# neuron 3.0 — project instructions

neuron is a C++ neural-modeling and statistical engine with a legacy menu CLI, a
loopback HTTP GUI, and small Python data/deployment tools.  The legacy source in
`../distro/` is read-only oracle material.  Current production source is `src/`.

## Load only the context the task needs

Start here, then open the smallest relevant authority:

| Task | Read before acting |
|---|---|
| Build, dataset grooming, training, deployment | `AGENTS.md` |
| Engine or numerical code | `docs/development_rules.md`; relevant Manifest class section |
| Optimizer or learning-algorithm research | `docs/optimizer_research.md`; rules 4, 6, 7 in `docs/development_rules.md` |
| GUI or REST fields | `docs/gui_cli_parity.md`; Manifest REST chapter |
| Cross-validation or inference | `docs/cross_validation.md`; `docs/evaluation_report_spec.md`; relevant Manifest services |
| ROC/statistics | `docs/roc_theory.md`; relevant cited Manifest section |
| Manifest editing | `docs/manifest_maintenance.md` in full |
| Why an old decision was made | Search `docs/HISTORY.md` and `docs/refactor_audit.md`; do not load either by default |

Plans retained for provenance (`docs/obd_plan.md`,
`docs/cv_refactoring_architecture.md`, `docs/b9_strict_parsing.md`) describe work
as it was undertaken.  They are historical, not the current public contract.
The Manifest and source are authoritative after implementation.

## Current state

- The behavior-preserving engine refactor and ROADMAP 4 are complete.
- Release builds and tests are the normal gate.  Current test count is discovered
  from CTest; never copy a remembered count into a status report.
- Current optimizer research and implementation follows
  `docs/learning_research/optimizer_implementation_plan.md`.  Resume at the
  first incomplete phase, and do not expose a candidate publicly before its
  matched-endpoint acceptance gate.  **Phase 0 and Phase 3 are complete:
  Shanno was the incumbent, and the retained L-BFGS implementation
  (`src/lbfgs.*`) now beats it 8.8x-13.0x to the committed practical endpoint
  across three row counts and four weight seeds.  Its research gate and public
  REST integration are complete: `/api/train` selects it with `algorithm=4`
  and optionally configures `lbfgs_memory`.  The retired CLI menus are not
  extended; GUI and automatic-selection integration remain deferred until the
  researched optimizer set is complete.  The next candidate is iRPROP+,
  measured against the plan's standing panel: L-BFGS as speed leader, Shanno
  as legacy quasi-Newton control, and canonical as behavioral/matched-objective
  reference.  L-BFGS is not a winner-takes-all retention gate.  Read
  `docs/learning_research/neural_optimizer_handoff.md` first.**
  Apply the plan's **large-workload speed scope governor** to every optimizer
  instruction: prioritize implementing and comparing plausible high-impact
  algorithms on representative scaled workloads. Existing methods establish
  honest baselines and canonical defines matched endpoints, but neither a fast
  pilot nor merely tractable training ends the search for material 10x--100x
  gains. The active program is neural computation: do not spend its budget on
  Logistic, DFA, or exhaustive baseline/workflow timing. Shanno is the incumbent
  neural speed baseline; move to novel neural candidate prototypes and use only
  the measurements needed to accept, reject, or scale them. Do not let timing
  minutiae displace candidate investigation.
  `docs/optimizer_research.md` remains the governing research protocol; no
  published formula may be altered merely to share code.
- The Design Manifest is normative and must stay synchronized with every public
  class, major object, principal method, algorithm, failure contract, and index entry.

## Non-negotiable project rules

The complete operational wording is in `docs/development_rules.md`; these are the
session-level triggers:

1. Update `AGENTS.md` when an operational recipe changes.  If the GUI or REST
   surface changes, update `docs/gui_cli_parity.md` in the same commit.
2. Prove a new test can fail.  Demonstrably recompile affected translation units
   after introducing a sabotage and again after restoring it; require the build
   log to show both compilations.  Guard against vacuous empty/default comparisons.
3. Measure a claimed defect or performance opportunity before acting on it.
4. Keep numerical vocabulary and contract checks in `Matrix`, `vector_ops`, and
   `Population`; extend that layer instead of dropping to raw arrays.
5. **Do no further feature work on the legacy CLI menus.** They remain frozen
   for compatibility and regression testing only. Every new interactive
   capability is designed REST-first and exposed through the GUI where a human
   control is appropriate; it does not receive a menu equivalent. Preserve the
   existing menu surface, and keep GUI controls and REST fields synchronized.
6. One class or service owns each mechanism.  Coordinators compose it; they do
   not reimplement it.
7. Speed is architectural.  Keep allocation, `std::function`, and virtual
   dispatch out of exemplar/element loops; use destination-taking operations and
   `const&`; measure before optimizing; keep published equations visible.
8. Behavior-preserving refactors keep the oracle/goldens byte-identical unless a
   separately characterized correctness change is explicitly authorized.
9. Manifest additions are explanatory contracts, not symbol inventories.  Follow
   the full Chapter 7 pattern and `docs/manifest_maintenance.md`; update the index
   gate and rebuild/inspect the PDF in the same commit.

## Build and verification

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
tests/golden/run_golden.sh
tests/gui/smoke.sh
tests/oracle/verify_oracle.sh
tests/tools/run_tools.sh
git diff --check
```

Run gates in proportion to the change while developing; run the complete chain
before a release-sized or cross-cutting commit.  Never re-bless a golden to make
a behavior-preserving change pass.  The oracle build instructions live in
`tests/oracle/README.md`.

For Manifest work, also run:

```bash
python3 tools/check_manifest_index.py
cd docs/tex
dot -Tpdf figures/objects.dot -o figures/objects.pdf
latexmk -pdf manifest.tex
```

Inspect affected PDF pages as images, not only the LaTeX log.  The complete
workflow and cleanup commands are in `docs/manifest_maintenance.md`.

## Source and ownership map

- `src/matrix.h`, `vector_ops.*`, `population.*`: numerical vocabulary.
- `src/dataset.*`, `twoset.*`: data ownership, partitions, metrics and ROC.
- `src/model.*`, `iterative.*`, `network.*`: model/training abstractions.
- `src/onehidden.*`, `simpleprop.*`, `bareprop.*`, `backprop.*`, `logistic.*`:
  neural and logistic models.
- `src/dfa.*`, `ldfa.*`, `qdfa.*`: discriminant analysis.
- `src/regressnet.*`: stepwise analysis over cloned networks.
- `src/nsplit.*`, `evaldesign.*`: partition planning and typed design policy.
- `src/crossval.*`, `cvadapters.*`, `cvreport.*`: repetition, model-family
  adaptation, and reporting.
- `src/auccov.*`, `delong.*`, `clustered_auc.*`: paired AUC algebra and covariance.
- `src/obd.*`, `autoalgo.*`, `plateau.h`: architecture and training control.
- `src/modelfactory.*`, `netclone.*`: cold-path construction and cloning.
- `src/asyncjob.*`, `procguard.*`, `gui.cpp`: process/thread boundaries and GUI.

## Repository hygiene

- Preserve unrelated user changes in a dirty worktree; stage by explicit path.
- Do not commit generated run artifacts (`model.txt`, `neuron.log`, sessions,
  uploaded data) unless they are intentional fixtures.
- Keep temporary probes and sabotage harnesses out of the final tree.
- `docs/HISTORY.md` is the append-only forensic archive.  Put current instructions
  here or in a focused authority, never in HISTORY.
- Commit only after relevant gates pass; push only when explicitly requested.

## Historical lookup

The detailed reanimation timeline, legacy defects, completed roadmaps, refactor
evidence, and superseded decisions remain searchable in `docs/HISTORY.md` and
`docs/refactor_audit.md`.  They were removed from the always-loaded instruction
file deliberately: use them when investigating provenance, not as ambient
context for new implementation work.
