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
| Page behavior (`gui_page.html` JavaScript) | `tests/gui/README.md` — drive it with `./tests/gui/browser.sh`, never only by curl |
| Cross-validation or inference | `docs/cross_validation.md`; `docs/evaluation_report_spec.md`; relevant Manifest services |
| ROC/statistics | `docs/roc_theory.md`; relevant cited Manifest section |
| Manifest editing | `docs/manifest_maintenance.md` in full |
| Why an old decision was made | Search `docs/HISTORY.md` and `docs/refactor_audit.md`; do not load either by default |

Plans retained for provenance (`docs/obd_plan.md`,
`docs/cv_refactoring_architecture.md`, `docs/b9_strict_parsing.md`,
`docs/stepwise_async_results_spec.md`) describe work
as it was undertaken.  They are historical, not the current public contract.
The Manifest and source are authoritative after implementation.

## Current state

- The behavior-preserving engine refactor and ROADMAP 4 are complete.
- Release builds and tests are the normal gate.  Current test count is discovered
  from CTest; never copy a remembered count into a status report.
- Optimizer research is neural-only and follows `docs/optimizer_research.md` and
  `docs/learning_research/optimizer_implementation_plan.md`. For optimizer work,
  read `docs/learning_research/neural_optimizer_handoff.md` first; it owns phase
  status, measurements, the standing panel and the exact next step. The durable
  session triggers are: apply the large-workload speed governor; lead with the
  portfolio policy rather than one speed figure; treat seed stability and the
  conditioning pair as first-class acceptance axes; expose nothing publicly
  before retention; and never alter a published formula merely to share code.
  Retained neural methods are L-BFGS, iRPROP+, and Levenberg--Marquardt (LM-v2).
  L-BFGS (`algorithm=4`), iRPROP+ (`algorithm=5`), and LM (`algorithm=6`) are
  public through REST and the GUI. `/api/train`'s `algorithm` field names ONE
  method (`5`), a SET of them (`1,4,5`), or `auto` (the set of everything), and
  the GUI's Train panel is one checkbox per method. Eligibility is resolved
  before the count is read: no eligible member is a refusal, exactly one trains
  directly with no probe budget spent, two or more compete under one fixed
  2250 ms total divided equally. The selector reports omissions by name. None
  of the retained methods is in the retired menus. BB was screened,
  rejected and removed; its exact historical tree is the annotated
  `research/bb-prototype-2026-08-08` tag. The completed LM evidence is
  `lm_source_decision.md`, `lm_v2_source_decision.md`, `lm_step0_results.md`,
  and `lm_screen_results.md`. Do not reopen completed candidates or spend this
  program on Logistic, DFA, exhaustive baseline timing or timing minutiae.
- The Design Manifest is normative and must stay synchronized with every public
  class, major object, principal method, algorithm, failure contract, and index entry.

## Non-negotiable project rules

The complete operational wording is in `docs/development_rules.md`; these are the
session-level triggers:

1. Update `AGENTS.md` when an operational recipe changes.  If the GUI or REST
   surface changes, update `docs/gui_cli_parity.md` in the same commit -- and if
   the page's JavaScript changed, CLICK IT: `./tests/gui/browser.sh`.  The gate
   greps the served page and cannot see it run.
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
10. A retained learning algorithm is not complete at engine retention. Before
    another candidate begins, expose it REST-first, add the synchronized GUI
    control where appropriate, decide and document automatic-selection
    membership, update Chapter 4 and Chapter 12 of the Manifest plus operational
    and parity docs, and pass the public-surface gates. Rejected candidates are
    removed instead; research-only status must be explicit.

## Build and verification

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
tests/golden/run_golden.sh
tests/gui/smoke.sh
tests/gui/asyncjob.sh
tests/gui/strictparse.sh
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
