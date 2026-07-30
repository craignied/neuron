# Maintaining the neuron Design Manifest

This document records what we learned while bringing the original neUROn2++
Design Manifest forward for neuron 3.0. Use it whenever adding a feature,
changing an interface, or revising `docs/manifest.pdf`.

## Purpose of the manifest

The manifest serves two related purposes:

1. It explains the current neuron 3.0 architecture and public interfaces.
2. It preserves the mathematical and object-design reasoning inherited from
   neUROn2++.

Those purposes must remain visibly separate. Current instructions should never
be mixed into historical implementation details in a way that makes an obsolete
procedure look supported.

The published PDF is the human-readable architecture reference. It is not a
development log, roadmap, or substitute for exact interface contracts maintained
elsewhere in the repository.

## Source layout

The root LaTeX source is:

```text
docs/tex/manifest.tex
```

Current neuron 3.0 chapters are split into focused files:

| Source | Contents |
|---|---|
| `docs/tex/modern_architecture.tex` | Layer ownership, data flow, models, selection, evaluation, artifacts, and source map |
| `docs/tex/current_driver.tex` | Current CLI role, reproducible sessions, menu organization, convergence, and reporting |
| `docs/tex/rest_api.tex` | Loopback HTTP endpoints, parameters, responses, async jobs, and downloads |
| `docs/tex/current_tooling.tex` | `mkdataset.py`, `neuron2web.py`, validation, and deployment |

Small PDF figures used by the retained foundational chapters live in:

```text
docs/tex/figures/
```

The generated, tracked artifact is:

```text
docs/manifest.pdf
```

LaTeX intermediates and the temporary `docs/tex/manifest.pdf` are build
artifacts. Do not commit them.

## Document organization

### Part I: Current neuron 3.0

Part I is normative at the architectural level. It should describe:

- current layer ownership and dependency direction;
- supported human and programmatic interfaces;
- current data preparation, splitting, modeling, training, and evaluation
  workflows;
- current REST endpoints and important parameter semantics;
- current helper tools and supported deployment paths;
- reproducibility, convergence, audit, and artifact behavior.

Write this part in present tense. A reader should be able to use it without
knowing the history of neUROn2++.

### Part II: Foundational design record

Part II preserves the mathematical derivations, numerical foundation,
documentation philosophy, and original object-design reasoning.

It is explanatory rather than normative. Detailed method lists and code examples
may reflect neUROn2++. When a foundational section conflicts with Part I or the
current source, Part I and the source tree are authoritative.

Do not casually rewrite mathematical or historical material merely to make it
sound newer. Update it only when:

- the underlying mathematics was corrected;
- the current implementation materially changes the architectural lesson; or
- an obsolete operational instruction could mislead a present-day reader.

## What was deliberately omitted

The published manifest no longer includes:

- MSVC project, workspace, RTTI, warning-pragma, or library-path instructions;
- hand-maintained Makefile and bundled-GSL setup;
- the old interactive driver manual;
- configurable trapezoidal ROC threshold counts;
- Perl helper scripts;
- Palm OS and early iPhone exporters;
- the old pre-menu data and network file structure;
- the old preprocessor-header configuration system.

Do not restore these to the published PDF. Their historical existence is recorded
in the repository history and legacy distribution.

Windows installation belongs in the root `README.md`, where it points to the
tested CI workflow. The manifest does not need a second Windows build guide.

## Authoritative sources

The manifest summarizes behavior; it must not invent or independently redefine
it. Check the authoritative source before changing prose.

| Subject | Authority |
|---|---|
| Build, dependencies, language standard | `CMakeLists.txt`, root `README.md`, CI workflow |
| GUI/CLI/API feature parity | `docs/gui_cli_parity.md` |
| REST route registration and exact validation | `src/gui.cpp` |
| Browser controls and fields sent to the API | `src/gui_page.html` |
| Engine layer ownership | `docs/cv_refactoring_architecture.md`, current C++ classes |
| Statistical interpretation | `docs/roc_theory.md`, engine reports, statistical tests |
| Dataset preparation | `tools/mkdataset.py`, `tools/README.md`, tool tests |
| HTML deployment | `tools/neuron2web.py`, `docs/deploy.md`, tool tests |
| Operational agent recipes | `AGENTS.md` |
| Dated development history | `docs/HISTORY.md` |

### API documentation rule

When adding or changing an endpoint:

1. Read the registered route and its handler in `src/gui.cpp`.
2. Record the HTTP verb and exact path.
3. Record accepted fields using their actual public spellings.
4. Distinguish required, optional, defaulted, and mutually exclusive fields.
5. State important validation and refusal behavior.
6. State whether the operation mutates the current model.
7. For long jobs, document async-only versus optional async behavior, status,
   cancellation, and HTTP 409 busy behavior.
8. Verify examples against the handler. Never infer a parameter from a plan or
   test name.

The audit that produced this guide caught a documented `repeats` field that did
not exist in `/api/cv`. Source inspection must precede publication.

### Helper-tool documentation rule

There are currently two user-facing programs in `tools/`:

- `mkdataset.py`
- `neuron2web.py`

Before revising their chapter, inspect the programs' argument parsers and
validation paths, not only their README prose. Document supported model and field
types precisely. For example, `neuron2web.py` supports `K %% N` numerical
missing-indicator pairs; it does not support categorical or binary `K` forms.

If a new user-facing helper is added:

1. Preserve the standard-library-only rule unless the project explicitly changes
   it.
2. Add it to `tools/README.md`.
3. Add committed fixture coverage to `tests/tools/run_tools.sh`.
4. Add it to the overview table in `current_tooling.tex`.
5. Document its inputs, outputs, validation, supported scope, and one verified
   example.

Internal test, fixture, migration, or maintenance scripts do not automatically
belong in the manifest.

## Adding new architecture

The manifest follows the project's ownership constitution:

```text
GUI and REST API
        |
experiment orchestration
        |
Model and DataSet interfaces
        |
Iterative, Network, and concrete models
        |
Matrix, vector operations, statistics, and utilities
```

When documenting a new capability, explain:

- which layer owns the mechanism;
- which higher layers call it;
- what state it reads or changes;
- what artifact or report it produces;
- how it handles reproducibility and failure;
- how it is reached through the GUI and REST API;
- whether it participates in CLI parity or is intentionally GUI-beyond-CLI.

Avoid describing a convenience caller as though it owns the computation.
Cross-validation owns repetition, not training. OBD owns architecture selection,
not final evaluation. Models own fitting, not fold management. DataSet owns fold
materialization, not modeling policy.

Update the source map in `modern_architecture.tex` when adding a new architectural
module or materially changing ownership.

## Writing conventions

- Preserve the existing report-style LaTeX organization and restrained visual
  design.
- Put substantial current chapters in separate `docs/tex/*.tex` files and include
  them from `manifest.tex`.
- Use `\texttt{}` for short identifiers and `\path{}` for long repository paths
  that need line wrapping.
- Use `verbatim` for commands and multi-line examples.
- Escape underscores in ordinary LaTeX text.
- Keep tables within the report class's narrow text width. Prefer `longtable` for
  multi-page reference tables.
- Use ASCII hyphens in new source text.
- Write current behavior in present tense.
- Explain what statistical numbers mean and state important assumptions.
- Do not paste roadmap status, dated work logs, or speculative features into the
  manifest.
- Do not duplicate long installation instructions already maintained in the
  README.

## Building the PDF

Build from `docs/tex/`:

```sh
cd docs/tex
latexmk -pdf -interaction=nonstopmode -halt-on-error manifest.tex
cp manifest.pdf ../manifest.pdf
```

The source now uses modern `pdfLaTeX`, `hyperref`, and the archived PDF figures.
Do not reintroduce the old `dvipdfm` driver, raw PDF specials, or missing BMP
figure references.

After copying the final PDF, remove intermediates:

```sh
latexmk -C manifest.tex
rm -f manifest.pdf
```

Only `docs/manifest.pdf` is the published PDF artifact.

## Verification workflow

Every meaningful manifest change requires all of the following.

### 1. Source audit

- Compare factual claims with the current code or authoritative document.
- Search the published text for obsolete terms relevant to the edit.
- Confirm endpoint names, field names, defaults, supported types, and refusal
  behavior.

### 2. Clean LaTeX build

Run:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error manifest.tex
```

The build must finish successfully. Fix new undefined references and new serious
layout warnings. Do not claim that the PDF is updated merely because a stale
tracked PDF still opens.

### 3. Extracted-text check

Use `pdftotext` to confirm that:

- new headings and important phrases appear;
- deliberately omitted sections do not appear;
- examples contain the expected endpoint and option spellings.

Example:

```sh
pdftotext docs/manifest.pdf /tmp/manifest.txt
rg "Loopback REST API|Data preparation and deployment tools" /tmp/manifest.txt
```

### 4. Visual review

Render every changed page and its neighboring transition pages:

```sh
mkdir -p tmp/pdfs/manifest-review
pdftoppm -f FIRST -l LAST -jpeg -r 110 \
  docs/manifest.pdf tmp/pdfs/manifest-review/page
```

Inspect the rendered images for:

- clipped or overlapping text;
- tables extending past margins;
- awkward chapter or part transitions;
- broken figures;
- unreadably small tables;
- unexpected blank pages;
- title, contents, page-number, and bookmark problems.

Text extraction is not a substitute for visual inspection.

### 5. Repository hygiene

- Run `git diff --check`.
- Do not commit LaTeX intermediates or rendered review images.
- Preserve unrelated working-tree files.
- Update this guide if the maintenance process itself changes.

## Checklist for a new feature

- [ ] Decide whether it belongs in Part I or the foundational record.
- [ ] Identify the authoritative implementation and interface contract.
- [ ] Update the architecture narrative and source map if ownership changed.
- [ ] Update the REST chapter if routes or parameters changed.
- [ ] Update the CLI chapter and `docs/gui_cli_parity.md` if parity is involved.
- [ ] Update the tooling chapter for a new or changed user-facing helper.
- [ ] Include reproducibility, mutation, failure, and artifact behavior.
- [ ] Build the PDF from a clean LaTeX state.
- [ ] Extract and search its text.
- [ ] Render and inspect all changed pages.
- [ ] Copy the final PDF to `docs/manifest.pdf`.
- [ ] Remove intermediates and run `git diff --check`.

## Final principle

The manifest is most valuable when it explains stable architecture and verified
public behavior. Keep history where it clarifies design, keep operational details
where they are maintained, and verify every current claim against the code before
publishing it.
