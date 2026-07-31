# Maintaining the neuron Design Manifest

> **NON-NEGOTIABLE OBJECT-DOCUMENTATION RULE:** The manifest is an
> architectural specification for programmers, not merely a user guide or REST
> reference. Before a source change is complete, audit every new or changed
> public class, struct, enum, namespace service, method, and caller-visible data
> member against Chapters 8 and later. Any object absent there must receive a
> manifest entry following Section 7.1: purpose, public methods, public variables,
> a usable example, and important notes. A REST endpoint never substitutes for
> the object specification that implements it.
>
> **PLACEMENT, HIERARCHY, AND INDEX ARE PART OF THE CHANGE.** Choose the entry's
> chapter and position from its architectural ownership and dependencies; do not
> append it wherever editing is convenient. A new high-level object must also be
> introduced in Chapter 12 and added to Figure 12.1. Every new object, important
> method, algorithm, and statistical estimator must receive an explicit
> `\\index{...}` entry, and the index must be rebuilt and inspected in the
> published PDF.

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

The other developer documents in `docs/` have narrower jobs:

- `evaluation_report_spec.md` is the normative presentation and artifact contract
  for cross-validation output. It tells implementers what Tier 1, Tier 2, Tier 3,
  and the locked-test layer must contain; the manifest explains the architecture
  that produces them.
- `gui_cli_parity.md` is an enforcement matrix. It makes every frozen CLI
  capability traceable to a GUI control and REST parameter and exposes a missing
  interface as a reviewable gap.
- `HISTORY.md` is the dated forensic archive: measurements, rejected approaches,
  completed roadmaps, and the reasons behind settled decisions. It is deliberately
  not a current specification.

All three belong under `docs/` because they are repository-wide contracts or
records read by maintainers, not source modules and not root-level onboarding.
The README should link only the material a new user needs; `CLAUDE.md`,
`AGENTS.md`, and this guide route maintainers to the rest.

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
| `docs/tex/modern_objects.tex` | Current Chapter 12 object/service contracts, examples, notes, and index entries |

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

### How to place object documentation

Placement follows the dependency direction, from foundations to orchestration:

1. Numerical containers and generic operators belong with `vector` and
   `Matrix` in Chapter 8.
2. General utilities belong in Chapter 9; reusable statistical primitives and
   estimators belong in Chapters 10 and 11.
3. Dataset, model, training-control, evaluation-design, selection, and reporting
   objects belong in Chapter 12, ordered from the objects that own data and
   mechanisms toward the coordinators that compose them.
4. A feature-specific compound object belongs in its own chapter only when its
   workflow needs a sustained explanation beyond an object contract, as with
   stepwise regression in Chapter 13. Chapter 12 must still locate that object in
   the overall hierarchy and point to the detailed chapter.

Do not organize by implementation date, REST route, or roadmap phase. Ask which
object owns the invariant, which lower-level objects it calls, and which
higher-level callers consume it. Put the contract beside its architectural
peers. When no existing chapter can hold it without mixing layers, add a focused
chapter and explain the dependency boundary at its start.

For each entry, apply the complete Chapter 7 documentation pattern even when the
implementation uses a namespace rather than a class. Do not reduce Section 7.1
to a list of names. The entry must contain, in this order:

1. **Natural-language purpose.** Say what problem the object or feature solves,
   what it owns, and how a caller uses it. Use the conversational tone required
   by Chapter 7.
2. **Algorithmic or mathematical description, when applicable.** State the
   transformation, estimator, selection rule, or data flow. Give an equation or
   ordered algorithm when that is clearer than prose. A method list alone does
   not document an algorithm.
3. **Complete public specification.** Give every caller-visible signature,
   parameter meaning, accepted range or units, return value, mutation or
   ownership effect, and documented failure behavior. Document overloads
   separately. Never make the reader infer inputs and outputs from a method name.
4. **Public types and state.** List every public enum value and every field of a
   public result, configuration, progress, or error structure, explaining its
   meaning and unavailable/sentinel states.
5. **Example.** Provide a small compilable or near-compilable call sequence that
   shows realistic inputs, the returned object or changed state, and how failure
   is checked. A declaration-only fragment is not an example.
6. **Notes.** State dimensional and cardinality requirements, prerequisites,
   invariants, exceptions/refusals, efficiency tradeoffs, mutation, ownership,
   reproducibility, numerical assumptions, and interactions with related methods.
   Include only the categories that apply, but never omit a known calling
   constraint.

Chapter 7 also requires documentation in three places: the Design Manifest,
immediately before the method in source as its specification, and within the
implementation where the algorithm needs explanation. A manifest review must
check all three. The Manifest is the reusable caller contract; the source comment
is the local contract; inline comments explain implementation decisions.

Follow the visual grammar of the original specifications (for example, the
Matrix dot-product and Gaussian-function entries). After a short explanatory or
mathematical introduction, put each callable signature in its own `itemize` item
and state its inputs, output, and effect. Put public variables in a separate
list, with one field or closely related field group per item. Do not compress a
dozen methods or fields into a prose paragraph: that may save pages, but it makes
the contract difficult to find, understand, and implement. Page count is not a
design constraint. Use visibly labeled `Example` and `Notes` subsections
consistently.

Before accepting an object section, perform a **vertical-list test**: a reader
scanning down the page must be able to find each method and public field without
parsing a dense paragraph. Then perform a **caller test**: for any listed method,
the reader must be able to tell what to pass, what comes back, what changes, and
what can fail without opening the header.

Private implementation fields need not be transcribed mechanically, but public
state must not be omitted. If a header deliberately exposes a result struct,
every field is part of the architectural contract and belongs in the entry.

### Chapter 12 and Figure 12.1

Chapter 12 is the authoritative map of neuron objects. Whenever a high-level
object or service is added, removed, renamed, or changes dependency direction:

1. add or revise its explanatory bullet and full contract in Chapter 12;
2. update the editable source `docs/tex/figures/objects.dot`;
3. regenerate `docs/tex/figures/objects.pdf` with Graphviz;
4. keep Figure 12.1 on its dedicated landscape page at approximately the full
   printable width; do not shrink it back into the Chapter 12 opening text;
5. check that the prose and arrows agree with the actual includes and calls.

Render Figure 12.1 by itself after every change and inspect it at normal page
scale. Every node label and edge label must be readable without zooming. If the
graph becomes too crowded, improve its layout or divide responsibilities into a
second explanatory figure; never solve crowding by reducing the principal map
to an unreadable thumbnail.

### Current source-to-manifest coverage map

The July 2026 full-header audit established this routing table. Re-run the audit
when a header or public type is added; do not treat the table as permission to
skip source inspection.

| Source objects/services | Manifest home |
|---|---|
| `vector_ops.h`, `matrix.h` | Chapter 9, vector and Matrix |
| `utility.h` | Chapter 10, Utility methods |
| `stats.h` | Chapter 11, Statistical methods |
| `function_defs.h` | Chapter 12, Function definitions |
| `dataset.h`, `twoset.h` | Chapter 12, established specifications plus current-contract additions |
| `model.h`, `iterative.h`, `network.h` | Chapter 12, inheritance specifications plus current training contract |
| `bareprop.h`, `simpleprop.h`, `backprop.h`, `logistic.h` | Chapter 12, concrete models plus current resizing/result additions |
| `dfa.h`, `ldfa.h`, `qdfa.h` | Chapter 12, discriminant model hierarchy |
| `split.h`, `evaldesign.h` | Chapter 12, partition planning and typed evaluation design |
| `auccov.h`, `delong.h`, `clustered_auc.h` | Chapter 12, common AUC algebra and sampling-unit-specific covariance |
| `plateau.h`, `autoalgo.h`, `modelfactory.h`, `netclone.h` | Chapter 12, training-control and construction services |
| `obd.h` | Chapter 12, architecture selection |
| `crossval.h`, `cvadapters.h`, `cvreport.h` | Chapter 12, repetition, model-family adapters, reporting, and artifacts |
| `regressnet.h` | Chapter 12 hierarchy/current contract and Chapter 13 workflow |
| `gui.h` | Chapters 2 and 4 interface/orchestration boundary; it declares no object |
| `version.h`, `stdafx.h` | Build/version constants only; no caller object contract |

Nested result, progress, error, and configuration types are covered with their
owning object or namespace. A new nested public type must be added there even if
the header itself already appears in this table.

### Index maintenance

The index is a required navigation layer, not an optional finishing touch.
Every added object and every principal method, algorithm, estimator, or design
term must have a deliberate `\\index{...}` entry at its defining discussion.
Run the normal `latexmk` build through `makeindex`, then inspect both the
generated index pages and representative links/page numbers in the PDF. Merely
seeing an `.idx` file is not verification.

Index for how programmers search, not merely for source-file names. Include the
object or namespace, each principal public method, public result/configuration
types, important algorithms and estimators, statistical assumptions, failure
concepts, and useful synonyms or hierarchical forms. A module-level entry alone
is insufficient when the section defines several independently useful methods.

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
| Cross-validation report layout and artifacts | `docs/evaluation_report_spec.md`, `src/cvreport.{h,cpp}` |
| Evaluation design and clustered covariance | `src/evaldesign.{h,cpp}`, `src/auccov.{h,cpp}`, `src/delong.{h,cpp}`, `src/clustered_auc.{h,cpp}` |
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

Evaluation policy and covariance are also separate. `evaldesign` names the achieved
partition, declared sampling unit, and permitted estimator. `delong` estimates
ordinary independent-row covariance; `clustered_auc` estimates Obuchowski clustered
covariance; `auccov` owns their shared placements, point AUC, and contrast algebra.
Do not describe grouping as though it performs clustered inference, or describe a
clustered interval as DeLong.

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
- [ ] Audit every new or changed public class, namespace, method, overload, enum,
      result/configuration field, and error contract.
- [ ] Document purpose, applicable algorithm or mathematics, complete signatures
      and parameter/return semantics, public state, a usable example, and notes.
- [ ] Verify the same contract immediately precedes the method in source and that
      implementation decisions needing explanation are commented in the code.
- [ ] Apply the vertical-list test and caller test; reject compressed inventories.
- [ ] Update the architecture narrative and source map if ownership changed.
- [ ] Update Figure 12.1 for a high-level object or dependency change, retain its
      full-page landscape layout, and inspect every label at normal page scale.
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
