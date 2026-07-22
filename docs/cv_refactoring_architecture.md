# Cross-validation refactoring: DRY within the neuron architecture

> **Standing architectural rule (adopted 2026-07-22).** This document was written for
> the Phase 4 cross-validation refactor, but its layer-ownership and dependency-direction
> principles are **permanent and repo-wide** — they govern every engine refactor, not just
> CV. Referenced from CLAUDE.md "Standing rules" (rule 6). Treat `docs/manifest.pdf` as the
> architectural constitution it names.

I approve the refactor and want the implementation as DRY as reasonably possible, but please treat the architecture documented in `docs/manifest.pdf` as an architectural constitution.

In this environment, DRY should mean that each operation has one authoritative implementation at the correct layer—not that distinct class responsibilities are collapsed into a generic procedure or large orchestration object.

## Preserve these boundaries

- `Matrix` remains the primitive numerical and data-manipulation layer. Matrix operations should be implemented there and reused, not recreated in CV, GUI, or model code.
- `DataSet` owns dataset state and transformations: training, validation, and test matrices; row selection; normalization; and split construction.
- `Model` remains the common modeling contract and owner of the model's `DataSet`.
- `Iterative` owns the common iterative-training lifecycle, stopping rules, and reporting mechanics.
- `Network` owns behavior common to neural and logistic network models.
- `Logistic`, `SimpleProp`, `BackProp`, and the other concrete classes retain responsibility for their particular mathematical architecture and implementation of the appropriate virtual methods.
- OBD owns architecture search, but should operate through the established model and training interfaces.
- Cross-validation should coordinate repeated dataset construction, model construction, training, and evaluation. It should not absorb the responsibilities of those objects.
- The GUI and HTTP handlers should translate user requests into calls to the class layer. They must not become the authoritative implementation of model building or training.

Therefore, extracting duplicated GUI-handler code such as `buildModel()` and `trainConfigured()` is desirable, but the reusable implementation should live at the lowest class-layer boundary that naturally owns it. The GUI handler may call it; CV may call it; nested OBD-CV may call it. None should copy it.

A procedure callback is acceptable as an orchestration mechanism, provided the callback composes the existing objects through their public interfaces. It should not become an alternate model hierarchy or reach into protected or private state.

Please preserve the required construction order described in the manifest—particularly loading the `DataSet` into a model before completing architecture-dependent sizing or initialization.

## Dependency direction

Dependencies should continue to point downward:

```text
GUI/API
   ↓
CV/OBD orchestration
   ↓
Model / DataSet interfaces
   ↓
Iterative / Network / concrete models
   ↓
Matrix and numerical primitives
```

Lower layers must not acquire knowledge of CV, folds, HTTP, GUI controls, or SEER. For example, `Matrix` should know how to gather rows, but it should not know what a test fold is. `DataSet` should know how to materialize a fold, but it should not decide which model to train. A concrete model should know how to train itself, but it should not run the outer CV loop.

Please distinguish genuine duplication from intentional polymorphism. Similar implementations in different concrete model classes are not automatically candidates for consolidation when they encode different model mathematics or satisfy different virtual contracts. Consolidate shared mechanisms only when their semantics and invariants are genuinely identical.

## Refactor before adding behavior

Keep the behavior-preserving refactor separate from the new Phase 4 behavior:

1. First perform a behavior-preserving refactor.
2. Prove existing seeded outputs, reports, model files, and tests remain unchanged.
3. Then implement CV and nested OBD-CV through the extracted interfaces.

Avoid a "god" training function with switches for every model type. Prefer virtual dispatch and small, explicitly owned services. If a proposed extraction makes it difficult to say which class owns an invariant, it is probably at the wrong layer.

The goal is not the fewest lines of code. The goal is one authoritative implementation of each mechanism, located in the class that conceptually owns it, while retaining the manifest's ordered progression from primitives to datasets to models to training procedures to user interfaces.

## Phase 4 ownership rule

> **Cross-validation owns repetition, not training.**  
> **OBD owns model selection, not evaluation.**  
> **Models own fitting, not fold management.**  
> **`DataSet` owns fold materialization, not modeling policy.**

This is the governing interpretation of DRY for the Phase 4 refactor.
