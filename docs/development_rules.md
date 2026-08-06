# Engine development rules

This is the detailed authority for changes to neuron’s C++ engine. Read it for
engine, numerical, training, optimizer, or model-selection work. `CLAUDE.md`
contains only the short triggers so ordinary operational tasks do not load the
history behind every rule.

## 1. Keep operational documentation synchronized

When a command, menu, API field, output interpretation, or working recipe
changes, update `AGENTS.md`. When the GUI or menus change, update
`docs/gui_cli_parity.md`; parity means a visible page control and HTTP parameter,
not merely an internal handler.

## 2. Demonstrate that tests guard the change

A new characterization must fail against the behavior it claims to detect. A
new implementation guard must fail when the new mechanism is sabotaged.
Demonstrably recompile every affected translation unit after introducing the
sabotage and again after restoring it. A clean source diff or “Built target” is
not evidence; require compilation lines in the build log.

Assert production success and non-empty artifacts before comparing values.
Equality between two absent files, empty arrays, null results, or default values
is vacuous. Isolate expected crashes one case per process so one failure cannot
hide later cases. Release/`NDEBUG` behavior is the shipped contract.

ASan is not a standing gate in this agent environment: even a trivial control
has hung at startup. Use a tiny hardened-libc++ probe for focused container
diagnostics, prove its valid control first, and remove it afterward.

## 3. Measure before acting

A document naming a defect or hot path is a hypothesis. Reproduce correctness
claims with the smallest discriminating fixture. For performance, time identical
work, verify identical numerical endpoints, interleave old/new trials, report
spread, and separate fixed overhead. Do not optimize noise.

## 4. Preserve the numerical vocabulary

`Matrix`, `vector_ops`, and `Population` exist so implementation can be read
against the published equations and so Release builds reject invalid shapes at
entry points. Extend this layer when a primitive is missing. Scalar code is
appropriate for order statistics and genuinely scalar published routines; say
why beside it.

`vector_ops` throws `SizeMismatch`, `RangeViolation`, or `EmptyVector`.
`Matrix` throws `BoundsViolation` for an index/range inside one container,
`DimensionMismatch` for incompatible objects, `BadSize` for an invalid intrinsic
size, and `Singular` for numerical non-invertibility. Preserve two deliberate
contracts: `dotprod(input, output)` permits a shorter destination prefix, and
both empty column selections are legal.

## 5. Develop REST and GUI only; freeze the legacy menus

Do no further feature work on the legacy CLI menu interface. It remains frozen
for compatibility, scripted operational use, oracle comparison, and regression
testing; it is not the design target for new capabilities. Do not add menu
entries, prompts, options, or menu-only configuration.

Every new interactive capability is designed and implemented through the REST
API first. Add a visible GUI control when the capability needs a human-facing
control, and keep that control and its REST contract synchronized. The historical
menu-to-GUI matrix remains evidence that the retired menu surface is covered,
while its forward-looking section records REST/GUI capabilities that intentionally
have no menu equivalent.

## 6. Give each mechanism one owner

Put behavior in the lowest class or service that has all information needed to
own it. Coordinators sequence mechanisms; they do not copy formulas, selection
rules, parsing, report interpretation, or type switches. Construction-time type
knowledge belongs in `modelfactory`/`netclone`, not in hot training paths.

## 7. Treat speed as architecture

Use destination-taking matrix/vector overloads and compound operators in loops.
Pass read-only containers by `const&`. Keep allocation, `std::function`, virtual
dispatch, type switches, and generic descriptors out of per-element and
per-exemplar paths. A once-per-run virtual fit hook is different from a
per-exemplar scoring hook. Never hide a published formula behind a sign flag,
comparator, generic index, or descriptor merely to reduce line count.

Measure before changing a hot path and retain the measurement with the decision,
including decisions not to optimize.

## 8. Separate correctness from refactoring

When characterization reveals a defect, land the fail-first correctness change
separately before extraction. A behavior-preserving refactor keeps goldens and
oracle comparisons unchanged. Never re-bless evidence to accommodate it.

## 9. Maintain the Design Manifest as a public contract

For every public class, major result/config/progress object, and principal
method, document purpose, ownership, caller use, complete signature, fields,
mutation, return/failure behavior, example, notes, and a verifiable citation for
named published methods. Distinguish published mathematics from neuron-specific
policy. Add method-level index entries and extend
`tools/check_manifest_index.py`. Follow `docs/manifest_maintenance.md` in full.
