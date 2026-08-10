# neuron / Sol restart handoff

Updated: 2026-08-09, after Claude's algorithm-checkbox GUI commit.

## Role of this file

This is Sol/Codex's versioned continuity note for work that may resume from a
different directory or a compacted conversation. It records the current audit
boundary, specialized review concerns, and the working history needed to resume
without reconstructing it from chat.

It supplements rather than overrides `AGENTS.md`, `CLAUDE.md`, the Manifest, or
the source. Those authorities win whenever this snapshot disagrees with current
repository state. Update this file at meaningful handoff points instead of
letting its commit, next task, or verified claims become stale.

## Repository

- Canonical working tree: `/Users/craign/code/neUROn2++/neuron-3.0`
- Branch: `main`
- Verified state at handoff: clean and synchronized with `origin/main`
- Current commit: `794dc673128ddb9c729d4f5c239c8d3928b1218e`
  (`Select a subset of algorithms; the set decides whether they compete`)

If a future session begins in another project that uses neuron, change to the
canonical working tree above before doing repository work. Confirm the state
again; do not assume this recorded commit is still current.

## Mandatory restart sequence

1. Run `git status -sb`, `git fetch origin`, and compare `HEAD`, `main`, and
   `origin/main` before acting.
2. Read root `CLAUDE.md` and `AGENTS.md` completely and follow their lazy-load
   routing.
3. For GUI/API work, read `docs/gui_cli_parity.md`, the Manifest Chapter 4 REST
   contract (`docs/tex/rest_api.tex`), and `tests/gui/README.md` completely.
4. Inspect commit `794dc67` and its tests before changing the new checkbox
   design. Claude reported all gates green, but Sol has not independently
   audited this new commit yet.
5. Preserve the standing prohibition on new legacy-menu work.

## Settled work immediately before the GUI commit

- `6352979`: retained LM-v2 engine implementation.
- `7e6ee88`: consolidated optimizer benchmark table plus the future-algorithm
  maintenance rule.
- `a4e65e1`: CLAUDE/handoff resynchronization.
- `d997db2`: Sol audited and completed LM's public REST/GUI integration and
  automatic-selection participation. Two defects were corrected: the GUI's
  false claim that LM forbids validation, and a selector edge case that could
  spend more than a very small declared total budget.
- `a3852c4`: Manifest title page changed to `Revised August 2026`; maintenance
  now requires checking and updating the month-year whenever the Manifest is
  touched.

LM is retained and public as REST/GUI token 6. It is eligible only for small
(`P <= 512`) single-hidden-layer LMS neural models with batch/epoch enabled and
automatic step-size search disabled. Validation and weight decay are allowed.
There is deliberately no legacy-menu entry.

## Settled checkbox strategy

The old single algorithm drop-down is replaced by one checkbox per learning
algorithm. Eligibility is resolved before deciding whether competition occurs:

- zero eligible checked methods: refuse, naming every reason;
- exactly one eligible method: train it directly, with no probe or competition;
- two or more eligible methods: compete under one fixed 2250 ms total budget,
  divided among the eligible competitors only;
- checked but ineligible methods are explicitly reported, never silently lost.

The GUI may disable currently ineligible choices and show their reasons, but the
REST API remains authoritative and defensive.

## Claude's just-landed GUI/API implementation

Commit `794dc67` implements the strategy above:

- `/api/train algorithm` accepts one token (`5`), a strict comma-separated set
  (`1,4,5`), or `auto` (the full requested universe).
- `autoalgo::plan()` is the shared resolver; `pick()` receives the requested
  set.
- `GET /api/algorithms` reports current eligibility and reasons to the page.
- The Train panel uses six method checkboxes; several checked methods mean
  competition, while one means direct training.
- Eligibility shown by the page is based on the configuration selecting the
  method would produce, avoiding the circular defect where controls that a
  method selection would correct caused that method to be disabled.
- A real-browser Playwright gate was added as `tests/gui/browser.sh`; it is
  required when page JavaScript changes but is intentionally outside the normal
  release gate.

Claude's commit reports: CTest 41/41, smoke, strict parsing (251 checks), async,
browser (25 checks), goldens, oracle, tools, Manifest index, whitespace, PDF
rebuild and visual inspection all green.

## Exact next task boundary

Before further GUI design, independently audit commit `794dc67` for correctness
and contract drift. Pay particular attention to:

- strict parsing of lists: empty members, duplicates, trailing commas, overflow,
  whitespace, and atomic refusal before mutation;
- direct-training behavior after eligibility reduces a requested set to one;
- fixed-total budget accounting only when at least two eligible methods remain;
- cancellation and winner adoption;
- `/api/algorithms` using proposed/effective GUI settings rather than stale live
  settings, without duplicating eligibility rules;
- OBD/CV behavior and optimizer naming after subset selection;
- blocking/async response parity and `autoAlgo`/`competed`/omission JSON;
- real browser behavior, including zero-selection prevention and eligibility
  refresh after model/error/control changes;
- Manifest Chapter 4, Chapter 12, index, `CLAUDE.md`, `AGENTS.md`, parity docs,
  and maintenance rules matching the code exactly.

Do not begin another optimizer candidate until this public integration remains
complete and clean. Do not broaden the GUI beyond the user's next explicit
decision.
