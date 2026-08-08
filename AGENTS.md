# Operating neuron — a guide for AI assistants

You are working in **neuron 3.0**, a neural computational modeling
environment: feed-forward neural networks and binary logistic regression
with serious statistics (ROC areas with 95% confidence intervals, Wald
tests, goodness-of-fit). This root file contains the rules and routing that
apply broadly. The longer build/data/train/deploy recipes are loaded on demand
from `docs/agent_data_workflows.md`. When you finish a task, tell the user what
the numbers mean, not just what they are.

Ground rules:

- **No further CLI-menu development (standing rule 5).** The legacy interactive
  menus are frozen for compatibility, scripted operational use, and regression
  testing only. Do not add menu entries, prompts, options, or equivalents for
  new features. Build every new interactive capability REST-first, add a GUI
  control where appropriate, and keep the GUI and REST contract synchronized.
  Update **`docs/gui_cli_parity.md`** in the same commit as a GUI or REST surface
  change; its legacy matrix is historical coverage, not a new-work checklist.
- **If you are about to write engine code, read standing rules 4, 6 and 7 in
  `CLAUDE.md` first.** In short: the class layer (`Matrix`, `vector_ops`,
  `Population`) is the numerical vocabulary and code must read against the
  equations it came from (rule 4); each mechanism has one authoritative
  implementation in the class that owns it (rule 6); and **speed is an
  architectural requirement** (rule 7) — prefer the destination-taking overload
  and the compound operators in loops, pass read-only containers by `const&`,
  keep `std::function`/virtual dispatch/allocation out of hot loops, measure
  before optimizing, and never hide a published formula to save lines.
- The engine is `build/neuron`, an interactive menu program. **Never drive
  it by typing interactively** — write the menu answers to a file, one per
  line, and run `./build/neuron --seed 42 < session.in > session.out`.
  Always pass `--seed N` so the run is reproducible; always read
  `session.out` afterward to harvest results (and check it doesn't end in
  an error or an unanswered prompt).
- The Python tools in `tools/` run on bare `python3` — never pip-install
  anything for them.
- The engine writes its logs (`model.txt`, `neuron.log`) into the
  directory of the first dataset file the session loads (falling back to
  the current directory if none is ever loaded) — so the run's artifacts
  end up next to the user's data, where they expect them.
- Costs: grooming is instant; the focused guide's trainings take seconds. If the
  *iteration table* is still printing after ~2 minutes, something is wrong —
  kill it and re-check the recipe (usually a missing iteration cap or
  missing `--refcat`). The statistical report AFTER training (including its
  2,000-resample ROC bootstrap per set) adds a few seconds — ~2 s on the
  a few seconds on datasets with thousands of rows — and scales gently
  (nothing in the report is quadratic any more). A short pause after
  "Total iterations" is the report working, not a hang.
- The train/test split (menu 5 / `/api/load` with a fraction) is stratified on
  the outcome and scales to large data — a 226,000-row raw load plus split is
  sub-second (it was O(n²) and would have taken minutes until ROADMAP 4).

## Build, data, training, and deployment workflows

For building the engine, grooming a dataset, scripted CLI training, interpreting
the resulting statistics, or producing a deployed browser calculator, read
`docs/agent_data_workflows.md` in full before acting. It owns the verified
recipes, iteration caps, deployment prerequisites, and report-interpretation
rules. Keep the root rules above in force while following it.

## 3b. The GUI and loopback API (`neuron --gui`)

`./build/neuron --gui` starts the browser interface on loopback; use
`--no-browser` for scripted smoke work. The REST API is the authoritative surface
for new interactive capabilities, and the GUI is its primary human client. The
retired menu remains available only for compatibility and scripted legacy runs.

Do not load the entire API contract for an unrelated training task. For GUI or
HTTP work, read these focused authorities:

- `docs/gui_cli_parity.md`: legacy coverage plus the current GUI ↔ REST contract.
- Manifest REST chapter: endpoint fields, defaults, strict parsing, response and
  busy/error behavior.
- `docs/cross_validation.md` and `docs/evaluation_report_spec.md`: CV, nested
  selection, locked-test inference and artifacts.
- `docs/obd_plan.md`: historical OBD implementation rationale; current public
  contract is the Manifest and source.
- `docs/b9_strict_parsing.md`: historical B9 evidence; current accepted field
  tokens are the Manifest REST tables.

Operational invariants that belong in every GUI session:

- Long training, OBD, CV and stepwise operations are asynchronous when requested.
  Poll `GET /api/train/status`; cancel with `POST /api/train/stop`.
- Send bodyless POSTs with an explicit empty body, for example
  `curl -X POST -d "" ...`; otherwise the HTTP library waits for its read timeout.
- While a long job owns the engine, other engine-touching endpoints return 409
  with `busy:true`.
- Training continues from current weights. Randomize explicitly for a fresh start.
- Direct retained neural optimizers use `/api/train` `algorithm=4` for L-BFGS
  and `algorithm=5` for iRPROP+. Both require `batch_epoch=1` and `autostep=0`;
  the GUI enforces those controls. They deliberately have no legacy-menu entry.
- Present boolean fields use exactly `1` and `0`. Unknown, trailing, overflowing,
  or non-finite numeric text is a field-specific error; omission and an empty
  field retain the endpoint-specific defaults documented in the Manifest.
- The GUI and API use one-based input-column numbers; C++ engine methods use
  zero-based positions.
- GUI actions are appended to `neuron_actions.log` in the run directory.
- Do not extend the menus. If a GUI or REST capability changes, update
  `docs/gui_cli_parity.md` in the same commit and exercise both blocking and
  asynchronous forms where applicable.

For agents, prefer the scripted CLI sessions in
`docs/agent_data_workflows.md` §2 unless the task specifically concerns GUI/API
behavior. The verified end-to-end API gate is
`tests/gui/smoke.sh`; focused GUI characterization lives under `tests/gui/`.

## 4. Verifying the installation

- Quick: `./tests/tools/run_tools.sh` (Python tools vs committed outputs,
  including the deployment forward-pass check against the engine) and
  `./tests/golden/run_golden.sh` (three seeded engine sessions -- `xor_seed42`,
  `regress_seed42`, `binormal_seed42` -- must match
  committed transcripts byte-for-byte).
- GUI: `./tests/gui/smoke.sh` (every endpoint and the page's own controls) and
  `./tests/gui/asyncjob.sh` (the async-job lifecycle underneath all four long
  jobs — the busy gate, the two open doors, cancellation, and the reset that
  keeps one job's progress out of the next one's status). Both start a real
  server on an OS-assigned port; the second is separate so that a hang or a
  terminated server there cannot take the endpoint coverage with it.
  The same lifecycle's *ordering* rules — a result is published before the job
  reports itself idle, and the job is marked running before `start()` returns —
  are pinned in-process by the `asyncjob_lifecycle` ctest case, because neither
  is observable through HTTP (`docs/refactor_audit.md` §20.4).
- Request fields: `./tests/gui/strictparse.sh` (every handler's omission rules,
  present-but-empty rules, domain refusals, and the malformed values each field
  must refuse by name). Third server-starting script, separate for the same
  reason as the second. The contract it pins is `docs/b9_strict_parsing.md`.

### How a request field is read (since 2026-08-03)

Every field on every endpoint goes through the same rules, so a malformed
request is refused by name instead of being silently reinterpreted:

- **Whitespace** around a value is trimmed, and the rest must be consumed
  entirely. `fraction=0.3` and `fraction=" 0.3 "` are the same; `fraction=0.3junk`
  is refused. There is no partial reading.
- **Whole numbers**: optional `+`, then digits. A negative is a sign error, and
  a value wider than the type is a range error, not a wrap.
- **Numbers**: anything `strtod` accepts, consumed entirely, and **finite** —
  `nan` and `inf` are refused on every field.
- **Flags**: exactly `1` or `0`, case-sensitively, everywhere. `true`, `yes`
  and `TRUE` are refused; before this they silently meant false.
- **Omitted** keeps the current value. **Present-but-empty** does too for
  numbers and counts — the page sends `fraction`, `seed` and `decay` on every
  request whether or not they are filled in — with two exceptions: on
  `/api/train` an empty `minerr`, `change`, `errwindow` or `gradmax` *disables*
  that stopping condition, and an empty **flag** is refused.
- **Refusals name the field**: `folds: '5junk' is not a whole number`,
  `gradmax: 'inf' is not a finite number`, `async: 'true' must be 1 or 0`. A
  syntax fault and a domain fault read differently on purpose.
- **A refusal applies nothing.** Each handler reads all of its fields before it
  changes anything, so a request refused for its last parameter has not applied
  its first.

Every maintained curl recipe already uses `=1`/`=0` and unambiguous numbers,
so none of them changes. Writing a new endpoint or parameter means using
`readUnsigned` / `readDouble` / `readBool` in `src/gui.cpp`;
`tools/check_strict_parsing.py` fails the build otherwise.
- The low-birth-weight dataset is a self-verifying reference: follow
  `docs/datasets/low-birth-weight/README.md` and the engine should report
  log likelihood −111.2865 on the committed betas.

## 5. Repository map (for when you need more)

| Where | What |
|---|---|
| `src/`, `build/neuron` | The C++ engine (menus: dataset → model → train → stats) |
| `tools/mkdataset.py` | CSV → neuron-ready dataset (`docs/agent_data_workflows.md` §1) |
| `tools/neuron2web.py` | trained model → standalone HTML calculator (`docs/agent_data_workflows.md` §3) |
| `docs/deploy.md` | Label-spec reference for deployment |
| `docs/roc_theory.md` | What the ROC statistics mean and how to cite them |
| `docs/manifest.pdf` | The complete manual (menu-by-menu) |
| `CLAUDE.md` | Standing rules, current state, settled decisions, roadmap (for working **on** neuron rather than **with** it) |
| `docs/HISTORY.md` | The dated development record — read on demand for the reasoning behind a decision |

Maintainers: keep this file and `docs/agent_data_workflows.md` synchronized when
their rules, tools, or recipes change; the routed recipes are promised to work.

**If you are changing the engine rather than using it**, read the standing rules
at the top of `CLAUDE.md` first. The short version, learned expensively: a green
test suite is not evidence until you know it executes what you changed — run a
new test against the old binary and watch it fail before you trust it — and a
doc that names a mechanism (including one in this repo) is a hypothesis, not a
finding. Measure it.
