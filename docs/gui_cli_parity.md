# GUI / CLI parity matrix

**This file is a contract, not a report.** Standing rule 5 (CLAUDE.md): every
capability in the CLI menu interface MUST have a GUI equivalent — a page control
*and* an HTTP API parameter. A CLI menu option with no GUI equivalent is a bug.
Any change to the menus (`src/neuron.cpp`) or the GUI (`src/gui.cpp`,
`src/gui_page.html`) updates the matching row here in the same commit.

The CLI is the authoritative feature list because its menus are frozen; the GUI
is the primary human interface and must be a superset. "API param" is the field
`POST` accepts; "GUI control" is the page element that sends it.

Status: ✅ present · 🔲 planned (parity gap being closed 2026-07-19) · — n/a.

## Main menu

| CLI menu option | GUI control | API | Status |
|---|---|---|---|
| 1 Specify dataset | § Dataset panel | `POST /api/load` | ✅ |
| 2 Specify model | § Model panel | `POST /api/model` | ✅ |
| 3 Use model (train) | § Train panel | `POST /api/train` | ✅ |
| 4 Discriminant function analysis | § DFA panel | `POST /api/dfa` | ✅ |
| 9 Quit | — (server lifecycle) | — | — |

## Dataset submenu

| CLI menu option | GUI control | API | Status |
|---|---|---|---|
| Load raw + randomize into train/test | mode=raw + Test fraction | `POST /api/load` `mode=raw&fraction=` | ✅ |
| Load training set / test set | mode=train + Test set upload | `POST /api/load` `mode=train`,`testfile` | ✅ |
| Convert raw → training set (scale, no split) | mode=raw + fraction=0 | `POST /api/load` `fraction=0` | ✅ |
| Save training set / test set | § Session files buttons | `GET /api/save/{train_set,test_set}` | ✅ |
| Save scaling factors | § Session files → Scaling factors | `GET /api/save/scales` | ✅ |
| Continuous (non-discrete) outcome | mode=raw fraction=0 / mode=train | `POST /api/load` `discrete=0` | ✅ |

## Model submenu

| CLI menu option | GUI control | API | Status |
|---|---|---|---|
| 1 Bias nodes on/off | Bias-nodes toggle | `POST /api/model` `bias=` (off → BareProp) | ✅ |
| 2 Hidden layers / nodes | Hidden nodes field | `POST /api/model` `hidden=` (comma list → BackProp) | ✅ |
| 3 Output error function (LMS / X-entropy) | Error-function select | `POST /api/model` `errfunc=` | ✅ |
| 4 Load network from a file | Load-network upload | `POST /api/model` `mode=load` | ✅ |
| 5 Binary logistic regression | Model-type select | `POST /api/model` `type=logistic` | ✅ |
| 6 Log last operation to file | Log-last-op toggle | `POST /api/model` `log_lastop=` | ✅ |
| 7 Log to history file | Log-history toggle | `POST /api/model` `log_history=` | ✅ |

## Use-model / Train submenu

| CLI menu option | GUI control | API | Status |
|---|---|---|---|
| 1 Randomize weights | Randomize weights button | `POST /api/randomize` | ✅ |
| 2 Learning rate (auto / manual η) | Auto-step toggle + η field | `POST /api/train` `eta=`,`autostep=` | ✅ |
| 3 Stopping: max iterations | Max iterations field | `POST /api/train` `maxiter=` | ✅ |
| 3 Stopping: lower error limit | Min-error field | `POST /api/train` `minerr=` | ✅ |
| 3 Stopping: change-in-error limit | Change field | `POST /api/train` `change=` | ✅ |
| 3 Stopping: error-window increase | Window field | `POST /api/train` `errwindow=` | ✅ |
| 3 Stopping: max absolute gradient | Grad-max field | `POST /api/train` `gradmax=` | ✅ |
| (new) Plateau auto-stop tol / window | Auto-stop checkbox + tol/window fields | `POST /api/train` `autostop`,`autostop_tol`,`autostop_window` | ✅ |
| 4 Batch/epoch on/off | Batch/epoch toggle | `POST /api/train` `batch_epoch=` | ✅ |
| 5 Weight decay (on/off + λ) | Weight-decay toggle + λ field | `POST /api/train` `weight_decay=`,`decay=` | ✅ |
| 6 Print counter (log / linear) | Print-counter select + count | `POST /api/train` `logprint=`,`printcount=` | ✅ |
| (train-time) Algorithm (GD/CGD/Shanno/auto) | Algorithm select | `POST /api/train` `algorithm=` | ✅ |
| 7 Train model | Train button | `POST /api/train` | ✅ |
| 8 Save network | § Session files → Network | `GET /api/save/network` | ✅ |
| 9 Stepwise regression | § Stepwise regression panel | `POST /api/regress` | ✅ |

## DFA submenu

| CLI menu option | GUI control | API | Status |
|---|---|---|---|
| 1 Linear DFA | DFA panel → Linear + Run | `POST /api/dfa` `type=linear` | ✅ |
| 2 Quadratic DFA | DFA panel → Quadratic + Run | `POST /api/dfa` `type=quadratic` | ✅ |

## Logging (cross-cutting)

Standing requirement (Craig, 2026-07-19): **every user action is logged.**

| Behavior | Mechanism | Status |
|---|---|---|
| Engine operations (dataset split, randomize, each training run incl. stopped) | `neuron.log` via `addHistory` | ✅ (pre-existing) |
| Per-action audit trail (load/model/set-param/randomize/train/stop) with timestamp + values | workspace audit log | 🔲 |
| Each training run's header records ALL params in effect (η, decay, stopping conditions, plateau) | run header (self-describing) | 🔲 |
