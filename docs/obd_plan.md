# ROADMAP 2 Phase 4 — OBD hidden-layer sizing: implementation plan

*Written by Fable 2026-07-19 as the handoff for the Opus session that builds this.
Read CLAUDE.md's standing rules first; rules 2 (prove tests fail), 3 (measure
before believing), 4 (class layer), 5 (GUI/CLI parity matrix), and 1 (AGENTS.md
same-commit) all bind here. The roadmap sketch lives in CLAUDE.md → ROADMAP 2 →
"OBD hybrid driver"; this document supersedes it where they differ.*

**Goal (Craig's decision, 2026-07-14): hybrid grow→prune hidden-layer sizing for
SimpleProp.** Grow until overlearning (test error diverging from train), prune
back by saliency, confirm. GUI/API only — the CLI menus are frozen and get no
new features.

---

## 1. Claims verified against the code (2026-07-19, this session)

These were the sketch's "measure first" items. All settled by reading; the
line numbers are current at commit `8220a10`.

1. **`setHidden` IS destructive** — `simpleprop.cpp:97-125`: bare
   `Matrix::resize` (documented GARBAGE fill, `matrix.h:100`) on `hW`/`hWup`/`hG`,
   vector `resize` on `hO`/`oW`/`oG`/`h_err`, then `weightsSetFlag = false`.
   So `growHidden` must snapshot the old weights itself and copy them back;
   never rely on resize preserving anything.
2. **Grow-with-zero-outgoing-weights = bit-identical forward pass — TRUE, with
   one trap: the bias slot moves.** `forward()` (`simpleprop.cpp:601-626`) ends
   in `x = dotprod( hO, oW )` where **the LAST element of `hO`/`oW` is the bias**
   (`hO[nHidden] = 1`). Growing from h to h+k shifts the bias from index h to
   h+k; the old bias weight must be relocated to the new last slot, old unit
   weights stay at 0..h−1, and the k new slots get 0. The new terms then add
   exactly 0.0 into the dot product (x + 0.0 is bit-exact for finite x) and the
   existing terms keep their summation order — output bits unchanged. **Botching
   the bias relocation is the likeliest bug in this phase; the ctest below is
   designed to catch exactly it.**
3. **CGD/Shanno state self-heals across train() calls** — `train()` restarts at
   `iteration = 0` every call (`iterative.cpp:218`), and both `CGD()` and
   `shanno()` reassign `lastG`/`lastF` wholesale at `t == 0`
   (`network.cpp:485-490, 526-531, 579-584, 609-613`) — vector assignment
   resizes. So grow/shrink **between** train() calls needs no manual clearing of
   `stackG`/`lastG`/`lastF` (the sketch said to clear them; do it anyway as
   one-line hygiene, but know it is belt-and-braces, not load-bearing — and say
   so in the comment).
4. **The PlateauDetector is per-run local** (`iterative.cpp:142`) — repeated
   size-trials cannot inherit stale plateau state. `autoStopPatience = 3` is a
   class constant.
5. **OBD needs a test set, and the primitive exists**:
   `Network::sampleTestError( stride )` (`network.cpp:291`) returns the mean
   test error without touching the TwoSet guesses (so no cached-statistics
   invalidation), and returns −1 when there is no 1-output test set — the
   driver's refusal check and its per-size test-error measurement are the same
   call.
6. **Pruning needs no new Matrix primitive.** There is no `excluderows`, but
   `includerows( keep )` (`matrix.h:166`, added 2026-07-16) gathers arbitrary
   rows in arbitrary order — removing hidden unit j is `includerows` with the
   keep-list. Rule 4 is satisfied by the existing layer.
7. **The trial-clone hygiene template is `autoalgo.cpp`**: probe clones get
   `setBootstrapResamples( 0 )` on both TwoSets (`autoalgo.cpp:77-79`) so a
   discarded report never pays for a bootstrap; `Iterative::copy()` nulls the
   observer (`iterative.cpp:61`) so clones can't drive the GUI's buffers;
   `cloneNetwork()` (`netclone.h`) is the typeid clone dispatch. The OBD driver
   snapshots ("best net so far") use `cloneNetwork`; every snapshot inherits
   that hygiene automatically via copy(), **except** bootstrap-disable, which is
   TwoSet state the driver must set once on its working net.
8. **`hO` is clobbered during training** (`oW -= ( hO *= ( eta * o_err ) )`,
   `simpleprop.cpp:480`) — saliency sampling must call `forward()` per exemplar
   and read `hO` immediately, never trust `hO` after a training step. (forward()
   restores `hO[nHidden] = 1` itself.)

**Nothing in this phase touches the train loop, the report, or any printed CLI
byte** — goldens should be byte-identical without re-blessing. If a golden
moves, something is wrong; stop and find out what.

## 2. Engine additions — `src/simpleprop.{h,cpp}`

All three are SimpleProp-only (1 hidden layer is the OBD target; BackProp
multi-layer sizing is out of scope, per the roadmap).

### `void growHidden( const unsigned extra )`
- `assert( builtFlag && weightsSetFlag && extra > 0 )`.
- Snapshot `hW`, `oW`, and `nHidden` (say `oldH`).
- Call `setHidden( oldH + extra )` — reuse the existing sizing logic rather
  than duplicating it.
- Restore: copy the snapshot's rows into `hW` rows 0..oldH−1; fill the new
  rows with small random weights via the layer (build a `Matrix< double >(
  extra, nInput+1 )`, `random( randomLimit )`, copy rows in — do NOT hand-roll
  element loops). `oW[0..oldH−1]` = snapshot, `oW[oldH..oldH+extra−1]` = 0,
  `oW[nHidden]` = the snapshot's **last** element (the relocated bias weight).
- `weightsSetFlag = true`; clear `stackG`/`lastG`/`lastF` (hygiene, see §1.3).
- Why small-random incoming + zero outgoing: the forward pass is unchanged
  (warm start), but gradients flow into the zero outgoing weight immediately,
  and random incoming weights break symmetry if extra > 1.

### `void removeHidden( const vector< unsigned >& v )`
Named to match `removeInputs`. Removes hidden units by index (0..nHidden−1;
the bias pseudo-unit is not addressable).
- `assert` every index < nHidden, and `v.size() < nHidden` (at least one unit
  survives).
- Build the keep-list; `hW = hW.includerows( keep )`; same for `hG`;
  `hWup.resize` to match. Rebuild `oW` keeping kept-unit weights in order and
  the bias weight last; resize `hO`/`oG`/`h_err`; update `nHidden`/`nH`;
  restore `hO[nHidden] = 1`; clear CGD state as above. `weightsSetFlag` stays
  true.
- Removing a unit whose outgoing weight is 0 must be bit-identical on
  forward() — that is the ctest's round-trip check.

### `vector< double > hiddenSaliency()`
- For each training exemplar: `forward( Train, i )`, record `hO[j]` for
  j = 0..nHidden−1.
- saliency_j = `|oW[j]| * Population( samples_j ).std()` — a unit is prunable
  when its output barely moves (std ≈ 0) or barely reaches the output
  (|oW| ≈ 0). Use `Population` (rule 4; note `var()` is the two-pass form since
  legacy bug #6, so identical samples give exactly 0, not NaN).
- Requires a loaded training set and set weights; assert both.

## 3. Engine test — `tests/obd/check_obd.cpp` (ctest `obd_grow_shrink`)

Model it on `tests/binormal/check_az.cpp` (no GSL needed). Small synthetic
1-output discrete dataset, seeded. Assertions, each **watched to fail via
sabotage** (rule 2 — and "fail for the measured reason", per the 2026-07-19
night lesson):

1. **Grow is invisible**: train a 4-hidden net a few hundred iterations, record
   `o` from `forward()` over every train exemplar; `growHidden( 2 )`; outputs
   must be `==` bit-identical (no tolerance). *Sabotage: skip the bias-weight
   relocation (leave it where resize's garbage put it) — must fail this
   assertion and no other.*
2. **Grow-then-remove round-trips**: `removeHidden` of exactly the new units →
   outputs bit-identical to the original again. *Sabotage: off-by-one the
   keep-list.*
3. **Saliency semantics**: after grow, the zero-outgoing units have saliency
   exactly 0 and every original unit's saliency is > 0. *This is the assertion
   that fails if `hiddenSaliency` indexes the bias slot.*
4. **Training still works after both ops**: one `train()` with each of
   trainingType 0/1/2 (a handful of iterations) — finite error, no assert, and
   error decreases from the grow point over a longer type-0 run. This exercises
   the t==0 CGD/Shanno reinit path at the new df().

## 4. The driver — `src/obd.{h,cpp}`, `ObdDriver`

Engine-side, UI-agnostic (RegressNet is the structural precedent: a driver that
owns clones and prints its trail through `util::screen()`).

```cpp
namespace obd {
struct SizeTrial { unsigned hidden; double trainErr, testErr;
                   Iterative::StopReason stop; bool phaseGrow; };
struct Result {
    bool ok = false; string message;       // refusals travel here
    unsigned selectedHidden = 0;
    vector< SizeTrial > history;
    unique_ptr< Network > winner;          // the trained selected net; adopt it
    bool cancelled = false;
};
Result run( DataSet& data, const Config& cfg,
            Iterative::Observer* obs, const atomic< bool >* cancel );
}
```

Config (all API-settable, defaults in parentheses): `hStart` (2), `hMax` (30),
`iterBudget` per size (2000 — the bank lesson: capped, not 1M), `riseTol`
(0.05), `pruneTol` (0.02), `algorithm` (0 = canonical; `auto` runs
`autoalgo::pick` ONCE on the first net and keeps that choice for every size),
plus eta/autostep/seed handled by the caller as for /api/train.

Refusals (return `ok=false`, message text): no 1-output discrete test set
(`sampleTestError` returning −1 is the check); dataset not loaded; hStart < 1
or hMax < hStart.

**GROW phase**: build a SimpleProp at `hStart` (setDataSet → setHidden →
randomize), disable bootstrap on both TwoSets (§1.7). For each size: `train()`
with plateau auto-stop FORCED ON (that is how features 3+4 compose — each size
trial self-terminates) and maxIterations = iterBudget; record
`SizeTrial{ h, finalTrainErr, sampleTestError(1), stopReason, true }`. Track
the best test error and snapshot the best net with `cloneNetwork`. Overlearning
= test error > bestTestErr·(1+riseTol) for **2 consecutive sizes** while train
error is non-increasing → stop growing. Otherwise `growHidden( 1 )` (warm
start — the whole point of §1.2) and continue to hMax.

**PRUNE phase**: from the best snapshot. Loop while nHidden > 1: compute
`hiddenSaliency`, `removeHidden` of the argmin, confirm-retrain with a fraction
of iterBudget (recommend iterBudget/4, floor 100), re-measure test error. If
≤ bestTestErr·(1+pruneTol): accept (this becomes the new candidate), record the
trial, continue. Else: restore the last accepted snapshot and stop.

**Finish**: one full `train()` of the selected net at the user's real
iteration budget with their plateau settings, re-enable the bootstrap
(`setBootstrapResamples` back to the default — read the default from the
TwoSet before zeroing it, don't hardcode 2000), and let the normal
`reportAccuracy` epilogue run. Print the size-vs-error table through
`util::screen()` before the final report:

```
OBD hidden-layer search:
  phase   hidden   train error   test error   stopped by
  grow         2      1.23e-01     1.51e-01   plateau
  ...
Selected: 5 hidden nodes (grew to 8, pruned back 3).
```

**Observer/cancel plumbing**: `run()` takes the caller's Observer and installs
it on whichever net is currently training (so the GUI's realtime chart streams
through every trial), plus the `cancel` atomic checked between trials. A cancel
mid-trial arrives through the observer (STOP_OBSERVER), and the driver then
returns `cancelled = true` with the best-so-far as winner.

## 5. GUI — `POST /api/obd` (+ status + page)

- **Async-only** (the roadmap decision): shares the TrainJob machinery.
  Follow the Phase 2 adoption pattern exactly — the worker re-derives every
  engine pointer inside the job because adoption REPLACES `modelPtr`
  (`gui.cpp:974` onward is the template; the 1b-era lambda-capture dangling bug
  is documented there).
- Params: `hidden_start`, `hidden_max`, `iter_budget`, `rise_tol`, `prune_tol`,
  `algorithm` (1|2|3|auto), `seed`, plus the standard train params that make
  sense (eta/autostep/weight decay), all validated in the handler with explicit
  refusals (asserts vanish in release — the Phase 3 lesson).
- Result JSON: `{ ok, selectedHidden, history:[{hidden,trainErr,testErr,stop,
  phase}], ... }` plus the normal `output`/`roc`/`stats` blocks from the
  adopted net's final report — reuse `jsonROCSeries`/`jsonStatsSet` untouched.
- `GET /api/train/status` gains optional fields while an OBD job runs:
  `{ obd: { phase:"grow"|"prune"|"final", hidden:N } }` — additive, absent for
  plain training (smoke must assert absence on a plain run, presence mid-OBD).
- Every action logged to `neuron_actions.log` with its param values (the
  audit-log rule, `logAction` in gui.cpp).
- **Page**: OBD section — the config fields, a Run OBD button (disabled while
  any job runs; Stop reuses /api/train/stop), and a size-vs-error chart
  (hidden count on x, train + test error on y, grow and prune phases visually
  distinct). **Invoke the `dataviz` skill before writing any chart code**
  (standing constraint), and keep the existing palette so color follows the
  entity. Status line shows "OBD: growing, h=6 …" from the status poll.

## 6. Smoke — `tests/gui/smoke.sh` additions

Each new assert individually **watched to fail** against `git stash push --
src/` per rule 2, and for the measured reason:

1. Happy path on the low-birth-weight fixture smoke already ships: raw load
   with a test split → `/api/obd` (small: hidden_max=5, iter_budget=300, seeded)
   → poll to done → JSON has `selectedHidden` ≥ 1, a `history` whose first
   entry is hidden=2, and real `stats`. Use python for the JSON asserts as the
   existing blocks do.
2. Refusal without a test set (fraction=0 load → obd → `ok:false` with the
   message).
3. Refusal on a continuous outcome, and 409 busy when an obd job is running
   (start async obd, immediate second obd → `busy:true`, then stop).
4. `stop` mid-OBD → `cancelled` result still carries best-so-far
   `selectedHidden` and history.
5. Plain `/api/train` status has NO `obd` field.

## 7. Docs, same commits (rules 1 and 5)

- `docs/gui_cli_parity.md`: OBD is a **GUI-only new feature** — add a row to a
  short "GUI-beyond-CLI" section (create it; DFA graded ROC could seed it) with
  CLI column "— n/a (menus frozen by decision, 2026-07-14)". The matrix's
  contract is GUI ⊇ CLI; a GUI-only feature is legal and should be visibly so.
- `AGENTS.md`: `/api/obd` params + a recipe ("size the hidden layer
  automatically"), the status-poll `obd` field, and the cost note (an OBD run
  is many training runs; iter_budget × sizes is the bill).
- `CLAUDE.md`: status entry (findings-first, per the house style) + tick
  Phase 4 in ROADMAP 2 + note this plan file as superseded once landed.

## 8. Work order

1. Read this plan's §1 citations to confirm nothing drifted (rule 3 — cheap:
   five greps).
2. Engine: `growHidden`/`removeHidden`/`hiddenSaliency` + `check_obd` ctest,
   sabotage each assertion (§3). Commit.
3. Driver: `obd.{h,cpp}` + wire the ctest to drive a tiny end-to-end
   `obd::run` on synthetic data (deterministic with a seed: assert it selects
   SOME size, history is monotone in the grow phase's hidden counts, and a
   cancel mid-run returns cancelled). Commit.
4. GUI endpoint + status + page + smoke (§5-6). Commit.
5. Docs (§7) folded into the commits that change the thing they document.
6. Gates, in the standing order: zero-warning clean build →
   `tests/golden/run_golden.sh` **byte-identical, no re-bless** → ctest 7/7 →
   smoke → `tests/oracle/verify_oracle.sh` → live `--gui` click-through of the
   OBD panel on the bank data if the Chrome extension is connected (otherwise
   note it as the standing browser-verification debt).

## 9. Pre-made decisions and open questions

Pre-decided here so the session doesn't stall (Craig can override any of them):
naming `removeHidden` (matches `removeInputs`); saliency = |oW_j|·std(hO_j)
(the sketch's formula); grow step 1; defaults in §4; `auto` probes once, not
per size (probing every size would spend 3× the budget for information that
rarely changes); the final full-budget train after selection (a sized-but-
half-trained net would be a strange deliverable).

Genuinely open for Craig, flag when reporting: (a) should the OBD table also
print per-size classification accuracy (cheap, `classAccuracy` exists)?
(b) is hMax=30 the right ceiling, or scale it with input count?
