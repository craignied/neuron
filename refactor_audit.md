# DRY / architecture audit — neuron-3.0 engine

**Status: part proposal, part implementation record.** Prepared 2026-07-31 as a
proposal; from §8.7's list onward, most of it has shipped. §8.7 carries the
authoritative per-item state with commit hashes — read that, not this line, for
what is done. The findings in §§1–7 are preserved **as originally written**, so
that what was proposed can be compared with what was built; where a finding has
since shipped or been corrected, §8 and §8.7 govern.

**Legacy bug #12 (§11.11) is FIXED** — `BackProp` discarded the conjugate
direction in batch/epoch mode, so CGD and Shanno were silently plain gradient
descent there. Diagnosed, escalated, and corrected as a standalone commit on
2026-08-01, guarded by `tests/backprop/check_bpoptimizer.cpp`.

> **REVISED after Sol's review, 2026-07-31.** Read **§8** before acting on anything below.
> It corrects finding 1.1 (which overstated equivalence), corrects §7's verification protocol
> (which was the wrong test for behavior-preserving work), records one claim in Sol's review
> that I checked and found **incorrect**, adds three findings neither of us had, and gives the
> agreed commit order. Where §8 and the body disagree, §8 governs.
> Standing rule 7 — the missing efficiency rule this audit's §0 recommended — **is now in
> `CLAUDE.md`** (commit `02870fd`), so §0 is history rather than a proposal.

Scope: every file in `src/` (26,965 lines), read in full or targeted-read. The brief was
Craig's: the architecture exists so that (a) the code reads like the matrix notation in the
paper it came from, and (b) a good compiler can make the fastest possible binary — and
therefore the code must be as DRY as reasonably possible, with `Matrix` / `vector_ops` /
`Population` used wherever matrix math is called for.

Each finding gives its **location**, the **measurement** that establishes it, the **proposed
change**, and the **risk**. Findings are ranked by lines removed × confidence. Section 6 lists
things I considered and deliberately rejected, so the reasoning is on the record.

Two rules govern every item here and are restated once rather than repeated:

- **Rule 6 (layer ownership).** Extraction goes to the *lowest class-layer boundary that
  naturally owns the mechanism* — never into a generic god object, never collapsing distinct
  polymorphic responsibilities. Per-class model math is not a consolidation target.
- **Behavior-preserving first.** Every item below is an extraction that must leave the three
  golden transcripts, all configured tests, `tests/gui/smoke.sh` and
  `tests/oracle/verify_oracle.sh` byte-identical. Where an item *cannot* be byte-identical,
  it says so.

---

## 0. First: what the docs actually say about this (Craig's question)

> *"I'm curious how much of that is made very clear in CLAUDE.md and related documents that
> Claude would read before coding."*

**Readability: stated, and stated well.** `CLAUDE.md` standing rule 4 says it outright — the
computational core was built on `Matrix` / `vector_ops` / `Population` "so that the code reads
like the matrix notation in the paper it came from — you can go from the page to the code and
back", plus the bounds-checking argument, plus the instruction to *extend the layer* rather
than drop to raw arrays, plus legacy bug #8 as the price of not doing it. That rule works; it
is why the engine still reads the way it does.

**DRY: stated, and interpreted precisely.** Rule 6 plus `docs/cv_refactoring_architecture.md`
give the operating definition (one authoritative implementation of each mechanism, in the class
that conceptually owns it; dependencies point downward). That is the right definition and it is
the one this audit applies.

**Efficiency: essentially NOT stated.** This is the gap. Searching `CLAUDE.md` and `AGENTS.md`
for the principle finds only:

- rule 4's *cautionary anecdote* — the O(n²) ROC threshold sweep that made the bootstrap a
  scale cliff — which teaches "leaving the layer cost performance twice", not "speed is a
  design constraint";
- a passing "the efficient index-gather rewrite" in the ROADMAP 4 history.

Nowhere does either file say what Craig said above: **the engine is C++ because neural
computation is compute-bound, and code must be written so a good optimizer can do its job.**
`AGENTS.md` has no coding-style or architecture section at all — its five headings are all
operational recipes (build, groom, train, deploy, verify, repo map).

Worse, the *specific idioms* that implement the efficiency principle are documented only in
scattered comments inside `vector_ops.h` and the model bodies, where nobody reads them before
writing code:

| Idiom | Where it is explained today |
|---|---|
| Prefer the unary (`*=`, `+=`) operators; the binary ones copy the LHS | a comment at `vector_ops.h:88-91` and again at `:179-182` |
| Prefer the overload that takes the **output object as an argument**; the value-returning form allocates | ~20 repetitions of "Use previously coded X method" in `matrix.h`, never stated as a rule |
| Range overloads (`func(v,f,out,a,b)`, `dotprod(v,out,a,b)`) exist to avoid touching bias slots without a second pass | a comment in `simpleprop.cpp:589-592` |
| Accumulate outside the exemplar loop, multiply by η once at the end | `simpleprop.cpp:534-536` |

**Recommendation (independent of any code change below).** Add a short standing rule 7 to
`CLAUDE.md` — *"The engine is compiled because speed is a requirement"* — stating the three
idioms above as the house style, and cross-reference it from the manifest's new Part I. Cost:
about 25 lines. Value: it is the half of the architecture that a new contributor (human or
model) currently has to infer from comments, and several findings below exist precisely because
it was inferred wrong.

---

## 1. Major structural duplication

### 1.1 `SimpleProp` and `BareProp` share most of their mechanisms — *superseded by §8.2*

> **This heading originally read "are the same class twice". That overstated it, and the
> proposal below is not the one to implement.** The measurements are sound; the conclusion
> drawn from them was too broad. See **§8.2** for the corrected, method-by-method position.

**Location:** `src/simpleprop.{h,cpp}` (830 + 113 lines), `src/bareprop.{h,cpp}` (619 + 86).

**Measurement.** I diffed the two bodies with whitespace and comments discounted:

- `trainSet()` — `simpleprop.cpp:443-516` vs `bareprop.cpp:471-545`. **76 lines, differing in
  nothing but the class name and stray whitespace** (`diff` reports 18 lines, every one of them
  a `SimpleProp::` / `BareProp::` token or a space).
- `innerTrainSet()` — `simpleprop.cpp:520-734` vs `bareprop.cpp:283-467`. ~185 lines. Executable
  difference: **one line.**

  ```cpp
  ( func( hO, d_sigmoidal(), h_err, 0, nH ) *= o_err ) *= oW;   // SimpleProp
  ( func( hO, d_sigmoidal(), h_err )        *= o_err ) *= oW;   // BareProp
  ```

  And `nH` is defined as `nHidden - 1` (`simpleprop.cpp:115`). For BareProp, `h_err.size() ==
  hO.size() == nHidden`, so the ranged form `func( hO, d_sigmoidal(), h_err, 0, nHidden - 1 )`
  transforms exactly the whole vector and asserts cleanly (`vector_ops.h:329-340`: needs
  `a <= b && b < vec_out.size()`; `0 <= nHidden-1 < nHidden` ✓). **The two lines are the same
  formula.** With `nH` replaced by `nHidden - 1`, `innerTrainSet` is shareable with *zero*
  parameterization.
- `forward()` — same story. `hW.dotprod( I, hO, 0, nHidden-1 )` satisfies the ranged assert for
  BareProp too (`matrix.h:1427`: `nrows_ == iEnd-iBegin+1` → `nHidden == nHidden` ✓). The one
  genuinely SimpleProp-only statement is `hO[ nHidden ] = 1;` (restore the bias slot), which is
  out of bounds for BareProp — and `Network::biasFlag` already distinguishes them correctly
  (set in each constructor).
- `randomize()`, `save()`, `pack()`, `outputHeader()` — identical modulo one string.
- `setHidden()`, `setDataSet()`, `df()`, `load()`, `removeInputs()`, `unpack()` — identical
  modulo `+ 1` on the input width and the presence of the bias column/slot.

**Proposal.** Introduce a shared implementation base — `OneHiddenNet` (or `PropNet`) — between
`Network` and the two concrete classes. It owns `nHidden`, `hW`, `hWup`, `hG`, `hO`, `oW`,
`h_err`, `oG`, `o_err`, and implements *all* of the above once, with the two bias-dependent
facts expressed as `biasFlag` (already a `Network` member) rather than as duplicated bodies:

- input width `nInput + ( biasFlag ? 1 : 0 )`,
- `hO` sized `nHidden + ( biasFlag ? 1 : 0 )` with the pinned bias slot guarded by `if ( biasFlag )`,
- `df()` returns the class's own formula (this stays virtual — it is real per-class math, not
  duplication).

`SimpleProp` and `BareProp` **remain distinct classes**, each reduced to a constructor that sets
`objType` and `biasFlag`, plus `df()`. That is deliberate and non-negotiable: `typeid` dispatch
in `netclone.cpp`, `modelfactory::createByTypeName`, the first line of every saved model file,
and the golden transcripts all depend on the two types existing separately. Merging them into
one class would be a *format* change; merging their bodies is not.

`growHidden` / `removeHidden` / `hiddenSaliency` (the OBD sizing methods, currently SimpleProp-only)
move to the base as well, guarded the same way — which incidentally makes OBD available to
BareProp, a capability gain, but that should land as a **separate commit after** the
behavior-preserving extraction, not inside it.

**Net reduction:** ~450 lines.
**Risk:** medium-low mechanically, but this touches the hottest loop in the engine. Byte-identity
is provable: the goldens `xor_seed42` and `regress_seed42` both exercise these paths.
**Order:** do this first — it removes one of the four copies in finding 1.2 for free.

### 1.2 Four copies of the automatic-stepsize wrapper (`trainSet`)

**Location:** `simpleprop.cpp:443`, `bareprop.cpp:471`, `backprop.cpp:327`, `logistic.cpp:357`.

**Measurement.** All four are the same ~50-line algorithm: buffer the weights and the CGD/Shanno
state (`lastG`, `lastF`), loop up to `maxLoops` shrinking η by `gamma` until the error difference
falls below `deltaError`, restore the buffered weights, then run the real `innerTrainSet()`. The
only thing that varies is **which containers get buffered**:

| Class | Buffered |
|---|---|
| SimpleProp / BareProp | `hW`, `oW` |
| BackProp | `Weights` (vector of Matrix) |
| Logistic | `W` |

There is one *apparent* behavioral difference: Logistic guards on `automaticStepSizeFlag` alone,
the other three on `batchEpochFlag && automaticStepSizeFlag`. **Verified equivalent** for the
shipped paths — Logistic sets `setBatchEpoch(true)` in its constructor (`logistic.cpp:13`), the
CLI refuses to turn it off (`neuron.cpp:1005-1010`), and the GUI refuses too
(`gui.cpp:1302-1304`). The shared implementation should use the stricter
`batchEpochFlag && automaticStepSizeFlag`; that changes behavior only in a state neither
interface can reach, and Sol should confirm that reading rather than take mine.

**Proposal.** Template method on `Network`:

```cpp
// Network (non-virtual): the automatic-stepsize search, once.
double Network::trainSet();
// New protected pure virtuals, each 2-4 lines per concrete class:
virtual void saveWeights() = 0;
virtual void restoreWeights() = 0;
```

`lastG` / `lastF` are already `Network` members, so the base buffers those itself.

**Net reduction:** ~230 lines before finding 1.1, ~155 after.
**Risk:** low. Pure motion; no arithmetic moves.

### 1.3 `RegressNet::forward_regress()` and `reverse_regress()`

**Location:** `regressnet.cpp:270-477` and `:480-699`. ~210 lines each.

**Measurement.** Identical skeletons: outer pass loop → inner candidate loop → `copy_network()`
→ `removeInputs()` → `announce("training candidate")` → `train()` → `announce("candidate
complete")` → `recordCandidate()` → `requireConvergedFit()` → `Wilks()` → `pX2()` → update the
best-so-far → `report()`. The genuine differences are five:

1. `removed` vs `added` bookkeeping (same vector, opposite meaning);
2. forward removes the *complement* of the admitted set (`set_difference`, `regressnet.cpp:568-576`);
3. `largest_p` / `p > largest_p` vs `smallest_p` / `p <= smallest_p`;
4. `Wilks( N, lastError, subError )` vs `Wilks( N, fullError, lastError )` — argument order;
5. the stop test `largest_p < threshold` vs `smallest_p > threshold`, and the wording.

Everything else — including the two carefully-commented invariants (*record the candidate
BEFORE the eligibility check that throws*, and *maintain `finalVariables` per pass so an early
exit still reports what completed*) — is written out twice. That is the specific danger: those
two invariants were established by a Sol review and are now maintained in two places.

**Proposal.** One private `void stepwise( Direction dir )` with a small `Direction` descriptor
carrying the five differences (a comparator, the verb strings, and the Wilks argument order).
Keep `forward_regress()` / `reverse_regress()` as two-line public entry points — the public API
and the report wording do not change.

**Net reduction:** ~180 lines.
**Risk:** medium. `regress_seed42` covers the reverse direction only; forward is covered by the
GUI walkthrough and by `smoke.sh`. Confirm forward coverage before trusting the goldens here —
this is exactly the hole standing rule 2 exists to catch, and legacy bug #11 (the
`computeCondNum` null dereference) survived for the same reason: *"every test and golden ran
only the REVERSE direction"* (`network.cpp:702-706`).

### 1.4 `DataSet::metricsReport()` — the training and test blocks are one function copied

**Location:** `dataset.cpp:1380-1466` (training) and `:1469-1553` (test).

**Measurement.** ~85 lines each. Difference: `TrainTwoSet`/`TestTwoSet`, the heading string
(`"Training set:"` / `"Test set:"`), its underline, and two label words. Everything else —
four `try`/`catch( DivisionByZero )` blocks, the classification table, the ROC report, K-S,
the Pearson-with-no-p line and its comment, Hosmer-Lemeshow — is verbatim.

**Proposal.** One private helper:

```cpp
void DataSet::metricsFor( ostream&, TwoSet&, const string& heading );
```

called twice. The Pearson comment ("a statistic, never a p") then exists once, which matters:
it is a settled decision and it is currently written in two places that could drift.

**Net reduction:** ~85 lines.
**Risk:** very low — pure output formatting, and the goldens pin the exact bytes.

---

## 2. Medium duplication

### 2.1 `DataSet::setRawMatrix` / `setTrainMatrix` / `setTestMatrix`

**Location:** `dataset.cpp:95-159`, `:178-218`, `:395-435`.

Three ~40-line methods with the same four-step shape: assert dimensions → check column count →
check discreteness → assign, set flag, log to history. The only variations are the target
Matrix/flag and the noun in the message ("raw dataset" / "training set" / "test set").
`setRawMatrix` additionally clears the three derived sets.

**Proposal.** A private `bool acceptMatrix( Matrix<double>& dest, Matrix<double>& in, const
char* noun )` doing the validation and assignment; each public setter becomes ~6 lines
(call it, set its own flag, do its own extra work). **~60 lines.** Very low risk.

### 2.2 `DataSet::saveTrain` / `saveTest` / `saveScales`, and `TwoSet::save`

**Location:** `dataset.cpp:222-257`, `:439-474`, `:261-315`; `twoset.cpp:135-165`.

Same open-truncate → `is_open()` check → write → screen message → close → history-log →
`return success` scaffold, four times (five if you count the model `save()` methods in 2.3).

**Proposal.** One private `bool DataSet::writeToFile( const string& filename, const char* what,
function< void( ostream& ) > body )`. **~50 lines.** Low risk.

### 2.3 `save()` / `load()` scaffolds across the four model types

**Location:** `simpleprop.cpp:294/342`, `bareprop.cpp:134/182`, `backprop.cpp:754/832`,
`logistic.cpp:138/188`.

Every `save()` is: open truncate → `is_open()` guard with the same message → `outputHeader()` →
*type-specific weight writing* → success message → close → history log. Every `load()` is:
`getGoodFile` → open → read and assert the type line → read and compare `nInput` with the same
three-line refusal → *type-specific weight reading* → `weightsSetFlag = true` → echo header →
history log.

**Proposal.** Two protected `Network` helpers taking the type-specific part as a callback:
`saveWithHeader(...)` / `loadWithHeader(...)`. Note that after finding 1.1 only three copies
remain. **~120 lines.** Low risk, but the file formats are pinned by
`tests/oracle/verify_oracle.sh` — byte-identity is mandatory and is testable.

### 2.4 `LDFA::train` / `QDFA::train` — the reporting scaffold

**Location:** `ldfa.cpp:32-98`, `qdfa.cpp:39-112`.

**Measurement.** 67 vs 74 lines; 35 lines differ, i.e. **roughly half of each is shared
boilerplate**: the two `ostringstream`s, the `"I'm running X:"` line, `outputHeader()`, the
`catch ( Matrix<double>::Singular& )`, `addHistory()`, the last-operation file write, and
`return -1`.

**Proposal.** `DFA::train()` becomes a template method owning the scaffold and calling a new
protected pure virtual `fitDiscriminant()` (the ~15 lines that are genuinely LDFA's or QDFA's
math). The per-class discriminant math stays per-class — that is intentional polymorphism, not
duplication. **~55 lines.**

### 2.5 `LDFA::reportAccuracy` / `QDFA::reportAccuracy`

**Location:** `ldfa.cpp:101-212`, `qdfa.cpp:115-225`.

**Measurement.** The multi-output half (66 vs 68 lines) differs in **14 lines**: the discriminant
expression and `max_element` vs `min_element`. The 1-output half differs only in the expression
and the sign of the margin fed to `sigmoidal()`.

**Proposal.** Move the loop structure to `DFA::reportAccuracy()`, with two protected virtuals:
`double discriminant( unsigned classIndex )` and `bool largerWins()`. **~55 lines.** The
carefully-argued "graded class-1 score, not a hard 0/1 decision" comment then lives once.

### 2.6 `Iterative::train()` — six copies of the stop-and-announce block

**Location:** `iterative.cpp:352-518`.

Each of the six exits (min error, change, window, gradmax, plateau, observer) is the same
12-line shape:

```cpp
screenStream.str( "" );
screenStream << <the message> << endl;
if ( !quietFlag ) { fileStream << screenStream.str(); util::screen() << screenStream.str(); }
stopReason = STOP_X;
break;
```

including the same four-line comment about quiet runs, repeated verbatim six times.

**Proposal.** A private `void announceStop( StopReason, const string& message, ostringstream&
screenStream, ostringstream& fileStream )`; each site becomes two lines. **~70 lines.** Low
risk; the goldens pin every one of these messages.

### 2.7 `Model` should own the last-operation file write

**Location:** `iterative.cpp:592-605`, `ldfa.cpp:82-95`, `qdfa.cpp:96-109` — three identical
15-line blocks (`run_path` → open truncate → `is_open` guard → write → close).

**Proposal.** `bool Model::writeLastop( const string& text )`, beside the existing
`Model::addHistory`. That is the class that owns `lastopFlag` and `lastopFilename`, so it is the
correct owner (rule 6). **~35 lines.**

### 2.8 `TwoSet` rate getters and confusion-count getters

**Location:** `twoset.cpp:305-425`.

`getClassAcc` / `getSens` / `getSpec` / `getPVP` / `getPVN` are five copies of *"if no threshold,
print a message and return 0; else `calculate(threshold)` and return a ratio, throwing
`DivisionByZero` on a zero denominator"*. `getTP` / `getTN` / `getFP` / `getFN` are four copies of
*"if no threshold, throw; else `calculate(threshold)` and return the member"*.

Note the existing latent oddity worth fixing while you are here: the ratio getters compute
`result` **before** testing the denominator for zero (`twoset.cpp:330-332` and its three
siblings), so the division happens first and the throw is a formality after the fact. Harmless
today, but backwards.

**Proposal.** One private `double rate( unsigned num, unsigned den, const char* what )`, and one
private `unsigned count( unsigned TwoSet::*member, const char* what )`. **~50 lines.**

### 2.9 The Hanley-McNeil trapezoidal-CI block is written twice inside one function

**Location:** `twoset.cpp:892-899` and `:926-933` — inside `ROCarea()`.

Eight lines, verbatim, including the `[0,1]` clamp. A caller reading this cannot tell that the
two branches are the same computation.

**Proposal.** A private `void reportTrapCI( ostream& )`. **~10 lines**, but the value is that the
clamp policy (which is a settled decision — the clamp applies here and deliberately does *not*
apply to the clustered interval) then exists once.

### 2.10 GUI: four copies of the async-job launch

**Location:** `gui.cpp:1446-1469` (train), `:1758-1776` (OBD), `:2854-2874` (CV),
`:3084-3110` (regress).

Each is: `if ( job.worker.joinable() ) join()` → lock → `resetForNewRun()` → `cancel = false`
→ `running = true` → `job.worker = thread( [...]{ result = runXJob(...); lock; job.result =
result; running = false; } )`. The comment *"publish BEFORE running goes false"* — which is the
whole synchronization contract — is repeated four times.

**Proposal.** One `void launchJob( function< string() > run )`. The publish-then-clear ordering
then has exactly one implementation, which is what it deserves. **~45 lines.**

### 2.11 GUI: B9 (already on the roadmap) is the same finding

`gui.cpp` calls `atol`/`atof` **51 times**, and `handleObd` defines *local lambdas*
`uintParam` / `fracParam` (`gui.cpp:1703-1719`) that no other handler can reach — while
`handleTrain` open-codes the same validation eight times (`:1349-1365`) and the same
present-only-apply shape four times (`:1412-1435`).

This audit adds nothing new to **ROADMAP 4 item B9** except the observation that B9 is
*also* the largest DRY item in `gui.cpp`, and that the parser should live in the class layer
(`util::`), not in `gui.cpp`, so the CLI and any future caller share it. Do B9 as specified;
promote those two lambdas into it.

---

## 3. Small / mechanical — the "one line changed, many deleted" kind

### 3.1 `RegressNet::network_name()` is `Model::getType()`

**Location:** `regressnet.cpp:251-267`.

Seventeen lines of `typeid` dispatch returning `"BareProp"` / `"SimpleProp"` / `"BackProp"` /
`"Binary logistic"` / `"unknown"`. Those are **exactly** the `objType` strings the four
constructors set (`simpleprop.cpp:11`, `bareprop.cpp:11`, `backprop.h:13`, `logistic.cpp:10`),
and `Model::getType()` already returns `objType` (`model.h:27`).

**Proposal.** `return netPtr->getType();`. **17 lines → 1.** A future Network type then reports
its real name instead of `"unknown"`, which is strictly better. Zero risk — the string is used
only in a report line.

### 3.2 Same dispatch, third copy, in the GUI

**Location:** `gui.cpp:1019-1020`.

```cpp
string kind = dynamic_cast< SimpleProp* >( modelPtr.get() ) ? "SimpleProp"
    : dynamic_cast< BareProp* >( modelPtr.get() ) ? "BareProp" : "BackProp";
```

→ `modelPtr->getType()`. **2 lines → 1**, and it stops being wrong the moment a fifth type
exists.

### 3.3 `BackProp::trainSet` rebuilds a buffer the assignment operator already builds

**Location:** `backprop.cpp:333-372`.

Forty lines that (a) branch on bias, (b) resize `WeightsBuffer` layer by layer with the boundary
cases spelled out twice, and then (c) copy element by element — to achieve what
`Matrix::operator=` already does, since it resizes when the dimensions differ
(`matrix.h:765-780`).

**Proposal.**

```cpp
vector< Matrix< double > > WeightsBuffer = Weights;   // replaces lines 333-372
...
Weights = WeightsBuffer;                              // replaces lines 419-420
```

**~35 lines → 2**, and `nInput` / `nOutput` become unused locals. This is subsumed by finding
1.2 (the shared `saveWeights`/`restoreWeights` becomes one line for BackProp), but is worth
noting on its own because it is the clearest single instance of hand-rolling what the class
layer already provides — rule 4's exact failure mode, in the direction of verbosity rather than
raw arrays.

### 3.4 `SimpleProp::nH` is a stored copy of a derived value

`nH` is set to `nHidden - 1` in `setHidden` (`simpleprop.cpp:115`), maintained in
`removeHidden` (`:216`), and copied in `copy()` (`:35`). It is never anything else. It is a
member that must be kept in sync for no reason. Delete it and write `nHidden - 1`; this is a
precondition for finding 1.1 anyway.

### 3.5 `stats::sqr` is defined twice

`stats::sqr` (`stats.h:111`) and `XY::sqr` (`stats.h:251`) are the same inline. `XY` should use
`stats::sqr`. Trivial, but it is two definitions of `x*x`.

### 3.6 `#ifdef STATS_DEBUG` blocks

`stats.cpp` contains ~30 copies of

```cpp
#ifdef STATS_DEBUG
    util::screen() << "In stats::<name>" << endl;
#endif
```

**Proposal.** One macro (`STATS_TRACE( name )`) defined once at the top, expanding to nothing
when the switch is off. **~90 lines → ~30 + 5.** Purely cosmetic; low priority. *(Note: these
blocks reference `util::screen()` but `stats.cpp` does not include `utility.h` — the debug
switch does not currently compile. Worth either fixing or deleting outright.)*

### 3.7 `util::askI` / `askD` overload pairs

`utility.cpp:158-218` and `:223-270`: the bounded and lower-bounded forms are the same loop
with a different message. Minor (~40 lines), and the CLI menus are frozen, so the *messages*
must not change. Low priority; listed for completeness.

---

## 4. Efficiency findings (in scope per the brief, not DRY)

These are cases where the code as written prevents the compiler from doing its job, or does
avoidable work. Each is a small diff.

### 4.1 `errorFunction`'s multi-output constructor takes three vectors **by value** — *fix this*

**Location:** `function_defs.h:121-122`.

```cpp
errorFunction( vector< double > y, vector< double > o, vector< double > x, bool errorType )
```

Three heap allocations and three copies **per exemplar, per iteration**. It is called from
`BackProp::innerTrainSet` (`backprop.cpp:468`) — the inner training loop — and from
`Network::reportAccuracy`'s multi-output paths (`network.cpp:212, 261`).

**Proposal.** `const vector< double >&` for all three. The body needs `const_iterator` instead
of `iterator`; the LMS branch (`vector<double> errors = y - o;`) still compiles because
`operator-` takes its LHS by value already (`vector_ops.h:103`). **One word per parameter.**
Zero behavioral change; measurable on any multi-output run.

### 4.2 `errorFunction::multOutput` is never initialized

**Location:** `function_defs.h:180`, read at `:187`.

Neither real constructor sets `multOutput`; `copy()` reads `rhs.multOutput`. Nothing depends on
its value, so there is no live bug — but it is an indeterminate read, and it is the same shape
as the uninitialized `Model::errorType` scalar that caused the nested-OBD nondeterminism hunt
(a settled decision in `CLAUDE.md`, and the reason `Iterative::copy` writes
`quietFlag = false` explicitly). Initialize it or delete it.

### 4.3 `Matrix::t()` and `Matrix::dotprod( Matrix, Matrix )` allocate per column

**Location:** `matrix.h:1337-1348` and `:1667-1678`.

```cpp
for ( unsigned c = 0; c < ncols_; c++ )
    M_in.replacerow( c, this->col( c ) );          // t(): one vector<T> allocated per column

for ( unsigned col = 0; col < B.cols(); col++ )
    C.replacecol( col, this->dotprod( B.col( col ) ) );   // dotprod: two per column
```

Both are used together in `Network::computeCondNum` (`network.cpp:690-691`), where `grads` is
`df × nTrain`. On a 12,000-row dataset that is 12,000 vector allocations in `t()` alone, plus a
column-strided (cache-hostile) product.

**Proposal.** Add in-place forms inside the class — a direct element loop for `t()`, and an
`ikj`-ordered triple loop for the Matrix-Matrix product. Both stay in the class layer, keep the
same signatures, and read no worse. This is the same category as the ROC threshold-sweep
reformulation (rule 4's own precedent): the notation is preserved and the cliff goes away.

**Caveat, stated plainly:** I have *not* measured this. Per standing rule 3 it is a hypothesis
until someone profiles `computeCondNum` on a large dataset. `./build/scale_probe` is the tool.
Do not act on this item on the strength of my reading it.

### 4.4 Six non-RAII `set_screen` save/restore pairs — an exception-safety hole, not just DRY

**Location:** `autoalgo.cpp:110/119`, `cvadapters.cpp:52/54` and `:294/296`,
`obd.cpp:190/192` and `:325/327`.

Each is:

```cpp
ostream& saved = util::screen();
ostringstream discard;
util::set_screen( discard );
... m.train() / obd::run() ...          // <-- can throw
util::set_screen( saved );
```

If the guarded call throws — and `train()` can throw `Matrix::BoundsViolation`,
`stats::statsErr`, `RegressNetErr` — the restore never runs, and the engine's output stream is
left pointing at an `ostringstream` that is about to be destroyed. Every subsequent engine print
in that process writes through a dangling reference.

`gui.cpp` has the RAII version (`struct Capture`, `gui.cpp:519-524`) but it restores to `cout`
unconditionally rather than to the previous stream, so it cannot nest correctly either.

**Proposal.** One `util::ScreenCapture` RAII guard in the class layer that saves and restores
*the previous stream*; replace all seven sites. **~15 lines removed, one real fault class
closed.** This is the highest-value item in section 4.

### 4.5 `Population` copies its dataset

**Location:** `stats.h:167`, `stats.cpp:583-602` — `Population( vector<double>& )` stores
`x = v_in`.

Every construction copies the whole vector. Called per hidden unit in
`SimpleProp::hiddenSaliency` (`simpleprop.cpp:257`), per bootstrap interval
(`twoset.cpp:781`), and twice inside the `XY` fitexy constructor over the full x and y vectors
(`stats.cpp:923`). Holding a `const vector<double>&` would remove the copies — at the cost of a
lifetime contract the current class does not have. **Judgment call; flagging, not recommending.**
The related API wart is cheaper and unambiguous: `stats::ttest`/`tutest`/`tptest`/`ftest` take
`vector<double>&` (non-const) for read-only arguments, so no caller can pass a temporary or a
const vector — make them `const&`.

---

## 5. Suggested sequence

Behavior-preserving extraction first, in dependency order, one commit each:

1. **3.1, 3.2, 3.4, 3.5** — the one-liners. Zero risk, immediate proof the gates are green.
2. **4.4** — `util::ScreenCapture`. Fixes a real fault; small.
3. **4.1, 4.2** — `errorFunction` signatures and initialization.
4. **2.6, 2.7** — `Iterative::announceStop`, `Model::writeLastop`.
5. **1.1** — the `OneHiddenNet` base. The big one; do it alone.
6. **1.2** (now three copies), **3.3** — the `trainSet` template method.
7. **2.1, 2.2, 2.3, 1.4, 2.8, 2.9** — the `DataSet` / `TwoSet` / model-file scaffolds.
8. **2.4, 2.5** — the DFA template methods.
9. **1.3** — stepwise. *Only after* confirming the forward direction is actually covered.
10. **2.10**, then **B9** (which subsumes 2.11).

---

## 6. Considered and rejected — with reasons

These would show up in a naive duplication scan. They are not defects.

- **The paired `matrix.h` overloads** (`submatrix`, `row`, `col`, `addrow`, `addcol`,
  `includecols`, `excludecols`, `includerows`, `dotprod`, `dotprodt`, `dotprod_row`, `t`,
  `inverse`, `colsums`, `toVector` — about twenty pairs, each a value-returning form that
  constructs a result and delegates to the reference-taking form in three lines). This is **the
  efficiency idiom itself**, and the delegation is already minimal. Collapsing it would destroy
  the ability to reuse a buffer in a loop. Leave it exactly as it is.

- **`matrix.cpp`'s explicit `Matrix<double>` specializations** (`random`, `covariance`,
  `inverse`, `ludcmp`, `lubksb`, `determinant`). These are `double`-only by nature; the
  `inverseGaussJordan` / `inverseLU` pair are two *different algorithms*, not a duplication.

- **Per-model `innerTrainSet` math** (SimpleProp/BareProp aside, which is finding 1.1).
  `BackProp` and `Logistic` compute genuinely different gradients. Rule 6 names this explicitly:
  per-class model math is not a consolidation target.

- **`Network::reportAccuracy`'s train/test blocks** (`network.cpp:137-193`). These *look* like
  finding 1.4, but the 1-output and multi-output halves differ structurally and the whole
  function is pinned byte-for-byte by three goldens and the oracle. The gain (~40 lines) does
  not justify the risk against `verify_oracle.sh`. Revisit only if it is being edited anyway.

- **`cloneNetwork` and `modelfactory::createByTypeName`** — inverse maps (type → object, name →
  object), both irreducibly needing the concrete types. Not duplication.

- **`crossval`, `cvadapters`, `evaldesign`, `auccov`, `clustered_auc`, `split`** — the ROADMAP 4
  layer. I found **no** duplication worth acting on. `strataKey` and `groupKey`
  (`dataset.cpp:1101, 1245`) both densify tuples through a `map`, and `splitDiagnostic` /
  `groupDiagnostic` both compute the train/test outcome-1 rate — about 20 lines between them.
  A shared helper is arguably right, but these are the two functions whose *differences* (binned
  vs exact matching; one prints strata, the other groups) are the settled distinction between
  stratification and grouping. I would rather they read separately than be unified and later be
  misread as substitutes.

- **The CLI menu bodies** (`neuron.cpp`). The train-set/test-set column repetition in
  `specify_ROC` and friends is real, but the menus are **frozen** by settled decision and the
  transcripts are golden-pinned. Not worth touching.

---

## 7. Verification protocol for any of this

Non-negotiable, per `CLAUDE.md` rules 2 and 6, for **every** commit above:

1. Zero-warning Release build.
2. `tests/golden/run_golden.sh` — **byte-identical**, all three transcripts. Any diff means the
   extraction changed behavior; read it, do not bless it.
3. `ctest` — all configured tests. (Do not write the count here: it changes with
   every batch of work. `ctest -N` answers it.)
4. `tests/gui/smoke.sh`.
5. `tests/oracle/verify_oracle.sh` — numerically identical.
6. **Prove the guard.** For each item, before believing the goldens cover it: `git stash push --
   src/`, rebuild, and confirm the test *fails* against the old code path — the practice rule 2
   exists for. Finding 1.3 in particular is where I expect this to reveal a hole.

None of these items adds a GUI control, so no click-through is required for any of them.

---

---

## 8. Revision after Sol's review (2026-07-31)

Sol reviewed the audit above. Most of it is accepted. This section records what changed, what
I checked and found wrong in the review itself, what neither of us had, and the plan to follow.

### 8.1 Accepted corrections to the audit

| # | Correction | Verdict |
|---|---|---|
| 1 | **Finding 1.1's conclusion was too broad.** "The same class twice" is not what the measurements support; see §8.2. | Accepted, corrected in place. |
| 2 | **§7's `git stash` proof was the wrong test.** A behavior-preserving extraction *should* pass the same tests as the code it replaces — requiring it to fail against the old binary contradicts the goal. Standing rule 2's real requirement is to sabotage the **new characterization tests** to prove they guard something, then restore. | Accepted; §8.4 replaces §7. |
| 3 | **`biasFlag` is publicly mutable** (`network.h:33`), so it is not a safe discriminator for object layout. *Verified:* `setBias` has exactly two callers, both in `modelfactory` and both before `setDataSet`, so no current path abuses it — but the type contract does not prevent it, and Sol is right that a shared base must fix its bias architecture at construction. | Accepted. |
| 4 | **Don't move OBD (`growHidden`/`removeHidden`/`hiddenSaliency`) to a shared base.** It would advertise BareProp OBD support that nothing tests. | Accepted — it stays SimpleProp-only; adding BareProp OBD is a feature decision, not a refactor. |
| 5 | **Don't silently retighten Logistic's auto-stepsize guard** from `automaticStepSizeFlag` to `batchEpochFlag && automaticStepSizeFlag` inside a behavior-preserving commit. My audit called the states "unreachable from today's interfaces", which is a statement about callers, not about the class contract. | Accepted — preserve every programmatic state; see §8.5. |
| 6 | **Don't reduce the DFA discriminants to boolean switches** (`largerWins()`). | Accepted — this is exactly what new standing rule 7 now forbids. My §2.5 proposal was wrong on its own stated principle. |
| 7 | **Don't use `std::function` for cold-path file-writing boilerplate** (my §2.2). Use typed helpers local to the owning class. | Accepted. |

### 8.2 The corrected `SimpleProp` / `BareProp` position — the equivalence table Sol asked for

Every pair below was diffed with comments and whitespace discounted. Categories are Sol's.

**A — identical mechanism (differs only by the class name in the signature):**

| Method | Evidence |
|---|---|
| `trainSet()` | 76 lines; `diff` shows only the class name and stray spaces |
| `randomize()` | 20 lines; only the class name — the message already reads `objType` |
| `save()` | 45 lines; only the class name |
| `load()` | 74 lines; only the class name and **two comments** describing the header line being skipped |
| `pack()` | identical |
| copy ctor / `operator=` | identical shape |

**B — one mechanism, bias-dependent dimensions:**

| Method | The bias dependency |
|---|---|
| `innerTrainSet()` | **one line**, and it is the *same formula*: `nH ≡ nHidden - 1`, and the ranged `func( hO, d_sigmoidal(), h_err, 0, nHidden-1 )` covers BareProp's whole vector while satisfying `vector_ops.h:334`'s assert. |
| `forward()` | the ranged `dotprod`/`func` likewise satisfy `matrix.h:1427` for both; the genuinely bias-only statement is `hO[ nHidden ] = 1;` |
| `setHidden()` | widths `nInput + 1` vs `nInput`; `hO`/`oW`/`oG` sized `nHidden + 1` vs `nHidden` |
| `unpack()` | offset `(nInput+1)*nHidden` vs `nInput*nHidden` |
| `removeInputs()` | SimpleProp re-appends the bias column and sizes `I` to `nInput+1` |
| `setDataSet()` | SimpleProp appends the bias column to `Train`/`Test`/`Validation` |

**C — genuinely different mathematics; keep visibly per-class (rule 7):**

| Method | Why |
|---|---|
| `df()` | `((nInput+2)*nHidden)+1` vs `(nInput+1)*nHidden` — two published parameter counts. Algebraically unifiable; deliberately not unified. |
| `outputHeader()` | `"Bias nodes on all layers by definition"` vs `"No bias nodes by definition"` — **these are model-file format lines**, read back by `load()`. Frozen. |
| the bias slot / bias column themselves | this *is* the biased-vs-unbiased distinction; hiding it behind a flag is the failure mode rule 7 names |

**D — asymmetries examined:**

| Asymmetry | Finding |
|---|---|
| SimpleProp has `growHidden`/`removeHidden`/`hiddenSaliency` | intended (OBD); leave SimpleProp-only |
| SimpleProp's `innerTrainSet` carries commented-out `storeGrads` blocks | dead comments, no behavior |
| **BareProp's `setDataSet` has no validation block** | **NOT a defect — see §8.3** |

**Corrected proposal.** Extract only category **A**, plus category **B** where the shared form
is *provably* the same expression (`innerTrainSet`, `forward`). Category C stays in the concrete
classes. If a shared intermediate base is built, its bias architecture is **fixed at
construction** (a constructor argument or a per-class constant), never re-read from the mutable
`biasFlag`, and the biased and unbiased forward/gradient equations stay legible either in the
concrete methods or immediately adjacent to the shared implementation. The Manifest documents
the base. Realistic reduction: **~250 lines, not ~450.**

### 8.3 One claim in the review that I checked and it is incorrect

> *"`SimpleProp::setDataSet()` prepares validation data; `BareProp::setDataSet()` currently does
> not. Determine whether that is a defect, fix and test it separately before consolidating."*

**There is no defect, and no commit is needed.** `BareProp::setDataSet` calls
`Model::extractInputMatrices()` (`bareprop.cpp:59`), and that base-class utility populates the
`Validation` submatrix for **every** model type (`model.cpp:143-160`, added in Phase 4c). What
SimpleProp, Logistic and BackProp do *in addition* is append the **bias column** to
`Validation` — which BareProp must not do, because it has no bias column.

The path is complete for BareProp: `DataSet::monitorSet()` returns `MONITOR_VALIDATION`,
`Network::sampleTestError` reads `Validation` and `getValMatrix()( r, nInput )`
(`network.cpp:325-334`), and `BareProp::forward` reads a row into an `I` sized `nInput`, which
is exactly `Validation`'s width. `Model::copy` carries `Validation` into clones
(`model.cpp:37`).

Sol read the four `setDataSet` bodies without the base-class utility they all call. Acting on
this would have spent a commit and a test on a non-issue. Recorded rather than silently
skipped, because the next reader of the review will hit the same instruction.

### 8.4 Verification protocol (replaces §7)

Per Sol, and correcting my §7:

**Before** any extraction, add characterization tests that pin behavior independently of the
implementation, covering: SimpleProp and BareProp forward propagation from fixed saved weights;
one training iteration each, asserting the exact resulting weights and error; batch and
non-batch; auto step-size on and off; CGD and Shanno state restoration; save/load round trips
for both concrete types; validation-set monitoring for every model type that claims it
(including BareProp — see §8.3); forward *and* reverse stepwise; nested and exception-unwound
screen redirection.

**Then sabotage each new test** — deliberately break the behavior it claims to guard, confirm
it fails, restore. That is standing rule 2 correctly applied. Do **not** require a pure
extraction commit to fail against the old binary; behavior preservation is the point.

**After every commit:** zero-warning Release build → `tests/golden/run_golden.sh`
byte-identical → `ctest` → `tests/gui/smoke.sh` → `tests/oracle/verify_oracle.sh` →
`git diff --check`. No item in this audit adds a GUI control, so no click-through is due.

### 8.5 Additional instructions from the review, adopted

- **Auto step-size (§1.2):** `Network` template method with protected snapshot/restore hooks;
  the base owns the η search, `lastG`/`lastF` restoration, loop and stopping behavior, and the
  final real iteration. Concrete classes own only their weight snapshot. **Preserve Logistic's
  guard exactly as written.** Verify exact η, errors, weights and optimizer state per class.
- **DFA (§2.4/2.5):** characterize LDFA and QDFA reports and predictions for binary *and*
  multi-output data first; then extract only the orchestration into `DFA`. A small virtual
  "score this class" is acceptable **only if** the concrete source still displays the published
  formula. No `largerWins()`.
- **Stepwise (§1.3):** not until forward coverage is real and sabotage-proven. Then an explicit
  `Direction` enum plus small direction-specific operations — not a descriptor stuffed with
  strings and comparators. The loop must keep its named invariants visible: candidate recorded
  before the eligibility check, Wilks argument order, `finalVariables` updated only after a
  completed pass, incomplete analyses never claiming a result.
- **GUI launcher (§2.10):** it owns concurrency state, so it gets tests — previous worker
  joined, reset under the right lock, result published before `running` clears, exception
  turned into a finished structured result, cancellation reset, no captured reference
  outliving the handler.
- **`errorFunction` (§4.1):** also drop the temporary `errors = y - o` in the LMS branch and
  accumulate directly; remove `multOutput` rather than merely initializing dead state; preserve
  the cross-entropy boundary behavior exactly; add a multi-output correctness test.
  *One judgment flagged for Craig:* replacing `0.5 * dotprod( errors, errors )` with a hand
  loop trades a line of vector notation for an allocation. Rule 7 says speed and legibility are
  both requirements; here they point opposite ways. A destination-taking form that reuses a
  member buffer would satisfy both, at the cost of a member. Craig's call.
- **`computeCondNum` (§4.3):** reframed. Do not optimize a general transpose plus a general
  product. Measure first (runtime, allocation count, time in each, peak memory); if it matters,
  add a class-layer `gramRows()` expressing $B = GG^{\mathsf{T}}/N$ directly, so the code still
  reads as the mathematics and never materializes $G^{\mathsf{T}}$. Compare exact matrix
  entries and condition numbers before and after.
- **`Population` (§4.5):** do **not** convert to borrowed storage; its value ownership avoids
  lifetime hazards. Consider move construction or a calculation-oriented free function instead.
  The `const&` fix to `stats::ttest`/`tutest`/`tptest`/`ftest` stands.
- **Deferred or rejected:** `STATS_DEBUG` macro compression (§3.6), the frozen CLI prompt
  overloads (§3.7), generic file-writing callbacks spanning unrelated owners (§2.2 as written),
  merging the paired `Matrix` overloads (already rejected in §6), moving OBD to a shared base,
  any model-file format change, any report-wording change during structural work.

### 8.6 Three findings neither the audit nor the review had

Found while building the §8.2 table and re-checking the review's claims.

1. **`SimpleProp::in_bias` and `BackProp::in_bias` are dead members.** Declared at
   `simpleprop.h:93` and `backprop.h:63`; every use site declares a **local** `in_bias` that
   shadows the member (`simpleprop.cpp:70, 785`; `backprop.cpp:92, 720`). The members are
   therefore always empty, and `copy()` faithfully copies the emptiness
   (`simpleprop.cpp:38`, `backprop.cpp:28`). Delete both declarations and both copy lines —
   4 lines, zero risk. Worth doing precisely *because* a member shadowed by a local is the kind
   of thing that later gets "fixed" by someone deleting the local.
2. **`Network::CGD()` and `Network::shanno()` allocate per iteration.**
   `vector<double> u = stackG - lastG;` plus `lastF * b` (and in `shanno`, `u * a`) are fresh
   vectors of packed-gradient length on **every training iteration** for optimizers 2 and 3 —
   the ones `algorithm=auto` picks most often. Sol caught this; my audit missed it entirely.
   The destination-taking forms and compound operators that rule 7 now names would remove them.
   **Measure before changing** — but this is a better-founded candidate than §4.3, because the
   allocation is per-iteration rather than per-report.
3. **`stats.cpp`'s `STATS_DEBUG` switch does not compile.** The ~30 blocks call `util::screen()`,
   but `stats.cpp` includes only `stdafx.h` (which is **empty**, 0 bytes), `<iostream>`,
   `<stdio.h>`, `<math.h>`, `<limits>` and `stats.h` — none of which reach `utility.h`.
   Defining the switch is a build failure. Either fix the include or delete the blocks; do not
   "tidy" them into a macro while leaving them broken (§3.6's original suggestion).

### 8.8 Defects found by the characterization work (2026-08-01)

Step 2 of the sequence found production defects before any refactoring could
obscure their origin. That is the strategy working; it also means the plan grew.

**D1 — a quiet run was a different run. FIXED, `113da40`.**
`Network::runHeader()` computed `regularizer` and `decayTerm` — constants the
training math reads — and `Iterative::train()` called it inside
`if ( !quietFlag )`. No constructor initialized either. A quiet run with weight
decay on therefore trained on uninitialized doubles: NaN under
`-ftrivial-auto-var-init=pattern`, a stable-but-wrong 0.673 against a correct
0.0223 under an ordinary Release build. Fixed by separating `prepareRun()` from
`runHeader()` per Sol's specification, plus constructor defence.
**Reach, stated precisely:** no shipped path is known to have hit it. The only
production `setQuiet(true)` is `RegressNet::copy_network()`, on a clone of an
already-trained model, and `Network::copy` carries both constants across — so
every stepwise candidate inherited good values. `regress_seed42` runs quiet
candidates with decay ON and is byte-identical across the fix, which is the
same fact from the other side. My first report of this said "every stepwise
candidate trained on garbage"; that was an over-claim and is corrected here and
in the test header.

**D2 — reporting the condition number moved the model. FIXED, `de16396`.**
`Network::computeCondNum()` calls `innerTrainSet()` to collect per-exemplar
gradients — and `innerTrainSet()` *applies a weight update*. `Logistic::
reportAccuracy` is the only caller of `reportCondNum`, so an **audible**
Logistic run ends one gradient step past the weights whose error it just
reported, while a quiet run does not. Demonstrated by deleting that one call:
the quiet-vs-audible weight comparison then passes.
Consequences: the saved model, the guesses, and any deployed calculator reflect
weights the reported error never described. Same family as D1 and legacy bug
#10 — reporting must not change the fit.
Fixed as a non-mutating `Logistic::collectGradients()`, **not** by snapshot and
restore — Sol's correction, and the right one: the old call also moved `G`,
`stackG`, `lastG`, `lastF` and the step-size accumulators, so restoring the
weights alone would have left a continued run different from a control. The
canonical value is preserved to the digit (3.7090611118615389 before and after);
the `check_quietprep` exclusion is gone.

**D6 — the condition number was built from search directions under CGD/Shanno.
FIXED in `de16396`, then made moot by `75e6f64`** (the diagnostic is now X'VX and
no longer involves gradients at all).** The old harvest stored `G` *after* `engine()`, and
under training types 1 and 2 `engine()` replaces the gradient with a conjugate
direction. The B matrix is defined as an outer product of gradients, so the
number those two optimizers reported was not a condition number of anything.
Now built from gradients. It appears in no golden, smoke or oracle output.

**D7 — with canonical backprop and gradient stopping OFF, the B matrix was
built from an unwritten Matrix. FIXED in `de16396`, then made moot by
`75e6f64`.** In that
configuration `innerTrainSet` took its no-separate-gradient branch, which never
called `storeGrads`, so `grads` held whatever `resize` left. `collectGradients`
always fills it.

**D3 — a layout-sensitive flake on the validation-monitor path. OPEN, UNEXPLAINED.**
Three assertions (SimpleProp, BareProp, Logistic — never BackProp) failed
together in ~8% of runs of one particular binary, and have not reproduced in any
build since. Ruled out by measurement: not run-to-run randomness in a fixed
binary (6/6 and 100/100 clean); **not D1** — reinstating D1 deliberately brings
its own assertion back 100/100 while these stay at 0/100, so they do not share a
cause; not visible under pattern- or zero-initialized stacks (0/40 each).
What remains is binary-layout sensitivity that nothing yet explains. Per
CLAUDE.md's settled decision on the nested-OBD flake, this is **recorded, not
closed** — "a fix that only reduces a heap-layout-sensitive flake is a suspect,
not a cure." The three assertions now print the returned value, the DataSet's
own monitor answer and the row counts on failure, so the next sighting is
evidence.

**D4 — batch weight decay is applied per exemplar. UNDER AUDIT (Sol step 4).**
In the canonical batch path the multiplier `w *= decayTerm` runs once per
exemplar inside the epoch loop, so the effective per-epoch decay is
`(1 - eta*lambda)^n` — exponential in dataset size. Measured consequence: at
`eta = 1` over 84 rows the weights fell to ~2e-4 of their initial size by
iteration 2000 and every batch arm parked at `ln 2`. The characterization suite
therefore pins the **on-line** decay path only and explicitly declines to pin
the batch one, pending the mathematical note Sol asked for.

**D5 — `vector_ops::func` has no Release-build bounds protection.** Its size
precondition is an `assert`, which vanishes under `NDEBUG`; `Matrix::operator()`
throws unconditionally *by design* for exactly this reason (rule 4). Proven:
giving `SimpleProp` the unranged `func( hO, d_sigmoidal(), h_err )` fires the
assert in a checked build and silently writes one past the end of `h_err` in a
Release one. Per Sol, recorded as a separate hardening finding — the whole
family of range primitives should get a consistent checked contract at once, not
one overload in isolation.

**D8 — the build recompiles the engine 149 times, and it stalled CI. MITIGATED,
root fix outstanding.** Every test target lists its own copy of the engine
sources: 149 compilation units of `src/` across seventeen targets, for 67 files.
With `cmake --build --parallel` (no job cap) the hosted **macOS** runner wedged
at ~90% of the build on 2026-08-01, sat silent for 19.5 minutes, and had every
in-flight compile SIGTERMed at once (exit 143). Ubuntu and Windows were
unaffected. The two test targets added that day made the build ~19% heavier and
appear to have crossed a threshold rather than created one.

*Mitigated* by bounding the build to `--parallel 4` and adding
`timeout-minutes: 25`, so a stall fails fast and legibly instead of burning
twenty minutes and producing an unreadable log.

*The root fix* is to compile the engine ONCE and link the tests against it —
the same DRY defect as everything else in this audit, in CMake. It must be
**layered**, not one monolithic library: several tests are deliberately narrow
and documented as such ("Links only the statistics core -- no GSL needed"),
which is an assertion that those layers do not depend on GSL. A single library
would destroy that proof. The layering already exists in rule 6's dependency
order: a numerical core (matrix / vector_ops / utility / stats / twoset), an
engine layer (model / iterative / network / concrete models), an evaluation
layer (split / crossval / auccov / ...). Each test links the lowest one it
needs. That takes 149 compilations to about 67 and speeds every platform.

### 8.7 Commit sequence — progress

1. ~~Architectural documentation rule~~ — **done, `02870fd`** (standing rule 7 +
   AGENTS.md + Manifest 2.1 + `manifest_maintenance.md` authority row).
2. ~~Characterization tests and sabotage proofs~~ — **done**; `tests/props/`
   (34 assertions) and `tests/iterative/check_quietprep.cpp`.
3. ~~Scoped screen redirection~~ — **done**, `util::ScreenCapture`; all six
   manual pairs and the GUI capture replaced (§4.4).
4. ~~BareProp validation decision~~ — **closed, no defect** (§8.3).
5. ~~D1 quiet-run fix~~ — **done, `113da40`**.
   ~~D2 condition-number mutation~~ — **done, `de16396`**.
   ~~D4 weight decay~~ — **done, `37585c5`**.
   ~~Condition-number definition (X'VX)~~ — **done, `75e6f64`**.
   ~~`errorFunction` allocation/DRY~~ — **done**; the temporary `( y - o )`
   vector became `vector_ops::sumSquaredDifference`.
   ~~`errorFunction` bounds aggregation~~ — **done**, as its own correctness
   commit: the multi-output cross-entropy loop cleared `boundsErrorFlag` in its
   ordinary branch, so an in-range component *after* a boundary one erased the
   record of it; the flag is an aggregate and was also never initialised before
   the loop. `tests/errorfunc/check_errorfunc.cpp` tests both orderings.
   ~~`TwoSet` division ordering~~ — **done**; `checkedRate()` validates the
   denominator before dividing, all four rates, with the ordering demonstrated
   under `-fsanitize=float-divide-by-zero` and the four zero-denominator throws
   and the no-threshold contract pinned in `tests/twoset/check_rates.cpp`.
6. ~~D8 — the build's repeated compilation~~ — **done, `f0c3006`**: 208
   compilation units to 31, layered so the no-GSL boundary is structural.
   Green on all three CI platforms.
7. ~~Mechanical DRY~~ — **done, commits A–D**: `0edbb93` (three dead members —
   `finalFlag`, both `in_bias`), `c87a35c` (§3.1 `RegressNet::network_name`,
   §3.2 the GUI's third dispatch copy, §3.5 `XY::sqr`), `cf0b1df` (§3.4
   `SimpleProp::nH`), `c70788a` (§3.3 the `BackProp` weight buffer).
   `grads`/`storeGrads` went with `75e6f64`.
7a. ~~D5 — the `vector_ops` bounds policy~~ — **done, `de6f00c` (inventory and
   measured gap) + `5c94cd2` (the checks)**: seventeen contracts that could read
   or write outside a container were assert-only in a project that builds Release
   by default; they now throw `nvec::SizeMismatch` / `RangeViolation` /
   `EmptyVector`. Full inventory and policy in §11.
7b. ~~LEGACY BUG #12 (§11.11)~~ — **done**, as its own correctness commit:
   `BackProp`'s batch weight update now consumes the post-`engine()` `Gradient`.
   Four invariants captured from the unfixed engine are bit-identical after it;
   `tests/backprop/check_bpoptimizer.cpp` is the guard, and was proven to fail
   against the unfixed engine before the fix was written. No golden re-blessed.
8. ~~`DataSet::metricsReport` / `Iterative::announceStop` / `Model::writeLastop`~~
   — **done**, three separate commits (`7c5d653`, `2fb6212`, `a553db4`), each
   with its characterization test written and passing BEFORE the extraction:
   `check_metricsreport` (both halves proven to execute, report proven
   byte-identical), `check_announcestop` (all SEVEN exits — reason, token,
   `converged()`, exact text, quiet contract — with two independent sabotages),
   `check_writelastop` (all three writers, truncation over a 20 KB leftover,
   the disabled case, the unopenable path).
9. Auto step-size template method (§1.2, §8.5).
10. Bounded SimpleProp/BareProp sharing (§8.2).
11. DFA extraction, then the measured per-exemplar scoring optimization.
12. Stepwise extraction — only after forward coverage exists.
13. GUI async launcher, with its concurrency tests.
14. Measured `gramRows()` / CGD-Shanno allocation work, only if justified.

**The governing sentence, Sol's:** remove duplicated mechanisms, but never trade away the
visible mathematics or introduce runtime abstraction costs merely to reduce line count. That is
now standing rule 7.

---

## 9. D4 — the weight-decay mathematics, across four models and both paths

Sol's step 4. Written before any decay baseline is accepted, so that a
characterization test cannot elevate a dataset-size-dependent defect into a
contract. Nothing here is fixed yet; this establishes what is wrong and how far
it reaches.

### 9.1 The intended update

The engine's own statement of it is a comment in every model's `innerTrainSet`:

    w_{t+1} = ( 1 - 2*eta*lambda ) w_t - eta * dE/dw

with, from `Network::prepareRun`, `regularizer = lambda = decay/2` and
`decayTerm = 1 - eta*decay = 1 - 2*eta*lambda`. So `decayTerm` is a factor
applied **once per weight update**. That is the standard L2 form: for the
penalized objective

    E(w) = (1/N) sum_k e_k(w) + lambda |w|^2

one gradient step is

    w <- w - eta [ (1/N) sum_k grad e_k + 2*lambda*w ]
       = ( 1 - 2*eta*lambda ) w - eta (1/N) sum_k grad e_k .

### 9.2 What is implemented — three separate mechanisms

Identical in structure in all four models (`simpleprop.cpp:560/600/637/681`,
`bareprop.cpp:317/355/392/425`, `backprop.cpp:480/524/563/607`,
`logistic.cpp:519/535/549`):

| | mechanism | where | guard |
|---|---|---|---|
| **A** | `setError += regularizer * \|w\|^2` | inside the exemplar loop | `weightDecayFlag` |
| **B** | `w *= decayTerm` | inside the exemplar loop, **before** the batch/on-line split | `weightDecayFlag && trainingType == 0 && !gradMaxFlag` |
| **C** | `G += w * decay` | inside the exemplar loop, into the accumulator | `weightDecayFlag`, separate-gradient branch only |

**A is correct.** It adds the penalty N times, and the epoch returns
`setError / nTrain`, so the penalty appears exactly once in the mean objective.

**C is correct.** The penalty gradient is added N times into the accumulator,
which is then divided by `nTrain`, so it contributes `decay*w` exactly once to
the averaged gradient — precisely the `2*lambda*w` term above.

**B is wrong in batch mode.** It sits above the `if ( !batchEpochFlag )` split,
so it runs once per *exemplar* in both modes. On-line does N updates per epoch,
so once per exemplar is right there. Batch does **one** update per epoch, so the
implemented per-epoch factor is

    ( 1 - eta*decay )^N   instead of   ( 1 - eta*decay )

— exponential in the number of training rows. The regularization strength
becomes a property of how many rows the user happened to have.

### 9.3 Reach

B is guarded by `!gradMaxFlag`, and gradient stopping is **on by default**
(`Iterative`'s constructor). With it on, canonical training takes the
separate-gradient branch and uses **C**, which is correct. So the defect is
reachable only when a user turns gradient stopping off — a non-default
configuration, available from the CLI stop-conditions menu and from `/api/train`
with an empty `gradmax`. It affects no golden, no smoke case and not the oracle.

That is the same shape as D1: real, reachable, not currently exercised by any
shipped default path.

### 9.4 Measurement

Batch, canonical, gradient stopping off, `eta = 1`, 500 epochs, SimpleProp 2-3-1,
comparing the engine as shipped against an experimental build with the
multiplier moved to once per update. `|net|` is the mean absolute pre-sigmoid
activation over the training rows — a scale-free proxy for weight magnitude.

| lambda' = decay | N | shipped error | shipped \|net\| | corrected error | corrected \|net\| |
|---|---|---|---|---|---|
| 5e-5 | 40 | 0.1056 | 3.397 | 0.0542 | 4.908 |
| 5e-5 | 160 | 0.2287 | 1.965 | 0.0539 | 4.909 |
| 1e-2 | 40 | 0.5961 | 0.577 | 0.4071 | 1.784 |
| 1e-2 | 160 | 0.6290 | 0.388 | 0.4157 | 1.746 |

The signature is in the N columns. **As shipped the result depends strongly on
dataset size** — at the shipped default lambda', quadrupling N moves the final
error from 0.106 to 0.229 and the weight scale from 3.40 to 1.97, from the decay
rule alone. **Corrected, it does not**: 4.908 against 4.909, identical to three
significant figures, which is what a regularizer whose strength is set by lambda
and eta should do.

The effect is material even at the shipped default, not only at large lambda.

*(A previous note in `tests/props/check_props.cpp` cited weights collapsing to
~2e-4 and the batch arms parking at ln 2. That was measured on a build still
carrying D1's uninitialised `decayTerm`, and is withdrawn. The table above is
measured on the fixed engine.)*

### 9.5 Proposed fix — not applied here

Move B out of the exemplar loop in the batch branch: apply `w *= decayTerm` once,
beside the single weight update at the end of the epoch, leaving the on-line
branch as it is. That makes both modes agree with the manifest's formula and
with mechanism C, and makes the regularizer independent of N. Behavior changes
only for `weightDecay ON && canonical && gradient stopping OFF && batch`, which
no gate exercises — so it needs its own characterization test, written first.

### 9.6 Sol's question: is the penalty gradient part of the B matrix?

Asked of `Logistic::collectGradients`, which adds the constant `decay*W` to
**every** exemplar column. Three different objects are being conflated in the
current definition, and they are not interchangeable:

1. **The training objective's gradient.** `grad E_k = (o_k - y_k) I_k` plus, for
   the penalized objective, a share of `2*lambda*w`. The penalty is a property
   of the *whole* objective, not of an observation; splitting it across N
   columns by giving each the full term (rather than 1/N of it) does not even
   sum to the right total — the accumulated gradient in `innerTrainSet` gets
   away with it only because it divides by `nTrain` afterwards.
2. **The per-observation score.** In likelihood terms the score for observation
   k is `s_k = (y_k - o_k) x_k`, with **no** penalty. The outer-product
   (BHHH) estimator of the information matrix is `sum_k s_k s_k'`. That is what
   an outer product of per-exemplar gradients is an estimator *of*, and it wants
   unpenalized scores.
3. **The penalized curvature.** For a penalized fit the conditioning question is
   about `X'VX + decay*I`, the Hessian of the penalized objective — not an outer
   product of anything.

Adding a constant vector `c = decay*W` to every score does not turn (2) into
(3). It produces `sum_k (s_k + c)(s_k + c)' = sum_k s_k s_k' + c(sum_k s_k)' +
(sum_k s_k)c' + N c c'`, i.e. the score information plus rank-one cross terms
plus an `N c c'` term that grows with the sample. That is not the penalized
Hessian and has no standard interpretation.

**Worth noting:** the engine *already computes* the right object elsewhere.
`Logistic::reportAccuracy` forms `Var(B) = inv( X'VX )` for the Wald tests
(`logistic.cpp`, Hosmer & Lemeshow eqn 2.8). The textbook conditioning
diagnostic for this model is the condition number of `X'VX` (plus `decay*I` when
penalized), which is available a few lines from where the condition number is
printed.

**Recommendation:** treat the condition-number *definition* as a product
decision for Craig, separate from D4's mechanical fix. The options are (a) leave
it as the outer product of penalized gradients, which is what shipped and what
D2 preserved; (b) drop the penalty term, making it the standard BHHH information
estimate; (c) compute it from `X'VX + decay*I`, the penalized Hessian, which is
the quantity a statistician would expect a "condition number" to describe.
D2 deliberately chose (a) because a behavior-preserving fix may not change a
definition. Changing it is a separate, and arguable, improvement.


---

## 10. D8 — the build compiles the engine 208 times. The layered plan.

Sol's step 6, written before any CMake edit.

### 10.1 Baseline, measured 2026-08-01

| | |
|---|---|
| Unique engine sources (`src/*.cpp`) | **31** |
| Engine compilation units across all targets | **208** (19 targets) |
| Redundancy | **6.7x** |
| Clean build, `--parallel 4` (CI's setting) | **36.4 s** wall, **128 s** CPU |
| Clean build, `--parallel` unbounded (24 cores) | **16.0 s** wall, **179 s** CPU |

The CPU figure is the one that matters: it is the work, and it is what stalled
the hosted macOS runner when 19 targets' worth of it ran at once.

### 10.2 The dependency map

Derived from the actual `#include` graph of every `.cpp` and its own header.

**The single fact that governs the design: only `network.cpp` includes GSL.**
Everything below it in the graph is GSL-free, and several tests are documented
as proving exactly that. A monolithic library would erase the proof; a layered
one *enforces* it, because a target that links only the lower layers cannot
acquire GSL even by accident.

```
neuron_core     utility  vector_ops  matrix  stats  twoset          (no GSL)
                        |
neuron_data     split  dataset                                      (no GSL)
                        |
neuron_infer    auccov  delong  clustered_auc  evaldesign           (no GSL)
                        |
neuron_engine   model iterative network* simpleprop bareprop        (* GSL enters here)
                backprop logistic dfa ldfa qdfa netclone
                modelfactory regressnet autoalgo obd
                        |
neuron_eval     crossval  cvadapters  cvreport
                        |
                neuron  ( gui.cpp + neuron.cpp )
```

`neuron_infer` sits beside `neuron_data` rather than above it: `auccov` needs
only `matrix` and `stats`, and `evaldesign` needs nothing at all.

### 10.3 Target -> library

| Target | Links | GSL |
|---|---|---|
| `check_matrix`, `check_az`, `check_wickens`, `check_hl` | `neuron_core` | no |
| `check_plateau` | (header-only, unchanged) | no |
| `check_capture`, `check_split` | `neuron_data` | no |
| `check_delong`, `check_clustered`, `scale_probe` | `neuron_infer` | no |
| `check_obd`, `check_props`, `check_gradcadence`, `check_quietcopy`, `check_quietprep`, `check_condnum`, `check_decay` | `neuron_engine` | yes |
| `check_crossval` | `neuron_eval` | yes |
| `neuron` | `neuron_eval` + `gui.cpp` + `neuron.cpp` | yes |

The nine targets that link GSL today are exactly the nine that link
`neuron_engine` or above tomorrow. The ten that do not, still do not.

### 10.4 Constraints this refactor must not break

- Every executable and every `add_test` NAME survives, unchanged.
- No lower layer acquires GSL because a higher one uses it.
- The no-GSL targets keep proving that boundary — now structurally.
- C++17, and the warning configuration, stay as they are (there are no
  per-target compile definitions or options today; verified).
- No engine behavior, report, model format or public interface changes. This is
  a CMake-only commit.
- `gui.cpp` keeps its private include directories (the generated `gui_page.h`
  and `third_party/`), which no library needs.

### 10.5 Result, measured

| | before | after |
|---|---|---|
| Engine compilation units | 208 | **31** |
| Clean build, `--parallel 4` | 36.4 s wall / 128 s CPU | **13.4 s / 32 s** |
| Clean build, unbounded (24 cores) | 16.0 s wall / 179 s CPU | **9.6 s / 34 s** |

CPU work drops **4x**. That is the number that mattered: it is what 19 targets
were duplicating, and what stalled the hosted macOS runner when they ran at once.

Two corrections to the plan, found by building it:

- **`scale_probe` spans two SIBLING layers.** It uses the clustered estimator
  (`neuron_infer`) *and* the group fold planner (`nsplit`, in `neuron_data`),
  and neither sits above the other, so it names both. The link failure was the
  map being wrong, not the layering.
- Nothing else moved. The nine targets that linked GSL before link it now; the
  ten that did not, still do not — verified by reading each target's generated
  link line, not by assuming.

Every `ctest` name at that time was byte-identical to the pre-refactor list, checked by
diffing `ctest -N` across a `git stash`.

---

## 11. D5 — the bounds policy of the numerical vector layer

**This section is the design. It was written before `vector_ops.h` was edited**
(Sol's instruction, 2026-08-01), because the question is not "replace the assert
in `func`" but "what does this layer promise, in the build we actually ship".

### 11.1 The finding that forces it, and the finding that widens it

**The motivating defect, proven.** `SimpleProp::trainSet` passes `hO`
(`nHidden + 1` elements, last one the bias slot) into `h_err` (`nHidden`). Given
the *unranged* `func( vec_in, fx, vec_out )`, `transform` writes `vec_in.size()`
elements into `vec_out` — one past the end of `h_err`. The guard is
`assert ( vec_in.size() == vec_out.size() )`. In a checked build the assert
fires; in the shipped build it does not exist.

**The wider finding, measured today.** `build/CMakeCache.txt`:

```
CMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG
```

and `CMakeLists.txt:11` forces `CMAKE_BUILD_TYPE=Release` when none is given. So
**every one of the 19 `ctest` targets already runs with `NDEBUG` defined**. Not
one `vector_ops` assertion has ever executed in the gate chain. The assertions
are not a weaker guard than they appear — in the configuration this project
builds, tests, and ships, they are *not a guard at all*.

That also makes `CMakeLists.txt:8` wrong where it reassures:

> asserts vanish under -DNDEBUG by design (handlers validate; Matrix bounds
> checks are unconditional throws, not asserts)

True of `Matrix`, and false of the layer beside it. Corrected in D5b.

### 11.2 The inventory

Every operation in `src/vector_ops.{h,cpp}`, not only the ones carrying an
`assert`. **Class** is: **A** = a violation can construct an invalid iterator,
read past a container, write past a destination, or dereference an empty one;
**B** = a violation gives a mathematically undefined *value* but touches no
memory it does not own; **—** = no precondition to violate.

| # | Operation | Precondition | Debug today | Release today | Class | Proposed |
|---|---|---|---|---|---|---|
| 1 | `nvec::random( v, n )` | none — fills existing elements | ok | ok | — | unchanged |
| 2 | `operator+= -= *= /=( vector<T>&, const vector<T>& )` (4) | `lhs.size() <= rhs.size()` — **prefix, deliberate**, see 11.3 | `assert` | **reads `lhs.size()` elements out of a shorter `rhs`** | A | `throw nvec::SizeMismatch` |
| 3 | `operator+ - * /( vector<T>, const vector<T>& )` (4) | as #2; `lhs` is a by-value copy, so the result takes `lhs`'s size | via #2 | via #2 | A | inherited from #2 |
| 4 | `operator+= -= *= /=( vector<T>&, const T )` (4) | none | ok | ok | B (`/=` by a zero scalar) | unchanged — see 11.4 |
| 5 | scalar `+ - * /` (6 overloads, both operand orders for `+`/`*`) | none | ok | ok | — | unchanged |
| 6 | `operator<<( ostream&, const vector<T>& )` | none | ok | ok | — | unchanged |
| 7 | `operator>>( istream&, vector<T>& )` | none — reads into existing elements only, so it silently consumes `rhs.size()` items | ok | ok | — | unchanged |
| 8 | `operator==`, `operator!=` | none — size difference is an explicit `if`, returning `false` | ok | ok | — | unchanged |
| 9 | `dotprod( v1, v2 )` | `v1.size() == v2.size()` — **equality, deliberate**, see 11.3 | `assert` | **`inner_product` reads `v1.size()` from a shorter `v2`** | A | `throw nvec::SizeMismatch` |
| 10 | `func( vec_in, fx, vec_out )` | `vec_in.size() == vec_out.size()` | `assert` | **writes `vec_in.size()` into a shorter `vec_out`** — the proven defect | A | `throw nvec::SizeMismatch` |
| 11 | `func( vec_in, fx, vec_out, a, b )` | `a <= b`, `b < vec_out.size()`, **and `b < vec_in.size()`** | `assert` — **and the input bound is not asserted at all** | reads past `vec_in`; `a > b` forms a reversed range | A | `throw nvec::RangeViolation` on **both** sides |
| 12 | `func( vec_in, fx )` | none — allocates `vec_in.size()`, so it always satisfies #10 | ok | ok | — | unchanged |
| 13 | `func( vec_in, fx, a, b )` | `a <= b`, `b < vec_in.size()` | `assert` | reads past `vec_in` | A | `throw nvec::RangeViolation` |
| 14 | `flatten( container, vec )` / `flatten( container )` | none — `clear()` then `push_back` | ok | ok | — | unchanged |
| 15 | `squared( v )` | none — empty sums to `0`, the correct identity | ok | ok | — | unchanged |
| 16 | `sumSquaredDifference( a, b )` | `a.size() == b.size()` | `assert` | **walks `b` in lockstep with `a`, past its end** | A | `throw nvec::SizeMismatch` |
| 17 | `maxabs( v )` | non-empty — see 11.5 | none; empty returns `0` | same | A′ | `throw nvec::EmptyVector` |
| 18 | `minabs( v )` | non-empty | none; **dereferences `*v.begin()` unconditionally** | same — UB on empty | A | `throw nvec::EmptyVector` |
| 19 | `bin( v_in, b, binFlag, v )` | `b > 0`; empty input is harmless | `assert( b > 0 )`, `assert( v_in.size() > 0 )` | **`v_in.size() / b` with `b == 0` is integer division by zero** — UB, `SIGFPE` on both our targets | A | `throw nvec::RangeViolation` on `b == 0` (`b` is an *extent*, not a container length); empty input **accepted** (yields one empty bin, as today) |
| 20 | `bin( v_in, b, binFlag )` | as #19 | via #19 | via #19 | A | inherited |

Counting overloads, that is 36 functions, of which **13 can misuse memory in the
shipped build** and are protected today by nothing.

### 11.3 The `<=` / `==` asymmetry is intentional. Do not normalize it.

Sol asked this to be settled from call sites and documented formulas rather than
by taste. It is settled, and it is load-bearing:

- `src/simpleprop.cpp:584` — *"Note also that although `oW` has 1 more element
  than `h_err`, its last element will be ignored in `*=`"*. `oW` is
  `nHidden + 1` (bias); `h_err` is `nHidden`.
- `src/backprop.cpp:476` — *"in the case of bias, where size `HErrors[]` <
  `Weights[]` cols, that the result of `dotprodt` will truncate to `HErrors[]`
  size, and that the result of `func`, also > size `HErrors[]`, will truncate to
  size of `HErrors[]` because `*=` results in the size of LHS"*.
- `src/vector_ops.h:88` — the binary operators take `lhs` **by value**, so the
  result carries `lhs`'s size. The prefix rule is the same rule.

So the compound operators implement a deliberate **prefix** semantic that the
bias-slot arithmetic of both networks depends on. `dotprod` requires **equality**
because a dot product over a prefix is a *different scalar*, and silently
returning it is the failure mode a bounds check exists to prevent.

`func( in, fx, out )` keeps equality for the same reason, and this is the point
of the whole exercise: the operation that truncates says so in its contract; the
operation that does not, refuses. Had `func` quietly accepted a longer input as a
prefix, `SimpleProp` would have been "correct" by definition and the bias slot
would have been silently squared into the hidden error. The fix was to make the
caller state its domain — `func( hO, d_sigmoidal(), h_err, 0, nHidden - 1 )` —
and the contract must keep forcing that.

### 11.4 What deliberately stays unchecked

- **Scalar `/=` by zero** (#4). A mathematical result, not a memory access;
  `double` gives IEEE infinity. Adding a branch would put a test inside a
  per-element scalar loop for a case no caller has. `TwoSet`'s rate accessors
  already show where a zero denominator genuinely needs refusing — at the
  formula, not in the vector layer.
- **Empty input to `squared`, `dotprod`, `sumSquaredDifference`** (#15, #9, #16).
  A sum over an empty index set is `0`, and that identity is already asserted:
  `tests/errorfunc/check_errorfunc.cpp:153`.
- **Empty input to `bin`** (#19). Memory-safe today (`n = 0`, the short-input
  branch appends the empty vector) and there is no reason to make it an error.
  Note the `assert( v_in.size() > 0 )` currently there is stricter than the code
  needs; it is dropped rather than promoted.

### 11.5 Two judgement calls, stated rather than buried

**`maxabs` on an empty vector.** It returns `0` today, because `result` is
initialised to `0` and the loop simply does not run. That is memory-safe but it
is a *lie*: the maximum of an empty set has no value, and `0` is the one value a
caller acts on — `Network::getGradMax()` feeds it straight to the gradient
stopping rule, where "the largest gradient is 0" means *converged*. A model with
no parameters would be certified as having converged.

I checked whether that path is live, rather than assuming either way. It is not:
`maxabs` has four live call sites (`network.cpp:510, 535, 576, 763`; the two in
the commented-out historical block do not count). Three take `stackG`, the packed
gradient, whose size is `df()` — and no model returns `0` from `df()`
(`Logistic::df()` is `getInput() + 1`, so even forward stepwise's empty baseline
has the intercept). The fourth takes `eigenv`, sized `dimension`, and
`conditionOf` returns early at `dimension == 0`. **So refusing empty changes no
reachable behavior**, and it removes a false convergence signal from a future
one. Proposed: `maxabs` and `minabs` both refuse, symmetrically.

**`bin` has no production caller at all.** Grepped: the only match outside
`vector_ops.h` is a local variable named `bin` in `tests/binormal/check_az.cpp:80`.
It is guarded here because it is public API of the layer and D5's job is the
layer's contract — but *whether it should exist* is a separate question for
Craig, not one to settle inside a bounds commit.

### 11.6 The exception contract

Declared in the vector layer, in the `nvec` namespace that already exists there:

```cpp
namespace nvec {
    // A vector operation refuses operands it cannot compute over without
    //    reading or writing outside them. These are CONTRACT violations --
    //    programming errors, not data conditions -- and they are checked
    //    unconditionally, because the engine ships with NDEBUG defined.
    class SizeMismatch    : public std::exception { ... };  // parallel containers of incompatible length
    class RangeViolation  : public std::exception { ... };  // positions outside the container they index
    class EmptyVector     : public std::exception { ... };  // an operation with no value on the empty set
}
```

Three flat types, no hierarchy beyond `std::exception`. They are distinguished
because they answer different diagnostic questions — *"your two vectors disagree"*,
*"your indices are outside"*, *"there is nothing to compute"* — which is the
condition Sol set for splitting them.

- **No upward dependency.** `vector_ops.h` and `matrix.h` are siblings (neither
  includes the other; only `matrix.cpp` includes `vector_ops.h`), so reusing
  `Matrix<T>::BoundsViolation` would invert the layering. Verified, not assumed.
- **They derive from `std::exception` and carry a `what()`**, unlike
  `Matrix::BoundsViolation`, whose `: public std::exception` is commented out at
  `matrix.h:121`. That is deliberate: `catch ( ... )` still works everywhere it
  works today, and the GUI's existing `catch ( const exception& e )` handlers
  (e.g. `gui.cpp:1070`) newly get a message instead of letting the throw reach
  cpp-httplib. Bringing `Matrix::BoundsViolation` into line is a **separate
  commit**, not smuggled into this one.

### 11.7 Policy, in the form it will be applied

1. Every Class-A precondition is checked **unconditionally**, in Release.
2. **The replaced asserts are removed, not kept beside the throws.** An `assert`
   next to an unconditional check adds nothing and re-teaches the reader the
   exact misconception that produced this defect — that these are debug-only
   concerns. After D5b, `vector_ops.h` contains no assertions.
3. **Checks happen once, before the loop, never per element.** All of them are
   integer comparisons ahead of an O(n) transform. Rule 7 says measure rather
   than assert that this is free; the measurement is in D5b's plan below.
4. **No destination is silently resized.** A destination-taking operation
   *rejects* an incompatible destination. Automatic resizing would hide exactly
   the model-shape defect this is here to surface.
5. **The successful path is untouched** — same equations, same iteration
   domains, same allocations. Goldens byte-identical, model files byte-identical.
6. Exact-size requirements are not weakened, and prefix requirements are not
   tightened (11.3).

### 11.8 The test, and why an ordinary test target would not do

A `ctest` case that passes proves nothing here unless it is compiled the way the
engine ships. `check_vector_bounds` therefore sets `NDEBUG` **explicitly**:

```cmake
add_executable(check_vector_bounds tests/vectorops/check_vector_bounds.cpp)
target_link_libraries(check_vector_bounds PRIVATE neuron_core)
target_compile_definitions(check_vector_bounds PRIVATE NDEBUG)
add_test(NAME vector_ops_bounds COMMAND check_vector_bounds)
```

Release already defines it, but a `Debug` configuration would not, and the
guarantee must not depend on how someone configured their build directory.

**One case per process.** Run with no argument it runs every case and is an
ordinary ctest case. Run as `check_vector_bounds <n>` it runs exactly one. That
second mode exists for the *demonstration*: against the current assert-only code
these cases do not throw, they corrupt memory, and a crash in case 3 would
otherwise hide cases 4–13. The driver runs each in its own process and records
the exit status.

Cases: the twelve Class-A violations of 11.2, plus the one Sol singled out
(`b` outside the **input** of the destination-range `func`, which no assert
covers today), plus positive controls — boundary-valid ranges, single-element
ranges, equal empty vectors where the contract permits them, a **longer** right
operand on each compound operator (the prefix semantic of 11.3, which must keep
working), and ordinary calculations.

**Three proofs of meaning, per Sol:**

1. the focused target is shown to have been rebuilt with `NDEBUG` (compile line
   recorded, not assumed — stale binaries have produced false results three
   times in this refactor);
2. the shipped assert-only `func` is restored and the Release-focused test is
   shown to fail, or a sanitizer is shown to catch the invalid access;
3. the input-bound check and the destination-bound check are sabotaged
   **independently**, so the test is proven to hold both contracts and not one
   twice.

Proofs 1 and 2 are measured in 11.12 — D5a *is* the restored state, so the
harness runs against the assert-only operations directly. Proof 3 belongs to
D5b, where there are checks to sabotage.

Sanitizers are a one-time memory-safety proof. The permanent suite asserts the
public exception contract and does not depend on their availability — the same
split used for the `TwoSet` denominator fix.

### 11.12 D5a, measured

**The compile line, read rather than assumed** (`build/CMakeFiles/check_vector_bounds.dir/flags.make`):

```
CXX_DEFINES = -DNDEBUG
CXX_FLAGS   = -O3 -DNDEBUG -std=c++17 -arch arm64
```

**`tests/vectorops/run_bounds_demo.sh`, against the unchanged operations:**

```
held: 10    unprotected: 15    crashed: 2
```

**All seventeen Class-A contracts are unenforced in the shipped build.** Fifteen
ran silently outside their operands and returned a value; two —
`func( in, f, out, a, b )` and `func( in, f, a, b )` with `a > b`, which form a
reversed iterator range — killed the process with `SIGBUS`. **All ten positive
controls held**, including both prefix cases: a longer right operand on `*=` and
`+=` still truncates to the left-hand size, which is what `SimpleProp`'s
`h_err *= oW` and `BackProp`'s hidden-error chain depend on.

**A second microscope, and one that is not available.** The same source built
with libc++'s hardened mode
(`-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG -DNDEBUG -O1`)
additionally traps case 15, `minabs` on an empty vector, with `SIGTRAP` — an
independent confirmation that `*v.begin()` there is a real invalid dereference
and not merely an unchecked contract. It does not catch the `transform` cases:
hardening validates a dereference, not the relationship between two iterators
handed to an algorithm.

**AddressSanitizer does not work on this machine, and the finding is that it is
broken, not that it is slow.** An ASan build of

```cpp
#include <cstdio>
int main(){ printf("hello\n"); return 0; }
```

compiles and then hangs at startup, producing no output, with and without the
tool sandbox. That is not instrumentation overhead — Sol's rule applies:
*"if even a tiny focused target times out, that indicates a broken invocation, a
hung program, or an overbroad build."* It was the runtime itself. ASan is
therefore not used here; proof 2 is satisfied by its other branch, the
Release-focused test failing, which it does on every one of the seventeen. Note
that `-fsanitize=float-divide-by-zero` **did** work earlier in this refactor (the
`TwoSet` ordering proof) — UBSan-style checks are inline instrumentation and need
no shadow-memory remapping, so "sanitizers" is not one capability here.

### 11.9 Commits

- **D5a** — this section; the `nvec` exception vocabulary; the harness, built
  under `NDEBUG` against the *unchanged* operations and recorded as failing every
  Class-A contract. **Not registered with `add_test` yet**, so the gate chain
  stays green while the gap is on the record.
- **D5b** — the checks; the harness wired into `ctest` (`vector_ops_bounds`,
  making it 20 cases); `CMakeLists.txt:8`'s claim corrected; the obsolete
  assert-only commentary removed; `CLAUDE.md` rule 4 amended (11.10); manifest
  updated, in its established vertical method-by-method format, because the
  layer's documented public contracts did change. Results in 11.13.

### 11.10 Rule 4 needs one sentence

Rule 4 argues bounds safety entirely through `Matrix::operator()`, and
`CMakeLists.txt` repeats it. Both read as though `Matrix` were the whole
guarantee. The amendment states that **both** `Matrix` and the numerical vector
operations reject bounds and dimension violations in Release — which is what
makes the class layer worth staying inside.

### 11.13 D5b, measured

**Which assertions remain in `vector_ops.h`: none.** `#include <cassert>` went
with them. An `assert` beside an unconditional check adds nothing and re-teaches
the exact misconception that produced the defect — that these are debug-only
concerns.

**No exercised production caller violated the newly explicit contract.** The
goldens are byte-identical (three transcripts covering SimpleProp, BareProp,
BackProp, Logistic, stepwise and the binormal path), 20 `ctest` cases pass
including OBD, CV, the characterization suite and all three optimizers, the full
GUI smoke passes every endpoint, and the oracle is numerically identical. Had any
live call site been relying on a violated precondition, it would now throw rather
than overrun. That is evidence about *exercised* callers, which is what a test
suite can give.

**Cost: none measurable.** Rule 7 says measure rather than assert that a check is
free. A hot-loop benchmark at SimpleProp's shapes (`hO` = nHidden + 1, `h_err` =
nHidden, `oW` = nHidden + 1, nHidden = 10), four guarded calls per iteration —
the ranged destination `func`, the prefix `*=`, the equal-size `func`, and
`dotprod` — 20 M iterations, A/B interleaved five times:

| | ns per iteration |
|---|---|
| before (assert-only, asserts compiled out) | 8.36, 8.40, 8.46, 8.48, 8.46 |
| after (unconditional checks) | 7.65, 7.80, 7.79, 7.73, 7.75 |

The checked build is reproducibly **~8% faster**, with an identical checksum.
**I have not explained that and am not claiming it as a benefit** — plausibly the
size comparison hands the optimizer a proven relationship, or it is code layout.
The claim that survives measurement is the one that matters: a check performed
once before an O(n) loop costs nothing detectable, even at n ≈ 10 on the hottest
path in the engine.

*(Two traps worth recording. The first three "before" runs read 16.3 ns — double
the truth — because a build was running concurrently; interleaving A/B is what
exposed it. And the "after" binary was confirmed to contain the checks by
`nm`: 12 `nvec` exception symbols against 0 in the "before" binary. Stale
binaries have produced false results three times in this refactor.)*

**Sabotage, both bounds independently** (proof 3). Snapshot-restore by file copy,
never `git checkout`, and each rebuild confirmed by its own `Building CXX object`
line:

| sabotage | case 4 (input bound) | case 5 (destination bound) |
|---|---|---|
| `b >= vec_in.size()` removed | **FAIL** | ok |
| `b >= vec_out.size()` removed | ok | **FAIL** |

Each check is therefore held by its own case, and the test is not asserting one
contract twice. Restored and verified byte-identical against the snapshot;
`grep -c SABOTAGE` returns 0.

**Documentation.** `CMakeLists.txt`'s reassurance and `CLAUDE.md` rule 4 both
named `Matrix` alone as the bounds guarantee; both now state that the numerical
vector operations refuse in Release too. The manifest gains
§"Bounds and dimension violations" plus per-method `Notes` on the unary
operators, `dotprod`/`sumSquaredDifference`, all four `func` forms, and
`maxabs`/`minabs`, in its established vertical format — the layer's documented
public contracts changed, which is the condition for touching it. Rebuilt and
text-checked: 202 pages, both new cross-references resolving.

### 11.11 The parked observation, resolved: LEGACY BUG #12

The observation was: while proving commit D (`c70788a`) behavior-preserving, a
`BackProp` fixture — batch + epoch, automatic step size, two hidden layers,
seeded — gave **the identical final error `0.58003708523755504` under canonical,
CGD and Shanno**. It was recorded as unverified. It is now diagnosed, and it is
a defect.

**`BackProp` computes the conjugate direction in batch/epoch mode and then
throws it away.** `src/backprop.cpp:606-616`:

```cpp
// Set the gradients to the accumulators so that pack() & unpack() work
for ( unsigned l = 0; l < Gradient.size(); l++ )
    Gradient[ l ] = WeightsAccumulate[ l ] /= ( double ) nTrain;

engine( trainingType, iteration );      // writes its result into Gradient

for ( unsigned m = 0; m < WeightsAccumulate.size(); m++ )
    Weights[ m ] -= ( WeightsAccumulate[ m ] *= eta );   // reads WeightsAccumulate
```

`Gradient[ l ] = ...` is a `Matrix` **copy**, so `Gradient` and
`WeightsAccumulate` are distinct objects holding equal values. `Network::CGD` /
`Network::shanno` `pack()` from `Gradient`, compute the direction, and `unpack()`
back into `Gradient`. The weight update then reads `WeightsAccumulate`, which the
optimizer never touched. Under CGD and Shanno, batch `BackProp` performs
**plain gradient descent**.

**It is not inert dispatch, and not a weak fixture.** Both were checked rather
than assumed:

- A temporary `fprintf` probe in `Network::CGD` and `Network::shanno` (snapshot,
  inserted, rebuilt with the compile line confirmed, restored byte-identical)
  shows **2720 entries each** during the batch runs. The optimizers run.
- A discriminating fixture — two hidden layers, batch, automatic step size
  **off**, fixed `eta`, gradient and plateau stopping off, all three optimizers
  loaded from **one saved weights file** rather than a re-seeded `randomize()`,
  weights compared bitwise as `%a` after 1, 2, 5 and 20 iterations — gives
  identical values at **every** checkpoint, from the first iteration.
- The same fixture in **on-line** mode diverges immediately (canonical
  `0x1.3739bae822973p+6`, CGD `0x1.1e20d1981552fp+6`, Shanno `0x0p+0` after one
  iteration). So the fixture discriminates and the dispatch works; on-line
  `BackProp` updates from `Gradient` (`backprop.cpp:542-545`) and is correct.

**Reach, measured.**

| configuration | canonical / CGD / Shanno |
|---|---|
| BackProp, batch, auto step size off | **identical** |
| BackProp, batch, auto step size **on** (the parked fixture) | **identical** |
| BackProp, batch, gradient stopping on | **identical** |
| BackProp, **on-line** | distinct |
| **SimpleProp**, batch, 3 hidden | distinct |
| **BareProp**, batch, 3 hidden | distinct |

So the defect is exactly **BackProp × batch/epoch × (CGD or Shanno)**.
`SimpleProp` and `BareProp` update from `oG`/`hG` *after* `engine()`
(`simpleprop.cpp:731-732`, `bareprop.cpp:471-472`) and are correct in both modes.
Canonical with gradient stopping takes the same separate-gradient branch but
dispatches to a `switch` with no `case 0`, so `Gradient` is unchanged and the
update is numerically right — measured identical to canonical without gradient
stopping.

**Reachable in production**, not a laboratory case. `autoalgo::pick` clones the
current model, sets each `trainingType`, and **forces `setBatchEpoch( true )` for
CGD and Shanno** (`autoalgo.cpp:98-100`) — precisely the defective combination.
On a `BackProp`, `algorithm=auto` therefore probes three optimizers that are all
secretly the same one, and selects a winner among identical fits. Any GUI user
who builds a multi-layer network and chooses CGD or Shanno with batch/epoch gets
canonical gradient descent under a different name.

**Why it survived.** `tests/props/check_props.cpp:404` covers CGD and Shanno for
**SimpleProp and BareProp only** — the `Case` struct literally has one expected
value per those two models and none for `BackProp`. No test, golden, or smoke
check has ever run CGD or Shanno on a `BackProp` and compared it against
canonical. This is standing rule 2 again: the optimizer tests execute the
dispatch, pass, and guard nothing for the one model where it does not work.

**It is legacy, not a regression.** The same six lines are in
`../distro/src/backprop.cpp:630-634`, and `git log -L` puts them in the initial
carry-forward commit `f87e9dd`. Commit D (`c70788a`) touched
`innerTrainSet`'s weight buffer and is not implicated — its own before/after
identity was separately proven load-bearing.

**The correction, applied 2026-08-01** as its own commit. It updates from the
structure the optimizer actually wrote, as the other two models already do:

```cpp
for ( unsigned m = 0; m < Gradient.size(); m++ )
    Weights[ m ] -= ( Gradient[ m ] * eta );
```

Multiplied **by value**, not with `*=`: `Gradient` is read again by
`getGradMax()` and by the next iteration's optimizer state, and must not be
scaled by `eta` in place.

The three questions raised before writing it, resolved:

1. It changes numerical output for BackProp + batch + CGD/Shanno. **No golden
   covers that combination** — no golden fixture references `BackProp` at all —
   so nothing was re-blessed and nothing moved.
2. `WeightsAccumulate` **keeps its lifetime and storage** (Sol's ruling). It is
   still the per-exemplar batch accumulator, reset at the top of every pass;
   becoming dead after the average is copied out does not justify changing it in
   a correctness commit.
3. It landed as its **own** commit, before the three DRY extractions and before
   the auto-step-size template method (§1.2).

### 11.14 The measured values — evidence, not a contract

The first `check_bpoptimizer` pinned these as exact hexadecimal literals. They
passed on macOS and **failed on Ubuntu/GCC and Windows/MSVC while every
structural assertion passed on all three** — which is the correct outcome, and
the right reading of it is that the correction is real everywhere and twenty
iterations of a conjugate method are allowed to differ in their low bits between
compilers. Pinning one machine's bytes as a universal contract was an overclaim;
a per-platform table would be the same overclaim three times. The numbers are
kept here as the sabotage evidence they are.

| arm | before the fix (macOS) | after |
|---|---|---|
| canonical batch | `0x1.760ffbe7973c2p+4` | `0x1.760ffbe7973c2p+4` |
| batch CGD | `0x1.760ffbe7973c2p+4` | `0x1.50564fab14283p+4` |
| batch Shanno | `0x1.760ffbe7973c2p+4` | `0x1.70220b4a8a31ap+4` |
| batch CGD, auto step | `0x1.750c75ff17d87p+4` | `0x1.a5cfd3a6729ffp+4` |
| batch Shanno, auto step | `0x1.750c75ff17d87p+4` | `0x1.6eebfea1aa607p+4` |
| on-line canonical | `0x1.225f05c606211p+5` | `0x1.225f05c606211p+5` |
| on-line CGD | `0x1.617604c2720ddp+2` | `0x1.617604c2720ddp+2` |
| on-line Shanno | `0x1.5p+6` | `0x1.5p+6` |

Before the fix the three batch arms are **one number**. After it they are three,
and every arm the correction was not allowed to touch is unchanged.

**The portable invariant that replaced them is stronger than a capture**,
because it is a statement about the code rather than about one machine:
canonical training with gradient stopping **armed** takes the separate-gradient
branch — the one that was wrong — but `engine( 0, ... )` dispatches to a `switch`
with no `case 0`, so nothing transforms the gradient and the update must agree
with the canonical branch's own, to within floating-point reassociation
(`( a * eta ) / n` against `( a / n ) * eta`). Proven to guard: dropping the
batch average from that branch fails it while the separation assertions still
pass, and restoring the old `WeightsAccumulate` update fails the separations
while the invariant still passes. The two are independent.

The separation threshold is **0.01**, chosen from measurement rather than to
make CI green: the smallest real gap is 0.3706 (canonical against Shanno,
batch), so the threshold sits ~37x below the true signal while double rounding
noise on a sum of magnitude ~23 is around 1e-14. Failures print hexadecimal, so
future cross-platform drift stays diagnosable.

**On-line CGD and Shanno are degenerate at every step size tried** — they
collapse to zero or saturate every output. That is a property of the algorithms,
not of the fixture: conjugate methods assume a true batch gradient, which is
exactly why `autoalgo::pick` forces batch/epoch for both. They are smoke in the
permanent test (they must run and stay finite); only the on-line canonical arm
is a quality control.


---

*Prepared by Claude (Opus 5). Reviewed by Sol (2026-07-31); revised §8 in response.
§9–§11 are the implementation record: §§8.7–8.8 and 9–10 describe work now
committed, and §11 is the D5 design, written before its code.*
