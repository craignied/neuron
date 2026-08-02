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
9. ~~Auto step-size template method (§1.2, §8.5)~~ — **done**, in two commits:
   `7bc89b8` (characterization, test-only) and the extraction that follows it.
   `Network::searchStepSize`, a protected member template; each model supplies
   its own guard at the call site and a private `WeightSnapshot`. 283 lines
   removed for 196 added, four models' behavior byte-identical on the
   refactoring machine, five sabotages each failing a distinct set.
10. Bounded SimpleProp/BareProp sharing (§8.2). **Commit 1 done**: `OneHiddenNet`
    (`src/onehidden.{h,cpp}`) owns the shared state and category A — `randomize`,
    `save`, `load`, `pack`, the copy utility and the now-identical
    `WeightSnapshot`. The two concrete models lose 456 lines and gain 44. Bias
    architecture is the concrete type's, never the mutable `biasFlag`; `load()`
    sizes through a narrow pure virtual `setHidden( unsigned )`. Proven by
    sabotaging the NEW shared code (§8.4): dropping `oW` from `copy()` fails 4
    assertions, letting `save()` read `biasFlag` fails 2. All five gates green,
    nothing re-blessed; the Manifest documents the base.
    **Commit 2 done**: `innerTrainSet()` and the forward equations, category B,
    shared with **zero parameterization** — with comments discounted the two
    `innerTrainSet` bodies are now identical and `forward` differs only in
    `hO[ nHidden ] = 1;`. The single executable difference is one formula once
    its domain is written as a range: elements 0 .. `nHidden - 1` are the hidden
    units, every element of an unbiased `hO` and all but the pinned bias slot of
    a biased one. The forward equations live in a non-virtual
    `OneHiddenNet::propagate()` — no dispatch added to a per-exemplar path
    (rule 7) — and each `forward()` keeps reading its exemplar, SimpleProp's
    keeping the bias pin. Proven the same expression BEFORE moving anything:
    giving BareProp the ranged forms alone reproduces every `check_props`
    literal captured at `02870fd`. Sabotages of the new shared code: narrowing
    the range to `nHidden - 2` fails 19 assertions across both models; deleting
    SimpleProp's bias pin fails 9 SimpleProp and **0** BareProp. The three dead
    `finalFlag` / `storeGrads` comment blocks went with the move (§8.2 asymmetry
    D) — both identifiers were already deleted and the condition number is X'VX.
    All gates green including `tests/tools/run_tools.sh`; nothing re-blessed.
10a. ~~**D9 — the `Matrix` bounds policy**~~ — **DONE**. Inventory (§12.2),
    Release characterization proven red (§12.6), catch sites enumerated (§12.7),
    Sol's policy mapped case by case (§12.8), the boundary helper (§12.9), and
    the implementation with its four sabotages (§12.10). `DimensionMismatch`
    added, all four classes derived from `std::exception`, 55 contracts enforced
    at the entry points, `runOnWorker` and the CLI `main` boundary. Originally
    recorded as:
    Discovered by Commit 2's verification and widened by Sol: `Matrix::operator()`
    and `includerows` throw in Release, and roughly forty-five other public entry
    points that can read or write outside the allocation are guarded by `assert`
    alone — which never executes in the build this project ships. Same class as
    D5, and it contradicts standing rule 4's sentence about *both* numerical
    classes. Inventory, policy and order of work in §12. **DFA depends heavily on
    `Matrix`, so this lands first.**

11. DFA extraction, then the measured per-exemplar scoring optimization.
    **Characterization DONE** (`tests/dfa/check_dfa.cpp`, 44 assertions): both
    models, binary and three-class, the reports, the graded guesses, ROC
    availability per output arity, save behavior, history and last-operation
    behavior, the singular and non-discrete refusals, running twice, and a
    standalone analysis leaving a trained network's guesses alone. **Six
    sabotages**, each failing a distinct set — see §12.12. No implementation
    code was touched.
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

### 11.15 FIXED: `Iterative::iteration` was not initialised

Found 2026-08-01 while writing `tests/onehidden/check_onehidden.cpp`, by D5's
own bounds check firing: five CGD iterations threw `nvec::SizeMismatch`.

`Iterative::Iterative()`'s initialiser list (`iterative.cpp:13-19`) sets
nineteen members and **does not set `iteration`**. It is an `unsigned`, so it
holds indeterminate memory until `train()` assigns it.

**The production training path is safe**, and that is established rather than
assumed: the only caller of `trainSet()` in `src/` is `iterative.cpp:312`, which
sits inside `for ( iteration = 0; iteration <= maxIterations; iteration++ )`. So
`iteration` is always assigned before any training pass reads it.

**Two places where it is nevertheless observable:**

1. `Iterative::copy()` copies it (`iterative.cpp:106`). A clone of a model that
   has never trained therefore copies indeterminate memory — the exact pattern
   of the settled decision *"'Not copied' must be WRITTEN in `copy()`, never left
   out"*, and of the `Matrix` value-initialisation misdiagnosis, whose real cause
   was an uninitialised `Model::errorType` scalar.
2. `getIterations()` is public and is read by `RegressNet` for its audit trail
   (`regressnet.cpp:77`, `:794`). Whether that can be reached before any training
   has run is **not established** — it needs checking, and I am not claiming
   either way.

**What it costs when it is read.** `Network::engine( type, iteration )` branches
on `t == 0 || t == df()`. An indeterminate `t` takes the conjugate-direction
branch on the very first call, where `lastF` and `lastG` are still empty, and
`dotprod( u, lastF )` then reads `df()` doubles out of an empty vector. Before
D5 that was a silent out-of-bounds read; it is now a thrown `SizeMismatch`,
which is the only reason this was noticed at all.

**Fixed 2026-08-01** in its own commit, not folded into the DRY extraction — the
same rule that kept legacy bug #12 separate. `iteration ( 0 )` is now first in
the initialiser list. It defines the **starting** value only: `train()` still
owns iteration progression, `trainSet()` still increments nothing, and a caller
driving `trainSet()` directly must still set the counter itself, which is why the
tests that establish an optimizer step keep doing so explicitly.

**Proven to fail two ways.** Against the old constructor an ordinary Release
build already failed 7 assertions on this machine — but that is luck, since the
stack could as easily have held zero. The deterministic evidence is a build with
`-ftrivial-auto-var-init=pattern`, which fills a stack-constructed model's
un-initialised members with a recognisable pattern: **the same 7 assertions
fail**, and both builds pass after the fix. (Not ASan, which does not work in
this environment — CLAUDE.md rule 2.) That is the same diagnostic that made the
earlier uninitialised-state defect deterministic.

**`train()` is untouched**, and the goldens say so: all three transcripts
byte-identical.

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

---

## 12. D9 — the bounds policy of the numerical Matrix layer

**This section is the inventory and the design. It is written before `matrix.h`
or `matrix.cpp` is edited**, exactly as §11 was for `vector_ops`, because the
question is again not "add a throw to `row()`" but *what does this class promise
in the build we actually ship*.

**Status: OPEN. Nothing below is implemented.** Sol's instruction (2026-08-01):
record it as its own bounded phase and do it **before the DFA extraction**,
because DFA leans heavily on `Matrix`. It is **not** folded into the
`OneHiddenNet` commits.

### 12.1 The finding

It surfaced from the Commit-2 verification. `Matrix`'s ranged `dotprod` is
`assert`-only, so the claim that BareProp satisfies its range contract could not
be tested in the shipped configuration; an assert-enabled build was needed to
check it, and that build *also* aborted on a defect in
`tests/onehidden/check_onehidden.cpp` that Release had hidden (`174e76e`).
Sol then widened it, correctly: this is not a defect in two methods, it is the
class's policy.

**Exactly two mechanisms enforce a bounds or dimension contract in Release:**
`Matrix::operator()`, which throws `BoundsViolation` on an out-of-range element,
and `includerows`, which throws the same on an out-of-range gather position.
(Counting mechanisms, not overloads, deliberately: `operator()` has a const and
a non-const form and `includerows` has a destination and a returning form, and
counting those separately would muddy the claim in either direction.) Every
other dimension and range contract — row and column access, row and column
replacement, the arithmetic shape checks, the transpose, all nine
`dotprod`/`dotprodt`/`dotprod_row` overloads, the outer product, the
conversions, the column sums — is guarded by `assert` alone.
`CMakeLists.txt` forces `CMAKE_BUILD_TYPE=Release` when none is given, and
`CMAKE_CXX_FLAGS_RELEASE` carries `-DNDEBUG`, so **none of those assertions has
ever executed in the gate chain**. Under `NDEBUG` an invalid argument is not an
exception; it is an out-of-bounds read or write.

**This contradicts standing rule 4 as written**, which says that *both* numerical
classes reject bounds and dimension violations in Release:

> **both** classes reject bounds and dimension violations in Release, where
> asserts vanish: `Matrix::operator()` throws `BoundsViolation`, and every
> `vector_ops` operation that walks two containers in lockstep or indexes one by
> position throws `nvec::SizeMismatch` / `RangeViolation` / `EmptyVector`.

That sentence is true of `vector_ops` since `5c94cd2`, and true of those two
`Matrix` mechanisms out of the whole class. It was written from the element
accessor and generalized to the class. Same defect class as D5 — a measured contract that the shipped binary does
not enforce — and the same origin: a rule stated from the part that was checked.

### 12.2 The inventory

Every public operation of `src/matrix.{h,cpp}`, including the free functions in
the header and the `double` specializations. **Class** is §11's: **A** = a
violation can read or write outside the allocation; **B** = a violation gives a
mathematically undefined *value* but touches no memory it does not own; **—** =
no precondition to violate.

| # | Operation | Precondition | Release today | Class |
|---|---|---|---|---|
| 1 | `Matrix()`, `~Matrix`, copy ctor, `operator=`, `clear`, `fill`, `setHeader`, `rows`, `cols` | none | ok | — |
| 2 | `Matrix( r, c )`, `resize( r, c )` | allocation succeeds | `assert ( data_ != 0 )` only | A′ |
| 3 | `Matrix( r, c, value )` | `nrows != 0` | **throws `BadSize`** | ✓ |
| 4 | `Matrix( Q, Pt )` (outer product) | both vectors non-empty | `assert` | A |
| 5 | `Matrix( filename, ncols )`, `loadfile( filename, ncols )` | file rows and `ncols` nonzero | `assert` | A |
| 6 | `Matrix( report, filename )`, `loadfile( report, filename )` | file non-empty | `assert` | A |
| 7 | `savefile` | none | ok | — |
| 8 | `operator()( r, c )` ×2 | `r < nrows_`, `c < ncols_` | **throws `BoundsViolation`** | ✓ |
| 9 | `submatrix( r1, rN, c1, cN, M )` | `r1 <= rN <= nrows_`, `c1 <= cN <= ncols_`, `M` sized to match | `assert` | A |
| 10 | `submatrix( r1, rN, c1, cN )` | as #9 | via #9 | A |
| 11 | `row( r, v )` | `v.size() == ncols_`, `r < nrows_` | copies `v.size()` from `data_ + r*ncols_` — **reads past the allocation**; this is the one the Debug build caught | A |
| 12 | `row( r )` | `r < nrows_` (the vector is allocated correctly) | via #11 | A |
| 13 | `col( c, v )`, `col( c )` | `v.size() == nrows_`, `c < ncols_` | strided read past the allocation | A |
| 14 | `replacerow( r, v )` | `v.size() == ncols_`, `r < nrows_` | **WRITES past the allocation** | A |
| 15 | `replacecol( c, v )` | `v.size() == nrows_`, `c < ncols_` | **WRITES past the allocation** | A |
| 16 | `addrow( v, bigM )` / `addcol( v, bigM )` (+ the returning forms) | vector matches the fixed dimension; `bigM` is one row/column larger | `assert` | A |
| 17 | `operator+= -= *= /=( const Matrix& )` (4) | identical dimensions | walks both in lockstep past the shorter | A |
| 18 | `operator+ - * /( const Matrix& )` (4) | as #17 | via #17 | A |
| 19 | scalar `+= -= *= /=` and binary scalar forms (8, both operand orders) | none | ok | B (`/=` by a zero scalar) |
| 20 | `operator==`, `operator!=` | none — a size difference returns `false` | ok | — |
| 21 | `t( M_in )` | `M_in.rows() == ncols_ && M_in.cols() == nrows_` | writes outside `M_in` | A |
| 22 | `t()` | none — allocates correctly | ok | — |
| 23 | `dotprod( iVec, oVec )` | `iVec.size() == ncols_`, `oVec.size() <= nrows_` (**prefix, deliberate** — same rule as §11.3) | reads `oVec.size() * iVec.size()` elements from `data_` | A |
| 24 | `dotprod( iVec, oVec, a, b )` | `iVec.size() == ncols_`, `nrows_ == b - a + 1`, `a <= b`, `b < oVec.size()` | as #23, plus writes into `oVec[ a .. b ]` | A |
| 25 | `dotprod( iVec )` | `iVec.size() == ncols_` | via #23 | A |
| 26 | `dotprodt( iVec, oVec )`, ranged, and returning (3) | transposed forms of #23–#25 | as #23 | A |
| 27 | `dotprod_row( D, r, oVec )`, ranged, and returning (3) | `ncols_ == D.ncols_`, `r < D.nrows_`, `oVec` sized | as #23, plus an unchecked row index into `D` | A |
| 28 | `dotprod( B, C )` (matrix product) | `ncols_ == B.rows()`, `C.rows() == nrows_`, `C.cols() == B.cols()` | reads and writes outside both | A |
| 29 | `dotprod( B )` | `ncols_ == B.rows()` | via #28 | A |
| 30 | `outprod( Q, Pt )` | `nrows_ == Q.size()`, `ncols_ == Pt.size()` | **WRITES past the allocation** — and it is on the per-exemplar training path | A |
| 31 | `squared()`, `maxabs()` | none — an empty matrix sums/maxes to `0` | ok | — |
| 32 | `colsums( sums )` | `sums.size() == ncols_` | writes past `sums` | A |
| 33 | `colsums()` | none — allocates correctly | ok | — |
| 34 | `rowindex( v )` | `v.size() == nrows_` | writes past `v` | A |
| 35 | `includecols( M, pos )` / `excludecols( M, pos )` (+ returning forms) | `max(pos) < ncols_`, `pos` unique, `M` sized | `assert` — **and the assert itself dereferences `max_element` on an empty `pos`**, so a Debug build has UB where Release has no check | A |
| 36 | `includerows( M, pos )` (+ returning form) | `max(pos) < nrows_`, `M.nrows_ == pos.size()` | **throws `BoundsViolation`** (added 2026-07-16 with the bootstrap resample) | ✓ |
| 37 | `toVector( v )` | `v.size() == nrows_ * ncols_` | writes past `v` | A |
| 38 | `toVector()` | none — allocates correctly | ok | — |
| 39 | `func( Mi, fx, Mo )` (free) | identical dimensions | writes past `Mo` | A |
| 40 | `func( Mi, fx )` (free) | none — allocates correctly | ok | — |
| 41 | `toMatrix( M_in, v_in )` / `toMatrix( v_in, r, c )` (free) | `v_in.size() == r * c` | reads past `v_in` | A |
| 42 | `operator<<` (free) | none | ok | — |
| 43 | `operator>>` (free) | `nrows_ != 0 && ncols_ != 0` | reads into a zero-sized matrix — no write, but silently consumes nothing | B |
| 44 | `random( n )` | none | ok | — |
| 45 | `covariance( V )` | `V` is `ncols_ × ncols_`; `nrows_ > 0`, `ncols_ > 1` | writes outside `V`; division by `nrows_ - 1` | A |
| 46 | `covariance()` | as #45's second half | via #45 | A |
| 47 | `inverse` (6 overloads), `inverseGaussJordan`, `inverseLU`, `ludcmp`, `lubksb`, `determinant` | square, and the destination the same shape | `assert` for shape; **`Singular` is already thrown** for the numerical failure | A (shape) / ✓ (singularity) |
| 48 | `begin()`, `end()` ×2 | documented "DO NOT USE THESE" | raw pointers, by design | out of scope |

Counting overloads, **roughly 60 public entry points, of which about 45 can read
or write outside an allocation in the shipped build.** Two mechanisms already
enforce a bounds or dimension contract and are the precedent to follow —
`operator()` and `includerows`, both throwing `BoundsViolation`. `BadSize` and
`Singular` are enforced in Release as well, but they report a construction
failure and a numerical one; they are the precedent for *how* this class raises
a refusal, not for bounds coverage.

### 12.3 What this phase must NOT do

- **Do not change `BoundsViolation`'s inheritance.** It is a nested class with
  `: public std::exception` deliberately commented out, as `BadSize` and
  `Singular` are. Whether these should derive from `std::exception` is a real
  question with catch-site consequences across the GUI, the CLI and the tests;
  it is not part of a bounds-enforcement phase. Sol's instruction, adopted.
- **Do not normalize the `<=` / `==` asymmetry.** `dotprod( iVec, oVec )` takes
  `oVec.size() <= nrows_` on purpose — the same prefix rule §11.3 settled for
  `vector_ops`, and the feed-forward bias arithmetic depends on it.
- **Do not add checks to a hot loop's inner body.** These are entry-point
  preconditions, checked once per call, exactly as `vector_ops` does it. The
  per-exemplar callers (`outprod`, `row`, the ranged `dotprod`s) are called once
  per exemplar, not once per element, so the cost is a comparison against a
  member — but this must be *measured* on the scale probe before and after, not
  assumed (rule 7).
- **Do not fold it into another commit**, and do not start it before the DFA
  work is explicitly sequenced after it.

### 12.4 The shape of the work, in order

1. **This inventory** (done, above) — reviewed before any code changes.
2. **Characterize under Release, in two halves that must not be conflated.**

   - **The contracts that already hold** — `operator()`'s `BoundsViolation`,
     `includerows`' `BoundsViolation`, `Matrix( r, c, value )`'s `BadSize`, and
     the `Singular` throws of the `inverse` / `ludcmp` / `lubksb` family. These
     **pass today and must keep passing**: they are the existing behavior this
     phase may not regress, and they belong in the characterization for that
     reason, not as red-test evidence.
   - **The contracts that are absent** — the Class-A rows of §12.2. Each of
     these is written as an expectation of a throw, and each must be **watched
     failing against today's `matrix.h`** before any policy is implemented. That
     is the D5 pattern, and it is what distinguishes an absent contract from an
     untested one.

   Stating it as "every expectation fails" would have been wrong and is
   corrected here: the first half cannot fail, and if it ever does, the phase has
   broken something rather than proved something.
3. **The smallest coherent exception policy.** `BoundsViolation` already exists
   and already means "an index or range is outside this matrix". A dimension
   mismatch between two objects is a different fact and probably wants its own
   type beside `BadSize`; that decision belongs in the design step, with the
   catch sites enumerated first.
4. **Then implement**, one commit, with the goldens, the oracle, `smoke.sh`,
   `run_tools.sh` and the scale probe as the evidence that nothing legal changed.
5. **Then rewrite standing rule 4 again.** Its second paragraph currently records
   the measured gap (corrected in `6f327a6`, before any code — the claim was
   false and could not be left standing while the phase ran). When `Matrix`
   reaches the goal, that paragraph collapses back into the single sentence about
   both numerical classes, which will then be true. D5b did the same for
   `CMakeLists.txt`'s comment.

**Only then, DFA.**

### 12.5 How the red tests are executed, so that one crash cannot hide the rest

Sol's third correction, and it is a test-*design* constraint rather than a
wording one. An absent contract in this class does not politely return without
throwing: it reads or writes outside an allocation. Several cases can therefore
**corrupt memory or terminate the process** under Release. Running them
sequentially in one executable would let an early overflow hide every later
case — and a suite that stops at case 3 and reports nothing about cases 4–45 is
the vacuous-comparison hole in another costume.

The mechanism `check_vector_bounds` already established (§11.8) is the one to
reuse, because it was built for exactly this:

- **One case per process.** With no argument the binary runs every case and is an
  ordinary `ctest` target; with `check_matrix_bounds <n>` it runs precisely one.
  A driver runs each dangerous case in its own process and records its exit
  status, so a crash is a *recorded result for that case* rather than the end of
  the run.
- **A valid control first, in every process.** Each case performs a legal
  operation of the same family and asserts the correct answer **before** it makes
  the invalid call. A case that dies before its control proves nothing about the
  contract — it proves the harness is broken — and this is what separates the
  two.
- **Positive controls as their own cases**: boundary-valid indices, single-row
  and single-column matrices, the deliberate `<=` prefix of
  `dotprod( iVec, oVec )` with a shorter destination (§11.3's rule, which must
  keep working), and the already-enforced contracts of step 2's first half.
- **`NDEBUG` asserted by the test itself**, as `check_vector_bounds` does, so the
  guarantee cannot silently become "we happened to build Debug".

**Order of evidence**: control passes → invalid call watched **failing to throw**
(or crashing) against today's `matrix.h`, per case → policy implemented → every
case now throws the expected type → **then sabotage the implemented policy** and
watch the corresponding cases go red. The last step is the one that proves the
new checks are what the tests are reading, and it is not optional.

### 12.6 The characterization, measured (2026-08-01)

`tests/matrix/check_matrix_bounds.cpp`, 52 cases, built with `NDEBUG` — the file
refuses to compile without it — and run **one process per case** by
`tests/matrix/run_matrix_bounds_demo.sh`. Every case runs a legal operation of
its own family and checks the answer *before* the invalid call.

```
held:            9
no exception:   39
wrong type:      0
crashed:         4
control failed:  0
```

**No control failed**, so every verdict below is about a contract rather than
about the harness.

**Read "held: 9" carefully — it is 8, not 9** (Sol's correction). Eight are the
declared positive contracts. The ninth is **case 50**, an ABSENT contract that
happened to throw the placeholder type from `operator()` downstream; case 52
shows the same call succeeding on a matrix shaped the other way. The honest
totals are **8 contracts held and 44 absent** — 39 silent, 4 fatal, 1
incidentally throwing. See §12.8.

**The eight HOLDS cases all held**, as they must: `operator()` on a row, a column
and the const overload; `includerows` on a gather position; `BadSize` on a
zero-row construction; `Singular` from the inverse family; and the two positive
controls — the deliberate `<=` prefix of `dotprod( iVec, oVec )` with a shorter
destination, and ordinary arithmetic / transpose / dot product answers.

**Thirty-nine absent contracts returned with no exception at all.** They are the
Class-A rows of §12.2, and the list is now executable rather than a reading:
`row` (index and destination width), `col` (index and destination height),
`replacerow` and `replacecol` (index — the *write* cases), `submatrix` (range and
destination), all four compound operators and the binary form, `t`, `dotprod`
(input length, over-long destination, both ranged violations), `dotprodt` and its
ranged form, `dotprod_row` (row index and disagreeing widths), the matrix product
(inner dimensions and destination shape), `colsums`, `rowindex`, `toVector`,
both `toMatrix` forms, the free `func`, `includecols` (bad position and the empty
vector), `excludecols`, `addrow`, `addcol`, `covariance`, and the outer-product
constructor on empty vectors.

**Four killed their process outright**, all with **SIGTRAP** (exit 133) and no
message of their own — which is exactly why the driver reads the case list up
front and records exit status rather than trusting the program to report:

| case | operation |
|---|---|
| 15 | `replacerow` with a source wider than the matrix — **writes** past the allocation |
| 36 | `outprod` with a left vector longer than `nrows_` — **writes**, and this is the per-exemplar training path (`hWup.outprod( h_err, I )`) |
| 37 | `outprod` with a right vector longer than `ncols_` — **writes** |
| 48 | `addcol` with a column taller than the matrix |

**Two results that change what the policy has to say.**

1. **`inverse( I )` on a non-square matrix is safe only by accident.** Case 50 (a
   2 × 3 input) **threw `BoundsViolation`** — but not from a precondition. LU
   decomposition loops over `ncols_` and indexes rows by that bound, so a wider
   than tall matrix runs `operator()` off the end and the one contract this class
   does enforce refuses it. Case 52 is the same call on a 3 × 2 matrix: **more**
   rows than the loop bound, every index in range, and it **runs to completion**
   on a matrix that has no inverse. The pair is the evidence that the safety is
   incidental. A shape check cannot be skipped on the grounds that "it throws
   anyway".
2. **`includecols` with an empty position vector does not fail in Release** (case
   45). The pathology is in the guard, not the operation: the `assert`
   dereferences `max_element` over an empty range, so a **checked** build has
   undefined behavior exactly where Release has no check at all. Whatever the
   policy does here, it must not reproduce that shape.

### 12.7 The catch sites, enumerated — and the fact they force into the design

Sol asked for these before any policy is proposed. They are decisive, and they
are the reason this section stops here rather than recommending a type.

**`Matrix`'s three exception classes do not derive from `std::exception`.** In
`matrix.h` each is written `class BoundsViolation /* : public std::exception */`
— inheritance present, commented out, for all of `BadSize`, `BoundsViolation` and
`Singular`. `nvec::SizeMismatch`, `RangeViolation` and `EmptyVector`, added by
D5, **do** derive from it.

Every site that catches a `Matrix` exception today, in the whole repository:

| Site | Catches | Notes |
|---|---|---|
| `src/ldfa.cpp:71`, `src/qdfa.cpp:85`, `src/logistic.cpp:323` | `Matrix< double >::Singular&` | by exact type; reports "singular matrix" and returns |
| `src/matrix.cpp:455, 541, 576` | `Singular&` | internal, inside the inverse family |
| `tests/matrix/check_matrix.cpp:120` | `Matrix< double >::BoundsViolation&` | the only `BoundsViolation` catch anywhere |

**Nothing in `src/` catches `BoundsViolation`.** And the generic handlers cannot:

- `src/gui.cpp:1070` (`dfa->train()`) and `src/gui.cpp:1150`
  (`modelPtr->train()`) catch **`const exception&`**. A `BoundsViolation` is not
  one, so it passes straight through both.
- The GUI's long jobs run on a **`std::thread`** (`src/gui.cpp:1469`, `:1777`).
  An exception that escapes a thread's function calls `std::terminate`. So an
  uncaught `BoundsViolation` inside async training, OBD, stepwise or CV **kills
  the server process**, with the page still polling a status endpoint that will
  never answer.
- Synchronous handlers are luckier: `third_party/httplib.h:7158` wraps routing in
  `catch (...)` and returns **500 with `EXCEPTION_WHAT: UNKNOWN`** — no message,
  because the `std::exception` branch above it is the one that reads `what()`.
- `src/neuron.cpp`'s `main` has no top-level catch; the CLI would terminate.

**This is not hypothetical, and it predates D9.** `Network::computeCondNum`
already throws `Matrix< double >::BoundsViolation` on a non-square argument
(`src/network.cpp:711`), and `src/utility.h:101` documents that `train()` can
throw it. One live throw site with no catcher is a latent defect; **forty-three
more, spread across the class the whole engine computes with, is a design
decision about process lifetime** — and it turns "silent memory corruption" into
"the server dies", which is better but is not automatically acceptable and is not
mine to choose quietly.

So the policy question is larger than picking a type name, and it has at least
these parts, all of which Sol's step 3 should settle together:

1. Whether `Matrix`'s exceptions derive from `std::exception`. **Deferred by Sol
   and still deferred here** — but the enumeration above shows it is load-bearing
   rather than cosmetic: it is the difference between the GUI reporting "training
   failed: …" and the GUI process dying.
2. Whether a cross-object **dimension mismatch** is `BoundsViolation` (an index
   outside *this* matrix) or a new sibling of `BadSize`. The characterization
   currently expects `BoundsViolation` everywhere, which is a placeholder, not a
   recommendation — 26 of the 39 are dimension mismatches, not bad indices.
3. Which boundaries acquire a catch, and what each reports.

Nothing in `matrix.h` or `matrix.cpp` has been changed.

### 12.8 The per-case exception mapping (Sol's policy, applied case by case)

The policy, decided by Sol 2026-08-01 and not reopened here:

- **`BoundsViolation`** — an invalid index, range or gather position: one number
  outside one container.
- **`DimensionMismatch`** — *new* — two shapes that cannot be combined:
  matrix/vector/destination disagreements, **including a non-square input to the
  inverse family**.
- **`BadSize`** — an invalid construction size.
- **`Singular`** — numerical non-invertibility.
- All four derive **directly from `std::exception`**, with
  `const char* what() const noexcept override`. Not from a more specialized
  standard exception. Existing exact-type catches stay valid.

**A correction to how §12.6 reported its total, per Sol.** The aggregate line
"held: 9" is eight declared positive contracts **plus case 50**, and case 50 is
not a ninth. It is an ABSENT contract that happened to throw the placeholder type
from `operator()` *downstream* — case 52, the same call on a taller-than-wide
matrix, runs to completion. Read the totals as **8 contracts held, 44 absent**
(39 silent + 4 fatal + 1 incidentally-throwing), 0 wrong type, 0 control failed.

| # | Operation and violation | Type | Today |
|---|---|---|---|
| 1 | `operator()`: row index | `BoundsViolation` | held |
| 2 | `operator()`: column index | `BoundsViolation` | held |
| 3 | `operator() const`: both | `BoundsViolation` | held |
| 4 | `includerows`: gather position | `BoundsViolation` | held |
| 5 | `Matrix( 0, c, value )` | `BadSize` | held |
| 6 | `inverse`: a singular matrix | `Singular` | held |
| 7 | `dotprod( in, out )`: a SHORTER destination | **legal** | held |
| 8 | ordinary arithmetic, transpose, dot product | **legal** | held |
| 9 | `row( r, v )`: row index | `BoundsViolation` | silent |
| 10 | `row( r, v )`: destination width ≠ `ncols_` | `DimensionMismatch` | silent |
| 11 | `row( r )`: row index | `BoundsViolation` | silent |
| 12 | `col( c, v )`: column index | `BoundsViolation` | silent |
| 13 | `col( c, v )`: destination height ≠ `nrows_` | `DimensionMismatch` | silent |
| 14 | `replacerow`: row index | `BoundsViolation` | silent |
| 15 | `replacerow`: source width ≠ `ncols_` | `DimensionMismatch` | **SIGTRAP** |
| 16 | `replacecol`: column index | `BoundsViolation` | silent |
| 17 | `replacecol`: source height ≠ `nrows_` | `DimensionMismatch` | silent |
| 18 | `submatrix`: row range past the end | `BoundsViolation` | silent |
| 19 | `submatrix`: destination ≠ the block | `DimensionMismatch` | silent |
| 20 | `operator+=`: differing dimensions | `DimensionMismatch` | silent |
| 21 | `operator-=`: differing dimensions | `DimensionMismatch` | silent |
| 22 | `operator*=`: differing dimensions | `DimensionMismatch` | silent |
| 23 | `operator/=`: differing dimensions | `DimensionMismatch` | silent |
| 24 | `operator+`: differing dimensions (inherits) | `DimensionMismatch` | silent |
| 25 | `t( M_in )`: destination ≠ transpose shape | `DimensionMismatch` | silent |
| 26 | `dotprod( in, out )`: input length ≠ `ncols_` | `DimensionMismatch` | silent |
| 27 | `dotprod( in, out )`: destination LONGER than `nrows_` | `DimensionMismatch` | silent |
| 28 | `dotprod` ranged: extent ≠ `nrows_` | `DimensionMismatch` | silent |
| 29 | `dotprod` ranged: range outside the destination | `BoundsViolation` | silent |
| 30 | `dotprodt( in, out )`: input length ≠ `nrows_` | `DimensionMismatch` | silent |
| 31 | `dotprodt` ranged: extent ≠ `ncols_` | `DimensionMismatch` | silent |
| 32 | `dotprod_row`: row index into the dataset | `BoundsViolation` | silent |
| 33 | `dotprod_row`: the two matrices disagree | `DimensionMismatch` | silent |
| 34 | `dotprod( B, C )`: inner dimensions | `DimensionMismatch` | silent |
| 35 | `dotprod( B, C )`: destination shape | `DimensionMismatch` | silent |
| 36 | `outprod`: left vector ≠ `nrows_` | `DimensionMismatch` | **SIGTRAP** |
| 37 | `outprod`: right vector ≠ `ncols_` | `DimensionMismatch` | **SIGTRAP** |
| 38 | `colsums`: destination ≠ `ncols_` | `DimensionMismatch` | silent |
| 39 | `rowindex`: destination ≠ `nrows_` | `DimensionMismatch` | silent |
| 40 | `toVector( v )`: destination ≠ `nrows_ * ncols_` | `DimensionMismatch` | silent |
| 41 | `toMatrix( M, v )`: source ≠ `rows * cols` | `DimensionMismatch` | silent |
| 42 | `toMatrix( v, r, c )`: source ≠ `r * c` | `DimensionMismatch` | silent |
| 43 | `func( Mi, fx, Mo )`: the two matrices disagree | `DimensionMismatch` | silent |
| 44 | `includecols`: position past the last column | `BoundsViolation` | silent |
| 45 | `includecols`: an EMPTY selection | **legal** — see below | silent |
| 46 | `excludecols`: position past the last column | `BoundsViolation` | silent |
| 47 | `addrow`: new row width ≠ `ncols_` | `DimensionMismatch` | silent |
| 48 | `addcol`: new column height ≠ `nrows_` | `DimensionMismatch` | **SIGTRAP** |
| 49 | `covariance( V )`: destination ≠ `ncols_` × `ncols_` | `DimensionMismatch` | silent |
| 50 | `inverse( I )`: a WIDER-than-tall matrix | `DimensionMismatch` | held, **incidentally** |
| 51 | `Matrix( Q, Pt )`: empty vectors | `BadSize` | silent |
| 52 | `inverse( I )`: a TALLER-than-wide matrix | `DimensionMismatch` | silent |
| **53** | `covariance`: fewer than two columns (**new case**) | `BadSize` | to be measured |
| **54** | `excludecols`: an EMPTY selection (**new case**) | **legal** — see below | to be measured |
| **55** | `covariance`: a single row (**new case**) | `BadSize` | to be measured |

Totals: **14 `BoundsViolation`**, **32 `DimensionMismatch`**, **4 `BadSize`**,
**1 `Singular`**, **4 deliberately legal** — 55 cases.

The counts moved twice under Sol's review, and both moves are recorded because
each was a wrong classification rather than a wording choice: `includecols( {} )`
left `BadSize` for **legal** (below), and `covariance`'s two *intrinsic* size
failures are `BadSize` rather than `DimensionMismatch` — a source matrix too
small to have a covariance is not two shapes that disagree, it is one shape that
is invalid. Its *destination* shape (case 49) stays `DimensionMismatch`.

**The empty include/exclude selections, resolved explicitly** — Sol's
instruction. **Both are legal**, and neither may inherit its behavior from the
broken `max_element` assert:

- **`includecols( {} )` returns an *n* × 0 matrix.** I first proposed refusing
  this as `BadSize`, arguing that a zero-column matrix is unusable. **That was
  wrong, and the class says so in its own words.** `matrix.h:337-343`:

  > *"I had to relax these conditions for stepwise regression of a network
  > without biases, as the weights of the baseline network (that with no input
  > nodes) will be a Matrix with 0 columns"*

  — and the `ncols != 0` half of both the assert and the `BadSize` throw is
  commented out beside it. Zero columns is a **supported** shape with a live
  reason, so "keep no columns" has a valid answer and refusing it would break the
  bias-free stepwise baseline. Case 45 pins the shape of the result.
- **`excludecols( {} )` returns an unchanged copy.** "Exclude nothing" is a
  well-defined request, and refusing it would break a caller whose removal list
  legitimately comes out empty — `removeInputs` is exactly that shape. Case 54
  pins it.

**`covariance` gets a row minimum it never had.** `matrix.cpp:44` asserts
`nrows_ > 0 && ncols_ > 1`, and the last line of the same function is
`V /= static_cast< double >( nrows_ - 1 )`. A **single-row** matrix therefore
passes the guard and divides by zero — the sample covariance of one observation
does not exist, and the code produces infinities instead of saying so. The
contract becomes `nrows_ >= 2 && ncols_ >= 2`, both `BadSize`, and case 55 is the
row half. It was found by writing this table, not by the harness, because no case
had asked.

**Deliberately not cased, with reasons.** `Matrix( filename, ncols )`,
`loadfile`, and `operator>>` have preconditions about file contents and about a
zero-sized destination; their failure is a construction size and is already the
`BadSize` contract, and casing them needs fixture files whose absence would test
the fixture rather than the class. The allocation asserts (`data_ != 0` in the
constructors and `resize`) are not caller-reachable contracts. `begin()` /
`end()` are documented "DO NOT USE THESE" and stay out of scope.

**What does not move**, restated because the mapping is where it would slip:
the short-destination prefix rule of `dotprod( iVec, oVec )` (case 7) stays
legal, and case 27 refuses only the *longer* destination.

### 12.9 The process boundary, proposed shape

Sol's policy: `Matrix` methods never catch their own contract exceptions; every
GUI worker-thread entry point stops an exception escaping the thread, publishes a
structured failure, and always clears `job.running`; one narrow helper rather
than four catch blocks; `catch ( const std::exception& )` for the message and
`catch ( ... )` as the process-survival boundary; an equivalent CLI boundary; and
the exact `Singular` catches stay where singularity is an expected analytical
outcome.

**Why a helper and not four blocks.** All four launch sites are the same six
lines today (`src/gui.cpp:1469`, `:1777`, `:2872`, `:3103`):

```cpp
job.worker = thread( [ cfg ]
{
    string result = runObdJob( cfg );
    {
        lock_guard< mutex > lock( job.progressMutex );
        job.result = result; // publish BEFORE running goes false
    }
    job.running = false;
} );
```

The publish-then-clear ordering is an invariant the status endpoint depends on,
and it is currently maintained in four places. Proposed:

```cpp
// THE WORKER BOUNDARY -- one implementation, four callers.
//
// Every long job runs on a std::thread, and an exception that escapes a
//    thread's function calls std::terminate: the server dies while the page is
//    still polling /api/train/status. Until D9 that was reachable --
//    Matrix's contract exceptions did not derive from std::exception, so
//    runTrainingAndBuildResult's catch( const exception& ) could not see one
//    (Network::computeCondNum throws BoundsViolation on a non-square argument).
//    They derive from it now, so the inner handlers report properly; this is
//    the last resort, and it exists so that the answer to "what if something
//    else throws" is never "the process".
//
// It also owns the publish-then-clear ordering that the status endpoint
//    depends on -- job.result under the mutex FIRST, job.running cleared
//    afterward, on every path including the throwing ones.
static void runOnWorker( const function< string() >& body )
{
    string result;

    try
    {
        result = body();
    }
    catch ( const exception& e )
    {
        result = jsonMsg( false, string( "the run failed: " ) + e.what() );
    }
    catch ( ... )
    {
        result = jsonMsg( false, "the run failed with an unrecognized error" );
    }

    {
        lock_guard< mutex > lock( job.progressMutex );
        job.result = result; // publish BEFORE running goes false
    }
    job.running = false;
}
```

and each site becomes one line:

```cpp
job.worker = thread( [ cfg ] { runOnWorker( [ cfg ] { return runObdJob( cfg ); } ); } );
```

Notes on the shape, each of which is a decision rather than a detail:

- **`std::function` is correct here** and not a rule-7 violation: this is called
  once per job, not once per exemplar. The hot path is inside `body`.
- **`jsonMsg` escapes its own message**, so the handler must not escape it again.
- **The inner handlers stay.** `runTrainingAndBuildResult`'s
  `catch ( const exception& )` gives the *specific* message and the captured
  output; this boundary is the last resort, not a replacement for it.
- **Cancellation is not an exception** in this engine (it is a flag the observer
  reads), so nothing here changes cancellation behavior.
- **It is not the async-launcher refactor.** The joinable/reset/lock sequence
  above each launch is untouched; only the thread body moves.

**The CLI boundary** (`src/neuron.cpp`) is the same idea at the other entry
point. `main` has no top-level catch today, so a contract failure terminates
opaquely. The minimal form keeps the menus untouched:

```cpp
static int neuronMain( int argc, char* argv[] ) { ...today's main body... }

int main( int argc, char* argv[] )
{
    try { return neuronMain( argc, argv ); }
    catch ( const exception& e )
    {
        cerr << "neuron: fatal: " << e.what() << endl;
        return 1;
    }
    catch ( ... )
    {
        cerr << "neuron: fatal: unrecognized error" << endl;
        return 1;
    }
}
```

**The inheritance change, and a claim of mine that was false.** Making the four
classes derive from `std::exception` means every existing
`catch ( const exception& )` starts seeing `Matrix` failures it previously let
past. At `src/gui.cpp:1070` and `:1150` that is a strict improvement — a DFA or
training run that hits a contract now returns `{"ok":false,"message":"..."}`
instead of httplib's `500 EXCEPTION_WHAT: UNKNOWN`.

I then wrote that `src/autoalgo.cpp:116` and `src/crossval.cpp:86-90` "would now
swallow a `Matrix` contract failure that today crosses them". **That is wrong.**
Both are `catch ( ... )`, which catches every exception regardless of what it
derives from; they have always swallowed these. Sol caught it. The inheritance
change does nothing at either site, and the sentence is corrected here rather
than quietly deleted because it was an argument for leaving them alone, and the
argument was empty.

**The real problem at those two sites is worse than the one I claimed, and this
phase fixes it.** Both convert *any* exception into a domain answer:

- `src/autoalgo.cpp:116` — *"a diverged probe is a result, not a failure"* —
  turns it into an unusable optimizer probe, so a `Matrix` contract failure is
  silently reported as "that optimizer diverged" and the search continues with
  the others.
- `src/crossval.cpp:86-90` — `try { m.az = ts.getStatROCarea(); } catch ( ... ) {}`
  — turns it into an absent metric, so a fold whose arithmetic violated a
  contract is reported as a fold whose AUC was not estimable.

A programmer-contract failure is neither of those things. Per Sol, both sites
**rethrow `BoundsViolation`, `DimensionMismatch` and `BadSize` before** their
existing catch-all, and keep their present handling of genuine numerical and
statistical failures — including `Singular`, which at these sites legitimately
means an unavailable fit or an unavailable metric. The contract errors then reach
the worker and CLI boundaries with their messages intact, which is the whole
point of giving them messages.



### 12.10 D9 implemented, and what it proved (2026-08-01)

`Matrix::DimensionMismatch` added; all four classes now derive from
`std::exception` with `what() const noexcept override`. Every `assert` that
guarded a caller-reachable precondition in `matrix.h` and `matrix.cpp` became a
typed throw **at the entry point** — none inside an element loop. **All 55 cases
hold**: 0 silent, 0 crashed, 0 wrong type, 0 control failed, against 8 held and
44 absent before. `check_matrix_bounds` is now a registered `ctest` case, which
is the line that changed when the policy landed.

**Four sabotages of the implemented policy**, each failing a distinct set:

| sabotage | result |
|---|---|
| drop `row()`'s index check | 2 cases, both `row` overloads, `NO THROW` |
| make `dotprod( in, out )`'s prefix rule an equality | **1 case, and it is the CONTROL** — verdict `CONTROL FAILED`, the distinct exit status doing its job |
| throw `DimensionMismatch` where `submatrix` should throw `BoundsViolation` | 1 case, `WRONG TYPE` — the types are discriminated, not merely "something threw" |
| remove `runOnWorker`'s handlers, with a fault injected in `runTrainJob` | the **server process dies** (`libc++abi: terminating due to uncaught exception`), where with the boundary it survives |

**The boundary, demonstrated end to end.** A `Matrix< double >::DimensionMismatch`
temporarily injected into `runTrainJob`, then `/api/train&async=1`:

```
{"ok":true,"running":false,"series":{...},
 "result":{"ok":false,"message":"the run failed: Matrix dimension mismatch"}}
```

`running` cleared, the result published carrying `what()`, and `/api/version`
still answering. The CLI the same way: a throw injected at the top of
`neuronMain` gives `neuron: fatal: Matrix bounds violation` and **exit 1**, where
before it terminated with no message at all. Both injections were reverted; they
are demonstrations, not tests — see the gap below.

**It found a defect immediately, in a test.** `check_autostep` failed the moment
the contract existed: its `Probe::signature()` passed the **DataSet's** training
matrix to `forward()`, whose last column is the outcome. For `BareProp` the row
copy was short and now throws; for every biased model the width happened to match
and the **label was being read into the bias slot**. That is the same defect
`check_onehidden` had, fixed a day earlier by an assert-enabled build — the
helper had been copied. `check_bpoptimizer` carried it too and was silent on both
counts. All three now use `Model::Train`. **No engine code needed to change**,
which is the outcome a behavior-preserving hardening should have: the contract
found three tests lying about what they measured, and nothing in the engine
violating it.

**The remaining gap, stated rather than papered over.** The worker and CLI
boundaries have **no automated test**. `runOnWorker` lives in `gui.cpp`, which is
not in a library, so nothing can link it; and no endpoint can be made to fail a
`Matrix` contract on purpose, so `smoke.sh` cannot reach it either. The evidence
above is a manual injection, recorded here with its exact output. Making it
permanent needs either a fault hook in the GUI or `runOnWorker` extracted to a
linkable unit, and both are larger than this phase. It is the one mechanism added
here that standing rule 2 does not yet guard, and it should be closed before the
next thing leans on it.


### 12.11 The performance measurement §12.3 required (2026-08-01)

Standing rule 7: *measure before you change an algorithm for speed* — and the
same obligation runs the other way, for a change that might cost speed. `row`,
the ranged `dotprod` and `outprod` are per-exemplar operations, so the entry
checks had to be measured on real training rather than argued about.

**`./build/scale_probe` was not used, and would not have answered this.** It
measures the clustered-AUC path — placements, tie handling and contrast algebra
over generated cluster data. It does not train a network, so it exercises none
of the three operations in question. Sol's condition, met by not relying on it.

**The measurement.** Both binaries built `Release` from the same generator, the
old one in a separate worktree at `31880dc^` so the working tree was never
touched. Identical dataset (4000 rows × 20 inputs, deterministic — no RNG in its
construction), identical seed 42, identical architecture (SimpleProp 2-12-1
sizing at 20 inputs), identical iteration ceiling of 3000, driven through the
HTTP API so that algorithm, batch mode, seed and ceiling are all set explicitly.
Timing covers the `/api/train` request only; dataset load and model construction
sit outside it. Seven trials per configuration, **interleaved** old/new so that
machine drift cannot land on one side.

**The two binaries do identical work**, which is what makes the comparison a
timing comparison: every run of configuration A ends at final error
`1.306535e-01` and every run of B at `1.748305e-01`, on both binaries.

**How much of the new code the runs executed.** 4000 rows × 3000 iterations =
**12,000,000 exemplar passes**, each one calling `row( r, I )`, the ranged
`hW.dotprod( I, hO, 0, nHidden - 1 )` and `hG.outprod( h_err, I )` — the
separate-gradient branch, which is the default because gradient stopping is on.
That is **36 million newly-checked entry points per run**; configuration B adds
`toVector` and `toMatrix` 3000 times each through `pack`/`unpack`.

| configuration | pre median | post median | delta | run-to-run spread |
|---|---|---|---|---|
| **A** on-line canonical | 8458 ms | 8442 ms | **−16 ms (−0.19%)** | pre 95 ms, post 266 ms |
| **B** batch CGD | 8495 ms | 8530 ms | **+35 ms (+0.41%)** | pre 163 ms, post 256 ms |

Subtracting the 510 ms of fixed overhead (server start plus dataset load,
measured at `maxiter=5`), the training-only deltas are **−0.20%** and
**+0.44%**.

**Reading it honestly: this is noise, not a small regression.** Both deltas are
smaller than the spread within either binary's own seven trials, and **they point
in opposite directions** — a real per-exemplar cost would appear in both
configurations with the same sign, since both run the same three checked
operations 12 million times. What the measurement establishes is an upper bound:
whatever the checks cost, it is under half a percent and under the run-to-run
variance of this machine. No optimization is warranted, and none was done.

**Why that is the expected result** rather than a lucky one: each check is a
comparison of two `unsigned` members already in cache, at the entry point of an
operation that then walks hundreds of doubles. The checks were deliberately not
placed inside any element loop (§12.3), which is the decision this measurement
confirms was the right one.
---

## 12.12 The DFA characterization, before any extraction (2026-08-01)

Written first, per Sol, and pushed on its own with no implementation change.
`tests/dfa/check_dfa.cpp`, 44 assertions across LDFA and QDFA, binary and
three-class. Portable by construction: structure (which lines a report has),
integers (row counts, distinct-score counts), same-process relations, and
inequalities a sign error would break. No multi-iteration float literals.

**What the six sabotages proved, each failing a distinct set:**

| sabotage | fails |
|---|---|
| flip the LINEAR discriminant's sense (`d1-d0` → `d0-d1`) | 2, both LDFA direction assertions, area 0.000 |
| flip the QUADRATIC's sense | 2, both QDFA direction assertions |
| revert the graded score to a hard 0/1 decision | 1 — "2 distinct scores, not a 0/1 decision" |
| let DFA keep a pointer to the caller's `DataSet` and write through it | 1 — the network's guesses move |
| remove `metricsReport` from the 1-output path | 4, including the last-operation file's content |
| silence the singular refusal | 1 |

**The direction pin is the one that matters for the extraction.** LDFA takes the
larger discriminant, QDFA the smaller, because one is a similarity and the other
a distance. A flipped sense does not crash, does not change the number of scores,
and leaves an entirely normal-looking report — with the area on the wrong side of
0.5. That is precisely what a shared `largerWins()` flag would risk, and the two
sabotages above show the guard bites in both directions.

**Two things the characterization changed about my own expectations.**

1. **QDFA's guesses are not strictly inside (0,1), and that is correct.** My first
   assertion said they were; it failed. Measured on this fixture: LDFA spans
   0.0502–0.9666 with 120 distinct scores and none saturated; **QDFA saturates 20
   of 120 to exactly 1.0**, with 101 distinct. The quadratic margin is a
   difference of Mahalanobis distances plus log-determinant constants, reaching
   the tens, and `sigmoidal()` of that is exactly 1.0 in double precision. The
   assertion was wrong, not the code. The contract pinned is the **closed** range
   plus graded-ness; the saturated count is deliberately *not* asserted, because
   it comes through a matrix inverse and a borderline row could move in the last
   bits on another toolchain. It is recorded because it costs ROC resolution at
   the top, and because anyone "simplifying" the graded score should see it.
2. **My first standalone test was weak, and a failed sabotage is what showed it.**
   It ran the analysis on a `DataSet` the network did not own, so value semantics
   made the assertion true structurally and no realistic sabotage could reach it —
   my first attempt crashed on an unrelated bounds violation instead of failing
   the assertion. The test now hands the DFA **the network's own `DataSet`**,
   which is the form a caller can actually reach (the GUI takes both from the same
   place), and the pointer sabotage then fails exactly the intended assertion.

**The singular fixture was WRONG, and CI on two platforms is what found it.**
`db435f7` passed everything locally on macOS and failed `dfa_characterization`
on **Ubuntu and Windows**: LDFA did not refuse the fixture there, so both the
`"Can't do LDFA"` assertion and the "no accuracy reported" assertion failed.
(QDFA refused on Ubuntu — equally by luck.)

*Why the original fixture was not deterministically singular.* It made column 1
exactly `2 x` column 0: singular in exact arithmetic, but with **no zero row**.
`ludcmp` has two singularity tests — a **structural** one, `big == 0`, when a
whole row is zero, and an **arithmetic** one, `L( j, j ) == 0`, after
eliminating. A collinear pair can only reach the second, and whether the
residual cancels to exactly zero depends on how the covariance sums were
accumulated: FP contraction into FMA, vectorization and reassociation all differ
between clang on arm64, gcc on x86-64 and MSVC. The fixture was therefore a bet
on one toolchain's rounding, dressed as mathematics.

*Why the replacement is.* A **zero-variance column** reaches the structural test
instead. Every deviation from that column's mean is exactly `0.0`, so every
covariance product involving it is exactly `0.0` and every sum of them is exactly
`0.0` — true under any IEEE rounding mode, with or without FMA, in any summation
order, because `0*x = 0` and `0+0 = 0` are exact. The constant is `0.0` rather
than any other value so the column's sum is exactly zero for **any** row count,
making the mean exactly zero without an argument about representability.
Measured: `V(0,0) = 0.225009...`, `V(0,1) = V(1,1) = 0` exactly. It is a real
degeneracy too — a predictor identically zero in the training data is the case
`DataSet::normalize` already documents as bug B4.

*And the fixture now proves itself.* Four assertions check, on whatever platform
is running, that the covariance row is exactly zero, that the other column still
carries variance, that inverting throws `Matrix::Singular`, and that it is **not**
some other exception — so the two refusal cases can never again pass for the
wrong reason or fail mysteriously. The suppression sabotage was re-run against
the new fixture: silencing LDFA's refusal fails exactly its assertion, and
silencing QDFA's fails exactly its own.

*Two process notes, both mine.* I pushed `db435f7` without waiting for CI, having
just been told to wait for it on the previous commit; the failure was public for
that reason. And `setRawMatrix` / `setTrainMatrix` turn out **not** to normalize
at all — I had reasoned about `inLowerLimit` affecting the constant column before
measuring, and the measurement showed the raw value passes straight through.

**One sabotage that changed nothing, and why that is informative.** Adding a
`metricsReport` call to the multi-output path did not move a single assertion —
because `DataSet::metricsReport` refuses more than one output at its own entry
and writes nothing. So the "no ROC for multi-output" property is enforced by
`DataSet`, not by the DFA classes, and an extraction cannot lose it by accident.
Worth knowing before designing the shared orchestration.

---

## 13. The DFA extraction — PROPOSAL ONLY, nothing implemented

Written after the characterization (`ad7ab14`, green on all three platforms) and
before any implementation change, to Sol's eight-point specification. **No source
file is modified by this section.**

### 13.1 What was measured, not assumed

Both method pairs were diffed with comments and whitespace discounted:

| pair | LDFA | QDFA | code lines that differ |
|---|---|---|---|
| `train()` | 54 lines (36 code) | 61 lines | **18** — and every one of them is the FIT |
| `reportAccuracy()` | 112 lines (70 code) | 111 lines | **22** — and every one of them is a DISCRIMINANT or `max_element`/`min_element` |

That split is the whole proposal: `train()`'s difference is a block that runs
**once per run**, and `reportAccuracy()`'s difference is inside loops that run
**once per exemplar**. They are therefore treated separately, per Sol's item 6.

### 13.2 Ownership table (item 1)

**Moves into `DFA` — the once-per-run scaffold, and nothing else.**

| Element | Why it may move |
|---|---|
| the two `ostringstream`s and their flush | Pure reporting mechanics; byte-identical in both. |
| `"I'm running " << objType << ":"` | The object's own name, which it already owns. Not a parameterization of mathematics — the same device `OneHiddenNet::randomize()` already uses. Produces `"I'm running LDFA:"` and `"I'm running QDFA:"` exactly, and the characterization asserts both strings. |
| `outputHeader( screenStream )` | Already `DFA`'s own method, called identically. |
| the `try` / `catch ( Matrix< double >::Singular& )` and `"Can't do " << objType << ": " << e.what()` | Identical control flow; identical message modulo `objType`. The characterization asserts both texts and that no accuracy follows a refusal. |
| the single `reportAccuracy( screenStream )` call | Today it is written twice, once inside each arity branch of each model — four copies. The fit no longer branches at the scaffold level, so one call after the fit is the same order of operations. |
| `screenStream << endl`, the file-stream copy, `util::screen()` write | Identical. |
| `addHistory( fileStream )`, `writeLastop( fileStream.str() )`, `return -1` | Identical; `Model` already owns both writers. |

**Stays in `LDFA` — the linear fit, entire.**

`C = Train.covariance()`, `S = C.inverse()`, and both constant forms, written
out: `K0 = 0.5 * dotprod( U0, S.dotprod( U0 ) ) - log( P0 )` and its `U1`
sibling, and the n-output `K[ o ] = 0.5 * dotprod( U[ o ], S.dotprod( U[ o ] ) )
- log( P[ o ] )`. **One pooled covariance** is a property of the linear model and
appears nowhere else.

**Stays in `QDFA` — the quadratic fit, entire.**

`C0 = D0.covariance()`, `C1 = D1.covariance()`, `S0 = C0.inverse( Det0 )`, its
`S1` sibling, `K0 = log( Det0 ) - log( P0 )`, `K1`, and the n-output loop over
`C.push_back( D[ o ].covariance() )` etc. **Per-class covariances and their
determinants** are the quadratic model's, and the determinant term is why its
constants have a different form.

**Stays in both, untouched by this commit — `reportAccuracy()` in full**, both
copies, including every discriminant expression and both `max_element` /
`min_element` selections. See §13.5.

### 13.3 Signatures and call flow (item 2)

Two declarations change. One is added.

```cpp
// dfa.h
class DFA : public Model {
public:
    // Was pure virtual. Becomes the shared scaffold -- NON-virtual, because
    //    there is nothing left for a subclass to override in it.
    double train();                                   // was: = 0

    virtual void reportAccuracy( ostream& ) = 0;      // UNCHANGED

protected:
    // The fit, and only the fit: covariances, inverses, constants. Runs ONCE
    //    per train(). Each model writes its own published formulae here.
    virtual void fitDiscriminant() = 0;               // NEW
};
```

```cpp
// ldfa.h / qdfa.h -- train() disappears from both; fitDiscriminant appears
    virtual void reportAccuracy( ostream& );          // unchanged
protected:
    virtual void fitDiscriminant();                   // replaces train()
```

Call flow, once per run, top to bottom:

```
caller -> DFA::train()                       non-virtual
            ostringstream setup
            "I'm running " << objType << ":"
            outputHeader( screenStream )     virtual, 1x   (already virtual today)
            try {
              fitDiscriminant()              virtual, 1x   (NEW)
              reportAccuracy( screenStream ) virtual, 1x   (already virtual today)
            }
            catch ( Matrix<double>::Singular& )
              "Can't do " << objType << ": " << what()
            endl; file copy; screen write
            addHistory( fileStream )
            writeLastop( fileStream.str() )
            return -1
```

One behavioral ordering note, stated because it is the only thing that moves:
today `reportAccuracy` is called *inside* each arity branch, immediately after
that branch's constants are computed; afterwards it is called once, immediately
after `fitDiscriminant()` returns. The sequence fit-then-report is unchanged, and
so is what the `try` block covers.

### 13.4 Virtual calls, and their frequency (item 3)

| call | virtual? | executions per `train()` |
|---|---|---|
| `outputHeader` | yes, already | 1 |
| `fitDiscriminant` | yes, **new** | 1 |
| `reportAccuracy` | yes, already | 1 |

**One new virtual call per run.** Nothing is introduced inside any loop:
per-exemplar dispatch is prohibited and this proposal adds none, because the
per-exemplar code is not being shared at all (§13.5). `DFA::train()` itself
becomes non-virtual — a subclass has no reason to override the scaffold, and
leaving it virtual would invite exactly that.

### 13.5 The per-exemplar loops are NOT shared here (item 6)

`reportAccuracy()`'s 22 differing lines are the discriminants themselves and the
`max_element` / `min_element` selection, all inside loops over exemplars. Every
mechanism that could share those loops is disqualified:

- **A virtual `score( X )` per exemplar** — per-exemplar virtual dispatch.
  Prohibited outright, and it is also rule 7's named failure mode.
- **A template or CRTP on a scoring functor** — zero-overhead, but it moves the
  discriminant out of `LDFA::reportAccuracy` into a functor and makes
  larger-versus-smaller a template parameter. That is the comparator flag in
  another costume; the formulae stop being readable where the model is.
- **A shared loop with a passed comparator or sign** — explicitly forbidden.

So the recommendation is to **leave both `reportAccuracy` bodies exactly as they
are**, and revisit them in the later measured scoring phase, where the question
is posed as performance with a measurement attached rather than as line count.
The honest consequence: this extraction removes far less duplication than
§2.4/§2.5 imagined, because most of the duplication is in the half that may not
be shared. That is the correct outcome, not a shortfall.

*One sub-option, offered rather than assumed.* Inside the multi-output branch of
each `reportAccuracy` there is an identical **once-per-run** block: size
`testClasses`, take the output submatrix, call `rowindex`, and print
`"Sorry, that test set had bad output columns."` on failure. It is ~8 identical
lines and touches no discriminant. It could become
`bool DFA::classesFromTestOutputs()`. It is *not* included in the recommendation
above because it edits the same function this section otherwise defers, and
Sol may prefer one function per commit. Sol's call.

### 13.6 The mathematics stays visible (items 4 and 5)

After the extraction, reading `ldfa.cpp` shows, with nothing between them and the
reader:

- the pooled covariance and its inverse;
- `K0 = 0.5 * dotprod( U0, S.dotprod( U0 ) ) - log( P0 )`;
- in `reportAccuracy`, unchanged: `d0 = dotprod( U0, S.dotprod( X ) ) - K0`,
  `d1 = ...`, `sigmoidal()( d1 - d0 )`, and `max_element` — **the larger
  discriminant wins**.

and `qdfa.cpp` shows:

- the per-class covariances, their inverses and determinants;
- `K0 = log( Det0 ) - log( P0 )`;
- in `reportAccuracy`, unchanged: `d0 = dotprod( X - U0, S0.dotprod( X - U0 ) ) +
  K0`, `d1 = ...`, `sigmoidal()( d0 - d1 )`, and `min_element` — **the smaller
  discriminant wins**.

**No `largerWins()`, no sign flag, no comparator, no formula descriptor, no
generic indexing convention, no type switch** appears anywhere in `DFA`. The only
thing the shared scaffold reads from the concrete class is `objType`, which is
its name, and the only thing it calls is `fitDiscriminant()`, which is the
model's own code. `DFA` after the change contains no discriminant arithmetic at
all — it contains streams, a `try`, and three virtual calls.

### 13.7 Sabotage plan (item 7)

Each newly shared mechanism, with the characterization assertions expected to go
red. To be run and recorded before the commit is offered.

| # | sabotage of the shared scaffold | expected failures |
|---|---|---|
| 1 | hard-code the banner to `"I'm running LDFA:"` for both | 1 — *QDFA: the report names itself* |
| 2 | hard-code the refusal to `"Can't do LDFA: "` | 1 — *QDFA: a singular covariance is refused* |
| 3 | remove the `catch ( Singular& )` entirely | both refusal cases, and the run terminates rather than reporting — 4 assertions plus a non-zero exit |
| 4 | call `reportAccuracy()` **before** `fitDiscriminant()` | the direction and separation assertions for both models (4), since the constants are not yet computed |
| 5 | drop `writeLastop( fileStream.str() )` | 2 — the last-operation content assertions, for both models |
| 6 | drop `addHistory( fileStream )` | 2 — the history-file content assertions |
| 7 | give `DFA` a non-virtual `fitDiscriminant` so `QDFA`'s is never reached | QDFA's direction, separation and multi-output accuracy assertions |

Sabotage 4 is the important one: it is the only way the scaffold could silently
break the *order* the models depend on, and nothing but the direction assertions
would notice.

### 13.8 Expected reduction and documentation (item 8)

**Source.** `LDFA::train` (54 lines) and `QDFA::train` (61) are replaced by
`DFA::train` (~40 with its comments) plus `LDFA::fitDiscriminant` (~22) and
`QDFA::fitDiscriminant` (~32). **Net ≈ −21 lines**, and one authoritative copy of
the scaffold instead of two. Deliberately modest: the large duplication is in
`reportAccuracy`, which §13.5 declines to touch. If the §13.5 sub-option is
included, a further ~8 lines.

**Documentation, in the same commit**: `refactor_audit.md` item 11 marked with
what shipped and what was deferred and why; the Manifest — `DFA` currently has no
Chapter 12 object entry at all, only a line in the architecture chapter's model
list, so this adds one covering `train()` as the shared scaffold,
`fitDiscriminant()` and `reportAccuracy()` as the per-class contracts, and the
statement that the discriminant mathematics is per-class by policy — plus a
`docs/manifest.pdf` rebuild with the affected pages inspected; `docs/HISTORY.md`;
and `CLAUDE.md` only if the roadmap line changes.

### 13.9 A separate possible defect, recorded and NOT touched here

Sol asked for this to be recorded rather than quietly fixed. It is worse than
suspected, so the measurement is given in full.

**Measured, by a read-only probe (no engine source touched), on a 3-class
fixture where `nOutput = 3` and every vector should stay at 3:**

```
QDFA after setDataSet:      D=3 U=3
QDFA after train 1:         C=3 S=3 K=3 Det=3
QDFA after train 2:         C=6 S=6 K=6 Det=3
QDFA after train 3:         C=9 S=9 K=9 Det=3
QDFA after 2nd setDataSet:  D=6 U=6
LDFA after train 1..3:      K=3 K=3 K=3          <- bounded
LDFA after 2nd setDataSet:  D=6 U=6
```

Two distinct accumulations:

1. **`QDFA::train()` multi-output** appends to `C`, `S` and `K` with
   `push_back` on every run. `Det` does not grow because it uses
   `Det.resize( nOutput )`. `LDFA::train()` does not grow because it uses
   `K.resize( nOutput )` and indexed assignment — so this is QDFA's alone.
2. **`DFA::setDataSet()` multi-output** appends to `D` and `U` with `push_back`
   on every call. This one is in the **shared base**, so it affects LDFA and QDFA
   equally.

**The consequence is not merely unbounded memory — it is a stale model.** Every
reader indexes `[ 0 .. nOutput - 1 ]`, so after a second load the leading entries
are still the *first* dataset's class matrices and means. Measured, same object,
loading a genuinely different 3-class dataset B after dataset A:

| | reused object on B | fresh object on B | (A alone) |
|---|---|---|---|
| LDFA training accuracy | **37.3%** | 100% | 100% |
| QDFA training accuracy | **30.0%** | 100% | 100% |

Chance is 33%. The reused object is fitting dataset A and scoring dataset B.

**Reachability, stated precisely.** No shipped path currently reloads different
data into an existing DFA object: `/api/dfa` constructs a fresh model per request
(`gui.cpp`), `cvadapters::dfaProcedure` constructs a fresh model per fold, and the
CLI's `dfa()` constructs `ldfaObj` / `qdfaObj` on entry to the submenu, which is
the only scope in which `*dataPtr` cannot change. What **is** reachable today is
the growth itself: repeating LDFA or QDFA from that CLI submenu grows `D` and `U`
(and, for QDFA, `C`, `S`, `K`) on every run, and every run after the first
silently reports the *first* run's fit — identical numbers, because the data
cannot change within that scope. Same shape as D1: real, reachable, not currently
producing a wrong answer on any shipped path, and one refactor away from doing so.

**And it exposes a weakness in my own characterization.** `test_train_twice_multi`
asserts that a second run reports the same thing — and it passes **because of**
this defect rather than despite it. The assertion is satisfied by staleness. Sol
identified exactly that: repeated reports being identical does not prove internal
state is bounded. A correctness commit must add an assertion on the *state*, not
only on the output.

**Disposition, per Sol: do not clear or otherwise fix this during the
behavior-preserving extraction.** It is a separate correctness commit, and it
should come with its own characterization — a bounded-state assertion, and a
reload-different-data assertion that fails against today's engine. Whether it
lands before or after the scaffold extraction is Sol's call; the extraction does
not touch `setDataSet` or either fit, so the two do not collide.

*Prepared by Claude (Opus 5). Reviewed by Sol (2026-07-31); revised §8 in response.
§9–§12 are the implementation record: §§8.7–8.8 and 9–10 describe work now
committed, §11 is the D5 design, written before its code, and §12 is D9's,
written the same way and now implemented (§12.10).*
