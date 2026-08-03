# B9 — the GUI-wide strict-parsing pass

ROADMAP 4's last open item. This file is the inventory, the measurements that
produced it, and the settled parser contract. It is written the way
`refactor_audit.md` is written: a record of what was measured and what was
decided, not a plan to be re-derived.

**The defect in one line.** Every numeric request field in `src/gui.cpp` is
converted with `atol` or `atof`, which stop at the first character they cannot
use and report nothing, so `folds=5junk` is five folds, `fraction=abc` is a
request for a train/test split that silently produces no test set, and
`maxiter=4294967297` trains for one iteration and then reports that it "hit the
maximum iteration count". Booleans are compared against a single literal, so
`async=true` runs synchronously and `discrete=false` declares a discrete
outcome.

---

## 1. Measured, not read

Everything in §2 was produced by driving a live pre-B9 server
(`build/neuron --gui`, HEAD `a8c07a7`) with the low-birth-weight dataset and
recording what came back. Rule 3 applies to inventories as much as to plans: a
reading of `atol`'s contract is a hypothesis about this program, and two of the
entries below (`fraction=1e`, `/api/regress threshold=nan`) did not behave the
way the source suggested at first reading.

Reproduce with `tests/gui/strictparse.sh`, which pins every row.

### The conversion sites

Counted from the current source rather than repeated from the roadmap's
historical figure — which, as it happens, is still right:

| what | count |
|---|---|
| `atol` / `atof` calls in `src/gui.cpp` | **51** |
| `strtol` / `strtod` / `stoi` / `stod` / stream extraction of a request field | **0** |
| handler-local numeric parser lambdas (`uintParam`, `fracParam`) | **3**, in 2 handlers |
| handler-local boolean parser lambdas (`boolParam`) | **1** |
| string comparisons against a literal that decide a **boolean** | **18** |
| string comparisons against a literal that decide an **enum** (`algorithm`) | **2** |
| numerically converted field-instances across all handlers | **55** (50 scalar, 5 comma-separated lists) |
| boolean field-instances across all handlers | **21** (16 distinct names) |

One stream extraction exists and is deliberately **out of scope**:
`handleModel` reads `f >> backpropBias` from a saved *network file*'s second
line (`src/gui.cpp:880`). That is model-file format, not a request field;
changing it would change what files the program can load.

---

## 2. The inventory

Legend for **empty**: what a present-but-empty value means today.
`—` means the field has no special empty handling and falls into the numeric
conversion, which yields 0.

### `/api/load`

| field | kind | domain checked today | omitted | empty | pre-B9 misparse (measured) |
|---|---|---|---|---|---|
| `mode` | enum | `raw`, else train | train | train | unknown token silently means train |
| `file`/`path`, `testfile`/`testpath` | file | exists | — | — | — |
| `inputs` | unsigned | `<1` → derive from columns | derive | derive | `5junk` → 5 |
| `outputs` | unsigned | `<1` → 1 | 1 | 1 | `-1` → huge → "too few columns" |
| `fraction` | double | engine's | 0.0 | 0.0 | `abc` → 0.0 → **ok:true, no test set**; `0.3junk` → 0.3; `1e` → 1.0 |
| `discrete` | bool | `!= "0"` | true | true | `false` → **true** |
| `threshold` | double | `(0,1)` | skip | skip | `0.5junk` → 0.5 |
| `in_lower`, `in_upper` | double | `lo < hi` | engine default | engine default | `-inf`/`inf` → **accepted** |
| `out_lower`, `out_upper` | double | `lo < hi` | engine default | engine default | as above |
| `history` | bool | `== "1"` | skip | skip | `true` → false |
| `roc_report` | enum | `both`\|`either` | skip | skip | validated already |
| `roc_min` | unsigned | `>= 2` | skip | skip | `2junk` → 2 |
| `test_n` | unsigned | none | skip | skip | `20junk` → 20; `-5` → huge → engine refusal |
| `strata`, `group` | list | 1..nInput | skip | skip | `parseColumnList` uses `atol` per token |
| `strata_bins` | unsigned | `>= 2` | skip | skip | `abc` → 0 → domain message for a syntax fault |
| `val_fraction` | double | `(0,1)` | skip | skip | `0.2junk` → 0.2 |
| `val_n` | unsigned | `>= 1` | skip | skip | needs `test_n` |

### `/api/model`

| field | kind | domain checked today | omitted | empty | pre-B9 misparse |
|---|---|---|---|---|---|
| `type` | enum | `logistic`\|`simpleprop` | error | error | validated already |
| `mode` | enum | `load`, else create | create | create | unknown token silently creates |
| `errfunc` | enum | `xentropy`\|`lms`, else default | default | default | `xent` → **silently the default** |
| `hidden` | list | each `>= 1` | error | error | `3junk` → 3 |
| `bias` | bool | `== "0"` → off | true | true | `false` → **bias on** |
| `log_lastop`, `log_history` | bool | `== "0"` → off | true | true | `no` → **on** |

### `/api/dfa`
`type` (enum, validated), `log_lastop`, `log_history` (bool, as above).

### `/api/randomize`
`seed` (unsigned, no domain check): `99junk` → 99, `-1` → 4294967295.

### `/api/train`

| field | kind | domain checked today | omitted | empty | pre-B9 misparse |
|---|---|---|---|---|---|
| `algorithm` | unsigned or `auto` | `1..3` | 0 → error | error | `1x` → 1 |
| `seed` | unsigned | none | skip | skip | as `/api/randomize` |
| `maxiter` | unsigned | `>= 1` | error | error | `4294967297` → **1** |
| `async` | bool | `== "1"` | false | false | `true` → **blocking** |
| `batch_epoch` | bool | `!= "1"` refused for logistic | keep | off | `true` → off |
| `autostop` | bool | `== "1"` | off | off | `true` → off |
| `autostop_tol` | double | `(0,1)` | 1e-4 | 1e-4 | read only when `autostop=1` |
| `autostop_window` | unsigned | `>= 2` | 100 | 100 | as above |
| `eta` | double | `(0,1]` | keep | keep | `0.5junk` → 0.5; `nan` refused |
| `weight_decay` | bool | `== "1"` | keep | off | `true` → off |
| `decay` | double | `[0,1]` | 5e-5 | 5e-5 | read only when `weight_decay=1` |
| `minerr`, `change` | double | `(0,1)` | keep | **disables** | `0.5junk` → 0.5 |
| `errwindow` | unsigned | `> 1` | keep | **disables** | `5junk` → 5 |
| `gradmax` | double | `> 0` | keep | **disables** | `inf` → **accepted**, a stop that can never fire |
| `printcount` | unsigned | `>= 1` | keep | **no-op** | `3junk` → 3 |
| `logprint` | bool | `== "1"` | keep | off | `true` → off |
| `autostep` | bool | `== "1"` | keep | off | `true` → off |

`printcount` present-and-empty is a **no-op**, not a disable — the only field
where present-empty differs from the four stopping conditions beside it. That
asymmetry is preserved.

### `/api/obd`
`hidden_start`, `hidden_max`, `iter_budget`, `sample_every`,
`early_stop_patience`, `grow_patience` (unsigned, `>= 1`, via `uintParam`);
`autostop_window` (`>= 2`); `early_stop_tol`, `prune_tol`, `autostop_tol`
(double, `(0,1)`, via `fracParam`); `algorithm` (`1`|`2`|`3`|`auto`, empty →
`1`); `seed` (unsigned). Measured: `hidden_max=4junk` and `sample_every=2junk`
both start a search; `iter_budget=-1` and `early_stop_tol=inf` are refused.

### `/api/cv`
`folds` (`>= 2`), `seed` (`>= 0`), `maxiter` (`>= 1`), `neural_hidden` (`>= 1`),
`hidden_max` (`>= 1`), `iter_budget` (`>= 1`) via a **second copy** of
`uintParam`; `autostop_tol` `(0,1)`; `autostop_window` `>= 2`; `algorithm`
(empty|`auto`|`1`|`2`|`3`); `logistic`, `ldfa`, `qdfa`, `neural`, `neural_obd`
(bool via `boolParam`, which accepts `1` **or** `true`); `inner_val` `(0,1)`;
`strata`, `group` (lists), `strata_bins` `>= 2`; `independence` (enum, already
strict); `locked_fraction` `[0,1)`; `locked_n` (`>= 0`); `primary`,
`reference` (enum, case-insensitive). Measured: `folds=5junk` starts a 5-fold
run, `inner_val=0.25junk` and `maxiter=10x` are accepted, `logistic=yes` and
`logistic=TRUE` both mean **false**, `seed=-1` and `locked_n=-1` are refused.

### `/api/regress`
`structure` (`util::variable_parse`), `direction` (enum, validated),
`threshold` (double, guarded by `<= 0 || >= 1`), `async` (bool).

**`threshold=nan` passes both guards** — every comparison with NaN is false —
and reaches `RegressNet`. Measured. `threshold=inf` is refused by `>= 1`.

### `/api/save/:what`, `/api/train/status`, `/api/train/stop`, `/api/stats`, `/api/version`
No numeric or boolean request fields. `:what` is a path token compared against
a fixed list.

---

## 3. The settled contract

### Where it lives
`util::` (`src/utility.{h,cpp}`), the lowest layer both the GUI and any future
caller already depend on. Syntax and representation only: the parsers know
nothing about HTTP, JSON, or which endpoint asked.

### The parsers

```cpp
enum class ParseStatus { Ok, Empty, Syntax, Trailing, Negative, Range, NotFinite };

ParseStatus parseUnsigned( const string& text, unsigned& out );
ParseStatus parseDouble  ( const string& text, double& out );
ParseStatus parseBool    ( const string& text, bool& out );

string unsignedError( const string& field, const string& text, ParseStatus );
string numberError  ( const string& field, const string& text, ParseStatus );
string boolError    ( const string& field, const string& text, ParseStatus );
```

`out` is written **only** on `Ok`, so a refused field can never leave a caller
holding a half-converted value.

### Whitespace — settled explicitly, not by accident
Leading and trailing spaces and tabs are **trimmed**, then the remainder must
be consumed entirely. This is what the pre-B9 code already accepted
(`atol`/`atof` skip leading whitespace and stop at trailing whitespace;
`parseColumnList` and `parseLayers` trim each token by hand), so no request
that works today stops working, while `5junk` and `5 junk` both become
`Trailing`. Interior whitespace is never permitted.

### Booleans — `1` and `0`, case-sensitive, everywhere
Approved 2026-08-03. The GUI page sends only `1` and `0`
(`src/gui_page.html`), and `AGENTS.md` documents only `1` and `0`. Any other
token is `Syntax` and is refused by name. This **narrows one undocumented
path**: `/api/cv logistic=true` works today and becomes an error. It is the
only currently-valid request the pass changes; every other boolean change
turns a silent misread into a refusal.

An absent boolean keeps its documented default (on for `bias`, `discrete`,
`log_lastop`, `log_history`; off for the rest). A **present but empty**
boolean is `Empty` and is refused — today it means false for `async`-style
fields and true for `discrete`-style ones, which is not a contract anyone
could have relied on deliberately.

### Non-finite doubles are refused
`nan`, `inf`, `-inf` and their signed and mixed-case variants are `NotFinite`.
No field's contract permits one: `gradmax=inf` is a stopping rule that cannot
fire and `in_lower=-inf` is not a variate bound. The two fields that appear to
refuse them today do so by accident — a NaN fails `lo < hi` and an infinity
fails `>= 1` — and `/api/regress threshold=nan` shows what happens when the
accident does not occur.

### Overflow and underflow
An unsigned value above `UINT_MAX` is `Range`. A floating value that overflows
to `HUGE_VAL` is `Range`. A floating value that **underflows** yields the
closest representable value, which may be zero, and is accepted: every field
whose domain excludes zero rejects it at the domain check one line later, and
the alternative would be to refuse a legitimately tiny tolerance. Negative
text for an unsigned field is `Negative`, distinct from `Syntax`, so the
message can say what is actually wrong.

### Locale
The program never calls `setlocale`, so the C locale is in force and
`strtod`'s decimal separator is `.` on every platform. Recorded rather than
assumed; a future `setlocale` call would change field parsing and must not be
added without revisiting this paragraph.

### Errors identify the field
Every refusal names the field and quotes the text it was given, e.g.

```
folds: '5junk' is not a whole number
gradmax: 'inf' is not a finite number
async: 'true' must be 1 or 0
maxiter: '4294967297' is out of range
```

Domain messages that exist today (`"folds must be at least 2"`,
`"autostop_tol must be between 0 and 1"`, …) are **unchanged**: a syntax fault
and a domain fault are different faults and now read differently.

### Parse before apply
`handleTrain` already validates every field before applying any of them, and
`handleLoad` builds a local `DataSet` it installs only on success. That
ordering becomes a stated requirement rather than an accident: each handler
parses all of its fields into locals first, and only then mutates engine
state, so a malformed later parameter cannot leave a coherent-looking partial
mutation. The five stopping-condition setters in `handleTrain` that re-parsed
their text at the apply site now consume the already-parsed values.

### What stays in the handlers
The domain policy — which minimum, which interval, which combination is a
conflict — stays legible at the call site. There is no descriptor table, no
reflection, no callback framework and no flag-driven mega-parser. The two
copies of `uintParam` and the one `fracParam` become one file-scope helper
each, built on the parsers above; `tools/check_strict_parsing.py` fails the
build if a permissive conversion or a handler-local parser copy returns.

---

## 4. Out of scope, recorded as findings

- **Enum tokens that silently default.** `errfunc=xent` uses the default error
  function; `mode=trian` loads a training set; `/api/model mode` anything-but-
  `load` creates. These are `==` comparisons, not conversions, and refusing
  them changes currently-accepted requests on three endpoints. Decided
  2026-08-03 to record and not change here.
- **`f >> backpropBias`** reads a saved network file, not a request (§1).
- **The CLI menus are frozen.** `util::askI` / `askD` keep their own prompting
  loops; the new parsers are available to them but no CLI interaction is
  rewritten merely because they exist.

---

## 5. As landed

`src/utility.{h,cpp}` gained `ParseStatus`, `parseUnsigned`, `parseDouble`,
`parseBool`, `unsignedError`, `numberError`, `boolError` — 130 lines, in
`neuron_core`, so the unit test links neither GSL nor the engine.

`src/gui.cpp` gained five file-scope readers — `given`, `readUnsigned`,
`readDouble`, `readBool`, plus `uintField` / `fracField` for the two domain
shapes that were already written out twice each — and lost all 51 permissive
conversions, the three parser lambdas, the `boolParam` lambda, and every
hand-written boolean comparison against a literal. Every handler migrated:
`/api/load`, `/api/model`, `/api/dfa`, `/api/randomize`, `/api/train`,
`/api/obd`, `/api/cv`, `/api/regress`. Net: **+412 / −239**.

Three deliberate consequences beyond the parsing itself:

- **`parseLayers` returns a message instead of an empty vector.** A hidden-layer
  spec that held an unreadable token used to collapse the whole vector and the
  caller could only say "one or more positive integers"; `hidden=3,junk` now
  names the token.
- **`/api/load` reads `strata_bins`, `val_n` and `val_fraction` before it opens
  the file**, so a malformed split control is refused without the cost of a load.
- **The apply sites stopped re-converting their text.** Five of `handleTrain`'s
  setters parsed the same field a second time where they applied it, which is
  two places for one field's meaning to live.

### The gates it added

| gate | what it holds |
|---|---|
| `ctest` case `strict_field_parsers` | the parser contract itself — 185 checks |
| `tests/gui/strictparse.sh` | that the **handlers** use it — 215 checks, ~1 s |
| `tools/check_strict_parsing.py` | that neither comes back — run by `tests/tools/run_tools.sh`, so by CI on all three platforms |

The three are not redundant. A parser test passes while every handler still
calls `atol`; an endpoint test passes while the gate is absent and the next
handler reintroduces one; the gate passes while the parsers are wrong.

### Sabotage evidence

Every case below was applied, the affected object file **deleted**, the build
re-run while **requiring the compile line in the log**, the test run, then the
source restored, the object deleted **again**, rebuilt with the same
requirement, and the control re-run. An inert edit carrying only the marker was
run first as a harness control and correctly reported NOT CAUGHT.

Parsers (`check_parse`):

| # | sabotage | result |
|---|---|---|
| S0 | inert marker (harness control) | not caught — correct |
| S1 | full-consumption check removed | caught, 15 checks |
| S2 | leading `-` allowed through to `strtoull` | caught, 5 |
| S3 | `UINT_MAX` ceiling removed | caught, 7 |
| S4 | `!isfinite` refusal removed | caught, 21 |
| S5 | ERANGE treated as failure for any result | caught, 3 |
| S6 | `parseBool` extended to accept `true` | caught, 4 |
| S7 | destination written before the status was decided | caught, 12 |

Handler migration (`strictparse.sh`) — none of these is visible to the parser
test, which is the point:

| # | sabotage | result |
|---|---|---|
| M1 | the unsigned reader stops reporting its status | caught, 48 checks |
| M2 | the boolean reader returns to `text != "0"` | caught, 34 |
| M3 | `given()` stops distinguishing present-but-empty | caught, **17 — all of them PIN rows** |
| M4 | one handler left unmigrated (`/api/train` `maxiter`) | caught, 4 |
| M5 | a non-finite value forwarded as a success | caught, 12 |
| M6 | a domain check moved below the first setter | **NOT caught at first** |

Repository gate (`check_strict_parsing.py`): a returning `atol`, an inline
boolean test on a `param()` result, a handler-local parser lambda, a deleted
reader and an unused reader were each introduced and each caught.

**M6 is the finding.** The parse-before-apply case as first written sent a
*syntax* fault in a later field, and a syntax fault is refused by the reader at
the top of the handler whatever the ordering — so it could not see a domain
check that had been moved below a setter. The case that can see it sends a
*domain* fault (`eta=0.6&printcount=0`) and then asks the engine's own run
header which learning rate it is using. Adding it turned M6 from NOT CAUGHT
into caught. M3 is the mirror image and the reason the PIN rows exist: breaking
the empty-field distinction failed seventeen positive controls and not one
malformed-input assertion.

### Behaviour intentionally changed for malformed requests

Everything in §2's "pre-B9 misparse" column now refuses by name. Beyond that,
three changes are worth stating plainly because they alter requests that were
previously *accepted*:

1. `/api/cv logistic=true` (and the other four procedure flags) — the approved
   narrowing, §3.
2. **A present-but-empty flag is refused** where it previously meant true for
   `discrete`-style fields and false for `async`-style ones.
3. **Non-finite doubles are refused everywhere**, including `gradmax=inf` and
   `in_lower=-inf`, which were accepted.

Two refusal *messages* changed without any change in what is refused: a negative
`locked_n` and a negative `seed` on `/api/cv` were already refused by a domain
comparison and now refuse as sign faults naming the field. And when a request
carries two bad fields, `/api/obd` and `/api/cv` now report the **first**; they
used to report the last, because each lambda overwrote the previous one's
message.
