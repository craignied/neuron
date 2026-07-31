# Evaluation report specification (ROADMAP 4 Phase 4)

> **STATUS (2026-07-23): this is the ASPIRATIONAL spec; the shipped single-run CV report
> (`src/cvreport.{h,cpp}`, reached via `/api/cv`) implements a subset.** SHIPPED: the
> three-tier structure; Tier-1 headline table (Procedure · AUC(CV) mean ± sd · Arch · Time)
> + verdict block + the standing caveat; Tier-2 per-fold AUC/sens/spec, per-procedure
> failures (with reasons) and validFolds, and OBD architecture- AND optimizer-selection
> frequency; Tier-3
> `cv_predictions.csv` / `cv_metrics.csv` (with a `status` column and both `n_valid`/`n_total`) / `cv_run.json`
> (with per-procedure `validFolds` + `failures`). A Tier-3 file that cannot be written (unwritable
> directory, full disk) is reported as a run WARNING naming the file + reason — never silently
> dropped, and never counted as written unless it opened/wrote/flushed/closed cleanly.
> **SHIPPED 2026-07-24 — the locked-test layer (inference is OPT-IN):** with
> `locked_fraction`/`locked_n` on `/api/cv`, an outcome-stratified ROW holdout is scored for
> point AUCs and predictions (`cv_locked_predictions.csv`; a `lockedTest` block in
> `cv_run.json`). Ordinary DeLong assumes *independent* observations, which a mechanical row
> holdout does not establish, so the AUC 95% CIs and the contrast *p* are produced **only when
> the caller declares `independence=rows`**; otherwise inference is **withheld** (point AUCs
> still shown) and the metadata records why. When inference runs, Tier 1 gains the
> `AUC (test) [95% CI]` column and the prespecified DeLong verdict (ΔAUC + two-sided *p* →
> significant/not; a deterministic area separation reports p≈0, equal areas "no testable
> difference"); Tier 2 gains a locked-test section with sampling-unit/inference metadata; the
> "frozen architecture" appears as each procedure's locked-test `arch`.
> **SHIPPED 2026-07-30 — fold policy and cluster-aware inference:** `/api/cv` takes its own
> `strata=`/`strata_bins=`/`group=`, so the fold plan may be outcome-stratified (the default,
> unchanged), outcome × covariate-stratified, or **group-disjoint** — and a grouped request's
> key governs the locked holdout and the nested search's inner validation split as well.
> `independence=cluster` runs **Obuchowski's clustered ROC covariance** (the cluster, not the
> row, is the independent unit); it never falls back to ordinary DeLong, and
> `independence=rows` over a grouped design is refused rather than relabelled. The design is
> machine-readable (`foldDesign` / `lockedTest.splitDesign`, with per-fold rows/events/groups,
> imbalance and largest-group counts recomputed from the final assignment), and both
> prediction files share ONE raw-row identity space so they can be joined.
> **NOT yet implemented (aspirational below):**
> Tier-2 calibration; per-fold timing; Tier-3 download buttons; a fully metric-agnostic
> Tier 1; composing `strata=` **and** `group=` into a single fold plan (refused today — there
> is no tested joint balancing objective).
> **Format note:**
> `cv_predictions.csv` ships as **one row per exemplar with one prediction column per
> procedure** (a paired wide format), NOT the "one row per (exemplar, procedure)" the body
> describes. Its `exemplar` is the ORIGINAL raw row id, the same identity space as
> `cv_locked_predictions.csv`'s `row`, so the two files join. And Tier-2 detail is returned
> only when the async CV job finishes; it does not stream fold-by-fold.

An automated evaluation run — several procedures, k outer folds, nested OBD, a per-exemplar
out-of-fold prediction for every patient — generates a great deal of output. Without a
deliberate structure it becomes a wall of numbers no one reads. This spec fixes the structure.

It is **general** — no dataset is the mold: every column and line below appears only when
its data exists and its policy applies.

## The governing principle: verbose underneath, ONE crisp table on top

The report is **three tiers**, and a reader who reads only the first tier still gets the answer:

- **Tier 1 — the headline summary.** One screen: a single table plus a short verdict block. It
  is the **last** thing printed to a log (scroll to the bottom = the answer) and the **first**
  thing shown in the GUI (pinned at the top). If you read nothing else, you read this.
- **Tier 2 — descriptive detail.** Printed, scrollable: per-fold tables, calibration, sens/spec,
  OBD architecture-selection frequencies, timing, failures, the fold-plan description.
- **Tier 3 — machine-readable artifacts.** Files in the run directory, **never printed**: the
  per-exemplar out-of-fold predictions, per-fold metrics, run metadata. The substrate other
  tools (and future inferential methods) consume.

Two hard rules: **never print Tier 3** (a quarter-million predictions × several procedures buries
everything), and **never let Tier 1 exceed one screen.**

**The report has two audiences, by design.** Tier 1 is for a human at a glance. Tiers 2 and 3
are for an **LLM fed the full report** to answer any deeper question ("how did fold 7 differ?",
"which clusters drove the neural net's edge?"). That is *why* Tier 2/3 must stay complete and
self-describing rather than pre-summarized: the machine reader needs the raw material, and
pre-digesting it would throw away exactly what it came for. Do not "helpfully" trim or
LLM-summarize Tier 2/3 — a human never has to read them, and a machine wants them whole.

## Tier 1 — the headline summary table (the part to get exactly right)

One row per procedure. A column appears only when it is meaningful:

| Column | Appears when | Meaning |
|---|---|---|
| **Procedure** | always | `Logistic`, `LDFA`, `QDFA`, `Neural (OBD)`, … |
| **AUC (CV)** | CV ran | mean ± sd across the outer folds. **The sd is descriptive spread across _dependent_ folds — NOT a confidence interval.** |
| **AUC (test)** / **AUC (test) [95% CI]** | a locked test set exists | point AUC on the untouched test set; a 95% CI is appended ONLY when a sampling unit was declared AND the achieved design permits an estimator — the Wald DeLong interval for `independence=rows` over a row-wise design, the clustered interval for `independence=cluster` over a group-aware one — else the point AUC alone. Needs both classes and finite predictions. |
| **Arch** | the procedure has architecture metadata | modal selection + frequency (e.g. `4-2 (7/10)`); `—` for procedures without it |
| **Time** | always | wall-clock for that procedure |

Directly beneath the table, a short verdict block (never prose paragraphs):

- **Primary contrast (prespecified):** the single contrast declared *before* the run (e.g.
  `Neural vs Logistic`) — ΔAUC + the *p* of whichever estimator the design permitted (the line
  names it: `DeLong p` or `clustered ROC p`, never one labelled as the other) **on the locked
  test set only**, with a plain verdict (`significant` / `not significant`), and for a clustered
  design the number of independent clusters it rests on. Any other contrast is labelled
  *exploratory* or carries a stated multiplicity correction.
- **Architecture footnote:** what OBD selected across folds (modal + range) and the frozen
  architecture used on the locked test.
- **One standing caveat line, always:** *"CV ± is descriptive spread across dependent folds, not
  a confidence interval; the only inferential comparison is on the locked test set."*

Illustrative rendering (the numbers are invented; the spec is general):

```
═══════════════════════════════════════════════════════════════════════════
 SUMMARY — 10-fold nested CV + locked test · 226,679 patients · 6,705 events (2.96%)
 Folds: group-disjoint outcome-stratified 10-fold, seed 42 grouped by columns 3, 4 (612 groups)
═══════════════════════════════════════════════════════════════════════════
 Procedure       AUC (CV)         AUC (test) [95% CI]      Arch          Time
 ───────────────────────────────────────────────────────────────────────────
 Logistic        0.873 ± 0.011    0.874 [0.835–0.914]      —             3 s
 LDFA            0.831 ± 0.017    0.830 [0.788–0.872]      —             2 s
 QDFA            0.845 ± 0.015    0.846 [0.804–0.888]      —             2 s
 Neural (OBD)    0.881 ± 0.010    0.882 [0.842–0.921]      4-2 *         96 s
 ───────────────────────────────────────────────────────────────────────────
 Primary contrast (prespecified): Neural vs Logistic
   Locked test: ΔAUC +0.008, DeLong p = 0.38  →  not significant
 * OBD selected 4-2 in 7/10 folds (range 3-2 … 5-3); frozen model: 4-2
 CV ± is descriptive spread across dependent folds, not a CI; inference is on the locked test.
═══════════════════════════════════════════════════════════════════════════
```

### Tier-1 rules — what appears when (so it stays general)

- **No locked test** (a pure-CV policy): drop the `AUC (test) [95% CI]` column and the DeLong
  contrast; the verdict line becomes *"no locked-test inference (CV policy) — see descriptive
  spread."*
- **Single procedure:** still a one-row table, no contrast line.
- **A procedure that failed on some folds:** its row shows the metric over the folds that
  completed with `(n/k folds)` and a failure note — **never a silently short average.**
- **A non-AUC metric** (future outcomes): the column header becomes that metric's name; the
  structure is unchanged. Tier 1 is metric-agnostic.

## Tier 2 — descriptive detail (printed, above the summary)

- **Per-fold table:** fold × procedure × (AUC, sensitivity, specificity, calibration).
- **OBD architecture-selection frequency** (neural procedures): hidden size → count over folds.
- **Optimizer selection** (neural procedures): optimizer → count over folds, and whether
  Auto chose it per fold or the caller fixed it. Reported BESIDE the architecture, never
  instead of it — both are selection metadata and both belong in the record. A fold that
  produced no model contributes neither (architecture and optimizer are one per-fold
  record, so a failed fold cannot report half of it).
- **Timing** per procedure and per fold.
- **Failures:** fold, procedure, reason (a diverged fit, an infeasible size, …).
- **The fold plan:** stratification/grouping, k, seed — everything needed to reproduce it.

## Tier 3 — machine-readable artifacts (files, never printed)

Written via `util::run_path` (beside the data, like `neuron.log`):

- **`cv_predictions.csv`** — one row per (exemplar, procedure): exemplar id, true outcome, fold,
  out-of-fold predicted probability. *These paired OOF predictions are the substrate for any
  future CV-aware inferential method — retained, never summarised away.*
- **`cv_metrics.csv`** — fold × procedure × metrics, with a `status` column and BOTH
  denominators (`n_valid`, `n_total`): they are equal on a clean fold/run, and on the pooled
  row after a fold fails `n_valid < n_total` with `status = partial` (so the row never claims
  more observations than were used).
- **`cv_run.json`** — fold plan, seed, procedures, per-fold timings, failures, software version,
  and per-procedure `arch` / `optimizer` / `optimizerAuto` arrays (positionally paired: a fold
  appears in all three or in none).
- **`cv_locked_predictions.csv`** (with a locked test) — `row`[`,cluster`]`,outcome`, then one
  prediction column per procedure. The `cluster` column appears **only** for a grouped design;
  its absence means ungrouped, not unknown. Cluster ids are joined **by position** with the
  locked rows, so a length or range defect refuses the whole file rather than emitting a
  mis-joined column — a cluster id off by one row silently re-labels which patients are
  correlated with which.

**The design is machine-readable, not only prose (DLG-8).** `cv_run.json` carries
`foldDesign` and `lockedTest.splitDesign` objects — `method`, `seed`, `k`, the 1-based
`strataColumns` / `strataBins` / `groupColumns`, `groups`, `developmentOnly`,
`requested` vs `achieved` size, `leakage`, `refusal`, `warnings` — beside the human
sentence, which is *derived from the same fields*. A consumer must never have to parse
the sentence, and a report can no longer describe a design that was not run. The
`lockedTest` block adds `clusters` (independent units present in the locked sample; 0 =
ungrouped) and `inferenceReason` (why no estimator ran — an undeclared sampling unit, a
locked test too sparse to estimate a covariance, and a design that forbids the estimator
are three distinct facts). Which estimator a (declared sampling unit × achieved
partition method) pair permits is decided in **one** function, `evaldesign::chooseInference`
— not re-derived by the handler, the renderer, and the JSON writer.

## Ordering & per-interface rendering (same Tier-1 content, one source)

- **CLI / log** (text via `util::screen`): Tier 2 detail streams as folds complete; the **Tier 1
  summary prints LAST**. ASCII box table (as above).
- **GUI:** the **Tier 1 summary is pinned at the top** of the results panel (an HTML table); Tier
  2 detail below it; Tier 3 offered as file downloads.

## Ownership (rule 6)

The **comparison coordinator** owns the summary — it holds the results joined by patient and
fold, renders Tier 1 and Tier 2, and writes Tier 3. The runner/adapters produce per-fold metrics
and procedure metadata; `DataSet`/`TwoSet` compute the metrics (`getStatROCarea`, sens/spec,
calibration). **The renderer has no model-family switches** — it iterates procedures and whatever
metadata each one carries. Nothing dataset-specific lives in the report; policy (locked-test vs
pure-CV, which metric, which contrast) decides which columns and lines appear, not the structure.
