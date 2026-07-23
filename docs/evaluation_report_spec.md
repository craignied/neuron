# Evaluation report specification (ROADMAP 4 Phase 4)

> **STATUS (2026-07-23): this is the ASPIRATIONAL spec; the shipped single-run CV report
> (`src/cvreport.{h,cpp}`, reached via `/api/cv`) implements a subset.** SHIPPED: the
> three-tier structure; Tier-1 headline table (Procedure · AUC(CV) mean ± sd · Arch · Time)
> + verdict block + the standing caveat; Tier-2 per-fold AUC/sens/spec, per-procedure
> failures (with reasons) and validFolds, and OBD architecture-selection frequency; Tier-3
> `cv_predictions.csv` / `cv_metrics.csv` (now with a per-fold `status` column) / `cv_run.json`
> (with per-procedure `validFolds` + `failures`). A Tier-3 file that cannot be written (unwritable
> directory, full disk) is reported as a run WARNING naming the file + reason — never silently
> dropped, and never counted as written unless it opened/wrote/flushed/closed cleanly.
> **NOT yet implemented (aspirational below):**
> a locked-test AUC/DeLong column and frozen-architecture result; Tier-2 calibration; per-fold
> timing; Tier-3 download buttons; a fully metric-agnostic Tier 1. **Format note:**
> `cv_predictions.csv` ships as **one row per exemplar with one prediction column per
> procedure** (a paired wide format), NOT the "one row per (exemplar, procedure)" the body
> describes. And Tier-2 detail is returned only when the async CV job finishes; it does not
> stream fold-by-fold. The locked-test inference layer is a separate future feature.

An automated evaluation run — several procedures, k outer folds, nested OBD, a per-exemplar
out-of-fold prediction for every patient — generates a great deal of output. Without a
deliberate structure it becomes a wall of numbers no one reads. This spec fixes the structure.

It is **general** (SEER is the illustrative example, never the mold): every column and line
below appears only when its data exists and its policy applies.

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
"which counties drove the neural net's edge?"). That is *why* Tier 2/3 must stay complete and
self-describing rather than pre-summarized: the machine reader needs the raw material, and
pre-digesting it would throw away exactly what it came for. Do not "helpfully" trim or
LLM-summarize Tier 2/3 — a human never has to read them, and a machine wants them whole.

## Tier 1 — the headline summary table (the part to get exactly right)

One row per procedure. A column appears only when it is meaningful:

| Column | Appears when | Meaning |
|---|---|---|
| **Procedure** | always | `Logistic`, `LDFA`, `QDFA`, `Neural (OBD)`, … |
| **AUC (CV)** | CV ran | mean ± sd across the outer folds. **The sd is descriptive spread across _dependent_ folds — NOT a confidence interval.** |
| **AUC (test) [95% CI]** | a locked test set exists | point AUC on the untouched test set + its DeLong 95% CI |
| **Arch** | the procedure has architecture metadata | modal selection + frequency (e.g. `4-2 (7/10)`); `—` for procedures without it |
| **Time** | always | wall-clock for that procedure |

Directly beneath the table, a short verdict block (never prose paragraphs):

- **Primary contrast (prespecified):** the single contrast declared *before* the run (e.g.
  `Neural vs Logistic`) — ΔAUC + DeLong p **on the locked test set only**, with a plain verdict
  (`significant` / `not significant`). Any other contrast is labelled *exploratory* or carries a
  stated multiplicity correction.
- **Architecture footnote:** what OBD selected across folds (modal + range) and the frozen
  architecture used on the locked test.
- **One standing caveat line, always:** *"CV ± is descriptive spread across dependent folds, not
  a confidence interval; the only inferential comparison is on the locked test set."*

Illustrative rendering (SEER-flavoured; the spec is general):

```
═══════════════════════════════════════════════════════════════════════════
 SUMMARY — 10-fold nested CV + locked test · 226,679 patients · 6,705 events (2.96%)
 Folds: outcome-stratified, group-aware (county), seed 42
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
- **Timing** per procedure and per fold.
- **Failures:** fold, procedure, reason (a diverged fit, an infeasible size, …).
- **The fold plan:** stratification/grouping, k, seed — everything needed to reproduce it.

## Tier 3 — machine-readable artifacts (files, never printed)

Written via `util::run_path` (beside the data, like `neuron.log`):

- **`cv_predictions.csv`** — one row per (exemplar, procedure): exemplar id, true outcome, fold,
  out-of-fold predicted probability. *These paired OOF predictions are the substrate for any
  future CV-aware inferential method — retained, never summarised away.*
- **`cv_metrics.csv`** — fold × procedure × metrics.
- **`cv_run.json`** — fold plan, seed, procedures, per-fold timings, failures, software version.

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
metadata each one carries. Nothing SEER-specific lives in the report; policy (locked-test vs
pure-CV, which metric, which contrast) decides which columns and lines appear, not the structure.
