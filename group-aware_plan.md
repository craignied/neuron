# Group-aware evaluation and cluster-aware inference — Claude implementation handoff

## Read this first

This is the live implementation plan for the remaining ROADMAP 4 work. It was
reconciled with the repository on 2026-07-30. Do not follow older phase numbers
or implement items merely because they appear in the historical record.

Before editing code, read:

1. `AGENTS.md`, especially GUI/CLI parity, reproducibility, convergence, and the
   descriptions of `strata=`, `group=`, `val_fraction=`, `/api/cv`, and locked
   inference.
2. `CLAUDE.md`, ROADMAP 4 "What remains."
3. `docs/HISTORY.md`, "ROADMAP 4 — the plan as executed" and the DLG audit
   (`DLG-1` through `DLG-9`). History explains why the shipped contracts exist;
   do not regress them.
4. `docs/evaluation_report_spec.md`.
5. `docs/roc_theory.md`.
6. The current interfaces and tests named in the source map below.

Whenever behavior, API parameters, or GUI controls change, update `AGENTS.md`,
`CLAUDE.md`, `docs/gui_cli_parity.md`, `docs/evaluation_report_spec.md`,
`docs/roc_theory.md`, and the session entry in `docs/HISTORY.md` in the same
commit. The CLI menus are frozen. These are GUI-beyond-CLI evaluation features,
but GUI controls and HTTP parameters must still remain in parity.

## What is already shipped — do not rebuild it

The old version of this plan predated a large amount of completed work.

- `nsplit::stratifiedHoldout`, `holdoutByStrata`, and `groupHoldout` exist.
  `/api/load` and the Dataset panel support `strata=`, `strata_bins=`, and
  `group=`. Group holdout keeps clusters intact and reports zero leakage.
- Three-way train/validation/test splitting exists through `val_fraction=` or
  `val_n=`. It is outcome-stratified and currently refuses composition with
  `strata=` or `group=`.
- Outcome-stratified `nsplit::kFold` exists and is the only CV fold planner
  currently used.
- `crossval::run`, `compare`, and `evaluateOnce` exist. They preserve paired
  row identity, isolate procedure RNG substreams, validate locked partitions,
  expose progress, and keep model-family knowledge in procedure adapters.
- `/api/cv`, its GUI panel, async progress/stop behavior, nested OBD, the
  three-tier report, and Tier-3 artifacts exist.
- The locked test exists through `locked_fraction=` or `locked_n=`. It is
  currently an outcome-stratified row holdout.
- Point locked-test AUCs and predictions are retained even when inference is
  unavailable.
- Ordinary paired DeLong is opt-in only through `independence=rows`.
  `independence=cluster` is deliberately refused. An unspecified sampling unit
  yields point AUCs but no confidence intervals or contrast p-value.
- The DLG audit fixes already shipped:
  - DLG-1: no automatic IID claim; inference is explicit.
  - DLG-2: zero-variance/nonzero-delta contrast logic is corrected.
  - DLG-3: achieved locked class counts and development-fold feasibility are
    checked before the job starts.
  - DLG-4: predictions are independent of AUC/CI availability.
  - DLG-5: `evaluateOnce` validates indices, coverage, leakage, procedure
    callables, and unique procedure identities.
  - DLG-6: pooled artifact denominators and partial status are correct.
  - DLG-7: the locked-specific parsing defects are fixed.
  - DLG-9: the locked AUC interval is explicitly the classic Wald DeLong
    interval.

Do not fold the GUI-wide B9 strict-parsing pass into the scientific work below.
It remains a separate backlog item after these paths are stable.

## The two remaining scientific capabilities

These mechanisms are related but not interchangeable.

1. **Covariate-stratified and group-aware CV/evaluation partitions** decide
   which rows may appear in the same development fold, nested validation
   partition, or locked test. Grouping prevents cluster leakage and changes the
   target to generalization to unseen groups.
2. **Cluster-aware locked-test AUC inference** estimates uncertainty when rows
   in the locked sample share a sampling unit. It changes the standard errors,
   confidence intervals, and paired procedure comparison.

Keeping a county wholly on one side prevents leakage. It does not make patients
within a held-out county independent. Ordinary DeLong is therefore still
invalid for a county-clustered claim.

Build both as generalized mechanisms. The SEER county key is an acceptance
fixture, never a special path or hard-coded policy.

## Statistical contract to document before implementation

Add a compact design record to `docs/roc_theory.md` before adding the clustered
estimator:

- Point AUC remains the exact patient-row Mann–Whitney probability: a randomly
  selected outcome-1 row scores above a randomly selected outcome-0 row, with
  half credit for ties.
- Clusters, not patient rows, are the independent sampling units for clustered
  variance.
- All compared procedures use the same immutable partitions and are paired on
  the same locked rows and cluster identities.
- No fitting, tuning, architecture selection, optimizer selection, or stopping
  decision may use locked outcomes or locked predictions.
- Row-wise evaluation estimates performance for new rows from the represented
  group mixture. Group-disjoint evaluation estimates performance for groups the
  model did not see. Reports must name the estimand actually used.
- Ordinary DeLong is available only for a declared independent-row sampling
  unit. Clustered inference is never selected merely because it gives a more
  favorable result.

Use Obuchowski's nonparametric clustered structural-component covariance as the
analytic design:

- Obuchowski NA. *Biometrics*. 1997;53:567–578.
  DOI: `10.2307/2533958`.
- DeLong ER, DeLong DM, Clarke-Pearson DL. *Biometrics*.
  1988;44:837–845.
- Public reproduction of Obuchowski Table 3:
  <https://lerner.ccf.org/quantitative-health/documents/clusteredROC_help.pdf>

Keep a paired whole-cluster bootstrap as an independent validation method and a
possible later alternative, not as an unexamined substitute.

## Architecture rules

- `nsplit` owns index assignment and partition validation.
- `DataSet` owns interpreting selected input columns, constructing stable
  stratum/group keys, quantile-binning continuous strata, and materializing
  matrices.
- `crossval` owns repetition over a supplied immutable fold plan.
- Procedure adapters own fitting and any inner model selection.
- A statistics module owns clustered AUC covariance.
- `gui.cpp` owns request policy and composition; it must not duplicate key
  construction or statistical formulas.
- `cvreport` renders structured metadata; it must not infer the design from
  prose or switch on model family.
- No split, report, or inference path may depend on which procedures were
  selected or their order.
- Preserve the current outcome-only CV path and independent-row DeLong outputs
  byte-for-byte when their existing policy is selected.
- Use the existing training and resampling RNG streams. Split/fitting RNG and
  inference resampling must remain isolated.
- Add no dependency. Extend the existing Matrix/class layer only where a
  genuine primitive is missing.

## Current source map

Read these before choosing signatures:

- `src/split.{h,cpp}` — shipped holdout planners, `partitionError`, and
  outcome-only `kFold`.
- `tests/split/check_split.cpp` — reproducibility, balance, leakage, and
  DataSet integration contracts.
- `src/dataset.{h,cpp}` — `buildStrata`, `buildGroups`, split configuration,
  diagnostics, and row materialization. Expose/reuse this ownership; do not
  copy it into the GUI.
- `src/crossval.{h,cpp}` — `run`, `compare`, `evaluateOnce`, shared fold plan,
  row pairing, progress, and RNG substreams.
- `src/cvadapters.{h,cpp}` — plain and nested-OBD fitting. Inner validation
  policy belongs at this boundary or in an injected partition plan.
- `src/gui.cpp` — `CvConfig`, `handleCv`, `runCvJob`, `buildLockedInfo`, and
  `lockedJson`.
- `src/gui_page.html` — the CV and sampling-unit controls.
- `src/delong.{h,cpp}` and `tests/delong/check_delong.cpp` — ordinary
  independent-row covariance, tie handling, interval, and contrast behavior.
- `src/cvreport.{h,cpp}` — `PlanInfo`, `LockedInfo`, Tier 1/2 rendering, and
  Tier-3 writers.
- `tests/crossval/check_crossval.cpp` — class-layer, report, and artifact
  contracts.
- `tests/gui/smoke.sh` — public API behavior and artifact integration.

## Delivery sequence

Implement in the order below. Do not start clustered inference before stable
cluster identity flows through the complete evaluation pipeline.

### 1. DLG-8: structured design and cluster identity

Replace the present free-text-only design plumbing with structured metadata
while keeping the rendered wording stable for existing requests.

Introduce method-neutral types at the lowest common policy/report boundary,
for example:

```cpp
enum class SamplingUnit { Unspecified, Row, Cluster };
enum class AucInference { None, DeLongIndependent, ObuchowskiClustered };

enum class PartitionMethod {
    OutcomeStratified,
    CovariateStratified,
    StratifiedGroup
};
```

Exact names may follow repository conventions, but do not retain booleans and
magic strings as the source of truth.

The structured design must carry:

- split/fold method;
- seed and `k`;
- selected 1-based user column numbers and `strata_bins`;
- group count and stable group ID per relevant raw row;
- sampling unit and requested/applied inference method;
- achieved rows, events, groups, and groups per partition;
- leakage count and degeneracy/warning fields;
- a refusal reason distinct from procedure-fit failure.

Thread stable group identity through:

- raw data → development/locked partition;
- outer fold plan;
- `crossval::LockedResult`;
- `cvreport::LockedInfo`;
- `cv_locked_predictions.csv`;
- the `lockedTest` and fold-plan objects in `cv_run.json`.

Do not derive cluster IDs from row numbers, formatted labels, or report text.
They need only be stable within the run and artifacts; raw source values need
not be exposed if they may contain sensitive identifiers.

Compatibility requirements:

- No group metadata requested: current pure-CV and independent-row output stays
  byte-identical where tests require it.
- `SamplingUnit::Unspecified`: predictions and point AUCs, no inferential CI/p.
- `SamplingUnit::Row`: ordinary DeLong, with the existing visible assumption.
- `SamplingUnit::Cluster` before the estimator lands: keep the existing explicit
  refusal. Never fall back to ordinary DeLong.

Tests:

- artifact row/group/outcome/prediction pairing;
- structured metadata JSON escaping;
- group-vector length and ID validation;
- prediction retention when inference is unavailable;
- a grouped request cannot reach ordinary DeLong.

### 2. Generalized covariate-stratified k-fold planning

Add a structured fold plan over arbitrary stratum IDs in `nsplit`. Suggested
shape:

```cpp
struct FoldPlan {
    bool ok = false;
    string reason;
    vector<unsigned> foldId;
    unsigned k = 0;
    vector<unsigned> foldSize;
    vector<vector<unsigned>> stratumByFold;
    vector<string> warnings;
};

FoldPlan stratifiedKFold(const vector<unsigned>& stratum, unsigned k);
```

Use repository types if preferable. The important contract is structured
assignment plus achieved counts and warnings.

Algorithm:

1. Validate `2 <= k <= n` and compact/validate stratum IDs.
2. Partition row indices by stratum.
3. Fisher–Yates shuffle each stratum with the existing split RNG stream.
4. Deal rows across folds while rotating the starting fold or choosing among
   currently smallest eligible folds. Do not send every stratum remainder to
   fold 0.
5. Compute achieved fold/stratum counts from the returned assignment.

Preserve the current `nsplit::kFold(label,k)` behavior for the no-parameter
outcome-only path. Either leave it untouched or make a compatibility wrapper
only after a golden test proves identical seeded assignments. Do not quietly
change all existing CV results while adding covariate strata.

`DataSet` must expose a read-only key-building operation that reuses its current
`buildStrata` interpretation:

- selected user columns become 0-based input columns internally;
- low-cardinality columns become levels;
- continuous columns use `strata_bins` quantile bins;
- the outcome remains part of the CV stratum key.

Policy:

- Warn, do not fabricate rows, when a stratum has fewer than `k` members.
- A fold without both outcomes cannot supply an AUC; retain the procedure/fold
  record under the existing missing-metric contract.
- Do not silently collapse selected strata.
- Report per-fold row/event and stratum counts.

Tests:

- every row assigned exactly once to a valid fold;
- fixed-seed reproducibility and different-seed divergence;
- unavoidable fold-size remainder only;
- sufficiently large strata balanced across folds;
- rare strata produce warnings;
- multiple categorical columns and continuous quantile bins through `DataSet`;
- outcome-only compatibility.

Sabotage proof: temporarily reduce the stratum key to outcome only and ensure a
covariate-balance assertion fails.

### 3. Generalized stratified-group k-fold planning

Add a separate planner because indivisible groups make this a bin-packing
problem:

```cpp
struct GroupFoldPlan : FoldPlan {
    unsigned nGroups = 0;
    vector<unsigned> groupsPerFold;
    unsigned leakageCount = 0;
    double imbalanceScore = 0;
};

GroupFoldPlan stratifiedGroupKFold(
    const vector<unsigned>& outcome,
    const vector<unsigned>& group,
    unsigned k);
```

Algorithm:

1. Build each group's row list and outcome-count vector.
2. Seed-shuffle only for deterministic tie-breaking.
3. Process difficult groups first: descending size, then outcome imbalance.
4. Score every candidate fold by prospective outcome-0, outcome-1, and total
   row imbalance.
5. Assign the entire group to the minimum-score fold; use seeded ordering only
   for exact ties.
6. Add a bounded whole-group move/swap improvement pass only if measured SEER
   balance needs it.
7. Independently recompute leakage and require zero.

Policy:

- Require at least `k` distinct groups.
- Warn when fewer than `k` groups contain an outcome class.
- Report the largest group relative to the target fold, actual per-fold class
  counts, group counts, and imbalance.
- Never split an oversized group to meet a target.
- Do not hard-code SEER, county, prevalence, column numbers, or ten folds.

Tests:

- every row assigned exactly once;
- no group appears in more than one fold;
- reproducibility and group-label permutation invariance;
- balance on unequal synthetic clusters;
- too few groups;
- rare-event groups, one-class groups, and an oversized group;
- keys formed from one and several input columns.

Sabotage proof: substitute row-wise k-fold and ensure the leakage assertion
fails.

SEER acceptance for this layer: the supplied four area-SES columns form 612
county keys; record zero leakage, usable folds, achieved rows/events/counties,
runtime, and memory.

### 4. Compose fold policy into `/api/cv` and the GUI

Add CV-specific parameters. A CV request must be self-contained and must not
silently borrow the previous `/api/load` split configuration:

- `strata=` — comma-separated 1-based input columns;
- `strata_bins=` — quantile-bin count;
- `group=` — comma-separated 1-based input columns;
- `independence=rows|cluster` — keep the shipped parameter name for API
  compatibility, but parse immediately into `SamplingUnit`.

Policy:

- Neither `strata` nor `group`: preserve current outcome-stratified folds.
- `strata` only: outcome × covariate-stratified folds.
- `group` only: outcome-balanced group folds.
- Initially refuse `strata` plus `group` unless a tested joint balancing
  objective is deliberately implemented. Do not silently let one win as
  `/api/load` currently does.
- `independence=cluster` requires `group`, a locked test, and the clustered
  inference implementation. Until all three exist, return a precise refusal.
- `independence=rows` plus `group` is contradictory for inferential reporting.
  Refuse it; do not call the grouping descriptive and then run row DeLong.

Add independent controls to the CV panel for stratification columns/bins,
group columns, and sampling unit. Explain in one sentence:

- stratification balances represented subgroups;
- grouping measures transfer to unseen groups and prevents leakage;
- clustering changes inferential uncertainty.

Do not invisibly read values from the Dataset panel.

Replace `PlanInfo::foldPlan` as the source of truth with the structured plan.
It may retain a derived display string for compatibility. Render:

- method, seed, `k`, selected columns/bins;
- rows/events/groups per fold;
- stratum counts where useful;
- leakage and imbalance warnings.

Write the full structure to `cv_run.json`. Add stable group ID to
`cv_predictions.csv` when grouping is active. Preserve the present paired-wide
prediction format; `docs/evaluation_report_spec.md` explicitly records that the
older long-format description is not the shipped format.

### 5. Group-aware locked holdout

The same CV request's `group=` key must govern the locked split as well as the
outer folds:

1. Build stable group IDs from the full raw data.
2. Create the locked partition first with whole groups using
   `nsplit::groupHoldout` or a measured improvement of that general primitive.
3. Build all CV folds only inside the development rows, retaining their raw row
   and group identity.
4. Verify independently that no locked group occurs in development and no
   development group crosses outer folds.
5. Pass the locked rows to `evaluateOnce`; do not put group policy inside
   `evaluateOnce`, which correctly accepts an already-formed partition.

The report must say `group-disjoint locked holdout`, not IID. Record requested
and achieved locked sizes because groups are indivisible.

Preflight using achieved partitions:

- at least two events and two non-events in the locked sample for AUC;
- nonempty development data;
- at least `k` development groups;
- enough event-bearing and non-event-bearing groups for usable folds;
- enough informative locked clusters for the clustered covariance policy;
- zero locked/development leakage.

Keep point predictions and point AUCs even when clustered inference is
infeasible.

### 6. Group-aware nested validation

This is required for an honest unseen-group neural procedure. Group-disjoint
outer folds are not sufficient if nested OBD selects architecture using rows
from groups also present in its inner training set.

Extend the nested-OBD adapter boundary to accept an inner partition policy or a
precomputed group vector aligned with raw rows. For a grouped CV request:

- inner training and validation must be group-disjoint;
- the outer held-out fold remains untouched;
- for the final locked-development refit, OBD's inner validation must also be
  group-disjoint;
- group IDs must be gathered by raw row identity, never reconstructed after
  materialization.

Use a generalized whole-group holdout within each outer training partition.
Record achieved inner sizes and explicit infeasibility. Never fall back to
row-wise inner validation on a grouped request.

Do not change ordinary nested OBD behavior for a non-grouped request.

Tests must prove:

- no inner train/validation group overlap on every outer fold;
- no outer-held-out row/group enters inner selection;
- deterministic procedure results under comparison membership/order changes;
- infeasible thin-group partitions fail that fold with a specific reason;
- sabotage with row-wise inner validation trips a leakage assertion.

### 7. Clustered AUC covariance class layer

Create a separate module such as `src/clustered_auc.{h,cpp}`. Do not hide
cluster behavior inside `delong` and do not copy its tie logic.

Suggested result:

```cpp
namespace clustered_auc {

struct Result {
    bool ok = false;
    string reason;
    unsigned nRows = 0;
    unsigned nClusters = 0;
    unsigned n0 = 0;
    unsigned n1 = 0;
    vector<double> auc;
    Matrix<double> cov;
    string method;
};

Result analyze(
    const vector<unsigned>& label,
    const vector<unsigned>& cluster,
    const vector<vector<double>>& pred);

}
```

Extract/reuse the common placement or mid-rank primitive at the lowest natural
statistics boundary so ordinary and clustered methods use identical tie
handling and identical point AUC.

Requirements:

- implement Obuchowski's 1997 clustered structural-component covariance;
- handle clusters containing only negatives, only positives, or both;
- compute the full covariance matrix across all fitted procedures;
- treat clusters as the independent units in any small-sample rule;
- return structured refusal for length mismatch, invalid IDs, non-finite
  predictions, too few informative clusters, or a degenerate class;
- use the corrected zero-variance contrast behavior already established by
  DLG-2;
- report AUC, clustered SE/CI, delta, SE(delta), statistic, p-value, cluster
  count, and method name.

Validation is not optional:

1. Published Obuchowski MRA fixture:
   - 36 clusters, 65 observations;
   - AUCs approximately 0.9837 and 0.9852;
   - SEs approximately 0.0108 and 0.0097;
   - difference SE approximately 0.0066;
   - CI/p approximately `(-0.0115, 0.0143)` and `0.8271`;
   - compare intermediate covariance components where the reference supplies
     them.
2. Reduction identity: one row per cluster reduces to ordinary independent-row
   DeLong within numerical tolerance.
3. Paired whole-cluster bootstrap: resample clusters with replacement and
   recompute both AUCs and their signed difference. Use the reserved resampling
   RNG stream. Set a prespecified simulation tolerance for analytic versus
   bootstrap spread.
4. Ties, unequal cluster sizes, one-class clusters, identical procedures,
   perfect/reversed scores, cluster-label permutation, and malformed input.

Sabotage proof: run the published clustered fixture through ordinary DeLong and
ensure at least the variance/SE assertion fails.

### 8. Inference policy and reporting integration

The policy chooses exactly one:

- `DeLongIndependent` for declared independent rows and a row-wise design;
- `ObuchowskiClustered` for a declared cluster unit with valid cluster IDs;
- `None` for unspecified or infeasible inference.

Never choose by significance.

Make `LockedInfo`, columns, and contrasts method-neutral. Preserve independent
flags for:

- prediction available;
- point AUC available;
- interval available;
- contrast inference available.

Carry:

- sampling unit;
- inference method;
- number of clusters;
- rows and clusters by outcome;
- assumptions and refusal reason;
- stable cluster ID for every locked row.

Tier 1 clustered example:

```text
Locked test: ΔAUC = +0.008, clustered ROC p = 0.41
95% CIs and contrast use Obuchowski clustered covariance; 126 counties.
```

Independent rows must continue to say DeLong explicitly. Do not label the
clustered method "DeLong."

Tier 2 must distinguish:

- no inference requested;
- metadata absent;
- too few classes or informative clusters;
- procedure fit failure;
- valid nonsignificant result;
- valid equal/zero-variance result.

Tier 3:

- `cv_locked_predictions.csv`: raw row ID, cluster ID when active, outcome,
  paired procedure predictions;
- `cv_run.json`: structured split/fold policy, sampling unit, group columns and
  count, inference method, row/cluster counts by outcome, covariance/contrast
  results, assumptions, and warnings.

The full report must remain model-family agnostic and preserve current output
for pure outcome CV and independent-row DeLong.

### 9. Documentation, verification, and SEER acceptance

Update all live documentation listed at the top. Remove false statements that a
mechanically generated row holdout is IID. Teach, consistently:

- grouping prevents leakage;
- clustering changes variance;
- group-aware splitting does not justify ordinary DeLong;
- row-generalization and group-generalization are different estimands.

Run the repository gates required by `CLAUDE.md` after every coherent phase:

1. zero-warning Release build;
2. focused new unit tests;
3. `tests/golden/run_golden.sh` and inspect any diff;
4. `tests/gui/smoke.sh`;
5. full `ctest`;
6. `tests/oracle/verify_oracle.sh`;
7. the tools suite;
8. live GUI click-through for new controls with zero browser/page errors;
9. Linux, Windows, and macOS CI.

For every new behavior, first demonstrate the test fails against the old path
or a targeted sabotage. Record that proof in the commit/handoff; a green test
that never observed the defect is not sufficient.

SEER acceptance must use only generalized parameters, with the county key
supplied as data configuration. Record:

- county count and size distribution;
- zero leakage for locked split, outer folds, and inner validation;
- requested versus achieved locked/development sizes;
- rows, events, and counties per outer fold;
- procedure invariance to comparison membership/order;
- all locked predictions paired by patient and county;
- refusal of ordinary DeLong for the clustered policy;
- clustered AUCs, intervals, contrast, and independent county count;
- runtime and peak memory at 226,679 rows.

You may show ordinary DeLong beside clustered results only as a diagnostic
illustration of design effect. Label ordinary DeLong invalid for the clustered
claim and never use it as the reported scientific inference.

## Commit plan

Keep commits independently reviewable. A sensible sequence is:

1. DLG-8 structured design/cluster metadata and compatibility tests.
2. Covariate-stratified k-fold planner and DataSet key access.
3. Stratified-group k-fold planner and SEER fold acceptance.
4. `/api/cv` + GUI fold controls, diagnostics, and artifacts.
5. Group-disjoint locked holdout.
6. Group-disjoint nested OBD validation.
7. Clustered AUC module with published fixture and bootstrap cross-check.
8. Inference/report integration and end-to-end SEER acceptance.
9. Documentation/status reconciliation.

For each commit: add and observe the failing/sabotaged test, implement, run
proportional gates, inspect the diff, update parity/history documentation, then
commit. Do not push unless Craig explicitly asks.
