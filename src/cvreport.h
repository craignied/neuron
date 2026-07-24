/* Cross-validation evaluation report (ROADMAP 4 Phase 4). Renders a
   crossval::Comparison into the three-tier report of docs/evaluation_report_spec.md:

     Tier 1 -- a one-screen headline table + verdict block (printed LAST to a log,
               so the bottom of the scroll is the answer; pinned at the TOP in the GUI).
     Tier 2 -- descriptive per-fold detail, architecture-selection frequency,
               timing, and the fold plan (printed, scrollable).
     Tier 3 -- machine-readable files (cv_predictions.csv / cv_metrics.csv /
               cv_run.json), written beside the data and NEVER printed.

   The report is CV-only in this slice: there is no locked test set, so the
   AUC(test)/DeLong column and contrast are omitted and the verdict states the
   CV policy (the locked-test comparison is a later policy layer above the
   mechanism). Ownership (rule 6): the coordinator owns the summary; this renderer
   iterates procedures and whatever metadata each carries -- it has NO model-family
   switches, and nothing here is SEER-specific. */

#ifndef CVREPORT_H
#define CVREPORT_H

#include <ostream>
#include <string>
#include <vector>

#include "crossval.h"

using namespace std;

namespace cvreport {

// The descriptive context a Comparison does not carry: what the run WAS. Every
//    field is optional and appears in the report only when set.
struct PlanInfo {
	string dataset;    // a label for the header (e.g. "lowbwt"); "" hides it
	unsigned n = 0;    // total exemplars
	unsigned events = 0; // outcome-1 count
	string foldPlan;   // how the folds were built (e.g. "outcome-stratified, seed 42")
	string primary;    // prespecified primary procedure name (for the contrast line)
	string reference;  // the procedure it is contrasted against
};

// The locked-test inference results (ROADMAP 4 Phase 4). A policy layer above the
//    CV mechanism (the GUI CV job) fills this from crossval::evaluateOnce + delong:
//    each procedure refit by its own rule on the development set, scored once on an
//    untouched locked test set, then DeLong on the paired predictions. When has is
//    false the report is CV-only and byte-identical to a run without a locked test.
//    A column is matched to a cmp.entries row by NAME, so column order is free.
//    DeLong assumes the locked-test rows are INDEPENDENT (see delong.h) -- the
//    renderer states that assumption; it does not decide the split method.
struct LockedColumn {
	string name;
	bool has = false;      // an AUC(test) with CI is available for this procedure
	double auc = 0, lo = 0, hi = 0;
	string arch;           // frozen architecture string ("" if none)
	string note;           // e.g. "failed: <reason>" when the procedure did not fit
	vector< double > pred; // per locked-test row (paired to LockedInfo.testRows);
	                       //    empty when the procedure did not fit
};

// The prespecified primary contrast on the locked test: delta is always
//    AUC(primary) - AUC(reference), with DeLong's two-sided p. significant is the
//    caller's decision at its alpha (a policy, passed in). degenerate marks an
//    untestable difference (identical predictions). note carries a refusal reason
//    when the contrast could not be computed at all.
struct LockedContrast {
	bool has = false;
	string primary, reference;
	double delta = 0, p = 0;
	bool significant = false;
	bool degenerate = false;
	string note;
};

struct LockedInfo {
	bool has = false;
	unsigned n = 0, events = 0;    // locked-test size and event count
	string splitPlan;              // how the locked test was formed (free text)
	vector< unsigned > testRows;   // locked-test raw row ids (identity, for the CSV)
	vector< unsigned > outcome;    // per locked-test row, true 0/1 (paired)
	vector< LockedColumn > columns;
	LockedContrast contrast;
};

// Tier 1: the one-screen headline summary (box table + verdict block) as text.
//    With a locked-test result (locked.has) the table gains an AUC (test) [95% CI]
//    column and the verdict states the DeLong contrast; otherwise byte-identical.
string tier1( const crossval::Comparison& cmp, const PlanInfo& info,
	const LockedInfo& locked = LockedInfo() );

// Tier 2: the descriptive per-fold detail as text (plus a locked-test section
//    when locked.has; byte-identical to before when it does not).
string tier2( const crossval::Comparison& cmp, const PlanInfo& info,
	const LockedInfo& locked = LockedInfo() );

// The outcome of writing one Tier-3 artifact. ok is true ONLY when the file was
//    opened, fully written, flushed, and closed without error -- so a mid-write
//    failure (a full disk, an I/O error) is reported, not silently dropped (B7).
struct ArtifactResult {
	string name;   // e.g. "cv_predictions.csv"
	string path;   // the full path attempted
	bool ok = false;
	string error;  // the reason when !ok
};

// Tier 3: write cv_predictions.csv / cv_metrics.csv / cv_run.json into dir
//    (a directory path; the files are dir + "/cv_*.{csv,json}"). Returns one
//    ArtifactResult PER attempted file, so the caller can report exactly which
//    failed and why. NEVER printed -- these are the machine-readable substrate.
vector< ArtifactResult > writeArtifacts( const crossval::Comparison& cmp,
	const PlanInfo& info, const string& dir,
	const LockedInfo& locked = LockedInfo() );

// Log ordering: stream Tier 2 then the Tier 1 summary LAST, and write the Tier 3
//    artifacts into dir (dir "" skips the files). One call for the CLI/log path.
void render( ostream& out, const crossval::Comparison& cmp, const PlanInfo& info,
	const string& dir, const LockedInfo& locked = LockedInfo() );

} // namespace cvreport

#endif
