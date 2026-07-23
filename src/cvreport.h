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

// Tier 1: the one-screen headline summary (box table + verdict block) as text.
string tier1( const crossval::Comparison& cmp, const PlanInfo& info );

// Tier 2: the descriptive per-fold detail as text.
string tier2( const crossval::Comparison& cmp, const PlanInfo& info );

// Tier 3: write cv_predictions.csv / cv_metrics.csv / cv_run.json into dir
//    (a directory path; the files are dir + "/cv_*.{csv,json}"). Returns the
//    paths written. NEVER printed -- these are the machine-readable substrate.
vector< string > writeArtifacts( const crossval::Comparison& cmp,
	const PlanInfo& info, const string& dir );

// Log ordering: stream Tier 2 then the Tier 1 summary LAST, and write the Tier 3
//    artifacts into dir (dir "" skips the files). One call for the CLI/log path.
void render( ostream& out, const crossval::Comparison& cmp, const PlanInfo& info,
	const string& dir );

} // namespace cvreport

#endif
