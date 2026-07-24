/* Cross-validation runner (ROADMAP 4 Phase 4b-CV). CV owns REPETITION only
   (rule 6 / the ownership constitution): it materializes each fold and calls a
   PROCEDURE, collecting a per-exemplar out-of-fold prediction for every row plus
   per-fold held-out metrics. It has NO model knowledge -- a procedure is the
   adapter that knows how to fit one model family (plain train, or nested OBD).

   The runner does not choose the model, does not train, and does not know about
   HTTP, folds-as-policy, or SEER. The comparison coordinator (crossval::compare)
   supplies one immutable fold plan to several procedures and joins their
   results; the runner is the shared engine underneath. */

#ifndef CROSSVAL_H
#define CROSSVAL_H

#include <atomic>
#include <functional>
#include <string>
#include <vector>

#include "dataset.h"

using namespace std;

namespace crossval {

// Held-out metrics for a subset of exemplars, computed from a TwoSet built on
//    their ( outcome, prediction ) pairs at a 0.5 threshold. Any metric not
//    computable on a degenerate subset (one class present, or empty) is -1.
//    ONE place builds a TwoSet from out-of-fold pairs and reads its metrics
//    (rule 6); the runner and the report both call it.
struct Metrics {
	unsigned n = 0;
	double az = -1;   // binormal ROC area
	double trap = -1; // trapezoidal (exact Mann-Whitney) ROC area
	double sens = -1; // sensitivity at 0.5
	double spec = -1; // specificity at 0.5
	double ca = -1;   // classification accuracy at 0.5
};

// Held-out metrics for rows (indices into outcome/pred) -- the shared helper.
Metrics metricsFor( const vector< unsigned >& outcome,
	const vector< double >& pred, const vector< unsigned >& rows );

// One fold's held-out result. ok is false when the procedure could not fit this
//    fold (a singular DFA, an OBD refusal, a divergence the adapter rejected) --
//    then reason explains it and az/trap are -1 and NO prediction was scattered
//    for the fold's rows. A failed fold is retained as MISSING and reported, never
//    silently averaged (the reporting contract). az/trap are also -1 on a fold
//    that fitted but is degenerate for that estimator (one held-out class).
struct FoldResult {
	unsigned fold;
	unsigned nHeldout;
	bool ok = true;     // did the procedure produce predictions for this fold?
	string reason;      // why not, when !ok
	double az = -1;     // held-out binormal ROC area
	double trap = -1;   // held-out trapezoidal (exact Mann-Whitney) ROC area
};

struct RunResult {
	bool ok = false;
	bool cancelled = false;         // the run was stopped before completing
	string message;                 // refusal text when !ok
	vector< FoldResult > folds;
	vector< double > oofPrediction; // per raw row: its out-of-fold predicted prob
	                                //    ( -1 for a row whose fold FAILED -- absent )
	vector< unsigned > outcome;     // per raw row: its true 0/1 outcome
	unsigned validFolds = 0;        // folds that produced predictions
	double oofAz = -1;              // pooled out-of-fold binormal ROC area
	double oofTrap = -1;           // pooled out-of-fold trapezoidal ROC area
};

// What a procedure returns for one fold: predictions in test-row order when ok;
//    a reason and (optionally) the cancelled flag when not. A procedure NEVER
//    fabricates predictions -- it says it failed and the runner records that.
struct ProcResult {
	bool ok = false;
	bool cancelled = false;
	vector< double > pred; // held-out predictions (test-row order) when ok
	string reason;         // failure text when !ok
};

// A procedure fits a model on the fold's TRAINING rows -- its ENTIRE fit and
//    selection, so nothing sees the held-out rows -- and returns the predicted
//    probabilities for the TEST rows, in test-row order. foldData arrives already
//    materialized ( train = trainRows, test = testRows ); trainRows and testRows
//    are the raw row indices, passed so an adapter that needs an INNER split of
//    the training rows -- nested OBD carves its own validation set from them --
//    can re-materialize the fold three ways without ever touching the held-out
//    rows. cancel (may be null) lets a long fit stop promptly; the adapter should
//    return { cancelled = true } when it fires. The plain adapters ignore the
//    index arguments.
using Procedure = function< ProcResult( DataSet& foldData,
	const vector< unsigned >& trainRows, const vector< unsigned >& testRows,
	const atomic< bool >* cancel ) >;

// Run ONE procedure over a fold plan. data must be a rawLoaded, discrete,
//    1-output DataSet; foldId[ r ] is raw row r's fold ( 0 .. k-1 ), e.g. from
//    nsplit::kFold. Every row is held out exactly once. cancel (may be null) is
//    checked between folds and passed to the procedure, so Stop ends the run
//    promptly with cancelled = true. A fold the procedure cannot fit is recorded
//    as a failed FoldResult and its rows get no out-of-fold prediction.
//    When substreams is true the engine RNG is re-seeded per fold from a mix of
//    ( seed, fold ), so a procedure's fitted result depends only on its own
//    (procedure, fold) key -- never on how much RNG earlier work consumed. This
//    isolation lives inside CV; it uses util::set_seed at controlled points and
//    adds no new RNG mechanism. With substreams false the ambient RNG is used
//    (the caller seeds once), preserving the standalone-run behavior.
RunResult run( DataSet& data, const vector< unsigned >& foldId, Procedure proc,
	const atomic< bool >* cancel = nullptr, bool substreams = false,
	unsigned seed = 0 );

// A named procedure, for the comparison coordinator.
struct ProcedureSpec {
	string name;      // "Logistic", "LDFA", "Neural (OBD)", ...
	Procedure proc;
	// Optional architecture-metadata sink: for a nested-OBD procedure, wire the
	//    SAME vector into cvadapters::nestedObdProcedure( ..., archHidden ) and
	//    here. After the run the coordinator snapshots it into the entry (the
	//    per-fold selected hidden size), so the report can summarize it. Leave
	//    null for procedures without architecture metadata.
	vector< unsigned >* archHidden = nullptr;
};

// Several procedures' results over ONE shared fold plan. Because every entry
//    used the same folds, their per-exemplar out-of-fold predictions are paired
//    row for row -- the substrate the report and any future paired comparison
//    need (join by patient and fold).
struct Comparison {
	bool ok = false;
	bool cancelled = false;        // a procedure's run was stopped
	string message;                // refusal text when !ok
	vector< unsigned > outcome;    // shared, per raw row
	vector< unsigned > foldId;     // the shared fold plan (per raw row) -- the
	                               //    report needs it to write per-fold rows
	unsigned k = 0;                // number of folds
	struct Entry {
		string name;
		RunResult result;
		double seconds = 0;            // wall-clock for this procedure
		vector< unsigned > archHidden; // per-fold selected size; empty if none
	};
	vector< Entry > entries;
};

// One procedure's result on a single locked-test split. ok is false when the
//    procedure could not fit -- then reason explains it and pred is empty and NO
//    prediction was fabricated (the same contract as a failed CV fold). pred is in
//    test-row order, paired across entries by position (i.e. by testRows[ i ]).
struct LockedEntry {
	string name;
	bool ok = false;
	string reason;
	vector< double > pred;         // per locked-test row (paired across entries)
	double seconds = 0;            // wall-clock for this procedure's fit + score
	vector< unsigned > archHidden; // frozen architecture metadata, if any
};

// Several procedures evaluated ONCE on one shared locked-test split -- the
//    paired substrate for a locked-test inference (e.g. DeLong). testRows is the
//    row identity (raw indices), outcome[ i ] is testRows[ i ]'s true 0/1 outcome,
//    and every entry's pred[ i ] is that same row's prediction, so the pairing is
//    externally auditable (row -> outcome -> one prediction per procedure).
struct LockedResult {
	bool ok = false;
	bool cancelled = false;
	string message;                 // refusal text when !ok
	vector< unsigned > testRows;    // locked-test raw row indices (identity)
	vector< unsigned > outcome;     // per locked-test row, true 0/1 (paired)
	vector< LockedEntry > entries;  // one per procedure, in request order
};

// Evaluate each procedure ONCE on a single (train, test) partition of data's Raw:
//    materialize the split (makeFold), apply the procedure to the TRAINING rows by
//    its OWN prespecified fitting rule -- for a nested-OBD procedure that includes
//    carving an inner validation split from the training rows for architecture
//    selection and early stopping, so the final model is whatever the procedure
//    produces (NOT necessarily trained on every training row) -- and collect its
//    predictions on the test rows, paired across procedures. This is FINAL
//    evaluation on a locked test set, not repetition, but it reuses the same
//    Procedure adapters (a procedure that fits on a fold's training rows fits
//    equally on the development rows). The caller (a policy layer) owns what to do
//    with the pairing (e.g. DeLong). The split METHOD is the caller's -- this
//    mechanism sees only the row-index partition, so an indivisible-group locked
//    split slots in later without changing it. Substream/seed isolate each
//    procedure's fit by NAME (bug B11), as in compare().
LockedResult evaluateOnce( DataSet& data, const vector< unsigned >& trainRows,
	const vector< unsigned >& testRows, const vector< ProcedureSpec >& procs,
	const atomic< bool >* cancel = nullptr, bool substreams = false,
	unsigned seed = 0 );

// The comparison coordinator: run each procedure over the SAME fold plan and
//    collect the results, so the procedures stay paired. It owns coordination
//    only -- it does not train, select, or know any model family (rule 6).
//    When substreams is true each procedure runs on its OWN deterministic RNG
//    substream keyed by ( seed, procedure index, fold ), so a procedure's CV
//    result is invariant to which OTHER procedures are compared and in what order
//    -- comparison membership/ordering is a presentation choice, not an input to
//    a procedure's fit. Pass the same seed used to build the shared fold plan.
Comparison compare( DataSet& data, const vector< unsigned >& foldId,
	const vector< ProcedureSpec >& procs, const atomic< bool >* cancel = nullptr,
	bool substreams = false, unsigned seed = 0 );

} // namespace crossval

#endif
