/* CV procedure adapters (ROADMAP 4 Phase 4b-CV). Each adapter builds a
   crossval::Procedure for one model family -- it knows how to fit that family on
   a fold's training set and return its held-out predictions. The generic runner
   (crossval::run) stays model-agnostic; the model knowledge lives here (rule 6:
   the adapters are separate from the runner). trainProcedure (any Network,
   including Logistic), dfaProcedure (LDFA/QDFA), and nestedObdProcedure are the
   families. */

#ifndef CVADAPTERS_H
#define CVADAPTERS_H

#include "crossval.h"
#include "network.h"
#include "obd.h"

namespace cvadapters {

// Plain train: clone a configured template network and retrain it from fresh
//    weights on the fold's training set (the clone carries the template's
//    training configuration via copy). maxIter caps the per-fold training.
crossval::Procedure trainProcedure( const Network& templateNet, unsigned maxIter );

// Discriminant function analysis: linear ( quadratic == false, LDFA ) or
//    quadratic ( QDFA ). Fit on the fold's training set; the held-out score is
//    the graded discriminant probability (the 2026-07-19 DFA ROC work).
crossval::Procedure dfaProcedure( bool quadratic );

// Nested OBD: for each fold, carve an inner validation set from the fold's
//    TRAINING rows and run the OBD hidden-layer search on that inner split alone.
//    The ENTIRE architecture selection happens inside the fold -- OBD early-stops
//    on the inner validation set (Network::sampleTestError reroutes to it, Phase
//    4c), so the outer held-out rows never influence which architecture is chosen
//    (the no-leakage invariant). The selected, early-stopped winner is then scored
//    once on the held-out rows. innerValFraction is the share of the fold's
//    training rows held out as the inner validation set.
//
//    cfg.algorithm travels with the search: a fixed optimizer reaches every fold
//    unchanged, and cfg.algorithm < 0 (auto) means each fold probes and chooses
//    INDEPENDENTLY from its own inner training data -- never reusing a choice
//    made by another fold, the standalone panel, or a previous run.
//
//    When selections is non-null, each fold that produced a model appends what it
//    chose (hidden units AND the optimizer it ran on) in fold order -- the
//    selection metadata the CV report summarizes. A failed fold appends nothing.
//
//    progress (may be null) is the INNER search's own callback, forwarded to
//    obd::run: it reports the phase and hidden-node trial inside the current fold.
//    CV's own ProgressFn reports the fold; this reports what is happening within
//    it, and a caller composes the two. The runner never learns about OBD to do it.
//    rowGroup (may be empty) is per-RAW-ROW group identity for the dataset the
//    procedure will receive, aligned by row index. When it is non-empty the inner
//    validation split is GROUP-DISJOINT too: without that, a grouped run still
//    chooses its architecture with rows from clusters that are also in its inner
//    training set, and the "generalizes to unseen groups" claim covers only the
//    outer score, not the selection that produced the model. There is NO fallback
//    to a row-wise inner split on a grouped request -- an infeasible inner
//    partition fails that fold with a reason.
crossval::Procedure nestedObdProcedure( const obd::Config& cfg,
	double innerValFraction,
	vector< crossval::FoldSelection >* selections = nullptr,
	obd::ProgressFn progress = nullptr,
	const vector< unsigned >& rowGroup = vector< unsigned >() );

// The inner validation split, as a separately testable decision (it is the whole
//    no-leakage argument for a nested search, so it is not left inline where only
//    an end-to-end run can reach it).
//
//    trainRows are the fold's training rows as RAW row indices; rowLabel and
//    rowGroup are indexed by raw row (rowGroup empty = ungrouped). Returns raw row
//    indices for the two inner sets. ok is false with a reason when no usable
//    split exists -- a grouped fold with too few groups, or a fraction that leaves
//    one side empty. Refusing is the contract: the alternative is selecting an
//    architecture on rows whose clusters it also trained on.
struct InnerSplit {
	bool ok = false;
	string reason;
	vector< unsigned > train, validation;
};
InnerSplit innerValidationSplit( const vector< unsigned >& trainRows,
	const vector< unsigned >& rowLabel, const vector< unsigned >& rowGroup,
	double fraction );

} // namespace cvadapters

#endif
