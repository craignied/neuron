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
crossval::Procedure nestedObdProcedure( const obd::Config& cfg,
	double innerValFraction,
	vector< crossval::FoldSelection >* selections = nullptr,
	obd::ProgressFn progress = nullptr );

} // namespace cvadapters

#endif
