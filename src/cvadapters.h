/* CV procedure adapters (ROADMAP 4 Phase 4b-CV). Each adapter builds a
   crossval::Procedure for one model family -- it knows how to fit that family on
   a fold's training set and return its held-out predictions. The generic runner
   (crossval::run) stays model-agnostic; the model knowledge lives here (rule 6:
   the adapters are separate from the runner). A nested-OBD adapter joins them
   in a later slice. */

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
//    training rows held out as the inner validation set. When selectedHidden is
//    non-null, each fold's chosen hidden-unit count is appended in fold order --
//    the architecture-selection metadata the CV report summarizes.
crossval::Procedure nestedObdProcedure( const obd::Config& cfg,
	double innerValFraction, vector< unsigned >* selectedHidden = nullptr );

} // namespace cvadapters

#endif
