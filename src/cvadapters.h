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

namespace cvadapters {

// Plain train: clone a configured template network and retrain it from fresh
//    weights on the fold's training set (the clone carries the template's
//    training configuration via copy). maxIter caps the per-fold training.
crossval::Procedure trainProcedure( const Network& templateNet, unsigned maxIter );

// Discriminant function analysis: linear ( quadratic == false, LDFA ) or
//    quadratic ( QDFA ). Fit on the fold's training set; the held-out score is
//    the graded discriminant probability (the 2026-07-19 DFA ROC work).
crossval::Procedure dfaProcedure( bool quadratic );

} // namespace cvadapters

#endif
