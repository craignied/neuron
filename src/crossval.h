/* Cross-validation runner (ROADMAP 4 Phase 4b-CV). CV owns REPETITION only
   (rule 6 / the ownership constitution): it materializes each fold and calls a
   PROCEDURE, collecting a per-exemplar out-of-fold prediction for every row plus
   per-fold held-out metrics. It has NO model knowledge -- a procedure is the
   adapter that knows how to fit one model family (plain train, or nested OBD).

   The runner does not choose the model, does not train, and does not know about
   HTTP, folds-as-policy, or SEER. The comparison coordinator (a later slice)
   supplies one immutable fold plan to several procedures and joins their
   results; the runner is the shared engine underneath. */

#ifndef CROSSVAL_H
#define CROSSVAL_H

#include <functional>
#include <string>
#include <vector>

#include "dataset.h"
#include "network.h"

using namespace std;

namespace crossval {

// One fold's held-out result. az/trap are -1 when the fold is too small or
//    degenerate for that estimator.
struct FoldResult {
	unsigned fold;
	unsigned nHeldout;
	double az;   // held-out binormal ROC area
	double trap; // held-out trapezoidal (exact Mann-Whitney) ROC area
};

struct RunResult {
	bool ok = false;
	string message;                 // refusal text when !ok
	vector< FoldResult > folds;
	vector< double > oofPrediction; // per raw row: its out-of-fold predicted prob
	vector< unsigned > outcome;     // per raw row: its true 0/1 outcome
	double oofAz = -1;              // pooled out-of-fold binormal ROC area
	double oofTrap = -1;           // pooled out-of-fold trapezoidal ROC area
};

// A procedure fits a model on foldData's TRAINING set -- its ENTIRE fit and
//    selection, so nothing sees the held-out rows -- and returns the predicted
//    probabilities for foldData's TEST rows, in test-row order.
using Procedure = function< vector< double >( DataSet& ) >;

// Run ONE procedure over a fold plan. data must be a rawLoaded, discrete,
//    1-output DataSet; foldId[ r ] is raw row r's fold ( 0 .. k-1 ), e.g. from
//    nsplit::kFold. Every row is held out exactly once, so every row receives an
//    out-of-fold prediction.
RunResult run( DataSet& data, const vector< unsigned >& foldId, Procedure proc );

// A ready-made procedure: clone a configured template network, retrain it from
//    fresh weights on the fold's training set (the clone carries the template's
//    training configuration via copy), and return its held-out predictions. The
//    per-fold ROC bootstrap is disabled (the runner needs point predictions).
Procedure trainProcedure( const Network& templateNet, unsigned maxIter );

} // namespace crossval

#endif
