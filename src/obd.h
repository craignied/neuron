// Optimal Brain Damage hidden-layer sizing (ROADMAP 2 Phase 4).
//
// A grow-then-prune search for the right number of hidden units in a SimpleProp
// network, using VALIDATION EARLY STOPPING for time efficiency: no model trains
// to completion. Overtraining is the point DURING a run where the held-out
// (test) error turns back up while training error keeps falling -- so the driver
// watches the test error via sampleTestError() through an Iterative::Observer,
// tracks its running minimum, and stops each size the moment the test error has
// risen past that minimum by a tolerance and stayed up for a short patience.
// That minimum is the size's score. Growth is a WARM START (SimpleProp::
// growHidden preserves the smaller net's fit exactly), so each size continues
// from the last rather than retraining from scratch; the search stops adding
// units when a larger net no longer lowers the best test error, then prunes back
// by saliency. See docs/obd_plan.md for the full rationale and the (not-built)
// continuous-trajectory fallback should this prove too slow on large data.

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "network.h"

#ifndef OBD_H
#define OBD_H

namespace obd {

// One iteration's progress during the search, for a realtime chart. testErr < 0
//    means test error was not sampled on this iteration. Called on whatever
//    thread runs the search (the GUI's worker), so a GUI callback must guard its
//    own buffers. phase is "grow", "prune", or "final".
using ProgressFn = std::function< void( const char* phase, unsigned hidden,
	unsigned iteration, double trainErr, double testErr ) >;

// One size the search evaluated. testErr is the MINIMUM held-out error reached
//    at this size (its score), not the final -- the net is early-stopped a short
//    patience past that minimum. phaseGrow distinguishes growth from pruning.
struct SizeTrial {
	unsigned hidden;
	double trainErr; // training error at the validation minimum
	double testErr;  // the minimum held-out error reached (the score)
	double trainCA;  // training-set classification accuracy at the early-stop point (0..1)
	double testCA;   // test-set classification accuracy at the early-stop point (0..1)
	Iterative::StopReason stop;
	bool phaseGrow;
};

struct Config {
	unsigned hStart = 2;   // first hidden-unit count to try
	unsigned hMax = 30;    // ceiling on growth
	unsigned iterBudget = 2000; // per-size iteration CEILING (early stop ends it first)
	unsigned sampleEvery = 20;  // iterations between test-error samples (deterministic)
	double earlyStopTol = 0.02; // test-err rise above its min that counts as overtraining
	unsigned earlyStopPatience = 3; // samples above min+tol before a size stops
	unsigned growPatience = 2;  // sizes without a min-test improvement before growth stops
	double pruneTol = 0.02;     // how much worse than best a pruned net may be and still be kept
	int algorithm = 0;          // 0/1/2 = trainingType; -1 = auto (probe once, keep the choice)
	// TRAIN-error plateau backstop (the engine's auto-stop, forced on for every
	//    size): a size that converges flat never trips the test-error rise, so
	//    without this it would burn the whole budget. Same semantics as
	//    /api/train's autostop_tol / autostop_window.
	double plateauTol = 1e-4;
	unsigned plateauWindow = 100;
};

struct Result {
	bool ok = false;
	std::string message;             // refusal text when !ok
	unsigned selectedHidden = 0;
	std::vector< SizeTrial > history;
	std::unique_ptr< Network > winner; // the trained (early-stopped) selected net; adopt it
	bool cancelled = false;
};

// Run the search on data (which must carry a 1-output discrete training AND test
//    set). progress may be null; cancel may be null and, when it fires, ends the
//    search with the best-so-far as winner. The winner's ROC bootstrap is
//    re-enabled and its final report has been printed to util::screen() ahead of
//    the returned result, preceded by the size-vs-error table.
Result run( DataSet& data, const Config& cfg,
	ProgressFn progress, const std::atomic< bool >* cancel );

} // namespace obd

#endif
