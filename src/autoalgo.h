// Automatic training-algorithm selection (ROADMAP 2 Phase 2).
//
// pick() probes all three optimizers -- canonical backpropagation,
// conjugate gradient descent, Shanno's algorithm -- on CLONES of the same
// network, each starting from identical weights and given an equal
// wall-clock budget. The winner is the lowest final training error; its
// clone is returned so the caller can ADOPT it (probe progress kept) and
// continue training to the real iteration budget. See docs: the bank
// walkthrough measured why the winner is not knowable a priori -- uncapped
// CGD once ran 80 minutes to a worse-than-useless model there, while
// canonical GD converged in seconds; on other data the order reverses.

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "network.h"

#ifndef AUTOALGO_H
#define AUTOALGO_H

namespace autoalgo {

// One probe's outcome
struct Probe {
	unsigned algorithm; // 1..3, the training-menu / API numbering
	std::string name;   // the algorithm's menu name
	double error;       // final training error inside the budget
	unsigned iterations; // iterations completed inside the budget
	bool usable;        // finite, non-negative error (a diverged probe is not)
	// Why the probe ended: STOP_OBSERVER = the budget; STOP_GRADMAX =
	//    it CONVERGED inside the budget (a strong signal)
	Iterative::StopReason stop = Iterative::STOP_NONE;
};

// The selection: the probes, the choice, and the winning clone itself
struct Result {
	unsigned selected = 0; // 1..3; 0 = no probe was usable (keep the original)
	std::string selectedName;
	std::vector< Probe > probes;
	std::unique_ptr< Network > winner; // adopt this; null when selected == 0
	bool cancelled = false; // the caller's cancel flag fired mid-probe
};

// Probe the three algorithms from 'start' (cloned; 'start' is not touched)
//    with budgetMs of wall clock each. 'cancel' may be null; when it fires,
//    probing stops and no winner is returned. Probes train into a discarded
//    screen; the one decision summary is printed to the caller's screen.
Result pick( const Network& start, unsigned budgetMs,
	const std::atomic< bool >* cancel );

} // namespace autoalgo

#endif
