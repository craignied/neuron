// Automatic training-algorithm selection (ROADMAP 2 Phase 2), and the ONE
// eligibility rule the whole public surface consults.
//
// pick() probes every CURATED CANDIDATE that is eligible for the actual model
// and configuration -- canonical backpropagation, conjugate gradient descent,
// Shanno's algorithm, L-BFGS, iRPROP+ and Levenberg-Marquardt -- on CLONES of
// the same network, each starting from identical weights and given an equal
// wall-clock budget. The winner is the lowest final training error; its clone is
// returned so the caller can ADOPT it (probe progress kept) and continue
// training to the real iteration budget. See docs: the bank walkthrough measured
// why the winner is not knowable a priori -- uncapped CGD once ran 80 minutes to
// a worse-than-useless model there, while canonical GD converged in seconds; on
// other data the order reverses.
//
// WHY ELIGIBILITY LIVES HERE AND NOWHERE ELSE (rule 6). Two callers need the
// same answer to the same question: /api/train must REFUSE a directly selected
// optimizer the model cannot run, and pick() must OMIT that optimizer from the
// candidate list. Those are one rule with two presentations. Written twice they
// drift, and the failure is silent in the worst direction -- a method probed
// under a configuration it refuses, its Ineligible caught by the divergence
// handler, and reported to the user as "diverged".

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "network.h"

#ifndef AUTOALGO_H
#define AUTOALGO_H

namespace autoalgo {

// THE CONFIGURATION A CANDIDATE WOULD RUN UNDER. Built from the model, or from
//    the model overlaid with whatever a pending request would change, so a
//    refusal is decided against what WOULD run and not against what happens to
//    be installed at the moment the question is asked.
struct Settings {
	bool batchEpoch = true;    // batch/epoch (off-line) training is on
	bool autoStep = false;     // the automatic step-size search is on

	// THE PARAMETER COUNT THE CEILING IS JUDGED AGAINST. Settings::of takes the
	//    model's own, which is the answer for an ordinary training run.
	//
	//    A procedure that RESIZES the model while the chosen optimizer runs must
	//    raise it to the largest architecture it will build. OBD's grow-and-
	//    prune search is the case: it probes once at the starting size, keeps
	//    that optimizer for every size, and then grows. Judging the ceiling at
	//    the starting size would let a search adopt Levenberg-Marquardt and then
	//    walk into LM::Ineligible partway up, with no handler between it and the
	//    worker. This is the DECLARED sixth restriction applied to the
	//    architecture that will actually run -- not a further one.
	unsigned parameters = 0;

	// The model's own current settings, unchanged.
	static Settings of( Network& model );
};

// WHY `trainingType` CANNOT RUN on `model` under `s`, or null when it can.
//
//    The reason is a complete sentence in the second person's absence -- "is
//    available for neural models only", "requires batch_epoch=1" -- so a caller
//    can prefix it with the token and the name and get a refusal a human can
//    act on. It NAMES the offending condition; it never says "ineligible".
//
//    Returning a reason is not the same as refusing: /api/train turns it into a
//    400, pick() turns it into a recorded omission. Neither invents its own.
const char* ineligible( const Network& model, unsigned trainingType,
	const Settings& s );

// THE CURATED CANDIDATE LIST for `model` under `s`: every training type that is
//    eligible, in ascending trainingType order. That order is the tie policy --
//    pick() compares strictly, so an exact tie keeps the SIMPLER method, and
//    simplicity is what the enumeration is ordered by.
std::vector< unsigned > candidates( const Network& model, const Settings& s );

// THE DEFAULT TOTAL PROBE BUDGET, and a compatibility contract rather than a
//    tuning constant: 2250 ms is exactly what the original three-arm selection
//    spent (3 x 750 ms), so an ordinary automatic selection costs today what it
//    always cost. The budget is divided across the ELIGIBLE candidates -- it is
//    not per candidate -- because retaining another algorithm must not make
//    every automatic selection, every OBD search and every nested cross-
//    validation fold linearly slower.
const unsigned DEFAULT_TOTAL_BUDGET_MS = 2250;

// One probe's outcome
struct Probe {
	unsigned algorithm; // the REST/GUI token, trainingType + 1
	std::string name;   // the algorithm's prose name
	double error;       // final training error inside the budget
	unsigned iterations; // iterations completed inside the budget
	bool usable;        // finite, non-negative error (a diverged probe is not)
	// Why the probe ended: STOP_PROBE_BUDGET = the probe window expired (a
	//    bounded experiment, NOT a converged fit); STOP_GRADMAX = it CONVERGED
	//    inside the budget (a strong signal). Whichever probe wins, the real
	//    training that follows must still reach an eligible stopping condition.
	Iterative::StopReason stop = Iterative::STOP_NONE;
};

// A method that was NOT probed, and why. An omission is a reported fact, not an
//    absence: without this the selection result cannot distinguish "we tried it
//    and it failed" from "it was never eligible here", and the second silently
//    reads as the first.
struct Omission {
	unsigned algorithm;  // the REST/GUI token, trainingType + 1
	std::string name;    // the algorithm's prose name
	std::string reason;  // ineligible()'s sentence, verbatim
};

// The selection: the probes, the omissions, the choice, and the winning clone
struct Result {
	unsigned selected = 0; // the winning token; 0 = no probe was usable
	std::string selectedName;
	std::vector< Probe > probes;
	std::vector< Omission > omitted;

	// THE BUDGET, both halves, because neither is derivable from the other
	//    without knowing how many candidates were eligible -- and that is
	//    precisely the number a reader is trying to check.
	unsigned totalBudgetMs = 0;        // the declared ceiling for the whole selection
	unsigned perCandidateBudgetMs = 0; // what EVERY attempted candidate got

	std::unique_ptr< Network > winner; // adopt this; null when selected == 0
	bool cancelled = false; // the caller's cancel flag fired mid-probe
};

// Probe the eligible candidates from 'start' (cloned; 'start' is not touched),
//    sharing totalBudgetMs of wall clock EQUALLY between them. 'cancel' may be
//    null; when it fires, probing stops and no winner is returned. Probes train
//    into a discarded screen; the one decision summary is printed to the
//    caller's screen.
//
//    THE BUDGET RULE, stated once here and asserted in the tests:
//
//        per candidate = totalBudgetMs / (number of ELIGIBLE candidates)
//
//    integer-divided, so every attempted candidate gets the SAME figure -- an
//    unequal budget makes the comparison unfair, so the remainder is LEFT
//    UNSPENT rather than handed to whichever candidates come first. An omitted
//    candidate consumes nothing and does not divide the total. A nonempty
//    selection whose total is smaller than its eligible-candidate count throws
//    std::invalid_argument: silently flooring the share to 1 ms would spend
//    more than the declared total, while a zero-budget probe is not a useful
//    comparison.
//
//    ONLY the facts this function cannot derive are passed in. Batch/epoch and
//    the automatic step-size search are read from 'start' itself, because a
//    caller that described them differently from the model it handed over would
//    produce a candidate list for one configuration and a probe under another --
//    and the probe would fail with Ineligible, which the divergence handler
//    below would report as "diverged".
//
//    'plannedParameters' is what the model cannot tell it: whether the caller
//    will RESIZE the model while the chosen optimizer runs. 0 means "the
//    model's own", which is the answer for an ordinary training run; OBD passes
//    the largest architecture its search will build. It is applied as a MAXIMUM
//    against the model's own count, never as a replacement, so a caller passing
//    0 cannot widen eligibility past what the model itself allows.
Result pick( const Network& start, unsigned plannedParameters,
	unsigned totalBudgetMs, const std::atomic< bool >* cancel );

} // namespace autoalgo

#endif
