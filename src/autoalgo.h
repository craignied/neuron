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

// A method that was NOT probed, and why. An omission is a reported fact, not an
//    absence: without this the selection result cannot distinguish "we tried it
//    and it failed" from "it was never eligible here", and the second silently
//    reads as the first.
struct Omission {
	unsigned algorithm;  // the REST/GUI token, trainingType + 1
	std::string name;    // the algorithm's prose name
	std::string reason;  // ineligible()'s sentence, verbatim
};

// WHAT A REQUESTED SET OF METHODS RESOLVES TO on this model: the ones that can
//    actually run, and the ones that cannot with the reason each was refused.
//
//    `competitors` is in ascending trainingType order whatever order the caller
//    asked in. That order is the tie policy -- pick() compares strictly, so an
//    exact tie keeps the SIMPLER method, and simplicity is what the enumeration
//    is ordered by -- and it is why the resolution sorts rather than preserving
//    the request: a set of methods has no meaningful user-supplied order, and
//    letting one in would make the winner depend on checkbox order.
//
//    THE THREE OUTCOMES A CALLER MUST DISTINGUISH, and the reason this is a
//    resolution step rather than a filter inside pick():
//
//      competitors.empty()      nothing the caller named can run here: REFUSE,
//                               naming every omission. Refusing is only correct
//                               BEFORE the request is applied to the model.
//      competitors.size() == 1  there is nothing to compare. Train on it
//                               directly -- no probe, no budget spent. A
//                               competition of one is not a competition; it is
//                               a slower way to run the only eligible method.
//      competitors.size() > 1   compete under the shared total budget.
struct Plan {
	std::vector< unsigned > competitors; // trainingTypes, ascending
	std::vector< Omission > omitted;     // requested but ineligible, with reasons
};

// Resolve `requested` (trainingTypes, any order) against `model` under `s`.
//    AN EMPTY REQUEST MEANS THE WHOLE CURATED LIST -- that is what algorithm=auto
//    asks for, and it makes "probe everything eligible" the same code path as
//    "probe these three", so the two can never disagree about eligibility.
//
//    Only requested methods are reported as omitted. A caller that named three
//    methods is told about those three; it is not handed the eligibility status
//    of methods it never asked about. (An empty request names all of them, so
//    algorithm=auto still reports every ineligible method, exactly as before.)
Plan plan( const Network& model, const std::vector< unsigned >& requested,
	const Settings& s );

// THE CURATED CANDIDATE LIST for `model` under `s`: every training type that is
//    eligible, in ascending trainingType order. plan() with an empty request.
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

// The selection: the probes, the omissions, the choice, and the winning clone
struct Result {
	unsigned selected = 0; // the winning token; 0 = no probe was usable
	std::string selectedName;
	std::vector< Probe > probes;
	std::vector< Omission > omitted;

	// WHETHER A COMPETITION ACTUALLY HAPPENED. False when the resolved list held
	//    a single eligible method: `selected` names it, no probe ran, no budget
	//    was spent and there is no winning clone to adopt, because the caller's
	//    own model is already the thing that will train. A reader cannot infer
	//    this from probes.empty() -- a cancelled selection is empty too.
	bool competed = false;

	// THE BUDGET, both halves, because neither is derivable from the other
	//    without knowing how many candidates were eligible -- and that is
	//    precisely the number a reader is trying to check.
	unsigned totalBudgetMs = 0;        // the declared ceiling for the whole selection
	unsigned perCandidateBudgetMs = 0; // what EVERY attempted candidate got

	std::unique_ptr< Network > winner; // adopt this; null when selected == 0
	bool cancelled = false; // the caller's cancel flag fired mid-probe
};

// Probe the eligible members of 'requested' from 'start' (cloned; 'start' is not
//    touched), sharing totalBudgetMs of wall clock EQUALLY between them. An empty
//    'requested' means every curated candidate. 'cancel' may be null; when it
//    fires, probing stops and no winner is returned. Probes train into a
//    discarded screen; the one decision summary is printed to the caller's
//    screen.
//
//    A SINGLE ELIGIBLE METHOD IS NOT PROBED. plan() resolves the request first;
//    when one method survives, there is nothing to compare it against, so the
//    result names it (competed = false, no probes, no budget, no winning clone)
//    and the caller trains on it directly. Probing it anyway would spend the
//    whole budget re-deriving an answer that was already known, and -- because a
//    probe's progress is adopted -- would silently make "train with L-BFGS" mean
//    something different from "train with L-BFGS" depending on how many boxes
//    happened to be ticked.
//
//    THE BUDGET RULE, stated once here and asserted in the tests:
//
//        per candidate = totalBudgetMs / (number of ELIGIBLE COMPETITORS)
//
//    integer-divided, so every attempted candidate gets the SAME figure -- an
//    unequal budget makes the comparison unfair, so the remainder is LEFT
//    UNSPENT rather than handed to whichever candidates come first. An omitted
//    candidate consumes nothing and does not divide the total, and neither does
//    a method the caller never named: narrowing the request buys the survivors a
//    LARGER share of the same fixed total, it does not shorten the selection.
//    A competition whose total is smaller than its competitor count throws
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
Result pick( const Network& start, const std::vector< unsigned >& requested,
	unsigned plannedParameters, unsigned totalBudgetMs,
	const std::atomic< bool >* cancel );

} // namespace autoalgo

#endif
