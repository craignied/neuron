// Header for IRpropState, the iRPROP+ optimizer.
//
// RESEARCH ONLY. Phase 4 of docs/learning_research/optimizer_implementation_plan.md.
// There is no REST field, no GUI control, no menu token, no automatic-selection
// entry and no saved-network field for it. Its public identifier, if it is
// retained, is assigned in Phase 5.
//
// SOURCES, fixed before any of this was written and recorded in full in
// docs/learning_research/irprop_source_decision.md:
//
//   Igel, C. & Husken, M. (2000), "Improving the Rprop learning algorithm",
//     Proc. Second Int. ICSC Symposium on Neural Computation, pp. 115-121
//     -- where iRPROP+ is introduced.
//   Igel, C. & Husken, M. (2003), "Empirical evaluation of the improved Rprop
//     learning algorithms", Neurocomputing 50:105-123 -- the evaluation, and
//     the pseudocode transcribed into computeStep() below.
//   Riedmiller, M. & Braun, H. (1993), "A direct adaptive method for faster
//     backpropagation learning: The RPROP algorithm", Proc. IEEE ICNN,
//     pp. 586-591 -- the original sign-based update and every constant.
//
// ONE NAMED ALGORITHM. RPROP-, RPROP+, iRPROP- and iRPROP+ are four different
// published methods. This is iRPROP+. There is no flag, mode or configuration
// value that turns this code into any of the other three, because a variant
// switch is how a benchmark comes to describe a method nobody named.
//
// WHAT SEPARATES iRPROP+ FROM RPROP+, and it is one line: on a sign flip
// RPROP+ always reverts the previous step, while iRPROP+ reverts it ONLY when
// the objective got worse. When the objective improved despite the flip the
// coordinate takes NO STEP AT ALL -- it neither reverts nor advances.
//
// WHAT THIS CLASS OWNS, and what it deliberately does not:
//
//   OWNS  the published table; the previous raw gradient (after the sign-flip
//         zeroing, which IS the algorithm's memory); the per-parameter Delta;
//         the previous applied step; the previous objective; the initialization
//         flag; the clamps; and the research counters.
//
//   NOT   the model. This class has no model dependency at all: it is handed an
//         objective and a raw gradient and produces an absolute step. That is
//         exactly what lets tests/network/check_irprop.cpp drive the published
//         table by hand instead of inferring it from a training run.
//
//   NOT   stopping, cancellation, the iteration counter or reporting. Iterative
//         keeps all of those. Unlike L-BFGS this method makes exactly ONE
//         evaluation per iteration, so Iterative's existing between-iteration
//         cancellation check is the whole story and none is added here.
//
//   NOT   eta. Delta is an absolute step magnitude in parameter units, never a
//         learning rate. Nothing here reads or writes eta -- the plan's
//         architecture decision 8 forbids expressing an absolute step by
//         dividing by eta or by mutating the user's learning rate.

#ifndef IRPROP_H
#define IRPROP_H

#include <exception>
#include <vector>

using namespace std;

class IRpropState {
public:
	// A configuration this method cannot run under. Raised rather than silently
	//    doing something else -- an optimizer that quietly becomes a different
	//    optimizer is the defect legacy bug #12 was.
	class Ineligible : public std::exception {
	public:
		explicit Ineligible( const char* why ) : reason( why ) { }
		virtual const char* what() const throw() { return reason; }
	private:
		const char* reason;
	};

	// A non-finite objective or raw gradient. REFUSED, with no state modified
	//    and no weight moved: never silently fall back to gradient descent,
	//    because the optimizer that ran is part of the result.
	class NotFinite : public std::exception {
	public:
		virtual const char* what() const throw()
		{ return "iRPROP+ refuses a non-finite objective or gradient"; }
	};

	// --- The published constants, all of them -----------------------------
	//    Recorded in the source decision before any measurement was taken, and
	//    NONE of them is configurable. Delta_0 in particular is not a knob: a
	//    tunable initial step would make the screen a tuning exercise.

	static const double DELTA_INIT; // 0.1   initial Delta, every parameter
	static const double ETA_PLUS;   // 1.2   growth factor
	static const double ETA_MINUS;  // 0.5   shrink factor
	static const double DELTA_MIN;  // 1e-6  lower clamp
	static const double DELTA_MAX;  // 50    upper clamp

	IRpropState();

	// PER-RUN reset: Delta, previous gradient, previous step, previous
	//    objective and counters. Called from Network::prepareRun(), which
	//    train() calls once before the loop and outside every reporting guard.
	//    There is no persistent configuration to preserve -- every constant
	//    above is published and fixed -- which is why there is no
	//    copyConfigurationFrom() here and why a copied model starts clean.
	void reset();

	// Has this run seen its first point yet?
	bool started() const { return startedFlag; }

	// THE PUBLISHED TABLE, one full application of it.
	//
	//    Given the objective and the RAW gradient at the currently installed
	//    parameters, writes the ABSOLUTE STEP to apply as `w += step`. It does
	//    not touch the model and does not apply anything; the caller owns the
	//    update path.
	//
	//    `rawGradient` is READ, never written. The published zeroing of the
	//    current gradient after a sign flip applies to what this class REMEMBERS
	//    for the next iteration, and the caller's raw gradient must survive it
	//    intact -- currGradMax is taken from that same vector, and a stopping
	//    rule reading a partially zeroed gradient would report a smaller maximum
	//    than the point actually has.
	//
	//    Throws NotFinite if the objective or any gradient element is not
	//    finite, before any state is modified.
	void computeStep( const double objective, const vector< double >& rawGradient,
		vector< double >& step );

	// --- Observation, for the deterministic tests --------------------------
	//    Read-only. The published table is asserted element by element against
	//    hand-computed values, which needs the state to be visible; nothing
	//    here can modify it.

	const vector< double >& deltas() const { return delta; }
	const vector< double >& previousGradient() const { return prevG; }
	const vector< double >& previousStep() const { return prevStep; }
	double previousObjective() const { return prevF; }

	// Research counters, for the phase report. They live in the per-PARAMETER
	//    loop, which runs once per epoch beside the epoch's single update, and
	//    never in an exemplar loop (rule 7).
	unsigned long long growths() const { return grew; }
	unsigned long long shrinks() const { return shrank; }
	unsigned long long rollbacks() const { return reverted; }
	unsigned long long heldFlips() const { return held; }
	unsigned long long iterations() const { return steps; }

private:
	vector< double > delta,    // Delta_ij, one per packed parameter
		prevG,                 // dE^(t-1)/dw, AFTER the sign-flip zeroing
		prevStep;              // dw^(t-1), the step actually applied
	double prevF;              // E^(t-1)
	bool startedFlag;

	unsigned long long grew, shrank, reverted, held, steps;
};

#endif
