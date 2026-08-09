// Header for LM, the Levenberg-Marquardt optimizer, and LMObjective, the
// evaluation interface it drives.
//
// Research phase 6 of docs/learning_research/optimizer_implementation_plan.md.
// The complete contract -- every equation, every constant, every declared
// adaptation, the tests and the screen -- was written BEFORE any of this code,
// in docs/learning_research/lm_source_decision.md, and may not be revised now
// that measurements exist.
//
// SOURCES, fixed before any of this was written:
//
//   Madsen, K., Nielsen, H.B. & Tingleff, O. (2004), "Methods for Non-Linear
//     Least Squares Problems", 2nd ed., Informatics and Mathematical Modelling,
//     Technical University of Denmark -- equation (3.13), the initial damping
//     (3.14), the gain ratio, and ALGORITHM 3.16, which is what iterate()
//     below transcribes. Also Appendix A.4, the Cholesky factorization that
//     Matrix::solveSPD implements.
//   Nielsen, H.B. (1999), "Damping Parameter in Marquardt's Method", IMM, DTU,
//     Report IMM-REP-1999-05 -- the nu-based damping update Algorithm 3.16
//     uses, cited by the above as its source.
//   Levenberg, K. (1944), Quart. Appl. Math. 2:164-168, and Marquardt, D.W.
//     (1963), SIAM J. Appl. Math. 11:431-441 -- the primaries.
//   Hagan, M.T. & Menhaj, M.B. (1994), IEEE Trans. Neural Networks 5(6):989-993
//     -- precedent for LM on feedforward-network training. CLOSED ACCESS, NOT
//     READ, and NO equation, constant or control-flow decision here comes from
//     it. The per-exemplar Jacobian row is derived from neuron's own forward
//     equations instead. The substitution is disclosed in the source decision.
//
// ONE NAMED ALGORITHM. Levenberg's mu*I damping, Marquardt's mu*diag(J'J), and
// Moré's trust-region scaling are three different published methods. This is
// Algorithm 3.16, with mu*I. There is no flag, mode or configuration value that
// turns it into any of the others, because a variant switch is how a benchmark
// comes to describe a method nobody named.
//
// WHAT THIS CLASS OWNS, and what it deliberately does not:
//
//   OWNS  Algorithm 3.16's ENTIRE control flow -- mu_0, the damped solve, the
//         gain ratio, the acceptance test, both mu updates, the nu schedule,
//         the rejection loop and its bound, exact restoration, the finiteness
//         policy, the per-run reset and the research counters.
//
//   NOT   the model. It has NO model dependency: it is handed a way to
//         evaluate F, A = J'J and g = J'f at a point and to install points,
//         and that is the whole of its contact with the world. That is exactly
//         what lets tests/network/check_lm.cpp drive Algorithm 3.16 by hand,
//         against analytic problems with no network anywhere near them.
//
//   NOT   stopping. Iterative decides when a run ends and records why; this
//         class only REPORTS facts to it. Algorithm 3.16's three criteria map
//         onto that split as follows:
//
//           (3.15a) ||g||_inf <= eps1   Iterative's STOP_GRADMAX, which has
//                                       always lived there and is unchanged;
//           (3.15b) small step          COMPUTED HERE, in its published
//                                       position, and reported through
//                                       stepConverged(); Iterative ends the run
//                                       as STOP_SMALL_STEP;
//           (3.15c) k >= k_max          Iterative's maxIterations.
//
//         v1 omitted (3.15b) entirely, on the reasoning that stopping belonged
//         to Iterative. That was a correctness defect, not a clean separation:
//         with no channel to report convergence, a converged run's only exit was
//         StepRejected -- a FAILURE reported at a gradient maximum of 2.2e-10.
//         The fix restores the criterion without moving the ownership.
//
//   NOT   eta. The LM step is ABSOLUTE -- a displacement in parameter units,
//         sign included -- so it is applied through Network::applyAbsoluteStep,
//         never `w -= g * eta`. Nothing here reads or writes eta.

#ifndef LM_H
#define LM_H

#include <exception>
#include <vector>

#include "matrix.h"

using namespace std;

// WHAT LM NEEDS FROM A MODEL, and nothing more. One consumer, Network; no
//    descriptor, no std::function, no registration. Modelled on LBFGSObjective,
//    which exists for the same reason and against the same boundary.
class LMObjective {
public:
	virtual ~LMObjective() { }

	// The currently installed parameters, in the packed layout.
	virtual void currentPoint( vector< double >& weights ) const = 0;

	// Install packed parameters.
	virtual void install( const vector< double >& weights ) = 0;

	// EVALUATE THE NORMAL EQUATIONS at the currently installed parameters.
	//    Returns the objective F, and writes
	//
	//        normal   = A = J'J, INCLUDING any penalty curvature
	//        gradient = g = J'f, which IS grad F
	//
	//    in one traversal of the training set. It must not train, and the
	//    Matrix it writes must be fully symmetric on return.
	virtual double evaluateNormal( Matrix< double >& normal,
		vector< double >& gradient ) = 0;

	// Has something outside asked for this run to stop? Checked between trial
	//    evaluations, because ONE LM iteration can traverse the training set
	//    several times -- exactly the case Iterative::Observer::cancelled()
	//    was added for when a Wolfe line search appeared.
	virtual bool cancelled() const = 0;
};

class LM {
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

	// A non-finite objective, gradient or normal matrix AT THE STARTING POINT,
	//    or an initial damping that cannot be formed. Refused with no state
	//    modified and no weight moved: never silently fall back to gradient
	//    descent, because the optimizer that ran is part of the result.
	class NotFinite : public std::exception {
	public:
		virtual const char* what() const throw()
		{ return "Levenberg-Marquardt refuses a non-finite starting point"; }
	};

	// MAX_REJECTIONS consecutive trials were rejected inside one iteration. The
	//    last accepted parameters are restored exactly before this is thrown.
	//
	//    Algorithm 3.16's only bound is k_max, which counts accepted and
	//    rejected steps together; neuron's iteration counter counts trainSet()
	//    calls, so an unbounded inner loop would be INVISIBLE to Iterative and a
	//    stalled run would look merely slow. Returning quietly instead would let
	//    STOP_CHANGE or the plateau rule call a total failure "converged", which
	//    the convergence contract forbids.
	class StepRejected : public std::exception {
	public:
		virtual const char* what() const throw()
		{ return "Levenberg-Marquardt could not accept a step at any damping"; }
	};

	// --- The published constants, all of them ------------------------------
	//    Recorded in the source decision before any measurement was taken, and
	//    NONE of them is configurable. tau in particular is not a knob: a
	//    tunable initial damping would make the screen a tuning exercise.

	static const double TAU;        // 1e-3  initial damping scale, mu_0 = tau*max(a_ii)
	static const double NU_INIT;    // 2     the rejection multiplier's start and reset
	static const double MU_FLOOR;   // 1/3   the floor in mu *= max( 1/3, 1-(2rho-1)^3 )

	// CRITERION (3.15b), the published small-step convergence test:
	//
	//     ||h_lm|| <= EPS_STEP * ( ||x|| + EPS_STEP )
	//
	//    Madsen, Nielsen & Tingleff (2004) leave eps_2 to the user; their worked
	//    examples use 1e-8 (Example 3.7), 1e-12 (Example 3.16, the Rosenbrock
	//    problem this project drives as a test oracle) and 1e-15 (Examples 3.8,
	//    3.17). 1e-12 is the middle of the source's own range and the value of
	//    the example the tests already use.
	//
	//    NOT CONFIGURABLE, and deliberately so: a tunable convergence tolerance
	//    would let a later measurement describe the tuning rather than the
	//    method. There is no setter, no field and no public control.
	static const double EPS_STEP;   // 1e-12

	// The parameter ceiling. Storage is 3 P^2 doubles and the accumulation is
	//    O(N P^2), so this is where the method stops being the right one; 512 is
	//    Hagan & Menhaj's published "no more than a few hundred weights" rounded
	//    to a power of two, and bounds this class's storage at 6.3 MB
	//    INDEPENDENT of the row count.
	static const unsigned MAX_PARAMETERS;

	// Consecutive rejected trials allowed within ONE iteration. After 20, mu has
	//    grown by 2^210 and the step is numerically indistinguishable from zero;
	//    the double range would not overflow until roughly 45.
	static const unsigned MAX_REJECTIONS;

	LM();

	// PER-RUN reset: the accepted point, its objective, A and g, the damping,
	//    nu, the counters and the initialization flag. Called from
	//    Network::prepareRun(), which train() calls once before the loop and
	//    outside every reporting guard.
	//
	//    There is no persistent configuration to preserve -- every constant
	//    above is published or declared and fixed -- which is why there is no
	//    copyConfigurationFrom() here and why a copied model starts clean.
	void reset();

	// Has this run evaluated its first point yet?
	bool started() const { return startedFlag; }

	// HAS THE STEP BECOME NUMERICALLY NEGLIGIBLE? A FACT, not an instruction:
	//    Iterative asks this through Network and decides what to do about it,
	//    exactly as it does with the gradient maximum. Cleared by reset().
	//
	//    When it is true, iterate() has ALREADY returned normally, having
	//    evaluated NO trial at that iteration -- criterion (3.15b) is tested in
	//    its published position, immediately after the solve and before the
	//    trial. The accepted parameters are installed and untouched.
	bool stepConverged() const { return smallStep; }

	// ONE OUTER ITERATION of Algorithm 3.16.
	//
	//    Returns the objective at the point the step DEPARTED FROM -- legacy
	//    pre-update reporting, matching canonical, CGD, Shanno, L-BFGS and
	//    iRPROP+. Moving it would silently move what every stopping rule in
	//    Iterative compares.
	//
	//    On return the accepted parameters are installed. On cancellation the
	//    last accepted parameters are restored exactly and the departing
	//    objective is returned. Throws NotFinite at a bad starting point and
	//    StepRejected when no damping accepts a step.
	double iterate( LMObjective& f );

	// maxabs of the RAW gradient at the point this iteration departed from --
	//    the same point the returned objective describes (the plan's
	//    architecture decision 7).
	double gradMax() const { return gradMaxValue; }

	// --- Observation, for the deterministic tests --------------------------
	//    Read-only. Algorithm 3.16 is asserted step by step against
	//    hand-computed values, which needs the state to be visible; nothing
	//    here can modify it.

	double damping() const { return mu; }
	double rejectionMultiplier() const { return nuValue; }
	double acceptedObjective() const { return acceptedF; }
	const vector< double >& acceptedPoint() const { return accepted; }
	const vector< double >& gradient() const { return gradG; }
	const vector< double >& lastStep() const { return stepH; }
	const Matrix< double >& normalMatrix() const { return normalA; }

	// Research counters, for the phase report. They live once per iteration or
	//    once per trial, never in an exemplar loop (rule 7).
	unsigned long long iterations() const { return iterationCount; }
	unsigned long long acceptances() const { return accepted_; }
	unsigned long long rejections() const { return rejected_; }
	unsigned long long factorizationFailures() const { return factorFail; }
	unsigned long long nonFiniteTrials() const { return nonFinite_; }
	unsigned long long cancellations() const { return cancelled_; }

private:
	// Every buffer is sized once per run and reused for every iteration.
	vector< double > accepted,  // x, the accepted parameters
		trial,                  // x + h_lm, the trial parameters
		gradG,                  // g = J'f at the accepted point
		trialG,                 // g at the trial point
		stepH,                  // h_lm
		rhs;                    // -g, the right-hand side of (3.13)

	Matrix< double > normalA,   // A = J'J at the accepted point
		trialA,                 // A at the trial point
		damped;                 // A + mu I, the working copy the solve reads

	double acceptedF,           // F(x)
		mu,                     // the damping parameter
		nuValue,                // the rejection multiplier
		gradMaxValue;           // maxabs( g ) at the departing point

	bool startedFlag,
		smallStep;              // criterion (3.15b) has fired

	unsigned long long iterationCount, accepted_, rejected_, factorFail,
		nonFinite_, cancelled_;

	// mu *= nu; nu *= 2 -- Algorithm 3.16's rejection branch, in one place so
	//    the three call sites that reject (a failed factorization, a non-finite
	//    trial, and rho <= 0) cannot drift apart.
	void rejectStep();
};

#endif
