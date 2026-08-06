// Header for LBFGS, the limited-memory BFGS optimizer.
//
// RESEARCH ONLY. Phase 3 of docs/learning_research/optimizer_implementation_plan.md.
// It is not selectable from the CLI, the GUI, the HTTP API or automatic
// selection, it is not written to a saved network, and it appears in no
// Manifest capability. It is reachable through Network::setTrainingType with
// Network::TRAIN_LBFGS, which the menus never produce.
//
// SOURCES, fixed before any of this was written and recorded in full in
// docs/learning_research/lbfgs_source_decision.md:
//
//   Nocedal, J. (1980), Math. Comp. 35(151):773-782 -- limited-memory updates.
//   Liu, D. C. & Nocedal, J. (1989), Math. Prog. 45:503-528 -- the method.
//   Nocedal, J. & Wright, S. J., Numerical Optimization, 2nd ed., Springer 2006:
//     Algorithm 7.4  (two-loop recursion, p. 178)
//     equation 7.20  (initial scaling gamma_k, p. 178)
//     Algorithm 7.5  (the outer loop, p. 179)
//     conditions 3.6a / 3.7b (sufficient decrease, STRONG curvature, pp. 33-34)
//     Algorithm 3.5  (line search, bracketing phase, p. 60)
//     Algorithm 3.6  (zoom, p. 61)
//   Wolfe, P. (1969), SIAM Review 11(2):226-235 -- the conditions themselves.
//
// THE LINE SEARCH IS A COMPLETE STRONG-WOLFE SEARCH -- Algorithm 3.5 with
// Algorithm 3.6 as its refinement phase. It is NOT Armijo backtracking. That
// distinction is the whole reason the packed boundary exists: a backtracking
// search never tests curvature, so it cannot guarantee s'y > 0, and a
// quasi-Newton method that cannot guarantee that is not the published one.
//
// WHAT THIS CLASS OWNS, and what it deliberately does not:
//
//   OWNS  the memory length m; the s, y and rho ring buffers; the last
//         ACCEPTED parameters, objective and raw gradient; the two-loop
//         scratch; the line search and its evaluation counters; curvature-pair
//         acceptance and history reset; the direction; and trial installation,
//         acceptance, failure and exact restoration.
//
//   NOT   stopping, cancellation, the iteration counter, reporting, or the
//         model's equations. Iterative keeps all of those. This class asks the
//         objective whether cancellation has been requested and restores if so;
//         it never decides that a run is over.
//
//   NOT   eta. The step length is this class's own variable. Mutating the
//         user's learning rate to express an absolute step is forbidden by the
//         plan's architecture decision 8, and nothing here touches it.
//
//   NOT   lastG / lastF. Those are CGD's and Shanno's, and one buffer with two
//         meanings is how two published algorithms come to share a defect.

#ifndef LBFGS_H
#define LBFGS_H

#include <exception>
#include <vector>

using namespace std;

// The model, as L-BFGS needs to see it. Four methods, one consumer: this is
//    the minimum boundary, not a callback framework. Each call below is ONE
//    full traversal of the training set at most, so the virtual dispatch is
//    once per evaluation and never inside an exemplar loop (rule 7).
class LBFGSObjective {
public:
	virtual ~LBFGSObjective() { }

	// The currently installed parameters, packed.
	virtual void currentPoint( vector< double >& w ) const = 0;

	// Install packed parameters. Used for trial points AND for the exact
	//    restoration a failed search owes the caller.
	virtual void install( const vector< double >& w ) = 0;

	// Objective and RAW gradient at the currently installed parameters.
	virtual double evaluate( vector< double >& g ) = 0;

	// Has something outside asked the run to stop? Checked before every trial
	//    evaluation, because one evaluation is a whole pass over the data and a
	//    cancelled run should not pay for twenty of them.
	virtual bool cancelled() const = 0;
};

class LBFGS {
public:
	// A configuration this method cannot run under. Raised rather than
	//    silently doing something else -- an optimizer that quietly becomes a
	//    different optimizer is the defect legacy bug #12 was.
	class Ineligible : public std::exception {
	public:
		explicit Ineligible( const char* why ) : reason( why ) { }
		virtual const char* what() const throw() { return reason; }
	private:
		const char* reason;
	};

	// --- The published constants, all of them ----------------------------
	//    Recorded in the source decision before any measurement was taken.
	//    Changing one after seeing a benchmark would make the benchmark a
	//    description of the tuning rather than of the algorithm.

	static const double C1;           // 1e-4  sufficient decrease  (N&W p. 33)
	static const double C2;           // 0.9   STRONG curvature     (N&W p. 34)
	static const double ALPHA_MAX;    // 1e10  bracketing ceiling
	static const double EXPAND;       // 2.0   bracketing expansion
	static const double CURV_EPS;     // 1e-10 relative s'y floor
	static const unsigned MAX_EVALS;  // 20    evaluations per line search

	LBFGS();

	// Memory length m. Liu & Nocedal (1989) section 5 report 3 <= m <= 7 as
	//    generally adequate; 5 is the default. Configuration, so it survives a
	//    reset and is copied explicitly.
	void setMemory( const unsigned m );
	unsigned getMemory() const { return memory; }

	// PER-RUN reset: history, accepted point, scratch and counters. Called
	//    from Network::prepareRun(), which train() calls once before the loop
	//    and outside every reporting guard.
	void reset();

	// Has this run evaluated its starting point yet? A direct research call
	//    that bypasses train() must either initialize or refuse; iterate()
	//    below initializes on its first call, so there is no state in which it
	//    reads an uninitialized accepted point.
	bool started() const { return startedFlag; }

	// ONE OUTER ITERATION, from the last accepted point.
	//
	//    Returns the objective AT THE POINT THE STEP DEPARTED FROM -- f_k, not
	//    f_{k+1}. That is legacy pre-update reporting: every other optimizer in
	//    this engine returns the error computed during the pass that produced
	//    the gradient it then steps along, and currGradMax describes that same
	//    point. Returning the post-step objective would silently move what
	//    every stopping rule in Iterative compares.
	double iterate( LBFGSObjective& obj );

	// maxabs of the RAW gradient at the accepted point -- set before the
	//    direction is computed, never from a transformed direction (the plan's
	//    architecture decision 7).
	double gradMax() const { return gMax; }

	// Counted separately from outer iterations, because they are the currency:
	//    one L-BFGS iteration may cost many full passes, and an optimizer that
	//    wins on iterations while losing on passes has won nothing.
	unsigned long long objectiveEvaluations() const { return objEvals; }
	unsigned long long gradientEvaluations() const { return gradEvals; }
	unsigned long long lineSearchFailures() const { return failures; }
	unsigned long long historyResets() const { return resets; }
	unsigned long long curvatureRejections() const { return curvSkips; }
	unsigned long long cancellations() const { return cancels; }

	// How many curvature pairs are currently stored (0 .. m).
	unsigned pairs() const { return count; }

	// --- The published pieces, callable without a model -------------------
	//    Exposed so the deterministic tests can compare them against an
	//    independent dense inverse-BFGS oracle rather than against this same
	//    code running twice.

	// Nocedal & Wright Algorithm 7.4. Writes H_k * g into `result`, using the
	//    stored history and the equation-7.20 scaling.
	void applyInverseHessian( const vector< double >& g,
		vector< double >& result ) const;

	// The equation-7.20 scaling from the NEWEST accepted pair; 1.0 when the
	//    history is empty.
	double scaling() const;

	// Store a curvature pair if it satisfies the acceptance condition.
	//    Returns false when the pair is SKIPPED -- Liu & Nocedal drop pairs,
	//    they do not damp them.
	bool pushPair( const vector< double >& s, const vector< double >& y );

	// Drop the whole history. Not a failure by itself: it is what a
	//    non-descent direction and a failed line search both fall back to.
	void resetHistory();

	// Configuration only -- see the class note on copying. The working history
	//    is NOT carried: train() begins a new run and prepareRun() resets it,
	//    so a copy cannot promise a mid-run trajectory and does not pretend to.
	void copyConfigurationFrom( const LBFGS& rhs );

private:
	unsigned memory; // m

	// The ring buffers. `next` is where the following pair will be written and
	//    `count` how many are live, so the newest pair is at
	//    ( next + memory - 1 ) % memory and the oldest at
	//    ( next + memory - count ) % memory.
	vector< vector< double > > S, Y;
	vector< double > Rho;
	unsigned next, count;

	// The accepted point: parameters, objective, raw gradient.
	vector< double > w, g;
	double f, gMax;
	bool startedFlag;

	// Scratch, sized once per run rather than per iteration.
	mutable vector< double > q;      // two-loop working vector
	mutable vector< double > alphas; // two-loop alpha_i, one per stored pair
	vector< double > dir;            // the search direction
	vector< double > trialW, trialG; // the point the line search is testing
	vector< double > sNew, yNew;     // the candidate curvature pair

	// The line search's accepted result, valid only when lineSearch returned true
	double acceptedF;

	unsigned long long objEvals, gradEvals, failures, resets, curvSkips, cancels;

	// The evaluation count when the current line search began. The declared
	//    ceiling is PER SEARCH, not per run, so it needs a mark rather than an
	//    absolute number.
	unsigned long long searchStartEvals;

	// Index of the i-th oldest live pair, i in [0, count)
	unsigned slot( const unsigned i ) const;

	// Fill `dir` with the search direction at the accepted point. Steepest
	//    descent when the history is empty (N&W Algorithm 7.5 step 2).
	void computeDirection();

	// Is `dir` a usable descent direction -- finite everywhere, and downhill?
	bool isDescent() const;

	// Nocedal & Wright Algorithm 3.5 with Algorithm 3.6. True on acceptance,
	//    with the accepted point INSTALLED and trialW/trialG/acceptedF holding
	//    it. False on cancellation, evaluation ceiling, a degenerate bracket or
	//    a bracket that reaches ALPHA_MAX -- in which case the caller restores.
	bool lineSearch( LBFGSObjective& obj, const double alpha1,
		const double dphi0 );

	// One trial evaluation at step `a`. False means the search must stop:
	//    cancellation or the evaluation ceiling. A non-finite objective is NOT
	//    a stop -- it is reported as +infinity, which is exactly the condition
	//    Algorithm 3.5 uses to conclude that the step was too long.
	bool evaluateAt( LBFGSObjective& obj, const double a, double& phi,
		double& dphi );

	// Algorithm 3.6. The bracket's slope at aLo is not a parameter: bisection
	//    does not read it, and passing a value nothing uses would suggest the
	//    interpolation is something other than what it is.
	bool zoom( LBFGSObjective& obj, double aLo, double fLo,
		double aHi, const double phi0, const double dphi0,
		double& aOut, double& fOut );
};

#endif
