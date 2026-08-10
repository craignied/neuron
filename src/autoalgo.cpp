// Automatic training-algorithm selection (ROADMAP 2 Phase 2), and the one
// eligibility rule the public surface consults. See autoalgo.h.

#include "stdafx.h" // For MSVC, must be first!

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

#include "autoalgo.h"
#include "logistic.h"
#include "netclone.h"
#include "utility.h"

using namespace std;

namespace {

// The observer that meters a probe: stop when the wall-clock budget is
//    spent or the caller cancels. Attached to the CLONE (Iterative::copy
//    nulls observers, so a clone never inherits one).
struct BudgetObserver : Iterative::Observer
{
	chrono::steady_clock::time_point deadline;
	const atomic< bool >* cancel;
	unsigned lastIteration = 0;

	BudgetObserver( unsigned budgetMs, const atomic< bool >* cancelFlag )
		: deadline( chrono::steady_clock::now()
			+ chrono::milliseconds( budgetMs ) ),
		cancel( cancelFlag ) {}

	// A probe is a bounded EXPERIMENT, not a final fit: when its window expires
	//    it must say so rather than borrow a convergence reason. A probe that
	//    ends this way has not converged and never claims to have.
	Iterative::StopReason reason = Iterative::STOP_PROBE_BUDGET;
	Iterative::StopReason whyStopped() const override { return reason; }

	bool onIteration( unsigned iteration, double ) override
	{
		lastIteration = iteration;
		if ( cancel && cancel->load() )
		{
			reason = Iterative::STOP_CANCELLED;
			return false;
		}
		if ( chrono::steady_clock::now() < deadline )
			return true;
		reason = Iterative::STOP_PROBE_BUDGET;
		return false;
	}
};

} // namespace

autoalgo::Settings autoalgo::Settings::of( Network& model )
{
	Settings s;
	s.batchEpoch = model.getBatchEpoch();
	s.autoStep = model.getAutoStepSize();
	s.parameters = model.packedSize();
	return s;
}

// THE ONE ELIGIBILITY RULE. Every condition below is one under which the named
//    method would have to become a different method to run, so the answer is a
//    refusal or an omission and never a coercion.
//
//    The engine states the same refusals again inside LBFGS::Ineligible,
//    IRpropState::Ineligible and LM::Ineligible, and that is deliberate defence
//    in depth rather than duplication: those guard the C++ boundary against any
//    caller, this one exists so a REQUEST is refused by name before a single
//    field is applied, and so an ineligible method is never probed at all.
const char* autoalgo::ineligible( const Network& model, unsigned trainingType,
	const Settings& s )
{
	// Canonical backpropagation, CGD and Shanno are the frozen menu trio. They
	//    run on every model this engine trains, in either mode, and the probe
	//    forces batch/epoch for the two that assume a true batch gradient.
	if ( trainingType <= 2 )
		return 0;

	if ( trainingType >= Network::TRAINING_TYPES )
		return "is not a training algorithm this engine has";

	// --- Shared by every retained modern optimizer -------------------------
	//    All three own an absolute step through the packed boundary, and none
	//    of them has a per-exemplar form.
	if ( dynamic_cast< const Logistic* >( &model ) )
		return "is available for neural models only";
	if ( model.packedSize() == 0 )
		return "needs a model that implements the packed parameter boundary";

	// --- Levenberg-Marquardt's own, narrower contract ----------------------
	if ( trainingType == Network::TRAIN_LM )
	{
		// THE STRICT LMS GATE. LM minimizes a sum of squares; cross-entropy is
		//    not one, and no residual vector's half f'f is that objective.
		if ( model.getXEerror() )
			return "requires least-squares error: cross-entropy is not a sum of "
				"squares, and no residual Jacobian represents it";
		if ( !model.normalEquationsAvailable() )
			return "is available for the single-hidden-layer least-squares "
				"network family only: this model does not implement the "
				"normal-equations boundary it needs";
	}

	if ( !s.batchEpoch )
		return "requires batch_epoch=1";
	if ( s.autoStep )
		return "requires autostep=0";

	// THE PARAMETER CEILING, and the LAST of Levenberg-Marquardt's six declared
	//    refusals. Judged against Settings::parameters, which is the model's own
	//    count for an ordinary run and the largest planned architecture for a
	//    procedure that resizes the model under the optimizer it chose.
	//
	//    THERE IS NO SEVENTH. A validation / early-stopping split does not appear
	//    here, and must not: it changes neither LM's objective nor Algorithm
	//    3.16, Iterative still owns stopping and can still monitor a held-out
	//    set, and lm_source_decision.md section 9 -- written before any of this
	//    code and frozen since -- declares exactly six. Narrowing a retained
	//    algorithm after its measurement is how a published method quietly
	//    becomes a different one.
	if ( trainingType == Network::TRAIN_LM
		&& s.parameters > LM::MAX_PARAMETERS )
		return "is limited to small least-squares networks: this model has "
			"more parameters than the declared ceiling";

	return 0;
}

autoalgo::Plan autoalgo::plan( const Network& model,
	const vector< unsigned >& requested, const Settings& s )
{
	Plan p;

	// ONE PASS IN ASCENDING ORDER over the enumeration, asking of each type
	//    whether it was requested. Walking the REQUEST instead would carry the
	//    caller's order into the competitor list -- and the tie policy is that
	//    order (autoalgo.h) -- and would have to defend against a repeated token
	//    producing a method that competes twice.
	for ( unsigned t = 0; t < Network::TRAINING_TYPES; t++ )
	{
		if ( !requested.empty()
			&& find( requested.begin(), requested.end(), t ) == requested.end() )
			continue; // not asked for: neither a competitor nor an omission

		const char* why = ineligible( model, t, s );
		if ( !why )
		{
			p.competitors.push_back( t );
			continue;
		}
		Omission o;
		o.algorithm = t + 1;
		o.name = Network::algorithmName( t );
		o.reason = why;
		p.omitted.push_back( o );
	}

	// A TOKEN THIS ENGINE HAS NO ALGORITHM FOR IS REFUSED, NOT DROPPED. The loop
	//    above can only see the enumeration, so anything past it would otherwise
	//    leave no trace at all -- and a caller that named four methods and got a
	//    three-way competition would never learn which one vanished. Callers
	//    validate the range before they get here; this is what makes the
	//    resolution total rather than trusting that they did.
	for ( vector< unsigned >::const_iterator r = requested.begin();
		r != requested.end(); r++ )
	{
		if ( *r < Network::TRAINING_TYPES )
			continue;
		Omission o;
		o.algorithm = *r + 1;
		o.name = "unknown algorithm";
		o.reason = ineligible( model, *r, s );
		p.omitted.push_back( o );
	}
	return p;
}

vector< unsigned > autoalgo::candidates( const Network& model, const Settings& s )
{
	return plan( model, vector< unsigned >(), s ).competitors;
}

autoalgo::Result autoalgo::pick( const Network& start,
	const vector< unsigned >& requested, unsigned plannedParameters,
	unsigned totalBudgetMs, const atomic< bool >* cancel )
{
	Result result;
	result.totalBudgetMs = totalBudgetMs;

	ostream& callerScreen = util::screen(); // where the summary belongs
	ostringstream discard; // probes train into here, then it is thrown away
	double bestError = 0; // meaningful only once result.selected != 0

	// One clone answers the eligibility question and then serves as the first
	//    probe, so asking costs no extra copy of the dataset.
	unique_ptr< Network > probe = cloneNetwork( start );
	if ( !probe ) // unknown Network type: nothing to select over
		return result;

	Settings settings = Settings::of( *probe );
	// A MAXIMUM, never an assignment: a caller that does not resize the model
	//    passes 0 and cannot thereby widen eligibility past the model's own
	//    parameter count.
	if ( plannedParameters > settings.parameters )
		settings.parameters = plannedParameters;

	// THE CANDIDATE LIST IS SETTLED BEFORE ANY PROBING, from the pristine clone.
	//    It cannot be decided inside the loop: the winning clone is MOVED into
	//    the result, so there is no model left to ask by the time the next
	//    candidate comes up. AN INELIGIBLE METHOD IS OMITTED WITH ITS REASON,
	//    never attempted -- a probe of a method that refuses its configuration
	//    throws Ineligible, the catch below turns any exception into a NaN, and
	//    the summary would then tell the user that a perfectly good optimizer
	//    diverged.
	const Plan resolved = plan( *probe, requested, settings );
	const vector< unsigned >& list = resolved.competitors;
	result.omitted = resolved.omitted;

	// NOTHING TO COMPARE. One eligible method is the answer already; probing it
	//    would spend the budget to rediscover it, and the run that followed would
	//    start from the probe's weights rather than the caller's. Say which
	//    method it is, spend nothing, and adopt nothing -- competed stays false
	//    and there is no winner, so the caller trains its OWN model on it.
	if ( list.size() == 1 )
	{
		result.selected = list[ 0 ] + 1;
		result.selectedName = Network::algorithmName( list[ 0 ] );
		callerScreen << "Algorithm: " << result.selectedName
			<< " (the only eligible method of those requested; no competition"
			" was run)." << endl;
		for ( vector< Omission >::iterator o = result.omitted.begin();
			o != result.omitted.end(); o++ )
			callerScreen << "   " << o->name << ": not eligible here -- it "
				<< o->reason << endl;
		return result;
	}
	if ( list.empty() ) // nothing requested can run here; the caller reports why
		return result;
	result.competed = true;

	// A POSITIVE, EQUAL SHARE IS PART OF THE CALLER CONTRACT. Silently flooring
	//    an impossible share to 1 ms would spend MORE than the declared total,
	//    so refuse the invalid budget rather than publish two contradictory
	//    numbers. Production passes 2250 ms and is far from this boundary.
	if ( totalBudgetMs < list.size() )
		throw invalid_argument( "automatic-selection total budget must provide "
			"at least 1 ms per eligible candidate" );

	// THE BUDGET IS DIVIDED, NOT MULTIPLIED. Retaining a fourth, fifth or sixth
	//    algorithm must not make every automatic selection -- and every OBD
	//    search, and every nested cross-validation fold -- linearly slower, so
	//    the total is fixed and the eligible candidates share it equally. The
	//    integer remainder is deliberately left unspent: handing it to the first
	//    few candidates would give them a longer window than the rest, and the
	//    comparison that follows is between final errors reached in equal time.
	result.perCandidateBudgetMs = totalBudgetMs / ( unsigned ) list.size();

	for ( vector< unsigned >::const_iterator it = list.begin();
		it != list.end(); it++ )
	{
		const unsigned t = *it;

		if ( cancel && cancel->load() )
		{
			result.cancelled = true;
			break;
		}

		if ( !probe ) // the previous candidate consumed the clone
		{
			probe = cloneNetwork( start );
			if ( !probe )
				break;
		}

		// The probe must leave no trace: no model.txt / neuron.log writes,
		//    and no 2000-resample ROC bootstrap in its (discarded) report
		probe->setLastop( false );
		probe->setHistory( false );
		DataSet& d = probe->getDataSet();
		if ( d.getDiscrete() && d.getOutput() == 1 && d.trainLoaded() )
		{
			d.getTrainTwoSet().setBootstrapResamples( 0 );
			if ( d.testLoaded() )
				d.getTestTwoSet().setBootstrapResamples( 0 );
		}

		// Configure THIS clone's algorithm (after copying, so the clones all
		//    start from identical weights). CGD and Shanno assume a true
		//    batch gradient; their probes force epoch training exactly as
		//    the menu warns interactive users to. NOTHING ELSE IS COERCED --
		//    the retained modern optimizers were eligible only because the
		//    configuration already satisfied them.
		probe->setTrainingType( t );
		if ( t == 1 || t == 2 )
			probe->setBatchEpoch( true );

		// The budget is the only limit that should ever fire; convergence
		//    (max-gradient) may legitimately beat it
		probe->setMaxIterations( 1000000000 );

		BudgetObserver meter( result.perCandidateBudgetMs, cancel );
		probe->setObserver( &meter );

		double finalError;
		{
			util::ScreenCapture quiet; // restores on the throw path too
			try
			{
				finalError = probe->train();
			}
			// A PROGRAMMER-CONTRACT FAILURE IS NOT A DIVERGED PROBE (D9).
			//    These three say a caller handed the numerical layer shapes or
			//    indices that cannot be right; turning one into a NaN would
			//    report "that optimizer diverged" and quietly continue with the
			//    others. They pass through to the worker or CLI boundary, which
			//    reports what() to the user. Everything else -- a numerical
			//    divergence, a singular matrix, a statistics failure -- keeps
			//    its old meaning below.
			catch ( const Matrix< double >::BoundsViolation& ) { throw; }
			catch ( const Matrix< double >::DimensionMismatch& ) { throw; }
			catch ( const Matrix< double >::BadSize& ) { throw; }
			catch ( ... ) // a diverged probe is a result, not a failure
			{
				finalError = numeric_limits< double >::quiet_NaN();
			}
		} // the probe's report text is released with the capture
		probe->setObserver( nullptr );

		Probe p;
		p.algorithm = t + 1;
		p.name = Network::algorithmName( t );
		p.error = finalError;
		p.iterations = meter.lastIteration;
		p.stop = probe->getStopReason();
		// train() returns -1 when something went wrong; NaN/inf is divergence
		p.usable = isfinite( finalError ) && finalError >= 0;
		result.probes.push_back( p );

		if ( cancel && cancel->load() ) // fired during this probe
		{
			result.cancelled = true;
			break;
		}

		// Lowest final training error wins; a strict comparison keeps the
		//    SIMPLER algorithm on ties (probes run in order of simplicity)
		if ( p.usable && ( result.selected == 0 || p.error < bestError ) )
		{
			bestError = p.error;
			result.selected = p.algorithm;
			result.selectedName = p.name;
			result.winner = std::move( probe );
		}
		probe.reset(); // a used probe is never reused; the next one clones fresh
	}

	// The one decision summary, on the caller's screen (the captured report)
	callerScreen << "Auto algorithm selection (" << totalBudgetMs
		<< " ms total, " << result.perCandidateBudgetMs << " ms each across "
		<< list.size() << " eligible candidates):" << endl;
	for ( vector< Probe >::iterator p = result.probes.begin();
		p != result.probes.end(); p++ )
	{
		callerScreen << "   " << p->name << ": ";
		if ( p->usable )
		{
			callerScreen << "error = " << resetiosflags( ios::fixed )
				<< setiosflags( ios::scientific ) << setprecision( 6 )
				<< p->error << resetiosflags( ios::scientific )
				<< " after " << p->iterations << " iterations";
			if ( p->stop == Iterative::STOP_PROBE_BUDGET )
				callerScreen << " (budget spent)";
			else if ( p->stop == Iterative::STOP_CANCELLED )
				callerScreen << " (cancelled)";
			else if ( p->stop == Iterative::STOP_GRADMAX )
				callerScreen << " (CONVERGED inside the budget)";
			callerScreen << endl;
		}
		else
			callerScreen << "diverged" << endl;
	}
	// Say what was NOT probed and why, in the same summary. A method missing
	//    from a list with no explanation reads as a method that lost.
	for ( vector< Omission >::iterator o = result.omitted.begin();
		o != result.omitted.end(); o++ )
		callerScreen << "   " << o->name << ": not eligible here -- it "
			<< o->reason << endl;
	if ( result.cancelled )
		callerScreen << "   Selection cancelled by request." << endl;
	else if ( result.selected )
		callerScreen << "Selected: " << result.selectedName << endl;
	else
		callerScreen << "No probe produced a usable error; keeping the "
			"configured algorithm." << endl;

	if ( result.cancelled ) // a cancelled selection adopts nothing
	{
		result.selected = 0;
		result.selectedName.clear();
		result.winner.reset();
	}

	return result;
}
