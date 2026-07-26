// Optimal Brain Damage hidden-layer sizing (ROADMAP 2 Phase 4). See obd.h.

#include "stdafx.h" // For MSVC, must be first!

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

#include "obd.h"
#include "simpleprop.h"
#include "netclone.h"
#include "autoalgo.h"
#include "utility.h"

using namespace std;

namespace {

// Watches the held-out error during one training run and stops it at the onset
//    of overtraining (validation early stopping). Sampled every cfg.sampleEvery
//    iterations (iteration-based, so a seeded run is reproducible); tracks the
//    running MINIMUM test error and stops when the test error has risen
//    earlyStopTol above that minimum for earlyStopPatience samples. Also carries
//    the Stop request (the cancel flag) into the loop and feeds the progress
//    callback. A non-finite test-error sample is not judged -- training simply
//    continues -- so a transiently diverged probe is not mistaken for overfit.
struct ValidationObserver : Iterative::Observer
{
	Network* net = nullptr;              // the net being trained (sampled for test error)
	unsigned testStride = 1;             // sampleTestError subsample stride
	const obd::Config* cfg = nullptr;
	obd::ProgressFn progress;
	const atomic< bool >* cancel = nullptr;
	const char* phase = "grow";
	unsigned hidden = 0;

	double minTestErr = numeric_limits< double >::infinity();
	double trainAtMin = numeric_limits< double >::infinity();
	unsigned samplesAbove = 0;
	unsigned lastIteration = 0; // iterations this trial actually used

	// Why this observer stopped the run. Both exits below return false, but a
	//    caller cancel and held-out deterioration are different facts: the first
	//    makes the trial INELIGIBLE, the second is a meaningful stopping rule.
	Iterative::StopReason reason = Iterative::STOP_CANCELLED;
	Iterative::StopReason whyStopped() const override { return reason; }

	bool onIteration( unsigned iteration, double setError ) override
	{
		lastIteration = iteration;
		if ( cancel && cancel->load() )
		{
			reason = Iterative::STOP_CANCELLED;
			return false;
		}

		if ( iteration % cfg->sampleEvery != 0 ) // sample on the cadence only
			return true;

		double testErr = net->sampleTestError( testStride );
		if ( progress )
			progress( phase, hidden, iteration, setError, testErr );

		if ( !isfinite( testErr ) || testErr < 0 ) // can't judge; keep training
			return true;

		if ( testErr < minTestErr ) // a new best -- the score, and the reset
		{
			minTestErr = testErr;
			trainAtMin = setError;
			samplesAbove = 0;
		}
		else if ( testErr > minTestErr * ( 1.0 + cfg->earlyStopTol ) )
		{
			// The test error has clearly turned back up: overtraining. This is
			//    a meaningful stopping rule -- the size is scored at its
			//    held-out minimum and the trial is eligible.
			if ( ++samplesAbove >= cfg->earlyStopPatience )
			{
				reason = Iterative::STOP_EARLY_STOP;
				return false;
			}
		}

		return true;
	}
};

// Silence the ROC bootstrap on a search net's TwoSets: each size's train() runs
//    reportAccuracy, and a 2000-resample bootstrap per size would dwarf the
//    training itself. Returns the prior resample count so the winner can restore it.
unsigned disableBootstrap( DataSet& d )
{
	unsigned saved = 2000;
	if ( d.getDiscrete() && d.getOutput() == 1 && d.trainLoaded() )
	{
		saved = d.getTrainTwoSet().getBootstrapResamples();
		d.getTrainTwoSet().setBootstrapResamples( 0 );
		if ( d.testLoaded() )
			d.getTestTwoSet().setBootstrapResamples( 0 );
	}
	return saved;
}

void restoreBootstrap( DataSet& d, unsigned resamples )
{
	if ( d.getDiscrete() && d.getOutput() == 1 && d.trainLoaded() )
	{
		d.getTrainTwoSet().setBootstrapResamples( resamples );
		if ( d.testLoaded() )
			d.getTestTwoSet().setBootstrapResamples( resamples );
	}
}

// Configure a search net: quiet (no logs/prints), the chosen optimizer, and
//    batch/epoch forced for CGD/Shanno (they assume a true batch gradient).
void configureNet( SimpleProp& net, unsigned trainingType )
{
	net.setLastop( false );
	net.setHistory( false );
	net.setLogPrint( false );
	net.setTrainingType( trainingType );
	if ( trainingType == 1 || trainingType == 2 )
		net.setBatchEpoch( true );
}

// A trial's stop reason, printed in the size table. Kept beside classify() so
//    the label and the eligibility decision can never drift apart.
const char* stopLabel( Iterative::StopReason s )
{
	return s == Iterative::STOP_EARLY_STOP ? "early stop"
		: s == Iterative::STOP_PLATEAU ? "plateau"
		: s == Iterative::STOP_GRADMAX ? "converged"
		: s == Iterative::STOP_MIN_ERROR ? "min error"
		: s == Iterative::STOP_CHANGE ? "min change"
		: s == Iterative::STOP_WINDOW ? "error window"
		: s == Iterative::STOP_CANCELLED ? "cancelled"
		: s == Iterative::STOP_MAX_ITERATIONS ? "CEILING"
		: "other";
}

unsigned argmin( const vector< double >& v )
{
	unsigned best = 0;
	for ( unsigned i = 1; i < v.size(); i++ )
		if ( v[ i ] < v[ best ] )
			best = i;
	return best;
}

// Read the classification accuracy the last train() epilogue's reportAccuracy
//    left in a net's TwoSets (the guesses at the early-stop point). Fractions in
//    0..1; -1 when the set has no usable TwoSet. Measured at the early-stop
//    endpoint, which is where the snapshot net sits -- a few patience samples
//    past the min-error point the testErr score records.
void readAccuracies( SimpleProp& net, double& trainCA, double& testCA )
{
	trainCA = testCA = -1;
	DataSet& d = net.getDataSet();
	if ( d.trainLoaded() && d.getTrainTwoSet().loaded() )
		trainCA = d.getTrainTwoSet().getClassAcc();
	if ( d.testLoaded() && d.getTestTwoSet().loaded() )
		testCA = d.getTestTwoSet().getClassAcc();
}

// Train one size to its validation minimum (early stopped). Fills score (the
//    minimum test error) and trainAtMin; returns the net's stop reason.
Iterative::StopReason trainToValidationMin( SimpleProp& net, const obd::Config& cfg,
	unsigned testStride, obd::ProgressFn progress, const atomic< bool >* cancel,
	const char* phase, unsigned hidden, unsigned budget,
	double& score, double& trainAtMin, unsigned& iterationsUsed,
	ostringstream& discard )
{
	ValidationObserver obs;
	obs.net = &net; obs.testStride = testStride; obs.cfg = &cfg;
	obs.progress = progress; obs.cancel = cancel; obs.phase = phase; obs.hidden = hidden;

	net.setObserver( &obs );
	net.setMaxIterations( budget );

	// The three ways a size ends, whichever fires first: overtraining onset
	//    (the observer above), a TRAIN-error plateau (a size that converges
	//    flat never trips the test-error rise, so without this it would burn
	//    the whole budget -- the engine's plateau detector ends it), or the
	//    iteration budget as the backstop.
	net.setAutoStop( true, cfg.plateauTol, cfg.plateauWindow );

	ostream& screen = util::screen();
	util::set_screen( discard );
	net.train();
	util::set_screen( screen );
	discard.str( "" ); // drop the per-size report
	net.setObserver( nullptr );

	// The score is the tracked minimum; fall back to a final sample if the run
	//    was too short to sample (budget < sampleEvery)
	score = isfinite( obs.minTestErr ) ? obs.minTestErr : net.sampleTestError( testStride );
	trainAtMin = isfinite( obs.trainAtMin ) ? obs.trainAtMin : score;
	iterationsUsed = obs.lastIteration;
	return net.getStopReason();
}

void printTable( ostream& out, const vector< obd::SizeTrial >& history,
	unsigned selected, unsigned grewTo )
{
	out << endl << "OBD hidden-layer search (validation early stopping):" << endl;
	out << "  phase   hidden   train error   test error   CA train   CA test   "
		"iters   stopped by" << endl;
	for ( vector< obd::SizeTrial >::const_iterator t = history.begin();
		t != history.end(); t++ )
	{
		out << "  " << ( t->phaseGrow ? "grow " : "prune" )
			<< setw( 8 ) << t->hidden
			<< "   " << resetiosflags( ios::fixed ) << setiosflags( ios::scientific )
			<< setprecision( 4 ) << setw( 11 ) << t->trainErr
			<< "  " << setw( 11 ) << t->testErr
			<< resetiosflags( ios::scientific ) << setiosflags( ios::fixed )
			<< setprecision( 1 );
		if ( t->trainCA >= 0 ) out << setw( 9 ) << t->trainCA * 100 << "%";
		else out << setw( 10 ) << "n/a";
		if ( t->testCA >= 0 ) out << setw( 8 ) << t->testCA * 100 << "%";
		else out << setw( 9 ) << "n/a";
		out << resetiosflags( ios::fixed ) << setw( 8 ) << t->iterations << "   "
			<< stopLabel( t->stop );
		// An ineligible trial is marked in the table itself: it took no part in
		//    the comparison, and a reader must not mistake its numbers for a fit.
		if ( t->eligibility != obd::ELIGIBLE )
			out << "  <- NOT a fitted model; excluded from comparison";
		out << endl;
	}
	if ( selected )
		out << "Selected: " << selected << " hidden nodes (grew to " << grewTo << ")." << endl;
	else
		out << "Selected: NONE -- no architecture was chosen (see below)." << endl;
}

} // namespace

obd::Eligibility obd::classify( Iterative::StopReason stop,
	double score, double trainErr )
{
	// A trial whose numbers are not finite tells us nothing, whatever ended it.
	if ( !isfinite( score ) || !isfinite( trainErr ) )
		return NUMERICAL_FAILURE;

	// The convergence question is the ENGINE's, not OBD's -- the training report
	//    and the CV adapters ask the same predicate, so "converged" cannot mean
	//    one thing here and another there (rule 6). OBD adds only the reasons a
	//    SEARCH cares about on top of it.
	if ( Iterative::converged( stop ) )
		return ELIGIBLE;
	if ( stop == Iterative::STOP_CANCELLED )
		return INCOMPLETE_CANCELLED;
	// MAX_ITERATIONS, PROBE_BUDGET and NONE all mean the fit never finished.
	return INCOMPLETE_CEILING;
}

const char* obd::stopToken( Iterative::StopReason stop, Eligibility e )
{
	// A trial whose numbers are not finite is reported as the numerical failure
	//    it is, whatever ended the loop; otherwise the engine's one spelling.
	if ( e == NUMERICAL_FAILURE ) return "numerical_failure";
	return Iterative::stopReasonToken( stop );
}

obd::Result obd::run( DataSet& data, const Config& cfg,
	ProgressFn progress, const atomic< bool >* cancel )
{
	Result result;

	// --- Refusals -----------------------------------------------------------
	if ( !data.trainLoaded() )
	{
		result.message = "load a training set first";
		return result;
	}
	if ( !( data.getDiscrete() && data.getOutput() == 1 ) )
	{
		result.message = "OBD needs a discrete, single-output dataset";
		return result;
	}
	if ( !data.testLoaded() )
	{
		result.message = "OBD needs a held-out test set -- it is the validation "
			"signal early stopping watches";
		return result;
	}
	if ( cfg.hStart < 1 || cfg.hMax < cfg.hStart )
	{
		result.message = "the hidden-unit range is empty (need 1 <= hStart <= hMax)";
		return result;
	}

	ostream& screen = util::screen();
	ostringstream discard; // per-size training reports are thrown away
	unsigned testStride = data.getNumTest() > 1000 ? data.getNumTest() / 1000 : 1;

	// --- Build the starting net ---------------------------------------------
	unique_ptr< SimpleProp > net = make_unique< SimpleProp >();
	net->setDataSet( data );
	net->setHidden( cfg.hStart );
	net->setLastop( false ); net->setHistory( false ); net->setLogPrint( false );
	net->randomize();

	// Optimizer: fixed, or probed once (autoalgo) and kept for every size. Auto
	//    is a procedure for CHOOSING an optimizer, not an optimizer: it runs once
	//    per independent search, and the winner is then used for every grow and
	//    prune trial below. A probe ending on its window is a bounded experiment
	//    (STOP_PROBE_BUDGET) and never counts as a converged fit.
	unsigned trainingType = ( cfg.algorithm >= 0 && cfg.algorithm <= 2 )
		? ( unsigned ) cfg.algorithm : 0;
	result.autoSelected = ( cfg.algorithm < 0 );
	if ( cfg.algorithm < 0 ) // auto
	{
		util::set_screen( discard );
		autoalgo::Result pick = autoalgo::pick( *net, 750, cancel );
		util::set_screen( screen );
		discard.str( "" );
		if ( pick.cancelled )
		{
			result.cancelled = true;
			result.message = "cancelled during algorithm selection";
			return result;
		}
		if ( pick.selected )
			trainingType = pick.selected - 1;
		screen << "OBD: auto-selected "
			<< ( pick.selected ? pick.selectedName : string( "canonical backpropagation" ) )
			<< " for the search." << endl;
	}
	configureNet( *net, trainingType );
	result.algorithm = ( int ) trainingType; // observable: what the search ran on

	// Silence the bootstrap during the search; restore it on the winner
	unsigned savedBoot = disableBootstrap( net->getDataSet() );

	// --- GROW phase (warm-start-then-settle) --------------------------------
	double bestTestErr = numeric_limits< double >::infinity();
	unique_ptr< Network > bestNet;
	unsigned bestHidden = 0, grewTo = cfg.hStart, sinceImprovement = 0;

	// A trial the search NEEDED could not be finished. Record it, abandon the
	//    search, and say exactly why -- never fall through to a nominal winner.
	bool halted = false;

	for ( unsigned h = cfg.hStart; h <= cfg.hMax; h++ )
	{
		if ( cancel && cancel->load() ) { result.cancelled = true; break; }

		grewTo = h;
		double score, trainAtMin, trainCA, testCA;
		unsigned used = 0;
		Iterative::StopReason stop = trainToValidationMin( *net, cfg, testStride,
			progress, cancel, "grow", h, cfg.iterBudget, score, trainAtMin,
			used, discard );
		readAccuracies( *net, trainCA, testCA );
		Eligibility el = classify( stop, score, trainAtMin );
		result.history.push_back( { h, trainAtMin, score, trainCA, testCA, stop,
			true, el, used } );

		// An INELIGIBLE trial is not a fit. Its loss is not compared, it is not
		//    snapshotted as the winner, and pruning never starts from it.
		if ( el != ELIGIBLE )
		{
			if ( el == INCOMPLETE_CANCELLED ) result.cancelled = true;
			else result.ceilingExhausted = ( el == INCOMPLETE_CEILING );
			halted = true;
			break;
		}

		if ( score < bestTestErr ) // a larger net helped: snapshot it
		{
			bestTestErr = score;
			bestHidden = h;
			bestNet = cloneNetwork( *net );
			sinceImprovement = 0;
		}
		else if ( ++sinceImprovement >= cfg.growPatience )
			break; // extra capacity stopped helping

		if ( cancel && cancel->load() ) { result.cancelled = true; break; }
		if ( h < cfg.hMax )
			net->growHidden( 1 ); // warm start the next size
	}

	// --- PRUNE phase --------------------------------------------------------
	if ( !halted && !result.cancelled && bestNet )
	{
		unique_ptr< Network > workBase = cloneNetwork( *bestNet );
		SimpleProp* work = dynamic_cast< SimpleProp* >( workBase.get() );
		unsigned hCur = bestHidden;

		while ( work && hCur > 1 )
		{
			if ( cancel && cancel->load() ) { result.cancelled = true; break; }

			vector< double > sal = work->hiddenSaliency();
			work->removeHidden( { argmin( sal ) } );
			hCur--;

			// The SAME configured ceiling as a grow trial. A quarter-budget would
			//    be a second, undocumented ceiling -- and since reaching a ceiling
			//    now refuses the search, an internal one would fail runs the user
			//    configured perfectly well. Pruning warm-starts from a fitted net,
			//    so it converges quickly anyway.
			double score, trainAtMin, trainCA, testCA;
			unsigned used = 0;
			Iterative::StopReason stop = trainToValidationMin( *work, cfg, testStride,
				progress, cancel, "prune", hCur, cfg.iterBudget, score, trainAtMin,
				used, discard );
			readAccuracies( *work, trainCA, testCA );
			Eligibility el = classify( stop, score, trainAtMin );
			result.history.push_back( { hCur, trainAtMin, score, trainCA, testCA, stop,
				false, el, used } );

			// An incomplete pruned candidate is never accepted -- and because its
			//    loss is meaningless the search cannot continue past it either.
			if ( el != ELIGIBLE )
			{
				if ( el == INCOMPLETE_CANCELLED ) result.cancelled = true;
				else result.ceilingExhausted = ( el == INCOMPLETE_CEILING );
				halted = true;
				break;
			}

			if ( score <= bestTestErr * ( 1.0 + cfg.pruneTol ) ) // small net still good
			{
				if ( score < bestTestErr )
					bestTestErr = score;
				bestHidden = hCur;
				bestNet = cloneNetwork( *work );
			}
			else
				break; // the last accepted net stands
		}
	}

	// --- Finish -------------------------------------------------------------
	// A search that could not finish a trial it needed reports THAT, with the
	//    whole trial table, and selects nothing. Comparing unfinished fits is the
	//    defect this refusal exists to prevent, so there is no "best effort" path.
	if ( halted && !result.cancelled )
	{
		const SizeTrial& t = result.history.back();
		ostringstream m;
		if ( t.eligibility == NUMERICAL_FAILURE )
			m << "the " << t.hidden << "-hidden trial produced a non-finite error "
				<< "(a diverged fit) -- try a different optimizer or a lower "
				<< "learning rate. No architecture was selected.";
		else
			m << "training did not converge: the " << t.hidden << "-hidden "
				<< ( t.phaseGrow ? "grow" : "prune" ) << " trial ran out of its "
				<< cfg.iterBudget << "-iteration ceiling before any stopping "
				<< "condition fired. An iteration ceiling is a safety limit, not "
				<< "a stopping rule, so its weights are not a fitted model and "
				<< "NO architecture was selected. Raise the iteration budget, or "
				<< "change the optimizer or stopping conditions, and run again.";
		result.message = m.str();
		printTable( screen, result.history, 0, grewTo );
		screen << "OBD REFUSED: " << result.message << endl;
		return result;
	}

	if ( !bestNet )
	{
		result.message = result.cancelled
			? "cancelled before any size completed"
			: "no hidden size produced a usable model";
		return result;
	}

	result.selectedHidden = bestHidden;
	printTable( screen, result.history, bestHidden, grewTo );

	// The winner is already at its validation minimum -- do NOT train it further
	//    (that is the overtraining the search avoided). Re-enable the bootstrap
	//    and run reportAccuracy once to produce the full final report/stats.
	restoreBootstrap( bestNet->getDataSet(), savedBoot );
	bestNet->reportAccuracy( screen );

	result.winner = std::move( bestNet );
	result.ok = true;
	return result;
}
