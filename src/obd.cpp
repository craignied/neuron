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

	bool onIteration( unsigned iteration, double setError ) override
	{
		if ( cancel && cancel->load() )
			return false;

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
			// The test error has clearly turned back up: overtraining
			if ( ++samplesAbove >= cfg->earlyStopPatience )
				return false;
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
	double& score, double& trainAtMin, ostringstream& discard )
{
	ValidationObserver obs;
	obs.net = &net; obs.testStride = testStride; obs.cfg = &cfg;
	obs.progress = progress; obs.cancel = cancel; obs.phase = phase; obs.hidden = hidden;

	net.setObserver( &obs );
	net.setMaxIterations( budget );

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
	return net.getStopReason();
}

void printTable( ostream& out, const vector< obd::SizeTrial >& history,
	unsigned selected, unsigned grewTo )
{
	out << endl << "OBD hidden-layer search (validation early stopping):" << endl;
	out << "  phase   hidden   train error   test error   CA train   CA test   "
		"stopped by" << endl;
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
		out << resetiosflags( ios::fixed ) << "   "
			<< ( t->stop == Iterative::STOP_OBSERVER ? "early stop"
				: t->stop == Iterative::STOP_PLATEAU ? "plateau"
				: t->stop == Iterative::STOP_GRADMAX ? "converged"
				: "budget" ) << endl;
	}
	out << "Selected: " << selected << " hidden nodes (grew to " << grewTo << ")." << endl;
}

} // namespace

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

	// Optimizer: fixed, or probed once (autoalgo) and kept for every size
	unsigned trainingType = ( cfg.algorithm >= 0 && cfg.algorithm <= 2 )
		? ( unsigned ) cfg.algorithm : 0;
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

	// Silence the bootstrap during the search; restore it on the winner
	unsigned savedBoot = disableBootstrap( net->getDataSet() );

	// --- GROW phase (warm-start-then-settle) --------------------------------
	double bestTestErr = numeric_limits< double >::infinity();
	unique_ptr< Network > bestNet;
	unsigned bestHidden = 0, grewTo = cfg.hStart, sinceImprovement = 0;

	for ( unsigned h = cfg.hStart; h <= cfg.hMax; h++ )
	{
		if ( cancel && cancel->load() ) { result.cancelled = true; break; }

		grewTo = h;
		double score, trainAtMin, trainCA, testCA;
		Iterative::StopReason stop = trainToValidationMin( *net, cfg, testStride,
			progress, cancel, "grow", h, cfg.iterBudget, score, trainAtMin, discard );
		readAccuracies( *net, trainCA, testCA );
		result.history.push_back( { h, trainAtMin, score, trainCA, testCA, stop, true } );

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
	if ( !result.cancelled && bestNet )
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

			unsigned budget = cfg.iterBudget / 4 < 100 ? 100 : cfg.iterBudget / 4;
			double score, trainAtMin, trainCA, testCA;
			Iterative::StopReason stop = trainToValidationMin( *work, cfg, testStride,
				progress, cancel, "prune", hCur, budget, score, trainAtMin, discard );
			readAccuracies( *work, trainCA, testCA );
			result.history.push_back( { hCur, trainAtMin, score, trainCA, testCA, stop, false } );

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
