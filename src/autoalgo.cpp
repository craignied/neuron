// Automatic training-algorithm selection (ROADMAP 2 Phase 2).

#include "stdafx.h" // For MSVC, must be first!

#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

#include "autoalgo.h"
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

	bool onIteration( unsigned iteration, double ) override
	{
		lastIteration = iteration;
		if ( cancel && cancel->load() )
			return false;
		return chrono::steady_clock::now() < deadline;
	}
};

const char* const ALGORITHM_NAMES[] = {
	"Canonical backpropagation",
	"Conjugate gradient descent",
	"Shanno's algorithm" };

} // namespace

autoalgo::Result autoalgo::pick( const Network& start, unsigned budgetMs,
	const atomic< bool >* cancel )
{
	Result result;

	ostream& callerScreen = util::screen(); // where the summary belongs
	ostringstream discard; // probes train into here, then it is thrown away
	double bestError = 0; // meaningful only once result.selected != 0

	for ( unsigned t = 0; t < 3; t++ )
	{
		if ( cancel && cancel->load() )
		{
			result.cancelled = true;
			break;
		}

		unique_ptr< Network > probe = cloneNetwork( start );
		if ( !probe ) // unknown Network type: nothing to select over
			break;

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
		//    the menu warns interactive users to.
		probe->setTrainingType( t );
		if ( t == 1 || t == 2 )
			probe->setBatchEpoch( true );

		// The budget is the only limit that should ever fire; convergence
		//    (max-gradient) may legitimately beat it
		probe->setMaxIterations( 1000000000 );

		BudgetObserver meter( budgetMs, cancel );
		probe->setObserver( &meter );

		double finalError;
		util::set_screen( discard );
		try
		{
			finalError = probe->train();
		}
		catch ( ... ) // a diverged probe is a result, not a failure
		{
			finalError = numeric_limits< double >::quiet_NaN();
		}
		util::set_screen( callerScreen );
		discard.str( "" ); // release the probe's report text
		probe->setObserver( nullptr );

		Probe p;
		p.algorithm = t + 1;
		p.name = ALGORITHM_NAMES[ t ];
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
	}

	// The one decision summary, on the caller's screen (the captured report)
	callerScreen << "Auto algorithm selection (" << budgetMs
		<< " ms per probe):" << endl;
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
			if ( p->stop == Iterative::STOP_OBSERVER )
				callerScreen << " (budget spent)";
			else if ( p->stop == Iterative::STOP_GRADMAX )
				callerScreen << " (CONVERGED inside the budget)";
			callerScreen << endl;
		}
		else
			callerScreen << "diverged" << endl;
	}
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
