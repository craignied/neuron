/* CV procedure adapters -- see cvadapters.h and rule 6. */

#include "cvadapters.h"

#include "netclone.h"
#include "twoset.h"
#include "ldfa.h"
#include "qdfa.h"
#include "obd.h"
#include "split.h"
#include "iterative.h"
#include "utility.h"

#include <atomic>
#include <memory>
#include <sstream>

using crossval::ProcResult;

// A value no sigmoidal guess in (0,1) can take -- used to detect whether a fit
//    actually WROTE the held-out guesses. A singular LDFA/QDFA catches the
//    exception and returns WITHOUT calling reportAccuracy, leaving the guess
//    column untouched; poisoning it first turns that silent non-write into a
//    detectable failure instead of reading unwritten storage (bug B3).
static const double GUESS_SENTINEL = 4.2424242e18;

// Read a fitted model's held-out predictions -- the guesses reportAccuracy (run
//    in the train epilogue) wrote into the test TwoSet, in test-row order.
static vector< double > heldoutPredictions( Model& m )
{
	TwoSet& te = m.getDataSet().getTestTwoSet();
	unsigned n = te.getNumElements();
	vector< double > preds( n );
	for ( unsigned i = 0; i < n; i++ ) preds[ i ] = te.test( i );
	return preds;
}

// Fit a model with its report discarded; return true only if the fit actually
//    produced held-out guesses. The guess column is poisoned first, so a fit
//    that never writes it (a singular DFA) is caught rather than read as garbage.
static bool fitQuietly( Model& m, double* finalError = nullptr )
{
	TwoSet& te = m.getDataSet().getTestTwoSet();
	te.setBootstrapResamples( 0 ); // point predictions
	unsigned n = te.getNumElements();
	for ( unsigned i = 0; i < n; i++ ) te.test( i ) = GUESS_SENTINEL;

	ostream& saved = util::screen();
	ostringstream discard;
	util::set_screen( discard );
	double err = m.train(); // epilogue writes the guesses -- unless the fit failed
	util::set_screen( saved );
	if ( finalError ) *finalError = err;

	if ( n == 0 ) return false;
	for ( unsigned i = 0; i < n; i++ )
		if ( te.test( i ) != GUESS_SENTINEL ) return true; // a guess was written
	return false; // nothing written -- the fit produced no predictions
}

// An observer that stops a training run the moment the Stop flag fires.
namespace {
struct CancelObserver : Iterative::Observer
{
	const atomic< bool >* cancel = nullptr;
	bool onIteration( unsigned, double ) override
	{
		return !( cancel && cancel->load() );
	}
	// The only way this observer stops a run is a cancel, which is the
	//    Observer default -- named here so the intent is explicit.
	Iterative::StopReason whyStopped() const override
	{
		return Iterative::STOP_CANCELLED;
	}
};
}

crossval::Procedure cvadapters::trainProcedure( const Network& templateNet,
	unsigned maxIter )
{
	return [ &templateNet, maxIter ]( DataSet& foldData,
		const vector< unsigned >&, const vector< unsigned >&,
		const atomic< bool >* cancel ) -> ProcResult
	{
		ProcResult pr;
		unique_ptr< Network > clone = cloneNetwork( templateNet );
		clone->setDataSet( foldData ); // retarget to this fold (weights survive)
		clone->randomize();            // fresh weights for an honest fit
		clone->setMaxIterations( maxIter );

		CancelObserver obs; obs.cancel = cancel;
		clone->setObserver( &obs );
		double finalError = 0;
		bool produced = fitQuietly( *clone, &finalError );
		clone->setObserver( nullptr );

		if ( cancel && cancel->load() ) { pr.cancelled = true; return pr; }
		if ( !produced ) { pr.reason = "training produced no predictions"; return pr; }

		// Writing predictions is not the same as having FITTED. A run that ended
		//    at the iteration ceiling did not converge; its weights are wherever
		//    the run happened to be, so its held-out predictions are not this
		//    procedure's answer and must not enter a pooled AUC, a fold mean, or
		//    a locked-test contrast. Same rule the engine report and OBD use.
		Iterative::StopReason stop = clone->getStopReason();
		if ( !isfinite( finalError ) )
		{
			pr.reason = "training produced a non-finite error (a diverged fit)";
			return pr;
		}
		if ( !Iterative::converged( stop ) )
		{
			pr.reason = string( "training did not converge (" )
				+ Iterative::stopReasonToken( stop ) + "): it stopped at the "
				+ to_string( maxIter ) + "-iteration ceiling, which is a safety "
				"limit, not a stopping condition, so this fold contributes no "
				"prediction. Raise the per-fold iteration cap, or set a stopping "
				"condition that can fire.";
			return pr;
		}

		pr.ok = true;
		pr.pred = heldoutPredictions( *clone );
		return pr;
	};
}

crossval::Procedure cvadapters::dfaProcedure( bool quadratic )
{
	return [ quadratic ]( DataSet& foldData,
		const vector< unsigned >&, const vector< unsigned >&,
		const atomic< bool >* ) -> ProcResult
	{
		ProcResult pr;
		unique_ptr< Model > dfa = quadratic
			? unique_ptr< Model >( make_unique< QDFA >() )
			: unique_ptr< Model >( make_unique< LDFA >() );
		dfa->setDataSet( foldData );
		dfa->setHistory( false );
		dfa->setLastop( false );

		// DFA::train computes the discriminant + writes guesses -- but on a
		//    singular within-class covariance it catches the exception and writes
		//    nothing. fitQuietly reports that as a failed fold (bug B3).
		if ( !fitQuietly( *dfa ) )
		{
			pr.reason = quadratic
				? "QDFA is singular on this fold (a class covariance is not invertible)"
				: "LDFA is singular on this fold (the pooled covariance is not invertible)";
			return pr;
		}
		pr.ok = true;
		pr.pred = heldoutPredictions( *dfa );
		return pr;
	};
}

crossval::Procedure cvadapters::nestedObdProcedure( const obd::Config& cfg,
	double innerValFraction, vector< crossval::FoldSelection >* selections )
{
	return [ cfg, innerValFraction, selections ]( DataSet& foldData,
		const vector< unsigned >& trainRows,
		const vector< unsigned >& testRows,
		const atomic< bool >* cancel ) -> ProcResult
	{
		ProcResult pr;

		// Inner validation split of the fold's TRAINING rows ONLY, stratified on
		//    the outcome. This is where leak-freeness is won: the held-out testRows
		//    are never among the rows OBD's early stopping watches.
		Matrix< double >& raw = foldData.getRawMatrix();
		unsigned outCol = raw.cols() - 1;
		vector< unsigned > innerLabel( trainRows.size() );
		for ( unsigned i = 0; i < trainRows.size(); i++ )
			innerLabel[ i ] = ( raw( trainRows[ i ], outCol ) != 0 ) ? 1u : 0u;

		unsigned nInnerVal = ( unsigned ) ( innerValFraction * trainRows.size() + 0.5 );
		if ( nInnerVal < 1 ) nInnerVal = 1;
		if ( trainRows.size() >= 2 && nInnerVal > trainRows.size() - 1 )
			nInnerVal = trainRows.size() - 1; // always leave an inner training row

		nsplit::Holdout h = nsplit::stratifiedHoldout( innerLabel, nInnerVal );
		vector< unsigned > innerTrain, innerVal;
		innerTrain.reserve( h.train.size() );
		innerVal.reserve( h.test.size() );
		for ( unsigned i = 0; i < h.train.size(); i++ )
			innerTrain.push_back( trainRows[ h.train[ i ] ] );
		for ( unsigned i = 0; i < h.test.size(); i++ )
			innerVal.push_back( trainRows[ h.test[ i ] ] );

		// Materialize the fold three ways: train = innerTrain, validation =
		//    innerVal (what OBD's early stopping watches), test = testRows (the
		//    outer held-out, evaluated once). Every set scaled from innerTrain.
		foldData.makeFold( innerTrain, testRows, innerVal );

		// Keep the per-fold cost sane: no ROC bootstrap during the search or in
		//    the winner's final report (the runner computes pooled ROC itself). OBD
		//    saves and restores this count, so setting 0 now makes the winner cheap.
		foldData.getTrainTwoSet().setBootstrapResamples( 0 );
		foldData.getTestTwoSet().setBootstrapResamples( 0 );

		// Run the OBD architecture search on THIS fold, its whole report discarded;
		//    cancel stops it promptly.
		ostream& saved = util::screen();
		ostringstream discard;
		util::set_screen( discard );
		obd::Result r = obd::run( foldData, cfg, nullptr, cancel );
		util::set_screen( saved );

		if ( r.cancelled ) { pr.cancelled = true; return pr; }
		if ( !r.ok || !r.winner )
		{
			// A failed fold is reported as such -- NEVER a fabricated 0.5 (bug B2).
			pr.reason = r.message.empty()
				? "OBD produced no model for this fold" : r.message;
			return pr;
		}

		// What this fold chose is recorded ONLY for a fold that actually produced
		//    a model -- a failed fold contributes no metadata at all (no fake 0
		//    size, no fabricated optimizer). Architecture and optimizer are
		//    appended together as one record so they cannot desync.
		if ( selections )
		{
			crossval::FoldSelection s;
			s.hidden = r.selectedHidden;
			s.algorithm = r.algorithm;
			s.autoSelected = r.autoSelected;
			selections->push_back( s );
		}

		// OBD ran reportAccuracy on the winner over the outer held-out test set (in
		//    test-row order); read those predictions from the winner's own DataSet.
		pr.ok = true;
		pr.pred = heldoutPredictions( *r.winner );
		return pr;
	};
}
