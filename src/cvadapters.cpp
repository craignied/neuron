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

#include <algorithm>
#include <atomic>
#include <map>
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

cvadapters::InnerSplit cvadapters::innerValidationSplit(
	const vector< unsigned >& trainRows, const vector< unsigned >& rowLabel,
	const vector< unsigned >& rowGroup, double fraction )
{
	InnerSplit out;
	unsigned nTrain = ( unsigned ) trainRows.size();
	if ( nTrain < 2 )
	{
		out.reason = "too few training rows in this fold for an inner validation split";
		return out;
	}

	vector< unsigned > label( nTrain );
	for ( unsigned i = 0; i < nTrain; i++ ) label[ i ] = rowLabel[ trainRows[ i ] ];

	unsigned nVal = ( unsigned ) ( fraction * nTrain + 0.5 );
	if ( nVal < 1 ) nVal = 1;
	if ( nVal > nTrain - 1 ) nVal = nTrain - 1; // always leave an inner training row

	vector< unsigned > pickTrain, pickVal; // positions within trainRows
	if ( rowGroup.empty() )
	{
		nsplit::Holdout h = nsplit::stratifiedHoldout( label, nVal );
		pickTrain = h.train; pickVal = h.test;
	}
	else
	{
		// Group-disjoint inner split. groupHoldout indexes by dense id, so the
		//    fold's ids are densified here -- in order of FIRST APPEARANCE among
		//    the training rows, never by ascending value. groupHoldout visits
		//    groups in a seeded permutation of the dense id list, so densifying by
		//    value would make the split depend on how the caller happened to NUMBER
		//    its clusters: relabel the same clustering and a different set of rows
		//    becomes the inner validation set. First appearance is a property of
		//    the data, so the same clustering always gives the same split.
		vector< unsigned > gid( nTrain );
		{
			map< unsigned, unsigned > dense;
			for ( unsigned i = 0; i < nTrain; i++ )
			{
				unsigned key = rowGroup[ trainRows[ i ] ];
				map< unsigned, unsigned >::iterator it = dense.find( key );
				if ( it == dense.end() )
				{
					unsigned id = ( unsigned ) dense.size();
					dense[ key ] = id;
					gid[ i ] = id;
				}
				else gid[ i ] = it->second;
			}
			if ( dense.size() < 2 )
			{
				out.reason = "this fold's training rows come from a single group, so no "
					"group-disjoint inner validation set exists";
				return out;
			}
		}

		nsplit::GroupHoldout h = nsplit::groupHoldout( label, gid, nVal );
		pickTrain = h.train; pickVal = h.test;
	}

	if ( pickTrain.empty() || pickVal.empty() )
	{
		out.reason = rowGroup.empty()
			? "the inner validation fraction leaves one side of the split empty"
			: "whole groups cannot be divided into a usable inner training and "
				"validation set at this fraction";
		return out;
	}

	out.train.reserve( pickTrain.size() );
	out.validation.reserve( pickVal.size() );
	for ( unsigned i = 0; i < pickTrain.size(); i++ )
		out.train.push_back( trainRows[ pickTrain[ i ] ] );
	for ( unsigned i = 0; i < pickVal.size(); i++ )
		out.validation.push_back( trainRows[ pickVal[ i ] ] );
	out.ok = true;
	return out;
}

crossval::Procedure cvadapters::nestedObdProcedure( const obd::Config& cfg,
	double innerValFraction, vector< crossval::FoldSelection >* selections,
	obd::ProgressFn progress, const vector< unsigned >& rowGroup )
{
	return [ cfg, innerValFraction, selections, progress, rowGroup ]( DataSet& foldData,
		const vector< unsigned >& trainRows,
		const vector< unsigned >& testRows,
		const atomic< bool >* cancel ) -> ProcResult
	{
		ProcResult pr;

		// Inner validation split of the fold's TRAINING rows ONLY. This is where
		//    leak-freeness is won: the held-out testRows are never among the rows
		//    OBD's early stopping watches. When the run is grouped the inner split
		//    is group-disjoint as well, so the architecture is not chosen on rows
		//    whose clusters are also in the inner training set.
		Matrix< double >& raw = foldData.getRawMatrix();
		unsigned outCol = raw.cols() - 1;
		vector< unsigned > rowLabel( raw.rows() );
		for ( unsigned r = 0; r < raw.rows(); r++ )
			rowLabel[ r ] = ( raw( r, outCol ) != 0 ) ? 1u : 0u;

		InnerSplit split = innerValidationSplit( trainRows, rowLabel, rowGroup,
			innerValFraction );
		if ( !split.ok )
		{
			// A fold whose inner partition is infeasible FAILS with its reason --
			//    it never silently falls back to a row-wise inner split, which
			//    would answer a different question than the one being asked.
			pr.reason = split.reason;
			return pr;
		}
		const vector< unsigned >& innerTrain = split.train;
		const vector< unsigned >& innerVal = split.validation;

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
		//    cancel stops it promptly. progress (when the caller wired one) carries
		//    the phase and hidden-node trial out of the search -- the only view a
		//    long nested run has of what it is doing inside the current fold.
		ostream& saved = util::screen();
		ostringstream discard;
		util::set_screen( discard );
		obd::Result r = obd::run( foldData, cfg, progress, cancel );
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
