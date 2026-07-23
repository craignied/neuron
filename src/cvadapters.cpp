/* CV procedure adapters -- see cvadapters.h and rule 6. */

#include "cvadapters.h"

#include "netclone.h"
#include "twoset.h"
#include "ldfa.h"
#include "qdfa.h"
#include "obd.h"
#include "split.h"
#include "utility.h"

#include <memory>
#include <sstream>

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

// Run a model's fit with its report discarded, restoring the caller's stream.
static void fitQuietly( Model& m )
{
	m.getDataSet().getTestTwoSet().setBootstrapResamples( 0 ); // point predictions
	ostream& saved = util::screen();
	ostringstream discard;
	util::set_screen( discard );
	m.train(); // epilogue writes the held-out guesses
	util::set_screen( saved );
}

crossval::Procedure cvadapters::trainProcedure( const Network& templateNet,
	unsigned maxIter )
{
	return [ &templateNet, maxIter ]( DataSet& foldData,
		const vector< unsigned >&, const vector< unsigned >& ) -> vector< double >
	{
		unique_ptr< Network > clone = cloneNetwork( templateNet );
		clone->setDataSet( foldData ); // retarget to this fold (weights survive)
		clone->randomize();            // fresh weights for an honest fit
		clone->setMaxIterations( maxIter );
		fitQuietly( *clone );
		return heldoutPredictions( *clone );
	};
}

crossval::Procedure cvadapters::dfaProcedure( bool quadratic )
{
	return [ quadratic ]( DataSet& foldData,
		const vector< unsigned >&, const vector< unsigned >& ) -> vector< double >
	{
		unique_ptr< Model > dfa = quadratic
			? unique_ptr< Model >( make_unique< QDFA >() )
			: unique_ptr< Model >( make_unique< LDFA >() );
		dfa->setDataSet( foldData );
		dfa->setHistory( false );
		dfa->setLastop( false );
		fitQuietly( *dfa ); // DFA::train computes the discriminant + writes guesses
		return heldoutPredictions( *dfa );
	};
}

crossval::Procedure cvadapters::nestedObdProcedure( const obd::Config& cfg,
	double innerValFraction, vector< unsigned >* selectedHidden )
{
	return [ cfg, innerValFraction, selectedHidden ]( DataSet& foldData,
		const vector< unsigned >& trainRows,
		const vector< unsigned >& testRows ) -> vector< double >
	{
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

		// Run the OBD architecture search on THIS fold, its whole report discarded.
		ostream& saved = util::screen();
		ostringstream discard;
		util::set_screen( discard );
		obd::Result r = obd::run( foldData, cfg, nullptr, nullptr );
		util::set_screen( saved );

		if ( selectedHidden )
			selectedHidden->push_back( r.ok ? r.selectedHidden : 0u );

		if ( !r.ok || !r.winner ) // a degenerate fold: a neutral 0.5 per held-out row
			return vector< double >( testRows.size(), 0.5 );

		// OBD ran reportAccuracy on the winner over the outer held-out test set (in
		//    test-row order); read those predictions from the winner's own DataSet.
		return heldoutPredictions( *r.winner );
	};
}
