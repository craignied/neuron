/* CV procedure adapters -- see cvadapters.h and rule 6. */

#include "cvadapters.h"

#include "netclone.h"
#include "twoset.h"
#include "ldfa.h"
#include "qdfa.h"
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
	return [ &templateNet, maxIter ]( DataSet& foldData ) -> vector< double >
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
	return [ quadratic ]( DataSet& foldData ) -> vector< double >
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
