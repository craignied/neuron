/* Cross-validation runner -- see crossval.h and rule 6. */

#include "crossval.h"

#include "twoset.h"
#include "netclone.h"
#include "utility.h"

#include <memory>
#include <sstream>

// Compute the ROC areas for a set of (outcome, prediction) pairs by loading
//    them into a TwoSet -- its column 0 is the true outcome, column 1 the guess.
//    Non-computable areas come back as -1 (a degenerate held-out set).
static void rocFromPairs( const vector< unsigned >& outcome,
	const vector< double >& pred, const vector< unsigned >& rows,
	double& az, double& trap )
{
	az = trap = -1;
	if ( rows.empty() ) return;

	Matrix< double > m( ( unsigned ) rows.size(), 2 );
	for ( unsigned i = 0; i < rows.size(); i++ )
	{
		m( i, 0 ) = ( double ) outcome[ rows[ i ] ];
		m( i, 1 ) = pred[ rows[ i ] ];
	}

	TwoSet ts;
	ts.setMatrix( m );
	ts.setThreshold( 0.5 );
	ts.setBootstrapResamples( 0 ); // point areas only

	try { trap = ts.getTrapROCarea(); } catch ( ... ) {}
	try { az = ts.getStatROCarea(); } catch ( ... ) {}
}

crossval::RunResult crossval::run( DataSet& data,
	const vector< unsigned >& foldId, Procedure proc )
{
	RunResult res;

	if ( !data.rawLoaded() )
	{
		res.message = "no raw dataset loaded";
		return res;
	}

	Matrix< double >& raw = data.getRawMatrix();
	unsigned n = raw.rows(), outCol = raw.cols() - 1;

	if ( foldId.size() != n )
	{
		res.message = "fold assignment size does not match the dataset";
		return res;
	}

	unsigned k = 0;
	for ( unsigned r = 0; r < n; r++ )
		if ( foldId[ r ] + 1 > k ) k = foldId[ r ] + 1;

	res.oofPrediction.assign( n, -1.0 );
	res.outcome.assign( n, 0 );
	for ( unsigned r = 0; r < n; r++ )
		res.outcome[ r ] = ( raw( r, outCol ) != 0 ) ? 1u : 0u;

	for ( unsigned f = 0; f < k; f++ )
	{
		vector< unsigned > trainRows, testRows;
		for ( unsigned r = 0; r < n; r++ )
			( foldId[ r ] == f ? testRows : trainRows ).push_back( r );
		if ( testRows.empty() ) continue;

		DataSet foldData = data; // a copy carrying Raw + the config
		foldData.makeFold( trainRows, testRows );

		vector< double > preds = proc( foldData ); // held-out preds, test order
		if ( preds.size() != testRows.size() )
		{
			res.message = "a procedure returned the wrong number of predictions";
			return res;
		}
		for ( unsigned i = 0; i < testRows.size(); i++ )
			res.oofPrediction[ testRows[ i ] ] = preds[ i ];

		FoldResult fr;
		fr.fold = f;
		fr.nHeldout = ( unsigned ) testRows.size();
		rocFromPairs( res.outcome, res.oofPrediction, testRows, fr.az, fr.trap );
		res.folds.push_back( fr );
	}

	// Pooled out-of-fold ROC: every row's out-of-fold prediction, together.
	vector< unsigned > allRows( n );
	for ( unsigned r = 0; r < n; r++ ) allRows[ r ] = r;
	rocFromPairs( res.outcome, res.oofPrediction, allRows, res.oofAz, res.oofTrap );

	res.ok = true;
	return res;
}

crossval::Procedure crossval::trainProcedure( const Network& templateNet,
	unsigned maxIter )
{
	return [ &templateNet, maxIter ]( DataSet& foldData ) -> vector< double >
	{
		unique_ptr< Network > clone = cloneNetwork( templateNet );
		clone->setDataSet( foldData );  // retarget to this fold (weights survive)
		clone->randomize();             // fresh weights for an honest fit
		clone->setMaxIterations( maxIter );
		clone->getDataSet().getTestTwoSet().setBootstrapResamples( 0 );

		// train() prints its report through util::screen(); discard it, then
		//    restore whatever stream was current.
		ostream& saved = util::screen();
		ostringstream discard;
		util::set_screen( discard );
		clone->train();                 // epilogue writes the held-out guesses
		util::set_screen( saved );

		TwoSet& te = clone->getDataSet().getTestTwoSet();
		unsigned m = te.getNumElements();
		vector< double > preds( m );
		for ( unsigned i = 0; i < m; i++ ) preds[ i ] = te.test( i );
		return preds;
	};
}
