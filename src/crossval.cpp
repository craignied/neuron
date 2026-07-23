/* Cross-validation runner -- see crossval.h and rule 6. */

#include "crossval.h"

#include <chrono>

#include "twoset.h"

// Build the held-out metrics for a set of (outcome, prediction) pairs by loading
//    them into a TwoSet -- its column 0 is the true outcome, column 1 the guess.
//    Any metric not computable on a degenerate held-out set (one class, empty)
//    comes back as -1. This is the ONE place out-of-fold pairs become metrics
//    (rule 6): the runner scores each fold with it and the report reuses it.
crossval::Metrics crossval::metricsFor( const vector< unsigned >& outcome,
	const vector< double >& pred, const vector< unsigned >& rows )
{
	Metrics m;
	m.n = ( unsigned ) rows.size();
	if ( rows.empty() ) return m;

	Matrix< double > a( ( unsigned ) rows.size(), 2 );
	for ( unsigned i = 0; i < rows.size(); i++ )
	{
		a( i, 0 ) = ( double ) outcome[ rows[ i ] ];
		a( i, 1 ) = pred[ rows[ i ] ];
	}

	TwoSet ts;
	ts.setMatrix( a );
	ts.setThreshold( 0.5 );
	ts.setBootstrapResamples( 0 ); // point areas only

	try { m.trap = ts.getTrapROCarea(); } catch ( ... ) {}
	try { m.az = ts.getStatROCarea(); } catch ( ... ) {}
	try { m.sens = ts.getSens(); } catch ( ... ) {}
	try { m.spec = ts.getSpec(); } catch ( ... ) {}
	try { m.ca = ts.getClassAcc(); } catch ( ... ) {}
	return m;
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

		// held-out preds in test-row order; trainRows/testRows let a nested
		//    procedure re-split the training rows for its own inner validation set
		vector< double > preds = proc( foldData, trainRows, testRows );
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
		Metrics fm = metricsFor( res.outcome, res.oofPrediction, testRows );
		fr.az = fm.az; fr.trap = fm.trap;
		res.folds.push_back( fr );
	}

	// Pooled out-of-fold ROC: every row's out-of-fold prediction, together.
	vector< unsigned > allRows( n );
	for ( unsigned r = 0; r < n; r++ ) allRows[ r ] = r;
	Metrics pooled = metricsFor( res.outcome, res.oofPrediction, allRows );
	res.oofAz = pooled.az; res.oofTrap = pooled.trap;

	res.ok = true;
	return res;
}

crossval::Comparison crossval::compare( DataSet& data,
	const vector< unsigned >& foldId, const vector< ProcedureSpec >& procs )
{
	Comparison c;

	if ( procs.empty() )
	{
		c.message = "no procedures to compare";
		return c;
	}

	for ( unsigned i = 0; i < procs.size(); i++ )
	{
		// The coordinator owns coordination -- including timing each procedure
		//    (rule 6). It does not train, select, or know the model family.
		chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
		RunResult rr = run( data, foldId, procs[ i ].proc );
		chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

		if ( !rr.ok )
		{
			c.message = "procedure '" + procs[ i ].name + "': " + rr.message;
			return c;
		}
		if ( c.outcome.empty() ) c.outcome = rr.outcome; // same across all procs
		Comparison::Entry e;
		e.name = procs[ i ].name;
		e.result = rr;
		e.seconds = chrono::duration< double >( t1 - t0 ).count();
		if ( procs[ i ].archHidden ) // a procedure that carries architecture metadata
			e.archHidden = *procs[ i ].archHidden;
		c.entries.push_back( e );
	}

	c.foldId = foldId; // the shared plan, for per-fold report rows
	c.k = 0;
	for ( unsigned r = 0; r < foldId.size(); r++ )
		if ( foldId[ r ] + 1 > c.k ) c.k = foldId[ r ] + 1;

	c.ok = true;
	return c;
}
