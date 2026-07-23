/* Cross-validation runner -- see crossval.h and rule 6. */

#include "crossval.h"

#include <chrono>

#include "twoset.h"
#include "utility.h"

// Deterministic seed mixing (a MurmurHash3 finalizer) -- keys an RNG substream by
//    an index so ( seed, procedure, fold ) map to well-separated seeds. Used only
//    to isolate CV's per-fold fits; it does not change the engine RNG mechanism.
static unsigned mixSeed( unsigned base, unsigned k )
{
	unsigned h = base ^ ( k + 0x9E3779B9u + ( base << 6 ) + ( base >> 2 ) );
	h ^= h >> 16; h *= 0x85EBCA6Bu;
	h ^= h >> 13; h *= 0xC2B2AE35u;
	h ^= h >> 16;
	return h;
}

// FNV-1a over the procedure NAME -- a procedure's substream is keyed by its
//    stable identity, NOT its position in the comparison, so adding, removing, or
//    reordering other procedures never shifts its RNG (bug B11).
static unsigned hashName( const string& s )
{
	unsigned h = 2166136261u;
	for ( unsigned i = 0; i < s.size(); i++ )
	{
		h ^= ( unsigned char ) s[ i ];
		h *= 16777619u;
	}
	return h;
}

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
	const vector< unsigned >& foldId, Procedure proc, const atomic< bool >* cancel,
	bool substreams, unsigned seed )
{
	RunResult res;

	if ( !data.rawLoaded() )
	{
		res.message = "no raw dataset loaded";
		return res;
	}
	if ( !proc )
	{
		res.message = "no procedure supplied";
		return res;
	}

	Matrix< double >& raw = data.getRawMatrix();
	unsigned n = raw.rows(), outCol = raw.cols() - 1;

	if ( n < 2 )
	{
		res.message = "cross-validation needs at least two rows";
		return res;
	}
	if ( foldId.size() != n )
	{
		res.message = "fold assignment size does not match the dataset";
		return res;
	}

	// The fold ids must be contiguous 0..k-1 and there must be >= 2 folds --
	//    the class-layer API defends its own contract (the GUI plan is valid).
	unsigned k = 0;
	for ( unsigned r = 0; r < n; r++ )
		if ( foldId[ r ] + 1 > k ) k = foldId[ r ] + 1;
	if ( k < 2 )
	{
		res.message = "a fold plan needs at least two folds";
		return res;
	}
	vector< unsigned > foldCount( k, 0 );
	for ( unsigned r = 0; r < n; r++ ) foldCount[ foldId[ r ] ]++;
	for ( unsigned f = 0; f < k; f++ )
		if ( foldCount[ f ] == 0 )
		{
			res.message = "the fold plan has an empty fold (ids must be contiguous)";
			return res;
		}

	res.oofPrediction.assign( n, -1.0 );
	res.outcome.assign( n, 0 );
	for ( unsigned r = 0; r < n; r++ )
		res.outcome[ r ] = ( raw( r, outCol ) != 0 ) ? 1u : 0u;

	// Rows whose fold produced a real prediction -- only these enter the pooled
	//    ROC, so a failed fold is never silently averaged in.
	vector< unsigned > pooledRows;

	for ( unsigned f = 0; f < k; f++ )
	{
		if ( cancel && cancel->load() )
		{
			res.cancelled = true;
			res.message = "cancelled";
			return res;
		}

		vector< unsigned > trainRows, testRows;
		for ( unsigned r = 0; r < n; r++ )
			( foldId[ r ] == f ? testRows : trainRows ).push_back( r );
		if ( testRows.empty() || trainRows.empty() )
		{
			res.message = "a fold left the training or test set empty";
			return res;
		}

		DataSet foldData = data; // a copy carrying Raw + the config
		foldData.makeFold( trainRows, testRows ); // deterministic (fixed indices)

		// Start this fold's fit on its own deterministic RNG substream, so the
		//    result depends only on ( seed, fold ) -- not on earlier consumption.
		if ( substreams )
			util::set_seed( mixSeed( seed, f + 1 ) );

		// trainRows/testRows let a nested procedure re-split the training rows for
		//    its own inner validation set; cancel lets a long fit stop promptly.
		ProcResult pr = proc( foldData, trainRows, testRows, cancel );

		if ( pr.cancelled )
		{
			res.cancelled = true;
			res.message = "cancelled";
			return res;
		}

		FoldResult fr;
		fr.fold = f;
		fr.nHeldout = ( unsigned ) testRows.size();

		if ( !pr.ok )
		{
			// A failed fold is retained as MISSING and reported -- never a
			//    fabricated prediction (the reporting contract, ROADMAP 4).
			fr.ok = false;
			fr.reason = pr.reason.empty() ? "the procedure failed on this fold"
				: pr.reason;
			res.folds.push_back( fr );
			continue;
		}
		if ( pr.pred.size() != testRows.size() )
		{
			res.message = "a procedure returned the wrong number of predictions";
			return res;
		}

		for ( unsigned i = 0; i < testRows.size(); i++ )
		{
			res.oofPrediction[ testRows[ i ] ] = pr.pred[ i ];
			pooledRows.push_back( testRows[ i ] );
		}

		Metrics fm = metricsFor( res.outcome, res.oofPrediction, testRows );
		fr.ok = true;
		fr.az = fm.az; fr.trap = fm.trap;
		res.folds.push_back( fr );
		res.validFolds++;
	}

	// Pooled out-of-fold ROC over the rows that got a real prediction only.
	Metrics pooled = metricsFor( res.outcome, res.oofPrediction, pooledRows );
	res.oofAz = pooled.az; res.oofTrap = pooled.trap;

	res.ok = true;
	return res;
}

crossval::Comparison crossval::compare( DataSet& data,
	const vector< unsigned >& foldId, const vector< ProcedureSpec >& procs,
	const atomic< bool >* cancel, bool substreams, unsigned seed )
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
		//    (rule 6). It does not train, select, or know the model family. Each
		//    procedure gets its own substream base ( keyed by procedure index ), so
		//    its result never depends on which OTHER procedures are in the run.
		chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
		RunResult rr = run( data, foldId, procs[ i ].proc, cancel,
			substreams, substreams ? mixSeed( seed, hashName( procs[ i ].name ) ) : 0 );
		chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

		if ( rr.cancelled )
		{
			c.cancelled = true;
			c.message = "cancelled during '" + procs[ i ].name + "'";
			return c;
		}
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
