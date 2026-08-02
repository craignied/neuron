/* Cross-validation runner -- see crossval.h and rule 6. */

#include "crossval.h"

#include <chrono>
#include <functional>
#include <set>

#include "split.h"
#include "twoset.h"
#include "utility.h"

// Defend the row-partition + procedure contract for a locked-test evaluation
//    (DLG-5). evaluateOnce is a public class-layer API meant to accept later group
//    partitions, so it validates rather than trusts its caller. The row-partition
//    invariants (in range, no duplicate, disjoint, and FULL coverage -- dev + locked
//    must partition the whole raw dataset, so no row silently vanishes) are the
//    shared nsplit::partitionError; here we add the procedure-identity checks: every
//    procedure callable with a unique, nonempty name (the name is the stable
//    RNG-substream and report identity). Returns "" when ok, else the reason.
static string validateOnceInputs( unsigned nRows,
	const vector< unsigned >& trainRows, const vector< unsigned >& testRows,
	const vector< crossval::ProcedureSpec >& procs )
{
	set< string > names;
	for ( unsigned i = 0; i < procs.size(); i++ )
	{
		if ( !procs[ i ].proc ) return "a procedure has no callable function";
		if ( procs[ i ].name.empty() ) return "a procedure has an empty name";
		if ( !names.insert( procs[ i ].name ).second )
			return "duplicate procedure name '" + procs[ i ].name + "'";
	}
	// A locked-test split must partition the ENTIRE dataset (require coverage).
	return nsplit::partitionError( nRows, trainRows, testRows, true );
}

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
// Runs one optional metric calculation, keeping an unavailable metric absent
//    while letting a Matrix CONTRACT failure through. See the note at the call
//    site: the difference is between "this fold has no estimable AUC", which is
//    a result, and "the arithmetic was invalid", which is a defect.
static void metricFor( const function< void() >& calculate )
{
	try { calculate(); }
	catch ( const Matrix< double >::BoundsViolation& ) { throw; }
	catch ( const Matrix< double >::DimensionMismatch& ) { throw; }
	catch ( const Matrix< double >::BadSize& ) { throw; }
	catch ( ... ) { } // an unavailable metric stays absent
}

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

	// Each metric is optional -- a degenerate fold has no estimable AUC, and a
	//    TwoSet with no positives has no sensitivity. Those are ANSWERS and the
	//    catch-all below records them as absent.
	//
	//    A Matrix contract failure is not one of them (D9). It says a caller
	//    handed the numerical layer shapes or indices that cannot be right, and
	//    swallowing it here would report "this fold's AUC was not estimable"
	//    when the truth is that the arithmetic was invalid. metricFor rethrows
	//    those three types so they reach the worker or CLI boundary with their
	//    message; Singular is deliberately NOT among them, because a singular
	//    fit legitimately means an unavailable metric.
	metricFor( [ & ] { m.trap = ts.getTrapROCarea(); } );
	metricFor( [ & ] { m.az = ts.getStatROCarea(); } );
	metricFor( [ & ] { m.sens = ts.getSens(); } );
	metricFor( [ & ] { m.spec = ts.getSpec(); } );
	metricFor( [ & ] { m.ca = ts.getClassAcc(); } );
	return m;
}

crossval::RunResult crossval::run( DataSet& data,
	const vector< unsigned >& foldId, Procedure proc, const atomic< bool >* cancel,
	bool substreams, unsigned seed, ProgressFn progress )
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

	// Fold-level progress. The runner reports only what repetition knows -- which
	//    fold, of how many, and how many are behind it. Announced BEFORE the fit so
	//    a display names the fold that is currently costing the time, not the one
	//    that just finished.
	Progress prog;
	prog.stage = Progress::CROSS_VALIDATION;
	prog.k = k;

	for ( unsigned f = 0; f < k; f++ )
	{
		if ( progress )
		{
			prog.fold = f + 1;
			prog.completedFolds = f;
			progress( prog );
		}

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

	// Every fold is behind us: report the finished grid before the pooled scoring,
	//    so a caller's "completed folds" reaches k rather than stopping at k-1.
	if ( progress )
	{
		prog.fold = k;
		prog.completedFolds = k;
		progress( prog );
	}

	// Pooled out-of-fold ROC over the rows that got a real prediction only.
	Metrics pooled = metricsFor( res.outcome, res.oofPrediction, pooledRows );
	res.oofAz = pooled.az; res.oofTrap = pooled.trap;
	res.pooledN = ( unsigned ) pooledRows.size(); // the honest pooled denominator

	res.ok = true;
	return res;
}

crossval::LockedResult crossval::evaluateOnce( DataSet& data,
	const vector< unsigned >& trainRows, const vector< unsigned >& testRows,
	const vector< ProcedureSpec >& procs, const atomic< bool >* cancel,
	bool substreams, unsigned seed, ProgressFn progress )
{
	LockedResult lr;

	if ( !data.rawLoaded() ) { lr.message = "no raw dataset loaded"; return lr; }
	if ( procs.empty() ) { lr.message = "no procedures to evaluate"; return lr; }
	if ( trainRows.empty() || testRows.empty() )
	{
		lr.message = "the locked-test split needs nonempty training and test sets";
		return lr;
	}

	Matrix< double >& raw = data.getRawMatrix();
	unsigned outCol = raw.cols() - 1;

	// Defend the partition + procedure contract BEFORE indexing raw or invoking a
	//    procedure (DLG-5): in range, disjoint, no dupes, callable + unique names.
	string bad = validateOnceInputs( raw.rows(), trainRows, testRows, procs );
	if ( !bad.empty() ) { lr.message = bad; return lr; }

	// Row identity + the paired true outcomes (the audit substrate).
	lr.testRows = testRows;
	lr.outcome.resize( testRows.size() );
	for ( unsigned i = 0; i < testRows.size(); i++ )
		lr.outcome[ i ] = ( raw( testRows[ i ], outCol ) != 0 ) ? 1u : 0u;

	// The locked evaluation folds NOTHING -- it refits each procedure once on the
	//    development rows and scores it once. fold and k stay 0 so a display cannot
	//    invent a fold number for a pass that has none.
	Progress prog;
	prog.stage = Progress::LOCKED_EVALUATION;
	prog.procCount = ( unsigned ) procs.size();

	for ( unsigned p = 0; p < procs.size(); p++ )
	{
		if ( progress )
		{
			prog.procedure = procs[ p ].name;
			prog.procIndex = p + 1;
			prog.completedProcedures = p;
			progress( prog );
		}

		if ( cancel && cancel->load() )
		{
			lr.cancelled = true; lr.message = "cancelled"; return lr;
		}

		DataSet foldData = data;                 // Raw + config
		foldData.makeFold( trainRows, testRows ); // deterministic (fixed indices)

		// Each procedure fits on its own name-keyed substream, so its locked-test
		//    result is invariant to which OTHER procedures are evaluated and in what
		//    order (bug B11) -- membership/order is a presentation choice.
		if ( substreams )
			util::set_seed( mixSeed( seed, hashName( procs[ p ].name ) ) );

		chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
		ProcResult pr = procs[ p ].proc( foldData, trainRows, testRows, cancel );
		chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

		if ( pr.cancelled )
		{
			lr.cancelled = true;
			lr.message = "cancelled during '" + procs[ p ].name + "'";
			return lr;
		}

		LockedEntry e;
		e.name = procs[ p ].name;
		e.seconds = chrono::duration< double >( t1 - t0 ).count();
		if ( !pr.ok )
		{
			// A failed procedure is retained as MISSING and reported, never a
			//    fabricated prediction -- the CV contract, applied to the locked test.
			e.ok = false;
			e.reason = pr.reason.empty() ? "the procedure failed on the locked test"
				: pr.reason;
		}
		else if ( pr.pred.size() != testRows.size() )
		{
			lr.message = "a procedure returned the wrong number of "
				"locked-test predictions";
			return lr;
		}
		else
		{
			e.ok = true;
			e.pred = pr.pred;
			if ( procs[ p ].selections ) e.selections = *procs[ p ].selections;
		}
		lr.entries.push_back( e );
	}

	// Every procedure refit and scored.
	if ( progress )
	{
		prog.completedProcedures = ( unsigned ) procs.size();
		progress( prog );
	}

	lr.ok = true;
	return lr;
}

crossval::Comparison crossval::compare( DataSet& data,
	const vector< unsigned >& foldId, const vector< ProcedureSpec >& procs,
	const atomic< bool >* cancel, bool substreams, unsigned seed,
	ProgressFn progress )
{
	Comparison c;

	if ( procs.empty() )
	{
		c.message = "no procedures to compare";
		return c;
	}

	for ( unsigned i = 0; i < procs.size(); i++ )
	{
		// Stamp the procedure identity onto whatever the runner reports about
		//    folds. Each layer fills only its own fields (rule 6): the runner does
		//    not know what a procedure is, and the coordinator does not know which
		//    fold is running until the runner says so.
		ProgressFn relay;
		if ( progress )
		{
			string name = procs[ i ].name;
			unsigned index = i + 1, count = ( unsigned ) procs.size();
			relay = [ progress, name, index, count ]( const Progress& p )
			{
				Progress q = p;
				q.procedure = name;
				q.procIndex = index;
				q.procCount = count;
				q.completedProcedures = index - 1;
				progress( q );
			};
		}

		// The coordinator owns coordination -- including timing each procedure
		//    (rule 6). It does not train, select, or know the model family. Each
		//    procedure gets its own substream base ( keyed by procedure index ), so
		//    its result never depends on which OTHER procedures are in the run.
		chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
		RunResult rr = run( data, foldId, procs[ i ].proc, cancel,
			substreams, substreams ? mixSeed( seed, hashName( procs[ i ].name ) ) : 0,
			relay );
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
		if ( procs[ i ].selections ) // a procedure that carries architecture metadata
			e.selections = *procs[ i ].selections;
		c.entries.push_back( e );
	}

	c.foldId = foldId; // the shared plan, for per-fold report rows
	c.k = 0;
	for ( unsigned r = 0; r < foldId.size(); r++ )
		if ( foldId[ r ] + 1 > c.k ) c.k = foldId[ r ] + 1;

	// The whole grid is behind us. Reported so a caller's completed-procedure
	//    count reaches the total rather than stopping one short -- the relay above
	//    can only ever say "index - 1 finished", because from inside a procedure
	//    that is all that is true.
	//
	//    This event is about the COMPARISON, not about any fold or procedure, so
	//    it names none: fold, completedFolds and procedure stay empty. Carrying a
	//    fold number here would hand a display a fold with no procedure to
	//    attribute it to.
	if ( progress )
	{
		Progress done;
		done.stage = Progress::CROSS_VALIDATION;
		done.procCount = ( unsigned ) procs.size();
		done.completedProcedures = ( unsigned ) procs.size();
		done.k = c.k;
		progress( done );
	}

	c.ok = true;
	return c;
}
