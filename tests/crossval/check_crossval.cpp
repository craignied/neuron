// check_crossval.cpp : the generic CV runner (ROADMAP 4 Phase 4b-CV).
//
// crossval::run materializes each fold, calls a procedure to fit on the fold's
// training set and predict its held-out rows, and collects a per-exemplar
// out-of-fold prediction for every row plus per-fold ROC. This pins:
//   - every row is held out exactly once (so every row gets an OOF prediction);
//   - a real fit gives a pooled OOF AUC well above chance on a learnable
//     problem (watched to FAIL against a procedure that skips training);
//   - the run is reproducible under a fixed seed.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "crossval.h"
#include "cvadapters.h"
#include "cvreport.h"
#include "obd.h"
#include "simpleprop.h"
#include "logistic.h"
#include "dataset.h"
#include "split.h"
#include "utility.h"

using namespace std;

int failures = 0;

void expect( bool ok, const string& what )
{
	if ( ok ) cout << "ok - " << what << endl;
	else { cout << "FAIL - " << what << endl; failures++; }
}

// A learnable linearly-separable 2-input problem (as in check_obd).
static Matrix< double > learnable( unsigned n )
{
	Matrix< double > raw( n, 3 );
	for ( unsigned i = 0; i < n; i++ )
	{
		double x0 = -1.0 + 2.0 * ( ( i * 37 ) % 100 ) / 99.0;
		double x1 = -1.0 + 2.0 * ( ( i * 53 ) % 100 ) / 99.0;
		raw( i, 0 ) = x0; raw( i, 1 ) = x1;
		raw( i, 2 ) = ( x0 + x1 > 0 ) ? 1 : 0;
	}
	return raw;
}

int main()
{
	unsigned n = 250;
	Matrix< double > raw = learnable( n );

	DataSet data;
	data.setInput( 2 ); data.setOutput( 1 ); data.setDiscrete( true );
	data.setHistory( false );
	data.setRawMatrix( raw );

	// Stratified 5-fold plan.
	vector< unsigned > label( n );
	for ( unsigned r = 0; r < n; r++ ) label[ r ] = ( raw( r, 2 ) == 0 ) ? 0u : 1u;
	util::set_seed( 1 );
	vector< unsigned > foldId = nsplit::kFold( label, 5 );

	// A configured template SimpleProp (all folds clone this architecture/config).
	DataSet tdata = data;
	vector< unsigned > allRows( n ), none;
	for ( unsigned r = 0; r < n; r++ ) allRows[ r ] = r;
	tdata.makeFold( allRows, none ); // train = everything (just to size the net)

	SimpleProp tmpl;
	tmpl.setDataSet( tdata );
	tmpl.setHidden( 4 );
	tmpl.setHistory( false ); tmpl.setLastop( false ); tmpl.setLogPrint( false );
	util::set_seed( 2 );
	tmpl.randomize();

	util::set_seed( 7 );
	crossval::RunResult r = crossval::run( data, foldId,
		cvadapters::trainProcedure( tmpl, 400 ) );

	expect( r.ok, "the CV run completes" );
	expect( r.folds.size() == 5, "one result per fold" );

	bool allPredicted = true;
	for ( unsigned i = 0; i < n; i++ )
		if ( r.oofPrediction[ i ] < 0.0 ) allPredicted = false;
	expect( allPredicted, "every row receives an out-of-fold prediction" );

	expect( r.oofTrap > 0.6,
		"pooled out-of-fold AUC is well above chance on a learnable problem" );

	// Reproducibility under a fixed seed.
	util::set_seed( 7 );
	crossval::RunResult r2 = crossval::run( data, foldId,
		cvadapters::trainProcedure( tmpl, 400 ) );
	expect( r.oofPrediction == r2.oofPrediction,
		"the same seed reproduces the out-of-fold predictions" );

	// The DFA adapter (LDFA) over the SAME fold plan -- a different model family
	// through the same generic runner. DFA is deterministic (no seed), and its
	// held-out score is the graded discriminant probability.
	crossval::RunResult rl = crossval::run( data, foldId,
		cvadapters::dfaProcedure( false ) );
	bool ldfaAll = true;
	for ( unsigned i = 0; i < n; i++ )
		if ( rl.oofPrediction[ i ] < 0.0 ) ldfaAll = false;
	expect( rl.ok && ldfaAll && rl.oofTrap > 0.6,
		"the LDFA adapter cross-validates: every row predicted, pooled AUC beats chance" );

	// Logistic regression through the SAME generic runner + trainProcedure: a
	// Logistic IS a Network (cloneNetwork handles it) and carries its own config
	// (cross-entropy, batch, auto-step) through copy, so no bespoke adapter is
	// needed -- trainProcedure fits and CV-scores it like any other network.
	Logistic ltmpl;
	ltmpl.setDataSet( tdata );
	ltmpl.setHistory( false ); ltmpl.setLastop( false ); ltmpl.setLogPrint( false );
	util::set_seed( 3 );
	ltmpl.randomize();

	util::set_seed( 7 );
	crossval::RunResult rlog = crossval::run( data, foldId,
		cvadapters::trainProcedure( ltmpl, 400 ) );
	bool logAll = true;
	for ( unsigned i = 0; i < n; i++ )
		if ( rlog.oofPrediction[ i ] < 0.0 ) logAll = false;
	expect( rlog.ok && logAll && rlog.oofTrap > 0.6,
		"logistic regression cross-validates through trainProcedure (no bespoke adapter)" );

	// The comparison coordinator: two model families over ONE shared fold plan,
	// so every patient carries a prediction from BOTH -- the paired substrate.
	vector< crossval::ProcedureSpec > procs;
	procs.push_back( { "Neural", cvadapters::trainProcedure( tmpl, 400 ) } );
	procs.push_back( { "LDFA", cvadapters::dfaProcedure( false ) } );

	util::set_seed( 7 );
	crossval::Comparison cmp = crossval::compare( data, foldId, procs );

	expect( cmp.ok && cmp.entries.size() == 2 &&
		cmp.entries[ 0 ].name == "Neural" && cmp.entries[ 1 ].name == "LDFA",
		"the coordinator runs both procedures over the shared fold plan" );

	bool paired = ( cmp.entries[ 0 ].result.oofPrediction.size() == n &&
		cmp.entries[ 1 ].result.oofPrediction.size() == n );
	for ( unsigned i = 0; i < n; i++ )
		if ( cmp.entries[ 0 ].result.oofPrediction[ i ] < 0.0 ||
			cmp.entries[ 1 ].result.oofPrediction[ i ] < 0.0 ) paired = false;
	expect( paired && cmp.entries[ 0 ].result.oofTrap > 0.6 &&
		cmp.entries[ 1 ].result.oofTrap > 0.6,
		"every patient has a paired out-of-fold prediction from each procedure" );

	// The nested-OBD adapter: for each fold the ENTIRE architecture search runs on
	// an inner train/validation split of the fold's training rows, and the winner
	// is scored on the outer held-out rows. Leak-free by construction (OBD early-
	// stops on the inner validation set, never the held-out rows), and every row
	// still gets exactly one out-of-fold prediction.
	obd::Config ocfg;
	ocfg.hStart = 2; ocfg.hMax = 4; ocfg.iterBudget = 300;
	ocfg.sampleEvery = 20; ocfg.algorithm = 0;
	vector< unsigned > pickedHidden;

	util::set_seed( 7 );
	crossval::RunResult ro = crossval::run( data, foldId,
		cvadapters::nestedObdProcedure( ocfg, 0.25, &pickedHidden ) );

	expect( ro.ok && ro.folds.size() == 5, "the nested-OBD CV run completes over every fold" );

	bool obdAll = true;
	for ( unsigned i = 0; i < n; i++ )
		if ( ro.oofPrediction[ i ] < 0.0 ) obdAll = false;
	expect( obdAll,
		"nested OBD gives every row an out-of-fold prediction (held out exactly once)" );

	expect( ro.oofTrap > 0.6,
		"nested-OBD pooled out-of-fold AUC beats chance (the search fits honestly)" );

	bool sizesSane = ( pickedHidden.size() == 5 );
	for ( unsigned i = 0; i < pickedHidden.size(); i++ )
		if ( pickedHidden[ i ] < 1 || pickedHidden[ i ] > ocfg.hMax ) sizesSane = false;
	expect( sizesSane,
		"each fold reports a selected hidden-unit count within the search range" );

	util::set_seed( 7 );
	vector< unsigned > pickedHidden2;
	crossval::RunResult ro2 = crossval::run( data, foldId,
		cvadapters::nestedObdProcedure( ocfg, 0.25, &pickedHidden2 ) );
	expect( ro.oofPrediction == ro2.oofPrediction && pickedHidden == pickedHidden2,
		"the same seed reproduces the nested-OBD out-of-fold predictions and sizes" );

	// B2: a fold OBD cannot fit is recorded as FAILED, never fabricated. hidden_max
	// below hStart (2) makes OBD refuse every fold; the runner must mark them failed
	// (no prediction, no fake 0.5), leave those rows absent, and pool nothing.
	obd::Config badCfg; badCfg.hStart = 2; badCfg.hMax = 1; // empty range
	vector< unsigned > badArch;
	crossval::RunResult rf = crossval::run( data, foldId,
		cvadapters::nestedObdProcedure( badCfg, 0.25, &badArch ) );
	bool allFoldsFailed = ( rf.ok && rf.validFolds == 0 && rf.folds.size() == 5 );
	for ( unsigned i = 0; i < rf.folds.size(); i++ )
		if ( rf.folds[ i ].ok ) allFoldsFailed = false;
	bool noFabrication = true;
	for ( unsigned i = 0; i < n; i++ )
		if ( rf.oofPrediction[ i ] != -1.0 ) noFabrication = false; // -1 = absent
	expect( allFoldsFailed && noFabrication && badArch.empty() && rf.oofTrap < 0,
		"a fold OBD cannot fit is failed, not fabricated (no 0.5, no fake size, absent from the pool)" );

	// B3: a singular DFA fold is recorded as failed, not read from unwritten guess
	// storage. Perfectly collinear inputs ( x1 = 2*x0 ) give a rank-deficient,
	// non-invertible covariance, so every LDFA/QDFA fold is singular.
	Matrix< double > craw( 120, 3 );
	for ( unsigned i = 0; i < 120; i++ )
	{
		double x = -1.0 + 2.0 * ( ( i * 41 ) % 100 ) / 99.0;
		craw( i, 0 ) = x;
		craw( i, 1 ) = 2.0 * x; // collinear with input 0 -> singular covariance
		craw( i, 2 ) = ( i % 2 ) ? 1 : 0;
	}
	DataSet cdata; cdata.setInput( 2 ); cdata.setOutput( 1 ); cdata.setDiscrete( true );
	cdata.setHistory( false ); cdata.setRawMatrix( craw );
	vector< unsigned > clabel( 120 );
	for ( unsigned r = 0; r < 120; r++ ) clabel[ r ] = ( craw( r, 2 ) == 0 ) ? 0u : 1u;
	util::set_seed( 5 );
	vector< unsigned > cfold = nsplit::kFold( clabel, 4 );
	crossval::RunResult rq = crossval::run( cdata, cfold, cvadapters::dfaProcedure( true ) );
	bool dfaFailed = ( rq.ok && rq.validFolds == 0 );
	for ( unsigned r = 0; r < 120; r++ )
		if ( rq.oofPrediction[ r ] != -1.0 ) dfaFailed = false;
	bool dfaReasoned = !rq.folds.empty() && !rq.folds[ 0 ].ok
		&& rq.folds[ 0 ].reason.find( "singular" ) != string::npos;
	expect( dfaFailed && dfaReasoned,
		"a singular DFA fold is failed with a reason, not read from unwritten storage" );

	// B4: a column CONSTANT within the TRAINING fold must normalize to a finite
	// value, not inf/NaN from a zero-range (0/0) division. Input 0 is constant in
	// the training rows here (all 5) but differs in the held-out rows (9).
	Matrix< double > kraw( 20, 2 );
	for ( unsigned i = 0; i < 20; i++ )
	{
		kraw( i, 0 ) = ( i < 15 ) ? 5.0 : 9.0; // constant in train, differs in test
		kraw( i, 1 ) = ( i % 2 ) ? 1 : 0;
	}
	DataSet kdata; kdata.setInput( 1 ); kdata.setOutput( 1 ); kdata.setDiscrete( true );
	kdata.setHistory( false ); kdata.setRawMatrix( kraw );
	vector< unsigned > kTrain, kTest;
	for ( unsigned i = 0; i < 15; i++ ) kTrain.push_back( i );
	for ( unsigned i = 15; i < 20; i++ ) kTest.push_back( i );
	kdata.makeFold( kTrain, kTest );
	bool allFinite = true;
	Matrix< double >& ktr = kdata.getTrainMatrix();
	Matrix< double >& kte = kdata.getTestMatrix();
	for ( unsigned r = 0; r < ktr.rows(); r++ ) if ( !isfinite( ktr( r, 0 ) ) ) allFinite = false;
	for ( unsigned r = 0; r < kte.rows(); r++ ) if ( !isfinite( kte( r, 0 ) ) ) allFinite = false;
	expect( allFinite,
		"a training-fold-constant input normalizes to a finite value (no 0/0 division)" );

	// B10: the class-layer runner defends its fold-plan contract.
	crossval::RunResult re1 = crossval::run( data, foldId, crossval::Procedure() );
	expect( !re1.ok && re1.oofPrediction.empty(), "run refuses an empty procedure" );
	vector< unsigned > oneFold( n, 0 ); // all rows in fold 0 -> only one fold
	crossval::RunResult re2 = crossval::run( data, oneFold,
		cvadapters::trainProcedure( tmpl, 50 ) );
	expect( !re2.ok, "run refuses a single-fold plan" );
	vector< unsigned > gap( n, 0 ); gap[ 0 ] = 2; // ids {0,2}, fold 1 empty
	crossval::RunResult re3 = crossval::run( data, gap,
		cvadapters::trainProcedure( tmpl, 50 ) );
	expect( !re3.ok, "run refuses a non-contiguous fold plan (an empty fold)" );

	// B11: with deterministic substreams, a procedure's CV predictions depend only
	// on its own (name, fold) key -- NOT on which other procedures are compared or
	// in what order. Neural's predictions must be identical whether it runs alone,
	// or after Logistic, or after Logistic+LDFA in a different order.
	auto neuralOf = []( DataSet& d, const vector< unsigned >& fid, SimpleProp& t,
		vector< crossval::ProcedureSpec > procs ) -> vector< double >
	{
		crossval::Comparison cm = crossval::compare( d, fid, procs, nullptr,
			true /*substreams*/, 99 );
		for ( unsigned i = 0; i < cm.entries.size(); i++ )
			if ( cm.entries[ i ].name == "Neural" ) return cm.entries[ i ].result.oofPrediction;
		return vector< double >();
	};
	// The OTHER procedures are stochastic networks (they consume RNG), so without
	// substreams they would shift Neural's stream -- which is exactly what the
	// name-keyed substreams must prevent.
	vector< crossval::ProcedureSpec > justNeural = {
		{ "Neural", cvadapters::trainProcedure( tmpl, 300 ), nullptr } };
	vector< crossval::ProcedureSpec > logThenNeural = {
		{ "OtherA", cvadapters::trainProcedure( tmpl, 300 ), nullptr },
		{ "Neural", cvadapters::trainProcedure( tmpl, 300 ), nullptr } };
	vector< crossval::ProcedureSpec > threeReordered = {
		{ "OtherB", cvadapters::trainProcedure( tmpl, 300 ), nullptr },
		{ "Neural", cvadapters::trainProcedure( tmpl, 300 ), nullptr },
		{ "OtherA", cvadapters::trainProcedure( tmpl, 300 ), nullptr } };
	vector< double > nAlone = neuralOf( data, foldId, tmpl, justNeural );
	vector< double > nAfter = neuralOf( data, foldId, tmpl, logThenNeural );
	vector< double > nReord = neuralOf( data, foldId, tmpl, threeReordered );
	expect( !nAlone.empty() && nAlone == nAfter && nAlone == nReord,
		"a procedure's CV predictions are invariant to other procedures' presence and order" );

	// The three-tier report (docs/evaluation_report_spec.md). Build a Comparison
	// of three procedures over the shared plan -- LDFA, plain neural, nested OBD
	// (with its architecture-metadata sink wired) -- and render it.
	vector< unsigned > obdArch;
	vector< crossval::ProcedureSpec > rprocs;
	rprocs.push_back( { "LDFA", cvadapters::dfaProcedure( false ), nullptr } );
	rprocs.push_back( { "Neural", cvadapters::trainProcedure( tmpl, 400 ), nullptr } );
	rprocs.push_back( { "Neural (OBD)",
		cvadapters::nestedObdProcedure( ocfg, 0.25, &obdArch ), &obdArch } );

	util::set_seed( 7 );
	crossval::Comparison rc = crossval::compare( data, foldId, rprocs );
	expect( rc.ok && rc.entries.size() == 3 && rc.k == 5,
		"the coordinator carries the fold plan and every procedure's results" );

	cvreport::PlanInfo info;
	info.n = n; info.foldPlan = "outcome-stratified, seed 7";
	unsigned events = 0;
	for ( unsigned r = 0; r < n; r++ ) if ( rc.outcome[ r ] ) events++;
	info.events = events;
	info.primary = "Neural (OBD)"; info.reference = "LDFA";

	string t1 = cvreport::tier1( rc, info );
	string t2 = cvreport::tier2( rc, info );

	// Tier 1 is the headline: one row per procedure, the CV caveat, and -- because
	// the nested-OBD entry carries architecture metadata -- an Arch footnote.
	expect( t1.find( "SUMMARY" ) != string::npos &&
		t1.find( "AUC (CV)" ) != string::npos &&
		t1.find( "LDFA" ) != string::npos &&
		t1.find( "Neural (OBD)" ) != string::npos,
		"Tier 1 names every procedure under a headline table" );
	expect( t1.find( "descriptive spread across dependent folds" ) != string::npos,
		"Tier 1 always carries the standing CV caveat" );
	expect( t1.find( "OBD selected" ) != string::npos,
		"Tier 1 footnotes the OBD architecture selection" );

	// The headline AUC must be the mean of the entry's per-fold exact AUCs -- the
	// number a reader trusts. Compute it independently and require it printed.
	double s = 0; unsigned v = 0;
	const crossval::RunResult& lr = rc.entries[ 0 ].result; // LDFA
	for ( unsigned i = 0; i < lr.folds.size(); i++ )
		if ( lr.folds[ i ].trap >= 0 ) { s += lr.folds[ i ].trap; v++; }
	char want[ 32 ];
	snprintf( want, sizeof want, "%.3f", v ? s / v : -1.0 );
	expect( v > 0 && t1.find( string( want ) ) != string::npos,
		"Tier 1's AUC(CV) equals the mean of the procedure's per-fold exact AUCs" );

	expect( t2.find( "Cross-validation detail" ) != string::npos &&
		t2.find( "pooled OOF AUC" ) != string::npos &&
		t2.find( "OBD architecture selection" ) != string::npos,
		"Tier 2 gives per-fold detail and the OBD selection frequency" );

	// Tier 3: three machine-readable files, predictions one row per exemplar.
	string dir = "/tmp/cvreport_check";
	system( ( "mkdir -p " + dir ).c_str() );
	vector< string > files = cvreport::writeArtifacts( rc, info, dir );
	expect( files.size() == 3, "Tier 3 writes three machine-readable files" );

	unsigned predLines = 0;
	{
		ifstream pf( ( dir + "/cv_predictions.csv" ).c_str() );
		string line;
		while ( getline( pf, line ) ) predLines++;
	}
	expect( predLines == n + 1,
		"cv_predictions.csv has a header plus one row per exemplar" );

	if ( failures == 0 )
	{
		cout << "check_crossval: the CV runner holds every row out once and fits honestly"
			<< endl;
		return 0;
	}
	cerr << "check_crossval: FAILED" << endl;
	return 1;
}
