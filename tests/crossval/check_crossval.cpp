// check_crossval.cpp : the generic CV runner (ROADMAP 4 Phase 4b-CV).
//
// crossval::run materializes each fold, calls a procedure to fit on the fold's
// training set and predict its held-out rows, and collects a per-exemplar
// out-of-fold prediction for every row plus per-fold ROC. This pins:
//   - every row is held out exactly once (so every row gets an OOF prediction);
//   - a real fit gives a pooled OOF AUC well above chance on a learnable
//     problem (watched to FAIL against a procedure that skips training);
//   - the run is reproducible under a fixed seed.

#include <iostream>
#include <string>
#include <vector>

#include "crossval.h"
#include "cvadapters.h"
#include "simpleprop.h"
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

	if ( failures == 0 )
	{
		cout << "check_crossval: the CV runner holds every row out once and fits honestly"
			<< endl;
		return 0;
	}
	cerr << "check_crossval: FAILED" << endl;
	return 1;
}
