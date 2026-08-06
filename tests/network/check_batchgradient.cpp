// check_batchgradient.cpp : CHARACTERIZATION of the batch objective and the
// RAW batch gradient, for every neural model that has one.
//
// WHY THIS FILE EXISTS. Phase 3 of the optimizer program extracts the batch
// separate-gradient pass out of OneHiddenNet::innerTrainSet() and
// BackProp::innerTrainSet() into one authoritative evaluation that L-BFGS can
// also call at a trial point. That extraction moves the two quantities every
// optimizer in the engine reads:
//
//     f(w) -- the mean training-set error PLUS the weight-decay penalty, which
//             is what innerTrainSet() returns;
//     g(w) -- the RAW mean gradient of that same f, which is what the gradient
//             stopping rule reads and what CGD/Shanno transform.
//
// Nothing pinned them TOGETHER before. tests/props pins returned errors,
// tests/backprop/check_bpoptimizer pins which structure the update consumes,
// and tests/iterative/check_decay pins that decay is applied once per update.
// None of them asserts that g is the gradient OF f. An extraction could
// therefore move both consistently-wrongly -- drop the decay term from the
// gradient and from the objective, say -- and every existing test would still
// pass. This file closes that.
//
// IT PASSES ON THE OLD ENGINE, and must keep passing after the extraction. It
// uses no method the extraction introduces: the objective is trainSet()'s
// return value, and the raw gradient is the model's own gradient structure
// after one separate-gradient pass (trainingType 0 with gradient stopping
// armed, where engine() dispatches through a switch with no case 0 and so
// leaves the gradient RAW).
//
// WHAT IS ASSERTED
//
//   1. SimpleProp and BareProp: every component of g agrees with a CENTRAL
//      finite difference of f, over LMS and cross-entropy, decay on and off.
//      Both architectures, so the bias column and the pinned bias slot are
//      covered. This is the strong form -- it pins the whole gradient, element
//      by element, against the objective the engine itself reports.
//
//   2. BackProp: the same statement in the one direction its private weights
//      allow. Weights is reachable read-only (weightMatrices()); Gradient is
//      not reachable at all. So the gradient is RECOVERED from the update the
//      engine performs, g = ( w_before - w_after ) / eta, and then checked two
//      ways: its maximum absolute element must equal getGradMax(), and the
//      objective must fall along it at the rate g'g as the step shrinks.
//      Recovering g from the update is not a weaker check by accident -- it is
//      also exactly the relation legacy bug #12 broke.
//
//   3. Batch and on-line remain separate: an on-line pass over N exemplars
//      makes N updates and cannot equal the batch pass's single one.
//
//   4. The pass does not consume its own output: two evaluations at the same
//      weights return the identical objective and the identical gradient.
//      That is the property a trial-point evaluator needs and the property a
//      careless extraction removes first.
//
// TOLERANCES ARE NOT PINNED LITERALS. Nothing here compares against a number
// captured on one machine (tests/backprop/check_bpoptimizer.cpp paid for that
// lesson on two other platforms). Every assertion is a RELATION computed in
// the same process, so it holds wherever exp() and log() are correctly
// rounded.
//
// Sabotage evidence for the guards this file provides is in the commit that
// introduces the extraction it guards.

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "backprop.h"
#include "bareprop.h"
#include "dataset.h"
#include "simpleprop.h"
#include "utility.h"

using namespace std;

static int failures = 0;

static void expect( bool ok, const string& what )
{
	if ( ok )
		cout << "ok - " << what << endl;
	else
	{
		cout << "FAIL - " << what << endl;
		failures++;
	}
}

// Swallow engine narration for the duration of a scope
struct Hush {
	ostringstream sink;
	ostream& prev;
	Hush() : prev( util::screen() ) { util::set_screen( sink ); }
	~Hush() { util::set_screen( prev ); }
};

// The same deterministic, learnable, deliberately OFF-CENTRE 2-input problem
//    tests/props uses. Balanced classes put full-batch descent on an exact
//    stationary point, where a gradient check would be checking zero against
//    zero -- the vacuous comparison standing rule 2 forbids.
static DataSet makeData( unsigned n )
{
	Matrix< double > raw( n, 3 );
	for ( unsigned i = 0; i < n; i++ )
	{
		double x0 = -1.0 + 2.0 * ( ( i * 37 ) % 100 ) / 99.0;
		double x1 = -1.0 + 2.0 * ( ( i * 53 ) % 100 ) / 99.0;
		raw( i, 0 ) = x0;
		raw( i, 1 ) = x1;
		raw( i, 2 ) = ( x0 + x1 > 0.55 ) ? 1 : 0;
	}

	DataSet d;
	d.setInput( 2 );
	d.setOutput( 1 );
	d.setDiscrete( true );
	d.setHistory( false );
	Hush quiet;
	d.setRawMatrix( raw );
	Matrix< double >& r = d.getRawMatrix();
	d.setTrainMatrix( r ); // train on everything: no split, nothing to stratify
	return d;
}

// ---------------------------------------------------------------------------
// The one-hidden probe. hW/oW and hG/oG are OneHiddenNet's protected members,
// so a subclass reaches them without widening anything in src/. Reading the
// gradient structure directly rather than through pack() keeps this test
// independent of the packing the extraction is about to give a second caller.

template < class NET >
class OneHiddenProbe : public NET {
public:
	void prepare() { this->prepareRun(); } // the per-run decay constants

	unsigned params() const
	{
		return this->hW.rows() * this->hW.cols()
			+ ( unsigned ) this->oW.size();
	}

	void getWeights( vector< double >& w ) const
	{
		w = this->hW.toVector();
		w.insert( w.end(), this->oW.begin(), this->oW.end() );
	}

	void setWeights( const vector< double >& w )
	{
		unsigned h = this->hW.rows() * this->hW.cols();
		vector< double > head( w.begin(), w.begin() + h );
		toMatrix( this->hW, head );
		this->oW.assign( w.begin() + h, w.end() );
	}

	// The RAW mean gradient left by the last separate-gradient pass, in the
	//    same order as getWeights above.
	void getGradient( vector< double >& g ) const
	{
		g = this->hG.toVector();
		g.insert( g.end(), this->oG.begin(), this->oG.end() );
	}
};

typedef OneHiddenProbe< SimpleProp > SimpleProbe;
typedef OneHiddenProbe< BareProp > BareProbe;

class BackPropProbe : public BackProp {
public:
	void prepare() { prepareRun(); }

	// Read-only, through the narrow accessor BackProp already publishes to
	//    subclasses. There is no write path, which is why the BackProp
	//    assertions below are directional rather than element-by-element.
	void getWeights( vector< double >& w ) const
	{
		w.clear();
		const vector< Matrix< double > >& W = weightMatrices();
		for ( unsigned l = 0; l < W.size(); l++ )
			w.insert( w.end(), W[ l ].begin(), W[ l ].end() );
	}

	// The value the gradient stopping rule reads, which is Network's and
	//    therefore protected.
	double gradMax() { return getGradMax(); }
};

// ---------------------------------------------------------------------------
// Configuration. Every arm runs the SEPARATE-GRADIENT branch -- canonical
// training with gradient stopping armed -- because that is the branch the
// extraction moves, and because engine() has no case 0, so the gradient the
// probe reads afterwards is raw.

template < class NET >
static void configure( NET& net, bool xentropy, bool decayOn, double eta )
{
	net.setHistory( false );
	net.setLastop( false );
	net.setLogPrint( false );
	net.setQuiet( true );
	if ( xentropy ) net.setXEerror(); else net.setLMSerror();
	net.setBatchEpoch( true );
	net.setAutoStepSize( false );
	net.setWeightDecay( decayOn );
	net.setDecay( 5e-5 );
	net.setEta( eta );
	net.setTrainingType( 0 );
	net.setGradStop( true );    // arms the separate-gradient branch
	net.setGradMaxLimit( 0.0 ); // ... with a rule that can never fire
	net.setMinStop( false );
	net.setChangeStop( false );
	net.setWindowStop( false );
}

// ---------------------------------------------------------------------------
// 1. SimpleProp and BareProp: every gradient component against a central
//    finite difference of the objective the engine reports.

template < class PROBE >
static void checkOneHidden( const string& label, bool xentropy, bool decayOn )
{
	DataSet d = makeData( 60 );

	PROBE net;
	Hush quiet;
	net.setDataSet( d );
	net.setHidden( 3 );
	configure( net, xentropy, decayOn, 0.05 );
	util::set_seed( 424242 );
	net.randomize();
	net.prepare(); // train() would do this; trainSet() alone does not

	vector< double > w0;
	net.getWeights( w0 );

	// One pass at w0: the return value is f( w0 ) and the gradient structure
	//    is left holding g( w0 ). The pass also UPDATES the weights, so w0 is
	//    reinstalled afterwards -- that restoration is what makes this a
	//    point evaluation on the old engine, which has no pure evaluator.
	double f0 = net.trainSet();
	vector< double > g0;
	net.getGradient( g0 );
	net.setWeights( w0 );

	expect( g0.size() == w0.size(),
		label + ": packed gradient and weights have the same length" );
	expect( w0.size() > 0 && std::isfinite( f0 ),
		label + ": the objective is finite and the model has parameters" );

	// The gradient must not be uniformly zero -- a check of zero against zero
	//    would pass whatever the extraction did to it.
	double gmax = 0;
	for ( unsigned i = 0; i < g0.size(); i++ )
		gmax = max( gmax, fabs( g0[ i ] ) );
	expect( gmax > 1e-8, label + ": the gradient is not identically zero" );

	// Central differences. h is chosen for a double-precision objective of
	//    order 1: the truncation error is O(h^2) and the cancellation error is
	//    O(eps/h), so h ~ eps^(1/3) ~ 6e-6 is the balance point.
	const double h = 1e-5;
	double worst = 0;
	unsigned worstAt = 0;
	for ( unsigned i = 0; i < w0.size(); i++ )
	{
		vector< double > w = w0;
		w[ i ] = w0[ i ] + h;
		net.setWeights( w );
		double fp = net.trainSet();

		w[ i ] = w0[ i ] - h;
		net.setWeights( w );
		double fm = net.trainSet();

		double fd = ( fp - fm ) / ( 2 * h );
		// Mixed absolute/relative: components differ by orders of magnitude,
		//    and a purely relative bound on a near-zero component would be a
		//    comparison of two roundings.
		double tol = 1e-6 + 1e-5 * fabs( g0[ i ] );
		double err = fabs( fd - g0[ i ] );
		if ( err > worst ) { worst = err; worstAt = i; }
		if ( err > tol )
		{
			cout << "         component " << i << ": analytic "
				<< setprecision( 12 ) << g0[ i ] << ", finite difference "
				<< fd << endl;
			expect( false, label + ": gradient component agrees with the objective" );
			net.setWeights( w0 );
			return;
		}
	}
	net.setWeights( w0 );

	ostringstream s;
	s << label << ": all " << w0.size()
		<< " gradient components agree with a central difference of the "
		<< "objective (worst " << setprecision( 3 ) << worst
		<< " at " << worstAt << ")";
	expect( true, s.str() );
}

// ---------------------------------------------------------------------------
// 2. The evaluation does not consume its own output: the same weights give the
//    same objective and the same gradient, twice.

template < class PROBE >
static void checkRepeatable( const string& label )
{
	DataSet d = makeData( 60 );

	PROBE net;
	Hush quiet;
	net.setDataSet( d );
	net.setHidden( 3 );
	configure( net, true, true, 0.05 );
	util::set_seed( 909090 );
	net.randomize();
	net.prepare();

	vector< double > w0;
	net.getWeights( w0 );

	double f1 = net.trainSet();
	vector< double > g1;
	net.getGradient( g1 );
	net.setWeights( w0 );

	double f2 = net.trainSet();
	vector< double > g2;
	net.getGradient( g2 );
	net.setWeights( w0 );

	expect( f1 == f2, label + ": the objective at the same weights is identical" );
	bool same = ( g1.size() == g2.size() );
	for ( unsigned i = 0; same && i < g1.size(); i++ )
		same = ( g1[ i ] == g2[ i ] );
	expect( same, label + ": the gradient at the same weights is identical" );
}

// ---------------------------------------------------------------------------
// 3. BackProp: the gradient recovered from the update it performs.

static void checkBackProp( bool xentropy, bool decayOn )
{
	string label = string( "BackProp " ) + ( xentropy ? "x-entropy" : "LMS" )
		+ ( decayOn ? " decay" : " no decay" );

	// Two independent networks constructed identically from the same seed, so
	//    both start at the same w0 -- the substitute for the write access
	//    BackProp does not give a subclass.
	const double eta = 1e-3;

	DataSet d = makeData( 60 );
	vector< unsigned > layers( 1, 3 );

	BackPropProbe a;
	{
		Hush quiet;
		a.setDataSet( d );
		a.setHidden( layers );
		configure( a, xentropy, decayOn, eta );
		util::set_seed( 515151 );
		a.randomize();
		a.prepare();
	}

	vector< double > w0;
	a.getWeights( w0 );
	double f0 = a.trainSet();      // returns f( w0 ), leaves w1 = w0 - eta*g
	double gradMax = a.gradMax();
	vector< double > w1;
	a.getWeights( w1 );

	expect( w0.size() == w1.size() && w0.size() > 0,
		label + ": the weight vector is non-empty and keeps its shape" );

	// g recovered from the update the engine performed
	vector< double > g( w0.size() );
	double recoveredMax = 0, gg = 0;
	for ( unsigned i = 0; i < w0.size(); i++ )
	{
		g[ i ] = ( w0[ i ] - w1[ i ] ) / eta;
		recoveredMax = max( recoveredMax, fabs( g[ i ] ) );
		gg += g[ i ] * g[ i ];
	}

	expect( recoveredMax > 1e-8, label + ": the recovered gradient is not zero" );

	// THE UPDATE AND THE STOPPING RULE MUST READ THE SAME STRUCTURE. This is
	//    legacy bug #12's relation: getGradMax() packs the model's Gradient,
	//    and the update must have moved the weights along exactly it.
	expect( fabs( recoveredMax - gradMax ) <= 1e-12 * max( 1.0, gradMax ),
		label + ": the update moved along the gradient the stopping rule reads" );

	// The objective must fall along that gradient at the rate g'g. A second
	//    pass from w1 returns f( w1 ); the first-order prediction is
	//    f(w1) = f(w0) - eta*g'g + O(eta^2), so with eta this small the
	//    relative agreement is dominated by the neglected second-order term.
	double f1 = a.trainSet();
	double predicted = f0 - eta * gg;
	double scale = max( 1e-12, fabs( f0 - f1 ) );
	expect( fabs( f1 - predicted ) < 0.02 * scale + 1e-12,
		label + ": the objective falls along the gradient at the rate g'g" );
}

// ---------------------------------------------------------------------------
// 4. Batch and on-line are different passes and must stay different.

static void checkBatchOnlineSeparation()
{
	DataSet d = makeData( 60 );

	SimpleProbe batch, online;
	{
		Hush quiet;
		batch.setDataSet( d );
		batch.setHidden( 3 );
		configure( batch, true, true, 0.05 );
		util::set_seed( 313131 );
		batch.randomize();
		batch.prepare();

		online.setDataSet( d );
		online.setHidden( 3 );
		configure( online, true, true, 0.05 );
		online.setBatchEpoch( false );
		util::set_seed( 313131 );
		online.randomize();
		online.prepare();
	}

	vector< double > w0a, w0b;
	batch.getWeights( w0a );
	online.getWeights( w0b );
	bool sameStart = ( w0a.size() == w0b.size() );
	for ( unsigned i = 0; sameStart && i < w0a.size(); i++ )
		sameStart = ( w0a[ i ] == w0b[ i ] );
	expect( sameStart, "batch and on-line arms start from identical weights" );

	double fBatch = batch.trainSet();
	double fOnline = online.trainSet();

	vector< double > w1a, w1b;
	batch.getWeights( w1a );
	online.getWeights( w1b );

	// The objective is accumulated over the same exemplars in both, but the
	//    on-line arm updates the weights DURING the pass, so from the second
	//    exemplar onwards it is forward-propagating a different model.
	bool moved = false;
	for ( unsigned i = 0; i < w1a.size(); i++ )
		if ( w1a[ i ] != w1b[ i ] ) moved = true;
	expect( moved, "one on-line pass and one batch pass reach different weights" );
	expect( fBatch != fOnline,
		"one on-line pass and one batch pass report different objectives" );
}

int main()
{
	cout << "batch objective and raw gradient characterization" << endl;

	checkOneHidden< SimpleProbe >( "SimpleProp LMS decay", false, true );
	checkOneHidden< SimpleProbe >( "SimpleProp LMS no decay", false, false );
	checkOneHidden< SimpleProbe >( "SimpleProp x-entropy decay", true, true );
	checkOneHidden< SimpleProbe >( "SimpleProp x-entropy no decay", true, false );

	checkOneHidden< BareProbe >( "BareProp LMS decay", false, true );
	checkOneHidden< BareProbe >( "BareProp LMS no decay", false, false );
	checkOneHidden< BareProbe >( "BareProp x-entropy decay", true, true );
	checkOneHidden< BareProbe >( "BareProp x-entropy no decay", true, false );

	checkRepeatable< SimpleProbe >( "SimpleProp" );
	checkRepeatable< BareProbe >( "BareProp" );

	checkBackProp( true, true );
	checkBackProp( false, true );
	checkBackProp( true, false );

	checkBatchOnlineSeparation();

	cout << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
