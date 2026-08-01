// check_quietprep.cpp : a QUIET run is a SILENT run, never a DIFFERENT one.
//
// THE BUG (found 2026-08-01, by the characterization suite in tests/props).
// Network::runHeader() does two unrelated jobs. Most of it prints the run's
// parameters -- reporting. But its first two statements compute
//
//     regularizer = decay / 2;
//     decayTerm   = 1 - ( eta * decay );
//
// and those are TRAINING CONSTANTS: innerTrainSet reads regularizer to add the
// weight-decay penalty to the reported error, and multiplies the weights by
// decayTerm on every exemplar of a canonical run.
//
// Iterative::train() called runHeader() from INSIDE its `if ( !quietFlag )`
// block. Neither constant was initialised by any constructor. So a quiet run
// with weight decay enabled trained on two uninitialised doubles.
//
// Measured before the fix, on the tests/props fixture: a build with
// -ftrivial-auto-var-init=pattern produced NaN; this machine's ordinary
// Release build happened to produce 0.673 instead of the correct 0.0223. Same
// source, same seed, different answer -- decided by what was on the stack.
//
// HOW FAR IT REACHED -- stated precisely, because the first reading of this
// was wrong. The only production caller of setQuiet( true ) is
// RegressNet::copy_network() (regressnet.cpp), on a CLONE of the user's
// already-trained model. Network::copy copies decayTerm and regularizer
// (network.cpp), and the parent computed them correctly during its own audible
// training run -- so the clone inherits good values and never needs to derive
// them. CV, OBD and autoalgo do not use quiet at all; they redirect the screen,
// so their runs call runHeader normally.
//
// So NO shipped path is known to have hit this. It was a latent defect: genuine
// undefined behaviour, reachable by any direct engine caller (this test reaches
// it), held harmless only by the accident that the sole quiet caller always
// clones a parent that has already trained. A first quiet run on a fresh model,
// or a clone of a template that never trained, would have hit it. The goldens
// confirm the point from the other side: regress_seed42 runs quiet stepwise
// candidates WITH weight decay on and is byte-identical across this fix,
// because every one of those candidates inherited its constants.
//
// This is legacy bug #10's sibling, and the third of its family: a quantity
// the training math depends on, computed inside a block that runs only when
// something is displayed. CLAUDE.md's settled decision on setQuiet says it in
// as many words -- "It is OUTPUT ONLY: it may never guard a calculation a
// stopping rule reads" -- and the offending call sat ten lines below it. A
// comment is not an implementation.
//
// THE FIX. Run preparation is separated from report generation:
// Iterative::prepareRun() is called once per train(), OUTSIDE every reporting
// guard; Network::prepareRun() computes the two constants; runHeader() is
// reporting only. Constructors also initialise both, so a direct caller that
// never trains cannot read indeterminate storage either -- but prepareRun()
// remains the authoritative per-run calculation, because eta and decay may
// change between runs.
//
// THE INVARIANT ASSERTED HERE: two otherwise identical runs, one quiet and one
// audible, must agree exactly -- final error, every forward output, stop
// reason, convergence, and iteration count. Weight decay ON is the
// configuration that exposed it; decay OFF is included as the control that
// must also hold.
//
// SABOTAGE: move the two assignments back into runHeader() (or move the
// prepareRun() call back inside the `if ( !quietFlag )` block) and the
// decay-ON cases below fail -- with NaN under pattern initialisation, and with
// a wrong-but-stable number under an ordinary build.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "simpleprop.h"
#include "logistic.h"
#include "dataset.h"
#include "utility.h"

using namespace std;

int failures = 0;

void expect( bool ok, const string& what )
{
	if ( ok )
		cout << "ok - " << what << endl;
	else
	{
		cout << "FAIL - " << what << endl;
		failures++;
	}
}

// Redirect engine narration for a scope, restoring the PREVIOUS stream
struct Hush {
	ostringstream sink;
	ostream& prev;
	Hush() : prev( util::screen() ) { util::set_screen( sink ); }
	~Hush() { util::set_screen( prev ); }
};

// Exposes the forward output of a training row (o and Train are protected)
class Probe : public SimpleProp {
public:
	unsigned rows() { return Train.rows(); }
	double out( unsigned r ) { forward( Train, r ); return o; }
};

class ProbeLogistic : public Logistic {
public:
	unsigned rows() { return Train.rows(); }
	double out( unsigned r ) { forward( Train, r ); return o; }
};

static DataSet makeData( unsigned n, unsigned nTest )
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
	d.randomize( nTest );
	return d;
}

// What a finished run is, observably: its error, its outputs, and how it ended
struct Outcome {
	double error = 0;
	vector< double > outputs;
	string stopReason;
	bool converged = false;
	unsigned iterations = 0;
};

// Train one SimpleProp under an explicit quiet setting. Everything else --
// seed, architecture, weights, optimizer, decay, ceiling -- is identical
// between the two calls; quiet is the ONLY difference.
static Outcome runProp( DataSet& d, bool quiet, bool decay )
{
	Probe net;
	net.setDataSet( d );
	net.setHidden( 3 );
	net.setHistory( false );
	net.setLastop( false );
	net.setLogPrint( false );
	net.setXEerror();
	net.setWeightDecay( decay );
	net.setDecay( decay ? 5e-5 : 0 );
	net.setBatchEpoch( false ); // canonical on-line: the decayTerm path
	net.setAutoStepSize( false );
	net.setEta( 0.05 );
	net.setGradStop( false );
	net.setMaxIterations( 200 );

	util::set_seed( 7 );
	net.randomize();
	net.setQuiet( quiet );

	Outcome r;
	{
		Hush hush; // an audible run must not spray the test's output
		r.error = net.train();
	}
	for ( unsigned i = 0; i < net.rows(); i++ )
		r.outputs.push_back( net.out( i ) );
	r.stopReason = Iterative::stopReasonToken( net.getStopReason() );
	r.converged = Iterative::converged( net.getStopReason() );
	r.iterations = net.getIterations();
	return r;
}

static Outcome runLogistic( DataSet& d, bool quiet, bool decay )
{
	ProbeLogistic net;
	net.setDataSet( d );
	net.setHistory( false );
	net.setLastop( false );
	net.setLogPrint( false );
	net.setWeightDecay( decay );
	net.setDecay( decay ? 5e-5 : 0 );
	net.setAutoStepSize( false );
	net.setEta( 0.05 );
	net.setGradStop( false );
	net.setMaxIterations( 200 );

	util::set_seed( 7 );
	net.randomize();
	net.setQuiet( quiet );

	Outcome r;
	{
		Hush hush;
		r.error = net.train();
	}
	for ( unsigned i = 0; i < net.rows(); i++ )
		r.outputs.push_back( net.out( i ) );
	r.stopReason = Iterative::stopReasonToken( net.getStopReason() );
	r.converged = Iterative::converged( net.getStopReason() );
	r.iterations = net.getIterations();
	return r;
}

// Exact comparison. There is no tolerance to argue about here: the two runs
// execute the same arithmetic in the same order on the same data, so anything
// but bit-identical means quiet changed the computation.
static void compare( const Outcome& q, const Outcome& a, const string& label,
	bool compareWeights = true )
{
	bool errorSame = ( q.error == a.error );
	if ( !errorSame )
		cout << "         quiet=" << setprecision( 17 ) << q.error
			<< "  audible=" << a.error << endl;
	expect( errorSame, label + ": identical final error" );

	bool bothFinite = std::isfinite( q.error ) && std::isfinite( a.error );
	expect( bothFinite, label + ": both runs produced a finite error" );

	if ( compareWeights )
	{
		bool outputsSame = ( q.outputs.size() == a.outputs.size() );
		for ( unsigned i = 0; outputsSame && i < q.outputs.size(); i++ )
			if ( q.outputs[ i ] != a.outputs[ i ] )
				outputsSame = false;
		expect( outputsSame, label + ": identical weights (every forward output)" );
	}

	expect( q.stopReason == a.stopReason, label + ": identical stop reason" );
	expect( q.converged == a.converged, label + ": identical convergence verdict" );
	expect( q.iterations == a.iterations, label + ": identical iteration count" );
}

int main()
{
	util::set_seed( 4242 );
	DataSet d = makeData( 120, 36 );

	// Weight decay ON: the configuration that read the uninitialised constants
	compare( runProp( d, true, true ), runProp( d, false, true ),
		"SimpleProp, weight decay ON" );

	// Weight decay OFF: the control. Neither constant is read, so this held
	// even before the fix -- which is precisely why the defect stayed hidden.
	compare( runProp( d, true, false ), runProp( d, false, false ),
		"SimpleProp, weight decay OFF" );

	// Logistic reads regularizer on the same path, and is the model stepwise
	// regression is most often run over.
	// Logistic, now including the weight comparison: D2 (computeCondNum's
	//    mutating gradient harvest) is fixed, so an audible run no longer ends
	//    a gradient step past a quiet one. See tests/iterative/check_condnum.cpp.
	compare( runLogistic( d, true, true ), runLogistic( d, false, true ),
		"Logistic, weight decay ON" );

	cout << endl << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
