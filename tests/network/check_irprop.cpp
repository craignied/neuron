// check_irprop.cpp : the research-only iRPROP+ prototype (src/irprop.*).
//
// THE PUBLISHED TABLE IS DRIVEN BY HAND. IRpropState has no model dependency --
// it is handed an objective and a raw gradient and produces an absolute step --
// so every branch is asserted against values computed here from Igel & Husken
// (2003)'s pseudocode, not inferred from a training run. A branch inferred from
// a training run is a branch nobody checked: a wrong sign, a missing clamp and a
// missing rollback all produce a finite objective that goes down.
//
// THE ONE LINE THAT MATTERS MOST. RPROP+ reverts the previous step on every
// sign flip; iRPROP+ reverts it ONLY when the objective got worse, and takes NO
// step at all when it improved. A test suite that exercises only the rollback
// half would pass unchanged if this code were RPROP+ wearing the iRPROP+ name,
// which is exactly what the plan forbids. So checkNoRollback() below is the
// load-bearing test, and it is one of the two sabotage targets in the log at the
// bottom of this file.
//
// WHAT IS DELIBERATELY NOT CLAIMED. There is no Delta-refusal branch to test.
// Delta starts finite, is only ever multiplied by a finite constant and
// immediately clamped into [DELTA_MIN, DELTA_MAX], so it cannot leave that
// range; the INVARIANT is asserted after every step of every sequence below
// instead of guarding a branch nothing can reach. Saying a guard is unreachable
// is a claim about the code, so it is stated here rather than asserted by a test
// that would pass whatever the guard did.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "backprop.h"
#include "bareprop.h"
#include "dataset.h"
#include "irprop.h"
#include "simpleprop.h"
#include "utility.h"
#include "vector_ops.h"

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

struct Hush {
	ostringstream sink;
	ostream& prev;
	Hush() : prev( util::screen() ) { util::set_screen( sink ); }
	~Hush() { util::set_screen( prev ); }
};

// The Delta invariant, checked after every single application of the table.
static bool deltasSound( const IRpropState& s )
{
	const vector< double >& d = s.deltas();
	if ( d.empty() )
		return false;
	for ( unsigned i = 0; i < d.size(); i++ )
		if ( !std::isfinite( d[ i ] )
			|| d[ i ] < IRpropState::DELTA_MIN
			|| d[ i ] > IRpropState::DELTA_MAX )
			return false;
	return true;
}

// ===========================================================================
// 1. THE HAND-COMPUTED SEQUENCE.
//
// Three coordinates, five iterations, every branch of the published table
// exercised and two of them exercised in the SAME iteration -- which is the
// point of a per-coordinate table and cannot be shown by a one-parameter
// fixture. Every expected value below is written as the paper writes it:
//
//     Delta^(t) = Delta^(t-1) * etaPlus   or   * etaMinus, clamped
//     dw^(t)    = -sign( g ) * Delta^(t)
//
// Exact equality is used, not a tolerance: each expectation is a single
// multiplication of the same two doubles the implementation multiplies, so an
// implementation that computes the published product reproduces it bit for bit,
// and one that computes something else has no reason to land within a
// tolerance either.

static void checkSequence()
{
	const double d0 = IRpropState::DELTA_INIT; // 0.1
	const double up = IRpropState::ETA_PLUS;   // 1.2
	const double dn = IRpropState::ETA_MINUS;  // 0.5

	IRpropState s;
	vector< double > step, w( 3, 0.0 );
	bool sound = true;

	// --- iteration 1: no previous gradient, so every product is zero -------
	//     Delta^(0) = Delta_0 means exactly this: the "= 0" branch at Delta_0.
	vector< double > g1( 3 );
	g1[ 0 ] = 2.0; g1[ 1 ] = -3.0; g1[ 2 ] = 0.0;
	s.computeStep( 10.0, g1, step );
	sound = sound && deltasSound( s );
	w += step;

	expect( s.deltas()[ 0 ] == d0 && s.deltas()[ 1 ] == d0 && s.deltas()[ 2 ] == d0,
		"iteration 1 starts every Delta at the published Delta_0 = 0.1" );
	expect( step[ 0 ] == -d0 && step[ 1 ] == d0,
		"iteration 1 takes the '= 0' branch: -sign(g) * Delta_0" );
	expect( step[ 2 ] == 0.0,
		"sign(0) = 0, so a zero-gradient coordinate takes no step" );
	expect( s.previousGradient() == g1,
		"... and the raw gradient is remembered for the next comparison" );

	// --- iteration 2: two products positive, one still zero ---------------
	vector< double > g2( 3 );
	g2[ 0 ] = 1.0; g2[ 1 ] = -2.0; g2[ 2 ] = 5.0;
	s.computeStep( 9.0, g2, step );
	sound = sound && deltasSound( s );
	w += step;

	expect( s.deltas()[ 0 ] == d0 * up && s.deltas()[ 1 ] == d0 * up,
		"a positive sign product grows Delta by etaPlus" );
	expect( s.deltas()[ 2 ] == d0,
		"... while a zero product leaves that coordinate's Delta alone" );
	expect( step[ 0 ] == -( d0 * up ) && step[ 1 ] == d0 * up,
		"... and the step is -sign(g) * the GROWN Delta" );
	expect( step[ 2 ] == -d0,
		"... and the zero-product coordinate steps by its unchanged Delta" );

	// --- iteration 3: A SIGN FLIP WITH THE OBJECTIVE WORSE -----------------
	//     9.5 > 9.0, so this is the rollback half of the "< 0" branch.
	const double prevStep0 = s.previousStep()[ 0 ];
	vector< double > g3( 3 );
	g3[ 0 ] = -4.0; g3[ 1 ] = -1.0; g3[ 2 ] = 5.0;
	s.computeStep( 9.5, g3, step );
	sound = sound && deltasSound( s );
	w += step;

	expect( s.deltas()[ 0 ] == ( d0 * up ) * dn,
		"a negative sign product shrinks Delta by etaMinus" );
	expect( step[ 0 ] == -prevStep0,
		"... and with E^(t) > E^(t-1) the previous step is reverted exactly" );
	expect( s.rollbacks() == 1, "... which is counted as one rollback" );
	expect( s.previousGradient()[ 0 ] == 0.0,
		"... and the current gradient is zeroed at that coordinate" );
	expect( s.previousGradient()[ 1 ] == -1.0 && s.previousGradient()[ 2 ] == 5.0,
		"... at that coordinate ONLY -- the others remember their gradients" );
	expect( s.deltas()[ 1 ] == ( d0 * up ) * up && step[ 1 ] == ( d0 * up ) * up,
		"... while a coordinate that did not flip grows and steps as before" );

	// --- iteration 4: THE BRANCH THE ZEROING FORCES ------------------------
	//     prevG[0] is 0, so the product is 0 whatever this gradient is, and the
	//     coordinate MUST take the "= 0" branch: it steps by the freshly shrunk
	//     Delta along the NEW gradient's sign. Not a further shrink, not another
	//     rollback. This is what makes the zeroing a mechanism rather than a
	//     cleanup, and it is what pins the claim that dw^(t-1) is never read
	//     after a no-rollback flip.
	const double shrunk0 = ( d0 * up ) * dn;
	vector< double > g4( 3 );
	g4[ 0 ] = 7.0; g4[ 1 ] = -1.0; g4[ 2 ] = 5.0;
	s.computeStep( 9.2, g4, step );
	sound = sound && deltasSound( s );
	w += step;

	expect( s.deltas()[ 0 ] == shrunk0,
		"the iteration after a flip leaves that Delta at its shrunk value" );
	expect( step[ 0 ] == -shrunk0,
		"... and steps by exactly that shrunk Delta, along the new sign" );
	expect( s.rollbacks() == 1,
		"... without a second rollback: the zeroed product cannot reach that branch" );

	// --- iteration 5: A SIGN FLIP WITH THE OBJECTIVE IMPROVED --------------
	//     THE iRPROP+ LINE. 9.0 < 9.2, so the coordinate takes NO step. RPROP+
	//     would revert here.
	vector< double > w4 = w;
	vector< double > g5( 3 );
	g5[ 0 ] = -7.0; g5[ 1 ] = -1.0; g5[ 2 ] = 5.0;
	s.computeStep( 9.0, g5, step );
	sound = sound && deltasSound( s );
	w += step;

	expect( step[ 0 ] == 0.0,
		"a sign flip with E^(t) <= E^(t-1) applies NO step -- the iRPROP+ line" );
	expect( w[ 0 ] == w4[ 0 ],
		"... leaving that weight bit-identical" );
	expect( s.deltas()[ 0 ] == shrunk0 * dn,
		"... while Delta still shrinks, exactly as the branch says" );
	expect( s.heldFlips() == 1 && s.rollbacks() == 1,
		"... and it is counted as a held flip, not a rollback" );

	expect( sound, "every Delta stayed finite and within [1e-6, 50] throughout" );
	expect( s.iterations() == 5, "five applications of the table were counted" );
}

// ===========================================================================
// 2 + 3. The clamps.

static void checkGrowthClamp()
{
	IRpropState s;
	vector< double > step, g( 2, 1.0 );
	bool sound = true, everAbove = false;

	// The same gradient every time, so every product after the first is
	//    positive and Delta grows by etaPlus without interruption.
	//    0.1 * 1.2^k first reaches 50 at k = 35, so 60 iterations is well past
	//    the clamp and the last 25 test that it HOLDS rather than merely arrives.
	for ( unsigned i = 0; i < 60; i++ )
	{
		s.computeStep( 100.0 - i, g, step );
		sound = sound && deltasSound( s );
		if ( s.deltas()[ 0 ] > IRpropState::DELTA_MAX )
			everAbove = true;
	}

	expect( s.deltas()[ 0 ] == IRpropState::DELTA_MAX,
		"repeated positive sign products drive Delta to DeltaMax = 50" );
	expect( !everAbove && sound, "... and it never exceeds it" );
	expect( step[ 0 ] == -IRpropState::DELTA_MAX,
		"... so the step is the clamped magnitude, not the unclamped product" );
}

static void checkShrinkClamp()
{
	IRpropState s;
	vector< double > step, g( 2, 1.0 );
	bool sound = true, everBelow = false;

	// Alternating gradient signs. The shrink happens on the flip; the NEXT
	//    iteration has a zeroed previous gradient and so takes the "= 0" branch
	//    and does NOT shrink. Delta therefore halves every TWO iterations, and
	//    that cadence is itself asserted below -- an implementation that omitted
	//    the zeroing would shrink on every iteration and reach the floor twice
	//    as fast.
	double after2 = 0, after3 = 0, after4 = 0, after5 = 0;
	for ( unsigned i = 0; i < 100; i++ )
	{
		g.assign( 2, ( i % 2 ) ? -1.0 : 1.0 );
		s.computeStep( 100.0 - i, g, step ); // improving, so flips are held
		sound = sound && deltasSound( s );
		if ( s.deltas()[ 0 ] < IRpropState::DELTA_MIN )
			everBelow = true;
		if ( i == 1 ) after2 = s.deltas()[ 0 ];
		if ( i == 2 ) after3 = s.deltas()[ 0 ];
		if ( i == 3 ) after4 = s.deltas()[ 0 ];
		if ( i == 4 ) after5 = s.deltas()[ 0 ];
	}

	const double d0 = IRpropState::DELTA_INIT, dn = IRpropState::ETA_MINUS;
	expect( after2 == d0 * dn, "a flip shrinks Delta by etaMinus" );
	expect( after3 == after2,
		"... and the iteration after it does NOT: the zeroed gradient sends it "
		"through the '= 0' branch" );
	expect( after4 == ( d0 * dn ) * dn && after5 == after4,
		"... so Delta halves every two iterations, not every one" );
	expect( s.deltas()[ 0 ] == IRpropState::DELTA_MIN,
		"repeated flips drive Delta to DeltaMin = 1e-6" );
	expect( !everBelow && sound, "... and it never falls below it" );
}

// ===========================================================================
// 5. The no-rollback branch, isolated.
//
// checkSequence() reaches it at the end of a five-step history. This reaches it
// in the shortest possible way and asserts the WEIGHT, so the claim "no step"
// does not rest on the step vector alone.

static void checkNoRollback()
{
	vector< double > step, w( 1, 4.0 ), g( 1, 1.0 );

	// The same three iterations, differing ONLY in the objective at the flip.
	//    Everything else -- gradients, Delta history, previous step -- is
	//    identical, so the two outcomes differ by the error test and nothing
	//    else. That is the control: without it, "no step" could be a fixture
	//    that never steps.
	for ( int worse = 0; worse < 2; worse++ )
	{
		IRpropState s;
		vector< double > wi = w;

		g[ 0 ] = 1.0;
		s.computeStep( 10.0, g, step ); wi += step;   // "= 0" branch
		g[ 0 ] = 1.0;
		s.computeStep( 9.0, g, step ); wi += step;    // "> 0" branch
		const double appliedBefore = s.previousStep()[ 0 ];
		const vector< double > wBefore = wi;

		g[ 0 ] = -1.0;                                // the flip
		s.computeStep( worse ? 9.5 : 8.5, g, step );
		wi += step;

		if ( worse )
		{
			expect( step[ 0 ] == -appliedBefore,
				"E^(t) > E^(t-1): the previous step is reverted exactly" );
			expect( s.rollbacks() == 1 && s.heldFlips() == 0,
				"... and counted as a rollback" );
		}
		else
		{
			expect( step[ 0 ] == 0.0 && wi[ 0 ] == wBefore[ 0 ],
				"E^(t) <= E^(t-1): no step is applied and the weight is unchanged" );
			expect( s.rollbacks() == 0 && s.heldFlips() == 1,
				"... and counted as a held flip, never a rollback" );
			expect( s.previousStep()[ 0 ] == 0.0,
				"... and the remembered applied step is the zero actually applied" );
		}

		// Common to both: the shrink and the zeroing happen either way.
		expect( s.deltas()[ 0 ] == ( IRpropState::DELTA_INIT
				* IRpropState::ETA_PLUS ) * IRpropState::ETA_MINUS,
			string( worse ? "rollback" : "held flip" )
				+ ": Delta shrinks on the flip regardless of the objective" );
		expect( s.previousGradient()[ 0 ] == 0.0,
			string( worse ? "rollback" : "held flip" )
				+ ": the current gradient is zeroed regardless of the objective" );
	}
}

// ===========================================================================
// 7. A wholly zero gradient.

static void checkZeroGradient()
{
	IRpropState s;
	vector< double > step, w( 3, 1.5 ), g( 3, 0.0 );

	s.computeStep( 5.0, g, step );
	vector< double > before = w;
	w += step;

	expect( step[ 0 ] == 0.0 && step[ 1 ] == 0.0 && step[ 2 ] == 0.0,
		"a wholly zero gradient produces a wholly zero step" );
	expect( w == before, "... leaving every weight bit-identical" );
	expect( s.deltas()[ 0 ] == IRpropState::DELTA_INIT,
		"... and every Delta unchanged" );

	// And again, so a zero product against a zero previous gradient is also
	//    the "= 0" branch rather than anything else.
	s.computeStep( 5.0, g, step );
	expect( step[ 0 ] == 0.0 && s.rollbacks() == 0 && s.heldFlips() == 0,
		"... twice over, with no branch other than '= 0' reached" );
}

// ===========================================================================
// 8. Non-finite refusal, with the state proven untouched.

static void checkNonFinite()
{
	const double nan = numeric_limits< double >::quiet_NaN();
	const double inf = numeric_limits< double >::infinity();

	// Before the first iteration: refused, and the state stays uninitialized.
	{
		IRpropState s;
		vector< double > step, g( 2, 1.0 );
		bool threw = false;
		try { s.computeStep( nan, g, step ); }
		catch ( IRpropState::NotFinite& ) { threw = true; }
		expect( threw && !s.started(),
			"a non-finite objective is refused before the state is initialized" );
	}

	// After a clean iteration: refused, with EVERY piece of state identical.
	const double bad[ 3 ] = { nan, inf, -inf };
	const char* what[ 3 ] = { "NaN", "+Inf", "-Inf" };
	for ( int k = 0; k < 3; k++ )
	{
		IRpropState s;
		vector< double > step, g( 2, 1.0 ), w( 2, 0.0 );
		s.computeStep( 10.0, g, step );
		w += step;

		const vector< double > d0 = s.deltas(), p0 = s.previousGradient(),
			s0 = s.previousStep(), w0 = w;
		const double f0 = s.previousObjective();
		const unsigned long long n0 = s.iterations();

		// (a) the objective
		vector< double > gOk( 2, 1.0 );
		bool threwF = false;
		try { s.computeStep( bad[ k ], gOk, step ); }
		catch ( IRpropState::NotFinite& ) { threwF = true; }

		// (b) one gradient element
		vector< double > gBad( 2, 1.0 );
		gBad[ 1 ] = bad[ k ];
		bool threwG = false;
		try { s.computeStep( 9.0, gBad, step ); }
		catch ( IRpropState::NotFinite& ) { threwG = true; }

		expect( threwF, string( "a " ) + what[ k ] + " objective is refused" );
		expect( threwG, string( "a " ) + what[ k ]
			+ " gradient element is refused" );
		expect( s.deltas() == d0 && s.previousGradient() == p0
				&& s.previousStep() == s0 && s.previousObjective() == f0
				&& s.iterations() == n0 && w == w0,
			string( "... and " ) + what[ k ]
				+ " leaves Delta, the remembered gradient and step, the "
				  "objective, the counter and the weights untouched" );
	}

	// The control: the same fixture with finite values does step, so "untouched"
	//    above is the refusal firing and not an inert fixture.
	{
		IRpropState s;
		vector< double > step, g( 2, 1.0 );
		s.computeStep( 10.0, g, step );
		const vector< double > d0 = s.deltas();
		vector< double > g2( 2, 1.0 );
		s.computeStep( 9.0, g2, step );
		expect( s.deltas() != d0 && s.iterations() == 2,
			"... while the finite control does advance the state" );
	}
}

// ===========================================================================
// The real-model side. One dataset, three architectures.

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
	d.setTrainMatrix( r );
	return d;
}

// The narrowest seam that lets a test read what production keeps protected.
//    innerTrainSet() is overridden in the idiom tests/network/check_autostep.cpp
//    established: count, then call the production implementation, so what is
//    counted is the dispatch that actually ran.
template < class NET >
class NetProbe : public NET {
public:
	NetProbe() : inners( 0 ) { }
	const IRpropState& optimizer() const { return this->irprop; }
	double gradMaxNow() { return this->getGradMax(); }
	unsigned packed() const { return this->packedSize(); }
	void packW( vector< double >& w ) const { this->packWeights( w ); }
	double evalAt( vector< double >& g ) { return this->batchObjectiveGradient( g ); }

	virtual double innerTrainSet() { inners++; return NET::innerTrainSet(); }
	unsigned long long inners;
};

// A model that reports no packed boundary. packedSize() is the eligibility
//    question, and this is the only way to ask it of a neural model -- all three
//    implement the boundary, so the refusal would otherwise be unreachable and
//    therefore untested.
class Boundaryless : public NetProbe< SimpleProp > {
protected:
	virtual unsigned packedSize() const { return 0; }
};

template < class NET >
static void setupNeural( NET& p, DataSet& d, unsigned seed, double eta )
{
	Hush quiet;
	p.setDataSet( d );
	p.setHidden( 3 );
	p.setHistory( false );
	p.setLastop( false );
	p.setLogPrint( false );
	p.setQuiet( true );
	p.setXEerror();
	p.setBatchEpoch( true );
	p.setAutoStepSize( false );
	p.setWeightDecay( true );
	p.setDecay( 5e-5 );
	p.setEta( eta );
	p.setTrainingType( Network::TRAIN_IRPROP );
	p.setMinStop( false );
	p.setChangeStop( false );
	p.setWindowStop( false );
	p.setGradStop( false );
	p.setMaxIterations( 40 );
	util::set_seed( seed );
	p.randomize();
}

// ===========================================================================
// 15 + 17. currGradMax is the RAW gradient, and the returned objective is the
// pre-update one.
//
// Both are read one iteration at a time against an INDEPENDENT evaluation taken
// at the same installed weights before the iteration runs.

static void checkRawGradMaxAndReturn()
{
	DataSet d = makeData( 60 );
	NetProbe< SimpleProp > p;
	setupNeural( p, d, 20260806, 0.05 );

	bool gradMaxRaw = true, returnedPreUpdate = true;
	for ( unsigned i = 0; i < 30; i++ )
	{
		vector< double > gIndep;
		double fIndep;
		{ Hush quiet; fIndep = p.evalAt( gIndep ); }
		const double rawMax = maxabs( gIndep );

		double returned;
		{ Hush quiet; returned = p.innerTrainSet(); }

		if ( p.gradMaxNow() != rawMax )
			gradMaxRaw = false;
		if ( returned != fIndep )
			returnedPreUpdate = false;
	}

	// NON-VACUITY. If no sign flip ever occurred, no gradient was ever zeroed
	//    and the assertion above could not have distinguished the raw gradient
	//    from the post-table one. The test says so rather than passing quietly.
	expect( p.optimizer().shrinks() > 0,
		"the run really did flip signs, so gradient zeroing was in play" );
	expect( gradMaxRaw,
		"currGradMax is maxabs of the RAW gradient, not of the zeroed one" );
	expect( returnedPreUpdate,
		"the returned objective is the one at the point the step departed from" );
	expect( p.optimizer().iterations() == p.inners,
		"exactly one application of the table per full training-set traversal" );
}

// ===========================================================================
// 10 + 11. eta is neither read nor written.

static void checkEtaIndependence()
{
	DataSet d = makeData( 60 );

	NetProbe< SimpleProp > slow, fast;
	setupNeural( slow, d, 31415, 0.05 );
	setupNeural( fast, d, 31415, 0.90 );

	vector< double > w0a, w0b;
	slow.packW( w0a );
	fast.packW( w0b );
	expect( w0a == w0b && !w0a.empty(), "both arms start from identical weights" );

	double fa, fb;
	{ Hush quiet; fa = slow.train(); }
	{ Hush quiet; fb = fast.train(); }

	vector< double > w1a, w1b;
	slow.packW( w1a );
	fast.packW( w1b );

	// An eta of 0.05 and an eta of 0.9 differ by a factor of eighteen. If a
	//    single step were scaled by it -- or divided by it -- these could not
	//    agree bit for bit.
	expect( fa == fb && w1a == w1b,
		"eta 0.05 and eta 0.90 reach bit-identical weights: the step is ABSOLUTE" );
	expect( slow.getEta() == 0.05 && fast.getEta() == 0.90,
		"... and neither run wrote the configured eta" );

	// And the configured eta still means what it meant to canonical training:
	//    a canonical run after an iRPROP+ run matches one that never saw it.
	NetProbe< SimpleProp > used, fresh;
	setupNeural( used, d, 27182, 0.05 );
	{ Hush quiet; used.train(); }
	used.setTrainingType( 0 );
	used.setMaxIterations( 20 );
	util::set_seed( 27182 );
	{ Hush quiet; used.randomize(); }

	setupNeural( fresh, d, 27182, 0.05 );
	fresh.setTrainingType( 0 );
	fresh.setMaxIterations( 20 );
	util::set_seed( 27182 );
	{ Hush quiet; fresh.randomize(); }

	double fu, ff;
	{ Hush quiet; fu = used.train(); }
	{ Hush quiet; ff = fresh.train(); }
	vector< double > wu, wf;
	used.packW( wu );
	fresh.packW( wf );
	expect( fu == ff && wu == wf,
		"canonical training after an iRPROP+ run is bit-identical to canonical "
		"that never saw one" );
}

// ===========================================================================
// 12, 13, 14. The refusals, each with the weights proven unmoved.

static void checkRefusals()
{
	DataSet d = makeData( 60 );

	struct Trial { const char* label; int kind; };
	const Trial trials[ 3 ] = {
		{ "on-line mode", 0 },
		{ "the automatic step-size search", 1 },
		{ "a model with no packed boundary", 2 }
	};

	for ( int k = 0; k < 3; k++ )
	{
		NetProbe< SimpleProp > p;
		Boundaryless b;
		Network* net;

		if ( trials[ k ].kind == 2 )
		{
			setupNeural( b, d, 31415, 0.05 );
			net = &b;
		}
		else
		{
			setupNeural( p, d, 31415, 0.05 );
			if ( trials[ k ].kind == 0 )
				p.setBatchEpoch( false );
			else
				p.setAutoStepSize( true );
			net = &p;
		}

		vector< double > before, after;
		if ( trials[ k ].kind == 2 )
			b.NetProbe< SimpleProp >::packW( before );
		else
			p.packW( before );

		bool threw = false;
		string why;
		try { Hush quiet; net->innerTrainSet(); }
		catch ( IRpropState::Ineligible& e ) { threw = true; why = e.what(); }

		if ( trials[ k ].kind == 2 )
			b.NetProbe< SimpleProp >::packW( after );
		else
			p.packW( after );

		expect( threw, string( "iRPROP+ refuses " ) + trials[ k ].label );
		expect( threw && !why.empty() && after == before,
			string( "... by name, before any weight moves (" ) + why + ")" );
	}

	// A direct call with no parameters at all.
	{
		IRpropState s;
		vector< double > step, empty;
		bool threw = false;
		try { s.computeStep( 1.0, empty, step ); }
		catch ( IRpropState::Ineligible& ) { threw = true; }
		expect( threw, "iRPROP+ refuses a model with no parameters" );
	}
}

// ===========================================================================
// 16. Per-run reset and copy.

static void checkLifecycle()
{
	DataSet d = makeData( 60 );

	NetProbe< SimpleProp > p;
	setupNeural( p, d, 20260806, 0.05 );

	double f1;
	{ Hush quiet; f1 = p.train(); }
	unsigned long long firstRun = p.optimizer().iterations();
	vector< double > w1;
	p.packW( w1 );
	expect( firstRun > 0, "a run applies the published table" );

	// THE RESET, asserted exactly. Back to the identical starting weights, the
	//    second run must reproduce the first in every respect. It can only do so
	//    if Delta, the remembered gradient, the remembered step and the previous
	//    objective were all reset by prepareRun(): a run inheriting a grown
	//    Delta would take a different first step and never converge on the same
	//    end state.
	util::set_seed( 20260806 );
	{ Hush quiet; p.randomize(); }
	double f2;
	{ Hush quiet; f2 = p.train(); }
	vector< double > w2;
	p.packW( w2 );

	expect( p.optimizer().iterations() == firstRun,
		"a second run from the same start applies the table the same number of times" );
	expect( f1 == f2 && w1 == w2,
		"... and reaches the identical objective and end weights" );

	// A copy carries no working state. There is no CONFIGURATION to carry
	//    either -- every constant is published and fixed -- so "clean" is the
	//    whole contract.
	NetProbe< SimpleProp > clone;
	clone = p;
	expect( !clone.optimizer().started() && clone.optimizer().iterations() == 0
			&& clone.optimizer().deltas().empty(),
		"a copy carries none of the original's mid-run state" );
}

// ===========================================================================
// 18. Fixed-start integration for the three neural models.

template < class NET >
static void checkIntegration( const string& label, DataSet& d )
{
	NetProbe< NET > a, b;
	setupNeural( a, d, 31415, 0.05 );
	setupNeural( b, d, 31415, 0.05 );

	vector< double > g0;
	double f0;
	{ Hush quiet; f0 = a.evalAt( g0 ); }

	double fa, fb;
	{ Hush quiet; fa = a.train(); }
	{ Hush quiet; fb = b.train(); }

	vector< double > w1a, w1b;
	a.packW( w1a );
	b.packW( w1b );
	bool finite = true;
	for ( unsigned i = 0; i < w1a.size(); i++ )
		if ( !std::isfinite( w1a[ i ] ) )
			finite = false;

	expect( std::isfinite( fa ) && finite,
		label + ": the run ends with a finite objective and finite weights" );
	expect( fa < f0, label + ": the objective fell below its starting value" );
	expect( fa == fb && w1a == w1b,
		label + ": the same fixed start reaches the identical end state" );
	expect( a.optimizer().iterations() == a.inners,
		label + ": one table application per full training-set traversal" );

	cout << "     " << label << ": " << a.inners << " traversals, objective "
		<< setprecision( 8 ) << f0 << " -> " << fa
		<< " (grew " << a.optimizer().growths()
		<< ", shrank " << a.optimizer().shrinks()
		<< ", rolled back " << a.optimizer().rollbacks()
		<< ", held " << a.optimizer().heldFlips() << ")" << endl;
}

static void checkBackPropIntegration( DataSet& d )
{
	NetProbe< BackProp > a;
	{
		Hush quiet;
		vector< unsigned > layers( 1, 3 );
		a.setDataSet( d );
		a.setHidden( layers );
		a.setHistory( false );
		a.setLastop( false );
		a.setLogPrint( false );
		a.setQuiet( true );
		a.setXEerror();
		a.setBatchEpoch( true );
		a.setAutoStepSize( false );
		a.setWeightDecay( true );
		a.setDecay( 5e-5 );
		a.setEta( 0.05 );
		a.setTrainingType( Network::TRAIN_IRPROP );
		a.setMinStop( false );
		a.setChangeStop( false );
		a.setWindowStop( false );
		a.setGradStop( false );
		a.setMaxIterations( 40 );
		util::set_seed( 31415 );
		a.randomize();
	}

	vector< double > g0;
	double f0, fa;
	{ Hush quiet; f0 = a.evalAt( g0 ); }
	{ Hush quiet; fa = a.train(); }

	expect( std::isfinite( fa ) && fa < f0,
		"BackProp: the objective fell to a finite value from a fixed start" );
	expect( a.optimizer().iterations() == a.inners,
		"BackProp: one table application per full training-set traversal" );

	cout << "     BackProp: " << a.inners << " traversals, objective "
		<< setprecision( 8 ) << f0 << " -> " << fa << endl;
}

int main()
{
	cout << "iRPROP+ (research prototype)" << endl;

	checkSequence();
	checkGrowthClamp();
	checkShrinkClamp();
	checkNoRollback();
	checkZeroGradient();
	checkNonFinite();

	checkRawGradMaxAndReturn();
	checkEtaIndependence();
	checkRefusals();
	checkLifecycle();

	DataSet d = makeData( 60 );
	checkIntegration< SimpleProp >( "SimpleProp", d );
	checkIntegration< BareProp >( "BareProp", d );
	checkBackPropIntegration( d );

	cout << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}

// ===========================================================================
// SABOTAGE LOG. Each was applied to src/irprop.cpp, watched to fail after a
// visible recompilation of that translation unit, then restored and watched to
// pass again after a second visible recompilation.
//
// 1. THE ERROR-DEPENDENT TEST DELETED from the "< 0" branch, so it always
//    reverts -- which is RPROP+ wearing the iRPROP+ name, the exact defect this
//    file exists to prevent. Six assertions failed, and the one that NAMES the
//    mechanism was among them:
//
//        FAIL - a sign flip with E^(t) <= E^(t-1) applies NO step
//               -- the iRPROP+ line
//        FAIL - ... leaving that weight bit-identical
//        FAIL - ... and it is counted as a held flip, not a rollback
//        FAIL - E^(t) <= E^(t-1): no step is applied and the weight is unchanged
//        FAIL - ... and counted as a held flip, never a rollback
//        FAIL - ... and the remembered applied step is the zero actually applied
//
//    THE CONTROL: the ROLLBACK assertions beside them still PASSED -- "E^(t) >
//    E^(t-1): the previous step is reverted exactly" is satisfied by RPROP+ by
//    construction, so a sabotage that failed those too would have meant the test
//    was detecting something other than the error test.
//
//    WHAT ALSO PASSED, and it is the reason this file drives the table by hand:
//    every real-model integration test. Under RPROP+ the objective still fell,
//    the run was still deterministic, the weights were still finite and the
//    traversal count was still one per iteration. A suite built only from
//    training runs would have shipped the wrong algorithm green.
//
// 2. THE CURRENT-GRADIENT ZEROING DELETED (`prevG[i] = g` instead of 0). Nine
//    assertions failed. Three name the mechanism directly:
//
//        FAIL - ... and the current gradient is zeroed at that coordinate
//        FAIL - held flip: the current gradient is zeroed regardless of the
//               objective
//        FAIL - rollback: the current gradient is zeroed regardless of the
//               objective
//
//    and six are its consequences -- the branch the zeroing forces on the next
//    iteration, and the two-iterations-per-shrink cadence that follows from it:
//
//        FAIL - the iteration after a flip leaves that Delta at its shrunk value
//        FAIL - ... and steps by exactly that shrunk Delta, along the new sign
//        FAIL - ... while Delta still shrinks, exactly as the branch says
//        FAIL - ... and it is counted as a held flip, not a rollback
//        FAIL - ... and the iteration after it does NOT: the zeroed gradient
//               sends it through the '= 0' branch
//        FAIL - ... so Delta halves every two iterations, not every one
