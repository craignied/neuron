// check_packedboundary.cpp : the packed parameter boundary (network.h).
//
// The boundary exists because a genuine Wolfe line search must evaluate the
// objective at TRIAL points, and innerTrainSet() cannot be that evaluator: it
// updates the weights. Three methods, one layout:
//
//     packedSize()               -- is the boundary available at all
//     packWeights / unpackWeights -- the parameter vector out and back
//     batchObjectiveGradient()   -- f and RAW g at the installed weights
//
// WHAT IS ASSERTED, and why each assertion is not vacuous:
//
//   1. ROUND TRIP. unpack( pack( w ) ) restores every weight EXACTLY, and the
//      model computes the identical function afterwards. Guarded against the
//      vacuous form: the vector must be non-empty, and the round trip is also
//      run through a PERTURBED vector, so a pack/unpack pair that ignored its
//      argument entirely would fail. (An identity round trip on untouched
//      weights passes trivially for a no-op implementation -- that is the
//      "composite that moves for the wrong reason" shape.)
//
//   2. ONE LAYOUT. The packed weights and the packed gradient have the same
//      length, for every model. A step computed in one and applied to the
//      other is only meaningful if that holds.
//
//   3. THE EXTRACTION'S CENTRAL CLAIM. batchObjectiveGradient() returns
//      BIT-IDENTICAL numbers to the legacy training pass at the same weights --
//      the same objective and the same gradient, ==, not near. That is what
//      "one authoritative implementation" means, and it is the assertion that
//      fails if the extraction ever grows a second copy of the model
//      equations.
//
//   4. IT EVALUATES, IT DOES NOT TRAIN. After a call the weights are
//      unchanged, and so are the optimizer's carried state (lastG, lastF),
//      currGradMax, eta and the iteration counter. Checked after a call that
//      demonstrably did work -- a finite objective and a non-zero gradient --
//      so "nothing moved" is not "nothing happened".
//
//   5. THE GRADIENT IS THE GRADIENT OF THE OBJECTIVE, by central finite
//      difference, for all three neural models. BackProp is included here and
//      not in check_batchgradient.cpp for a specific reason: before the
//      boundary existed there was no way to write a BackProp weight from a
//      subclass, so the old-engine characterization could only check one
//      direction. unpackWeights is what makes the element-by-element check
//      possible, so this is a NEW-mechanism test, not a characterization.
//
//   6. REFUSAL. A model that does not implement the boundary reports
//      packedSize() == 0 and throws from the other three rather than returning
//      something plausible. A wrong-sized parameter vector is refused by name.
//
// No captured literals: every assertion is a same-process relation.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "backprop.h"
#include "bareprop.h"
#include "dataset.h"
#include "logistic.h"
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

struct Hush {
	ostringstream sink;
	ostream& prev;
	Hush() : prev( util::screen() ) { util::set_screen( sink ); }
	~Hush() { util::set_screen( prev ); }
};

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

// The boundary is protected, as it must be -- it is not a public capability.
// A subclass reaches it, which is exactly how src/lbfgs.cpp's adapter reaches
// it too.
template < class NET >
class BoundaryProbe : public NET {
public:
	// The refusal type is protected, like the methods that throw it -- a
	//    subclass may name it, which is how the one real consumer catches it
	//    too. Re-exported here so this file's free functions can.
	typedef typename NET::NoPackedBoundary Refused;

	void prepare() { this->prepareRun(); }

	unsigned size() const { return this->packedSize(); }
	void pack( vector< double >& w ) const { this->packWeights( w ); }
	void unpack( const vector< double >& w ) { this->unpackWeights( w ); }
	double evaluate( vector< double >& g ) { return this->batchObjectiveGradient( g ); }

	// The optimizer state a pure evaluation must not touch
	const vector< double >& carriedG() const { return this->lastG; }
	const vector< double >& carriedF() const { return this->lastF; }
	double carriedGradMax() const { return this->currGradMax; }

	// The function the current weights compute, for the round-trip check
	double forwardAt( unsigned r ) { this->forward( this->Train, r ); return this->o; }
	unsigned trainRows() const { return this->Train.rows(); }
};

template < class NET >
static void configure( NET& net, bool decayOn )
{
	net.setHistory( false );
	net.setLastop( false );
	net.setLogPrint( false );
	net.setQuiet( true );
	net.setXEerror();
	net.setBatchEpoch( true );
	net.setAutoStepSize( false );
	net.setWeightDecay( decayOn );
	net.setDecay( 5e-5 );
	net.setEta( 0.05 );
	net.setTrainingType( 0 );
	net.setGradStop( true );
	net.setGradMaxLimit( 0.0 );
	net.setMinStop( false );
	net.setChangeStop( false );
	net.setWindowStop( false );
}

// Build one of each neural model on the same data, at the same seed.
static void buildSimple( BoundaryProbe< SimpleProp >& p, DataSet& d, bool decay )
{
	Hush quiet;
	p.setDataSet( d );
	p.setHidden( 3 );
	configure( p, decay );
	util::set_seed( 20260806 );
	p.randomize();
	p.prepare();
}

static void buildBare( BoundaryProbe< BareProp >& p, DataSet& d, bool decay )
{
	Hush quiet;
	p.setDataSet( d );
	p.setHidden( 3 );
	configure( p, decay );
	util::set_seed( 20260806 );
	p.randomize();
	p.prepare();
}

static void buildBack( BoundaryProbe< BackProp >& p, DataSet& d, bool decay )
{
	Hush quiet;
	vector< unsigned > layers( 1, 3 );
	p.setDataSet( d );
	p.setHidden( layers );
	configure( p, decay );
	util::set_seed( 20260806 );
	p.randomize();
	p.prepare();
}

// ---------------------------------------------------------------------------
// 1 + 2. Round trip and one layout.

template < class PROBE >
static void checkRoundTrip( PROBE& net, const string& label )
{
	expect( net.size() > 0, label + ": the boundary is available (packedSize > 0)" );

	vector< double > w0;
	net.pack( w0 );
	expect( w0.size() == net.size() && !w0.empty(),
		label + ": packWeights writes exactly packedSize doubles" );

	vector< double > g;
	double f = net.evaluate( g );
	expect( g.size() == w0.size(),
		label + ": the packed gradient has the packed weights' length" );
	expect( std::isfinite( f ), label + ": the objective is finite" );

	// Record the function BEFORE the round trip
	vector< double > before;
	for ( unsigned r = 0; r < net.trainRows(); r++ )
		before.push_back( net.forwardAt( r ) );

	// A PERTURBED round trip: unpack something different, then unpack the
	//    original back. An implementation that ignored its argument would
	//    survive an identity round trip and dies here.
	vector< double > w1 = w0;
	for ( unsigned i = 0; i < w1.size(); i++ )
		w1[ i ] += 0.25 + 0.01 * i;
	net.unpack( w1 );

	vector< double > w1back;
	net.pack( w1back );
	bool perturbedHeld = ( w1back.size() == w1.size() );
	for ( unsigned i = 0; perturbedHeld && i < w1.size(); i++ )
		perturbedHeld = ( w1back[ i ] == w1[ i ] );
	expect( perturbedHeld,
		label + ": unpackWeights installs the vector it was given, exactly" );

	bool moved = false;
	for ( unsigned r = 0; r < net.trainRows(); r++ )
		if ( net.forwardAt( r ) != before[ r ] ) moved = true;
	expect( moved,
		label + ": installing different weights changes what the model computes" );

	net.unpack( w0 );
	vector< double > w0back;
	net.pack( w0back );
	bool exact = ( w0back.size() == w0.size() );
	for ( unsigned i = 0; exact && i < w0.size(); i++ )
		exact = ( w0back[ i ] == w0[ i ] );
	expect( exact, label + ": the round trip restores every weight bit-exactly" );

	bool restored = true;
	for ( unsigned r = 0; r < net.trainRows(); r++ )
		if ( net.forwardAt( r ) != before[ r ] ) restored = false;
	expect( restored,
		label + ": the round trip restores the function the model computes" );
}

// ---------------------------------------------------------------------------
// 3 + 4. The same numbers as the legacy pass, and no training.

template < class PROBE >
static void checkAuthoritative( PROBE& legacy, PROBE& boundary, const string& label )
{
	// Two identically built models. The first takes one legacy training
	//    iteration; the second evaluates the boundary at the same weights.
	vector< double > w0;
	boundary.pack( w0 );

	double fLegacy = legacy.trainSet();   // returns f( w0 ), then updates
	double gradMaxLegacy = 0;             // recovered below from the boundary

	vector< double > g;
	double fBoundary = boundary.evaluate( g );

	expect( fLegacy == fBoundary,
		label + ": the boundary's objective is bit-identical to the legacy pass" );

	// The legacy pass's gradient is not directly readable for every model, so
	//    the comparison that IS available everywhere is the objective above
	//    plus the update the legacy pass performed: with trainingType 0 the
	//    engine leaves the gradient raw, so w0 - w1 = eta * g.
	vector< double > w1;
	{
		PROBE& l = legacy;
		l.pack( w1 );
	}
	double eta = legacy.getEta();
	bool sameGradient = ( w1.size() == g.size() );
	double worst = 0;
	for ( unsigned i = 0; sameGradient && i < g.size(); i++ )
	{
		double recovered = ( w0[ i ] - w1[ i ] ) / eta;
		worst = max( worst, fabs( recovered - g[ i ] ) );
		gradMaxLegacy = max( gradMaxLegacy, fabs( recovered ) );
	}
	// Recovering g through a subtraction and a division loses bits that the
	//    boundary's own value never had, so this is a tight relative bound
	//    rather than ==. The bit-identical claim is carried by the objective
	//    above and by the goldens.
	expect( sameGradient && worst <= 1e-10 * max( 1.0, gradMaxLegacy ),
		label + ": the boundary's gradient is the one the legacy update consumed" );

	// It did real work ...
	double gmax = 0;
	for ( unsigned i = 0; i < g.size(); i++ )
		gmax = max( gmax, fabs( g[ i ] ) );
	expect( gmax > 1e-8, label + ": the evaluation produced a non-zero gradient" );

	// ... and it trained nothing.
	vector< double > lastG = boundary.carriedG(), lastF = boundary.carriedF();
	double gradMaxBefore = boundary.carriedGradMax();
	unsigned iterBefore = boundary.getIterations();
	double etaBefore = boundary.getEta();

	vector< double > w2;
	boundary.pack( w2 );
	bool weightsHeld = ( w2.size() == w0.size() );
	for ( unsigned i = 0; weightsHeld && i < w0.size(); i++ )
		weightsHeld = ( w2[ i ] == w0[ i ] );
	expect( weightsHeld, label + ": the evaluation left the weights untouched" );

	vector< double > g2;
	double f2 = boundary.evaluate( g2 );
	expect( f2 == fBoundary,
		label + ": a second evaluation at the same weights returns the same objective" );
	bool gHeld = ( g2.size() == g.size() );
	for ( unsigned i = 0; gHeld && i < g.size(); i++ )
		gHeld = ( g2[ i ] == g[ i ] );
	expect( gHeld,
		label + ": a second evaluation at the same weights returns the same gradient" );

	expect( boundary.carriedG() == lastG && boundary.carriedF() == lastF,
		label + ": the evaluation left the optimizer's carried state untouched" );
	expect( boundary.carriedGradMax() == gradMaxBefore,
		label + ": the evaluation left currGradMax untouched" );
	expect( boundary.getIterations() == iterBefore,
		label + ": the evaluation did not advance the iteration counter" );
	expect( boundary.getEta() == etaBefore,
		label + ": the evaluation did not touch eta" );
}

// ---------------------------------------------------------------------------
// 5. The gradient is the gradient of the objective, element by element.

template < class PROBE >
static void checkFiniteDifference( PROBE& net, const string& label )
{
	vector< double > w0;
	net.pack( w0 );

	vector< double > g;
	net.evaluate( g );

	const double h = 1e-5;
	double worst = 0;
	unsigned worstAt = 0;
	vector< double > scratch;
	for ( unsigned i = 0; i < w0.size(); i++ )
	{
		vector< double > w = w0;
		w[ i ] = w0[ i ] + h;
		net.unpack( w );
		double fp = net.evaluate( scratch );

		w[ i ] = w0[ i ] - h;
		net.unpack( w );
		double fm = net.evaluate( scratch );

		double fd = ( fp - fm ) / ( 2 * h );
		double tol = 1e-6 + 1e-5 * fabs( g[ i ] );
		double err = fabs( fd - g[ i ] );
		if ( err > worst ) { worst = err; worstAt = i; }
		if ( err > tol )
		{
			cout << "         component " << i << ": analytic "
				<< setprecision( 12 ) << g[ i ] << ", finite difference "
				<< fd << endl;
			expect( false, label + ": gradient component agrees with the objective" );
			net.unpack( w0 );
			return;
		}
	}
	net.unpack( w0 );

	ostringstream s;
	s << label << ": all " << w0.size()
		<< " gradient components agree with a central difference (worst "
		<< setprecision( 3 ) << worst << " at " << worstAt << ")";
	expect( true, s.str() );
}

// ---------------------------------------------------------------------------
// 6. Refusal.

static void checkRefusals()
{
	DataSet d = makeData( 60 );

	// Logistic deliberately does not implement the boundary.
	BoundaryProbe< Logistic > lg;
	{
		Hush quiet;
		lg.setDataSet( d );
	}
	expect( lg.size() == 0,
		"Logistic reports packedSize 0: the boundary is unavailable" );

	bool threwPack = false, threwUnpack = false, threwEval = false;
	vector< double > v( 3, 0.0 );
	try { lg.pack( v ); } catch ( BoundaryProbe< Logistic >::Refused& ) { threwPack = true; }
	try { lg.unpack( v ); } catch ( BoundaryProbe< Logistic >::Refused& ) { threwUnpack = true; }
	try { lg.evaluate( v ); } catch ( BoundaryProbe< Logistic >::Refused& ) { threwEval = true; }
	expect( threwPack && threwUnpack && threwEval,
		"a model without the boundary refuses all three calls rather than "
		"returning a plausible value" );

	// A wrong-sized parameter vector is refused, on a model that HAS the
	//    boundary -- so this tests the size check, not the availability check.
	BoundaryProbe< SimpleProp > sp;
	buildSimple( sp, d, true );

	vector< double > w;
	sp.pack( w );

	bool threwShort = false, threwLong = false;
	vector< double > shortV( w.begin(), w.end() - 1 );
	vector< double > longV = w;
	longV.push_back( 0.0 );
	try { sp.unpack( shortV ); } catch ( nvec::SizeMismatch& ) { threwShort = true; }
	try { sp.unpack( longV ); } catch ( nvec::SizeMismatch& ) { threwLong = true; }
	expect( threwShort && threwLong,
		"unpackWeights refuses a parameter vector of the wrong length" );

	// ... and refusing left the model alone
	vector< double > after;
	sp.pack( after );
	bool held = ( after.size() == w.size() );
	for ( unsigned i = 0; held && i < w.size(); i++ )
		held = ( after[ i ] == w[ i ] );
	expect( held, "a refused unpackWeights installed nothing" );
}

int main()
{
	cout << "packed parameter boundary" << endl;

	DataSet d = makeData( 60 );

	{
		BoundaryProbe< SimpleProp > p; buildSimple( p, d, true );
		checkRoundTrip( p, "SimpleProp" );
		checkFiniteDifference( p, "SimpleProp decay" );
	}
	{
		BoundaryProbe< BareProp > p; buildBare( p, d, true );
		checkRoundTrip( p, "BareProp" );
		checkFiniteDifference( p, "BareProp decay" );
	}
	{
		BoundaryProbe< BackProp > p; buildBack( p, d, true );
		checkRoundTrip( p, "BackProp" );
		checkFiniteDifference( p, "BackProp decay" );
	}
	{
		BoundaryProbe< BackProp > p; buildBack( p, d, false );
		checkFiniteDifference( p, "BackProp no decay" );
	}

	{
		BoundaryProbe< SimpleProp > a, b;
		buildSimple( a, d, true ); buildSimple( b, d, true );
		checkAuthoritative( a, b, "SimpleProp" );
	}
	{
		BoundaryProbe< BareProp > a, b;
		buildBare( a, d, true ); buildBare( b, d, true );
		checkAuthoritative( a, b, "BareProp" );
	}
	{
		BoundaryProbe< BackProp > a, b;
		buildBack( a, d, true ); buildBack( b, d, true );
		checkAuthoritative( a, b, "BackProp" );
	}

	checkRefusals();

	cout << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
