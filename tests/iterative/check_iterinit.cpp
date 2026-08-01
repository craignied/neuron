// check_iterinit.cpp : a fresh iterative model has a DEFINED iteration count.
//
// Iterative::Iterative()'s initialiser list set nineteen members and did not set
// `iteration`. It is an unsigned, so until train() assigned it, it held
// indeterminate memory.
//
// THIS WAS NOT A TRAINING REGRESSION, and the commit that fixes it should not be
// described as one. The only caller of trainSet() in src/ sits inside train()'s
//
//     for ( iteration = 0; iteration <= maxIterations; iteration++ )
//
// so the normal path always assigned the counter before anything read it, and
// remains byte-identical. What was reachable is the PUBLIC object contract:
//
//   * getIterations() on a model that has not trained returned whatever was in
//     that memory. RegressNet reads it for its audit trail.
//   * Iterative::copy() propagates it, so copy-constructing or assigning from a
//     fresh model copied indeterminate memory into a second object -- the same
//     pattern as the settled "'not copied' must be WRITTEN in copy()" decision,
//     and as the errorType scalar that turned out to be the real cause behind
//     the reverted Matrix value-initialisation misdiagnosis.
//   * a direct trainSet() call -- which the class layer permits -- read it.
//     Network::engine() branches on `t == 0 || t == df()`, so an indeterminate
//     value takes the CONJUGATE-DIRECTION branch on the very first call, where
//     lastF and lastG are still empty, and dotprod() then reads df() doubles out
//     of an empty vector.
//
// That last one is how this was found at all: since the D5 bounds work it throws
// nvec::SizeMismatch instead of quietly reading rubbish. The fix is to define the
// member, never to weaken that check.
//
// train() STILL OWNS ITERATION PROGRESSION. trainSet() does not increment
// anything, and a direct caller that wants a later optimizer step must still
// establish it explicitly, as tests/onehidden and tests/network/check_autostep
// do. Zero-initialisation makes the STARTING value defined; it does not make the
// counter self-advancing.
//
// PROVING IT. A normal build may accidentally observe zero and pass against the
// old constructor, so the sabotage evidence is a build with
// -ftrivial-auto-var-init=pattern, which fills automatic storage -- including a
// stack-constructed model's un-initialised members -- with a recognisable
// pattern. That made the earlier uninitialised-state defect deterministic too.
// (Not ASan: it does not work in this environment, see CLAUDE.md rule 2.)

#include <iostream>
#include <string>
#include <vector>

#include "iterative.h"
#include "simpleprop.h"
#include "dataset.h"
#include "utility.h"

using namespace std;

int failures = 0;

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

// A minimal Iterative: a scripted error trace and nothing else.
class Scripted : public Iterative {
public:
	Scripted() : calls ( 0 )
	{
		setHistory( false );
		setLastop( false );
		setQuiet( true );
		theData.setDiscrete( false );
		setMaxIterations( 4 );
		setGradStop( false );
	}
	double trainSet() override { return 1.0 / ++calls; }
	double getGradMax() override { return 1.0; }
	void setDataSet( DataSet& ) override { }
	void outputHeader( ostream& ) override { }
	void reportAccuracy( ostream& ) override { }
	void classAccuracy( ostream& ) override { }
protected:
	void runHeader( ostream& ) override { }
private:
	unsigned calls;
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
	util::ScreenCapture hush;
	d.setRawMatrix( raw );
	d.setTrainMatrix( raw );
	return d;
}

// --- 1, 2, 3. a fresh object, and fresh copies of it ----------------------

static void test_fresh_is_zero()
{
	cout << "-- a fresh model --" << endl;

	Scripted fresh;
	expect( fresh.getIterations() == 0,
		"a freshly constructed model reports 0 iterations" );

	Scripted copied( fresh );
	expect( copied.getIterations() == 0,
		"copy-constructing a fresh model reports 0" );

	Scripted assigned;
	assigned = fresh;
	expect( assigned.getIterations() == 0,
		"assigning from a fresh model reports 0" );

	// A real network too, not only the double: the member is on Iterative, but
	// a concrete model is what a caller actually holds.
	DataSet d = makeData( 60 );
	SimpleProp net;
	{
		util::ScreenCapture hush;
		net.setDataSet( d );
		net.setHidden( 3 );
	}
	expect( net.getIterations() == 0,
		"a freshly built SimpleProp reports 0 iterations" );

	SimpleProp netCopy( net );
	expect( netCopy.getIterations() == 0,
		"...and so does a copy of it" );
}

// --- 4. a trained model's count survives copying --------------------------

static void test_trained_count_is_carried()
{
	cout << "-- a trained model --" << endl;

	Scripted n;
	{
		util::ScreenCapture hush;
		n.train();
	}
	unsigned ran = n.getIterations();

	// The ceiling is 4 and nothing else can stop this double, so train() ends
	// having gone one past it -- whatever that is, a copy must agree.
	expect( ran > 0, "a trained model reports a non-zero iteration count" );

	Scripted copied( n );
	expect( copied.getIterations() == ran,
		"copy construction carries the trained iteration count" );

	Scripted assigned;
	assigned = n;
	expect( assigned.getIterations() == ran,
		"assignment carries the trained iteration count" );
}

// --- 5. a first DIRECT trainSet() is Golden's step 1 -----------------------
//
// Network::engine() treats t == 0 as "start a fresh search direction". With the
// counter defined, a direct first call takes that branch and never reads the
// empty lastF/lastG. This is the case that threw.

static void test_first_direct_trainset()
{
	cout << "-- a first direct trainSet(), CGD and Shanno --" << endl;

	for ( unsigned type = 1; type <= 2; type++ )
	{
		const char* name = ( type == 1 ) ? "CGD" : "Shanno";

		DataSet d = makeData( 60 );
		SimpleProp net;
		bool threw = false;
		double err = 0;
		{
			util::ScreenCapture hush;
			net.setDataSet( d );
			net.setHidden( 3 );
			net.setHistory( false );
			net.setLastop( false );
			net.setQuiet( true );
			net.setBatchEpoch( true );
			net.setTrainingType( type );
			net.setEta( 0.1 );
			util::set_seed( 7 );
			net.randomize();
			try { err = net.trainSet(); }
			catch ( ... ) { threw = true; }
		}
		expect( !threw, string( "a first direct trainSet() under " ) + name
			+ " does not read empty optimizer history" );
		expect( err == err, string( "...and returns a number (" ) + name + ")" );
	}
}

// --- 6. the ordinary train() path is untouched ----------------------------

static void test_train_path_unchanged()
{
	cout << "-- the ordinary train() path --" << endl;

	// train() assigns the counter before the first trainSet(), so it never
	// depended on the constructor. Two runs of the same configuration must
	// agree exactly -- and the goldens cover the byte-level statement.
	Scripted a, b;
	{
		util::ScreenCapture hush;
		a.train();
		b.train();
	}
	expect( a.getIterations() == b.getIterations(),
		"train() reports the same iteration count for identical runs" );
	expect( a.getStopReason() == b.getStopReason(),
		"...and stops for the same reason" );
}

int main()
{
	test_fresh_is_zero();
	test_trained_count_is_carried();
	test_first_direct_trainset();
	test_train_path_unchanged();

	cout << endl << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
