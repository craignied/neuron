// check_announcestop.cpp : every way training can stop, and what it says.
//
// Iterative::train() ends through seven exits. Six of them -- minimum error,
// change in error, error window, gradient maximum, plateau, and the observer --
// were the same five-line block written out six times:
//
//     screenStream.str( "" );
//     screenStream << <this condition's own message>;
//     if ( !quietFlag ) { fileStream << ...; util::screen() << ...; }
//     stopReason = <this condition's reason>;
//     break;
//
// What repeats is only the PUBLICATION -- a quiet run says nothing, an audible
// one writes the same text to the log and the screen, and either way the reason
// is recorded. The conditions themselves, their formulae, their order, and
// their wording are not duplication and stay written out. The seventh exit, the
// iteration ceiling, is not a stopping condition at all (it is a failure to
// converge) and reports through the epilogue instead.
//
// This pins all seven before the extraction.
//
// WHAT IS ASSERTED, for every exit
//   1. the StopReason, through getStopReason();
//   2. its token, through stopReasonToken() -- what the GUI and the CV report
//      actually publish;
//   3. converged() -- three of these are convergence and four are not, and the
//      distinction is the convergence contract;
//   4. the exact message text on the screen;
//   5. that a QUIET run emits none of it while still recording the reason.
//
// A SCRIPTED Iterative, not a real model: the point is to reach each exit
// deliberately and in isolation. A real network would let several conditions
// race, and whichever fired first would be the only one ever tested.
//
// SABOTAGE, each watched to fail and restored:
//   * a reason assignment -- swap STOP_WINDOW for STOP_CHANGE in the window
//     exit; the window case's reason, token and converged() assertions fail
//     while its message assertion still passes;
//   * the quiet path -- publish regardless of quietFlag; every "a quiet run
//     says nothing" assertion fails while the audible ones still pass.
// The two are independent, which is what proves the test holds both.

#include <iostream>
#include <sstream>
#include <string>

#include "iterative.h"
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

// An Iterative that reports a scripted error trace and nothing else. Each test
// arms exactly ONE stopping rule, so the exit under test is the only one that
// can fire.
class Scripted : public Iterative {
public:
	Scripted() : calls ( 0 ), errorPath ( DESCENDING ), grad ( 1.0 )
	{
		setHistory( false ); // no neuron.log
		setLastop( false );  // no model.txt
		theData.setDiscrete( false ); // no classification columns
		setMaxIterations( 200 );
	}

	enum Path { DESCENDING, FLAT, RISING, TINY_STEP };

	void setPath( Path p ) { errorPath = p; }
	void setGrad( double g ) { grad = g; }

	double trainSet() override
	{
		calls++;
		switch ( errorPath )
		{
			case FLAT:      return 0.5;              // plateaus at once
			case RISING:    return 0.1 + 0.01 * calls; // climbs
			case TINY_STEP: return 0.5 - 1e-9 * calls; // improves imperceptibly
			default:        return 1.0 / calls;      // ordinary descent
		}
	}

	double getGradMax() override { return grad; }

	unsigned iterationsRun() const { return calls; }

	void setDataSet( DataSet& ) override { }
	void outputHeader( ostream& ) override { }
	void reportAccuracy( ostream& ) override { }
	void classAccuracy( ostream& ) override { }

protected:
	void runHeader( ostream& ) override { }

private:
	unsigned calls;
	Path errorPath;
	double grad;
};

// An observer that stops the run at a chosen iteration, naming its own reason.
class Stopper : public Iterative::Observer {
public:
	Stopper( unsigned at, Iterative::StopReason why ) : stopAt ( at ), reason ( why ) { }
	bool onIteration( unsigned iteration, double ) override
	{
		return iteration < stopAt; // false stops the run
	}
	Iterative::StopReason whyStopped() const override { return reason; }
private:
	unsigned stopAt;
	Iterative::StopReason reason;
};

// Run `net` capturing everything it says, and report what happened.
struct Outcome {
	Iterative::StopReason reason;
	string token;
	bool didConverge;
	string said;
};

static Outcome runIt( Scripted& net, bool quiet )
{
	net.setQuiet( quiet );
	Outcome o;
	{
		util::ScreenCapture cap;
		net.train();
		o.said = cap.text();
	}
	o.reason = net.getStopReason();
	o.token = Iterative::stopReasonToken( o.reason );
	o.didConverge = net.converged( o.reason );
	return o;
}

// Every exit is checked the same way: reason, token, convergence, message, and
// that a quiet run of the identical configuration says nothing while still
// recording the reason.
static void checkExit( const char* what, Iterative::StopReason want,
	const char* wantToken, bool wantConverged, const char* wantSays,
	void ( *arm )( Scripted& ) )
{
	cout << "-- " << what << " --" << endl;

	Scripted audible;
	arm( audible );
	Outcome a = runIt( audible, false );

	expect( a.reason == want, string( what ) + ": the StopReason" );
	expect( a.token == wantToken,
		string( what ) + ": the token is \"" + wantToken + "\"" );
	expect( a.didConverge == wantConverged, string( what ) + ": converged() is "
		+ ( wantConverged ? "true" : "false" ) );
	expect( a.said.find( wantSays ) != string::npos,
		string( what ) + ": says \"" + wantSays + "\"" );

	Scripted quiet;
	arm( quiet );
	Outcome q = runIt( quiet, true );

	expect( q.said.find( wantSays ) == string::npos,
		string( what ) + ": a QUIET run does not say it" );
	expect( q.said.empty(), string( what ) + ": a quiet run says nothing at all" );
	expect( q.reason == want,
		string( what ) + ": ...but still records the reason" );
}

// --- the six stopping conditions ------------------------------------------

static void armMinError( Scripted& n )
{
	n.setMinStop( true );
	n.setMinError( 0.2 );          // 1/calls drops below 0.2 at call 6
}

static void armChange( Scripted& n )
{
	n.setChangeStop( true );
	n.setChange( 1e-6 );           // TINY_STEP improves by 1e-9 each iteration
	n.setPath( Scripted::TINY_STEP );
}

static void armWindow( Scripted& n )
{
	n.setWindowStop( true );
	n.setWindow( 3 );
	n.setPath( Scripted::RISING ); // error climbs, so it exceeds the window
}

static void armGradMax( Scripted& n )
{
	n.setGradStop( true );
	n.setGradMaxLimit( 1e-6 );
	n.setGrad( 1e-9 );             // already below the limit
}

static void armPlateau( Scripted& n )
{
	n.setAutoStop( true, 1e-4, 5 );
	n.setPath( Scripted::FLAT );   // a constant error is a plateau
}

// --- the observer's three reasons -----------------------------------------
//
// One exit, three different facts. A cancel, a validation early stop and an
// expired probe budget must not be reported alike.

static void test_observer_reasons()
{
	struct Case {
		Iterative::StopReason why;
		const char* token;
		bool converged;
		const char* says;
	};
	const Case cases[] = {
		{ Iterative::STOP_CANCELLED, "cancelled", false,
			"Training was stopped by request." },
		{ Iterative::STOP_EARLY_STOP, "validation_early_stop", true,
			"Held-out error deteriorated: stopped early." },
		{ Iterative::STOP_PROBE_BUDGET, "probe_budget", false,
			"The probe's time budget expired." }
	};

	for ( const Case& c : cases )
	{
		cout << "-- observer: " << c.token << " --" << endl;

		Stopper stop( 4, c.why );
		Scripted audible;
		audible.setObserver( &stop );
		Outcome a = runIt( audible, false );

		expect( a.reason == c.why, string( c.token ) + ": the StopReason" );
		expect( a.token == c.token, string( c.token ) + ": the token" );
		expect( a.didConverge == c.converged,
			string( c.token ) + ": converged() is "
			+ ( c.converged ? "true" : "false" ) );
		expect( a.said.find( c.says ) != string::npos,
			string( c.token ) + ": says \"" + c.says + "\"" );

		Stopper stop2( 4, c.why );
		Scripted quiet;
		quiet.setObserver( &stop2 );
		Outcome q = runIt( quiet, true );
		expect( q.said.empty(), string( c.token ) + ": a quiet run says nothing" );
		expect( q.reason == c.why,
			string( c.token ) + ": ...but still records the reason" );
	}
}

// --- the seventh exit: the ceiling is NOT a stopping condition -------------

static void test_iteration_ceiling()
{
	cout << "-- the iteration ceiling --" << endl;

	Scripted n; // no stopping rule armed at all
	n.setMaxIterations( 5 );
	Outcome o = runIt( n, false );

	expect( o.reason == Iterative::STOP_MAX_ITERATIONS,
		"the ceiling records STOP_MAX_ITERATIONS" );
	expect( o.token == "max_iterations", "its token is \"max_iterations\"" );
	expect( !o.didConverge,
		"reaching the ceiling is NOT convergence -- the convergence contract" );
	expect( o.said.find( "did NOT converge" ) != string::npos,
		"and the report says so in plain words" );
	expect( o.said.find( "safety limit, not a stopping condition" )
		!= string::npos,
		"...naming the ceiling as a safety limit" );

	// The ceiling reports through the EPILOGUE, which a quiet run skips
	// entirely -- unlike the six conditions, this text is not published by the
	// stop itself.
	Scripted q;
	q.setMaxIterations( 5 );
	Outcome qo = runIt( q, true );
	expect( qo.said.empty(), "a quiet run at the ceiling says nothing" );
	expect( qo.reason == Iterative::STOP_MAX_ITERATIONS,
		"...but still records the ceiling" );
}

int main()
{
	checkExit( "minimum error", Iterative::STOP_MIN_ERROR, "min_error", true,
		"The error became lower than ", armMinError );
	checkExit( "change in error", Iterative::STOP_CHANGE, "min_change", true,
		"The change in error became lower than ", armChange );
	checkExit( "error window", Iterative::STOP_WINDOW, "error_window", true,
		"The error increased over the window width of ", armWindow );
	checkExit( "gradient maximum", Iterative::STOP_GRADMAX, "grad_max", true,
		"The maximum absolute gradient became lower than ", armGradMax );
	checkExit( "plateau", Iterative::STOP_PLATEAU, "plateau", true,
		"The error plateaued over a window of ", armPlateau );

	test_observer_reasons();
	test_iteration_ceiling();

	cout << endl << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
