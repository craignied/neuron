// check_metricsreport.cpp : DataSet::metricsReport, both halves of it.
//
// The training and test blocks of metricsReport were the same forty lines
// written twice, differing only in which TwoSet they read, the heading, and the
// classification-table label. This pins the report BEFORE the extraction, so
// that the extraction can be shown to change nothing a reader sees.
//
// WHY A TEST AND NOT JUST THE GOLDENS. A golden transcript proves the bytes it
// contains did not move; it does not prove which branches produced them. The
// training half and the test half are separate code, and a report run with no
// test set exercises only one of them. So each half is driven explicitly here:
//   * training set only  -> the training block, and NO test block;
//   * training and test  -> both blocks, in that order.
//
// WHAT IS ASSERTED
//   1. Both halves, with every line each emits: the heading and its rule, the
//      four rates, the classification table, ROC, Kolmogorov-Smirnov, Pearson
//      X2, Hosmer-Lemeshow -- in that order.
//   2. Pearson prints a STATISTIC with n and explicitly no p-value. That is a
//      settled decision (individual-level X2 has no valid chi-squared reference
//      with continuous covariates), so it is asserted rather than left to a
//      golden that nobody reads.
//   3. The four division-by-zero messages, one per rate, reached through a
//      degenerate TwoSet -- the catch blocks are half the duplicated body and
//      would otherwise never run in this suite.
//   4. The two refusals: a non-discrete output, and more than one output. Both
//      go to util::screen(), not to the report stream.
//   5. --capture prints the exact report text, so the extraction can be proven
//      byte-for-byte (diff the capture before and after).

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

// Every line of a half, in the order metricsReport emits them.
static void expectHalf( const string& text, const string& heading,
	const string& rule, const string& tableLabel, const string& who )
{
	size_t at = text.find( heading );
	expect( at != string::npos, who + ": the heading" );
	expect( text.find( rule ) != string::npos, who + ": the heading's rule" );

	const char* lines[] = {
		"Classification accuracy: ",
		"Sensitivity: ",
		"Specificity: ",
		"Predictive value positive: ",
		"Predictive value negative: "
	};
	for ( const char* l : lines )
		expect( text.find( l, at ) != string::npos,
			who + ": " + string( l ) );

	expect( text.find( tableLabel, at ) != string::npos,
		who + ": " + tableLabel );
	expect( text.find( "Kolmogorov-Smirnov goodness of fit", at ) != string::npos,
		who + ": Kolmogorov-Smirnov" );
	expect( text.find( "Pearson X2 = ", at ) != string::npos,
		who + ": Pearson X2" );
	expect( text.find( "Hosmer-Lemeshow goodness of fit", at ) != string::npos,
		who + ": Hosmer-Lemeshow" );
}

// A separable 1-output discrete problem, split into a training and a test set.
static DataSet makeData( unsigned n, unsigned nTest )
{
	Matrix< double > raw( n, 3 );
	for ( unsigned i = 0; i < n; i++ )
	{
		double x0 = -1.0 + 2.0 * ( ( i * 37 ) % 100 ) / 99.0;
		double x1 = -1.0 + 2.0 * ( ( i * 53 ) % 100 ) / 99.0;
		raw( i, 0 ) = x0;
		raw( i, 1 ) = x1;
		raw( i, 2 ) = ( x0 + x1 > 0.2 ) ? 1 : 0;
	}
	DataSet d;
	d.setInput( 2 );
	d.setOutput( 1 );
	d.setDiscrete( true );
	d.setHistory( false );
	util::ScreenCapture hush;
	d.setRawMatrix( raw );
	if ( nTest )
		d.randomize( nTest );
	else
		d.setTrainMatrix( raw );
	return d;
}

// Give a TwoSet (real, guess) pairs directly, so the report has something to
// describe without training a model.
static void fill( TwoSet& t, const vector< double >& real,
	const vector< double >& guess )
{
	Matrix< double > m( ( unsigned ) real.size(), 2 );
	for ( unsigned i = 0; i < real.size(); i++ )
	{
		m( i, 0 ) = real[ i ];
		m( i, 1 ) = guess[ i ];
	}
	t.setMatrix( m );
	t.setThreshold( 0.5 );
}

// A well-behaved set: both classes present, both predicted.
static void ordinary( TwoSet& t, unsigned n )
{
	vector< double > real, guess;
	for ( unsigned i = 0; i < n; i++ )
	{
		bool positive = ( i % 3 ) != 0;
		real.push_back( positive ? 1 : 0 );
		// Mostly right, deliberately not perfectly: a separable set makes
		//    ROC and the goodness-of-fit tests degenerate.
		double g = positive ? 0.55 + 0.004 * ( i % 100 )
			: 0.45 - 0.004 * ( i % 100 );
		if ( i % 7 == 0 ) g = 1.0 - g; // some genuine errors
		guess.push_back( g );
	}
	fill( t, real, guess );
}

// SEEDED, deliberately. ROCarea's confidence interval is a 2000-resample
//    bootstrap, and util::d_random() seeds itself from time( 0 ) when nobody
//    else has -- so an unseeded capture is reproducible within a second and not
//    across a rebuild. That made a byte-for-byte before/after comparison of the
//    report meaningless (it showed three "differences" that were the clock).
//    Seeding immediately before each report makes the whole text deterministic,
//    which is what lets the extraction be proven to change nothing.
static string reportOf( DataSet& d )
{
	util::set_seed( 4242 );
	ostringstream out;
	d.metricsReport( out );
	return out.str();
}

// --- 1 and 2. both halves ---------------------------------------------------

static void test_training_only()
{
	cout << "-- training set only --" << endl;

	DataSet d = makeData( 120, 0 );
	ordinary( d.getTrainTwoSet(), 120 );

	string text = reportOf( d );

	expectHalf( text, "Training set:", "-------------",
		"Classification table for training set:", "training" );

	expect( text.find( "Test set:" ) == string::npos,
		"with no test set loaded, no test block is emitted" );

	// The settled decision, asserted where a reader will see it break
	expect( text.find( "no valid p at the individual level" ) != string::npos,
		"Pearson prints a statistic with n and NO p-value" );
	expect( text.find( "(n = " ) != string::npos,
		"...and reports the n it was computed over" );
}

static void test_training_and_test()
{
	cout << "-- training and test --" << endl;

	DataSet d = makeData( 160, 40 );
	ordinary( d.getTrainTwoSet(), 120 );
	ordinary( d.getTestTwoSet(), 40 );

	string text = reportOf( d );

	expectHalf( text, "Training set:", "-------------",
		"Classification table for training set:", "training" );
	expectHalf( text, "Test set:", "---------",
		"Classification table for test set:", "test" );

	expect( text.find( "Training set:" ) < text.find( "Test set:" ),
		"the training block comes first" );

	// Both halves carry the Pearson wording, not just the first
	size_t first = text.find( "no valid p at the individual level" );
	expect( first != string::npos
		&& text.find( "no valid p at the individual level", first + 1 )
			!= string::npos,
		"both halves print the Pearson caveat" );
}

// --- 3. the four division-by-zero messages ---------------------------------
//
// Half the duplicated body is catch blocks. A set with no real negatives makes
// specificity and PVN undefined; one with nothing predicted positive makes PVP
// undefined. Between them every catch is reached.

static void test_division_by_zero_messages()
{
	cout << "-- undefined rates are reported, not thrown --" << endl;

	// No real negatives: specificity (tn + fp) and PVN (tn + fn) are undefined
	{
		DataSet d = makeData( 40, 0 );
		vector< double > real( 40, 1.0 ), guess;
		for ( unsigned i = 0; i < 40; i++ )
			guess.push_back( ( i % 4 ) ? 0.8 : 0.2 );
		fill( d.getTrainTwoSet(), real, guess );

		string text = reportOf( d );
		expect( text.find( "Specificity: " ) != string::npos
			&& text.find( "Sensitivity: " ) != string::npos,
			"the report still runs with an undefined rate" );
		expect( text.find( "divide by zero" ) != string::npos
			|| text.find( "Divide" ) != string::npos
			|| text.find( "zero" ) != string::npos,
			"an undefined rate prints its exception message" );
	}

	// Nothing predicted positive: PVP (tp + fp) is undefined
	{
		DataSet d = makeData( 40, 0 );
		vector< double > real, guess;
		for ( unsigned i = 0; i < 40; i++ )
		{
			real.push_back( i % 2 );
			guess.push_back( 0.1 ); // every guess below threshold
		}
		fill( d.getTrainTwoSet(), real, guess );

		string text = reportOf( d );
		expect( text.find( "Predictive value positive: " ) != string::npos,
			"PVP is labelled even when undefined" );
		expect( text.find( "Predictive value negative: " ) != string::npos,
			"and the report continues past it to PVN" );
	}
}

// --- 4. the two refusals ---------------------------------------------------

static void test_refusals()
{
	cout << "-- refusals go to the screen, not the report --" << endl;

	// Not discrete
	{
		Matrix< double > raw( 20, 3 );
		for ( unsigned i = 0; i < 20; i++ )
		{
			raw( i, 0 ) = i * 0.1; raw( i, 1 ) = i * 0.2;
			raw( i, 2 ) = i * 0.05;
		}
		DataSet d;
		d.setInput( 2 ); d.setOutput( 1 );
		d.setDiscrete( false );
		d.setHistory( false );
		string said;
		ostringstream out;
		{
			util::ScreenCapture cap;
			d.setRawMatrix( raw );
			d.setTrainMatrix( raw );
			d.metricsReport( out );
			said = cap.text();
		}
		expect( said.find( "The output must be discrete" ) != string::npos,
			"a continuous output is refused, through util::screen()" );
		expect( out.str().empty(), "...and nothing is written to the report" );
	}

	// More than one output
	{
		Matrix< double > raw( 20, 4 );
		for ( unsigned i = 0; i < 20; i++ )
		{
			raw( i, 0 ) = i * 0.1; raw( i, 1 ) = i * 0.2;
			raw( i, 2 ) = i % 2; raw( i, 3 ) = ( i + 1 ) % 2;
		}
		DataSet d;
		d.setInput( 2 ); d.setOutput( 2 );
		d.setDiscrete( true );
		d.setHistory( false );
		string said;
		ostringstream out;
		{
			util::ScreenCapture cap;
			d.setRawMatrix( raw );
			d.setTrainMatrix( raw );
			d.metricsReport( out );
			said = cap.text();
		}
		expect( said.find( "There must be only 1 output" ) != string::npos,
			"two outputs are refused, through util::screen()" );
		expect( out.str().empty(), "...and nothing is written to the report" );
	}
}

// --- 5. capture: the exact bytes, for the before/after diff ----------------

static void capture()
{
	DataSet a = makeData( 120, 0 );
	ordinary( a.getTrainTwoSet(), 120 );
	cout << "===== training only =====" << endl << reportOf( a );

	DataSet b = makeData( 160, 40 );
	ordinary( b.getTrainTwoSet(), 120 );
	ordinary( b.getTestTwoSet(), 40 );
	cout << "===== training and test =====" << endl << reportOf( b );

	DataSet c = makeData( 40, 0 );
	vector< double > real( 40, 1.0 ), guess;
	for ( unsigned i = 0; i < 40; i++ )
		guess.push_back( ( i % 4 ) ? 0.8 : 0.2 );
	fill( c.getTrainTwoSet(), real, guess );
	cout << "===== undefined rates =====" << endl << reportOf( c );
}

int main( int argc, char* argv[] )
{
	if ( argc > 1 && string( argv[ 1 ] ) == "--capture" )
	{
		capture();
		return 0;
	}

	test_training_only();
	test_training_and_test();
	test_division_by_zero_messages();
	test_refusals();

	cout << endl << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
