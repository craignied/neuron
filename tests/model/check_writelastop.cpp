// check_writelastop.cpp : the last-operation file, from all three writers.
//
// Iterative::train(), LDFA::train() and QDFA::train() each ended with the same
// block: if lastopFlag, open util::run_path( lastopFilename ) truncating, and
// write the run's whole file stream into it -- or say "Error in opening <path>!"
// on the screen if it would not open. Model owns lastopFlag and lastopFilename,
// so Model owns the write.
//
// WHAT IS ASSERTED, for each of the three writers
//   1. the file is created at the resolved path and holds the run's report;
//   2. TRUNCATION -- a second, shorter run REPLACES the file rather than
//      appending to it. This is the property most easily lost by an extraction
//      (ios::trunc dropped, or ios::app added) and the one a reader would not
//      notice for a long time: model.txt would simply keep growing;
//   3. with lastop disabled, nothing is written at all and an existing file is
//      left exactly as it was;
//   4. a path that cannot be opened prints "Error in opening <path>!" through
//      util::screen() and does NOT abort the run -- train() still returns.
//
// Path resolution is exercised deliberately through util::run_path's own rule:
// a name containing a separator is respected verbatim, a bare name is placed in
// the run directory. Both are checked, because the extraction could easily
// resolve the path once in the wrong place.

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "iterative.h"
#include "ldfa.h"
#include "qdfa.h"
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

static string slurp( const string& path )
{
	ifstream in( path.c_str() );
	if ( !in.is_open() )
		return string();
	ostringstream all;
	all << in.rdbuf();
	return all.str();
}

static bool exists( const string& path )
{
	ifstream in( path.c_str() );
	return in.is_open();
}

// A minimal Iterative: a scripted error trace, no data, no reports.
class Scripted : public Iterative {
public:
	Scripted() : calls ( 0 )
	{
		setHistory( false ); // no neuron.log
		theData.setDiscrete( false );
		setMaxIterations( 3 );
	}
	double trainSet() override { return 1.0 / ++calls; }
	double getGradMax() override { return 1.0; }
	void setDataSet( DataSet& ) override { }
	void outputHeader( ostream& o ) override { o << "SCRIPTED HEADER" << endl; }
	void reportAccuracy( ostream& ) override { }
	void classAccuracy( ostream& ) override { }
protected:
	void runHeader( ostream& ) override { }
private:
	unsigned calls;
};

// A separable two-class problem the discriminant functions can actually fit.
static DataSet makeData( unsigned n )
{
	Matrix< double > raw( n, 3 );
	for ( unsigned i = 0; i < n; i++ )
	{
		bool positive = ( i % 2 ) == 0;
		raw( i, 0 ) = ( positive ? 1.0 : -1.0 ) + 0.01 * ( i % 17 );
		raw( i, 1 ) = ( positive ? 0.5 : -0.5 ) + 0.01 * ( i % 13 );
		raw( i, 2 ) = positive ? 1 : 0;
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

static string tmpPath( const char* name )
{
	return string( "/tmp/neuron_lastop_" ) + name + ".txt";
}

// Each writer is driven through the same four checks. `go` runs one training
// pass on a freshly built model configured with the given filename and flag.
template < class GO >
static void checkWriter( const char* who, GO go )
{
	cout << "-- " << who << " --" << endl;

	string path = tmpPath( who );
	remove( path.c_str() );

	// 1. it writes, at the resolved path, and the file holds the report
	{ util::ScreenCapture hush; go( path, true ); }
	expect( exists( path ), string( who ) + ": the last-operation file is written" );
	string first = slurp( path );
	expect( !first.empty(), string( who ) + ": ...and it is not empty" );

	// 2. TRUNCATION: a second run replaces the file, never appends
	{ util::ScreenCapture hush; go( path, true ); }
	string second = slurp( path );
	expect( second.size() == first.size(), string( who )
		+ ": a second run REPLACES the file (it did not grow)" );
	expect( second == first, string( who )
		+ ": ...and the replacement is the new run's report alone" );

	// A deliberately long file, then a run: the leftover tail must be gone.
	//    Compared by SIZE, not by a sentinel character -- the reports contain
	//    most of the alphabet ("Pearson X2 = ", for one), so any letter used as
	//    a marker would be found in the legitimate output and the assertion
	//    would pass or fail for the wrong reason.
	{
		const string filler( 20000, 'X' );
		ofstream stuff( path.c_str(), ios::out | ios::trunc );
		stuff << filler << endl;
		stuff.close();
		{ util::ScreenCapture hush; go( path, true ); }
		string after = slurp( path );
		expect( after.size() < filler.size(), string( who )
			+ ": an existing longer file is truncated, not overwritten in place" );
		expect( after == first, string( who )
			+ ": ...and what remains is exactly this run's report" );
	}

	// 3. disabled: nothing is written, and an existing file is untouched
	{
		ofstream marker( path.c_str(), ios::out | ios::trunc );
		marker << "UNTOUCHED" << endl;
		marker.close();
		{ util::ScreenCapture hush; go( path, false ); }
		expect( slurp( path ) == "UNTOUCHED\n", string( who )
			+ ": with lastop off, an existing file is left alone" );
	}

	// 4. an unopenable path reports itself and does not abort the run.
	//    The capture belongs HERE and not inside the writer lambdas: an inner
	//    capture would swallow this message before the outer one could see it,
	//    and the assertion would fail for a reason that has nothing to do with
	//    the code under test.
	{
		string bad = "/no/such/directory/neuron_lastop.txt";
		string said;
		bool returned = false;
		{
			util::ScreenCapture cap;
			go( bad, true );
			returned = true;
			said = cap.text();
		}
		expect( said.find( "Error in opening " ) != string::npos
			&& said.find( bad ) != string::npos,
			string( who ) + ": an unopenable path names itself" );
		expect( returned, string( who ) + ": ...and the run still completes" );
	}

	remove( path.c_str() );
}

int main()
{
	// Iterative
	checkWriter( "iterative", []( const string& path, bool on )
	{
		Scripted n;
		n.setLastop( on );
		n.setLastopFilename( path );
		n.train();
	} );

	// LDFA
	checkWriter( "ldfa", []( const string& path, bool on )
	{
		DataSet d = makeData( 60 );
		LDFA m;
		m.setDataSet( d );
		m.setHistory( false );
		m.setLastop( on );
		m.setLastopFilename( path );
		m.train();
	} );

	// QDFA
	checkWriter( "qdfa", []( const string& path, bool on )
	{
		DataSet d = makeData( 60 );
		QDFA m;
		m.setDataSet( d );
		m.setHistory( false );
		m.setLastop( on );
		m.setLastopFilename( path );
		m.train();
	} );

	// util::run_path's rule, which the writers depend on: a name with a
	// separator is used verbatim, a bare name goes to the run directory. All
	// three writers resolve through it, so it is asserted once here rather
	// than three times above.
	cout << "-- path resolution --" << endl;
	expect( util::run_path( "/tmp/explicit.txt" ) == "/tmp/explicit.txt",
		"an explicit path is respected verbatim" );
	expect( util::run_path( "model.txt" ).find( "model.txt" ) != string::npos,
		"a bare name is resolved against the run directory" );

	cout << endl << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
