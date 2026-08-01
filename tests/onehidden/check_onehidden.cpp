// check_onehidden.cpp : SimpleProp and BareProp, pinned before they share a base.
//
// The two one-hidden-layer models carry byte-identical state and six methods
// whose bodies differ only by the class name in the signature (refactor_audit.md
// section 8.2, category A): randomize, save, load, pack, the copy utility, and
// the automatic step-size weight snapshot. Those move to a narrow intermediate
// base, OneHiddenNet.
//
// WHAT MUST NOT MOVE, and what this file exists to hold still:
//
//   * THEY REMAIN TWO CONCRETE TYPES. Same runtime type, same objType, same
//     factory dispatch, same first line in a saved model file.
//   * THEIR BIAS ARCHITECTURE IS FIXED AT CONSTRUCTION. Network::biasFlag is a
//     mutable, publicly settable flag; if shared storage or a shared save path
//     ever consulted it to decide layout, then calling setBias() on a live
//     SimpleProp would silently reshape it, or change what it writes to disk.
//     The test flips that flag on both types and requires nothing observable
//     to move.
//   * df() and outputHeader() stay per-class. outputHeader's text is a MODEL
//     FILE FORMAT line that load() reads back, so it is frozen, and df() is two
//     published parameter counts that are deliberately not algebraically
//     unified (rule 7).
//   * SHARED STORAGE MUST NOT IMPLY ASSIGNABILITY. A SimpleProp is not a
//     BareProp with a flag set; if a common base ever made cross-type
//     assignment compile, one would silently become the other. Asserted at
//     COMPILE time, since a runtime test cannot catch what does not compile.
//
// Portable by construction, as check_autostep is: saved-file BYTES, packed
// gradient ordering, integer sizes, and same-process before/after relations --
// no multi-iteration floating-point literals.

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <memory>

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

#include "simpleprop.h"
#include "bareprop.h"
#include "modelfactory.h"
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

// --- SHARED STORAGE MUST NOT MEAN INTERCHANGEABLE ---------------------------
//
// A compile-time assertion, because the failure it guards against is a program
// that COMPILES: if SimpleProp and BareProp ever became assignable to one
// another -- which a careless common base with a public operator= would do --
// then a caller could turn one model into the other and the saved file would
// disagree with the object that wrote it. No runtime test can catch that.

static_assert( !std::is_assignable< SimpleProp&, const BareProp& >::value,
	"a BareProp must never be assignable to a SimpleProp" );
static_assert( !std::is_assignable< BareProp&, const SimpleProp& >::value,
	"a SimpleProp must never be assignable to a BareProp" );
static_assert( !std::is_constructible< SimpleProp, const BareProp& >::value,
	"a SimpleProp must never be constructible from a BareProp" );
static_assert( !std::is_constructible< BareProp, const SimpleProp& >::value,
	"a BareProp must never be constructible from a SimpleProp" );

// Only the forward outputs, which are reachable. pack() and hO are private in
// both classes today, and this test must compile BEFORE the extraction that
// would make them protected -- so pack is checked through what it observably
// controls instead (see test_pack below).
template < class NET >
class Probe : public NET {
public:
	// train() owns the iteration counter -- `for ( iteration = 0; ... )` -- so a
	//    test that calls trainSet() DIRECTLY must set it, exactly as
	//    check_autostep does. It is not initialised by the constructor, and
	//    Network::engine() branches on it: an indeterminate value takes the
	//    conjugate-direction branch on the very first call, where lastF and
	//    lastG are still empty, and dotprod() then reads past the end of an
	//    empty vector. Since 2026-08-01 that throws nvec::SizeMismatch instead
	//    of quietly reading rubbish, which is how this surfaced at all.
	void setIteration( unsigned t ) { this->iteration = t; }

	// forward() takes the MODEL's own input matrix -- Model::Train, which each
	//    model built for itself in setDataSet, with the bias column appended
	//    where its architecture has one. It is NOT the DataSet's training
	//    matrix: that one still carries the outcome in its last column, and its
	//    width happens to equal a SimpleProp's input width, so passing it read
	//    the LABEL into the bias slot and an assert-enabled build aborted on
	//    BareProp, whose input vector is narrower still. Both were true of this
	//    file as first written.
	double signature()
	{
		double sum = 0;
		for ( unsigned r = 0; r < this->Train.rows(); r++ )
		{
			this->forward( this->Train, r );
			sum += this->o;
		}
		return sum;
	}
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

template < class NET >
static void build( Probe< NET >& p, DataSet& d, unsigned hidden = 3 )
{
	p.setDataSet( d );
	p.setHidden( hidden );
	p.setHistory( false );
	p.setLastop( false );
	p.setLogPrint( false );
	p.setQuiet( true );
	p.setXEerror();
	p.setWeightDecay( false );
	p.setDecay( 0 );
	p.setEta( 0.5 );
	util::set_seed( 7 );
	p.randomize();
}

static string slurp( const string& path )
{
	ifstream in( path.c_str() );
	ostringstream all;
	all << in.rdbuf();
	return all.str();
}

// A private directory under the platform's own temporary root. /tmp is NOT a
//    temporary directory on Windows -- check_writelastop had exactly this
//    defect and it was fixed hours before this file was written; repeating it
//    here cost a red CI run. The per-process name also keeps concurrent runs
//    from colliding on a shared filename.
class TempDir {
public:
	TempDir()
	{
		namespace fs = std::filesystem;
		root = fs::temp_directory_path()
			/ ( "neuron_onehidden_" + to_string( ( long ) getpid() ) );
		fs::create_directories( root );
	}
	~TempDir()
	{
		std::error_code ec;
		std::filesystem::remove_all( root, ec );
	}
	string file( const char* name ) const
	{
		return ( root / ( string( name ) + ".txt" ) ).string();
	}
private:
	std::filesystem::path root;
};

static TempDir tempDir;

static string tmp( const char* name ) { return tempDir.file( name ); }

// --- 1. construction, copying, assignment ---------------------------------

template < class NET >
static void test_copying( const char* who )
{
	cout << "-- " << who << ": copying --" << endl;

	DataSet d = makeData( 90 );
	Probe< NET > a;
	build( a, d );

	// Copy construction
	NET copied( a );
	expect( copied.getType() == a.getType(),
		string( who ) + ": a copy keeps the type name" );

	// Assignment
	NET assigned;
	assigned = a;
	expect( assigned.getType() == a.getType(),
		string( who ) + ": an assigned model keeps the type name" );

	// The copies must be the same MODEL, which the saved bytes settle
	string pa = tmp( "orig" ), pc = tmp( "copy" ), pas = tmp( "assign" );
	{
		util::ScreenCapture hush;
		a.save( pa );
		copied.save( pc );
		assigned.save( pas );
	}
	// NON-EMPTY FIRST. Two failed saves both slurp to "" and every equality
	//    below then passes while proving nothing -- which is exactly what
	//    happened on Windows when this file hardcoded /tmp: only the one
	//    assertion that looked at CONTENT caught it.
	string bytesA = slurp( pa );
	expect( !bytesA.empty(), string( who ) + ": the model file has content" );
	expect( bytesA == slurp( pc ),
		string( who ) + ": a copy-constructed model saves identical bytes" );
	expect( bytesA == slurp( pas ),
		string( who ) + ": an assigned model saves identical bytes" );

	remove( pa.c_str() ); remove( pc.c_str() ); remove( pas.c_str() );
}

// --- 2, 3, 4, 5. seed, save bytes, round trip, factory type ---------------

template < class NET >
static void test_file_identity( const char* who, const char* firstLine )
{
	cout << "-- " << who << ": the model file --" << endl;

	DataSet d = makeData( 90 );
	Probe< NET > a;
	build( a, d );

	string p1 = tmp( "a" ), p2 = tmp( "b" );
	{
		util::ScreenCapture hush;
		a.save( p1 );
	}
	string bytes = slurp( p1 );
	expect( !bytes.empty(), string( who ) + ": the model file has content" );

	// The first line IS the format's type identifier, and load() reads it back
	expect( bytes.compare( 0, string( firstLine ).size(), firstLine ) == 0,
		string( who ) + ": the file begins with \"" + firstLine + "\"" );

	// The same seed must give the same weights, so the same bytes
	Probe< NET > again;
	DataSet d2 = makeData( 90 );
	build( again, d2 );
	{
		util::ScreenCapture hush;
		again.save( p2 );
	}
	expect( slurp( p2 ) == bytes,
		string( who ) + ": the same seed writes identical bytes" );

	// Load and save again: a round trip must be a fixed point
	Probe< NET > loaded;
	loaded.setDataSet( d );
	{
		util::ScreenCapture hush;
		loaded.load( p1 );
		loaded.save( p2 );
	}
	expect( slurp( p2 ) == bytes,
		string( who ) + ": load then save reproduces the file exactly" );

	remove( p1.c_str() ); remove( p2.c_str() );
}

// --- 6. pack, through what it observably controls ------------------------
//
// pack() flattens the gradient structure into stackG and unpack() puts it back;
// both are private, and this file must compile before the extraction that would
// make them protected. They are not untested for that reason:
//
//   * the SIZE of the packed vector is df(), which is public, and is a
//     different number for the two models -- so a pack that moved to a shared
//     implementation and lost the bias column would change it;
//   * the ORDER is exercised end to end by CGD, which packs, rewrites the
//     vector as a conjugate direction, and unpacks it every single iteration.
//     A permuted ordering would scatter the direction across the wrong weights
//     and the trained model would differ. That is what --capture records, as a
//     same-machine before/after artifact.

template < class NET >
static void test_pack( const char* who, unsigned wantDf )
{
	cout << "-- " << who << ": pack --" << endl;

	DataSet d = makeData( 90 );
	Probe< NET > p;
	build( p, d );

	expect( p.df() == wantDf, string( who ) + ": df() -- and so the packed "
		"gradient -- is " + to_string( wantDf ) + " parameters" );

	// A CGD run drives pack/unpack every iteration. Two identical runs must
	// agree exactly; a pack and an unpack that disagreed about ordering would
	// still be deterministic, so this is a floor, and --capture is the ceiling.
	double a, b;
	{
		util::ScreenCapture hush;
		p.setBatchEpoch( true );
		p.setTrainingType( 1 );
		for ( unsigned i = 0; i < 5; i++ ) { p.setIteration( i ); p.trainSet(); }
		a = p.signature();
	}
	DataSet d2 = makeData( 90 );
	Probe< NET > q;
	build( q, d2 );
	{
		util::ScreenCapture hush;
		q.setBatchEpoch( true );
		q.setTrainingType( 1 );
		for ( unsigned i = 0; i < 5; i++ ) { q.setIteration( i ); q.trainSet(); }
		b = q.signature();
	}
	expect( a == b, string( who ) + ": five CGD iterations are reproducible" );
}

// --- 7. the automatic step-size snapshot, and continued training ----------

template < class NET >
static void test_autostep_snapshot( const char* who )
{
	cout << "-- " << who << ": the step-size snapshot --" << endl;

	// A searched run must land where a run given the same eta lands: the
	// snapshot restored the trial weights. Same relation check_autostep uses,
	// repeated here because the snapshot type is what is about to move.
	DataSet d1 = makeData( 90 ), d2 = makeData( 90 );
	Probe< NET > searched, given;
	build( searched, d1 );
	build( given, d2 );

	{
		util::ScreenCapture hush;
		searched.setBatchEpoch( true );
		searched.setAutoStepSize( true );
		searched.setIteration( 0 ); searched.trainSet();
		searched.setIteration( 1 ); searched.trainSet();
	}

	// The control repeats the first iteration identically, then is handed the
	// searcher's eta for the second.
	{
		util::ScreenCapture hush;
		given.setBatchEpoch( true );
		given.setAutoStepSize( true );
		given.setIteration( 0 ); given.trainSet();   // identical first iteration
		given.setAutoStepSize( false );
		given.setEta( searched.getEta() );
		given.setIteration( 1 ); given.trainSet();
	}

	expect( fabs( searched.signature() - given.signature() ) < 1e-9,
		string( who ) + ": a searched run lands where an eta-given run lands" );
}

// --- 8. the bias architecture is NOT the mutable flag ---------------------
//
// Network::biasFlag is public through setBias(). If shared storage, a shared
// save(), or a shared pack() ever consulted it, then flipping it on a live
// model would reshape that model or change what it writes to disk. Nothing
// observable may move.

template < class NET >
static void test_bias_flag_is_not_the_architecture( const char* who )
{
	cout << "-- " << who << ": setBias does not reshape an existing model --" << endl;

	DataSet d = makeData( 90 );
	Probe< NET > p;
	build( p, d );

	unsigned dfBefore = p.df();
	string path = tmp( "bias" );
	{
		util::ScreenCapture hush;
		p.save( path );
	}
	string before = slurp( path );

	// Flip it, both ways
	p.setBias( !p.getBias() );
	{
		util::ScreenCapture hush;
		p.save( path );
	}

	expect( p.df() == dfBefore, string( who )
		+ ": the parameter count is unchanged" );
	expect( slurp( path ) == before, string( who )
		+ ": the saved model file is byte-identical" );

	remove( path.c_str() );
}

// --- 5 (factory) ----------------------------------------------------------

static void test_factory_types()
{
	cout << "-- the factory still builds two distinct types --" << endl;

	DataSet d = makeData( 90 );

	// From a SPEC: one hidden layer, bias on or off, is the whole rule that
	//    picks between these two classes.
	struct Case { bool bias; const char* type; };
	const Case cases[] = { { true, "SimpleProp" }, { false, "BareProp" } };

	for ( const Case& c : cases )
	{
		modelfactory::Spec spec;
		spec.bias = c.bias;
		spec.hidden.push_back( 3 );

		string err;
		unique_ptr< Model > m;
		{
			util::ScreenCapture hush;
			m = modelfactory::build( spec, d, err );
		}
		expect( m && m->getType() == c.type,
			string( "spec bias=" ) + ( c.bias ? "on" : "off" ) + " builds a "
			+ c.type );
	}

	// And from a saved file's TYPE NAME, which is the load path
	for ( const Case& c : cases )
	{
		unique_ptr< Model > m;
		{
			util::ScreenCapture hush;
			m = modelfactory::createByTypeName( c.type, c.bias );
		}
		expect( m && m->getType() == c.type,
			string( "the type name \"" ) + c.type + "\" loads back as one" );
	}
}

int main()
{
	test_copying< SimpleProp >( "SimpleProp" );
	test_copying< BareProp >  ( "BareProp" );

	test_file_identity< SimpleProp >( "SimpleProp", "SimpleProp" );
	test_file_identity< BareProp >  ( "BareProp",   "BareProp" );

	// The two published parameter counts, deliberately not unified (rule 7):
	//    SimpleProp ((nInput+2)*nHidden)+1 = (2+2)*3+1 = 13
	//    BareProp   (nInput+1)*nHidden     = (2+1)*3   = 9
	test_pack< SimpleProp >( "SimpleProp", 13 );
	test_pack< BareProp >  ( "BareProp",   9 );

	test_autostep_snapshot< SimpleProp >( "SimpleProp" );
	test_autostep_snapshot< BareProp >  ( "BareProp" );

	test_bias_flag_is_not_the_architecture< SimpleProp >( "SimpleProp" );
	test_bias_flag_is_not_the_architecture< BareProp >  ( "BareProp" );

	test_factory_types();

	cout << endl << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
