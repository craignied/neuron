// harness.h : the shared mechanics of the optimizer benchmark (Phase 0, Step 0A).
//
// Two binaries include this file and nothing else shares it:
//
//   optimizer_probe            measures wall time. Built, never a ctest case --
//                              a timing assertion in the suite is a flake
//                              generator (the scale_probe precedent).
//   check_optimizer_harness    asserts the MECHANICS deterministically, with no
//                              timing assertion anywhere, and IS a ctest case so
//                              the harness cannot rot.
//
// The measuring binary and the binary that proves the measurement is trustworthy
// must run the same code, or the proof is about something other than the tool.
// Hence one header rather than two copies.
//
// NOTHING HERE TOUCHES PRODUCTION. Every reach into a model is through an
// existing public or protected member, using the benchmark-subclass idiom that
// tests/network/check_autostep.cpp already established and CI already runs.
//
// WHAT THIS FILE DELIBERATELY DOES NOT DO, because Phase 0 forbids it: no packed
// weight API, no pure objective/gradient boundary, no new optimizer, no change to
// Network::engine() or to what a trainingType means. Those arrive in the L-BFGS
// phase, shaped by their first real consumer.

#ifndef OPTBENCH_HARNESS_H
#define OPTBENCH_HARNESS_H

#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/resource.h>
#endif

#include "backprop.h"
#include "bareprop.h"
#include "dataset.h"
#include "logistic.h"
#include "simpleprop.h"
#include "utility.h"

#ifndef OPTBENCH_GIT_REV
#define OPTBENCH_GIT_REV "unknown"
#endif
#ifndef OPTBENCH_GIT_DIRTY
#define OPTBENCH_GIT_DIRTY 1
#endif
#ifndef OPTBENCH_SOURCE_ID
#define OPTBENCH_SOURCE_ID "unknown"
#endif
#ifndef OPTBENCH_SOURCE_COUNT
#define OPTBENCH_SOURCE_COUNT 0
#endif
#ifndef OPTBENCH_BUILD_TYPE
#define OPTBENCH_BUILD_TYPE "unknown"
#endif

namespace optbench {

using namespace std;

// Schema 2. Bumped from 1 because field MEANINGS changed, not merely their
//    number: the start identity is now a parameter-state identity rather than a
//    function fingerprint, the iteration count is now the completed count
//    rather than a zero-based index, and data_seed is gone because it had no
//    effect on anything.
const unsigned SCHEMA_VERSION = 2;

// ---------------------------------------------------------------------------
// FNV-1a over raw bytes. Used for three distinct identities answering three
// distinct questions -- keeping them apart is the whole point:
//
//   split          a full-precision content hash of the training matrix: WHICH
//                  DATA. A seed alone is not sufficient, because two
//                  configurations can share a seed and differ in split.
//   weight_*_id    a full-precision hash of the model's ACTUAL WEIGHT
//                  STRUCTURES, including dimensions, ordering and the concrete
//                  model identity: WHICH PARAMETER STATE. This is what an arm is
//                  compared on.
//   function_*_id  a hash of forward() outputs over the training inputs: WHICH
//                  FUNCTION. Secondary and diagnostic only. It cannot serve as a
//                  parameter identity, because distinct weight vectors can agree
//                  on every training row -- a hidden-unit permutation is exactly
//                  such a collision -- and because two fixtures sharing input
//                  columns produce the same value from the same weights.
struct Fnv {
	unsigned long long h;
	Fnv() : h( 14695981039346656037ULL ) { }
	void raw( const void* p, size_t n )
	{
		const unsigned char* q = ( const unsigned char* ) p;
		for ( size_t i = 0; i < n; i++ )
		{
			h ^= q[ i ];
			h *= 1099511628211ULL;
		}
	}
	void d( double v ) { raw( &v, sizeof v ); }
	void u( unsigned v ) { raw( &v, sizeof v ); }
	void s( const string& v ) { u( ( unsigned ) v.size() ); raw( v.data(), v.size() ); }
	// An untouched hash is the offset basis. Comparing two of THOSE is comparing
	//    two absent artifacts, the vacuous-equality trap rule 2 names.
	bool empty() const { return h == 14695981039346656037ULL; }
};

// THE build identity, composed in ONE place. It was composed twice -- once into
//    every row and once into --identity -- and the two disagreed, because only
//    the row half appended the NDEBUG suffix. The runner's stale-binary check
//    caught it immediately, which is the check working: two spellings of one
//    fact is exactly what it exists to refuse.
static inline string buildIdentity()
{
	string b = OPTBENCH_BUILD_TYPE;
#ifdef NDEBUG
	b += "/NDEBUG";
#else
	b += "/asserts";
#endif
	return b;
}

static inline string hex64( unsigned long long v )
{
	ostringstream s;
	s << "0x" << setw( 16 ) << setfill( '0' ) << hex << v;
	return s.str();
}

// ---------------------------------------------------------------------------
// A parameter-state identity, or an explicit statement of why there isn't one.
//
// EVERY MODEL NOW HAS ONE. Logistic through its public getBetas(), the
// one-hidden pair through OneHiddenNet's protected hW/oW, and BackProp through
// the narrow protected BackProp::weightMatrices() added for exactly this
// purpose. `available` remains, and remains checked, because an unavailable or
// empty identity must be REFUSED rather than papered over with the weaker
// function fingerprint: a weaker identity wearing the stronger one's name is the
// defect this whole mechanism exists to prevent.
struct WeightIdentity {
	bool available;
	unsigned long long hash;
	unsigned elements;   // how many weight scalars went into the hash
	string reason;       // why unavailable, or why refused
	WeightIdentity() : available( false ), hash( 0 ), elements( 0 ) { }
};

static inline WeightIdentity unavailableIdentity( const string& why )
{
	WeightIdentity w;
	w.reason = why;
	return w;
}

// The one-hidden-layer weight traversal, written once for SimpleProp and
//    BareProp, whose weight structures are identical -- OneHiddenNet exists for
//    exactly that reason. Dimensions and ordering are hashed alongside the
//    values, so a reshaped model cannot collide with a differently reshaped one
//    holding the same numbers.
static inline WeightIdentity oneHiddenIdentity( const string& tag,
	const Matrix< double >& hW, const vector< double >& oW )
{
	if ( hW.rows() == 0 || hW.cols() == 0 || oW.empty() )
		return unavailableIdentity( "weight structures are empty (hW "
			+ to_string( hW.rows() ) + "x" + to_string( hW.cols() )
			+ ", oW " + to_string( oW.size() ) + ")" );

	Fnv f;
	f.s( tag );
	f.u( hW.rows() );
	f.u( hW.cols() );
	for ( unsigned r = 0; r < hW.rows(); r++ )
		for ( unsigned c = 0; c < hW.cols(); c++ )
			f.d( hW( r, c ) );
	f.u( ( unsigned ) oW.size() );
	for ( size_t i = 0; i < oW.size(); i++ )
		f.d( oW[ i ] );

	WeightIdentity w;
	w.available = true;
	w.hash = f.h;
	w.elements = hW.rows() * hW.cols() + ( unsigned ) oW.size();
	return w;
}

// The vector-of-matrices weight traversal: BackProp's layout. Matrix count,
//    every matrix's dimensions, and every element in deterministic
//    matrix/row/column order, so neither a reshaped layer nor a reordered stack
//    can collide with a different state holding the same numbers. Free-standing
//    rather than inline in the specialization, so a test can vary one element at
//    a time without any mutation access to a production model.
static inline WeightIdentity matricesIdentity( const string& tag,
	const vector< Matrix< double > >& W )
{
	if ( W.empty() )
		return unavailableIdentity( "weight matrix stack is empty" );

	Fnv f;
	f.s( tag );
	f.u( ( unsigned ) W.size() );
	unsigned elements = 0;
	for ( size_t m = 0; m < W.size(); m++ )
	{
		if ( W[ m ].rows() == 0 || W[ m ].cols() == 0 )
			return unavailableIdentity( "weight matrix " + to_string( m )
				+ " is empty (" + to_string( W[ m ].rows() ) + "x"
				+ to_string( W[ m ].cols() ) + ")" );
		f.u( W[ m ].rows() );
		f.u( W[ m ].cols() );
		for ( unsigned r = 0; r < W[ m ].rows(); r++ )
			for ( unsigned c = 0; c < W[ m ].cols(); c++ )
				f.d( W[ m ]( r, c ) );
		elements += W[ m ].rows() * W[ m ].cols();
	}

	WeightIdentity w;
	w.available = true;
	w.hash = f.h;
	w.elements = elements;
	return w;
}

// ---------------------------------------------------------------------------
// One benchmark arm's configuration.

struct Case {
	string name;
	// THE COMPARISON GROUP. Arms sharing this name are asserted to be a fair
	//    optimizer comparison: every field defining the WORK must match and only
	//    the optimizer may differ, unless groupAxis documents another intended
	//    axis. A group name is not decoration -- run_probe.py REFUSES a result
	//    set whose group members disagree, rather than printing a warning.
	string group;
	string groupAxis; // "optimizer", or the other axis this group varies
	string model;     // logistic | simpleprop | bareprop | backprop
	string fixture;   // linear2 | xor2
	unsigned rows;
	unsigned weightSeed;
	unsigned hidden;            // one-hidden models
	vector< unsigned > layers;  // backprop
	unsigned optimizer;         // 0 canonical, 1 CGD, 2 Shanno
	bool batch;
	bool autoStep;
	double eta;
	bool decayOn;
	double decay;
	// Gradient stopping does two things at once: it arms STOP_GRADMAX, and it
	//    selects which production branch canonical training takes -- with it off
	//    the canonical accumulator branch runs, with it on the separate-gradient
	//    branch runs and engine() dispatches through a switch with no case 0
	//    (tests/backprop/check_bpoptimizer.cpp). CGD and Shanno REQUIRE the
	//    separate-gradient branch, so a canonical arm compared against them must
	//    use the same instrumentation or the group is timing two different code
	//    paths under one label. The limit is 0, so the rule can never fire.
	bool gradStop;
	double target;    // matched endpoint, strictly inside (0,1)
	unsigned ceiling; // safety ceiling: reaching it is failure, never convergence
	bool xentropy;

	// BENCHMARK-ONLY fault injection, reachable from the self-test alone: no
	//    pilot case sets it and no command-line flag exposes it. It proves that
	//    the MEASURING path emits a failed row instead of terminating the
	//    process. There is no production throwing hook.
	enum Inject { INJECT_NONE, INJECT_SETUP, INJECT_TRAINING };
	Inject inject;
};

// ---------------------------------------------------------------------------
// One measured row.

struct Row {
	unsigned schema;
	string rev, sourceId, buildType;
	unsigned sourceFiles;   // how many source files the identity covers
	bool dirty;
	string caseName, group, groupAxis, fixture, splitId, model, arch, loss;
	unsigned rows, inputs, params, weightSeed;
	// The parameter-state identity: what an arm is actually compared on.
	bool weightIdAvailable;
	string weightStartId, weightEndId, weightIdNote;
	unsigned weightElements;
	// Secondary, diagnostic: the function the weights compute on the inputs.
	string functionStartId, functionEndId;
	unsigned optimizer;
	string optimizerName, mode;
	double eta;
	bool autoStep, decayOn;
	double decay;
	bool gradStop;
	double target, achieved;
	unsigned ceiling;
	// iterationIndex is Iterative::getIterations(), whose meaning DIFFERS by
	//    exit path: on a stopping-rule break it is the zero-based index of the
	//    iteration that just finished, while on ceiling exhaustion the loop
	//    leaves it at maxIterations+1, which is a count. Two meanings in one
	//    accessor, so it is reported as an index and never used as a count.
	//    iterationsCompleted is the honest measure: trainSet() calls, counted.
	unsigned iterationIndex;
	long long iterationsCompleted;
	long long fullPasses;   // innerTrainSet() calls; -1 = unavailable
	long long elapsedNs;
	long long peakRssKb;    // -1 = unavailable on this platform
	string stopReason;
	bool converged, targetReached, finite, usable;
	string failureStage;    // none | refused | setup | training
	string error;
};

static inline const char* optimizerName( unsigned t )
{
	switch ( t )
	{
	case 0: return "canonical";
	case 1: return "cgd";
	case 2: return "shanno";
	default: return "unknown";
	}
}

static inline string jsonEscape( const string& s )
{
	string out;
	for ( size_t i = 0; i < s.size(); i++ )
	{
		char c = s[ i ];
		if ( c == '"' || c == '\\' ) { out += '\\'; out += c; }
		else if ( c == '\n' ) out += "\\n";
		else if ( c == '\r' ) out += "\\r";
		else if ( c == '\t' ) out += "\\t";
		else if ( ( unsigned char ) c < 0x20 ) { } // drop other controls
		else out += c;
	}
	return out;
}

// Non-finite has no JSON spelling, so it becomes null rather than the bare token
//    `nan`, which no strict parser accepts. The `finite` flag distinguishes a
//    null from an absence.
static inline string jnum( double v )
{
	if ( !std::isfinite( v ) )
		return "null";
	ostringstream s;
	s << setprecision( 17 ) << v;
	return s.str();
}

static inline string jstr( const string& k, const string& v )
{
	return ",\"" + k + "\":\"" + jsonEscape( v ) + "\"";
}

static inline string toJsonLine( const Row& r )
{
	ostringstream o;
	o << "{\"schema\":" << r.schema;
	o << jstr( "rev", r.rev );
	o << ",\"dirty\":" << ( r.dirty ? "true" : "false" );
	o << jstr( "source_id", r.sourceId );
	o << ",\"source_files\":" << r.sourceFiles;
	o << jstr( "build", r.buildType );
	o << jstr( "case", r.caseName );
	o << jstr( "comparison_group", r.group );
	o << jstr( "group_axis", r.groupAxis );
	o << jstr( "fixture", r.fixture );
	o << jstr( "split", r.splitId );
	o << jstr( "model", r.model );
	o << jstr( "arch", r.arch );
	o << jstr( "loss", r.loss );
	o << ",\"rows\":" << r.rows;
	o << ",\"inputs\":" << r.inputs;
	o << ",\"params\":" << r.params;
	o << ",\"weight_seed\":" << r.weightSeed;
	o << ",\"weight_id_available\":" << ( r.weightIdAvailable ? "true" : "false" );
	o << jstr( "weight_start_id", r.weightStartId );
	o << jstr( "weight_end_id", r.weightEndId );
	o << ",\"weight_elements\":" << r.weightElements;
	o << jstr( "weight_id_note", r.weightIdNote );
	o << jstr( "function_start_id", r.functionStartId );
	o << jstr( "function_end_id", r.functionEndId );
	o << ",\"optimizer\":" << r.optimizer;
	o << jstr( "optimizer_name", r.optimizerName );
	o << jstr( "mode", r.mode );
	o << ",\"eta\":" << jnum( r.eta );
	o << ",\"auto_step\":" << ( r.autoStep ? "true" : "false" );
	o << ",\"decay_on\":" << ( r.decayOn ? "true" : "false" );
	o << ",\"decay\":" << jnum( r.decay );
	o << ",\"grad_stop\":" << ( r.gradStop ? "true" : "false" );
	o << ",\"target\":" << jnum( r.target );
	o << ",\"achieved\":" << jnum( r.achieved );
	o << ",\"ceiling\":" << r.ceiling;
	o << ",\"iteration_index\":" << r.iterationIndex;
	if ( r.iterationsCompleted < 0 ) o << ",\"iterations_completed\":null";
	else o << ",\"iterations_completed\":" << r.iterationsCompleted;
	if ( r.fullPasses < 0 ) o << ",\"full_passes\":null";
	else o << ",\"full_passes\":" << r.fullPasses;
	o << ",\"elapsed_ns\":" << r.elapsedNs;
	if ( r.peakRssKb < 0 ) o << ",\"peak_rss_kb\":null";
	else o << ",\"peak_rss_kb\":" << r.peakRssKb;
	o << jstr( "stop_reason", r.stopReason );
	o << ",\"converged\":" << ( r.converged ? "true" : "false" );
	o << ",\"target_reached\":" << ( r.targetReached ? "true" : "false" );
	o << ",\"finite\":" << ( r.finite ? "true" : "false" );
	o << ",\"usable\":" << ( r.usable ? "true" : "false" );
	o << jstr( "failure_stage", r.failureStage );
	o << jstr( "error", r.error );
	o << "}";
	return o.str();
}

// ---------------------------------------------------------------------------
// The benchmark subclass. Two counters and two identities.
//
// innerTrainSet() and trainSet() are each called once per pass and once per
// iteration respectively -- never per exemplar -- so both counters sit outside
// every hot loop and rule 7 is untouched. The innerTrainSet override is the
// idiom tests/network/check_autostep.cpp already established and CI runs.

template < class NET >
class Probe : public NET {
public:
	Probe() : innerCalls( 0 ), outerCalls( 0 ), injectTraining( false ) { }

	// Every full pass through the training set. The step-size search runs
	//    maxLoops trial passes and then one real one, so this is what makes
	//    those extra passes visible instead of hidden inside one iteration.
	double innerTrainSet() override
	{
		innerCalls++;
		if ( injectTraining && innerCalls == 2 )
			throw runtime_error( "benchmark-only injected training fault" );
		return NET::innerTrainSet();
	}

	// Every training ITERATION. Iterative::getIterations() cannot answer this:
	//    its value is a zero-based index on a rule break but a count on ceiling
	//    exhaustion. Counting the calls is unambiguous on every exit path.
	double trainSet() override
	{
		outerCalls++;
		return NET::trainSet();
	}

	unsigned innerCalls, outerCalls;
	bool injectTraining;

	// THE PARAMETER-STATE IDENTITY. Specialized per model below, because each
	//    model's weight structures are its own and are reached differently:
	//    Logistic through its public getBetas(), the one-hidden models through
	//    OneHiddenNet's protected hW/oW, and BackProp not at all.
	WeightIdentity weightIdentity() const;

	// SECONDARY: the function the current weights compute on the training
	//    inputs. Diagnostic. Never the identity an arm is compared on.
	unsigned long long functionIdentity( unsigned& rowsSeen, bool& allFinite )
	{
		Fnv f;
		rowsSeen = 0;
		allFinite = true;
		for ( unsigned r = 0; r < this->Train.rows(); r++ )
		{
			this->forward( this->Train, r );
			if ( !std::isfinite( this->o ) )
				allFinite = false;
			f.d( this->o );
			rowsSeen++;
		}
		return f.h;
	}
};

// Logistic: W is private, but getBetas() is a public const accessor to it.
template <>
inline WeightIdentity Probe< Logistic >::weightIdentity() const
{
	const vector< double >& W = this->getBetas();
	if ( W.empty() )
		return unavailableIdentity( "Logistic weight vector is empty" );
	Fnv f;
	f.s( "Logistic" );
	f.u( ( unsigned ) W.size() );
	for ( size_t i = 0; i < W.size(); i++ )
		f.d( W[ i ] );
	WeightIdentity w;
	w.available = true;
	w.hash = f.h;
	w.elements = ( unsigned ) W.size();
	return w;
}

// SimpleProp / BareProp: hW and oW are OneHiddenNet's protected members, so a
//    subclass reads them directly. One traversal, called from both.
template <>
inline WeightIdentity Probe< SimpleProp >::weightIdentity() const
{
	return oneHiddenIdentity( "SimpleProp", this->hW, this->oW );
}

template <>
inline WeightIdentity Probe< BareProp >::weightIdentity() const
{
	return oneHiddenIdentity( "BareProp", this->hW, this->oW );
}

// BackProp: through the narrow protected read-only seam BackProp::weightMatrices(),
//    which returns the authoritative weight stack by const reference and reaches
//    no training workspace. Added 2026-08-04 because BackProp was the one model
//    whose parameters a subclass could not read, and an optimizer comparison
//    whose arms cannot be shown to share a starting state is not a comparison.
template <>
inline WeightIdentity Probe< BackProp >::weightIdentity() const
{
	return matricesIdentity( "BackProp", this->weightMatrices() );
}

// ---------------------------------------------------------------------------
// Deterministic fixtures, generated rather than committed.
//
//   linear2  a 2-input threshold problem, the construction the existing
//            optimizer tests use, so a surprise here is a change in the engine
//            rather than in a new dataset nobody has looked at.
//   xor2     a 2-input XOR-shaped problem: no linear boundary separates it.
//
// NO DATA SEED. These fixtures are deterministic by construction and nothing
// about them varies with a seed, so the schema no longer carries one. Reporting
// a seed that has no effect is false provenance. Step 0B introduces a real
// split/data seed together with real holdout splits.

static inline Matrix< double > fixtureMatrix( const string& which, unsigned n )
{
	Matrix< double > raw( n, 3 );
	for ( unsigned i = 0; i < n; i++ )
	{
		double x0 = -1.0 + 2.0 * ( ( i * 37 ) % 100 ) / 99.0;
		double x1 = -1.0 + 2.0 * ( ( i * 53 ) % 100 ) / 99.0;
		raw( i, 0 ) = x0;
		raw( i, 1 ) = x1;
		if ( which == "xor2" )
			raw( i, 2 ) = ( ( x0 > 0 ) != ( x1 > 0 ) ) ? 1 : 0;
		else
			raw( i, 2 ) = ( x0 + x1 > 0.55 ) ? 1 : 0;
	}
	return raw;
}

static inline unsigned long long matrixIdentity( const Matrix< double >& m )
{
	Fnv f;
	f.u( m.rows() );
	f.u( m.cols() );
	for ( unsigned r = 0; r < m.rows(); r++ )
		for ( unsigned c = 0; c < m.cols(); c++ )
			f.d( m( r, c ) );
	return f.h;
}

static inline DataSet makeDataSet( const Case& c, unsigned long long& fixtureId )
{
	Matrix< double > raw = fixtureMatrix( c.fixture, c.rows );
	DataSet d;
	d.setInput( 2 );
	d.setOutput( 1 );
	d.setDiscrete( true );
	d.setHistory( false );
	util::ScreenCapture hush;
	d.setRawMatrix( raw );
	// Step 0A trains on the whole fixture: no holdout, so the split identity is
	//    the fixture identity and there is one fewer moving part while the
	//    mechanics are proven. Step 0B introduces real splits with Civic Choice.
	d.setTrainMatrix( raw );
	fixtureId = matrixIdentity( raw );
	return d;
}

// ---------------------------------------------------------------------------
// Architecture, per model. Overloads rather than a switch: each model's
// architecture call has its own type, and the compiler should say so.

static inline void arch( Probe< Logistic >&, const Case& ) { }
static inline void arch( Probe< SimpleProp >& p, const Case& c ) { p.setHidden( c.hidden ); }
static inline void arch( Probe< BareProp >& p, const Case& c ) { p.setHidden( c.hidden ); }
static inline void arch( Probe< BackProp >& p, const Case& c ) { p.setHidden( c.layers ); }

static inline string archLabel( const Case& c )
{
	if ( c.model == "logistic" )
		return "linear";
	if ( c.model == "backprop" )
	{
		ostringstream s;
		for ( size_t i = 0; i < c.layers.size(); i++ )
			s << ( i ? "-" : "" ) << c.layers[ i ];
		return s.str();
	}
	ostringstream s;
	s << c.hidden;
	return s.str();
}

static inline long long peakRssKb()
{
#ifdef _WIN32
	return -1; // no portable equivalent wired up; reported as null, never as 0
#else
	struct rusage ru;
	if ( getrusage( RUSAGE_SELF, &ru ) != 0 )
		return -1;
#ifdef __APPLE__
	return ( long long ) ( ru.ru_maxrss / 1024 ); // Darwin reports bytes
#else
	return ( long long ) ru.ru_maxrss;            // Linux reports kilobytes
#endif
#endif
}

// ---------------------------------------------------------------------------
// Configuration refusal. Release builds strip every assert in the setters
// (NDEBUG is the shipped contract), so a bad value would be installed silently
// and the run would report a number for a configuration nobody asked for. Each
// refusal names the field.

static inline string validate( const Case& c )
{
	if ( c.model != "logistic" && c.model != "simpleprop"
		&& c.model != "bareprop" && c.model != "backprop" )
		return "model: unknown model '" + c.model + "'";
	if ( c.fixture != "linear2" && c.fixture != "xor2" )
		return "fixture: unknown fixture '" + c.fixture + "'";
	if ( c.optimizer > 2 )
		return "optimizer: must be 0, 1 or 2";
	if ( c.rows < 2 )
		return "rows: must be at least 2";
	if ( !( c.target > 0.0 && c.target < 1.0 ) )
		return "target: must be strictly between 0 and 1 (Iterative::setMinError)";
	if ( c.ceiling < 1 )
		return "ceiling: must be at least 1";
	if ( !std::isfinite( c.eta ) || c.eta < 0.0 || c.eta > 1.0 )
		return "eta: must be finite and within [0,1] (Network::setEta)";
	if ( !std::isfinite( c.decay ) || c.decay < 0.0 || c.decay > 1.0 )
		return "decay: must be finite and within [0,1] (Network::setDecay)";
	if ( c.model == "backprop" && c.layers.empty() )
		return "layers: backprop needs at least one hidden layer";
	if ( ( c.model == "simpleprop" || c.model == "bareprop" ) && c.hidden < 1 )
		return "hidden: must be at least 1";
	if ( c.group.empty() )
		return "comparison_group: every case must name one";
	return "";
}

// Fill the configuration-describing half of a row. Shared by the measured path
//    and the refusal path, so a refused arm is never a differently shaped row.
static inline void describe( Row& r, const Case& c, const string& rev )
{
	r.schema = SCHEMA_VERSION;
	r.rev = rev;
	r.dirty = OPTBENCH_GIT_DIRTY ? true : false;
	r.sourceId = OPTBENCH_SOURCE_ID;
	r.sourceFiles = OPTBENCH_SOURCE_COUNT;
	r.buildType = buildIdentity();
	r.caseName = c.name;
	r.group = c.group;
	r.groupAxis = c.groupAxis;
	r.fixture = c.fixture;
	r.splitId = "";
	r.model = c.model;
	r.arch = archLabel( c );
	r.loss = c.xentropy ? "xentropy" : "lms";
	r.rows = r.inputs = r.params = 0;
	r.weightSeed = c.weightSeed;
	r.weightIdAvailable = false;
	r.weightStartId = r.weightEndId = r.weightIdNote = "";
	r.weightElements = 0;
	r.functionStartId = r.functionEndId = "";
	r.optimizer = c.optimizer;
	r.optimizerName = optimizerName( c.optimizer );
	r.mode = c.batch ? "batch" : "online";
	r.eta = c.eta;
	r.autoStep = c.autoStep;
	r.decayOn = c.decayOn;
	r.decay = c.decay;
	r.gradStop = c.gradStop;
	r.target = c.target;
	r.achieved = numeric_limits< double >::quiet_NaN();
	r.ceiling = c.ceiling;
	r.iterationIndex = 0;
	r.iterationsCompleted = -1;
	r.fullPasses = -1;
	r.elapsedNs = 0;
	r.peakRssKb = -1;
	r.stopReason = "none";
	r.converged = r.targetReached = r.finite = r.usable = false;
	r.failureStage = "none";
	r.error = "";
}

// ---------------------------------------------------------------------------
// One arm.
//
// THE TIMED REGION IS EXACTLY train(), AND NOTHING ELSE. Dataset construction,
// architecture, randomization, both identities at both ends, serialization and
// the memory reading are all outside it. Quiet mode is on, so the epilogue --
// the accuracy report, the classification tables, the ROC fit and its
// 2000-resample bootstrap -- never runs inside the measurement.
//
// EVERY RUNTIME FAILURE STILL EMITS EXACTLY ONE ROW. An exception out of setup
// or out of train() is caught here, recorded with its stage and message, and
// returned as a failed row: the process does not terminate and the campaign does
// not lose the arm. That contract matters immediately for Step 0B's singular,
// separated and non-finite cases, where throwing is the correct engine behavior.

template < class NET >
static Row runTyped( const Case& c, const string& rev )
{
	Row r;
	describe( r, c, rev );

	Probe< NET > p;
	unsigned long long fixtureId = 0;

	// --- setup, outside the timed region -----------------------------------
	try
	{
		DataSet d = makeDataSet( c, fixtureId );
		r.splitId = hex64( fixtureId );
		r.rows = d.getNumTrain();
		r.inputs = d.getInput();

		if ( c.inject == Case::INJECT_SETUP )
			throw runtime_error( "benchmark-only injected setup fault" );

		p.setDataSet( d );
		arch( p, c );

		p.setHistory( false );  // no neuron.log
		p.setLastop( false );   // no model.txt
		p.setLogPrint( false );
		p.setQuiet( true );     // no epilogue: no report, no ROC bootstrap
		if ( c.xentropy ) p.setXEerror(); else p.setLMSerror();
		p.setWeightDecay( c.decayOn );
		p.setDecay( c.decay );
		p.setBatchEpoch( c.batch );
		p.setAutoStepSize( c.autoStep );
		p.setEta( c.eta );
		p.setTrainingType( c.optimizer );

		// THE MATCHED ENDPOINT, and only it.
		p.setMinStop( true );
		p.setMinError( c.target );
		p.setChangeStop( false );
		p.setWindowStop( false );
		p.setGradStop( c.gradStop );
		p.setGradMaxLimit( 0.0 ); // armed branch, but a rule that can never fire
		p.setMaxIterations( c.ceiling );

		util::set_seed( c.weightSeed );
		p.randomize();
		r.params = p.df();

		WeightIdentity ws = p.weightIdentity();
		r.weightIdAvailable = ws.available;
		r.weightIdNote = ws.reason;
		r.weightElements = ws.elements;
		if ( ws.available )
			r.weightStartId = hex64( ws.hash );

		unsigned rowsSeen = 0;
		bool startFinite = true;
		Fnv fp;
		fp.h = p.functionIdentity( rowsSeen, startFinite );

		// Non-vacuity, before any timing is reported: no rows traversed, or an
		//    untouched hash, is not evidence of a starting state.
		if ( rowsSeen == 0 || fp.empty() )
		{
			r.failureStage = "setup";
			r.error = "initial state produced no evidence (rows="
				+ to_string( rowsSeen ) + ")";
			return r;
		}
		if ( !startFinite )
		{
			r.failureStage = "setup";
			r.error = "initial state is not finite";
			return r;
		}
		r.functionStartId = hex64( fp.h );
	}
	catch ( const exception& e )
	{
		r.failureStage = "setup";
		r.error = string( "exception during setup: " ) + e.what();
		return r;
	}
	catch ( ... )
	{
		r.failureStage = "setup";
		r.error = "unknown exception during setup";
		return r;
	}

	// --- the measurement ----------------------------------------------------
	p.innerCalls = 0;
	p.outerCalls = 0;
	p.injectTraining = ( c.inject == Case::INJECT_TRAINING );

	double achieved = 0;
	bool threw = false;
	try
	{
		util::ScreenCapture hush; // a quiet run says nothing, but stdout is ours
		chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
		achieved = p.train();
		chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
		r.elapsedNs = chrono::duration_cast< chrono::nanoseconds >( t1 - t0 ).count();
	}
	catch ( const exception& e )
	{
		threw = true;
		r.failureStage = "training";
		r.error = string( "exception during training: " ) + e.what();
	}
	catch ( ... )
	{
		threw = true;
		r.failureStage = "training";
		r.error = "unknown exception during training";
	}

	// Whatever survived the attempt is still worth recording.
	r.fullPasses = ( long long ) p.innerCalls;
	r.iterationsCompleted = ( long long ) p.outerCalls;
	r.iterationIndex = p.getIterations();
	r.peakRssKb = peakRssKb();

	if ( threw )
		// A model left mid-update is not a state to fingerprint. No end identity
		//    is taken, and the row says why rather than reporting a number
		//    derived from partially applied weights.
		return r;

	r.achieved = achieved;
	Iterative::StopReason why = p.getStopReason();
	r.stopReason = Iterative::stopReasonToken( why );
	r.converged = Iterative::converged( why );
	r.finite = std::isfinite( achieved );

	WeightIdentity we = p.weightIdentity();
	if ( we.available )
		r.weightEndId = hex64( we.hash );

	unsigned endRows = 0;
	bool endFinite = true;
	r.functionEndId = hex64( p.functionIdentity( endRows, endFinite ) );

	r.targetReached = r.finite && achieved < c.target
		&& why == Iterative::STOP_MIN_ERROR;

	// THE USABLE PREDICATE. A row earns a place in a timing summary only by
	//    satisfying every one of these. The END-STATE facts are part of it: a
	//    finite returned objective sitting on non-finite final outputs, or a
	//    traversal that covered fewer rows than the training set, is not a
	//    completed fit however fast it was.
	r.usable = r.finite
		&& r.targetReached
		&& r.converged
		&& why != Iterative::STOP_MAX_ITERATIONS
		&& why != Iterative::STOP_CANCELLED
		&& why != Iterative::STOP_PROBE_BUDGET
		&& r.iterationsCompleted > 0
		&& r.fullPasses > 0
		&& !r.functionStartId.empty()
		&& endFinite
		&& endRows == r.rows
		&& ( !r.weightIdAvailable || !r.weightEndId.empty() );

	if ( !r.usable && r.error.empty() )
	{
		if ( !r.finite ) r.error = "non-finite objective";
		else if ( !endFinite ) r.error = "final outputs are not finite";
		else if ( endRows != r.rows ) r.error = "final traversal covered "
			+ to_string( endRows ) + " of " + to_string( r.rows ) + " rows";
		else if ( why == Iterative::STOP_MAX_ITERATIONS )
			r.error = "ceiling exhausted without reaching target";
		else if ( !r.targetReached ) r.error = "stopped without reaching target";
		else r.error = "unusable";
	}

	return r;
}

static inline Row runCase( const Case& c, const string& rev )
{
	string bad = validate( c );
	if ( !bad.empty() )
	{
		Row r;
		describe( r, c, rev );
		r.failureStage = "refused";
		r.error = "refused -- " + bad;
		return r;
	}

	if ( c.model == "logistic" )   return runTyped< Logistic >( c, rev );
	if ( c.model == "simpleprop" ) return runTyped< SimpleProp >( c, rev );
	if ( c.model == "bareprop" )   return runTyped< BareProp >( c, rev );
	return runTyped< BackProp >( c, rev );
}

// ---------------------------------------------------------------------------
// The pilot case table. Step 0A only: enough cases to exercise every mechanic
// the validation asserts, and no more. The committed workload matrix and its
// measured target table belong to Step 0B.

static inline Case baseCase()
{
	Case c;
	c.name = "";
	c.group = "";
	c.groupAxis = "optimizer";
	c.model = "logistic";
	c.fixture = "linear2";
	c.rows = 240;
	c.weightSeed = 7;
	c.hidden = 3;
	c.optimizer = 0;
	c.batch = true;
	c.autoStep = false;
	c.eta = 0.5;
	c.decayOn = false;
	c.decay = 0.0;
	// THE GROUP DEFAULT IS THE SEPARATE-GRADIENT BRANCH. CGD and Shanno require
	//    it, so a canonical arm compared with them must run it too, or the group
	//    is timing two different code paths. The limit is 0, so it never fires.
	c.gradStop = true;
	c.target = 0.35;
	c.ceiling = 4000;
	c.xentropy = true;
	c.inject = Case::INJECT_NONE;
	return c;
}

static inline vector< Case > pilotCases()
{
	vector< Case > v;

	// --- group simpleprop-opt: THE optimizer comparison. Every field that
	//     defines the work is identical; only trainingType differs.
	{
		Case c = baseCase();
		c.name = "simpleprop-canonical";
		c.group = "simpleprop-opt";
		c.model = "simpleprop";
		v.push_back( c );
	}
	{
		Case c = baseCase();
		c.name = "simpleprop-cgd";
		c.group = "simpleprop-opt";
		c.model = "simpleprop";
		c.optimizer = 1;
		v.push_back( c );
	}
	{
		Case c = baseCase();
		c.name = "simpleprop-shanno";
		c.group = "simpleprop-opt";
		c.model = "simpleprop";
		c.optimizer = 2;
		v.push_back( c );
	}

	// --- group logistic-opt
	{
		Case c = baseCase();
		c.name = "logistic-canonical";
		c.group = "logistic-opt";
		v.push_back( c );
	}
	{
		Case c = baseCase();
		c.name = "logistic-shanno";
		c.group = "logistic-opt";
		c.optimizer = 2;
		v.push_back( c );
	}

	// --- group bareprop-opt: the unbiased sibling at the same scale
	{
		Case c = baseCase();
		c.name = "bareprop-canonical";
		c.group = "bareprop-opt";
		c.model = "bareprop";
		v.push_back( c );
	}
	{
		Case c = baseCase();
		c.name = "bareprop-shanno";
		c.group = "bareprop-opt";
		c.model = "bareprop";
		c.optimizer = 2;
		v.push_back( c );
	}

	// --- group backprop-opt: certifiable since BackProp::weightMatrices() gave
	//     the multi-layer model a parameter-state identity like every other.
	{
		Case c = baseCase();
		c.name = "backprop-canonical";
		c.group = "backprop-opt";
		c.model = "backprop";
		c.layers.push_back( 4 );
		c.layers.push_back( 3 );
		v.push_back( c );
	}
	{
		Case c = baseCase();
		c.name = "backprop-shanno";
		c.group = "backprop-opt";
		c.model = "backprop";
		c.optimizer = 2;
		c.layers.push_back( 4 );
		c.layers.push_back( 3 );
		v.push_back( c );
	}

	// --- group canonical-branch: the FAST canonical accumulator path against
	//     the separate-gradient path. A different code path, so it gets its own
	//     group and an honest axis label instead of being timed against CGD and
	//     Shanno as though it were the same work.
	{
		Case c = baseCase();
		c.name = "simpleprop-canonical-accumulator";
		c.group = "canonical-branch";
		c.groupAxis = "grad_stop branch";
		c.model = "simpleprop";
		c.gradStop = false;
		v.push_back( c );
	}
	{
		Case c = baseCase();
		c.name = "simpleprop-canonical-separate";
		c.group = "canonical-branch";
		c.groupAxis = "grad_stop branch";
		c.model = "simpleprop";
		c.gradStop = true;
		v.push_back( c );
	}

	// --- group autostep: the step-size search off and on, at a FIXED ceiling
	//     with an unreachable target so both arms run the same number of
	//     iterations and the pass count is the only free variable.
	{
		Case c = baseCase();
		c.name = "passcount-nosearch";
		c.group = "autostep";
		c.groupAxis = "auto_step";
		c.model = "simpleprop";
		c.target = 1e-30;
		c.ceiling = 20;
		v.push_back( c );
	}
	{
		Case c = baseCase();
		c.name = "passcount-search";
		c.group = "autostep";
		c.groupAxis = "auto_step";
		c.model = "simpleprop";
		c.autoStep = true;
		c.target = 1e-30;
		c.ceiling = 20;
		v.push_back( c );
	}

	// --- group ceiling: an impossible target. Reaching the ceiling is a FAILURE
	//     TO CONVERGE and must never be reported as a fast result.
	{
		Case c = baseCase();
		c.name = "impossible-target";
		c.group = "ceiling";
		c.groupAxis = "none";
		c.target = 1e-30;
		c.ceiling = 50;
		v.push_back( c );
	}

	// --- group nonlinear: the XOR fixture. Target from the canonical control
	//     (--characterize at 927d6dc reached 0.2184 in 4000 iterations).
	{
		Case c = baseCase();
		c.name = "simpleprop-xor";
		c.group = "nonlinear";
		c.groupAxis = "none";
		c.model = "simpleprop";
		c.fixture = "xor2";
		c.target = 0.30;
		v.push_back( c );
	}

	return v;
}

// THE CANONICAL REFERENCE of a comparison group: the arm a matched target is
//    characterized from. --characterize is allowed only on one of these, because
//    a target must come from a canonical control and never from the candidate
//    being timed.
static inline bool isCanonicalReference( const Case& c )
{
	return c.optimizer == 0 && !c.autoStep;
}

// Which canonical reference case belongs to a given group. Empty when the group
//    has none, so the caller can say so rather than guess.
static inline string canonicalReferenceFor( const string& group )
{
	vector< Case > all = pilotCases();
	for ( size_t i = 0; i < all.size(); i++ )
		if ( all[ i ].group == group && isCanonicalReference( all[ i ] ) )
			return all[ i ].name;
	return "";
}

static inline bool findCase( const string& name, Case& out )
{
	vector< Case > all = pilotCases();
	for ( size_t i = 0; i < all.size(); i++ )
		if ( all[ i ].name == name )
		{
			out = all[ i ];
			return true;
		}
	return false;
}

} // namespace optbench

#endif
