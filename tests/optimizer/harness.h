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
#include <filesystem>
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
#include "crossval.h"
#include "cvadapters.h"
#include "dataset.h"
#include "logistic.h"
#include "plateau.h"
#include "simpleprop.h"
#include "split.h"
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
// THE ENGINE the harness links, identified apart from the harness itself.
//    source_id covers everything that can change what is measured, which is the
//    right authority for an ARM ROW. It is the wrong one for a TARGET: writing a
//    measured target into this file changes source_id, so a target could never
//    share an identity with the arms judged against it. What a target depends on
//    is src/, and engine_id is exactly that -- stable across harness edits, and
//    invalidated loudly by a change to the engine.
#ifndef OPTBENCH_ENGINE_ID
#define OPTBENCH_ENGINE_ID "unknown"
#endif
#ifndef OPTBENCH_ENGINE_COUNT
#define OPTBENCH_ENGINE_COUNT 0
#endif

namespace optbench {

using namespace std;

// Schema 2. Bumped from 1 because field MEANINGS changed, not merely their
//    number: the start identity is now a parameter-state identity rather than a
//    function fingerprint, the iteration count is now the completed count
//    rather than a zero-based index, and data_seed is gone because it had no
//    effect on anything.
// Schema 3 (Step 0B). Bumped because field MEANINGS changed again and new
//    facts became load-bearing: `rows` is now the TRAINING row count of a real
//    holdout split rather than the whole fixture, `data_seed` returns as a seed
//    that actually selects the split, and a row now declares WHICH ENDPOINT it
//    was aimed at and WHAT SCOPE was timed. A reader that does not know those
//    two fields can mistake a whole-workflow measurement for an optimizer-only
//    one, which is precisely the confusion Step 0B must not create.
const unsigned SCHEMA_VERSION = 3;

// THE TWO PREDECLARED ENDPOINTS. Both are training-objective values and both
//    come from a canonical characterization run, never from the arm being
//    timed. They answer different questions and must never be averaged together:
//
//      practical  the objective canonical had reached when further training
//                 stopped materially improving the HELD-OUT model. Reaching it
//                 fast is what makes a real workload tractable.
//      strict     the objective canonical reaches when run out to its own
//                 floor. It is the late-stage failure detector: an optimizer
//                 that races to the practical endpoint and then cannot finish
//                 is a different animal from one that does both.
//
//    A third value, "none", is for arms that are not endpoint comparisons at
//    all (the pass-count and ceiling mechanics cases).
static const char* const ENDPOINT_PRACTICAL = "practical";
static const char* const ENDPOINT_STRICT = "strict";
static const char* const ENDPOINT_NONE = "none";

// WHAT THE CLOCK COVERED. A complete-workflow number is a legitimate and
//    necessary measurement -- it is the one the user actually waits for -- but
//    it is not an optimizer timing, and a row that does not say which it is can
//    be quoted as either. So every row says.
//
//      optimizer  exactly train() on one model, epilogue suppressed
//      workflow   a whole repeated-fit consumer: every fold's fit AND its
//                 scoring epilogue, plus the locked refit
static const char* const SCOPE_OPTIMIZER = "optimizer";
static const char* const SCOPE_WORKFLOW = "workflow";

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
	string fixture;   // linear2 | xor2, or the name of a prepared data file

	// A FILE-BACKED FIXTURE. Non-empty means the fixture is a groomed dataset on
	//    disk -- Civic Choice and its row-count series -- loaded through the
	//    maintained recipe rather than generated here. `rows` is then ignored:
	//    the file decides how many rows there are, and the row reports what was
	//    actually loaded rather than what was requested.
	string dataFile;
	unsigned inputs;  // input nodes of that file; 0 for a generated fixture

	// THE SPLIT SEED, which now genuinely selects a split. Step 0A carried a
	//    data_seed that nothing consumed, so it was removed as false provenance;
	//    it returns here because a real stratified holdout depends on it.
	unsigned dataSeed;
	double testFraction; // 0 = train on everything (the Step 0A behavior)

	unsigned rows;
	unsigned weightSeed;

	// WHICH ENDPOINT this arm is aimed at, and WHAT ITS CLOCK COVERED. Both are
	//    part of the comparison group's invariant: an arm racing to the
	//    practical endpoint and an arm racing to the strict one are not
	//    comparable, and neither is an optimizer-only timing against a
	//    whole-workflow one.
	string endpoint;    // practical | strict | none
	string timingScope; // optimizer | workflow

	// THE REPEATED-FIT CONSUMER. "fit" is one train() call; "cv" is repeated
	//    cross-validation over a shared fold plan plus the locked refit, which
	//    is the operation that actually dominates the intended workflow. A cv
	//    arm is timed at workflow scope by construction -- its clock necessarily
	//    covers each fold's scoring epilogue, because that is where a fold's
	//    held-out predictions come from.
	string workload;    // fit | cv
	unsigned cvFolds;
	unsigned cvRepeats;
	unsigned hidden;            // one-hidden models
	vector< unsigned > layers;  // backprop
	unsigned optimizer;         // 0 canonical, 1 CGD, 2 Shanno, 3 L-BFGS (research)

	// L-BFGS memory length m. Ignored by every other optimizer, and a GROUP
	//    INVARIANT: two L-BFGS arms at different m are not the same method, so
	//    comparing them needs its own group with its own declared axis.
	unsigned lbfgsMemory;
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

	// THE PLATEAU AUTO-STOP, the engine's own fold-relative stopping rule.
	//    A matched OBJECTIVE endpoint is the right instrument for a single fit
	//    on a fixed training set. It is the wrong one for cross-validation, and
	//    that was measured rather than assumed: the endpoint characterized on
	//    the full development set was unreachable on 4 of 10 folds, because a
	//    fold trains on 80% of those rows and its achievable objective is its
	//    own. Racing four methods to an objective that 40% of the work cannot
	//    reach produces no timing at all.
	//
	//    So a cv arm stops where the ENGINE says a fit has stopped improving --
	//    setAutoStop, at Iterative's own default tolerance and window. Every arm
	//    of the group uses the identical rule, so the comparison stays fair; it
	//    is simply not a matched-objective race, and the row says so by
	//    declaring endpoint "none" rather than borrowing a name it has not
	//    earned. Model quality is then checked directly, through the pooled
	//    out-of-fold and locked-test ROC areas the arm reports.
	bool autoStop;
	bool minStop;     // is the matched objective armed at all?

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
	string rev, sourceId, engineId, buildType;
	unsigned sourceFiles;   // how many source files the identity covers
	unsigned engineFiles;   // how many src/ files the ENGINE identity covers
	bool dirty;
	string caseName, group, groupAxis, fixture, splitId, model, arch, loss;
	// dataId is the content of the RAW dataset, BEFORE any split: which
	//    observations. splitId is the content of the TRAINING matrix after the
	//    split and its scaling: which rows, in which order, on which scale. They
	//    are separate questions -- two arms can share observations and still be
	//    trained on different rows -- so both are reported.
	string dataId;
	unsigned dataSeed;
	double testFraction;
	unsigned rowsTotal, rowsTest; // rows = TRAINING rows, as it always did
	string endpoint, timingScope, workload;
	unsigned cvFolds, cvRepeats;
	unsigned rows, inputs, params, weightSeed;
	// The parameter-state identity: what an arm is actually compared on.
	bool weightIdAvailable;
	string weightStartId, weightEndId, weightIdNote;
	unsigned weightElements;
	// Secondary, diagnostic: the function the weights compute on the inputs.
	string functionStartId, functionEndId;
	unsigned optimizer;
	unsigned lbfgsMemory;
	// THE METHOD, as one name. "canonical fixed eta" and "canonical automatic
	//    step size" are the same trainingType and are NOT the same method: the
	//    search costs maxLoops extra full passes per iteration and can land on a
	//    different step. The brief compares four methods, so a row names the
	//    method rather than making every reader recombine two fields.
	string method;
	string optimizerName, mode;
	double eta;
	bool autoStep, decayOn;
	double decay;
	bool gradStop, autoStop, minStop;
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
	// The HELD-OUT reading at the end of the run: the quality the endpoint
	//    definitions are ultimately about. -1 when there is no held-out set.
	//    Diagnostic -- it is never a stopping rule and never enters `usable`,
	//    because a rule that read it would be selecting on the test set.
	double heldoutError;
	// Workflow (cv) arms only; -1 otherwise.
	double cvAuc;      // pooled out-of-fold ROC area over all repetitions
	double lockedAuc;  // the locked refit's ROC area on the untouched test set
	unsigned cvFoldsOk, cvFoldsTotal;
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
	case 3: return "lbfgs";
	case 4: return "irprop";
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
	o << jstr( "engine_id", r.engineId );
	o << ",\"engine_files\":" << r.engineFiles;
	o << jstr( "build", r.buildType );
	o << jstr( "case", r.caseName );
	o << jstr( "comparison_group", r.group );
	o << jstr( "group_axis", r.groupAxis );
	o << jstr( "fixture", r.fixture );
	o << jstr( "split", r.splitId );
	o << jstr( "data_id", r.dataId );
	o << ",\"data_seed\":" << r.dataSeed;
	o << ",\"test_fraction\":" << jnum( r.testFraction );
	o << jstr( "endpoint", r.endpoint );
	o << jstr( "timing_scope", r.timingScope );
	o << jstr( "workload", r.workload );
	o << ",\"cv_folds\":" << r.cvFolds;
	o << ",\"cv_repeats\":" << r.cvRepeats;
	o << jstr( "model", r.model );
	o << jstr( "arch", r.arch );
	o << jstr( "loss", r.loss );
	o << ",\"rows\":" << r.rows;
	o << ",\"rows_total\":" << r.rowsTotal;
	o << ",\"rows_test\":" << r.rowsTest;
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
	o << jstr( "method", r.method );
	o << ",\"lbfgs_memory\":" << r.lbfgsMemory;
	o << jstr( "optimizer_name", r.optimizerName );
	o << jstr( "mode", r.mode );
	o << ",\"eta\":" << jnum( r.eta );
	o << ",\"auto_step\":" << ( r.autoStep ? "true" : "false" );
	o << ",\"decay_on\":" << ( r.decayOn ? "true" : "false" );
	o << ",\"decay\":" << jnum( r.decay );
	o << ",\"grad_stop\":" << ( r.gradStop ? "true" : "false" );
	o << ",\"auto_stop\":" << ( r.autoStop ? "true" : "false" );
	o << ",\"min_stop\":" << ( r.minStop ? "true" : "false" );
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
	o << ",\"heldout_error\":" << jnum( r.heldoutError );
	o << ",\"cv_auc\":" << jnum( r.cvAuc );
	o << ",\"locked_auc\":" << jnum( r.lockedAuc );
	o << ",\"cv_folds_ok\":" << r.cvFoldsOk;
	o << ",\"cv_folds_total\":" << r.cvFoldsTotal;
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

// THE TWO READINGS CHARACTERIZATION NEEDS, behind an interface, because both are
//    PROTECTED on Network and only a subclass can take them. Declaring the seam
//    explicitly is better than templating the recorder on the model: it says in
//    one place exactly what the trajectory recorder is allowed to touch, and it
//    is read-only.
struct Sampler {
	virtual ~Sampler() { }
	virtual double sampleGradMax() = 0;   // the engine's own convergence measure
	virtual double sampleHeldoutError() = 0; // -1 when there is no held-out set
};

template < class NET >
class Probe : public NET, public Sampler {
public:
	Probe() : innerCalls( 0 ), outerCalls( 0 ), evaluationCalls( 0 ),
		injectTraining( false ) { }

	double sampleGradMax() override { return this->getGradMax(); }
	double sampleHeldoutError() override { return this->sampleTestError( 1 ); }

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

	// EVERY FULL TRAVERSAL AN OPTIMIZER MAKES THROUGH THE PACKED BOUNDARY.
	//
	//    innerTrainSet() above counts a traversal for every method that makes
	//    exactly one per call, which was all of them until L-BFGS. A Wolfe
	//    line search makes SEVERAL per iteration -- one per trial point -- and
	//    they all go through batchObjectiveGradient(). Counting them here, in
	//    the same override idiom, is what stops an optimizer from appearing to
	//    win because its extra passes were invisible. The plan is explicit:
	//    a lower outer-iteration count is not a win.
	//
	//    Zero for canonical, CGD and Shanno: their batch path calls the
	//    non-virtual batchGradient() directly and never enters this. So the
	//    reported pass count for those arms is unchanged, bit for bit, from
	//    the committed Step 0B campaign.
	double batchObjectiveGradient( vector< double >& g ) override
	{
		evaluationCalls++;
		return NET::batchObjectiveGradient( g );
	}

	// The one L-BFGS knob the screen varies. lbfgs is Network's protected
	//    member, so a subclass reaches it -- no public setter is added to the
	//    engine for a research-only comparison.
	void setLBFGSMemory( const unsigned m ) { this->lbfgs.setMemory( m ); }

	unsigned innerCalls, outerCalls;
	unsigned long long evaluationCalls;
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

//   well4    a 4-input threshold problem whose inputs all share one scale.
//   poor4    THE SAME PROBLEM, with the inputs on scales spanning 1000x. Phase
//            4's poorly-scaled fixture, deferred by Step 0B to "the phase that
//            has a candidate to discriminate" -- iRPROP+'s hypothesized
//            advantage is exactly poorly scaled objectives.
//
// WHY THE SCALING IS BUILT THIS WAY, and it is not the obvious way. The obvious
// construction -- multiply each input column by a different constant -- produces
// a fixture BIT-IDENTICAL to the well-scaled one. DataSet::normalize
// (src/dataset.cpp:692) min-max normalizes every input column onto
// [inLowerLimit, inUpperLimit], so a per-column LINEAR rescale of the raw data
// is exactly cancelled before training ever sees it, and the arm would have
// reported "no difference on poorly scaled data" from a fixture that was not
// poorly scaled.
//
// What survives that normalization is where the BULK of a column sits inside its
// own range. So each column keeps anchors at -1 and +1 -- fixing what min-max
// maps to the endpoints -- while its remaining values are compressed by 10^-j.
// After normalization column 0 spans the full [-0.9, 0.9] and column 3 occupies
// about +/-0.0009 of it, so the weight on input 3 must be roughly 1000x the
// weight on input 0 to contribute equally. That is genuine ill-conditioning, and
// it is what the two fixtures differ by: same rows, same outcome rule, same
// count, different conditioning. The pair is checked for non-vacuity by
// requiring their split identities to differ (check_optimizer_harness.cpp).
static inline Matrix< double > fixtureMatrix( const string& which, unsigned n )
{
	if ( which == "well4" || which == "poor4" )
	{
		const bool poor = ( which == "poor4" );
		const unsigned primes[ 4 ] = { 37, 53, 71, 89 };
		Matrix< double > raw( n, 5 );
		for ( unsigned i = 0; i < n; i++ )
		{
			// Rows 0 and 1 anchor every column's min and max at -1 and +1, so
			//    normalization has a fixed range to map and the compression
			//    below is not undone by it.
			double base[ 4 ];
			for ( unsigned j = 0; j < 4; j++ )
			{
				if ( i == 0 ) base[ j ] = -1.0;
				else if ( i == 1 ) base[ j ] = 1.0;
				else base[ j ] = -1.0
					+ 2.0 * ( ( i * primes[ j ] ) % 100 ) / 99.0;

				double scale = 1.0;
				if ( poor && i > 1 )
					for ( unsigned k = 0; k < j; k++ )
						scale *= 0.1;
				raw( i, j ) = base[ j ] * scale;
			}
			// THE OUTCOME IS THE SAME FUNCTION OF THE SAME UNDERLYING VALUES in
			//    both fixtures, so the two arms are solving one problem at two
			//    conditionings rather than two different problems.
			raw( i, 4 ) = ( base[ 0 ] + base[ 1 ] + base[ 2 ] + base[ 3 ] > 0.55 )
				? 1 : 0;
		}
		return raw;
	}

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

// Where the prepared Step 0B datasets live. Compiled in rather than resolved
//    relative to the working directory, so a campaign launched from anywhere
//    reads the same files -- and so a MISSING file is a clear refusal naming the
//    path rather than a silently empty dataset.
#ifndef OPTBENCH_DATA_DIR
#define OPTBENCH_DATA_DIR "."
#endif

static inline string dataPath( const string& file )
{
	// An ABSOLUTE path passes through. The deterministic gate writes its own
	//    tiny fixture to a temporary file and names it absolutely, so proving
	//    the split mechanics does not require the prepared Civic Choice data to
	//    exist -- a ctest case that depends on a generated directory is a ctest
	//    case that fails on a fresh clone. Use path::is_absolute so a Windows
	//    drive-letter path (C:\...) is not rewritten under OPTBENCH_DATA_DIR;
	//    a leading-'/' check alone is Unix-only and broke the Windows CI gate.
	if ( std::filesystem::path( file ).is_absolute() )
		return file;
	return string( OPTBENCH_DATA_DIR ) + "/" + file;
}

// TWO IDENTITIES, TAKEN AT TWO MOMENTS, because they answer two questions.
//
//    dataId   the RAW matrix as loaded, before anything is split or scaled:
//             WHICH OBSERVATIONS.
//    splitId  the TRAINING matrix after the split and its scaling: WHICH ROWS,
//             on WHICH SCALE. Two arms can share every observation and still
//             train on different rows, so sharing dataId is not sharing a split.
//
// The split is REAL now: util::set_seed( dataSeed ) then DataSet::randomizeD,
// the stratified holdout the maintained recipe uses (menu 5 / /api/load with a
// fraction). Step 0A had no holdout and therefore no honest data seed; the
// schema carried one anyway and it was removed as false provenance. It returns
// here because it now selects something.
static inline DataSet makeDataSet( const Case& c, unsigned long long& dataId,
	unsigned long long& splitId, unsigned& rowsTotal, unsigned& rowsTest )
{
	DataSet d;
	d.setOutput( 1 );
	d.setDiscrete( true );
	d.setHistory( false );
	util::ScreenCapture hush;

	if ( c.dataFile.empty() )
	{
		Matrix< double > raw = fixtureMatrix( c.fixture, c.rows );
		// The FIXTURE decides its input count, read from the matrix it built
		//    rather than hard-coded here: a four-input fixture declared through a
		//    two-input constant would train on a design nobody described.
		d.setInput( raw.cols() - 1 );
		d.setRawMatrix( raw );
		dataId = matrixIdentity( raw );
	}
	else
	{
		d.setInput( c.inputs );
		string path = dataPath( c.dataFile );
		d.loadRaw( path );
		if ( !d.rawLoaded() )
			throw runtime_error( "could not load prepared dataset '" + path
				+ "'. Run tests/optimizer/prepare_data.py." );
		dataId = matrixIdentity( d.getRawMatrix() );
	}

	rowsTotal = d.getRawMatrix().rows();

	// A cv arm does NOT pre-split here: cross-validation materializes its own
	//    folds from Raw, and the locked test is carved by the same planner the
	//    GUI uses. Splitting twice would give it a training set nothing trains on.
	if ( c.workload == "cv" )
	{
		rowsTest = 0;
		splitId = dataId;
		return d;
	}

	if ( c.testFraction > 0.0 )
	{
		util::set_seed( c.dataSeed );
		if ( !d.randomizeD( c.testFraction ) )
			throw runtime_error( "the stratified holdout split failed" );
	}
	else
	{
		// No holdout: train on everything. The Step 0A behavior, retained so the
		//    mechanics fixtures keep measuring exactly what they measured.
		Matrix< double >& raw = d.getRawMatrix();
		d.setTrainMatrix( raw );
	}

	rowsTest = d.getNumTest();
	splitId = matrixIdentity( d.getTrainMatrix() );
	return d;
}

// ---------------------------------------------------------------------------
// Architecture, per model. Overloads rather than a switch: each model's
// architecture call has its own type, and the compiler should say so.

// Taken by CONCRETE MODEL reference rather than by Probe<>, so the same four
//    overloads serve both the instrumented arm and the plain template network a
//    cv workload clones -- cloneNetwork dispatches on typeid, so a cv template
//    cannot be a Probe. One definition of "what architecture does this case
//    ask for" (rule 6): a second one is how a cv arm comes to benchmark a
//    different network than the fit arm it is compared against.
static inline void arch( Logistic&, const Case& ) { }
static inline void arch( SimpleProp& p, const Case& c ) { p.setHidden( c.hidden ); }
static inline void arch( BareProp& p, const Case& c ) { p.setHidden( c.hidden ); }
static inline void arch( BackProp& p, const Case& c ) { p.setHidden( c.layers ); }

// THE TRAINING CONFIGURATION A CASE ASKS FOR, in one place, for the same reason.
//    A cv template is configured by this and then cloned per fold, so every fold
//    runs the optimizer, loss, step rule and matched endpoint the case declares.
template < class NET >
static inline void configure( NET& p, const Case& c )
{
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
	p.setMinStop( c.minStop );
	p.setMinError( c.target );
	p.setChangeStop( false );
	p.setWindowStop( false );
	p.setGradStop( c.gradStop );
	p.setGradMaxLimit( 0.0 ); // armed branch, but a rule that can never fire
	// Iterative's own defaults (iterative.cpp:35-36), for the same reason the
	//    strict endpoint uses its gradient limit: the project already decided
	//    what "stopped improving" means.
	p.setAutoStop( c.autoStop, 1e-4, 100 );
	p.setMaxIterations( c.ceiling );
}

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
	if ( c.dataFile.empty() && c.fixture != "linear2" && c.fixture != "xor2"
		&& c.fixture != "well4" && c.fixture != "poor4" )
		return "fixture: unknown generated fixture '" + c.fixture + "'";
	if ( !c.dataFile.empty() && c.inputs < 1 )
		return "inputs: a file-backed fixture must declare its input count";
	if ( !std::isfinite( c.testFraction ) || c.testFraction < 0.0
		|| c.testFraction >= 1.0 )
		return "test_fraction: must be finite and within [0,1)";
	if ( c.endpoint != ENDPOINT_PRACTICAL && c.endpoint != ENDPOINT_STRICT
		&& c.endpoint != ENDPOINT_NONE )
		return "endpoint: must be practical, strict or none";
	if ( c.timingScope != SCOPE_OPTIMIZER && c.timingScope != SCOPE_WORKFLOW )
		return "timing_scope: must be optimizer or workflow";
	if ( c.workload != "fit" && c.workload != "cv" )
		return "workload: must be fit or cv";
	// A CV arm's clock NECESSARILY covers each fold's scoring epilogue -- that is
	//    where a fold's held-out predictions come from -- so it cannot be an
	//    optimizer-only timing. Refusing the combination is what stops a
	//    whole-workflow number from being labelled as one.
	if ( c.workload == "cv" && c.timingScope != SCOPE_WORKFLOW )
		return "timing_scope: a cv workload is workflow scope, never optimizer";
	if ( c.workload == "cv" && c.cvFolds < 2 )
		return "cv_folds: a cv workload needs at least 2 folds";
	if ( c.workload == "cv" && c.cvRepeats < 1 )
		return "cv_repeats: a cv workload needs at least 1 repetition";
	if ( c.workload == "cv" && !( c.testFraction > 0.0 ) )
		return "test_fraction: a cv workload needs a locked test set to refit onto";
	// A FIT MUST HAVE A STOPPING RULE THAT CAN FIRE. Without one every fold
	//    runs to its ceiling, which the convergence contract calls a failure --
	//    so an arm with no armed rule cannot produce a result, only a wait.
	if ( !c.minStop && !c.autoStop )
		return "stopping: no rule is armed that can fire (min_error or auto_stop)";
	// An arm not racing to an objective must not claim an objective endpoint.
	if ( !c.minStop && c.endpoint != ENDPOINT_NONE )
		return "endpoint: an arm with no objective target cannot declare the '"
			+ c.endpoint + "' endpoint";
	if ( c.workload != "cv" && ( c.cvFolds || c.cvRepeats ) )
		return "cv_folds/cv_repeats: only a cv workload may set these";
	if ( c.optimizer > Network::TRAIN_IRPROP )
		return "optimizer: must be 0 (canonical), 1 (CGD), 2 (Shanno), "
			"3 (L-BFGS) or 4 (the research-only iRPROP+ prototype)";
	// L-BFGS OWNS ITS OWN STEP AND ITS OWN GRADIENT. It refuses the automatic
	//    step-size search and on-line mode in the engine; declaring either here
	//    would produce an arm that throws at its first pass rather than a row
	//    that says what is wrong with the case.
	if ( c.optimizer == Network::TRAIN_LBFGS && c.autoStep )
		return "auto_step: L-BFGS chooses its own step and cannot run with the "
			"automatic step-size search";
	if ( c.optimizer == Network::TRAIN_LBFGS && !c.batch )
		return "mode: L-BFGS requires batch/epoch training";
	if ( c.optimizer == Network::TRAIN_LBFGS && c.lbfgsMemory < 1 )
		return "lbfgs_memory: must be at least 1";
	if ( c.optimizer == Network::TRAIN_LBFGS && c.model == "logistic" )
		return "model: Logistic does not implement the packed parameter "
			"boundary L-BFGS needs";
	// iRPROP+ REFUSES THE SAME THREE CONFIGURATIONS, for its own reasons: its
	//    step is absolute so a second step rule cannot own the iteration, the
	//    published method compares one full-batch gradient with the previous
	//    one, and it reads and writes parameters only through the packed
	//    boundary. Declared here so a bad case produces a row that says what is
	//    wrong rather than an arm that throws at its first pass.
	if ( c.optimizer == Network::TRAIN_IRPROP && c.autoStep )
		return "auto_step: iRPROP+ owns its own absolute step and cannot run "
			"with the automatic step-size search";
	if ( c.optimizer == Network::TRAIN_IRPROP && !c.batch )
		return "mode: iRPROP+ requires batch/epoch training";
	if ( c.optimizer == Network::TRAIN_IRPROP && c.model == "logistic" )
		return "model: Logistic does not implement the packed parameter "
			"boundary iRPROP+ needs";
	if ( c.dataFile.empty() && c.rows < 2 )
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
	r.engineId = OPTBENCH_ENGINE_ID;
	r.engineFiles = OPTBENCH_ENGINE_COUNT;
	r.buildType = buildIdentity();
	r.caseName = c.name;
	r.group = c.group;
	r.groupAxis = c.groupAxis;
	r.fixture = c.dataFile.empty() ? c.fixture : c.dataFile;
	r.splitId = "";
	r.dataId = "";
	r.dataSeed = c.dataSeed;
	r.testFraction = c.testFraction;
	r.rowsTotal = r.rowsTest = 0;
	r.endpoint = c.endpoint;
	r.timingScope = c.timingScope;
	r.workload = c.workload;
	r.cvFolds = c.cvFolds;
	r.cvRepeats = c.cvRepeats;
	r.heldoutError = -1;
	r.cvAuc = r.lockedAuc = -1;
	r.cvFoldsOk = r.cvFoldsTotal = 0;
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
	r.method = string( optimizerName( c.optimizer ) )
		+ ( c.autoStep ? "-autostep" : "" );
	r.mode = c.batch ? "batch" : "online";
	r.eta = c.eta;
	r.autoStep = c.autoStep;
	r.decayOn = c.decayOn;
	r.decay = c.decay;
	r.gradStop = c.gradStop;
	r.lbfgsMemory = c.lbfgsMemory;
	r.autoStop = c.autoStop;
	r.minStop = c.minStop;
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
	unsigned long long dataId = 0, splitId = 0;

	// --- setup, outside the timed region -----------------------------------
	try
	{
		DataSet d = makeDataSet( c, dataId, splitId, r.rowsTotal, r.rowsTest );
		r.dataId = hex64( dataId );
		r.splitId = hex64( splitId );
		r.rows = d.getNumTrain();
		r.inputs = d.getInput();

		if ( c.inject == Case::INJECT_SETUP )
			throw runtime_error( "benchmark-only injected setup fault" );

		p.setDataSet( d );
		arch( p, c );
		configure( p, c );
		if ( c.optimizer == Network::TRAIN_LBFGS )
			p.setLBFGSMemory( c.lbfgsMemory );

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
	p.evaluationCalls = 0;
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
	// FULL PASSES MEANS FULL TRAVERSALS OF THE TRAINING SET, which is the
	//    currency a speed comparison is denominated in. For every method that
	//    performs exactly one traversal per innerTrainSet() call -- canonical,
	//    canonical-autostep, CGD, Shanno -- that is the inner-call count, and
	//    this is unchanged from Step 0B.
	//
	//    For L-BFGS it is NOT. Its innerTrainSet() traverses nothing itself: it
	//    delegates, and every traversal is a batchObjectiveGradient() call, one
	//    per trial point of the line search. The test is what the run actually
	//    did, not which model it is -- an arm that made evaluation calls is an
	//    arm whose traversals are counted there.
	r.fullPasses = p.evaluationCalls > 0
		? ( long long ) p.evaluationCalls
		: ( long long ) p.innerCalls;
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

	// THE HELD-OUT READING, taken AFTER the clock stopped and used by nothing
	//    that decides the run. It is what the practical endpoint is ultimately
	//    about, so a row that reached its endpoint fast can be checked against
	//    the model it actually produced. Deliberately NOT part of `usable` and
	//    deliberately not a stopping rule: a rule reading it would be selecting
	//    on held-out data. -1 when there is no held-out set to sample.
	r.heldoutError = p.sampleTestError( 1 );

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

// ---------------------------------------------------------------------------
// THE REPEATED-FIT CONSUMER: repeated cross-validation plus the locked refit.
//
// This is the operation the intended workflow actually spends its hours in, and
// it is the reason the whole program cares about single-fit speed: a k-fold,
// r-repetition comparison pays for k*r fits plus one, so a per-fit saving is
// multiplied by k*r+1 before the user sees it.
//
// IT IS A WORKFLOW MEASUREMENT AND SAYS SO. The clock covers each fold's fit AND
// the scoring epilogue that produces its held-out predictions, because a fold
// with no predictions is not a fold. Comparing this number against an
// optimizer-only one would be comparing two different jobs, so validate()
// refuses a cv arm that claims optimizer scope and the runner keeps the two
// scopes in different comparison groups.
//
// THE POLICY IS THE MAINTAINED ONE, not a benchmark invention: a stratified
// locked holdout (nsplit::stratifiedHoldout), cross-validation over the
// DEVELOPMENT rows only, k-fold stratified on the outcome (nsplit::kFold), and
// the locked refit through crossval::evaluateOnce -- the same sequence
// gui.cpp's /api/cv performs. Reimplementing it here would benchmark a
// procedure no user runs.
//
// The convergence contract is inherited, not re-stated: cvadapters::
// trainProcedure already fails any fold whose fit ended at the iteration
// ceiling, so an unreachable matched endpoint makes this arm FAIL rather than
// return a fast time for a set of models nobody would use.

// The 0/1 outcome of every row of a raw matrix -- the label a stratified
//    planner balances on. Written once because three callers need it and a
//    second spelling of "which column is the outcome" is a defect waiting.
static inline vector< unsigned > outcomeLabels( const Matrix< double >& raw )
{
	unsigned n = raw.rows(), outCol = raw.cols() - 1;
	vector< unsigned > label( n );
	for ( unsigned i = 0; i < n; i++ )
		label[ i ] = ( raw( i, outCol ) != 0 ) ? 1u : 0u;
	return label;
}

template < class NET >
static Row runCv( const Case& c, const string& rev )
{
	Row r;
	describe( r, c, rev );

	DataSet data;
	DataSet devData;
	vector< unsigned > devRows, lockedRows;
	NET templateNet;

	try
	{
		unsigned long long dataId = 0, splitId = 0;
		data = makeDataSet( c, dataId, splitId, r.rowsTotal, r.rowsTest );
		r.dataId = hex64( dataId );
		r.inputs = data.getInput();

		util::ScreenCapture hush;
		Matrix< double >& raw = data.getRawMatrix();
		unsigned n = raw.rows();
		vector< unsigned > label = outcomeLabels( raw );

		// The locked test set, carved once and never touched by any fold.
		util::set_seed( c.dataSeed );
		nsplit::Holdout h = nsplit::stratifiedHoldout( label,
			( unsigned ) ( c.testFraction * ( double ) n ) );
		devRows = h.train;
		lockedRows = h.test;
		string bad = nsplit::partitionError( n, devRows, lockedRows, true );
		if ( !bad.empty() )
			throw runtime_error( "the locked-test split is not a valid partition: "
				+ bad );

		// Cross-validation sees the development rows ONLY.
		Matrix< double > devRaw = raw.includerows( devRows );
		devData = data;
		devData.setRawMatrix( devRaw );

		r.rows = devRaw.rows();
		r.rowsTest = ( unsigned ) lockedRows.size();
		r.splitId = hex64( matrixIdentity( devRaw ) );

		// The template every fold clones. Its architecture, loss, optimizer and
		//    matched endpoint are the case's, through the SAME two helpers the
		//    fit arms use -- so a cv arm and a fit arm on one case describe one
		//    configuration.
		arch( templateNet, c );
		configure( templateNet, c );
		templateNet.setDataSet( devData );
		util::set_seed( c.weightSeed );
		templateNet.randomize();
		r.params = templateNet.df();

		// NO WEIGHT IDENTITY IS CLAIMED. Every fold randomizes its own fresh
		//    weights inside the procedure (that is what an honest per-fold fit
		//    does), so there is no single starting parameter state for this arm
		//    and reporting the template's would name a state nothing trained
		//    from. An absent identity is refused by the runner for optimizer
		//    groups; a workflow group is compared on its declared axis instead.
		r.weightIdAvailable = false;
		r.weightIdNote = "a cv arm has no single starting state: each fold "
			"randomizes its own weights inside the procedure";

		if ( c.inject == Case::INJECT_SETUP )
			throw runtime_error( "benchmark-only injected setup fault" );
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

	double aucSum = 0;
	unsigned aucCount = 0;
	bool everyFoldOk = true, lockedOk = false;

	try
	{
		util::ScreenCapture hush;
		chrono::steady_clock::time_point t0 = chrono::steady_clock::now();

		for ( unsigned rep = 0; rep < c.cvRepeats; rep++ )
		{
			vector< crossval::ProcedureSpec > procs( 1 );
			procs[ 0 ].name = c.name;
			procs[ 0 ].proc = cvadapters::trainProcedure( templateNet, c.ceiling );

			// Each repetition is a DIFFERENT fold plan of the same development
			//    rows -- that is what repetition means. The locked test is not
			//    re-drawn: it is locked.
			util::set_seed( c.dataSeed + 1000u + rep );
			vector< unsigned > repFolds = nsplit::kFold(
				outcomeLabels( devData.getRawMatrix() ), c.cvFolds );

			crossval::Comparison cmp = crossval::compare( devData, repFolds, procs,
				nullptr, true, c.dataSeed + 1000u + rep );
			if ( !cmp.ok || cmp.entries.empty() )
				throw runtime_error( "cross-validation refused: " + cmp.message );

			const crossval::RunResult& rr = cmp.entries[ 0 ].result;
			r.cvFoldsTotal += ( unsigned ) rr.folds.size();
			r.cvFoldsOk += rr.validFolds;
			if ( rr.validFolds != rr.folds.size() )
				everyFoldOk = false;
			if ( rr.oofTrap >= 0 ) { aucSum += rr.oofTrap; aucCount++; }
		}

		// The locked refit: one final fit on every development row, scored once
		//    on the untouched locked test set.
		{
			vector< crossval::ProcedureSpec > procs( 1 );
			procs[ 0 ].name = c.name;
			procs[ 0 ].proc = cvadapters::trainProcedure( templateNet, c.ceiling );
			crossval::LockedResult lr = crossval::evaluateOnce( data, devRows,
				lockedRows, procs, nullptr, true, c.dataSeed );
			if ( lr.ok && !lr.entries.empty() && lr.entries[ 0 ].ok )
			{
				lockedOk = true;
				vector< unsigned > all( lr.outcome.size() );
				for ( size_t i = 0; i < all.size(); i++ ) all[ i ] = ( unsigned ) i;
				crossval::Metrics m = crossval::metricsFor( lr.outcome,
					lr.entries[ 0 ].pred, all );
				r.lockedAuc = m.trap;
			}
			else if ( !lr.entries.empty() )
				r.error = "locked refit failed: " + lr.entries[ 0 ].reason;
			else
				r.error = "locked refit refused: " + lr.message;
		}

		chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
		r.elapsedNs = chrono::duration_cast< chrono::nanoseconds >( t1 - t0 ).count();
	}
	catch ( const exception& e )
	{
		r.failureStage = "training";
		r.error = string( "exception during training: " ) + e.what();
		return r;
	}
	catch ( ... )
	{
		r.failureStage = "training";
		r.error = "unknown exception during training";
		return r;
	}

	r.peakRssKb = peakRssKb();
	if ( aucCount ) r.cvAuc = aucSum / aucCount;

	// A cv arm reports NO iteration or pass count. Those are per-fit facts and
	//    this arm ran k*repeats+1 fits; summing them would invite a per-pass
	//    ratio that describes no single training run. Left null rather than
	//    filled with a number that means something else.
	r.iterationsCompleted = -1;
	r.fullPasses = -1;
	r.stopReason = everyFoldOk && lockedOk ? "cv_complete" : "cv_incomplete";
	r.finite = std::isfinite( r.cvAuc ) && r.cvAuc >= 0;
	r.converged = everyFoldOk && lockedOk;
	r.targetReached = r.converged;
	r.achieved = r.cvAuc;

	// USABLE MEANS THE WHOLE CONSUMER SUCCEEDED. Every fold of every repetition
	//    produced predictions -- which, by the adapter's own contract, means
	//    every fold's fit CONVERGED to the matched endpoint -- and the locked
	//    refit produced a scored model. A partial run is retained and reported,
	//    never averaged.
	r.usable = r.finite && everyFoldOk && lockedOk
		&& r.cvFoldsTotal == c.cvFolds * c.cvRepeats
		&& r.cvFoldsOk == r.cvFoldsTotal
		&& r.lockedAuc >= 0;

	if ( !r.usable && r.error.empty() )
		r.error = "cross-validation completed " + to_string( r.cvFoldsOk )
			+ " of " + to_string( r.cvFoldsTotal ) + " folds";

	return r;
}

// ---------------------------------------------------------------------------
// ENDPOINT CHARACTERIZATION: where the two predeclared targets come from.
//
// Run the CANONICAL REFERENCE arm alone, with an unreachable target so it runs
// its whole ceiling, and record two series per iteration: the training
// objective, and the held-out error. Then:
//
//   strict     the training objective canonical reached at its floor. The
//              late-stage failure detector.
//   practical  the training objective canonical had reached at the iteration
//              where the HELD-OUT error stopped improving. Past that point more
//              training buys the usable model nothing, so reaching it fast is
//              what makes a real workload tractable.
//
// THE PLATEAU IS THE ENGINE'S OWN DETECTOR, NOT A RULE INVENTED HERE.
// PlateauDetector (src/plateau.h) already owns "has this series stopped
// improving" for the whole project -- it is what setAutoStop uses -- and it is
// used here at its own default window, tolerance and patience. Two reasons, and
// the second is the load-bearing one:
//
//   1. rule 6. A second definition of "plateaued" would be a second
//      implementation of the one thing the project already decided.
//   2. IT IS HORIZON-INDEPENDENT, and the first version of this was not. That
//      version took the best held-out error over the WHOLE run and asked when
//      the series first came within 1% of it -- so a longer characterization
//      found a better best, moved the band, and moved the endpoint. Measured:
//      the same neural workload put its practical endpoint at iteration 11,299
//      under a 20,000 ceiling and at 78,764 under a 100,000 one. An endpoint
//      that moves when you watch it longer is not an endpoint. The detector
//      fires where the series flattens and does not care what happens after.

// THE STRICT ENDPOINT IS A CONVERGED RUN, NOT A CEILING. Running canonical with
//    an unreachable min_error and calling wherever it stopped "the floor" would
//    violate the convergence contract in the one place it matters most: the
//    ceiling is a failure to converge, never a stopping condition, so a target
//    derived from an exhausted ceiling describes an unfinished fit.
//
//    So characterization arms the engine's own mathematical convergence rule --
//    STOP_GRADMAX, the maximum absolute gradient -- at a limit declared here,
//    and REFUSES to publish a strict endpoint for a run that ended at the
//    ceiling anyway. A characterization that cannot converge is a stop
//    condition to report, not a number to round off.
//    THE LIMIT IS THE ENGINE'S OWN DEFAULT, not a number chosen here.
//    Iterative constructs with gradMaxFlag on and gradMaxLimit 1e-6
//    (iterative.cpp:33-35), so "converged" already has a shipped meaning in this
//    project and the strict endpoint uses it rather than inventing a second one.
static const double STRICT_GRADMAX = 1e-6;

// Likewise the ceiling: Iterative's own default maxIterations. It is a SAFETY
//    LIMIT, and a characterization that reaches it is reported as a failure to
//    converge, never rounded off into a floor.
static const unsigned STRICT_CEILING = 1000000;

// THE GUARD RUNS SHORT, AND SAYS SO. Proving that watching the held-out set does
//    not change the fit is a proof about a MECHANISM -- sampling calls forward()
//    on held-out rows and writes scratch the next training pass overwrites -- and
//    a mechanism that perturbed the fit would perturb it on the first iteration,
//    not the hundred-thousandth. So the two-run comparison uses a short ceiling
//    and the long characterization runs once. What is NOT claimed: that the
//    trajectories were compared out to the strict endpoint.
static const unsigned GUARD_ITERATIONS = 400;

// The strict TARGET is the converged objective raised by one part in a hundred
//    thousand, because reaching an endpoint is `achieved < target` and a target
//    set exactly at the floor is one the reference itself cannot satisfy.
static const double STRICT_HEADROOM = 1e-5;

// The trajectory recorder. It runs ONLY during characterization, never inside a
//    timed arm, so nothing it costs can appear in a benchmark number.
//
//    IT MUST NOT CHANGE THE FIT. Sampling the held-out set calls forward() on
//    held-out rows, which writes the network's scratch output -- exactly the
//    shape of legacy bug #10, where recalculating for a REPORT changed which
//    model a run produced. So characterization runs twice, with and without the
//    held-out sampling, and REFUSES unless the two objective trajectories are
//    bit-identical. A rationale is not a measurement (2026-08-02).
struct Trajectory : Iterative::Observer {
	Sampler* net;
	bool sampleHeldout;
	vector< double > objective, heldout, gradmax;
	Trajectory() : net( 0 ), sampleHeldout( false ) { }
	bool onIteration( unsigned, double setError ) override
	{
		objective.push_back( setError );
		// BOTH passes record the gradient, so it cannot confound the guard
		//    below: the only difference between the two passes is the held-out
		//    sampling, which is the thing being tested. getGradMax() is the same
		//    call the armed stopping rule already makes each iteration.
		if ( net ) gradmax.push_back( net->sampleGradMax() );
		if ( sampleHeldout && net )
			heldout.push_back( net->sampleHeldoutError() );
		return true;
	}
};

// WHERE THE PRACTICAL ENDPOINT IS, given the two series a characterization
//    recorded. Free-standing rather than inline in characterizeTyped, for the
//    same reason oneHiddenIdentity is: a rule that can only be exercised by
//    running a model for tens of thousands of iterations is a rule that gets
//    tested by eye. This one is a pure function of two vectors, so the gate can
//    hand it a series whose right answer is known and vary the horizon alone.
//
//    A held-out sample of -1 means "not sampled" and is skipped; the returned
//    iteration indexes the ORIGINAL series, so it still names a training
//    iteration.
struct PracticalPoint {
	bool fired;
	unsigned iteration;
	double objective;
	double heldout;
	PracticalPoint() : fired( false ), iteration( 0 ), objective( 0 ),
		heldout( -1 ) { }
};

static inline PracticalPoint practicalEndpoint( const vector< double >& objective,
	const vector< double >& heldout )
{
	PracticalPoint p;
	PlateauDetector det;   // the engine's own defaults
	for ( size_t i = 0; i < heldout.size() && i < objective.size(); i++ )
	{
		if ( heldout[ i ] < 0 ) continue;
		if ( det.update( heldout[ i ] ) )
		{
			p.fired = true;
			p.iteration = ( unsigned ) i;
			p.objective = objective[ i ];
			p.heldout = heldout[ i ];
			return p;
		}
	}
	return p;
}

struct Characterization {
	bool ok;
	string error;
	string caseName, model, dataId, splitId;
	unsigned rows, rowsTotal, rowsTest, inputs, params;
	unsigned iterations, ceiling, guardIterations;
	long long elapsedNs;
	bool practicalOk, strictOk, plateauFired;
	string stopReason;
	bool converged;          // did canonical reach the gradient rule, or the ceiling?
	double finalObjective;   // the floor canonical reached
	double bestHeldout;      // the best held-out error anywhere in the run
	double finalGradmax;     // the gradient the run actually reached
	double bestGradmax;      // the smallest gradient anywhere in the run
	unsigned practicalIteration; // first iteration within tolerance of it
	double practicalObjective;   // the training objective THERE
	double practicalHeldout;
	bool samplingWasFree;    // the two objective trajectories were identical
	Characterization() : ok( false ), rows( 0 ), rowsTotal( 0 ), rowsTest( 0 ),
		inputs( 0 ), params( 0 ), iterations( 0 ), ceiling( 0 ),
		guardIterations( 0 ), elapsedNs( 0 ),
		practicalOk( false ), strictOk( false ), plateauFired( false ),
		converged( false ),
		finalObjective( 0 ), bestHeldout( -1 ),
		finalGradmax( -1 ), bestGradmax( -1 ),
		practicalIteration( 0 ), practicalObjective( 0 ), practicalHeldout( -1 ),
		samplingWasFree( false ) { }
};

template < class NET >
static Characterization characterizeTyped( const Case& c, unsigned ceiling )
{
	Characterization ch;
	ch.caseName = c.name;
	ch.model = c.model;
	ch.ceiling = ceiling;

	// One prepared reference model. Written once and called three times, so the
	//    guard runs and the characterization run cannot drift apart -- if they
	//    could, the guard would be proving something about a different fit.
	struct Prepared {
		Probe< NET > net;
		DataSet data;
	};

	try
	{
		// --- the guard: the same short run twice, watched and unwatched -------
		vector< double > guardQuiet, guardWatched;
		for ( int pass = 0; pass < 2; pass++ )
		{
			Probe< NET > p;
			unsigned long long dataId = 0, splitId = 0;
			unsigned rt = 0, rs = 0;
			DataSet d = makeDataSet( c, dataId, splitId, rt, rs );
			p.setDataSet( d );
			arch( p, c );
			configure( p, c );
			p.setMinStop( false );
			p.setAutoStop( false, 1e-4, 100 );
			p.setGradStop( true );
			p.setGradMaxLimit( STRICT_GRADMAX );
			p.setMaxIterations( GUARD_ITERATIONS );
			util::set_seed( c.weightSeed );
			p.randomize();

			Trajectory t;
			t.net = &p;
			t.sampleHeldout = ( pass == 1 );
			p.setObserver( &t );
			{
				util::ScreenCapture hush;
				p.train();
			}
			p.setObserver( 0 );
			( pass ? guardWatched : guardQuiet ) = t.objective;
		}

		// If watching the held-out set moved the fit by a single bit, the
		//    trajectory it reports is not the trajectory the timed arms follow,
		//    and every endpoint derived from it would describe a different run.
		//    This is legacy bug #10's exact shape: a reporting action that
		//    changes which model a run produces.
		ch.guardIterations = ( unsigned ) guardQuiet.size();
		ch.samplingWasFree = !guardQuiet.empty() && ( guardQuiet == guardWatched );
		if ( !ch.samplingWasFree )
		{
			ch.error = guardQuiet.empty()
				? "the guard run completed no iterations"
				: "sampling the held-out set CHANGED the fit: the two objective "
				  "trajectories differ over " + to_string( guardQuiet.size() )
				  + " iterations. An endpoint derived from the watched run would "
				  "not describe the runs being timed (legacy bug #10).";
			return ch;
		}

		// --- the characterization: one long run, watched ---------------------
		Probe< NET > p;
		unsigned long long dataId = 0, splitId = 0;
		DataSet d = makeDataSet( c, dataId, splitId, ch.rowsTotal, ch.rowsTest );
		ch.dataId = hex64( dataId );
		ch.splitId = hex64( splitId );
		ch.inputs = d.getInput();

		p.setDataSet( d );
		arch( p, c );
		configure( p, c );
		p.setMinStop( false );
		p.setAutoStop( false, 1e-4, 100 );  // the reference runs to its own floor
		p.setGradStop( true );
		p.setGradMaxLimit( STRICT_GRADMAX );
		p.setMaxIterations( ceiling );
		util::set_seed( c.weightSeed );
		p.randomize();
		ch.params = p.df();
		ch.rows = d.getNumTrain();

		Trajectory t;
		t.net = &p;
		t.sampleHeldout = true;
		p.setObserver( &t );
		chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
		{
			util::ScreenCapture hush;
			p.train();
		}
		chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
		p.setObserver( 0 );
		ch.elapsedNs = chrono::duration_cast< chrono::nanoseconds >( t1 - t0 ).count();
		ch.stopReason = Iterative::stopReasonToken( p.getStopReason() );
		ch.converged = Iterative::converged( p.getStopReason() );

		if ( t.objective.empty() )
		{
			ch.error = "the canonical reference completed no iterations";
			return ch;
		}

		ch.iterations = ( unsigned ) t.objective.size();
		ch.finalObjective = t.objective.back();
		if ( !t.gradmax.empty() )
		{
			ch.finalGradmax = t.gradmax.back();
			ch.bestGradmax = t.gradmax[ 0 ];
			for ( size_t i = 1; i < t.gradmax.size(); i++ )
				if ( t.gradmax[ i ] < ch.bestGradmax ) ch.bestGradmax = t.gradmax[ i ];
		}

		// THE PRACTICAL ENDPOINT is derived whether or not the strict one is
		//    available: "when did the useful model arrive" is answerable from a
		//    run that has not yet reached its floor, and it is the endpoint the
		//    tractability question actually turns on.
		bool haveHeldout = false;
		for ( size_t i = 0; i < t.heldout.size(); i++ )
			if ( t.heldout[ i ] >= 0 ) { haveHeldout = true; break; }
		if ( !haveHeldout )
		{
			ch.error = "no held-out set: a practical endpoint cannot be derived "
				"without one (set test_fraction)";
			return ch;
		}

		ch.bestHeldout = -1;
		double watchedHeldoutBack = -1;
		for ( size_t i = 0; i < t.heldout.size(); i++ )
			if ( t.heldout[ i ] >= 0 )
			{
				watchedHeldoutBack = t.heldout[ i ];
				if ( ch.bestHeldout < 0 || t.heldout[ i ] < ch.bestHeldout )
					ch.bestHeldout = t.heldout[ i ];
			}

		// THE PLATEAU, by the engine's own detector at its own defaults,
		//    through the one function that owns the rule.
		PracticalPoint pp = practicalEndpoint( t.objective, t.heldout );
		ch.plateauFired = pp.fired;
		if ( pp.fired )
		{
			ch.practicalIteration = pp.iteration;
			ch.practicalHeldout = pp.heldout;
			ch.practicalObjective = pp.objective;
		}
		else
		{
			ch.practicalIteration = ch.iterations - 1;
			ch.practicalHeldout = watchedHeldoutBack;
			ch.practicalObjective = ch.finalObjective;
		}
		// A run that never plateaued has no practical endpoint to publish: the
		//    useful model had not stopped improving when the run ended, so the
		//    objective at the end is a budget, not a plateau.
		if ( !ch.plateauFired )
		{
			ch.error = "the held-out error had not plateaued when the run ended "
				"after " + to_string( ch.iterations ) + " iterations, so no "
				"practical endpoint is published: the last objective is where a "
				"budget ran out, not where the useful model stopped improving.";
			return ch;
		}

		ch.practicalOk = true;

		// THE CEILING IS NOT A FLOOR. A reference that exhausted its iteration
		//    ceiling did not converge, and the objective it is sitting on is
		//    where an unfinished fit happened to be. No strict endpoint is
		//    published for it -- the practical one above still is, and the row
		//    says which of the two it is offering.
		if ( !ch.converged )
		{
			ch.error = "the canonical reference did NOT converge (" + ch.stopReason
				+ ") within " + to_string( ch.iterations ) + " iterations; its "
				"gradient reached " + to_string( ch.finalGradmax ) + " against the "
				"engine's own " + to_string( STRICT_GRADMAX ) + " limit. No strict "
				"endpoint is published for this workload.";
			return ch;
		}

		ch.strictOk = true;
		ch.ok = true;
		return ch;
	}
	catch ( const exception& e )
	{
		ch.error = string( "exception during characterization: " ) + e.what();
		return ch;
	}
	catch ( ... )
	{
		ch.error = "unknown exception during characterization";
		return ch;
	}
}

static inline Characterization characterize( const Case& c, unsigned ceiling )
{
	Characterization bad;
	string why = validate( c );
	if ( !why.empty() ) { bad.error = "refused -- " + why; return bad; }
	if ( c.workload != "fit" )
	{
		bad.error = "refused -- characterize: only a fit workload has a "
			"training-objective trajectory to characterize";
		return bad;
	}
	if ( c.model == "logistic" )   return characterizeTyped< Logistic >( c, ceiling );
	if ( c.model == "simpleprop" ) return characterizeTyped< SimpleProp >( c, ceiling );
	if ( c.model == "bareprop" )   return characterizeTyped< BareProp >( c, ceiling );
	return characterizeTyped< BackProp >( c, ceiling );
}

static inline string toJsonLine( const Characterization& ch )
{
	ostringstream o;
	o << "{\"record\":\"characterization\"";
	// A CHARACTERIZATION IS EVIDENCE, so it identifies the binary that produced
	//    it exactly as an arm row does. It sets the targets every later arm is
	//    judged against; a target whose provenance cannot be traced to a
	//    specific engine is a target nobody can re-derive.
	o << ",\"schema\":" << SCHEMA_VERSION;
	o << jstr( "rev", OPTBENCH_GIT_REV );
	o << ",\"dirty\":" << ( OPTBENCH_GIT_DIRTY ? "true" : "false" );
	o << jstr( "source_id", OPTBENCH_SOURCE_ID );
	o << ",\"source_files\":" << ( unsigned ) OPTBENCH_SOURCE_COUNT;
	o << jstr( "engine_id", OPTBENCH_ENGINE_ID );
	o << ",\"engine_files\":" << ( unsigned ) OPTBENCH_ENGINE_COUNT;
	o << jstr( "build", buildIdentity() );
	o << jstr( "case", ch.caseName );
	o << jstr( "model", ch.model );
	o << jstr( "data_id", ch.dataId );
	o << jstr( "split", ch.splitId );
	o << ",\"rows\":" << ch.rows;
	o << ",\"rows_total\":" << ch.rowsTotal;
	o << ",\"rows_test\":" << ch.rowsTest;
	o << ",\"inputs\":" << ch.inputs;
	o << ",\"params\":" << ch.params;
	o << ",\"iterations\":" << ch.iterations;
	o << ",\"ceiling\":" << ch.ceiling;
	o << ",\"guard_iterations\":" << ch.guardIterations;
	o << ",\"elapsed_ns\":" << ch.elapsedNs;
	o << ",\"practical_ok\":" << ( ch.practicalOk ? "true" : "false" );
	o << ",\"strict_ok\":" << ( ch.strictOk ? "true" : "false" );
	o << jstr( "stop_reason", ch.stopReason );
	o << ",\"converged\":" << ( ch.converged ? "true" : "false" );
	o << ",\"plateau_fired\":" << ( ch.plateauFired ? "true" : "false" );
	o << ",\"strict_gradmax\":" << jnum( STRICT_GRADMAX );
	o << ",\"final_gradmax\":" << jnum( ch.finalGradmax );
	o << ",\"best_gradmax\":" << jnum( ch.bestGradmax );
	o << ",\"strict_objective\":" << jnum( ch.finalObjective );
	o << ",\"strict_target\":" << jnum( ch.finalObjective * ( 1.0 + STRICT_HEADROOM ) );
	o << ",\"practical_objective\":" << jnum( ch.practicalObjective );
	o << ",\"practical_iteration\":" << ch.practicalIteration;
	o << ",\"practical_heldout\":" << jnum( ch.practicalHeldout );
	o << ",\"best_heldout\":" << jnum( ch.bestHeldout );
	o << ",\"sampling_was_free\":" << ( ch.samplingWasFree ? "true" : "false" );
	o << ",\"ok\":" << ( ch.ok ? "true" : "false" );
	o << jstr( "error", ch.error );
	o << "}";
	return o.str();
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

	if ( c.workload == "cv" )
	{
		if ( c.model == "logistic" )   return runCv< Logistic >( c, rev );
		if ( c.model == "simpleprop" ) return runCv< SimpleProp >( c, rev );
		if ( c.model == "bareprop" )   return runCv< BareProp >( c, rev );
		return runCv< BackProp >( c, rev );
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
	c.dataFile = "";
	c.inputs = 0;
	c.dataSeed = 0;
	c.testFraction = 0.0;
	c.endpoint = ENDPOINT_NONE;
	c.timingScope = SCOPE_OPTIMIZER;
	c.workload = "fit";
	c.cvFolds = 0;
	c.cvRepeats = 0;
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
	c.lbfgsMemory = 5; // Liu & Nocedal (1989) section 5: 3 <= m <= 7
	c.autoStop = false;
	c.minStop = true;
	c.target = 0.35;
	c.ceiling = 4000;
	c.xentropy = true;
	c.inject = Case::INJECT_NONE;
	return c;
}

// ---------------------------------------------------------------------------
// THE COMMITTED TARGET TABLE (Step 0B).
//
// Every entry is a MEASUREMENT, produced by `optimizer_probe --characterize` on
// the canonical reference of that workload and then written down here. It is
// committed rather than recomputed at run time for one reason: a target
// recomputed by the campaign is not a control. If the engine changes, these
// values must be re-characterized deliberately and the change recorded --
// which is exactly the visibility a silently-recomputed target would remove.
//
// A `strict` value is the objective canonical reached at its ceiling, raised by
// one part in a thousand so the reference itself can satisfy `achieved <
// target`. A `practical` value is the objective canonical had reached when the
// held-out error first came within PRACTICAL_TOLERANCE of its best.
//
// UNSET (0) MEANS NOT YET CHARACTERIZED, and validate() refuses a target of 0,
// so a case referring to an uncharacterized workload cannot silently run
// against a meaningless endpoint.
struct Endpoints {
	const char* key;
	double practical;      // 0 = no practical endpoint is published
	double strict;         // 0 = NO STRICT ENDPOINT EXISTS; strictNote says why
	unsigned ceiling;      // the safety limit the timed arms run under
	unsigned charCeiling;  // the ceiling the characterization itself ran under
	const char* strictNote;
};

// Measured on Apple Silicon, Release/NDEBUG, at the source_id recorded in
//    docs/learning_research/optimizer_baseline_results.md. Keys are
//    <model>-<rows>-<arch>.
//
// A ZERO STRICT ENDPOINT IS A RESULT, NOT AN OMISSION. It means the canonical
//    reference did not converge on that workload, so there is no converged
//    objective to match -- and publishing wherever its ceiling left it would be
//    the ceiling-is-a-floor error the convergence contract exists to forbid.
static const Endpoints ENDPOINT_TABLE[] = {
	// Measured against engine_id 20233b71ed257605 over 71 src/ files, build Release/NDEBUG.
	// Regenerate with tests/optimizer/fill_targets.py after any change
	// to src/ -- these are measurements of THAT engine's behavior.
	{ "logistic-6000-linear",       0.665221248, 0.664939378, 50000, 1000000,
	  "" },
	{ "logistic-25000-linear",      0.662455956, 0.661943767, 50000, 1000000,
	  "" },
	{ "logistic-100000-linear",     0.665042606, 0.664748927, 50000, 1000000,
	  "" },
	{ "logistic-400000-linear",     0.663610259, 0.663446411, 40000, 1000000,
	  "" },
	{ "simpleprop-6000-4",          0.118124155, 0, 40000, 40000,
	  "canonical did not converge: gradient 7.52e-04 against the engine's 1e-6 rule after 40001 iterations" },
	{ "simpleprop-25000-4",         0.117689406, 0, 40000, 40000,
	  "canonical did not converge: gradient 4.29e-04 against the engine's 1e-6 rule after 40001 iterations" },
	{ "simpleprop-100000-4",        0.118292976, 0, 40000, 40000,
	  "canonical did not converge: gradient 6.11e-04 against the engine's 1e-6 rule after 40001 iterations" },
	{ "simpleprop-25000-2",         0.1075966, 0, 130000, 200000,
	  "canonical did not converge: gradient 4.26e-05 against the engine's 1e-6 rule after 200001 iterations" },
	{ "simpleprop-25000-8",         0.106751914, 0, 120000, 200000,
	  "canonical did not converge: gradient 3.75e-04 against the engine's 1e-6 rule after 200001 iterations" },
	{ "simpleprop-25000-16",        0.118096561, 0, 30000, 40000,
	  "canonical did not converge: gradient 1.67e-03 against the engine's 1e-6 rule after 40001 iterations" },
	// THE PHASE 4 CONDITIONING PAIR. Written by hand rather than by
	// fill_targets.py, whose case regex matches Civic names only -- but by
	// EXACTLY its formulae: practical = practical_objective * (1 + 1e-5), and
	// ceiling = max(20000, int(practical_iteration * 2.5/10000 + 1) * 10000).
	// Characterized at ceiling 400,000 against engine_id 4e11285b904cd4f9
	// (75 src/ files); at 40,000 NEITHER had plateaued and no endpoint was
	// published, which is the ceiling-is-not-a-floor rule doing its job.
	//
	// THE PAIR IS THE MEASUREMENT. Same rows, same outcome rule, same
	// architecture; only the input conditioning differs -- and canonical's own
	// plateau is 21x worse on the ill-conditioned twin (0.0762 against 0.00366),
	// which is what establishes that the fixture is genuinely harder rather than
	// merely different.
	{ "well4-simpleprop-6000-4",    0.00365853677, 0, 370000, 400000,
	  "canonical did not converge: gradient 1.17e-05 against the engine's 1e-6 rule after 400001 iterations" },
	{ "poor4-simpleprop-6000-4",    0.0761685276, 0, 210000, 400000,
	  "canonical did not converge: gradient 1.82e-04 against the engine's 1e-6 rule after 400001 iterations" },
	{ 0, 0, 0, 0, 0, "" }
};

static inline const Endpoints* endpointsFor( const string& key )
{
	for ( unsigned i = 0; ENDPOINT_TABLE[ i ].key; i++ )
		if ( key == ENDPOINT_TABLE[ i ].key )
			return &ENDPOINT_TABLE[ i ];
	return 0;
}

// WHETHER THIS WORKLOAD HAS THAT ENDPOINT AT ALL. A strict endpoint exists only
//    where the canonical reference actually CONVERGED. On the neural workloads
//    it did not: measured, its maximum gradient settles around 5e-4 and does not
//    approach the engine's own 1e-6 rule -- 4.5e-4 after 20,000 iterations and
//    still 6.2e-4 after 100,000, having risen in between. Declaring a strict arm
//    there would mean racing four methods to an objective no canonical run ever
//    established, so those arms are NOT DECLARED and the reason travels with the
//    table rather than living only in a document.
static inline bool endpointAvailable( const string& key, const string& endpoint )
{
	const Endpoints* e = endpointsFor( key );
	if ( !e ) return false;
	if ( endpoint == ENDPOINT_STRICT ) return e->strict > 0;
	return e->practical > 0;
}

// A GENERATED fixture prefixes its key. Without that, "simpleprop-6000-4" would
//    name two different workloads -- Civic Choice and a generated one at the same
//    size and architecture -- and one would silently race to the other's endpoint.
static inline string endpointKey( const string& model, unsigned rows,
	unsigned hidden, const string& fixture = "" )
{
	string archKey = ( model == "logistic" ) ? "linear" : to_string( hidden );
	return ( fixture.empty() ? "" : fixture + "-" )
		+ model + "-" + to_string( rows ) + "-" + archKey;
}

// The Civic Choice workload's fixed facts, in one place.
static const unsigned CIVIC_INPUTS = 14;      // from the maintained grooming
static const unsigned CIVIC_HIDDEN = 4;       // the walkthrough's OBD selection
static const unsigned CIVIC_DATA_SEED = 20260804;
static const double CIVIC_TEST_FRACTION = 0.25;

static inline string civicFile( unsigned rows )
{
	return "civic_" + to_string( rows ) + ".txt";
}

// One Civic Choice arm. Everything that defines the WORK comes from (model,
//    rows, hidden, endpoint); only the method varies within a group.
static inline Case civicCase( const string& model, unsigned rows,
	unsigned hidden, const string& endpoint, unsigned optimizer, bool autoStep )
{
	Case c = baseCase();
	c.model = model;
	c.dataFile = civicFile( rows );
	c.inputs = CIVIC_INPUTS;
	c.dataSeed = CIVIC_DATA_SEED;
	c.testFraction = CIVIC_TEST_FRACTION;
	c.hidden = hidden;
	c.rows = 0;               // the file decides
	c.optimizer = optimizer;
	c.autoStep = autoStep;
	c.endpoint = endpoint;
	c.groupAxis = "method";

	// EACH MODEL'S OWN CONSTRUCTED DEFAULTS, written down rather than inherited
	//    silently, so the row reports them and the group invariant checks them.
	//    A benchmark of a configuration no user runs answers a question nobody
	//    asked, and these are the values a user gets:
	//
	//      Network      eta 0.05, batch epoch, weight decay ON at 5e-5, LMS,
	//                   no step-size search      (network.cpp:20-46, model.cpp:9)
	//      Logistic     cross-entropy and batch BY DEFINITION, weight decay OFF,
	//                   and the step-size search ON       (logistic.cpp:8-18)
	//
	//    autoStep is the one the caller varies, because "canonical fixed eta"
	//    and "canonical automatic step size" are two of the four methods the
	//    comparison is about -- and for Logistic the automatic one is the
	//    default, which is itself worth knowing.
	c.eta = 0.05;
	c.batch = true;
	if ( model == "logistic" )
	{
		c.xentropy = true;
		c.decayOn = false;
		c.decay = 0.0;
	}
	else
	{
		c.xentropy = false;
		c.decayOn = true;
		c.decay = 5e-5;
	}

	string archKey = ( model == "logistic" ) ? "linear" : to_string( hidden );
	const Endpoints* e = endpointsFor( endpointKey( model, rows, hidden ) );
	if ( e )
	{
		c.target = ( endpoint == ENDPOINT_STRICT ) ? e->strict : e->practical;
		c.ceiling = e->ceiling;
	}
	else
	{
		c.target = 0; // refused by validate(): an uncharacterized workload
		c.ceiling = 1;
	}

	c.group = "civic-" + model + "-r" + to_string( rows ) + "-h" + archKey
		+ "-" + endpoint;
	c.name = c.group + "-"
		+ optimizerName( optimizer ) + ( autoStep ? "-autostep" : "" );
	return c;
}

// ---------------------------------------------------------------------------
// THE CONDITIONING PAIR (Phase 4): one problem at two conditionings.
//
// `well4` and `poor4` differ ONLY in how the same underlying inputs are scaled
// (see fixtureMatrix). Each is its own comparison group with its own
// characterized endpoint, and the reading is ACROSS the two groups: what does
// each method's cost do when the conditioning degrades? That question cannot be
// answered inside one group, and a single ill-conditioned arm without its
// well-conditioned twin answers it either.
static const unsigned COND_ROWS = 6000;
static const unsigned COND_HIDDEN = 4;
static const unsigned COND_DATA_SEED = 20260806;

static inline Case conditioningCase( const string& fixture, unsigned optimizer )
{
	Case c = baseCase();
	c.model = "simpleprop";
	c.fixture = fixture;
	c.dataFile = "";
	c.inputs = 0;             // a generated fixture declares its own width
	c.rows = COND_ROWS;
	c.dataSeed = COND_DATA_SEED;
	c.testFraction = 0.25;
	c.hidden = COND_HIDDEN;
	c.optimizer = optimizer;
	c.autoStep = false;
	c.endpoint = ENDPOINT_PRACTICAL;
	c.groupAxis = "method";

	// Network's own constructed defaults, as every other neural arm uses.
	c.eta = 0.05;
	c.batch = true;
	c.xentropy = false;
	c.decayOn = true;
	c.decay = 5e-5;

	const string key = endpointKey( "simpleprop", COND_ROWS, COND_HIDDEN,
		fixture );
	const Endpoints* e = endpointsFor( key );
	if ( e )
	{
		c.target = e->practical;
		c.ceiling = e->ceiling;
	}
	else
	{
		// UNCHARACTERIZED, and therefore only a control to characterize WITH.
		//    The placeholder keeps validate() satisfied so `--characterize
		//    --case ...` can name this arm; characterize() replaces it. Without
		//    this the table is a one-way door -- the same reason
		//    referenceCaseFor() exists.
		c.target = 0.5;
		c.ceiling = 40000;
	}

	c.group = "cond-" + fixture + "-simpleprop-r" + to_string( COND_ROWS )
		+ "-h" + to_string( COND_HIDDEN ) + "-practical";
	c.name = c.group + "-" + optimizerName( optimizer );
	return c;
}

// The four methods the brief compares for a neural model. Logistic takes the
//    three that apply to it -- the step-size search is a Network mechanism and
//    Logistic's own fit is the same three trainingTypes.
// THE CANONICAL REFERENCE FOR A WORKLOAD, built from its key alone and
//    independent of whether that workload currently has any declared arms.
//
//    Without this the table is a one-way door: a workload whose endpoint could
//    not be established declares no arms, so its canonical reference case stops
//    existing, so `--characterize --case ...` cannot name it, so it can never be
//    re-characterized at a larger budget. A control that disappears when its
//    measurement fails is the wrong shape for a control.
static inline bool referenceCaseFor( const string& key, Case& out )
{
	string model;
	unsigned rows = 0, hidden = 0;
	size_t r = key.find( "-r" ) == string::npos ? key.find( '-' ) : 0;
	// key is <model>-<rows>-<arch>; arch is "linear" or a hidden-unit count.
	size_t a = key.find( '-' );
	if ( a == string::npos ) return false;
	size_t b = key.find( '-', a + 1 );
	if ( b == string::npos ) return false;
	model = key.substr( 0, a );
	string rowsStr = key.substr( a + 1, b - a - 1 );
	string archStr = key.substr( b + 1 );
	for ( size_t i = 0; i < rowsStr.size(); i++ )
		if ( !isdigit( ( unsigned char ) rowsStr[ i ] ) ) return false;
	rows = ( unsigned ) atoi( rowsStr.c_str() );
	if ( archStr != "linear" )
	{
		for ( size_t i = 0; i < archStr.size(); i++ )
			if ( !isdigit( ( unsigned char ) archStr[ i ] ) ) return false;
		hidden = ( unsigned ) atoi( archStr.c_str() );
	}
	( void ) r;
	if ( !endpointsFor( key ) ) return false;
	out = civicCase( model, rows, hidden, ENDPOINT_PRACTICAL, 0, false );
	// The declared target may be 0 -- that is exactly the case this exists for.
	//    characterize() replaces it, and refuses nothing on account of it.
	out.target = 0.5;
	out.name = key + "-reference";
	return true;
}

static inline void addMethods( vector< Case >& v, const string& model,
	unsigned rows, unsigned hidden, const string& endpoint,
	bool includeQuasiNewton = true )
{
	if ( !endpointAvailable( endpointKey( model, rows, hidden ), endpoint ) )
		return;
	v.push_back( civicCase( model, rows, hidden, endpoint, 0, false ) );
	v.push_back( civicCase( model, rows, hidden, endpoint, 0, true ) );
	if ( !includeQuasiNewton )
		return;
	v.push_back( civicCase( model, rows, hidden, endpoint, 1, false ) );
	v.push_back( civicCase( model, rows, hidden, endpoint, 2, false ) );
}

// The repeated-fit consumer: k-fold cross-validation repeated r times over the
//    development rows, plus the locked refit. Aimed at the PRACTICAL endpoint --
//    a repeated workload run to the strict floor is a different (and much more
//    expensive) question, and conflating them is how a "workflow is now
//    tractable" claim comes to rest on an endpoint nobody would use in a fit.
static inline Case civicCvCase( const string& model, unsigned rows,
	unsigned hidden, unsigned optimizer, bool autoStep )
{
	Case c = civicCase( model, rows, hidden, ENDPOINT_PRACTICAL, optimizer, autoStep );
	c.workload = "cv";
	c.timingScope = SCOPE_WORKFLOW;
	c.cvFolds = 5;
	c.cvRepeats = 2;
	// Fold-relative, for the reason recorded on Case::autoStop: the matched
	//    objective was measurably unreachable on 4 of 10 folds.
	c.endpoint = ENDPOINT_NONE;
	c.minStop = false;
	c.autoStop = true;
	c.target = 0.5;   // carried, unused: no objective rule is armed
	string archKey = ( model == "logistic" ) ? "linear" : to_string( hidden );
	c.group = "civic-cv-" + model + "-r" + to_string( rows ) + "-h" + archKey;
	c.name = c.group + "-" + optimizerName( optimizer )
		+ ( autoStep ? "-autostep" : "" );
	return c;
}

// ---------------------------------------------------------------------------
// The Step 0B workload matrix.
//
// DELIBERATELY NOT AN EXHAUSTIVE TAXONOMY. Correlation, separation, non-finite
// and poorly-scaled fixtures are candidate-specific correctness questions and
// belong to the phase that has a candidate to discriminate. The plan's scope
// governor is explicit that timing minutiae must not displace candidate
// investigation, and a taxonomy no pending decision depends on is exactly that.
// What is here is what establishes the baseline a candidate must beat:
//
//   1  Civic Choice logistic, both endpoints
//   2  Civic Choice neural at the walkthrough's own architecture, both endpoints
//   3  a row-count series over the same problem at 6k / 25k / 100k / 400k
//   4  a parameter-count series over hidden = 2 / 4 / 8 / 16
//   5  the repeated-fit consumer: repeated CV plus the locked refit
static inline vector< Case > step0bCases()
{
	vector< Case > v;

	// 1 + 2: the application benchmark at the committed walkthrough size.
	const string ends[ 2 ] = { ENDPOINT_PRACTICAL, ENDPOINT_STRICT };
	for ( int e = 0; e < 2; e++ )
	{
		addMethods( v, "logistic", 6000, 0, ends[ e ] );
		addMethods( v, "simpleprop", 6000, CIVIC_HIDDEN, ends[ e ] );
	}

	// 3: row-count scaling. Each size is its OWN comparison group -- an endpoint
	//    characterized at 6,000 rows is not the endpoint at 400,000, and forcing
	//    one target across sizes would compare arms against a target only some of
	//    them can reach. Scaling is read ACROSS groups, in the report.
	//    THE NEURAL SERIES STOPS AT 100,000 AND THE LOGISTIC ONE DOES NOT.
	//    Characterizing the 400,000-row neural workload alone costs hours before
	//    a single arm is timed, and the plan's scope governor says to run a
	//    smaller series and label the extrapolation rather than spend the hours.
	//    The logistic series runs the whole way because it is cheap: its
	//    canonical reference converges in ~17,000 iterations at every size.
	// CGD AND SHANNO ARE NOT RE-RUN ON THE LARGEST LOGISTIC WORKLOADS, and this
	//    is a measured decision rather than an omission. Both fail to reach the
	//    Logistic endpoint at 6,000 and at 25,000 rows, on BOTH endpoints -- six
	//    independent demonstrations -- and a failing arm burns its entire
	//    iteration ceiling on every run. At 100,000 rows that is ~11 minutes per
	//    run and at 400,000 it is ~45; re-establishing the same failure at the
	//    two largest sizes costs about four hours of machine time to learn
	//    nothing that is not already established. The plan's scope governor is
	//    explicit that timing minutiae must not displace candidate
	//    investigation. What IS measured at every size is the pair that
	//    succeeds, canonical and canonical-autostep, which is where the
	//    Logistic scaling answer lives.
	//
	//    The neural sizes keep all four methods: that is where the interesting
	//    result is, and there CGD and Shanno do not fail.
	const unsigned sizes[ 3 ] = { 25000, 100000, 400000 };
	for ( int i = 0; i < 3; i++ )
	{
		addMethods( v, "logistic", sizes[ i ], 0, ENDPOINT_PRACTICAL,
			sizes[ i ] <= 25000 );
		addMethods( v, "simpleprop", sizes[ i ], CIVIC_HIDDEN, ENDPOINT_PRACTICAL );
	}
	// The strict endpoint at the largest sizes too, so late-stage failure has
	//    somewhere to show up rather than being assumed absent.
	addMethods( v, "simpleprop", 100000, CIVIC_HIDDEN, ENDPOINT_STRICT );
	addMethods( v, "logistic", 100000, 0, ENDPOINT_STRICT, false );

	// 4: parameter-count scaling, at a size whose runs are long enough to
	//    measure and short enough to repeat. Sizes relevant to intended use --
	//    the walkthrough's own search ran 2..12 hidden nodes.
	const unsigned hiddens[ 3 ] = { 2, 8, 16 };
	for ( int i = 0; i < 3; i++ )
		addMethods( v, "simpleprop", 25000, hiddens[ i ], ENDPOINT_PRACTICAL );
	// hidden = 4 at 25,000 is already in the row series; it is the same group.

	// 5: the repeated-fit consumer.
	//    Aimed at the practical endpoint, so they are declared only where one
	//    exists -- the same rule the fit arms follow.
	if ( endpointAvailable( endpointKey( "logistic", 6000, 0 ), ENDPOINT_PRACTICAL ) )
	{
		v.push_back( civicCvCase( "logistic", 6000, 0, 0, false ) );
		v.push_back( civicCvCase( "logistic", 6000, 0, 2, false ) );
	}
	if ( endpointAvailable( endpointKey( "simpleprop", 6000, CIVIC_HIDDEN ),
		ENDPOINT_PRACTICAL ) )
	{
		v.push_back( civicCvCase( "simpleprop", 6000, CIVIC_HIDDEN, 0, false ) );
		v.push_back( civicCvCase( "simpleprop", 6000, CIVIC_HIDDEN, 2, false ) );
	}
	if ( endpointAvailable( endpointKey( "simpleprop", 25000, CIVIC_HIDDEN ),
		ENDPOINT_PRACTICAL ) )
	{
		v.push_back( civicCvCase( "simpleprop", 25000, CIVIC_HIDDEN, 0, false ) );
		v.push_back( civicCvCase( "simpleprop", 25000, CIVIC_HIDDEN, 2, false ) );
	}

	return v;
}

// ---------------------------------------------------------------------------
// THE PHASE 3 SCREEN: L-BFGS against the incumbent, and nothing else.
//
// The plan's scope governor is explicit that a candidate is screened on ONE
// cheap representative neural workload before any large-data time is spent on
// it, and that an obvious loser is rejected there rather than tuned into
// contention. So this is deliberately two arms, not six:
//
//   * Civic Choice at 6,000 rows, the walkthrough's own four hidden units, the
//     committed 25% stratified holdout and the committed practical objective --
//     the identical workload, split and endpoint Shanno's 238.7 ms / 194 passes
//     were measured on, so the comparison is against a published number rather
//     than a re-measured one;
//   * the same comparison group, so run_probe.py REFUSES the pair if anything
//     defining the work differs between them;
//   * Shanno as the incumbent control, re-run here rather than quoted, because
//     the engine has changed since Step 0B (the packed-boundary extraction) and
//     an incumbent quoted from a different binary is not a control.
//
// Canonical and CGD are NOT included. Their standing is already measured and
// the plan forbids scaling every historical arm; adding them would spend
// roughly a minute of ceiling per repetition to re-establish a settled result.
static inline vector< Case > screenCases()
{
	vector< Case > v;
	if ( !endpointAvailable( endpointKey( "simpleprop", 6000, CIVIC_HIDDEN ),
		ENDPOINT_PRACTICAL ) )
		return v;

	// 2 = Shanno, the incumbent; 3 = the L-BFGS prototype.
	v.push_back( civicCase( "simpleprop", 6000, CIVIC_HIDDEN,
		ENDPOINT_PRACTICAL, 2, false ) );
	v.push_back( civicCase( "simpleprop", 6000, CIVIC_HIDDEN,
		ENDPOINT_PRACTICAL, Network::TRAIN_LBFGS, false ) );

	// THE PREDECLARED MEMORY COMPARISON, and nothing beyond it. The plan names
	//    m = 5, 10 and 20 and forbids an unbounded tuning exercise, so these
	//    three are declared here rather than chosen after seeing a result.
	//    Their own comparison group with its own axis: two L-BFGS arms at
	//    different m are not the same method, and putting them in the method
	//    group would let run_probe.py's invariant check pass a group that is
	//    varying two things at once.
	const unsigned mem[ 3 ] = { 5, 10, 20 };
	for ( int i = 0; i < 3; i++ )
	{
		Case c = civicCase( "simpleprop", 6000, CIVIC_HIDDEN,
			ENDPOINT_PRACTICAL, Network::TRAIN_LBFGS, false );
		c.lbfgsMemory = mem[ i ];
		c.group = "civic-lbfgs-memory-r6000-h4-practical";
		c.groupAxis = "lbfgs_memory";
		c.name = c.group + "-m" + to_string( mem[ i ] );
		v.push_back( c );
	}

	// THE WEIGHT-SEED TEST, predeclared and small. Every result above rests on
	//    ONE initialization, and repeated runs from one fixed start establish
	//    deterministic reproducibility rather than reliability -- a distinction
	//    Step 0B's report was explicit about and this must not quietly drop.
	//    Three further seeds, fixed here before any of them was run.
	//
	//    EACH SEED IS ITS OWN COMPARISON GROUP. weight_seed and
	//    weight_start_id are group invariants precisely because two arms from
	//    different starting weights are not racing the same race; the matched
	//    comparison is Shanno against L-BFGS at the SAME start, repeated at
	//    several starts.
	const unsigned seeds[ 3 ] = { 101, 202, 303 };
	for ( int i = 0; i < 3; i++ )
	{
		const unsigned methods[ 2 ] = { 2, Network::TRAIN_LBFGS };
		for ( int k = 0; k < 2; k++ )
		{
			Case c = civicCase( "simpleprop", 6000, CIVIC_HIDDEN,
				ENDPOINT_PRACTICAL, methods[ k ], false );
			c.weightSeed = seeds[ i ];
			c.group = "civic-seed" + to_string( seeds[ i ] )
				+ "-simpleprop-r6000-h4-practical";
			c.name = c.group + "-" + optimizerName( methods[ k ] );
			v.push_back( c );
		}
	}

	// THE SURVIVOR/INCUMBENT PAIR ONLY, scaled. The plan is explicit that a
	//    survivor and the incumbent are scaled, not every historical arm, and
	//    each size is its own comparison group because an endpoint
	//    characterized at 6,000 rows is not the endpoint at 100,000.
	const unsigned sizes[ 2 ] = { 25000, 100000 };
	for ( int i = 0; i < 2; i++ )
	{
		if ( !endpointAvailable( endpointKey( "simpleprop", sizes[ i ],
			CIVIC_HIDDEN ), ENDPOINT_PRACTICAL ) )
			continue;
		v.push_back( civicCase( "simpleprop", sizes[ i ], CIVIC_HIDDEN,
			ENDPOINT_PRACTICAL, 2, false ) );
		v.push_back( civicCase( "simpleprop", sizes[ i ], CIVIC_HIDDEN,
			ENDPOINT_PRACTICAL, Network::TRAIN_LBFGS, false ) );
	}

	// THE LATE-STAGE QUESTION, asked in the only way this workload permits.
	//
	//    Step 0B established that canonical gradient descent does not converge
	//    on the neural workloads at the engine's own 1e-6 rule, so there is no
	//    canonical strict endpoint to race to and none is invented here. What
	//    can be asked is what each method does when it is allowed to run until
	//    THE ENGINE'S OWN plateau rule says it has stopped improving --
	//    setAutoStop at Iterative's shipped tolerance and window, identical for
	//    both arms.
	//
	//    THIS IS NOT A MATCHED-ENDPOINT RACE and does not pretend to be: the
	//    arms stop at their own plateaus, so the row declares endpoint `none`
	//    rather than borrowing a name it has not earned, and the comparison to
	//    read is WHERE each lands, not only how fast it got there. A method
	//    that stops earlier at a worse objective has not won.
	const unsigned lateMethods[ 2 ] = { 2, Network::TRAIN_LBFGS };
	for ( int k = 0; k < 2; k++ )
	{
		Case c = civicCase( "simpleprop", 6000, CIVIC_HIDDEN,
			ENDPOINT_PRACTICAL, lateMethods[ k ], false );
		c.minStop = false;
		c.autoStop = true;
		c.endpoint = ENDPOINT_NONE;
		c.target = 0.5;  // carried, unused: no objective rule is armed
		c.group = "civic-latestage-simpleprop-r6000-h4";
		c.name = c.group + "-" + optimizerName( lateMethods[ k ] );
		v.push_back( c );
	}

	return v;
}

// ---------------------------------------------------------------------------
// THE PHASE 4 SCREEN: iRPROP+ against the STANDING PORTFOLIO PANEL.
//
// The panel is four arms with four distinct roles, and it is not a race with
// one winner (the plan's Phase 5 portfolio policy):
//
//   L-BFGS      the current speed leader, and the primary modern reference --
//               NOT a requirement every retained method must beat;
//   Shanno      the established legacy quasi-Newton control, which is what
//               exposes a regression specific to the newer implementation path;
//   canonical   the behavioral and matched-objective reference. It is the
//               source of the committed endpoint every arm here races to, so it
//               belongs in the panel even though the Phase 3 screen omitted it;
//               at 6,000 rows one run is ~19 s, which is affordable;
//   iRPROP+     the candidate.
//
// CGD is not a panel member and is not run: its standing is settled (2x slower
// than canonical) and the plan forbids scaling every historical arm.
//
// The workload is the one the panel already lives on -- Civic Choice at 6,000
// rows, the walkthrough's own four hidden units, the committed 25% stratified
// holdout and the committed practical objective -- so nothing here is
// re-characterized and no endpoint is invented for the candidate.
static inline vector< Case > screen4Cases()
{
	vector< Case > v;
	if ( !endpointAvailable( endpointKey( "simpleprop", 6000, CIVIC_HIDDEN ),
		ENDPOINT_PRACTICAL ) )
		return v;

	// THE PANEL, one comparison group. run_probe.py REFUSES the set if anything
	//    defining the work differs between its members.
	const unsigned panel[ 4 ] = { 0, 2, Network::TRAIN_LBFGS,
		Network::TRAIN_IRPROP };
	for ( int k = 0; k < 4; k++ )
		v.push_back( civicCase( "simpleprop", 6000, CIVIC_HIDDEN,
			ENDPOINT_PRACTICAL, panel[ k ], false ) );

	// THE WEIGHT-SEED TEST, at the three seeds Phase 3 predeclared, so the
	//    candidate is read on the same starts its reference was. Each seed is
	//    its own comparison group: two arms from different starting weights are
	//    not racing the same race. Canonical is omitted at the extra seeds --
	//    it is the endpoint's source, not a per-seed control, and three more
	//    19-second arms per repetition buys nothing the base group has not
	//    already established.
	const unsigned seeds[ 3 ] = { 101, 202, 303 };
	for ( int i = 0; i < 3; i++ )
	{
		const unsigned methods[ 3 ] = { 2, Network::TRAIN_LBFGS,
			Network::TRAIN_IRPROP };
		for ( int k = 0; k < 3; k++ )
		{
			Case c = civicCase( "simpleprop", 6000, CIVIC_HIDDEN,
				ENDPOINT_PRACTICAL, methods[ k ], false );
			c.weightSeed = seeds[ i ];
			c.group = "civic-seed" + to_string( seeds[ i ] )
				+ "-simpleprop-r6000-h4-practical";
			c.name = c.group + "-" + optimizerName( methods[ k ] );
			v.push_back( c );
		}
	}

	// THE LATE-STAGE QUESTION, asked the only way this workload permits. There
	//    is no canonical strict endpoint on the neural workloads (canonical does
	//    not converge on them at the engine's 1e-6 rule), so all three run to
	//    THE ENGINE'S OWN plateau rule instead, identically configured, and the
	//    row declares endpoint `none` rather than borrowing a name it has not
	//    earned. WHAT EACH ARM LANDS ON is read beside how fast it got there: a
	//    method that stops earlier at a worse objective has not won. Phase 3
	//    found exactly that shape for L-BFGS, so the question is live.
	const unsigned lateMethods[ 3 ] = { 2, Network::TRAIN_LBFGS,
		Network::TRAIN_IRPROP };
	for ( int k = 0; k < 3; k++ )
	{
		Case c = civicCase( "simpleprop", 6000, CIVIC_HIDDEN,
			ENDPOINT_PRACTICAL, lateMethods[ k ], false );
		c.minStop = false;
		c.autoStop = true;
		c.endpoint = ENDPOINT_NONE;
		c.target = 0.5;  // carried, unused: no objective rule is armed
		c.group = "civic-latestage-simpleprop-r6000-h4";
		c.name = c.group + "-" + optimizerName( lateMethods[ k ] );
		v.push_back( c );
	}

	// THE CONDITIONING PAIR. iRPROP+'s hypothesized advantage is poorly scaled
	//    objectives, so this is the arm that could earn it a place in the
	//    portfolio on a DISTINCT profile rather than on raw speed -- and the
	//    well-scaled twin is what makes that reading possible. The canonical
	//    reference of each is declared UNCONDITIONALLY, because it is the arm
	//    `--characterize` must be able to name before any endpoint exists.
	const string conds[ 2 ] = { "well4", "poor4" };
	for ( int i = 0; i < 2; i++ )
	{
		const string key = endpointKey( "simpleprop", COND_ROWS, COND_HIDDEN,
			conds[ i ] );
		if ( endpointAvailable( key, ENDPOINT_PRACTICAL ) )
		{
			const unsigned condPanel[ 4 ] = { 0, 2, Network::TRAIN_LBFGS,
				Network::TRAIN_IRPROP };
			for ( int k = 0; k < 4; k++ )
				v.push_back( conditioningCase( conds[ i ], condPanel[ k ] ) );
		}
		else
			v.push_back( conditioningCase( conds[ i ], 0 ) );
	}

	// SCALING, added only after the 6,000-row screen was run and left iRPROP+ a
	//    plausible portfolio candidate -- the plan's staged gate, in that order.
	//    The candidate and the two quasi-Newton references only; canonical is
	//    not re-run at these sizes, where one arm costs minutes to re-establish
	//    a settled reference. Each size is its own comparison group, because an
	//    endpoint characterized at 6,000 rows is not the endpoint at 100,000.
	const unsigned sizes[ 2 ] = { 25000, 100000 };
	for ( int i = 0; i < 2; i++ )
	{
		if ( !endpointAvailable( endpointKey( "simpleprop", sizes[ i ],
			CIVIC_HIDDEN ), ENDPOINT_PRACTICAL ) )
			continue;
		const unsigned scaled[ 3 ] = { 2, Network::TRAIN_LBFGS,
			Network::TRAIN_IRPROP };
		for ( int k = 0; k < 3; k++ )
			v.push_back( civicCase( "simpleprop", sizes[ i ], CIVIC_HIDDEN,
				ENDPOINT_PRACTICAL, scaled[ k ], false ) );
	}

	return v;
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

// EVERY case the probe knows: the Step 0A mechanics pilot and the Step 0B
//    workload matrix. One list, because a name must resolve to exactly one case
//    no matter which selection the caller asked for.
static inline vector< Case > allCases()
{
	vector< Case > v = pilotCases();
	vector< Case > b = step0bCases();
	v.insert( v.end(), b.begin(), b.end() );

	// The Phase 3 screen's Shanno arm IS a Step 0B arm -- same group, same
	//    name -- so only the candidate is new. Appended by name rather than
	//    wholesale, because a duplicate case name would make --case ambiguous
	//    and would run the same arm twice under --all.
	vector< Case > sc = screenCases();
	for ( size_t i = 0; i < sc.size(); i++ )
	{
		bool known = false;
		for ( size_t j = 0; j < v.size() && !known; j++ )
			known = ( v[ j ].name == sc[ i ].name );
		if ( !known )
			v.push_back( sc[ i ] );
	}

	// The Phase 4 panel overlaps both of the above -- its Shanno, canonical,
	//    L-BFGS and late-stage arms are already declared -- so only the iRPROP+
	//    arms are new. Appended by name for the same reason.
	vector< Case > s4 = screen4Cases();
	for ( size_t i = 0; i < s4.size(); i++ )
	{
		bool known = false;
		for ( size_t j = 0; j < v.size() && !known; j++ )
			known = ( v[ j ].name == s4[ i ].name );
		if ( !known )
			v.push_back( s4[ i ] );
	}
	return v;
}

// Which canonical reference case belongs to a given group. Empty when the group
//    has none, so the caller can say so rather than guess.
static inline string canonicalReferenceFor( const string& group )
{
	vector< Case > all = allCases();
	for ( size_t i = 0; i < all.size(); i++ )
		if ( all[ i ].group == group && isCanonicalReference( all[ i ] ) )
			return all[ i ].name;
	return "";
}

static inline bool findCase( const string& name, Case& out )
{
	vector< Case > all = allCases();
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
