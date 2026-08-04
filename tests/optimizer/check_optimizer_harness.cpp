// check_optimizer_harness.cpp : the benchmark harness's mechanics, asserted.
//
// This IS a ctest case and contains NO TIMING ASSERTION of any kind. That
// separation is the design: optimizer_probe measures wall time and is never a
// gate, while everything the measurement DEPENDS ON is deterministic and gated
// here. Both binaries include the same harness.h, so this checks the tool that
// actually runs.
//
// WHAT EACH GROUP OF CHECKS GUARDS:
//
//   1. THE PARAMETER-STATE IDENTITY. The weight hash is over actual weight
//      elements, is nonempty, moves when the seed moves, moves when the
//      architecture moves, moves when training moves the weights, and is not a
//      constant. Empty weight structures are refused rather than hashed
//      vacuously. Every model has one: Logistic through getBetas(), the
//      one-hidden pair through OneHiddenNet's hW/oW, BackProp through the
//      protected read-only BackProp::weightMatrices() seam. An unavailable
//      identity is REFUSED, never replaced by the weaker function fingerprint.
//   2. THE FUNCTION FINGERPRINT is secondary and is pinned as such: it agrees
//      across two fixtures that share input columns, which is exactly why it
//      cannot be the parameter identity.
//   3. Dataset/split identity.
//   4. ITERATION SEMANTICS. getIterations() means a zero-based index on a rule
//      break and a count on ceiling exhaustion -- two meanings in one accessor.
//      iterations_completed counts trainSet() calls and is correct on both
//      paths; this asserts the discrepancy exists and that the counter is right.
//   5. The ceiling is a failure to converge, never a fast result.
//   6. A reachable target is usable, end state included.
//   7. The full-pass counter sees the step-size search, as an exact ratio.
//   8. Every runtime failure still emits exactly ONE well-formed row: setup
//      faults and training faults are distinguished, and neither terminates the
//      process.
//   9. --characterize is refused on a non-canonical case, by name.
//  10. Row schema completeness and refusal-by-name.
//  11. No artifacts left behind.

#include <cstdio>
#include <iostream>
#include <string>
#include <sys/stat.h>

#include "harness.h"

using namespace std;
using namespace optbench;

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

static void expectEq( long long got, long long want, const string& what )
{
	if ( got != want )
		cout << "         got " << got << ", expected " << want << endl;
	expect( got == want, what );
}

static const string REV = "harness-check";

// A case that runs one iteration and stops nowhere useful: these tests are about
// the START, not the endpoint.
static Case shortCase( const string& name, const string& model )
{
	Case c = baseCase();
	c.name = name;
	c.group = "unit";
	c.model = model;
	c.ceiling = 1;
	c.target = 1e-30;
	return c;
}

// --- 1. the parameter-state identity ---------------------------------------

static void test_weight_identity()
{
	cout << "-- the PARAMETER-STATE identity (actual weights) --" << endl;

	Row a = runCase( shortCase( "w-a", "simpleprop" ), REV );
	Row b = runCase( shortCase( "w-b", "simpleprop" ), REV );

	// NONEMPTY first, and over a nonempty container. Equality between two
	//    absent artifacts proves nothing.
	expect( a.weightIdAvailable, "SimpleProp exposes a weight identity" );
	expect( !a.weightStartId.empty()
		&& a.weightStartId != "0x0000000000000000",
		"...which is nonempty" );
	// 3 hidden units x (2 inputs + 1 bias) = 9, plus oW of 3+1 = 4 -> 13
	expectEq( a.weightElements, 13,
		"...hashed over every weight element (9 hidden + 4 output)" );
	expectEq( a.weightElements, a.params,
		"...which is exactly the model's parameter count df()" );
	expect( a.weightStartId == b.weightStartId,
		"two identically constructed arms hold the SAME weights" );

	// Non-vacuity: a constant would pass the equality above.
	Case d = shortCase( "w-c", "simpleprop" );
	d.weightSeed = 8;
	Row e = runCase( d, REV );
	expect( e.weightStartId != a.weightStartId,
		"a DIFFERENT weight seed changes the weight identity" );

	Case f = shortCase( "w-d", "simpleprop" );
	f.hidden = 4;
	Row g = runCase( f, REV );
	expect( g.weightStartId != a.weightStartId,
		"a different architecture changes it" );
	expect( g.weightElements != a.weightElements,
		"...and changes how many elements were hashed" );

	// Training must move it.
	Case h = baseCase();
	h.name = "w-trained";
	h.group = "unit";
	h.model = "simpleprop";
	Row k = runCase( h, REV );
	expect( k.usable, "a trained SimpleProp arm is usable" );
	expect( !k.weightEndId.empty() && k.weightEndId != k.weightStartId,
		"training changes the weight identity" );

	// Two different models must not collide, because the model tag is hashed.
	Row bare = runCase( shortCase( "w-bare", "bareprop" ), REV );
	expect( bare.weightIdAvailable, "BareProp exposes a weight identity too" );
	expect( bare.weightStartId != a.weightStartId,
		"SimpleProp and BareProp identities do not collide" );

	// Logistic, through its public accessor.
	Row lg = runCase( shortCase( "w-log", "logistic" ), REV );
	expect( lg.weightIdAvailable, "Logistic exposes a weight identity" );
	expectEq( lg.weightElements, lg.params,
		"...over df() elements (inputs + intercept)" );

	// BACKPROP, through the protected read-only seam BackProp::weightMatrices().
	//    This was the one model without a parameter-state identity, and its
	//    comparison group was refused as uncertifiable until the seam existed.
	Case bp = shortCase( "w-bp", "backprop" );
	bp.layers.push_back( 4 );
	bp.layers.push_back( 3 );
	Row bpr = runCase( bp, REV );
	expect( bpr.weightIdAvailable, "BackProp exposes a weight identity" );
	expect( !bpr.weightStartId.empty()
		&& bpr.weightStartId != "0x0000000000000000",
		"...which is nonempty" );
	expect( bpr.weightIdNote.empty(),
		"...with no unavailability note left behind" );
	// layers 4,3 with bias over 2 inputs, 1 output:
	//    (2+1)*4 = 12, (4+1)*3 = 15, (3+1)*1 = 4  ->  31
	expectEq( bpr.weightElements, 31,
		"...hashed over every element of every weight matrix" );
	expectEq( bpr.weightElements, bpr.params,
		"...which is exactly df()" );

	Case bp2 = bp;
	bp2.name = "w-bp2";
	expect( runCase( bp2, REV ).weightStartId == bpr.weightStartId,
		"two identically constructed BackProps hold the same weights" );

	Case bp3 = bp;
	bp3.name = "w-bp3";
	bp3.weightSeed = 8;
	expect( runCase( bp3, REV ).weightStartId != bpr.weightStartId,
		"a different BackProp weight seed changes it" );

	Case bp4 = shortCase( "w-bp4", "backprop" );
	bp4.layers.push_back( 5 );
	bp4.layers.push_back( 3 );
	Row bp4r = runCase( bp4, REV );
	expect( bp4r.weightStartId != bpr.weightStartId,
		"a different BackProp architecture changes it" );
	expect( bp4r.weightElements != bpr.weightElements,
		"...and changes how many elements were hashed" );

	Case bpTrained = baseCase();
	bpTrained.name = "w-bp-trained";
	bpTrained.group = "unit";
	bpTrained.model = "backprop";
	bpTrained.layers.push_back( 4 );
	bpTrained.layers.push_back( 3 );
	Row bpt = runCase( bpTrained, REV );
	expect( bpt.usable, "a trained BackProp arm is usable" );
	expect( !bpt.weightEndId.empty() && bpt.weightEndId != bpt.weightStartId,
		"training changes the BackProp weight identity" );
}

// Every element of every BackProp weight matrix must participate. Exercised
// through the free-standing traversal, so the test needs no mutation access to a
// production model -- the same reason oneHiddenIdentity is reachable directly.
// Without this, a traversal that skipped a whole matrix would still pass every
// model-level assertion above, exactly as the hidden-matrix sabotage once did.
static void test_backprop_traversal()
{
	cout << "-- every BackProp weight matrix and element participates --" << endl;

	vector< Matrix< double > > base;
	base.push_back( Matrix< double >( 2, 3, 1.0 ) );
	base.push_back( Matrix< double >( 1, 2, 1.0 ) );

	WeightIdentity ref = matricesIdentity( "BackProp", base );
	expect( ref.available, "a populated stack is accepted (the control)" );
	expectEq( ref.elements, 8, "...over 2*3 + 1*2 = 8 elements" );

	for ( size_t m = 0; m < base.size(); m++ )
		for ( unsigned r = 0; r < base[ m ].rows(); r++ )
			for ( unsigned c = 0; c < base[ m ].cols(); c++ )
			{
				vector< Matrix< double > > w = base;
				w[ m ]( r, c ) = 9.0;
				expect( matricesIdentity( "BackProp", w ).hash != ref.hash,
					"matrix " + to_string( m ) + " element (" + to_string( r )
						+ "," + to_string( c ) + ") participates" );
			}

	// Shape and stack order participate too.
	{
		vector< Matrix< double > > w;
		w.push_back( Matrix< double >( 3, 2, 1.0 ) ); // same 6 values, new shape
		w.push_back( Matrix< double >( 1, 2, 1.0 ) );
		expect( matricesIdentity( "BackProp", w ).hash != ref.hash,
			"the same values in a different SHAPE differ" );
	}
	{
		vector< Matrix< double > > w;
		w.push_back( Matrix< double >( 1, 2, 1.0 ) ); // stack reversed
		w.push_back( Matrix< double >( 2, 3, 1.0 ) );
		expect( matricesIdentity( "BackProp", w ).hash != ref.hash,
			"the same matrices in a different ORDER differ" );
	}
	{
		vector< Matrix< double > > w = base;
		w.push_back( Matrix< double >( 1, 1, 1.0 ) ); // an extra layer
		expect( matricesIdentity( "BackProp", w ).hash != ref.hash,
			"an extra layer differs" );
	}

	// Empty structures are refused, never hashed vacuously.
	expect( !matricesIdentity( "BackProp",
		vector< Matrix< double > >() ).available,
		"an empty stack is refused" );
	{
		vector< Matrix< double > > w = base;
		w.push_back( Matrix< double >() );
		WeightIdentity bad = matricesIdentity( "BackProp", w );
		expect( !bad.available, "a stack containing an empty matrix is refused" );
		expect( !bad.reason.empty(), "...with a reason, not a silent zero" );
	}
	expect( matricesIdentity( "A", base ).hash != matricesIdentity( "B", base ).hash,
		"the model tag participates" );
}

// Empty weight structures must be refused, not hashed. Reached directly,
// because no configured case can produce an unbuilt model through runCase.
static void test_empty_weights_refused()
{
	cout << "-- empty weight structures are refused, not hashed --" << endl;

	Matrix< double > emptyM;
	vector< double > emptyV;
	vector< double > someV( 3, 1.0 );
	Matrix< double > someM( 2, 2, 1.0 );

	expect( !oneHiddenIdentity( "t", emptyM, someV ).available,
		"an empty hidden weight matrix is refused" );
	expect( !oneHiddenIdentity( "t", someM, emptyV ).available,
		"an empty output weight vector is refused" );
	expect( !oneHiddenIdentity( "t", emptyM, emptyV ).reason.empty(),
		"...with a reason, not a silent zero" );
	expect( oneHiddenIdentity( "t", someM, someV ).available,
		"a populated pair IS accepted (the control for the three above)" );
	expect( oneHiddenIdentity( "a", someM, someV ).hash
		!= oneHiddenIdentity( "b", someM, someV ).hash,
		"the model tag participates: same numbers, different model, different id" );

	// EVERY STRUCTURE MUST PARTICIPATE, ONE AT A TIME.
	//
	//    This exists because a sabotage proved the earlier checks vacuous. With
	//    the hidden matrix traversal replaced by a constant, every model-level
	//    assertion above still passed -- different seed, different architecture
	//    and post-training identities all still moved, because the OUTPUT vector
	//    alone was enough to move them. A test that cannot tell whether hW was
	//    read does not guard hW.
	//
	//    So each structure is varied on its own, with the other held fixed.
	{
		Matrix< double > hA( 2, 2, 1.0 ), hB( 2, 2, 1.0 );
		hB( 1, 1 ) = 2.0;                       // ONE hidden weight differs
		vector< double > o( 3, 1.0 );
		expect( oneHiddenIdentity( "t", hA, o ).hash
			!= oneHiddenIdentity( "t", hB, o ).hash,
			"one changed HIDDEN weight changes the identity (hW is read)" );

		vector< double > oA( 3, 1.0 ), oB( 3, 1.0 );
		oB[ 2 ] = 2.0;                          // ONE output weight differs
		expect( oneHiddenIdentity( "t", hA, oA ).hash
			!= oneHiddenIdentity( "t", hA, oB ).hash,
			"one changed OUTPUT weight changes the identity (oW is read)" );

		// Every element of hW, not merely the first or last: a partial traversal
		//    would pass the single-element check above depending on where it
		//    stopped.
		for ( unsigned r = 0; r < 2; r++ )
			for ( unsigned c = 0; c < 2; c++ )
			{
				Matrix< double > h( 2, 2, 1.0 );
				h( r, c ) = 9.0;
				expect( oneHiddenIdentity( "t", h, oA ).hash
					!= oneHiddenIdentity( "t", hA, oA ).hash,
					"hidden weight (" + to_string( r ) + "," + to_string( c )
						+ ") participates" );
			}
		for ( size_t i = 0; i < oA.size(); i++ )
		{
			vector< double > o2( 3, 1.0 );
			o2[ i ] = 9.0;
			expect( oneHiddenIdentity( "t", hA, o2 ).hash
				!= oneHiddenIdentity( "t", hA, oA ).hash,
				"output weight " + to_string( i ) + " participates" );
		}

		// Dimensions participate, so the same numbers in a different shape are a
		//    different state.
		Matrix< double > wide( 1, 4, 1.0 ), tall( 4, 1, 1.0 );
		expect( oneHiddenIdentity( "t", wide, oA ).hash
			!= oneHiddenIdentity( "t", tall, oA ).hash,
			"the same values in a different SHAPE give a different identity" );
	}

	// The same one-at-a-time question for Logistic, whose weights are one
	//    vector reached through getBetas().
	{
		Case c = shortCase( "w-log-a", "logistic" );
		Row a = runCase( c, REV );
		Case d = shortCase( "w-log-b", "logistic" );
		d.weightSeed = 99;
		Row b = runCase( d, REV );
		expect( a.weightStartId != b.weightStartId,
			"Logistic's identity moves with its weights" );
	}
}

// --- 2. the function fingerprint is secondary -------------------------------

static void test_function_fingerprint_is_secondary()
{
	cout << "-- the function fingerprint is SECONDARY, and pinned as such --" << endl;

	Case a = shortCase( "fn-a", "simpleprop" );
	Case b = shortCase( "fn-b", "simpleprop" );
	b.fixture = "xor2"; // same x0/x1 columns, different label column

	Row ra = runCase( a, REV );
	Row rb = runCase( b, REV );

	expect( !ra.functionStartId.empty(), "it is populated" );
	expect( ra.functionStartId == rb.functionStartId,
		"two fixtures sharing input columns give the SAME function fingerprint" );
	expect( ra.weightStartId == rb.weightStartId,
		"...as do their weights, which is why that alone is not enough" );
	expect( ra.splitId != rb.splitId,
		"...while the SPLIT identity distinguishes the data" );
}

// --- 3. dataset and split identity -----------------------------------------

static void test_split_identity()
{
	cout << "-- the dataset/split identity --" << endl;

	Row a = runCase( shortCase( "s-a", "logistic" ), REV );
	Row b = runCase( shortCase( "s-b", "logistic" ), REV );

	expect( !a.splitId.empty() && a.splitId != "0x0000000000000000",
		"the split identity is nonempty" );
	expect( a.splitId == b.splitId,
		"the same fixture and configuration give the same split identity" );

	Case c = shortCase( "s-c", "logistic" );
	c.rows += 20;
	expect( runCase( c, REV ).splitId != a.splitId,
		"a different row count changes it" );
}

// --- 4. ITERATION SEMANTICS -------------------------------------------------
//
// The two accessors genuinely disagree, and this pins both halves of that fact:
// on a rule break getIterations() is one BEHIND the completed count, while on
// ceiling exhaustion it equals it. Reporting the index as though it were the
// count -- which the first version of this harness did -- understates the work
// by exactly one iteration on every converged arm.

static void test_iteration_semantics()
{
	cout << "-- iteration index versus iterations COMPLETED --" << endl;

	Case c;
	if ( !findCase( "logistic-canonical", c ) )
	{
		expect( false, "the logistic-canonical case exists" );
		return;
	}
	Row r = runCase( c, REV );
	expect( r.usable, "the reference arm is usable" );
	expectEq( r.iterationsCompleted, ( long long ) r.iterationIndex + 1,
		"on a stopping-rule break the INDEX is one behind the COUNT" );
	expectEq( r.fullPasses, r.iterationsCompleted,
		"...and with no step-size search, one pass per completed iteration" );

	Case ceil;
	findCase( "impossible-target", ceil );
	Row rc = runCase( ceil, REV );
	expectEq( rc.iterationsCompleted, ( long long ) ceil.ceiling + 1,
		"on ceiling exhaustion the count is ceiling+1 (train runs 0..max)" );
	expectEq( rc.iterationIndex, ( long long ) rc.iterationsCompleted,
		"...where the index happens to EQUAL the count -- the other meaning" );
}

// --- 5. the ceiling ---------------------------------------------------------

static void test_ceiling_is_failure()
{
	cout << "-- an impossible target is a FAILURE, never a fast result --" << endl;

	Case c;
	findCase( "impossible-target", c );
	Row r = runCase( c, REV );

	expect( r.stopReason == string( "max_iterations" ), "it ends at the ceiling" );
	expect( !r.converged, "...which converged() reports as NOT converged" );
	expect( !r.targetReached, "...the target was not reached" );
	expect( !r.usable, "...and the row is NOT usable for a speed comparison" );
	expect( !r.error.empty(), "...with an explicit reason" );
	expect( r.failureStage == string( "none" ),
		"...and no failure STAGE, because nothing threw" );
	expect( r.elapsedNs > 0, "...while its elapsed time is still recorded" );
}

// --- 6. a reachable target --------------------------------------------------

static void test_reachable_target()
{
	cout << "-- a reachable target is usable, end state included --" << endl;

	Case c;
	findCase( "simpleprop-shanno", c );
	Row r = runCase( c, REV );

	expect( r.finite, "the objective is finite" );
	expect( r.stopReason == string( "min_error" ), "it stops on the target rule" );
	expect( r.converged, "...which is a convergence" );
	expect( r.achieved < c.target, "...below the target" );
	expect( r.iterationsCompleted > 0 && r.fullPasses > 0, "...with real work counted" );
	expect( !r.weightEndId.empty(), "...a final weight identity was taken" );
	expect( !r.functionEndId.empty(), "...and a final function fingerprint" );
	expect( r.usable, "...so the row is usable" );
}

// --- 7. the pass counter sees the search ------------------------------------

static void test_pass_counting()
{
	cout << "-- the counter distinguishes step-size trial passes --" << endl;

	Case off, on;
	findCase( "passcount-nosearch", off );
	findCase( "passcount-search", on );

	Row a = runCase( off, REV );
	Row b = runCase( on, REV );

	expectEq( a.iterationsCompleted, b.iterationsCompleted,
		"both arms complete the same number of iterations" );
	expect( a.fullPasses > 0 && b.fullPasses > 0, "both counted passes" );
	expectEq( a.fullPasses, a.iterationsCompleted,
		"without the search, one pass per iteration" );
	expectEq( b.fullPasses, 4 * a.fullPasses,
		"with it, four -- three trials and one real pass (maxLoops = 3)" );
}

// --- 8. every runtime failure still emits one row ---------------------------
//
// The injected faults are benchmark-only and unreachable from the pilot table or
// any command-line flag. Without this the probe would TERMINATE on a throwing
// arm and the orchestrator would abandon the whole campaign -- which matters
// immediately for Step 0B's singular, separated and non-finite cases, where
// throwing is the engine behaving correctly.

static void test_failure_rows()
{
	cout << "-- a throwing arm emits a failed row instead of terminating --" << endl;

	{
		Case c = baseCase();
		c.name = "inject-setup";
		c.group = "unit";
		c.model = "simpleprop";
		c.inject = Case::INJECT_SETUP;
		Row r = runCase( c, REV );

		expect( r.failureStage == string( "setup" ), "a setup fault is staged as setup" );
		expect( r.error.find( "exception during setup" ) != string::npos,
			"...with the exception's own message" );
		expect( !r.usable && !r.finite && !r.converged && !r.targetReached,
			"...and is consistently not usable" );
		expect( r.elapsedNs == 0, "...with no timing, because none happened" );
		expect( r.weightEndId.empty(),
			"...and no end identity from a model that never ran" );
		string line = toJsonLine( r );
		expect( line.find( "\"case\":\"inject-setup\"" ) != string::npos
			&& line.find( "\"usable\":false" ) != string::npos,
			"...emitted as one complete, marked row" );
	}

	{
		Case c = baseCase();
		c.name = "inject-training";
		c.group = "unit";
		c.model = "simpleprop";
		c.inject = Case::INJECT_TRAINING;
		Row r = runCase( c, REV );

		expect( r.failureStage == string( "training" ),
			"a training fault is staged as training" );
		expect( r.error.find( "exception during training" ) != string::npos,
			"...with the exception's own message" );
		expect( !r.usable, "...and is not usable" );
		expect( !r.weightStartId.empty(),
			"...while the START identity, taken before the fault, is retained" );
		expect( r.weightEndId.empty(),
			"...and NO end identity is taken from partially updated weights" );
		expect( r.fullPasses >= 1,
			"...with the pass count that did complete preserved" );
	}

	// The control: the same case without injection must succeed, or the two
	//    tests above would pass for the wrong reason.
	{
		Case c = baseCase();
		c.name = "inject-none";
		c.group = "unit";
		c.model = "simpleprop";
		Row r = runCase( c, REV );
		expect( r.usable && r.failureStage == string( "none" ),
			"the un-injected control succeeds (so the faults above are the cause)" );
	}
}

// --- 9. --characterize is refused on a non-canonical case -------------------
//
// The first version of this harness documented characterization as a canonical
// control and then only changed the target, so `--characterize --case
// simpleprop-shanno` cheerfully characterized Shanno. A target taken from the
// candidate being timed is not a matched endpoint.

static void test_characterize_requires_canonical()
{
	cout << "-- characterization is a CANONICAL control, and refuses not to be --" << endl;

	Case shan, canon;
	findCase( "simpleprop-shanno", shan );
	findCase( "simpleprop-canonical", canon );

	expect( !isCanonicalReference( shan ),
		"a Shanno arm is not a canonical reference" );
	expect( isCanonicalReference( canon ),
		"the canonical arm of the same group is" );
	expect( canonicalReferenceFor( shan.group ) == string( "simpleprop-canonical" ),
		"...and the group names which case to characterize instead" );

	Case search;
	findCase( "passcount-search", search );
	expect( !isCanonicalReference( search ),
		"an automatic-step arm is not a canonical reference either" );

	expect( canonicalReferenceFor( "no-such-group" ).empty(),
		"an unknown group yields no reference rather than a guess" );
}

// --- 10. schema and refusal -------------------------------------------------

static void test_row_schema()
{
	cout << "-- the emitted row carries every required field --" << endl;

	Case c;
	findCase( "logistic-canonical", c );
	string line = toJsonLine( runCase( c, REV ) );

	const char* required[] = {
		"\"schema\":", "\"rev\":", "\"dirty\":", "\"source_id\":", "\"build\":",
		"\"case\":", "\"comparison_group\":", "\"group_axis\":", "\"fixture\":",
		"\"split\":", "\"model\":", "\"arch\":", "\"loss\":", "\"rows\":",
		"\"inputs\":", "\"params\":", "\"weight_seed\":",
		"\"weight_id_available\":", "\"weight_start_id\":", "\"weight_end_id\":",
		"\"weight_elements\":", "\"weight_id_note\":", "\"function_start_id\":",
		"\"function_end_id\":", "\"optimizer\":", "\"optimizer_name\":",
		"\"mode\":", "\"eta\":", "\"auto_step\":", "\"decay_on\":", "\"decay\":",
		"\"grad_stop\":", "\"target\":", "\"achieved\":", "\"ceiling\":",
		"\"iteration_index\":", "\"iterations_completed\":", "\"full_passes\":",
		"\"elapsed_ns\":", "\"peak_rss_kb\":", "\"stop_reason\":",
		"\"converged\":", "\"target_reached\":", "\"finite\":", "\"usable\":",
		"\"failure_stage\":", "\"error\":", 0
	};
	for ( unsigned i = 0; required[ i ]; i++ )
		expect( line.find( required[ i ] ) != string::npos,
			string( "the row contains " ) + required[ i ] );

	expect( line.find( "\"data_seed\"" ) == string::npos,
		"data_seed is GONE: it had no effect on the fixture" );
	expect( line.find( "nan" ) == string::npos && line.find( "inf" ) == string::npos,
		"no bare nan/inf token: a non-finite number is emitted as null" );
	expect( line.find( "\"schema\":2" ) != string::npos,
		"the schema version is 2 (field meanings changed)" );
}

static void test_refusals()
{
	cout << "-- bad configuration is refused BY NAME --" << endl;

	struct Bad { const char* field; Case ( *make )(); };

	{
		Case c = baseCase(); c.name = "b1"; c.group = "u"; c.model = "perceptron";
		Row r = runCase( c, REV );
		expect( !r.usable && r.error.find( "model" ) != string::npos
			&& r.error.find( "perceptron" ) != string::npos,
			"an unknown model is refused, naming field and value" );
		expect( r.failureStage == string( "refused" ),
			"...staged as a refusal, not as a setup fault" );
	}
	{
		Case c = baseCase(); c.name = "b2"; c.group = "u"; c.fixture = "moons";
		expect( runCase( c, REV ).error.find( "fixture" ) != string::npos,
			"an unknown fixture is refused by name" );
	}
	{
		Case c = baseCase(); c.name = "b3"; c.group = "u"; c.optimizer = 7;
		expect( runCase( c, REV ).error.find( "optimizer" ) != string::npos,
			"an unknown optimizer is refused by name" );
	}
	{
		// NDEBUG strips setMinError's assert, so the harness is the only thing
		//    between a nonsense target and a silently installed one.
		Case c = baseCase(); c.name = "b4"; c.group = "u"; c.target = 4.0;
		expect( runCase( c, REV ).error.find( "target" ) != string::npos,
			"a target outside (0,1) is refused by name" );
	}
	{
		Case c = baseCase(); c.name = "b5"; c.group = "u";
		c.target = numeric_limits< double >::quiet_NaN();
		expect( runCase( c, REV ).error.find( "target" ) != string::npos,
			"a NaN target is refused by name" );
	}
	{
		Case c = baseCase(); c.name = "b6"; c.group = "u"; c.eta = 5.0;
		expect( runCase( c, REV ).error.find( "eta" ) != string::npos,
			"an eta outside [0,1] is refused by name" );
	}
	{
		Case c = baseCase(); c.name = "b7"; c.group = "u"; c.ceiling = 0;
		expect( runCase( c, REV ).error.find( "ceiling" ) != string::npos,
			"a zero ceiling is refused by name" );
	}
	{
		Case c = baseCase(); c.name = "b8"; c.group = "u"; c.model = "backprop";
		expect( runCase( c, REV ).error.find( "layers" ) != string::npos,
			"a backprop with no hidden layer is refused by name" );
	}
	{
		Case c = baseCase(); c.name = "b9"; c.group = "";
		expect( runCase( c, REV ).error.find( "comparison_group" ) != string::npos,
			"a case with no comparison group is refused by name" );
	}
	{
		Case c;
		expect( !findCase( "no-such-case", c ), "an unknown case name is not found" );
	}
}

// --- 11. every pilot case declares a coherent comparison group --------------

static void test_pilot_groups_are_fair()
{
	cout << "-- every pilot comparison group is a FAIR comparison --" << endl;

	vector< Case > all = pilotCases();
	unsigned checked = 0;

	for ( size_t i = 0; i < all.size(); i++ )
		for ( size_t j = i + 1; j < all.size(); j++ )
		{
			if ( all[ i ].group != all[ j ].group ) continue;
			const Case& a = all[ i ];
			const Case& b = all[ j ];
			checked++;

			// Everything that defines the WORK must match. Only the declared
			//    axis may differ.
			bool sameWork = a.model == b.model && a.fixture == b.fixture
				&& a.rows == b.rows && a.weightSeed == b.weightSeed
				&& a.hidden == b.hidden && a.layers == b.layers
				&& a.batch == b.batch && a.decayOn == b.decayOn
				&& a.decay == b.decay && a.target == b.target
				&& a.ceiling == b.ceiling && a.xentropy == b.xentropy
				&& a.eta == b.eta;
			expect( sameWork, "group '" + a.group + "': " + a.name + " and "
				+ b.name + " define the same work" );

			if ( a.groupAxis == "optimizer" )
			{
				expect( a.gradStop == b.gradStop && a.autoStep == b.autoStep,
					"group '" + a.group + "': same branch and step policy, so "
					"only the optimizer differs" );
				expect( a.optimizer != b.optimizer,
					"group '" + a.group + "': the arms actually differ in optimizer" );
			}
			else if ( a.groupAxis == "auto_step" )
				expect( a.autoStep != b.autoStep && a.optimizer == b.optimizer,
					"group '" + a.group + "': varies auto_step alone" );
			else if ( a.groupAxis == "grad_stop branch" )
				expect( a.gradStop != b.gradStop && a.optimizer == b.optimizer,
					"group '" + a.group + "': varies the grad_stop branch alone" );
		}

	expect( checked > 0, "there are multi-arm groups to check (non-vacuity)" );

	// The specific fairness defect that prompted this: canonical timed against
	//    CGD/Shanno while running a different production branch.
	Case canon, cgd, shan;
	findCase( "simpleprop-canonical", canon );
	findCase( "simpleprop-cgd", cgd );
	findCase( "simpleprop-shanno", shan );
	expect( canon.gradStop && cgd.gradStop && shan.gradStop,
		"the SimpleProp optimizer group all use the separate-gradient branch" );
	expect( canon.group == cgd.group && cgd.group == shan.group,
		"...and are declared in one comparison group" );

	// The fast accumulator branch is retained, but in its OWN group.
	Case acc;
	findCase( "simpleprop-canonical-accumulator", acc );
	expect( !acc.gradStop && acc.group != canon.group,
		"the fast accumulator branch is kept in a separate, honestly labelled group" );
}

// --- 12. no artifacts left behind ------------------------------------------

static bool exists( const char* path )
{
	struct stat st;
	return stat( path, &st ) == 0;
}

static void test_no_artifacts()
{
	cout << "-- the harness leaves nothing behind --" << endl;

	bool hadModel = exists( "model.txt" );
	bool hadLog = exists( "neuron.log" );

	Case c;
	findCase( "logistic-canonical", c );
	runCase( c, REV );

	expect( hadModel || !exists( "model.txt" ), "no model.txt is created" );
	expect( hadLog || !exists( "neuron.log" ), "no neuron.log is created" );
}

int main()
{
	test_weight_identity();
	test_backprop_traversal();
	test_empty_weights_refused();
	test_function_fingerprint_is_secondary();
	test_split_identity();
	test_iteration_semantics();
	test_ceiling_is_failure();
	test_reachable_target();
	test_pass_counting();
	test_failure_rows();
	test_characterize_requires_canonical();
	test_row_schema();
	test_refusals();
	test_pilot_groups_are_fair();
	test_no_artifacts();

	cout << endl << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
