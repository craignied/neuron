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
#include <filesystem>
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

// --- 2b. the Phase 4 conditioning pair is not a no-op -----------------------
//
// THE FIXTURE THIS GUARDS ALMOST DID NOT WORK. The obvious way to build a
// poorly scaled fixture -- multiply each input column by a different constant --
// produces a design matrix BIT-IDENTICAL to the well-scaled one, because
// DataSet::normalize min-max normalizes every input column and exactly cancels
// any per-column linear rescale. An arm built that way would have reported "no
// difference on poorly scaled data" from a fixture that was not poorly scaled.
//
// So the pair is checked for the one property the whole comparison rests on:
// that the two fixtures reach training with DIFFERENT data. Sharing the outcome
// rule and the row count is intended; sharing the split identity would mean the
// experiment has no independent variable.

static void test_conditioning_pair_differs()
{
	cout << "-- the conditioning pair really is two conditionings --" << endl;

	Case a = shortCase( "cond-well", "simpleprop" );
	a.fixture = "well4";
	Case b = shortCase( "cond-poor", "simpleprop" );
	b.fixture = "poor4";

	Row ra = runCase( a, REV );
	Row rb = runCase( b, REV );

	expect( ra.rows == rb.rows && ra.inputs == rb.inputs
			&& ra.params == rb.params,
		"both fixtures present the same shape: rows, inputs and parameters" );
	expect( ra.inputs == 4,
		"...and the fixture's own width is read from it, not assumed to be 2" );
	expect( ra.splitId != rb.splitId,
		"...while their SPLIT identities differ, so the scaling survived "
		"normalization and the comparison has an independent variable" );
	expect( ra.weightStartId == rb.weightStartId,
		"...from identical starting weights, so only the data differs" );
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
		"\"failure_stage\":", "\"error\":",
		// Schema 3 (Step 0B)
		"\"data_id\":", "\"data_seed\":", "\"test_fraction\":",
		"\"endpoint\":", "\"timing_scope\":", "\"workload\":",
		"\"cv_folds\":", "\"cv_repeats\":", "\"rows_total\":",
		"\"rows_test\":", "\"method\":", "\"heldout_error\":",
		"\"cv_auc\":", "\"locked_auc\":", "\"cv_folds_ok\":",
		"\"cv_folds_total\":", "\"engine_id\":", "\"engine_files\":", 0
	};
	for ( unsigned i = 0; required[ i ]; i++ )
		expect( line.find( required[ i ] ) != string::npos,
			string( "the row contains " ) + required[ i ] );

	// data_seed RETURNED in schema 3, and it now selects something. In schema 2
	//    it was removed as false provenance because fixtureMatrix() ignored it;
	//    a Step 0B arm's holdout is chosen by it, which is why test_real_split
	//    can show two seeds producing two different splits.
	expect( line.find( "\"data_seed\"" ) != string::npos,
		"data_seed is BACK, because it now chooses the split" );
	expect( line.find( "nan" ) == string::npos && line.find( "inf" ) == string::npos,
		"no bare nan/inf token: a non-finite number is emitted as null" );
	expect( line.find( "\"schema\":3" ) != string::npos,
		"the schema version is 3 (rows is now a training count, and a row "
		"declares its endpoint and its timing scope)" );
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


// --- 12. STEP 0B: a REAL split, with a seed that selects it -----------------
//
// Step 0A trained on the whole fixture, so `split` was the fixture identity and
// the schema's data seed selected nothing -- which is why it was removed as
// false provenance. These pin the replacement: the seed genuinely chooses which
// rows train, the SAME seed reproduces that choice, and the observations are a
// separate fact from the partition of them.
//
// The fixture is written HERE, to a temporary file, rather than read from the
// prepared Civic Choice directory: a ctest case that depends on generated data
// fails on a fresh clone, and these mechanics do not need real data to be true.

static string writeSplitFixture()
{
	// /tmp is not a temporary directory on Windows; the hardcoded path made
	//    fopen fail on windows-latest CI and emptied the fixture path for the
	//    three split-unit checks. Use the platform temp root, as
	//    check_writelastop.cpp already does for the same reason.
	string path = ( std::filesystem::temp_directory_path()
		/ "optbench_split_fixture.txt" ).string();
	FILE* f = fopen( path.c_str(), "w" );
	if ( !f ) return "";
	// Three inputs and a 0/1 outcome, deterministic, balanced enough for a
	//    stratified holdout to have both classes on both sides.
	//
	//    DELIBERATELY NOISY. A separable problem never plateaus: its loss keeps
	//    falling and its held-out error keeps creeping down as the weights grow,
	//    so the horizon-independence test below had nothing to detect and failed
	//    at every ceiling tried. Flipping a fixed one row in seven gives the
	//    held-out error a floor to reach -- which is what a real dataset has and
	//    what a practical endpoint is defined against.
	for ( unsigned i = 0; i < 400; i++ )
	{
		double x0 = ( double ) ( ( i * 37 ) % 100 ) / 99.0;
		double x1 = ( double ) ( ( i * 53 ) % 100 ) / 99.0;
		double x2 = ( double ) ( ( i * 17 ) % 100 ) / 99.0;
		unsigned y = ( x0 + x1 > 1.0 ) ? 1u : 0u;
		if ( ( i * 13 ) % 7 == 0 ) y = 1u - y;   // deterministic label noise
		fprintf( f, "%.6f, %.6f, %.6f, %u\n", x0, x1, x2, y );
	}
	fclose( f );
	return path;
}

static Case splitCase( const string& name, unsigned dataSeed, const string& path )
{
	Case c = baseCase();
	c.name = name;
	c.group = "split-unit";
	c.groupAxis = "method";
	c.model = "logistic";
	c.dataFile = path;
	c.inputs = 3;
	c.dataSeed = dataSeed;
	c.testFraction = 0.25;
	c.rows = 0;
	c.ceiling = 2;
	c.target = 0.999;   // reachable or not; these tests are about the SPLIT
	c.endpoint = ENDPOINT_PRACTICAL;
	return c;
}

static void test_real_split()
{
	cout << "-- Step 0B: a real holdout split, chosen by a real seed --" << endl;

	string path = writeSplitFixture();
	expect( !path.empty(), "the split fixture was written" );
	if ( path.empty() ) return;

	Row a = runCase( splitCase( "sp-a", 101, path ), REV );
	Row again = runCase( splitCase( "sp-again", 101, path ), REV );
	Row b = runCase( splitCase( "sp-b", 202, path ), REV );

	expect( a.failureStage == "none", "the file-backed case ran" );
	if ( a.failureStage != "none" ) cout << "         " << a.error << endl;

	expectEq( a.rowsTotal, 400, "every row of the file was loaded" );
	expectEq( a.rowsTest, 100, "a 0.25 holdout took a quarter of them" );
	expectEq( a.rows, 300, "the rest are the training rows" );
	expectEq( ( long long ) a.rows + ( long long ) a.rowsTest,
		( long long ) a.rowsTotal, "the partition accounts for every row" );

	// THE TWO IDENTITIES ANSWER TWO QUESTIONS. Same observations, different
	//    partition of them: dataId must hold still while splitId moves. If one
	//    hash served both, a reseeded campaign would look like the same run.
	expect( a.dataId == b.dataId,
		"a different split seed does NOT change the data identity" );
	expect( a.splitId != b.splitId,
		"a different split seed DOES change the split identity" );
	expect( a.splitId == again.splitId,
		"the same split seed reproduces the same split identity" );
	expect( a.dataId != a.splitId,
		"the data identity and the split identity are different hashes" );

	// The split seed is now REPORTED, and reports the value that was used.
	expectEq( a.dataSeed, 101, "the row carries the seed that chose its split" );
	expectEq( b.dataSeed, 202, "and a different arm carries its own" );

	remove( path.c_str() );
}

// --- 13. STEP 0B: arms of one group start from ONE parameter state ----------
//
// The whole comparison rests on it. Step 0A proved it on generated fixtures;
// this proves it survives a real file-backed split, where the DataSet is built
// by a different path and the weight count comes from a real design matrix.

static void test_split_arms_share_a_start()
{
	cout << "-- Step 0B: every method starts from one parameter state --" << endl;

	string path = writeSplitFixture();
	if ( path.empty() ) { expect( false, "fixture" ); return; }

	Case base = splitCase( "start-canonical", 101, path );
	Case cgd = base; cgd.name = "start-cgd"; cgd.optimizer = 1;
	Case shanno = base; shanno.name = "start-shanno"; shanno.optimizer = 2;
	Case autostep = base; autostep.name = "start-auto"; autostep.autoStep = true;

	Row a = runCase( base, REV );
	Row b = runCase( cgd, REV );
	Row c = runCase( shanno, REV );
	Row d = runCase( autostep, REV );

	expect( a.weightIdAvailable && !a.weightStartId.empty(),
		"the file-backed arm carries a parameter-state identity" );
	expect( a.weightStartId == b.weightStartId
		&& a.weightStartId == c.weightStartId
		&& a.weightStartId == d.weightStartId,
		"all four methods start from the identical weight state" );
	expect( a.weightElements == a.params && a.params > 0,
		"the hash covered exactly the model's parameters" );

	// AND THE METHOD NAMES ARE DISTINCT, because two of these four share a
	//    trainingType. A comparison that called both "canonical" would report
	//    two different jobs under one name.
	expect( a.method == "canonical" && d.method == "canonical-autostep",
		"the step-size search is a different METHOD, not a different setting" );
	expect( b.method == "cgd" && c.method == "shanno",
		"the other two methods name themselves" );

	remove( path.c_str() );
}

// --- 14. STEP 0B: the endpoint and the scope are part of the configuration --
//
// Two fields whose whole job is to stop a number being read as something it is
// not. Both are refused by NAME when they are impossible, in the C++ layer, so
// a bad configuration never produces a row at all.

static void test_endpoint_and_scope_refusals()
{
	cout << "-- Step 0B: endpoint, scope and workload refusals --" << endl;

	Case c = shortCase( "e-1", "logistic" );
	c.endpoint = "eventually";
	Row r = runCase( c, REV );
	expect( r.failureStage == "refused"
		&& r.error.find( "endpoint:" ) != string::npos,
		"an unknown endpoint is refused BY NAME" );

	c = shortCase( "e-2", "logistic" );
	c.timingScope = "wall";
	r = runCase( c, REV );
	expect( r.failureStage == "refused"
		&& r.error.find( "timing_scope:" ) != string::npos,
		"an unknown timing scope is refused by name" );

	c = shortCase( "e-3", "logistic" );
	c.workload = "stepwise";
	r = runCase( c, REV );
	expect( r.failureStage == "refused"
		&& r.error.find( "workload:" ) != string::npos,
		"an unknown workload is refused by name" );

	// THE MISLABELLING REFUSAL. A cv arm's clock necessarily covers each fold's
	//    scoring epilogue, so it cannot be an optimizer-only timing -- and the
	//    engine layer refuses the combination rather than leaving it to a
	//    reader to notice.
	c = shortCase( "e-4", "logistic" );
	c.workload = "cv";
	c.timingScope = SCOPE_OPTIMIZER;
	c.cvFolds = 5; c.cvRepeats = 2; c.testFraction = 0.25;
	r = runCase( c, REV );
	expect( r.failureStage == "refused"
		&& r.error.find( "timing_scope: a cv workload" ) != string::npos,
		"a cv workload claiming optimizer scope is REFUSED, not annotated" );

	c = shortCase( "e-5", "logistic" );
	c.cvFolds = 5;
	r = runCase( c, REV );
	expect( r.failureStage == "refused"
		&& r.error.find( "cv_folds/cv_repeats" ) != string::npos,
		"a non-cv case setting fold counts is refused by name" );

	c = shortCase( "e-6", "logistic" );
	c.workload = "cv"; c.timingScope = SCOPE_WORKFLOW;
	c.cvFolds = 5; c.cvRepeats = 2; c.testFraction = 0.0;
	r = runCase( c, REV );
	expect( r.failureStage == "refused"
		&& r.error.find( "locked test set" ) != string::npos,
		"a cv workload with nothing locked away is refused by name" );

	c = shortCase( "e-7", "logistic" );
	c.testFraction = 1.5;
	r = runCase( c, REV );
	expect( r.failureStage == "refused"
		&& r.error.find( "test_fraction:" ) != string::npos,
		"an impossible holdout fraction is refused by name" );
}

// --- 15. STEP 0B: an UNCHARACTERIZED workload cannot run --------------------
//
// The committed target table is a table of MEASUREMENTS, and 0 means "not yet
// measured". A case referring to an unmeasured workload must not quietly run
// against a meaningless endpoint -- which is the same failure as a target
// chosen to flatter an arm, arrived at by neglect instead of intent.

static void test_uncharacterized_target_refused()
{
	cout << "-- Step 0B: an uncharacterized workload is refused --" << endl;

	Case c = civicCase( "logistic", 999999, 0, ENDPOINT_PRACTICAL, 0, false );
	expectEq( ( long long ) ( c.target * 1e9 ), 0,
		"a workload absent from the committed table gets no target" );
	Row r = runCase( c, REV );
	expect( r.failureStage == "refused"
		&& r.error.find( "target:" ) != string::npos,
		"and it is refused by name rather than run" );

	// The endpoint table is a lookup, not a guess: a key it does not hold
	//    returns nothing at all.
	expect( endpointsFor( "no-such-workload" ) == 0,
		"an unknown endpoint key resolves to nothing" );
	expect( endpointsFor( "logistic-6000-linear" ) != 0,
		"a known one resolves" );
}

// --- 16. STEP 0B: every declared case is well formed -------------------------
//
// The Step 0B table is generated by helper functions rather than written out by
// hand, so a mistake in a helper would produce dozens of malformed cases at
// once. validate() is the same gate the probe applies; running it over the
// whole table here means the campaign cannot begin with a case that would be
// refused halfway through it.

static void test_step0b_table_is_well_formed()
{
	cout << "-- Step 0B: the declared workload matrix --" << endl;

	vector< Case > all = step0bCases();
	expect( all.size() > 20, "the Step 0B table is populated" );

	unsigned bad = 0, cvArms = 0, practical = 0, strict = 0;
	for ( size_t i = 0; i < all.size(); i++ )
	{
		string why = validate( all[ i ] );
		if ( !why.empty() )
		{
			if ( bad < 3 )
				cout << "         " << all[ i ].name << ": " << why << endl;
			bad++;
		}
		if ( all[ i ].workload == "cv" ) cvArms++;
		if ( all[ i ].endpoint == ENDPOINT_PRACTICAL ) practical++;
		if ( all[ i ].endpoint == ENDPOINT_STRICT ) strict++;
	}
	expectEq( bad, 0, "every declared Step 0B case passes validate()" );
	expect( cvArms >= 2, "the repeated-fit consumer is represented" );
	expect( practical > 0 && strict > 0, "both endpoints are represented" );

	// NAMES ARE UNIQUE. --case resolves by name, so a duplicate would silently
	//    make one of two different configurations unreachable.
	vector< Case > every = allCases();
	unsigned dupes = 0;
	for ( size_t i = 0; i < every.size(); i++ )
		for ( size_t j = i + 1; j < every.size(); j++ )
			if ( every[ i ].name == every[ j ].name ) dupes++;
	expectEq( dupes, 0, "every case name in the whole table is unique" );

	// EVERY GROUP HAS A CANONICAL REFERENCE to characterize from, except the
	//    ones that deliberately have none. A group whose target could not have
	//    come from a canonical control is a group with no matched endpoint.
	unsigned groupless = 0;
	for ( size_t i = 0; i < all.size(); i++ )
		if ( canonicalReferenceFor( all[ i ].group ).empty() ) groupless++;
	expectEq( groupless, 0,
		"every Step 0B group declares a canonical reference case" );
}


// --- 17. STEP 0B: the practical endpoint does not move when you watch longer -
//
// THE DEFECT THIS PINS WAS REAL AND WAS SHIPPED IN AN EARLIER DRAFT OF THIS
// FILE'S SIBLING. The first practical-endpoint rule took the BEST held-out
// error over the whole characterization and asked when the series first came
// within 1% of it. That makes the endpoint a function of how long you looked: a
// longer run finds a better best, moves the band, and moves the endpoint.
// Measured on the Civic Choice neural workload, the same configuration put its
// practical endpoint at iteration 11,299 under a 20,000 ceiling and 78,764
// under a 100,000 one -- two different endpoints for one workload.
//
// The replacement is PlateauDetector, the engine's own local detector, which
// fires where the series flattens and cannot see what happens afterwards.
//
// The test needs a workload whose CEILING ACTUALLY BINDS. On a run that
// converges before either ceiling, both characterizations are the same run and
// the comparison proves nothing -- which is exactly what a first attempt at this
// test did, passing under the sabotage.

// A held-out trace with a KNOWN shape: a fast improvement that has flattened by
// a few hundred iterations, and then keeps creeping down by a hair forever. The
// creep is the whole point -- it is what a global-best rule chases and a local
// plateau rule ignores.
static vector< double > heldoutTrace( unsigned n )
{
	vector< double > v( n );
	for ( unsigned i = 0; i < n; i++ )
		v[ i ] = 0.40 + 0.20 * exp( -( double ) i / 60.0 ) - 1e-7 * i;
	return v;
}

static vector< double > objectiveTrace( unsigned n )
{
	vector< double > v( n );
	for ( unsigned i = 0; i < n; i++ )
		v[ i ] = 0.30 + 0.30 * exp( -( double ) i / 60.0 );
	return v;
}

// The rule the FIRST version of this used, kept here as the test's own control.
// It is not called by anything else and is not a fallback: its only job is to
// show that the horizon really does move an endpoint under a global rule, so
// that the plateau rule holding still is a result and not an artifact of a
// trace nothing could have moved.
static unsigned globalBestEndpoint( const vector< double >& heldout )
{
	double best = heldout[ 0 ];
	for ( size_t i = 1; i < heldout.size(); i++ )
		if ( heldout[ i ] < best ) best = heldout[ i ];
	double threshold = best * 1.01;
	for ( size_t i = 0; i < heldout.size(); i++ )
		if ( heldout[ i ] <= threshold ) return ( unsigned ) i;
	return ( unsigned ) heldout.size() - 1;
}

static void test_practical_endpoint_is_horizon_independent()
{
	cout << "-- Step 0B: the practical endpoint is horizon-independent --" << endl;

	PracticalPoint shortRun = practicalEndpoint( objectiveTrace( 3000 ),
		heldoutTrace( 3000 ) );
	PracticalPoint longRun = practicalEndpoint( objectiveTrace( 30000 ),
		heldoutTrace( 30000 ) );

	expect( shortRun.fired && longRun.fired,
		"the series plateaus under both horizons" );
	expectEq( shortRun.iteration, longRun.iteration,
		"a TEN-FOLD longer horizon does not move the practical endpoint" );
	expect( shortRun.objective == longRun.objective,
		"and the objective it names is bit-identical" );

	// THE CONTROL. Without it this test could pass on a trace no rule could
	//    have moved, and would be asserting nothing. The rule this replaced --
	//    "within 1% of the best held-out error the whole run achieved" -- must
	//    demonstrably move on the same trace.
	unsigned gShort = globalBestEndpoint( heldoutTrace( 3000 ) );
	unsigned gLong = globalBestEndpoint( heldoutTrace( 30000 ) );
	expect( gShort != gLong,
		"the discarded global-best rule DOES move on this same trace, so the "
		"horizon is genuinely capable of moving an endpoint here" );
	if ( gShort != gLong )
		cout << "         (global-best would say " << gShort << " then "
			<< gLong << "; the plateau says " << shortRun.iteration
			<< " both times)" << endl;

	// The same defect, as it was actually measured on the real workload before
	//    the rule changed: Civic Choice neural, 4 hidden units, put its
	//    practical endpoint at iteration 11,299 under a 20,000 ceiling and
	//    78,764 under a 100,000 one. Recorded in
	//    docs/learning_research/optimizer_baseline_results.md.
}

// --- 18. STEP 0B: watching the held-out set must not change the fit ---------
//
// Legacy bug #10's exact shape, in a new place: the gradient used to be
// recalculated only inside the print block, so the REPORTING CADENCE chose the
// model. Characterization samples the held-out set every iteration, which calls
// forward() on held-out rows and writes the network's scratch output. If that
// perturbed training, every endpoint derived from the watched run would
// describe a different fit from the one the arms perform.
//
// characterize() proves it per run and REFUSES on a mismatch. This asserts the
// guard exists, ran, and passed -- an unverified guard is a comment.

static void test_heldout_sampling_does_not_change_the_fit()
{
	cout << "-- Step 0B: sampling the held-out set does not move the fit --" << endl;

	string path = writeSplitFixture();
	if ( path.empty() ) { expect( false, "fixture" ); return; }

	Case c = splitCase( "guard", 101, path );
	Characterization ch = characterize( c, 1500 );

	expect( ch.samplingWasFree,
		"the watched and unwatched objective trajectories are identical" );
	expect( ch.guardIterations > 1,
		"the guard actually ran iterations to compare" );
	// The ceiling leaves the loop at maxIterations+1 -- the same two-meanings
	//    accessor Step 0A pinned -- so the guard runs the whole window and one
	//    more. Asserted as "at least", because the point is that it ran the
	//    window, not which side of the off-by-one the engine lands on.
	expect( ch.guardIterations >= GUARD_ITERATIONS,
		"over the whole declared guard window" );

	remove( path.c_str() );
}

// --- 19. STEP 0B: characterization is still a canonical control -------------

static void test_characterize_rejects_a_workflow_arm()
{
	cout << "-- Step 0B: characterization refuses what it cannot characterize --"
		<< endl;

	Case c = shortCase( "chz-cv", "logistic" );
	c.workload = "cv";
	c.timingScope = SCOPE_WORKFLOW;
	c.cvFolds = 5; c.cvRepeats = 1; c.testFraction = 0.25;
	c.target = 0.5;
	Characterization ch = characterize( c, 100 );
	expect( !ch.ok && ch.error.find( "only a fit workload" ) != string::npos,
		"a cv arm has no training-objective trajectory and is refused by name" );

	// A workload with no held-out set has no practical endpoint to derive, and
	//    says so rather than inventing one from the training objective alone.
	Case d = shortCase( "chz-nosplit", "logistic" );
	d.target = 0.5;
	d.testFraction = 0.0;
	Characterization dh = characterize( d, 600 );
	expect( !dh.ok && dh.error.find( "no held-out set" ) != string::npos,
		"a workload with no holdout yields no practical endpoint, by name" );
}

int main()
{
	test_weight_identity();
	test_backprop_traversal();
	test_empty_weights_refused();
	test_function_fingerprint_is_secondary();
	test_conditioning_pair_differs();
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
	test_real_split();
	test_split_arms_share_a_start();
	test_endpoint_and_scope_refusals();
	test_uncharacterized_target_refused();
	test_step0b_table_is_well_formed();
	test_practical_endpoint_is_horizon_independent();
	test_heldout_sampling_does_not_change_the_fit();
	test_characterize_rejects_a_workflow_arm();
	test_no_artifacts();

	cout << endl << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}
