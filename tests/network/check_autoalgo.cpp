// check_autoalgo.cpp : the ONE eligibility rule (autoalgo::ineligible), the
// curated candidate list it produces, and the automatic selector that probes it.
//
// WHY THIS FILE EXISTS. Until Levenberg-Marquardt became public there were two
// separate hand-written copies of "which optimizer can run here": one in
// gui.cpp, refusing a directly selected method, and one in autoalgo::pick,
// which had no such notion at all because its three candidates could always
// run. Those are one rule. Written twice they drift, and the drift is silent in
// the worst possible direction -- a method probed under a configuration it
// refuses throws Ineligible, pick()'s divergence handler turns any exception
// into a NaN, and the user is told that a perfectly good optimizer diverged.
//
// SO THE ASSERTIONS HERE ARE PAIRED. Every refusal is asserted together with a
// CONTROL that differs in exactly the offending field and must be eligible.
// A test that only ever asserts "ineligible" would pass against a rule that
// refused everything, which is the same defect one layer up.
//
// AND THE CANDIDATE LIST IS ASSERTED AS A SET, not as a count. A count passes
// when one method is silently swapped for another; the sabotage log at the
// bottom of this file records that the inclusion of LM in an ELIGIBLE list is
// one of the two production-wiring sabotage targets for this change.

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "autoalgo.h"
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

// Engine chatter belongs nowhere near the assertions.
struct Hush {
	ostringstream sink;
	ostream& prev;
	Hush() : prev( util::screen() ) { util::set_screen( sink ); }
	~Hush() { util::set_screen( prev ); }
};

// A separable, learnable problem. Small on purpose: the probes below are
//    metered in wall clock, so the point is that they all make progress, not
//    that any of them finishes.
static DataSet makeData( unsigned rows, unsigned inputs, unsigned seed )
{
	Matrix< double > raw( rows, inputs + 1 );
	unsigned s = seed;
	for ( unsigned i = 0; i < rows; i++ )
	{
		double acc = 0;
		for ( unsigned j = 0; j < inputs; j++ )
		{
			s = s * 1103515245u + 12345u;
			const double v = -1.0 + 2.0 * ( double ) ( ( s >> 16 ) % 1000 ) / 999.0;
			raw( i, j ) = v;
			acc += v * ( ( j % 2 ) ? -1.0 : 1.0 );
		}
		raw( i, inputs ) = ( acc > 0.0 ) ? 1.0 : 0.0;
	}
	DataSet d;
	d.setOutput( 1 );
	d.setDiscrete( true );
	d.setHistory( false );
	d.setInput( inputs );
	d.setRawMatrix( raw );
	Matrix< double >& r = d.getRawMatrix();
	d.setTrainMatrix( r );
	return d;
}

// The same problem, split three ways, so the VALIDATION half of Levenberg-
//    Marquardt's public contract has a dataset that actually carries one.
static DataSet makeThreeWayData( unsigned rows, unsigned inputs, unsigned seed )
{
	DataSet d = makeData( rows, inputs, seed );
	d.randomize3( rows / 5, rows / 5 ); // test, validation
	return d;
}

// THE WHOLE CURATED LIST, spelled as an empty request. algorithm=auto means
//    this; a named subset is any other vector. Written out rather than passed as
//    a literal so that every call site says which of the two it is asking for.
static const vector< unsigned > ALL;

// A named subset, in the order a caller might have written it.
static vector< unsigned > setOfTypes( unsigned a, unsigned b,
	unsigned c = ( unsigned ) -1 )
{
	vector< unsigned > v;
	v.push_back( a );
	v.push_back( b );
	if ( c != ( unsigned ) -1 ) v.push_back( c );
	return v;
}

static string setOf( const vector< unsigned >& list )
{
	string s;
	for ( unsigned i = 0; i < list.size(); i++ )
	{
		if ( i ) s += ",";
		s += to_string( list[ i ] );
	}
	return s;
}

// ---------------------------------------------------------------------------
// 1. THE NAMING OWNER. Both vocabularies must cover every training type, and
//    the prose names of the three legacy methods are pinned by the goldens.
// ---------------------------------------------------------------------------

static void checkNaming()
{
	expect( string( Network::algorithmName( 0 ) ) == "canonical backpropagation"
		&& string( Network::algorithmName( 1 ) ) == "conjugate gradient descent"
		&& string( Network::algorithmName( 2 ) ) == "Shanno",
		"the three legacy prose names are byte-identical to the goldens'" );
	expect( string( Network::algorithmName( Network::TRAIN_LBFGS ) ) == "L-BFGS"
		&& string( Network::algorithmName( Network::TRAIN_IRPROP ) ) == "iRPROP+"
		&& string( Network::algorithmName( Network::TRAIN_LM ) )
			== "Levenberg-Marquardt",
		"the retained modern methods are named by their published names" );
	expect( string( Network::algorithmLabel( Network::TRAIN_LM ) )
		== "Levenberg-Marquardt",
		"the compact label spells Levenberg-Marquardt in full, as L-BFGS is" );

	// Every type in the enumeration is named in BOTH vocabularies, and no two
	//    share a name -- a table that silently reused an entry would make one
	//    optimizer report as another.
	bool named = true, distinct = true;
	for ( unsigned t = 0; t < Network::TRAINING_TYPES; t++ )
	{
		if ( string( Network::algorithmName( t ) ) == "unknown"
			|| string( Network::algorithmLabel( t ) ) == "unknown" )
			named = false;
		for ( unsigned u = t + 1; u < Network::TRAINING_TYPES; u++ )
			if ( string( Network::algorithmName( t ) )
					== string( Network::algorithmName( u ) )
				|| string( Network::algorithmLabel( t ) )
					== string( Network::algorithmLabel( u ) ) )
				distinct = false;
	}
	expect( named, "every training type is named in both vocabularies" );
	expect( distinct, "no two training types share a name in either vocabulary" );
	expect( string( Network::algorithmName( Network::TRAINING_TYPES ) )
			== "unknown"
		&& string( Network::algorithmLabel( Network::TRAINING_TYPES ) )
			== "unknown",
		"a value past the enumeration reports unknown, it does not index past "
		"the table" );
}

// ---------------------------------------------------------------------------
// 2. THE ELIGIBILITY MATRIX, every refusal beside its control.
// ---------------------------------------------------------------------------

static void checkEligibility()
{
	Hush hush;
	DataSet data = makeData( 80, 4, 20260809u );

	SimpleProp lms;
	lms.setDataSet( data );
	lms.setHidden( 3 );
	lms.randomize();

	autoalgo::Settings ok = autoalgo::Settings::of( lms );
	expect( ok.batchEpoch && !ok.autoStep && ok.parameters == lms.packedSize(),
		"Settings::of reads the model's own batch/autostep/parameter state" );

	// --- THE CONTROL. Every one of the six must be eligible here, or every
	//     refusal below is vacuous.
	bool allEligible = true;
	for ( unsigned t = 0; t < Network::TRAINING_TYPES; t++ )
		if ( autoalgo::ineligible( lms, t, ok ) )
			allEligible = false;
	expect( allEligible, "CONTROL: on a batch LMS SimpleProp every one of the "
		"six methods is eligible" );

	// --- Batch/epoch off
	autoalgo::Settings online = ok;
	online.batchEpoch = false;
	expect( !autoalgo::ineligible( lms, 0, online )
			&& !autoalgo::ineligible( lms, 1, online )
			&& !autoalgo::ineligible( lms, 2, online ),
		"CONTROL: the three legacy methods still run on-line" );
	expect( autoalgo::ineligible( lms, Network::TRAIN_LBFGS, online )
		&& autoalgo::ineligible( lms, Network::TRAIN_IRPROP, online )
		&& autoalgo::ineligible( lms, Network::TRAIN_LM, online ),
		"batch_epoch=0 refuses all three retained modern optimizers" );
	expect( string( autoalgo::ineligible( lms, Network::TRAIN_LM, online ) )
		== "requires batch_epoch=1",
		"and the refusal names the field, so a caller can act on it" );

	// --- Automatic step-size search on
	autoalgo::Settings autostep = ok;
	autostep.autoStep = true;
	expect( !autoalgo::ineligible( lms, 0, autostep ),
		"CONTROL: canonical backpropagation still runs with the step search" );
	expect( autoalgo::ineligible( lms, Network::TRAIN_LBFGS, autostep )
		&& autoalgo::ineligible( lms, Network::TRAIN_IRPROP, autostep )
		&& string( autoalgo::ineligible( lms, Network::TRAIN_LM, autostep ) )
			== "requires autostep=0",
		"autostep=1 refuses all three retained modern optimizers by name" );

	// --- A VALIDATION / EARLY-STOPPING SPLIT IS NOT A RESTRICTION.
	//     lm_source_decision.md section 9 declares six refusals and this is not
	//     among them: a held-out split changes neither LM's objective nor
	//     Algorithm 3.16, and Iterative still owns stopping. The assertion is
	//     positive on purpose -- a restriction that was never declared can only
	//     be guarded by requiring the method to REMAIN eligible.
	DataSet threeWay = makeThreeWayData( 200, 4, 20260809u );
	expect( threeWay.valLoaded(), "CONTROL: the three-way fixture really did "
		"load a validation set" );
	SimpleProp split;
	split.setDataSet( threeWay );
	split.setHidden( 3 );
	split.randomize();
	autoalgo::Settings fromData = autoalgo::Settings::of( split );
	bool allEligibleUnderSplit = true;
	for ( unsigned t = 0; t < Network::TRAINING_TYPES; t++ )
		if ( autoalgo::ineligible( split, t, fromData ) )
			allEligibleUnderSplit = false;
	expect( allEligibleUnderSplit,
		"a loaded validation split refuses NOTHING -- Levenberg-Marquardt "
		"included; it is not one of the six declared restrictions" );

	// --- Cross-entropy: the strict LMS gate
	SimpleProp xent;
	xent.setDataSet( data );
	xent.setHidden( 3 );
	xent.randomize();
	xent.setXEerror();
	autoalgo::Settings xs = autoalgo::Settings::of( xent );
	expect( !autoalgo::ineligible( xent, Network::TRAIN_LBFGS, xs )
		&& !autoalgo::ineligible( xent, Network::TRAIN_IRPROP, xs ),
		"CONTROL: L-BFGS and iRPROP+ minimize whatever objective the model has" );
	expect( autoalgo::ineligible( xent, Network::TRAIN_LM, xs )
		&& string( autoalgo::ineligible( xent, Network::TRAIN_LM, xs ) ).find(
			"least-squares" ) != string::npos,
		"cross-entropy refuses Levenberg-Marquardt, naming the sum of squares" );

	// --- Model family. BackProp packs parameters but implements no normal
	//     equations; Logistic implements neither boundary.
	BackProp deep;
	deep.setBias( true );
	deep.setDataSet( data );
	vector< unsigned > layers;
	layers.push_back( 3 );
	deep.setHidden( layers );
	deep.randomize();
	autoalgo::Settings ds = autoalgo::Settings::of( deep );
	expect( !autoalgo::ineligible( deep, Network::TRAIN_LBFGS, ds )
		&& !autoalgo::ineligible( deep, Network::TRAIN_IRPROP, ds ),
		"CONTROL: a multi-layer BackProp runs L-BFGS and iRPROP+" );
	expect( autoalgo::ineligible( deep, Network::TRAIN_LM, ds )
		&& string( autoalgo::ineligible( deep, Network::TRAIN_LM, ds ) ).find(
			"single-hidden-layer" ) != string::npos,
		"BackProp refuses Levenberg-Marquardt: no normal-equations boundary" );

	BareProp bare;
	bare.setDataSet( data );
	bare.setHidden( 3 );
	bare.randomize();
	autoalgo::Settings bs = autoalgo::Settings::of( bare );
	expect( !autoalgo::ineligible( bare, Network::TRAIN_LM, bs ),
		"CONTROL: BareProp is the other half of the eligible LM family" );

	Logistic logit;
	logit.setDataSet( data );
	autoalgo::Settings gs = autoalgo::Settings::of( logit );
	expect( !autoalgo::ineligible( logit, 0, gs )
			&& !autoalgo::ineligible( logit, 1, gs )
			&& !autoalgo::ineligible( logit, 2, gs ),
		"CONTROL: logistic regression still trains on the legacy three" );
	expect( string( autoalgo::ineligible( logit, Network::TRAIN_LBFGS, gs ) )
			== "is available for neural models only"
		&& string( autoalgo::ineligible( logit, Network::TRAIN_IRPROP, gs ) )
			== "is available for neural models only"
		&& string( autoalgo::ineligible( logit, Network::TRAIN_LM, gs ) )
			== "is available for neural models only",
		"logistic regression refuses all three modern optimizers, one sentence" );

	// --- The parameter ceiling. P = hidden*(inputs+1) + (hidden+1).
	SimpleProp big;
	big.setDataSet( data );
	big.setHidden( 103 ); // 103*5 + 104 = 619 > 512
	big.randomize();
	expect( big.packedSize() > LM::MAX_PARAMETERS,
		"CONTROL: the wide fixture really is past the declared ceiling" );
	autoalgo::Settings bigs = autoalgo::Settings::of( big );
	expect( !autoalgo::ineligible( big, Network::TRAIN_LBFGS, bigs ),
		"CONTROL: L-BFGS has no parameter ceiling" );
	expect( autoalgo::ineligible( big, Network::TRAIN_LM, bigs )
		&& string( autoalgo::ineligible( big, Network::TRAIN_LM, bigs ) ).find(
			"ceiling" ) != string::npos,
		"more than MAX_PARAMETERS weights refuses Levenberg-Marquardt" );

	// The boundary is stated exactly rather than approached: a model AT the
	//    ceiling is eligible, one past it is not. inputs=4 gives P = 6h + 1,
	//    so h = 85 packs 511 weights and h = 86 packs 517.
	SimpleProp under, over;
	under.setDataSet( data ); under.setHidden( 85 ); under.randomize();
	over.setDataSet( data );  over.setHidden( 86 );  over.randomize();
	expect( under.packedSize() <= LM::MAX_PARAMETERS
		&& over.packedSize() > LM::MAX_PARAMETERS,
		"CONTROL: the pair straddles the ceiling ("
			+ to_string( under.packedSize() ) + " and "
			+ to_string( over.packedSize() ) + ")" );
	expect( !autoalgo::ineligible( under, Network::TRAIN_LM,
			autoalgo::Settings::of( under ) )
		&& autoalgo::ineligible( over, Network::TRAIN_LM,
			autoalgo::Settings::of( over ) ),
		"the ceiling is a boundary, not a region: at it eligible, past it not" );

	// A PROCEDURE THAT RESIZES THE MODEL is judged at the size it will reach,
	//    not the size it starts from. OBD probes at hStart and then grows, and
	//    the optimizer it adopts has to survive hMax -- there is no handler
	//    between LM::Ineligible and the worker thread.
	autoalgo::Settings growing = autoalgo::Settings::of( under );
	growing.parameters = over.packedSize();
	expect( autoalgo::ineligible( under, Network::TRAIN_LM, growing ),
		"a search that will grow past the ceiling refuses LM at the START" );
	expect( !autoalgo::ineligible( under, Network::TRAIN_LBFGS, growing ),
		"CONTROL: growth does not disturb an optimizer with no ceiling" );

	// --- A training type this engine does not have
	expect( autoalgo::ineligible( lms, Network::TRAINING_TYPES, ok ),
		"a training type past the enumeration is refused, not indexed" );
}

// ---------------------------------------------------------------------------
// 3. THE CURATED CANDIDATE LIST, asserted as a set.
// ---------------------------------------------------------------------------

static void checkCandidates()
{
	Hush hush;
	DataSet data = makeData( 80, 4, 20260809u );

	SimpleProp lms;
	lms.setDataSet( data );
	lms.setHidden( 3 );
	lms.randomize();
	autoalgo::Settings s = autoalgo::Settings::of( lms );

	expect( setOf( autoalgo::candidates( lms, s ) ) == "0,1,2,3,4,5",
		"an eligible small LMS network offers all six candidates, in order of "
		"simplicity (which IS the tie policy)" );

	// THE SIX DECLARED RESTRICTIONS, AND ONLY THOSE SIX. Each drops LM (and
	//    whatever else shares it); nothing outside the list drops anything.
	//    Stated as a table so a seventh restriction cannot be added without a
	//    row that admits to it.
	DataSet threeWay = makeThreeWayData( 200, 4, 20260809u );
	SimpleProp split;
	split.setDataSet( threeWay );
	split.setHidden( 3 );
	split.randomize();
	expect( setOf( autoalgo::candidates( split,
			autoalgo::Settings::of( split ) ) ) == "0,1,2,3,4,5",
		"a VALIDATION SPLIT drops nothing: it is not one of the six" );

	SimpleProp decayed;
	decayed.setDataSet( data );
	decayed.setHidden( 3 );
	decayed.randomize();
	decayed.setWeightDecay( true );
	decayed.setDecay( 1e-3 );
	expect( setOf( autoalgo::candidates( decayed,
			autoalgo::Settings::of( decayed ) ) ) == "0,1,2,3,4,5",
		"WEIGHT DECAY drops nothing either: LM carries the penalty curvature" );

	autoalgo::Settings grown = s;
	grown.parameters = LM::MAX_PARAMETERS + 1;
	expect( setOf( autoalgo::candidates( lms, grown ) ) == "0,1,2,3,4",
		"a search that will GROW past the ceiling drops exactly LM" );

	autoalgo::Settings online = s;
	online.batchEpoch = false;
	expect( setOf( autoalgo::candidates( lms, online ) ) == "0,1,2",
		"on-line mode leaves exactly the legacy three" );

	Logistic logit;
	logit.setDataSet( data );
	expect( setOf( autoalgo::candidates( logit,
			autoalgo::Settings::of( logit ) ) ) == "0,1,2",
		"logistic regression leaves exactly the legacy three" );
}

// ---------------------------------------------------------------------------
// 4. THE SELECTOR. That an eligible Levenberg-Marquardt is actually PROBED --
//    and that an ineligible one is OMITTED WITH A REASON rather than attempted
//    and reported as divergence.
// ---------------------------------------------------------------------------

static bool probed( const autoalgo::Result& r, unsigned token )
{
	for ( unsigned i = 0; i < r.probes.size(); i++ )
		if ( r.probes[ i ].algorithm == token ) return true;
	return false;
}

static string omissionReason( const autoalgo::Result& r, unsigned token )
{
	for ( unsigned i = 0; i < r.omitted.size(); i++ )
		if ( r.omitted[ i ].algorithm == token ) return r.omitted[ i ].reason;
	return "";
}

static const autoalgo::Probe* probeOf( const autoalgo::Result& r, unsigned token )
{
	for ( unsigned i = 0; i < r.probes.size(); i++ )
		if ( r.probes[ i ].algorithm == token ) return &r.probes[ i ];
	return 0;
}

static void checkSelection()
{
	Hush hush;
	DataSet data = makeData( 120, 4, 20260809u );

	SimpleProp start;
	start.setDataSet( data );
	start.setHidden( 3 );
	start.setLastop( false );
	start.setHistory( false );
	start.randomize();

	// The objective AT THE STARTING WEIGHTS. Training reports pre-update, so a
	//    single-iteration run returns F(x_0) -- the reference every probe below
	//    must have moved away from.
	double startError;
	{
		SimpleProp probeStart( start );
		probeStart.setMaxIterations( 1 );
		probeStart.setQuiet( true );
		startError = probeStart.train();
	}
	expect( startError > 0 && startError == startError,
		"CONTROL: the starting objective is finite and positive ("
			+ to_string( startError ) + ")" );

	autoalgo::Result r = autoalgo::pick( start, ALL, 0, 60, 0 );

	expect( r.probes.size() == 6 && r.omitted.empty(),
		"an eligible small LMS network probes all six and omits none" );
	expect( probed( r, Network::TRAIN_LM + 1 ),
		"LEVENBERG-MARQUARDT IS PROBED when it is eligible" );

	const autoalgo::Probe* lm = probeOf( r, Network::TRAIN_LM + 1 );
	expect( lm && lm->name == string( Network::algorithmName( Network::TRAIN_LM ) ),
		"and it is named by the one naming owner, not by a local string" );
	expect( lm && lm->usable,
		"and it produced a usable error -- it was NOT reported as diverged, "
		"which is what probing an ineligible method looks like" );
	expect( lm && lm->iterations >= 1 && lm->error < startError,
		"and it actually trained: at least one iteration, and an objective "
		"below the starting one" );

	bool everyProbeMoved = true, everyProbeNamed = true;
	for ( unsigned i = 0; i < r.probes.size(); i++ )
	{
		if ( !( r.probes[ i ].usable && r.probes[ i ].error < startError ) )
			everyProbeMoved = false;
		if ( r.probes[ i ].name
			!= string( Network::algorithmName( r.probes[ i ].algorithm - 1 ) ) )
			everyProbeNamed = false;
	}
	expect( everyProbeMoved, "every probe reduced the objective from the same "
		"starting weights" );
	expect( everyProbeNamed, "every probe carries the one owner's name" );
	expect( r.selected >= 1 && r.selected <= Network::TRAINING_TYPES
		&& r.winner != 0,
		"a winner was selected and its clone returned for adoption" );
	expect( r.totalBudgetMs == 60 && r.perCandidateBudgetMs == 10,
		"the total budget is shared: 60 ms across six candidates is 10 ms each" );

	// THE ADOPTION CONTRACT, and what makes the blocking and asynchronous
	//    results name the winner correctly whichever method wins: the returned
	//    clone is already CONFIGURED with the winning optimizer, so the caller
	//    reads its training type rather than being told a token to trust.
	expect( r.winner
		&& r.winner->getTrainingType() == r.selected - 1,
		string( "the adopted clone is configured with the winning optimizer (" )
			+ Network::algorithmName( r.selected - 1 ) + ")" );

	// AND THE CALLER'S MODEL IS UNTOUCHED. Probing trains clones; if it trained
	//    'start' itself, the objective here would have moved.
	double afterError;
	{
		SimpleProp probeStart( start );
		probeStart.setMaxIterations( 1 );
		probeStart.setQuiet( true );
		afterError = probeStart.train();
	}
	expect( afterError == startError,
		"the caller's own network is bit-identical after probing" );

	// --- A VALIDATION SPLIT DOES NOT REMOVE A CANDIDATE. The same shape as the
	//     omission cases below, asserted in the positive direction, because a
	//     restriction that was never declared can only be guarded by requiring
	//     the method to survive it.
	DataSet threeWay = makeThreeWayData( 200, 4, 20260809u );
	SimpleProp split;
	split.setDataSet( threeWay );
	split.setHidden( 3 );
	split.setLastop( false );
	split.setHistory( false );
	split.randomize();
	autoalgo::Result v = autoalgo::pick( split, ALL, 0, 60, 0 );
	expect( v.probes.size() == 6 && v.omitted.empty()
		&& probed( v, Network::TRAIN_LM + 1 ),
		"a dataset with a validation split still probes all six, LM included" );

	// --- A PROCEDURE THAT WILL GROW PAST THE CEILING omits LM, and says so.
	//     This is OBD's case: probe small, then grow.
	autoalgo::Result g2 = autoalgo::pick( start, ALL, LM::MAX_PARAMETERS + 1, 60, 0 );
	expect( g2.probes.size() == 5 && !probed( g2, Network::TRAIN_LM + 1 ),
		"a search planning to grow past the ceiling does not probe LM at all" );
	expect( omissionReason( g2, Network::TRAIN_LM + 1 ).find( "ceiling" )
		!= string::npos,
		"and the result carries WHY, so its absence cannot read as a loss" );
	expect( g2.selected >= 1 && g2.selected <= Network::TRAIN_IRPROP + 1
		&& g2.selected != Network::TRAIN_LM + 1,
		"and an omitted method can never win" );

	// --- Cross-entropy: the same shape, a different reason.
	SimpleProp xent( start );
	xent.setXEerror();
	autoalgo::Result x = autoalgo::pick( xent, ALL, 0, 60, 0 );
	expect( !probed( x, Network::TRAIN_LM + 1 )
		&& omissionReason( x, Network::TRAIN_LM + 1 ).find( "least-squares" )
			!= string::npos,
		"cross-entropy omits Levenberg-Marquardt, naming the sum of squares" );

	// --- Logistic: three probed, three omitted, none of them "diverged".
	Logistic logit;
	logit.setDataSet( data );
	logit.setLastop( false );
	logit.setHistory( false );
	autoalgo::Result g = autoalgo::pick( logit, ALL, 0, 60, 0 );
	expect( g.probes.size() == 3 && g.omitted.size() == 3,
		"logistic regression probes the legacy three and omits the modern three" );
	bool allNeuralReason = true;
	for ( unsigned i = 0; i < g.omitted.size(); i++ )
		if ( g.omitted[ i ].reason != "is available for neural models only" )
			allNeuralReason = false;
	expect( allNeuralReason, "each with the same one-sentence reason" );
}

// ---------------------------------------------------------------------------
// 5. THE BUDGET IS BOUNDED AND SHARED, not multiplied.
//
//    This is the property that makes "retain another algorithm" free at the
//    call sites that hurt: every OBD search and every nested cross-validation
//    fold runs an automatic selection, so a per-candidate budget would make the
//    whole suite linearly slower each time the portfolio grew. The assertions
//    below compare selections with DIFFERENT numbers of eligible candidates and
//    require the declared total to be identical.
// ---------------------------------------------------------------------------

static void checkBudget()
{
	Hush hush;
	DataSet data = makeData( 120, 4, 20260809u );

	SimpleProp six;
	six.setDataSet( data );
	six.setHidden( 3 );
	six.setLastop( false );
	six.setHistory( false );
	six.randomize();

	Logistic three;
	three.setDataSet( data );
	three.setLastop( false );
	three.setHistory( false );

	expect( autoalgo::DEFAULT_TOTAL_BUDGET_MS == 2250,
		"the default total is the shipped three-arm cost, 3 x 750 ms -- a "
		"compatibility contract, not a tuning constant" );

	autoalgo::Result a = autoalgo::pick( six, ALL, 0, 60, 0 );
	autoalgo::Result b = autoalgo::pick( three, ALL, 0, 60, 0 );

	expect( a.probes.size() == 6 && b.probes.size() == 3,
		"CONTROL: the two fixtures really do offer different candidate counts ("
			+ to_string( a.probes.size() ) + " and "
			+ to_string( b.probes.size() ) + ")" );
	// The DECLARED total. Measured against sabotage C (the division removed, so
	//    every candidate got the whole total): this assertion still PASSED,
	//    because totalBudgetMs is recorded from the argument and says nothing
	//    about what was spent. It is kept as documentation of intent, not as a
	//    guard, and the two assertions after it are the ones that fail.
	expect( a.totalBudgetMs == b.totalBudgetMs && a.totalBudgetMs == 60,
		"the declared total does not move when the candidate count doubles" );
	expect( a.perCandidateBudgetMs == 10 && b.perCandidateBudgetMs == 20,
		"the per-candidate figure absorbs the difference instead: 60/6 and 60/3" );
	// THE LOAD-BEARING ONE. What the selection actually spends is the share
	//    times the number of candidates, and THAT is what must not grow.
	expect( a.perCandidateBudgetMs * a.probes.size() <= a.totalBudgetMs
		&& b.perCandidateBudgetMs * b.probes.size() <= b.totalBudgetMs,
		"AND THE SHARES NEVER SUM PAST THE TOTAL, whatever the candidate count" );

	// EQUAL SHARES, and the remainder rule. 50 across 6 is 8 each with 2 ms
	//    left unspent -- NOT 9 for the first two. An unequal window makes the
	//    comparison that follows a comparison of budgets.
	autoalgo::Result r = autoalgo::pick( six, ALL, 0, 50, 0 );
	expect( r.perCandidateBudgetMs == 8,
		"the remainder is floored, not distributed: 50 across six is 8 each" );
	expect( r.perCandidateBudgetMs * 6 == 48 && r.totalBudgetMs == 50,
		"leaving 2 ms deliberately unspent rather than giving two candidates "
		"a longer window than the other four" );

	// An impossible equal-share budget is refused rather than silently spending
	//    more than the value reported as the total.
	bool tinyRefused = false;
	try { ( void ) autoalgo::pick( six, ALL, 0, 3, 0 ); }
	catch ( const invalid_argument& ) { tinyRefused = true; }
	expect( tinyRefused,
		"a total below one millisecond per candidate is refused, not overspent" );

	// An OMITTED candidate consumes no budget and does not divide the total:
	//    the logistic run above shared 60 between three, not between six.
	expect( b.omitted.size() == 3 && b.perCandidateBudgetMs == 20,
		"omitted candidates take no share of the total" );
}

// ---------------------------------------------------------------------------
// 6. A REQUESTED SUBSET. The training panel's checkboxes name a set of methods,
//    and the set decides three different things: WHO competes, WHETHER anything
//    competes at all, and how large each competitor's share of the fixed total
//    is.
//
//    THE ORDER OF THE TWO QUESTIONS IS THE CONTRACT. Eligibility is resolved
//    FIRST and the count is read from the survivors -- so three boxes ticked
//    with two of them impossible here is the same run as one box ticked, and
//    neither spends a probe budget. Asserting only "three requested, three
//    probed" would pass against an implementation that counted the request.
// ---------------------------------------------------------------------------

static void checkRequestedSubset()
{
	Hush hush;
	DataSet data = makeData( 120, 4, 20260809u );

	SimpleProp net;
	net.setDataSet( data );
	net.setHidden( 3 );
	net.setLastop( false );
	net.setHistory( false );
	net.randomize();

	autoalgo::Settings s = autoalgo::Settings::of( net );

	// --- plan(): who competes, who is left out, and in whose order -----------
	expect( setOf( autoalgo::plan( net, ALL, s ).competitors ) == "0,1,2,3,4,5"
		&& autoalgo::plan( net, ALL, s ).omitted.empty(),
		"CONTROL: an empty request is the whole curated list, as auto means" );

	autoalgo::Plan two = autoalgo::plan( net,
		setOfTypes( Network::TRAIN_IRPROP, 0 ), s );
	expect( setOf( two.competitors ) == "0,4",
		"a named subset yields exactly its members -- and in ASCENDING order, "
		"not the order they were written, because that order is the tie policy" );
	expect( two.omitted.empty(),
		"and a method that was never requested is not reported as omitted: the "
		"caller is told about the methods it asked about, not the other four" );

	// The property the previous assertion is really guarding, stated so it
	//    cannot be satisfied by an empty omission list in general.
	SimpleProp xent( net );
	xent.setXEerror();
	autoalgo::Settings xs = autoalgo::Settings::of( xent );
	autoalgo::Plan mixed = autoalgo::plan( xent,
		setOfTypes( 0, Network::TRAIN_LM ), xs );
	expect( setOf( mixed.competitors ) == "0" && mixed.omitted.size() == 1
		&& mixed.omitted[ 0 ].algorithm == Network::TRAIN_LM + 1,
		"a REQUESTED method that cannot run here is omitted by name" );
	expect( mixed.omitted.size() == 1
		&& mixed.omitted[ 0 ].reason.find( "least-squares" ) != string::npos,
		"with the one rule's own sentence, so the refusal and the omission read "
		"identically" );
	expect( autoalgo::plan( xent, ALL, xs ).omitted.size() == 1,
		"CONTROL: on this same model the full request omits exactly that one "
		"method -- the subset changed WHO was asked about, not the rule" );

	// --- Nothing eligible: the caller must be able to refuse -----------------
	Logistic logit;
	logit.setDataSet( data );
	logit.setLastop( false );
	logit.setHistory( false );
	autoalgo::Settings ls = autoalgo::Settings::of( logit );
	autoalgo::Plan none = autoalgo::plan( logit,
		setOfTypes( Network::TRAIN_LBFGS, Network::TRAIN_LM ), ls );
	expect( none.competitors.empty() && none.omitted.size() == 2,
		"a request naming only methods this model cannot run resolves to NO "
		"competitors, with a reason for each" );
	expect( !autoalgo::plan( logit, setOfTypes( 0, 1 ), ls ).competitors.empty(),
		"CONTROL: the same model with two runnable methods named still competes" );

	// A token past the enumeration is refused rather than dropped: a caller that
	//    named three methods and got a two-way competition would never learn why.
	autoalgo::Plan bogus = autoalgo::plan( net,
		setOfTypes( 0, Network::TRAINING_TYPES ), s );
	expect( setOf( bogus.competitors ) == "0" && bogus.omitted.size() == 1
		&& bogus.omitted[ 0 ].algorithm == Network::TRAINING_TYPES + 1,
		"a token this engine has no algorithm for is reported, not ignored" );

	// --- pick(): one survivor is not a competition --------------------------
	//     Asserted through the ELIGIBILITY resolution, not the request size:
	//     three methods are named and two of them cannot run here.
	autoalgo::Result lone = autoalgo::pick( xent,
		setOfTypes( Network::TRAIN_LM, 0 ), 0, 60, 0 );
	expect( lone.selected == 1 && !lone.competed,
		"TWO METHODS REQUESTED, ONE ELIGIBLE: the survivor is selected and NO "
		"competition is run" );
	expect( lone.probes.empty() && lone.totalBudgetMs == 60
		&& lone.perCandidateBudgetMs == 0,
		"and nothing is spent probing it -- there is nothing to compare it to" );
	expect( lone.winner == 0,
		"and no clone is adopted, so the run continues on the caller's own "
		"weights rather than a probe's" );
	expect( lone.omitted.size() == 1
		&& lone.omitted[ 0 ].algorithm == Network::TRAIN_LM + 1,
		"while the method that could not run is still reported by name" );

	// The same shape with one method named, which is how a single ticked box
	//    reaches the selector at all.
	autoalgo::Result single = autoalgo::pick( net,
		vector< unsigned >( 1, Network::TRAIN_IRPROP ), 0, 60, 0 );
	expect( single.selected == Network::TRAIN_IRPROP + 1 && !single.competed
		&& single.probes.empty() && single.perCandidateBudgetMs == 0,
		"one method requested and eligible: named, not probed" );

	// CONTROL, and the assertion that keeps the two above from passing against
	//    an implementation that never probes anything.
	autoalgo::Result pair = autoalgo::pick( net,
		setOfTypes( 0, Network::TRAIN_IRPROP ), 0, 60, 0 );
	expect( pair.competed && pair.probes.size() == 2 && pair.winner != 0,
		"CONTROL: two ELIGIBLE methods requested really do compete, and a "
		"winning clone comes back" );

	// --- The narrowed set gets a LARGER share of the SAME total -------------
	//     Narrowing the field must not shorten the selection: the total is a
	//     fixed compatibility contract, so fewer competitors means each is
	//     probed for longer, and the sum never grows.
	autoalgo::Result all6 = autoalgo::pick( net, ALL, 0, 60, 0 );
	expect( all6.probes.size() == 6 && all6.perCandidateBudgetMs == 10,
		"CONTROL: all six share 60 ms as 10 ms each" );
	expect( pair.totalBudgetMs == all6.totalBudgetMs
		&& pair.perCandidateBudgetMs == 30,
		"the same total across two competitors is 30 ms each -- a subset buys a "
		"longer probe, it does not shorten the selection" );
	expect( pair.perCandidateBudgetMs * pair.probes.size() <= pair.totalBudgetMs
		&& all6.perCandidateBudgetMs * all6.probes.size() <= all6.totalBudgetMs,
		"AND THE SHARES NEVER SUM PAST THE TOTAL, at either width" );

	// --- A subset cannot elect a method it does not contain ------------------
	bool onlyRequested = true;
	for ( unsigned i = 0; i < pair.probes.size(); i++ )
		if ( pair.probes[ i ].algorithm != 1
			&& pair.probes[ i ].algorithm != Network::TRAIN_IRPROP + 1 )
			onlyRequested = false;
	expect( onlyRequested && ( pair.selected == 1
		|| pair.selected == Network::TRAIN_IRPROP + 1 ),
		"only requested methods are probed, and only a requested method can win" );
	expect( pair.winner && pair.winner->getTrainingType() == pair.selected - 1,
		"and the adopted clone carries the winning method, not a token to trust" );
}

int main()
{
	cout << "Automatic selection: eligibility, candidates, probes" << endl;

	checkNaming();
	checkEligibility();
	checkCandidates();
	checkSelection();
	checkBudget();
	checkRequestedSubset();

	cout << ( failures ? "FAILURES: " : "all passed (" ) << failures
		<< ( failures ? "" : " failures)" ) << endl;
	return failures ? 1 : 0;
}

// ===========================================================================
// SABOTAGE LOG. Applied to production source, watched to fail after a visible
// recompilation of the affected translation units, then restored and watched to
// pass again after a second visible recompilation. Both objects -- the library's
// and this test's -- were removed each way, because make's timestamp
// granularity has silently skipped a rebuild on this project before.
//
// I. THE ELIGIBILITY QUERY ASKED ABOUT THE LIVE STEP CONTROLS, in
//    gui_page.html's refreshEligibility(): the query string rebuilt from
//    $("batch_epoch").checked and $("autostep").checked instead of the fixed
//    "?batch_epoch=1&autostep=0". This is the defect the panel actually shipped
//    for review with, found by clicking the page in Chrome: with Automatic
//    learning rate on, L-BFGS, iRPROP+ and LM grey out, and the control the
//    user must change first is the one their tick was going to change for them.
//    Caught by tests/gui/browser.sh on the assertion that names it -- "with
//    Automatic learning rate ON, the methods that own their own step are STILL
//    available" -- and, statically, by tests/gui/smoke.sh on "the Train panel
//    asks about the live controls, not the forced ones". src/gui_page.html is
//    compiled into the binary, so the neuron target visibly recompiled each way.
//    CONTROL: the discriminating assertion beside it still passed -- LM stayed
//    greyed on that cross-entropy model for the one reason a tick cannot fix --
//    so the failure is the panel greying out the WRONG methods, not the panel
//    having stopped greying anything out.
//
// H. THE SAME SABOTAGE, AND IT WAS NOT CAUGHT. Recorded because the null result
//    is the measurement that produced assertion 2b. The first attempt reverted
//    the query AND left the autostep/batch_epoch listeners removed, so nothing
//    re-ran the query when those controls moved -- and TWO independent things
//    prevent this defect, either of which is sufficient. browser.sh passed.
//    The lock assertions could not see it because they only watch the controls
//    the tick sets, not what the panel does with the answer. The guard was then
//    stated as the invariant itself (re-run the query BY HAND with Automatic
//    learning rate on, and require the step-owning methods to survive), after
//    which sabotage I -- one mechanism, the other left intact -- fails it.
//    The lesson is the standing one: sabotage ONE mechanism at a time, and a
//    sabotage that is not caught is a measurement about the TEST.
//
// G. THE REQUESTED SET DISCARDED AT THE REST BOUNDARY, in gui.cpp's
//    readAlgorithmSet(): `set = false` instead of `set = ( types.size() > 1 )`,
//    so a request naming several methods trained with the FIRST one and never
//    competed at all -- the silent-wrong-optimizer failure, since the run still
//    succeeds and reports a perfectly true label for the method it did use.
//    This file cannot see it (gui.cpp is not linked here) and neither can the
//    engine tests above, which is exactly why it is recorded: tests/gui/smoke.sh
//    owns that boundary and failed on "a requested subset must compete over
//    exactly its members", the autoAlgo block being absent entirely.
//    CONTROL: everything before it in the same script passed, including the
//    algorithm=auto selection and both single-token trains -- auto sets the flag
//    on the line above the sabotage, so it was unaffected, which is what makes
//    this a sabotage of the SET wiring and not of selection.
//    src/gui.cpp and the neuron binary visibly recompiled each way.
//
// F. THE NO-COMPETITION RULE REMOVED, in autoalgo::pick(): `if ( false )` in
//    place of `if ( list.size() == 1 )`, so a lone eligible method was probed
//    with the entire budget and the run continued from the PROBE's weights
//    rather than the caller's. Four assertions failed, and they are the four
//    that name the mechanism -- the survivor being selected without a
//    competition, nothing being spent, no clone being adopted, and the
//    single-method request. CONTROL: 80 assertions passed, including every
//    competing case, so the sabotage did not simply break selection.
//
// E. THE REQUESTED SET IGNORED INSIDE THE SELECTOR, in autoalgo::pick():
//    `plan( *probe, vector< unsigned >(), settings )`, which resolves the whole
//    curated list whatever the caller asked for. Seven assertions failed and the
//    one that names the mechanism was among them ("only requested methods are
//    probed, and only a requested method can win"), together with the budget
//    share -- six competitors cannot each get a two-way share. CONTROL: 77
//    assertions passed, including the entire eligibility matrix, the naming
//    tables, and every algorithm=auto (empty-request) assertion, which is what
//    makes this a sabotage of the request wiring rather than of the rule.
//
// D. THE IMPOSSIBLE-BUDGET REFUSAL DISABLED, in autoalgo::pick(): the guard
//    was made false so six candidates shared a declared 3 ms total. Both
//    autoalgo.cpp and this test recompiled visibly. Exactly the assertion
//    "a total below one millisecond per candidate is refused, not overspent"
//    failed; all 63 controls passed. Restored, both translation units visibly
//    recompiled again, and the complete test passed.
//
// C. THE BUDGET DIVISION REMOVED, in autoalgo::pick():
//    `result.perCandidateBudgetMs = totalBudgetMs`, so every candidate got the
//    whole total and a six-arm selection cost six times a three-arm one -- the
//    exact regression the shared budget exists to prevent. Seven assertions
//    failed. READ WHICH: "the declared total does not move when the candidate
//    count doubles" still PASSED, because the declared total is recorded from
//    the argument and says nothing about the spend. The guard is
//    "AND THE SHARES NEVER SUM PAST THE TOTAL", which failed, together with the
//    floor, the remainder rule and the omitted-candidates rule.
//    CONTROL: all 57 remaining assertions passed -- the eligibility matrix, the
//    naming tables, the candidate lists and every probe assertion.
//
// B. THE ELIGIBLE CANDIDATE DROPPED FROM THE CURATED LIST. In
//    autoalgo::candidates(), `t != Network::TRAIN_LM &&` added to the
//    eligibility test, so LM was never a candidate even where the rule said it
//    was eligible. This crosses production wiring rather than a component: the
//    RULE was untouched and every eligibility assertion still passed. Six
//    assertions failed, and the one that NAMES the mechanism was among them:
//
//        FAIL - an eligible small LMS network offers all six candidates ...
//        FAIL - an eligible small LMS network probes all six and omits none
//        FAIL - LEVENBERG-MARQUARDT IS PROBED when it is eligible
//        FAIL - and it is named by the one naming owner, not by a local string
//        FAIL - and it produced a usable error -- it was NOT reported as
//               diverged ...
//        FAIL - and it actually trained: at least one iteration, and an
//               objective below the starting one
//
//    CONTROL: all 49 remaining assertions -- the whole eligibility matrix, the
//    naming tables, and the non-LM candidate lists -- still passed, which is
//    what makes this a sabotage of the wiring and not of the rule. (Re-run
//    after the validation restriction was removed and the budget shared; the
//    failing set grew from six to fifteen because LM is now eligible in more
//    fixtures, and the budget arithmetic depends on the candidate count.)
//
// A. THE REST TOKEN WIRED TO THE WRONG OPTIMIZER, in gui.cpp:
//    `net->setTrainingType( algorithm == 6 ? 0 : algorithm - 1 )`. This file
//    cannot see it -- gui.cpp is not linked here -- and that is the point of
//    recording it: tests/gui/smoke.sh owns that boundary and failed on
//    "REST Levenberg-Marquardt selection/result contract", with the dumped
//    result showing `"algorithmName": "Canonical"` and the engine's own run
//    header reading "Training algorithm is canonical backpropagation".
//    CONTROL: the L-BFGS and iRPROP+ REST contracts immediately above it in the
//    same script passed, which is how the script reached the LM case at all.
