// lm_step0.cpp : STEP L0 of the Levenberg-Marquardt phase -- the measured
// per-traversal cost ratio, and nothing else.
//
// RESEARCH EVIDENCE. Committed so the measurement can be reproduced, but NOT
// registered with add_test: wall time is machine-dependent and a timing assertion
// in the suite is a flake generator (the scale_probe precedent). It adds no REST
// field, GUI control, optimizer token, automatic-selection entry or menu change,
// and touches no file under src/.
//
// THE QUESTION, from docs/learning_research/lm_source_decision.md section 12:
//
//   how long does ONE COMPLETE proposed LM normal-equations traversal take,
//   against ONE COMPLETE production batchObjectiveGradient() traversal, from
//   IDENTICAL installed weights, at P = 65 and P = 25?
//
// The cost model in section 0 estimates the ratio at P/4 -- roughly 16x and 6x.
// That is an ESTIMATE and decides nothing: constant factors, cache behaviour,
// symmetry handling and vectorization are all unmeasured. This program measures
// it. The measured ratio R then replaces P/4 in the break-even table:
//
//   LM must reach the endpoint in fewer than (panel traversals / R) traversals
//   to win on wall-clock. If that break-even is below TWO on every eligible
//   workload, LM is rejected here, at a cost of hours rather than a campaign.
//
// MEASURED 2026-08-08: 1.8x at P=65 and 0.87x at P=25 against the hypothesis's
// 16.2x and 6.2x -- so the budgets are 5.5 traversals against iRPROP+ on Civic
// Choice and 30-37 against L-BFGS on the conditioning pair. The gate passed and
// the implementation phase was authorized. See lm_step0_results.md.
//
// WHY THE NEW TRAVERSAL LIVES IN A SUBCLASS RATHER THAN IN OneHiddenNet:
//    it is the same code that would go into the model layer -- it reaches the
//    same protected state through the same forward() -- but placed where
//    deleting it is `rm`, not a revert. Step L0 exists to decide whether that
//    code should exist at all, so it may not require editing src/ to find out.
//    Once OneHiddenNet::batchNormalEquations() exists, this stays as the
//    INDEPENDENT restatement the timing was taken against.
//
//   ./build/lm_step0

#include "stdafx.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "harness.h"

using namespace std;
using namespace optbench;

// ---------------------------------------------------------------------------
// THE TWO NUMERICAL PRIMITIVES THE PROPOSED TRAVERSAL NEEDS.
//
// In the disposable tree, per the user's ruling: if LM survives Step L0 these
// are promoted into Matrix as a separate, fully tested, Manifest-synchronized
// commit BEFORE Algorithm 3.16 is implemented; if it does not, they go with the
// prototype. They are written the way the Matrix primitive would be written --
// walking the row-major block directly -- because a per-element bounds-checked
// operator()(i,j) in a P(P+1)/2 inner loop would make this measure the bounds
// check rather than the algorithm.

// A += v v', UPPER TRIANGLE ONLY. That triangle is where the P(P+1)/2 versus P^2
//    saving actually lives, and it is why this is not `A += scratch.outprod(v,v)`
//    (which allocates or traverses twice, and fills both triangles per exemplar).
//    Deliberately NOT skipping zero elements of v: a data-dependent shortcut
//    would make the measured cost a property of this fixture's sparsity.
static inline void accumulateUpperOuter( Matrix< double >& A,
	const vector< double >& v )
{
	const unsigned P = A.rows();
	double* base = A.begin();
	const double* pv = &v[ 0 ];

	for ( unsigned i = 0; i < P; i++ )
	{
		const double vi = pv[ i ];
		double* dst = base + ( size_t ) i * P + i;
		const double* src = pv + i;
		for ( unsigned j = i; j < P; j++ )
			*dst++ += vi * *src++;
	}
}

// THE SYMMETRY COMPLETION, once per traversal and outside the exemplar loop.
//    A general dense Matrix may not be handed to a caller with one triangle
//    stale while being called symmetric -- a caller reading A(i,j) for i > j
//    would silently read zero and nothing in the type would say so. O(P^2) here
//    against O(N P^2) in the loop, and it is INSIDE the timed region.
static inline void mirrorUpper( Matrix< double >& A )
{
	const unsigned P = A.rows();
	double* base = A.begin();
	for ( unsigned i = 0; i < P; i++ )
		for ( unsigned j = i + 1; j < P; j++ )
			base[ ( size_t ) j * P + i ] = base[ ( size_t ) i * P + j ];
}

// ---------------------------------------------------------------------------
// THE MODEL, with both traversals reachable.
//
// SimpleProp rather than a template: both Step L0 workloads are SimpleProp, and
// the eligible model set for this phase is OneHiddenNet's two concrete classes.

class LMStep0 : public SimpleProp {
public:
	// prepareRun() derives regularizer and decayTerm from the CURRENT eta and
	//    decay, and train() is what normally calls it. Nothing here trains, so
	//    it is called explicitly -- omitting it would evaluate the penalty
	//    against an uninitialised regularizer, which is the exact defect
	//    Iterative::prepareRun was extracted to fix.
	void prepare() { prepareRun(); }

	unsigned packed() const { return packedSize(); }

	void currentWeights( vector< double >& w ) const { packWeights( w ); }

	// THE PRODUCTION REFERENCE, unmodified: the same virtual the panel's L-BFGS
	//    and iRPROP+ arms call. Not a reimplementation of it.
	double productionPass( vector< double >& g )
	{ return batchObjectiveGradient( g ); }

	// THE COMPLETE PROPOSED LM TRAVERSAL, at the currently installed weights:
	//    forward propagation, the output and hidden sensitivities, the packed
	//    Jacobian row, J'J, J'r, the objective, the 1/N division, the decay
	//    terms, and the symmetry completion. Returns the objective.
	//
	//    F(w) = (1/N) sum_k 0.5 ( o_k - y_k )^2 + (decay/2) ||w||^2, expressed
	//    as 0.5 f'f through the augmented residual of the source decision's
	//    section 3, so that
	//
	//        A = (1/N) sum_k a_k a_k'  +  decay I
	//        g = (1/N) sum_k r_k a_k   +  decay w    ( = grad F, exactly )
	//
	//    with a_k = d o_k / d w -- the sensitivity of the OUTPUT, not of the
	//    error, which is why it cannot be had from o_err and h_err without
	//    dividing by r_k.
	double normalPass( Matrix< double >& A, vector< double >& g );

private:
	vector< double > jrow,  // a_k, the packed Jacobian row
		hsens,              // the hidden-unit sensitivities
		wpack;              // the packed weights, for the decay gradient term
};

double LMStep0::normalPass( Matrix< double >& A, vector< double >& g )
{
	const unsigned nTrain = theData.getNumTrain();
	const unsigned hRows = hW.rows(), hCols = hW.cols();
	const unsigned oSize = ( unsigned ) oW.size();
	const unsigned P = hRows * hCols + oSize;

	// Sized once per run in the real implementation; sized here on first use and
	//    reused for every repetition, so no allocation happens per traversal.
	if ( A.rows() != P || A.cols() != P )
		A.resize( P, P );
	A.fill( 0.0 );
	if ( g.size() != P ) g.resize( P );
	fill( g.begin(), g.end(), 0.0 );
	if ( jrow.size() != P ) jrow.resize( P );
	if ( hsens.size() != nHidden ) hsens.resize( nHidden );

	double setError = 0.0;

	for ( unsigned k = 0; k < nTrain; k++ )
	{
		// The same forward the rest of the engine runs: it reads the exemplar
		//    into I, pins the bias slot, and leaves hO, x and o set.
		forward( Train, k );

		const double r = o - y[ k ];
		setError += 0.5 * r * r;

		// The output sensitivity, d_sigmoidal()( o ) -- the sigmoid derivative
		//    written on its OUTPUT, exactly as function_defs.h defines it.
		const double s = o * ( 1.0 - o );

		// The hidden sensitivities. Elements 0 .. nHidden-1 are the hidden
		//    units; a biased model's hO carries the pinned bias slot after them
		//    and it has no sensitivity of its own.
		for ( unsigned j = 0; j < nHidden; j++ )
			hsens[ j ] = s * oW[ j ] * hO[ j ] * ( 1.0 - hO[ j ] );

		// The packed Jacobian row, in the ONE layout packPair() defines: hW row
		//    by row, then oW. d o / d hW(j,i) = hs_j I_i, d o / d oW_j = s hO_j.
		double* a = &jrow[ 0 ];
		for ( unsigned j = 0; j < hRows; j++ )
		{
			const double hj = hsens[ j ];
			for ( unsigned i = 0; i < hCols; i++ )
				*a++ = hj * I[ i ];
		}
		for ( unsigned j = 0; j < oSize; j++ )
			*a++ = s * hO[ j ];

		accumulateUpperOuter( A, jrow );
		axpy( r, jrow, g );
	}

	// The 1/N, applied to the populated triangle only.
	const double invN = 1.0 / ( double ) nTrain;
	{
		double* base = A.begin();
		for ( unsigned i = 0; i < P; i++ )
			for ( unsigned j = i; j < P; j++ )
				base[ ( size_t ) i * P + j ] *= invN;
	}
	g *= invN;
	setError *= invN;

	// WEIGHT DECAY. The penalty rows of the augmented residual are LINEAR in w,
	//    so their contribution to A is the EXACT Hessian of the ridge penalty --
	//    the Gauss-Newton approximation drops nothing from this term.
	if ( weightDecayFlag )
	{
		double* base = A.begin();
		for ( unsigned i = 0; i < P; i++ )
			base[ ( size_t ) i * P + i ] += decay;

		packWeights( wpack );
		axpy( decay, wpack, g );

		setError += regularizer * ( hW.squared() + squared( oW ) );
	}

	mirrorUpper( A );

	return setError;
}

// ---------------------------------------------------------------------------
// Order statistics. Median, MAD, p10, p90 -- the plan's reporting list.

struct Stats { double median, mad, p10, p90; };

static double quantile( vector< double > v, double q )
{
	sort( v.begin(), v.end() );
	if ( v.empty() ) return 0.0;
	const double pos = q * ( double ) ( v.size() - 1 );
	const size_t lo = ( size_t ) floor( pos );
	const size_t hi = ( size_t ) ceil( pos );
	if ( lo == hi ) return v[ lo ];
	return v[ lo ] + ( pos - ( double ) lo ) * ( v[ hi ] - v[ lo ] );
}

static Stats summarize( const vector< double >& v )
{
	Stats s;
	s.median = quantile( v, 0.5 );
	s.p10 = quantile( v, 0.10 );
	s.p90 = quantile( v, 0.90 );
	vector< double > dev;
	dev.reserve( v.size() );
	for ( size_t i = 0; i < v.size(); i++ )
		dev.push_back( fabs( v[ i ] - s.median ) );
	s.mad = quantile( dev, 0.5 );
	return s;
}

// ---------------------------------------------------------------------------
// One workload.

struct Workload {
	string name;
	Case c;
	// Previously measured panel traversal counts on this exact workload, from
	//    docs/learning_research/irprop_screen_results.md. Carried so the gate is
	//    applied to MEASURED counts rather than to remembered ones.
	string fastestArm;
	unsigned fastestTraversals;
	string secondArm;
	unsigned secondTraversals;
};

static const double AGREEMENT_TOLERANCE = 1e-12;

struct Outcome {
	bool valid;
	string refusal;
	unsigned params, rows;
	Stats lm, prod;
	double ratio, ratioLow, ratioHigh;
	double objectiveLm, objectiveProd, objectiveRel;
	double gradientRel;
};

// decayOverride: -1 keeps the workload's own setting (the DECLARED protocol);
//    0 forces weight decay off. The forced-off run is a DIAGNOSTIC, never the
//    gate: it exists because batchGradient() builds a temporary Matrix and a
//    temporary vector PER EXEMPLAR for its decay term (`hG += ( hW * decay )`,
//    `oG += ( oW * decay )`, both returning by value), while the proposed
//    traversal adds decay once per traversal outside the loop. Turning decay off
//    removes that difference, so comparing the two runs ATTRIBUTES the ratio
//    between the algorithm's extra work and the existing path's allocation.
static Outcome measure( const Workload& w, unsigned reps, unsigned orchSeed,
	int decayOverride = -1 )
{
	Outcome out;
	out.valid = false;
	out.params = out.rows = 0;
	out.ratio = out.ratioLow = out.ratioHigh = 0.0;
	out.objectiveLm = out.objectiveProd = out.objectiveRel = 0.0;
	out.gradientRel = 0.0;

	LMStep0 net;
	unsigned long long dataId = 0, splitId = 0;
	unsigned rowsTotal = 0, rowsTest = 0;

	Case c = w.c;
	if ( decayOverride == 0 ) { c.decayOn = false; c.decay = 0.0; }

	DataSet d = makeDataSet( c, dataId, splitId, rowsTotal, rowsTest );
	net.setDataSet( d );
	arch( net, c );
	configure( net, c );

	util::set_seed( c.weightSeed );
	net.randomize();
	net.prepare();

	out.rows = d.getNumTrain();
	out.params = net.packed();

	// --- the validity gates, BEFORE any timing is reported ------------------
	//     A ratio between two paths that do not compute the same thing is not
	//     evidence, and equality between two absent or default results is
	//     vacuous (rule 2).

	vector< double > weightsBefore, weightsAfter, gProd, gLm;
	Matrix< double > A;
	net.currentWeights( weightsBefore );

	const double fProd = net.productionPass( gProd );
	const double fLm = net.normalPass( A, gLm );

	net.currentWeights( weightsAfter );

	if ( weightsBefore.empty() || out.params == 0 )
	{
		out.refusal = "no packed parameter state";
		return out;
	}
	if ( weightsAfter != weightsBefore )
	{
		out.refusal = "an evaluation moved the weights -- it trained";
		return out;
	}
	if ( gProd.size() != out.params || gLm.size() != out.params )
	{
		out.refusal = "a gradient is not the packed length";
		return out;
	}
	if ( !std::isfinite( fProd ) || !std::isfinite( fLm ) )
	{
		out.refusal = "a traversal returned a non-finite objective";
		return out;
	}

	bool allZero = true;
	for ( unsigned i = 0; i < out.params; i++ )
	{
		if ( !std::isfinite( gProd[ i ] ) || !std::isfinite( gLm[ i ] ) )
		{
			out.refusal = "a gradient element is not finite";
			return out;
		}
		if ( gLm[ i ] != 0.0 ) allZero = false;
	}
	if ( allZero )
	{
		out.refusal = "the proposed traversal produced an all-zero gradient";
		return out;
	}

	// A must be symmetric AFTER the mirror, and its diagonal must carry decay.
	for ( unsigned i = 0; i < out.params; i++ )
	{
		if ( c.decayOn && !( A( i, i ) >= c.decay ) )
		{
			out.refusal = "A's diagonal does not carry the decay term";
			return out;
		}
		for ( unsigned j = i + 1; j < out.params; j++ )
			if ( A( i, j ) != A( j, i ) )
			{
				out.refusal = "A is not symmetric: the mirror did not run";
				return out;
			}
	}

	// The objectives and gradients must AGREE. This is the section 3 identity
	//    -- J'f is grad F -- used as the check that the new traversal computes
	//    the same thing the production one does.
	out.objectiveLm = fLm;
	out.objectiveProd = fProd;
	out.objectiveRel = fabs( fLm - fProd ) / max( 1.0, fabs( fProd ) );
	if ( out.objectiveRel > AGREEMENT_TOLERANCE )
	{
		out.refusal = "the two traversals disagree on the objective";
		return out;
	}

	double worst = 0.0;
	for ( unsigned i = 0; i < out.params; i++ )
	{
		const double rel = fabs( gLm[ i ] - gProd[ i ] )
			/ max( 1.0, fabs( gProd[ i ] ) );
		if ( rel > worst ) worst = rel;
	}
	out.gradientRel = worst;
	if ( worst > AGREEMENT_TOLERANCE )
	{
		out.refusal = "the two traversals disagree on the gradient";
		return out;
	}

	// --- timing --------------------------------------------------------------
	//     Randomized and interleaved: arm order is drawn per repetition from the
	//     recorded orchestration seed, so neither arm always runs into a warm
	//     cache the other left.

	vector< double > lmMs, prodMs;
	std::mt19937 orch( orchSeed );

	// Warm-up, discarded.
	net.productionPass( gProd );
	net.normalPass( A, gLm );

	for ( unsigned rep = 0; rep < reps; rep++ )
	{
		bool lmFirst = ( orch() & 1u ) != 0;
		for ( int turn = 0; turn < 2; turn++ )
		{
			const bool runLm = ( turn == 0 ) ? lmFirst : !lmFirst;
			const auto t0 = std::chrono::steady_clock::now();
			if ( runLm ) net.normalPass( A, gLm );
			else net.productionPass( gProd );
			const auto t1 = std::chrono::steady_clock::now();
			const double ms = std::chrono::duration< double, std::milli >(
				t1 - t0 ).count();
			if ( runLm ) lmMs.push_back( ms ); else prodMs.push_back( ms );
		}
	}

	out.lm = summarize( lmMs );
	out.prod = summarize( prodMs );
	out.ratio = out.lm.median / out.prod.median;
	out.ratioLow = out.lm.p10 / out.prod.p90;
	out.ratioHigh = out.lm.p90 / out.prod.p10;
	out.valid = true;
	return out;
}

// ---------------------------------------------------------------------------

int main( int argc, char** argv )
{
	unsigned reps = 15, orchSeed = 20260808;
	for ( int i = 1; i < argc; i++ )
	{
		const string a = argv[ i ];
		if ( a == "--reps" && i + 1 < argc ) reps = ( unsigned ) atoi( argv[ ++i ] );
		else if ( a == "--seed" && i + 1 < argc ) orchSeed = ( unsigned ) atoi( argv[ ++i ] );
		else
		{
			fprintf( stderr, "usage: lm_step0 [--reps N] [--seed S]\n" );
			return 2;
		}
	}

	vector< Workload > loads;
	{
		Workload w;
		w.name = "civic-6000-h4";
		w.c = civicCase( "simpleprop", 6000, CIVIC_HIDDEN, ENDPOINT_PRACTICAL,
			0, false );
		w.fastestArm = "iRPROP+"; w.fastestTraversals = 10;
		w.secondArm = "L-BFGS";  w.secondTraversals = 20;
		loads.push_back( w );
	}
	{
		Workload w;
		w.name = "well4";
		w.c = conditioningCase( "well4", 0 );
		w.fastestArm = "L-BFGS"; w.fastestTraversals = 26;
		w.secondArm = "iRPROP+"; w.secondTraversals = 53;
		loads.push_back( w );
	}
	{
		Workload w;
		w.name = "poor4";
		w.c = conditioningCase( "poor4", 0 );
		w.fastestArm = "L-BFGS"; w.fastestTraversals = 32;
		w.secondArm = "iRPROP+"; w.secondTraversals = 59;
		loads.push_back( w );
	}

	printf( "STEP L0 -- the measured per-traversal cost ratio\n" );
	printf( "build %s, engine %s\n", OPTBENCH_BUILD_TYPE, OPTBENCH_ENGINE_ID );
	printf( "%u randomized interleaved repetitions, orchestration seed %u\n",
		reps, orchSeed );
	printf( "agreement tolerance %.0e (objective and every gradient element)\n\n",
		AGREEMENT_TOLERANCE );

	bool anyWinRegion = false, anyInvalid = false;

	for ( size_t i = 0; i < loads.size(); i++ )
	{
		Outcome o;
		try
		{
			o = measure( loads[ i ], reps, orchSeed + ( unsigned ) i );
		}
		catch ( const exception& e )
		{
			printf( "%-16s REFUSED: %s\n\n", loads[ i ].name.c_str(), e.what() );
			anyInvalid = true;
			continue;
		}

		if ( !o.valid )
		{
			printf( "%-16s REFUSED: %s\n\n", loads[ i ].name.c_str(),
				o.refusal.c_str() );
			anyInvalid = true;
			continue;
		}

		printf( "%s  (P = %u, %u training rows)\n", loads[ i ].name.c_str(),
			o.params, o.rows );
		printf( "  objectives agree to %.2e, gradients to %.2e  [gate %.0e]\n",
			o.objectiveRel, o.gradientRel, AGREEMENT_TOLERANCE );
		printf( "  %-24s median %9.4f ms   MAD %7.4f   p10 %9.4f   p90 %9.4f\n",
			"batchObjectiveGradient", o.prod.median, o.prod.mad, o.prod.p10,
			o.prod.p90 );
		printf( "  %-24s median %9.4f ms   MAD %7.4f   p10 %9.4f   p90 %9.4f\n",
			"normal equations (LM)", o.lm.median, o.lm.mad, o.lm.p10, o.lm.p90 );
		printf( "  RATIO  %.2fx   (p10/p90 band %.2fx .. %.2fx)   cost model said %.1fx\n",
			o.ratio, o.ratioLow, o.ratioHigh, o.params / 4.0 );

		const double beFast = loads[ i ].fastestTraversals / o.ratio;
		const double beSecond = loads[ i ].secondTraversals / o.ratio;
		printf( "  break-even vs %-8s (%u traversals): %5.2f LM traversals\n",
			loads[ i ].fastestArm.c_str(), loads[ i ].fastestTraversals, beFast );
		printf( "  break-even vs %-8s (%u traversals): %5.2f LM traversals\n",
			loads[ i ].secondArm.c_str(), loads[ i ].secondTraversals, beSecond );

		if ( beFast >= 2.0 )
		{
			anyWinRegion = true;
			printf( "  => a credible win region: LM has %.2f traversals to beat the"
				" fastest arm\n", beFast );
		}
		else
			printf( "  => no win region: LM must finish in under %.2f traversals\n",
				beFast );
		printf( "\n" );
	}

	// --- DIAGNOSIS, not the gate --------------------------------------------
	//     The ratios above are a property of the engine AS IT IS, which is the
	//     right comparison: the panel's traversal counts were produced by this
	//     same production path. But a ratio has two possible causes -- LM's extra
	//     work, and the existing path's overhead -- and reporting one number
	//     without attributing it would leave the reader unable to tell which.
	//
	//     batchGradient() adds its decay term PER EXEMPLAR as `hG += ( hW * decay )`
	//     and `oG += ( oW * decay )`. Both operators return BY VALUE, so each
	//     exemplar heap-allocates one Matrix and one vector. The proposed
	//     traversal adds decay ONCE, outside the loop. Re-running with weight
	//     decay off removes that asymmetry: what the ratio does between the two
	//     runs is the attribution.
	printf( "----------------------------------------------------------------\n" );
	printf( "DIAGNOSIS (not the gate): the same measurement with weight decay OFF.\n"
		"batchGradient() allocates a temporary Matrix and vector PER EXEMPLAR for\n"
		"its decay term; the proposed traversal adds decay once per traversal.\n\n" );
	printf( "  %-16s %12s %12s %10s %10s\n", "workload", "prod off", "LM off",
		"ratio off", "ratio on" );
	for ( size_t i = 0; i < loads.size(); i++ )
	{
		Outcome on, off;
		try
		{
			on = measure( loads[ i ], reps, orchSeed + ( unsigned ) i );
			off = measure( loads[ i ], reps, orchSeed + ( unsigned ) i, 0 );
		}
		catch ( const exception& e )
		{
			printf( "  %-16s REFUSED: %s\n", loads[ i ].name.c_str(), e.what() );
			continue;
		}
		if ( !on.valid || !off.valid )
		{
			printf( "  %-16s REFUSED: %s\n", loads[ i ].name.c_str(),
				on.valid ? off.refusal.c_str() : on.refusal.c_str() );
			continue;
		}
		printf( "  %-16s %9.4f ms %9.4f ms %9.2fx %9.2fx\n",
			loads[ i ].name.c_str(), off.prod.median, off.lm.median,
			off.ratio, on.ratio );
	}
	printf( "\n" );

	printf( "----------------------------------------------------------------\n" );
	if ( anyInvalid )
	{
		printf( "VERDICT: INCONCLUSIVE -- a workload was refused before timing.\n"
			"No ratio from a refused workload may be used as evidence.\n" );
		return 1;
	}
	if ( anyWinRegion )
		printf( "VERDICT: a two-or-more-traversal win region EXISTS.\n"
			"Per the predeclared gate: STOP and report the evidence before\n"
			"implementing Algorithm 3.16.\n" );
	else
		printf( "VERDICT: break-even is below two LM traversals on EVERY eligible\n"
			"workload. Per the predeclared gate: REJECT LM at Step L0, record the\n"
			"negative result, and remove every disposable source change.\n" );
	return 0;
}
