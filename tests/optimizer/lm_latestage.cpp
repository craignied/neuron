// lm_latestage.cpp : CHARACTERIZE the Levenberg-Marquardt late-stage failure.
//
// It does not fix anything, and it changes no declared constant. It answers a
// list of questions about a failure that has already been observed, and it is
// deliberately separate from the screen so that nothing here can be mistaken
// for a tuning pass.
//
// THE OBSERVED FACT, stated precisely. On the late-stage arm -- the engine's own
// plateau rule, no objective target -- LM raised StepRejected after 68
// iterations and 110 traversals. What was observed is ONLY that twenty
// consecutive trials failed the acceptance rule `rho > 0`. It is NOT established
// that no damping could improve F, and this program exists because that
// distinction is the whole question.
//
// WHAT IT REPORTS, per the characterization list:
//
//   * the last accepted objective and its gradient maximum;
//   * accepted and rejected counts, and the factorization/non-finite counts;
//   * final mu, nu, step norm, and the gain ratio's NUMERATOR and DENOMINATOR
//     for every rejected trial;
//   * whether the trial objectives are finite but indistinguishable from the
//     accepted one at machine precision;
//   * whether the parameters are restored exactly after StepRejected;
//   * where the run stands against the successful late-stage panel arms.
//
// HOW IT SEES INSIDE. LM's own iterate() is driven directly through a wrapper
// LMObjective that delegates to the model's real boundary and LOGS every trial
// objective on the way past. No engine source is modified and no constant is
// touched; the wrapper is a caller, exactly as Network::NormalEvaluator is.
//
//   ./build/lm_latestage

#include "stdafx.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include "harness.h"

using namespace std;
using namespace optbench;

// The model, with the boundary and the optimizer reachable.
class LateProbe : public SimpleProp {
public:
	void prepare() { prepareRun(); }
	unsigned packed() const { return packedSize(); }
	void weightsOut( vector< double >& w ) const { packWeights( w ); }
	void weightsIn( const vector< double >& w ) { unpackWeights( w ); }
	double normalPass( Matrix< double >& A, vector< double >& g )
	{ return batchNormalEquations( A, g ); }
	double gradientPass( vector< double >& g )
	{ return batchObjectiveGradient( g ); }
	LM& optimizer() { return lm; }
};

// A wrapper that does exactly what Network::NormalEvaluator does, and records
//    every objective it returns. Nothing here changes what LM computes.
class Logging : public LMObjective {
public:
	explicit Logging( LateProbe& p ) : net( p ), traversals( 0 ) { }

	LateProbe& net;
	unsigned traversals;
	vector< double > seen;      // every objective returned, in order

	virtual void currentPoint( vector< double >& w ) const { net.weightsOut( w ); }
	virtual void install( const vector< double >& w ) { net.weightsIn( w ); }
	virtual bool cancelled() const { return false; }
	virtual double evaluateNormal( Matrix< double >& A, vector< double >& g )
	{
		traversals++;
		const double f = net.normalPass( A, g );
		seen.push_back( f );
		return f;
	}
};

static double norm2( const vector< double >& v )
{
	double s = 0;
	for ( size_t i = 0; i < v.size(); i++ ) s += v[ i ] * v[ i ];
	return sqrt( s );
}

int main()
{
	util::ScreenCapture hush;

	// THE LATE-STAGE ARM'S OWN CONFIGURATION, taken from the harness so this
	//    characterizes the arm that failed rather than a workload invented for
	//    the occasion.
	Case c = civicCase( "simpleprop", 6000, CIVIC_HIDDEN, ENDPOINT_PRACTICAL,
		Network::TRAIN_LM, false );
	c.minStop = false;
	c.autoStop = true;
	c.endpoint = ENDPOINT_NONE;

	unsigned long long dataId = 0, splitId = 0;
	unsigned rowsTotal = 0, rowsTest = 0;
	DataSet d = makeDataSet( c, dataId, splitId, rowsTotal, rowsTest );

	LateProbe p;
	p.setHistory( false );
	p.setLastop( false );
	p.setQuiet( true );
	p.setLMSerror();
	p.setWeightDecay( c.decayOn );
	p.setDecay( c.decay );
	p.setBatchEpoch( true );
	p.setAutoStepSize( false );
	p.setEta( c.eta );
	p.setDataSet( d );
	p.setHidden( c.hidden );
	p.setTrainingType( Network::TRAIN_LM );
	util::set_seed( c.weightSeed );
	p.randomize();
	p.prepare();

	Logging f( p );
	LM& lm = p.optimizer();

	printf( "LM LATE-STAGE CHARACTERIZATION\n" );
	printf( "workload: civic 6000, SimpleProp h%u, P = %u, %u training rows\n",
		c.hidden, p.packed(), d.getNumTrain() );
	printf( "build %s, engine %s\n\n", OPTBENCH_BUILD_TYPE, OPTBENCH_ENGINE_ID );

	// Iterate until LM refuses, capturing the state of the LAST SUCCESSFUL
	//    iteration before the failing one.
	unsigned iterations = 0;
	double lastReturned = 0, muBeforeFailure = 0, nuBeforeFailure = 0;
	double gradMaxBefore = 0, acceptedBefore = 0;
	vector< double > pointBeforeFailure, gradBefore;
	size_t seenBeforeFailure = 0;
	bool refused = false;

	for ( ; iterations < 100000; iterations++ )
	{
		muBeforeFailure = lm.started() ? lm.damping() : 0.0;
		nuBeforeFailure = lm.started() ? lm.rejectionMultiplier() : 0.0;
		acceptedBefore = lm.started() ? lm.acceptedObjective() : 0.0;
		gradMaxBefore = lm.started() ? lm.gradMax() : 0.0;
		if ( lm.started() ) gradBefore = lm.gradient();
		p.weightsOut( pointBeforeFailure );
		seenBeforeFailure = f.seen.size();

		try { lastReturned = lm.iterate( f ); }
		catch ( const LM::StepRejected& ) { refused = true; break; }
		catch ( const LM::NotFinite& )
		{ printf( "STOPPED: NotFinite\n" ); return 1; }
	}

	if ( !refused )
	{
		printf( "no refusal within %u iterations -- nothing to characterize\n",
			iterations );
		return 1;
	}

	printf( "REFUSED after %u completed iterations and %u traversals\n\n",
		iterations, f.traversals );

	// --- 1. the last accepted point -----------------------------------------
	printf( "1. LAST ACCEPTED STATE (the point the failing iteration departed from)\n" );
	printf( "   accepted objective   %.17g\n", acceptedBefore );
	printf( "   gradient maximum     %.17g\n", gradMaxBefore );
	printf( "   ||g||                %.17g\n", norm2( gradBefore ) );
	printf( "   engine gradmax rule  %s\n\n",
		gradMaxBefore < 1e-6 ? "WOULD have fired (< 1e-6)"
			: "would NOT have fired (>= 1e-6)" );

	// --- 2. counts ----------------------------------------------------------
	printf( "2. COUNTS OVER THE WHOLE RUN\n" );
	printf( "   iterations %llu   accepted %llu   rejected %llu\n",
		lm.iterations(), lm.acceptances(), lm.rejections() );
	printf( "   factorization failures %llu   non-finite trials %llu\n\n",
		lm.factorizationFailures(), lm.nonFiniteTrials() );

	// --- 3. the failing iteration, trial by trial ---------------------------
	//     The trial objectives of the failing iteration are exactly the entries
	//     the wrapper recorded after the last accepted one.
	printf( "3. THE FAILING ITERATION, TRIAL BY TRIAL\n" );
	printf( "   departing objective  %.17g\n", acceptedBefore );
	printf( "   mu at entry          %.17g\n", muBeforeFailure );
	printf( "   nu at entry          %.17g\n", nuBeforeFailure );
	printf( "   final mu             %.17g\n", lm.damping() );
	printf( "   final nu             %.17g\n", lm.rejectionMultiplier() );
	printf( "   final ||h||          %.17g\n\n", norm2( lm.lastStep() ) );

	printf( "   %-4s %-24s %-14s %-14s %-9s %s\n",
		"k", "trial objective", "numerator", "denominator", "rho", "ulps below F" );

	// Replay the damping schedule to recover each trial's mu, and with it the
	//    denominator 0.5 h'( mu h - g ). Every quantity here is recomputed from
	//    the SAME published formulas the optimizer uses, on the accepted point's
	//    A and g, which LM still holds.
	{
		Matrix< double > A = lm.normalMatrix();
		vector< double > g = lm.gradient();
		double mu = muBeforeFailure, nu = nuBeforeFailure;

		for ( size_t k = seenBeforeFailure; k < f.seen.size(); k++ )
		{
			Matrix< double > damped = A;
			for ( unsigned i = 0; i < A.rows(); i++ ) damped( i, i ) += mu;
			vector< double > rhs( g.size() ), h;
			for ( size_t i = 0; i < g.size(); i++ ) rhs[ i ] = -g[ i ];

			double denom = 0, num = 0, rho = 0;
			bool solved = true;
			try { damped.solveSPD( rhs, h ); }
			catch ( const Matrix< double >::Singular& ) { solved = false; }

			const double trialF = f.seen[ k ];
			if ( solved )
			{
				for ( size_t i = 0; i < h.size(); i++ )
					denom += h[ i ] * ( mu * h[ i ] - g[ i ] );
				denom *= 0.5;
				num = acceptedBefore - trialF;
				rho = num / denom;
			}

			// How many representable doubles separate the trial objective from
			//    the accepted one? THIS is the question "finite but
			//    indistinguishable at machine precision" asks.
			double ulps = 0;
			{
				const double a = acceptedBefore, b = trialF;
				if ( std::isfinite( a ) && std::isfinite( b ) )
				{
					const double spacing = nextafter( a, HUGE_VAL ) - a;
					ulps = spacing > 0 ? ( b - a ) / spacing : 0.0;
				}
			}

			printf( "   %-4zu %-24.17g %-14.6g %-14.6g %-9.3g %.3g\n",
				k - seenBeforeFailure + 1, trialF, num, denom, rho, ulps );

			mu *= nu;
			nu *= 2.0;
		}
	}
	printf( "\n" );

	// --- 4. exact restoration ------------------------------------------------
	vector< double > after;
	p.weightsOut( after );
	printf( "4. RESTORATION AFTER StepRejected\n" );
	printf( "   parameters bit-identical to the last accepted point: %s\n\n",
		( after == pointBeforeFailure ) ? "YES" : "NO" );

	// --- 5. where the run stands against the successful panel arms ----------
	Matrix< double > A;
	vector< double > g;
	const double objectiveNow = p.normalPass( A, g );
	printf( "5. WHERE THE RUN STANDS\n" );
	printf( "   training objective   %.17g\n", objectiveNow );
	printf( "   held-out error       %.17g\n", p.sampleTestError( 1 ) );
	printf( "   (late-stage panel, one repetition: Shanno 0.0914732 train /"
		" 0.0870610 held-out,\n"
		"    L-BFGS 0.0916086 / 0.0874695, iRPROP+ 0.0940683 / 0.0884944)\n\n" );

	printf( "   value LM returned from its last successful iteration: %.17g\n",
		lastReturned );
	return 0;
}
