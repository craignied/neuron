// iRPROP+. See irprop.h for the sources, the ownership boundary and why this is
// one named algorithm rather than a family behind variant flags.

#include "stdafx.h" // For MSVC, must be first!

#include <algorithm>
#include <cmath>
#include <limits>

#include "irprop.h"
#include "vector_ops.h"

// The published constants. Every one of them was fixed in
//    docs/learning_research/irprop_source_decision.md before a single
//    measurement was taken. Riedmiller & Braun (1993), carried unchanged by
//    Igel & Husken (2003).
const double IRpropState::DELTA_INIT = 0.1;
const double IRpropState::ETA_PLUS = 1.2;
const double IRpropState::ETA_MINUS = 0.5;
const double IRpropState::DELTA_MIN = 1e-6;
const double IRpropState::DELTA_MAX = 50.0;

// sign(), with sign(0) = 0. That third value is not a rounding decision: it is
//    what makes a coordinate with a zero gradient take no step in either
//    stepping branch, which is what the published table says.
static inline double signOf( const double v )
{
	return v > 0 ? 1.0 : ( v < 0 ? -1.0 : 0.0 );
}

IRpropState::IRpropState()
{
	reset();
}

void IRpropState::reset()
{
	delta.clear();
	prevG.clear();
	prevStep.clear();

	// Unread on the first iteration -- the zero previous gradient below sends
	//    every coordinate through the "= 0" branch, which does not compare
	//    objectives. Set to +infinity so that no rollback could be expressed
	//    even if the branch were somehow reached.
	prevF = numeric_limits< double >::infinity();

	startedFlag = false;
	grew = shrank = reverted = held = steps = 0;
}

// THE PUBLISHED TABLE, Igel & Husken (2003):
//
//   if   dE^(t-1)/dw * dE^(t)/dw > 0 :
//            Delta = min( Delta * etaPlus, DeltaMax )
//            dw    = -sign( dE^(t)/dw ) * Delta ;   w += dw
//   elif dE^(t-1)/dw * dE^(t)/dw < 0 :
//            Delta = max( Delta * etaMinus, DeltaMin )
//            if E^(t) > E^(t-1) :  w -= dw^(t-1)
//            dE^(t)/dw = 0
//   elif dE^(t-1)/dw * dE^(t)/dw = 0 :
//            dw    = -sign( dE^(t)/dw ) * Delta ;   w += dw
//
// WHY THIS IS A VISIBLE PER-PARAMETER LOOP rather than a sequence of whole-
//    vector operations. The algorithm IS a per-coordinate branch table: the
//    branch a parameter takes is chosen by its own sign product, and two
//    parameters in the same iteration routinely take different branches.
//    Expressing that with the vector vocabulary would need three masked
//    temporaries per iteration and would hide exactly the branch structure the
//    plan requires to stay visible. The loop is over PARAMETERS and runs once
//    per epoch -- the same place the epoch's single weight update already runs
//    -- so it is not an exemplar loop and rule 7 is untouched.
void IRpropState::computeStep( const double objective,
	const vector< double >& rawGradient, vector< double >& step )
{
	const unsigned n = ( unsigned ) rawGradient.size();

	if ( n == 0 )
		throw Ineligible( "iRPROP+ needs a model with parameters" );

	// REFUSED BEFORE ANYTHING IS MODIFIED, so a refused iteration has changed
	//    no Delta, no remembered gradient and no weight. A diverged run must not
	//    step on garbage and must not quietly become gradient descent.
	if ( !std::isfinite( objective ) )
		throw NotFinite();
	for ( unsigned i = 0; i < n; i++ )
		if ( !std::isfinite( rawGradient[ i ] ) )
			throw NotFinite();

	if ( !startedFlag )
	{
		// Delta_ij^(0) = Delta_0 for every parameter, and no previous gradient.
		//    A zero previous gradient makes every sign product zero, so the
		//    first iteration takes the "= 0" branch and steps by Delta_0 along
		//    the current gradient's sign -- which is precisely what
		//    Delta_ij^(0) = Delta_0 means.
		delta.assign( n, DELTA_INIT );
		prevG.assign( n, 0.0 );
		prevStep.assign( n, 0.0 );
		startedFlag = true;
	}
	else if ( delta.size() != n )
		// The parameter count changed inside a run. reset() is the only
		//    supported way for that to happen, and it goes through prepareRun().
		throw nvec::SizeMismatch();

	step.assign( n, 0.0 );

	// E^(t) > E^(t-1): the ERROR-DEPENDENT rollback condition, and the one line
	//    that separates iRPROP+ from RPROP+. Evaluated once for the whole
	//    iteration because the objective is a property of the point, not of a
	//    coordinate.
	const bool worse = objective > prevF;

	for ( unsigned i = 0; i < n; i++ )
	{
		const double g = rawGradient[ i ];
		const double product = prevG[ i ] * g;

		if ( product > 0 )
		{
			delta[ i ] = min( delta[ i ] * ETA_PLUS, DELTA_MAX );
			step[ i ] = -signOf( g ) * delta[ i ];
			prevG[ i ] = g;
			grew++;
		}
		else if ( product < 0 )
		{
			delta[ i ] = max( delta[ i ] * ETA_MINUS, DELTA_MIN );

			if ( worse )
			{
				// Revert the previous step -- w^(t+1) = w^(t) - dw^(t-1).
				step[ i ] = -prevStep[ i ];
				reverted++;
			}
			else
			{
				// THE iRPROP+ LINE. The objective improved despite the flip, so
				//    this coordinate takes NO step: it neither reverts nor
				//    advances. RPROP+ would have reverted here.
				step[ i ] = 0;
				held++;
			}

			// dE^(t)/dw := 0. This is the algorithm's MEMORY, not a cleanup:
			//    it forces the next iteration's sign product to zero, so that
			//    iteration necessarily takes the "= 0" branch and steps by the
			//    freshly shrunk Delta along the NEW gradient's sign. It is
			//    written into what this class remembers; the caller's raw
			//    gradient is untouched, because currGradMax is taken from it.
			prevG[ i ] = 0;
			shrank++;
		}
		else
		{
			// Delta is carried unchanged: the "= 0" branch of the published
			//    table updates the step, not the update value.
			step[ i ] = -signOf( g ) * delta[ i ];
			prevG[ i ] = g;
		}
	}

	// dw^(t): the step ACTUALLY APPLIED, which is what the next iteration's
	//    rollback branch would revert. After a no-rollback flip that value is 0,
	//    and it is provably never read -- the same branch zeroed the gradient,
	//    so the next iteration takes the "= 0" branch, which does not read it,
	//    and overwrites it before the "< 0" branch can fire again. That is a
	//    claim about the code, so tests/network/check_irprop.cpp pins it rather
	//    than leaving it as a comment.
	prevStep = step;
	prevF = objective;
	steps++;
}
