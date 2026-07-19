// PlateauDetector -- a pure, engine-agnostic early-stop signal for iterative
//    training. Feed it one training-set error per iteration via update(); it
//    returns true when the error has stopped improving and the run should stop.
//
// Header-only and dependency-free by design (ROADMAP 2 Phase 3): it knows
//    nothing about Network/Iterative, so it is trivially unit-testable against
//    synthetic error traces without building the engine.
//
// Mechanism: a two-window moving-average comparison. It holds the most recent
//    2*window errors as two adjacent windows -- the "previous" window and the
//    "recent" window -- and compares their means. Relative improvement
//        (mean_prev - mean_recent) / |mean_prev|
//    below tol is a STRIKE. A falling-but-flat trace strikes (improvement below
//    tol); a RISING trace strikes too (improvement is negative, hence below
//    tol), so the detector doubles as crude overlearning protection. `patience`
//    consecutive strikes fire the stop.
//
// Why two windows and not a single slope: averaging over a full window on each
//    side smooths the oscillation that CGD/Shanno and auto-step-size training
//    produce (their traces are noisy, though -- verified 2026-07-19 -- NOT
//    periodic with df(), so there is deliberately no period-based window
//    widening here; a window of ~100 already spans many teeth).
//
// NaN/inf handling (a diverged run feeds non-finite errors): a non-finite error
//    is counted as a strike directly -- it is certainly not improvement -- and
//    is kept OUT of the moving-average windows so it cannot poison later means.
//    Persistent divergence therefore plateau-stops within `patience` steps
//    rather than sailing to maxIterations. Every comparison against a NaN in the
//    means would be false (never triggering), which is exactly the trap that bit
//    the autoalgo probes twice; handling it before the windows avoids it here.

#ifndef PLATEAU_H
#define PLATEAU_H

#include <deque>
#include <cmath>
#include <cstddef>

class PlateauDetector {
public:
	// window   = width of each of the two averaging windows (>= 2)
	// tol      = relative-improvement threshold below which an update strikes
	// patience = consecutive strikes required to fire the stop
	PlateauDetector( unsigned window = 100, double tol = 1e-4,
		unsigned patience = 3 )
		: window_( window < 2 ? 2 : window ), tol_( tol ),
		  patience_( patience < 1 ? 1 : patience )
	{
		reset();
	}

	// Forget all history (a fresh training run reuses the same detector).
	void reset()
	{
		prev_.clear();
		recent_.clear();
		sumPrev_ = 0.0;
		sumRecent_ = 0.0;
		strikeCount_ = 0;
	}

	// Feed one iteration's error; returns true when the stop should fire.
	//    Once it has fired it keeps returning true until reset().
	bool update( double error )
	{
		// A non-finite error is a strike outright and never enters the
		//    windows (it would poison every future mean). See header note.
		if ( !std::isfinite( error ) )
		{
			strikeCount_++;
			return strikeCount_ >= patience_;
		}

		push( error );

		// Need both windows full before the comparison means anything.
		if ( prev_.size() < window_ || recent_.size() < window_ )
			return false;

		double meanPrev = sumPrev_ / ( double ) window_;
		double meanRecent = sumRecent_ / ( double ) window_;

		// Denominator guards a zero mean (error genuinely at 0 -> converged;
		//    any residual counts as no improvement, i.e. a strike).
		double denom = std::fabs( meanPrev );
		bool improved = ( denom > 0.0 )
			? ( ( meanPrev - meanRecent ) / denom >= tol_ )
			: ( meanPrev - meanRecent >= tol_ ); // meanPrev == 0

		if ( improved )
			strikeCount_ = 0;
		else
			strikeCount_++;

		return strikeCount_ >= patience_;
	}

	unsigned strikes() const { return strikeCount_; }

private:
	// Slide the newest error into `recent_`; overflow of `recent_` cascades
	//    the oldest recent value into `prev_`, whose overflow is discarded.
	//    Two running sums keep update() O(1).
	void push( double error )
	{
		recent_.push_back( error );
		sumRecent_ += error;

		if ( recent_.size() > window_ )
		{
			double moved = recent_.front();
			recent_.pop_front();
			sumRecent_ -= moved;

			prev_.push_back( moved );
			sumPrev_ += moved;

			if ( prev_.size() > window_ )
			{
				double dropped = prev_.front();
				prev_.pop_front();
				sumPrev_ -= dropped;
			}
		}
	}

	unsigned window_, patience_;
	double tol_;

	std::deque< double > prev_, recent_; // the two adjacent averaging windows
	double sumPrev_, sumRecent_;         // running sums, so update() is O(1)
	unsigned strikeCount_;               // consecutive strikes so far
};

#endif
