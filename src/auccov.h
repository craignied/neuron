/* The parts of an AUC covariance analysis that do NOT depend on the sampling
   unit (ROADMAP 4).

   Two estimators live above this file -- ordinary DeLong for independent rows
   (delong.h) and Obuchowski's clustered covariance (clustered_auc.h). They
   differ ONLY in how they combine the structural components into a covariance
   matrix. Everything else must be identical between them or their results are
   not comparable:

     - the point areas, and the tie handling that produces them. A "clustered
       AUC" that disagreed with the ordinary AUC on the same data in the fourth
       decimal because one used mid-ranks and the other a pairwise loop would be
       a defect that no test of either estimator alone could see;
     - the placements (DeLong's V10 / V01), which both estimators sum;
     - the algebra that turns ( areas, covariance ) into an interval or a paired
       contrast -- including the zero-variance case law established by DLG-2,
       which is about the covariance matrix and not about how it was estimated.

   So those live here, once, and each estimator supplies only its own
   combination step. Class-layer note (rule 4): the covariance is a
   Matrix< double >; the mid-rank computation is order-statistic code and is the
   one genuinely index-based part, marked where it lives. */

#ifndef AUCCOV_H
#define AUCCOV_H

#include <string>
#include <vector>

#include "matrix.h"

using namespace std;

namespace auccov {

// 1-based mid-ranks of x (ties share the average of the ranks they span).
vector< double > midranks( const vector< double >& x );

// DeLong's structural components for m procedures scored on one paired sample.
//    pos / neg are the row indices of the outcome-1 / outcome-0 observations, in
//    input order. v10[ k ][ a ] is the fraction of outcome-0 rows that procedure
//    k's score for positive pos[ a ] beats (half credit for ties); v01[ k ][ b ]
//    the fraction of outcome-1 rows that negative neg[ b ] loses to. auc[ k ] is
//    the mean of v10[ k ] -- the exact Mann-Whitney area.
//
//    ok is false, with a reason, when there are no procedures, a length does not
//    match, a score is not finite (a diverged model has no meaningful area), or a
//    class is empty. It does NOT impose a minimum class size: how many
//    observations an estimator needs is the ESTIMATOR's rule (ordinary DeLong
//    needs two of each class for its divisors; the clustered estimator counts
//    clusters instead), so each states its own.
struct Placements {
	bool ok = false;
	string reason;
	unsigned n0 = 0, n1 = 0;
	vector< unsigned > pos, neg;
	vector< vector< double > > v10, v01;
	vector< double > auc;
};
Placements compute( const vector< unsigned >& label,
	const vector< vector< double > >& pred );

// A single area with its 95% normal (Wald) interval on the area scale.
//    clampToUnit reflects a DELIBERATE difference between the two estimators:
//    the ordinary-DeLong path clamps to [0,1] (classic DeLong 1988, DLG-9), while
//    the clustered path does not, because the published reference implementation
//    reports an unclamped interval and matching it is part of that acceptance
//    test -- and because an interval that has run past 1 is telling the reader
//    the normal approximation is straining, which truncation hides.
struct Interval {
	double auc = 0, se = 0, lo = 0, hi = 0;
	bool valid = false;
};
Interval interval( const vector< double >& auc, const Matrix< double >& cov,
	unsigned i, bool clampToUnit );

// A pairwise contrast of areas i and j. delta is always auc[ i ] - auc[ j ] (the
//    caller passes i = the prespecified PRIMARY), p is two-sided (H0: equal
//    areas). valid is false when i/j are out of range or the difference variance
//    is materially negative (an invalid covariance, refused not silently zeroed).
//
//    The zero-difference-variance case has TWO distinct meanings and they must not
//    be conflated (DLG-2):
//      - degenerate: seDelta == 0 AND delta == 0 -- equal areas with no spread, so
//        there is genuinely NO testable difference (p reported as 1).
//      - separated:  seDelta == 0 AND delta != 0 -- the placements are constant and
//        the areas differ deterministically (e.g. AUC 0 vs AUC 1): the difference
//        is real and total, not absent. p is 0 and the flag says so.
//    The split uses a SCALE-AWARE tolerance, never an exact == 0: this function
//    accepts an arbitrary covariance, so floating-point cancellation can clamp the
//    difference variance to zero while delta is a sub-ULP nonzero, and an exact
//    test would report that numerical noise as a deterministic separation with
//    p = 0 (Sol's counterexample: auc { 0.5, nextafter( 0.5, 1 ) }, equal
//    diagonals, off-diagonal one ULP above).
struct Contrast {
	double aucI = 0, aucJ = 0, delta = 0;
	double seI = 0, seJ = 0, seDelta = 0;
	double ciLoI = 0, ciHiI = 0;
	double ciLoJ = 0, ciHiJ = 0;
	double ciLoDelta = 0, ciHiDelta = 0;
	double z = 0, p = 1;
	bool degenerate = false;
	bool separated = false;
	string note;
	bool valid = false;
};
Contrast contrast( const vector< double >& auc, const Matrix< double >& cov,
	unsigned i, unsigned j, bool clampToUnit );

} // namespace auccov

#endif
