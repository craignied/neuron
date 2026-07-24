/* DeLong's test for correlated ROC areas (ROADMAP 4 Phase 4 -- the locked-test
   inference layer). Given several classifiers scored on the SAME test
   observations (paired), DeLong (1988) estimates each area under the ROC curve,
   the full covariance among those areas, and thence the difference between any
   two areas with its standard error, z, and two-sided p-value. It is the standard
   inferential comparison of correlated AUCs.

   INDEPENDENCE SCOPE -- read this before using the result inferentially. Ordinary
   DeLong assumes the test OBSERVATIONS are independent (one row = one independent
   subject). It is valid on a single untouched locked test set drawn IID. It is NOT
   valid on:
     - out-of-fold cross-validation predictions (k overlapping fitted models inject
       dependence its variance ignores -- see docs/cross_validation.md), or
     - clustered test data (e.g. SEER patients sharing a county). Group-aware
       SPLITTING prevents train/test leakage but does NOT make the test rows
       independent, so it does not satisfy DeLong's assumption. Cluster-aware test
       inference (a clustered/aggregated variance) is a deliberate FOLLOW-ON item,
       not this layer. Use this only when the locked test rows are independent.

   Class-layer note (rule 4): the covariance is a Matrix< double > so the algebra
   reads like DeLong's paper and every access is bounds-checked. The placement
   ("structural component") computation is order-statistic code -- mid-ranks over
   sorted scores -- and is genuinely index-based; it is the one scalar part, marked
   where it lives.

   Reference: DeLong, DeLong & Clarke-Pearson (1988), Biometrics 44:837-845; the
   fast mid-rank formulation follows Sun & Xu (2014), IEEE SPL 21:1389-1393. */

#ifndef DELONG_H
#define DELONG_H

#include <string>
#include <vector>

#include "matrix.h"

using namespace std;

namespace delong {

// The AUCs and their covariance for a set of m classifiers scored on one paired
//    test set. auc[ k ] is classifier k's area; cov( k, l ) is the DeLong
//    covariance of areas k and l (cov( k, k ) is Var of area k). ok is false when
//    the set is degenerate for the estimator -- fewer than two positives or two
//    negatives, so no covariance divisor exists -- with reason set.
struct Result {
	unsigned n0 = 0;          // negatives (outcome 0) in the test set
	unsigned n1 = 0;          // positives (outcome 1)
	vector< double > auc;     // per classifier
	Matrix< double > cov;     // m x m covariance of the areas (empty if !ok)
	bool ok = false;
	string reason;
};

// Compute the AUCs and their covariance. label[ r ] is observation r's true
//    outcome (nonzero = positive, the outcome-1 class); pred[ k ] is classifier
//    k's predicted score for every observation, oriented so a HIGHER score means
//    more positive (the engine's convention). Each pred[ k ].size() must equal
//    label.size() (the observations are paired). The result is refused (ok false,
//    reason set) when: there are no classifiers, a length does not match, a score
//    is non-finite (a diverged model has no meaningful area), or a class has fewer
//    than two observations (no covariance divisor). n0/n1 are always reported.
Result analyze( const vector< unsigned >& label,
	const vector< vector< double > >& pred );

// A pairwise contrast of areas i and j from a Result. The difference is always
//    delta = auc[ i ] - auc[ j ] (the caller passes i = the prespecified PRIMARY,
//    j = the REFERENCE, so delta reads AUC(primary) - AUC(reference) everywhere).
//    p is two-sided (H0: equal areas). ci* are 95% normal intervals on the area
//    scale, clamped to [0,1] (classic DeLong; a logit interval is a noted future
//    refinement). valid is false when i/j are out of range or the Result is not
//    ok. degenerate is true when the two classifiers give the SAME area with zero
//    difference-variance (e.g. identical predictions) -- there is then no testable
//    difference: delta is 0, se(delta) is 0, and p is reported as 1, but the flag
//    lets the caller say so explicitly rather than present a fabricated z/p.
struct Contrast {
	double aucI = 0, aucJ = 0, delta = 0;   // areas and their difference (i - j)
	double seI = 0, seJ = 0, seDelta = 0;   // standard errors
	double ciLoI = 0, ciHiI = 0;            // 95% CI on area i
	double ciLoJ = 0, ciHiJ = 0;            // 95% CI on area j
	double ciLoDelta = 0, ciHiDelta = 0;    // 95% CI on the difference
	double z = 0, p = 1;                    // test statistic and two-sided p
	bool degenerate = false;                // zero difference-variance (no test)
	bool valid = false;
};
Contrast contrast( const Result& r, unsigned i, unsigned j );

// A single area with its 95% DeLong interval (the AUC (test) [95% CI] column).
struct Interval {
	double auc = 0, se = 0, lo = 0, hi = 0;
	bool valid = false;
};
Interval interval( const Result& r, unsigned i );

} // namespace delong

#endif
