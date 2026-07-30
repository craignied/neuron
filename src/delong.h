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

#include "auccov.h"
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
//    The type and ALL of its case law -- the two meanings of a zero difference
//    variance, the scale-aware split between them, the refusal of a materially
//    negative variance -- live in auccov, shared with the clustered estimator:
//    that reasoning is about a covariance matrix, not about how it was estimated.
//    Area intervals here ARE clamped to [0,1] (classic DeLong 1988; the clustered
//    path deliberately does not clamp -- see auccov.h).
typedef auccov::Contrast Contrast;
Contrast contrast( const Result& r, unsigned i, unsigned j );

// A single area with its 95% DeLong interval: a NORMAL (Wald) interval on the
//    area scale, auc +/- 1.96*se, clamped to [0,1]. Truncation near the boundary
//    shifts nominal coverage; a logit-scale or bootstrap interval for
//    near-boundary areas is noted future work (DLG-9). The report labels this the
//    Wald DeLong interval.
typedef auccov::Interval Interval;
Interval interval( const Result& r, unsigned i );

} // namespace delong

#endif
