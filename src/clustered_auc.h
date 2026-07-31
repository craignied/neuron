/* Obuchowski's nonparametric clustered ROC-area covariance (ROADMAP 4).

   The successor to ordinary DeLong for test data whose rows share a SAMPLING
   UNIT -- two measurements on one subject, two pupils in one classroom, patients
   under one clinician, households in a survey. Rows within a cluster are
   correlated, so the row count overstates how much independent information the
   sample carries, and ordinary DeLong -- which divides by rows and has no term
   for the within-cluster correlation -- simply does not estimate the right
   covariance.

   Its standard error can come out TOO SMALL OR TOO LARGE, depending on the
   within-cluster covariance structure. Measured both ways here: when a cluster
   effect shifts the two outcome classes together, within-cluster pairs become
   more concordant and the row-based SE is too LARGE; when each cluster carries a
   single outcome class, the positives and negatives clump and the row-based SE is
   badly too SMALL (measured: 0.0428 against a correct 0.0967). The published
   reference example has the cross term positive for one reader and negative for
   the other. So "clustering widens the interval" is NOT the rule -- the rule is
   that the row-based estimator is answering a different question.

   A cluster is whatever the caller's group key says it is; nothing here knows or
   cares what it represents.

   WHAT DOES NOT CHANGE. The point area is still the exact patient-ROW
   Mann-Whitney probability over all pairs, computed by the SAME placement code
   as ordinary DeLong (auccov::compute) so the two estimators can never disagree
   about the area or about tie handling. Clustering is a statement about the
   variance, not about the estimand. Do NOT cluster-average the area itself --
   that estimates a different quantity.

   WHAT THIS IS NOT. Group-aware SPLITTING and clustered INFERENCE are different
   mechanisms and are not substitutes. Keeping a cluster wholly on one side of a
   split prevents leakage; it does not make the rows within a held-out cluster
   independent. See docs/roc_theory.md, "Clustered ROC data".

   Reference: Obuchowski NA, Nonparametric analysis of clustered ROC curve data,
   Biometrics 1997;53:567-578 (DOI 10.2307/2533958); the clustered generalization
   of DeLong, DeLong & Clarke-Pearson (1988), Biometrics 44:837-845. The
   implementation is validated against the paper's own worked example, including
   its published intermediate components -- tests/clustered/check_clustered.cpp. */

#ifndef CLUSTERED_AUC_H
#define CLUSTERED_AUC_H

#include <string>
#include <vector>

#include "auccov.h"
#include "matrix.h"

using namespace std;

namespace clustered_auc {

// The areas and their clustered covariance for m procedures scored on one paired
//    sample. auc[ k ] is procedure k's area (identical to the ordinary
//    Mann-Whitney area on the same data); cov( k, l ) is the clustered
//    covariance. ok is false with a reason when the estimator cannot be formed.
//
//    The counts are all reported even on a refusal, because they are what a
//    caller needs to explain it: nClusters, and how many of those clusters carry
//    each outcome class ( clusters10 / clusters01 -- the divisors this estimator
//    actually uses, which is why "too few clusters" is a different condition from
//    "too few rows" ).
struct Result {
	bool ok = false;
	string reason;
	unsigned nRows = 0;
	unsigned nClusters = 0;
	unsigned clusters10 = 0;  // clusters holding >= 1 outcome-1 row
	unsigned clusters01 = 0;  // clusters holding >= 1 outcome-0 row
	unsigned n0 = 0, n1 = 0;
	vector< double > auc;
	Matrix< double > cov;
	string method;            // the estimator's own name, for the report
};

// Compute the areas and the clustered covariance. label[ r ] is observation r's
//    outcome (nonzero = the outcome-1 class); cluster[ r ] is its sampling-unit
//    id (ids need not be dense or ordered -- they are compacted internally, and
//    the result is invariant to how they are numbered); pred[ k ] is procedure
//    k's score for every observation, higher = more positive, all the same length
//    as label (the observations are paired across procedures).
//
//    Refused, with a reason, when: there are no procedures, a length does not
//    match, a score is not finite, a class is absent, or fewer than two clusters
//    carry a given outcome class (the S10 / S01 divisors are ( I10 - 1 ) and
//    ( I01 - 1 ), so one informative cluster leaves no spread to estimate --
//    a refusal, never a division by zero dressed as an answer).
Result analyze( const vector< unsigned >& label,
	const vector< unsigned >& cluster,
	const vector< vector< double > >& pred );

// The paired contrast and the single-area interval, over the clustered
//    covariance. Both delegate to the shared algebra (auccov), so the
//    zero-variance case law is identical to the ordinary path by construction.
//
//    The area interval is NOT clamped to [0,1] here, deliberately: the reference
//    implementation reports an unclamped interval for the published example and
//    matching it is part of the acceptance test, and an interval that has run
//    past the boundary is telling the reader the normal approximation is
//    straining -- which truncation hides. The ordinary-DeLong path clamps
//    (classic DeLong 1988, DLG-9); the difference is intentional and noted in
//    both places.
auccov::Contrast contrast( const Result& r, unsigned i, unsigned j );
auccov::Interval interval( const Result& r, unsigned i );

} // namespace clustered_auc

#endif
