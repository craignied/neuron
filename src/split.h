/* Representative test-set splitting (ROADMAP 4).

Index-level train/test splitting, deliberately decoupled from the data Matrix so
it can be unit-tested in isolation and reused by every splitting mode. The caller
passes label / stratum / group vectors built from its own data; the splitter
returns row indices, which the caller gathers (DataSet uses Matrix::includerows).

Phase 1 provides the simplest cell of the design -- an outcome-stratified single
holdout. Later phases extend this namespace with named-covariate strata,
group-aware assignment, and k-fold estimators (see ROADMAP 4 in CLAUDE.md). */

#ifndef SPLIT_H
#define SPLIT_H

#include <vector>

using namespace std;

namespace nsplit {

// The result of a stratified single holdout at the index level. test and train
//    hold row indices into the caller's data; together they partition
//    [ 0, label.size() ) with no overlap and no repeats. The count fields let
//    the caller report class frequencies without a second pass over the data.
struct Holdout {
	vector< unsigned > test;  // row indices assigned to the test set
	vector< unsigned > train; // row indices assigned to the training set
	unsigned n0, n1;          // number of outcome-0 and outcome-1 rows
	unsigned n0Test, n1Test;  // number of each placed in the test set
};

// Stratified single holdout on a binary outcome. label[ r ] must be 0 or 1;
//    nTest is the total number of rows for the test set ( nTest <= n ). Test
//    slots are apportioned to the two classes in proportion to prevalence --
//    round-half-up on the 0 class, exactly as the legacy splitter did, so the
//    population base rate is preserved -- then each class's row indices are
//    drawn by a partial Fisher-Yates shuffle using util::i_random (the training
//    RNG stream, seeded by util::set_seed). O(n) time and one allocation per
//    output vector; reproducible under a fixed seed.
Holdout stratifiedHoldout( const vector< unsigned >& label, unsigned nTest );

// The result of a general multi-cell stratified holdout. test and train
//    partition the rows as above; cellTotal[ s ] and cellTest[ s ] give each
//    stratum's size and test count, for the representativeness diagnostic.
struct StratHoldout {
	vector< unsigned > test, train;
	vector< unsigned > cellTotal, cellTest; // indexed by stratum id
};

// General stratified holdout. stratum[ r ] is the stratum id ( 0 .. S-1 ) for
//    row r -- a cell of the outcome x named-covariate cross-classification.
//    The nTest test slots are apportioned across strata in proportion to each
//    stratum's size by largest-remainder (Hamilton) apportionment -- the whole
//    remainder goes to the strata with the largest fractional quotas, ties to
//    the lower stratum id -- then drawn per stratum by partial Fisher-Yates.
//    This is the ROADMAP 4 Phase 2 generalization of stratifiedHoldout above;
//    the two-class outcome-only case ( strata = the outcome itself ) is left on
//    the dedicated function above so its byte-for-byte behavior never moves.
StratHoldout holdoutByStrata( const vector< unsigned >& stratum, unsigned nTest );

}

#endif
