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

#include <string>
#include <vector>

using namespace std;

namespace nsplit {

// Validate a two-way row partition of a population of nRows rows (DLG-5). Returns
//    "" when well formed, else a human-readable reason. Checks, in this order:
//    every index < nRows, no duplicate within either side, the sides are DISJOINT
//    (an overlap would leak test rows into training), and -- only when
//    requireCoverage -- that train and test together cover EVERY row (so no row is
//    silently dropped). The shared partition-contract check for row-index entry
//    points; a locked-test evaluation, whose train+test must partition the whole
//    raw dataset, calls it with requireCoverage = true. It owns index bookkeeping
//    only, no fold or model policy (rule 6), and does NOT force coverage on callers
//    that intentionally select a subset -- that is the caller's declared choice.
string partitionError( unsigned nRows, const vector< unsigned >& train,
	const vector< unsigned >& test, bool requireCoverage );

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

// The result of a group-aware (outcome-stratified group) holdout. test and
//    train partition the rows with the guarantee that no group straddles them.
//    nGroups / groupsInTest describe the group partition for the diagnostic.
struct GroupHoldout {
	vector< unsigned > test, train;
	unsigned nGroups, groupsInTest;
};

// Group-aware holdout: every row of a group lands in the SAME set, so a cluster
//    (a hospital, a county) is never split across train and test. label[ r ] is
//    the binary outcome and group[ r ] the group id ( 0 .. G-1 ) for row r; the
//    caller keeps groups intact by giving rows that must stay together the same
//    id. Groups are visited in a seeded-random order and each whole group is
//    placed in the test set while that keeps the test class counts at or below
//    their outcome-proportional targets ( round( nTest * n_c / n ) ), otherwise
//    in training -- a greedy stratified-group assignment. Because groups are
//    indivisible the achieved test size only approximates nTest (a group larger
//    than the whole test target can never be tested); the split diagnostic
//    reports the achieved balance. This is the ROADMAP 4 Phase 3 tool for a
//    HARDER test -- generalization to groups the model never trained on --
//    whereas stratification makes the test resemble the sample.
GroupHoldout groupHoldout( const vector< unsigned >& label,
	const vector< unsigned >& group, unsigned nTest );

// A structured k-fold assignment: the plan, its achieved counts, and anything
//    the caller should be told about it. ok is false when no plan could be built
//    (with reason); warnings carry facts that do NOT invalidate the plan -- a
//    stratum too small to appear in every fold, most often.
//
//    foldId is the assignment ( row -> 0 .. k-1 ); foldSize[ f ] and
//    stratumByFold[ f ][ s ] are RECOMPUTED from that assignment rather than
//    accumulated while dealing, so the reported balance is what was actually
//    produced and not what the algorithm intended (a displayed fact is derived
//    from the object that owns it).
struct FoldPlan {
	bool ok = false;
	string reason;                              // why no plan, when !ok
	vector< unsigned > foldId;                  // per row, 0 .. k-1
	unsigned k = 0;
	unsigned nStrata = 0;
	vector< unsigned > foldSize;                // rows per fold
	vector< vector< unsigned > > stratumByFold; // [ fold ][ stratum ] achieved
	vector< string > warnings;
};

// General stratified k-fold assignment over ARBITRARY stratum ids (ROADMAP 4).
//    stratum[ r ] is row r's cell -- the outcome alone, or an outcome x named
//    covariate cross-classification (DataSet::strataKey builds those). Ids need
//    not be dense; they are compacted in ascending order, which is the order the
//    strata are dealt in.
//
//    Each stratum's rows are fully shuffled (seeded Fisher-Yates) and dealt
//    round-robin through ONE counter shared across strata, so a stratum's
//    remainder starts where the previous stratum stopped instead of every
//    remainder landing on fold 0 -- that bias is what makes small strata pile up
//    in the low-numbered folds. Every fold then holds ~n/k rows and ~each
//    stratum's population share.
//
//    kFold( label, k ) is exactly this function on the outcome alone; the
//    identity is asserted by tests/split/check_split.cpp against a range of
//    seeds, which is what licensed making one a wrapper for the other.
FoldPlan stratifiedKFold( const vector< unsigned >& stratum, unsigned k );

// A group-aware fold plan: a FoldPlan whose folds are made of WHOLE groups.
//    groupsPerFold and leakageCount describe the group partition; leakageCount
//    is recomputed independently from the assignment and must be 0 (a nonzero
//    value refuses the plan rather than reporting it).
//
//    imbalanceScore is the worst relative deviation of any fold's row count or
//    per-class count from its fair share -- 0.08 means "the worst fold is 8% off".
//    Groups are indivisible, so it is never 0 in general and a large value is a
//    fact about the group sizes, not a defect to tune away.
struct GroupFoldPlan : FoldPlan {
	unsigned nGroups = 0;
	vector< unsigned > groupsPerFold;
	vector< vector< unsigned > > classByFold; // [ fold ][ 0/1 ] achieved counts
	unsigned leakageCount = 0;
	double imbalanceScore = 0;
	unsigned largestGroup = 0;                // rows in the biggest group
};

// Outcome-balanced GROUP k-fold (ROADMAP 4). Every row of a group lands in the
//    same fold, so a cluster (a county, a hospital) is never split across folds
//    and each held-out fold measures generalization to groups the model never
//    trained on. label[ r ] is the binary outcome and group[ r ] the group id.
//
//    Indivisible groups make this bin packing, not dealing, so it is a separate
//    planner rather than a mode of stratifiedKFold. Groups are processed
//    hardest-first (descending size, then descending outcome imbalance) and each
//    whole group goes to the fold that minimizes the prospective deviation of
//    that fold's outcome-0 count, outcome-1 count, and total size from their fair
//    shares. Ties are broken by a seeded permutation of the fold order -- the ONLY
//    place randomness enters, so two runs of the same seed agree and the plan does
//    not depend on the arbitrary numbering of the groups.
//
//    An oversized group is never split to meet a target: the plan reports
//    largestGroup and imbalanceScore and lets the caller judge. Fewer than k
//    distinct groups is a refusal (some fold would be empty); fewer than k groups
//    carrying an outcome class is a warning, because those folds cannot supply an
//    AUC but the run is still meaningful.
GroupFoldPlan stratifiedGroupKFold( const vector< unsigned >& label,
	const vector< unsigned >& group, unsigned k );

// Stratified k-fold assignment on a binary outcome (ROADMAP 4 Phase 4).
//    label[ r ] must be 0 or 1; returns fold[ r ] in 0 .. k-1 with every row in
//    exactly one fold. Reproducible under util::set_seed; 2 <= k <= n. This is
//    the outcome-only case of stratifiedKFold above and delegates to it, so the
//    two can never drift apart.
vector< unsigned > kFold( const vector< unsigned >& label, unsigned k );

}

#endif
