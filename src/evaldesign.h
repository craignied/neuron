/* The evaluation DESIGN vocabulary (ROADMAP 4, DLG-8).

   How an evaluation was partitioned, what the independent sampling unit is, and
   which AUC inference (if any) that combination permits -- as TYPES, in one
   place, rather than as booleans and free text authored at each call site.

   Why this exists. The locked-test layer shipped with its design recorded as
   four strings the GUI filled in by hand ( samplingUnit / independenceStatus /
   inferenceMethod / splitPlan ) and a bool. Free text cannot be checked: nothing
   stops a caller writing "outcome-stratified row holdout" over a group-disjoint
   split, and nothing can ASK a report whether its design permits ordinary
   DeLong. The clustered work that follows needs exactly that question answered
   mechanically, so the vocabulary becomes structured first and the display
   strings become DERIVED (this file is their single source -- rule 6: a rule
   with two kinds of consumer is stated once, in the class layer).

   Nothing here computes a statistic or touches data; it is the description of a
   design plus the ONE policy function that maps a design to a permitted
   inference. Ownership: nsplit assigns indices, DataSet builds keys, crossval
   repeats, this names what was done and what may be concluded from it. */

#ifndef EVALDESIGN_H
#define EVALDESIGN_H

#include <string>
#include <vector>

using namespace std;

namespace evaldesign {

// What the caller declares to be the INDEPENDENT unit of observation. This is a
//    statement about the data, not about the split: a group-aware split keeps a
//    county whole but does not make two patients in that county independent.
//    Unspecified is the default and withholds inference -- silence is never read
//    as a declaration of independence (DLG-1).
enum class SamplingUnit { Unspecified, Row, Cluster };

// Which AUC covariance estimator was (or may be) applied. None is a first-class
//    outcome carrying a reason, not an error.
enum class AucInference { None, DeLongIndependent, ObuchowskiClustered };

// How rows were assigned to partitions (a holdout's two sides, or k folds).
enum class PartitionMethod {
	OutcomeStratified,   // balance the binary outcome only
	CovariateStratified, // balance outcome x named-covariate cells
	StratifiedGroup      // whole groups, outcome-balanced across partitions
};

// Parse the API's `independence=` value ( "rows" | "cluster" | "" ). Returns
//    false on anything else, so a handler reports a field error instead of
//    silently treating a typo as "not declared" -- the failure mode that would
//    quietly withhold inference the caller asked for.
bool parseSamplingUnit( const string& text, SamplingUnit& out );

// Display names -- the single source for the wording the report and the JSON
//    both show. samplingUnitName / independenceStatus describe the DECLARATION;
//    inferenceName names an estimator (never "none", which needs a reason --
//    use inferenceText).
string samplingUnitName( SamplingUnit u );
string independenceStatus( SamplingUnit u );
// The unit as an ADJECTIVE, for use inside a sentence ("the declared
//    independent-row sampling unit") -- samplingUnitName's parenthetical reads
//    as noise there.
string samplingUnitPhrase( SamplingUnit u );
string inferenceName( AucInference m );
string partitionMethodName( PartitionMethod m );

// "DeLong (ordinary, independent rows)" for an estimator that ran; "none (why)"
//    when it did not. A method of None with an empty reason is a bug in the
//    caller, not a display case: every refusal states its cause.
string inferenceText( AucInference m, const string& reason );

// The estimator's name where a verdict line has room for one word: "DeLong p =
//    0.653", "clustered ROC p = 0.41". Empty for None, so a caller that has no
//    inference cannot accidentally print a method name beside a point estimate.
string inferenceShortName( AucInference m );

// The structured description of ONE partitioning act: a locked holdout, or the
//    outer fold plan. Fields not applicable to the act are left at their
//    defaults and do not appear in the derived text.
struct Partition {
	PartitionMethod method = PartitionMethod::OutcomeStratified;
	unsigned seed = 0;
	unsigned k = 0;                    // folds; 0 for a two-way holdout

	// The user's 1-based input column numbers -- as TYPED, so a report can be
	//    read against the request. Densification to 0-based happens in DataSet.
	vector< unsigned > strataColumns;
	unsigned strataBins = 0;
	vector< unsigned > groupColumns;
	unsigned nGroups = 0;              // distinct groups over the rows partitioned

	bool developmentOnly = false;      // this plan folds development rows only

	// Requested vs ACHIEVED size, for a holdout. Groups are indivisible, so a
	//    group-aware holdout can only approximate a requested row count; the
	//    report states both rather than implying the request was honored.
	unsigned nRequested = 0, nAchieved = 0;

	unsigned leakage = 0;              // rows/groups found on both sides (must be 0)
	vector< string > warnings;
	string refusal;                    // why the partition could not be built at all
};

// The fold plan / holdout as one line of prose, derived from the fields above.
//    These are the ONLY places that wording is composed.
string describeFolds( const Partition& p );
string describeHoldout( const Partition& p );

// The inference policy: which estimator a declared sampling unit and an achieved
//    partition method permit. ONE function, so the rule can be unit-tested and
//    cannot drift between the report, the JSON, and the handler.
//
//    estimable / unestimableReason are the ESTIMATOR's own verdict on the data
//    it was given (too few of a class, a non-finite score); the policy folds it
//    in so a caller never has to combine "was it permitted" with "was it
//    computable" itself and get the precedence wrong.
struct InferenceChoice {
	AucInference method = AucInference::None;
	string reason;   // why None; empty when a method was chosen
};
InferenceChoice chooseInference( SamplingUnit unit, PartitionMethod split,
	bool estimable, const string& unestimableReason );

} // namespace evaldesign

#endif
