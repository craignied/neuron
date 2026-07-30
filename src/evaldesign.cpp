// The evaluation design vocabulary (ROADMAP 4, DLG-8). See evaldesign.h.

#include "evaldesign.h"

#include <cctype>

// The user's 1-based column numbers as "3, 7" -- the form they typed them in.
static string columnList( const vector< unsigned >& cols )
{
	string s;
	for ( unsigned i = 0; i < cols.size(); i++ )
	{
		if ( i ) s += ", ";
		s += to_string( cols[ i ] );
	}
	return s;
}

bool evaldesign::parseSamplingUnit( const string& text, SamplingUnit& out )
{
	string t = text;
	for ( unsigned i = 0; i < t.size(); i++ )
		t[ i ] = ( char ) tolower( ( unsigned char ) t[ i ] );

	if ( t.empty() ) { out = SamplingUnit::Unspecified; return true; }
	if ( t == "rows" ) { out = SamplingUnit::Row; return true; }
	if ( t == "cluster" ) { out = SamplingUnit::Cluster; return true; }
	return false;
}

string evaldesign::samplingUnitName( SamplingUnit u )
{
	switch ( u )
	{
		case SamplingUnit::Row: return "row (declared independent)";
		case SamplingUnit::Cluster: return "cluster (declared clustered)";
		default: return "unspecified";
	}
}

string evaldesign::samplingUnitPhrase( SamplingUnit u )
{
	switch ( u )
	{
		case SamplingUnit::Row: return "independent-row";
		case SamplingUnit::Cluster: return "cluster";
		default: return "undeclared";
	}
}

string evaldesign::independenceStatus( SamplingUnit u )
{
	switch ( u )
	{
		case SamplingUnit::Row: return "declared: independent rows";
		case SamplingUnit::Cluster: return "declared: clustered observations";
		default: return "not declared";
	}
}

string evaldesign::inferenceName( AucInference m )
{
	switch ( m )
	{
		case AucInference::DeLongIndependent: return "DeLong (ordinary, independent rows)";
		case AucInference::ObuchowskiClustered: return "Obuchowski (clustered ROC covariance)";
		default: return "none";
	}
}

string evaldesign::inferenceShortName( AucInference m )
{
	switch ( m )
	{
		case AucInference::DeLongIndependent: return "DeLong";
		case AucInference::ObuchowskiClustered: return "clustered ROC";
		default: return "";
	}
}

string evaldesign::inferenceText( AucInference m, const string& reason )
{
	if ( m != AucInference::None ) return inferenceName( m );
	return reason.empty() ? string( "none" ) : "none (" + reason + ")";
}

string evaldesign::partitionMethodName( PartitionMethod m )
{
	switch ( m )
	{
		case PartitionMethod::CovariateStratified: return "outcome x covariate-stratified";
		case PartitionMethod::StratifiedGroup: return "group-disjoint outcome-stratified";
		default: return "outcome-stratified";
	}
}

// What both describers append about the keys a partition was built from: the
//    columns the user named, so the plan can be read back against the request.
static string keyClause( const evaldesign::Partition& p )
{
	string s;
	if ( !p.strataColumns.empty() )
	{
		s += " on column" + string( p.strataColumns.size() == 1 ? "" : "s" ) + " "
			+ columnList( p.strataColumns );
		if ( p.strataBins ) s += " (" + to_string( p.strataBins ) + " bins)";
	}
	if ( !p.groupColumns.empty() )
	{
		s += " grouped by column" + string( p.groupColumns.size() == 1 ? "" : "s" ) + " "
			+ columnList( p.groupColumns );
		if ( p.nGroups ) s += " (" + to_string( p.nGroups ) + " groups)";
	}
	return s;
}

string evaldesign::describeFolds( const Partition& p )
{
	string s = partitionMethodName( p.method ) + " " + to_string( p.k ) + "-fold, seed "
		+ to_string( p.seed ) + keyClause( p );
	if ( p.developmentOnly ) s += " (development rows only)";
	return s;
}

string evaldesign::describeHoldout( const Partition& p )
{
	// A group-aware holdout is NOT a row holdout and must not be described as
	//    one: what it holds out is whole groups, and that is the estimand.
	string s = ( p.method == PartitionMethod::StratifiedGroup )
		? string( "group-disjoint locked holdout" )
		: partitionMethodName( p.method ) + " row holdout";
	s += ", seed " + to_string( p.seed ) + keyClause( p );

	// Only when the request could not be honored exactly -- which happens when
	//    groups are indivisible. Silence means requested == achieved.
	if ( p.nRequested && p.nAchieved != p.nRequested )
		s += "; " + to_string( p.nAchieved ) + " rows achieved of "
			+ to_string( p.nRequested ) + " requested";
	return s;
}

evaldesign::InferenceChoice evaldesign::chooseInference( SamplingUnit unit,
	PartitionMethod split, bool estimable, const string& unestimableReason )
{
	InferenceChoice ch;

	// Order matters. A design that forbids an estimator is refused BEFORE asking
	//    whether the data could support one, so a sparse clustered test never
	//    reports "too sparse" and thereby implies it would otherwise have run
	//    ordinary DeLong.
	if ( unit == SamplingUnit::Unspecified )
	{
		ch.reason = "sampling unit not declared independent";
		return ch;
	}

	if ( unit == SamplingUnit::Row )
	{
		// Group-aware splitting stops leakage; it does not make the held-out rows
		//    independent. Declaring independent rows over a group-disjoint design
		//    is a contradiction and is refused rather than quietly honored -- the
		//    metadata cannot repair an invalid p-value after it is presented.
		if ( split == PartitionMethod::StratifiedGroup )
		{
			ch.reason = "independent rows were declared over a group-disjoint design; "
				"grouping prevents leakage but does not make rows independent";
			return ch;
		}
		if ( !estimable )
		{
			ch.reason = "locked test too sparse: " + unestimableReason;
			return ch;
		}
		ch.method = AucInference::DeLongIndependent;
		return ch;
	}

	// SamplingUnit::Cluster. The clustered estimator is not built yet; until it
	//    is, this REFUSES rather than falling back to ordinary DeLong (a settled
	//    decision, not a gap to paper over).
	ch.reason = "clustered inference is a follow-on";
	return ch;
}
