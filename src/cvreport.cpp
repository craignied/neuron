/* Cross-validation evaluation report -- see cvreport.h and
   docs/evaluation_report_spec.md. */

#include "cvreport.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>

#include "version.h"

using namespace std;

// --- derived design text (DLG-8) --------------------------------------------
//
// The report DESCRIBES a design; it does not author one. Every line below is a
//    projection of the structured fields, composed in evaldesign so the wording
//    has exactly one source (rule 6). A caller can no longer hand the report a
//    sentence that contradicts what it ran.

string cvreport::PlanInfo::foldPlanText() const
{
	return folds.k ? evaldesign::describeFolds( folds ) : string();
}

string cvreport::LockedInfo::splitPlanText() const
{
	return evaldesign::describeHoldout( split );
}

string cvreport::LockedInfo::samplingUnitText() const
{
	return evaldesign::samplingUnitName( unit );
}

string cvreport::LockedInfo::independenceText() const
{
	return evaldesign::independenceStatus( unit );
}

string cvreport::LockedInfo::inferenceText() const
{
	return evaldesign::inferenceText( inference, inferenceReason );
}

string cvreport::LockedInfo::clusterError() const
{
	if ( cluster.empty() )
		return nClusters ? "cluster count reported without per-row cluster ids"
			: string();

	if ( cluster.size() != testRows.size() )
		return "cluster ids (" + to_string( cluster.size() ) + ") do not pair with "
			"the locked-test rows (" + to_string( testRows.size() ) + ")";

	set< unsigned > seen;
	for ( unsigned i = 0; i < cluster.size(); i++ )
	{
		if ( nClusters && cluster[ i ] >= nClusters )
			return "cluster id " + to_string( cluster[ i ] ) + " is out of range for "
				+ to_string( nClusters ) + " clusters";
		seen.insert( cluster[ i ] );
	}
	// The locked sample need not contain every group in the dataset, but it can
	//    never contain MORE distinct ids than the design says exist.
	if ( nClusters && seen.size() > nClusters )
		return "more distinct cluster ids than the declared cluster count";
	if ( !nClusters )
		return "per-row cluster ids without a cluster count";

	return "";
}

namespace {

// --- small text helpers -----------------------------------------------------

// Display width in terminal columns: count bytes that begin a UTF-8 code point
//    (skip continuation bytes 10xxxxxx), so "±" and the box-drawing rules count
//    as one column each rather than their byte length -- keeps columns aligned.
unsigned displayWidth( const string& s )
{
	unsigned w = 0;
	for ( unsigned i = 0; i < s.size(); i++ )
		if ( ( ( unsigned char ) s[ i ] & 0xC0 ) != 0x80 ) w++;
	return w;
}

string padRight( const string& s, unsigned width )
{
	unsigned w = displayWidth( s );
	return w >= width ? s : s + string( width - w, ' ' );
}

string rule( unsigned cols, const char* glyph )
{
	string r;
	for ( unsigned i = 0; i < cols; i++ ) r += glyph;
	return r;
}

string fixed3( double x )
{
	ostringstream os;
	os << setiosflags( ios::fixed ) << setprecision( 3 ) << x;
	return os.str();
}

string fmtTime( double seconds )
{
	ostringstream os;
	os << setiosflags( ios::fixed )
		<< setprecision( seconds < 10 ? 1 : 0 ) << seconds << " s";
	return os.str();
}

// The rows the FOLD PLAN actually covered, and how many of them carried the
//    outcome. Read from the Comparison -- the object that knows which rows it
//    folded -- and NEVER from PlanInfo, whose n/events describe the whole dataset
//    for the Tier-1 summary line.
//
//    With a locked test those are two different sets: CV folds the development
//    rows while the dataset totals include the locked rows too. Taking the
//    fold-plan counts from PlanInfo printed "(development rows only) ... n = 6000"
//    directly above five folds of 900 rows each -- a header contradicting the
//    table beneath it, and the wrong number to check the fold sizes against
//    (2026-07-29). Deriving them here makes that contradiction impossible rather
//    than merely correcting the one caller that got it wrong.
void planCounts( const crossval::Comparison& cmp, unsigned& n, unsigned& events )
{
	n = ( unsigned ) cmp.outcome.size();
	events = 0;
	for ( unsigned r = 0; r < cmp.outcome.size(); r++ )
		if ( cmp.outcome[ r ] ) events++;
}

// --- procedure-level summaries (no model-family knowledge) -------------------

// Mean and sample sd of a procedure's per-fold (exact) AUCs, valid folds only.
struct CvAuc { double mean; double sd; unsigned valid; unsigned folds; };

CvAuc cvAuc( const crossval::RunResult& r )
{
	CvAuc a; a.mean = -1; a.sd = -1; a.valid = 0; a.folds = ( unsigned ) r.folds.size();
	vector< double > v;
	for ( unsigned i = 0; i < r.folds.size(); i++ )
		if ( r.folds[ i ].trap >= 0 ) v.push_back( r.folds[ i ].trap );
	a.valid = ( unsigned ) v.size();
	if ( v.empty() ) return a;

	double s = 0;
	for ( unsigned i = 0; i < v.size(); i++ ) s += v[ i ];
	a.mean = s / v.size();

	if ( v.size() >= 2 )
	{
		double ss = 0;
		for ( unsigned i = 0; i < v.size(); i++ )
			ss += ( v[ i ] - a.mean ) * ( v[ i ] - a.mean );
		a.sd = sqrt( ss / ( v.size() - 1 ) ); // descriptive spread across dependent folds
	}
	else a.sd = 0;
	return a;
}

// "0.881 ± 0.010", or "0.881 (m/k folds)" when some folds were degenerate.
string aucCell( const CvAuc& a )
{
	if ( a.valid == 0 ) return "n/a";
	string cell = fixed3( a.mean );
	if ( a.sd >= 0 && a.valid >= 2 ) cell += " \xC2\xB1 " + fixed3( a.sd ); // U+00B1
	if ( a.valid < a.folds )
	{
		ostringstream os;
		os << " (" << a.valid << "/" << a.folds << " folds)";
		cell += os.str();
	}
	return cell;
}

// Architecture metadata: modal hidden size + its frequency, and the range.
struct ArchInfo { bool has; unsigned modal, modalCount, total, lo, hi; };

ArchInfo archInfo( const vector< crossval::FoldSelection >& sel )
{
	ArchInfo a; a.has = !sel.empty();
	a.modal = a.modalCount = 0; a.total = ( unsigned ) sel.size(); a.lo = a.hi = 0;
	if ( sel.empty() ) return a;

	map< unsigned, unsigned > freq;
	a.lo = a.hi = sel[ 0 ].hidden;
	for ( unsigned i = 0; i < sel.size(); i++ )
	{
		freq[ sel[ i ].hidden ]++;
		if ( sel[ i ].hidden < a.lo ) a.lo = sel[ i ].hidden;
		if ( sel[ i ].hidden > a.hi ) a.hi = sel[ i ].hidden;
	}
	for ( map< unsigned, unsigned >::const_iterator it = freq.begin();
		it != freq.end(); it++ )
		if ( it->second > a.modalCount ) { a.modalCount = it->second; a.modal = it->first; }
	return a;
}

// The optimizer names the report uses, matching the CLI/GUI menu terminology.
const char* algorithmName( int a )
{
	return a == 0 ? "Canonical" : a == 1 ? "CGD" : a == 2 ? "Shanno" : "unknown";
}

// "CGD 3/5 folds, Shanno 2/5" -- how often each optimizer was chosen, in
//    fold order of first appearance. Empty when nothing recorded an optimizer.
string optimizerSummary( const vector< crossval::FoldSelection >& sel )
{
	map< int, unsigned > freq;
	unsigned total = 0;
	for ( unsigned i = 0; i < sel.size(); i++ )
		if ( sel[ i ].algorithm >= 0 ) { freq[ sel[ i ].algorithm ]++; total++; }
	if ( !total ) return "";

	string out;
	for ( map< int, unsigned >::const_iterator it = freq.begin();
		it != freq.end(); it++ )
	{
		if ( !out.empty() ) out += ", ";
		out += string( algorithmName( it->first ) ) + " "
			+ to_string( it->second ) + "/" + to_string( total );
	}
	return out;
}

string archCell( const ArchInfo& a )
{
	if ( !a.has ) return "\xE2\x80\x94"; // em dash
	ostringstream os;
	os << a.modal << " (" << a.modalCount << "/" << a.total << ")";
	return os.str();
}

// Rows in a given fold (indices into the per-row vectors).
vector< unsigned > rowsInFold( const vector< unsigned >& foldId, unsigned f )
{
	vector< unsigned > rows;
	for ( unsigned r = 0; r < foldId.size(); r++ )
		if ( foldId[ r ] == f ) rows.push_back( r );
	return rows;
}

// CSV field: quote when it holds a comma or a quote (procedure names may).
string csv( const string& s )
{
	if ( s.find( ',' ) == string::npos && s.find( '"' ) == string::npos ) return s;
	string q = "\"";
	for ( unsigned i = 0; i < s.size(); i++ )
	{
		if ( s[ i ] == '"' ) q += '"';
		q += s[ i ];
	}
	return q + "\"";
}

string num( double x ) // a metric or -1 / non-finite for "not computable"
{
	if ( !( x >= 0 ) || !isfinite( x ) ) return ""; // NaN/inf/negative -> empty (never "nan")
	ostringstream os;
	os << setiosflags( ios::fixed ) << setprecision( 6 ) << x;
	return os.str();
}

// JSON string escaping -- PlanInfo/procedure names are general class-layer
//    inputs, so quotes/backslashes/control chars must not break cv_run.json (B13).
string jsonStr( const string& s )
{
	string out = "\"";
	for ( unsigned i = 0; i < s.size(); i++ )
	{
		unsigned char c = ( unsigned char ) s[ i ];
		if ( c == '"' ) out += "\\\"";
		else if ( c == '\\' ) out += "\\\\";
		else if ( c == '\n' ) out += "\\n";
		else if ( c == '\r' ) out += "\\r";
		else if ( c == '\t' ) out += "\\t";
		else if ( c < 0x20 ) { char b[ 8 ]; snprintf( b, sizeof b, "\\u%04x", c ); out += b; }
		else out += ( char ) c;
	}
	return out + "\"";
}

string jnumOrNull( double x ) // a finite non-negative number, or JSON null
{
	string s = num( x );
	return s.empty() ? "null" : s;
}

string uintArray( const vector< unsigned >& v )
{
	string s = "[";
	for ( unsigned i = 0; i < v.size(); i++ )
		s += ( i ? ", " : "" ) + to_string( v[ i ] );
	return s + "]";
}

// The structured partition design, machine-readable (DLG-8). A consumer can now
//    read WHAT was done -- method, keys, achieved counts, leakage -- instead of
//    parsing the human sentence beside it, which is the defect this replaces.
//    Optional parts appear as empty arrays / zeros, never as absent keys, so a
//    reader never has to distinguish "not applicable" from "forgot to write it".
string partitionJson( const evaldesign::Partition& p )
{
	string s = "{ \"method\": "
		+ jsonStr( evaldesign::partitionMethodName( p.method ) )
		+ ", \"seed\": " + to_string( p.seed )
		+ ", \"k\": " + to_string( p.k )
		+ ", \"strataColumns\": " + uintArray( p.strataColumns )
		+ ", \"strataBins\": " + to_string( p.strataBins )
		+ ", \"groupColumns\": " + uintArray( p.groupColumns )
		+ ", \"groups\": " + to_string( p.nGroups )
		+ ", \"developmentOnly\": " + ( p.developmentOnly ? "true" : "false" )
		+ ", \"requested\": " + to_string( p.nRequested )
		+ ", \"achieved\": " + to_string( p.nAchieved )
		+ ", \"leakage\": " + to_string( p.leakage )
		+ ", \"refusal\": " + jsonStr( p.refusal )
		+ ", \"warnings\": [";
	for ( unsigned i = 0; i < p.warnings.size(); i++ )
		s += ( i ? ", " : "" ) + jsonStr( p.warnings[ i ] );
	return s + "] }";
}

// --- locked-test helpers ----------------------------------------------------

// The locked column for a procedure, matched by NAME (column order is free).
const cvreport::LockedColumn* findCol(
	const cvreport::LockedInfo& L, const string& name )
{
	for ( unsigned i = 0; i < L.columns.size(); i++ )
		if ( L.columns[ i ].name == name ) return &L.columns[ i ];
	return nullptr;
}

// "0.874 [0.835-0.914]" with a CI, "0.874" (point AUC only, no inference), a note,
//    or an em-dash. The CI appears only when inference was declared (hasCi).
string testCell( const cvreport::LockedColumn* c )
{
	if ( !c ) return "\xE2\x80\x94"; // em dash
	if ( c->hasAuc )
	{
		string s = fixed3( c->auc );
		if ( c->hasCi )
			s += " [" + fixed3( c->lo ) + "\xE2\x80\x93" + fixed3( c->hi ) + "]";
		return s;
	}
	// Tier 1 is a ONE-SCREEN table, and a failure reason can run to several
	//    sentences (a ceiling refusal names the trial and the remedy). The
	//    headline keeps its columns aligned and says only that the procedure
	//    failed; the full reason is printed in the Tier 2 locked-test section.
	if ( !c->note.empty() )
	{
		string n = c->note;
		if ( displayWidth( n ) > 22 ) n = n.substr( 0, 19 ) + "...";
		return n;
	}
	return "\xE2\x80\x94";
}

} // namespace

// ---------------------------------------------------------------------------
// Tier 1 -- the one-screen headline summary
// ---------------------------------------------------------------------------
string cvreport::tier1( const crossval::Comparison& cmp, const PlanInfo& info,
	const LockedInfo& locked )
{
	const bool L = locked.has; // a locked-test column + DeLong verdict are present
	const unsigned COLS = L ? 100 : 76;
	// Column layout (display columns): name 16, AUC(CV) 20, [AUC(test) 24], Arch 14.
	const unsigned wName = 16, wAuc = 20, wTest = L ? 24 : 0, wArch = 14;

	ostringstream os;
	os << rule( COLS, "\xE2\x95\x90" ) << "\n"; // heavy rule

	os << " SUMMARY \xE2\x80\x94 " << cmp.k << "-fold cross-validation";
	if ( L ) os << " + locked test";
	if ( info.n ) os << " \xC2\xB7 " << info.n << " exemplars";
	if ( info.events && info.n )
	{
		ostringstream pct;
		pct << setiosflags( ios::fixed ) << setprecision( 2 )
			<< ( 100.0 * info.events / info.n );
		os << " \xC2\xB7 " << info.events << " events (" << pct.str() << "%)";
	}
	os << "\n";
	if ( !info.foldPlanText().empty() ) os << " Folds: " << info.foldPlanText() << "\n";
	if ( L )
	{
		os << " Locked test: " << locked.n << " rows";
		if ( locked.n ) os << " (" << locked.events << " events)";
		if ( !locked.splitPlanText().empty() ) os << "; " << locked.splitPlanText();
		os << "\n";
	}
	os << rule( COLS, "\xE2\x95\x90" ) << "\n";

	os << " " << padRight( "Procedure", wName ) << padRight( "AUC (CV)", wAuc );
	// The CI only appears when inference was declared; otherwise the column is the
	//    point AUC alone (DLG-1).
	if ( L ) os << padRight( locked.inferenceRan() ? "AUC (test) [95% CI]"
		: "AUC (test)", wTest );
	os << padRight( "Arch", wArch ) << "Time" << "\n";
	os << " " << rule( COLS - 1, "\xE2\x94\x80" ) << "\n";

	for ( unsigned i = 0; i < cmp.entries.size(); i++ )
	{
		const crossval::Comparison::Entry& e = cmp.entries[ i ];
		os << " " << padRight( e.name, wName )
			<< padRight( aucCell( cvAuc( e.result ) ), wAuc );
		if ( L ) os << padRight( testCell( findCol( locked, e.name ) ), wTest );
		os << padRight( archCell( archInfo( e.selections ) ), wArch )
			<< fmtTime( e.seconds ) << "\n";
	}
	os << " " << rule( COLS - 1, "\xE2\x94\x80" ) << "\n";

	// Verdict block. With a locked test AND a declared sampling unit, the
	//    prespecified DeLong contrast IS the inference. With a locked test but no
	//    declaration, the point difference is descriptive and inference is withheld.
	//    Without a locked test, a descriptive CV contrast (no p), never inference.
	if ( L && locked.contrast.hasInference )
	{
		const LockedContrast& c = locked.contrast;
		os << " Primary contrast (prespecified): " << c.primary
			<< " \xE2\x88\x92 " << c.reference << "\n";
		// The estimator NAMES ITSELF here: a clustered p must never be printed as
		//    "DeLong p" (they are different estimators with different assumptions).
		const string method = evaldesign::inferenceShortName( locked.inference );
		if ( c.degenerate )
			os << "   Locked test: equal areas \xE2\x80\x94 no testable difference.\n";
		else if ( c.separated )
			os << "   Locked test: \xCE\x94" << "AUC = "
				<< ( c.delta >= 0 ? "+" : "" ) << fixed3( c.delta )
				<< ", deterministic separation (" << method
				<< " p \xE2\x89\x88 0)  \xE2\x86\x92  significant.\n";
		else
		{
			ostringstream pv;
			pv << setiosflags( ios::fixed ) << setprecision( 3 ) << c.p;
			os << "   Locked test: \xCE\x94" << "AUC = "
				<< ( c.delta >= 0 ? "+" : "" ) << fixed3( c.delta )
				<< ", " << method << " p = " << pv.str() << "  \xE2\x86\x92  "
				<< ( c.significant ? "significant" : "not significant" ) << "\n";
		}
	}
	else if ( L && locked.contrast.hasDelta )
	{
		const LockedContrast& c = locked.contrast;
		os << " Primary contrast (prespecified): " << c.primary
			<< " \xE2\x88\x92 " << c.reference << "\n";
		os << "   \xCE\x94" << "AUC (point) = " << ( c.delta >= 0 ? "+" : "" )
			<< fixed3( c.delta ) << " \xE2\x80\x94 inference unavailable ("
			<< ( c.note.empty() ? "sampling unit not declared" : c.note ) << ").\n";
	}
	else if ( L && !locked.contrast.note.empty() )
		os << " Primary contrast: " << locked.contrast.note << "\n";
	else if ( !L && !info.primary.empty() && !info.reference.empty() )
	{
		double mp = -1, mr = -1;
		for ( unsigned i = 0; i < cmp.entries.size(); i++ )
		{
			if ( cmp.entries[ i ].name == info.primary )
				mp = cvAuc( cmp.entries[ i ].result ).mean;
			if ( cmp.entries[ i ].name == info.reference )
				mr = cvAuc( cmp.entries[ i ].result ).mean;
		}
		if ( mp >= 0 && mr >= 0 )
		{
			double d = mp - mr;
			os << " Primary contrast (descriptive, CV): " << info.primary
				<< " \xE2\x88\x92 " << info.reference << "  \xCE\x94" << "AUC = "
				<< ( d >= 0 ? "+" : "" ) << fixed3( d ) << "\n";
			os << "   no locked-test inference (CV policy) \xE2\x80\x94 "
				"see the descriptive spread above.\n";
		}
	}

	// Architecture footnote(s).
	for ( unsigned i = 0; i < cmp.entries.size(); i++ )
	{
		ArchInfo a = archInfo( cmp.entries[ i ].selections );
		if ( !a.has ) continue;
		os << " * " << cmp.entries[ i ].name << ": OBD selected " << a.modal
			<< " hidden in " << a.modalCount << "/" << a.total << " folds (range "
			<< a.lo << "\xE2\x80\x93" << a.hi << ").\n";

		// The optimizer each fold ran on, beside the architecture -- never
		//    instead of it (both are selection metadata and both are reported).
		string opt = optimizerSummary( cmp.entries[ i ].selections );
		if ( !opt.empty() )
			os << "   optimizer: " << opt << ".\n";
	}

	// The one standing caveat, always. When inference was withheld the caveat
	//    states the ACTUAL reason (DLG-8): it used to assert "the sampling unit was
	//    not declared independent" even when the unit HAD been declared and the
	//    locked test was merely too sparse to estimate a covariance.
	if ( L && locked.inferenceRan() )
		os << " CV \xC2\xB1 is descriptive spread across dependent folds, not a CI; "
			"the inferential\n comparison is on the locked test set ("
			<< evaldesign::inferenceShortName( locked.inference )
			<< ", which assumes the declared\n "
			<< evaldesign::samplingUnitPhrase( locked.unit ) << " sampling unit).\n";
	else if ( L )
		os << " CV \xC2\xB1 is descriptive spread across dependent folds, not a CI; "
			"AUC inference is\n withheld -- " << locked.inferenceReason
			<< ".\n Point AUCs above are descriptive.\n";
	else
		os << " CV \xC2\xB1 is descriptive spread across dependent folds, not a CI; "
			"this run performs no\n inferential comparison -- locked-test inference is "
			"a separate policy.\n";
	os << rule( COLS, "\xE2\x95\x90" ) << "\n";
	return os.str();
}

// ---------------------------------------------------------------------------
// Tier 2 -- descriptive per-fold detail
// ---------------------------------------------------------------------------
string cvreport::tier2( const crossval::Comparison& cmp, const PlanInfo& info,
	const LockedInfo& locked )
{
	ostringstream os;
	os << "Cross-validation detail\n";
	// The fold plan's own counts, from the rows it covered (see planCounts) --
	//    not the dataset totals, which with a locked test describe a larger set.
	unsigned planN = 0, planEvents = 0;
	planCounts( cmp, planN, planEvents );
	os << "Fold plan: "
		<< ( info.foldPlanText().empty() ? "(unspecified)" : info.foldPlanText() )
		<< "   k = " << cmp.k;
	if ( planN ) os << "   n = " << planN << "   events = " << planEvents;
	os << "\n";

	for ( unsigned p = 0; p < cmp.entries.size(); p++ )
	{
		const crossval::Comparison::Entry& e = cmp.entries[ p ];
		os << "\n" << e.name << "  (" << fmtTime( e.seconds ) << ")\n";
		os << "  fold      n      AUC      sens      spec\n";

		unsigned degenerate = 0;
		vector< string > failed; // fold-failure reasons (a fit that did not run)
		for ( unsigned f = 0; f < cmp.k; f++ )
		{
			// The runner already recorded per-fold status; a FAILED fold has no
			//    predictions, so its metrics must NOT be recomputed from absent data.
			const crossval::FoldResult* fr = nullptr;
			for ( unsigned i = 0; i < e.result.folds.size(); i++ )
				if ( e.result.folds[ i ].fold == f ) { fr = &e.result.folds[ i ]; break; }

			if ( fr && !fr->ok )
			{
				os << "  " << setw( 4 ) << f << setw( 8 ) << fr->nHeldout
					<< "   " << padRight( "failed", 8 ) << " " << padRight( "-", 9 )
					<< "-\n";
				ostringstream fr_s; fr_s << "fold " << f << ": " << fr->reason;
				failed.push_back( fr_s.str() );
				continue;
			}

			vector< unsigned > rows = rowsInFold( cmp.foldId, f );
			crossval::Metrics m = crossval::metricsFor(
				e.result.outcome, e.result.oofPrediction, rows );
			if ( m.trap < 0 ) degenerate++;
			os << "  " << setw( 4 ) << f << setw( 8 ) << m.n
				<< "   " << padRight( m.trap < 0 ? "n/a" : fixed3( m.trap ), 8 )
				<< " " << padRight( m.sens < 0 ? "n/a" : fixed3( m.sens ), 9 )
				<< ( m.spec < 0 ? "n/a" : fixed3( m.spec ) ) << "\n";
		}
		os << "  pooled OOF AUC (exact) = "
			<< ( e.result.oofTrap < 0 ? "n/a" : fixed3( e.result.oofTrap ) )
			<< ", binormal = "
			<< ( e.result.oofAz < 0 ? "n/a" : fixed3( e.result.oofAz ) )
			<< "   (" << e.result.validFolds << "/" << cmp.k << " folds fitted)\n";
		for ( unsigned i = 0; i < failed.size(); i++ )
			os << "  FAILED " << failed[ i ] << "\n";
		if ( degenerate )
			os << "  note: " << degenerate << " fitted fold(s) had a degenerate "
				"held-out set (one class), so no AUC\n";

		ArchInfo a = archInfo( e.selections );
		if ( a.has )
		{
			map< unsigned, unsigned > freq;
			for ( unsigned i = 0; i < e.selections.size(); i++ )
				freq[ e.selections[ i ].hidden ]++;
			os << "  OBD architecture selection:";
			for ( map< unsigned, unsigned >::const_iterator it = freq.begin();
				it != freq.end(); it++ )
				os << " " << it->first << "\xE2\x86\x92" << it->second << " fold(s)";
			os << "\n";

			// Which optimizer each fold's search actually ran on. With Auto this
			//    is chosen independently per fold, so the spread is informative.
			string opt = optimizerSummary( e.selections );
			if ( !opt.empty() )
			{
				bool anyAuto = false;
				for ( unsigned i = 0; i < e.selections.size(); i++ )
					if ( e.selections[ i ].autoSelected ) anyAuto = true;
				os << "  Optimizer selection: " << opt
					<< ( anyAuto ? "  (Auto, chosen independently per fold)"
						: "  (fixed by request)" ) << "\n";
			}
		}
	}

	// Locked-test evaluation section (only when a locked test was run).
	if ( locked.has )
	{
		os << "\nLocked-test evaluation (each procedure refit on the development set "
			"by its\nown rule, scored once on the untouched locked test)\n";
		os << "Locked test: " << locked.n << " rows, " << locked.events << " events";
		if ( !locked.splitPlanText().empty() ) os << "   " << locked.splitPlanText();
		os << "\n";
		// Design metadata: what the inference assumes (DLG-1), derived from the
		//    structured design rather than repeated as prose (DLG-8). The cluster
		//    line appears only for a grouped design -- absence is meaningful.
		os << "  sampling unit: " << locked.samplingUnitText()
			<< "   independence: " << locked.independenceText() << "\n";
		if ( locked.nClusters )
			os << "  clusters in the locked sample: " << locked.nClusters << "\n";
		os << "  inference: " << locked.inferenceText() << "\n";
		os << "  procedure         AUC(test)   " << ( locked.inferenceRan() ? "95% CI"
			: "(no CI -- inference withheld)" ) << "\n";
		for ( unsigned i = 0; i < locked.columns.size(); i++ )
		{
			const LockedColumn& c = locked.columns[ i ];
			os << "  " << padRight( c.name, 16 ) << "  ";
			if ( c.hasAuc )
			{
				os << padRight( fixed3( c.auc ), 10 ) << "  ";
				if ( c.hasCi )
					os << padRight( "[" + fixed3( c.lo ) + ", " + fixed3( c.hi ) + "]", 20 );
				else
					os << padRight( "-", 20 );
				os << "  " << ( c.arch.empty() ? "-" : c.arch );
			}
			else
				os << ( c.note.empty() ? "n/a" : c.note );
			os << "\n";
		}
		if ( locked.contrast.hasDelta )
		{
			const LockedContrast& c = locked.contrast;
			os << "  Prespecified contrast: " << c.primary << " - " << c.reference
				<< "  (delta = AUC(primary) - AUC(reference))\n";
			os << "    delta AUC = " << ( c.delta >= 0 ? "+" : "" ) << fixed3( c.delta );
			if ( !c.hasInference )
				os << "  (point difference; inference withheld"
					<< ( c.note.empty() ? "" : " -- " + c.note ) << ")\n";
			else if ( c.degenerate )
				os << "  (equal areas: no testable difference)\n";
			else if ( c.separated )
				os << "  (deterministic separation, "
					<< evaldesign::inferenceShortName( locked.inference )
					<< " p ~ 0: significant)\n";
			else
			{
				ostringstream pv;
				pv << setiosflags( ios::fixed ) << setprecision( 4 ) << c.p;
				os << ", " << evaldesign::inferenceShortName( locked.inference )
					<< " two-sided p = " << pv.str()
					<< "  (" << ( c.significant ? "significant" : "not significant" )
					<< ")\n";
			}
		}
		else if ( !locked.contrast.note.empty() )
			os << "  Prespecified contrast: " << locked.contrast.note << "\n";
		// The standing scope note for whichever estimator this design permits.
		if ( locked.inference == evaldesign::AucInference::ObuchowskiClustered )
			os << "  Clustered inference treats the cluster, not the row, as the "
				"independent\n  sampling unit; the point areas remain patient-row "
				"Mann-Whitney areas.\n";
		else
			os << "  DeLong assumes independent test observations; it does not apply to "
				"clustered\n  test data (e.g. shared county) -- cluster-aware inference "
				"is a follow-on.\n";
	}
	return os.str();
}

// ---------------------------------------------------------------------------
// Tier 3 -- machine-readable artifacts (files, never printed)
// ---------------------------------------------------------------------------
// Write one artifact through body(), reporting success ONLY if the file opened,
//    wrote, flushed, and closed cleanly. Detects post-open failure (a full disk /
//    I/O error surfaces as a stream failbit/badbit on flush or close), not merely
//    an unwritable directory (B7). A stale/partial file from a failed write is
//    removed so no half-written artifact is mistaken for a good one.
static cvreport::ArtifactResult writeOne( const string& dir, const string& name,
	const function< void( ostream& ) >& body )
{
	cvreport::ArtifactResult r;
	r.name = name;
	r.path = ( dir.empty() ? string() : dir + "/" ) + name;

	ofstream f( r.path.c_str() );
	if ( !f.is_open() )
	{
		r.error = "could not open for writing";
		return r;
	}

	body( f );
	f.flush();   // force buffered data out so a disk-full error surfaces now
	f.close();   // close can also fail (deferred write-back); it sets failbit

	if ( f.good() ) { r.ok = true; return r; }

	r.error = "write/flush/close failed (disk full or I/O error?)";
	remove( r.path.c_str() ); // drop the partial file
	return r;
}

vector< cvreport::ArtifactResult > cvreport::writeArtifacts(
	const crossval::Comparison& cmp, const PlanInfo& info, const string& dir,
	const LockedInfo& locked )
{
	// The fold plan's own counts. n and events are ONE fact about ONE set of rows,
	//    so they come from the same place (see planCounts): cv_run.json's n was
	//    already the folded count while its events came from PlanInfo, which with a
	//    locked test meant a row count and an event count describing different sets.
	unsigned n = 0, planEvents = 0;
	planCounts( cmp, n, planEvents );
	vector< ArtifactResult > results;

	// cv_predictions.csv -- one row per exemplar, one column per procedure.
	results.push_back( writeOne( dir, "cv_predictions.csv", [&]( ostream& f )
	{
		f << "exemplar,outcome,fold";
		for ( unsigned p = 0; p < cmp.entries.size(); p++ )
			f << "," << csv( cmp.entries[ p ].name );
		f << "\n";
		for ( unsigned r = 0; r < n; r++ )
		{
			f << r << "," << cmp.outcome[ r ] << ","
				<< ( r < cmp.foldId.size() ? cmp.foldId[ r ] : 0 );
			for ( unsigned p = 0; p < cmp.entries.size(); p++ )
				f << "," << num( cmp.entries[ p ].result.oofPrediction[ r ] );
			f << "\n";
		}
	} ) );

	// cv_metrics.csv -- fold x procedure metrics, plus a pooled row per procedure.
	results.push_back( writeOne( dir, "cv_metrics.csv", [&]( ostream& f )
	{
		// n_valid = rows with a computed metric; n_total = rows the row is ABOUT.
		//    They differ only on the pooled row after a fold failed (DLG-6): every
		//    row is then self-describing (status + both denominators), never claiming
		//    more observations than were actually used.
		f << "fold,procedure,status,n_valid,n_total,auc_trap,auc_binormal,sens,spec\n";
		for ( unsigned p = 0; p < cmp.entries.size(); p++ )
		{
			const crossval::Comparison::Entry& e = cmp.entries[ p ];
			for ( unsigned fold = 0; fold < cmp.k; fold++ )
			{
				const crossval::FoldResult* fr = nullptr;
				for ( unsigned i = 0; i < e.result.folds.size(); i++ )
					if ( e.result.folds[ i ].fold == fold )
						{ fr = &e.result.folds[ i ]; break; }

				if ( fr && !fr->ok ) // a failed fold: reason, no metrics, 0 valid
				{
					f << fold << "," << csv( e.name ) << "," << csv( fr->reason )
						<< ",0," << fr->nHeldout << ",,,,\n";
					continue;
				}
				vector< unsigned > rows = rowsInFold( cmp.foldId, fold );
				crossval::Metrics m = crossval::metricsFor(
					e.result.outcome, e.result.oofPrediction, rows );
				f << fold << "," << csv( e.name ) << ",ok," << m.n << "," << m.n << ","
					<< num( m.trap ) << "," << num( m.az ) << ","
					<< num( m.sens ) << "," << num( m.spec ) << "\n";
			}
			// The pooled denominators: n_valid is the rows that actually got an
			//    out-of-fold prediction; n_total is the whole dataset. After a failed
			//    fold they differ and status is 'partial'; on a clean run they are
			//    equal and status is 'ok' (DLG-6).
			unsigned pooledN = e.result.pooledN;
			const char* pooledStatus = ( pooledN < n ) ? "partial" : "ok";
			f << "pooled," << csv( e.name ) << "," << pooledStatus << "," << pooledN
				<< "," << n << "," << num( e.result.oofTrap ) << ","
				<< num( e.result.oofAz ) << ",,\n";
		}
	} ) );

	// cv_run.json -- the fold plan, procedures, timings, arch, version.
	results.push_back( writeOne( dir, "cv_run.json", [&]( ostream& f )
	{
		f << "{\n";
		f << "  \"software\": " << jsonStr( NEURON_PACKAGE_STRING ) << ",\n";
		f << "  \"n\": " << n << ",\n";
		f << "  \"events\": " << planEvents << ",\n";
		f << "  \"k\": " << cmp.k << ",\n";
		f << "  \"foldPlan\": " << jsonStr( info.foldPlanText() ) << ",\n";
		f << "  \"foldDesign\": " << partitionJson( info.folds ) << ",\n";
		f << "  \"procedures\": [\n";
		for ( unsigned p = 0; p < cmp.entries.size(); p++ )
		{
			const crossval::Comparison::Entry& e = cmp.entries[ p ];
			f << "    { \"name\": " << jsonStr( e.name )
				<< ", \"seconds\": " << fixed3( e.seconds )
				<< ", \"validFolds\": " << e.result.validFolds
				<< ", \"pooledAUC\": " << jnumOrNull( e.result.oofTrap )
				<< ", \"arch\": [";
			for ( unsigned i = 0; i < e.selections.size(); i++ )
				f << ( i ? ", " : "" ) << e.selections[ i ].hidden;
			// Per-fold optimizer, paired positionally with "arch" above: both come
			//    from the same per-fold record, so a fold appears in both or neither.
			f << "], \"optimizer\": [";
			for ( unsigned i = 0; i < e.selections.size(); i++ )
				f << ( i ? ", " : "" )
					<< jsonStr( algorithmName( e.selections[ i ].algorithm ) );
			f << "], \"optimizerAuto\": [";
			for ( unsigned i = 0; i < e.selections.size(); i++ )
				f << ( i ? ", " : "" )
					<< ( e.selections[ i ].autoSelected ? "true" : "false" );
			f << "], \"failures\": [";
			bool firstFail = true;
			for ( unsigned i = 0; i < e.result.folds.size(); i++ )
				if ( !e.result.folds[ i ].ok )
				{
					f << ( firstFail ? "" : ", " ) << "{ \"fold\": "
						<< e.result.folds[ i ].fold << ", \"reason\": "
						<< jsonStr( e.result.folds[ i ].reason ) << " }";
					firstFail = false;
				}
			f << "] }" << ( p + 1 < cmp.entries.size() ? "," : "" ) << "\n";
		}
		f << "  ]";
		// The locked-test inference block, only when a locked test was run. Omitted
		//    entirely otherwise, so a pure-CV cv_run.json is byte-identical to before.
		if ( locked.has )
		{
			f << ",\n  \"lockedTest\": {\n";
			f << "    \"n\": " << locked.n << ", \"events\": " << locked.events << ",\n";
			f << "    \"splitPlan\": " << jsonStr( locked.splitPlanText() ) << ",\n";
			// Structured design metadata -- ordinary DeLong assumes independent test
			//    observations; these say whether that was DECLARED and what inference
			//    (if any) ran, replacing the old hardcoded independenceAssumed=true.
			//    The prose fields stay for compatibility; splitDesign is the machine
			//    form, and inferenceReason says WHY when no estimator ran (DLG-8).
			f << "    \"samplingUnit\": " << jsonStr( locked.samplingUnitText() ) << ",\n";
			f << "    \"independenceStatus\": " << jsonStr( locked.independenceText() ) << ",\n";
			f << "    \"inferenceMethod\": " << jsonStr( locked.inferenceText() ) << ",\n";
			f << "    \"inferenceRan\": " << ( locked.inferenceRan() ? "true" : "false" ) << ",\n";
			f << "    \"inferenceReason\": " << jsonStr( locked.inferenceReason ) << ",\n";
			f << "    \"clusters\": " << locked.nClusters << ",\n";
			f << "    \"splitDesign\": " << partitionJson( locked.split ) << ",\n";
			f << "    \"areas\": [";
			for ( unsigned i = 0; i < locked.columns.size(); i++ )
			{
				const LockedColumn& c = locked.columns[ i ];
				f << ( i ? ", " : "" ) << "{ \"name\": " << jsonStr( c.name )
					<< ", \"auc\": " << ( c.hasAuc ? jnumOrNull( c.auc ) : string( "null" ) )
					<< ", \"lo\": " << ( c.hasCi ? jnumOrNull( c.lo ) : string( "null" ) )
					<< ", \"hi\": " << ( c.hasCi ? jnumOrNull( c.hi ) : string( "null" ) )
					<< ", \"note\": " << jsonStr( c.note ) << " }";
			}
			f << "]";
			if ( locked.contrast.hasDelta )
			{
				const LockedContrast& c = locked.contrast;
				f << ",\n    \"contrast\": { \"primary\": " << jsonStr( c.primary )
					<< ", \"reference\": " << jsonStr( c.reference )
					<< ", \"delta\": " << c.delta // signed; not the num()/-1 convention
					<< ", \"inferenceRan\": " << ( c.hasInference ? "true" : "false" )
					<< ", \"p\": " << ( ( c.hasInference && !c.degenerate ) ? jnumOrNull( c.p )
						: string( "null" ) )
					<< ", \"significant\": " << ( ( c.hasInference && c.significant )
						? "true" : "false" )
					<< ", \"degenerate\": " << ( c.degenerate ? "true" : "false" )
					<< ", \"separated\": " << ( c.separated ? "true" : "false" )
					<< ", \"note\": " << jsonStr( c.note ) << " }";
			}
			else if ( !locked.contrast.note.empty() )
				f << ",\n    \"contrast\": { \"note\": "
					<< jsonStr( locked.contrast.note ) << " }";
			f << "\n  }";
		}
		f << "\n}\n";
	} ) );

	// cv_locked_predictions.csv -- one row per locked-test exemplar: raw row id,
	//    true outcome, one prediction column per procedure. Row identity is
	//    preserved so the pairing is externally auditable (Sol's caution). Written
	//    only when a locked test was run, and through the SAME B7 machinery.
	if ( locked.has )
	{
		// The predictions are the audit/future-analysis substrate and must be
		//    written whenever they EXIST -- independent of whether an AUC/CI or
		//    contrast was estimable (DLG-4: the old gate on LockedColumn::has blanked
		//    perfectly valid predictions exactly when a one-class/sparse locked test
		//    made DeLong unavailable). First refuse on a structural length mismatch,
		//    rather than pad a short outcome/prediction column with a fabricated 0.
		bool consistent = ( locked.outcome.size() == locked.testRows.size() );
		for ( unsigned p = 0; p < locked.columns.size(); p++ )
			if ( !locked.columns[ p ].pred.empty()
				&& locked.columns[ p ].pred.size() != locked.testRows.size() )
				consistent = false;

		// Cluster identity is joined by POSITION like everything else here, so a
		//    length or id defect would silently re-label which patients are
		//    correlated with which -- refuse the whole file instead (DLG-8).
		const string clusterBad = locked.clusterError();
		const bool hasCluster = !locked.cluster.empty();

		if ( !consistent || !clusterBad.empty() )
		{
			ArtifactResult r;
			r.name = "cv_locked_predictions.csv";
			r.path = ( dir.empty() ? string() : dir + "/" ) + r.name;
			r.error = consistent ? clusterBad + " (not written)"
				: "locked-test rows, outcomes, and predictions have inconsistent "
					"lengths (not written)";
			results.push_back( r );
		}
		else
			results.push_back( writeOne( dir, "cv_locked_predictions.csv", [&]( ostream& f )
			{
				// The cluster column appears ONLY for a grouped design: an
				//    ungrouped run's file stays byte-identical to before.
				f << "row";
				if ( hasCluster ) f << ",cluster";
				f << ",outcome";
				for ( unsigned p = 0; p < locked.columns.size(); p++ )
					f << "," << csv( locked.columns[ p ].name );
				f << "\n";
				for ( unsigned i = 0; i < locked.testRows.size(); i++ )
				{
					f << locked.testRows[ i ];
					if ( hasCluster ) f << "," << locked.cluster[ i ];
					f << "," << locked.outcome[ i ];
					for ( unsigned p = 0; p < locked.columns.size(); p++ )
					{
						const LockedColumn& c = locked.columns[ p ];
						f << ",";
						// write every FINITE prediction that exists, regardless of has
						if ( i < c.pred.size() && isfinite( c.pred[ i ] ) )
						{
							ostringstream v;
							v << setiosflags( ios::fixed ) << setprecision( 6 ) << c.pred[ i ];
							f << v.str();
						}
					}
					f << "\n";
				}
			} ) );
	}

	return results;
}

// ---------------------------------------------------------------------------
// render -- Tier 2 detail, then the Tier 1 summary LAST; write Tier 3.
// ---------------------------------------------------------------------------
void cvreport::render( ostream& out, const crossval::Comparison& cmp,
	const PlanInfo& info, const string& dir, const LockedInfo& locked )
{
	out << tier2( cmp, info, locked ) << "\n";
	if ( !dir.empty() )
	{
		vector< ArtifactResult > files = writeArtifacts( cmp, info, dir, locked );
		unsigned okCount = 0;
		for ( unsigned i = 0; i < files.size(); i++ ) if ( files[ i ].ok ) okCount++;
		out << "Wrote " << okCount << " of " << files.size()
			<< " machine-readable file(s) to " << dir << "/.\n";
		for ( unsigned i = 0; i < files.size(); i++ )
			if ( !files[ i ].ok )
				out << "  WARNING: could not write " << files[ i ].name << " -- "
					<< files[ i ].error << "\n";
		out << "\n";
	}
	out << tier1( cmp, info, locked ); // the answer, last
}
