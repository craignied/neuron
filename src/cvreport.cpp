/* Cross-validation evaluation report -- see cvreport.h and
   docs/evaluation_report_spec.md. */

#include "cvreport.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>

#include "version.h"

using namespace std;

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

ArchInfo archInfo( const vector< unsigned >& arch )
{
	ArchInfo a; a.has = !arch.empty();
	a.modal = a.modalCount = 0; a.total = ( unsigned ) arch.size(); a.lo = a.hi = 0;
	if ( arch.empty() ) return a;

	map< unsigned, unsigned > freq;
	a.lo = a.hi = arch[ 0 ];
	for ( unsigned i = 0; i < arch.size(); i++ )
	{
		freq[ arch[ i ] ]++;
		if ( arch[ i ] < a.lo ) a.lo = arch[ i ];
		if ( arch[ i ] > a.hi ) a.hi = arch[ i ];
	}
	for ( map< unsigned, unsigned >::const_iterator it = freq.begin();
		it != freq.end(); it++ )
		if ( it->second > a.modalCount ) { a.modalCount = it->second; a.modal = it->first; }
	return a;
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

} // namespace

// ---------------------------------------------------------------------------
// Tier 1 -- the one-screen headline summary
// ---------------------------------------------------------------------------
string cvreport::tier1( const crossval::Comparison& cmp, const PlanInfo& info )
{
	const unsigned COLS = 76;
	// Column layout (display columns): name 16, AUC 20, Arch 14, Time.
	const unsigned wName = 16, wAuc = 20, wArch = 14;

	ostringstream os;
	os << rule( COLS, "\xE2\x95\x90" ) << "\n"; // heavy rule

	os << " SUMMARY \xE2\x80\x94 " << cmp.k << "-fold cross-validation";
	if ( info.n ) os << " \xC2\xB7 " << info.n << " exemplars";
	if ( info.events && info.n )
	{
		ostringstream pct;
		pct << setiosflags( ios::fixed ) << setprecision( 2 )
			<< ( 100.0 * info.events / info.n );
		os << " \xC2\xB7 " << info.events << " events (" << pct.str() << "%)";
	}
	os << "\n";
	if ( !info.foldPlan.empty() ) os << " Folds: " << info.foldPlan << "\n";
	os << rule( COLS, "\xE2\x95\x90" ) << "\n";

	os << " " << padRight( "Procedure", wName ) << padRight( "AUC (CV)", wAuc )
		<< padRight( "Arch", wArch ) << "Time" << "\n";
	os << " " << rule( COLS - 1, "\xE2\x94\x80" ) << "\n";

	for ( unsigned i = 0; i < cmp.entries.size(); i++ )
	{
		const crossval::Comparison::Entry& e = cmp.entries[ i ];
		os << " " << padRight( e.name, wName )
			<< padRight( aucCell( cvAuc( e.result ) ), wAuc )
			<< padRight( archCell( archInfo( e.archHidden ) ), wArch )
			<< fmtTime( e.seconds ) << "\n";
	}
	os << " " << rule( COLS - 1, "\xE2\x94\x80" ) << "\n";

	// Verdict block. CV-only: a descriptive contrast (no p), never inference.
	if ( !info.primary.empty() && !info.reference.empty() )
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
		ArchInfo a = archInfo( cmp.entries[ i ].archHidden );
		if ( !a.has ) continue;
		os << " * " << cmp.entries[ i ].name << ": OBD selected " << a.modal
			<< " hidden in " << a.modalCount << "/" << a.total << " folds (range "
			<< a.lo << "\xE2\x80\x93" << a.hi << ").\n";
	}

	// The one standing caveat, always.
	os << " CV \xC2\xB1 is descriptive spread across dependent folds, not a CI; "
		"this run performs no\n inferential comparison -- locked-test inference is "
		"a separate policy.\n";
	os << rule( COLS, "\xE2\x95\x90" ) << "\n";
	return os.str();
}

// ---------------------------------------------------------------------------
// Tier 2 -- descriptive per-fold detail
// ---------------------------------------------------------------------------
string cvreport::tier2( const crossval::Comparison& cmp, const PlanInfo& info )
{
	ostringstream os;
	os << "Cross-validation detail\n";
	os << "Fold plan: " << ( info.foldPlan.empty() ? "(unspecified)" : info.foldPlan )
		<< "   k = " << cmp.k;
	if ( info.n ) os << "   n = " << info.n;
	if ( info.events ) os << "   events = " << info.events;
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

		ArchInfo a = archInfo( e.archHidden );
		if ( a.has )
		{
			map< unsigned, unsigned > freq;
			for ( unsigned i = 0; i < e.archHidden.size(); i++ ) freq[ e.archHidden[ i ] ]++;
			os << "  OBD architecture selection:";
			for ( map< unsigned, unsigned >::const_iterator it = freq.begin();
				it != freq.end(); it++ )
				os << " " << it->first << "\xE2\x86\x92" << it->second << " fold(s)";
			os << "\n";
		}
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
	const crossval::Comparison& cmp, const PlanInfo& info, const string& dir )
{
	unsigned n = ( unsigned ) cmp.outcome.size();
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
		f << "fold,procedure,status,n,auc_trap,auc_binormal,sens,spec\n";
		for ( unsigned p = 0; p < cmp.entries.size(); p++ )
		{
			const crossval::Comparison::Entry& e = cmp.entries[ p ];
			for ( unsigned fold = 0; fold < cmp.k; fold++ )
			{
				const crossval::FoldResult* fr = nullptr;
				for ( unsigned i = 0; i < e.result.folds.size(); i++ )
					if ( e.result.folds[ i ].fold == fold )
						{ fr = &e.result.folds[ i ]; break; }

				if ( fr && !fr->ok ) // a failed fold: reason, no metrics
				{
					f << fold << "," << csv( e.name ) << "," << csv( fr->reason )
						<< "," << fr->nHeldout << ",,,,\n";
					continue;
				}
				vector< unsigned > rows = rowsInFold( cmp.foldId, fold );
				crossval::Metrics m = crossval::metricsFor(
					e.result.outcome, e.result.oofPrediction, rows );
				f << fold << "," << csv( e.name ) << ",ok," << m.n << ","
					<< num( m.trap ) << "," << num( m.az ) << ","
					<< num( m.sens ) << "," << num( m.spec ) << "\n";
			}
			f << "pooled," << csv( e.name ) << ",ok," << n << ","
				<< num( e.result.oofTrap ) << "," << num( e.result.oofAz ) << ",,\n";
		}
	} ) );

	// cv_run.json -- the fold plan, procedures, timings, arch, version.
	results.push_back( writeOne( dir, "cv_run.json", [&]( ostream& f )
	{
		f << "{\n";
		f << "  \"software\": " << jsonStr( NEURON_PACKAGE_STRING ) << ",\n";
		f << "  \"n\": " << n << ",\n";
		f << "  \"events\": " << info.events << ",\n";
		f << "  \"k\": " << cmp.k << ",\n";
		f << "  \"foldPlan\": " << jsonStr( info.foldPlan ) << ",\n";
		f << "  \"procedures\": [\n";
		for ( unsigned p = 0; p < cmp.entries.size(); p++ )
		{
			const crossval::Comparison::Entry& e = cmp.entries[ p ];
			f << "    { \"name\": " << jsonStr( e.name )
				<< ", \"seconds\": " << fixed3( e.seconds )
				<< ", \"validFolds\": " << e.result.validFolds
				<< ", \"pooledAUC\": " << jnumOrNull( e.result.oofTrap )
				<< ", \"arch\": [";
			for ( unsigned i = 0; i < e.archHidden.size(); i++ )
				f << ( i ? ", " : "" ) << e.archHidden[ i ];
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
		f << "  ]\n}\n";
	} ) );

	return results;
}

// ---------------------------------------------------------------------------
// render -- Tier 2 detail, then the Tier 1 summary LAST; write Tier 3.
// ---------------------------------------------------------------------------
void cvreport::render( ostream& out, const crossval::Comparison& cmp,
	const PlanInfo& info, const string& dir )
{
	out << tier2( cmp, info ) << "\n";
	if ( !dir.empty() )
	{
		vector< ArtifactResult > files = writeArtifacts( cmp, info, dir );
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
	out << tier1( cmp, info ); // the answer, last
}
