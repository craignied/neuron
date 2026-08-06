// optimizer_probe.cpp : the wall-clock measuring end of the optimizer benchmark.
//
// Built by the normal build so it cannot rot, and deliberately NOT registered
// with add_test: it measures wall time, which is machine-dependent, and a timing
// assertion inside ctest is a flake generator. Running it is a deliberate act.
// tests/clustered/scale_probe.cpp is the precedent.
//
// The mechanics it depends on ARE tested, deterministically and without a single
// timing assertion, by check_optimizer_harness -- which includes the same
// harness.h this does. A measuring tool nobody checks is not evidence.
//
//   ./build/optimizer_probe --list
//   ./build/optimizer_probe --identity                 what binary is this?
//   ./build/optimizer_probe --all                      every pilot case, one row
//   ./build/optimizer_probe --case logistic-canonical
//   ./build/optimizer_probe --characterize --case simpleprop-canonical
//
// One JSON Lines row per arm on stdout; diagnostics on stderr, so a caller can
// redirect stdout straight into a results file. A refused, failed or throwing
// arm still emits its row, with usable=false and an explicit reason -- never
// silently omitted, because a candidate that fails on a workload is a result
// about that candidate.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "harness.h"

using namespace std;
using namespace optbench;

static void usage()
{
	fprintf( stderr,
		"usage: optimizer_probe [--list] [--identity] [--all|--pilot|--step0b]\n"
		"                       [--case NAME] [--rep N] [--rev REV]\n"
		"                       [--characterize [--ceiling N]]\n"
		"  --list          print every case name, its group and its axis\n"
		"  --identity      print this binary's revision/source identity and exit\n"
		"  --all           run every case once (pilot mechanics + Step 0B workloads)\n"
		"  --pilot         run only the Step 0A mechanics pilot\n"
		"  --step0b        run only the Step 0B workload matrix\n"
		"  --case NAME     run one named case (repeatable)\n"
		"  --workload KEY  characterize a WORKLOAD's canonical control by its\n"
		"                  endpoint-table key, whether or not it currently has\n"
		"                  declared arms (repeatable; needs --characterize)\n"
		"  --rep N         run each selected case N times (default 1)\n"
		"  --rev REV       revision string recorded in every row\n"
		"  --characterize  run a CANONICAL REFERENCE case out to its ceiling and\n"
		"                  report BOTH predeclared endpoints -- the strict floor and\n"
		"                  the practical plateau -- plus the proof that watching the\n"
		"                  held-out set did not change the fit. Refused on any\n"
		"                  non-canonical arm.\n" );
}

int main( int argc, char** argv )
{
	// The DEFAULT revision is the one compiled into this binary, alongside the
	//    dirty flag and the source identity. --rev annotates; it cannot make a
	//    stale binary claim a different source, because source_id and dirty are
	//    compiled in and are emitted regardless.
	string rev = OPTBENCH_GIT_REV;
	unsigned reps = 1;
	bool all = false, pilot = false, step0b = false;
	bool list = false, identity = false, wantCharacterize = false;
	unsigned ceiling = 0;   // 0 = the engine's own default (STRICT_CEILING)
	vector< string > names, workloads;

	for ( int i = 1; i < argc; i++ )
	{
		string a = argv[ i ];
		if ( a == "--list" ) list = true;
		else if ( a == "--identity" ) identity = true;
		else if ( a == "--characterize" ) wantCharacterize = true;
		else if ( a == "--all" ) all = true;
		else if ( a == "--pilot" ) pilot = true;
		else if ( a == "--step0b" ) step0b = true;
		else if ( a == "--case" && i + 1 < argc ) names.push_back( argv[ ++i ] );
		else if ( a == "--workload" && i + 1 < argc ) workloads.push_back( argv[ ++i ] );
		else if ( a == "--rep" && i + 1 < argc )
		{
			// Refused by name. atoi would accept "abc" as 0 and run nothing
			//    while reporting success -- the permissive-conversion defect the
			//    engine already removed from its HTTP surface (ROADMAP 4 B9).
			string v = argv[ ++i ];
			char* end = 0;
			long n = strtol( v.c_str(), &end, 10 );
			if ( end == v.c_str() || *end != '\0' || n < 1 || n > 100000 )
			{
				fprintf( stderr, "refused -- rep: '%s' is not an integer in [1,100000]\n",
					v.c_str() );
				return 2;
			}
			reps = ( unsigned ) n;
		}
		else if ( a == "--ceiling" && i + 1 < argc )
		{
			// CHARACTERIZATION ONLY, and it can only LOWER the safety limit --
			//    a shortened characterization that hits the limit reports "did
			//    not converge", which is exactly what should happen. It cannot
			//    change any arm's ceiling, which comes from the committed table.
			string v = argv[ ++i ];
			char* end = 0;
			long n = strtol( v.c_str(), &end, 10 );
			if ( end == v.c_str() || *end != '\0' || n < 1 || n > 100000000 )
			{
				fprintf( stderr, "refused -- ceiling: '%s' is not an integer in "
					"[1,100000000]\n", v.c_str() );
				return 2;
			}
			ceiling = ( unsigned ) n;
		}
		else if ( a == "--rev" && i + 1 < argc ) rev = argv[ ++i ];
		else
		{
			fprintf( stderr, "refused -- unknown argument '%s'\n", a.c_str() );
			usage();
			return 2;
		}
	}

	if ( identity )
	{
		printf( "{\"schema\":%u,\"rev\":\"%s\",\"dirty\":%s,\"source_id\":\"%s\","
			"\"source_files\":%u,\"engine_id\":\"%s\",\"engine_files\":%u,"
			"\"build\":\"%s\"}\n",
			SCHEMA_VERSION, OPTBENCH_GIT_REV,
			OPTBENCH_GIT_DIRTY ? "true" : "false",
			OPTBENCH_SOURCE_ID, ( unsigned ) OPTBENCH_SOURCE_COUNT,
			OPTBENCH_ENGINE_ID, ( unsigned ) OPTBENCH_ENGINE_COUNT,
			buildIdentity().c_str() );
		return 0;
	}

	vector< Case > table = allCases();

	if ( list )
	{
		// --pilot / --step0b NARROW the listing, so a caller that wants one
		//    half of the table gets exactly those names and does not have to
		//    re-derive the membership rule from a name prefix.
		vector< Case > shown = table;
		if ( pilot ) shown = pilotCases();
		else if ( step0b ) shown = step0bCases();
		for ( size_t i = 0; i < shown.size(); i++ )
			printf( "%-56s group=%-44s axis=%-16s scope=%-9s endpoint=%s\n",
				shown[ i ].name.c_str(), shown[ i ].group.c_str(),
				shown[ i ].groupAxis.c_str(), shown[ i ].timingScope.c_str(),
				shown[ i ].endpoint.c_str() );
		return 0;
	}

	// --workload NAMES A WORKLOAD, not an arm, and builds that workload's
	//    canonical reference directly. It is how a workload whose endpoint could
	//    not be established gets re-characterized at a larger budget: its arms
	//    are undeclared, so no --case can name it.
	vector< Case > selected;
	for ( size_t i = 0; i < workloads.size(); i++ )
	{
		Case c;
		if ( !referenceCaseFor( workloads[ i ], c ) )
		{
			fprintf( stderr, "refused -- workload: '%s' is not a key in the "
				"committed endpoint table\n", workloads[ i ].c_str() );
			return 2;
		}
		selected.push_back( c );
	}
	if ( !workloads.empty() && !wantCharacterize )
	{
		fprintf( stderr, "refused -- workload: --workload builds a canonical "
			"CONTROL, which is only meaningful with --characterize. Use --case "
			"to run a declared arm.\n" );
		return 2;
	}
	if ( all ) selected = table;
	else if ( pilot ) selected = pilotCases();
	else if ( step0b ) selected = step0bCases();
	else
		for ( size_t i = 0; i < names.size(); i++ )
		{
			Case c;
			if ( !findCase( names[ i ], c ) )
			{
				fprintf( stderr, "refused -- case: unknown case '%s'\n",
					names[ i ].c_str() );
				return 2;
			}
			selected.push_back( c );
		}

	if ( selected.empty() )
	{
		fprintf( stderr, "refused -- nothing selected: pass --all, --pilot, "
			"--step0b or --case NAME\n" );
		usage();
		return 2;
	}

	// CHARACTERIZATION IS A CANONICAL CONTROL, and refuses to be anything else.
	//
	//    A matched target must come from a canonical reference run, never from
	//    the candidate arm being timed. Until this refusal existed the flag only
	//    lowered the target, so `--characterize --case simpleprop-shanno`
	//    cheerfully characterized Shanno and reported it as a canonical control
	//    -- the documentation said one thing and the code did another.
	//
	//    Refusing is preferred over silently substituting the group's canonical
	//    case: the caller asked for a specific arm, and quietly running a
	//    different one is how a result comes to describe a run nobody requested.
	//    The message names the case that SHOULD be characterized instead.
	if ( wantCharacterize )
	{
		for ( size_t i = 0; i < selected.size(); i++ )
			if ( !isCanonicalReference( selected[ i ] ) )
			{
				string ref = canonicalReferenceFor( selected[ i ].group );
				fprintf( stderr, "refused -- characterize: '%s' is not a canonical "
					"reference (optimizer=%s, auto_step=%s). A matched target must "
					"come from a canonical control, not from the arm being timed.\n",
					selected[ i ].name.c_str(),
					optimizerName( selected[ i ].optimizer ),
					selected[ i ].autoStep ? "on" : "off" );
				if ( ref.empty() )
					fprintf( stderr, "            group '%s' declares no canonical "
						"reference case.\n", selected[ i ].group.c_str() );
				else
					fprintf( stderr, "            characterize '%s' instead.\n",
						ref.c_str() );
				return 2;
			}

		// A characterization is not a timed arm and does not pretend to be one:
		//    it emits its OWN record type, so nothing downstream can mistake the
		//    control run for a measurement of the arm it derives targets for.
		//    The declared target and ceiling are overridden inside characterize()
		//    -- the whole point is to run past whatever endpoint is currently
		//    committed and see the floor.
		int rc = 0;
		for ( size_t i = 0; i < selected.size(); i++ )
		{
			Case c = selected[ i ];
			c.target = 0.5;      // any valid value; characterize() replaces it
			if ( c.ceiling < 2 ) c.ceiling = STRICT_CEILING;
			Characterization ch = characterize( c, ceiling ? ceiling : STRICT_CEILING );
			printf( "%s\n", toJsonLine( ch ).c_str() );
			fflush( stdout );
			if ( !ch.ok ) rc = 3;
		}
		return rc;
	}

	for ( unsigned rep = 0; rep < reps; rep++ )
		for ( size_t i = 0; i < selected.size(); i++ )
		{
			Row r = runCase( selected[ i ], rev );
			printf( "%s\n", toJsonLine( r ).c_str() );
			fflush( stdout );
		}

	return 0;
}
