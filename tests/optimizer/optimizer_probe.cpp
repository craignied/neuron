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
		"usage: optimizer_probe [--list] [--identity] [--all] [--case NAME]\n"
		"                       [--rep N] [--rev REV] [--characterize]\n"
		"  --list          print the pilot case names and their groups\n"
		"  --identity      print this binary's revision/source identity and exit\n"
		"  --all           run every pilot case once\n"
		"  --case NAME     run one named case (repeatable)\n"
		"  --rep N         run each selected case N times (default 1)\n"
		"  --rev REV       revision string recorded in every row\n"
		"  --characterize  run a CANONICAL REFERENCE case with an unreachable\n"
		"                  target, to obtain the objective a matched target is\n"
		"                  then chosen from. Refused on any other case.\n" );
}

int main( int argc, char** argv )
{
	// The DEFAULT revision is the one compiled into this binary, alongside the
	//    dirty flag and the source identity. --rev annotates; it cannot make a
	//    stale binary claim a different source, because source_id and dirty are
	//    compiled in and are emitted regardless.
	string rev = OPTBENCH_GIT_REV;
	unsigned reps = 1;
	bool all = false, list = false, identity = false, characterize = false;
	vector< string > names;

	for ( int i = 1; i < argc; i++ )
	{
		string a = argv[ i ];
		if ( a == "--list" ) list = true;
		else if ( a == "--identity" ) identity = true;
		else if ( a == "--characterize" ) characterize = true;
		else if ( a == "--all" ) all = true;
		else if ( a == "--case" && i + 1 < argc ) names.push_back( argv[ ++i ] );
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
			"\"source_files\":%u,\"build\":\"%s\"}\n",
			SCHEMA_VERSION, OPTBENCH_GIT_REV,
			OPTBENCH_GIT_DIRTY ? "true" : "false",
			OPTBENCH_SOURCE_ID, ( unsigned ) OPTBENCH_SOURCE_COUNT,
			buildIdentity().c_str() );
		return 0;
	}

	vector< Case > table = pilotCases();

	if ( list )
	{
		for ( size_t i = 0; i < table.size(); i++ )
			printf( "%-34s group=%-18s axis=%s\n", table[ i ].name.c_str(),
				table[ i ].group.c_str(), table[ i ].groupAxis.c_str() );
		return 0;
	}

	vector< Case > selected;
	if ( all )
		selected = table;
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
		fprintf( stderr, "refused -- nothing selected: pass --all or --case NAME\n" );
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
	if ( characterize )
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

		for ( size_t i = 0; i < selected.size(); i++ )
		{
			selected[ i ].target = 1e-30;   // unreachable: run the whole ceiling
			selected[ i ].name += "-characterize";
		}
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
