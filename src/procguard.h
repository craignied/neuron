#ifndef PROCGUARD_H
#define PROCGUARD_H

// THE PROCESS BOUNDARY (D9). The last resort at the top of a program: run the
// body, and if something reaches this far, say what it was and exit non-zero
// instead of terminating opaquely.
//
// This is deliberately its OWN unit rather than part of AsyncJob. Both are
// boundaries and both exist because an uncaught exception ends a process, but
// the CLI has no worker, no cancellation and no result to publish -- it has an
// exit status and a diagnostic. Folding it into the job type would be filing
// command-line exception handling under asynchronous-job behavior, which is
// not what it is.
//
// It is a function taking a body rather than a macro or a hand-written try
// block in main() because that is what makes it TESTABLE: a test supplies its
// own body and its own stream, and the shipped binary gains no injectable
// fault path.

#include <functional>
#include <ostream>

namespace procguard
{
	// Runs `body` and returns its exit status. An escaping std::exception
	//    becomes "<program>: fatal: <what>" on `err` and status 1; anything
	//    else becomes "<program>: fatal: unrecognized error" and status 1.
	int run( const std::function< int() >& body, std::ostream& err,
		const char* program );
}

#endif
