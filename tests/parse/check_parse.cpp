// The strict request-field parsers (ROADMAP 4 item B9): util::parseUnsigned,
// util::parseDouble, util::parseBool and the three message builders.
//
// These replaced atol and atof, which answer with a value and no error. This
// file pins the whole contract stated in utility.h -- the whitespace rule, the
// boolean token set, the sign and overflow policy, the non-finite refusal, the
// underflow acceptance, and the rule that a refused parse never writes its
// destination.
//
// Passing this file does NOT prove the handlers use these parsers. That is
// what tests/gui/strictparse.sh is for; a parser test and a migration test are
// different claims, and only the second one can catch an atol that was left
// behind.
//
// SABOTAGE (each watched to fail, then restored, with the object file deleted
// both times so the build could not believe a cached one):
//   * full-consumption check removed from parseUnsigned  -> the trailing cases;
//   * leading '-' allowed to reach strtoull              -> the sign cases;
//   * the UINT_MAX ceiling removed                       -> the range cases;
//   * !isfinite refusal removed from parseDouble         -> the nan/inf cases;
//   * ERANGE treated as a failure for any result         -> the underflow case;
//   * parseBool extended to accept "true"                -> the token cases;
//   * destination written before the status was decided  -> the untouched
//     cases, which are the ones a caller's `if ( status != Ok )` cannot see.
// No GSL, no engine: this links neuron_core alone.

#include <cfloat>
#include <cmath>
#include <iostream>
#include <string>

#include "utility.h"

using namespace std;

static int failures = 0;

static void expect( bool ok, const string& what )
{
	if ( ok )
		cout << "ok - " << what << endl;
	else
	{
		cout << "FAIL - " << what << endl;
		failures++;
	}
}

typedef util::ParseStatus PS;

static const char* name( PS s )
{
	switch ( s )
	{
		case PS::Ok:        return "Ok";
		case PS::Empty:     return "Empty";
		case PS::Syntax:    return "Syntax";
		case PS::Trailing:  return "Trailing";
		case PS::Negative:  return "Negative";
		case PS::Range:     return "Range";
		case PS::NotFinite: return "NotFinite";
	}
	return "?";
}

// --- unsigned ---------------------------------------------------------------

// The destination is seeded with a value no case produces, so "untouched"
//    is a real observation rather than a coincidence of defaults.
static const unsigned SENTINEL_U = 4242424u;

static void uOk( const string& text, unsigned want )
{
	unsigned got = SENTINEL_U;
	PS s = util::parseUnsigned( text, got );
	expect( s == PS::Ok && got == want,
		"parseUnsigned('" + text + "') = " + to_string( want )
			+ " [status " + name( s ) + ", got " + to_string( got ) + "]" );
}

static void uBad( const string& text, PS want )
{
	unsigned got = SENTINEL_U;
	PS s = util::parseUnsigned( text, got );
	expect( s == want, "parseUnsigned('" + text + "') is " + name( want )
		+ " [got " + name( s ) + "]" );
	expect( got == SENTINEL_U, "parseUnsigned('" + text
		+ "') left its destination untouched" );
}

static void checkUnsigned()
{
	cout << "--- parseUnsigned ---" << endl;

	// ordinary values, and the boundaries of the type
	uOk( "0", 0 );
	uOk( "1", 1 );
	uOk( "5", 5 );
	uOk( "42", 42 );
	uOk( "+7", 7 );                       // an explicit plus is a number
	uOk( "0042", 42 );                    // leading zeros are decimal, not octal
	uOk( "4294967295", 4294967295u );     // UINT_MAX exactly

	// the whitespace rule: surrounding is trimmed, interior is not whitespace
	//    the parser may ignore
	uOk( " 5", 5 );
	uOk( "5 ", 5 );
	uOk( "\t 5 \t", 5 );
	uBad( "5 5", PS::Trailing );
	uBad( "5\n", PS::Trailing );          // a newline is not surrounding space

	// trailing and leading junk -- the whole reason this function exists
	uBad( "5junk", PS::Trailing );
	uBad( "junk5", PS::Syntax );
	uBad( "5.0", PS::Trailing );          // a whole number is not a decimal one
	uBad( "5,3", PS::Trailing );
	uBad( "0x10", PS::Trailing );         // no hex, however C would read it
	uBad( "1e3", PS::Trailing );

	// nothing at all
	uBad( "", PS::Empty );
	uBad( "   ", PS::Empty );
	uBad( "\t", PS::Empty );

	// sign errors, distinguished from syntax so the message can say so
	uBad( "-1", PS::Negative );
	uBad( "-0", PS::Negative );
	uBad( " -5 ", PS::Negative );
	uBad( "-", PS::Negative );

	// overflow: just outside the type, and far outside it
	uBad( "4294967296", PS::Range );      // UINT_MAX + 1
	uBad( "4294967297", PS::Range );      // the pre-B9 "one iteration" case
	uBad( "99999999999999999999999999", PS::Range );

	// not numbers at all
	uBad( "abc", PS::Syntax );
	uBad( "nan", PS::Syntax );
	uBad( "inf", PS::Syntax );
	uBad( "+", PS::Syntax );
	uBad( ".", PS::Syntax );
}

// --- double -----------------------------------------------------------------

static const double SENTINEL_D = -98765.25;

static void dOk( const string& text, double want )
{
	double got = SENTINEL_D;
	PS s = util::parseDouble( text, got );
	expect( s == PS::Ok && got == want,
		"parseDouble('" + text + "') = " + to_string( want )
			+ " [status " + name( s ) + ", got " + to_string( got ) + "]" );
}

static void dBad( const string& text, PS want )
{
	double got = SENTINEL_D;
	PS s = util::parseDouble( text, got );
	expect( s == want, "parseDouble('" + text + "') is " + name( want )
		+ " [got " + name( s ) + "]" );
	expect( got == SENTINEL_D, "parseDouble('" + text
		+ "') left its destination untouched" );
}

static void checkDouble()
{
	cout << "--- parseDouble ---" << endl;

	// ordinary values, including every form a request actually carries
	dOk( "0", 0.0 );
	dOk( "1", 1.0 );
	dOk( "0.3", 0.3 );
	dOk( ".5", 0.5 );
	dOk( "5.", 5.0 );
	dOk( "-0.5", -0.5 );                  // negatives are legal here (bounds)
	dOk( "1e-4", 1e-4 );
	dOk( "1E-4", 1e-4 );
	dOk( "5e-5", 5e-5 );                  // the shipped weight-decay default
	dOk( "+2.5", 2.5 );

	// whitespace, as for the whole numbers
	dOk( " 0.3 ", 0.3 );
	dBad( "0.3 0.4", PS::Trailing );

	// trailing junk, and the exponent forms that are and are not complete
	dBad( "0.3junk", PS::Trailing );
	dBad( "0.05junk", PS::Trailing );
	dBad( "1e", PS::Trailing );           // pre-B9 this was 1.0
	dBad( "1e+", PS::Trailing );
	dBad( "1e-", PS::Trailing );
	dBad( "1.2.3", PS::Trailing );
	dBad( "--1", PS::Syntax );

	// nothing at all
	dBad( "", PS::Empty );
	dBad( "  ", PS::Empty );

	// non-finite, in every spelling strtod would otherwise accept
	dBad( "nan", PS::NotFinite );
	dBad( "NaN", PS::NotFinite );
	dBad( "-nan", PS::NotFinite );
	dBad( "inf", PS::NotFinite );
	dBad( "INF", PS::NotFinite );
	dBad( "-inf", PS::NotFinite );
	dBad( "+inf", PS::NotFinite );
	dBad( "infinity", PS::NotFinite );

	// overflow is a range error rather than a silent infinity
	dBad( "1e400", PS::Range );
	dBad( "-1e400", PS::Range );

	// underflow is accepted and yields the closest representable value, which
	//    may be zero. The domain check one line later in the caller is what
	//    refuses a zero where zero is not allowed.
	double tiny = SENTINEL_D;
	PS s = util::parseDouble( "1e-400", tiny );
	expect( s == PS::Ok, "parseDouble('1e-400') underflows to Ok" );
	expect( tiny >= 0.0 && tiny < DBL_MIN,
		"parseDouble('1e-400') yields a value below DBL_MIN" );

	// and the boundaries of the type itself
	dOk( "1.7976931348623157e308", DBL_MAX );
}

// --- bool -------------------------------------------------------------------

static void bOk( const string& text, bool want )
{
	bool got = !want;
	PS s = util::parseBool( text, got );
	expect( s == PS::Ok && got == want, "parseBool('" + text + "') = "
		+ ( want ? "true" : "false" ) + " [status " + name( s ) + "]" );
}

static void bBad( const string& text, PS want )
{
	// Both starting values, because a parser that writes before it decides
	//    would otherwise look correct for whichever one it happened to write.
	for ( int i = 0; i < 2; i++ )
	{
		bool got = ( i == 0 );
		PS s = util::parseBool( text, got );
		expect( s == want, "parseBool('" + text + "') is " + name( want )
			+ " [got " + name( s ) + "]" );
		expect( got == ( i == 0 ), "parseBool('" + text
			+ "') left its destination untouched" );
	}
}

static void checkBool()
{
	cout << "--- parseBool ---" << endl;

	bOk( "1", true );
	bOk( "0", false );
	bOk( " 1 ", true );
	bOk( "\t0\t", false );

	// The tokens that are NOT accepted. Every one of these silently meant
	//    false somewhere in the pre-B9 GUI, and "true" meant true in exactly
	//    one handler -- which is the inconsistency this token set removes.
	bBad( "true", PS::Syntax );
	bBad( "false", PS::Syntax );
	bBad( "TRUE", PS::Syntax );
	bBad( "True", PS::Syntax );
	bBad( "yes", PS::Syntax );
	bBad( "no", PS::Syntax );
	bBad( "on", PS::Syntax );
	bBad( "off", PS::Syntax );
	bBad( "2", PS::Syntax );
	bBad( "-1", PS::Syntax );
	bBad( "01", PS::Syntax );
	bBad( "1.0", PS::Syntax );
	bBad( "", PS::Empty );
	bBad( "   ", PS::Empty );
}

// --- the messages -----------------------------------------------------------

static void has( const string& text, const string& piece, const string& what )
{
	bool found = text.find( piece ) != string::npos;
	expect( found, found ? what
		: ( what + " -- '" + piece + "' is not in \"" + text + "\"" ) );
}

static void checkMessages()
{
	cout << "--- error messages ---" << endl;

	// Every message names its field. That is the requirement: a malformed
	//    request must say WHICH field, and must never be mistaken for a valid
	//    zero, a false, or a truncated number.
	string m = util::unsignedError( "folds", "5junk", PS::Trailing );
	has( m, "folds", "unsignedError names the field" );
	has( m, "5junk", "unsignedError quotes the text" );
	has( m, "whole number", "unsignedError says what was expected" );

	m = util::unsignedError( "test_n", "-5", PS::Negative );
	has( m, "test_n", "the negative message names the field" );
	has( m, "cannot be negative", "a negative reads as a sign fault" );

	m = util::unsignedError( "maxiter", "4294967296", PS::Range );
	has( m, "out of range", "an overflow reads as a range fault" );

	m = util::numberError( "gradmax", "inf", PS::NotFinite );
	has( m, "gradmax", "numberError names the field" );
	has( m, "finite", "a non-finite value says so" );

	m = util::numberError( "eta", "0.5junk", PS::Trailing );
	has( m, "is not a number", "numberError says what was expected" );

	m = util::boolError( "async", "true", PS::Syntax );
	has( m, "async", "boolError names the field" );
	has( m, "true", "boolError quotes the token" );
	has( m, "1 or 0", "boolError states the token set" );

	// An empty field's text is nothing, so the message must still be legible
	m = util::boolError( "logistic", "", PS::Empty );
	has( m, "logistic", "the empty message names the field" );
	has( m, "empty", "the empty message says the field is empty" );

	// A syntax fault and a domain fault must read differently -- the pre-B9
	//    strata_bins=abc was reported as "must be at least 2", which sent the
	//    reader looking for a value problem in a value that was never read.
	string syntax = util::unsignedError( "strata_bins", "abc", PS::Syntax );
	expect( syntax.find( "at least" ) == string::npos,
		"a syntax fault does not borrow the domain's wording" );
}

int main()
{
	checkUnsigned();
	checkDouble();
	checkBool();
	checkMessages();

	cout << ( failures ? "FAILURES: " : "all parser checks passed: " )
		<< failures << endl;
	return failures ? 1 : 0;
}
