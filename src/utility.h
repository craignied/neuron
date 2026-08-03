// Utility methods header for neUROn2++

// NOTE: Template functions and headers are here, everything else is
//    utility.cpp

#ifndef UTILITY_H
#define UTILITY_H

// Because, apparently, vector< vector< unsigned > > symbol names exceeds 255
//    characters in length, and MSCV will error with that (ugly!)
#ifdef _MSC_VER	// MSVC
#pragma warning (disable: 4786)
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cassert>
#include <stdexcept>

using namespace std;

namespace util {
	// util exception, extends runtime_error class
	class utilErr : public runtime_error {
	public:
		utilErr( const char* message ) throw() : runtime_error( message ) {}
	};

	// Random number methods:
	// Seeds the generator with a user-specified seed for reproducible runs;
	//    overrides the timestamp seeding in d_random()
	void set_seed( unsigned );
	// No arguments seeds generator with timestamp (unless set_seed was called)
	void d_random();
	// 1 argument returns random number between -arg and +arg
	double d_random( double );
	// 2 arguments returns random number between arguments
	double d_random( double, double );
	// Returns a uniform random integer in [0, n-1]
	unsigned i_random( unsigned );

	// Independent stream for resampling (the bootstrap). Kept separate from
	//    the generator above on purpose: statistics must never perturb the
	//    weight-init / train-test-split draws, or computing a confidence
	//    interval would change the next model trained. set_seed() seeds both,
	//    so --seed N stays fully reproducible.
	// Returns a uniform random integer in [0, n-1] from the resampling stream
	unsigned i_resample( unsigned );

	// Rounds to number of significant digits
	double round( double, unsigned );

	// Ask methods:
	// asks question with a lower bound, returns int
	int askI( const string&, const int );
	// asks question with bounds, returns int
	int askI( const string&, const int, const int );
	// asks question with a lower bound, returns double
	double askD( const string&, const double );
	// asks question with bounds, returns double
	double askD( const string&, const double, const double );
	// asks yes, no question
	bool askYesNo( const string& );

	// Utility function which tests a filename to see if the file exists,
	//    returns the filename if it does, and keeps querying for a good
	//    filename if it doesn't.
	string& getGoodFile( string& );

	// Run directory for engine log files (neuron.log, model.txt): they are
	//    written next to the session's first-loaded dataset file, not the cwd
	void set_run_dir( const string& ); // remember a dataset file's directory
	string run_path( const string& ); // resolve a bare log filename into it

	// Engine-core output stream: everything the engine classes print goes
	//    through screen() (default cout). A GUI/server redirects it with
	//    set_screen() to capture reports. The menu driver and the ask*
	//    prompts above deliberately stay on cout/cin.
	ostream& screen(); // the current engine output stream
	void set_screen( ostream& ); // redirect engine output

	// Removes carriage return from end of string if exists, returns string
	string& chopEndl( string& );

	// --- Strict text-to-value parsing (ROADMAP 4 item B9) -----------------
	//    A request field arrives as text and has to become a number, a count or
	//    a flag. The C conversions this replaced -- atol and atof -- answer with
	//    a value and no error: they consume as much of the text as they can
	//    understand, stop, and return what they got. So "5junk" is five,
	//    "abc" is zero, and a count above the width of the type wraps. Three
	//    live consequences of that, measured on the pre-B9 server and written
	//    up in docs/b9_strict_parsing.md: fraction=abc asked for a train/test
	//    split and silently produced no test set, maxiter=4294967297 trained
	//    for one iteration, and gradmax=inf installed a stopping rule that can
	//    never fire.
	//
	//    These parsers answer with a STATUS and write the destination only when
	//    that status is Ok, so a caller can neither miss a failure nor be left
	//    holding half a conversion. They know nothing about HTTP, JSON or which
	//    endpoint asked; the caller owns the field's name and its domain.

	// Why a field could not be read as the type its contract names. Ordered
	//    from "nothing was there" to "the text cannot be represented".
	enum class ParseStatus
	{
		Ok,         // parsed, and the destination was written
		Empty,      // nothing but surrounding whitespace
		Syntax,     // not a number, or not a permitted token, at all
		Trailing,   // a valid value followed by something else ("5junk")
		Negative,   // a negative value where the contract has no negatives
		Range,      // representable as text, not as the destination type
		NotFinite   // a floating value of nan or +/-inf
	};

	// Leading and trailing spaces and tabs are trimmed, then the remainder must
	//    be consumed entirely -- which is what atol/atof already accepted, so
	//    no request that worked before stops working, while "5junk" and "5 junk"
	//    are both refused. Interior whitespace is never permitted.
	//
	// parseUnsigned: refuses a negative as Negative rather than Syntax, so the
	//    message can say what is actually wrong, and refuses anything above
	//    UINT_MAX as Range rather than wrapping.
	ParseStatus parseUnsigned( const string& text, unsigned& out );

	// parseDouble: refuses nan and every spelling of infinity (NotFinite) --
	//    no field's contract permits one. Overflow to HUGE_VAL is Range;
	//    UNDERFLOW is accepted and yields the closest representable value,
	//    which may be zero, because every field whose domain excludes zero
	//    refuses it at the domain check one line later and a legitimately tiny
	//    tolerance must not be rejected as a syntax fault.
	ParseStatus parseDouble( const string& text, double& out );

	// parseBool: exactly "1" and "0", case-sensitively. That is what the GUI
	//    page sends and what AGENTS.md documents; anything else is Syntax, so a
	//    misspelled flag can no longer silently mean false.
	ParseStatus parseBool( const string& text, bool& out );

	// The message for a failed parse of `field`, quoting the text it was given:
	//    "folds: '5junk' is not a whole number". One per destination type
	//    because the expectation differs; each names the field first, so a
	//    caller composing a response never has to say which one failed.
	string unsignedError( const string& field, const string& text, ParseStatus );
	string numberError( const string& field, const string& text, ParseStatus );
	string boolError( const string& field, const string& text, ParseStatus );

	// SCOPED redirection of the engine's output stream. Construct one and every
	//    util::screen() write in this thread goes to the supplied stream until it
	//    goes out of scope; then the PREVIOUS stream is restored -- not cout.
	//
	//    Why this exists rather than a save/restore pair at each site. Six places
	//    in the engine did it by hand:
	//
	//        ostream& saved = util::screen();
	//        ostringstream discard;
	//        util::set_screen( discard );
	//        ... train() / obd::run() ...        <-- can throw
	//        util::set_screen( saved );
	//
	//    train() can throw (Matrix::BoundsViolation, stats::statsErr,
	//    RegressNetErr). When it did, the restore never ran and the engine's stream
	//    was left pointing at an ostringstream that was about to be destroyed --
	//    every subsequent engine print in that process writing through a dangling
	//    reference. A destructor runs during unwinding; an assignment does not.
	//
	//    Restoring the PREVIOUS stream rather than cout is what makes nesting
	//    correct. The GUI's own capture restored unconditionally to cout, so a
	//    capture inside a capture silently sent the outer scope's remaining output
	//    to the console instead of into the outer buffer.
	//
	//    Thread-local, like the stream it guards: each engine thread owns its own
	//    redirection and starts at cout (see utility.cpp).
	class ScreenCapture
	{
	public:
		// Redirect to the caller's stream
		explicit ScreenCapture( ostream& to );
		// Redirect to an internal buffer, readable through text()
		ScreenCapture();
		~ScreenCapture();

		// Non-copyable and non-movable: two objects restoring one saved stream
		//    would restore it twice, in an order nothing guarantees
		ScreenCapture( const ScreenCapture& ) = delete;
		ScreenCapture& operator= ( const ScreenCapture& ) = delete;

		// What was captured, when the default constructor was used
		string text() const { return own.str(); }
		ostringstream& buffer() { return own; }

	private:
		ostringstream own;      // used only by the default constructor
		ostream* previous;      // the stream to put back
	};


	// Method which parses a string to determine variable representation from
	//    nodes, and returns a *new* vector of vector of unsigned
	//    ";" delimits variables
	//    "," delimits nodes
	//    "-" specifies a range of nodes
	vector< vector< unsigned > > variable_parse( const string& );
	// Same, passed ifstream holds string
	vector< vector< unsigned > > variable_parse( ifstream& );

	// Method which removes a variable from a vector of vector of unsigned
	//    representation of variables (1st argument), returns a reference to
	//    the now manipulated vector of vector of unsigned, takes position of
	//    variable as 2nd argument, populates vector of vector of unsigned as
	//    3rd argument with result
	vector< vector< unsigned > >& remove_variable( const vector< vector< unsigned > >&,
		const unsigned, vector< vector< unsigned > >& );

	// As above, except returns *new* vector of vector of unsigned (no 3rd argument)
	vector< vector< unsigned > > remove_variable( const vector< vector< unsigned > >&,
		const unsigned );
}

// timestamp class and overloaded << operator allow timestamp method
//    to be inserted in an ostream
// Calling sequence: timestamp ( unsigned x )
//    where x is the number of seconds to print out in HHHH:MM:SS format
// Example:
//    cout << timestamp( 7261 ) << endl;
// prints out 02:01:01 then carriage return
class timestamp
{
	friend ostream& operator << ( ostream& output, const timestamp& t );

	public:
		timestamp ( const unsigned time_value ) { private_time = time_value; }

	private:
		unsigned private_time; // the passed number of seconds
};

#endif
