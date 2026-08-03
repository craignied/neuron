#!/usr/bin/env python3
"""Keep the GUI's request boundary strict (ROADMAP 4 item B9).

Every request field in src/gui.cpp is read through util::parseUnsigned /
parseDouble / parseBool, behind the readUnsigned / readDouble / readBool
helpers in that file. This gate fails if the permissive conversions come back,
or if a handler grows its own copy of a numeric or boolean parser -- which is
how the file came to hold 51 atol/atof calls, three uintParam/fracParam
lambdas and a boolParam in the first place.

Scope is src/gui.cpp alone, deliberately. The CLI menus are frozen and
src/neuron.cpp keeps its own conversions; this is about the HTTP boundary,
where the caller is a script or an LLM and there is nobody to re-read a prompt.

A forbidding gate can be satisfied by deleting the parsing altogether, so the
positive half runs too: the readers must still be defined and still be used.

Run by tests/tools/run_tools.sh, therefore by CI on all three platforms.
"""

import re
import sys
from pathlib import Path

GUI = Path( __file__ ).resolve().parent.parent / "src" / "gui.cpp"

# The C conversions that answer with a value and no error, plus the C++ ones
# that would throw where a handler must return a message. Word-bounded so that
# an identifier merely containing one of these names is not flagged.
FORBIDDEN_CALLS = re.compile(
    r"\b(atol|atoi|atof|strtol|strtoul|strtoll|strtoull|strtod|strtof"
    r"|std::stoi|std::stol|std::stoul|std::stod|std::stof)\s*\(" )

# A request field compared straight against a literal is a hand-rolled boolean
# parser: param( req, "async" ) == "1". A comparison of a LOCAL string against
# a token is not flagged -- the algorithm field really is an enum with the
# spellings 1, 2, 3 and auto, and it is read that way on purpose.
INLINE_TOKEN_TEST = re.compile( r'param\(\s*req\s*,\s*"[\w_]+"\s*\)\s*[!=]=\s*"' )

# A handler-local parser lambda. The two that existed were spelled uintParam,
# fracParam and boolParam; the pattern is the shape, not those three names.
LOCAL_PARSER_LAMBDA = re.compile( r"auto\s+\w*(Param|Field|Parse)\w*\s*=\s*\[" )

REQUIRED = [
    ( "static string readUnsigned(", "the shared unsigned reader" ),
    ( "static string readDouble(", "the shared floating-point reader" ),
    ( "static string readBool(", "the shared boolean reader" ),
    ( "util::parseUnsigned(", "util's strict whole-number parser" ),
    ( "util::parseDouble(", "util's strict number parser" ),
    ( "util::parseBool(", "util's strict flag parser" ),
]


def main():
    if not GUI.exists():
        print( "check_strict_parsing: %s not found" % GUI )
        return 1

    text = GUI.read_text( encoding = "utf-8" )
    lines = text.split( "\n" )
    problems = []

    for n, line in enumerate( lines, 1 ):
        code = line.split( "//" )[ 0 ]     # a comment may name what it replaced
        for pattern, what in (
            ( FORBIDDEN_CALLS, "a permissive conversion" ),
            ( INLINE_TOKEN_TEST, "a hand-rolled boolean test on a request field" ),
            ( LOCAL_PARSER_LAMBDA, "a handler-local parser lambda" ),
        ):
            m = pattern.search( code )
            if m:
                problems.append( "src/gui.cpp:%d: %s -- %s"
                    % ( n, what, m.group( 0 ).strip() ) )

    for needle, what in REQUIRED:
        if needle not in text:
            problems.append( "src/gui.cpp: %s (%s) is gone" % ( needle, what ) )

    # Used, not merely defined: a reader nobody calls is a deleted contract.
    for reader in ( "readUnsigned(", "readDouble(", "readBool(" ):
        if text.count( reader ) < 3:
            problems.append( "src/gui.cpp: %s appears %d time(s) -- the handlers "
                "are not calling it" % ( reader, text.count( reader ) ) )

    if problems:
        print( "Strict request parsing has regressed (ROADMAP 4 B9,"
            " docs/b9_strict_parsing.md):" )
        for p in problems:
            print( "   ", p )
        return 1

    print( "Strict request parsing OK: no permissive conversion, no handler-local"
        " parser, all three readers present and in use" )
    return 0


sys.exit( main() )
