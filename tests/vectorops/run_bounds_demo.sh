#!/bin/sh
# run_bounds_demo.sh -- runs each check_vector_bounds case in its OWN process.
#
# Against assert-only vector_ops.h (D5a) a violated precondition does not throw;
# it reads or writes outside a container. Some cases return a wrong answer, some
# abort the process. Run in one process, the first crash would hide every case
# after it -- so each is run alone and its exit status recorded.
#
# Exit status per case:
#     0    the contract held (the expected exception was thrown)
#     1    NO exception -- the operation ran outside its operands
#   >1     the process died (signal); see the status column
#
# Usage:  tests/vectorops/run_bounds_demo.sh [path-to-check_vector_bounds]
#
# This is a DEMONSTRATION driver, not a gate. The permanent assertion is the
# ctest case, which runs every case in one process once they all throw (D5b).

BIN="${1:-./build/check_vector_bounds}"

if [ ! -x "$BIN" ]; then
	echo "no such executable: $BIN" >&2
	echo "build it with: cmake --build build --target check_vector_bounds" >&2
	exit 2
fi

# Ask the binary for its case list, rather than hard-coding a count that rots
# the moment a case is added. Reading the names UP FRONT is what lets a case
# that kills its own process still be named below.
LIST=$( "$BIN" -l ) || { echo "could not list cases" >&2; exit 2; }
total=$( printf '%s\n' "$LIST" | wc -l | tr -d ' ' )

echo "check_vector_bounds: $total cases, one process each"
echo "  binary: $BIN"
echo ""

held=0
unprotected=0
crashed=0

n=1
while [ "$n" -le "$total" ]; do
	name=$( printf '%s\n' "$LIST" | sed -n "${n}p" | cut -d' ' -f2- )

	"$BIN" "$n" > /dev/null 2>&1
	status=$?

	case "$status" in
		0 ) verdict="   held"; held=$(( held + 1 )) ;;
		1 ) verdict="  NO THROW"; unprotected=$(( unprotected + 1 )) ;;
		* ) verdict="  CRASHED($status)"; crashed=$(( crashed + 1 )) ;;
	esac

	printf '%2s %-14s %s\n' "$n" "$verdict" "$name"
	n=$(( n + 1 ))
# The loop's stderr goes nowhere: a fatal signal is reported by the SHELL, not
#    by the program, so a case that takes a SIGBUS -- which is the finding --
#    otherwise prints "Bus error" ahead of its own row and shreds the report.
#    The verdict column carries the same fact, with the status number.
done 2>/dev/null

echo ""
echo "held: $held    unprotected: $unprotected    crashed: $crashed"

# The driver reports; it does not judge. A run against assert-only code is
# EXPECTED to show unprotected cases -- that is the finding being recorded.
exit 0
