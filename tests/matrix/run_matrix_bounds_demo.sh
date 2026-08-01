#!/bin/sh
# run_matrix_bounds_demo.sh -- runs each check_matrix_bounds case in its OWN
# process, for the D9 characterization.
#
# Against assert-only matrix.h a violated precondition does not throw: it reads
# or WRITES outside an allocation. Some cases return a wrong answer, some kill
# the process. Run in one process, the first crash would hide every case after
# it -- so each is run alone and its exit status recorded. That is what makes
# this evidence rather than a single "it crashed" line.
#
# Every case runs a LEGAL operation of the same family and checks its answer
# BEFORE it makes the invalid call, so a process that dies before its control
# is a broken harness, not an absent contract. That is exit status 3.
#
# Exit status per case:
#     0    the contract HELD (the expected exception was thrown)
#     1    NO exception -- the operation ran outside its operands
#     3    the CONTROL failed -- the case proves nothing; fix the case
#     4    threw the WRONG exception type
#   >4     the process died (signal); see the status column
#
# Usage:  tests/matrix/run_matrix_bounds_demo.sh [path-to-check_matrix_bounds]
#
# This is a DEMONSTRATION driver, not a gate. The permanent assertion is the
# ctest case, which runs every case in one process and is registered in
# CMakeLists.txt only once the policy exists and they all throw.

BIN="${1:-./build/check_matrix_bounds}"

if [ ! -x "$BIN" ]; then
	echo "no such executable: $BIN" >&2
	echo "build it with: cmake --build build --target check_matrix_bounds" >&2
	exit 2
fi

# Ask the binary for its case list rather than hard-coding a count that rots the
# moment a case is added. Reading the names UP FRONT is what lets a case that
# kills its own process still be named below.
LIST=$( "$BIN" -l ) || { echo "could not list cases" >&2; exit 2; }
total=$( printf '%s\n' "$LIST" | wc -l | tr -d ' ' )

echo "check_matrix_bounds: $total cases, one process each"
echo "  binary: $BIN"
echo ""

held=0
unprotected=0
control=0
wrongtype=0
crashed=0

n=1
while [ "$n" -le "$total" ]; do
	name=$( printf '%s\n' "$LIST" | sed -n "${n}p" | cut -d' ' -f2- )

	"$BIN" "$n" > /dev/null 2>&1
	status=$?

	case "$status" in
		0 ) verdict="held";           held=$(( held + 1 )) ;;
		1 ) verdict="NO THROW";       unprotected=$(( unprotected + 1 )) ;;
		3 ) verdict="CONTROL FAILED"; control=$(( control + 1 )) ;;
		4 ) verdict="WRONG TYPE";     wrongtype=$(( wrongtype + 1 )) ;;
		* ) verdict="CRASHED($status)"; crashed=$(( crashed + 1 )) ;;
	esac

	printf '%3s  %-16s %s\n' "$n" "$verdict" "$name"
	n=$(( n + 1 ))
# The loop's stderr goes nowhere: a fatal signal is reported by the SHELL, not
#    by the program, so a case that takes a SIGSEGV or SIGABRT -- which is the
#    finding -- would otherwise print "Segmentation fault" ahead of its own row
#    and shred the report. The verdict column carries the same fact, with the
#    status number.
done 2>/dev/null

echo ""
echo "held:            $held"
echo "no exception:    $unprotected"
echo "wrong type:      $wrongtype"
echo "crashed:         $crashed"
echo "control failed:  $control"

if [ "$control" -gt 0 ]; then
	echo ""
	echo "A CONTROL FAILED. Those cases prove nothing about the contract --" >&2
	echo "the legal operation inside them did not give the right answer." >&2
	exit 2
fi

exit 0
