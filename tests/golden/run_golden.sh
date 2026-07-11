#!/bin/bash
# Golden-transcript test: run fully seeded sessions and require byte-identical
# output to the committed transcripts (wall-clock "That took" line excluded).
# std::mt19937's standard-specified stream makes the transcripts valid on every
# platform and compiler — this is the cross-platform regression test CI runs.
#
# Usage: ./run_golden.sh [path-to-neuron-binary]
#        ./run_golden.sh --bless [path]   # regenerate the expected transcripts
#
# Cases: xor_seed42 (dataset load, SimpleProp training, full statistics)
#        regress_seed42 (same + stepwise reverse regression through RegressNet)
set -e
cd "$(dirname "$0")"

BLESS=0
if [ "$1" = "--bless" ]; then BLESS=1; shift; fi

BIN=$1
if [ -z "$BIN" ]; then
    for c in ../../build/neuron ../../build/Release/neuron.exe ../../build/neuron.exe; do
        if [ -x "$c" ]; then BIN=$c; break; fi
    done
fi
[ -n "$BIN" ] && [ -x "$BIN" ] || { echo "neuron binary not found — build it first (cmake -B build && cmake --build build)" >&2; exit 1; }
BIN=$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")

mkdir -p runs && cd runs

# Strip the one nondeterministic line (elapsed wall-clock time) and any CR
# a Windows shell might append.
strip() { grep -v 'That took' "$1" | sed 's/\r$//'; }

fail=0
for case in xor_seed42 regress_seed42; do
    "$BIN" --seed 42 < ../$case.in > $case.out 2>&1 \
        || { echo "FAIL: $case exited nonzero" >&2; fail=1; continue; }
    if [ $BLESS -eq 1 ]; then
        strip $case.out > ../$case.expected
        echo "blessed $case.expected"
    elif diff <(strip ../$case.expected) <(strip $case.out); then
        echo "OK: $case matches golden transcript"
    else
        echo "FAIL: $case diverged from golden transcript (see tests/golden/runs/$case.out)" >&2
        fail=1
    fi
done
exit $fail
