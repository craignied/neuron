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
#        binormal_seed42 (the statistical ROC path: binormal Az, the bin search,
#          and the bootstrap intervals — NONE of which the two cases above reach.
#          They have too few exemplars (goodData < calcThresh) and print "Cannot
#          calculate ROC statistically" instead, so before this case existed the
#          whole ROC report was invisible to the goldens: the delta-method
#          interval was replaced by a bootstrap with every invariant green.
#          Low-birth-weight is 189 rows, split 142/47, which reaches the binned
#          path on train and the unbinned one on test — both in one transcript.)
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
for case in xor_seed42 regress_seed42 binormal_seed42; do
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

# --- Coverage: does this suite execute what it claims to guard? -------------
#
# A green golden run is worthless if the transcripts never reach the code you
# changed. That is not hypothetical: on 2026-07-15 the entire ROC confidence
# interval was replaced -- delta method out, bootstrap in -- with all five
# project invariants passing, because xor_seed42 and regress_seed42 have four
# exemplars and print "Cannot calculate ROC statistically". None of them
# contained a single "By statistical method" line. The green was pure noise.
#
# So the suite asserts its own reach. Each entry is a feature that some
# transcript MUST exercise; if a case is deleted, retargeted, or quietly stops
# reaching a path, this fails instead of going green. Add an entry whenever you
# add a case that is the sole cover for something -- the point is that this list
# cannot rot silently the way a comment can.
#
# Format: <pattern><TAB><what it proves is covered>
coverage() {
    cat <<'FEATURES'
By statistical method	the binormal ROC report (binormal_seed42)
95% CI = 	ROC confidence intervals
bootstrap resamples	the bootstrap interval, not an analytic one
Operating points fitted	the zROC fit over distinct operating points
Kolmogorov-Smirnov	the K-S goodness of fit
Hosmer-Lemeshow	the Hosmer-Lemeshow goodness of fit
Reverse regressing	stepwise regression through RegressNet (regress_seed42)
FEATURES
}
if [ $BLESS -eq 0 ]; then
    all=$(cat xor_seed42.out regress_seed42.out binormal_seed42.out 2>/dev/null)
    while IFS=$'\t' read -r pattern what; do
        [ -z "$pattern" ] && continue
        case "$all" in
            *"$pattern"*) ;;
            *) echo "FAIL: no golden transcript covers $what (no \"$pattern\" anywhere)" >&2
               fail=1 ;;
        esac
    done <<< "$(coverage)"
    [ $fail -eq 0 ] && echo "OK: the transcripts cover every path this suite claims to guard"
fi

exit $fail
