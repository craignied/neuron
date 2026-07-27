#!/bin/bash
# Deterministic oracle cross-check for the 3.0 engine.
# Loads the committed reference network (xor_net.txt) into BOTH binaries; entering
# "Use model" runs one eta-0 forward pass, printing error and statistics. The two
# sessions must be identical except for the Kolmogorov-Smirnov line:
#
#   KNOWN ORACLE BUG (fixed in 3.0, found 2026-07-11): legacy KScalc() kept
#   Numerical Recipes' 1-based indices on 0-based vectors, skipping element 0 and
#   reading one past the end — its K-S statistic is partly heap garbage. The oracle's
#   K-S line is therefore excluded from the diff; 3.0's K-S is instead checked
#   against the known-correct value for this network (D = 1 for perfectly
#   separated XOR).
#
# To regenerate the reference network (shouldn't normally be needed):
#   cd runs && ../bin/oracle < ../xor_train_save.in && cp xor_net.txt ../
set -e
cd "$(dirname "$0")"
ORACLE=bin/oracle
NEW=../../build/neuron
[ -x "$ORACLE" ] || { echo "Oracle not built — run ./build_oracle.sh" >&2; exit 1; }
[ -x "$NEW" ] || { echo "3.0 engine not built — cmake -B build && cmake --build build (repo root)" >&2; exit 1; }
mkdir -p runs && cd runs
cp ../xor_net.txt .
../"$ORACLE" < ../xor_verify.in > verify_oracle.txt
../"$NEW"    < ../xor_verify.in > verify_30.txt
# Excluded lines: version banner/farewell; Kolmogorov-Smirnov (known oracle
# bug, see README); "95% CI" (3.0 enhancement — the oracle never reported
# confidence intervals on ROC areas); the oracle's two-line "Maximum number of
# bins to be searched ... exceeds data" warning (3.0 removed the binning
# entirely in favour of Wickens' binomial error bars — there is no bin count to
# search over, so the warning has nothing left to warn about; see
# docs/roc_theory.md). The exclusion is the whole warning, both its lines.
# Hosmer-Lemeshow (legacy bug #9, fixed in 3.0 2026-07-16): the oracle's
# H-L accumulated junk terms across a group-count scan and reported the most
# favorable p — on these 4 exemplars its own gammq happens to throw. 3.0
# computes the textbook C-hat (g=10 deciles of risk) and refuses honestly
# when there are fewer exemplars than groups; that refusal is asserted below.
# Pearson (same audit): the oracle printed a "p" from a hardcoded
# chi-squared(2) over a statistic that scales with n — noise. 3.0 prints the
# correct individual-level X2 with NO p (none validly exists at cells of
# size 1); on this near-perfectly-fitted XOR the residuals are tiny, so the
# known-correct X2 is near zero — asserted below. (3.0's Pearson line ends
# "see Hosmer-Lemeshow", so that exclusion catches it; the Pearson exclusion
# below is for the oracle's own line.)
# "Number thresholds" (2026-07-19): 3.0's trapezoidal ROC area is now the exact
# AUC integrated over every operating point (the same operatingPoints() sweep
# the statistical method and bootstrap use) instead of a fixed-count grid, so
# there is no threshold count to print. The AREA itself is unchanged on this
# perfectly separated XOR (1.0 either way), so only the oracle's now-orphaned
# "Number thresholds = 4" line is excluded.
# The ITERATION COUNT and the gradient-stop message (legacy bug #10, fixed in
# 3.0 2026-07-26): the oracle recalculated the maximum absolute gradient only
# inside the block that PRINTS an iteration row, and compared that cached value
# against the limit on every iteration -- so a stopping condition depended on a
# presentation setting. This verification is the smallest possible instance.
# xor_net.txt is loaded ALREADY converged (its gradient is 9.06e-08, below the
# 1e-06 limit) and eta is 0, so nothing can change: the correct answer is to
# stop immediately, at iteration 0. The oracle cannot -- iteration 0 is not a
# logarithmic print point, so it never looks, takes a (no-op) training step,
# prints its row at iteration 1 and only then notices. Excluded, therefore: the
# oracle's now-unmatched iteration row, "Total iterations", and the stop message
# (whose float precision is inherited from whatever last formatted the stream,
# so printing a row changes its spelling: "1.0e-06" vs "1.000000e-06").
# BOTH sides are asserted below, so this exclusion cannot quietly start hiding
# some other divergence. Because eta is 0 the extra step is a no-op and every
# statistic that follows is identical -- which is what makes this a clean
# demonstration rather than a re-blessing.
strip() { grep -v -e 'Welcome to' -e 'Thank you for using' -e 'Kolmogorov-Smirnov' \
    -e '95% CI' -e 'Maximum number of bins' -e 'Setting Maximum number of bins' \
    -e 'Hosmer-Lemeshow' -e 'Pearson' -e 'Number thresholds' \
    -e 'Total iterations' -e 'maximum absolute gradient became lower' "$1" \
    | grep -Ev '^ +[0-9]+ +[0-9]+\.[0-9]+e[+-][0-9]+ '; }
fail=0
diff <(strip verify_oracle.txt) <(strip verify_30.txt) || fail=1
grep -q 'Kolmogorov-Smirnov goodness of fit D = 1, p = 0.0970269' verify_30.txt \
    || { echo "FAIL: 3.0 K-S line differs from known-correct value" >&2; fail=1; }
grep -q 'fewer exemplars than Hosmer-Lemeshow groups (10)' verify_30.txt \
    || { echo "FAIL: 3.0 H-L should refuse on 4 exemplars (known-correct behavior)" >&2; fail=1; }
grep -q 'Pearson X2 = 0.0103491 (n = 4; no valid p at the individual level' verify_30.txt \
    || { echo "FAIL: 3.0 Pearson line differs from known-correct statistic" >&2; fail=1; }
grep -q 'maximum absolute gradient became lower' verify_30.txt \
    && grep -q 'Total iterations = 0' verify_30.txt \
    || { echo "FAIL: 3.0 should stop on the gradient limit at iteration 0 -- the loaded network is already below it" >&2; fail=1; }
grep -q 'Total iterations = 1' verify_oracle.txt \
    || { echo "FAIL: the oracle no longer shows the print-cadence iteration (the exclusion above is now hiding something else)" >&2; fail=1; }
if [ $fail -eq 0 ]; then
    echo "OK: oracle and 3.0 outputs identical (K-S checked against known-correct value)"
else
    echo "FAIL: outputs differ (see tests/oracle/runs/verify_*.txt)" >&2
    exit 1
fi
