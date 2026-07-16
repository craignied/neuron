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
strip() { grep -v -e 'Welcome to' -e 'Thank you for using' -e 'Kolmogorov-Smirnov' \
    -e '95% CI' -e 'Maximum number of bins' -e 'Setting Maximum number of bins' \
    -e 'Hosmer-Lemeshow' "$1"; }
fail=0
diff <(strip verify_oracle.txt) <(strip verify_30.txt) || fail=1
grep -q 'Kolmogorov-Smirnov goodness of fit D = 1, p = 0.0970269' verify_30.txt \
    || { echo "FAIL: 3.0 K-S line differs from known-correct value" >&2; fail=1; }
grep -q 'fewer exemplars than Hosmer-Lemeshow groups (10)' verify_30.txt \
    || { echo "FAIL: 3.0 H-L should refuse on 4 exemplars (known-correct behavior)" >&2; fail=1; }
if [ $fail -eq 0 ]; then
    echo "OK: oracle and 3.0 outputs identical (K-S checked against known-correct value)"
else
    echo "FAIL: outputs differ (see tests/oracle/runs/verify_*.txt)" >&2
    exit 1
fi
