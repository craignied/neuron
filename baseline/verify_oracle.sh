#!/bin/bash
# Deterministic oracle cross-check for the 3.0 engine.
# 1. Train XOR with the legacy binary and save the network (xor_train_save.in).
# 2. Load the identical saved weights into BOTH binaries (xor_verify.in); entering
#    "Use model" runs one eta-0 forward pass, printing error and statistics.
# 3. Diff the two sessions (version banner/farewell excluded). Any numeric drift fails.
set -e
cd "$(dirname "$0")"
LEGACY=build/neuron-2.64/src/neuron
NEW=../build/neuron
[ -x "$LEGACY" ] || { echo "Legacy oracle not built — run ./build_legacy.sh" >&2; exit 1; }
[ -x "$NEW" ] || { echo "3.0 engine not built — cmake -B build && cmake --build build (repo root)" >&2; exit 1; }
mkdir -p runs && cd runs
../"$LEGACY" < ../xor_train_save.in > /dev/null
../"$LEGACY" < ../xor_verify.in > verify_legacy.txt
../"$NEW"    < ../xor_verify.in > verify_30.txt
strip() { grep -v -e 'Welcome to' -e 'Thank you for using' "$1"; }
if diff <(strip verify_legacy.txt) <(strip verify_30.txt); then
    echo "OK: legacy and 3.0 outputs are numerically identical"
else
    echo "FAIL: outputs differ (see baseline/runs/verify_*.txt)" >&2
    exit 1
fi
