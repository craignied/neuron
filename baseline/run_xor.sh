#!/bin/sh
# Run the scripted XOR training session against the legacy binary.
# Usage: ./run_xor.sh [> my_output.txt]
set -e
cd "$(dirname "$0")"
BIN=build/neuron-2.64/src/neuron
[ -x "$BIN" ] || { echo "Legacy binary not built — run ./build_legacy.sh first" >&2; exit 1; }
mkdir -p runs && cd runs
../"$BIN" < ../xor.in
