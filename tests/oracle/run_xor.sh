#!/bin/sh
# Run the scripted XOR training session.
# Usage: ./run_xor.sh [path-to-neuron-binary]
# Default binary is the legacy oracle; pass ../../build/neuron to test the 3.0 engine.
set -e
cd "$(dirname "$0")"
BIN=${1:-bin/neuron-2.64}
BIN=$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")
[ -x "$BIN" ] || { echo "Binary not found: $BIN (for the oracle, run ./build_oracle.sh first)" >&2; exit 1; }
mkdir -p runs && cd runs
"$BIN" < ../xor.in
