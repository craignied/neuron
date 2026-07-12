#!/bin/bash
# Fixture test for the Python tools: run each tool on committed sample input
# and require byte-identical output files and stdout (CR stripped for Windows).
# Usage: ./run_tools.sh [--bless]
set -e
cd "$(dirname "$0")"

PY=$(command -v python3 || command -v python)
[ -n "$PY" ] || { echo "python3 not found" >&2; exit 1; }

BLESS=0
[ "$1" = "--bless" ] && BLESS=1

mkdir -p runs && cd runs

"$PY" ../../../tools/mkdataset.py --key key.txt --inputs inputs.txt -o data.txt \
    ../sample.csv > stdout.txt 2>&1

# Semicolon-delimited categorical input: --delimiter + --onehot (auto-detect),
# yes/no outcome mapping, blank categorical value, numeric indicator pair.
"$PY" ../../../tools/mkdataset.py --delimiter ';' --onehot \
    --key cat_key.txt --inputs cat_inputs.txt -o cat_data.txt \
    ../sample_categorical.csv > cat_stdout.txt 2>&1

# Same input with --refcat: reference-category (k-1) coding for the one-hot
# group, plus the blank-vs-reference warning.
"$PY" ../../../tools/mkdataset.py --delimiter ';' --onehot --refcat \
    --key ref_key.txt --inputs ref_inputs.txt -o ref_data.txt \
    ../sample_categorical.csv > ref_stdout.txt 2>&1

# neuron2web.py: deploy the committed XOR reference network. The scaling
# and guesses files were produced by the engine itself (raw2train on
# docs/xor.csv; eta-0 pass over tests/oracle/xor_net.txt).
"$PY" ../../../tools/neuron2web.py --network ../../oracle/xor_net.txt \
    --scales ../xor_scales.txt --spec ../xor_spec.txt \
    -o xor.html > xor_stdout.txt 2>&1

# Cross-check the tool's forward pass against the engine's saved guesses
# for all four XOR rows (max abs difference must be < 1e-6).
{
  while read -r outcome guess; do
    [ -n "$guess" ] && echo "$guess"
  done < ../xor_guesses.txt
  for row in "0,0" "0,1" "1,0" "1,1"; do
    "$PY" ../../../tools/neuron2web.py --network ../../oracle/xor_net.txt \
        --scales ../xor_scales.txt --spec ../xor_spec.txt --eval "$row"
  done
} | "$PY" -c '
import sys
v = [float(x) for x in sys.stdin.read().split()]
n = len(v) // 2
worst = max(abs(a - b) for a, b in zip(v[:n], v[n:]))
print(f"forward pass vs engine guesses: max |diff| = {worst:.2e}")
sys.exit(0 if worst < 1e-6 else 1)
' || { echo "FAIL: neuron2web forward pass disagrees with the engine" >&2; exit 1; }

strip() { sed 's/\r$//' "$1"; }

fail=0
for f in data.txt key.txt inputs.txt stdout.txt \
         cat_data.txt cat_key.txt cat_inputs.txt cat_stdout.txt \
         ref_data.txt ref_key.txt ref_inputs.txt ref_stdout.txt \
         xor.html xor_stdout.txt; do
    if [ $BLESS -eq 1 ]; then
        strip $f > ../expected_$f
        echo "blessed expected_$f"
    elif diff <(strip ../expected_$f) <(strip $f); then
        echo "OK: $f matches"
    else
        echo "FAIL: $f differs (see tests/tools/runs/$f)" >&2
        fail=1
    fi
done
exit $fail
