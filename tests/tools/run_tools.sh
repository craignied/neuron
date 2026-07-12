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

strip() { sed 's/\r$//' "$1"; }

fail=0
for f in data.txt key.txt inputs.txt stdout.txt; do
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
