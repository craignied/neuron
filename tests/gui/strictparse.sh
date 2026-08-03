#!/bin/bash
# How every GUI/HTTP handler reads its request fields (ROADMAP 4 item B9).
#
# Separate from smoke.sh, which characterizes what each endpoint RETURNS, and
# from asyncjob.sh, which characterizes the machinery under the four long jobs.
# This one characterizes the request boundary: the omission rules, the
# present-but-empty rules, the domain refusals, and -- before B9 -- the values
# that are silently misread.
#
# Everything is driven from one python3 process (stdlib only, like tools/) so
# that every wait has a deadline and every response is really parsed as JSON.
# No assertion is about how long anything took.
set -e
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)

BIN=$ROOT/build/neuron
[ -x "$BIN" ] || BIN=$ROOT/build/Release/neuron.exe
[ -x "$BIN" ] || { echo "FAIL: no neuron binary to test" >&2; exit 1; }

PY=python3; command -v python3 >/dev/null || PY=python

rm -rf runs/strictparse
mkdir -p runs/strictparse && cd runs/strictparse
cp ../../../oracle/xor_discrete.set .
cp ../../../../docs/datasets/low-birth-weight/lowbwt2-2train.txt .

"$BIN" --gui --no-browser > gui.out 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT

URL=""
for i in $(seq 1 50); do
    URL=$(grep -o 'http://127.0.0.1:[0-9]*' gui.out | head -1)
    [ -n "$URL" ] && break
    sleep 0.2
done
[ -n "$URL" ] || { echo "FAIL: server URL never appeared" >&2; cat gui.out; exit 1; }

$PY ../../strictparse_driver.py "$URL"
rc=$?

# The server must still be answering. A handler that killed the process is a
#    failure this file must not report as an assertion failure.
if ! curl -s --max-time 5 "$URL/api/version" | grep -q .; then
    echo "FAIL: the server stopped answering during the run" >&2
    tail -20 gui.out >&2
    exit 1
fi

exit $rc
