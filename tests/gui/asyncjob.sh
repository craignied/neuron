#!/bin/bash
# Characterization of the GUI's asynchronous-job lifecycle (refactor item 13).
#
# This is DELIBERATELY a separate script from smoke.sh. smoke.sh characterizes
# what each endpoint returns; this one characterizes the machinery underneath
# all four long jobs -- the single worker thread, the busy gate, the
# publish-then-clear ordering the status endpoint depends on, the reap of a
# finished worker, and the reset that keeps one job's progress out of the next
# one's status. A hang or crash here must not take smoke.sh's coverage with it.
#
# Everything is driven from one python3 process (stdlib only, like tools/) so
# that every wait has a deadline and every response is really parsed as JSON.
# There are no sleep-based correctness assertions: a sleep only paces a poll
# loop, and every assertion is about what was observed, never about how long
# something took.
#
# WHAT THIS CANNOT REACH, measured rather than assumed -- see docs/refactor_audit.md
# item 13. runOnWorker's two exception handlers and the CLI's main boundary have
# no automated coverage, because no legitimate request makes an async body throw:
# runTrainJob's exceptions are caught by runTrainingAndBuildResult, stepwise
# catches its own RegressNetErr, and OBD and CV report their refusals as
# ok:false results rather than throwing. That is the D9 debt item 13 exists to
# pay, and it is payable only once the launcher is linkable.
set -e
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)

BIN=$ROOT/build/neuron
[ -x "$BIN" ] || BIN=$ROOT/build/Release/neuron.exe
[ -x "$BIN" ] || { echo "FAIL: no neuron binary to test" >&2; exit 1; }

PY=python3; command -v python3 >/dev/null || PY=python

mkdir -p runs/asyncjob && cd runs/asyncjob
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

$PY ../../asyncjob_driver.py "$URL"
rc=$?

# The server must still be answering after every case. A worker that terminated
#    the process is the failure this whole file exists to notice, and a driver
#    that failed for some other reason must not be reported as that.
if ! curl -s --max-time 5 "$URL/api/version" | grep -q .; then
    echo "FAIL: the server stopped answering during the run" >&2
    tail -20 gui.out >&2
    exit 1
fi

exit $rc
