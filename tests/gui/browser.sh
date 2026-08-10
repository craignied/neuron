#!/bin/bash
# Drive the GUI page in a REAL browser.  Manual; not part of the release gate.
#
# Same shape as smoke.sh and strictparse.sh -- start a server in a scratch run
# directory, hand its URL to a python driver, tear it down -- but this one drives
# Chrome with Playwright instead of curl, because the page's JavaScript is the
# one part of the GUI that no HTTP-level test can see.  See the header of
# browser_driver.py for what that buys and what it costs.
#
#   ./tests/gui/browser.sh            headless
#   ./tests/gui/browser.sh --headed   watch it click
#
# HOW TO DRIVE THE PAGE, if you are reaching for something else first: use this.
# The "Claude in Chrome" extension bridge is NOT available in every environment
# (in the VSCode extension environment the command that connects it does not
# exist at all, so no browser tools ever appear).  Playwright needs no bridge,
# launches the Chrome that is already installed, and works everywhere.
set -e
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)

BIN=$ROOT/build/neuron
[ -x "$BIN" ] || BIN=$ROOT/build/Release/neuron.exe
[ -x "$BIN" ] || { echo "FAIL: no neuron binary to test" >&2; exit 1; }

PY=python3; command -v python3 >/dev/null || PY=python

rm -rf runs/browser
mkdir -p runs/browser && cd runs/browser
cp ../../../../docs/datasets/low-birth-weight/lowbwt2-2train.txt .

# The GUI writes neuron.log, model.txt and any saved session file into its own
#    working directory, which is why every one of these harnesses gets its own.
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

$PY ../../browser_driver.py "$URL" "$@"
rc=$?

# The server must still be answering: a page interaction that killed the process
#    is a failure this file must not report as an assertion failure.
if ! curl -s --max-time 5 "$URL/api/version" | grep -q .; then
    echo "FAIL: the server stopped answering during the run" >&2
    tail -20 gui.out >&2
    exit 1
fi

exit $rc
