#!/bin/bash
# Endpoint smoke test for neuron --gui: start the server on its OS-assigned
# port, drive the JSON API through a complete XOR training run, and require
# a served page, ok:true responses, and an ROC area in the train JSON.
set -e
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)

BIN=$ROOT/build/neuron
[ -x "$BIN" ] || BIN=$ROOT/build/Release/neuron.exe

mkdir -p runs && cd runs
cp ../../oracle/xor_discrete.set .

"$BIN" --gui --no-browser > gui.out 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT

# Wait for the server to announce its port
URL=""
for i in $(seq 1 50); do
    URL=$(grep -o 'http://127.0.0.1:[0-9]*' gui.out | head -1)
    [ -n "$URL" ] && break
    sleep 0.2
done
[ -n "$URL" ] || { echo "FAIL: server URL never appeared" >&2; cat gui.out; exit 1; }

fail() { echo "FAIL: $1" >&2; exit 1; }

curl -s "$URL/api/version" | grep -q "neuron" || fail "version endpoint"
curl -s "$URL/" | grep -q "<title>neuron</title>" || fail "page not served"

# No inputs/outputs given: the server derives them from the file's columns
#    (xor_discrete.set is 2 inputs + 1 output) — the page relies on this
load1=$(curl -s -X POST "$URL/api/load" -d "mode=train&path=xor_discrete.set")
echo "$load1" | grep -q '"ok":true' || fail "load endpoint (path)"
echo "$load1" | grep -q '2 inputs, 1 output' || fail "input count not derived from file"

# The page's file picker uploads content as multipart — test that path too.
#    An explicit inputs= override must still be honored (scripts rely on it).
curl -s -X POST "$URL/api/load" -F "file=@xor_discrete.set;filename=uploaded.set" \
    -F mode=train -F inputs=2 \
    | grep -q '"ok":true' || fail "load endpoint (multipart upload)"
[ -f uploaded.set ] || fail "upload was not saved beside the server"

curl -s -X POST "$URL/api/model" -d "type=simpleprop&hidden=3" \
    | grep -q '"ok":true' || fail "model endpoint"

curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=100000&seed=42" \
    > train.json
grep -q '"ok":true' train.json || fail "train endpoint"
grep -q '"area":' train.json || fail "no ROC area in train response"
grep -q '"acc":' train.json || fail "no structured stats (acc) in train response"
grep -q '"se":' train.json || fail "no ROC SE in train response"
# Full statistics panel: the stats object with confusion counts (Phase 1a)
grep -q '"stats":' train.json || fail "no stats object in train response"
grep -q '"confusion":' train.json || fail "no confusion table in stats"
grep -q '"tp":' train.json || fail "no confusion counts in stats"
# Four exemplars cannot support a binormal fit -- the report says so ("Cannot
#    calculate ROC statistically"), so the panel must say so too, with null.
#    It used to publish the search's zero-initialized fit here instead: an Az of
#    0, meaning perfectly anti-predictive, presented as a real result while the
#    report alongside it declined to give one.
grep -q '"binormal":null' train.json || fail "binormal must be null when no fit is possible"
grep -q '"bestP":' train.json && fail "fabricated a binormal fit on 4 exemplars"
# /api/stats recomputes the same object without retraining
curl -s "$URL/api/stats" > stats.json
grep -q '"ok":true' stats.json || fail "stats endpoint"
grep -q '"confusion":' stats.json || fail "no confusion table from /api/stats"
grep -q '"binormal":null' stats.json || fail "binormal must be null from /api/stats too"

# Train continues from the current weights: a second Train reports so
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=1&seed=42" \
    | grep -q 'continued training' || fail "second train should continue, not restart"
# Randomize weights explicitly (with and without a seed), then Train restarts
curl -s -X POST "$URL/api/randomize" -d "seed=7" \
    | grep -q '"ok":true' || fail "randomize endpoint (seeded)"
curl -s -X POST "$URL/api/randomize" -d "" \
    | grep -q '"ok":true' || fail "randomize endpoint (no seed)"
# Re-train (fresh weights just set count as continued) to restore trained state
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=100000&seed=42" \
    | grep -q '"ok":true' || fail "train after randomize"

# Stepwise regression on the trained network (XOR: 2 inputs -> 2 variables)
curl -s -X POST "$URL/api/regress" \
    --data-urlencode "structure=0;1" \
    --data-urlencode "direction=reverse" \
    --data-urlencode "threshold=0.05" > regress.json
grep -q '"ok":true' regress.json || fail "regress endpoint"
grep -q 'Reverse regressing' regress.json || fail "no regression report"
# Untrained-model guard: a fresh model must refuse regression
curl -s -X POST "$URL/api/model" -d "type=simpleprop&hidden=3" > /dev/null
curl -s -X POST "$URL/api/regress" \
    --data-urlencode "structure=0;1" --data-urlencode "direction=reverse" \
    --data-urlencode "threshold=0.05" | grep -q '"ok":false' \
    || fail "regress should refuse an untrained model"
# Re-train so the later save checks still have a trained network
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=100000&seed=42" \
    | grep -q '"ok":true' || fail "re-train after regress"

# Session artifacts: written into the workspace AND served as downloads
curl -s "$URL/api/save/network" -o dl_network
head -1 dl_network | grep -q "SimpleProp" || fail "network download"
[ -f network.txt ] || fail "network.txt not written to the workspace"
curl -s "$URL/api/save/train_guesses" -o dl_guesses
[ -s dl_guesses ] || fail "train guesses download"
curl -s "$URL/api/save/report" -o dl_report
grep -q "Iteration" dl_report || fail "report download"
# No test set was loaded, so its artifact must refuse cleanly
curl -s "$URL/api/save/test_set" | grep -q '"ok":false' \
    || fail "test_set should refuse when no test set exists"

# A matched, pre-split train/test pair loads through one call (both uploads)
curl -s -X POST "$URL/api/load" \
    -F "file=@xor_discrete.set;filename=pair_train.set" \
    -F "testfile=@xor_discrete.set;filename=pair_test.set" \
    -F mode=train > pair.json
grep -q '"ok":true' pair.json || fail "pre-split pair load"
grep -q 'test exemplars' pair.json || fail "test set not loaded from the pair"
[ -f pair_test.set ] || fail "uploaded test set not saved beside the server"

# Logistic regression exposes the Wald table and condition number (Phase 1a).
#    xor_discrete.set has a discrete outcome, so logistic applies.
curl -s -X POST "$URL/api/model" -d "type=logistic" \
    | grep -q '"ok":true' || fail "logistic model endpoint"
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=5000&seed=42" > logi.json
grep -q '"ok":true' logi.json || fail "logistic train"
grep -q '"logistic":' logi.json || fail "no logistic stats block"
grep -q '"waldP":' logi.json || fail "no Wald p-values in logistic stats"
grep -q '"condNumber":' logi.json || fail "no condition number in logistic stats"

# The binormal path proper. Everything above is too small to reach it (4
#    exemplars), so nothing above says anything about the ROC statistics --
#    this is the case that does. The Hosmer-Lemeshow low-birth-weight set is
#    committed in the repo and is large enough to be binned, so the panel must
#    carry both searched fits, each labelled with the bin count that produced
#    it (an Az is meaningless without it), plus the bootstrap interval.
cp ../../../docs/datasets/low-birth-weight/lowbwt2-2train.txt .
curl -s -X POST "$URL/api/load" -d "mode=raw&path=lowbwt2-2train.txt&fraction=0.25" \
    | grep -q '"ok":true' || fail "load low-birth-weight raw"
curl -s -X POST "$URL/api/model" -d "type=logistic" \
    | grep -q '"ok":true' || fail "logistic model on low-birth-weight"
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=2000&seed=42" > lbw.json
grep -q '"ok":true' lbw.json || fail "low-birth-weight train"
grep -q '"bestP":' lbw.json || fail "no best-p binormal fit where one is possible"
grep -q '"bestAUC":' lbw.json || fail "no best-AUC binormal fit where one is possible"
grep -q '"nBins":' lbw.json || fail "no bin count on the binormal fits"
# A fit must not report the zero-initialized bin count it never searched with
grep -q '"nBins":0' lbw.json && fail "binormal fit reported with no binning"
# Each binormal Az carries the bootstrap interval, with the resamples behind it
grep -q '"ci":{"lo":' lbw.json || fail "no bootstrap CI on the binormal fits"
grep -q '"resamples":' lbw.json || fail "no resample count on the binormal CI"
# The retired delta-method SE must not reappear as a bare fit SE
python3 - <<'PY' || fail "binormal fit still carries a non-bootstrap se"
import json, sys
d = json.load(open("lbw.json"))
for setname, s in (d.get("stats") or {}).items():
    b = (s or {}).get("binormal")
    if not b: continue
    for which, f in b.items():
        if f and "se" in f:
            sys.exit(1)
PY

echo "OK: GUI endpoints (version, page, load incl. pre-split pair, model, train + ROC + full stats JSON, /api/stats, binormal fits + null when impossible, logistic Wald/condition number, regress, saves)"
