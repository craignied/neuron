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
grep -q '"az":' train.json && fail "fabricated a binormal fit on 4 exemplars"
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

# Auto algorithm selection (ROADMAP 2 Phase 2): probe all three optimizers
#    from identical weights, adopt the winner, continue training to maxiter.
#    The result carries the structured selection and the report carries the
#    human-readable decision summary.
curl -s -X POST "$URL/api/train" -d "algorithm=auto&maxiter=2000&seed=42" > auto.json
grep -q '"ok":true' auto.json || fail "algorithm=auto train"
grep -q '"autoAlgo":{"selected":' auto.json || fail "no autoAlgo selection block"
[ $(grep -o '"algorithm":' auto.json | wc -l) -ge 3 ] || fail "want 3 probes in autoAlgo"
grep -q 'Auto algorithm selection' auto.json || fail "no probe summary in the report"
grep -q 'Selected: ' auto.json || fail "no decision line in the report"
# The user-visible message must name the winner (the page shows the message,
#    not the report)
grep -q '"message":"auto selected ' auto.json || fail "winner not named in the message"

# The binormal path proper. Everything above is too small to reach it (4
#    exemplars), so nothing above says anything about the ROC statistics --
#    this is the case that does. The Hosmer-Lemeshow low-birth-weight set is
#    committed in the repo and is large enough to fit, so the panel must carry
#    the Az, the operating-point count qualifying it, and the bootstrap interval.
cp ../../../docs/datasets/low-birth-weight/lowbwt2-2train.txt .
curl -s -X POST "$URL/api/load" -d "mode=raw&path=lowbwt2-2train.txt&fraction=0.25" \
    | grep -q '"ok":true' || fail "load low-birth-weight raw"
curl -s -X POST "$URL/api/model" -d "type=logistic" \
    | grep -q '"ok":true' || fail "logistic model on low-birth-weight"
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=2000&seed=42" > lbw.json
grep -q '"ok":true' lbw.json || fail "low-birth-weight train"
grep -q '"binormal":{"az":' lbw.json || fail "no binormal fit where one is possible"
# The report gives ONE fit now -- the best-p/best-AUC pair went with the binning
grep -q '"bestP":' lbw.json && fail "best-p/best-AUC pair should be gone"
grep -q '"nBins":' lbw.json && fail "bin count should be gone"
# The fit is qualified by the points behind it, not by a bin count
grep -q '"points":' lbw.json || fail "no operating-point count on the binormal fit"
grep -q '"points":0' lbw.json && fail "binormal fit reported with no operating points"
# The Az carries the bootstrap interval, with the resamples behind it
grep -q '"ci":{"lo":' lbw.json || fail "no bootstrap CI on the binormal fit"
grep -q '"resamples":' lbw.json || fail "no resample count on the binormal CI"
# The retired delta-method SE must not reappear as a bare fit SE
python3 - <<'PY' || fail "binormal fit still carries a non-bootstrap se"
import json, sys
d = json.load(open("lbw.json"))
for setname, s in (d.get("stats") or {}).items():
    b = (s or {}).get("binormal")
    if b and "se" in b:
        sys.exit(1)
PY

# --- Plateau auto-stop (ROADMAP 2 Phase 3) ---------------------------------
# On the low-birth-weight logistic (loaded above), an aggressive plateau
#    detector must stop the run early, reporting stopReason "plateau" and the
#    report line. The SAME run WITHOUT autostop must NOT -- it converges to
#    grad_max -- which proves the feature is off by default and is doing real
#    work, not relabeling a stop that would have happened anyway.
curl -s -X POST "$URL/api/model" -d "type=logistic" > /dev/null
curl -s -X POST "$URL/api/train" \
    -d "algorithm=1&maxiter=2000&seed=42&autostop=1&autostop_tol=0.001&autostop_window=10" \
    > plateau.json
grep -q '"ok":true' plateau.json || fail "autostop train"
grep -q '"stopReason":"plateau"' plateau.json || fail "autostop run should stop on plateau"
grep -q 'The error plateaued' plateau.json || fail "no plateau line in the report"
curl -s -X POST "$URL/api/model" -d "type=logistic" > /dev/null
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=2000&seed=42" > plateau_ctl.json
grep -q '"stopReason":"plateau"' plateau_ctl.json \
    && fail "a run without autostop must never plateau-stop"
# Invalid auto-stop parameters are rejected cleanly, not asserted away (asserts
#    vanish in release builds, so the handler must guard the ranges itself)
curl -s -X POST "$URL/api/train" \
    -d "algorithm=1&maxiter=100&seed=42&autostop=1&autostop_tol=5" \
    | grep -q '"ok":false' || fail "autostop_tol out of range should be rejected"
curl -s -X POST "$URL/api/train" \
    -d "algorithm=1&maxiter=100&seed=42&autostop=1&autostop_window=1" \
    | grep -q '"ok":false' || fail "autostop_window < 2 should be rejected"

# --- Train-panel parity controls (GUI/CLI parity, 2026-07-19) --------------
# Learning rate, weight decay, batch/epoch, the stopping conditions, and the
# print counter are now settable through /api/train, matching the CLI model +
# stopping-conditions menus. Behavioral proof: a change-in-error limit of 0.5
# stops almost immediately (min_change), and the run header must announce the
# condition -- both prove the param reached the engine, not just parsed.
curl -s -X POST "$URL/api/model" -d "type=logistic" > /dev/null
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=2000&seed=42&change=0.5" > tp.json
grep -q '"ok":true' tp.json || fail "train with change limit"
grep -q '"stopReason":"min_change"' tp.json || fail "change limit did not take effect"
grep -q 'change in error over 1 iteration becomes less than 0.5' tp.json \
    || fail "run header did not record the change-limit condition"
# Learning rate + weight decay + batch/epoch accepted together
curl -s -X POST "$URL/api/model" -d "type=logistic" > /dev/null
curl -s -X POST "$URL/api/train" \
    -d "algorithm=1&maxiter=200&seed=42&eta=0.5&weight_decay=1&decay=0.001&batch_epoch=1" \
    | grep -q '"ok":true' || fail "train with eta / weight-decay / batch-epoch"
# Out-of-range values are rejected cleanly (asserts vanish in release builds)
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=10&eta=5" \
    | grep -q '"ok":false' || fail "learning rate > 1 should be rejected"
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=10&weight_decay=1&decay=9" \
    | grep -q '"ok":false' || fail "weight decay lambda > 1 should be rejected"
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=10&errwindow=1" \
    | grep -q '"ok":false' || fail "error window <= 1 should be rejected"
# A bare train (no parity fields) is unaffected -- the model keeps its defaults
curl -s -X POST "$URL/api/model" -d "type=logistic" > /dev/null
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=2000&seed=42" \
    | grep -q '"stopReason":"grad_max"' || fail "a bare train must keep default stopping"

# --- Model-panel parity controls (GUI/CLI parity, 2026-07-19) --------------
# Bias toggle (SimpleProp vs BareProp), multiple hidden layers (BackProp), the
# output error function, and loading a saved network -- the CLI model menu.
curl -s -X POST "$URL/api/load" -d "mode=train&path=xor_discrete.set" > /dev/null
curl -s -X POST "$URL/api/model" -d "type=simpleprop&hidden=3&bias=0" \
    | grep -q 'BareProp' || fail "bias off should build a BareProp"
curl -s -X POST "$URL/api/model" -d "type=simpleprop&hidden=4,2" \
    | grep -q 'BackProp' || fail "multiple hidden layers should build a BackProp"
curl -s -X POST "$URL/api/model" -d "type=simpleprop&hidden=3&errfunc=lms" \
    | grep -q 'LMS' || fail "output error function (lms) not honored"
curl -s -X POST "$URL/api/model" -d "type=simpleprop&hidden=3&errfunc=xentropy" \
    | grep -q '"ok":true' || fail "errfunc=xentropy on discrete data should work"
curl -s -X POST "$URL/api/model" -d "type=simpleprop&hidden=5,0" \
    | grep -q '"ok":false' || fail "a non-positive hidden layer must be rejected"
# Load a saved network from a file: type read from line 1, weights load, and a
# loaded network is a trained one (it produces guesses).
cp ../../oracle/xor_net.txt .
curl -s -X POST "$URL/api/model" -F "file=@xor_net.txt;filename=net.txt" -F "mode=load" \
    | grep -q 'loaded SimpleProp' || fail "load saved network from file"
curl -s "$URL/api/save/train_guesses" -o dl_loaded
[ -s dl_loaded ] || fail "a loaded network should produce guesses"

# --- Discriminant function analysis (GUI/CLI parity, main menu 4) ----------
# Linear and quadratic DFA on the loaded discrete dataset -> report + ROC +
# stats, and it does not disturb the model. Load the fixed training set
# (mode=train, no random split) so the ROC Az below is deterministic.
curl -s -X POST "$URL/api/load" -d "mode=train&path=lowbwt2-2train.txt" > /dev/null
curl -s -X POST "$URL/api/dfa" -d "type=linear" > dfa.json
grep -q '"ok":true' dfa.json || fail "linear DFA"
grep -q 'running LDFA' dfa.json || fail "no LDFA report in the output"
grep -q '"stats":' dfa.json || fail "DFA should return the stats panel"
# DFA now stores a GRADED discriminant score, so it has a real statistical ROC
# Az -- not the single-operating-point degeneracy of the old hard 0/1 guess
# (which gave binormal:null). Proven to fail against the pre-graded code.
grep -q '"binormal":{"az":' dfa.json || fail "DFA should have a statistical ROC Az now"
grep -q '"binormal":null' dfa.json && fail "DFA binormal must not be null on a fittable set"
curl -s -X POST "$URL/api/dfa" -d "type=quadratic" \
    | grep -q 'quadratic discriminant' || fail "quadratic DFA"
curl -s -X POST "$URL/api/dfa" -d "type=cubic" \
    | grep -q '"ok":false' || fail "an unknown DFA type must be rejected"

# --- Async training (ROADMAP 2 Phase 1b) -----------------------------------
# A slow, continuous-outcome regression set: iterations are heavy enough that
#    a 50M-iteration budget cannot finish during the test, and a non-discrete
#    outcome keeps the cancelled run's epilogue cheap (no classification
#    statistics, no ROC bootstrap).
PY=python3; command -v python3 >/dev/null || PY=python
$PY - <<'PY'
import random
random.seed(3)
def rows(n):
    for i in range(n):
        a, b, c = (random.uniform(-0.9, 0.9) for _ in range(3))
        y = 0.5 + 0.3 * a - 0.2 * b + 0.1 * c + random.gauss(0, 0.1)
        y = min(max(y, 0.05), 0.95)
        yield "%.6f %.6f %.6f %.6f\n" % (a, b, c, y)
with open("slow_train.set", "w", newline="\n") as f:
    f.writelines(rows(3750))
with open("slow_test.set", "w", newline="\n") as f:
    f.writelines(rows(1250))
PY
curl -s -X POST "$URL/api/load" \
    -d "mode=train&path=slow_train.set&testpath=slow_test.set&discrete=0" \
    | grep -q '"ok":true' || fail "load slow regression pair (discrete=0)"
curl -s -X POST "$URL/api/model" -d "type=simpleprop&hidden=8" \
    | grep -q '"ok":true' || fail "model on slow set"

# async=1 returns immediately with the run still going
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=50000000&seed=42&async=1" \
    | grep -q '"ok":true' || fail "async train did not start"
curl -s "$URL/api/train/status" | grep -q '"running":true' \
    || fail "status should say running right after an async start"

# While it runs, every engine-touching endpoint refuses with 409 + busy:true
busy=$(curl -s -w '\n%{http_code}' -X POST "$URL/api/load" -d "mode=train&path=xor_discrete.set")
echo "$busy" | grep -q '"busy":true' || fail "no busy flag while training"
echo "$busy" | tail -1 | grep -q '409' || fail "busy refusal should be HTTP 409"

# The observer publishes a decimated error series while training
sleep 1
curl -s "$URL/api/train/status" > status.json
grep -q '"iter":\[[0-9]' status.json || fail "no error series while training"
grep -q '"train":\[' status.json || fail "no training-error series"

# Stop: the run finishes normally (report and all) with stopReason cancelled
curl -s -X POST "$URL/api/train/stop" | grep -q '"ok":true' || fail "stop endpoint"
for i in $(seq 1 120); do
    curl -s "$URL/api/train/status" > status.json
    grep -q '"running":false' status.json && break
    sleep 0.5
done
grep -q '"running":false' status.json || fail "run never stopped after cancel"
grep -q '"stopReason":"cancelled"' status.json || fail "cancelled run should say so"
grep -q '"ok":true' status.json || fail "cancelled run should still be a completed run"
# Stop with nothing running refuses cleanly
curl -s -X POST "$URL/api/train/stop" | grep -q '"ok":false' \
    || fail "stop should refuse when nothing is running"

# An async run left to finish carries the same full result as blocking mode
curl -s -X POST "$URL/api/load" -d "mode=train&path=xor_discrete.set" \
    | grep -q '"ok":true' || fail "engine not reusable after a cancelled run"
curl -s -X POST "$URL/api/model" -d "type=simpleprop&hidden=3" > /dev/null
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=100000&seed=42&async=1" \
    | grep -q '"ok":true' || fail "second async train did not start"
for i in $(seq 1 240); do
    curl -s "$URL/api/train/status" > status2.json
    grep -q '"running":false' status2.json && break
    sleep 0.5
done
grep -q '"running":false' status2.json || fail "async run never completed"
grep -q '"stopReason":"' status2.json || fail "no stopReason on the async result"
grep -q '"stats":' status2.json || fail "async result missing the stats object"

# Async + auto together: the adoption REPLACES the model mid-job, so the
#    worker must re-derive its pointers -- this is the path that would break
curl -s -X POST "$URL/api/train" -d "algorithm=auto&maxiter=200&seed=42&async=1" \
    | grep -q '"ok":true' || fail "async auto train did not start"
for i in $(seq 1 240); do
    curl -s "$URL/api/train/status" > status3.json
    grep -q '"running":false' status3.json && break
    sleep 0.5
done
grep -q '"running":false' status3.json || fail "async auto run never completed"
grep -q '"autoAlgo":{"selected":' status3.json || fail "async result missing autoAlgo"

# --- Per-action audit log (every user action logged, 2026-07-19) -----------
# Every GUI action lands in neuron_actions.log beside the data, timestamped,
# with the exact parameter values it carried. The session above drove load,
# model, train (with params), dfa, and saves -- all must be there.
[ -f neuron_actions.log ] || fail "no per-action audit log was written"
grep -qE '^[0-9-]+T[0-9:]+ load '  neuron_actions.log || fail "load not audited"
grep -qE '^[0-9-]+T[0-9:]+ model ' neuron_actions.log || fail "model not audited"
grep -qE '^[0-9-]+T[0-9:]+ train ' neuron_actions.log || fail "train not audited"
grep -q 'algorithm=' neuron_actions.log || fail "train parameters not recorded in the audit log"
grep -qE '^[0-9-]+T[0-9:]+ dfa '   neuron_actions.log || fail "dfa not audited"

echo "OK: GUI endpoints (version, page, load incl. pre-split pair, model, train + ROC + full stats JSON, /api/stats, binormal fits + null when impossible, logistic Wald/condition number, regress, saves, plateau auto-stop + control + validation, train-panel parity controls (learning rate/weight decay/batch-epoch/stopping conditions/print counter) + behavioral proof + validation, model-panel parity (bias->BareProp/multi-layer->BackProp/error function/load-network), DFA (linear/quadratic + report/stats), async train/status/stop + 409 busy + cancel, algorithm=auto blocking + async)"
