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
PY=python3; command -v python3 >/dev/null || PY=python

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

# A no-bias BackProp survives a save/load round trip WITH ITS WEIGHTS. The
# assertion is error continuity, because the message-level checks alone cannot
# see this bug: BackProp::load reads its line-2 bias flag but never APPLIES
# it, so without setBias before load() the weight matrices are sized for bias,
# operator>> desyncs on the file, load still returns true -- and the trained
# net silently comes back as garbage (error snaps to ln 2 ~ 0.693). Measured
# against the pre-fix handler: 0.587 -> 0.692; with the fix: 0.587 -> 0.587.
curl -s -X POST "$URL/api/load" -d "mode=train&path=lowbwt2-2train.txt" > /dev/null
curl -s -X POST "$URL/api/model" -d "type=simpleprop&hidden=4,2&bias=0" \
    | grep -q 'BackProp' || fail "no-bias multi-layer should build a BackProp"
curl -s -X POST "$URL/api/train" -d "algorithm=1&autostep=1&maxiter=3000&seed=42" \
    > nb_pre.json
grep -q '"ok":true' nb_pre.json || fail "train the no-bias BackProp"
curl -s "$URL/api/save/network" -o /dev/null
sed -n 2p network.txt | grep -q '^0' || fail "saved network should record bias absent"
curl -s -X POST "$URL/api/model" -d "mode=load&path=network.txt" \
    | grep -q 'loaded BackProp' || fail "no-bias BackProp did not load back"
curl -s -X POST "$URL/api/train" \
    -d "algorithm=1&autostep=0&batch_epoch=0&eta=0.0001&maxiter=1&seed=42" \
    > nb_post.json
grep -q 'continued training' nb_post.json || fail "loaded no-bias BackProp should train on"
$PY - <<'PY' || fail "no-bias BackProp weights did not survive the save/load round trip"
import json, re
err = lambda f: float(re.search(r"final error ([0-9.eE+-]+)",
                                json.load(open(f))["message"]).group(1))
e1, e2 = err("nb_pre.json"), err("nb_post.json")
assert abs(e2 - e1) / e1 < 0.05, (e1, e2)
PY

# Multipart posts must honor the log toggles too (the page's file-upload posts
# are multipart, where req.has_param alone is false and the toggles used to be
# silently ignored). With log_lastop=0 a training run must NOT write model.txt.
rm -f model.txt
curl -s -X POST "$URL/api/model" -F type=simpleprop -F hidden=3 -F log_lastop=0 \
    | grep -q '"ok":true' || fail "multipart model create"
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=100&seed=42" \
    | grep -q '"ok":true' || fail "train with lastop logging off"
[ -f model.txt ] && fail "log_lastop=0 sent as multipart was ignored"

# Logistic regression is batch/epoch by definition: the CLI refuses to turn it
# off, so the API must refuse batch_epoch=0 rather than mistrain.
curl -s -X POST "$URL/api/model" -d "type=logistic" > /dev/null
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=50&batch_epoch=0" \
    | grep -q '"ok":false' || fail "batch_epoch=0 on logistic must be refused"

# --- Discriminant function analysis (GUI/CLI parity, main menu 4) ----------
# Linear and quadratic DFA on the loaded discrete dataset -> report + ROC +
# stats, and it does not disturb the model. Load the fixed training set
# (mode=train, no random split) so the ROC Az below is deterministic.
curl -s -X POST "$URL/api/load" -d "mode=train&path=lowbwt2-2train.txt" > /dev/null
# No DFA has run since that load, so its guesses must refuse cleanly
curl -s "$URL/api/save/dfa_train_guesses" | grep -q '"ok":false' \
    || fail "dfa guesses should refuse before any DFA has run"
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
# The analysis's guesses stay savable after the run (the CLI offers this save
# right after a DFA; the handler used to destroy the analysis object)
curl -s "$URL/api/save/dfa_train_guesses" -o dl_dfa_guesses
[ -s dl_dfa_guesses ] || fail "DFA train guesses download"
grep -q '"ok":false' dl_dfa_guesses && fail "DFA train guesses refused after a run"
[ -f dfa_train_guesses.txt ] || fail "dfa_train_guesses.txt not written to the workspace"
# No test set on this dataset -> the test guesses refuse with the engine's message
curl -s "$URL/api/save/dfa_test_guesses" | grep -q '"ok":false' \
    || fail "dfa test guesses should refuse without a test set"

# --- Dataset characteristics + ROC reporting (CLI dataset menus 11/12/13) ---
# The load-time params: invalid values are rejected before anything loads;
# trap_thresholds is proven behaviorally (5 thresholds -> exactly 5 ROC curve
# points in the train JSON, not the auto-tuned count).
curl -s -X POST "$URL/api/load" -d "mode=train&path=lowbwt2-2train.txt&out_lower=0.2" \
    | grep -q '"ok":false' || fail "output bounds on a discrete outcome must be rejected"
curl -s -X POST "$URL/api/load" -d "mode=train&path=lowbwt2-2train.txt&threshold=1.5" \
    | grep -q '"ok":false' || fail "threshold outside (0,1) must be rejected"
curl -s -X POST "$URL/api/load" -d "mode=train&path=lowbwt2-2train.txt&trap_thresholds=1" \
    | grep -q '"ok":false' || fail "trap_thresholds < 2 must be rejected"
curl -s -X POST "$URL/api/load" -d "mode=train&path=lowbwt2-2train.txt&roc_report=maybe" \
    | grep -q '"ok":false' || fail "roc_report must be both or either"
curl -s -X POST "$URL/api/load" -d "mode=train&path=lowbwt2-2train.txt&in_lower=1&in_upper=-1" \
    | grep -q '"ok":false' || fail "inverted input bounds must be rejected"
curl -s -X POST "$URL/api/load" \
    -d "mode=train&path=lowbwt2-2train.txt&threshold=0.4&in_lower=-0.8&in_upper=0.8&history=1&trap_thresholds=5" \
    | grep -q '"ok":true' || fail "valid characteristics + ROC settings must load"
curl -s -X POST "$URL/api/model" -d "type=logistic" > /dev/null
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=500&seed=42" > tp5.json
$PY - <<'PY' || fail "trap_thresholds=5 did not reach the ROC sweep"
import json
d = json.load(open("tp5.json"))
assert len(d["roc"]["train"]["x"]) == 5, len(d["roc"]["train"]["x"])
PY

# The whole-number split form: test_n places EXACTLY n exemplars in the test
# set (randomizeD truncates ratio*N, so a fraction cannot promise this)
curl -s -X POST "$URL/api/load" -d "mode=raw&path=lowbwt2-2train.txt&test_n=47" > testn.json
grep -q '"ok":true' testn.json || fail "test_n load"
grep -q '47 test exemplars' testn.json || fail "test_n=47 should yield exactly 47"

# --- Multi-output (the CLI supports it; the GUI must too) -------------------
# A 2-output one-hot pair: loads with outputs=2, models as a BackProp, trains,
# and DFA reports per-set accuracy -- all accuracy-only, as in the CLI. The
# 1-output-only machinery must refuse it honestly, not crash.
$PY - <<'PY'
import random
random.seed(7)
def rows(n):
    for _ in range(n):
        x = [random.uniform(-0.9, 0.9) for _ in range(4)]
        cls = 1 if x[0] + 0.5 * x[1] > 0 else 0
        o = "1 0" if cls == 0 else "0 1"
        yield " ".join("%.6f" % v for v in x) + " " + o + " \n"
with open("mo_train.set", "w", newline="\n") as f:
    f.writelines(rows(40))
with open("mo_test.set", "w", newline="\n") as f:
    f.writelines(rows(20))
PY
curl -s -X POST "$URL/api/load" -d "mode=train&path=mo_train.set&testpath=mo_test.set&outputs=2" \
    > mo.json
grep -q '"ok":true' mo.json || fail "multi-output load"
grep -q '2 outputs' mo.json || fail "output count not reported"
curl -s -X POST "$URL/api/model" -d "type=simpleprop&hidden=3" > mo_model.json
grep -q 'BackProp 4-3-2' mo_model.json || fail "multi-output should build a BackProp"
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=100&seed=42" > mo_train.json
grep -q '"ok":true' mo_train.json || fail "multi-output train"
grep -q 'classification accuracy in the training set' mo_train.json \
    || fail "no multi-output accuracy report"
curl -s -X POST "$URL/api/dfa" -d "type=linear" > mo_dfa.json
grep -q '"ok":true' mo_dfa.json || fail "multi-output DFA"
grep -q 'Classification accuracy in the test set' mo_dfa.json \
    || fail "no multi-output DFA test accuracy"
# Logistic is 1-output by definition -- the CLI's own refusal, kept
curl -s -X POST "$URL/api/model" -d "type=logistic" \
    | grep -q '"ok":false' || fail "logistic must refuse a multi-output dataset"

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

echo "OK: GUI endpoints (version, page, load incl. pre-split pair, model, train + ROC + full stats JSON, /api/stats, binormal fits + null when impossible, logistic Wald/condition number, regress, saves, plateau auto-stop + control + validation, train-panel parity controls (learning rate/weight decay/batch-epoch/stopping conditions/print counter) + behavioral proof + validation, model-panel parity (bias->BareProp/multi-layer->BackProp/error function/load-network), no-bias BackProp save/load round trip, multipart log toggles, logistic batch/epoch guard, DFA (linear/quadratic + report/stats + guesses saves), dataset characteristics + ROC reporting load params + trap_thresholds behavioral proof, test_n exact split, multi-output load/BackProp/train/DFA + logistic refusal, async train/status/stop + 409 busy + cancel, algorithm=auto blocking + async, per-action audit log)"
