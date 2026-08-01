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
# GUI/CLI parity (rule 5): the CV panel must expose the nested-OBD optimizer with
#    the SAME four choices and terminology as the standalone OBD panel, and must
#    submit it -- a control the page never sends is not parity.
curl -s "$URL/" > page.html
grep -q 'id="cv_algorithm"' page.html || fail "CV panel has no optimizer control"
for opt in 'value="auto"' 'value="1"' 'value="2"' 'value="3"'; do
    $PY - "$opt" <<'PY' || fail "CV optimizer control is missing an option"
import re, sys
html = open("page.html", encoding="utf-8").read()
sel = re.search(r'id="cv_algorithm".*?</select>', html, re.S)
assert sel and sys.argv[1] in sel.group(0), sys.argv[1]
PY
done
grep -q 'algorithm: \$("cv_algorithm").value' page.html \
    || fail "the CV panel does not submit its optimizer choice"

# --- Page structure and terminology (live walkthrough findings, 2026-07-29) ---
# These are presentation contracts a user's reading of the page depends on, so
# they are asserted against the served page rather than trusted to review.
$PY - <<'PY' || fail "page structure/terminology contract broken"
import re, sys
html = open("page.html", encoding="utf-8").read()
def need(cond, why):
    if not cond:
        print("FAIL:", why, file=sys.stderr); sys.exit(1)

# (1) Every panel's PRIMARY ACTION is its LAST row. A Run/Create/Load button in
#     the first row invites acting before the form is finished (Craig's GUI-wide
#     rule). Checked structurally, for every fieldset, so a new panel cannot
#     quietly reintroduce the pattern.
panels = re.findall(r'<fieldset>(.*?)</fieldset>', html, re.S)
need(len(panels) >= 8, f"expected the full panel set, found {len(panels)}")
for body in panels:
    legend = re.search(r'<legend>(.*?)</legend>', body).group(1)
    rows = re.findall(r'<div class="row([^"]*)"', body)
    need(rows, f"panel {legend!r} has no rows")
    need("action" in rows[-1], f"panel {legend!r} does not end with its action row")
    # and nothing after the action row may be another control-bearing row
    need("action" not in " ".join(rows[:-1]),
         f"panel {legend!r} has more than one action row")

# (2) OBD controls name their UNIT. "Start hidden 2" was read as two hidden
#     LAYERS; OBD sizes the node count of ONE hidden layer.
obd = re.search(r'<legend>4c.*?</fieldset>', html, re.S).group(0)
need("Starting\n    hidden nodes" in obd or "Starting hidden nodes" in obd
     or re.search(r'Starting\s+hidden nodes', obd), "OBD start control not named in nodes")
need(re.search(r'Maximum\s+hidden nodes', obd), "OBD max control not named in nodes")
need("Start hidden <" not in obd and "Max hidden <" not in obd,
     "the old ambiguous OBD labels are still present")
need("sizes ONE hidden layer" in obd, "the OBD panel does not state it sizes one hidden layer")
need("never adds a second layer" in obd, "the OBD panel does not deny adding layers")

# (3) The same wording reaches the NESTED-OBD controls in the CV panel.
cv = re.search(r'<legend>4d.*?</fieldset>', html, re.S).group(0)
need(re.search(r'OBD\s+maximum hidden nodes', cv), "CV nested-OBD max is not named in nodes")
need(re.search(r'Fixed\s+hidden nodes', cv), "CV fixed hidden is not named in nodes")
need("never adds a layer" in cv, "the CV nested-OBD control does not deny adding layers")

# (4) The contrast keeps its ROLES and subtraction DIRECTION after selection --
#     they used to live only in the placeholder text, which selection erased.
need('id="cv_contrast_note"' in cv, "no persistent contrast-direction note")
need("updateContrastNote" in html, "the contrast note is never updated")
need("Primary (${pn})" in html and "Reference (${rn})" in html,
     "the contrast note does not name both roles from the current selection")
need("positive" in html and "favors the primary" in html,
     "the contrast note does not say which direction favors the primary")

# (5) Chart legends are FILLED IN from the server's answer, never hard-coded --
#     with a validation set loaded they must not claim the test set was watched.
need('id="errlegend"' in html and 'id="obdlegend"' in html,
     "chart legends are not addressable, so they cannot be made validation-aware")
need("setMonitorLabels" in html, "no monitor-aware legend function")
need(">test set (sampled)" not in html and ">test error (the score)" not in html,
     "a hard-coded test-set legend is still present")

# (6) Result provenance: the ROC/statistics panels name what produced them, and
#     a standalone analysis says it did not become the current model.
need('id="provbox"' in html and "setProvenance" in html, "no result-provenance banner")
need('id="cvscope"' in html, "the CV report does not scope itself away from the panels below")
for op in ["Train", "OBD hidden-layer sizing (standalone)", "discriminant function analysis"]:
    need(f'setProvenance("{op}' in html or f"setProvenance((" in html,
         f"no provenance set for {op}")
print("page structure/terminology ok")
PY

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

# --- Stepwise regression ----------------------------------------------------
# The fixture is the low-birth-weight logistic (5 inputs -> 5 variables), NOT
#    XOR, because every candidate subnetwork here genuinely converges on
#    grad_max in ~2000 iterations. XOR cannot serve: dropping either input
#    leaves a 1-input network being asked to learn XOR, which sits at ln 2
#    (0.693) with 50% accuracy and a gradient that never reaches 1e-6, so the
#    candidate runs to the ceiling. That is a real refusal, not a fixture
#    nuisance -- and until the convergence rule below was enforced, this suite's
#    happy path was quietly comparing that unfinished fit and passing.
cp ../../../docs/datasets/low-birth-weight/lowbwt2-2train.txt .
curl -s -X POST "$URL/api/load" -d "mode=train&path=lowbwt2-2train.txt" \
    | grep -q '"ok":true' || fail "load low-birth-weight for stepwise"
curl -s -X POST "$URL/api/model" -d "type=logistic" > /dev/null
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=2000&seed=42" > reg_train.json
grep -q '"stopReason":"grad_max"' reg_train.json \
    || fail "the stepwise fixture's own fit must converge before it is regressed"
curl -s -X POST "$URL/api/regress" \
    --data-urlencode "structure=0;1;2;3;4" \
    --data-urlencode "direction=reverse" \
    --data-urlencode "threshold=0.05" > regress.json
grep -q '"ok":true' regress.json || fail "regress endpoint"
grep -q 'Reverse regressing' regress.json || fail "no regression report"
# The ANALYSIS RESULT must be visible where the user ran it (2026-07-27). The
#    server has always returned the full stepwise report in `output` -- and the
#    page threw it away, rendering only j.message ("reverse stepwise regression
#    complete"). The comment above the server's return even claimed the output
#    was "shown on the page": it described intent, not behavior. A user's only
#    route to their own result was to find and read neuron.log.
#    The control (output is nonempty) passes pre-fix; the page assertions below
#    are the ones that fail, so this block proves the RENDERING, not the API.
$PY - <<'PY' || fail "the regression report must come back nonempty (control)"
import json
d = json.load(open("regress.json", encoding="utf-8"))
assert d["output"].strip(), "the server sent no stepwise report at all"
assert "p-value" in d["output"], d["output"][-400:]
PY
$PY - <<'PY' || fail "the page must render the stepwise report in a persistent pane"
import re
html = open("page.html", encoding="utf-8").read()
# A destination element that survives after the run, like the DFA pane
assert re.search(r'<pre id="regressout"', html), \
    "no persistent stepwise results pane on the page"
# ... and regress() must actually put the report INTO it. Keying on the
#     assignment (not merely the element's presence) is what fails pre-fix.
# ... and the page must actually put the report INTO it. Keying on the
#     assignment (not merely the element's presence) is what fails pre-fix.
assert re.search(r'\$\("regressout"\)\.textContent\s*=\s*\w+\.output', html), \
    "the page never renders the stepwise output; it discards the report"
PY
# The forward direction over the same grouped structure must complete too --
#    both directions carry the same convergence and result contracts.
curl -s -X POST "$URL/api/model" -d "type=logistic" > /dev/null
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=2000&seed=42" > /dev/null
curl -s -X POST "$URL/api/regress" \
    --data-urlencode "structure=0;1;2,3;4" \
    --data-urlencode "direction=forward" \
    --data-urlencode "threshold=0.05" > regress_fwd.json
grep -q '"ok":true' regress_fwd.json || fail "forward stepwise regression"
grep -q 'Forward regressing' regress_fwd.json || fail "no forward regression report"
# A grouped variable must never be split: variable 2 is nodes 2 AND 3 together
grep -q 'node(s) 2 3' regress_fwd.json || fail "grouped variable was not kept intact"
# The forward direction states its own result, in its own words
$PY - <<'PY' || fail "a forward run must state its final SELECTED variables"
import json, re
d = json.load(open("regress_fwd.json", encoding="utf-8"))
o = d["output"]
assert "Forward stepwise regression complete." in o, o[-500:]
assert "Variables added, in order:" in o, o[-500:]
m = re.search(r"Final selected variables: (.+)", o)
assert m, o[-500:]
assert [int(x) for x in m.group(1).split(",")] == d["stepwise"]["finalVariables"], \
    (m.group(1), d["stepwise"]["finalVariables"])
assert "Final retained variables:" not in o, o[-500:]
PY

# Untrained-model guard: a fresh model must refuse regression
curl -s -X POST "$URL/api/model" -d "type=logistic" > /dev/null
curl -s -X POST "$URL/api/regress" \
    --data-urlencode "structure=0;1;2;3;4" --data-urlencode "direction=reverse" \
    --data-urlencode "threshold=0.05" | grep -q '"ok":false' \
    || fail "regress should refuse an untrained model"
# An UNCONVERGED CANDIDATE CANNOT ENTER SELECTION (2026-07-27). Stepwise used
#    train()'s return value directly and checked nothing, so a candidate that
#    merely ran out of iteration ceiling produced a chi-square and a p-value
#    indistinguishable from a finished fit -- the one engine path still exempt
#    from the convergence rule that OBD, CV and ordinary training all obey.
#    The candidate clone inherits the trained model's configuration through the
#    established copy path, so training the ORIGINAL at a tiny ceiling is what
#    puts every candidate at that same ceiling.
#    It must FAIL the analysis, not skip the candidate: a pass selects the
#    largest p-value among all candidates, so dropping one silently changes
#    which variable wins.
curl -s -X POST "$URL/api/model" -d "type=logistic" > /dev/null
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=3&seed=42" > ceil_train.json
grep -q '"converged":false' ceil_train.json \
    || fail "the tiny-ceiling setup run was supposed to NOT converge"
curl -s -X POST "$URL/api/regress" \
    --data-urlencode "structure=0;1;2;3;4" --data-urlencode "direction=reverse" \
    --data-urlencode "threshold=0.05" > regress_ceiling.json
$PY - <<'PY' || fail "a ceiling-exhausted candidate must stop the stepwise analysis"
import json, re
d = json.load(open("regress_ceiling.json", encoding="utf-8"))
assert d["ok"] is False, d                       # the analysis failed
out, msg = d.get("output", ""), d["message"]
assert "max_iterations" in msg, msg              # named, machine-readable reason
assert "did not converge" in msg, msg
# The candidate is identified by its conceptual variable AND its input nodes
assert re.search(r"variable \d+ = node\(s\)", msg), msg
assert "not a finished fit" in out, out[-500:]
assert "iterations, which is not a stopping condition" in out, out[-500:]
# No statistic may be produced for the rejected candidate, and no variable may
#    be selected out of the incomplete pass
assert "no variable is selected" in out, out[-500:]
tail = out[out.index("STOPPING:"):]
assert "Chi-square" not in tail and "p = " not in tail, tail
assert "the largest was variable" not in out, out[-500:]
# ... and the user is told how to proceed
assert "Raise the maximum iterations" in out, out[-500:]
# An analysis that never reached a decision must NOT present a final variable
#    set: it does not have one, and printing an empty or partial list as a
#    conclusion would claim a result the procedure never produced.
assert "Final retained variables:" not in out, out[-500:]
assert "Final selected variables:" not in out, out[-500:]
assert "stepwise regression complete." not in out, out[-500:]
PY
# THE REJECTED CANDIDATE ITSELF MUST BE IN THE AUDIT TRAIL (Sol, 2026-07-27).
#    recordCandidate() ran AFTER requireConvergedFit(), which throws -- so the
#    one candidate a reader most needs, the fit that stopped the analysis, was
#    the only one missing. A first-candidate ceiling failure reported
#    fitsCompleted:1 and candidates:[].
$PY - <<'PY' || fail "the candidate that failed must appear in the audit trail"
import json
d = json.load(open("regress_ceiling.json", encoding="utf-8"))
sw = d["stepwise"]
c = sw["candidates"]
assert c, "the failing candidate was dropped from the audit trail"
bad = [x for x in c if not x["converged"]]
assert bad, "no unconverged candidate recorded, yet the analysis failed"
x = bad[-1]
assert x["stopReason"] == "max_iterations", x
# It contributes NO statistic: a rejected fit is recorded, never compared
assert x["p"] is None, x
assert x["g2"] is None, x
# The count and the trail agree with each other
assert sw["fitsCompleted"] == len(c), (sw["fitsCompleted"], len(c))
# An analysis that failed did not reach a decision, and must say so rather
#    than returning an empty finalVariables that reads as "kept nothing"
assert sw["complete"] is False, sw
PY
# The partial prose report survives the refusal, including the rejected
#    candidate's stopping explanation.
$PY - <<'PY' || fail "the partial audit trail must survive a failed stepwise run"
import json
d = json.load(open("regress_ceiling.json", encoding="utf-8"))
assert "Reverse regressing" in d["output"], d["output"][:300]
assert "Variable structure:" in d["output"], d["output"][:300]
PY

# CANDIDATE FITS MUST NOT EVALUATE THE TEST SET (2026-07-27). Every candidate
#    refit ran train()'s full reporting epilogue, which re-derived the
#    classification tables, the ROC fit and a 2000-resample bootstrap over the
#    TEST set. Wilks selection reads the TRAINING likelihood and consumes none
#    of it. Beyond being a large avoidable cost, it meant model selection
#    repeatedly evaluated the set that is supposed to stay untouched until a
#    procedure has been chosen.
#    Needs a dataset WITH a test set, so the bootstrap would actually run: the
#    mode=train fixture above has none, and would prove nothing.
curl -s -X POST "$URL/api/load" -d "mode=raw&path=lowbwt2-2train.txt&fraction=0.25&seed=1" \
    | grep -q '"ok":true' || fail "load low-birth-weight split for the quiet-candidate proof"
curl -s -X POST "$URL/api/model" -d "type=logistic" > /dev/null
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=2000&seed=42" > quiet_train.json
# The control: the ORDINARY training run above is unaffected -- it still
#    produces the full report including the test-set bootstrap. Only candidate
#    refits go quiet, so this must remain true.
$PY - <<'PY' || fail "an ordinary training run must still report its test statistics"
import json
o = json.load(open("quiet_train.json", encoding="utf-8"))["output"]
assert "Test set" in o, o[-600:]
assert "bootstrap resamples" in o, o[-600:]
PY
curl -s -X POST "$URL/api/regress" \
    --data-urlencode "structure=0;1;2;3;4" --data-urlencode "direction=reverse" \
    --data-urlencode "threshold=0.05" > quiet_regress.json
$PY - <<'PY' || fail "candidate refits must not evaluate or bootstrap the test set"
import json
d = json.load(open("quiet_regress.json", encoding="utf-8"))
assert d["ok"], d["message"]
o = d["output"]
# Not one candidate may have touched the held-out set
assert "bootstrap resamples" not in o, "a candidate refit ran the ROC bootstrap"
assert "Test set" not in o, "a candidate refit reported test-set statistics"
assert "Classification accuracy" not in o, "a candidate refit reported accuracy"
# ... nor emitted the training-run narration nobody consumes
assert "I'm running an iterative model" not in o, "candidate refits still narrate"
assert "Total iterations" not in o, "candidate refits still print their epilogue"
# What the analysis DOES report is unchanged and complete: every candidate's
#    comparison, and the selection built from it
assert "Removing variable(s)" in o, o[:400]
assert "Chi-square = " in o, o[:400]
assert o.count("p = ") >= 5, o.count("p = ")
assert "p-values of all removed variables" in o, o[-400:]
PY

# WILKS NEEDS LOG LIKELIHOODS, AND THE ORIGINAL IS NOT OURS TO MUTATE (Sol,
#    2026-07-27). Both directions used to call netPtr->setXEerror() on the
#    USER'S trained model. That mutated a model stepwise is supposed to treat
#    as read-only, and it did not even achieve what it looked like: e_in, the
#    prior error every first-pass comparison is made against, was captured from
#    the original fit and stayed in THAT fit's objective. So an LMS-trained
#    network had candidates training under cross-entropy while their baseline
#    error was least-mean-squares, and G2 = 2N(Esub - Efull) subtracted two
#    different objectives and called the result a chi-square. Logistic is
#    cross-entropy by construction, which is why the walkthrough never saw it.
curl -s -X POST "$URL/api/load" -d "mode=train&path=lowbwt2-2train.txt" >/dev/null
curl -s -X POST "$URL/api/model" -d "type=simpleprop&hidden=3&errfunc=lms" \
    | grep -q 'LMS' || fail "could not build an LMS-trained network"
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=300&seed=42" >/dev/null
curl -s -X POST "$URL/api/regress" \
    --data-urlencode "structure=0;1;2;3;4" --data-urlencode "direction=reverse" \
    --data-urlencode "threshold=0.05" > lms_regress.json
$PY - <<'PY' || fail "stepwise must refuse an LMS fit rather than mix objectives"
import json
d = json.load(open("lms_regress.json", encoding="utf-8"))
assert d["ok"] is False, d
m = d["message"]
assert "cross-entropy" in m, m
assert "least-mean-squares" in m or "least mean squares" in m, m
PY
# ... and it must have left the user's model exactly as it found it: still LMS,
#     still trained, still usable. (Pre-fix, the refused analysis had
#     already flipped the model's error function underneath the user.)
curl -s "$URL/api/stats" | grep -q '"ok":true' \
    || fail "the model must remain usable after a refused regression"
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=1&seed=42" > lms_after.json
$PY - <<'PY' || fail "stepwise must not change the original model's error function"
import json
o = json.load(open("lms_after.json", encoding="utf-8"))["output"]
assert "LMS error" in o, o[:600]
assert "X-entropy error" not in o, o[:600]
PY

# --- Asynchronous stepwise regression (2026-07-27) --------------------------
# Stepwise is many training runs, so it must behave like the other long engine
#    jobs. It used to be a single blocking POST that showed "regressing…" for
#    the whole operation with no progress, no Stop and no way to tell whether
#    it was still advancing.
#
# Blocking and async must agree exactly (same seed, same configuration): async
#    is the same call on a worker thread, so merely observing it must not change
#    RNG state, candidate order, p-values or the selection. Run on the fast
#    low-birth-weight logistic, whose candidates all converge.
curl -s -X POST "$URL/api/load" -d "mode=train&path=lowbwt2-2train.txt" >/dev/null
curl -s -X POST "$URL/api/model" -d "type=logistic" > /dev/null
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=2000&seed=42" > /dev/null
curl -s -X POST "$URL/api/regress" \
    --data-urlencode "structure=0;1;2;3;4" --data-urlencode "direction=reverse" \
    --data-urlencode "threshold=0.05" > sw_block.json
curl -s -X POST "$URL/api/model" -d "type=logistic" > /dev/null
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=2000&seed=42" > /dev/null
curl -s -X POST "$URL/api/regress" \
    --data-urlencode "structure=0;1;2;3;4" --data-urlencode "direction=reverse" \
    --data-urlencode "threshold=0.05" -d "async=1" \
    | grep -q '"ok":true' || fail "async stepwise did not start"
for i in $(seq 1 240); do
    curl -s "$URL/api/train/status" > sw_async.json
    grep -q '"running":false' sw_async.json && break
    sleep 0.3
done
grep -q '"running":false' sw_async.json || fail "async stepwise never completed"
$PY - <<'PY' || fail "async and blocking stepwise must produce identical results"
import json
b = json.load(open("sw_block.json", encoding="utf-8"))
a = json.load(open("sw_async.json", encoding="utf-8"))["result"]
assert b["ok"] and a["ok"], (b["message"], a["message"])
assert b["stepwise"] == a["stepwise"], (b["stepwise"], a["stepwise"])
assert b["output"] == a["output"], "the reports differ between blocking and async"
# The structured result must actually carry a selection, or this proves nothing
assert b["stepwise"]["path"], b["stepwise"]
assert b["stepwise"]["fitsCompleted"] > 0, b["stepwise"]
PY

# Progress, the busy contract and a real Stop. TWO SHORT async runs rather than
#    one long one: since candidate refits went quiet they are fast, so the whole
#    analysis is ~1.5 s and a test that waited for several progress states
#    before sending Stop could race the run to completion (measured: it did --
#    "no training in progress").
$PY - <<'PYEOF'
import random
random.seed(11)
def rows(n):
    for _ in range(n):
        x = [random.uniform(-0.9, 0.9) for _ in range(6)]
        z = 1.2 * x[0] - 0.9 * x[1] + 0.6 * x[2] + 0.05 * x[3]
        p = 1 / (1 + pow(2.718281828, -z))
        yield " ".join("%.6f" % v for v in x) + " " \
            + ("1" if random.random() < p else "0") + " \n"
with open("sw_slow.set", "w", newline="\n") as f:
    f.writelines(rows(4000))
PYEOF
curl -s -X POST "$URL/api/load" -d "mode=train&path=sw_slow.set" \
    | grep -q '"ok":true' || fail "load the async stepwise fixture"
curl -s -X POST "$URL/api/model" -d "type=logistic" > /dev/null
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=20000&seed=42" > sw_setup.json
grep -q '"converged":true' sw_setup.json \
    || fail "the async stepwise fixture's own fit must converge"

# RUN 1 -- observed to completion: progress must advance, and the busy contract
#    must hold while it owns the engine.
curl -s -X POST "$URL/api/regress" \
    --data-urlencode "structure=0;1;2;3;4;5" --data-urlencode "direction=reverse" \
    --data-urlencode "threshold=0.05" -d "async=1" > sw_start.json
grep -q '"ok":true' sw_start.json || fail "async stepwise did not start"
# It returned BEFORE finishing -- that is the whole point of async
curl -s "$URL/api/train/status" | grep -q '"running":true' \
    || fail "status should say running right after an async stepwise start"
swbusy=$(curl -s -w '\n%{http_code}' -X POST "$URL/api/model" -d "type=logistic")
echo "$swbusy" | grep -q '"busy":true' || fail "no busy flag while stepwise runs"
echo "$swbusy" | tail -1 | grep -q '409' || fail "stepwise busy refusal should be HTTP 409"
curl -s -X POST "$URL/api/regress" \
    --data-urlencode "structure=0;1;2;3;4;5" --data-urlencode "direction=reverse" \
    --data-urlencode "threshold=0.05" -d "async=1" \
    | grep -q '"busy":true' || fail "a second stepwise start must be refused"
: > sw_progress.txt
for i in $(seq 1 400); do
    curl -s "$URL/api/train/status" >> sw_progress.txt
    echo >> sw_progress.txt
    grep -q '"running":false' sw_progress.txt && break
    sleep 0.1
done
grep -q '"running":false' sw_progress.txt || fail "async stepwise never completed"
$PY - <<'PYEOF' || fail "stepwise progress must advance and be internally consistent"
import json
states = []
for line in open("sw_progress.txt", encoding="utf-8"):
    line = line.strip()
    if not line: continue
    s = json.loads(line).get("stepwise")
    if s: states.append(s)
assert states, "no stepwise progress was ever published"
distinct = [s for i, s in enumerate(states) if i == 0 or s != states[i - 1]]
assert len(distinct) >= 2, "progress never advanced: %r" % distinct
# Completed fits only ever go up
fits = [s["fitsCompleted"] for s in states]
assert fits == sorted(fits), fits
# The candidate accounting agrees with the structure that was submitted: six
#    conceptual variables, each a single input node, and a candidate index that
#    never exceeds the count of candidates in its own pass
for s in states:
    assert s["direction"] == "reverse", s
    assert 1 <= s["candidate"] <= s["candidatesThisStep"], s
    assert s["candidatesThisStep"] == 6 - s["step"], s
    assert s["inputs"] == [s["variable"]], s
    assert s["phase"], s
PYEOF

# THE COMPLETED CANDIDATE REPORTS HOW IT ENDED (Sol, 2026-07-27). Progress was
#    announced only BEFORE each fit, so the completed count lagged by one and
#    the last candidate stayed labelled "training" with no convergence status
#    anywhere -- which the spec requires. It has to be a STICKY field, not a
#    transient phase: a finished announcement is superseded by the next
#    candidate's starting announcement within microseconds, so a poller could
#    never observe it otherwise.
$PY - <<'PYEOF' || fail "progress must report each finished candidate's status"
import json
states = []
for line in open("sw_progress.txt", encoding="utf-8"):
    line = line.strip()
    if not line: continue
    s = json.loads(line).get("stepwise")
    if s: states.append(s)
done = [s["lastCompleted"] for s in states if s.get("lastCompleted")]
assert done, "no completed-candidate status was ever published"
for s in done:
    assert s["stopReason"], s
    assert isinstance(s["converged"], bool), s
    assert isinstance(s["variable"], int), s
# Every candidate converges on this fixture
assert all(s["converged"] for s in done), [s for s in done if not s["converged"]]
# The completed count must not lag: any state carrying a completed candidate's
#    status must already count it
assert all(s["fitsCompleted"] >= 1 for s in states if s.get("lastCompleted")), states
PYEOF

# RUN 2 -- stopped as soon as it is running. Stop must reach the candidate
#    training at that moment and end the job promptly; it must not merely
#    relabel work that ran to completion.
curl -s -X POST "$URL/api/regress" \
    --data-urlencode "structure=0;1;2;3;4;5" --data-urlencode "direction=reverse" \
    --data-urlencode "threshold=0.05" -d "async=1" \
    | grep -q '"ok":true' || fail "second async stepwise did not start"
curl -s "$URL/api/train/status" | grep -q '"running":true' \
    || fail "the run to be stopped was not running"
curl -s -X POST -d "" "$URL/api/train/stop" | grep -q '"ok":true' \
    || fail "stop did not accept during stepwise"
stopped=0
for i in $(seq 1 60); do
    curl -s "$URL/api/train/status" > sw_cancel.json
    grep -q '"running":false' sw_cancel.json && { stopped=$i; break; }
    sleep 0.5
done
[ "$stopped" != 0 ] || fail "cancelled stepwise never stopped"
$PY - <<'PYEOF' || fail "a stopped stepwise must report cancellation and keep its audit trail"
import json
d = json.load(open("sw_cancel.json", encoding="utf-8"))["result"]
assert d["cancelled"] is True, d
assert "cancelled" in d["message"], d["message"]
# It stopped inside an incomplete pass, so it selected nothing from that pass
assert "no variable is selected" in d["output"], d["output"][-500:]
# ... and whatever it did finish is still there: a cancelled run keeps its
#     partial audit trail rather than throwing the work away
assert "Reverse regressing" in d["output"], d["output"][:200]
# It really was cut short: 18 candidate fits (6+5+4+3) is the complete reverse
#    structure for six variables, so a "Stop" that let everything finish would
#    land on 18 here
assert d["stepwise"]["fitsCompleted"] < 18, d["stepwise"]
# A cancelled analysis reached no decision, and says so rather than returning
#    an empty finalVariables that would read as "the procedure kept nothing"
assert d["stepwise"]["complete"] is False, d["stepwise"]
# ... and a cancelled run likewise presents no final variable set
assert "Final retained variables:" not in d["output"], d["output"][-400:]
assert "stepwise regression complete." not in d["output"], d["output"][-400:]
# The cancelled candidate is itself in the trail, with its reason -- it is the
#    one that stopped the run, so leaving it out would hide the explanation
c = d["stepwise"]["candidates"]
assert c, "the cancelled candidate was dropped from the audit trail"
assert c[-1]["converged"] is False, c[-1]
assert c[-1]["stopReason"] == "cancelled", c[-1]
assert c[-1]["p"] is None, c[-1]
assert d["stepwise"]["fitsCompleted"] == len(c), (d["stepwise"]["fitsCompleted"], len(c))
PYEOF
# The user's trained model is untouched by a stepwise run, cancelled or not --
#    stepwise is a standalone analysis over clones. It must still be usable.
curl -s "$URL/api/stats" | grep -q '"ok":true' \
    || fail "the trained model must survive a cancelled stepwise run"
# Stop with nothing running refuses cleanly
curl -s -X POST -d "" "$URL/api/train/stop" | grep -q '"ok":false' \
    || fail "stop should refuse when no stepwise is running"

# A STOP MUST NOT POISON THE NEXT RUN (Sol, 2026-07-27). job.cancel is a
#    process-global latch that only an async START cleared, and blocking
#    regression installs the same cancel observer -- so immediately after the
#    cancelled run above, a BLOCKING regression cancelled its own first
#    candidate before training an iteration, and reported a failure the user
#    never asked for. This is the exact sequence a walkthrough performs: try
#    Stop, then run the analysis for real.
curl -s -X POST "$URL/api/regress" \
    --data-urlencode "structure=0;1;2;3;4;5" --data-urlencode "direction=reverse" \
    --data-urlencode "threshold=0.05" > after_stop.json
$PY - <<'PY' || fail "a blocking regression after a Stop must run normally"
import json
d = json.load(open("after_stop.json", encoding="utf-8"))
assert d["ok"] is True, d["message"]
assert d["cancelled"] is False, d
assert d["stepwise"]["fitsCompleted"] > 1, d["stepwise"]
assert "cancelled" not in d["output"], d["output"][-400:]
PY

# STALE PROGRESS MUST NOT FOLLOW A LATER JOB (Sol, 2026-07-27). The stepwise
#    object is cleared by every job start, so a subsequent plain training run
#    reports no stepwise accounting -- exactly as a plain train reports no obd
#    phase. Without the clear, /api/train/status attached the finished
#    stepwise progress to whatever ran next.
curl -s -X POST "$URL/api/model" -d "type=logistic" >/dev/null
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=200&seed=42&async=1" >/dev/null
for i in $(seq 1 120); do
    curl -s "$URL/api/train/status" > post_sw_status.json
    grep -q '"running":false' post_sw_status.json && break
    sleep 0.3
done
grep -q '"stepwise":' post_sw_status.json \
    && fail "a plain train must not carry stale stepwise progress"

# THE COMPLETED CANDIDATE REPORTS HOW IT ENDED (Sol, 2026-07-27). Progress was
#    announced only BEFORE each fit, so the completed count lagged by one and
#    the last candidate stayed labelled "training" until the result replaced
#    it -- with no convergence status anywhere, which the spec requires.
$PY - <<'PY' || fail "progress must report each finished candidate's status"
import json
states = []
for line in open("sw_progress.txt", encoding="utf-8"):
    line = line.strip()
    if not line: continue
    s = json.loads(line).get("stepwise")
    if s: states.append(s)
# The status of the last COMPLETED candidate must persist while the next one
#    trains -- a finished announcement is superseded within microseconds, so a
#    transient phase could never be observed by a poller.
done = [s["lastCompleted"] for s in states if s.get("lastCompleted")]
assert done, "no completed-candidate status was ever published"
for s in done:
    assert s["stopReason"], s
    assert isinstance(s["converged"], bool), s
    assert isinstance(s["variable"], int), s
# On this fixture every candidate converges
assert all(s["converged"] for s in done), [s for s in done if not s["converged"]]
# The completed count must not lag: once a candidate has finished, any state
#    carrying its status must already count it
withDone = [s for s in states if s.get("lastCompleted")]
assert all(s["fitsCompleted"] >= 1 for s in withDone), withDone
PY

# THE STRUCTURED RESULT IS THE AUDIT TRAIL (Sol, 2026-07-27). It used to carry
#    only the winning path and the final variables, so a consumer could not see
#    what was considered, what the comparison was, or how each fit ended
#    without parsing the prose report.
$PY - <<'PY' || fail "the structured result must carry every candidate considered"
import json, re
d = json.load(open("after_stop.json", encoding="utf-8"))
sw = d["stepwise"]
c = sw["candidates"]
assert c, "no candidate audit trail"
# Six variables, so the first pass considers six candidates, numbered 1..6
first = [x for x in c if x["step"] == 0]
assert len(first) == 6, first
assert sorted(x["candidate"] for x in first) == list(range(1, 7)), first
for x in c:
    for k in ("variable", "inputs", "priorError", "error", "df", "g2", "p",
              "converged", "stopReason", "iterations", "selected"):
        assert k in x, (k, x)
    assert x["inputs"], x            # the full input group, never empty
    # 0 is a legitimate count: dropping a null variable leaves the subnetwork
    #    at the trained optimum, so the gradient rule fires on the first check
    assert isinstance(x["iterations"], int) and x["iterations"] >= 0, x
    assert x["stopReason"], x
# Exactly one candidate per completed pass wins it, and the winners are the
#    selection path in order
won = [x for x in c if x["selected"]]
assert [x["variable"] for x in won] == [p["variable"] for p in sw["path"]], (won, sw["path"])
# A run that reached a decision says so, and then finalVariables is a claim
#    about the procedure's result rather than about how far it got
assert sw["complete"] is True, sw
assert sw["finalVariables"], sw
# Retained + removed must together account for every variable offered
assert sorted(sw["finalVariables"] + [p["variable"] for p in sw["path"]]) == list(range(6)), sw
# THE VISIBLE REPORT MUST STATE THE RESULT (Codex click-through, 2026-07-27).
#    The structured result carried finalVariables all along, but the report a
#    human actually reads ended at the p-value table: it answered "what
#    happened" and never "what did I end up with". A reader had to work out
#    which variables were ABSENT from the removal table.
o = d["output"]
assert "Reverse stepwise regression complete." in o, o[-500:]
assert "Variables removed, in order:" in o, o[-500:]
m = re.search(r"Final retained variables: (.+)", o)
assert m, o[-500:]
# ... and it must agree with the structured result, not merely exist
assert [int(x) for x in m.group(1).split(",")] == sw["finalVariables"], \
    (m.group(1), sw["finalVariables"])
# "retained" and "selected" are not interchangeable: this was a reverse run
assert "Final selected variables:" not in o, o[-500:]
PY

# Restore the XOR SimpleProp the save checks below expect (the stepwise section
#    above deliberately moved the engine onto the low-birth-weight logistic)
curl -s -X POST "$URL/api/load" -d "mode=train&path=xor_discrete.set" \
    | grep -q '"ok":true' || fail "reload XOR after regress"
curl -s -X POST "$URL/api/model" -d "type=simpleprop&hidden=3" > /dev/null
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

# --- The condition number uses EVERY eigenvalue (2026-07-27) ----------------
# computeCondNum built its eigenvalue vector from the half-open range
#    [&e[0], &e[dimension-1]) -- which is dimension-1 elements, so the LAST
#    eigenvalue was dropped from every condition number the engine ever
#    reported. The decisive case is a 2-parameter model (one input + bias):
#    one eigenvalue survives, so the maximum and the minimum are the same
#    number and the condition number is exactly 1 -- a perfectly conditioned
#    model, reported for data that is nothing of the kind. Measured pre-fix on
#    the 1-input logistic below: maxEig = minEig = 0.047, cond = 1.0.
#
#    The MATRIX changed on 2026-08-01 -- it is now the unpenalized observed
#    Fisher information X'VX, shared with the Wald covariance, rather than an
#    outer product of per-exemplar gradients -- so the label and the magnitudes
#    moved. The property asserted here did not: a 2-parameter design has two
#    eigenvalues, and consuming only one collapses the ratio to exactly 1.
$PY - <<'PY'
import random
random.seed(5)
with open("onein.set", "w", newline="\n") as f:
    for _ in range(600):
        x = random.uniform(-0.9, 0.9)
        p = 1 / (1 + pow(2.718281828, -(2.0 * x)))
        f.write("%.6f %s \n" % (x, "1" if random.random() < p else "0"))
PY
curl -s -X POST "$URL/api/load" -d "mode=train&path=onein.set" \
    | grep -q '1 inputs' || fail "load the single-input condition-number fixture"
curl -s -X POST "$URL/api/model" -d "type=logistic" > /dev/null
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=20000&seed=42" > cond2.json
$PY - <<'PY' || fail "the condition number must use every eigenvalue"
import json, re
d = json.load(open("cond2.json", encoding="utf-8"))
o = d["output"]
m = re.search(r"Information matrix maximum eigenvalue = (\S+)\n\s+minimum eigenvalue = (\S+)\n"
              r"Condition number = (\S+)", o)
assert m, o[-600:]
hi, lo, cond = (float(x.rstrip(".")) for x in m.groups())
# A 2-parameter model has TWO eigenvalues. If only one is consumed they
#    collapse and the ratio is exactly 1.
assert hi != lo, "max and min eigenvalue are identical: an eigenvalue was dropped"
assert cond > 1.0, "condition number is %r; a 2-parameter design is not perfectly conditioned" % cond
# The reported ratio must be the reported eigenvalues' ratio
assert abs(cond - hi / lo) < 0.05 * cond, (cond, hi, lo)
# ... and the JSON diagnostic must agree with the report
assert abs(d["stats"]["logistic"]["condNumber"] - cond) < 0.05 * cond, \
    (d["stats"]["logistic"]["condNumber"], cond)
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

# Ordinary training that ends at the iteration ceiling: the OPERATION succeeds
#    and the weights stay resumable, but the FIT is not valid and must say so.
#    ok and convergence are different facts, reported separately so nothing has
#    to infer one from the other.
curl -s -X POST "$URL/api/model" -d "type=logistic" > /dev/null
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=5&seed=42" > ceiling.json
grep -q '"ok":true' ceiling.json \
    || fail "a ceiling-limited train is still a successful operation (resumable weights)"
grep -q '"stopReason":"max_iterations"' ceiling.json || fail "expected a ceiling stop"
grep -q '"converged":false' ceiling.json \
    || fail "a ceiling-limited run must report converged:false"
grep -q '"ceilingExhausted":true' ceiling.json \
    || fail "a ceiling-limited run must report ceilingExhausted:true"
# The warning belongs to the ENGINE report, not only the JSON wrapper, so the
#    CLI transcript, neuron.log and any captured report carry it too. Checked
#    against the report FIELD, not the whole payload: the wrapper message also
#    says it, and matching that would pass even with the engine silent.
$PY - <<'PY' || fail "the engine training report must state that training did not converge"
import json
d = json.load(open("ceiling.json", encoding="utf-8"))
assert "did NOT converge" in d["output"], d["output"][-400:]
assert "safety limit" in d["output"], d["output"][-400:]
PY
# A converged run says so, and prints no such warning (the control).
curl -s -X POST "$URL/api/model" -d "type=logistic" > /dev/null
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=20000&seed=42" > converged.json
grep -q '"converged":true' converged.json \
    || fail "a run that stops on a real rule must report converged:true"
grep -q '"ceilingExhausted":false' converged.json \
    || fail "a converged run must not claim ceiling exhaustion"
$PY - <<'PY' || fail "a converged run must not print the non-convergence warning"
import json
d = json.load(open("converged.json", encoding="utf-8"))
assert "did NOT converge" not in d["output"], d["output"][-400:]
PY
# The page must render an unconverged run as a WARNING state, not the green
#    "ok" state that res.ok alone would have produced -- and it must key on
#    CONVERGENCE, not on one particular way of failing to converge. Keying on
#    ceilingExhausted alone painted a cancelled run green while the report under
#    it said the run had not converged.
$PY - <<'PY' || fail "the page must key its warning on convergence, not on ceiling exhaustion"
import re
html = open("page.html", encoding="utf-8").read()
m = re.search(r'^\s*(?:const\s+)?(\w+)\s*=\s*\(res\.converged === false\);',
              html, re.M)
assert m, "the page does not derive an unconverged flag from res.converged"
flag = m.group(1)
warn = re.search(r'if \(res\.ok && (\w+)\) setWarnStatus\("tstatus"', html)
assert warn, "no warning-state branch for the training status"
assert warn.group(1) == flag, \
    "the warning branch is gated on %r, not on the convergence flag" % warn.group(1)
PY

# Reporting cadence must not change the FIT (2026-07-26). The print counter is
#    a presentation setting; it may change how many rows come back and nothing
#    else. It used to change everything: the gradient was recalculated only
#    inside the block that PRINTS a row, so the stopping rule compared a value
#    cached at the last printed iteration. converged.json above is the same
#    seeded logistic under the default LOGARITHMIC counter; run it again
#    printing every iteration and require the same endpoint. The engine test
#    (tests/iterative) pins the mechanism -- this pins the API surface, where
#    the print counter is a user-facing control on the training panel.
curl -s -X POST "$URL/api/model" -d "type=logistic" > /dev/null
curl -s -X POST "$URL/api/train" \
    -d "algorithm=1&maxiter=20000&seed=42&logprint=0&printcount=1" > cadence_linear.json
$PY - <<'PY' || fail "the printing schedule must not change the fit"
import json, re
def load(name):
    d = json.load(open(name, encoding="utf-8"))
    out = d["output"]
    m = re.search(r"Total iterations = (\d+)", out)
    assert m, "no iteration count in " + name
    err = re.search(r"error in the training set = (\S+)\.", out)
    assert err, "no final training error in " + name
    rows = len(re.findall(r"^ +\d+ +\d\.\d+e[+-]\d+ ", out, re.M))
    return d, int(m.group(1)), err.group(1), rows
logd, logit, logerr, logrows = load("converged.json")       # logarithmic (default)
lind, linit, linerr, linrows = load("cadence_linear.json")  # every iteration
# The comparison only means something if the two schedules really did print
#    differently and the run really did stop on the gradient rule past the
#    dense head of the logarithmic schedule -- otherwise it would hold with the
#    bug present and guard nothing.
assert logd["stopReason"] == "grad_max", logd["stopReason"]
assert linrows > logrows, (linrows, logrows)
assert logit > 10, logit
assert logit == linit, ("the stopping iteration moved with the print counter", logit, linit)
assert logerr == linerr, ("the final error moved with the print counter", logerr, linerr)
assert logd["stopReason"] == lind["stopReason"]
assert logd["converged"] is True and lind["converged"] is True
PY

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
# Weight decay on with a blank lambda falls back to the engine default (5e-5)
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=10&weight_decay=1" \
    | grep -q '"ok":true' || fail "weight decay with no lambda should default, not error"
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
# The load-time params: invalid values are rejected before anything loads. (The
# trapezoidal ROC area is now the exact AUC over every operating point, so there
# is no trap_thresholds count to set -- that former param and its test are gone.)
curl -s -X POST "$URL/api/load" -d "mode=train&path=lowbwt2-2train.txt&out_lower=0.2" \
    | grep -q '"ok":false' || fail "output bounds on a discrete outcome must be rejected"
curl -s -X POST "$URL/api/load" -d "mode=train&path=lowbwt2-2train.txt&threshold=1.5" \
    | grep -q '"ok":false' || fail "threshold outside (0,1) must be rejected"
curl -s -X POST "$URL/api/load" -d "mode=train&path=lowbwt2-2train.txt&roc_report=maybe" \
    | grep -q '"ok":false' || fail "roc_report must be both or either"
curl -s -X POST "$URL/api/load" -d "mode=train&path=lowbwt2-2train.txt&in_lower=1&in_upper=-1" \
    | grep -q '"ok":false' || fail "inverted input bounds must be rejected"
curl -s -X POST "$URL/api/load" \
    -d "mode=train&path=lowbwt2-2train.txt&threshold=0.4&in_lower=-0.8&in_upper=0.8&history=1&roc_report=both" \
    | grep -q '"ok":true' || fail "valid characteristics + ROC settings must load"

# The whole-number split form: test_n places EXACTLY n exemplars in the test
# set (randomizeD truncates ratio*N, so a fraction cannot promise this)
curl -s -X POST "$URL/api/load" -d "mode=raw&path=lowbwt2-2train.txt&test_n=47" > testn.json
grep -q '"ok":true' testn.json || fail "test_n load"
grep -q '47 test exemplars' testn.json || fail "test_n=47 should yield exactly 47"

# --- Covariate stratification (ROADMAP 4 Phase 2, a GUI-beyond-CLI feature) --
# Stratifying the raw split on an input column (in addition to the outcome)
# returns the representativeness diagnostic; a bad column or bin count is
# rejected before loading.
curl -s -X POST "$URL/api/load" \
    -d "mode=raw&path=lowbwt2-2train.txt&fraction=0.25&seed=1&strata=1" > strata.json
grep -q '"ok":true' strata.json || fail "strata load"
grep -q 'Representativeness diagnostic' strata.json || fail "strata diagnostic missing"
grep -q 'Stratified on: outcome, input column 1' strata.json || fail "strata diagnostic wrong"
curl -s -X POST "$URL/api/load" -d "mode=raw&path=lowbwt2-2train.txt&fraction=0.25&strata=999" \
    | grep -q '"ok":false' || fail "an out-of-range strata column must be rejected"
curl -s -X POST "$URL/api/load" -d "mode=raw&path=lowbwt2-2train.txt&fraction=0.25&strata=1&strata_bins=1" \
    | grep -q '"ok":false' || fail "strata_bins < 2 must be rejected"

# --- Group-aware split (ROADMAP 4 Phase 3, a GUI-beyond-CLI feature) ---------
# Grouping on an input column keeps every cluster (rows with identical values)
# intact across the split, and the diagnostic states the zero-leakage guarantee.
curl -s -X POST "$URL/api/load" \
    -d "mode=raw&path=lowbwt2-2train.txt&fraction=0.25&seed=1&group=3" > group.json
grep -q '"ok":true' group.json || fail "group load"
grep -q 'group-aware split' group.json || fail "group diagnostic missing"
grep -q 'leakage = 0 by construction' group.json || fail "group zero-leakage guarantee missing"
curl -s -X POST "$URL/api/load" -d "mode=raw&path=lowbwt2-2train.txt&fraction=0.25&group=999" \
    | grep -q '"ok":false' || fail "an out-of-range group column must be rejected"

# --- Three-way split: train/validation/test (ROADMAP 4 Phase 4c) -------------
# A validation fraction makes an outcome-stratified three-way split so selection
# (OBD) monitors the validation set and the test set stays untouched.
curl -s -X POST "$URL/api/load" \
    -d "mode=raw&path=lowbwt2-2train.txt&fraction=0.25&val_fraction=0.15&seed=1" > threeway.json
grep -q '"ok":true' threeway.json || fail "three-way split load"
grep -q 'validation exemplars' threeway.json || fail "three-way split should report a validation set"
curl -s -X POST "$URL/api/load" \
    -d "mode=raw&path=lowbwt2-2train.txt&fraction=0.25&val_fraction=0.15&strata=1" \
    | grep -q '"ok":false' || fail "a three-way split with strata=/group= must be refused"

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
curl -s -X POST -d "" "$URL/api/train/stop" | grep -q '"ok":true' || fail "stop endpoint"
for i in $(seq 1 120); do
    curl -s "$URL/api/train/status" > status.json
    grep -q '"running":false' status.json && break
    sleep 0.5
done
grep -q '"running":false' status.json || fail "run never stopped after cancel"
grep -q '"stopReason":"cancelled"' status.json || fail "cancelled run should say so"
grep -q '"ok":true' status.json || fail "cancelled run should still be a completed run"
# A cancelled run is a completed OPERATION but not a fitted model either: the
#    engine report must say so for EVERY non-converged outcome, not only for the
#    ceiling, and the JSON must report converged:false without claiming the
#    ceiling was reached.
$PY - <<'PY' || fail "a cancelled run must be reported as not converged"
import json
d = json.load(open("status.json", encoding="utf-8"))["result"]
assert d["converged"] is False, d
assert d["ceilingExhausted"] is False, d      # cancelled, not ceiling-exhausted
assert "did NOT converge" in d["output"], d["output"][-400:]
assert "cancelled" in d["output"], d["output"][-400:]
PY
# Both ways of NOT converging reach the page's single warning branch: the
#    ceiling-exhausted run captured earlier and this cancelled one both report
#    ok:true with converged:false, differing only in ceilingExhausted (which
#    selects the wording of the note, not whether the warning appears).
$PY - <<'PY' || fail "ceiling and cancellation must both present as unconverged"
import json
ceil = json.load(open("ceiling.json", encoding="utf-8"))
canc = json.load(open("status.json", encoding="utf-8"))["result"]
assert ceil["ok"] is True and canc["ok"] is True, "both are successful operations"
assert ceil["converged"] is False and canc["converged"] is False, (ceil, canc)
assert ceil["ceilingExhausted"] is True, ceil
assert canc["ceilingExhausted"] is False, canc
PY
# Stop with nothing running refuses cleanly
curl -s -X POST -d "" "$URL/api/train/stop" | grep -q '"ok":false' \
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

# --- OBD hidden-layer sizing (ROADMAP 2 Phase 4) ---------------------------
# Refusal without a held-out test set: it is the validation signal early
#    stopping watches, so there is nothing to size against
curl -s -X POST "$URL/api/load" -d "mode=raw&path=lowbwt2-2train.txt&fraction=0" >/dev/null
curl -s -X POST "$URL/api/obd" -d "hidden_start=2&hidden_max=4" \
    | grep -q '"ok":false' || fail "OBD must refuse a dataset with no test set"

# A reached safety ceiling is a FAILURE TO CONVERGE, presented as such: no
#    architecture is selected, every trial is reported with its stop reason and
#    eligibility, and pruning never begins. A tiny budget with the default
#    stopping tolerances guarantees the ceiling fires first.
curl -s -X POST "$URL/api/load" -d "mode=raw&path=lowbwt2-2train.txt&fraction=0.3&seed=1" >/dev/null
curl -s -X POST "$URL/api/obd" \
    -d "hidden_start=2&hidden_max=6&iter_budget=40&sample_every=5&seed=1" \
    | grep -q '"ok":true' || fail "OBD ceiling search did not start"
for i in $(seq 1 120); do
    curl -s "$URL/api/train/status" > obd_ceiling.json
    grep -q '"running":false' obd_ceiling.json && break
    sleep 0.3
done
$PY - <<'PY' || fail "a ceiling-exhausted OBD run must refuse, not select"
import json
d = json.load(open("obd_ceiling.json"))["result"]
assert d["ok"] is False, d                      # not a success
assert d.get("ceilingExhausted") is True, d     # and specifically this failure
assert "selectedHidden" not in d, d             # NO architecture reported
assert "ceiling" in d["message"], d["message"]
t = d["trials"]
assert t and all(x["eligible"] is False for x in t), t
assert all(x["stopReason"] == "max_iterations" for x in t), t
assert all(x["phase"] == "grow" for x in t), t  # pruning never began
PY

# Happy path: an async grow-then-prune search that reports its phase while it
#    runs and returns a size-vs-error history plus the winner's ROC and stats.
#    autostop_tol is raised from the 1e-4 default because on this dataset no
#    stopping rule fires at the default within any practical budget (measured:
#    not at 20,000 iterations), and a search whose trials cannot finish now
#    correctly refuses rather than comparing unfinished fits.
curl -s -X POST "$URL/api/load" -d "mode=raw&path=lowbwt2-2train.txt&fraction=0.3&seed=1" >/dev/null
curl -s -X POST "$URL/api/obd" \
    -d "hidden_start=2&hidden_max=4&iter_budget=2000&sample_every=10&grow_patience=1&seed=1&autostop_tol=0.01" \
    | grep -q '"ok":true' || fail "OBD search did not start"
sawObd=0
for i in $(seq 1 120); do
    curl -s "$URL/api/train/status" > obd_status.json
    grep -q '"obd":{"phase"' obd_status.json && sawObd=1
    grep -q '"running":false' obd_status.json && break
    sleep 0.3
done
[ "$sawObd" = 1 ] || fail "status never reported an obd phase during the search"
grep -q '"running":false' obd_status.json || fail "OBD search never completed"
$PY - <<'PY' || fail "OBD result malformed"
import json
d = json.load(open("obd_status.json"))["result"]
assert d["ok"], d
assert d["selectedHidden"] >= 1, d["selectedHidden"]
h = d["obd"]["history"]
assert h and h[0]["hidden"] == 2, h                 # grow starts at hidden_start
assert "stats" in d and "roc" in d, list(d)
# Every trial admitted to the comparison ended on a real stopping rule, and the
# run names the optimizer it ran on (Auto records that it chose).
t = d["obd"]["trials"]
assert t and all(x["eligible"] is True for x in t), t
assert all(x["stopReason"] != "max_iterations" for x in t), t
assert d["optimizer"] in ("Canonical", "CGD", "Shanno"), d.get("optimizer")
assert d["optimizerAuto"] is False, d               # this run fixed it by default
PY

# A plain training run reports NO obd field in its status (that field is
#    OBD-only, so its absence proves a plain train is not mislabelled)
curl -s -X POST "$URL/api/model" -d "type=logistic" >/dev/null
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=200&seed=42&async=1" >/dev/null
for i in $(seq 1 120); do
    curl -s "$URL/api/train/status" > pt_status.json
    grep -q '"running":false' pt_status.json && break
    sleep 0.3
done
grep -q '"obd":' pt_status.json && fail "a plain train must not report an obd status field"

# 409 busy while a search runs, and Stop cancels it. TWO traps live here,
#    both caught when CI's Release build ran 15x faster than a local -O0 dir:
#    (1) a bodyless curl -X POST sends NO Content-Length, so the server waits
#    its ~5 s read timeout for a body before dispatching -- every stop in this
#    file therefore sends -d "" (without it, the stop was handled 5 s late,
#    after the search had already finished); (2) the search itself must
#    outlive the stop by a wide margin: sample_every=1 makes every iteration
#    pay a test-set sweep, autostop_window=5000 keeps the plateau backstop
#    from ending a size before iteration 10000, and 30 sizes keep the search
#    alive many seconds against these two immediate curls.
curl -s -X POST "$URL/api/obd" \
    -d "hidden_start=2&hidden_max=30&iter_budget=200000&sample_every=1&autostop_window=5000&seed=1" \
    | grep -q '"ok":true' || fail "long OBD search did not start"
obdbusy=$(curl -s -w '\n%{http_code}' -X POST "$URL/api/obd" -d "hidden_start=2&hidden_max=3")
echo "$obdbusy" | grep -q '"busy":true' || fail "no busy flag while an OBD search runs"
echo "$obdbusy" | tail -1 | grep -q '409' || fail "OBD busy refusal should be HTTP 409"
curl -s -X POST -d "" "$URL/api/train/stop" | grep -q '"ok":true' || fail "stop did not accept during OBD"
for i in $(seq 1 120); do
    curl -s "$URL/api/train/status" > obd_cancel.json
    grep -q '"running":false' obd_cancel.json && break
    sleep 0.3
done
grep -q '"running":false' obd_cancel.json || fail "cancelled OBD never completed"
grep -q '"cancelled":true' obd_cancel.json || fail "a stopped OBD result must say cancelled"

# --- Cross-validation model comparison (ROADMAP 4 Phase 4b-CV) --------------
# Refusal: no procedure selected is nothing to compare
curl -s -X POST "$URL/api/load" -d "mode=raw&path=lowbwt2-2train.txt&fraction=0.25&seed=1" >/dev/null
curl -s -X POST "$URL/api/cv" -d "logistic=0&ldfa=0&qdfa=0&neural=0" \
    | grep -q '"ok":false' || fail "CV must refuse when no procedure is selected"
# Refusal (B2): nested OBD with hidden_max below the start size (2) is an empty range
curl -s -X POST "$URL/api/cv" -d "neural=1&neural_obd=1&hidden_max=1" \
    | grep -q '"ok":false' || fail "CV must refuse hidden_max below the OBD start size"
# Refusal (B5): a validation fraction too small to yield a row must be rejected,
#    NOT silently produce a two-way split that reopens the OBD-on-test leak
curl -s -X POST "$URL/api/load" -d "mode=raw&path=lowbwt2-2train.txt&fraction=0.25&val_fraction=0.001" \
    | grep -q '"ok":false' || fail "a val_fraction that rounds to zero rows must be refused"
# Refusal (B8): CV needs a RAW dataset; a pre-split training set has no Raw
curl -s -X POST "$URL/api/load" -d "mode=train&path=lowbwt2-2train.txt" >/dev/null
curl -s -X POST "$URL/api/cv" -d "logistic=1" \
    | grep -q '"ok":false' || fail "CV must refuse a pre-split (mode=train) dataset"
# Optimizer parity: an invalid token is rejected BEFORE any job starts, with the
#    same message and encoding as standalone /api/obd (one public spelling)
curl -s -X POST "$URL/api/cv" -d "logistic=1&algorithm=bogus" \
    | grep -q '"ok":false' || fail "CV must reject an invalid algorithm token"
curl -s -X POST "$URL/api/cv" -d "logistic=1&algorithm=4" \
    | grep -q '"ok":false' || fail "CV must reject an out-of-range algorithm token"
curl -s -X POST "$URL/api/load" -d "mode=raw&path=lowbwt2-2train.txt&fraction=0.25&seed=1" >/dev/null

# Happy path: an async comparison of two procedures over one shared 5-fold plan,
#    returning the three-tier report (Tier 1 headline text, Tier 2 detail, Tier 3
#    files) and writing the machine-readable CSVs beside the data
curl -s -X POST "$URL/api/cv" \
    -d "folds=5&seed=42&logistic=1&ldfa=0&qdfa=0&neural=1&neural_obd=1&hidden_max=4&iter_budget=2000&inner_val=0.25&autostop_tol=0.01" \
    | grep -q '"ok":true' || fail "CV run did not start"
sawCv=0
for i in $(seq 1 120); do
    curl -s "$URL/api/train/status" > cv_status.json
    grep -q '"phase":"cross-validating"' cv_status.json && sawCv=1
    grep -q '"running":false' cv_status.json && break
    sleep 0.3
done
[ "$sawCv" = 1 ] || fail "status never reported a cross-validating phase"
grep -q '"running":false' cv_status.json || fail "CV never completed"
$PY - <<'PY' || fail "CV result malformed"
import json
d = json.load(open("cv_status.json", encoding="utf-8"))["result"]  # Tier-1 has UTF-8 box glyphs; Windows open() defaults to cp1252
assert d["ok"], d
cv = d["cv"]
assert "SUMMARY" in cv["tier1"] and "AUC (CV)" in cv["tier1"], cv["tier1"][:200]
# the standing caveat must always be present (it is policy, not decoration)
assert "descriptive spread across dependent folds" in cv["tier1"], cv["tier1"]
assert "Cross-validation detail" in cv["tier2"], cv["tier2"][:200]
assert len(cv["files"]) == 3, cv["files"]           # all three Tier-3 files written
assert cv.get("warnings", None) == [], cv.get("warnings")  # none failed (B7)
# Nested OBD reports BOTH selection facts: the architecture it chose and the
# optimizer it ran on. Absent an explicit algorithm= the default is Auto, which
# selects independently inside each fold -- so the report must say so.
assert "OBD selected" in cv["tier1"], cv["tier1"]        # architecture kept
assert "optimizer:" in cv["tier1"], cv["tier1"]          # optimizer beside it
assert "Optimizer selection:" in cv["tier2"], cv["tier2"][:400]
assert "chosen independently per fold" in cv["tier2"], cv["tier2"][:400]
PY
# Tier 3 predictions: header + one row per exemplar (189 in lowbwt), one column
#    per compared procedure -- the paired out-of-fold substrate
lines=$(wc -l < cv_predictions.csv | tr -d ' ')
[ "$lines" -eq 190 ] || fail "cv_predictions.csv should have 190 lines (header + 189), got $lines"
head -1 cv_predictions.csv | grep -q "Logistic" || fail "cv_predictions.csv missing a procedure column"

# Nested OBD that cannot finish a trial inside its ceiling: every fold FAILS.
#    The neural procedure must contribute no AUC, no architecture and no
#    optimizer metadata -- a failed fold fabricates nothing (and the comparison
#    beside it still reports the procedures that did fit).
curl -s -X POST "$URL/api/cv" \
    -d "folds=5&seed=42&logistic=1&neural=1&neural_obd=1&hidden_max=4&iter_budget=40&inner_val=0.25" \
    | grep -q '"ok":true' || fail "ceiling-limited CV run did not start"
for i in $(seq 1 200); do
    curl -s "$URL/api/train/status" > cv_ceiling.json
    grep -q '"running":false' cv_ceiling.json && break
    sleep 0.3
done
$PY - <<'PY' || fail "a ceiling-exhausted nested OBD must fail its folds, not fabricate"
import json
cv = json.load(open("cv_ceiling.json", encoding="utf-8"))["result"]["cv"]
t1, t2 = cv["tier1"], cv["tier2"]
assert "Neural (OBD)" in t1, t1
# no architecture footnote and no optimizer summary for a procedure that never fitted
assert "OBD selected" not in t1, t1
assert "Optimizer selection:" not in t2, t2
assert "ceiling" in t2, t2[:800]      # the failure reason is reported, per fold
PY

# 409 busy while a CV run owns the engine (shares the OBD/train job machinery)
# AND Stop must PROPAGATE into the running work (B1): a plain-neural CV with a
#    huge per-fold iteration cap and NO early stopping runs for many seconds if
#    uncancelled, so a prompt stop proves the cancel token reaches the fold's
#    training loop (the observer) rather than merely relabelling a finished run.
curl -s -X POST "$URL/api/cv" -d "folds=10&logistic=0&neural=1&neural_obd=0&neural_hidden=5&maxiter=2000000" \
    | grep -q '"ok":true' || fail "long CV run did not start"
cvbusy=$(curl -s -w '\n%{http_code}' -X POST "$URL/api/cv" -d "logistic=1")
echo "$cvbusy" | grep -q '"busy":true' || fail "no busy flag while a CV run runs"
echo "$cvbusy" | tail -1 | grep -q '409' || fail "CV busy refusal should be HTTP 409"
sleep 1
curl -s -X POST -d "" "$URL/api/train/stop" | grep -q '"ok":true' || fail "stop did not accept during CV"
cvStopped=0
for i in $(seq 1 40); do   # must finish well within ~12 s if Stop truly propagates
    curl -s "$URL/api/train/status" > cv_cancel.json
    grep -q '"running":false' cv_cancel.json && { cvStopped=1; break; }
    sleep 0.3
done
[ "$cvStopped" = 1 ] || fail "a long CV did not stop promptly after Stop (cancellation not propagated)"
grep -q '"cancelled":true' cv_cancel.json || fail "a stopped CV result must say cancelled"

# --- Locked-test inference (ROADMAP 4 Phase 4) -----------------------------
# Validation refusals first (Sol's caution: a prespecified contrast must name
#    SELECTED procedures; the split must be sized/composed correctly).
curl -s -X POST "$URL/api/cv" -d "logistic=1&neural=1&primary=qdfa&reference=logistic&locked_fraction=0.2" \
    | grep -q '"ok":false' || fail "a contrast naming an unselected procedure must be refused"
curl -s -X POST "$URL/api/cv" -d "logistic=1&neural=1&primary=neural&locked_fraction=0.2" \
    | grep -q '"ok":false' || fail "a contrast with primary but no reference must be refused"
# DLG-3: a locked test too small to hold >= 2 of each class is refused with counts,
#    not accepted then reported "AUC not computable" after an expensive job.
curl -s -X POST "$URL/api/cv" -d "logistic=1&neural=1&locked_n=3" \
    | grep -q '"ok":false' || fail "a locked test with < 2 of a class must be refused (DLG-3)"
# DLG-7: locked_fraction and locked_n are alternatives; both is a conflict.
curl -s -X POST "$URL/api/cv" -d "logistic=1&neural=1&locked_fraction=0.2&locked_n=40" \
    | grep -q '"ok":false' || fail "supplying both locked_fraction and locked_n must be refused"
# DLG-1: clustered inference is a follow-on -- it must be refused, never silently
#    run as ordinary DeLong.
curl -s -X POST "$URL/api/cv" -d "logistic=1&neural=1&locked_fraction=0.25&independence=cluster" \
    | grep -q '"ok":false' || fail "independence=cluster must be refused (not yet available)"

# DLG-1: WITHOUT a declared sampling unit, the locked test still scores + gives point
#    AUCs, but ordinary DeLong (CI + p) is WITHHELD -- an invalid p is never produced.
curl -s -X POST "$URL/api/cv" \
    -d "folds=5&seed=42&maxiter=4000&autostop_tol=0.01&logistic=1&neural=1&neural_obd=0&neural_hidden=4&locked_fraction=0.25" \
    | grep -q '"ok":true' || fail "locked-test CV (no declaration) did not start"
for i in $(seq 1 120); do curl -s "$URL/api/train/status" > cv_wh.json; grep -q '"running":false' cv_wh.json && break; sleep 0.3; done
$PY - <<'PY' || fail "locked-test withheld-inference result malformed"
import json
cv = json.load(open("cv_wh.json", encoding="utf-8"))["result"]["cv"]
assert "AUC (test)" in cv["tier1"] and "[95% CI]" not in cv["tier1"], cv["tier1"]  # point AUC, no CI
assert "DeLong p" not in cv["tier1"] and "withheld" in cv["tier1"], cv["tier1"]
lk = cv["locked"]
assert lk["inferenceRan"] is False, lk
assert lk["areas"][0]["auc"] is not None and lk["areas"][0]["lo"] is None, lk  # point AUC, no CI
assert lk["contrast"]["inferenceRan"] is False and lk["contrast"]["p"] is None, lk["contrast"]
PY

# Happy path: DECLARE independent rows -> DeLong runs. Default contrast Neural vs Logistic.
curl -s -X POST "$URL/api/cv" \
    -d "folds=5&seed=42&maxiter=4000&autostop_tol=0.01&logistic=1&neural=1&neural_obd=0&neural_hidden=4&locked_fraction=0.25&independence=rows" \
    | grep -q '"ok":true' || fail "locked-test CV (declared) did not start"
for i in $(seq 1 120); do
    curl -s "$URL/api/train/status" > cv_locked.json
    grep -q '"running":false' cv_locked.json && break
    sleep 0.3
done
grep -q '"running":false' cv_locked.json || fail "locked-test CV never completed"
$PY - <<'PY' || fail "locked-test CV result malformed"
import json
d = json.load(open("cv_locked.json", encoding="utf-8"))["result"]
assert d["ok"], d
cv = d["cv"]
# Declared -> Tier 1 gains the AUC(test) [95% CI] column, the DeLong verdict, the caveat.
assert "AUC (test) [95% CI]" in cv["tier1"], cv["tier1"]
assert ("DeLong p" in cv["tier1"]) or ("deterministic separation" in cv["tier1"]) \
    or ("no testable difference" in cv["tier1"]), cv["tier1"]
assert "inferential" in cv["tier1"] and "locked test set" in cv["tier1"], cv["tier1"]
lk = cv["locked"]
assert lk["inferenceRan"] is True and lk["n"] >= 4 and len(lk["areas"]) == 2, lk
assert lk["areas"][0]["lo"] is not None, lk  # a CI is present
# The prespecified contrast defaults to neural vs logistic, primary - reference.
c = lk["contrast"]
assert c and c["primary"] == "Neural" and c["reference"] == "Logistic", c
assert c["inferenceRan"] is True, c
assert c.get("degenerate") or c.get("separated") or (c["p"] is not None), c
# Tier 3 now writes FOUR files (adds cv_locked_predictions.csv), all reported ok.
assert len(cv["files"]) == 4, cv["files"]
assert cv.get("warnings", None) == [], cv.get("warnings")
PY
# The locked predictions file preserves row identity (raw id first), one column
#    per procedure -- the externally auditable pairing DeLong consumed.
[ -f cv_locked_predictions.csv ] || fail "no cv_locked_predictions.csv written"
head -1 cv_locked_predictions.csv | grep -q "^row,outcome,Logistic,Neural" \
    || fail "cv_locked_predictions.csv header wrong (row identity + one column per procedure)"

# --- Cross-validation FOLD POLICY (ROADMAP 4) -------------------------------
#     The CV request carries its own strata=/group=; it must never inherit the
#     Dataset panel's split configuration, and the two modes cannot be combined.
curl -s -X POST "$URL/api/cv" -d "folds=3&strata=3&group=4" \
    | grep -q "cannot be combined" || fail "strata + group must be refused for a fold plan"
curl -s -X POST "$URL/api/cv" -d "folds=3&group=99" \
    | grep -q "out of range" || fail "an out-of-range group column must be refused"
curl -s -X POST "$URL/api/cv" -d "folds=3&strata=3&strata_bins=1" \
    | grep -q "at least 2" || fail "strata_bins < 2 must be refused"

#     Declaring independent ROWS over a group-disjoint design is a contradiction:
#     grouping stops leakage, it does not make rows independent. Refused, never
#     relabelled "descriptive grouping" and handed to ordinary DeLong.
curl -s -X POST "$URL/api/cv" -d "folds=3&group=3,4,5&locked_fraction=0.25&independence=rows" \
    | grep -q "does not make rows independent" \
    || fail "independence=rows over a grouped design must be refused"

#     The clustered refusal names the MISSING prerequisite rather than one wall.
curl -s -X POST "$URL/api/cv" -d "folds=3&independence=cluster" \
    | grep -q "needs a group= key" || fail "clustered refusal must name the missing group key"
curl -s -X POST "$URL/api/cv" -d "folds=3&group=3,4,5&independence=cluster" \
    | grep -q "locked test" || fail "clustered refusal must name the missing locked test"

#     Covariate-stratified folds: the plan says what it did, in the user's own
#     1-based column numbers, and the machine-readable design says it structurally.
curl -s -X POST "$URL/api/cv" \
    -d "folds=3&seed=42&maxiter=4000&autostop_tol=0.01&logistic=1&neural=0&strata=3,4&strata_bins=4" \
    | grep -q '"ok":true' || fail "covariate-stratified CV did not start"
for i in $(seq 1 120); do curl -s "$URL/api/train/status" > cv_strata.json; grep -q '"running":false' cv_strata.json && break; sleep 0.3; done
$PY - <<'PYSTRATA' || fail "covariate-stratified CV result malformed"
import json
d = json.load(open("cv_strata.json", encoding="utf-8"))["result"]
assert d["ok"], d
t2 = d["cv"]["tier2"]
assert "outcome x covariate-stratified 3-fold" in t2, t2[:300]
assert "on columns 3, 4" in t2, t2[:300]      # the numbers the user typed
run = json.load(open("cv_run.json", encoding="utf-8"))
fd = run["foldDesign"]
assert fd["method"] == "outcome x covariate-stratified", fd
assert fd["strataColumns"] == [3, 4] and fd["strataBins"] == 4 and fd["strata"] > 2, fd
assert fd["groupColumns"] == [], fd
PYSTRATA

#     Group-aware folds AND a group-disjoint locked holdout from the SAME key.
#     Zero leakage is asserted from the ARTIFACTS, independently of the planner's
#     own count: no cluster id in two folds, and none in both the folded rows and
#     the locked test.
curl -s -X POST "$URL/api/cv" \
    -d "folds=3&seed=42&maxiter=4000&autostop_tol=0.01&logistic=1&neural=0&group=3,4,5&locked_fraction=0.25" \
    | grep -q '"ok":true' || fail "group-aware CV did not start"
for i in $(seq 1 120); do curl -s "$URL/api/train/status" > cv_group.json; grep -q '"running":false' cv_group.json && break; sleep 0.3; done
$PY - <<'PYGROUP' || fail "group-aware CV result malformed"
import csv, json
d = json.load(open("cv_group.json", encoding="utf-8"))["result"]
assert d["ok"], d
t1 = d["cv"]["tier1"]
# The locked holdout is NOT described as a row holdout -- what it holds out is groups.
assert "group-disjoint locked holdout" in t1, t1[:800]
assert "group-disjoint outcome-stratified" in t1, t1[:800]
lk = d["cv"]["locked"]
assert lk["splitMethod"] == "group-disjoint outcome-stratified", lk
assert lk["clusters"] >= 1, lk
assert lk["inferenceRan"] is False, lk   # no sampling unit declared

run = json.load(open("cv_run.json", encoding="utf-8"))
assert run["foldDesign"]["groupColumns"] == [3, 4, 5], run["foldDesign"]
assert run["foldDesign"]["leakage"] == 0, run["foldDesign"]
assert run["lockedTest"]["splitDesign"]["leakage"] == 0, run["lockedTest"]

folds = {}
with open("cv_predictions.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
assert "cluster" in rows[0], rows[0]
for r in rows:
    folds.setdefault(r["cluster"], set()).add(r["fold"])
assert all(len(v) == 1 for v in folds.values()), [c for c, v in folds.items() if len(v) > 1]

with open("cv_locked_predictions.csv", encoding="utf-8") as f:
    lrows = list(csv.DictReader(f))
assert "cluster" in lrows[0], lrows[0]
lockedClusters = {r["cluster"] for r in lrows}
assert not (lockedClusters & set(folds)), lockedClusters & set(folds)
PYGROUP

#     Nested OBD under a grouped fold plan: the architecture search runs on a
#     GROUP-DISJOINT inner validation split (unit-tested in check_crossval), and
#     end to end every fold must fit and no cluster may cross a fold.
curl -s -X POST "$URL/api/cv" \
    -d "folds=2&seed=42&group=3,4,5&logistic=0&neural=1&neural_obd=1&hidden_max=2&iter_budget=4000&autostop_tol=0.01&algorithm=2&inner_val=0.3" \
    | grep -q '"ok":true' || fail "grouped nested-OBD CV did not start"
for i in $(seq 1 300); do curl -s "$URL/api/train/status" > cv_gobd.json; grep -q '"running":false' cv_gobd.json && break; sleep 0.3; done
$PY - <<'PYGOBD' || fail "grouped nested-OBD CV result malformed"
import csv, json
d = json.load(open("cv_gobd.json", encoding="utf-8"))["result"]
assert d["ok"], d
run = json.load(open("cv_run.json", encoding="utf-8"))
p = run["procedures"][0]
assert p["validFolds"] == 2, p          # both folds fitted through the grouped inner split
assert len(p["arch"]) == 2, p           # and each recorded what it selected
assert run["foldDesign"]["method"] == "group-disjoint outcome-stratified", run["foldDesign"]
folds = {}
for r in csv.DictReader(open("cv_predictions.csv", encoding="utf-8")):
    folds.setdefault(r["cluster"], set()).add(r["fold"])
assert all(len(v) == 1 for v in folds.values()), [c for c, v in folds.items() if len(v) > 1]
PYGOBD

#     A grouped fold whose training rows are ONE group has no group-disjoint inner
#     split. It must FAIL that fold with the reason -- never fall back to a
#     row-wise inner validation, which would select on rows whose clusters the
#     model also trained on.
curl -s -X POST "$URL/api/cv" \
    -d "folds=2&seed=42&group=4&logistic=0&neural=1&neural_obd=1&hidden_max=2&iter_budget=4000&autostop_tol=0.01&algorithm=2" \
    | grep -q '"ok":true' || fail "single-group-fold CV did not start"
for i in $(seq 1 200); do curl -s "$URL/api/train/status" > cv_gthin.json; grep -q '"running":false' cv_gthin.json && break; sleep 0.3; done
$PY - <<'PYTHIN' || fail "infeasible grouped inner split not reported"
import json
run = json.load(open("cv_run.json", encoding="utf-8"))
p = run["procedures"][0]
assert p["validFolds"] == 0, p
assert p["failures"], p
assert all("single group" in f["reason"] for f in p["failures"]), p["failures"]
PYTHIN

#     CLUSTERED locked-test inference (Obuchowski). Declaring a clustered
#     sampling unit routes to the clustered covariance -- never to DeLong, and
#     never labelled as DeLong. The refusals name the missing prerequisite.
curl -s -X POST "$URL/api/cv" -d "folds=3&locked_fraction=0.3&independence=cluster" \
    | grep -q "needs a group= key" || fail "clustered without group= must be refused"
curl -s -X POST "$URL/api/cv" -d "folds=3&group=3,4,5&independence=cluster" \
    | grep -q "locked test" || fail "clustered without a locked test must be refused"

curl -s -X POST "$URL/api/cv" \
    -d "folds=3&seed=42&group=3,4,5&locked_fraction=0.3&independence=cluster&logistic=1&neural=1&neural_obd=0&neural_hidden=3&maxiter=4000&autostop_tol=0.01" \
    | grep -q '"ok":true' || fail "clustered CV did not start"
for i in $(seq 1 300); do curl -s "$URL/api/train/status" > cv_clust.json; grep -q '"running":false' cv_clust.json && break; sleep 0.3; done
$PY - <<'PYCLUST' || fail "clustered locked-test result malformed"
import json
d = json.load(open("cv_clust.json", encoding="utf-8"))["result"]
assert d["ok"], d
lk = d["cv"]["locked"]
assert lk["inferenceRan"] is True, lk
assert "Obuchowski" in lk["inferenceMethod"], lk
assert "DeLong" not in lk["inferenceMethod"], lk        # never mislabelled
assert lk["samplingUnit"].startswith("cluster"), lk
assert lk["independenceStatus"] == "declared: clustered observations", lk
assert lk["clusters"] >= 2, lk                          # independent units, not rows
assert lk["areas"][0]["lo"] is not None, lk             # a clustered interval
assert lk["contrast"]["inferenceRan"] is True, lk["contrast"]

t1 = d["cv"]["tier1"]
# The estimator names itself, and the headline says how many INDEPENDENT UNITS
# the p rests on -- a p from four clusters and one from four hundred otherwise
# read identically.
assert "clustered ROC p" in t1 or "no testable difference" in t1 or "deterministic separation" in t1, t1
assert "DeLong p" not in t1, t1
assert "independent cluster" in t1, t1
t2 = d["cv"]["tier2"]
assert "clusters in the locked sample:" in t2, t2[t2.find("Locked-test"):][:500]
assert "Obuchowski" in t2, t2[t2.find("Locked-test"):][:500]
# The standing scope note is the clustered one, not the DeLong one.
assert "treats the cluster, not the row" in t2, t2[t2.find("Locked-test"):][:900]

run = json.load(open("cv_run.json", encoding="utf-8"))
assert run["lockedTest"]["clusters"] >= 2, run["lockedTest"]
assert "Obuchowski" in run["lockedTest"]["inferenceMethod"], run["lockedTest"]

# THE NUMBERS, not just the label. Asserting only that the report SAYS
# "Obuchowski" leaves the label correct while the interval and the p come from
# the row-based estimator -- a sabotage that swapped the covariance while
# keeping the wording passed every structural assertion above. These are the
# clustered estimator's values on this seeded fixture; the row-based estimator
# on exactly the same fit gives [0.3695, 0.7017] and p = 0.2283, so a fallback
# cannot hide here.
a0 = lk["areas"][0]
assert abs(a0["auc"] - 0.535613) < 1e-5, a0        # the AREA is shared by both
assert abs(a0["lo"] - 0.320824) < 1e-4, a0         # the INTERVAL is not
assert abs(a0["hi"] - 0.750401) < 1e-4, a0
assert abs(lk["contrast"]["delta"] - 0.119658) < 1e-5, lk["contrast"]
assert lk["contrast"]["p"] < 1e-3, lk["contrast"]  # clustered 8.7e-05 vs row 0.23
PYCLUST

#     The clustered PREFLIGHT must actually run. c.unit used to be parsed AFTER
#     the locked-partition preflight, so the cluster check read an unset value
#     and never fired (Sol, 2026-07-30). This request has ample rows of each
#     class -- the identical request without the declaration runs fine -- and is
#     refused solely on the CLUSTER counts, before any async job starts.
cvclust_refusal=$(curl -s -X POST "$URL/api/cv" \
    -d "folds=2&group=3,4,5&locked_fraction=0.08&independence=cluster&logistic=1&neural=0")
echo "$cvclust_refusal" | grep -q '"ok":false' \
    || fail "a locked sample with too few informative clusters must be refused"
echo "$cvclust_refusal" | grep -q "two locked-test clusters carrying each outcome class" \
    || fail "the clustered refusal must explain the cluster counts"
curl -s "$URL/api/train/status" | grep -q '"running":false' \
    || fail "the clustered refusal must happen BEFORE the async job starts"
#     ...and the same locked sample is fine when no clustered inference is asked for,
#     which is what makes the refusal a cluster condition rather than a row one.
curl -s -X POST "$URL/api/cv" \
    -d "folds=2&seed=42&group=3,4,5&locked_fraction=0.08&logistic=1&neural=0&maxiter=4000&autostop_tol=0.01" \
    | grep -q '"ok":true' || fail "the same locked sample must run without a clustered declaration"
for i in $(seq 1 200); do curl -s "$URL/api/train/status" > cv_pf.json; grep -q '"running":false' cv_pf.json && break; sleep 0.3; done

#     ONE row-identity space across the two prediction files, and the structured
#     per-fold diagnostics DLG-8 promised.
curl -s -X POST "$URL/api/cv" \
    -d "folds=3&seed=42&group=3,4,5&locked_fraction=0.25&logistic=1&neural=0&maxiter=4000&autostop_tol=0.01" \
    | grep -q '"ok":true' || fail "identity/diagnostics CV did not start"
for i in $(seq 1 300); do curl -s "$URL/api/train/status" > cv_ident.json; grep -q '"running":false' cv_ident.json && break; sleep 0.3; done
$PY - <<'PYIDENT' || fail "raw-row identity or fold diagnostics malformed"
import csv, json
d = json.load(open("cv_ident.json", encoding="utf-8"))["result"]
assert d["ok"], d

# cv_predictions.csv's exemplar must be the ORIGINAL raw row, so the two files
# share one identity space. With a locked test the comparison is indexed by
# development row, and writing that index made row 7 a different patient in each.
dev = [int(r["exemplar"]) for r in csv.DictReader(open("cv_predictions.csv", encoding="utf-8"))]
lock = [int(r["row"]) for r in csv.DictReader(open("cv_locked_predictions.csv", encoding="utf-8"))]
assert len(set(dev)) == len(dev), "development raw ids repeat"
assert len(set(lock)) == len(lock), "locked raw ids repeat"
assert not (set(dev) & set(lock)), sorted(set(dev) & set(lock))[:10]
total = len(dev) + len(lock)
assert set(dev) | set(lock) == set(range(total)), "dev + locked do not cover every original row once"

# The per-fold diagnostics are present, recomputed, and consistent with the plan.
fd = json.load(open("cv_run.json", encoding="utf-8"))["foldDesign"]
k = fd["k"]
assert len(fd["foldRows"]) == k and len(fd["foldEvents"]) == k, fd
assert len(fd["foldGroups"]) == k, fd                 # a grouped plan reports clusters/fold
assert sum(fd["foldRows"]) == len(dev), (fd["foldRows"], len(dev))
assert sum(fd["foldGroups"]) == fd["groups"], fd      # every group in exactly one fold
assert all(e <= r for e, r in zip(fd["foldEvents"], fd["foldRows"])), fd
assert fd["largestGroup"] > 0 and fd["imbalance"] >= 0, fd
PYIDENT

# DLG-3 (k-aware development feasibility): a rare-event set where the locked test
#    can hold >= 2 of each class, but the development set is left with fewer than k
#    events -- so some outer fold could not contain an event. The request must be
#    refused BEFORE the async job, naming k and the achieved counts. 30 rows, 5
#    events: locked_n=12 -> 2 events / 10 non locked (ok), dev = 3 events < k=5.
$PY - <<'PY'
import random
random.seed(1)
rows=[]
for i in range(30):
    y = 1 if i < 5 else 0                 # exactly 5 events
    rows.append(f"{(i%7)/7.0:.4f} {(i%5)/5.0:.4f} {y}")
open("rare_events.set","w",newline="\n").write("\n".join(rows)+"\n")
PY
curl -s -X POST "$URL/api/load" -d "mode=raw&path=rare_events.set&fraction=0&inputs=2" >/dev/null
rareResp=$(curl -s -X POST "$URL/api/cv" -d "folds=5&logistic=1&neural=1&neural_obd=0&neural_hidden=3&locked_n=12&independence=rows")
echo "$rareResp" | grep -q '"ok":false' || fail "DLG-3: too few dev events for k folds must be refused"
echo "$rareResp" | grep -q "5-fold development" || fail "DLG-3 refusal must name the fold count and counts"
# reload the working dataset for anything after
curl -s -X POST "$URL/api/load" -d "mode=raw&path=lowbwt2-2train.txt&fraction=0.25&seed=1" >/dev/null

# --- The monitored held-out set is NAMED by the engine (2026-07-29) ----------
# The GUI's live chart and the OBD size search watch the VALIDATION set when one
# is loaded and the test set otherwise (DataSet::monitorSet). The page labels its
# charts from this answer instead of re-deriving the rule -- the legends were
# hard-coded to "test set" and so announced that a three-way split was being
# tuned on the untouched test set. Both arms are exercised.
twoway=$(curl -s -X POST "$URL/api/load" -d "mode=raw&path=lowbwt2-2train.txt&fraction=0.25&seed=1")
echo "$twoway" | grep -q '"monitor":"test"' \
    || fail "a two-way split must report the TEST set as the monitored set"
curl -s -X POST "$URL/api/model" -d "type=simpleprop&hidden=2" >/dev/null
curl -s -X POST "$URL/api/train" -d "algorithm=1&maxiter=20000&seed=42" > mon_two.json
grep -q '"monitor":"test"' mon_two.json \
    || fail "a two-way training run must name the test set as its monitor"

threeway=$(curl -s -X POST "$URL/api/load" \
    -d "mode=raw&path=lowbwt2-2train.txt&fraction=0.25&val_fraction=0.25&seed=1")
echo "$threeway" | grep -q '"monitor":"validation"' \
    || fail "a three-way split must report the VALIDATION set as the monitored set"
echo "$threeway" | grep -q 'validation exemplars' \
    || fail "the three-way load did not actually create a validation set"
curl -s -X POST "$URL/api/model" -d "type=simpleprop&hidden=2" >/dev/null
curl -s -X POST "$URL/api/obd" \
    -d "hidden_start=2&hidden_max=3&iter_budget=20000&algorithm=1&seed=42" \
    | grep -q '"ok":true' || fail "three-way OBD did not start"
for i in $(seq 1 300); do
    curl -s "$URL/api/train/status" > obd_val.json
    grep -q '"running":false' obd_val.json && break
    sleep 0.2
done
$PY - <<'PY' || fail "three-way OBD did not name validation as the scored set"
import json
r = json.load(open("obd_val.json", encoding="utf-8"))["result"]
assert r["ok"], r.get("message")
# The size search early-stops and SCORES on validation here, so the chart legend
# built from this must say validation -- not test, which is untouched.
assert r["monitor"] == "validation", r["monitor"]
PY

# --- Structured cross-validation progress (2026-07-29) ----------------------
# A four-procedure, five-fold nested-OBD run used to publish one unchanging phase
# word ("cross-validating") for its entire duration. The status poll now carries
# the repetition grid from the coordinator/runner (crossval::Progress) and the
# architecture trial from the nested search (obd::ProgressFn) -- composed, never
# scraped from report prose. Polled tightly for the whole run, then the union of
# what was seen is asserted, so this does not race one particular instant.
curl -s -X POST "$URL/api/load" -d "mode=raw&path=lowbwt2-2train.txt&fraction=0&seed=1" >/dev/null
curl -s -X POST "$URL/api/cv" \
    -d "folds=5&seed=42&maxiter=4000&autostop_tol=0.01&logistic=1&neural=1&neural_obd=1&hidden_max=4&iter_budget=1500&inner_val=0.25&algorithm=auto&locked_fraction=0.25&independence=rows" \
    | grep -q '"ok":true' || fail "progress-instrumented CV did not start"
: > cv_prog.ndjson
for i in $(seq 1 4000); do
    curl -s "$URL/api/train/status" > cv_prog_last.json
    $PY -c "
import json,sys
d = json.load(open('cv_prog_last.json', encoding='utf-8'))
cv = d.get('cv')
if cv: print(json.dumps(cv))
" >> cv_prog.ndjson 2>/dev/null
    grep -q '"running":false' cv_prog_last.json && break
done
grep -q '"running":false' cv_prog_last.json || fail "instrumented CV never completed"
$PY - <<'PY' || fail "cross-validation published no usable structured progress"
import json
seen = [json.loads(l) for l in open("cv_prog.ndjson", encoding="utf-8") if l.strip()]
assert seen, "no cv progress object was ever published"

cvs = [s for s in seen if s["stage"] == "cross-validation"]
lks = [s for s in seen if s["stage"] == "locked-test evaluation"]
assert cvs, "the cross-validation stage never reported"
assert lks, "the locked-test evaluation stage never reported"

# The repetition grid: both procedures named, folds inside 1..k, k reported.
#    This is SAMPLED progress from a running job, so it asserts what polling can
#    reliably see -- that the counters are live and advance. The exhaustive grid
#    (every procedure x every fold, in order, completedFolds reaching k) is pinned
#    deterministically in check_crossval, which observes every event instead of
#    sampling; asserting a specific transient here would only be a flaky test.
names = {s["procedure"] for s in cvs if s["procedure"]}
assert "Logistic" in names and "Neural (OBD)" in names, names
assert all(s["folds"] == 5 for s in cvs), "k was not reported as 5"
assert all(1 <= s["fold"] <= 5 for s in cvs if s["fold"]), "a fold outside 1..k"
assert len({s["fold"] for s in cvs}) > 1, "the fold counter never advanced"
assert max(s["completedFolds"] for s in cvs) >= 1, "completed folds never advanced"
assert all(s["completedFolds"] < s["folds"] or s["fold"] == s["folds"] for s in cvs), \
    "completed folds ran past k"
assert all(s["procedureCount"] == 2 for s in cvs if s["procedureCount"]), cvs[0]

# The nested architecture search inside the CURRENT fold. Its absence means "no
# nested search running", never "a search at zero nodes" -- so when present the
# phase must be a real one and the node count nonzero.
inner = [s["inner"] for s in cvs if s.get("inner")]
assert inner, "the nested OBD search never reported a trial"
assert all(i["phase"] in ("probing optimizers", "grow", "prune", "final")
           for i in inner), inner[:3]
assert all(i["hidden"] >= 1 for i in inner), inner[:3]
# The optimizer probe is wall-clock budgeted per candidate and can dominate a
# fold's elapsed time; with algorithm=auto it must be visible, not silence.
assert any(i["phase"] == "probing optimizers" for i in inner), \
    "the auto-optimizer probe inside a fold reported nothing"
# Only the nested-OBD procedure may report inner trials -- Logistic has none.
assert all(s["procedure"] == "Neural (OBD)" for s in cvs if s.get("inner")), \
    "a non-nested procedure reported an architecture trial"

# The locked pass folds NOTHING: it refits once on the development rows and
# scores once, so it must not invent a fold number.
assert all(s["fold"] == 0 and s["folds"] == 0 for s in lks), lks[:3]
assert {s["procedure"] for s in lks if s["procedure"]} == {"Logistic", "Neural (OBD)"}, lks
print("structured CV progress ok:", len(seen), "samples")
PY

# --- Tier 2's fold-plan counts describe the rows it FOLDED (2026-07-29) ------
# With a locked test the Tier-2 header says "(development rows only)" and used to
# print the whole dataset's n and events beside fold rows a quarter that size.
$PY - <<'PY' || fail "Tier-2 development counts disagree with the fold plan"
import json, re
d = json.load(open("cv_prog_last.json", encoding="utf-8"))["result"]
cv, lk = d["cv"], d["cv"]["locked"]
t1, t2 = cv["tier1"], cv["tier2"]

# Tier 1 is about the DATASET and correctly reports its totals.
total = int(re.search(r"(\d+) exemplars", t1).group(1))
totalEvents = int(re.search(r"(\d+) events", t1).group(1))

# Tier 2's fold-plan header is about the rows the plan covered.
m = re.search(r"Fold plan: (.*?)\s+k = (\d+)\s+n = (\d+)\s+events = (\d+)", t2)
assert m, t2[:300]
plan, k, n, events = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
assert "development rows only" in plan, plan
# Development = everything the locked test did not take. Both counts, not just n:
# n was already the folded count while events came from the dataset totals, so a
# row count and an event count described different sets of rows.
assert n == total - lk["n"], f"development n {n} != total {total} - locked {lk['n']}"
assert events == totalEvents - lk["events"], \
    f"development events {events} != total {totalEvents} - locked {lk['events']}"
assert n != total and events != totalEvents, \
    "the fixture has no locked test, so this proves nothing"

# ...and it must agree with the per-fold rows printed directly beneath it.
folds = [int(x) for x in re.findall(r"^\s+(?:\d+)\s+(\d+)\s", t2, re.M)]
assert folds, t2[:400]
assert sum(folds[:k]) == n, f"fold rows sum to {sum(folds[:k])}, header says {n}"

# The machine-readable artifact carries the same two numbers.
run = json.load(open("cv_run.json", encoding="utf-8"))
assert run["n"] == n and run["events"] == events, (run["n"], run["events"], n, events)
print(f"Tier 2 development counts ok: n={n} events={events} locked={lk['n']} total={total}")
PY

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

echo "OK: GUI endpoints (version, page, load incl. pre-split pair, model, train + ROC + full stats JSON, /api/stats, binormal fits + null when impossible, logistic Wald/condition number, stepwise visible report + async progress/Stop + structured audit/completeness + convergence/LMS refusals + quiet candidates, saves, plateau auto-stop + control + validation, train-panel parity controls (learning rate/weight decay/batch-epoch/stopping conditions/print counter) + behavioral proof + validation, model-panel parity (bias->BareProp/multi-layer->BackProp/error function/load-network), no-bias BackProp save/load round trip, multipart log toggles, logistic batch/epoch guard, DFA (linear/quadratic + report/stats + guesses saves), dataset characteristics + ROC reporting load params, test_n exact split, covariate stratification + diagnostic, group-aware split + zero-leakage diagnostic, three-way validation split, multi-output load/BackProp/train/DFA + logistic refusal, async train/status/stop + 409 busy + cancel, algorithm=auto blocking + async, OBD sizing (refusal/async grow-then-prune/obd status phase/plain-train has none/409 busy/cancel), locked-test DeLong inference (contrast validation refusals + AUC(test) column + prespecified contrast p + cv_locked_predictions.csv row identity + four Tier-3 files), per-action audit log)"
