#!/usr/bin/env python3
"""Deploy a trained neuron model as a self-contained HTML calculator.

The rebirth of neUROn2++'s neuron2html.pl (manifest.pdf ch. 11). Standard
library only — runs on a bare python3 (see tools/README.md).

Takes the same trio of files the original exporter did:

  1. the network file saved by the engine ("Save the network to a file");
     SimpleProp and Binary logistic models are supported;
  2. the scaling-factors file ("Save scaling factors") — the page's users
     type natural units, and the forward pass needs the engine's scaling
     x_norm = S * (x - x_min) + lbound applied first;
  3. a label spec describing how to present each input and the output,
     in the legacy tag format (docs/deploy.md has the full reference):

         N %% Age %% years
         C %% Education %% *primary %% secondary %% tertiary %% unknown
         B %% Housing loan %% yes %% no
         K %% N %% PSA %% ng/mL
         O %% No subscription %% Subscription
         R %% Odds of subscription

     A choice marked with '*' is the reference level of a --refcat-groomed
     one-hot group: it appears in the menu but contributes no input column.

The output is ONE .html file with no external dependencies: form, scaling,
and forward pass (sigmoids exactly as src/function_defs.h) all inline. It
works opened straight from disk; --serve additionally serves it on
127.0.0.1 with an OS-assigned free port and opens the browser.

--eval "v1, v2, ..." skips the HTML and prints the model's probability for
one row of natural-unit column values (the same expanded columns a groomed
data row has, without the outcome) — used by the fixture tests to verify
this forward pass against the engine's, and handy for spot checks.
"""

import argparse
import html
import sys
from math import exp


def die(message):
    sys.stderr.write("ERROR: " + message + "\n")
    sys.exit(1)


def read_lines(path, what):
    try:
        with open(path, "r") as f:
            return f.read().splitlines()
    except OSError:
        die("cannot open " + what + " file " + path)


def floats(line):
    return [float(tok) for tok in line.split()]


# --- The three input files ----------------------------------------------

def parse_network(path):
    """Return a dict describing the saved model."""
    lines = read_lines(path, "network")
    if not lines:
        die(path + " is empty")
    kind = lines[0].strip()

    if kind == "Binary logistic":
        n_in = int(lines[1].split()[0])
        if "y-intercept:" not in lines[3]:
            die(path + ": expected 'y-intercept:' on line 4")
        intercept = float(lines[4])
        betas = floats(lines[6])
        if len(betas) != n_in:
            die(path + f": {len(betas)} beta weights for {n_in} inputs")
        return {"kind": kind, "n_in": n_in,
                "betas": betas, "intercept": intercept}

    if kind == "SimpleProp":
        n_in = int(lines[2].split()[0])
        n_hid = int(lines[4].split()[0])
        if lines[6].strip() != "Hidden layer:":
            die(path + ": expected 'Hidden layer:' on line 7")
        hw = [floats(lines[7 + j]) for j in range(n_hid)]
        if lines[7 + n_hid].strip() != "Output layer:":
            die(path + ": expected 'Output layer:' after the hidden rows")
        ow = floats(lines[8 + n_hid])
        for j, row in enumerate(hw):
            if len(row) != n_in + 1:
                die(path + f": hidden row {j + 1} has {len(row)} weights, "
                    f"expected {n_in + 1} (inputs + bias)")
        if len(ow) != n_hid + 1:
            die(path + f": output row has {len(ow)} weights, "
                f"expected {n_hid + 1} (hidden + bias)")
        return {"kind": kind, "n_in": n_in, "hw": hw, "ow": ow}

    die(path + ": network type '" + kind + "' is not supported yet "
        "(SimpleProp and Binary logistic are)")


def parse_scales(path, n_in):
    """Return (lbound, S list, x_min list) from a saved scaling file."""
    lines = read_lines(path, "scaling-factors")
    if len(lines) < 5 or "x_norm" not in lines[0]:
        die(path + " does not look like a saved scaling-factors file")
    try:
        lbound = float(lines[1].split("lbound =")[1].split(",")[0])
    except (IndexError, ValueError):
        die(path + ": cannot parse lbound from line 2")
    S = floats(lines[2])
    xmin = floats(lines[4])
    if len(S) != n_in or len(xmin) != n_in:
        die(path + f": {len(S)} scale factors / {len(xmin)} minima "
            f"for a {n_in}-input network")
    for i, s in enumerate(S):
        if s != s or s in (float("inf"), float("-inf")):
            die(path + f": scale factor {i + 1} is not finite "
                "(constant column in the training set?)")
    return lbound, S, xmin


def parse_spec(path):
    """Return (variables list, outcome labels, odds label).

    Each variable is a dict with kind N/C/B/KN and n_cols set.
    """
    variables = []
    outcome = None
    odds = None

    for lineno, raw in enumerate(read_lines(path, "spec"), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = [f.strip() for f in line.split("%%")]
        tag = fields[0]

        def need(n, usage):
            if len(fields) < n:
                die(f"{path} line {lineno}: {usage}")

        if tag == "N":
            need(3, "N %% name %% units")
            variables.append({"kind": "N", "name": fields[1],
                              "units": fields[2], "n_cols": 1})
        elif tag == "C":
            need(3, "C %% name %% choice1 %% choice2 ...")
            choices = []
            n_ref = 0
            for c in fields[2:]:
                is_ref = c.startswith("*")
                n_ref += is_ref
                choices.append((c[1:].strip() if is_ref else c, is_ref))
            if n_ref > 1:
                die(f"{path} line {lineno}: more than one '*' reference level")
            if len(choices) - n_ref < 1:
                die(f"{path} line {lineno}: no non-reference choices left")
            variables.append({"kind": "C", "name": fields[1],
                              "choices": choices,
                              "n_cols": len(choices) - n_ref})
        elif tag == "B":
            need(4, "B %% name %% true-label %% false-label")
            variables.append({"kind": "B", "name": fields[1],
                              "true": fields[2], "false": fields[3],
                              "n_cols": 1})
        elif tag == "K":
            need(2, "K %% N %% name %% units")
            if fields[1] != "N":
                die(f"{path} line {lineno}: only K %% N (numerical "
                    "missing-indicator pairs) is supported")
            need(4, "K %% N %% name %% units")
            variables.append({"kind": "KN", "name": fields[2],
                              "units": fields[3], "n_cols": 2})
        elif tag == "O":
            need(3, "O %% 0-label %% 1-label")
            outcome = (fields[1], fields[2])
        elif tag == "R":
            need(2, "R %% odds label")
            odds = fields[1]
        else:
            die(f"{path} line {lineno}: unknown tag '{tag}'")

    if outcome is None:
        die(path + ": an O %% 0-label %% 1-label line is required")
    return variables, outcome, odds


def check_columns(variables, n_in, spec_path):
    """Verify the spec expands to exactly the network's input count."""
    col = 0
    spans = []
    for v in variables:
        first = col + 1
        col += v["n_cols"]
        spans.append(f"  columns {first}-{col}: {v['name']} ({v['kind']})"
                     if v["n_cols"] > 1 else
                     f"  column {first}: {v['name']} ({v['kind']})")
    if col != n_in:
        die(f"{spec_path} describes {col} input columns, but the network "
            f"has {n_in} inputs. The spec's expansion:\n" + "\n".join(spans))


# --- The forward pass (Python side; the generated JS mirrors it) ---------

def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


def predict(net, lbound, S, xmin, cols):
    xs = [S[i] * (cols[i] - xmin[i]) + lbound for i in range(len(cols))]
    if net["kind"] == "Binary logistic":
        z = net["intercept"]
        for i, b in enumerate(net["betas"]):
            z += b * xs[i]
        return sigmoid(z)
    # SimpleProp: bias is the last weight of every row (function_defs.h
    # sigmoid on hidden and output layers, hidden bias node fixed at 1)
    h = []
    for row in net["hw"]:
        z = row[-1]
        for i in range(net["n_in"]):
            z += row[i] * xs[i]
        h.append(sigmoid(z))
    z = net["ow"][-1]
    for j, hj in enumerate(h):
        z += net["ow"][j] * hj
    return sigmoid(z)


# --- HTML generation ------------------------------------------------------

def js_list(values):
    return "[" + ", ".join(repr(v) for v in values) + "]"


def build_form(variables):
    """Static form HTML plus the straight-line JS that collects columns."""
    rows = []
    collect = []
    for n, v in enumerate(variables):
        vid = f"v{n}"
        name = html.escape(v["name"])
        if v["kind"] == "N":
            rows.append(
                f'<label for="{vid}">{name}</label>'
                f'<span><input type="number" step="any" id="{vid}" value="0">'
                f' {html.escape(v["units"])}</span>')
            collect.append(f'  cols.push(num("{vid}"));')
        elif v["kind"] == "B":
            rows.append(
                f'<label for="{vid}">{name}</label>'
                f'<span><select id="{vid}">'
                f'<option value="1">{html.escape(v["true"])}</option>'
                f'<option value="0">{html.escape(v["false"])}</option>'
                f'</select></span>')
            collect.append(f'  cols.push(num("{vid}"));')
        elif v["kind"] == "C":
            opts = []
            col_of_choice = []  # 0-based column within the group, or -1
            k = 0
            for label, is_ref in v["choices"]:
                col_of_choice.append(-1 if is_ref else k)
                k += not is_ref
                opts.append(f'<option value="{col_of_choice[-1]}">'
                            f'{html.escape(label)}</option>')
            rows.append(
                f'<label for="{vid}">{name}</label>'
                f'<span><select id="{vid}">{"".join(opts)}</select></span>')
            collect.append(
                f'  onehot(cols, num("{vid}"), {v["n_cols"]});')
        elif v["kind"] == "KN":
            rows.append(
                f'<label for="{vid}">{name}</label>'
                f'<span><input type="number" step="any" id="{vid}" value="0">'
                f' {html.escape(v["units"])} '
                f'<label class="unk"><input type="checkbox" id="{vid}u">'
                f' unknown</label></span>')
            collect.append(
                f'  if (document.getElementById("{vid}u").checked)'
                f' {{ cols.push(0, 0); }} else'
                f' {{ cols.push(1, num("{vid}")); }}')
    return "\n".join(rows), "\n".join(collect)


def build_model_js(net):
    if net["kind"] == "Binary logistic":
        return (f'const BETA = {js_list(net["betas"])};\n'
                f'const INTERCEPT = {net["intercept"]!r};\n'
                'function model(xs) {\n'
                '  let z = INTERCEPT;\n'
                '  for (let i = 0; i < BETA.length; i++) z += BETA[i] * xs[i];\n'
                '  return sigmoid(z);\n'
                '}')
    hw = "[" + ",\n  ".join(js_list(r) for r in net["hw"]) + "]"
    return (f'const HW = {hw};\n'
            f'const OW = {js_list(net["ow"])};\n'
            'function model(xs) {\n'
            '  let z = OW[OW.length - 1];\n'
            '  for (let j = 0; j < HW.length; j++) {\n'
            '    let h = HW[j][HW[j].length - 1];\n'
            '    for (let i = 0; i < xs.length; i++) h += HW[j][i] * xs[i];\n'
            '    z += OW[j] * sigmoid(h);\n'
            '  }\n'
            '  return sigmoid(z);\n'
            '}')


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<!-- Generated by neuron2web.py (neuron 3.0) from {netfile}. -->
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 34rem;
       padding: 0 1rem; color: #1a1a2e; background: #fafafa; }}
h1 {{ font-size: 1.3rem; }}
form {{ display: grid; grid-template-columns: 1fr auto; gap: .55rem .9rem;
       align-items: center; background: #fff; border: 1px solid #ddd;
       border-radius: 8px; padding: 1.1rem; }}
label {{ font-size: .95rem; }}
label.unk {{ font-size: .8rem; color: #555; }}
input[type=number] {{ width: 7rem; }}
#go {{ grid-column: 1 / -1; padding: .5rem; font-size: 1rem; cursor: pointer; }}
#out {{ margin-top: 1.2rem; padding: 1.1rem; background: #fff;
       border: 1px solid #ddd; border-radius: 8px; display: none; }}
#out .p {{ font-size: 1.5rem; font-weight: 600; }}
.foot {{ margin-top: 1.5rem; font-size: .75rem; color: #777; }}
</style>
</head>
<body>
<h1>{title}</h1>
<form onsubmit="run(); return false;">
{form_rows}
<button id="go" type="submit">Compute</button>
</form>
<div id="out">
  <div class="p" id="prob"></div>
  <div id="oddsline"></div>
</div>
<p class="foot">Model deployed with neuron2web.py (neuron 3.0). The
computation runs entirely in this page; no data leaves your machine.</p>
<script>
"use strict";
const LB = {lbound!r};
const S = {S};
const XMIN = {xmin};
{model_js}
function sigmoid(x) {{ return 1 / (1 + Math.exp(-x)); }}
function num(id) {{ return parseFloat(document.getElementById(id).value) || 0; }}
function onehot(cols, hot, k) {{
  for (let i = 0; i < k; i++) cols.push(i === hot ? 1 : 0);
}}
function run() {{
  const cols = [];
{collect_js}
  const xs = cols.map((x, i) => S[i] * (x - XMIN[i]) + LB);
  const p = model(xs);
  const win = p >= 0.5;
  const pct = (100 * (win ? p : 1 - p)).toFixed(1);
  document.getElementById("prob").textContent =
    (win ? "{label1}" : "{label0}") + ": " + pct + "% probability";
  const odds = p / (1 - p);
  document.getElementById("oddsline").textContent = {odds_expr};
  document.getElementById("out").style.display = "block";
}}
</script>
</body>
</html>
"""


def build_page(net, netfile, lbound, S, xmin, variables, outcome, odds,
               title):
    form_rows, collect_js = build_form(variables)
    if odds:
        odds_expr = (f'"{html.escape(odds)}: " + '
                     '(odds >= 1 ? odds.toFixed(2) + " to 1"'
                     ' : "1 to " + (1 / odds).toFixed(2))')
    else:
        # No R tag: the second line carries the *less* probable outcome,
        # complementing the headline.
        odds_expr = ('(win ? "' + html.escape(outcome[0]) + '"'
                     ' : "' + html.escape(outcome[1]) + '") + ": " + '
                     '(100 * (win ? 1 - p : p)).toFixed(1) + "% probability"')
    return PAGE.format(
        title=html.escape(title),
        netfile=html.escape(netfile),
        form_rows=form_rows,
        lbound=lbound,
        S=js_list(S),
        xmin=js_list(xmin),
        model_js=build_model_js(net),
        collect_js=collect_js,
        label0=html.escape(outcome[0]),
        label1=html.escape(outcome[1]),
        odds_expr=odds_expr,
    )


# --- Serving --------------------------------------------------------------

def serve(page_bytes, quiet):
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page_bytes)))
            self.end_headers()
            self.wfile.write(page_bytes)

        def log_message(self, *args):
            pass

    # Port 0: the OS assigns a free port, so this never collides with
    # anything else running locally. Loopback only.
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"Serving the calculator at {url}  (Ctrl-C to stop)")
    webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        if not quiet:
            print("\nStopped.")


# --- Main -----------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Deploy a trained neuron model as a self-contained "
                    "HTML calculator (rebirth of neuron2html.pl)")
    parser.add_argument("-network", "--network", required=True, metavar="FILE",
                        help="network saved by the engine")
    parser.add_argument("-scales", "--scales", required=True, metavar="FILE",
                        help="scaling factors saved by the engine")
    parser.add_argument("-spec", "--spec", required=True, metavar="FILE",
                        help="input/output label spec (see docs/deploy.md)")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="output .html file (default: <network>.html)")
    parser.add_argument("-title", "--title", metavar="TEXT",
                        help="page title (default: the O tag's 1-label)")
    parser.add_argument("-serve", "--serve", action="store_true",
                        help="serve on 127.0.0.1, OS-assigned free port, "
                             "and open the browser")
    parser.add_argument("-eval", "--eval", metavar="ROW", dest="eval_row",
                        help="print the probability for one comma-separated "
                             "row of natural-unit column values and exit")
    parser.add_argument("-quiet", "--quiet", action="store_true",
                        help="suppress messages")
    return parser.parse_args()


def main():
    args = parse_args()

    net = parse_network(args.network)
    lbound, S, xmin = parse_scales(args.scales, net["n_in"])
    variables, outcome, odds = parse_spec(args.spec)
    check_columns(variables, net["n_in"], args.spec)

    if args.eval_row is not None:
        cols = [float(tok) for tok in args.eval_row.split(",")]
        if len(cols) != net["n_in"]:
            die(f"--eval row has {len(cols)} values; "
                f"the network has {net['n_in']} inputs")
        print(f"{predict(net, lbound, S, xmin, cols):.6g}")
        return

    title = args.title or outcome[1]
    page = build_page(net, args.network, lbound, S, xmin,
                      variables, outcome, odds, title)

    out_path = args.output or (args.network.rsplit(".", 1)[0] + ".html")
    try:
        with open(out_path, "w", newline="\n") as f:
            f.write(page)
    except OSError:
        die("cannot open output file " + out_path)
    if not args.quiet:
        print(f"Deployed {net['kind']} ({net['n_in']} inputs) to {out_path}")

    if args.serve:
        serve(page.encode("utf-8"), args.quiet)


if __name__ == "__main__":
    main()
