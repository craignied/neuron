# Deploying a trained model: `tools/neuron2web.py`

`neuron2web.py` turns a trained neuron model into a **single self-contained
HTML file** — a calculator with your labels on it, the scaling factors and
weights embedded, and the forward pass in inline JavaScript. Nothing else is
needed to run it: open it from disk, email it to a colleague, or upload it
to any static web host (Dreamhost, GitHub Pages, a departmental server) and
it works. All computation happens in the visitor's browser; no data is sent
anywhere.

This is the successor to neUROn2++'s `neuron2html.pl` (manifest.pdf ch. 11),
with the same three inputs and the same philosophy.

```
python3 tools/neuron2web.py --network NET.txt --scales SCALES.txt \
    --spec SPEC.txt -o calculator.html [--serve] [--title TEXT]
```

## The three input files

### 1. The network file (`--network`)

What the engine writes from *Use model → Save the network to a file* (or
the save prompt after training). Supported types: **SimpleProp** and
**Binary logistic** (the first line of the file says which). BareProp and
BackProp are not supported yet.

### 2. The scaling-factors file (`--scales`)

What the engine writes from *Specify dataset → Save scaling factors*. The
engine trains on inputs scaled to [−0.9, 0.9]; your calculator's users type
natural units (age 44, balance 1787). This file carries the
`x_norm = S·(x − x_min) + lbound` transform that connects them.

**Save it in the same session that converts or randomizes the raw dataset**
— the option only becomes available then, and the factors must come from
the same training set the network was trained on. If you train a model and
save only the network, you cannot deploy it: train again and save both.

### 3. The label spec (`--spec`)

A plain-text file, one tag per line, fields separated by `%%`. Lines
starting with `#` are comments. The tags are the legacy neUROn2++ exporter
format, carried forward:

| Tag | Form | Meaning | Columns |
|---|---|---|---|
| `N` | `N %% Age %% years` | Numerical input with units | 1 |
| `C` | `C %% Education %% *primary %% secondary %% tertiary` | Categorical input: a dropdown of choices, one-hot encoded | one per non-starred choice |
| `B` | `B %% Housing loan %% yes %% no` | Binary input: *true-label first*; the column is 1 when true | 1 |
| `K` | `K %% N %% PSA %% ng/mL` | Numerical input that may be unknown: a missing-indicator **pair**, `1, value` when known, `0, 0` when unknown (what `mkdataset.py` emits for columns with blanks) | 2 |
| `O` | `O %% Cancer absent %% Cancer present` | Outcome labels (0-label, then 1-label). **Required, once.** | — |
| `R` | `R %% Odds of cancer on biopsy` | Label for the odds line (optional; omit it and the page shows both outcome probabilities instead) | — |

**The `*` reference marker** (new in 3.0): in a `C` group, star exactly one
choice to mark it as the **reference level** of a `--refcat`-groomed
variable. It appears in the dropdown like any other choice but contributes
no input column — selecting it means "all this group's columns are 0",
exactly how the engine saw the reference category during training.

- Groomed with `--onehot --refcat` → star the first (alphabetical) level of
  every categorical variable, because that is the level `mkdataset.py`
  dropped. Your `*_key.txt` file says so explicitly
  (`reference = admin.: ...`).
- Groomed with `--onehot` alone (a neural network on full one-hot) → no
  stars; list every level.
- A two-level variable groomed with `--refcat` collapses to one column —
  spec it as `B` (true-label = the non-reference level).

**Order matters.** Variables must appear in groomed-column order, and a
`C` group's choices must be listed in column order (alphabetical, when the
data came from `mkdataset.py`). Write the spec with the key file open next
to you — it lists exactly which output columns belong to which variable, in
order. The tool verifies that the spec expands to exactly the network's
input count and prints the per-variable expansion when it doesn't.

## Checking a deployment: `--eval`

```
python3 tools/neuron2web.py --network NET --scales SCALES --spec SPEC \
    --eval "30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, ..."
```

Prints the model's probability for one row of natural-unit column values
(a groomed data row without its outcome column) using the same arithmetic
the page embeds. Feed it a few rows whose outcomes you know before you
publish a calculator. The fixture test in `tests/tools/run_tools.sh` uses
this to hold the tool's forward pass equal to the engine's own guesses on
the committed XOR reference network (agreement to 6 significant digits).

## Previewing: `--serve`

`--serve` starts a loopback-only HTTP server on an **OS-assigned free
port** (no collisions with anything else you run), prints the URL, and
opens your browser. It is purely a local preview: the .html file is the
deployment artifact, and uploading it anywhere static is the deployment.

## Worked example

The bank-marketing walkthrough ends by deploying its trained logistic
model; `docs/datasets/bank-marketing/bank_spec.txt` is a complete
committed spec for a 42-input `--refcat` model. The short version:

```
$ python3 tools/neuron2web.py \
    --network bank_logistic_net.txt --scales bank_scales.txt \
    --spec docs/datasets/bank-marketing/bank_spec.txt \
    -o bank_calculator.html --serve
Deployed Binary logistic (42 inputs) to bank_calculator.html
Serving the calculator at http://127.0.0.1:<OS-assigned port>/  (Ctrl-C to stop)
```

## What the page computes

The embedded JavaScript is a line-for-line mirror of the engine's forward
pass (`src/function_defs.h` sigmoids, bias as the last weight of each
row): scale each column, then for logistic `p = σ(β·x + b)`, for SimpleProp
`p = σ(w·σ(Hx + h) + b)`. Probability and odds are displayed with your `O`
and `R` labels. The numbers agree with the engine to the precision the
engine saves its weights with (6 significant digits).
