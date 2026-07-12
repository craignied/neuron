# tools/ — Python data-preparation utilities

Companions to the C++ engine for grooming data before modeling.

**Hard rule: standard library only.** Every tool here runs on a bare `python3`
(3.8+) with **no pip installs, no venv, no requirements.txt** — the heavy math
belongs to the engine, so the tools never need numpy. CI runs the tools with
the system `python3` on macOS, Linux, and Windows on every push; an import
outside the standard library fails the build. If a future tool genuinely needs
third-party packages, it goes in a clearly marked optional subdirectory — it
does not change this rule.

## mkdataset.py

Port of neUROn2++'s `mkdataset.pl` (manifest.pdf ch. 11). Converts an Excel
`.csv` export into a neuron-ready raw dataset:

```sh
python3 tools/mkdataset.py --key key.txt --inputs inputs.txt -o data.txt mydata.csv
```

- Drops rows whose last column (the outcome) is blank, with a warning
  (`--noelim` to keep them).
- Drops rows containing non-numeric data, naming the offending columns.
- Columns with missing values become indicator **pairs** — `0, 0` when blank,
  `1, value` when present — so the model can learn from missingness.
- `--key` records which output-file columns correspond to which labels;
  `--inputs` writes the variable-structure string for stepwise regression
  (one-hot groups appear as one variable, `1-4` range style).
- Improvement over the Perl: input is parsed with Python's `csv` module, so
  quoted Excel fields (embedded commas) are handled correctly.

Beyond the Perl original (added for real-world exports like the UCI
bank-marketing data — see `docs/datasets/bank-marketing/WALKTHROUGH.md`):

- `--delimiter ';'` — semicolon- (or tab-: `'\t'`) delimited input.
- `--onehot` — encode categorical text columns as one-hot indicator groups,
  one 0/1 column per distinct value (blank = all zeros). Auto-detects every
  non-numeric column; pass `--onehot 2,5,7` (1-based) to choose explicitly.
  A text **outcome** with exactly two values becomes a single 0/1 column
  (`no -> 0, yes -> 1`).
- `--refcat` — drop each one-hot input group's first (alphabetical) level as
  the reference category, k−1 columns per variable. **Use this whenever the
  model has an intercept — logistic regression above all:** with all k
  indicators, every group sums to a constant and the fit is perfectly
  collinear with the intercept, so the coefficients are not identifiable
  (and the optimizer can grind indefinitely). Neural networks tolerate the
  full k columns.

Tested by `tests/tools/run_tools.sh` against committed expected outputs.

## neuron2web.py

The successor to neUROn2++'s `neuron2html.pl`: deploys a trained model
(SimpleProp or Binary logistic) as **one self-contained HTML calculator** —
form, scaling factors, weights, and forward pass all inline JavaScript.
Open it from disk, email it, or upload it to any static web host; it runs
entirely in the browser.

```sh
python3 tools/neuron2web.py --network net.txt --scales scales.txt \
    --spec spec.txt -o calculator.html [--serve] [--eval "row"]
```

- Inputs: the engine's saved network + saved scaling factors + a label
  spec in the legacy tag format (`N`/`C`/`B`/`K`/`O`/`R`, with `*` marking
  a `--refcat` reference level). Full reference: `docs/deploy.md`.
- `--serve` previews on 127.0.0.1 with an OS-assigned free port.
- `--eval` prints the probability for one natural-unit input row — the
  fixture test uses it to hold the tool's forward pass equal to the
  engine's guesses on the committed XOR reference network.
