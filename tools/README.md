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
  `--inputs` writes the variable-structure string for stepwise regression.
- Improvement over the Perl: input is parsed with Python's `csv` module, so
  quoted Excel fields (embedded commas) are handled correctly.

Tested by `tests/tools/run_tools.sh` against committed expected outputs.
