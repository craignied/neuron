# Sample datasets

Example datasets for learning and testing neuron 3.0.

| Directory | Task | Format | Notes |
|---|---|---|---|
| `prostate-biopsy/` | Predict cancer on prostate biopsy (binary) | neuron-ready (scaled) | Real clinical modeling data from the neUROn2++ era; loads directly |
| `bank-marketing/` | Predict term-deposit subscription (binary) | raw `.csv` (needs grooming) | Public UCI dataset; a realistic "messy data" example for `tools/mkdataset.py` |

See each directory's README for columns, provenance, and how to use it.

The tiny `xor.csv` used by the tutorial lives in `docs/`, and the ROC/statistics
background is in `docs/roc_theory.md`.
