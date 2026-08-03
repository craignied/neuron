# Sample datasets

Example datasets for learning and testing neuron 3.0.

| Directory | Task | Format | Notes |
|---|---|---|---|
| `prostate-biopsy/` | Predict cancer on prostate biopsy (binary) | neuron-ready (scaled) | Real clinical modeling data from the neUROn2++ era; loads directly |
| `low-birth-weight/` | Predict low birth weight (binary) | neuron-ready (scaled) | The Hosmer & Lemeshow classic, with committed reference betas that double as a logistic-regression engine check |
| `civic-choice/` | Synthetic civic-participation classification | raw `.csv` plus generator | Maintained illustrated GUI walkthrough; synthetic so every modeling choice is inspectable |
| `bank-marketing/` | Predict term-deposit subscription (binary) | raw `.csv` (needs grooming) | Public UCI dataset retained as a larger raw-data example; it is not the walkthrough |

See each directory's README for columns, provenance, and how to use it.

ROC/statistics background is in `docs/roc_theory.md`.
