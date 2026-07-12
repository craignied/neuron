# Bank marketing dataset

A public UCI dataset: predict whether a bank client will **subscribe to a term deposit
(`yes`/`no`)** following a direct-marketing phone campaign. Included here as a realistic
**"messy real-world data"** example — unlike `prostate-biopsy`, this data needs grooming
before neuron can use it, which is exactly what the `tools/` utilities are for.

## Files

| File | Rows | What |
|---|---|---|
| `bank.csv` | 4,521 | 10% sample of the full dataset (521 `yes`, 4,000 `no`) |
| `bank-names.txt` | — | Original UCI variable descriptions and citation |

## Source & citation

From the UCI Machine Learning Repository:
<https://archive.ics.uci.edu/dataset/222/bank+marketing>

> [Moro et al., 2011] S. Moro, R. Laureano and P. Cortez. *Using Data Mining for Bank
> Direct Marketing: An Application of the CRISP-DM Methodology.* Proceedings of the
> European Simulation and Modelling Conference — ESM'2011, pp. 117-121, Guimarães,
> Portugal, October 2011. EUROSIS.

See `bank-names.txt` for the full attribute list and citation request.

## Why it isn't neuron-ready (and that's the point)

`bank.csv` is raw the way real exports are:

- **Semicolon-delimited**, not comma-delimited.
- **Quoted fields** (`"management"`, `"married"`).
- **Categorical text** in most columns — job, marital status, education, month, etc.
- The outcome is the word `"yes"`/`"no"`, not `1`/`0`.

neuron's engine consumes numeric matrices, so this data must first be encoded: convert
the delimiter, one-hot-encode the categorical variables into indicator columns, and map
the outcome to 1/0. `tools/mkdataset.py` handles delimiter/blank/missing-value mechanics
and emits the `--inputs` variable structure, but **categorical text encoding is not yet
automated** — that's a natural next tool (or `mkdataset.py` extension). Until then this
dataset is a worked example of the grooming problem, and a motivating target for the
tooling.

This dataset is redistributed for research/education under the terms described at the
UCI link above; please cite Moro et al. (2011) if you use it.
