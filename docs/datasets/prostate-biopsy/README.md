# Prostate biopsy dataset (BP40)

A binary classification dataset from the neUROn2++ era: predict whether a prostate
biopsy will find **cancer present (1)** or **cancer absent (0)** from clinical inputs.
This is the kind of medical prediction problem neUROn was built for — the manual's ROC
specification example is literally "Odds of cancer on prostate biopsy."

> **On the name:** these files have been called `BP40` since the original distribution.
> "BP" most plausibly abbreviates *biopsy prediction*; the exact expansion isn't recorded
> in the surviving documentation, so treat that as informed guess, not gospel.

## Files

| File | Rows | What |
|---|---|---|
| `BP40train.txt` | 218 | Training set (75 cancer, 143 no-cancer) |
| `BP40test.txt` | 144 | Held-out test set (50 cancer, 94 no-cancer) |
| `psa_defs.txt` | — | Input-structure specification: `0;1-4;5;6,7;8;9;10` |
| `Run9` | — | A saved SimpleProp network (11 inputs, 3 hidden nodes) trained on this data |

## Format

Already a **neuron-ready training set**: whitespace-delimited, all inputs pre-scaled to
roughly [−0.9, 0.9], with the **outcome in the last column**. 11 input columns + 1 output
= 12 columns. No grooming needed — load it directly with menu option 6 ("Load training
set"), specifying 11 input nodes and 1 output.

The `psa_defs.txt` string `0;1-4;5;6,7;8;9;10` is the **input-variable structure** for
stepwise regression (the same format `tools/mkdataset.py --inputs` produces): each
semicolon-separated group is one conceptual variable. Groups spanning several columns
(e.g. `1-4`) are one categorical variable one-hot-encoded across those columns; a pair
like `6,7` is a missing-value indicator pair (`0,0` absent / `1,value` present). So the
11 raw input columns represent 7 underlying clinical variables.

## Quick start

```
$ ./build/neuron
Menu choice: 1          # specify dataset
  1 -> 11               # 11 input nodes
  6 -> BP40train.txt    # load training set
  9 -> BP40test.txt     # load test set (optional)
  14                    # back to main menu
Menu choice: 2          # specify model: SimpleProp, 3 hidden nodes, X-entropy
Menu choice: 3          # train
```

With enough exemplars this is where the statistical ROC report (binormal Az with a 95%
confidence interval, plus the goodness-of-fit statistics) actually fires — see
`docs/roc_theory.md`. For reproducible training, add `--seed N`.
