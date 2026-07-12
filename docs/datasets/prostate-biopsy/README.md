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
| `psa_spec.txt` | — | Label spec for `tools/neuron2web.py` — deploys a trained model as a web calculator |
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

## The 11 input columns

The eleven columns group into seven clinical variables (`psa_defs.txt`:
`0;1-4;5;6,7;8;9;10`). Their identities were recovered from the PSAnet iPhone
app's input vector, the authoritative record:

| Column(s) | Variable | Encoding |
|---|---|---|
| 0 | Age | years |
| 1–4 | Ethnicity | one-hot: Caucasian, African American, Hispanic, Asian/Other |
| 5 | Family history of prostate cancer | 1 = yes, 0 = no |
| 6, 7 | IPSS | missing-indicator pair: `1, value` known / `0, 0` unknown |
| 8 | Total PSA | ng/mL |
| 9 | Complexed PSA | ng/mL |
| 10 | Digital rectal exam | 1 = abnormal, 0 = normal |

## Deploying a trained model

`psa_spec.txt` is the ready-made label spec for `tools/neuron2web.py`. After
training a model and saving its network + scaling factors (the GUI's
"Session files", or the CLI save prompts):

```
python3 tools/neuron2web.py --network network.txt --scales scales.txt \
    --spec docs/datasets/prostate-biopsy/psa_spec.txt -o psa_calculator.html
```

The spec uses full 4-column one-hot for ethnicity (no reference level),
matching a SimpleProp network like `Run9`. If you train **logistic
regression** instead, groom with `mkdataset.py --refcat` and star one
ethnicity level in the spec (see `docs/deploy.md`).
