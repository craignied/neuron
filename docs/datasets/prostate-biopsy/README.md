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
| `psa_scales.txt` | — | Natural-units scaling factors for this model (see "Deploying," below) |
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
11 raw input columns represent 7 underlying clinical variables. Paste that string into
the GUI's **Stepwise regression** panel (or the CLI's "Specify variables") to rank the
seven variables by significance on a trained network.

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

**In the GUI** (`./build/neuron --gui`): set the dataset mode to *already a
training set*, pick `BP40train.txt` as the data file and `BP40test.txt` in the
*Test set (optional)* picker, and Load — the input count (11) is read from the
file, and the pair loads together (218 train + 144 test) so training reports a
held-out test ROC.

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

Deploying to natural units (age in years, PSA in ng/mL) needs three files:
the network, a **label spec** (`psa_spec.txt`, ready-made), and a
**scaling-factors file** that maps natural units into the network's
normalized input space. All three are committed, so the model deploys with
one command:

```
python3 tools/neuron2web.py --network Run9 --scales psa_scales.txt \
    --spec psa_spec.txt -o psa_calculator.html
```

Verify before shipping — `--eval` takes a row in natural units, in
network-column order (no outcome column):

```
$ python3 tools/neuron2web.py --network Run9 --scales psa_scales.txt \
      --spec psa_spec.txt --eval "65, 1,0,0,0, 0, 1, 8, 4.2, 1.1, 0"
0.193435
```

That 0.193435 (65 y/o, Caucasian, no family history, IPSS 8, total PSA 4.2,
complexed PSA 1.1, normal DRE) reproduces the PSAnet iPhone app's own
forward pass for the same inputs to six significant figures — `Run9` **is**
the app's network, and `psa_scales.txt` **is** its scaling.

### Why the scaling factors are a separate file here

`BP40train.txt`/`BP40test.txt` arrived **already normalized** to
[−0.9, 0.9] — the natural-units values were scaled away before the dataset
was ever saved. So you cannot recover deployment scaling by loading BP40 and
saving factors: that run only sees normalized numbers and would produce a
near-identity mapping, and the resulting calculator would demand
already-normalized inputs (useless to a clinician typing an age and a PSA).

The genuine natural→normalized mapping survived only in the PSAnet iPhone
app (`~/code/iPhone/PSAnet`), which takes real clinical values and predicts.
`psa_scales.txt` is that mapping, transcribed into the engine's
scaling-file format:

- `S`      = `0.0529412 1.8 1.8 1.8 1.8 1.8 1.8 0.0545455 0.00188547 0.0024129 1.8`
- `x_min`  = `50 0 0 0 0 0 0 0 0.33 0.21 0`
- `lbound` = `−0.9`   (`x_norm = S·(x − x_min) + lbound`)

The lesson generalizes: to deploy in natural units you need the scaling from
the run that first normalized the data. When you groom a raw CSV yourself,
save the scaling factors at that point (the CLI's menu 8, or the GUI's
Session files) — for a dataset that was normalized upstream, hunt down
wherever that normalization was recorded.

The spec uses full 4-column one-hot for ethnicity (no reference level),
matching a SimpleProp network like `Run9`. If you train **logistic
regression** instead, groom with `mkdataset.py --refcat` and star one
ethnicity level in the spec (see `docs/deploy.md`).
