# Walkthrough: from messy .csv to a trained model with confidence intervals

This is the complete path from `bank.csv` — a raw, semicolon-delimited,
categorical-text export — to trained neuron models with ROC areas and 95%
confidence intervals. Every command and every output excerpt below was
produced by actually running it (macOS, `--seed 42`); your numbers will match
exactly on the same machine, and on any platform for the same seed (the RNG
is `std::mt19937`, whose stream the C++ standard specifies).

Commands are run from the repository root, with the engine built at
`build/neuron` (see the top-level README).

## 1. The raw material

`bank.csv` is 4,521 rows of a Portuguese bank's telemarketing records
(see README.md for provenance): did the client subscribe to a term deposit?

```
$ head -3 docs/datasets/bank-marketing/bank.csv
"age";"job";"marital";"education";"default";"balance";"housing";"loan";"contact";"day";"month";"duration";"campaign";"pdays";"previous";"poutcome";"y"
30;"unemployed";"married";"primary";"no";1787;"no";"no";"cellular";19;"oct";79;1;-1;0;"unknown";"no"
33;"services";"married";"secondary";"no";4789;"yes";"yes";"cellular";11;"may";220;1;339;4;"failure";"yes"
```

Three problems for a numeric engine: semicolon delimiters, categorical text
in most columns (`"unemployed"`, `"married"`…), and an outcome that is the
word `"yes"` or `"no"`.

## 2. One grooming command

```
$ python3 tools/mkdataset.py --delimiter ';' --onehot --refcat \
      --key bank_key.txt --inputs bank_inputs.txt -o bank_data.txt \
      docs/datasets/bank-marketing/bank.csv
```

- `--delimiter ';'` — parse semicolons (quoted fields are handled by
  Python's csv module).
- `--onehot` — auto-detect every categorical text column and encode it as a
  group of 0/1 indicator columns; the two-valued text outcome becomes a
  single 0/1 column.
- `--refcat` — drop each group's first level as the **reference category**
  (k−1 columns per variable). This matters: with all k indicators, every
  group sums to 1 in every row, which is perfectly collinear with the
  model's intercept — logistic regression coefficients stop being
  identifiable, and the optimizer can grind for a very long time chasing a
  fit that isn't unique. (Neural networks tolerate the full k columns; the
  k−1 coding is correct for both, so this walkthrough uses one file for
  both models.)

The tool reports what it found and did (excerpts):

```
17 columns found on first line
...
One-hot encoding:
Column 2 (job): 12 levels, reference = admin.: blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown
Column 3 (marital): 3 levels, reference = divorced: married, single
Column 4 (education): 4 levels, reference = primary: secondary, tertiary, unknown
...
Outcome column 17 (y): no -> 0, yes -> 1
...
Number of rows recorded in bank_data.txt is 4521
Number of columns was 43, which would be 42 input nodes in a 1-output dataset
```

The first raw row above becomes pure numbers (42 inputs + outcome):

```
$ head -1 bank_data.txt
30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1787, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 79, 1, -1, 0, 0, 0, 1, 0
```

Two companion files keep the encoding human-readable. `bank_key.txt` maps
every output column back to a variable (excerpt):

```
Column 1: age
Columns 2-12: job (one-hot, reference = admin.: blue-collar, entrepreneur, ...)
Columns 13-14: marital (one-hot, reference = divorced: married, single)
...
Column 43: y (no = 0, yes = 1)
```

`bank_inputs.txt` is the variable-structure string for the engine's stepwise
regression — each one-hot group counts as ONE variable:

```
0; 1-11; 12, 13; 14-16; 17; 18; 19; 20; 21, 22; 23; 24-34; 35; 36; 37; 38; 39-41
```

## 3. Load and split in neuron

Start the engine with a seed so the run is reproducible:

```
$ ./build/neuron --seed 42
```

Then, from the main menu (`1` = Specify dataset):

| Prompt | Answer | Why |
|---|---|---|
| Menu choice | `1` | Specify dataset |
| ( 1 ) Number of input nodes | `42` | From the grooming report |
| ( 2 ) Number of output nodes | `1` | Subscribe yes/no |
| ( 3 ) Load raw dataset | `bank_data.txt` | The groomed file |
| ( 5 ) Randomize … → ( 3 ) decimal fraction | `0.25` | Hold out 25% as a test set |
| Save prompts | `no`, `no` | (Say yes to keep the split files) |
| ( 14 ) | | Back to the main menu |

Loading it "raw" matters: the engine scales the columns itself (inputs to
[−0.9, 0.9]) when it converts or randomizes a raw dataset, so the groomed
file stays in natural units. The split is stratified and reported:

```
The training set has 3391 exemplars.
   The number of 1s in the training set is 391.
   The frequency of 1s in the training set is 0.115305.
The test set has 1130 exemplars.
   The number of 1s in the test set is 130.
   The frequency of 1s in the test set is 0.115044.
```

Both halves keep the population's 11.5% subscription rate.

## 4. Model 1: binary logistic regression

Main menu `2` (Specify model) → `5` (Binary logistic regression) — it
returns to the main menu immediately. Then `3` (Use model):

| Prompt | Answer | Why |
|---|---|---|
| Menu choice | `1` | Randomize weights |
| Menu choice | `3` → `1` | Stopping conditions → max iterations |
| Maximum number of iterations | `20000` | See the note below |
| Menu choice | `6`, then `7` | Back, then Train model |
| Choose training algorithm | `1` | Canonical backpropagation |

**Why cap the iterations?** The engine's default limit is 1,000,000, with a
stop when the maximum absolute gradient falls below 1e-6 — sensible for tiny
datasets, but on 3,391 rows you should set the budget yourself. And prefer
canonical gradient descent here: it converges in seconds, while Shanno's and
conjugate-gradient line searches can grind for a very long time on this data.
The run (training takes 12 seconds on an M2 Ultra; the statistical report
that follows — including two 2,000-resample ROC bootstraps — adds about 2
more):

```
Training algorithm is canonical backpropagation
Training is off-line (batch/epoch is on)
Automatic stepsize selection is on
...
    Iteration: X-entropy error:  Max abs grad:  CA Train %   CA Test %
             1     8.864913e-01   1.146914e-01        88.5        88.5
            10     3.394604e-01   2.301262e-02        88.3        88.4
           100     2.852821e-01   1.524149e-02        89.0        89.3
          1000     2.371346e-01   1.898204e-03        90.5        90.1
         10000     2.359370e-01   4.784566e-06        90.7        90.2
         20000     2.359369e-01   2.088162e-06        90.7        90.2

Total iterations = 20001
That took 0000:00:12
Final X-entropy error in the training set = 2.359369e-01.
Log likelihood = -8.000619e+02
```

The error is flat and the gradient is at 2e-6 — this is the converged
maximum-likelihood fit. The held-out test set report:

```
Test set:
---------
Classification accuracy: 90.2%
Sensitivity: 35.4%
Specificity: 97.3%
...
By statistical method, ROC area = 0.873922
95% CI = 0.844467 - 0.901292 (SE = 0.014801, 2000 bootstrap resamples)
Chi-squared = 258.501854
p = 1.000000e+00 (closer to 1 is better)
Operating points fitted = 920
By trapezoidal method, ROC area = 0.874854
95% CI = 0.835384 - 0.914323 (SE = 0.020138, Hanley-McNeil)
```

Both ROC methods agree (0.874) and their confidence intervals overlap almost
exactly — a healthy sign (and meaningful, because the two intervals rest on
different assumptions: the binormal CI is a stratified bootstrap, the
trapezoidal one is the closed-form Hanley–McNeil estimator). Don't over-read
the `p = 1.000000` fit line: on a continuous score it is an artifact of the
operating points being cumulated from one sample, not a perfect fit — see
`docs/roc_theory.md`. Note the sensitivity: at the default 0.5 cutoff the
model catches only 35% of subscribers, because only 11.5% of clients
subscribe; the ROC area is the threshold-free measure of how well the model
*ranks* them.

Logistic regression also prints a Wald test per coefficient. Some highlights
(the input numbers are the 0-based columns from `bank_key.txt`):

```
   Input:       Beta weight:     Std Err:    Wald p(W!=0):
-----------    -------------   ------------  -------------
         19        -0.255182      0.0891196         0.0042   housing = yes
         20        -0.343084       0.129837         0.0082   loan = yes
         22        -0.841783       0.146935        1.0e-08   contact = unknown
         35          7.50801       0.408643            0.0   duration
         40          1.39447       0.177059        3.3e-15   poutcome = success
```

(The right-hand annotations are added here from the key file.) Clients with
housing or personal loans subscribe less; an unknown contact channel is a
strong negative; a previous successful campaign is a strong positive; and
`duration` towers over everything — see §8 for why that's a warning, not a
triumph. The report ends with the condition number of the information matrix
(2.5e+04 here — high but workable; it would be infinite without `--refcat`).

## 5. Model 2: a neural network (SimpleProp)

Same dataset, second model. Main menu `2` → option `2` (hidden layers):
1 layer, 5 nodes → `8` (return). With one output, one hidden layer, and
bias nodes this builds a **SimpleProp** 42-5-1 network with X-entropy
output error. Then `3` (Use model) → `1` (Randomize weights) → `3` → `1` →
`2000` iterations → `6` → `7` (Train model) → `1` (Canonical
backpropagation). Four seconds later:

```
    Iteration: X-entropy error:  Max abs grad:  CA Train %   CA Test %
             1     5.734830e-01   2.975672e-01        88.5        88.5
           100     3.318139e-01   8.166189e-03        88.5        88.5
           500     2.452803e-01   6.410695e-03        89.9        89.6
          1000     2.309610e-01   3.076842e-02        90.9        90.0
          2000     2.123636e-01   1.772323e-02        91.1        89.4

Total iterations = 2001
That took 0000:00:04

Training set:  Classification accuracy: 91.1%
By statistical method, ROC area = 0.923369
95% CI = 0.909801 - 0.934962 (SE = 0.006530, 2000 bootstrap resamples)
By trapezoidal method, ROC area = 0.924339
95% CI = 0.905997 - 0.942682 (SE = 0.009359, Hanley-McNeil)

Test set:      Classification accuracy: 89.4%
By statistical method, ROC area = 0.871037
95% CI = 0.838943 - 0.899129 (SE = 0.015410, 2000 bootstrap resamples)
By trapezoidal method, ROC area = 0.872377
95% CI = 0.832594 - 0.912160 (SE = 0.020297, Hanley-McNeil)
```

Two things worth noticing:

- **Train vs test**: 0.923 vs 0.871. The network fits the training data
  better than logistic regression did (0.907) but generalizes the same —
  the classic mild-overfit signature, caught because we held out a test set.
  The network's test CI (0.839–0.899) and logistic's (0.844–0.901) overlap
  almost completely: on this data the extra flexibility buys nothing, which
  is itself a publishable observation.
- Builds before 2026-07-15 printed `a too large, ITMAX too small in gcf`
  here instead of a statistical ROC area — a failure of the old binned fit
  that had been part of neuron since the 2002 tutorial. Its cause is fixed;
  a current build prints the full statistical report above. If you ever see
  that message on a current build, report it as a bug rather than shrugging
  — the trapezoidal area with its Hanley–McNeil CI is still always printed
  and is the fallback number in the meantime.

**Don't skip the iteration cap.** Trained instead with conjugate gradients
and no cap, this exact configuration ran for 80 minutes: 40,000 iterations
stuck at the base rate, then real learning, then the automatic step size
destabilized and the final network was worse than useless (test ROC 0.57).
The cap is not a nicety — unattended optimizers need budgets.

## 6. Reading the ROC report

Every model prints, for the training set and the test set:

- **Classification table** — accuracy, sensitivity, specificity, predictive
  values at the 0.5 cutoff.
- **Statistical (binormal) ROC area** — the Wickens signal-detection-theory
  method: fit a line to the ROC curve in z-space, then
  Az = Φ(a/√(1+b²)) (Wickens 2002, Eq. 4.7, p. 68), fitted directly to the
  distinct operating points ("Operating points fitted" — quote that count with
  the area). Its **95% CI is a stratified bootstrap** (the resample count and
  any failures are printed on the CI line — quote those too).
- **Trapezoidal ROC area** — the empirical area, with a **95% CI by
  Hanley–McNeil**. A_z is the primary measure (Wickens holds the trapezoidal
  area negatively biased, pp. 70–72); the trapezoidal interval rests on
  different assumptions, so agreement between the two is worth reporting.
- **On what the intervals mean — and what the fit p does not — read
  `docs/roc_theory.md` before quoting anything in print.** (Builds before
  2026-07-15 printed a delta-method CI here that was mis-specified and ~5×
  too narrow, plus a second "best AUC" fit from an arbitrary binning; if your
  output looks like that, you are on an old binary — use the trapezoidal
  Hanley–McNeil interval and rebuild.)
- **Goodness-of-fit** — Kolmogorov–Smirnov, Pearson, and Hosmer–Lemeshow.
  The **Hosmer–Lemeshow line is the calibration test to quote**: Ĉ over
  g = 10 deciles of risk against χ² on 8 df, exactly as in Hosmer &
  Lemeshow, *Applied Logistic Regression*, 2nd ed. (Wiley, 2000), §5.2.2 —
  the canonical g every published reference value uses. The **Pearson line
  is a statistic, deliberately without a p-value**: with continuous inputs
  every covariate pattern is a single exemplar, and the individual-level X²
  then has no valid χ² reference — the very problem the Hosmer–Lemeshow
  grouping was invented to solve. Read it as a scale check instead: under a
  well-fitted model E[X²] ≈ n, so X² far above n suggests overdispersion or
  misfit. (Builds before 2026-07-16 printed a defective Hosmer–Lemeshow
  variant that rejected true models about half the time, and a Pearson "p"
  that shrank with sample size regardless of fit — see CLAUDE.md, legacy
  bug #9. Do not quote either from an old binary.)
- For logistic regression: **Wald tests** on every coefficient and the
  **condition number** of the information matrix (large values warn of
  collinearity — try re-grooming without `--refcat` to see it blow up).

## 7. Scripting sessions

Everything above can run unattended: put the menu answers in a text file,
one per line, and redirect. This is exactly how the transcripts in this
walkthrough were produced, and how `tests/golden/` guards the engine:

```
$ ./build/neuron --seed 42 < session.in > session.out
```

The complete `session.in` for the logistic model of §4 — every line is one
menu answer (§3's dataset steps, then the model, training, and quit):

```
1
1
42
2
1
3
bank_data.txt
5
3
0.25
no
no
14
2
5
3
1
3
1
20000
6
7
1
no
10
9
yes
```

With `--seed 42`, every number in §4 reproduces byte-for-byte (only the
elapsed-time line varies) — on any platform, thanks to mt19937.

## 8. A caveat before you take the model seriously

The `duration` input (length of the marketing call) is only known **after**
the call ends — at which point the outcome is nearly known too. The UCI
documentation itself warns that a realistic pre-call model should discard
it. Two good exercises: re-groom without that column and compare ROC areas,
or keep it and let **stepwise regression** (Use model → `9`, feeding it
`bank_inputs.txt`) tell you which variables actually carry the signal.

## 9. Deploying the model as a web calculator

To deploy, the training session must save two files the walkthrough's
earlier sessions skipped: the **scaling factors** (dataset menu option `8`,
available right after the randomize) and the **network** (answer `yes` to
the save prompt after training). With `--seed 42` the training is identical
to §4 — the extra saves consume no randomness.

Then one command turns them into a standalone page, using the committed
label spec `bank_spec.txt` (written from `bank_key.txt`; every `*` marks a
reference level from `--refcat`):

```
$ python3 tools/neuron2web.py \
      --network bank_logistic_net.txt --scales bank_scales.txt \
      --spec docs/datasets/bank-marketing/bank_spec.txt \
      -o bank_calculator.html
Deployed Binary logistic (42 inputs) to bank_calculator.html
```

`bank_calculator.html` is self-contained — form, scaling, weights, and
forward pass all inline. Open it from disk, or upload it to any static web
host and share the URL; the computation runs in the visitor's browser and
no data leaves their machine. Add `--serve` to preview it locally on an
OS-assigned port. Spot-check before publishing:

```
$ python3 tools/neuron2web.py --network bank_logistic_net.txt \
      --scales bank_scales.txt --spec .../bank_spec.txt \
      --eval "30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, ..."
0.153694
```

(That's the first client in `bank.csv` — a 79-second October call, true
outcome "no": 15.4%, correctly below threshold.) The spec format reference
is `docs/deploy.md`.
