# Low birth weight dataset (Hosmer & Lemeshow)

The classic teaching dataset from Hosmer & Lemeshow's *Applied Logistic
Regression*: predict whether a newborn will have **low birth weight (< 2500 g,
outcome 1)** from characteristics of the mother. The study data were collected
at Baystate Medical Center, Springfield, Massachusetts, in 1986; virtually
every logistic-regression textbook example descends from it. This copy comes
from the original neUROn2++ distribution.

## Files

| File | Rows | What |
|---|---|---|
| `lowbwt2-2train.txt` | 189 | Training set (59 low birth weight, 130 normal) |
| `lowbwt2-2betas.txt` | — | A saved neuron **Binary logistic** model: the fitted y-intercept and 5 beta weights |

## Format

Already a **neuron-ready training set**: whitespace-delimited, inputs scaled to
[−0.9, 0.9], outcome in the last column. 5 input columns + 1 output = 6 columns.
Load it with 5 input nodes and 1 output.

The five inputs correspond to Hosmer & Lemeshow's textbook model: mother's
**age**, mother's **weight** at last menstrual period (LWT), **race** as two
indicator columns (3 levels, reference-coded), and number of first-trimester
physician **visits** (FTV). The engine reproduces this model's published
log likelihood of **−111.286** exactly (see below), which is how the input
identities were confirmed — the original column key did not survive.

## Why it's here: a reference check for logistic regression

`lowbwt2-2betas.txt` is a **committed known-good fit**. Load it and the engine
verifies itself:

```
$ ./build/neuron
Menu choice: 1            # specify dataset: 5 inputs, 1 output
  6 -> lowbwt2-2train.txt
  14
Menu choice: 2            # specify model
  4 -> lowbwt2-2betas.txt # load the saved logistic model
Menu choice: 3            # use model: entry does a 1-iteration, eta-0 pass
```

The eta-0 pass evaluates the loaded coefficients without changing them, and
prints (verified output):

- **Max abs gradient = 3.4e-07** — below the engine's 1e-6 convergence
  criterion, i.e. the committed betas are the converged maximum-likelihood
  estimate for this training set;
- **Log likelihood = −111.2865** — matching the value published by Hosmer &
  Lemeshow for this model;
- the full report: classification table, ROC areas with 95% CIs, Wald tests
  on every coefficient, and the condition number.

Any change to the logistic engine that alters these numbers is a regression.

## Citation

> Hosmer, D.W. and Lemeshow, S. *Applied Logistic Regression*, 2nd ed.
> New York: Wiley, 2000. (Low birth weight data: Section 1.6.2 and throughout.)
