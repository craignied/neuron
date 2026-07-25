# Civic Choice — synthetic demonstration data

This directory contains a **fictional, synthetic** binary-classification dataset
designed to exercise neuron's public GUI workflow. It does not describe real
voters, political parties, demographic behavior, or voting patterns, and it
must not be used to draw conclusions about them.

`generate.py` creates mixed numeric and categorical inputs for a fictional
Party A versus Party B choice. Its outcome mechanism combines two kinds of
nonlinearity that an ordinary main-effects logistic regression cannot express:

- non-monotone effects within a variable (young and old ages versus middle
  ages; middle incomes versus both income tails); and
- an interaction whose direction reverses (a soft XOR of marital status and
  home ownership).

The construction borrows the geometry of standard synthetic benchmarks without
copying their abstract coordinates. XOR places the same class in opposing
quadrants, while `make_circles` and `make_moons` create curved or disconnected
class regions. Here, broad age and income bands make those ideas readable in a
walkthrough. See scikit-learn's official
[nonlinear XOR example](https://scikit-learn.org/stable/auto_examples/svm/plot_svm_nonlinear.html)
and [generated-dataset reference](https://scikit-learn.org/stable/datasets/sample_generators.html).

Three inputs are deliberately null:

- ethnicity;
- preferred ice cream; and
- car ownership.

Ethnicity is generated independently and never enters the outcome equation. It
is present to demonstrate that the availability of a protected attribute does
not justify assigning it predictive or causal meaning.

Generate the canonical candidate dataset with:

```bash
python3 generate.py --seed 20260724 --rows 6000 -o civic_choice.csv
```

The generator uses only Python's standard library. The canonical dataset is not
accepted as a walkthrough fixture until measured logistic, neural-network, OBD,
stepwise, cross-validation, and locked-test behavior satisfies the acceptance
criteria recorded in the walkthrough working ledger.
