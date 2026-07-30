# Civic Choice — synthetic demonstration data

This directory contains a **fictional, synthetic** binary-classification dataset
designed to exercise neuron's public GUI workflow. It does not describe real
voters, political parties, demographic behavior, or voting patterns, and it
must not be used to draw conclusions about them.

`civic_choice.csv` is the canonical 6,000-row walkthrough dataset. It is
committed so every reader works from identical observations and can reproduce
the documented results. `generate.py` is the transparent recipe that created
it; regenerating the CSV is optional, not a walkthrough prerequisite.

The generator creates mixed numeric and categorical inputs for a fictional
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

The committed file was generated with:

```bash
python3 generate.py --seed 20260724 --rows 6000 -o civic_choice.csv
```

The generator uses only Python's standard library. Its SHA-256 digest is:

```text
412f58f9a5cbdbcfb093731ad888bb76ee41c00b754e7051ea25bc392ceaf2ca
```

The canonical dataset passed the measured logistic, neural-network, OBD,
cross-validation, and locked-test acceptance criteria recorded in the
walkthrough working ledger.

For the complete illustrated GUI workflow, see
[Civic Choice: a complete GUI model-comparison walkthrough](WALKTHROUGH.md).
