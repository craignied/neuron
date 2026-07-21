# neuron 3.0

[![CI](https://github.com/craignied/neuron/actions/workflows/ci.yml/badge.svg)](https://github.com/craignied/neuron/actions/workflows/ci.yml)

A neural computational modeling environment — the third major revision of **neUROn**,
begun in 1992 by Craig Niederberger.

## History

| Era | Name | Notes |
|---|---|---|
| 1992 | neUROn | Original neural computational environment |
| 1996–2016 | neUROn2++ | Migrated to C++ (1996), fully object-oriented redesign (2000), GNU autotools distribution (2007). Final release 2.6.4 (2016). |
| 2026– | neuron 3.0 | This project — a modern reanimation |

neUROn2++ was a full-featured, menu-driven statistical modeling workbench built around
a common dataset/model architecture:

- **Feed-forward neural networks** trained by backpropagation (SimpleProp, BareProp, BackProp)
- **Binary logistic regression** with weight decay, conjugate gradient descent, and
  Shanno's algorithm; Wald tests and condition-number diagnostics
- **Linear and quadratic discriminant function analysis**
- **Stepwise regression** over model inputs
- **Goodness-of-fit and validation statistics**: ROC analysis, classification tables,
  Kolmogorov–Smirnov, Pearson chi-square, Hosmer–Lemeshow
- **Model exporters** that turned trained models into standalone calculators for the
  web (HTML/JavaScript), Palm OS, and iPhone

It was used extensively in medical research — notably neural-network prediction models
in urology and reproductive medicine.

## Design

- **Hybrid stack** — a C++ engine for speed, Python for data preparation and tooling
- **Simple, portable interface** — plain CLI, no heavy frameworks, easy to port
- **Full engine scope** — the neural networks and all of the statistical machinery
  (the novel and useful core) are carried forward; the legacy model exporters are not

## Building

Requires CMake and the [GNU Scientific Library](https://www.gnu.org/software/gsl/):

```sh
brew install gsl cmake    # macOS
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/neuron
```

```
Usage: neuron [--seed N] [--gui [--no-browser]] [--version]
```

`--gui` starts the same engine behind a **local web page** instead of the
menus: the binary embeds a small HTTP server (cpp-httplib, vendored in
`third_party/`), binds 127.0.0.1 on an **OS-assigned free port** — it can
never collide with anything else you run — prints the URL, and opens your
browser. Pick a data file, pick a model, train — watching a realtime
error-vs-iteration chart, with a Stop button, automatic algorithm selection
(`auto` probes all three optimizers and adopts the best), plateau auto-stop,
and **OBD hidden-layer sizing** (grow-then-prune with validation early
stopping picks the hidden-unit count for you) — then read the full statistics
beside a live ROC plot — classification and confusion counts, trapezoidal
and binormal areas with their intervals, goodness of fit, and (for logistic
models) the coefficient table with Wald p values and the condition number —
then save the **session files**: the network and
scaling factors (everything `tools/neuron2web.py` needs to deploy the
model) plus the training/test sets, guesses, and report (everything a
write-up needs). Each file is written into the directory you started
`neuron` from and downloaded by the browser. Loopback only; the CLI
remains fully functional.

The binary is self-contained (the GUI page is embedded), so a good habit
is one directory per experiment: `cd` there — or copy `build/neuron`
there — and everything the session produces stays together.

`--seed N` makes runs reproducible: the same seed with the same inputs produces
bit-identical training (weight initialization and train/test splits both draw from
the seeded generator). The generator is `std::mt19937`, whose output stream the
C++ standard specifies exactly — so seeded runs reproduce across platforms and
compilers, not just on one machine. Without `--seed`, the generator is seeded from
the clock, as neUROn2++ always was.

The engine is C++17 with no dependencies beyond GSL — it builds anywhere those exist.

> **If you copied or moved this directory:** delete any `build/` that came along
> before configuring (`rm -rf build`). CMake caches the absolute source path and
> refuses to reuse a cache created at a different location.

## Training algorithms

Every gradient-trained model — the feed-forward neural nets (SimpleProp, BareProp,
BackProp) and binary logistic regression — can be trained by any of **three
optimization algorithms**, offered at train time as a menu (and as `algorithm` 1–3
in the GUI). All three minimize the same error over the same weights; they differ
only in how each iteration chooses its search direction. (The DFA and stepwise
models are not trained by this trio.)

1. **Canonical backpropagation (gradient descent)** — textbook backprop: step
   directly downhill along the negative gradient, scaled by the step size. Cheapest
   per iteration and the most robust in practice; the workhorse.
2. **Conjugate gradient descent (CGD)** — each direction is the negative gradient
   plus a correction from the previous direction, so successive steps stay conjugate
   (Golden pp. 221–222). Fewer *iterations* on well-behaved problems, but more work
   per iteration.
3. **Shanno's algorithm** — a memoryless quasi-Newton method that approximates
   curvature from the last gradient/direction pair, without storing a full Hessian
   (Golden pp. 217–218). Strongest convergence per iteration when it works, most
   expensive per iteration.

**Batch/epoch training is required for CGD and Shanno.** Both assume a true (batch)
gradient accumulated over the whole training set; on an online/per-exemplar update
their conjugacy and curvature estimates are meaningless. The engine warns you if you
select either without epoch training on. Canonical gradient descent is fine either
way.

**Fewer iterations is not less wall-clock, and can be worse.** The per-iteration cost
of CGD and Shanno (line searches, automatic step-size selection) is high, and their
step control can go unstable on large datasets. On the bank-marketing walkthrough,
canonical GD reached the logistic MLE in about 12 seconds, while **uncapped CGD ran
80 minutes and ended worse than useless** (test ROC 0.57, step-size instability,
"Numerical out of bounds"). The practical recipe: **cap the iteration count** and
prefer canonical gradient descent unless you have a specific reason not to. See
`docs/datasets/bank-marketing/WALKTHROUGH.md` for the full comparison.

## Testing

Every push is built and tested on macOS, Linux, and Windows by the GitHub Actions
matrix in `.github/workflows/ci.yml`. The core check is the **golden transcript
test** (`tests/golden/run_golden.sh`): fully seeded sessions must reproduce their
committed transcripts byte for byte (only the elapsed-time line is excluded). Because
the RNG is `std::mt19937`, the same transcripts must match on all three platforms;
any numerical drift anywhere in the engine fails the build. After an intentional
output change, regenerate with `./tests/golden/run_golden.sh --bless`.

The suite has three cases: `xor_seed42` (SimpleProp training and statistics),
`regress_seed42` (plus stepwise regression), and `binormal_seed42`, which runs the
low-birth-weight data through a seeded logistic fit to cover the **statistical ROC
report** — the binormal area, its bootstrap interval, and the goodness-of-fit tests. The
first two are too small to reach that path at all, so before the third existed the entire
ROC report was invisible to the goldens.

**The suite also asserts its own coverage.** A green run means nothing if the transcripts
never execute what you changed, so `run_golden.sh` carries a list of paths that some
transcript must reach — the binormal report, the bootstrap interval, the zROC fit, K–S,
Hosmer–Lemeshow, stepwise. Delete or retarget a case and the suite fails and names what
it stopped covering, rather than going quietly green. That check exists because the ROC
confidence interval was once replaced entirely with every test passing and none of them
running the code.

A second, local-only check (`tests/oracle/`) cross-verifies the engine against the
original neUROn2++ binary; see its README.

## ROC area confidence intervals

Every ROC area the engine reports carries a 95% confidence interval. The binormal
area A_z uses a **stratified bootstrap percentile interval**; the empirical
(trapezoidal) area keeps the closed-form **Hanley–McNeil** estimator as an independent
cross-check. **`docs/roc_theory.md` is the authority** — it carries the full derivation,
the Methods-section language, and a page-level citation table. Read it before quoting an
interval in print.

The signal-detection basis is Wickens, *Elementary Signal Detection Theory* (Oxford,
2002). A_z = Φ(a/√(1+b²)) is his Eq. 4.7 (p. 68).

**Why the interval is bootstrapped rather than analytic.** neuron sweeps thresholds
across one continuous score on one sample, so its operating points are formed by
cumulating the same observations and are **not statistically independent** (Wickens
2002, pp. 87–88) — the rating case, not the independent-bias-condition case. Wickens
says that dependence must be accommodated in the fitting algorithm and that it
*increases* sampling variability (§5.3, p. 88); he offers no analytic standard error
for a least-squares z-ROC line, routing multi-point data to maximum likelihood, which
supplies the SEs (§3.6, p. 57). Resampling cases reproduces the dependence rather than
modelling it, and the percentile interval is asymmetric by construction — which A_z
requires, since its standard error varies with the parameter (§11.4, pp. 206–207).

The earlier delta-method interval assumed independent points and was ~5× too narrow;
it has been retired. Validation: on the low-birth-weight data the bootstrap SE agrees
with Hanley–McNeil to within ~3% (0.0524 vs 0.0522 at n = 142; 0.0844 vs 0.0870 at
n = 47) and scales with n as it must — two estimators with different assumptions
agreeing, where the delta method agreed with neither. Against simulation the mean
reported bootstrap SE matches the empirical SD of A_z to within 1%.

The interval line reports the effective number of resamples and how many failed —
quote both. Failures track ties rather than occurring at random, so a nonzero count
means a slightly narrow interval; on the low-birth-weight data a current build uses
all 2000. (Until 2026-07-15 it discarded 27% of them there, because the variance of
a bin holding a flat run of the ROC was computed by a formula that returned NaN
instead of zero. Fixing that widened the interval to its honest width.)

**A_z is the primary measure, not the trapezoidal area.** The trapezoidal area is
negatively biased — it connects operating points by straight lines where real
isosensitivity contours are bowed — and the bias grows when points are few or bunched
(Wickens 2002, pp. 70–71). Unless there is specific reason to doubt the Gaussian model,
A_z is preferable (p. 72). The trapezoidal area is retained because its variance
estimator rests on different assumptions, making agreement between the two meaningful.

**Validated against the literature.** Wickens' own worked example (Table 5.1, p. 84 —
rating data he analyses by hand through chapter 5, publishing A_z = 0.784 on p. 90) is a
ctest: the engine answers **0.7839**. Each operating point carries Wickens' binomial
error bar, σ²_z ≈ p(1−p)/N ÷ φ²(z) (Eq. 11.2 + 11.3, p. 202), and the z-ROC line is
fitted to the distinct operating points directly — there is no binning.

Until 2026-07-15 the z-points were grouped into fixed-count bins whose within-bin
standard deviations served as error bars, a device with no counterpart in Wickens that
measured bin width rather than sampling error. On Wickens' data that was worth 0.011 of
A_z — as much as the whole confidence interval — with the arbitrary bin count deciding
the answer. It is gone.

## Layout

- `src/` — the C++ engine (carried forward from neUROn2++, modernized incrementally
  in place), including the embedded web GUI (`gui.cpp`, `gui_page.html`)
- `third_party/` — vendored single-header dependencies (cpp-httplib, MIT)
- `tests/oracle/` — the legacy oracle harness: builds the original neUROn2++ binary
  from its release tarball and cross-checks the 3.0 engine against it
  (`verify_oracle.sh` requires numerically identical output)
- `docs/` — the neUROn2++ documentation: `manifest.pdf` is the full manual,
  `spin.html` the tutorial, `tex/` the LaTeX sources; `roc_theory.md` explains
  the signal-detection-theory (Wickens) basis of the statistical ROC area and
  how the engine implements it; `datasets/` holds sample datasets with READMEs
  (prostate-biopsy, ready to load; bank-marketing, a raw-data grooming example)
- `tools/` — Python utilities around the engine. **Standard library only** — they
  run on a bare `python3`, no pip installs or venv ever required (CI enforces
  this on all three platforms). `mkdataset.py` converts `.csv` exports into
  neuron-ready datasets (delimiters, one-hot encoding, reference-category
  coding, missing-value indicator pairs); `neuron2web.py` deploys a trained
  model as a single self-contained HTML calculator you can open from disk or
  post on any static web host (`docs/deploy.md`).
- `AGENTS.md` — the operating manual for AI assistants: verified recipes that
  take "I have a dataset" to a groomed file, a trained model, and a deployed
  calculator.

## Status

The full neUROn2++ engine builds as `neuron 3.0.0-dev` — C++17-clean, zero warnings,
verified numerically identical to the legacy binary on a saved-network forward pass
and equivalent on endpoint statistics for XOR training. Modernization proceeds from
this verified base. See `CLAUDE.md` for current project state.
