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
browser. Load a dataset, pick a model, train, and read the captured report
next to a live ROC plot. Loopback only; the CLI remains fully functional.

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

## Testing

Every push is built and tested on macOS, Linux, and Windows by the GitHub Actions
matrix in `.github/workflows/ci.yml`. The core check is the **golden transcript
test** (`tests/golden/run_golden.sh`): two fully seeded sessions — XOR training and
stepwise reverse regression — must reproduce their committed transcripts byte for
byte (only the elapsed-time line is excluded). Because the RNG is `std::mt19937`,
the same transcripts must match on all three platforms; any numerical drift anywhere
in the engine fails the build. After an intentional output change, regenerate with
`./tests/golden/run_golden.sh --bless`.

A second, local-only check (`tests/oracle/`) cross-verifies the engine against the
original neUROn2++ binary; see its README.

## ROC area confidence intervals

Every ROC area the engine reports carries a 95% confidence interval, computed by an
**analytic (parametric) method** in each case — not a bootstrap. `docs/roc_theory.md`
gives the full account; the signal-detection-theory basis of the binormal area is
Wickens, *Elementary Signal Detection Theory* (Oxford, 2002).

**Methods-section language:**

> A 95% confidence interval for the binormal area under the ROC curve (Az) was
> obtained by the delta method, propagating the standard errors of the fitted z-ROC
> intercept and slope through Az = Φ(a/√(1+b²)). For empirical (trapezoidal) areas,
> the confidence interval used the Hanley–McNeil variance estimator based on the
> equivalence of the area to the Mann–Whitney U statistic.

**Caveat before relying on the binormal interval in print:** on the binned fit path
(fitexy, errors in both coordinates) the routine doesn't expose the a–b covariance, so
that cross term is set to zero. In the calibration test the delta-method SE ran a bit
narrow (reported SE ≈ 0.56× the empirical SD) — somewhat anti-conservative — while the
Hanley–McNeil interval calibrated cleanly (ratio ≈ 1.09). The trapezoidal CI is the more
trustworthy of the two as it stands; recover the covariance term, or validate the
binormal CI further, before leaning on it in print. Both are calibration-checked by
simulation in `tests/binormal/check_az.cpp`.

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
