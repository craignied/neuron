# neuron 3.0

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
cmake --build build
./build/neuron
```

The engine is C++17 with no dependencies beyond GSL — it builds anywhere those exist.

## Layout

- `src/` — the C++ engine (carried forward from neUROn2++, modernized incrementally
  in place)
- `tests/oracle/` — the legacy oracle harness: builds the original neUROn2++ binary
  from its release tarball and cross-checks the 3.0 engine against it
  (`verify_oracle.sh` requires numerically identical output)
- `docs/` — the neUROn2++ documentation: `manifest.pdf` is the full manual,
  `spin.html` the tutorial, `tex/` the LaTeX sources
- `tools/` — Python utilities for data preparation (planned)

## Status

The full neUROn2++ engine builds as `neuron 3.0.0-dev` — C++17-clean, zero warnings,
verified numerically identical to the legacy binary on a saved-network forward pass
and equivalent on endpoint statistics for XOR training. Modernization proceeds from
this verified base. See `CLAUDE.md` for current project state.
