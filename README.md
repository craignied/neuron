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

## Status

Early reanimation. The legacy 2.6.4 codebase (C++, GSL, autotools) is preserved alongside
this repository as reference; the architecture and stack for 3.0 are being decided.
See `CLAUDE.md` for current project state and open decisions.

## Legacy build (reference)

neUROn2++ 2.6.4 builds with the standard GNU procedure and requires the
[GNU Scientific Library](https://www.gnu.org/software/gsl/):

```sh
brew install gsl        # macOS
./configure && make && sudo make install
```

Documentation for the legacy system: `doc/manifest.pdf` (manual) and `doc/spin.html`
(tutorial) in the neUROn2++ distribution.
