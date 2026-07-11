Read and follow the instructions in /Users/craign/code/locker/CLAUDE.md before proceeding.

# neuron-3.0

Reanimation of **neUROn2++** — Craig's neural computational modeling environment — as a
modern project. This directory is the new home; the legacy code is read-only reference.

## Lineage

- **neUROn** (1992) — original, begun by Craig Niederberger
- **neUROn2++** (1996 → C++, 2000 → full OO redesign, 2007 → GNU autotools distribution)
- Last release: **2.6.4** (Sep 2016, macOS/Homebrew build instructions). configure.ac says 2.64.
- Substantial contributors over the years: Vinod Kutty, Yuan Qin, Joe Jovero, Young Hong,
  Kai Liu, Richard Schoor, Gaurav Bansal, Ashay Kparker, Hui Liu, Tony Makhlouf.
- **neuron-3.0** (2026) — this reanimation.

## Legacy codebase (`../distro/`)

Interactive menu-driven CLI (`neuron`): specify dataset → specify model → use model →
discriminant function analysis. C++ with GNU autotools, depends on **GSL** (tested against
GSL 1.9–1.10; modern GSL is 2.x — untested).

### Source map (`../distro/src/`)

| Files | What |
|---|---|
| `neuron.cpp` | Main driver: menus, program flow |
| `model.{h,cpp}`, `network.{h,cpp}`, `iterative.{h,cpp}` | Abstract model / network / iterative-training base classes |
| `simpleprop`, `bareprop`, `backprop` | Feed-forward neural nets trained by backpropagation (three implementations of increasing sophistication) |
| `logistic.{h,cpp}` | Binary logistic regression — weight decay, gradient methods, conjugate gradient descent, Shanno's algorithm, Wald tests, condition number |
| `ldfa`, `qdfa`, `dfa` | Linear & quadratic discriminant function analysis |
| `regressnet.{h,cpp}` | Stepwise regression over network inputs |
| `dataset.{h,cpp}`, `twoset.{h,cpp}` | Dataset loading, train/test randomization, ROC reporting, classification tables |
| `stats.{h,cpp}` | Statistics: Kolmogorov–Smirnov, Pearson chi-square, Hosmer–Lemeshow, error functions |
| `matrix.{h,cpp}`, `vector_ops.{h,cpp}` | Templated matrix/vector math (matrix.h is 67 KB of templates) |
| `utility.{h,cpp}`, `function_defs.h` | Console I/O helpers, activation function definitions |

### Other legacy assets

- `../distro/doc/manifest.pdf` — the full manual (manifest.tex source). `spin.html` is a tutorial.
- `../distro/scripts/` — model exporters: `neuron2html.pl` (model → standalone HTML calculator),
  `neuron2palm.pl`, `neuron2iphone.rb`, plus `mkdataset.pl`. Documented in manifest.pdf ch. 11.
- `../distro/data/` — sample datasets: PSA (`psa_defs.txt`, `ordata`), low birthweight
  (`lowbwt2-2*`), XOR, BP40 train/test.
- `../distro/neuron-2.6*.tar.gz` — release tarballs 2.6 through 2.64.

### Legacy build status on this machine (checked 2026-07-11)

- `../distro/src/neuron` binary is **x86_64** (pre-Apple-Silicon) — would need Rosetta.
- **GSL is not installed** (`brew install gsl` would be needed to rebuild).
- Untested whether 2.64 compiles under modern clang++ / GSL 2.x.

## neuron-3.0 direction — OPEN DECISIONS

Not yet decided with Craig:

1. **Language/stack** — modernize the C++ (CMake, C++20, current GSL or Eigen/Accelerate)
   vs. rewrite in Python (numpy/scipy, fits the global venv workflow) vs. hybrid.
2. **Interface** — keep the interactive terminal menu, move to a CLI with flags/config files,
   or a workbench-mounted web UI (see locker CLAUDE.md for the blueprint convention).
3. **Scope** — full port (nets + logistic + DFA + stats + exporters) or start with a core
   (datasets + backprop + logistic) and grow.
4. **Model exporters** — HTML/Palm/iPhone exporters are of their era; decide what a 2026
   equivalent looks like (if any).

## Status

- **2026-07-11** — Project reanimated: explored `../distro`, wrote this CLAUDE.md and
  README.md, initialized git. No code yet. Next: decide the open questions above with Craig,
  then verify the legacy build (install GSL, try compiling 2.64) as a baseline reference.
