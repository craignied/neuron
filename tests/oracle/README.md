# Legacy oracle harness

The legacy neUROn2++ 2.64 binary serves as the numerical oracle for validating the 3.0
engine. It builds clean on Apple Silicon with modern clang++ and GSL 2.8 — zero source
changes needed (verified 2026-07-11).

Only the oracle *binary* lives here (`bin/`, gitignored); its source is never extracted
into this repo. Building the oracle requires the release tarball at
`../../../distro/neuron-2.64.tar.gz` (Craig's machines only — distribution users don't
need the oracle).

## Usage

```sh
./build_oracle.sh                 # builds bin/neuron-2.64 from the tarball (temp-dir build)
./run_xor.sh                      # scripted XOR training session against the oracle
./run_xor.sh ../../build/neuron   # same session against the 3.0 engine
./verify_oracle.sh                # deterministic cross-check: oracle vs 3.0 must match exactly
```

`verify_oracle.sh` is the real acceptance test: it trains XOR with the oracle, saves the
network, loads the identical weights into both binaries (which triggers one eta-0
forward pass printing error + statistics), and diffs the sessions. Passing means the
3.0 engine is numerically identical to neUROn2++ on that path. First passed 2026-07-11.

## Files

- `build_oracle.sh` — builds `bin/neuron-2.64` (extracts the tarball to a temp dir,
  keeps only the binary)
- `xor_discrete.set` — the 4-exemplar XOR training set (2 inputs / 1 output), copied
  from `distro/data/` so the harness is self-contained
- `xor.in` — line-by-line menu answers driving a full session: load the XOR set,
  SimpleProp with 3 hidden nodes, X-entropy error, randomize weights, canonical
  backprop, quit
- `xor_train_save.in` / `xor_verify.in` — the train-and-save / load-and-forward-pass
  scripts used by `verify_oracle.sh`
- `run_xor.sh` — pipes `xor.in` into a binary (default: oracle) from `runs/`
  (gitignored — the program writes `model.txt` and `neuron.log` into its cwd)
- `verify_oracle.sh` — deterministic oracle-vs-3.0 comparison (see Usage)
- `xor_reference_output.txt` — one captured oracle session

## Notes

- **Nondeterminism:** "Randomize weights" is unseeded, so iteration counts and final
  weights differ between runs (observed 90k and 300k iterations to converge on XOR).
  The invariants to compare are the endpoint statistics: 100% classification accuracy,
  ROC area 1.000000 (trapezoidal), Kolmogorov-Smirnov D = 1 p = 0.0970269.
  `verify_oracle.sh` sidesteps this entirely by comparing forward passes on identical
  saved weights.
- **Version string:** the oracle greets as "neuron 2.63" although its configure.ac says
  2.64 — the tarball's pre-generated `configure` predates the version bump. Cosmetic.
- **Legacy C++14 dependency:** the untouched legacy source relies on clang's C++14
  default (`std::unary_function`, dynamic exception specs). The 3.0 engine in `src/`
  has those removed and builds at `-std=c++17` with zero warnings.
