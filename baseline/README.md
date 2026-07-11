# Legacy baseline harness

The legacy neUROn2++ 2.64 binary serves as the numerical oracle for validating the 3.0
engine. It builds clean on Apple Silicon with modern clang++ and GSL 2.8 — zero source
changes needed (verified 2026-07-11).

## Usage

```sh
./build_legacy.sh              # extracts ../../distro/neuron-2.64.tar.gz into build/, configure + make
./run_xor.sh                   # scripted XOR training session against the legacy oracle
./run_xor.sh ../build/neuron   # same session against the 3.0 engine
./verify_oracle.sh             # deterministic cross-check: legacy vs 3.0 must match exactly
```

`xor_reference_output.txt` is a captured session for reference.

`verify_oracle.sh` is the real acceptance test: it trains XOR with the legacy binary,
saves the network, loads the identical weights into both binaries (which triggers one
eta-0 forward pass printing error + statistics), and diffs the sessions. Passing means
the 3.0 engine is numerically identical to neUROn2++ on that path.
First passed 2026-07-11.

## Files

- `build_legacy.sh` — builds `build/neuron-2.64/src/neuron` (build/ is gitignored)
- `xor.in` — line-by-line menu answers driving a full session: load
  `distro/data/xor_discrete.set` (2 in / 1 out), SimpleProp with 3 hidden nodes,
  X-entropy error, randomize weights, canonical backprop, quit
- `xor_train_save.in` / `xor_verify.in` — the train-and-save / load-and-forward-pass
  scripts used by `verify_oracle.sh`
- `run_xor.sh` — pipes `xor.in` into a binary (default: legacy) from `runs/`
  (gitignored — the program writes `model.txt` and `neuron.log` into its cwd)
- `verify_oracle.sh` — deterministic legacy-vs-3.0 comparison (see Usage)
- `xor_reference_output.txt` — one captured run

## Notes

- **Nondeterminism:** "Randomize weights" is unseeded, so iteration counts and final
  weights differ between runs (observed 90k and 300k iterations to converge on XOR).
  The invariants to compare against 3.0 are the endpoint statistics: 100% classification
  accuracy, ROC area 1.000000 (trapezoidal), Kolmogorov-Smirnov D = 1 p = 0.0970269.
  For deterministic comparisons, load a saved network (e.g. `distro/data/Run9`) and use
  the eta-0 forward-pass trick the driver itself uses (`use_model()` in neuron.cpp).
- **Version string:** the binary greets as "neuron 2.63" although configure.ac says 2.64 —
  the tarball's pre-generated `configure` predates the version bump. Cosmetic.
- **One warning class only:** `std::unary_function` deprecation at five spots in
  `function_defs.h` (the activation-function functors). That template was removed in
  C++17, so the clean build relies on clang's C++14 default — building with
  `-std=c++17` or later would fail until those functors drop the inheritance
  (it's vestigial; a five-line fix if ever needed).
