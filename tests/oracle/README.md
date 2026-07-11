# Legacy oracle harness

The final neUROn2++ release binary serves as the numerical oracle for validating the
3.0 engine: any engine change must still produce output identical to the original.

Only the oracle *binary* lives here (`bin/oracle`, gitignored); its source is never
extracted into this repo. `build_oracle.sh` compiles it in a temp dir from the release
tarball kept at `../../../distro/neuron-2.64.tar.gz` — that path is the only place the
old version number appears. (Craig's machines only — distribution users don't need
the oracle.)

## Usage

```sh
./build_oracle.sh                 # builds bin/oracle from the release tarball
./run_xor.sh                      # scripted XOR training session against the oracle
./run_xor.sh ../../build/neuron   # same session against the 3.0 engine
./verify_oracle.sh                # deterministic cross-check: oracle vs 3.0 must match exactly
```

`verify_oracle.sh` is the real acceptance test: it loads the committed reference
network (`xor_net.txt`) into both binaries (which triggers one eta-0 forward pass
printing error + statistics) and diffs the sessions — fully deterministic, no
training involved. Passing means the 3.0 engine is numerically identical to
neUROn2++ on that path. First passed 2026-07-11.

## Known oracle bug: Kolmogorov-Smirnov (fixed in 3.0)

Found 2026-07-11 by this very harness: legacy `TwoSet::KScalc()` ported the
Numerical Recipes two-sample K-S routine keeping NR's **1-based indices on 0-based
vectors** — it skips element 0 and reads one past the end of both arrays, so the
oracle's K-S statistic is partly heap garbage (observed D flapping between 0.5 and
the correct 1 on identical inputs across builds). 3.0's `src/twoset.cpp` fixes the
indexing. Consequently `verify_oracle.sh` excludes the K-S line from the diff and
instead checks 3.0's K-S against the known-correct value for the reference network
(D = 1, p = 0.0970269 — XOR separates perfectly). Every other output line must
match the oracle exactly.

## Files

- `build_oracle.sh` — builds `bin/oracle` (extracts the tarball to a temp dir, keeps
  only the binary)
- `xor_discrete.set` — the 4-exemplar XOR training set (2 inputs / 1 output), copied
  from the legacy sample data so the harness is self-contained
- `xor.in` — line-by-line menu answers driving a full session: load the XOR set,
  SimpleProp with 3 hidden nodes, X-entropy error, randomize weights, canonical
  backprop, quit
- `xor_net.txt` — the committed reference network (SimpleProp, 3 hidden nodes,
  trained on XOR by the oracle); the fixed input to `verify_oracle.sh`
- `xor_train_save.in` / `xor_verify.in` — train-and-save (for regenerating the
  reference network) / load-and-forward-pass session scripts
- `run_xor.sh` — pipes `xor.in` into a binary (default: the oracle) from `runs/`
  (gitignored — the program writes `model.txt` and `neuron.log` into its cwd)
- `verify_oracle.sh` — deterministic oracle-vs-3.0 comparison (see Usage)
- `xor_reference_output.txt` — one captured XOR session from the 3.0 engine

## Notes

- **Nondeterminism:** "Randomize weights" is unseeded, so iteration counts and final
  weights differ between runs (observed 90k and 300k iterations to converge on XOR).
  The invariants to compare are the endpoint statistics: 100% classification accuracy,
  ROC area 1.000000 (trapezoidal), Kolmogorov-Smirnov D = 1 p = 0.0970269.
  `verify_oracle.sh` sidesteps this entirely by comparing forward passes on identical
  saved weights.
- **Oracle banner:** the oracle greets with an older version string than its tarball
  name suggests (its pre-generated `configure` predates the final version bump).
  Cosmetic — ignore it.
- **Legacy C++14 dependency:** the untouched legacy source relies on clang's C++14
  default (`std::unary_function`, dynamic exception specs). The 3.0 engine in `src/`
  has those removed and builds at `-std=c++17` with zero warnings.
