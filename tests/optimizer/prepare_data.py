#!/usr/bin/env python3
"""Materialize the Step 0B benchmark datasets.

THIS SCRIPT REUSES THE MAINTAINED RECIPE, it does not reimplement it.  Every
file it writes comes from `docs/datasets/civic-choice/generate.py` followed by
`tools/mkdataset.py --onehot --refcat`, which is exactly the grooming AGENTS.md
and the Civic Choice walkthrough prescribe.  A second encoder written here would
be a second implementation of the one thing that decides what the input columns
MEAN (rule 6), and a benchmark whose data is encoded differently from the way a
user encodes theirs is measuring a different problem.

The row-count series is the SAME PROBLEM AT DIFFERENT SIZES.  That is a claim,
so it is checked rather than assumed: the generator is called with one seed and
different `--rows`, and this script REFUSES to proceed unless every groomed file
carries a byte-identical column key.  If a larger draw ever introduced a
category the 6,000-row draw did not contain, one-hot would silently widen the
design matrix and the "scaling" series would be scaling two different models.

Nothing here is committed.  `tests/optimizer/data/` is generated, and a
benchmark row identifies its data by CONTENT hash (`data_id`) rather than by
path, so provenance does not depend on this directory surviving.

    python3 tests/optimizer/prepare_data.py            # write and verify
    python3 tests/optimizer/prepare_data.py --check    # verify only
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(HERE, "data")

GENERATOR = os.path.join(ROOT, "docs", "datasets", "civic-choice", "generate.py")
GROOMER = os.path.join(ROOT, "tools", "mkdataset.py")
COMMITTED_CSV = os.path.join(ROOT, "docs", "datasets", "civic-choice", "civic_choice.csv")

# The committed walkthrough dataset's own seed.  Using it means the 6,000-row
#    benchmark file is the SAME observations the committed CSV holds, which is
#    checked below rather than asserted.
SEED = 20260724

# The row-count scaling series.  6,000 is the committed walkthrough size; the
#    rest are the same generator drawing more rows.  400,000 is the largest that
#    still fits a reasonable characterization budget on this hardware -- see
#    docs/learning_research/optimizer_baseline_results.md for the measurement
#    that chose it.
ROW_COUNTS = [6000, 25000, 100000, 400000]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def content_sha256(path):
    """SHA-256 of the OBSERVATIONS, with line endings normalized.

    The committed `civic_choice.csv` is stored with CRLF endings while the
    generator writes LF, so a raw byte comparison of the two reports a
    difference that no model can see.  The claim being checked is "these are the
    same observations", so the digest is taken over the same normalization the
    reader applies.  (This is also why the committed CSV's raw digest does not
    equal the one its README publishes: the published value is the LF form.)
    """
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit("failed: " + " ".join(cmd))
    return r.stdout


def groomed_name(rows):
    return os.path.join(DATA, "civic_%d.txt" % rows)


def build(check_only):
    if not os.path.isfile(GENERATOR):
        raise SystemExit("missing generator: " + GENERATOR)
    if not os.path.isfile(GROOMER):
        raise SystemExit("missing groomer: " + GROOMER)

    if check_only:
        missing = [groomed_name(n) for n in ROW_COUNTS
                   if not os.path.isfile(groomed_name(n))]
        if missing:
            raise SystemExit("not prepared; run without --check:\n  "
                             + "\n  ".join(missing))
        for n in ROW_COUNTS:
            p = groomed_name(n)
            with open(p) as f:
                have = sum(1 for _ in f)
            if have != n:
                raise SystemExit("%s holds %d rows, expected %d" % (p, have, n))
        print("prepared: " + ", ".join(str(n) for n in ROW_COUNTS))
        return

    os.makedirs(DATA, exist_ok=True)
    work = tempfile.mkdtemp(prefix="optbench-data-")
    keys = {}
    try:
        for n in ROW_COUNTS:
            # THE 6,000-ROW SIZE IS THE COMMITTED WALKTHROUGH FILE ITSELF, not a
            #    regeneration of it.  The primary application benchmark should be
            #    the artifact readers actually have.  The generator is still run
            #    at that size, and its observations must match, so a generator
            #    change that silently diverged from the committed CSV is caught
            #    here rather than in a results table nobody can reproduce.
            csv = os.path.join(work, "civic_%d.csv" % n)
            run([sys.executable, GENERATOR, "--seed", str(SEED),
                 "--rows", str(n), "-o", csv])

            if n == 6000 and os.path.isfile(COMMITTED_CSV):
                if content_sha256(csv) != content_sha256(COMMITTED_CSV):
                    raise SystemExit(
                        "the 6000-row draw no longer matches the committed\n"
                        "docs/datasets/civic-choice/civic_choice.csv. The\n"
                        "benchmark and the walkthrough would describe different\n"
                        "data. Resolve before benchmarking.")
                csv = COMMITTED_CSV

            key = os.path.join(work, "key_%d.txt" % n)
            inputs = os.path.join(work, "inputs_%d.txt" % n)
            out = groomed_name(n)
            run([sys.executable, GROOMER, "--onehot", "--refcat",
                 "--key", key, "--inputs", inputs, "-o", out, csv])
            with open(key) as f:
                keys[n] = f.read()

        # THE SAME PROBLEM AT DIFFERENT SIZES, checked.  A differing key means
        #    the encoded design matrix differs, and a row-count comparison over
        #    two different design matrices is not a scaling measurement.
        reference = keys[ROW_COUNTS[0]]
        for n in ROW_COUNTS[1:]:
            if keys[n] != reference:
                raise SystemExit(
                    "the %d-row draw groomed to a DIFFERENT column key than the\n"
                    "%d-row draw. The scaling series would not be one problem at\n"
                    "several sizes. Refusing to prepare." % (n, ROW_COUNTS[0]))

        shutil.copyfile(os.path.join(work, "key_%d.txt" % ROW_COUNTS[0]),
                        os.path.join(DATA, "civic_key.txt"))
        shutil.copyfile(os.path.join(work, "inputs_%d.txt" % ROW_COUNTS[0]),
                        os.path.join(DATA, "civic_inputs.txt"))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("column key identical across all %d sizes" % len(ROW_COUNTS))
    print("%-28s %8s  %s" % ("file", "rows", "sha256"))
    for n in ROW_COUNTS:
        p = groomed_name(n)
        print("%-28s %8d  %s" % (os.path.relpath(p, ROOT), n, sha256(p)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify the prepared files exist and have the right row counts")
    build(ap.parse_args().check)


if __name__ == "__main__":
    main()
