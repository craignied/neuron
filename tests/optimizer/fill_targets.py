#!/usr/bin/env python3
"""Write measured characterizations into the committed target table.

THE TABLE IS COMMITTED SOURCE, NOT A CACHE.  A target recomputed by the campaign
is not a control: it would move whenever the engine moved, silently, and every
comparison would still look matched.  So characterization is a separate,
deliberate act and this script is the only way its results reach the table --
which means a changed endpoint shows up as a source diff a reviewer can see.

It reads `--characterize` records (JSON Lines, as the probe emits them) and
rewrites the ENDPOINT_TABLE block in harness.h.  Nothing else in the file is
touched, and a record the probe marked unusable is refused rather than written.

    ./build/optimizer_probe --characterize --case NAME > chz.jsonl
    python3 tests/optimizer/fill_targets.py chz.jsonl [more.jsonl ...]
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "harness.h")

# The same headroom the harness declares: an endpoint is reached by
# `achieved < target`, so a target set exactly at the measured objective is one
# the reference itself could not satisfy.
HEADROOM = 1e-5


def key_for(case):
    """The endpoint key a characterization case belongs to.

    Derived from the case NAME, which is built by civicCase() from exactly the
    same three facts, so the two cannot disagree without the name changing.
    """
    # Two spellings, because a characterization can come from a declared ARM's
    # canonical reference or from `--workload KEY`, which names a workload whose
    # arms are not declared at all.  The second is how a workload that lost its
    # endpoint gets re-measured.
    m = re.match(r"civic-(\w+)-r(\d+)-h(\w+)-(practical|strict)-canonical", case)
    if m:
        model, rows, arch, _ = m.groups()
        return "%s-%s-%s" % (model, rows, arch)
    m = re.match(r"([a-z]+)-(\d+)-(\w+)-reference$", case)
    if m:
        return "%s-%s-%s" % m.groups()
    return None


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)

    measured = {}
    binaries = set()
    for path in sys.argv[1:]:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                d = json.loads(line)
                if d.get("record") != "characterization":
                    continue
                k = key_for(d["case"])
                if k is None:
                    raise SystemExit("cannot place case %r in the table"
                                     % d["case"])
                # ONE TABLE, ONE ENGINE.  Targets measured against two
                # different engines are not a matched set: an arm held to one
                # and an arm held to the other are being judged by different
                # code.  Refused rather than merged.
                #
                # ENGINE_ID, NOT SOURCE_ID.  Writing these measurements into
                # harness.h moves source_id by construction, so requiring that
                # would be requiring something that can never hold. What the
                # endpoints depend on is src/, and engine_id is that alone.
                binaries.add((d.get("engine_id"), d.get("engine_files"),
                              d.get("build")))
                measured[k] = d

    if not measured:
        raise SystemExit("no characterization records found")
    if len(binaries) != 1:
        raise SystemExit(
            "these characterizations came from %d different binaries:\n  %s\n"
            "A committed target table must describe one engine."
            % (len(binaries), "\n  ".join(repr(b) for b in sorted(
                binaries, key=lambda x: str(x)))))
    engine_id, engine_files, build = list(binaries)[0]
    if not engine_id or engine_id == "unknown":
        raise SystemExit("these characterizations carry no engine identity; "
                         "they cannot be traced to an engine and are refused")
    print("characterized against engine_id=%s (%s files), build=%s"
          % (engine_id, engine_files, build))

    src = open(HARNESS).read()
    start = src.index("static const Endpoints ENDPOINT_TABLE[] = {")
    end = src.index("};", start) + 2
    block = src[start:end]

    rows = []
    existing_values = {}
    for line in block.splitlines():
        m = re.match(r'\s*\{ "([^"]+)",\s*([^,]+),\s*([^,]+),\s*(\d+),\s*(\d+),',
                     line)
        if not m:
            continue
        rows.append(m.group(1))
        existing_values[m.group(1)] = (m.group(2), m.group(3))

    out = ["static const Endpoints ENDPOINT_TABLE[] = {",
           "\t// Measured against engine_id %s over %s src/ files, build %s."
           % (engine_id, engine_files, build),
           "\t// Regenerate with tests/optimizer/fill_targets.py after any change",
           "\t// to src/ -- these are measurements of THAT engine's behavior."]
    report = []
    for k in rows:
        d = measured.get(k)
        if d is None:
            # A workload with nothing supplied keeps whatever it has -- but ONLY
            # if what it has is "no endpoint".  Keeping a real value measured
            # against some other engine is the mixing this script exists to
            # refuse; keeping a zero is keeping an honest absence.
            existing = existing_values.get(k, ("0", "0"))
            if existing[0].strip() not in ("0", "0.0") or \
                    existing[1].strip() not in ("0", "0.0"):
                raise SystemExit(
                    "no characterization supplied for %r, and it already holds "
                    "measured endpoints (%s, %s). Supply its characterization "
                    "or the table would mix two engines." % (k, existing[0],
                                                             existing[1]))
            out.append('\t{ "%s",%s 0, 0, 20000, 0,\n\t  "%s" },'
                       % (k, " " * max(1, 26 - len(k)),
                          "not characterized against this engine"))
            report.append((k, 0.0, 0.0, 20000, 0, False, False))
            continue

        # A PRACTICAL ENDPOINT EXISTS ONLY IF THE SERIES PLATEAUED.  Where it
        # did not, the last objective is where a budget ran out, and writing it
        # as an endpoint would be the ceiling-is-a-floor error in a new place.
        practical = 0.0
        if d["plateau_fired"]:
            practical = d["practical_objective"] * (1.0 + HEADROOM)

        # A STRICT ENDPOINT EXISTS ONLY IF THE REFERENCE CONVERGED.
        strict = 0.0
        note = ""
        if d["converged"]:
            strict = d["strict_objective"] * (1.0 + HEADROOM)
        else:
            note = ("canonical did not converge: gradient %.2e against the "
                    "engine's 1e-6 rule after %d iterations"
                    % (d["final_gradmax"], d["iterations"]))

        # The arms' safety ceiling, from what the reference actually needed:
        # generous enough that reaching it is a real failure to converge, not a
        # budget this table set too tightly.
        need = d["practical_iteration"] if d["plateau_fired"] else 0
        if d["converged"]:
            need = max(need, d["iterations"])
        ceiling = max(20000, int(need * 2.5 / 10000 + 1) * 10000)

        out.append('\t{ "%s",%s %.9g, %.9g, %d, %d,\n\t  "%s" },'
                   % (k, " " * max(1, 26 - len(k)), practical, strict,
                      ceiling, d["ceiling"], note))
        report.append((k, practical, strict, ceiling, d["ceiling"],
                       d["plateau_fired"], d["converged"]))
    out.append("\t{ 0, 0, 0, 0, 0, \"\" }")
    out.append("};")

    open(HARNESS, "w").write(src[:start] + "\n".join(out) + src[end:])

    print("%-26s %12s %12s %9s %9s" %
          ("workload", "practical", "strict", "ceiling", "char ceil"))
    for k, p, s, c, cc, fired, conv in report:
        print("%-26s %12s %12s %9d %9d" %
              (k,
               ("%.8f" % p) if fired else "none (no plateau)",
               ("%.8f" % s) if conv else "none (not conv.)",
               c, cc))


if __name__ == "__main__":
    main()
