#!/usr/bin/env python3
"""Convert a delimited text file (presumably .csv from Excel) into a
neuron-ready raw dataset.

Python port of neUROn2++'s mkdataset.pl. Standard library only — this tool
must run on a bare python3 with no pip installs (see tools/README.md).

Behavior, matching the Perl original:
  * The first line is a header of column labels unless --noheader.
  * Rows whose last column (the outcome) is blank are dropped (unless
    --noelim), with a warning.
  * Rows containing non-numeric data are dropped from the output, with a
    warning naming the offending column(s).
  * Any column that has blanks in the surviving rows is emitted as an
    indicator PAIR: "0, 0" when blank, "1, value" when present, so the
    network can learn from missingness itself.
  * --key FILE writes "Column N: label" for the OUTPUT file's columns.
  * --inputs FILE writes the input-node structure string (e.g. "0; 1, 2; 3")
    for the engine's stepwise regression variable definitions.

Beyond the Perl original:
  * Input is parsed with the csv module, so quoted Excel fields (including
    embedded delimiters) are handled correctly.
  * --delimiter lets the input use something other than a comma (e.g. the
    semicolons of the UCI bank-marketing export; use '\\t' for tab).
  * --onehot encodes categorical text columns as one-hot indicator groups
    (one 0/1 column per distinct value; a blank encodes as all zeros).
    With no value it auto-detects every non-numeric column; alternatively
    give an explicit comma-separated list of 1-based column numbers.
    A text OUTCOME (last column) with exactly two values is mapped to a
    single 0/1 column instead (e.g. "no" -> 0, "yes" -> 1).
    The --key and --inputs files group one-hot columns as one variable.
  * --refcat drops the first (alphabetical) level of each one-hot input
    group, making it the reference category (k-1 columns per variable).
    Use this for models with an intercept — logistic regression most of
    all: with all k indicator columns, each group sums to a constant, the
    fit is perfectly collinear with the intercept, and the coefficients
    are not identifiable. Neural networks tolerate the full k columns.
"""

import argparse
import csv
import sys


def die(message):
    sys.stderr.write("ERROR: " + message + "\n")
    sys.exit(1)


def is_blank(field):
    return field.strip() == ""


def is_numeric(field):
    try:
        float(field)
        return True
    except ValueError:
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a .csv file into a neuron-ready raw dataset "
                    "(port of neUROn2++'s mkdataset.pl)")
    # Single-dash aliases keep the Perl script's flag spelling working.
    parser.add_argument("-quiet", "--quiet", action="store_true",
                        help="suppress messages")
    parser.add_argument("-noheader", "--noheader", action="store_true",
                        help="no first line of labels will be processed")
    parser.add_argument("-key", "--key", metavar="FILE",
                        help="create a key file from the label line")
    parser.add_argument("-noelim", "--noelim", action="store_true",
                        help="do not eliminate rows with the last column blank")
    parser.add_argument("-inputs", "--inputs", metavar="FILE",
                        help="print the input node structure to file")
    parser.add_argument("-delimiter", "--delimiter", metavar="CHAR", default=",",
                        help="input field delimiter (default ','; use '\\t' for tab)")
    parser.add_argument("-onehot", "--onehot", metavar="COLS", nargs="?",
                        const="auto", default=None,
                        help="one-hot encode categorical text columns; give "
                             "comma-separated 1-based column numbers, or no "
                             "value to auto-detect all non-numeric columns")
    parser.add_argument("-refcat", "--refcat", action="store_true",
                        help="drop the first level of each one-hot input "
                             "group as the reference category (k-1 columns; "
                             "use for models with an intercept, e.g. "
                             "logistic regression)")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="output file")
    parser.add_argument("file", help="the delimited input file")
    args = parser.parse_args()

    # Same conflict rules as the Perl original
    if args.key and args.noheader:
        die("a key is specified, but a label line is not")
    if args.key and not args.output:
        die("a key only makes sense if an output file is specified")
    if args.inputs and not args.output:
        die("an input structure file only makes sense if an output file is specified")

    args.delimiter = args.delimiter.replace("\\t", "\t")
    if len(args.delimiter) != 1:
        die("the delimiter must be a single character")
    if args.refcat and args.onehot is None:
        die("--refcat only makes sense together with --onehot")

    return args


def read_rows(filename, delimiter):
    """Read the delimited file as (physical_line_number, [fields]) tuples."""
    try:
        f = open(filename, "r", newline="")
    except OSError:
        die(filename + " is not a valid file name")
    with f:
        reader = csv.reader(f, delimiter=delimiter)
        return [(reader.line_num, row) for row in reader if row]


def main():
    args = parse_args()
    verbose = not args.quiet
    has_header = not args.noheader
    eliminate = not args.noelim

    rows = read_rows(args.file, args.delimiter)
    if not rows:
        die(args.file + " is empty")

    # --- Pass 1: column count, labels, and per-column blank counts ---------
    n_col = len(rows[0][1])
    if verbose:
        print(f"{n_col} columns found on first line")

    labels = []
    if has_header:
        labels = [field.strip() for field in rows[0][1]]

    def label_of(i):
        return labels[i] if i < len(labels) else f"(column {i + 1})"

    blank_count = [0] * n_col
    has_text = [False] * n_col  # column contains non-blank non-numeric data
    data_rows = rows[1:] if has_header else rows

    for line_num, row in data_rows:
        if eliminate and is_blank(row[-1]):
            if verbose:
                print(f"WARNING: Ignoring line {line_num}, missing output")
            continue
        if len(row) != n_col:
            die(f"{len(row)} columns found on line {line_num}")
        for i, field in enumerate(row):
            if is_blank(field):
                blank_count[i] += 1
            elif not is_numeric(field):
                has_text[i] = True

    # --- Categorical (one-hot) columns --------------------------------------
    categorical = set()
    if args.onehot == "auto":
        categorical = {i for i in range(n_col) if has_text[i]}
    elif args.onehot is not None:
        for piece in args.onehot.split(","):
            piece = piece.strip()
            if not piece.isdigit() or not 1 <= int(piece) <= n_col:
                die(f"--onehot column '{piece}' is not a column number "
                    f"between 1 and {n_col}")
            categorical.add(int(piece) - 1)

    levels = {}  # categorical column -> sorted list of distinct values
    if categorical:
        seen = {i: set() for i in categorical}
        for line_num, row in data_rows:
            if eliminate and is_blank(row[-1]):
                continue
            for i in categorical:
                if not is_blank(row[i]):
                    seen[i].add(row[i].strip())
        levels = {i: sorted(seen[i]) for i in categorical}

    # A categorical outcome must be binary; it maps to one 0/1 column.
    outcome_map = None
    out_i = n_col - 1
    if out_i in categorical:
        if len(levels[out_i]) != 2:
            die(f"outcome column {n_col} has {len(levels[out_i])} distinct "
                "values; a text outcome must have exactly 2")
        outcome_map = {levels[out_i][0]: "0", levels[out_i][1]: "1"}

    def enc_levels(i):
        """The levels of column i that get their own output column."""
        return levels[i][1:] if args.refcat else levels[i]

    if args.refcat:
        for i in sorted(categorical):
            if i == out_i and outcome_map:
                continue
            if len(levels[i]) < 2:
                die(f"--refcat: column {i + 1} has only "
                    f"{len(levels[i])} level(s); nothing would remain")
            if blank_count[i] and verbose:
                print(f"WARNING: Column {i + 1} has blanks, which --refcat "
                      "makes indistinguishable from the reference level")

    if verbose and has_header:
        print("\nColumn labels:")
        for i, label in enumerate(labels):
            print(f"Column {i + 1}: {label}")

    if verbose:
        print("\nPopulated report:")
        for i, count in enumerate(blank_count):
            status = "completely populated" if count == 0 else f"{count} rows blank"
            print(f"Column {i + 1}: {status}")

    if verbose and categorical:
        print("\nOne-hot encoding:")
        for i in sorted(categorical):
            if i == out_i and outcome_map:
                v0, v1 = levels[i]
                print(f"Outcome column {i + 1} ({label_of(i)}): "
                      f"{v0} -> 0, {v1} -> 1")
            elif args.refcat:
                print(f"Column {i + 1} ({label_of(i)}): "
                      f"{len(levels[i])} levels, reference = {levels[i][0]}: "
                      f"{', '.join(enc_levels(i))}")
            else:
                note = " (blank = all zeros)" if blank_count[i] else ""
                print(f"Column {i + 1} ({label_of(i)}): "
                      f"{len(levels[i])} levels{note}: {', '.join(levels[i])}")

    # --- Pass 2: write the output file --------------------------------------
    if args.output:
        try:
            out = open(args.output, "w", newline="\n")
        except OSError:
            die("cannot open output file " + args.output)

        if verbose:
            print(f"\nFile {args.output} will record output")
            if not eliminate:
                print("Rows with blank last columns will NOT be eliminated")

        row_count = 0
        with out:
            for line_num, row in data_rows:
                if eliminate and is_blank(row[-1]):
                    continue

                fields = [field.strip() for field in row]
                bad_columns = [str(i + 1) for i, field in enumerate(fields)
                               if i not in categorical
                               and not is_blank(field) and not is_numeric(field)]
                if bad_columns:
                    if verbose:
                        print(f"WARNING: Ignoring line {line_num}, "
                              f"non-numeric data in column(s) {', '.join(bad_columns)}")
                    continue
                if outcome_map and fields[out_i] not in outcome_map:
                    if verbose:
                        print(f"WARNING: Ignoring line {line_num}, "
                              f"outcome not one of the 2 known values")
                    continue

                pieces = []
                for i, field in enumerate(fields):
                    if i == out_i and outcome_map:
                        pieces.append(outcome_map[field])
                    elif i in categorical:
                        pieces.extend("1" if field == v else "0"
                                      for v in enc_levels(i))
                    elif blank_count[i] == 0:
                        pieces.append(field)
                    elif is_blank(field):
                        pieces.append("0, 0")
                    else:
                        pieces.append("1, " + field)
                out.write(", ".join(pieces) + "\n")
                row_count += 1

        if verbose:
            print(f"\nNumber of rows recorded in {args.output} is {row_count}")
            out_cols = 0
            for i, count in enumerate(blank_count):
                if i == out_i and outcome_map:
                    out_cols += 1
                elif i in categorical:
                    out_cols += len(enc_levels(i))
                else:
                    out_cols += 1 if count == 0 else 2
            message = f"Number of columns was {out_cols}"
            if eliminate:
                message += (f", which would be {out_cols - 1} input nodes "
                            "in a 1-output dataset")
            print(message)

    # --- Key and input-structure files --------------------------------------
    if args.inputs and verbose:
        note = " (assuming 1 output node)" if eliminate else ""
        print(f"File {args.inputs} will record input node structure{note}")
    if args.key and verbose:
        print(f"Writing key to file {args.key}")

    if args.output and (args.key or args.inputs or verbose):
        key_lines = []
        structure = []  # one entry per original column: list of output-file column indices
        col = 0  # 0-based column index in the output file
        for i, count in enumerate(blank_count):
            label = label_of(i)
            if i == out_i and outcome_map:
                v0, v1 = levels[i]
                structure.append([col])
                key_lines.append(f"Column {col + 1}: {label} ({v0} = 0, {v1} = 1)")
                col += 1
            elif i in categorical:
                k = len(enc_levels(i))
                structure.append(list(range(col, col + k)))
                ref = (f", reference = {levels[i][0]}" if args.refcat else "")
                span = (f"Columns {col + 1}-{col + k}" if k > 1
                        else f"Column {col + 1}")
                key_lines.append(f"{span}: {label} "
                                 f"(one-hot{ref}: {', '.join(enc_levels(i))})")
                col += k
            elif count == 0:
                structure.append([col])
                key_lines.append(f"Column {col + 1}: {label}")
                col += 1
            else:
                structure.append([col, col + 1])
                key_lines.append(f"Columns {col + 1}, {col + 2}: {label}")
                col += 2

        if verbose:
            print("\nColumn labels in output file:")
            for line in key_lines:
                print(line)

        if args.key:
            try:
                with open(args.key, "w", newline="\n") as keyfile:
                    for line in key_lines:
                        keyfile.write(line + "\n")
            except OSError:
                die("cannot open key file " + args.key)

        if args.inputs:
            # The last column is the output node when eliminating, so it is
            # not part of the input structure.
            variables = structure[:-1] if eliminate else structure

            def group_str(group):
                # Contiguous runs of 3+ columns use the engine's hyphen-range
                # syntax (e.g. "1-4"), as in the classic psa_defs.txt.
                if len(group) >= 3:
                    return f"{group[0]}-{group[-1]}"
                return ", ".join(str(c) for c in group)

            try:
                with open(args.inputs, "w", newline="\n") as inputsfile:
                    inputsfile.write("; ".join(group_str(g) for g in variables))
                    inputsfile.write("\n")
            except OSError:
                die("cannot open file " + args.inputs)


if __name__ == "__main__":
    main()
