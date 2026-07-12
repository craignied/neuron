#!/usr/bin/env python3
"""Convert a comma-delimited file (presumably .csv from Excel) into a
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

Improvement over the Perl: input is parsed with the csv module, so quoted
Excel fields (including embedded commas) are handled correctly.
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
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="output file")
    parser.add_argument("file", help="the comma-delimited input file")
    args = parser.parse_args()

    # Same conflict rules as the Perl original
    if args.key and args.noheader:
        die("a key is specified, but a label line is not")
    if args.key and not args.output:
        die("a key only makes sense if an output file is specified")
    if args.inputs and not args.output:
        die("an input structure file only makes sense if an output file is specified")

    return args


def read_rows(filename):
    """Read the csv file as (physical_line_number, [fields]) tuples."""
    try:
        f = open(filename, "r", newline="")
    except OSError:
        die(filename + " is not a valid file name")
    with f:
        reader = csv.reader(f)
        return [(reader.line_num, row) for row in reader if row]


def main():
    args = parse_args()
    verbose = not args.quiet
    has_header = not args.noheader
    eliminate = not args.noelim

    rows = read_rows(args.file)
    if not rows:
        die(args.file + " is empty")

    # --- Pass 1: column count, labels, and per-column blank counts ---------
    n_col = len(rows[0][1])
    if verbose:
        print(f"{n_col} columns found on first line")

    labels = []
    if has_header:
        labels = [field.strip() for field in rows[0][1]]

    blank_count = [0] * n_col
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

    if verbose and has_header:
        print("\nColumn labels:")
        for i, label in enumerate(labels):
            print(f"Column {i + 1}: {label}")

    if verbose:
        print("\nPopulated report:")
        for i, count in enumerate(blank_count):
            status = "completely populated" if count == 0 else f"{count} rows blank"
            print(f"Column {i + 1}: {status}")

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
                               if not is_blank(field) and not is_numeric(field)]
                if bad_columns:
                    if verbose:
                        print(f"WARNING: Ignoring line {line_num}, "
                              f"non-numeric data in column(s) {', '.join(bad_columns)}")
                    continue

                pieces = []
                for i, field in enumerate(fields):
                    if blank_count[i] == 0:
                        pieces.append(field)
                    elif is_blank(field):
                        pieces.append("0, 0")
                    else:
                        pieces.append("1, " + field)
                out.write(", ".join(pieces) + "\n")
                row_count += 1

        if verbose:
            print(f"\nNumber of rows recorded in {args.output} is {row_count}")
            out_cols = sum(1 if count == 0 else 2 for count in blank_count)
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
            label = labels[i] if i < len(labels) else f"(column {i + 1})"
            if count == 0:
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
            try:
                with open(args.inputs, "w", newline="\n") as inputsfile:
                    inputsfile.write("; ".join(
                        ", ".join(str(c) for c in group) for group in variables))
                    inputsfile.write("\n")
            except OSError:
                die("cannot open file " + args.inputs)


if __name__ == "__main__":
    main()
