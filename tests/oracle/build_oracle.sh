#!/bin/sh
# Build the legacy neUROn2++ oracle binary from its release tarball.
# The tarball is extracted into a throwaway temp dir; only the binary is kept
# (bin/oracle), so no legacy source tree lives inside this repo.
# Note: harmless "aclocal/automake missing" warnings appear on stderr —
# extraction timestamps make the 2007 build think autotools files need regenerating;
# the `missing` shim warns and continues. Legacy unary_function warnings are expected too.
set -e
cd "$(dirname "$0")"
TARBALL=../../../distro/neuron-2.64.tar.gz   # the final neUROn2++ release
[ -f "$TARBALL" ] || { echo "Tarball not found: $TARBALL" >&2; exit 1; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
tar xzf "$TARBALL" -C "$TMP"
SRC=$(echo "$TMP"/*)
( cd "$SRC" \
  && ./configure CPPFLAGS=-I/opt/homebrew/include LDFLAGS=-L/opt/homebrew/lib > /dev/null \
  && make > /dev/null )
mkdir -p bin
cp "$SRC/src/neuron" bin/oracle
echo "Oracle binary: $(pwd)/bin/oracle"
