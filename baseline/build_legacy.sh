#!/bin/sh
# Build the legacy neUROn2++ 2.64 reference binary from the release tarball.
# Produces baseline/build/neuron-2.64/src/neuron (arm64, linked against Homebrew GSL).
set -e
cd "$(dirname "$0")"
TARBALL=../../distro/neuron-2.64.tar.gz
mkdir -p build
tar xzf "$TARBALL" -C build
cd build/neuron-2.64
./configure CPPFLAGS=-I/opt/homebrew/include LDFLAGS=-L/opt/homebrew/lib
make
echo "Built: $(pwd)/src/neuron"
