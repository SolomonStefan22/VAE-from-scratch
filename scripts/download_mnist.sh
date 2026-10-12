#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p data
cd data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    if [ ! -f "$f" ]; then
        wget -q https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz
        gunzip "$f.gz"
    fi
done
echo "MNIST e in data/"