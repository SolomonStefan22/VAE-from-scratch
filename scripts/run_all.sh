#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/show_mnist
./build/train_ae
./build/train_vae 2
./build/train_vae 16
./build/train_classifier
./build/evaluate 2
./build/evaluate 16