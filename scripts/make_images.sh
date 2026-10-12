#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p docs/images
for f in ae_harta vae_harta ae_reconstructii vae_reconstructii_d16 vae_generate_d2 vae_generate_d16; do
    convert "results/$f.pgm" -filter point -resize 300% "docs/images/$f.png"
done
convert results/vae_puncte.ppm docs/images/vae_puncte.png