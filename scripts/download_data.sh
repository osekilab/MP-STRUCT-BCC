#!/usr/bin/env bash
set -euo pipefail

# Download BabyLM 2026 multilingual training data from HuggingFace.
# Requires: pip install huggingface-hub

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

DATASETS=(
    BabyLM-community/babylm-nld
    BabyLM-community/babylm-zho
    BabyLM-community/BabyLM-2026-Strict
)

for repo in "${DATASETS[@]}"; do
    name="${repo##*/}"
    echo "Downloading: $repo -> $DATA_DIR/$name"
    huggingface-cli download "$repo" --repo-type dataset --local-dir "$DATA_DIR/$name"
done

echo ""
echo "All datasets downloaded into $DATA_DIR/"
echo ""
echo "Next step: build the trilingual dataset with:"
echo "  python build_multilingual_dataset.py --data_dir $DATA_DIR"
