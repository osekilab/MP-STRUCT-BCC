#!/usr/bin/env bash
set -euo pipefail

# Evaluate a trained model on BLiMP (eng), BLiMP-NL (nld), and ZhoBLiMP (zho).
# Requires: pip install lm-eval
#
# Usage:
#   bash scripts/evaluate.sh <hf_model_dir> [GPU_ID]
#
# Example:
#   bash scripts/evaluate.sh hf_models/bcc_seed42 0

MODEL_DIR="${1:?Usage: $0 <hf_model_dir> [GPU_ID]}"
GPU="${2:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"

echo "=========================================="
echo "  Evaluating: ${MODEL_DIR}"
echo "=========================================="

echo ""
echo "--- BLiMP (English, 67 subtasks) ---"
lm_eval --model hf \
    --model_args "pretrained=${MODEL_DIR}" \
    --tasks blimp \
    --batch_size 16 \
    --output_path "results/${MODEL_DIR##*/}_eng"

echo ""
echo "--- BLiMP-NL (Dutch, 22 subtasks) ---"
lm_eval --model hf \
    --model_args "pretrained=${MODEL_DIR}" \
    --tasks blimp_nl \
    --batch_size 16 \
    --output_path "results/${MODEL_DIR##*/}_nld"

echo ""
echo "--- ZhoBLiMP (Chinese, 118 subtasks) ---"
lm_eval --model hf \
    --model_args "pretrained=${MODEL_DIR}" \
    --tasks zhoblimp \
    --batch_size 16 \
    --output_path "results/${MODEL_DIR##*/}_zho"

echo ""
echo "Evaluation complete. Results saved in results/"
