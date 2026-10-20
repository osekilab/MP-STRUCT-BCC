#!/usr/bin/env bash
set -euo pipefail

# Train all 4 conditions (BL, SD, MS, BCC) for a single seed.
#
# Usage:
#   bash scripts/train.sh [SEED] [GPU_ID]
#
# Example:
#   bash scripts/train.sh 42 0

SEED="${1:-42}"
GPU="${2:-0}"
DATASET="en_nld_zho_equal"
PPT_DIR="data/ppt"

export CUDA_VISIBLE_DEVICES="$GPU"

echo "=========================================="
echo "  [BL] Baseline (no PPT), seed=${SEED}"
echo "=========================================="
python training.py \
    --dataset "$DATASET" \
    --words_per_epoch 100000000 \
    --experiment_name "baseline_seed${SEED}" \
    --seed "$SEED" \
    --no_deduct_ppt

echo ""
echo "=========================================="
echo "  [SD] k-Shuffle Dyck, seed=${SEED}"
echo "=========================================="
python training.py \
    --dataset "$DATASET" \
    --ppt_data_path "${PPT_DIR}/shuff_dyck/shuff_dyck_k64_100000_1024.txt" \
    --ppt_steps 500 \
    --ppt_learning_rate 0.0005 \
    --ppt_warmup_steps 100 \
    --words_per_epoch 100000000 \
    --experiment_name "shuff_dyck_seed${SEED}" \
    --seed "$SEED" \
    --no_deduct_ppt

echo ""
echo "=========================================="
echo "  [MS] MP-STRUCT, seed=${SEED}"
echo "=========================================="
python training.py \
    --dataset "$DATASET" \
    --ppt_data_path "${PPT_DIR}/mp_struct/mg_ids_100000_1024.txt" \
    --ppt_steps 500 \
    --ppt_learning_rate 0.0005 \
    --ppt_warmup_steps 100 \
    --words_per_epoch 100000000 \
    --experiment_name "mp_struct_seed${SEED}" \
    --seed "$SEED" \
    --no_deduct_ppt

echo ""
echo "=========================================="
echo "  [BCC] MP-STRUCT BCC, seed=${SEED}"
echo "=========================================="
python training.py \
    --dataset "$DATASET" \
    --ppt_data_path "${PPT_DIR}/mp_struct_universal/universal_v5_p1_ids.txt" \
    --ppt_steps 500 \
    --ppt_learning_rate 0.0005 \
    --ppt_warmup_steps 100 \
    --words_per_epoch 100000000 \
    --experiment_name "bcc_seed${SEED}" \
    --seed "$SEED" \
    --no_deduct_ppt

echo ""
echo "All 4 conditions trained with seed=${SEED}."
