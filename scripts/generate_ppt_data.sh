#!/usr/bin/env bash
set -euo pipefail

# Generate formal language data for all three PPT conditions.
# This script produces the pre-pretraining data used in the paper:
#   SD:  k-Shuffle Dyck (k=64) — Hu et al. (2025)
#   MS:  MP-STRUCT — fixed English-like typological parameters
#   BCC: MP-STRUCT BCC — sampled typological profiles

PPT_DIR="${1:-data/ppt}"
mkdir -p "$PPT_DIR"

echo "=========================================="
echo "  [SD] k-Shuffle Dyck (k=64)"
echo "=========================================="
python -m ppt.k_shuffle_dyck generate \
    --file_dir "${PPT_DIR}/shuff_dyck" \
    --k 64 \
    --n 100000 \
    --max_length 1024 \
    --p_open 0.51 \
    --seed 42

echo ""
echo "=========================================="
echo "  [MS] MP-STRUCT"
echo "=========================================="
python -m ppt.minimalist_grammar_v3 generate_mg_sentences \
    --file_dir "${PPT_DIR}/mp_struct" \
    --n 100000 \
    --max_length 1024 \
    --strip_lexical True \
    --p_c_wh 0.2 \
    --p_wh_goal 0.2

echo ""
echo "=========================================="
echo "  [BCC] MP-STRUCT BCC (profile_prob=1.0, strip_lexical)"
echo "=========================================="
python -m ppt.mp_struct_universal \
    --num_sentences 100000 \
    --max_length 1024 \
    --output_dir "${PPT_DIR}/mp_struct_universal" \
    --output_file "bcc_ids.txt" \
    --profile_prob 1.0 \
    --strip_lexical \
    --seed 42

echo ""
echo "All PPT data generated in ${PPT_DIR}/"
echo "  SD:  ${PPT_DIR}/shuff_dyck/shuff_dyck_k64_100000_1024.txt"
echo "  MS:  ${PPT_DIR}/mp_struct/mg_ids_100000_1024.txt"
echo "  BCC: ${PPT_DIR}/mp_struct_universal/bcc_ids.txt"
