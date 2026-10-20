# MP-STRUCT BCC

Code and scripts for reproducing the main experiments in:

> **Linguistically Inspired Multilingual Pre-Pretraining**
> Masato Mita, Max Müller-Eberstein, Misaki Yamaguchi, Yohei Oseki.
> BabyLM Workshop @ EMNLP2026

## Overview

This repository provides the minimal code to reproduce the four experimental conditions reported in the paper:

| Condition | PPT Data | Description |
|-----------|----------|-------------|
| **BL** (Baseline) | — | Standard pretraining, no PPT |
| **SD** (k-Shuffle Dyck) | `k_shuffle_dyck.py` | Hierarchical structure via cross-serial dependencies (k=64; Hu et al., ACL2025) |
| **MS** (MP-STRUCT) | `minimalist_grammar_v3.py` | Fixed English-like typological parameters (Mita et al.,ACL2026 )|
| **BCC** (MP-STRUCT BCC) | `mp_struct_universal.py` | Sampled typological profiles following the Borer–Chomsky Conjecture |

All conditions use GPT-2 small (124M parameters), 500 PPT steps, and 100M words × 10 epochs of trilingual (English/Dutch/Chinese) pretraining data from BabyLM 2026.

## Setup

### Requirements

- Python 3.10+
- CUDA-compatible GPU
- ~50 GB disk space for training data

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Step 1: Download training data

Download the BabyLM 2026 datasets:

```bash
bash scripts/download_data.sh
```

This downloads:
- `BabyLM-2026-Strict` (English)
- `babylm-nld` (Dutch)
- `babylm-zho` (Chinese)

### Step 2: Build trilingual dataset

Construct the balanced trilingual dataset (`en_nld_zho_equal`):

```bash
python build_multilingual_dataset.py \
    --languages en nld zho \
    --words-per-lang 33333333 \
    --output data/en_nld_zho_equal
```

### Step 3: Train tokenizer

Train a BPE tokenizer on the trilingual data:

```bash
python train_tokenizer.py en_nld_zho_equal
```

This creates `tokenizers/en_nld_zho_equal/` and `configs/en_nld_zho_equal/`.

### Step 4: Generate PPT data

Generate formal language data for the three PPT conditions:

```bash
bash scripts/generate_ppt_data.sh
```

## Training

### Train all 4 conditions (single seed)

```bash
bash scripts/train.sh 42 0   # seed=42, GPU=0
```

### Train individual conditions

**Baseline (BL):**
```bash
python training.py \
    --dataset en_nld_zho_equal \
    --words_per_epoch 100000000 \
    --experiment_name baseline_seed42 \
    --seed 42 \
    --no_deduct_ppt
```

**k-Shuffle Dyck (SD):**
```bash
python training.py \
    --dataset en_nld_zho_equal \
    --ppt_data_path data/ppt/shuff_dyck/shuff_dyck_k64_100000_1024.txt \
    --ppt_steps 500 --ppt_learning_rate 0.0005 --ppt_warmup_steps 100 \
    --words_per_epoch 100000000 \
    --experiment_name shuff_dyck_seed42 \
    --seed 42 --no_deduct_ppt
```

**MP-STRUCT (MS):**
```bash
python training.py \
    --dataset en_nld_zho_equal \
    --ppt_data_path data/ppt/mp_struct/mg_ids_100000_1024.txt \
    --ppt_steps 500 --ppt_learning_rate 0.0005 --ppt_warmup_steps 100 \
    --words_per_epoch 100000000 \
    --experiment_name mp_struct_seed42 \
    --seed 42 --no_deduct_ppt
```

**MP-STRUCT BCC:**
```bash
python training.py \
    --dataset en_nld_zho_equal \
    --ppt_data_path data/ppt/mp_struct_universal/bcc_ids.txt \
    --ppt_steps 500 --ppt_learning_rate 0.0005 --ppt_warmup_steps 100 \
    --words_per_epoch 100000000 \
    --experiment_name bcc_seed42 \
    --seed 42 --no_deduct_ppt
```

### Multi-seed replication

The paper reports results averaged over seeds 0, 42, and 126:

```bash
for seed in 0 42 126; do
    bash scripts/train.sh $seed $GPU_ID
done
```

## Evaluation

### Convert checkpoints to HuggingFace format

```bash
python scripts/convert_to_hf.py \
    --experiment_dir experiments/bcc_seed42 \
    --dataset en_nld_zho_equal \
    --output_dir hf_models/bcc_seed42
```

### Run BLiMP evaluation

Evaluate on BLiMP (English), BLiMP-NL (Dutch), and ZhoBLiMP (Chinese) using [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
bash scripts/evaluate.sh hf_models/bcc_seed42 0
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Model | GPT-2 small (124M) |
| Sequence length | 512 tokens |
| PPT steps | 500 |
| PPT learning rate | 5e-4 |
| PPT warmup steps | 100 |
| PT learning rate | 5e-5 |
| PT epochs | 10 |
| PT words/epoch | 100M |
| Batch size | 16 |
| Warmup ratio | 1% |
| Gradient clipping | 1.0 |
| Vocabulary size | 16,384 |

## Repository Structure

```
MP-STRUCT-BCC/
├── training.py                  # Main training loop (PPT + PT)
├── models.py                    # GPT-2 model initialization
├── data_utils.py                # BabyLM and PPT data loading
├── utils.py                     # Configuration and experiment setup
├── config.yaml                  # Default hyperparameters
├── build_multilingual_dataset.py # Trilingual dataset construction
├── train_tokenizer.py           # BPE tokenizer training
├── extract_results.py           # Parse lm-eval JSON results to CSV
├── requirements.txt
├── ppt/
│   ├── k_shuffle_dyck.py        # SD: k-Shuffle Dyck generator
│   ├── minimalist_grammar_v3.py # MS: MP-STRUCT generator
│   └── mp_struct_universal.py   # BCC: MP-STRUCT BCC generator
└── scripts/
    ├── download_data.sh         # Download BabyLM datasets
    ├── generate_ppt_data.sh     # Generate PPT formal language data
    ├── train.sh                 # Train all 4 conditions
    ├── convert_to_hf.py         # Convert checkpoints to HF format
    └── evaluate.sh              # BLiMP evaluation
```




