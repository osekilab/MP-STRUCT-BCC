"""Convert a training checkpoint to HuggingFace format for evaluation.

Usage:
    python scripts/convert_to_hf.py \
        --experiment_dir experiments/bcc_seed42 \
        --dataset en_nld_zho_equal \
        --output_dir hf_models/bcc_seed42
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer


def convert(experiment_dir, dataset, output_dir, config_dir="configs", tokenizer_dir="tokenizers", epoch="9"):
    checkpoint_dir = Path(experiment_dir) / "checkpoints" / f"epoch_{epoch}"
    weight_path = checkpoint_dir / "latest_student.pt"

    if not weight_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}")

    config_path = Path(config_dir) / dataset
    config = GPT2Config.from_pretrained(str(config_path))
    model = GPT2LMHeadModel(config)
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(Path(tokenizer_dir) / dataset))
    tokenizer.save_pretrained(output_dir)

    print(f"Saved HF model to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--dataset", default="en_nld_zho_equal")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--tokenizer_dir", default="tokenizers")
    parser.add_argument("--epoch", default="9", help="Epoch checkpoint to convert (default: 9, i.e., final)")
    args = parser.parse_args()

    convert(args.experiment_dir, args.dataset, args.output_dir,
            args.config_dir, args.tokenizer_dir, args.epoch)
