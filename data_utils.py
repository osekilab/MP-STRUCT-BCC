# File: data_utils.py
# -------------------
# Function for dataset loading, construction and saving + collation functions

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

import math
import random
import os
from pathlib import Path
from tqdm import tqdm
import pickle

import pandas as pd


def load_texts_from_dir(data_dir: Path) -> list[tuple[str, str]]:
    """Load all training texts from a dataset directory.
    Returns list of (source_name, text) pairs.
    Handles both .txt and .parquet files."""
    texts = []

    for f in sorted(data_dir.glob('*.train.txt')):
        source = f.stem.replace('.train', '')
        text = f.read_text()
        texts.append((source, text))

    for f in sorted(data_dir.glob('*.train.parquet')):
        source = f.stem.replace('.train', '')
        df = pd.read_parquet(f)
        text = '\n'.join(df['text'].tolist())
        texts.append((source, text))

    return texts


def _resolve_tokenizer_path(cfg):
    """Resolve tokenizer directory to an absolute path."""
    tokenizer_dir = cfg.get('tokenizer_dir', 'tokenizers')
    return str(Path(tokenizer_dir, cfg['dataset']).resolve())


class FullBabyLMDataset(Dataset):

    def __init__(self, cfg):
        dataset_name = cfg['dataset']

        # Load the tokenizer
        tokenizer_path = _resolve_tokenizer_path(cfg)
        self.processor = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model_bos = self.processor.bos_token_id
        self.model_eos = self.processor.eos_token_id

        # Load and tokenize each source file
        self.data = []
        data_dir = Path(cfg.get('data_dir', 'data')) / dataset_name
        texts = load_texts_from_dir(data_dir)

        for source_name, all_text in texts:
            print(f'Opened {source_name} ({len(all_text):,} chars)')

            # Process full text into tokens (no special tokens; bos/eos added in __getitem__)
            tokenized_dataset = self.processor(text=[all_text], add_special_tokens=False)['input_ids'][0]
            print(f'Tokenized {source_name}; {len(tokenized_dataset):,} tokens total')

            # Chunk and add (reserve 2 tokens for bos/eos)
            chunk_size = cfg["datapoint_length"] - 2
            num_chunks = math.ceil(len(tokenized_dataset) / chunk_size)
            for curr_chunk in tqdm(range(num_chunks)):
                start = curr_chunk * chunk_size
                end = (curr_chunk+1) * chunk_size
                chunk_tokens = tokenized_dataset[start:end]
                self.data.append(chunk_tokens)
            print(f"Chunked {source_name}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor([self.model_bos] + self.data[idx] + [self.model_eos])

class PPTDataset(Dataset):
    """Dataset for Pre-pretraining on formal language data (MP-STRUCT / MP-STRUCT CORE).
    Reads a text file where each line is space-separated integers or tokens,
    then tokenizes with the PT tokenizer so embeddings are shared across phases."""

    def __init__(self, ppt_data_path, tokenizer_path, datapoint_length=512):
        processor = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model_bos = processor.bos_token_id
        self.model_eos = processor.eos_token_id

        self.total_word_count = 0
        self.data = []
        with open(ppt_data_path, 'r') as f:
            for line in tqdm(f, desc="Loading PPT data"):
                line = line.strip()
                if not line:
                    continue
                self.total_word_count += len(line.split())
                tokens = processor(text=line, add_special_tokens=False)['input_ids']
                chunk_size = datapoint_length - 2
                for i in range(0, len(tokens), chunk_size):
                    chunk = tokens[i:i+chunk_size]
                    if chunk:
                        self.data.append(chunk)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor([self.model_bos] + self.data[idx] + [self.model_eos])


def load_ppt_data(cfg):
    """Load PPT data and return a DataLoader."""
    tokenizer_path = _resolve_tokenizer_path(cfg)
    dataset = PPTDataset(
        cfg['ppt_data_path'], tokenizer_path, cfg['datapoint_length']
    )
    collate_fn = get_collate_fn(dataset.model_eos)
    return DataLoader(dataset, batch_size=cfg['batch_size'],
                      shuffle=True, collate_fn=collate_fn)


## General utilities ##
def load_babylm_data(cfg):
    dataset_name = cfg['dataset']
    data_dir = Path(cfg.get('data_dir', 'data'))
    cache_dir = data_dir / 'cached_train'
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = cache_dir / f'train_gpt2_{dataset_name}.pkl'

    if filename.exists():
        with open(filename, 'rb') as f:
            full_babylm_dset = pickle.load(f)
    else:
        full_babylm_dset = FullBabyLMDataset(cfg)
        with open(filename, 'wb') as f:
            pickle.dump(full_babylm_dset, f)

    collate_fn = get_collate_fn(full_babylm_dset.model_eos)
    dataloader = DataLoader(full_babylm_dset, batch_size=cfg["batch_size"],
                            shuffle=True, collate_fn=collate_fn)
    return dataloader

def get_collate_fn(model_eos):
    def collate_fn(batch):
        tokens = pad_sequence([item for item in batch], padding_value=model_eos, batch_first=True)
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        target_mask = input_tokens != model_eos
        target_mask[:, 0] = 1

        return input_tokens, target_tokens, target_mask
    return collate_fn
    
