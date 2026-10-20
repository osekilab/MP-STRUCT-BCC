"""k-Shuffle Dyck formal language generator for PPT.

Generates k-shuffle Dyck sequences following Hu et al. (2025).
Unlike standard (nested) Dyck languages, k-shuffle Dyck allows
cross-serial dependencies: any currently-open bracket type can be
closed at any step, not just the most recently opened one.

Symbol encoding:
  0 .. k-1     : opening brackets for types 0..k-1
  k .. 2k-1    : closing brackets (type i closes with symbol i+k)

Output format matches the existing PPT pipeline: one sequence per
line, space-separated integer IDs.

Reference:
  Hu, Huang & Linzen (2025). Between Circuits and Chomsky:
  Pre-pretraining Formal Language Encodings of Syntactic Structures
  Facilitate Syntactic Generalization. ACL 2025.
"""

from __future__ import annotations

import os
import random
from typing import List, Optional
from tqdm import trange
import fire


def generate_shuff_dyck(
    k: int,
    max_length: int = 1024,
    p_open: float = 0.51,
    min_depth: int = 1,
    max_depth: int = 8,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Generate a single k-shuffle Dyck sequence.

    At each step the generator either opens a random bracket type
    (with probability p_open) or closes a uniformly-chosen currently-open
    bracket type.  Depth bounds are enforced: depth 0 forces an open,
    depth >= max_depth forces a close.

    The "shuffle" property means closing does NOT respect stack order --
    any open bracket may be closed, producing cross-serial dependencies.
    """
    if rng is None:
        rng = random.Random()

    counts = [0] * k
    seq = []
    depth = 0

    for _ in range(max_length):
        if depth == 0:
            do_open = True
        elif depth >= max_depth:
            do_open = False
        else:
            do_open = rng.random() < p_open

        if do_open:
            bracket_type = rng.randrange(k)
            seq.append(bracket_type)
            counts[bracket_type] += 1
            depth += 1
        else:
            open_types = [i for i in range(k) if counts[i] > 0]
            bracket_type = rng.choice(open_types)
            seq.append(bracket_type + k)
            counts[bracket_type] -= 1
            depth -= 1

    return seq


def generate(
    file_dir: str,
    k: int = 64,
    n: int = 100000,
    max_length: int = 1024,
    p_open: float = 0.51,
    min_depth: int = 1,
    max_depth: int = 8,
    seed: int = 42,
):
    """Generate k-shuffle Dyck sequences and write to disk.

    Args:
        file_dir:    Output directory.
        k:           Number of distinct bracket types (default 64).
        n:           Number of sequences to generate (default 100000).
        max_length:  Tokens per sequence (default 1024).
        p_open:      Probability of opening a bracket (default 0.51).
        min_depth:   Minimum depth before allowing close (default 1).
        max_depth:   Maximum nesting depth (default 8).
        seed:        Random seed for reproducibility.

    Writes:
        <file_dir>/shuff_dyck_k<k>_<n>_<max_length>.txt
    """
    rng = random.Random(seed)
    os.makedirs(file_dir, exist_ok=True)

    out_path = os.path.join(file_dir, f"shuff_dyck_k{k}_{n}_{max_length}.txt")

    with open(out_path, "w") as f:
        for _ in trange(n, desc=f"k-shuffle Dyck (k={k}, len={max_length})"):
            seq = generate_shuff_dyck(
                k=k,
                max_length=max_length,
                p_open=p_open,
                min_depth=min_depth,
                max_depth=max_depth,
                rng=rng,
            )
            f.write(" ".join(map(str, seq)) + "\n")

    print(f"Wrote {n} sequences to {out_path}")
    print(f"  k={k}, max_length={max_length}, p_open={p_open}, seed={seed}")
    print(f"  Vocabulary size: {2 * k} (symbols 0..{2*k-1})")


if __name__ == "__main__":
    fire.Fire({"generate": generate})
