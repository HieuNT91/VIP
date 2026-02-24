#!/usr/bin/env python3
"""
Prepare question embeddings (and optional pairwise Gaussian kernel matrix) from a parquet dataset.

Usage:
  python prepare_embedding.py \
    --embedder Qwen/Qwen3-Embedding-0.6B \
    --dataset_file_path /path/to/dataset.parquet \
    [--saved_path data] \
    [--num_samples 100] [--pairwise] [--sigma 1.0] [--chunk_size 2048]

Outputs (naming convention):
  - Embeddings: embeddings_{embedder}_{dataset}_{num}.npy
  - Indices:    indices_{embedder}_{dataset}_{num}.json
    - Pairwise:   pairwise_{embedder}_{dataset}_{num}_matrix.npy (when --pairwise)

Notes:
  - Pairwise computation is O(N^2) in time and storage; uses chunked memmap writing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import List


import numpy as np
import torch
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

try:
    import datasets  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("Please install 'datasets' package to load parquet datasets.") from e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("Please install 'sentence-transformers' to compute embeddings.") from e


# DEFAULT_PROMPT_TEMPLATE = (
#     "Solve the following math problem step by step. The last line of your response should be "
#     "of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
#     "{{question}}\n\nRemember to put your answer on its own line after \"Answer:\"."
# )

DEFAULT_PROMPT_TEMPLATE = r"""Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem. Do not wrap $Answer with \boxed{}.

current question: {{question}}

Below are two examples for format reference.
Example question 1: Solve for x: 3x - 5 = 16.

Response:
Add 5 to both sides: 3x = 21.
Divide both sides by 3: x = 7.
Answer: 7

Example question 2: A jacket costs $80 and is on sale for 25% off. What is the sale price?

Response:
25% of 80 is 0.25 × 80 = 20.
Subtract the discount from the original price: 80 − 20 = 60.
Answer: 60

Solve the current question. Remember to put your answer on its own line after "Answer:".
"""

def sanitize_name(name: str) -> str:
    """Sanitize a string for safe filenames: replace slashes/spaces, keep [A-Za-z0-9_.-]."""
    name = name.replace("/", "-").replace(" ", "-")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name)


def extract_question_from_prompt(prompt_content: str, template: str) -> str:
    """Extract the question text from a prompt content using the known template.

    This heuristic assumes the template embeds the question between double newlines.
    Falls back to extracting the penultimate block between two double newlines.
    """
    split_token = "\n\n"
    if split_token in template:
        parts = template.split(split_token)
        if "{{question}}" in parts[1]:
            prefix = split_token.join(parts[:1]) + split_token
            question = prompt_content.replace(prefix, "").strip()
            if split_token in question:
                question = question.split(split_token)[0].strip()
            return question
    # Fallback: take the second last block
    blocks = re.split(r"\n\n", prompt_content)
    if len(blocks) > 1:
        return blocks[1].strip()
    return prompt_content.strip()


def load_dataset_questions(
    parquet_path: str, template: str, num_samples: int | None
) -> tuple[List[str], List[int]]:
    """Load prompts/questions and their indices from parquet via HF datasets.

    Returns:
        questions: list[str] extracted questions
        indices: list[int] prompt indices aligned with questions
    """
    ds = datasets.load_dataset("parquet", data_files=[parquet_path])["train"]

    total = len(ds)
    if num_samples is None or num_samples < 0 or num_samples > total:
        num_samples = total

    from concurrent.futures import ThreadPoolExecutor

    def extract_one(i: int):
        prompt_list = ds["prompt"][i]
        content = prompt_list[0]["content"] if prompt_list else ""
        idx = ds["extra_info"][i]["index"]
        q = extract_question_from_prompt(content, template)
        return q, idx

    questions: List[str] = [None] * num_samples  # type: ignore
    indices: List[int] = [None] * num_samples  # type: ignore
    with ThreadPoolExecutor(max_workers=64) as executor:
        results = list(tqdm(executor.map(extract_one, range(num_samples)), total=num_samples, desc="Extracting questions (parallel)"))
    for i, (q, idx) in enumerate(results):
        questions[i] = q.replace("current question: ","")
        indices[i] = idx

    return questions, indices


def compute_embeddings(embedder_name: str, questions: List[str]) -> np.ndarray:
    model = SentenceTransformer(embedder_name)
    batch_size = 8  # You can adjust this based on your GPU/CPU memory
    all_embs = []
    try:
        from tqdm import tqdm
        iterator = tqdm(range(0, len(questions), batch_size), desc="Encoding batches")
    except ImportError:
        iterator = range(0, len(questions), batch_size)

    for i in iterator:
        batch = questions[i:i+batch_size]
        try:
            emb = model.encode(batch, prompt_name="query", show_progress_bar=False)
        except Exception:
            emb = model.encode(batch, prompt_name="query")
        all_embs.append(np.asarray(emb, dtype=np.float32))
        del emb
        torch.cuda.empty_cache()
        
    return np.concatenate(all_embs, axis=0)


def compute_pairwise_gaussian_memmap(
    embeddings: np.ndarray,
    sigma: float,
    out_npy_path: str,
    chunk_size: int = 2048,
) -> str:
    """Compute and cache Euclidean distance matrix using chunked memmap writing.

    D[i,j] = ||x_i - x_j||_2
    """
    from numpy.lib.format import open_memmap

    X = embeddings.astype(np.float32, copy=False)
    n = X.shape[0]
    # Create a .npy memmap file for distances
    D = open_memmap(out_npy_path, mode="w+", dtype=np.float32, shape=(n, n))
    r = np.sum(X * X, axis=1, dtype=np.float32)

    outer = tqdm(range(0, n, chunk_size), desc="Pairwise rows", leave=True)
    for i0 in outer:
        i1 = min(i0 + chunk_size, n)
        A = X[i0:i1]
        Ai = r[i0:i1][:, None]
        inner = tqdm(range(i0, n, chunk_size), desc=f"Cols for rows {i0}-{i1}", leave=False)
        for j0 in inner:
            j1 = min(j0 + chunk_size, n)
            B = X[j0:j1]
            Bj = r[j0:j1][None, :]
            # dist^2 matrix
            dist_sq = Ai + Bj - 2.0 * (A @ B.T)
            dist_sq = np.maximum(dist_sq, 0.0)  # numerical safety
            dist = np.sqrt(dist_sq, dtype=np.float32)
            D[i0:i1, j0:j1] = dist
            if j0 != i0:
                D[j0:j1, i0:i1] = dist.T  # reflect to lower triangle

    # Ensure data is written
    D.flush()
    return out_npy_path


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare embeddings and optional pairwise matrix.")
    p.add_argument("--embedder", required=True, help="Sentence-Transformer repo name, e.g. Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--dataset_file_path", required=True, help="Path to parquet dataset file")
    p.add_argument("--saved_path", default="data/embedding_data", help="Directory to save outputs (default: data)")
    p.add_argument("--num_samples", type=int, default=-1, help="Limit number of samples (default: all)")
    p.add_argument("--pairwise", action="store_true", help="Also compute/save pairwise Gaussian kernel matrix")
    p.add_argument("--sigma", type=float, default=1.0, help="Sigma for Gaussian kernel (default: 1.0)")
    p.add_argument("--chunk_size", type=int, default=216, help="Chunk size for pairwise computation")
    p.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE, help="Prompt template containing {{question}} placeholder")

    args = p.parse_args()

    embedder_name = args.embedder
    parquet_path = args.dataset_file_path
    save_dir = Path(args.saved_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = Path(parquet_path).stem
    num_samples = None if args.num_samples is None else int(args.num_samples)

    # Load and extract questions/indices
    questions, indices = load_dataset_questions(parquet_path, args.prompt_template, num_samples)
    n = len(questions)
    # Compute embeddings
    embeddings = compute_embeddings(embedder_name, questions)

    # Build filenames with convention
    emb_name = f"embeddings_{sanitize_name(embedder_name)}_{sanitize_name(dataset_name)}_{n}.npy"
    idx_name = f"indices_{sanitize_name(embedder_name)}_{sanitize_name(dataset_name)}_{n}.json"
    emb_path = str(save_dir / emb_name)
    idx_path = str(save_dir / idx_name)

    # Save embeddings and indices
    np.save(emb_path, embeddings)
    with open(idx_path, "w") as f:
        json.dump(indices, f, indent=2)

    print(f"Saved embeddings to {emb_path}")
    print(f"Saved indices to    {idx_path}")

    # Optional pairwise
    if args.pairwise:
        pair_name = f"pairwise_{sanitize_name(embedder_name)}_{sanitize_name(dataset_name)}_{n}_matrix.npy"
        pair_path = str(save_dir / pair_name)
        print("Computing pairwise Gaussian kernel matrix ...")
        compute_pairwise_gaussian_memmap(embeddings, args.sigma, pair_path, chunk_size=args.chunk_size)
        print(f"Saved pairwise matrix to {pair_path}")


if __name__ == "__main__":
    main()