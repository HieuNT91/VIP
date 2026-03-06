import json 
from dataclasses import dataclass, field
from dotenv import load_dotenv
import fire
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
from vip.utils import profile
from vip import allocate_rollout
from matplotlib import pyplot as plt
import numpy as np 


@dataclass 
class Config:
    embedding_model: str = field(default="all-MiniLM-L6-v2")
    dataset: str = field(default="data/vip-dapo-math-6k.parquet")
    accuracy_series_data_path: str = field(default="logs/file_logs/rloo-Qwen2.5-Math-7B-rolloutn4-seed1_rollout_data.jsonl")
    
def clean_name(name: str):
    clean_name = name.split("/")[-1]
    clean_name = clean_name.replace("-", "_")
    return clean_name 

def atomic_save(data, path): # avoid corrupted files
    temp_path = path + ".tmp"
    torch.save(data, temp_path)
    os.replace(temp_path, path)
    
@profile
def compute_embeddings(model_name_or_path: str,
                       questions: list[str],
                       question_uids: list[str],
                       embedding_cache_path: str,
                       batch_size: int = 32) -> dict[str, torch.Tensor]:
    assert len(questions) == len(question_uids), "Questions and UIDs must have the same length"
    model = SentenceTransformer(model_name_or_path).to("cuda")
    embeddings = {}
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i+batch_size]
        batch_uids      = question_uids[i:i+batch_size]

        with torch.no_grad():
            emb = model.encode(batch_questions, convert_to_tensor=True)
            emb = emb.to("cpu")

        for uid, vec in zip(batch_uids, emb):
            embeddings[uid] = vec
        
        batch_idx = i // batch_size 
        if (batch_idx + 1) % 10 == 0 or (i + batch_size) >= len(questions):
            print(f"Processed {min(i + batch_size, len(questions))}/{len(questions)} questions")
            torch.cuda.empty_cache()
            atomic_save(embeddings, embedding_cache_path)
    return embeddings

@profile
def compute_pairwise_distances(embeddings: dict[str, torch.Tensor], 
                               pairwise_cache_path: str) -> torch.Tensor:
    uids = list(embeddings.keys())
    embs = torch.stack([embeddings[uid] for uid in uids], dim=0)  # (N, D)
    dists = torch.cdist(embs, embs, p=2)  # (N, N)
    atomic_save(dists, pairwise_cache_path)
    return dists

@profile
def compute_rollout_allocation(accuracies):
    allocation = allocate_rollout(accuracies, batch_budget=512*2, lower =1, upper=32)
    return allocation

def gather_data_by_steps(accuracy_series_data: dict, steps: list[int]):
    gathered_data = {"index": [], "accuracy": []}
    for step in steps:
        data = accuracy_series_data[step]
        gathered_data["index"].extend(data["index"])
        gathered_data["accuracy"].extend(data["accuracy"])
    return gathered_data

def main(**kwargs):
    config = Config(**kwargs)
    print(f"Running benchmark with config: {config}")
    
    dataset = load_dataset("parquet", data_files=[config.dataset])["train"]
    questions = [item['question'] for item in dataset]
    question_uids = [item['extra_info']['index'] for item in dataset]
    
    
    accuracy_series_data = {}
    with open(config.accuracy_series_data_path, "r") as f:
        for line in f.readlines():
            entry = json.loads(line)
            step = entry['step']
            data = entry['data']
            if step not in accuracy_series_data:
                accuracy_series_data[step] = {
                    "index": [],
                    "accuracy": [],
                }
            accuracy_series_data[step]["index"].append(data['index'])
            accuracy_series_data[step]["accuracy"].append(data['accuracy'])
    
    qid_to_idx = {uid: idx for idx, uid in enumerate(question_uids)}
    
    steps = sorted(accuracy_series_data.keys())
    
    for step in steps[20:]: # use data of previous step to train gpr, then predict current step
        test_data  = accuracy_series_data[step]
        test_question_uids = test_data["index"]
        test_accuracies    = test_data["accuracy"]
        allocation = compute_rollout_allocation(test_accuracies)
        print(allocation)
        print()
    
    


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(main) 
    