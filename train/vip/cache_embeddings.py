from dataclasses import dataclass, field
from dotenv import load_dotenv
import fire
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
from vip.utils import profile

@dataclass 
class Config:
    embedding_model: str = field(default="all-MiniLM-L6-v2")
    dataset: str = field(default="data/vip-dapo-math-17k.parquet")
    embedding_cache_dir: str = field(default="tmp/")
    batch_size: int = field(default=32)

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

def main(**kwargs):
    config = Config(**kwargs)
    print(f"Running benchmark with config: {config}")
    
    dataset = load_dataset("parquet", data_files=[config.dataset])["train"]
    questions = [item['question'] for item in dataset]
    question_uids = [item['extra_info']['index'] for item in dataset]
    print(questions[0])
    
    os.makedirs(config.embedding_cache_dir, exist_ok=True)
    embedding_cache_path = os.path.join(
        config.embedding_cache_dir,
        f"embeddings_{os.path.basename(config.dataset).replace('.parquet','')}_{clean_name(config.embedding_model)}.pt"
    )
    pairwise_cache_path = embedding_cache_path.replace("embeddings_", "pairwise_distances_")
    
    if os.path.exists(embedding_cache_path):
        print(f"Loading cached embeddings from {embedding_cache_path}")
        embeddings = torch.load(embedding_cache_path)
    else:
        print(f"Computing embeddings using model {config.embedding_model}")
        embeddings = compute_embeddings(
            model_name_or_path=config.embedding_model,
            questions=questions,
            question_uids=question_uids,
            embedding_cache_path=embedding_cache_path,
            batch_size=config.batch_size
        )
    assert len(embeddings) == len(questions), "Cached embeddings size mismatch"
    
    if os.path.exists(pairwise_cache_path):
        print(f"Loading cached pairwise distances from {pairwise_cache_path}")
        pairwise_dists = torch.load(pairwise_cache_path)
    else:
        print("Computing pairwise distances")
        pairwise_dists = compute_pairwise_distances(
            embeddings=embeddings,
            pairwise_cache_path=pairwise_cache_path
        )
    assert pairwise_dists.size(0) == len(questions), "Cached pairwise distances size mismatch"
    pairwise_dists = pairwise_dists.numpy()
    

if __name__ == "__main__":
    load_dotenv()
    fire.Fire(main) 
    