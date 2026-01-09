import json 
from dataclasses import dataclass, field
from dotenv import load_dotenv
from vip import GPR
import fire
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
from vip.utils import profile
from vip.success_rate_prediction.gaussian_process import GPR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
import numpy as np 
from sklearn.linear_model import LinearRegression

@dataclass 
class Config:
    embedding_model: str = field(default="all-MiniLM-L6-v2")
    dataset: str = field(default="data/vip-dapo-math-6k.parquet")
    accuracy_series_data_path: str = field(default="logs/file_logs/rloo-Qwen2.5-Math-7B-rolloutn4-seed1_rollout_data.jsonl")
    embedding_cache_dir: str = field(default="tmp/")
    batch_size: int = field(default=32)
    prior_value: float = field(default=-5.0)
    reuse_mean: bool = field(default=True)
    reuse_covariance: bool = field(default=False)
    return_std:  bool = field(default=True)
    window_size: int = field(default=1)
    end_step: int = field(default=60)
    
    
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
    
    gpr = GPR(
        pairwise_dists, 
        return_std=config.return_std,
        reuse_covariance=config.reuse_covariance,
        reuse_mean=config.reuse_mean,
        qid_to_idx=qid_to_idx,
        prior_value=config.prior_value,
    )

    steps = sorted(accuracy_series_data.keys())
    
    errors = []
    mean_baseline_errors = []
    linear_regression_errors = []
    linreg = LinearRegression() 
    END = config.end_step
    for step in steps[config.window_size:END]: # use data of previous step to train gpr, then predict current step
        
        training_steps = list(range(max(0, step - config.window_size), step))
        train_data = gather_data_by_steps(accuracy_series_data, training_steps)
        test_data  = accuracy_series_data[step]
        train_question_uids = train_data["index"]
        train_accuracies    = train_data["accuracy"]
        test_question_uids  = test_data["index"]
        test_accuracies     = test_data["accuracy"]
        
        # print(f"\n=== Step {step} ===")
        # print(f"Training on {len(train_question_uids)} questions, testing on {len(test_question_uids)} questions")  
        # print(f"len train_accuracies: {len(train_accuracies)}")
        # print(f"len test_accuracies: {len(test_accuracies)}")
        gpr.fit_qids(train_question_uids, train_accuracies)
        preds, stds = gpr.predict_qids(test_question_uids)
        
        mean_baseline_pred = np.mean(train_accuracies)
        mean_baseline_preds = [mean_baseline_pred] * len(test_accuracies)
        mean_baseline_mae = mean_absolute_error(test_accuracies, mean_baseline_preds)
        mean_baseline_errors.append(mean_baseline_mae)
        
        X_train = torch.stack([embeddings[uid] for uid in train_question_uids]).numpy()
        y_train = np.array(train_accuracies)
        linreg.fit(X_train, y_train)
        X_test = torch.stack([embeddings[uid] for uid in test_question_uids]).numpy()
        linear_regression_preds = linreg.predict(X_test)
        linear_regression_mae = mean_absolute_error(test_accuracies, linear_regression_preds)
        linear_regression_errors.append(linear_regression_mae)
        
        mae = mean_absolute_error(test_accuracies, preds)
        errors.append(mae)
        print(preds[:10])
        print(f"Step {step} MAE: {mae:.4f} | Mean Baseline MAE: {mean_baseline_mae:.4f} | Linear Regression MAE: {linear_regression_mae:.4f}")

    # plot overall error trend time 
    plt.figure(figsize=(8,6))
    plt.plot(steps[config.window_size:END], errors, marker='o', label="VIP (Gaussian Process)")
    plt.plot(steps[config.window_size:END], mean_baseline_errors, marker='x', label="Moving Average")
    plt.plot(steps[config.window_size:END], linear_regression_errors, marker='^', label="Linear Regression")

    plt.title("Prompt success rate prediction error", fontsize=18)
    plt.xlabel("Gradient step", fontsize=18)
    plt.ylabel("Mean Absolute Error", fontsize=18)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(fontsize=14, loc='upper left') 

    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tmp/gpr_prediction_mae.pdf")

            
            
            
    
    
        
    
        
        


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(main) 
    