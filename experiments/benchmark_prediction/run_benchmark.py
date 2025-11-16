from dataclasses import dataclass, field
from dotenv import load_dotenv
from vip import GPR
import fire
import os

@dataclass 
class Config:
    embedding_model: str = field(default="all-MiniLM-L6-v2")
    dataset: str = field(default="sbert_sts")

def main(**kwargs):
    config = Config(**kwargs)
    print(f"Running benchmark with config: {config}")


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(main) 