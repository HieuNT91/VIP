# python prepare_embedding.py \
#     --embedder Qwen/Qwen3-Embedding-0.6B \
#     --dataset_file_path /home/hieunt/verl/data/dapo-math-17k.parquet \
#     --saved_path /home/hieunt/verl/data/dapo-math-17k-embeddings \
#     --num_samples 17000 \
#     --chunk_size 512
    

# [--saved_path data] \
#     [--num_samples 100] [--pairwise] [--sigma 1.0] [--chunk_size 2048]

python3 gaussian_allocation/prepare_embedding.py \
        --embedder all-MiniLM-L6-v2 \
        --dataset_file_path /home/hieunt/verl/data/fixprompt-dapo-math-17k.parquet \
        --num_samples -1 \
        --pairwise \