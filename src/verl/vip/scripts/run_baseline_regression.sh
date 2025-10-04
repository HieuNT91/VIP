EMBEDDER="Qwen-Qwen3-Embedding-0.6B"
WINDOW_SIZE=1
DATASET="fixprompt-dapo-math-17k_17398"

# Run Ridge Regression
# python gaussian_allocation/cores/model_baseline.py \
#     --model_type ridge \
#     --alpha 1.0 \
#     --window_size $WINDOW_SIZE \
#     --embedder $EMBEDDER \
#     --dataset $DATASET

# Run Lasso Regression
python gaussian_allocation/cores/model_baseline.py \
    --model_type lasso \
    --alpha 0.0001 \
    --window_size $WINDOW_SIZE \
    --embedder $EMBEDDER \
    --dataset $DATASET

# Run Logistic Regression
# python gaussian_allocation/cores/model_baseline.py \
#     --model_type logistic \
#     --alpha 0.5 \
#     --window_size $WINDOW_SIZE \
#     --embedder $EMBEDDER \
#     --dataset $DATASET