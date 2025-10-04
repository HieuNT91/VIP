#!/usr/bin/env bash
set -euo pipefail

EMBEDDER="all-MiniLM-L6-v2"
DATASET="fixprompt-dapo-math-17k_17398"
WINDOW_SIZE=1
PRIOR_VALUE=-1

python gaussian_allocation/cores/model_v2.py \
  --embedder "$EMBEDDER" \
  --dataset "$DATASET" \
  --regression_data allo_grpo_4e \
  --window_size "$WINDOW_SIZE" \
  --prior_value "$PRIOR_VALUE" \
  --target_key mean_acc_per_epoch \
  --start_step 1 \
  --end_step 66 \
  --step_size 1 \
  --output_dir /home/hieunt/verl/results/gbr

# python gaussian_allocation/cores/model_v2.py \
#   --embedder "$EMBEDDER" \
#   --dataset "$DATASET" \
#   --regression_data allo_grpo_4e \
#   --window_size "$WINDOW_SIZE" \
#   --reuse_mean \
#   --prior_value "$PRIOR_VALUE" \
#   --target_key mean_acc_per_epoch \
#   --start_step 1 \
#   --end_step 66 \
#   --step_size 1 \
#   --output_dir /home/hieunt/verl/results/gbr

# python gaussian_allocation/cores/model_v2.py \
#   --embedder "$EMBEDDER" \
#   --dataset "$DATASET" \
#   --regression_data allo_grpo_4e \
#   --window_size "$WINDOW_SIZE" \
#   --reuse_covariance \
#   --reuse_mean \
#   --prior_value "$PRIOR_VALUE" \
#   --target_key mean_acc_per_epoch \
#   --start_step 1 \
#   --end_step 66 \
#   --step_size 1 \
#   --output_dir /home/hieunt/verl/results/gbr