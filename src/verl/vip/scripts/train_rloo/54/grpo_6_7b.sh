#!/bin/bash
project_name='GaussianAllocation'
exp_name='Qwen2.5-Math-7B-dpp-dr-grpo-6-ga'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024))
max_response_length=$((1024 * 10))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 3))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"
enable_adaptive_repeat=True
# GPR allocation params
enable_gpr_allocation=True
gpr_upper=12
gpr_embedder="sentence-transformers/all-MiniLM-L6-v2"
gpr_dataset="fixprompt-nodup-dapo-math-17k.dpp_ordered_17398"
gpr_data_root="/root/code_space/verl/data/embedding_data"

allocation_rule="grpo"
filter_groups_metric=acc
max_num_gen_batches=1
train_prompt_bsz=256
gen_prompt_bsz=$((train_prompt_bsz*1))
n_resp_per_prompt=6
train_prompt_mini_bsz=32
min_repeat_times=4
ema_decay=0.9

# Ray
export no_proxy="localhost"
export RAY_ADDRESS="http://${RAY_IP:-localhost}:8265" # The Ray cluster address to connect to
export WORKING_DIR="${PWD}" # The local directory to package to the Ray cluster
# Set the runtime environment like env vars and pip packages for the Ray cluster in yaml
export RUNTIME_ENV="./my_recipe/runtime_env.yaml" # This sets environment variables for the Ray cluster
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"/root/code_space/verl"}
MODEL_PATH=${MODEL_PATH:-"/root/verl/models/Qwen2.5-Math-7B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/fixprompt-nodup-dapo-math-17k.dpp_ordered.parquet"}
AIME_2024=${AIME_2024:-"${RAY_DATA_HOME}/data/fixprompt-aime-2024.parquet"}
AIME_2025=${AIME_2025:-"${RAY_DATA_HOME}/data/fixprompt-aime-2025.parquet"}
MATH_500=${MATH_500:-"${RAY_DATA_HOME}/data/fixprompt-math-500.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
offload=True
gen_tp=1

ray job submit --runtime-env="${RUNTIME_ENV}" \
    -- python3 -m gaussian_allocation.main_gaussian_allocation \
    data.train_files="${TRAIN_FILE}" \
    data.val_files=["${AIME_2024}","${AIME_2025}"] \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.ema_decay=${ema_decay} \
    algorithm.min_repeat_times=${min_repeat_times} \
    algorithm.enable_selection_allocation=False \
    algorithm.enable_adaptive_repeat=${enable_adaptive_repeat} \
    algorithm.enable_gpr_allocation=${enable_gpr_allocation} \
    algorithm.gpr_upper=${gpr_upper} \
    algorithm.gpr_embedder=${gpr_embedder} \
    algorithm.gpr_dataset=${gpr_dataset} \
    algorithm.gpr_data_root=${gpr_data_root} \
    algorithm.allocation_rule=${allocation_rule} \
    algorithm.election_random_seed=1234 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.enable_adaptive_repeat=${enable_adaptive_repeat} \
    algorithm.norm_adv_by_std_in_grpo=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['tensorboard','console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.test_freq=10 \
    trainer.save_freq=70 \
    trainer.total_epochs=2 \
    trainer.total_training_steps=70 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    data.shuffle=False \
    trainer.val_before_train=False \
    trainer.log_val_generations=17920 \
    