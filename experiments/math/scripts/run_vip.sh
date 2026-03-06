#!/usr/bin/env bash
set -xeuo pipefail
set -a 
source .env
set +a

adv_estimator=${ADVANTAGE_ESTIMATOR:-"grpo"}
# very important! please modify the max_position_embeddings in config.json to 32768 after downloading from huggingface
BASE_MODEL=${BASE_MODEL:-"Qwen2.5-Math-1.5B"}
NGPUS=${NGPUS:-4}
SEED=${SEED:-42}
GPU_UTIL=${GPU_UTIL:-0.8}
total_epochs=${TOTAL_EPOCHS:-2}
project_name="VIP-${BASE_MODEL}"
train_prompt_bsz=${BATCH_SIZE:-512}

budget_per_question=${BUDGET:-8}
n_resp_per_prompt=${budget_per_question}
upper_budget=${UPPER_BUDGET:-16}
lower_budget=${LOWER_BUDGET:-4}
length_scale=${LENGTH_SCALE:-0.5}
window_size=${WINDOW_SIZE:-3}
allocation_rule=${RULE:-vip}
difficult_bias=${BIAS:-0.00004}

# Rollout Correction parameters (sequence-level TIS + geometric RS)
rollout_is=sequence
rollout_is_threshold=2.0
rollout_is_batch_normalize=true
rollout_rs=geometric
rollout_rs_threshold=1.01
rollout_rs_threshold_lower=0.99
rollout_token_veto_threshold=1e-4

verbose=${VERBOSE:-True}
learning_rate=${LEARNING_RATE:-1e-6}
data_split=${DATA_SPLIT:-"17"} # can be 6 or 17
exp_name="${adv_estimator}-${BASE_MODEL}-${allocation_rule}-budgetperq${budget_per_question}l${lower_budget}u${upper_budget}-lengthscale${length_scale}-bz${train_prompt_bsz}-e${total_epochs}-lr${learning_rate}--math${data_split}k-rolloutcorr-seed${SEED}"
# exp_name="${adv_estimator}-${BASE_MODEL}-${allocation_rule}b${difficult_bias}-budgetperq${budget_per_question}l${lower_budget}u${upper_budget}-lengthscale${length_scale}-bz${train_prompt_bsz}-e${total_epochs}-lr${learning_rate}--math${data_split}k-rolloutcorr-seed${SEED}"
TENSORBOARD_DIR=${TENSORBOARD_DIR}/${project_name}/${exp_name}

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024))
response_length_multiple=${RESPONSE_LENGTH_MULTIPLE:-15}
max_response_length=$((1024 * $response_length_multiple))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 1))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10
gen_prompt_bsz=$((train_prompt_bsz * 1))
train_prompt_mini_bsz=${MINI_BATCH_SIZE:-64} # set this equal to train_prompt_bsz to enable on_policy

# Paths
MODEL_PATH="${BASE_MODEL_DIR}/${BASE_MODEL}"
CKPTS_DIR=${CKPTS_DIR}/${project_name}/${exp_name}
TRAIN_FILE=${DATA_DIR}/vip-dapo-math-${data_split}k.parquet
AIME24_FILE=${DATA_DIR}/vip-aime-2024.parquet
AIME25_FILE=${DATA_DIR}/vip-aime-2025.parquet


# Algorithm
temperature=1.0
top_p=0.99
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=1

python3 -m train.vip.main_vip \
    data.train_files="${TRAIN_FILE}" \
    data.val_files=["${AIME24_FILE}","${AIME25_FILE}"] \
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
    algorithm.norm_adv_by_std_in_grpo=False \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.allocation.budget_per_question=${budget_per_question} \
    algorithm.allocation.upper=${upper_budget} \
    algorithm.allocation.lower=${lower_budget} \
    algorithm.allocation.verbose=${verbose} \
    algorithm.prediction.length_scale=${length_scale} \
    algorithm.prediction.window_size=${window_size} \
    algorithm.allocation.allocation_rule=${allocation_rule} \
    algorithm.allocation.difficult_bias=${difficult_bias} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${learning_rate} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    algorithm.rollout_correction.rollout_rs_threshold_lower=${rollout_rs_threshold_lower} \
    algorithm.rollout_correction.rollout_token_veto_threshold=${rollout_token_veto_threshold} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_UTIL} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.nnodes="1" \
    trainer.val_before_train=False \
    trainer.test_freq=3 \
    trainer.save_freq=30 \
    trainer.total_epochs=${total_epochs} \
    data.shuffle=False \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
