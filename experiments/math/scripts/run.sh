ADVANTAGE_ESTIMATOR=rloo BASE_MODEL=Qwen2.5-Math-1.5B ROLLOUT_SIZE=8 NGPUS=8 SEED=1 bash experiments/math/scripts/run_baselines.sh
ADVANTAGE_ESTIMATOR=rloo BASE_MODEL=Qwen2.5-Math-1.5B ROLLOUT_SIZE=8 NGPUS=8 SEED=2 bash experiments/math/scripts/run_baselines.sh
ADVANTAGE_ESTIMATOR=rloo BASE_MODEL=Qwen2.5-Math-1.5B ROLLOUT_SIZE=8 NGPUS=8 SEED=3 bash experiments/math/scripts/run_baselines.sh
ADVANTAGE_ESTIMATOR=grpo BASE_MODEL=Qwen2.5-Math-1.5B ROLLOUT_SIZE=8 NGPUS=8 SEED=1 bash experiments/math/scripts/run_baselines.sh
ADVANTAGE_ESTIMATOR=grpo BASE_MODEL=Qwen2.5-Math-1.5B ROLLOUT_SIZE=8 NGPUS=8 SEED=2 bash experiments/math/scripts/run_baselines.sh
ADVANTAGE_ESTIMATOR=grpo BASE_MODEL=Qwen2.5-Math-1.5B ROLLOUT_SIZE=8 NGPUS=8 SEED=3 bash experiments/math/scripts/run_baselines.sh