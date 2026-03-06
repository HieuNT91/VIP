set -a 
source .env
set +a


hf download Qwen/Qwen2.5-Math-1.5B --local-dir ${BASE_MODEL_DIR}/Qwen2.5-Math-1.5B
hf download Qwen/Qwen2.5-Math-7B --local-dir ${BASE_MODEL_DIR}/Qwen2.5-Math-7B
hf download meta-llama/Llama-3.2-3B-Instruct --local-dir ${BASE_MODEL_DIR}/Llama3.2-3B-Instruct