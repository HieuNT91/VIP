# Implementation for Variance Informed Predictive allocation strategy (VIP)


The main codes is inside VIP/src/verl/vip/

DEFINE YOUR ENV like this
```
HOME_DIR=/storage2/xxx/VIP
BASE_MODEL_DIR=${HOME_DIR}/base_models
CKPTS_DIR=${HOME_DIR}/ckpts
DATA_DIR=${HOME_DIR}/data
TENSORBOARD_DIR=${HOME_DIR}/logs/tensorboard_logs
RAY_TMPDIR=${HOME_DIR}/ray_tmp
VERL_FILE_LOGGER_ROOT=${HOME_DIR}/logs/file_logs
VLLM_USE_V1=1
```


Installation:
```
pip install -e . 
cd src/verl/ 
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install ninja 
MAX_JOBS=32 pip install --no-build-isolation flash-attn==2.7.2.post1
pip install --no-deps -e .
```
