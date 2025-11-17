# Implementation for Variance Informed Predictive allocation strategy (VIP)


The main codes is inside VIP/src/verl/vip/

DEFINE YOUR ENV like this
```
HOME_DIR=/storage2/hiu/VIP
BASE_MODEL_DIR=${HOME_DIR}/base_models
CKPTS_DIR=${HOME_DIR}/ckpts
DATA_DIR=${HOME_DIR}/data
TENSORBOARD_DIR=${HOME_DIR}/tensorboard_logs
```


Installation:
```
pip install -e . 
cd src/verl/ 
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install ninja 
MAX_JOB=32 pip install --no-build-isolation flash-attn=2.7.4.post1
pip install --no-deps -e .
```