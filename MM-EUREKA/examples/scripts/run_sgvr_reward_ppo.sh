#!/bin/bash
set -x

ray stop --force
pkill -f ray
pkill -f python
pkill -f vllm
sleep 10

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=14400 
export NCCL_P2P_DISABLE=1 
export NCCL_SHM_DISABLE=1  
export NCCL_TREE_THRESHOLD=0  
export NCCL_ALGO=Ring  
export NCCL_PROTO=simple  
export NCCL_CROSS_NIC=0  

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,COMM



# ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-gpus 8

KL_COEF=1e-2
BATCH_SIZE=16
MICRO_BATCH_SIZE=2
SAVE_STEP=30
N_SAMPLES_PER_PROMPT=4
export VLLM_USE_TRUST_REMOTE_CODE=1
export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8265
export MASTER_ADDR=127.0.0.1 
export NODE_RANK=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR=""
SFT_MODEL=''
TRAIN_DATA=''
export REWARD_LOG_PATH="${OUTPUT_DIR}/reward.log"
export DEBUG_LOG_PATH="${OUTPUT_DIR}/debug.log"
export WORKING_DIR=$PWD

echo "Driver REWARD_LOG_PATH=$REWARD_LOG_PATH"
echo "Driver DEBUG_LOG_PATH=$DEBUG_LOG_PATH"

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

if [ "$NODE_RANK" -eq 0 ]; then
    ray start --head  --port=$RAY_MASTER_PORT --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus 8
else
    sleep 30
    ray start --address="$MASTER_ADDR:$RAY_MASTER_PORT" --num-gpus 8 --block
fi

sleep 30

# 在主节点上提交任务
if [ "$NODE_RANK" -eq 0 ]; then
  echo "使用远程奖励服务进行强化学习训练..."
  
  RUNTIME_ENV="{\"env_vars\": {\"PYTHONPATH\": \"$PYTHONPATH\", \"VLLM_USE_CUDA_IPC\": \"0\", \"VLLM_WORKER_USE_CUDA_IPC\": \"0\", \"RAY_DEDUP_LOGS\": \"0\", \"CUDA_IPC_ENABLE\": \"0\", \"REWARD_LOG_PATH\": \"$REWARD_LOG_PATH\", \"DEBUG_LOG_PATH\": \"$DEBUG_LOG_PATH\", \"NCCL_IB_DISABLE\": \"1\", \"NCCL_SOCKET_IFNAME\": \"eth0\", \"TORCH_NCCL_BLOCKING_WAIT\": \"1\", \"NCCL_ASYNC_ERROR_HANDLING\": \"1\", \"NCCL_TIMEOUT\": \"14400\", \"NCCL_P2P_DISABLE\": \"1\", \"NCCL_SHM_DISABLE\": \"1\", \"NCCL_TREE_THRESHOLD\": \"0\", \"NCCL_ALGO\": \"Ring\", \"NCCL_PROTO\": \"simple\", \"NCCL_CROSS_NIC\": \"0\", \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"}}"

  RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit \
  --working-dir $WORKING_DIR \
  --runtime-env-json "$RUNTIME_ENV" \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 8 \
  --remote_rm_url ./sgvr_reward.py \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 8 \
  --critic_num_nodes 1 \
  --critic_num_gpus_per_node 8 \
  --vllm_num_engines 8 \
  --vllm_tensor_parallel_size 1 \
  --colocate_all_models \
  --vllm_gpu_memory_utilization 0.4 \
  --pretrain "${SFT_MODEL}" \
  --save_path ${OUTPUT_DIR} \
  --micro_train_batch_size ${MICRO_BATCH_SIZE} \
  --train_batch_size ${BATCH_SIZE} \
  --micro_rollout_batch_size ${MICRO_BATCH_SIZE} \
  --rollout_batch_size ${BATCH_SIZE} \
  --vllm_sync_backend gloo \
  --temperature 0.9 \
  --n_samples_per_prompt ${N_SAMPLES_PER_PROMPT} \
  --lambd 1.0 \
  --gamma 1.0 \
  --max_epochs 1 \
  --num_episodes 10 \
  --prompt_max_len 4096 \
  --max_samples 100000 \
  --generate_max_len 4096 \
  --advantage_estimator gae \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 1e-6 \
  --critic_learning_rate 9e-6 \
  --critic_pretrain ${SFT_MODEL} \
  --init_kl_coef ${KL_COEF} \
  --use_kl_loss \
  --prompt_data ${TRAIN_DATA} \
  --disable_fast_tokenizer \
  --input_key message \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --save_steps ${SAVE_STEP} \
  --ckpt_path "${OUTPUT_DIR}/ckpt" \
  --max_ckpt_num 1000000 \
  --save_hf_ckpt \
  --freeze_prefix visual \
  --use_tensorboard "${OUTPUT_DIR}/tensorboard" \
  --load_checkpoint | tee ${OUTPUT_DIR}/training.log 
fi

