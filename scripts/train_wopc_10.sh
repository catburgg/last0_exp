#!/bin/bash
# Exp5: pure latent recon (no sim), latent_size=16, stacked small convs (+SiLU). Baseline: exp4-style + --latent_downsample_mode single
# 多机多卡训练脚本，需在各节点上运行；单机多卡时保持默认 NNODES=1 即可
set -e

cd /mnt/workspace/caojiajun/code/last0_exp/scripts
source /mnt/workspace/caojiajun/miniconda3/bin/activate
conda activate /mnt/workspace/caojiajun/miniconda3/envs/cosmos_mot

export PATH=/mnt/workspace/caojiajun/miniconda3/envs/cosmos_mot/bin:$PATH
export HF_HOME=/mnt/dataset/share/hwb/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/workspace/caojiajun/code/last0_exp:/mnt/workspace/caojiajun/code/last0_exp/transformers:$PYTHONPATH
export WANDB_MODE=online
export WANDB_API_KEY="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"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 多机/多卡：每节点 GPU 数、总节点数、当前节点 rank（由调度或手动设置）
GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
# accelerate --num_processes 是所有节点的总进程数
TOTAL_PROCESSES=$((GPUS_PER_NODE * NNODES))
echo ">>> ACCELERATE multi-node: num_processes=${TOTAL_PROCESSES} num_machines=${NNODES} machine_rank=${NODE_RANK} main_process_ip=${MASTER_ADDR} main_process_port=${MASTER_PORT}"

BASE_RUN_NAME="libero_10_query_query_kd0.1_ls16"
EXPERIMENT_NAME="libero_10_query"
OUTPUT_ROOT_DIR="../ckpt/${EXPERIMENT_NAME}/${BASE_RUN_NAME}"

DATA_JSON="/mnt/dataset/share/hwb/dataset/libero_training_data/libero_json/libero_10_no_noops_view2_chunk4_16_stride8_fast1_sparse_fastslow_train.json"
PRETRAIN_PATH="/mnt/dataset/share/hwb/hf_cache/Janus-Pro-1B"
PRETRAIN_ACTION_PATH="/mnt/dataset/share/hwb/hf_cache/LaST0_Pretrain_AE_chunk8/tfmr"
COSMOS_TOKENIZER_DIR="/mnt/dataset/share/hwb/hf_cache/Cosmos-0.1-Tokenizer-CI8x8"

TRAIN_BSZ=8
GRAD_ACCUM=4
LR=1e-4


echo ">>> Starting Training: ${BASE_RUN_NAME}"

accelerate launch --config_file ../config/sft.yaml \
    --num_processes ${TOTAL_PROCESSES} \
    --num_machines ${NNODES} \
    --machine_rank ${NODE_RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --deepspeed_multinode_launcher standard train_wopc.py \
    --pretrain_path ${PRETRAIN_PATH} \
    --pretrain_action_path ${PRETRAIN_ACTION_PATH} \
    --data_path ${DATA_JSON} \
    --data_root "" \
    --n_epochs 200 \
    --save_freq 20 \
    --action_dim 7 \
    --action_chunk 8 \
    --train_bsz_per_gpu ${TRAIN_BSZ} \
    --learning_rate ${LR} \
    --min_lr_ratio 0 \
    --weight_decay 0 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --output_dir ${OUTPUT_ROOT_DIR} \
    --log_dir ${OUTPUT_ROOT_DIR} \
    --experiment_name ${EXPERIMENT_NAME} \
    --load_action_from_latent 0 \
    --load_action_from_pretrain 1 \
    --use_latent 1 \
    --latent_size 16 \
    --latent_downsample_mode cross_attn \
    --recon_mode latent \
    --recon_weight 0.0 \
    --sim_weight 0.1 \
    --run_name ${BASE_RUN_NAME} \
    --cosmos_tokenizer_dir ${COSMOS_TOKENIZER_DIR} \
    --max_ckpts 0

echo ">>> Exp5 Finished."
