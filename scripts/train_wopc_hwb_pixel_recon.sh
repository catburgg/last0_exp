#!/bin/bash

set -e

cd /mnt/dataset/share_code/code/last0_exp/scripts
source /root/miniforge3/bin/activate /mnt/dataset/share_code/conda_envs/last05_hwb

export PYTHONPATH=/mnt/dataset/share_code/code/last0_exp:/mnt/dataset/share_code/code/last0_exp/transformers:$PYTHONPATH
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1


BASE_RUN_NAME="libero_spatial_mlp"
EXPERIMENT_NAME="last0_libero_spatial"
OUTPUT_ROOT_DIR="/mnt/dataset/share_code/code/last0_exp/ckpt/"

DATA_JSON="/mnt/dataset/share_code/dataset/libero_training_data/libero_json/libero_spatial_no_noops_view2_chunk4_16_stride8_fast1_sparse_fastslow_train.json"
PRETRAIN_PATH="/mnt/dataset/share_code/hf_cache/Janus-Pro-1B"
PRETRAIN_ACTION_PATH="/mnt/dataset/share_code/hf_cache/LaST0_Pretrain_AE_chunk8/tfmr"

NUM_PROCESSES=8
TRAIN_BSZ=4
LR=1e-4

accelerate launch --config_file ../config/sft.yaml \
    --num_processes ${NUM_PROCESSES}  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard train_wopc.py \
    --pretrain_path ${PRETRAIN_PATH} \
    --pretrain_action_path ${PRETRAIN_ACTION_PATH} \
    --data_path ${DATA_JSON} \
    --data_root "" \
    --n_epochs 200 \
    --save_freq 10 \
    --action_dim 7 \
    --action_chunk 8 \
    --train_bsz_per_gpu ${TRAIN_BSZ} \
    --learning_rate ${LR} \
    --min_lr_ratio 0 \
    --weight_decay 0 \
    --gradient_accumulation_steps 2 \
    --output_dir ${OUTPUT_ROOT_DIR} \
    --log_dir ${OUTPUT_ROOT_DIR} \
    --experiment_name ${EXPERIMENT_NAME} \
    --load_action_from_latent 0 \
    --load_action_from_pretrain 1 \
    --use_latent 1 \
    --latent_size 4 \
    --recon_mode pixel \
    --recon_weight 1.0 \
    --run_name ${BASE_RUN_NAME}

echo ">>> LaST0 Training Finished."
