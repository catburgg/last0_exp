#!/bin/bash

set -e

cd /path/to/last0
export PYTHONPATH=/path/to/last0:/path/to/last0/transformers:$PYTHONPATH
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

BASE_RUN_NAME="your_run_name"
EXPERIMENT_NAME="your_experiment_name"
OUTPUT_ROOT_DIR="../exp"

DATA_JSON="/path/to/data.json"
PRETRAIN_PATH="/path/to/pretrain_path"
PRETRAIN_ACTION_PATH="/path/to/pretrain_action_path"

NUM_PROCESSES=8
TRAIN_BSZ=8
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
    --save_freq 50 \
    --action_dim 7 \
    --action_chunk 8 \
    --train_bsz_per_gpu ${TRAIN_BSZ} \
    --learning_rate ${LR} \
    --min_lr_ratio 0 \
    --weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --output_dir ${OUTPUT_ROOT_DIR} \
    --log_dir ${OUTPUT_ROOT_DIR} \
    --experiment_name ${EXPERIMENT_NAME} \
    --load_action_from_latent 0 \
    --load_action_from_pretrain 1 \
    --use_latent 1 \
    --latent_size 8 \
    --run_name ${BASE_RUN_NAME}

echo ">>> LaST0 Training Finished."