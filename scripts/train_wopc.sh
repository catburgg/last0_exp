#!/bin/bash
# Exp5: pure latent recon (no sim), latent_size=16, stacked small convs (+SiLU). Baseline: exp4-style + --latent_downsample_mode single
set -e

cd /mnt/wfm/code/zxh/last0_exp/scripts
export PYTHONPATH=/mnt/wfm/code/zxh/last0_exp:/mnt/wfm/code/zxh/last0_exp/transformers:$PYTHONPATH
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

BASE_RUN_NAME="pure_latent_recon_ls4_stacked"
EXPERIMENT_NAME="libero_spatial_ablation"
OUTPUT_ROOT_DIR="/mnt/wfm/ckpt/ckpt/last0_exp"

DATA_JSON="/mnt/wfm/ckpt/data/data_libero/libero_training_data/libero_json/libero_spatial_no_noops_view2_chunk4_16_stride8_fast1_sparse_fastslow_train.json"
PRETRAIN_PATH="/mnt/wfm/ckpt/ckpt/pretrained/Janus-Pro-1B"
PRETRAIN_ACTION_PATH="/mnt/wfm/ckpt/ckpt/pretrained/LaST0_Pretrain_AE_chunk8/tfmr"
COSMOS_TOKENIZER_DIR="/mnt/wfm/ckpt/ckpt/pretrained/Cosmos-Tokenizer-CI8x8"

NUM_PROCESSES=8
TRAIN_BSZ=2
LR=1e-4

accelerate launch --config_file ../config/sft.yaml \
    --num_processes ${NUM_PROCESSES} \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard train_wopc.py \
    --pretrain_path ${PRETRAIN_PATH} \
    --pretrain_action_path ${PRETRAIN_ACTION_PATH} \
    --data_path ${DATA_JSON} \
    --data_root "" \
    --n_epochs 100 \
    --save_freq 5 \
    --action_dim 7 \
    --action_chunk 8 \
    --train_bsz_per_gpu ${TRAIN_BSZ} \
    --learning_rate ${LR} \
    --min_lr_ratio 0 \
    --weight_decay 0 \
    --gradient_accumulation_steps 4 \
    --output_dir ${OUTPUT_ROOT_DIR} \
    --log_dir ${OUTPUT_ROOT_DIR} \
    --experiment_name ${EXPERIMENT_NAME} \
    --load_action_from_latent 0 \
    --load_action_from_pretrain 1 \
    --use_latent 1 \
    --latent_size 4 \
    --latent_downsample_mode stacked \
    --recon_mode latent \
    --recon_weight 1.0 \
    --sim_weight 0.0 \
    --run_name ${BASE_RUN_NAME}

echo ">>> Exp5 Finished."
