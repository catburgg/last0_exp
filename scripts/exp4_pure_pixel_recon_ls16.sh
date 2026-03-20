#!/bin/bash
# Exp4: pure latent recon loss (no sim), latent_size=16
set -e

cd /mnt/nas/zhangxuheng/last0/scripts
export PYTHONPATH=/mnt/nas/zhangxuheng/last0:/mnt/nas/zhangxuheng/last0/transformers:$PYTHONPATH
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

BASE_RUN_NAME="pure_latent_recon_ls16"
EXPERIMENT_NAME="libero_spatial_ablation"
OUTPUT_ROOT_DIR="/mnt/data/zhangxuheng/ckpt/exp/"

DATA_JSON="/mnt/data/zhangxuheng/data/libero_training_data/libero_json/libero_spatial_no_noops_view2_chunk4_16_stride8_fast1_sparse_fastslow_train.json"
PRETRAIN_PATH="/mnt/data/zhangxuheng/ckpt/pretrained/Janus-Pro-1B"
PRETRAIN_ACTION_PATH="/mnt/data/zhangxuheng/ckpt/pretrained/LaST0_Pretrain_AE_chunk8/tfmr"

NUM_PROCESSES=8
TRAIN_BSZ=4
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
    --gradient_accumulation_steps 2 \
    --output_dir ${OUTPUT_ROOT_DIR} \
    --log_dir ${OUTPUT_ROOT_DIR} \
    --experiment_name ${EXPERIMENT_NAME} \
    --load_action_from_latent 0 \
    --load_action_from_pretrain 1 \
    --use_latent 1 \
    --latent_size 16 \
    --recon_mode latent \
    --recon_weight 1.0 \
    --sim_weight 0.0 \
    --run_name ${BASE_RUN_NAME}

echo ">>> Exp4 Finished."
