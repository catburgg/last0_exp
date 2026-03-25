#!/bin/bash
# Exp5: pure latent recon (no sim), latent_size=16, stacked small convs (+SiLU). Baseline: exp4-style + --latent_downsample_mode single
set -e

cd /mnt/workspace/caojiajun/code/last0_exp/scripts
source /mnt/workspace/caojiajun/miniconda3/bin/activate
conda activate /mnt/workspace/caojiajun/miniconda3/envs/cosmos_mot

export PATH=/mnt/workspace/caojiajun/miniconda3/envs/cosmos_mot/bin:$PATH
export HF_HOME=/mnt/dataset/share/hwb/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/workspace/caojiajun/code/last0_exp:/mnt/workspace/caojiajun/code/last0_exp/transformers:$PYTHONPATH
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1

BASE_RUN_NAME="libero_spatial_query"
EXPERIMENT_NAME="debug"
OUTPUT_ROOT_DIR="../ckpt"

DATA_JSON="/mnt/dataset/share/hwb/dataset/libero_training_data/libero_json/libero_spatial_no_noops_view2_chunk4_16_stride8_fast1_sparse_fastslow_train.json"
PRETRAIN_PATH="/mnt/dataset/share/hwb/hf_cache/Janus-Pro-1B"
PRETRAIN_ACTION_PATH="/mnt/dataset/share/hwb/hf_cache/LaST0_Pretrain_AE_chunk8/tfmr"
COSMOS_TOKENIZER_DIR="/mnt/dataset/share/hwb/hf_cache/Cosmos-0.1-Tokenizer-CI8x8"

NUM_PROCESSES=2
TRAIN_BSZ=1
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
    --latent_downsample_mode cross_attn \
    --recon_mode latent \
    --recon_weight 0.0 \
    --sim_weight 1.0 \
    --run_name ${BASE_RUN_NAME} \
    --cosmos_tokenizer_dir ${COSMOS_TOKENIZER_DIR}

echo ">>> Exp5 Finished."
