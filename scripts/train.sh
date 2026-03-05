#!/bin/bash

set -e

export http_proxy=http://192.168.32.28:18000 && export https_proxy=http://192.168.32.28:18000

cd /media/liuzhuoyang/LCoT_VLA_MOT/scripts
source /media/miniconda3/bin/activate /media/miniconda3/envs/janus_cot
export PATH=/media/miniconda3/envs/janus_cot/bin:$PATH
export HF_HOME=/media/huggingFace
export PYTHONPATH=/media/liuzhuoyang/LCoT_VLA_MOT:/media/liuzhuoyang/LCoT_VLA_MOT/transformers:$PYTHONPATH
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

BASE_RUN_NAME="janus_pro_siglip_uni3d_1B_1e-4_mot_pretrain0220e0_12tasks_view1_action_latent_sparse_slow1_fast4_pc_state_12_0303"
EXPERIMENT_NAME="latent_cot_mot_flow"
OUTPUT_ROOT_DIR="../exp"

DATA_JSON="/media/liuzhuoyang/LCoT_VLA_MOT/training_data/rlbench_json/12tasks_1view_chunk4_fast4_sparse_fastslow_train.json"
PRETRAIN_PATH="/media/liuzhuoyang/LCoT_VLA_MOT/exp_pretrain/latent_cot_mot_flow_rtx/janus_pro_siglip_uni3d_1B_1e-4_mot_pretrainvlm_rtx_joint_chunk8_stride8_view1+1_wrist_slow1_fast1_point_state_12_0212/stage3/checkpoint-0-319624/tfmr"
PC_EMBEDDER_CKPT="/media/liuzhuoyang/Uni3D/checkpoints/modelzoo/uni3d-b/model.pt"

NUM_PROCESSES=8
TRAIN_BSZ=4
LR=1e-4

accelerate launch --config_file ../config/sft.yaml \
    --num_processes ${NUM_PROCESSES}  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard train.py \
    --pretrain_path ${PRETRAIN_PATH} \
    --data_path ${DATA_JSON} \
    --data_root "" \
    --n_epochs 400 \
    --save_freq 100 \
    --action_dim 7 \
    --action_chunk 1 \
    --train_bsz_per_gpu ${TRAIN_BSZ} \
    --learning_rate ${LR} \
    --min_lr_ratio 0 \
    --weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --output_dir ${OUTPUT_ROOT_DIR} \
    --log_dir ${OUTPUT_ROOT_DIR} \
    --experiment_name ${EXPERIMENT_NAME} \
    --use_latent 1 \
    --latent_size 12 \
    --compress_strategy average \
    --pointcloud_embedder_ckpt_path ${PC_EMBEDDER_CKPT} \
    --run_name ${BASE_RUN_NAME}

echo ">>> Stage 1 Finished."
