#!/bin/bash
# Exp16: same as Exp10 but dit_align_mode=attn_query (DitPatchVectorizerQueryStyle: pre_mlp → MHA → post_mlp).
set -e

if ACCELERATE_BIN="$(command -v accelerate 2>/dev/null)"; then
  _PY="${ACCELERATE_BIN%/*}/python"
elif command -v python3 >/dev/null 2>&1; then
  _PY="$(command -v python3)"
else
  _PY=python3
fi
_NVJITLINK_LIB="$("$_PY" -c "import pathlib; import nvidia.nvjitlink; print(pathlib.Path(nvidia.nvjitlink.__file__).parent / 'lib')" 2>/dev/null || true)"
if [[ -n "$_NVJITLINK_LIB" && -d "$_NVJITLINK_LIB" ]]; then
  export LD_LIBRARY_PATH="${_NVJITLINK_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
unset _PY _NVJITLINK_LIB

cd /mnt/wfm/code/zxh/last0_exp/scripts
export PYTHONPATH=/mnt/wfm/code/zxh/last0_exp:$PYTHONPATH
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

BASE_RUN_NAME="libero_spatial_dit_attn_query_style"
EXPERIMENT_NAME="libero_spatial_ablation"
OUTPUT_ROOT_DIR="/mnt/wfm/ckpt/ckpt/last0_exp/"

DATA_JSON="/mnt/wfm/ckpt/data/data_libero/libero_training_data/libero_json/libero_spatial_no_noops_view2_chunk4_16_stride8_fast1_sparse_fastslow_train.json"
PRETRAIN_PATH="/mnt/wfm/ckpt/ckpt/pretrained/Janus-Pro-1B"
PRETRAIN_ACTION_PATH="/mnt/wfm/ckpt/ckpt/pretrained/LaST0_Pretrain_AE_chunk8/tfmr"
COSMOS_DIT_PATH="/mnt/wfm/ckpt/ckpt/pretrained/Cosmos-Predict2.5-2B/robot/policy/libero/model.pt"
WAN_VAE_PATH="/mnt/wfm/ckpt/ckpt/pretrained/Cosmos-Predict2.5-2B/tokenizer.pth"

NUM_PROCESSES=8
TRAIN_BSZ=8
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
    --save_freq 10 \
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
    --latent_size 4 \
    --vision_backend wan_dit \
    --dit_align_mode attn_query \
    --latent_downsample_mode single \
    --recon_mode latent \
    --recon_weight 0.0 \
    --sim_weight 1.0 \
    --cosmos_dit_path ${COSMOS_DIT_PATH} \
    --wan_vae_path ${WAN_VAE_PATH} \
    --run_name ${BASE_RUN_NAME}

echo ">>> Exp16 DiT attn_query (query-style stack, 8-GPU) finished."
