#!/bin/bash
# DiT Early-Exit pipeline: WAN VAE + Cosmos-Predict2.5 DiT (vision_backend=wan_dit).
# tokenizer.pth and model.pt are the matched pair; no Cosmos .jit / gen conv on this path.
# Debug one loss term: add e.g. --sim_weight 0 (only action) or --action_loss_weight 0 (only sim; keep recon_weight 0).
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
# cosmos-policy removed from PYTHONPATH — dit_lib.py is fully self-contained
export PYTHONPATH=/mnt/wfm/code/zxh/last0_exp:$PYTHONPATH
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1

BASE_RUN_NAME="libero_spatial_dit25"
EXPERIMENT_NAME="libero_spatial_dit25"
OUTPUT_ROOT_DIR="/mnt/wfm/code/zxh/last0_exp/ckpt/"

DATA_JSON="/mnt/wfm/ckpt/data/data_libero/libero_training_data/libero_json/libero_spatial_no_noops_view2_chunk4_16_stride8_fast1_sparse_fastslow_train.json"
PRETRAIN_PATH="/mnt/wfm/ckpt/ckpt/pretrained/Janus-Pro-1B"
PRETRAIN_ACTION_PATH="/mnt/wfm/ckpt/ckpt/pretrained/LaST0_Pretrain_AE_chunk8/tfmr"
COSMOS_DIT_PATH="/mnt/wfm/ckpt/ckpt/pretrained/Cosmos-Predict2.5-2B/robot/policy/libero/model.pt"
WAN_VAE_PATH="/mnt/wfm/ckpt/ckpt/pretrained/Cosmos-Predict2.5-2B/tokenizer.pth"

NUM_PROCESSES=2
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
    --vision_backend wan_dit \
    --dit_align_mode conv \
    --latent_downsample_mode single \
    --recon_mode latent \
    --recon_weight 0.0 \
    --sim_weight 1.0 \
    --cosmos_dit_path ${COSMOS_DIT_PATH} \
    --wan_vae_path ${WAN_VAE_PATH} \
    --run_name ${BASE_RUN_NAME}

echo ">>> DiT Early-Exit (Predict2.5 + WAN VAE) Training Finished."
