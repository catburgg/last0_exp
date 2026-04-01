#!/usr/bin/env bash
set -euo pipefail

_LAST0="${LAST0_ENV:-/mnt/wfm/ckpt/env/last0}"
if [[ -f "${_LAST0}/source_cuda_ld_path.sh" ]]; then
  # shellcheck source=/dev/null
  source "${_LAST0}/source_cuda_ld_path.sh"
fi
PYTHON_BIN="${PYTHON_BIN:-${_LAST0}/bin/python}"

cd /mnt/wfm/code/zxh/last0_exp/visualization
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-/mnt/wfm/ckpt/ckpt/pretrained/Cosmos-Predict2.5-2B/tokenizer.pth}"
export COSMOS_DIT_PATH="${COSMOS_DIT_PATH:-/mnt/wfm/ckpt/ckpt/pretrained/Cosmos-Predict2.5-2B/robot/policy/libero/model.pt}"

# VLChatProcessor / tokenizer live under .../checkpoint-*/tfmr (not the checkpoint parent dir).
"$PYTHON_BIN" eval_latent_similarity.py \
  --checkpoint_path /mnt/wfm/code/zxh/last0_exp/ckpt/libero_spatial_ablation/libero_spatial_dit_avg/checkpoint-99-82800/tfmr \
  --data_path /mnt/wfm/ckpt/data/data_libero/libero_training_data/libero_json/libero_spatial_no_noops_view2_chunk4_16_stride8_fast1_sparse_fastslow_train.json \
  --output_dir /mnt/wfm/code/zxh/last0_exp/eval_latent_sim_out \
  --dit_full_num_blocks 28 \
  --dit_num_blocks_run 11 \
  --num_samples 200 \
  --tokens_per_modality 100 \
  --future_steps 4
