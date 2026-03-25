#!/usr/bin/env bash
set -euo pipefail

# Pip CUDA wheels: system libnvjitlink/cusparse order can break torch._C load.
_LAST0="${LAST0_ENV:-/mnt/wfm/ckpt/env/last0}"
if [[ -f "${_LAST0}/source_cuda_ld_path.sh" ]]; then
  # shellcheck source=/dev/null
  source "${_LAST0}/source_cuda_ld_path.sh"
fi
PYTHON_BIN="${PYTHON_BIN:-${_LAST0}/bin/python}"

cd /mnt/wfm/code/zxh/last0_exp/scripts
export CUDA_VISIBLE_DEVICES=1
export COSMOS_TOKENIZER_DIR="${COSMOS_TOKENIZER_DIR:-/mnt/wfm/ckpt/ckpt/pretrained/Cosmos-Tokenizer-CI8x8}"
"$PYTHON_BIN" visualize_latent_recon.py \
  --checkpoint_path /mnt/wfm/ckpt/ckpt/last0_exp/libero_spatial_ablation/pure_latent_recon_ls4_stacked/checkpoint-19-66220/tfmr \
  --data_path /mnt/wfm/ckpt/data/data_libero/libero_training_data/libero_json/libero_spatial_no_noops_view2_chunk4_16_stride8_fast1_sparse_fastslow_train.json \
  --data_root "" \
  --output_dir /mnt/wfm/code/zxh/last0_exp/ \
  --num_samples 1 \
  --latent_size 4 \
  --action_dim 7 \
  --action_chunk 8