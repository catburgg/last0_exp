#!/usr/bin/env bash
# WAN VAE + DiT visualization from SftDataset (same data flow as visualize_latent_recon / visualize_recon.sh).
set -euo pipefail

# Pip CUDA wheels: prepend nvjitlink so torch can load (cusparse symbol __nvJitLinkComplete_*).
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

_LAST0="${LAST0_ENV:-/mnt/wfm/ckpt/env/last0}"
if [[ -f "${_LAST0}/source_cuda_ld_path.sh" ]]; then
  # shellcheck source=/dev/null
  source "${_LAST0}/source_cuda_ld_path.sh"
fi
PYTHON_BIN="${PYTHON_BIN:-${_LAST0}/bin/python}"

cd /mnt/wfm/code/zxh/last0_exp/visualization
export PYTHONPATH="/mnt/wfm/code/zxh/last0_exp${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-/mnt/wfm/ckpt/ckpt/pretrained/Cosmos-Predict2.5-2B/tokenizer.pth}"
export COSMOS_DIT_PATH="${COSMOS_DIT_PATH:-/mnt/wfm/ckpt/ckpt/pretrained/Cosmos-Predict2.5-2B/robot/policy/libero/model.pt}"

# Outputs per sample:
#   sample_XXXX.png       — 3 rows × num_frames cols: GT | WAN recon | decode_mu (full DiT head + precondition)
#   sample_XXXX_blocks*.png when --dit_block_viz set (PCA grid + pca_on_gt + heatmap_on_gt)
#   sample_XXXX_probe_final_decode.png when --dit_probe_final_after_blocks K1 K2 … (final_layer after K blocks → decode)
"$PYTHON_BIN" visualize_wan_dit_vae.py \
  --checkpoint_path /mnt/wfm/ckpt/ckpt/last0_exp/libero_spatial_ablation/pure_latent_recon_ls4_stacked/checkpoint-19-66220/tfmr \
  --data_path /mnt/wfm/ckpt/data/data_libero/libero_training_data/libero_json/libero_spatial_no_noops_view2_chunk4_16_stride8_fast1_sparse_fastslow_train.json \
  --data_root "" \
  --output_dir /mnt/wfm/code/zxh/last0_exp/wan_vae_viz_out \
  --num_samples 1 \
  --latent_size 4 \
  --action_dim 7 \
  --action_chunk 8 \
  --dit_sigma 0.3 \
  --dit_probe_final_after_blocks 2 4 6 8 10 12 14 16 18 20 22 24 26 28 \
  --dit_block_viz 2 4 6 8 10 12 14 16 18 20 22 24 26 28

echo ">>> Wrote under: /mnt/wfm/code/zxh/last0_exp/wan_vae_viz_out"
