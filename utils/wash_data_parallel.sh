#!/bin/bash
# Launch N wash jobs: same output_dir, disjoint samples, manifest.shardXXofYY.json per job.
# Multi-GPU: GPU_IDS="0 1 2 ...". Single-GPU: GPU_IDS="0" (watch VRAM).
set -euo pipefail

cd /mnt/wfm/code/zxh/last0_exp/scripts
export PYTHONPATH="/mnt/wfm/code/zxh/last0_exp${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${PYTHON:-/mnt/wfm/ckpt/env/last0/bin/python3}"
NUM_SHARDS="${NUM_SHARDS:-8}"
# Space- or comma-separated GPU ids (one device per shard, round-robin). read -a only splits on spaces — commas are normalized.
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
GPU_IDS_NORM="${GPU_IDS//,/ }"
read -r -a _GPUS <<< "${GPU_IDS_NORM}"
_NG="${#_GPUS[@]}"

PRETRAIN_PATH="/mnt/wfm/ckpt/ckpt/pretrained/Janus-Pro-1B"
COSMOS_DIT_PATH="/mnt/wfm/ckpt/ckpt/pretrained/Cosmos-Predict2.5-2B/robot/policy/libero/model.pt"
WAN_VAE_PATH="/mnt/wfm/ckpt/ckpt/pretrained/Cosmos-Predict2.5-2B/tokenizer.pth"
DATA_JSON="/mnt/wfm/ckpt/data/data_libero/libero_training_data/libero_json/libero_spatial_no_noops_view2_chunk4_16_stride8_fast1_sparse_fastslow_train.json"
OUTPUT_GT_ROOT="/mnt/wfm/ckpt/data/data_libero/libero_training_data_w_latent23"

COSMOS_K="23"
DIT_NUM_BLOCKS="28"

for ((sid = 0; sid < NUM_SHARDS; sid++)); do
  g="${_GPUS[$((sid % _NG))]}"
  (
    export CUDA_VISIBLE_DEVICES="${g}"
    "${PYTHON}" ../utils/wash_cosmos_dit_gt_cache.py \
      --data_path "${DATA_JSON}" \
      --checkpoint "${PRETRAIN_PATH}" \
      --output_dir "${OUTPUT_GT_ROOT}" \
      --source_format auto \
      --device cuda \
      --dtype bfloat16 \
      --wan_vae_path "${WAN_VAE_PATH}" \
      --cosmos_dit_path "${COSMOS_DIT_PATH}" \
      --dit_num_blocks "${DIT_NUM_BLOCKS}" \
      --cosmos_k_steps "${COSMOS_K}" \
      --policy_num_steps 35 \
      --policy_sigma_min 0.002 \
      --policy_sigma_max 80.0 \
      --policy_rho 7.0 \
      --dit_sigma_data 0.5 \
      --rng_seed 0 \
      --num_shards "${NUM_SHARDS}" \
      --shard_id "${sid}"
  ) &
  echo "started shard ${sid} on CUDA_VISIBLE_DEVICES=${g}"
done

wait
echo ">>> All ${NUM_SHARDS} shards finished."

if [[ "${MERGE_MANIFEST:-1}" == "1" ]]; then
  "${PYTHON}" ../utils/merge_wash_gt_manifests.py \
    --output_dir "${OUTPUT_GT_ROOT}" \
    --num_shards "${NUM_SHARDS}"
fi

echo ">>> Done."
