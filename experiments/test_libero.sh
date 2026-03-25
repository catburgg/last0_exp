
cd /mnt/wfm/code/zxh/last0_exp
export PYTHONPATH=/mnt/wfm/code/zxh/LIBERO:$PYTHONPATH
export PYTHONPATH=/mnt/wfm/code/zxh/last0_exp:/mnt/wfm/code/zxh:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

source /mnt/wfm/ckpt/env/last0/source_cuda_ld_path.sh
export PATH="/mnt/wfm/ckpt/env/last0/bin:$PATH"
export CUDA_VISIBLE_DEVICES=1

# EGL headless rendering setup (required in containerized environments)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
mkdir -p /usr/share/glvnd/egl_vendor.d/
echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Launch LIBERO evals
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /mnt/wfm/ckpt/ckpt/last0_exp/libero_spatial_ablation/pure_latent_recon_ls4_stacked/checkpoint-19-66220/tfmr \
  --task_suite_name libero_spatial \
  --cuda "0" \
  --vision_backend cosmos_vae \
  --latent_size 4 \
  --num_open_loop_steps 8 \
  --save_videos False \
  --seed 0

# libero_spatial libero_object libero_goal libero_10
