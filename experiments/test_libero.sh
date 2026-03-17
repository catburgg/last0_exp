
cd /mnt/nas/zhangxuheng/last0
export PYTHONPATH=/mnt/nas/zhangxuheng/LIBERO:$PYTHONPATH
export PYTHONPATH=/mnt/nas/zhangxuheng:/mnt/nas/zhangxuheng/last0/transformers:$PYTHONPATH

# EGL headless rendering setup (required in containerized environments)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
mkdir -p /usr/share/glvnd/egl_vendor.d/
echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Launch LIBERO evals
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /mnt/data/zhangxuheng/ckpt/exp/libero_spatial_baseline/libero_spatial_conv/checkpoint-59-49680/tfmr \
  --task_suite_name libero_spatial \
  --cuda "0" \
  --use_latent True \
  --latent_size 16 \
  # --num_open_loop_steps 8 \
  --seed 0

# libero_spatial libero_object libero_goal libero_10

