export LIBERO_CONFIG_PATH=/mnt/workspace/caojiajun/.libero
cd /mnt/workspace/caojiajun/code/last0_exp
source /mnt/workspace/caojiajun/miniconda3/bin/activate
conda activate /mnt/workspace/caojiajun/miniconda3/envs/cosmos_mot
export PATH=/mnt/workspace/caojiajun/miniconda3/envs/cosmos_mot/bin:$PATH
export HF_HOME=/mnt/dataset/share/hwb/hf_cache
export PYTHONPATH=/mnt/workspace/caojiajun/code/last0_exp/LIBERO:$PYTHONPATH
export PYTHONPATH=/mnt/workspace/caojiajun/code/last0_exp:$PYTHONPATH
export WANDB_MODE=offline

# EGL headless rendering setup (required in containerized environments)
N=3
pkill -f "Xvfb :$N" || true

Xvfb :$N -screen 0 1024x768x24 &
export DISPLAY=:$N

unset LD_PRELOAD
unset EGL_DEVICE_ID
unset PYOPENGL_PLATFORM
# # export MUJOCO_GL=egl
# # export EGL_DEVICE_ID=0
# # export __NV_PRIME_RENDER_OFFLOAD=1
# # export __GLX_VENDOR_LIBRARY_NAME=nvidia
# # export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# # export __GLVND_DISALLOW_PATCHING=1
# # export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

export CUDA_VISIBLE_DEVICES=1

OUTPUT_DIR=/mnt/workspace/caojiajun/code/last0_exp/ckpt/libero_object_query/libero_object_query_kd0.1_ls16/test_output/test
mkdir -p $OUTPUT_DIR
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /mnt/workspace/caojiajun/code/last0_exp/ckpt/libero_object_query/libero_object_query_kd0.1_ls16/checkpoint-9-10470/tfmr \
  --task_suite_name libero_object \
  --cuda "0" \
  --vision_backend cosmos_vae \
  --latent_size 16 \
  --num_open_loop_steps 8 \
  --save_videos False \
  --seed 0 \
  --output_dir $OUTPUT_DIR

# libero_spatial libero_object libero_goal libero_10
