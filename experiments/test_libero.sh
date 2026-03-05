
cd /path/to/last0
export PYTHONPATH=/path/to/last0/LIBERO:$PYTHONPATH
export PYTHONPATH=/path/to/last0:/path/to/last0/transformers:$PYTHONPATH

# Launch LIBERO evals
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /path/to/ckpt \
  --task_suite_name libero_spatial \
  --cuda "0" \
  --use_latent True \
  --latent_size 8 \
  --seed 0

# libero_spatial libero_object libero_goal libero_10

