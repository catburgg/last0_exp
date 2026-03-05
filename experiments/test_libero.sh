
cd /media/liuzhuoyang/LCoT_VLA_MOT
source /media/miniconda3/bin/activate /media/miniconda3/envs/janus_cot
export PATH=/media/miniconda3/envs/janus_cot/bin:$PATH
export HF_HOME=/media/huggingFace
export PYTHONPATH=/media/liuzhuoyang/LCoT_VLA_MOT/LIBERO:$PYTHONPATH
export PYTHONPATH=/media/liuzhuoyang/LCoT_VLA_MOT:/media/liuzhuoyang/LCoT_VLA_MOT/transformers:$PYTHONPATH
export WANDB_MODE=offline

# Launch LIBERO-Spatial evals
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /media/liuzhuoyang/LCoT_VLA_MOT/exp/latent_cot_mot_flow_libero/janus_pro_siglip_uni3d_1B_1e-4_mot_pretrain0123e0_libero_10_joint_chunk32_stride8_view1+1_wrist_slow1_fast1_state_8_oneshot_0220/stage3/checkpoint-149-475650/tfmr \
  --task_suite_name libero_10 \
  --cuda "0" \
  --use_latent True \
  --latent_size 8 \
  --seed 0

# # Launch LIBERO-Object evals
# python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-object \
#   --task_suite_name libero_object

# # Launch LIBERO-Goal evals
# python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-goal \
#   --task_suite_name libero_goal

# # Launch LIBERO-10 (LIBERO-Long) evals
# python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
#   --task_suite_name libero_10


