
export http_proxy=http://192.168.32.28:18000 && export https_proxy=http://192.168.32.28:18000 # for baidu

cd /media/liuzhuoyang/LCoT_VLA_MOT/scripts
source /media/miniconda3/bin/activate /media/miniconda3/envs/janus_cot
export PATH=/media/miniconda3/envs/janus_cot/bin:$PATH
export HF_HOME=/media/huggingFace
export PYTHONPATH=/media/liuzhuoyang/LCoT_VLA_MOT:$PYTHONPATH
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

run_name="janus_pro_siglip_1B_1e-4_mot_s2_fix_1123_debug"

accelerate launch --config_file ../config/sft.yaml \
    --num_processes 1  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard train_janus_siglip.py \
    --model_path /media/liuzhuoyang/LCoT_VLA_MOT/exp/latent_cot_mot/janus_pro_siglip_1B_1e-4_mot_s0_fix_1123/checkpoint-59-1740/tfmr \
    --data_path /media/liuzhuoyang/LCoT_VLA/training_data/json/4tasks_line_train.json \
    --data_root "" \
    --n_epochs 120 \
    --action_dim 7 \
    --train_bsz_per_gpu 8 \
    --learning_rate 1e-4 \
    --min_lr_ratio 0 \
    --weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --output_dir ../exp \
    --log_dir ../exp \
    --experiment_name latent_cot_mot \
    --load_action_from_latent 1 \
    --freeze_latent 1 \
    --use_latent 1 \
    --latent_size 4 \
    --compress_strategy average \
    --run_name ${run_name} \

# FreedomIntelligence/Janus-4o-7B   deepseek-ai/Janus-Pro-7B