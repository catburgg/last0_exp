# cd /gpfs/0607-cluster/chenhao/clash-for-linux-backup
# bash start.sh
# source /etc/profile.d/clash.sh
# proxy_on

export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897

cd /gpfs/0607-cluster/chenhao/DoubleRL-VLA/scripts
source /gpfs/0607-cluster/miniconda3/bin/activate /gpfs/0607-cluster/miniconda3/envs/double_rl
export PATH=/gpfs/0607-cluster/miniconda3/envs/double_rl/bin:$PATH
export HF_HOME=/gpfs/0607-cluster/HuggingFace
export PYTHONPATH=/gpfs/0607-cluster/chenhao/DoubleRL-VLA:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=4,5,6,7

accelerate launch --config_file ../config/sft.yaml \
    --num_processes 8  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard train_janus_no_siglip_encoder.py \
    --model_path deepseek-ai/Janus-Pro-1B \
    --data_path /gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/json/4tasks_train.json \
    --data_root /gpfs/0607-cluster/chenhao/data/rlbench/keyframe_fast_slow_chunk8_addlast_0806/for_rlds \
    --n_epochs 100 \
    --action_dim 7 \
    --train_bsz_per_gpu 8 \
    --learning_rate 1e-4 \
    --min_lr_ratio 0 \
    --weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --output_dir ../exp \
    --log_dir ../exp \
    --experiment_name action_only \
    --image_generation 0 \
    --run_name "janus_pro_no_siglip_encoder_1B_no_state_lr_1e-4" \


# FreedomIntelligence/Janus-4o-7B   deepseek-ai/Janus-Pro-7B