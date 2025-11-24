source /gpfs/0607-cluster/miniconda3/bin/activate /gpfs/0607-cluster/miniconda3/envs/double_rl
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7  ## for our machine
export PATH=/gpfs/0607-cluster/miniconda3/envs/double_rl/bin:$PATH
export HF_HOME=/gpfs/0607-cluster/HuggingFace
export PYTHONPATH=/gpfs/0607-cluster/chenhao/DoubleRL-VLA-MOT:$PYTHONPATH

# export CUDA_VISIBLE_DEVICES=4,5,6,7

python /gpfs/0607-cluster/chenhao/DoubleRL-VLA-MOT/scripts/test_mot.py \
    --model-path "/gpfs/0607-cluster/chenhao/DoubleRL-VLA-MOT/exp/simpler_image_action_mot/janus_pro_no_siglip_encoder_1B_1e-4/checkpoint-99-21900/tfmr" \
    --cuda 0 \
    --result-dir "/gpfs/0607-cluster/chenhao/DoubleRL-VLA-MOT" \
    --image_generation 1 \
    --dataset-name rlbench \