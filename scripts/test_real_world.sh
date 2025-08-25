source /gpfs/0607-cluster/miniconda3/bin/activate /gpfs/0607-cluster/miniconda3/envs/double_rl
export COPPELIASIM_ROOT=/gpfs/0607-cluster/chenhao/Programs/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7  ## for our machine
export PATH=/gpfs/0607-cluster/miniconda3/envs/double_rl/bin:$PATH
export HF_HOME=/gpfs/0607-cluster/HuggingFace
export PYTHONPATH=/gpfs/0607-cluster/chenhao/DoubleRL-VLA:$PYTHONPATH

# export CUDA_VISIBLE_DEVICES=4,5,6,7

python /gpfs/0607-cluster/chenhao/DoubleRL-VLA/scripts/test_real_world_no_siglip_encoder.py \
    --model-path "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/exp/action_only_0821_pick_place/janus_pro_no_siglip_encoder_1B_no_state_lr_1e-4/checkpoint-49-8850/tfmr" \
    --cuda 0 \
    --use_robot_state 0 \
    --max-steps 20 \
    --num-episodes 20 \
    --load-pointcloud 0 \
    --dataset-name 'real_world' \
    --image_generation 0 \
    --action-chunk 1
