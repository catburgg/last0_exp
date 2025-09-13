source /gpfs/0607-cluster/miniconda3/bin/activate /gpfs/0607-cluster/miniconda3/envs/double_rl
export COPPELIASIM_ROOT=/gpfs/0607-cluster/chenhao/Programs/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7  ## for our machine
export PATH=/gpfs/0607-cluster/miniconda3/envs/double_rl/bin:$PATH
export HF_HOME=/gpfs/0607-cluster/HuggingFace
export PYTHONPATH=/gpfs/0607-cluster/chenhao/DoubleRL-VLA:$PYTHONPATH

# export CUDA_VISIBLE_DEVICES=4,5,6,7

N=0
Xvfb :$N -screen 0 1024x768x24 &
export DISPLAY=:$N

models=("/gpfs/0607-cluster/chenhao/DoubleRL-VLA/exp/action_only/janus_pro_no_siglip_encoder_1B_no_state_lr_1e-4/checkpoint-99-2900/tfmr")
# tasks=("close_box" "close_laptop_lid")
# tasks=("toilet_seat_down" "sweep_to_dustpan")
# tasks=("close_fridge" "place_wine_at_rack_location")
# tasks=("water_plants" "phone_on_base")
# tasks=("take_umbrella_out_of_umbrella_stand" "take_frame_off_hanger")

tasks=("close_laptop_lid" "sweep_to_dustpan" "phone_on_base" "close_box")

# tasks=("close_box")

for model in "${models[@]}"; do
  exp_name=$(echo "$model" | awk -F'/' '{print $(NF-3)"_"$(NF-2)"_"$(NF-1)}')
  for task in "${tasks[@]}"; do
    python /gpfs/0607-cluster/chenhao/DoubleRL-VLA/scripts/test_rlbench_no_siglip_encoder.py \
      --model-path ${model} \
      --task-name ${task} \
      --exp-name ${exp_name} \
      --replay-or-predict 'predict' \
      --result-dir ${model} \
      --cuda $N \
      --use_robot_state 0 \
      --max-steps 10 \
      --num-episodes 20 \
      --load-pointcloud 0 \
      --dataset-name 'rlbench' \
      --image_generation 0 \
      --result-dir /gpfs/0607-cluster/chenhao/test_results/ch_test_0827 \
      --action-chunk 1
  done
done