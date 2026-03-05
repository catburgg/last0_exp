#!/bin/bash

set -e

cd /path/to/last0
export COPPELIASIM_ROOT=/path/to/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

export PYTHONPATH=/path/to/last0:$PYTHONPATH

N=0
Xvfb :$N -screen 0 1024x768x24 &
export DISPLAY=:$N

models=("/path/to/your/model") 

tasks=("close_box" "close_laptop_lid" "sweep_to_dustpan" "phone_on_base" "toilet_seat_down" "close_fridge" "place_wine_at_rack_location" "water_plants" "take_umbrella_out_of_umbrella_stand" "take_frame_off_hanger")

for model in "${models[@]}"; do
  exp_name=$(echo "$model" | awk -F'/' '{print $(NF-3)"_"$(NF-2)"_"$(NF-1)}')
  for task in "${tasks[@]}"; do
    python test_rlbench.py \
      --model-path ${model} \
      --task-name ${task} \
      --exp-name ${exp_name} \
      --result-dir ${model} \
      --cuda $N \
      --use_robot_state 0 \
      --max-steps 10 \
      --num-episodes 20 \
      --load-pointcloud 0 \
      --dataset-name 'rlbench' \
      --use_latent 1 \
      --latent_size 12 \
      --compress_strategy average \
      --fs_ratio 4 \
      --result-dir /path/to/your/result \
      --action-chunk 1
  done
done

