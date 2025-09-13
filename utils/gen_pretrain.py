import numpy as np
from PIL import Image
import json
import os
import re
from scipy.spatial.transform import Rotation as R

def unique_euler_xyz_rad(angles, range_style="2pi"):
    """
    输入: 欧拉角 (xyz 顺序)，弧度制 (任意范围, 可正可负, 可超过 2π)
    输出: 欧拉角 (xyz 顺序)，弧度制，严格唯一表示

    参数:
        precision: 保留小数位数
        range_style: "negpi" -> (-π, π], "2pi" -> [0, 2π)
    """
    # 输入是弧度
    rot = R.from_euler('xyz', angles, degrees=False)
    
    # 转回 xyz (弧度制)
    euler = rot.as_euler('xyz', degrees=False)
    
    # wrap 到 (-π, π]
    euler = (euler + np.pi) % (2 * np.pi) - np.pi
    
    # 约束: y ∈ [-π/2, π/2]
    if euler[1] > np.pi/2:
        euler[1] = np.pi - euler[1]
        euler[0] += np.pi
        euler[2] += np.pi
    elif euler[1] < -np.pi/2:
        euler[1] = -np.pi - euler[1]
        euler[0] += np.pi
        euler[2] += np.pi
    
    # 再 wrap 一次
    euler = (euler + np.pi) % (2 * np.pi) - np.pi
    
    # 如果要求 [0, 2π)，再转换
    if range_style == "2pi":
        euler = euler % (2 * np.pi)
    
    return euler

def npy_2_jsonl(data_root, jsonl_filename, task_lists):
    episode_num = 0

    with open(jsonl_filename, 'w') as f:
        for task in task_lists:
            print(f'Processing task: {task}')
            for file in os.listdir(f'{data_root}/{task}'):
                if not file.endswith('.npy'): 
                    continue

                print('generating:', file, end=' ')
                episode = np.load(f'{data_root}/{task}/{file}', allow_pickle=True)
                episode_num += 1

                episode_length = len(episode)
                print('episode_length:', episode_length)

                for i in range(episode_length-1):
                    episode_data = {
                        'idx': i,
                        'npy': f'{task}/{file}',
                    }
                    f.write(json.dumps(episode_data) + '\n')
    return episode_num

def jsonl_2_json(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    output_data = []
    for line in lines:
        item = json.loads(line)
        new_item = {
            "idx": item["idx"],
            "npy": item['npy'],
        }
        output_data.append(new_item)
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f)


def cal_stats(data_root, jsonl_filename, episode_num):
    actions = []

    with open(jsonl_filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            episode = np.load(f"{data_root}/{data['npy']}", allow_pickle=True)

            action = episode[data['idx']]['action']
            actions_7d = [action[i*7:(i+1)*7] for i in range(8)]
            delta_positions = [action[:3] for action in actions_7d]
            abs_rots = [action[3:6] for action in actions_7d]
            grippers = [action[-1] for action in actions_7d]
            delta_position_total = np.sum(delta_positions, axis=0)
            action = np.concatenate([delta_position_total, abs_rots[-1], [grippers[-1]]])
            action[3:6] = unique_euler_xyz_rad(action[3:6])

            actions.append(action)

    actions = np.array(actions)

    def calculate_stats(data, mask=None):
        if mask is None:
            mask = [True] * data.shape[1]
        
        stats = {
            'mean': np.mean(data, axis=0).tolist(),
            'std': np.std(data, axis=0).tolist(),
            'max': np.max(data, axis=0).tolist(),
            'min': np.min(data, axis=0).tolist(),
            'q01': np.quantile(data, 0.01, axis=0).tolist(),
            'q99': np.quantile(data, 0.99, axis=0).tolist(),
            'mask': mask,
        }
        return stats

    action_mask = [True, True, True, True, True, True, False]
    action_stats = calculate_stats(actions, action_mask)

    result = {
        "rlbench": {
            "action": action_stats,
            "num_transitions": len(actions),
            "num_trajectories": episode_num,
        }
    }

    output_path = jsonl_filename.replace("train.jsonl", "train_statistics.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Statistics have been saved to {output_path}")



######## ---------main---------- #########

data_root = "/gpfs/0607-cluster/chenhao/data/rlbench/keyframe_fast_slow_chunk8_addlast_0806/for_rlds"
json_save_root = "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/json"
jsonl_filename = f'{json_save_root}/4tasks_train.jsonl'
json_file = f'{json_save_root}/4tasks_train.json'

task_lists = [
  'close_box',
#   'close_fridge',
  'close_laptop_lid',
  'phone_on_base',
#   'place_wine_at_rack_location',
  'sweep_to_dustpan',
#   'take_frame_off_hanger',
#   'take_umbrella_out_of_umbrella_stand',
#   'toilet_seat_down',
#   'water_plants'
]

episode_num = npy_2_jsonl(data_root, jsonl_filename, task_lists)
cal_stats(data_root, jsonl_filename, episode_num)
jsonl_2_json(jsonl_filename, json_file)