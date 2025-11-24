# import numpy as np
# from PIL import Image
# import json
# import os
# import re
# from scipy.spatial.transform import Rotation as R


# def npz_2_jsonl(img_save_root, jsonl_filename, task_lists, npz_file):
#     num = 0
#     with open(jsonl_filename, 'w') as f:
        
#         for task in task_lists:
#             print(f'Processing task: {task}')

#             if not os.path.exists(f'{img_save_root}/{task}'):
#                 os.mkdir(f'{img_save_root}/{task}')

#             for file in os.listdir(npz_file):
#                 if not file.endswith('.npz'): 
#                     continue

#                 print(num, '  generating:', file, end=' ')

#                 episode = np.load(f'{npz_file}/{file}', allow_pickle=True)
#                 episode = episode['arr_0'].item()

#                 file = file.replace('.npz', '')
#                 episode_length = len(episode['image'])
#                 print('episode_length:', episode_length)

#                 if not os.path.exists(f'{img_save_root}/{task}/{file}'):
#                     os.mkdir(f'{img_save_root}/{task}/{file}')

#                     for i in range(episode_length):
#                         image = episode['image'][i]
#                         # image = Image.fromarray(image_array)
#                         image.resize((384, 384), Image.BICUBIC).save(f'{img_save_root}/{task}/{file}/image{i}.png')

#                 for i in range(1, episode_length-1):
#                     if np.isclose(episode['action'][i][:6].sum(), 0.0) and i!=episode_length-2:
#                         episode['action'][i+1][-1] = episode['action'][i][-1]
#                         continue
                    
#                     action = episode['action'][i]
#                     image_old = f'{img_save_root}/{task}/{file}/image{i}.png'
#                     image_new = f'{img_save_root}/{task}/{file}/image{i+1}.png'

#                     # Create dictionary for this step
#                     episode_data = {
#                         'image_old': image_old,
#                         'image_new': image_new,
#                         'action': action.tolist(),
#                         'state': [],
#                         'language_instruction': episode['instruction'][0]
#                     }

#                     f.write(json.dumps(episode_data) + '\n')

#                 num += 1

# def jsonl_2_json(input_file, output_file):
#     with open(input_file, 'r') as f:
#         lines = f.readlines()

#     output_data = []
#     for line in lines:
#         item = json.loads(line)
        
#         new_item = {
#             "input_prompt": item["language_instruction"],
#             "input_image": [item["image_old"]],
#             "input_image_resolution": [384, 384],
#             "output_image": item["image_new"],
#             "output_image_resolution": [384, 384],
#             "action": item["action"],
#             "state": item["state"]
#         }
        
#         output_data.append(new_item)
    
#     with open(output_file, 'w') as f:
#         json.dump(output_data, f, indent=2)


# def cal_stats(jsonl_filename):
#     actions = []
#     states = []
#     episode_ids = set()

#     with open(jsonl_filename, 'r') as f:
#         for line in f:
#             data = json.loads(line)
#             actions.append(data['action'])
#             states.append(data['state'])

#             image_old = data['image_old']
#             episode_id = os.path.basename(os.path.dirname(image_old))
#             episode_ids.add(episode_id)

#     actions = np.array(actions)
#     states = np.array(states)

#     def calculate_stats(data, mask=None):
#         if mask is None:
#             mask = [True] * data.shape[1]
        
#         stats = {
#             'mean': np.mean(data, axis=0).tolist(),
#             'std': np.std(data, axis=0).tolist(),
#             'max': np.max(data, axis=0).tolist(),
#             'min': np.min(data, axis=0).tolist(),
#             'q01': np.quantile(data, 0.01, axis=0).tolist(),
#             'q99': np.quantile(data, 0.99, axis=0).tolist(),
#             'mask': mask,
#         }
#         return stats

#     action_mask = [True, True, True, True, True, True, False]
#     state_mask = [True, True, True, True, True, True, False]

#     action_stats = calculate_stats(actions, action_mask)
#     state_stats = calculate_stats(states, state_mask)

#     result = {
#         "rlbench": {
#             "action": action_stats,
#             "state": state_stats,
#             "num_transitions": len(actions),
#             "num_trajectories": len(episode_ids)  # episode编号从0开始
#         }
#     }

#     output_path = jsonl_filename.replace(".jsonl", "_statistics.json")
#     with open(output_path, 'w') as f:
#         json.dump(result, f, indent=2)

#     print(f"Statistics have been saved to {output_path}")



# ######## ---------main---------- #########

# img_save_root = "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/simpler"
# json_save_root = "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/json"
# jsonl_filename = f'{json_save_root}/simpler_train_PutOnPlateInScene25Main-v3_16400_freq5.jsonl'
# json_file = f'{json_save_root}/simpler_train_PutOnPlateInScene25Main-v3_16400_freq5.json'

# npz_file = "/gpfs/0607-cluster/chenhao/RL4VLA/sft_data/PutOnPlateInScene25Main-v3/16400_freq5/data"

# task_lists = [
#   'PutOnPlateInScene25Main-v3_16400_freq5',
# ]

# npz_2_jsonl(img_save_root, jsonl_filename, task_lists, npz_file)
# cal_stats(jsonl_filename)
# jsonl_2_json(jsonl_filename, json_file)






import numpy as np
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import time

def convert_npz_to_npy(npz_file):
    npy_path = npz_file.replace('.npz', '.npy')
    
    try:
        with np.load(npz_file, allow_pickle=True) as data:
            episode_data = data['arr_0'].item()
        
        L = len(episode_data['action'])
        episode_list = []

        for i in range(1, L):
            if np.isclose(episode_data['action'][i][:6].sum(), 0.0) and i < L-2:
                episode_data['action'][i+1][-1] = episode_data['action'][i][-1]
                continue

            img = episode_data['image'][i]
            if isinstance(img, Image.Image):
                img = np.array(img.resize((384, 384), Image.BICUBIC))

            step_dict = {
                'image': img,
                'instruction': str(episode_data['instruction'][0]),
                'action': episode_data['action'][i],
                'info': episode_data['info'][i],
            }
            episode_list.append(step_dict)

        np.save(npy_path, episode_list)
        return npz_file, len(episode_list), "success"
    except Exception as e:
        return npz_file, 0, f"error: {e}"



def process_single_file(file):
    episode = np.load(file, allow_pickle=True)
    if len(episode)<=3: return None
    actions = np.stack([step["action"] for step in episode[:-1]])
    L = len(episode) - 1
    json_items = [{"idx": idx, "npy": file} for idx in range(L)]
    return actions, json_items, L


def process_npy_and_save(data_root, jsonl_path, json_path, stats_path, num_workers=16):
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    npy_files = []
    for p in data_root:
        npy_files += [os.path.join(p,f) for f in os.listdir(p) if f.endswith(".npy")]

    episode_num = len(npy_files)
    print(f"Found {episode_num} npy files.")


    all_actions = []
    total_transitions = 0
    t0 = time.time()
    valid_num = 0
    with open(jsonl_path, "w") as jsonl_file:
        json_items = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for result in tqdm(executor.map(lambda f: process_single_file(f), npy_files),
                            total=len(npy_files), desc="Processing npy", ncols=100):
                if result is None:
                    continue
                actions, items, L = result

                all_actions.append(actions)
                total_transitions += L
                valid_num += 1

                for item in items:
                    jsonl_file.write(json.dumps(item) + "\n")
                json_items.extend(items)
    print(f"Saved jsonl index to {jsonl_path}")


    # 直接用收集好的 json_items 写 json
    with open(json_path, "w") as f:
        json.dump(json_items, f)
    print(f"Saved json to {json_path}")

    # 计算统计信息
    all_actions = np.vstack(all_actions)
    print(f"Loaded {total_transitions} transitions in {(time.time()-t0):.2f}s, start computing statistics...")

    stats = {
        'mean': np.mean(all_actions, axis=0).tolist(),
        'std': np.std(all_actions, axis=0).tolist(),
        'max': np.max(all_actions, axis=0).tolist(),
        'min': np.min(all_actions, axis=0).tolist(),
        'q01': np.quantile(all_actions, 0.01, axis=0).tolist(),
        'q99': np.quantile(all_actions, 0.99, axis=0).tolist(),
        'mask': [True, True, True, True, True, True, False],
    }

    result = {
        "oxe_pretrain": {
            "action": stats,
            "num_transitions": total_transitions,
            "num_trajectories": episode_num,
        }
    }

    with open(stats_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Statistics have been saved to {stats_path}")
    print(f"Processed {valid_num}/{episode_num} valid npy files.")



if __name__ == "__main__":

    json_save_root = "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/json"
    jsonl_filename = f'{json_save_root}/simpler_train_PutOnPlateInScene25Main-v3_16400_freq5_new.jsonl'
    json_file = f'{json_save_root}/simpler_train_PutOnPlateInScene25Main-v3_16400_freq5_new.json'
    stats_file = os.path.join(json_save_root, f"simpler_train_PutOnPlateInScene25Main-v3_16400_freq5_statistics_new.json")
    data_root = ['/gpfs/0607-cluster/chenhao/RL4VLA/sft_data/PutOnPlateInScene25Main-v3/16400_freq5/data']

    need_npz2npy = 0
    if need_npz2npy:
        npz_files = []
        for p in data_root:
            npz_files += [os.path.join(p, f) for f in os.listdir(p) if f.endswith('.npz')]
        num_workers = 32
        results = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(convert_npz_to_npy, f) for f in npz_files]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Converting npz to npy"):
                results.append(fut.result())

        for fname, length, status in results:
            print(f"{fname}: {status}, length={length}")


    process_npy_and_save(data_root, jsonl_filename, json_file, stats_file, num_workers=32)
