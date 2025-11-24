import numpy as np
from PIL import Image
import json
import os
import re
from scipy.spatial.transform import Rotation as R
# from num2words import num2words
from pathlib import Path

def folder_2_jsonl(data_root, jsonl_filename, task_lists):
    files = []
    for task in task_lists:
        print(f'Processing task: {task}')
        for file in sorted(os.listdir(f'{data_root}/{task}')):
            files.append(f'{data_root}/{task}/{file}')

    with open(jsonl_filename, 'w') as f:
        for file in sorted(files)[-10:]:
            print('generating:', file, end=' \n')

            # imgs = [f'{data_root}/{task}/{file}/{img}' for img in os.listdir(f'{data_root}/{task}/{file}') 
            #         if img.endswith('.png') and '_' in img]
            # imgs.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
            
            img_dir = Path(data_root) / task / file / 'images'
            imgs = sorted(
                [str(img_path) for img_path in img_dir.glob('front_*.png')],
                key=lambda x: int(Path(x).stem.split('_')[-1]),
                reverse=True
            )[:-1]

            imgs_left = sorted(
                [str(img_path) for img_path in img_dir.glob('left_*.png')],
                key=lambda x: int(Path(x).stem.split('_')[-1]),
                reverse=True
            )[:-1]

            imgs_right = sorted(
                [str(img_path) for img_path in img_dir.glob('right_*.png')],
                key=lambda x: int(Path(x).stem.split('_')[-1]),
                reverse=True
            )[:-1]

            for img_i in range(len(imgs)-1):
                num_left_word = num2words(6 - 2 * (img_i + 1)).upper()
                episode_data = {
                    'image_old': [imgs_left[img_i], imgs_right[img_i], imgs[img_i]],
                    'image_new': imgs[img_i+1],
                    # 'language_instruction': f"Step {img_i+1}: Remove TWO Lego bricks from the green board and place them beside it, leaving {num_left_word} bricks on the board."
                    'language_instruction': "Pick up the two Lego bricks from the green board and place them beside the board."
                }

                f.write(json.dumps(episode_data) + '\n')


def jsonl_2_json(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    output_data = []
    for line in lines:
        item = json.loads(line)
        
        new_item = {
            "input_prompt": item["language_instruction"],
            "input_image": item["image_old"],
            "output_image": item["image_new"],
        }
        
        output_data.append(new_item)
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)


def cal_stats(jsonl_filename):
    actions = []
    states = []
    episode_numbers = set()

    with open(jsonl_filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            actions.append(data['action'])
            states.append(data['state'])
            
            # 从image_old中提取episode编号
            match = re.search(r'episode(\d+)', data['image_old'])
            if match:
                episode_numbers.add(int(match.group(1)))

    actions = np.array(actions)
    states = np.array(states)

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
    state_mask = [True, True, True, True, True, True, False]

    action_stats = calculate_stats(actions, action_mask)
    state_stats = calculate_stats(states, state_mask)

    result = {
        "rlbench": {
            "action": action_stats,
            "state": state_stats,
            "num_transitions": len(actions),
            "num_trajectories": max(episode_numbers) + 1  # episode编号从0开始
        }
    }

    output_path = jsonl_filename.replace("train.jsonl", "train_statistics.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Statistics have been saved to {output_path}")




def crop_by_saturation(input_path, out_cropped, out_mask, s_thresh=0.15, pad=8):
    # 打开并确保 RGB
    img = Image.open(input_path).convert("RGB")
    W, H = img.size

    # 转 HSV 并提取 S 通道（PIL 的 HSV 范围在 0..255）
    hsv = np.array(img.convert("HSV"))
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    # 构造掩码：饱和度超过阈值 或 明显不够亮的像素（v < 0.95）都视为前景
    mask = ((s > s_thresh) | (v < 0.4)).astype(np.uint8)

    # 行/列投影，找到有前景像素的行列
    rows = np.where(mask.sum(axis=1) > 0)[0]
    cols = np.where(mask.sum(axis=0) > 0)[0]

    if rows.size and cols.size:
        y0, y1 = int(rows[0]), int(rows[-1] + 1)
        x0, x1 = int(cols[0]), int(cols[-1] + 1)
    else:
        # 万一投影失败，回退到不裁剪
        x0, y0, x1, y1 = 0, 0, W, H

    # 加 padding 并裁剪到图像边界
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad)
    y1 = min(H, y1 + pad)

    cropped = img.crop((x0, y0, x1, y1)).resize((384, 384), Image.BICUBIC)
    cropped.save(out_cropped)

    # 保存掩码图（便于调参）
    mask_img = Image.fromarray((mask * 255).astype('uint8'))
    mask_img.save(out_mask)

    info = {
        "input": input_path,
        "out_cropped": out_cropped,
        "out_mask": out_mask,
        "s_threshold": s_thresh,
        "pad": pad,
        "crop_box": (x0, y0, x1, y1),
        "crop_size": (x1 - x0, y1 - y0),
        "orig_size": (W, H),
    }
    return info

def center_crop(img_path, out_path):
    img = Image.open(img_path).convert("RGB")

    # img.resize((384, 384), Image.BICUBIC).save(out_path)

    left = 160
    top = 60
    right = 450
    bottom = 350
    
    img_cropped = img.crop((left, top, right, bottom))#.resize((384, 384), Image.BICUBIC)
    img_cropped.save(out_path)

    
    # w, h = img_cropped.size

    # new_h = h // 2
    # top = (h - new_h) // 2 - 30
    # bottom = top + new_h
    # img_cropped = img_cropped.crop((0, top, w, bottom)).resize((384, 384), Image.BICUBIC)
    # img_cropped.save(out_path)



if __name__ == "__main__":

    # # ######## ---------main---------- #########

    # data_root = "/gpfs/0607-cluster/guchenyang/Data/Janus/demo3_hand_1500"
    # json_save_root = "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/json"
    # jsonl_filename = f'{json_save_root}/lego_new_1500_three_test.jsonl'
    # json_file = f'{json_save_root}/lego_new_1500_three_test.json'

    # task_lists = [
    # 'demo3_hand_0927',
    # 'demo3_hand_0928',
    # ]

    # folder_2_jsonl(data_root, jsonl_filename, task_lists)
    # # cal_stats(jsonl_filename)
    # jsonl_2_json(jsonl_filename, json_file)

    # ######## ---------main---------- #########



    # # -------------------------
    # # 配置（按需修改）
    # INPUT_PATH = "/gpfs/0607-cluster/guchenyang/Data/Janus/output_small_base/019985d0-7abc-77ea-b99e-c99ac54b5f46/keyframe_149.png"
    # OUT_CROPPED_PATH = "/gpfs/0607-cluster/keyframe_10_cropped_final.png"
    # OUT_MASK_PATH = "mask_used_150.png"
    # S_THRESHOLD = 0.15                         # 饱和度阈值（0..1），经验值 0.15
    # PAD = 8                                    # 裁剪框额外扩张像素
    

    # if not os.path.exists(INPUT_PATH):
    #     raise FileNotFoundError(f"输入文件不存在: {INPUT_PATH}")

    # info = crop_by_saturation(INPUT_PATH, OUT_CROPPED_PATH, OUT_MASK_PATH,
    #                           s_thresh=S_THRESHOLD, pad=PAD)
    # print("裁剪完成：")
    # for k, v in info.items():
    #     print(f"  {k}: {v}")
    # print("\n输出文件：")
    # print("  裁剪图:", OUT_CROPPED_PATH)
    # -------------------------



    # # ---------------------------- # 
    img_list = [
    "/gpfs/0607-cluster/guchenyang/Data/Janus/Raw/output_small_base/019985d0-f68c-7041-b0a5-ce74bf64c67f/keyframe_142.png",
    "/gpfs/0607-cluster/guchenyang/Data/Janus/Raw/output_small_base/019985d0-f68c-7041-b0a5-ce74bf64c67f/keyframe_208.png",
    ]
    i = 2
    for img in img_list:
        center_crop(img, f"/gpfs/0607-cluster/chenhao/DoubleRL-VLA-MOT/keyframe_{i}_cropped_final.png")
        i += 1
    # # ---------------------------- # 
