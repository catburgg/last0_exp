from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from PIL import Image


def center_crop(img):
    img = Image.open(img).convert("RGB")
    # return img
    left = 160
    top = 60
    right = 450
    bottom = 350
    img_cropped = img.crop((left, top, right, bottom))
    img_cropped = img_cropped.resize((384, 384), Image.BICUBIC)
    return img_cropped

def estimate_background_hsv(img, center_frac=0.3):
    h, w = img.shape[:2]
    cx1 = int(w*(0.5-center_frac/2))
    cy1 = int(h*(0.5-center_frac/2))
    cx2 = int(w*(0.5+center_frac/2))
    cy2 = int(h*(0.5+center_frac/2))
    patch = img[cy1:cy2, cx1:cx2]
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # 使用中位数减少小物体影响
    bg_h = int(np.median(hsv_patch[:, :, 0]))
    bg_s = int(np.median(hsv_patch[:, :, 1]))
    bg_v = int(np.median(hsv_patch[:, :, 2]))
    return (bg_h, bg_s, bg_v)

def board_mask_from_hsv(img, bg_hsv, h_thresh=15, s_thresh=60, v_thresh=60):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    bg_h, bg_s, bg_v = bg_hsv
    # 处理 hue 环形差
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    dh = np.abs(h - bg_h)
    dh = np.minimum(dh, 180 - dh)
    ds = np.abs(s - bg_s)
    dv = np.abs(v - bg_v)
    # 背景（板子）像素：h/s/v 都接近背景值
    mask = (dh <= h_thresh) & (ds <= s_thresh) & (dv <= v_thresh)
    return mask.astype(np.uint8) * 255
    
def gen_prompt_box(img1, img2, prompt_img_save_path, mask_path, binary_mask):

    img1_cropped = center_crop(img1)
    img2_cropped = center_crop(img2)

    # 将PIL图像转换为OpenCV格式
    img1 = cv2.cvtColor(np.array(img1_cropped), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.array(img2_cropped), cv2.COLOR_RGB2BGR)

    # 统一尺寸（若已经一致，这里不会变）
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))

    # 估计每张图的背景 HSV（从中心区域）
    bg1 = estimate_background_hsv(img1, center_frac=0.3)
    bg2 = estimate_background_hsv(img2, center_frac=0.3)

    # print("Estimated background HSV image1:", bg1)
    # print("Estimated background HSV image2:", bg2)

    # 得到背景（板子）掩码：1 表示背景（绿板）
    board_mask1 = board_mask_from_hsv(img1, bg1, h_thresh=15, s_thresh=60, v_thresh=60)
    board_mask2 = board_mask_from_hsv(img2, bg2, h_thresh=15, s_thresh=60, v_thresh=60)

    # 计算新出现的区域：在第一张图是背景，而在第二张图不是背景（即被积木覆盖）
    new_mask = ((board_mask1 == 255) & (board_mask2 == 0)).astype(np.uint8) * 255
    # 形态学处理，去噪并填洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imwrite(binary_mask, new_mask)

    split_and_draw_by_watershed(img1, new_mask, prompt_img_save_path)



    # # 去掉小连通域，只保留较大的候选（例如面积>400像素）
    # contours, _ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # filtered_mask = np.zeros_like(new_mask)
    # min_area = 400
    # kept_boxes = []
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area >= min_area:
    #         x,y,ww,hh = cv2.boundingRect(cnt)
    #         # 忽略触及图像边缘的大区域（可能是框外板边）
    #         if x <= 2 or y <= 2 or (x+ww) >= (w-2) or (y+hh) >= (h-2):
    #             # 仍可保留，视情况而定；这里我们保留但打印提示
    #             pass
    #         cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
    #         kept_boxes.append((x,y,ww,hh))

    # print("Kept boxes (x,y,w,h):", kept_boxes)

    # # 在第一张图上绘制红色矩形（厚度3）并保存结果
    # result = img1.copy()
    # for (x,y,ww,hh) in kept_boxes:
    #     cv2.rectangle(result, (x,y), (x+ww, y+hh), (0,0,255), 3)  # BGR red

    # out_path = Path(prompt_img_save_path)
    # cv2.imwrite(str(out_path), result)
    # print("Saved result to", out_path)

    # 同时将新掩码可视化保存，便于检查
    vis_mask = cv2.cvtColor(new_mask, cv2.COLOR_GRAY2BGR)
    vis_overlay = cv2.addWeighted(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), 0.8, cv2.cvtColor(vis_mask, cv2.COLOR_BGR2RGB), 0.6, 0)
    cv2.imwrite(mask_path, cv2.cvtColor(vis_overlay, cv2.COLOR_RGB2BGR))


def split_and_draw_by_watershed(img_bgr, new_mask, out_path="/gpfs/0607-cluster/chenhao/DoubleRL-VLA-MOT/difference_hsv_mask_watershed.png",
                                min_area=200, dist_threshold_ratio=0.5):
    """
    img_bgr: 原始图像 (BGR)
    new_mask: 二值掩码 uint8 (0 or 255)，表示在 img1 为板子但在 img2 非板子（即被覆盖/新增）
    min_area: 忽略面积小于该值的连通域（像素）
    dist_threshold_ratio: 在距离变换上选取内核阈值的比例 (0~1)，越大越保守（更少种子）
    """
    h, w = new_mask.shape[:2]

    # 1) 去噪、填洞（可根据需要调节结构元大小）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 2) 距离变换（距离到最近背景），并找“确定前景(sure foreground)”的区域作为种子
    # 要求 mask 是 0/255
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # 阈值：可以用 dist.max() 的比例
    dt_max = dist.max() if dist.max() > 0 else 1.0
    _, sure_fg = cv2.threshold(dist, dist_threshold_ratio * dt_max, 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    # 3) 确定未知区域 (unknown = mask - sure_fg)
    sure_bg = cv2.dilate(mask, kernel, iterations=3)  # 背景扩张
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 4) 连接成 markers（连通域标号）
    num_markers, markers = cv2.connectedComponents(sure_fg)
    # connectedComponents 给 0 背景，1..N 前景 —— 我们需要把背景编号为 1, 前景编号从 2 开始（为了和 watershed 约定）
    markers = markers + 1
    markers[unknown==255] = 0

    # 5) 对彩色图像执行 watershed（需要 3 通道 BGR）
    img_w = img_bgr.copy()
    cv2.watershed(img_w, markers)  # 处理后，边界像素值为 -1

    # 6) 每个 marker (>1) 提取区域并画 bbox
    result = img_bgr.copy()
    boxes = []
    # markers 中 label 从 2..(num_markers+1) 对应各前景
    unique_labels = np.unique(markers)
    for lab in unique_labels:
        if lab <= 1:  # 忽略背景和边界(0,1)
            continue
        # 创建该 label 的二值 mask
        lab_mask = np.uint8(markers == lab) * 255
        # 进一步过滤小连通域（有时一个 label 可能包含很小区域）
        cnts, _ = cv2.findContours(lab_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x,y,ww,hh = cv2.boundingRect(cnt)
            # 如果需要，可以过滤掉接触图像边缘的大区域（视情况）
            # if x <=2 or y <=2 or (x+ww) >= (w-2) or (y+hh) >= (h-2): continue
            boxes.append((x,y,ww,hh))
            cv2.rectangle(result, (x,y), (x+ww, y+hh), (0,0,255), 3)  # BGR 红色

    # 如果没有分出 >1 个 box，可尝试降阈或改用更强的 erosion 再尝试
    cv2.imwrite(str(Path(out_path)), result)
    return result, boxes, markers, dist


def folder_2_json(data_root, task_lists, output_file, img_save_path):
    files = []
    for task in task_lists:
        print(f'Processing task: {task}')
        for file in sorted(os.listdir(f'{data_root}/{task}')):
            files.append(f'{file}')
    
    output_data = []
    
    
    for file in sorted(files)[-10:]:
        print('generating:', file, end=' \n')
        os.makedirs(f"{img_save_path}/{task}/{file}", exist_ok=True)

        
        for img_i in range(3,0,-1):
            box = gen_prompt_box(img1 = f"{data_root}/{task}/{file}/{img_i-1}.png",
                        img2 = f"{data_root}/{task}/{file}/{img_i}.png",
                        prompt_img_save_path = f"{img_save_path}/{task}/{file}/{img_i-1}_prompt.png",
                        mask_path=f"{img_save_path}/{task}/{file}/{img_i-1}_mask.png",
                        binary_mask=f"{img_save_path}/{task}/{file}/{img_i-1}_binary.png"
                        )
            episode_data = {
                'input_img': [f"{img_save_path}/{task}/{file}/{img_i}.png"],
                'output_img': f"{img_save_path}/{task}/{file}/{img_i-1}_prompt.png",
                'language_instruction': "Erase two Lego bricks on the given image and mark the erased areas with red boxes."
            }
            output_data.append(episode_data)

            center_crop(f"{data_root}/{task}/{file}/{img_i}.png").save(f"{img_save_path}/{task}/{file}/{img_i}.png")
        center_crop(f"{data_root}/{task}/{file}/0.png").save(f"{img_save_path}/{task}/{file}/0.png")

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)





### --batch-- ###

# data_root = "/gpfs/0607-cluster/guchenyang/Data/Janus/Raw"
# img_save_path = "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/lego"
# json_save_root = "/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/json"

# json_file = f'{json_save_root}/lego_small_1000_test.json'
# task_lists = [
# 'outputSmall_0123',
# ]

# folder_2_json(data_root, task_lists, json_file, img_save_path)


### --batch-- ###




### --single-- ###


# np.set_printoptions(threshold=np.inf)
# img1 = '/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/lego/outputSmall_0123/01998953-b845-716e-99ec-4b58e8c4affa/2.png'
# img2 = '/gpfs/0607-cluster/chenhao/DoubleRL-VLA/training_data/lego/outputSmall_0123/01998953-b845-716e-99ec-4b58e8c4affa/3.png'
# img_prompt = '/gpfs/0607-cluster/chenhao/DoubleRL-VLA-MOT/prompt.png'
# img_mask = '/gpfs/0607-cluster/chenhao/DoubleRL-VLA-MOT/mask.png'
# binary_mask = '/gpfs/0607-cluster/chenhao/DoubleRL-VLA-MOT/binary.png'

# gen_prompt_box(img1, img2, img_prompt, img_mask, binary_mask)

### --single-- ###