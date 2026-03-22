import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import sys
# 如果报错找不到 janus，请确保当前环境变量 PYTHONPATH 包含了 janus 的路径
sys.path.append("/mnt/dataset/share_code/code/last0_exp")
from janus.models.cosmos_tokenizer.image_lib import ImageTokenizer

def process_and_save_latents(
    src_dir: str, 
    tgt_dir: str, 
    tokenizer_dir: str, 
    device: str = "cuda"
):
    print("Loading Cosmos ImageTokenizer...")
    # 参考 train_wopc.py 的加载方式
    tokenizer = ImageTokenizer(
        checkpoint_enc=os.path.join(tokenizer_dir, "encoder.jit"),
        checkpoint_dec=os.path.join(tokenizer_dir, "decoder.jit"),
    ).to(device)
    
    # 冻结模型
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad = False

    # 图片预处理逻辑 (参考 train_wopc.py 中的 convert_to_cosmos_input)
    # Cosmos VAE 要求的输入大小为 256x256，并且数值范围通常是 [-1, 1]
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(), # 变为 [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 变为 [-1, 1]
    ])

    task_lists = [
        # 'libero_spatial_no_noops',
        # 'libero_goal_no_noops',
        # 'libero_object_no_noops',
        'libero_10_no_noops'
    ]

    for task in task_lists:
        task_src_dir = os.path.join(src_dir, task)
        task_tgt_dir = os.path.join(tgt_dir, task)

        if not os.path.exists(task_src_dir):
            print(f"Warning: Source path {task_src_dir} does not exist. Skipping.")
            continue
        
        # 确保目标 task 路径存在
        os.makedirs(task_tgt_dir, exist_ok=True)
        print(f"Processing task: {task}")

        # 获取所有 episode 目录
        episodes = [d for d in os.listdir(task_src_dir) if os.path.isdir(os.path.join(task_src_dir, d))]
        
        for ep in tqdm(episodes, desc=f"Episodes in {task}"):
            ep_src_dir = os.path.join(task_src_dir, ep)
            ep_tgt_dir = os.path.join(task_tgt_dir, ep)
            
            os.makedirs(ep_tgt_dir, exist_ok=True)

            # 遍历 episode 中所有的 png 图片，这会自动包含普通的图片以及 padding_black_*.png
            for img_file in os.listdir(ep_src_dir):
                if not img_file.endswith('.png'):
                    continue

                img_path = os.path.join(ep_src_dir, img_file)
                base_name = img_file.replace('.png', '')
                save_path = os.path.join(ep_tgt_dir, f"{base_name}.pt")
                
                # 如果已经处理过（支持断点续传），则跳过
                if os.path.exists(save_path):
                    continue

                try:
                    # 读取图片并做 Transform
                    image = Image.open(img_path).convert('RGB')
                    pixel_values = preprocess(image).unsqueeze(0).to(device, dtype=torch.bfloat16) # shape: (1, 3, 256, 256)

                    # 提取 Latent 
                    with torch.no_grad():
                        # 与 train 代码一样，使用 encode 方法
                        latent = tokenizer.encode(pixel_values) # shape 通常为 (1, C, H, W)
                    
                    # 可以在保存时去掉 batch 维度，仅保留 (C, H, W)
                    latent_to_save = latent.squeeze(0).cpu().clone()
                    
                    # 保存为 .pt 文件
                    torch.save(latent_to_save, save_path)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    SRC_IMAGES_DIR = "/mnt/dataset/share_code/dataset/libero_training_data/libero_images"
    TGT_LATENT_DIR = "/mnt/dataset/share_code/dataset/libero_training_data/cosmos_latent_GT"
    TOKENIZER_DIR = "/mnt/dataset/share_code/hf_cache/Cosmos-0.1-Tokenizer-CI8x8"
    
    # 确保保存根目录已创建
    os.makedirs(TGT_LATENT_DIR, exist_ok=True)

    print(f"Start generating latent representations...")
    print(f"Source Directory: {SRC_IMAGES_DIR}")
    print(f"Target Directory: {TGT_LATENT_DIR}")

    # 以 bfloat16 的混合精度加载模型会更贴近你在 train_wopc.py 中的数据格式
    torch.set_default_dtype(torch.bfloat16)

    process_and_save_latents(
        src_dir=SRC_IMAGES_DIR,
        tgt_dir=TGT_LATENT_DIR,
        tokenizer_dir=TOKENIZER_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Generation complete.")