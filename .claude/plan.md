# Plan: 搬运InternVLA的Cosmos VAE latent encode→decode pipeline，添加recon_loss

## 背景对比

### 当前pipeline (last0)
图片 → vision_model + aligner → [B*4, 576, 2048] (VLM特征) → depthwise conv压缩 → latent tokens → LLM预测 → sim_loss(cosine) + action_loss

### InternVLA pipeline
图片 → resize 256×256 → cosmos_tokenizer.encode → [B, 16, 32, 32] (VAE latent) → gen_in_proj(1×1 conv) → downsample_conv → tokens → LLM → upsample_conv → gen_out_proj → pred latent → MSE loss with GT latent

### 核心区别
当前用VLM特征做latent压缩，InternVLA用Cosmos VAE的latent space。需要把latent来源从VLM特征切换到Cosmos VAE，并添加upsample+decode的重建loss。

---

## 改动计划 (4个文件，最小改动)

### 1. `scripts/train_wopc.py` — 训练循环改造 (主要改动)

**新增args:**
- `--recon_loss_type`: 选项 `none` / `latent` / `pixel`，默认 `none`（向后兼容）
- `--recon_weight`: float，默认 1.0

**改造 `if args.use_latent:` 分支 (约lines 628-723):**

当前流程:
```
helper_images → vision_model + aligner → depthwise conv压缩 → LLM → sim_loss
```

新流程:
```python
# Step 1: Cosmos VAE encode
helper_images = rearrange(batch['latent_pixel_values'], "b n c h w -> (b n) c h w")
cosmos_input = convert_to_cosmos_input(helper_images, image_mean, image_std)
with torch.no_grad():
    gt_latent = model.cosmos_tokenizer.encode(cosmos_input)  # [B*4, 16, 32, 32]

# Step 2: Project + Downsample → compressed tokens
projected = model.gen_in_proj(gt_latent.float())        # [B*4, hidden_size, 32, 32]
compressed = model.downsample_conv(projected)            # [B*4, hidden_size, side, side]
compressed_flat = rearrange(compressed, "bn d h w -> bn (h w) d")
compressed_latent_embeds = rearrange(compressed_flat, "(b n) k d -> b (n k) d", b=bs, n=num_frames)

# Step 3: 注入LLM，forward，提取predicted embeddings (逻辑不变)
# ... sim_loss = 1 - cosine_similarity(inferred, compressed) ...

# Step 4: 第二次forward for action (逻辑不变)
# ... action_loss ...

# Step 5: recon_loss (新增)
if args.recon_loss_type != "none":
    pred_latent = build_pred_latent_features(inferred, bs, num_frames, target_side, model)
    if args.recon_loss_type == "latent":
        recon_loss = F.mse_loss(pred_latent, gt_latent.float())
    elif args.recon_loss_type == "pixel":
        with torch.no_grad():
            gt_pixels = cosmos_input  # [-1, 1]
        pred_pixels = model.cosmos_tokenizer.decode(pred_latent)
        recon_loss = F.mse_loss(pred_pixels, gt_pixels)
else:
    recon_loss = torch.tensor(0.0)

# Step 6: 总loss
loss = sim_loss + action_loss + recon_loss * args.recon_weight
```

**latent_size → scale_factor 映射:**
- latent_size=4 → 1×1/frame → sf=32
- latent_size=16 → 2×2/frame → sf=16
- latent_size=64 → 4×4/frame → sf=8

公式: `sf = 32 // isqrt(latent_size // num_frames)`

**wandb logging:** 新增 `recon_loss` 字段

### 2. `janus/models/modeling_vlm.py` — 模型定义微调

当前 `cosmos_scale_factor` 硬编码默认8。需要改为根据 `latent_size` 和 `num_frames` 动态计算:
- 新增config属性 `latent_size` 和 `num_frames`（默认16和4）
- `sf = 32 // isqrt(latent_size // num_frames)`
- 其余模块 (gen_in_proj, downsample_conv, upsample_conv, gen_out_layer_norm, gen_out_proj) 已经存在，只需确保sf正确

### 3. `scripts/train_wopc.sh` — 新增训练参数

```bash
--recon_loss_type latent   # 或 pixel 或 none
--recon_weight 1.0
```

### 4. `scripts/visualize_latent.py` — 新建可视化脚本

功能: 给定checkpoint + 训练数据，可视化:
- GT原始图片 (反归一化后)
- GT cosmos latent decode回的图片
- 模型预测的latent decode回的图片

流程:
1. 加载checkpoint和一条训练数据
2. 图片 → convert_to_cosmos_input → cosmos_tokenizer.encode → gt_latent
3. gt_latent → cosmos_tokenizer.decode → 反归一化 → GT重建图
4. 走一遍LLM forward → 提取predicted embeddings → build_pred_latent_features → pred_latent
5. pred_latent → cosmos_tokenizer.decode → 反归一化 → 预测重建图
6. 三张图并排保存

---

## 不改动的部分
- Cosmos tokenizer本身 (frozen, 不训练)
- VLM vision_model + aligner (仍用于理解分支，不用于latent分支)
- Action prediction逻辑 (flow matching部分不变)
- 数据加载 (latent_pixel_values已有)

## 解耦设计
- recon_loss_type=none 时行为完全等价于当前代码（只是latent来源从VLM特征变为Cosmos VAE）
- recon_loss的计算独立于sim_loss和action_loss
- 可视化脚本完全独立，不影响训练代码
