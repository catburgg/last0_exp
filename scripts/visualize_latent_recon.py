import argparse
import os
import math
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from einops import rearrange
from transformers import AutoModelForCausalLM

from train_wopc import SftDataset, create_component_indexes
from janus.models import VLChatProcessor
from janus.models.cosmos_tokenizer.image_lib import ImageTokenizer


def infer_latent_side(latent_size: int, num_frames: int) -> int:
    if latent_size % num_frames != 0:
        raise ValueError(f"latent_size ({latent_size}) must be divisible by num_frames ({num_frames})")
    tokens_per_frame = latent_size // num_frames
    side = math.isqrt(tokens_per_frame)
    if side * side != tokens_per_frame:
        raise ValueError(f"tokens_per_frame ({tokens_per_frame}) must be perfect square")
    return side


def denormalize_to_01(images: torch.Tensor, image_mean, image_std) -> torch.Tensor:
    mean = torch.tensor(image_mean, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(image_std, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def convert_to_cosmos_input(images: torch.Tensor, image_mean, image_std) -> torch.Tensor:
    pixels01 = denormalize_to_01(images.float(), image_mean, image_std)
    pixels256 = F.interpolate(pixels01, size=(256, 256), mode="bilinear", align_corners=False)
    return pixels256 * 2.0 - 1.0


def build_pred_latent_features(inferred_embeddings_all, batch_size, num_frames, target_side, model):
    hidden_size = inferred_embeddings_all.shape[-1]
    inferred_2d = rearrange(
        inferred_embeddings_all,
        "b (n h w) d -> (b n) d h w",
        n=num_frames,
        h=target_side,
        w=target_side,
    )
    upsampled = model.upsample_conv(inferred_2d)
    h_upsampled, w_upsampled = upsampled.shape[-2:]
    upsampled = upsampled.permute(0, 2, 3, 1).reshape(batch_size * num_frames, -1, hidden_size)
    pred_latent = model.gen_out_proj(model.gen_out_layer_norm(upsampled))
    pred_latent = pred_latent.view(batch_size * num_frames, h_upsampled, w_upsampled, pred_latent.shape[-1])
    pred_latent = pred_latent.permute(0, 3, 1, 2).contiguous()
    return pred_latent


class DummyAccelerator:
    @staticmethod
    def print(*args, **kwargs):
        return None


def probe_cosmos_scale_factor(latent_size: int, num_frames: int) -> int:
    target_side = infer_latent_side(latent_size, num_frames)
    cosmos_ckpt_dir = os.environ.get(
        "COSMOS_TOKENIZER_DIR",
        "/mnt/data/zhangxuheng/ckpt/pretrained/Cosmos-Tokenizer-CI8x8",
    )
    tokenizer = ImageTokenizer(
        checkpoint_enc=f"{cosmos_ckpt_dir}/encoder.jit",
        checkpoint_dec=f"{cosmos_ckpt_dir}/decoder.jit",
    )
    with torch.no_grad():
        probe = tokenizer.encode(torch.zeros(1, 3, 256, 256, dtype=torch.float32))
    cosmos_side = int(probe.shape[-1])
    if cosmos_side % target_side != 0:
        raise ValueError(f"cosmos_side ({cosmos_side}) is not divisible by target_side ({target_side})")
    return cosmos_side // target_side


def save_four_frame_grid(gt_images, pred_images, save_path):
    # Input shape: [4, 3, H, W], values expected in [-1, 1]
    gt_vis = ((gt_images + 1.0) / 2.0).clamp(0.0, 1.0)
    pred_vis = ((pred_images + 1.0) / 2.0).clamp(0.0, 1.0)
    panels = []
    for i in range(gt_vis.shape[0]):
        panels.append(gt_vis[i])
        panels.append(pred_vis[i])
    grid = vutils.make_grid(torch.stack(panels, dim=0), nrow=2, padding=4)
    vutils.save_image(grid, save_path)


def main():
    parser = argparse.ArgumentParser("Visualize latent reconstruction from checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint tfmr directory")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--latent_size", type=int, default=16, choices=[4, 16, 64])
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--action_chunk", type=int, default=8)
    parser.add_argument("--robot_state", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_frames = 4
    scale_factor = probe_cosmos_scale_factor(args.latent_size, num_frames)
    processor = VLChatProcessor.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        flow=True,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
        use_pointcloud=False,
        use_latent=1,
        cosmos_scale_factor=scale_factor,
        ignore_mismatched_sizes=True,
    ).to(device)
    model.eval()

    ds = SftDataset(args, processor, DummyAccelerator(), model)
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std

    with torch.no_grad():
        max_samples = min(args.num_samples, len(ds))
        for idx in range(max_samples):
            batch = ds.collate_fn([ds[idx]])
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            inputs_embeds = model.prepare_inputs_embeds(
                input_ids=batch["input_ids"],
                pixel_values=batch["encoder_pixel_values"],
                images_emb_mask=batch["images_emb_mask"],
                images_seq_mask=batch["images_seq_mask"],
            )
            noisy_actions = model.x_embedder(batch["noisy_actions"].to(inputs_embeds.dtype))
            timesteps = model.t_embedder(batch["timesteps"].to(inputs_embeds.dtype)).unsqueeze(1)
            inputs_embeds = torch.cat([inputs_embeds, timesteps, noisy_actions], dim=1)
            batch["attention_mask"] = torch.cat(
                [
                    batch["attention_mask"],
                    torch.ones((1, timesteps.shape[1]), dtype=torch.bool, device=device),
                    torch.ones((1, noisy_actions.shape[1]), dtype=torch.bool, device=device),
                ],
                dim=1,
            )

            fast_img_len = batch["fast_img_len"]
            action_len = 1 + 578 * fast_img_len + 1 + args.action_chunk
            latent_indexes, action_indexes = create_component_indexes(inputs_embeds.shape[1], action_len)

            helper_images = rearrange(batch["latent_pixel_values"], "b n c h w -> (b n) c h w")
            cosmos_input = convert_to_cosmos_input(helper_images, image_mean, image_std)
            gt_latent_features = model.cosmos_tokenizer.encode(cosmos_input).to(torch.float32)

            latent_start_id = processor.tokenizer.convert_tokens_to_ids("<|latent_start|>")
            latent_pad_id = processor.tokenizer.convert_tokens_to_ids("<|latent_pad|>")
            target_side = infer_latent_side(args.latent_size, num_frames)

            gt_latent_for_conv = gt_latent_features.to(model.gen_in_proj.weight.dtype)
            compressed_2d = model.downsample_conv(model.gen_in_proj(gt_latent_for_conv))
            compressed_flat = rearrange(compressed_2d, "bn d h w -> bn (h w) d")
            compressed_latent_embeds = rearrange(compressed_flat, "(b n) k d -> b (n k) d", b=1, n=num_frames)
            compressed_latent_embeds = compressed_latent_embeds.to(inputs_embeds.dtype)

            pad_mask = (batch["input_ids"] == latent_pad_id)
            extra_len = inputs_embeds.shape[1] - pad_mask.shape[1]
            if extra_len > 0:
                pad_mask = torch.cat(
                    [pad_mask, torch.zeros((pad_mask.shape[0], extra_len), dtype=torch.bool, device=device)],
                    dim=1,
                )
            inputs_embeds[pad_mask] = compressed_latent_embeds.reshape(-1, compressed_latent_embeds.shape[-1])

            outputs = model.language_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=batch["attention_mask"],
                return_dict=True,
                use_cache=False,
                latent_indexes=latent_indexes.to(device),
                action_indexes=action_indexes.to(device),
                use_latent=1,
            )
            hidden_states = outputs.last_hidden_state
            start_idx = (batch["input_ids"][0] == latent_start_id).nonzero(as_tuple=True)[0]
            pad_idxs = (batch["input_ids"][0] == latent_pad_id).nonzero(as_tuple=True)[0]
            pred_input_idxs = torch.cat([start_idx, pad_idxs[:-1]]) if len(pad_idxs) > 0 else start_idx
            inferred_embeddings_all = hidden_states[0, pred_input_idxs, :].unsqueeze(0)

            infer_for_recon = inferred_embeddings_all.to(model.upsample_conv.weight.dtype)
            pred_latent_features = build_pred_latent_features(
                infer_for_recon,
                batch_size=1,
                num_frames=num_frames,
                target_side=target_side,
                model=model,
            ).to(torch.float32)

            pred_pixels = model.cosmos_tokenizer.decode(pred_latent_features)
            gt_pixels = model.cosmos_tokenizer.decode(gt_latent_features)
            pred_pixels = rearrange(pred_pixels, "(b n) c h w -> b n c h w", b=1, n=num_frames)[0]
            gt_pixels = rearrange(gt_pixels, "(b n) c h w -> b n c h w", b=1, n=num_frames)[0]

            save_path = os.path.join(args.output_dir, f"sample_{idx:04d}.png")
            save_four_frame_grid(gt_pixels, pred_pixels, save_path)
            print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
