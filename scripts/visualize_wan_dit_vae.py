#!/usr/bin/env python3
"""
WAN VAE + Cosmos DiT from SftDataset. Per sample:
  sample_XXXX.png — GT | recon | decode_mu (full DiT + EDM precondition).
  With --dit_block_viz: blocks.png, blocks_pca_on_gt.png, blocks_heatmap_on_gt.png (same grid).
  With --dit_probe_final_after_blocks K1 K2 …: probe_final_decode.png — row = “after K DiT blocks then
  same final_layer”, col = frame (WAN decode_mu x0_8ch + μ[8:16]; shallow K is a rough probe, not trained match).
"""

from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for _p in (REPO_ROOT, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

from train_wopc import SftDataset


class DummyAccelerator:
    @staticmethod
    def print(*args, **kwargs):
        return None


from janus.models import VLChatProcessor
from janus.models.cosmos_tokenizer.dit_lib import (
    CROP_SIZE,
    NUM_DIT_BLOCKS,
    CosmosDiTEarlyExit,
    CosmosDiTFullHead,
    FULL_PATCH_TEMPORAL,
    pad_latent_temporal,
)
from janus.models.cosmos_tokenizer.wan_vae_lib import WanVAECodec, WanVAEEncoder


def denormalize_to_01(images: torch.Tensor, image_mean, image_std) -> torch.Tensor:
    mean = torch.tensor(image_mean, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(image_std, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def latent_pixel_values_to_wan_m11(
    latent_b_n_c_hw: torch.Tensor,
    image_mean,
    image_std,
) -> torch.Tensor:
    b, n, c, h, w = latent_b_n_c_hw.shape
    x = rearrange(latent_b_n_c_hw.float(), "b n c h w -> (b n) c h w")
    x01 = denormalize_to_01(x, image_mean, image_std)
    if x01.shape[-2:] != (240, 240):
        x01 = F.interpolate(x01, size=(240, 240), mode="bilinear", align_corners=False)
    return rearrange((x01 * 2.0 - 1.0), "(b n) c h w -> b n c h w", b=b, n=n)


def center_crop_mu_bcthw(mu: torch.Tensor, crop: int = CROP_SIZE) -> torch.Tensor:
    _, _, _, h, w = mu.shape
    if h == crop and w == crop:
        return mu
    h0, w0 = (h - crop) // 2, (w - crop) // 2
    return mu[:, :, :, h0 : h0 + crop, w0 : w0 + crop]


def _spatial_hw3(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 5:
        t = t.squeeze(1)
    if t.dim() == 4:
        t = t[0]
    return t


def patch_mean_abs_energy_hw(spatial: torch.Tensor) -> torch.Tensor:
    x = _spatial_hw3(spatial).float()
    e = x.abs().mean(dim=-1)
    e = e - e.amin()
    return (e / (e.amax() - e.amin() + 1e-8)).clamp(0, 1)


def blend_rgb_overlay(gt_3hw: torch.Tensor, top_3hw: torch.Tensor, alpha: float) -> torch.Tensor:
    a = float(alpha)
    return ((1.0 - a) * gt_3hw.float() + a * top_3hw.float()).clamp(0.0, 1.0)


def scalar_hw_to_heatmap_rgb(scalar_hw: torch.Tensor, out_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
    s4 = scalar_hw.float().to(device).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(s4, size=out_hw, mode="bilinear", align_corners=False).squeeze(0).squeeze(0).clamp(0, 1)
    r = torch.clamp(1.5 - (4 * t - 3).abs(), 0, 1)
    g = torch.clamp(1.5 - (4 * t - 2).abs(), 0, 1)
    b = torch.clamp(1.5 - (4 * t - 1).abs(), 0, 1)
    return torch.stack([r, g, b], dim=0)


def dit_spatial_pca_rgb(spatial: torch.Tensor, out_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
    x = _spatial_hw3(spatial).float().to(device)
    H, W, D = x.shape
    X = x.reshape(H * W, D)
    n, d = X.shape
    X = X - X.mean(dim=0, keepdim=True)
    if n < 2:
        rgb = torch.zeros(3, H, W, device=device, dtype=torch.float32)
    else:
        U, S, _ = torch.linalg.svd(X, full_matrices=False)
        k = min(3, U.shape[1], S.shape[0])
        proj = U[:, :k] * S[:k].unsqueeze(0)
        if k < 3:
            proj = F.pad(proj, (0, 3 - k))
        proj = proj[:, :3].reshape(H, W, 3)
        rgb = proj.permute(2, 0, 1).contiguous()
        for c in range(3):
            ch = rgb[c]
            rgb[c] = (ch - ch.amin()) / (ch.amax() - ch.amin() + 1e-8)
    return F.interpolate(rgb.unsqueeze(0), size=out_hw, mode="bilinear", align_corners=False).squeeze(0)


def _save_panel_grid(panels_nchw: torch.Tensor, ncol: int, save_path: str, padding: int = 4) -> None:
    t = panels_nchw.float().clamp(0, 1).cpu()
    n, c, h, w = t.shape
    nrow = (n + ncol - 1) // ncol
    pad = padding
    gh, gw = nrow * h + max(0, nrow - 1) * pad, ncol * w + max(0, ncol - 1) * pad
    grid = torch.zeros(c, gh, gw, dtype=t.dtype)
    for i in range(n):
        r, col = divmod(i, ncol)
        y, x0 = r * (h + pad), col * (w + pad)
        grid[:, y : y + h, x0 : x0 + w] = t[i]
    if c == 1:
        grid = grid.repeat(3, 1, 1)
    arr = (grid.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(save_path)


def wan_recon_per_frame(codec: WanVAECodec, wan_m11_bnchw: torch.Tensor, w_dtype: torch.dtype) -> torch.Tensor:
    b, n, _, _, _ = wan_m11_bnchw.shape
    x = wan_m11_bnchw.to(device=codec.encoder.conv1.weight.device, dtype=w_dtype)
    recons = []
    for i in range(n):
        xi = x[:, i : i + 1].transpose(1, 2)
        recons.append(codec.recon(xi.float()))
    return torch.cat(recons, dim=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--latent_size", type=int, default=16, choices=[4, 16, 64])
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--action_chunk", type=int, default=8)
    p.add_argument("--robot_state", action="store_true")
    p.add_argument("--use_latent", type=int, default=1)
    p.add_argument(
        "--wan_vae_path",
        type=str,
        default=os.environ.get(
            "WAN_VAE_PATH",
            "/mnt/wfm/ckpt/ckpt/pretrained/Cosmos-Predict2.5-2B/tokenizer.pth",
        ),
    )
    p.add_argument(
        "--cosmos_dit_path",
        type=str,
        default=os.environ.get(
            "COSMOS_DIT_PATH",
            "/mnt/wfm/ckpt/ckpt/pretrained/Cosmos-Predict2.5-2B/robot/policy/libero/model.pt",
        ),
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dit_full_num_blocks", type=int, default=NUM_DIT_BLOCKS)
    p.add_argument("--dit_sigma", type=float, default=1e-5)
    p.add_argument("--dit_sigma_data", type=float, default=0.5)
    p.add_argument("--dit_timestep", type=int, default=0)
    p.add_argument("--dit_block_viz", type=int, nargs="+", default=None)
    p.add_argument(
        "--dit_probe_final_after_blocks",
        type=int,
        nargs="+",
        default=None,
        metavar="K",
        help="Decode after K blocks + final_layer (grid: one row per K, cols = frames).",
    )
    p.add_argument("--block_overlay_alpha", type=float, default=0.5)
    args = p.parse_args()

    block_depths = list(dict.fromkeys(args.dit_block_viz)) if args.dit_block_viz else []
    max_d = max(block_depths) if block_depths else 0
    probe_final = (
        list(dict.fromkeys(args.dit_probe_final_after_blocks))
        if args.dit_probe_final_after_blocks
        else []
    )
    dit_full_num_blocks = args.dit_full_num_blocks
    if probe_final:
        dit_full_num_blocks = min(NUM_DIT_BLOCKS, max(dit_full_num_blocks, max(probe_final)))

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    processor = VLChatProcessor.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    im = processor.image_processor
    ds = SftDataset(args, processor, DummyAccelerator(), None)
    codec = WanVAECodec(args.wan_vae_path, device="cpu").to(device)
    w_dtype = next(codec.encoder.conv1.parameters()).dtype
    codec = codec.to(w_dtype)

    dit_full = CosmosDiTFullHead(args.cosmos_dit_path, num_blocks=dit_full_num_blocks, device="cpu")
    dit_full = dit_full.to(device=device, dtype=torch.bfloat16).eval()
    dit_early = None
    if block_depths:
        dit_early = CosmosDiTEarlyExit(args.cosmos_dit_path, num_blocks=max_d, device="cpu")
        dit_early = dit_early.to(device=device, dtype=torch.bfloat16).eval()

    h240, w240 = 240, 240
    alpha = args.block_overlay_alpha

    with torch.no_grad():
        for idx in range(min(args.num_samples, len(ds))):
            lp = ds.collate_fn([ds[idx]])["latent_pixel_values"].to(device)
            wan_bn = latent_pixel_values_to_wan_m11(lp, im.image_mean, im.image_std).to(device)
            num_frames = wan_bn.shape[1]

            recon_b3thw = wan_recon_per_frame(codec, wan_bn, w_dtype)
            gt_nchw = wan_bn[0]
            recon_nchw = recon_b3thw[0].permute(1, 0, 2, 3)
            to01 = lambda t: ((t.float() + 1.0) * 0.5).clamp(0.0, 1.0)

            decode_row: list[torch.Tensor] = []
            block_panels: list[torch.Tensor] = []
            pca_gt: list[torch.Tensor] = []
            hm_gt: list[torch.Tensor] = []
            stash_mu8p: list[torch.Tensor] = []
            stash_tail16: list[torch.Tensor] = []

            for fi in range(num_frames):
                xi = wan_bn[:, fi : fi + 1].transpose(1, 2).to(w_dtype).float()
                lat = codec.encode_latent(xi)
                mu16 = lat[:, : WanVAEEncoder.LATENT_Z_DIM]
                mu8 = center_crop_mu_bcthw(lat[:, :8].to(torch.bfloat16))
                mu8p = pad_latent_temporal(mu8, FULL_PATCH_TEMPORAL)
                _, x0p = dit_full(
                    mu8p, sigma=args.dit_sigma, use_precondition=True, sigma_data=args.dit_sigma_data
                )
                mu16f = center_crop_mu_bcthw(mu16.to(w_dtype))
                stash_mu8p.append(mu8p)
                stash_tail16.append(mu16f[:, 8:16, :1])
                dec = codec.decode_mu(
                    torch.cat([x0p[:, :, :1].to(w_dtype), mu16f[:, 8:16, :1]], dim=1)
                )
                dec01 = ((dec[0, :, 0].float() + 1.0) * 0.5).clamp(0.0, 1.0)
                decode_row.append(
                    F.interpolate(dec01.unsqueeze(0), size=(h240, w240), mode="bilinear", align_corners=False)
                    .squeeze(0)
                    .cpu()
                )

                if dit_early is not None:
                    gt01_fi = to01(gt_nchw[fi]).cpu()
                    snaps = dit_early.forward_spatial_at_depths(
                        lat[:, :8].to(torch.bfloat16), args.dit_timestep, sorted(set(block_depths))
                    )
                    for d in block_depths:
                        spatial = snaps[d]
                        panel = dit_spatial_pca_rgb(spatial, (h240, w240), device).cpu()
                        block_panels.append(panel)
                        pca_gt.append(blend_rgb_overlay(gt01_fi, panel, alpha))
                        hm = scalar_hw_to_heatmap_rgb(
                            patch_mean_abs_energy_hw(spatial), (h240, w240), device
                        ).cpu()
                        hm_gt.append(blend_rgb_overlay(gt01_fi, hm, alpha))

            main_panels = [to01(gt_nchw[i]).cpu() for i in range(num_frames)]
            main_panels += [to01(recon_nchw[i]).cpu() for i in range(num_frames)]
            main_panels += decode_row
            main_path = os.path.join(args.output_dir, f"sample_{idx:04d}.png")
            _save_panel_grid(torch.stack(main_panels, dim=0), num_frames, main_path)
            out_paths = [main_path]

            if block_panels:
                blk = os.path.join(args.output_dir, f"sample_{idx:04d}_blocks.png")
                _save_panel_grid(torch.stack(block_panels, dim=0), num_frames, blk)
                pca_path = os.path.join(args.output_dir, f"sample_{idx:04d}_blocks_pca_on_gt.png")
                _save_panel_grid(torch.stack(pca_gt, dim=0), num_frames, pca_path)
                hm_path = os.path.join(args.output_dir, f"sample_{idx:04d}_blocks_heatmap_on_gt.png")
                _save_panel_grid(torch.stack(hm_gt, dim=0), num_frames, hm_path)
                out_paths += [blk, pca_path, hm_path]

            if probe_final:
                probe_panels: list[torch.Tensor] = []
                for k in probe_final:
                    for fi in range(num_frames):
                        _, x0pk = dit_full.forward_upto_then_final(
                            stash_mu8p[fi],
                            num_blocks_run=k,
                            sigma=args.dit_sigma,
                            use_precondition=True,
                            sigma_data=args.dit_sigma_data,
                        )
                        dec = codec.decode_mu(
                            torch.cat([x0pk[:, :, :1].to(w_dtype), stash_tail16[fi]], dim=1)
                        )
                        dec01 = ((dec[0, :, 0].float() + 1.0) * 0.5).clamp(0.0, 1.0)
                        probe_panels.append(
                            F.interpolate(dec01.unsqueeze(0), size=(h240, w240), mode="bilinear", align_corners=False)
                            .squeeze(0)
                            .cpu()
                        )
                pr_path = os.path.join(args.output_dir, f"sample_{idx:04d}_probe_final_decode.png")
                _save_panel_grid(torch.stack(probe_panels, dim=0), num_frames, pr_path)
                out_paths.append(pr_path)

            print(*out_paths)


if __name__ == "__main__":
    main()
