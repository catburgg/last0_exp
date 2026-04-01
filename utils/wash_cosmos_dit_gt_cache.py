#!/usr/bin/env python3
"""
Precompute Cosmos DiT GT vectors aligned with ``train_wopc.py`` ``vision_backend == cosmos_denoise``.

Reads either:
  - JSONL: one record per line (``input_images_slow`` / ``output_images`` or JSON-export keys)
  - JSON array: same keys as SftDataset (``input_image_slow`` / ``output_image``)
  - Legacy: JSON list of ints + ``--dataset_root`` ``{idx}/frames`` layout

Writes under ``--output_dir``:
  ``gt_tensors/{key}.pt`` and ``manifest.json`` (or ``manifest.shardXXofYY.json`` if ``--num_shards`` > 1).

Parallel runs: same ``--output_dir``, ``--num_shards N``, distinct ``--shard_id`` 0…N-1; see ``wash_data_parallel.sh``.

Default ``--output_dir``: ``/mnt/wfm/ckpt/data/data_libero/libero_train_data_w_latent``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from tqdm import tqdm

_LAST0 = Path(__file__).resolve().parents[1]
if str(_LAST0) not in sys.path:
    sys.path.insert(0, str(_LAST0))
_UTILS_DIR = Path(__file__).resolve().parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from cosmos_policy_res_steps import order2_fn, policy_sigmas_like_cosmos_sampler

from janus.models import VLChatProcessor
from janus.models.cosmos_tokenizer.dit_lib import CosmosDiTFullHead
from janus.models.cosmos_tokenizer.wan_vae_lib import WanVAEEncoder


def denormalize_to_01(images: torch.Tensor, image_mean, image_std) -> torch.Tensor:
    mean = torch.tensor(image_mean, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(image_std, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    pixels = images * std + mean
    return pixels.clamp(0.0, 1.0)


def convert_to_cosmos_input(images: torch.Tensor, image_mean, image_std) -> torch.Tensor:
    pixels01 = denormalize_to_01(images.float(), image_mean, image_std)
    pixels256 = F.interpolate(pixels01, size=(256, 256), mode="bilinear", align_corners=False)
    return pixels256 * 2.0 - 1.0


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_indices(data: Any) -> List[int]:
    if isinstance(data, list) and (len(data) == 0 or isinstance(data[0], int)):
        return [int(x) for x in data]
    if isinstance(data, dict) and "indices" in data:
        return [int(x) for x in data["indices"]]
    raise ValueError("Legacy JSON must be a list of indices or {\"indices\": [...]}")


def _maybe_remap_path(
    p: str,
    json_dir: Path,
    old_prefix: Optional[str],
    new_prefix: Optional[str],
) -> Path:
    s = p
    if old_prefix and new_prefix is not None and s.startswith(old_prefix):
        s = new_prefix + s[len(old_prefix) :]
    pp = Path(s)
    if pp.is_file():
        return pp
    if not pp.is_absolute():
        cand = (json_dir / pp).resolve()
        if cand.is_file():
            return cand
    return pp


def _parse_image_keys(row: Dict[str, Any]) -> Tuple[str, List[str]]:
    if "input_image_slow" in row and "output_image" in row:
        slow = row["input_image_slow"]
        out = row["output_image"]
    elif "input_images_slow" in row and "output_images" in row:
        slow = row["input_images_slow"]
        out = row["output_images"]
    else:
        raise KeyError("expected input_image_slow/output_image or input_images_slow/output_images")
    if isinstance(slow, str):
        slow_list = [slow]
    else:
        slow_list = list(slow)
    if not slow_list:
        raise ValueError("empty slow image list")
    current_path = slow_list[0]
    outputs = list(out)
    return current_path, outputs


def iter_manifest_rows(
    data_path: Path,
    source_format: str,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    if source_format == "jsonl":
        with open(data_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                yield f"{line_no:08d}", json.loads(line)
    elif source_format == "json":
        data = _load_json(data_path)
        if isinstance(data, list):
            for i, row in enumerate(data):
                yield f"{i:08d}", row
        else:
            raise ValueError("JSON mode expects a list of records")
    else:
        raise ValueError(source_format)


def _load_rgb_frames(frames_dir: Path, num_frames: int) -> torch.Tensor:
    frames = []
    for i in range(num_frames):
        p = frames_dir / f"frame_{i:06d}.png"
        if not p.exists():
            raise FileNotFoundError(p)
        img = Image.open(p).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        frames.append(arr)
    return torch.from_numpy(np.stack(frames, axis=0))


def process_image_paths(paths: List[str], processor: VLChatProcessor) -> torch.Tensor:
    images = [Image.open(p).convert("RGB") for p in paths]
    out = processor.image_processor(images, return_tensors="pt")
    return out["pixel_values"]


def compute_dit_gt_cosmos_denoise(
    *,
    processor: VLChatProcessor,
    wan_vae: WanVAEEncoder,
    cosmos_dit: CosmosDiTFullHead,
    current_pixel: torch.Tensor,
    future_pixels: torch.Tensor,
    device: torch.device,
    sigma: float,
    dit_num_blocks: Optional[int],
    dit_dtype: torch.dtype,
    rng_seed: Optional[int] = None,
) -> torch.Tensor:
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std

    current_cosmos = convert_to_cosmos_input(current_pixel.float(), image_mean, image_std)
    current_240 = F.interpolate(current_cosmos, size=(240, 240), mode="bilinear", align_corners=False).unsqueeze(
        2
    )

    future_imgs = future_pixels.float()
    if future_imgs.dim() == 3:
        future_imgs = future_imgs.unsqueeze(0)
    future_cosmos = convert_to_cosmos_input(future_imgs, image_mean, image_std)
    future_240 = F.interpolate(future_cosmos, size=(240, 240), mode="bilinear", align_corners=False).unsqueeze(
        2
    )

    with torch.no_grad():
        current_latent = wan_vae.encode(current_240.to(device))
        future_latent = wan_vae.encode(future_240.to(device))
    mu_current = current_latent[:, :8]
    mu_future = future_latent[:, :8]

    num_frames = mu_future.shape[0]
    if rng_seed is not None:
        torch.manual_seed(int(rng_seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(rng_seed))
    epsilon = torch.randn_like(mu_future)
    xt_future = float(sigma) * epsilon + (1.0 - float(sigma)) * mu_future

    mu_current_bn = mu_current.unsqueeze(1).expand(-1, num_frames, -1, -1, -1, -1)
    mu_current_bn = rearrange(mu_current_bn, "b n c t h w -> (b n) c t h w")
    xt_pair = torch.cat([mu_current_bn, xt_future], dim=2)

    with torch.no_grad():
        hidden = cosmos_dit.forward_hidden(
            xt_pair.to(dit_dtype),
            num_blocks_run=dit_num_blocks,
            sigma=float(sigma),
        )
    dit_vec = hidden.mean(dim=[1, 2, 3])
    return dit_vec.float()


def compute_dit_gt_cosmos_k_schedule(
    *,
    processor: VLChatProcessor,
    wan_vae: WanVAEEncoder,
    cosmos_dit: CosmosDiTFullHead,
    current_pixel: torch.Tensor,
    future_pixels: torch.Tensor,
    device: torch.device,
    cosmos_k_steps: int,
    policy_num_steps: int = 35,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    dit_num_blocks: Optional[int],
    dit_dtype: torch.dtype,
    rng_seed: Optional[int] = None,
    sigma_data: float = 0.5,
) -> torch.Tensor:
    """
    K-1 RES 2ab steps with future-only re-projection of the clean half, then ``forward_hidden``
    at ``σ = sigmas_L[K-1]`` (total K DiT evaluations: K-1 ``forward`` + one ``forward_hidden``).
    """
    k = int(cosmos_k_steps)
    if k < 1:
        raise ValueError("cosmos_k_steps must be >= 1")

    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std

    current_cosmos = convert_to_cosmos_input(current_pixel.float(), image_mean, image_std)
    current_240 = F.interpolate(current_cosmos, size=(240, 240), mode="bilinear", align_corners=False).unsqueeze(
        2
    )

    future_imgs = future_pixels.float()
    if future_imgs.dim() == 3:
        future_imgs = future_imgs.unsqueeze(0)
    future_cosmos = convert_to_cosmos_input(future_imgs, image_mean, image_std)
    future_240 = F.interpolate(future_cosmos, size=(240, 240), mode="bilinear", align_corners=False).unsqueeze(
        2
    )

    with torch.no_grad():
        current_latent = wan_vae.encode(current_240.to(device))
        future_latent = wan_vae.encode(future_240.to(device))
    mu_current = current_latent[:, :8]
    mu_future = future_latent[:, :8]
    num_frames = mu_future.shape[0]

    if rng_seed is not None:
        torch.manual_seed(int(rng_seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(rng_seed))

    sigmas_L = policy_sigmas_like_cosmos_sampler(
        policy_num_steps=policy_num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
    ).to(device=device, dtype=torch.float64)

    if k > sigmas_L.shape[0]:
        raise ValueError(
            f"cosmos_k_steps={k} exceeds policy schedule length {sigmas_L.shape[0]} "
            f"(policy_num_steps={policy_num_steps})"
        )

    epsilon = torch.randn_like(mu_future)
    sigma0 = float(sigmas_L[0].item())
    xt_future = mu_future + sigma0 * epsilon

    mu_current_bn = mu_current.unsqueeze(1).expand(-1, num_frames, -1, -1, -1, -1)
    mu_current_bn = rearrange(mu_current_bn, "b n c t h w -> (b n) c t h w")

    bn = xt_future.shape[0]
    ones_b = torch.ones(bn, device=device, dtype=torch.float64)

    x_pair = torch.cat([mu_current_bn, xt_future], dim=2)
    x0_preds: Optional[List] = None

    for i_th in range(k - 1):
        sigma_cur = sigmas_L[i_th]
        sigma_next = sigmas_L[i_th + 1]
        with torch.no_grad():
            _, x0_full = cosmos_dit.forward(
                x_pair.to(dit_dtype),
                sigma=float(sigma_cur.item()),
                use_precondition=True,
                sigma_data=sigma_data,
            )
        x_s = x_pair.to(torch.float64)
        s = sigma_cur * ones_b
        t = sigma_next * ones_b
        x0_s = x0_full.to(torch.float64)
        x_pair_new, x0_preds = order2_fn(x_s, s, t, x0_s, x0_preds)
        x_pair_new = x_pair_new.to(dtype=x_pair.dtype)
        x_pair_new[:, :, 0:1, :, :] = mu_current_bn[:, :, 0:1, :, :]
        x_pair = x_pair_new

    readout_sigma = float(sigmas_L[k - 1].item())
    with torch.no_grad():
        hidden = cosmos_dit.forward_hidden(
            x_pair.to(dit_dtype),
            num_blocks_run=dit_num_blocks,
            sigma=readout_sigma,
            sigma_data=sigma_data,
        )
    dit_vec = hidden.mean(dim=[1, 2, 3])
    return dit_vec.float()


def _compute_dit_vec(
    *,
    processor: VLChatProcessor,
    wan_vae: WanVAEEncoder,
    cosmos_dit: CosmosDiTFullHead,
    current_pixel: torch.Tensor,
    future_pixels: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> torch.Tensor:
    if args.cosmos_k_steps is not None:
        return compute_dit_gt_cosmos_k_schedule(
            processor=processor,
            wan_vae=wan_vae,
            cosmos_dit=cosmos_dit,
            current_pixel=current_pixel,
            future_pixels=future_pixels,
            device=device,
            cosmos_k_steps=args.cosmos_k_steps,
            policy_num_steps=args.policy_num_steps,
            sigma_min=args.policy_sigma_min,
            sigma_max=args.policy_sigma_max,
            rho=args.policy_rho,
            dit_num_blocks=args.dit_num_blocks,
            dit_dtype=dtype,
            rng_seed=args.rng_seed,
            sigma_data=args.dit_sigma_data,
        )
    return compute_dit_gt_cosmos_denoise(
        processor=processor,
        wan_vae=wan_vae,
        cosmos_dit=cosmos_dit,
        current_pixel=current_pixel,
        future_pixels=future_pixels,
        device=device,
        sigma=args.cosmos_denoise_sigma,
        dit_num_blocks=args.dit_num_blocks,
        dit_dtype=dtype,
        rng_seed=args.rng_seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--source_format",
        type=str,
        choices=["auto", "jsonl", "json", "indices"],
        default="auto",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/wfm/ckpt/data/data_libero/libero_train_data_w_latent",
    )
    parser.add_argument("--dataset_root", type=str, default="/mnt/wfm/ckpt/data/data_libero/libero_train_data")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="HF folder for VLChatProcessor.from_pretrained",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--cosmos_denoise_sigma", type=float, default=0.5)
    parser.add_argument("--dit_num_blocks", type=int, default=None)
    parser.add_argument("--wan_vae_path", type=str, default=None)
    parser.add_argument("--cosmos_dit_path", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--path_prefix_old", type=str, default=None)
    parser.add_argument("--path_prefix_new", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=None,
        help="If set, torch RNG before ε so GT is reproducible and matches verify_wash_vs_train_dit_gt.py",
    )
    parser.add_argument(
        "--cosmos_k_steps",
        type=int,
        default=None,
        help="If set (>=1), policy RES schedule + 2ab steps then forward_hidden; if None, single-σ path.",
    )
    parser.add_argument(
        "--policy_num_steps",
        type=int,
        default=35,
        help="CosmosPolicySampler-style num_steps; nfe uses num_steps-1 when >1.",
    )
    parser.add_argument("--policy_sigma_min", type=float, default=0.002)
    parser.add_argument("--policy_sigma_max", type=float, default=80.0)
    parser.add_argument("--policy_rho", type=float, default=7.0)
    parser.add_argument(
        "--dit_sigma_data",
        type=float,
        default=0.5,
        help="EDM sigma_data for forward / forward_hidden.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split the (post --start/--end/--max_samples) list into this many disjoint shards.",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Keep items where list_index %% num_shards == shard_id.",
    )
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")

    data_path = Path(args.data_path).resolve()
    json_dir = data_path.parent
    output_dir = Path(args.output_dir)
    gt_dir = output_dir / "gt_tensors"
    gt_dir.mkdir(parents=True, exist_ok=True)

    source_format = args.source_format
    if source_format == "auto":
        suf = data_path.suffix.lower()
        if suf == ".jsonl":
            source_format = "jsonl"
        elif suf == ".json":
            try:
                peek = _load_json(data_path)
                source_format = "indices" if isinstance(peek, list) and (
                    len(peek) == 0 or isinstance(peek[0], int)
                ) else "json"
            except Exception:
                source_format = "json"
        else:
            source_format = "indices"

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    default_wan = "/mnt/wfm/ckpt/ckpt/pretrained/Cosmos-Predict2.5-2B/tokenizer.pth"
    default_dit = "/mnt/wfm/ckpt/ckpt/pretrained/Cosmos-Predict2.5-2B/robot/policy/libero/model.pt"
    wan_path = args.wan_vae_path or default_wan
    dit_path = args.cosmos_dit_path or default_dit

    processor = VLChatProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    wan_vae = WanVAEEncoder(wan_path, device=str(device))
    wan_vae = wan_vae.to(device=device, dtype=dtype)
    cosmos_dit = CosmosDiTFullHead(dit_path, device=str(device))
    cosmos_dit = cosmos_dit.to(device=device, dtype=dtype)
    cosmos_dit.eval()

    tqdm_desc = (
        f"wash_gt[shard {args.shard_id}/{args.num_shards}]"
        if args.num_shards > 1
        else "wash_gt"
    )

    manifest_entries: List[Dict[str, Any]] = []

    policy_schedule_meta: Optional[Dict[str, Any]] = None
    if args.cosmos_k_steps is not None:
        sl = policy_sigmas_like_cosmos_sampler(
            policy_num_steps=args.policy_num_steps,
            sigma_min=args.policy_sigma_min,
            sigma_max=args.policy_sigma_max,
            rho=args.policy_rho,
        )
        k = int(args.cosmos_k_steps)
        policy_schedule_meta = {
            "readout_step": k - 1,
            "readout_sigma": float(sl[k - 1].item()),
            "sigmas_L_prefix": [float(x) for x in sl[:k].tolist()],
            "sigmas_L_len": int(sl.shape[0]),
        }

    if source_format == "indices":
        indices = _resolve_indices(_load_json(data_path))
        if args.end is not None:
            indices = indices[args.start : args.end]
        else:
            indices = indices[args.start :]
        if args.max_samples is not None:
            indices = indices[: args.max_samples]
        if args.num_shards > 1:
            indices = [x for i, x in enumerate(indices) if i % args.num_shards == args.shard_id]
        dataset_root = Path(args.dataset_root)

        print(f"[wash_gt] shard {args.shard_id}/{args.num_shards}: {len(indices)} samples (indices)", flush=True)
        for idx in tqdm(indices, desc=tqdm_desc, dynamic_ncols=True):
            frames_dir = dataset_root / str(idx) / "frames"
            meta_file = dataset_root / str(idx) / "metadata.json"
            meta = _load_json(meta_file)
            num_frames = int(meta.get("num_frames", meta.get("length", 0)))
            rgb = _load_rgb_frames(frames_dir, num_frames)
            current_pixel = processor.image_processor(
                [Image.fromarray(rgb[0].numpy(), mode="RGB")], return_tensors="pt"
            )["pixel_values"]
            future_list = [Image.fromarray(rgb[i].numpy(), mode="RGB") for i in range(1, num_frames)]
            if not future_list:
                future_list = [Image.fromarray(rgb[0].numpy(), mode="RGB")]
            future_pixels = processor.image_processor(future_list, return_tensors="pt")["pixel_values"]

            dit_vec = _compute_dit_vec(
                processor=processor,
                wan_vae=wan_vae,
                cosmos_dit=cosmos_dit,
                current_pixel=current_pixel,
                future_pixels=future_pixels,
                device=device,
                args=args,
                dtype=dtype,
            )
            out_name = f"{int(idx):08d}.pt"
            out_path = gt_dir / out_name
            payload = {
                "dataset_index": int(idx),
                "num_frames": int(dit_vec.shape[0]),
                "dit_vec": dit_vec.cpu(),
                "shape": list(dit_vec.shape),
                "rng_seed": args.rng_seed,
            }
            torch.save(payload, out_path)
            manifest_entries.append(
                {
                    "id": str(idx),
                    "file": f"gt_tensors/{out_name}",
                    "num_frames": int(dit_vec.shape[0]),
                    "dit_shape": list(dit_vec.shape),
                }
            )
    else:
        rows = list(iter_manifest_rows(data_path, source_format))
        if args.end is not None:
            rows = rows[args.start : args.end]
        else:
            rows = rows[args.start :]
        if args.max_samples is not None:
            rows = rows[: args.max_samples]
        if args.num_shards > 1:
            rows = [x for i, x in enumerate(rows) if i % args.num_shards == args.shard_id]

        print(f"[wash_gt] shard {args.shard_id}/{args.num_shards}: {len(rows)} samples", flush=True)
        for record_id, row in tqdm(rows, desc=tqdm_desc, dynamic_ncols=True):
            cur_rel, out_rels = _parse_image_keys(row)
            cur_p = _maybe_remap_path(cur_rel, json_dir, args.path_prefix_old, args.path_prefix_new)
            out_paths = [_maybe_remap_path(p, json_dir, args.path_prefix_old, args.path_prefix_new) for p in out_rels]
            for p in [cur_p] + out_paths:
                if not Path(p).is_file():
                    raise FileNotFoundError(f"missing image: {p}")

            current_pixel = process_image_paths([str(cur_p)], processor)
            future_pixels = process_image_paths([str(p) for p in out_paths], processor)

            dit_vec = _compute_dit_vec(
                processor=processor,
                wan_vae=wan_vae,
                cosmos_dit=cosmos_dit,
                current_pixel=current_pixel,
                future_pixels=future_pixels,
                device=device,
                args=args,
                dtype=dtype,
            )
            out_name = f"{record_id}.pt"
            out_path = gt_dir / out_name
            payload = {
                "record_id": record_id,
                "num_frames": int(dit_vec.shape[0]),
                "dit_vec": dit_vec.cpu(),
                "shape": list(dit_vec.shape),
                "rng_seed": args.rng_seed,
            }
            torch.save(payload, out_path)
            manifest_entries.append(
                {
                    "record_id": record_id,
                    "file": f"gt_tensors/{out_name}",
                    "num_frames": int(dit_vec.shape[0]),
                    "dit_shape": list(dit_vec.shape),
                }
            )

    manifest = {
        "data_path": str(data_path),
        "output_dir": str(output_dir.resolve()),
        "source_format": source_format,
        "checkpoint": args.checkpoint,
        "cosmos_denoise_sigma": args.cosmos_denoise_sigma,
        "cosmos_k_steps": args.cosmos_k_steps,
        "policy_num_steps": args.policy_num_steps,
        "policy_sigma_min": args.policy_sigma_min,
        "policy_sigma_max": args.policy_sigma_max,
        "policy_rho": args.policy_rho,
        "dit_sigma_data": args.dit_sigma_data,
        "pair_update": "future_only",
        "policy_schedule": policy_schedule_meta,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "dit_num_blocks": args.dit_num_blocks,
        "rng_seed": args.rng_seed,
        "wan_vae_path": wan_path,
        "cosmos_dit_path": dit_path,
        "num_samples": len(manifest_entries),
        "entries": manifest_entries,
    }
    if args.num_shards > 1:
        manifest_path = output_dir / f"manifest.shard{args.shard_id:02d}of{args.num_shards:02d}.json"
    else:
        manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest_entries)} tensors under {gt_dir}")
    print(f"Manifest: {manifest_path}")
    if args.num_shards > 1:
        print(
            "Sharding: merge manifest.shard*of*.json `entries` after all shards finish "
            "(order may differ from single-process manifest)."
        )


if __name__ == "__main__":
    main()
