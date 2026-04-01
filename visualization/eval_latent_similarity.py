"""
Image-only latent similarity: WAN VAE encode + Cosmos DiT **block-stack hidden states**
(``forward_hidden``: after DiT blocks, **before** ``final_layer`` / unpatchify — not x0 from full head).
Compares cosine similarity between current frame (t) and future frames (t+k).
"""
from __future__ import annotations

import argparse
from typing import Optional
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from janus.models import VLChatProcessor
from janus.models.cosmos_tokenizer.dit_lib import (
    NUM_DIT_BLOCKS,
    CosmosDiTFullHead,
    FULL_PATCH_TEMPORAL,
    pad_latent_temporal,
)
from janus.models.cosmos_tokenizer.wan_vae_lib import WanVAECodec
from train_wopc import SftDataset

from visualize_wan_dit_vae import (
    center_crop_mu_bcthw,
    latent_pixel_values_to_wan_m11,
)


class DummyAccelerator:
    @staticmethod
    def print(*args, **kwargs):
        print(*args, **kwargs)


def _compress_tokens_like_original(
    hidden_b_tp_hp_wp_d: torch.Tensor,
    tokens_per_modality: int,
) -> torch.Tensor:
    """
    Same pattern as the original ``eval_latent_similarity`` image branch on
    ``img_embeds_flat`` (sequence on dim=1):

    - ``embeds``: [B, seq_len, D] with ``seq_len = Tp*Hp*Wp``
    - ``group_size = seq_len // tokens_per_modality``
    - split into ``tokens_per_modality`` chunks along dim=1, mean over each chunk's sequence dim
    - ``compressed``: [B, tokens_per_modality, D] -> flatten to [B, tokens_per_modality * D]
    """
    b, nt, nh, nw, d = hidden_b_tp_hp_wp_d.shape
    embeds = hidden_b_tp_hp_wp_d.reshape(b, nt * nh * nw, d).float()
    seq_len = embeds.shape[1]
    group_size = seq_len // tokens_per_modality
    chunks = torch.split(embeds, group_size, dim=1)
    compressed = torch.cat(
        [c.mean(dim=1, keepdim=True) for c in chunks[:tokens_per_modality]],
        dim=1,
    )
    return compressed.reshape(b, -1)


def encode_to_latent_tokens(
    codec: WanVAECodec,
    dit_full: CosmosDiTFullHead,
    processor,
    dataset,
    img_paths: list[str],
    tokens_per_modality: int,
    dit_sigma: float,
    dit_sigma_data: float,
    device: torch.device,
    dit_num_blocks_run: Optional[int],
) -> torch.Tensor:
    """
    WAN VAE encode -> pad -> ``CosmosDiTFullHead.forward_hidden`` (DiT blocks only; no ``final_layer``)
    -> per frame: reshape DiT hidden to ``[B, seq_len, D]``, then same
    ``group_size / split / mean`` as the original image ``aligner(vision_model)`` path.
    Returns [L, D_flat] float32.
    """
    w_dtype = next(codec.encoder.conv1.parameters()).dtype
    im = processor.image_processor
    L = len(img_paths)

    with torch.no_grad():
        pixel_values = dataset.process_image(img_paths).to(device=device, dtype=torch.bfloat16)
        lp = pixel_values.unsqueeze(0)
        wan_bn = latent_pixel_values_to_wan_m11(lp, im.image_mean, im.image_std).to(device)

        rows: list[torch.Tensor] = []
        for fi in range(L):
            xi = wan_bn[:, fi : fi + 1].transpose(1, 2).to(w_dtype).float()
            lat = codec.encode_latent(xi)
            mu8 = center_crop_mu_bcthw(lat[:, :8].to(torch.bfloat16))
            mu8p = pad_latent_temporal(mu8, FULL_PATCH_TEMPORAL)
            hidden = dit_full.forward_hidden(
                mu8p,
                num_blocks_run=dit_num_blocks_run,
                sigma=float(dit_sigma),
                sigma_data=float(dit_sigma_data),
            )
            row = _compress_tokens_like_original(hidden, tokens_per_modality)
            rows.append(row)

        return torch.cat(rows, dim=0)


def main() -> None:
    p = argparse.ArgumentParser(description="Cosine similarity t vs t+k in WAN VAE + DiT latent space")
    p.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="VLChatProcessor checkpoint (e.g. training tfmr dir)",
    )
    p.add_argument("--data_path", type=str, required=True, help="Training JSON (LIBERO-style with input_image_slow / output_image)")
    p.add_argument("--data_root", type=str, default="", help="Unused; kept for parity with train_wopc args")
    p.add_argument("--output_dir", type=str, default="./eval_latent_sim_out")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument(
        "--tokens_per_modality",
        type=int,
        default=1,
        help="Split seq_len=Tp*Hp*Wp into this many groups; mean-pool along sequence within each "
        "(same as original script on vision token sequence).",
    )
    p.add_argument("--latent_size", type=int, default=16, choices=[4, 16, 64])
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--action_chunk", type=int, default=8)
    p.add_argument("--future_steps", type=int, default=4, help="Max horizon k for t+k (capped by len(output_image))")
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
    p.add_argument("--dit_full_num_blocks", type=int, default=NUM_DIT_BLOCKS)
    p.add_argument(
        "--dit_num_blocks_run",
        type=int,
        default=None,
        help="Use only first N DiT blocks for hidden features (default: all blocks loaded in CosmosDiTFullHead).",
    )
    p.add_argument("--dit_sigma", type=float, default=1e-5)
    p.add_argument("--dit_sigma_data", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading VLChatProcessor from {args.checkpoint_path} ...")
    processor = VLChatProcessor.from_pretrained(args.checkpoint_path, trust_remote_code=True)

    print(f"Loading WanVAECodec ({args.wan_vae_path}) ...")
    codec = WanVAECodec(args.wan_vae_path, device="cpu").to(device)
    w_dtype = next(codec.encoder.conv1.parameters()).dtype
    codec = codec.to(w_dtype)

    print(f"Loading CosmosDiTFullHead ({args.cosmos_dit_path}, blocks={args.dit_full_num_blocks}) ...")
    dit_full = CosmosDiTFullHead(args.cosmos_dit_path, num_blocks=args.dit_full_num_blocks, device="cpu")
    dit_full = dit_full.to(device=device, dtype=torch.bfloat16).eval()

    cfg = argparse.Namespace(
        data_path=args.data_path,
        data_root=args.data_root,
        use_latent=1,
        latent_size=args.latent_size,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
        robot_state=False,
    )
    dataset = SftDataset(cfg, processor, DummyAccelerator(), None)
    img_dir = dataset.img_dir

    future_steps = args.future_steps
    sims_img = {k: [] for k in range(1, future_steps + 1)}
    features_img: list[np.ndarray] = []

    print(
        f"Feature extraction (WAN + DiT block hidden, pre-final_layer), "
        f"dit_num_blocks_run={args.dit_num_blocks_run}, tokens_per_modality={args.tokens_per_modality}, "
        f"future_steps cap={future_steps} ..."
    )
    indices = np.random.permutation(len(dataset))[: args.num_samples]

    for idx in tqdm(indices):
        data_item = dataset[idx]

        t_img_path = [os.path.join(img_dir, data_item["input_image_slow"][0])]
        future_img_paths = [os.path.join(img_dir, p) for p in data_item["output_image"]]

        latent_t_img = encode_to_latent_tokens(
            codec,
            dit_full,
            processor,
            dataset,
            t_img_path,
            args.tokens_per_modality,
            args.dit_sigma,
            args.dit_sigma_data,
            device,
            args.dit_num_blocks_run,
        )
        latent_fut_img = encode_to_latent_tokens(
            codec,
            dit_full,
            processor,
            dataset,
            future_img_paths,
            args.tokens_per_modality,
            args.dit_sigma,
            args.dit_sigma_data,
            device,
            args.dit_num_blocks_run,
        )

        features_img.append(
            torch.cat([latent_t_img, latent_fut_img], dim=0).float().cpu().numpy()
        )

        n_fut = latent_fut_img.shape[0]
        for k in range(min(future_steps, n_fut)):
            sim_img_val = F.cosine_similarity(
                latent_t_img.float(),
                latent_fut_img[k : k + 1].float(),
                dim=1,
            ).item()
            sims_img[k + 1].append(sim_img_val)

    def print_sims(title: str, sims_dict: dict) -> None:
        print("\n" + "=" * 50)
        print(f"Cosine Similarity to Current Frame (t) — {title}")
        print("=" * 50)
        for kk in range(1, future_steps + 1):
            if len(sims_dict[kk]) > 0:
                mean_sim = float(np.mean(sims_dict[kk]))
                std_sim = float(np.std(sims_dict[kk]))
                print(f"Step t+{kk} : {mean_sim:.4f} ± {std_sim:.4f}")

    print_sims("IMAGE (WAN VAE + DiT block hidden, pre-final_layer)", sims_img)

    os.makedirs(args.output_dir, exist_ok=True)
    K = args.tokens_per_modality
    path_img = os.path.join(args.output_dir, f"latent_features_img_{K}.npy")
    arrs = [np.asarray(x, dtype=np.float32) for x in features_img]
    try:
        to_save = np.stack(arrs, axis=0)
    except ValueError:
        to_save = np.array(arrs, dtype=object)
    np.save(path_img, to_save)
    print("\nSaved features for PCA / downstream analysis:")
    print(f" - {path_img} (shape {to_save.shape})")


if __name__ == "__main__":
    main()
