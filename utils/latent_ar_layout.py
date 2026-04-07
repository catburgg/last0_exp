"""
Shared layout for intra-frame parallel + inter-frame autoregressive latent training/inference.

Sequence structure: [prefix incl. <|latent_start|>] [optional GT for frames 0..f-1]
[zeros for current frame f] [<|latent_end|> + suffix + action tokens].
"""

from __future__ import annotations

import torch


def find_latent_bounds(
    input_ids_row: torch.Tensor,
    latent_start_id: int,
    latent_end_id: int,
) -> tuple[int, int]:
    """Return (latent_start_idx, latent_end_idx) for one sequence row."""
    ls = (input_ids_row == latent_start_id).nonzero(as_tuple=True)[0]
    le = (input_ids_row == latent_end_id).nonzero(as_tuple=True)[0]
    if len(ls) == 0 or len(le) == 0:
        raise ValueError("Missing <|latent_start|> or <|latent_end|> in input_ids")
    return int(ls[0].item()), int(le[0].item())


def build_latent_ar_step_embeds(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    frame_idx: int,
    tokens_per_frame: int,
    latent_start_id: int,
    latent_end_id: int,
    prev_frames_gt: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Build one AR step: prefix + latent_start + (optional previous-frame GT) + current-frame zeros + suffix.

    Args:
        inputs_embeds: Full sequence from prepare_inputs_embeds + flow tokens, **before** GT latent injection.
        input_ids: Full token ids (batch must share layout; uses row 0 for bounds).
        frame_idx: 0 .. num_frames-1
        tokens_per_frame: latent_size // num_frames
        prev_frames_gt: [B, frame_idx * tokens_per_frame, D] or None if frame_idx == 0

    Returns:
        step_embeds, attention_mask (bool), cur_frame_start (index of first zero placeholder for current frame)
    """
    bs, _, d = inputs_embeds.shape
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype

    row = input_ids if input_ids.dim() == 1 else input_ids[0]
    ls, le = find_latent_bounds(row, latent_start_id, latent_end_id)

    prefix = inputs_embeds[:, : ls + 1, :]
    suffix = inputs_embeds[:, le + 1 :, :]

    zeros = torch.zeros(bs, tokens_per_frame, d, device=device, dtype=dtype)

    if frame_idx == 0:
        step = torch.cat([prefix, zeros, suffix], dim=1)
        cur_start = ls + 1
    else:
        if prev_frames_gt is None:
            raise ValueError("prev_frames_gt required when frame_idx > 0")
        expected = frame_idx * tokens_per_frame
        if prev_frames_gt.shape[1] != expected:
            raise ValueError(
                f"prev_frames_gt has wrong length: got {prev_frames_gt.shape[1]}, expected {expected}"
            )
        step = torch.cat([prefix, prev_frames_gt, zeros, suffix], dim=1)
        cur_start = ls + 1 + expected

    attn = torch.ones(bs, step.shape[1], dtype=torch.bool, device=device)
    return step, attn, cur_start


def gather_frame_gt(
    compressed_latent_embeds: torch.Tensor,
    frame_idx: int,
    tokens_per_frame: int,
) -> torch.Tensor:
    """Slice GT embeddings for one frame: [B, tokens_per_frame, D]."""
    s = frame_idx * tokens_per_frame
    e = s + tokens_per_frame
    return compressed_latent_embeds[:, s:e, :].detach()
