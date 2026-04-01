"""
CosmosDiTEarlyExit: self-contained early-exit wrapper for Cosmos-Policy DiT.

No dependency on transformer_engine or cosmos-policy source code.
All components are reimplemented from checkpoint weight inspection.

Checkpoint facts (Cosmos-Policy-LIBERO-Predict2-2B):
  - CosmosDiTEarlyExit (training path): treats 72 = 8×3×3×1 (legacy layout; see below).
  - Full ``net.final_layer`` + unpatchify (policy MiniTrain geometry, consistent 72 & 64):
      72 = 9×2×2×2 → 8 latent + 1 padding-mask channel, patch_spatial=2, patch_temporal=2
      64 = 8×2×2×2 → out_channels=8 (same patch sizes; see ``CosmosDiTFullHead``).
  - 28 DiT blocks; default early-exit uses first 11 blocks (indices 0–10)
  - model_channels = 2048, num_heads = 16, head_dim = 128
  - crossattn_emb_channels = 1024
  - use_adaln_lora = True, adaln_lora_dim = 256

Input: CI8x8 VAE latent (B, 16, H, W)
  → vae_to_dit_proj Linear(16,8) → (B, 8, H, W)
  → center-crop to 30×30 (divisible by patch_spatial=3)
  → unsqueeze T=1 → (B, 8, 1, 30, 30)
  → DiT early exit → (B, L=100, 2048)
  → GAP → (B*num_frames, 2048)
"""

import math
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

CROP_SIZE = 30        # 30 is divisible by patch_spatial=3; nearest below 32
PATCH_SPATIAL = 3
PATCH_TEMPORAL = 1
IN_CHANNELS = 8       # DiT x_embedder in_channels (from checkpoint: 72 = 8×3×3×1)
HIDDEN_DIM = 2048
NUM_HEADS = 16
HEAD_DIM = HIDDEN_DIM // NUM_HEADS  # 128
NUM_DIT_BLOCKS = 28
DEFAULT_NUM_DIT_BLOCKS = 11
CROSSATTN_DIM = 1024
ADALN_LORA_DIM = 256

DIT_HP = CROP_SIZE // PATCH_SPATIAL # 10
DIT_WP = CROP_SIZE // PATCH_SPATIAL # 10

# Geometry for ``net.final_layer`` + unpatchify (matches checkpoint 72 & 64 together).
FULL_PATCH_SPATIAL = 2
FULL_PATCH_TEMPORAL = 2
FULL_IN_CHANNELS = 9   # 8 latent + 1 padding mask (policy ``concat_padding_mask``)
FULL_OUT_CHANNELS = 8
FULL_DIT_HP = CROP_SIZE // FULL_PATCH_SPATIAL  # 15
FULL_DIT_WP = CROP_SIZE // FULL_PATCH_SPATIAL  # 15

# ---------------------------------------------------------------------------
# Minimal self-contained DiT components (no transformer_engine required)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm matching TransformerEngine's behavior (weight only, no bias)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        x = x * rms * self.weight.float()
        return x.to(orig_dtype)


class PatchEmbed(nn.Module):
    """Patch embedding: Rearrange → Linear (no bias)."""
    def __init__(self, spatial_patch_size: int, temporal_patch_size: int,
                 in_channels: int, out_channels: int):
        super().__init__()
        self.sp = spatial_patch_size
        self.tp = temporal_patch_size
        dim = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size
        # proj[1] matches checkpoint key "x_embedder.proj.1.weight"
        self.proj = nn.Sequential(
            nn.Identity(),  # placeholder for Rearrange (we do it manually)
            nn.Linear(dim, out_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        p, r = self.sp, self.tp
        assert H % p == 0 and W % p == 0 and T % r == 0
        # Fold patches: (B, C, T, H, W) → (B, T//r, H//p, W//p, C*r*p*p)
        x = x.reshape(B, C, T // r, r, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.reshape(B, T // r, H // p, W // p, C * r * p * p)
        x = self.proj[1](x)  # (B, T//r, H//p, W//p, out_channels)
        return x


class Timesteps(nn.Module):
    """Sinusoidal timestep embedding (no parameters)."""
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        # timesteps_B_T: (B, T) float
        assert timesteps_B_T.ndim == 2
        B, T = timesteps_B_T.shape
        in_dtype = timesteps_B_T.dtype
        ts = timesteps_B_T.flatten().float()
        half = self.num_channels // 2
        exp = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=ts.device) / (half - 0.0)
        emb = ts[:, None] * torch.exp(exp)[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)  # (B*T, num_channels)
        return emb.to(in_dtype).reshape(B, T, self.num_channels)


class TimestepEmbedding(nn.Module):
    """
    Maps sinusoidal timestep features → (emb_B_T_D, adaln_lora_B_T_3D).
    With use_adaln_lora=True:
      emb_B_T_D = input (Fourier features)
      adaln_lora_B_T_3D = SiLU(linear_1(input)) → linear_2  [6144-dim]
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)

    def forward(self, sample: torch.Tensor):
        # sample: (B, T, D)
        emb = self.activation(self.linear_1(sample))
        adaln_lora = self.linear_2(emb)  # (B, T, 3D)
        return sample, adaln_lora  # emb_B_T_D = original sample, adaln_lora_B_T_3D


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.output_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, D)
        orig_shape = x.shape
        B = orig_shape[0]
        x_flat = x.reshape(B, -1, orig_shape[-1])  # (B, S, D)
        S = x_flat.shape[1]

        q = self.q_proj(x_flat).reshape(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x_flat).reshape(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(x_flat).reshape(B, S, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # (B, S, H, D) → (B, H, S, D) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)  # (B, H, S, D)
        out = out.transpose(1, 2).reshape(B, S, self.num_heads * self.head_dim)
        out = self.output_proj(out)
        return out.reshape(orig_shape)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(context_dim, dim, bias=False)
        self.v_proj = nn.Linear(context_dim, dim, bias=False)
        self.output_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, D), context: (B, N, context_dim)
        orig_shape = x.shape
        B = orig_shape[0]
        x_flat = x.reshape(B, -1, orig_shape[-1])  # (B, S, D)
        S = x_flat.shape[1]
        N = context.shape[1]

        q = self.q_proj(x_flat).reshape(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(context).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(context).reshape(B, N, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)  # (B, H, S, D)
        out = out.transpose(1, 2).reshape(B, S, self.num_heads * self.head_dim)
        out = self.output_proj(out)
        return out.reshape(orig_shape)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden_dim, bias=False)
        self.layer2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.gelu(self.layer1(x)))


class AdaLNModulation(nn.Module):
    """AdaLN with LoRA: SiLU → Linear(dim, lora_dim) → Linear(lora_dim, 3*dim)."""
    def __init__(self, dim: int, lora_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, lora_dim, bias=False),
            nn.Linear(lora_dim, 3 * dim, bias=False),
        )

    def forward(self, emb: torch.Tensor, adaln_lora: torch.Tensor):
        # emb: (B, T, D), adaln_lora: (B, T, 3D)
        out = self.layers(emb) + adaln_lora  # (B, T, 3D)
        shift, scale, gate = out.chunk(3, dim=-1)
        return shift, scale, gate


class DiTBlock(nn.Module):
    """
    Full DiT transformer block matching Cosmos-Policy checkpoint structure.
    Self-attn + Cross-attn + MLP, each with AdaLN-LoRA modulation.
    """
    def __init__(self, dim: int, context_dim: int, num_heads: int,
                 head_dim: int, lora_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.layer_norm_self_attn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = SelfAttention(dim, num_heads, head_dim)
        self.adaln_self_attn = AdaLNModulation(dim, lora_dim)

        self.layer_norm_cross_attn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(dim, context_dim, num_heads, head_dim)
        self.adaln_cross_attn = AdaLNModulation(dim, lora_dim)

        self.layer_norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.adaln_mlp = AdaLNModulation(dim, lora_dim)

    def forward(self, x: torch.Tensor, emb: torch.Tensor,
                adaln_lora: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, D)
        # emb: (B, T, D)  — Fourier timestep features
        # adaln_lora: (B, T, 3D)
        # context: (B, N, context_dim)

        def _broadcast(t):
            # (B, T, D) → (B, T, 1, 1, D)
            return t.unsqueeze(2).unsqueeze(3)

        # Self-attention
        shift, scale, gate = self.adaln_self_attn(emb, adaln_lora)
        xn = self.layer_norm_self_attn(x) * (1 + _broadcast(scale)) + _broadcast(shift)
        x = x + _broadcast(gate) * self.self_attn(xn)

        # Cross-attention
        shift, scale, gate = self.adaln_cross_attn(emb, adaln_lora)
        xn = self.layer_norm_cross_attn(x) * (1 + _broadcast(scale)) + _broadcast(shift)
        x = x + _broadcast(gate) * self.cross_attn(xn, context)

        # MLP
        shift, scale, gate = self.adaln_mlp(emb, adaln_lora)
        xn = self.layer_norm_mlp(x) * (1 + _broadcast(scale)) + _broadcast(shift)
        x = x + _broadcast(gate) * self.mlp(xn)

        return x


def dit_unpatchify(
    x_B_T_H_W_M: torch.Tensor,
    patch_spatial: int,
    patch_temporal: int,
    out_channels: int,
) -> torch.Tensor:
    """Inverse of patch embedding on the channel grid (no learned weights). From MiniTrainDIT.unpatchify."""
    # Use distinct names for token-time Tp vs patch_temporal pt (einops reuses repeated axis names).
    return rearrange(
        x_B_T_H_W_M,
        "B Tp Hp Wp (p1 p2 pt c) -> B c (Tp pt) (Hp p1) (Wp p2)",
        p1=patch_spatial,
        p2=patch_spatial,
        pt=patch_temporal,
        c=out_channels,
    )


class FinalLayer(nn.Module):
    """
    Final DiT head (MiniTrainDIT.final_layer in minimal_v4_dit.py).
    AdaLN-LoRA uses only the first 2*hidden_size components of adaln_lora_B_T_3D.
    """

    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = True,
        adaln_lora_dim: int = ADALN_LORA_DIM,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        flat = spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, flat, bias=False)
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 2 * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=False),
            )

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        adaln_lora_B_T_3D: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_adaln_lora:
            shift_B_T_D, scale_B_T_D = (
                self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
            ).chunk(2, dim=-1)
        else:
            shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

        shift_B_T_1_1_D = rearrange(shift_B_T_D, "b t d -> b t 1 1 d")
        scale_B_T_1_1_D = rearrange(scale_B_T_D, "b t d -> b t 1 1 d")
        x = self.layer_norm(x_B_T_H_W_D) * (1 + scale_B_T_1_1_D) + shift_B_T_1_1_D
        return self.linear(x)


class EDMScaling:
    """c_skip / c_out / c_in / c_noise from continuous σ (cosmos_policy EDMScaling)."""

    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data

    def __call__(
        self, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1.0 / (sigma**2 + self.sigma_data**2).sqrt()
        c_noise = 0.25 * sigma.clamp_min(1e-20).log()
        return c_skip, c_out, c_in, c_noise


def pad_latent_temporal(x_B_C_T_H_W: torch.Tensor, factor: int) -> torch.Tensor:
    """Right-pad T so T % factor == 0 (duplicate last frame)."""
    _, _, t, _, _ = x_B_C_T_H_W.shape
    if t % factor == 0:
        return x_B_C_T_H_W
    pad = factor - (t % factor)
    last = x_B_C_T_H_W[:, :, -1:, :, :]
    return torch.cat([x_B_C_T_H_W, last.repeat(1, 1, pad, 1, 1)], dim=2)


def concat_padding_mask_channel(
    x_B_C_T_H_W: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Append padding mask channel (B,1,T,H,W); default ones = valid."""
    b, _, t, h, w = x_B_C_T_H_W.shape
    if mask is None:
        mask = torch.ones(b, 1, t, h, w, dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
    return torch.cat([x_B_C_T_H_W, mask], dim=1)


def pool_timesteps_to_patch_tokens(
    values_B_1_T_1_1: torch.Tensor, patch_temporal: int
) -> torch.Tensor:
    """
    Map per-frame values (B,1,T,1,1) to one timestep per temporal patch (B, T//pt).
    Matches MiniTrain when patch_temporal>1: x_embedder shrinks T, but t_embedder must align to token T.
    """
    b, _, t, _, _ = values_B_1_T_1_1.shape
    if t % patch_temporal != 0:
        raise ValueError(f"latent T={t} must be divisible by patch_temporal={patch_temporal}")
    flat = values_B_1_T_1_1[:, 0, :, 0, 0]
    return flat.view(b, t // patch_temporal, patch_temporal).mean(dim=-1)


# ---------------------------------------------------------------------------
# Full blocks + final_layer + unpatchify (policy geometry)
# ---------------------------------------------------------------------------


class CosmosDiTFullHead(nn.Module):
    """
    MiniTrain-style ``blocks → final_layer → unpatchify`` with checkpoint ``net.final_layer.*``.
    Uses patch (2,2) and 9 input channels (8 latent + mask) so x_embedder (72) and final linear (64) match.
    Temporal length must be divisible by ``FULL_PATCH_TEMPORAL`` (pad with ``pad_latent_temporal``).
    """

    def __init__(self, ckpt_path: str, num_blocks: int = NUM_DIT_BLOCKS, device: str = "cpu"):
        super().__init__()
        if not (1 <= num_blocks <= NUM_DIT_BLOCKS):
            raise ValueError(f"num_blocks must be in [1, {NUM_DIT_BLOCKS}], got {num_blocks}")
        self.num_blocks = num_blocks
        self.patch_spatial = FULL_PATCH_SPATIAL
        self.patch_temporal = FULL_PATCH_TEMPORAL
        self.out_channels = FULL_OUT_CHANNELS

        self.x_embedder = PatchEmbed(
            FULL_PATCH_SPATIAL, FULL_PATCH_TEMPORAL, FULL_IN_CHANNELS, HIDDEN_DIM
        )
        self.t_embedder = nn.Sequential(
            Timesteps(HIDDEN_DIM),
            TimestepEmbedding(HIDDEN_DIM, HIDDEN_DIM),
        )
        self.t_embedding_norm = RMSNorm(HIDDEN_DIM, eps=1e-6)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=HIDDEN_DIM,
                    context_dim=CROSSATTN_DIM,
                    num_heads=NUM_HEADS,
                    head_dim=HEAD_DIM,
                    lora_dim=ADALN_LORA_DIM,
                    mlp_ratio=4.0,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size=HIDDEN_DIM,
            spatial_patch_size=FULL_PATCH_SPATIAL,
            temporal_patch_size=FULL_PATCH_TEMPORAL,
            out_channels=FULL_OUT_CHANNELS,
            use_adaln_lora=True,
            adaln_lora_dim=ADALN_LORA_DIM,
        )
        self._load_weights(ckpt_path, device)
        for p in self.parameters():
            p.requires_grad_(False)

    def _load_weights(self, ckpt_path: str, device: str):
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        net = {k[4:]: v for k, v in sd.items() if k.startswith("net.")}

        self.x_embedder.proj[1].weight.data.copy_(net["x_embedder.proj.1.weight"])
        self.t_embedder[1].linear_1.weight.data.copy_(net["t_embedder.1.linear_1.weight"])
        self.t_embedder[1].linear_2.weight.data.copy_(net["t_embedder.1.linear_2.weight"])
        self.t_embedding_norm.weight.data.copy_(net["t_embedding_norm.weight"])

        for i, block in enumerate(self.blocks):
            prefix = f"blocks.{i}."
            b = {k[len(prefix) :]: v for k, v in net.items() if k.startswith(prefix)}
            block.self_attn.q_proj.weight.data.copy_(b["self_attn.q_proj.weight"])
            block.self_attn.k_proj.weight.data.copy_(b["self_attn.k_proj.weight"])
            block.self_attn.v_proj.weight.data.copy_(b["self_attn.v_proj.weight"])
            block.self_attn.output_proj.weight.data.copy_(b["self_attn.output_proj.weight"])
            block.self_attn.q_norm.weight.data.copy_(b["self_attn.q_norm.weight"])
            block.self_attn.k_norm.weight.data.copy_(b["self_attn.k_norm.weight"])
            block.cross_attn.q_proj.weight.data.copy_(b["cross_attn.q_proj.weight"])
            block.cross_attn.k_proj.weight.data.copy_(b["cross_attn.k_proj.weight"])
            block.cross_attn.v_proj.weight.data.copy_(b["cross_attn.v_proj.weight"])
            block.cross_attn.output_proj.weight.data.copy_(b["cross_attn.output_proj.weight"])
            block.cross_attn.q_norm.weight.data.copy_(b["cross_attn.q_norm.weight"])
            block.cross_attn.k_norm.weight.data.copy_(b["cross_attn.k_norm.weight"])
            block.mlp.layer1.weight.data.copy_(b["mlp.layer1.weight"])
            block.mlp.layer2.weight.data.copy_(b["mlp.layer2.weight"])
            block.adaln_self_attn.layers[1].weight.data.copy_(b["adaln_modulation_self_attn.1.weight"])
            block.adaln_self_attn.layers[2].weight.data.copy_(b["adaln_modulation_self_attn.2.weight"])
            block.adaln_cross_attn.layers[1].weight.data.copy_(b["adaln_modulation_cross_attn.1.weight"])
            block.adaln_cross_attn.layers[2].weight.data.copy_(b["adaln_modulation_cross_attn.2.weight"])
            block.adaln_mlp.layers[1].weight.data.copy_(b["adaln_modulation_mlp.1.weight"])
            block.adaln_mlp.layers[2].weight.data.copy_(b["adaln_modulation_mlp.2.weight"])

        self.final_layer.linear.weight.data.copy_(net["final_layer.linear.weight"])
        self.final_layer.adaln_modulation[1].weight.data.copy_(net["final_layer.adaln_modulation.1.weight"])
        self.final_layer.adaln_modulation[2].weight.data.copy_(net["final_layer.adaln_modulation.2.weight"])

    def _embed_forward(
        self,
        latent_9_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
    ) -> torch.Tensor:
        x = self.x_embedder(latent_9_B_C_T_H_W)
        ts_dtype = next(self.t_embedder.parameters()).dtype
        timesteps_B_T = timesteps_B_T.to(dtype=ts_dtype, device=latent_9_B_C_T_H_W.device)
        emb_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        emb_B_T_D = self.t_embedding_norm(emb_B_T_D)
        dtype = latent_9_B_C_T_H_W.dtype
        emb_B_T_D = emb_B_T_D.to(dtype)
        adaln_lora_B_T_3D = adaln_lora_B_T_3D.to(dtype)
        b = latent_9_B_C_T_H_W.shape[0]
        crossattn = torch.zeros(b, 1, CROSSATTN_DIM, dtype=dtype, device=latent_9_B_C_T_H_W.device)
        for block in self.blocks:
            x = block(x, emb_B_T_D, adaln_lora_B_T_3D, crossattn)
        x = self.final_layer(x, emb_B_T_D, adaln_lora_B_T_3D)
        out = dit_unpatchify(x, self.patch_spatial, self.patch_temporal, self.out_channels)
        return out

    def _embed_forward_partial(
        self,
        latent_9_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        num_blocks_run: int,
    ) -> torch.Tensor:
        """Same as ``_embed_forward`` but run only the first ``num_blocks_run`` DiT blocks (0 = embed → final only)."""
        x = self.x_embedder(latent_9_B_C_T_H_W)
        ts_dtype = next(self.t_embedder.parameters()).dtype
        timesteps_B_T = timesteps_B_T.to(dtype=ts_dtype, device=latent_9_B_C_T_H_W.device)
        emb_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        emb_B_T_D = self.t_embedding_norm(emb_B_T_D)
        dtype = latent_9_B_C_T_H_W.dtype
        emb_B_T_D = emb_B_T_D.to(dtype)
        adaln_lora_B_T_3D = adaln_lora_B_T_3D.to(dtype)
        b = latent_9_B_C_T_H_W.shape[0]
        crossattn = torch.zeros(b, 1, CROSSATTN_DIM, dtype=dtype, device=latent_9_B_C_T_H_W.device)
        nb = max(0, min(int(num_blocks_run), len(self.blocks)))
        for i in range(nb):
            x = self.blocks[i](x, emb_B_T_D, adaln_lora_B_T_3D, crossattn)
        x = self.final_layer(x, emb_B_T_D, adaln_lora_B_T_3D)
        return dit_unpatchify(x, self.patch_spatial, self.patch_temporal, self.out_channels)

    def _full_head_forward_core(
        self,
        latent_8_B_C_T_H_W: torch.Tensor,
        sigma: Optional[torch.Tensor],
        use_precondition: bool,
        sigma_data: float,
        padding_mask: Optional[torch.Tensor],
        num_blocks_run: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
        """
        Shared preconditioning + embed path. If ``num_blocks_run`` is None, run all blocks then final;
        else run that many blocks (partial stack) then final.
        Returns ``(net_out, xt, sigma_t_or_None, t_latent)`` for ``x0`` composition.
        """
        b, c8, t, h, w = latent_8_B_C_T_H_W.shape
        assert c8 == 8, f"expected 8 latent channels, got {c8}"
        xt = latent_8_B_C_T_H_W
        sigma_t: Optional[torch.Tensor] = None

        if use_precondition:
            if sigma is None:
                raise ValueError("sigma is required when use_precondition=True")
            edm = EDMScaling(sigma_data)
            if not isinstance(sigma, torch.Tensor):
                sigma_t = torch.full((b, 1), float(sigma), device=xt.device, dtype=xt.dtype)
            else:
                sigma_t = sigma.to(device=xt.device, dtype=xt.dtype)
                if sigma_t.ndim == 1:
                    sigma_t = sigma_t.unsqueeze(-1)
            sigma_b_1_t_1_1 = sigma_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(b, 1, t, 1, 1)
            c_skip, c_out, c_in, c_noise = edm(sigma_b_1_t_1_1)
            net_in = xt * c_in
            timesteps_B_T = pool_timesteps_to_patch_tokens(c_noise, self.patch_temporal)
        else:
            net_in = xt
            tok_t = t // self.patch_temporal
            if sigma is None:
                timesteps_B_T = torch.zeros(b, tok_t, device=xt.device, dtype=xt.dtype)
            else:
                edm = EDMScaling(sigma_data)
                if not isinstance(sigma, torch.Tensor):
                    sigma_t = torch.full((b, 1), float(sigma), device=xt.device, dtype=xt.dtype)
                else:
                    sigma_t = sigma.to(device=xt.device, dtype=xt.dtype)
                    if sigma_t.ndim == 1:
                        sigma_t = sigma_t.unsqueeze(-1)
                sigma_b_1_t_1_1 = sigma_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(b, 1, t, 1, 1)
                _, _, _, c_noise = edm(sigma_b_1_t_1_1)
                timesteps_B_T = pool_timesteps_to_patch_tokens(c_noise, self.patch_temporal)

        latent_9 = concat_padding_mask_channel(net_in, padding_mask)
        if num_blocks_run is None:
            net_out = self._embed_forward(latent_9, timesteps_B_T)
        else:
            net_out = self._embed_forward_partial(latent_9, timesteps_B_T, num_blocks_run)
        return net_out, xt, sigma_t, t

    @torch.no_grad()
    def forward(
        self,
        latent_8_B_C_T_H_W: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        use_precondition: bool = False,
        sigma_data: float = 0.5,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            latent_8: (B, 8, T, H, W), H/W divisible by ``FULL_PATCH_SPATIAL``; T padded to multiple of 2.
            sigma: optional (B, 1) or scalar; used for EDM c_in / c_noise and optional x0 composition.
            use_precondition: if True, apply c_in to latent before embed, timesteps = c_noise, and return x0_pred.
            sigma_data: EDM sigma_data (Text2World default 0.5).

        Returns:
            net_out: unpatchified network output (B, 8, T', H', W').
            x0_pred: if ``use_precondition``, ``c_skip * x_t + c_out * net_out`` on the 8-channel latent; else None.
        """
        net_out, xt, sigma_t, t = self._full_head_forward_core(
            latent_8_B_C_T_H_W, sigma, use_precondition, sigma_data, padding_mask, num_blocks_run=None
        )
        if not use_precondition:
            return net_out, None
        assert sigma_t is not None
        b = xt.shape[0]
        sigma_b_1_t_1_1 = sigma_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(b, 1, t, 1, 1)
        c_skip, c_out, _, _ = EDMScaling(sigma_data)(sigma_b_1_t_1_1)
        x0 = c_skip * xt + c_out * net_out
        return net_out, x0

    @torch.no_grad()
    def forward_upto_then_final(
        self,
        latent_8_B_C_T_H_W: torch.Tensor,
        num_blocks_run: int,
        sigma: Optional[torch.Tensor] = None,
        use_precondition: bool = True,
        sigma_data: float = 0.5,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run the first ``num_blocks_run`` transformer blocks (0 allowed), then ``final_layer`` + unpatchify.
        Same EDM preconditioning and ``x0_pred`` composition as ``forward`` when ``use_precondition``.

        Note: This is a **probe** — the final head was trained after the full depth; shallow activations are
        not guaranteed to be compatible with it.
        """
        net_out, xt, sigma_t, t = self._full_head_forward_core(
            latent_8_B_C_T_H_W,
            sigma,
            use_precondition,
            sigma_data,
            padding_mask,
            num_blocks_run=int(num_blocks_run),
        )
        if not use_precondition:
            return net_out, None
        assert sigma_t is not None
        b = xt.shape[0]
        sigma_b_1_t_1_1 = sigma_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(b, 1, t, 1, 1)
        c_skip, c_out, _, _ = EDMScaling(sigma_data)(sigma_b_1_t_1_1)
        x0 = c_skip * xt + c_out * net_out
        return net_out, x0

    @torch.no_grad()
    def forward_hidden(
        self,
        latent_8_B_C_T_H_W: torch.Tensor,
        num_blocks_run: Optional[int] = None,
        sigma: float = 0.5,
        sigma_data: float = 0.5,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        EDM-preconditioned forward through ``num_blocks_run`` DiT blocks (None = all).
        Returns raw hidden state ``[B, Tp, Hp, Wp, D]`` **before** ``final_layer``.

        Args:
            latent_8_B_C_T_H_W: (B, 8, T, H, W). T must be a multiple of ``FULL_PATCH_TEMPORAL`` (2).
            num_blocks_run: number of blocks to run (None = all self.blocks).
            sigma: noise level, used for EDM c_in / c_noise (default 0.5).
            sigma_data: EDM sigma_data (default 0.5).
            padding_mask: optional (B, 1, T, H, W) padding mask.

        Returns:
            x: (B, Tp, Hp, Wp, D=2048) hidden state after the last requested block.
        """
        b, c8, t, h, w = latent_8_B_C_T_H_W.shape
        assert c8 == 8, f"expected 8 latent channels, got {c8}"

        edm = EDMScaling(sigma_data)
        sigma_t = torch.full(
            (b, 1), float(sigma),
            device=latent_8_B_C_T_H_W.device, dtype=latent_8_B_C_T_H_W.dtype,
        )
        sigma_b1t11 = sigma_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(b, 1, t, 1, 1)
        _, _, c_in, c_noise = edm(sigma_b1t11)
        net_in = latent_8_B_C_T_H_W * c_in
        timesteps_B_T = pool_timesteps_to_patch_tokens(c_noise, self.patch_temporal)

        latent_9 = concat_padding_mask_channel(net_in, padding_mask)
        x = self.x_embedder(latent_9)
        ts_dtype = next(self.t_embedder.parameters()).dtype
        emb_B_T_D, adaln_lora_B_T_3D = self.t_embedder(
            timesteps_B_T.to(dtype=ts_dtype, device=latent_9.device)
        )
        emb_B_T_D = self.t_embedding_norm(emb_B_T_D).to(latent_9.dtype)
        adaln_lora_B_T_3D = adaln_lora_B_T_3D.to(latent_9.dtype)
        crossattn = torch.zeros(b, 1, CROSSATTN_DIM, dtype=latent_9.dtype, device=latent_9.device)

        n = len(self.blocks) if num_blocks_run is None else int(num_blocks_run)
        for block in self.blocks[:n]:
            x = block(x, emb_B_T_D, adaln_lora_B_T_3D, crossattn)
        return x  # [B, Tp, Hp, Wp, D]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CosmosDiTEarlyExit(nn.Module):
    """
    Loads patch embed + the first ``num_blocks`` transformer blocks (indices 0 .. num_blocks-1).
    If ``num_blocks == 0``, only patch embed + timestep embed run (no DiT blocks). All parameters
    are frozen. No transformer_engine dependency.
    """

    def __init__(self, ckpt_path: str, num_blocks: int = DEFAULT_NUM_DIT_BLOCKS,
                 device: str = "cpu"):
        super().__init__()
        if not (0 <= num_blocks <= NUM_DIT_BLOCKS):
            raise ValueError(
                f"num_blocks must be in [0, {NUM_DIT_BLOCKS}], got {num_blocks}"
            )
        self.num_blocks = num_blocks

        self.x_embedder = PatchEmbed(PATCH_SPATIAL, PATCH_TEMPORAL, IN_CHANNELS, HIDDEN_DIM)
        self.t_embedder = nn.Sequential(
            Timesteps(HIDDEN_DIM),
            TimestepEmbedding(HIDDEN_DIM, HIDDEN_DIM),
        )
        self.t_embedding_norm = RMSNorm(HIDDEN_DIM, eps=1e-6)

        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=HIDDEN_DIM,
                context_dim=CROSSATTN_DIM,
                num_heads=NUM_HEADS,
                head_dim=HEAD_DIM,
                lora_dim=ADALN_LORA_DIM,
                mlp_ratio=4.0,
            )
            for _ in range(num_blocks)
        ])

        self._load_weights(ckpt_path, device)

        for p in self.parameters():
            p.requires_grad_(False)

    def _load_weights(self, ckpt_path: str, device: str):
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        net = {k[4:]: v for k, v in sd.items() if k.startswith("net.")}

        # x_embedder: proj[1].weight
        self.x_embedder.proj[1].weight.data.copy_(net["x_embedder.proj.1.weight"])

        # t_embedder[1]: linear_1.weight, linear_2.weight
        self.t_embedder[1].linear_1.weight.data.copy_(net["t_embedder.1.linear_1.weight"])
        self.t_embedder[1].linear_2.weight.data.copy_(net["t_embedder.1.linear_2.weight"])

        # t_embedding_norm: weight (ignore _extra_state from TE)
        self.t_embedding_norm.weight.data.copy_(net["t_embedding_norm.weight"])

        # blocks
        for i, block in enumerate(self.blocks):
            p = f"blocks.{i}."
            b = {k[len(p):]: v for k, v in net.items() if k.startswith(p)}

            # self_attn
            block.self_attn.q_proj.weight.data.copy_(b["self_attn.q_proj.weight"])
            block.self_attn.k_proj.weight.data.copy_(b["self_attn.k_proj.weight"])
            block.self_attn.v_proj.weight.data.copy_(b["self_attn.v_proj.weight"])
            block.self_attn.output_proj.weight.data.copy_(b["self_attn.output_proj.weight"])
            block.self_attn.q_norm.weight.data.copy_(b["self_attn.q_norm.weight"])
            block.self_attn.k_norm.weight.data.copy_(b["self_attn.k_norm.weight"])

            # cross_attn
            block.cross_attn.q_proj.weight.data.copy_(b["cross_attn.q_proj.weight"])
            block.cross_attn.k_proj.weight.data.copy_(b["cross_attn.k_proj.weight"])
            block.cross_attn.v_proj.weight.data.copy_(b["cross_attn.v_proj.weight"])
            block.cross_attn.output_proj.weight.data.copy_(b["cross_attn.output_proj.weight"])
            block.cross_attn.q_norm.weight.data.copy_(b["cross_attn.q_norm.weight"])
            block.cross_attn.k_norm.weight.data.copy_(b["cross_attn.k_norm.weight"])

            # mlp
            block.mlp.layer1.weight.data.copy_(b["mlp.layer1.weight"])
            block.mlp.layer2.weight.data.copy_(b["mlp.layer2.weight"])

            # adaln_modulation_* → .1.weight (lora_in), .2.weight (lora_out)
            block.adaln_self_attn.layers[1].weight.data.copy_(b["adaln_modulation_self_attn.1.weight"])
            block.adaln_self_attn.layers[2].weight.data.copy_(b["adaln_modulation_self_attn.2.weight"])
            block.adaln_cross_attn.layers[1].weight.data.copy_(b["adaln_modulation_cross_attn.1.weight"])
            block.adaln_cross_attn.layers[2].weight.data.copy_(b["adaln_modulation_cross_attn.2.weight"])
            block.adaln_mlp.layers[1].weight.data.copy_(b["adaln_modulation_mlp.1.weight"])
            block.adaln_mlp.layers[2].weight.data.copy_(b["adaln_modulation_mlp.2.weight"])

    @torch.no_grad()
    def forward(self, latent: torch.Tensor, timestep: int = 0) -> torch.Tensor:
        """
        Args:
            latent: (B, 8, 1, H, W) — H and W divisible by 3.
            timestep: int, default 0.

        Returns:
            (B, T*Hp*Wp, 2048) — flattened patch tokens after ``num_blocks`` blocks (indices 0..num_blocks-1).
            For T=1, H=W=30, patch=3: shape = (B, 100, 2048).
        """
        B = latent.shape[0]
        device = latent.device
        dtype = latent.dtype

        # 1. Patch embed → (B, T, Hp, Wp, D)
        x = self.x_embedder(latent)

        # 2. Timestep embedding (match t_embedder weight dtype for bf16 models)
        ts_dtype = next(self.t_embedder.parameters()).dtype
        ts = torch.full((B, 1), float(timestep), dtype=ts_dtype, device=device)
        emb_B_T_D, adaln_lora_B_T_3D = self.t_embedder(ts)
        emb_B_T_D = self.t_embedding_norm(emb_B_T_D)
        emb_B_T_D = emb_B_T_D.to(dtype)
        adaln_lora_B_T_3D = adaln_lora_B_T_3D.to(dtype)

        # 3. Dummy cross-attention context (zeros since we extract features, not generate)
        crossattn = torch.zeros(B, 1, CROSSATTN_DIM, dtype=dtype, device=device)

        # 4. Run blocks
        for block in self.blocks:
            x = block(x, emb_B_T_D, adaln_lora_B_T_3D, crossattn)

        # 5. Flatten: (B, T, Hp, Wp, D) → (B, T*Hp*Wp, D)
        B_, T, Hp, Wp, D = x.shape
        return x.reshape(B_, T * Hp * Wp, D)

    @torch.no_grad()
    def forward_spatial(self, latent: torch.Tensor, timestep: int = 0) -> torch.Tensor:
        """Returns (B, T, Hp, Wp, D) before token flattening."""
        B = latent.shape[0]
        device = latent.device
        dtype = latent.dtype
        x = self.x_embedder(latent)
        ts_dtype = next(self.t_embedder.parameters()).dtype
        ts = torch.full((B, 1), float(timestep), dtype=ts_dtype, device=device)
        emb_B_T_D, adaln_lora_B_T_3D = self.t_embedder(ts)
        emb_B_T_D = self.t_embedding_norm(emb_B_T_D).to(dtype)
        adaln_lora_B_T_3D = adaln_lora_B_T_3D.to(dtype)
        crossattn = torch.zeros(B, 1, CROSSATTN_DIM, dtype=dtype, device=device)
        for block in self.blocks:
            x = block(x, emb_B_T_D, adaln_lora_B_T_3D, crossattn)
        return x

    @torch.no_grad()
    def forward_spatial_at_depths(
        self,
        latent: torch.Tensor,
        timestep: int,
        block_depths: list[int],
    ) -> dict[int, torch.Tensor]:
        """
        Run patch embed + blocks; after each block index k (1-based count), save activations.

        Args:
            latent: (B, 8, 1, H, W)
            block_depths: e.g. ``[5, 10, 15]`` = snapshot after 5 / 10 / 15 blocks (must be ≤ ``self.num_blocks``).

        Returns:
            ``{depth: x_B_T_Hp_Wp_D}`` for each requested depth.
        """
        want = sorted(set(block_depths))
        for d in want:
            if not (1 <= d <= self.num_blocks):
                raise ValueError(f"block depth {d} not in [1, {self.num_blocks}]")
        want_set = set(want)
        B = latent.shape[0]
        device = latent.device
        dtype = latent.dtype
        x = self.x_embedder(latent)
        ts_dtype = next(self.t_embedder.parameters()).dtype
        ts = torch.full((B, 1), float(timestep), dtype=ts_dtype, device=device)
        emb_B_T_D, adaln_lora_B_T_3D = self.t_embedder(ts)
        emb_B_T_D = self.t_embedding_norm(emb_B_T_D).to(dtype)
        adaln_lora_B_T_3D = adaln_lora_B_T_3D.to(dtype)
        crossattn = torch.zeros(B, 1, CROSSATTN_DIM, dtype=dtype, device=device)
        out: dict[int, torch.Tensor] = {}
        for i, block in enumerate(self.blocks):
            x = block(x, emb_B_T_D, adaln_lora_B_T_3D, crossattn)
            if (i + 1) in want_set:
                out[i + 1] = x.clone()
        return out


class DitPatchVectorizerConv(nn.Module):
    """(B, Hp, Wp, D) → (B, D) via depthwise spatial conv + pointwise 1×1."""

    def __init__(self, dim: int = HIDDEN_DIM, hp: int = DIT_HP, wp: int = DIT_WP):
        super().__init__()
        self.hp = hp
        self.wp = wp
        self.dw = nn.Conv2d(dim, dim, kernel_size=(hp, wp), groups=dim)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Hp, Wp, D)
        x = x.permute(0, 3, 1, 2)
        x = self.dw(x)
        x = self.pw(x)
        return x.flatten(1)


class DitPatchVectorizerAttn(nn.Module):
    """Single learnable query cross-attends over flattened patch tokens → (B, D)."""

    def __init__(self, dim: int = HIDDEN_DIM, num_heads: int = NUM_HEADS):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, d = x.shape
        seq = x.reshape(b, -1, d)
        q = self.query.expand(b, -1, -1)
        out, _ = self.mha(q, seq, seq, need_weights=False)
        return out.squeeze(1)


class DitPatchVectorizerQueryStyle(nn.Module):
    """
    Query-branch style stack: pre_mlp → cross-attn (learnable query) → GELU → post_mlp → (B, D).

    Mirrors ``LatentDownsampleCrossAttn`` on the remote ``query`` branch (pre → MHA → post_act →
    post_mlp); last linear is D→D so outputs stay in DiT feature space for ``dit_gt_to_llm`` / sim_loss.
    """

    def __init__(self, dim: int = HIDDEN_DIM, num_heads: int = NUM_HEADS):
        super().__init__()
        self.pre_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.query_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=0.0
        )
        self.post_act = nn.GELU()
        self.post_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Hp, Wp, D)
        b, _, _, d = x.shape
        kv = x.reshape(b, -1, d)
        kv = self.pre_mlp(kv)
        q = self.query_token.expand(b, -1, -1)
        out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        out = self.post_act(out)
        out = self.post_mlp(out)
        return out.squeeze(1)


class DitPatchVectorizerAvgPool(nn.Module):
    """Spatial average pooling over DiT patch grid.

    - ``grid_h == grid_w == 1``: global average → ``(B, D)`` (legacy).
    - Otherwise: ``adaptive_avg_pool2d`` to ``(grid_h, grid_w)`` → ``(B, grid_h*grid_w, D)``.
    """

    def forward(
        self,
        x: torch.Tensor,
        grid_h: int = 1,
        grid_w: int = 1,
    ) -> torch.Tensor:
        if grid_h == 1 and grid_w == 1:
            return x.mean(dim=(1, 2))
        x = x.permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, (grid_h, grid_w))
        x = x.permute(0, 2, 3, 1)
        return x.reshape(x.shape[0], grid_h * grid_w, -1)


# ---------------------------------------------------------------------------
# Pipeline utilities
# ---------------------------------------------------------------------------

def build_dit_gt_features(
    vae_latent_BN_16_H_W: torch.Tensor,
    vae_to_dit_proj: nn.Linear,
    dit_model: CosmosDiTEarlyExit,
    num_frames: int = 4,
    crop_size: int = CROP_SIZE,
) -> torch.Tensor:
    """
    Compute GT features: CI8x8 VAE latent → DiT early-exit → GAP → (B*num_frames, 2048).

    vae_to_dit_proj is trainable (receives gradients); dit_model blocks are frozen.
    """
    assert crop_size % PATCH_SPATIAL == 0
    BN, C, H, W = vae_latent_BN_16_H_W.shape
    assert C == 16

    x = vae_latent_BN_16_H_W.permute(0, 2, 3, 1)
    x = vae_to_dit_proj(x)
    x = x.permute(0, 3, 1, 2)

    h0 = (H - crop_size) // 2
    w0 = (W - crop_size) // 2
    x = x[:, :, h0:h0 + crop_size, w0:w0 + crop_size]
    x = x.unsqueeze(2)  # (BN, 8, 1, crop, crop)

    features = dit_model(x, timestep=0)   # (BN, L, 2048)
    features = features.mean(dim=1)       # (BN, 2048)
    return features


def build_dit_pred_features(
    inferred_embeddings_all: torch.Tensor,
    batch_size: int,
    num_frames: int,
    latent_size: int,
    dit_out_proj: nn.Linear,
) -> torch.Tensor:
    """
    Map LLM output → DiT feature space: (B, latent_size, D_llm) → (B*num_frames, 2048).
    latent_size = num_frames * tokens_per_frame (e.g. 4 frames × 1 token = 4).
    """
    tpf = latent_size // num_frames
    x = inferred_embeddings_all.view(batch_size, num_frames, tpf, -1)
    x = x.mean(dim=2)
    x = x.reshape(batch_size * num_frames, -1)
    wdtype = next(dit_out_proj.parameters()).dtype
    return dit_out_proj(x.to(wdtype)).float()
