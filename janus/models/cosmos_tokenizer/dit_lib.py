"""
CosmosDiTEarlyExit: self-contained early-exit wrapper for Cosmos-Policy DiT.

No dependency on transformer_engine or cosmos-policy source code.
All components are reimplemented from checkpoint weight inspection.

Checkpoint facts (Cosmos-Policy-LIBERO-Predict2-2B):
  - x_embedder.proj.1.weight: (2048, 72) → C=8, patch_spatial=3, patch_temporal=1
  - 28 total blocks; early exit after block index 10 (11 blocks: 0–10 inclusive)
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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

CROP_SIZE = 30        # 30 is divisible by patch_spatial=3; nearest below 32
PATCH_SPATIAL = 3
PATCH_TEMPORAL = 1
IN_CHANNELS = 8       # DiT x_embedder in_channels (from checkpoint: 72 = 8×3×3×1)
HIDDEN_DIM = 2048
NUM_HEADS = 16
HEAD_DIM = HIDDEN_DIM // NUM_HEADS  # 128
EARLY_EXIT_BLOCK = 10  # exit after block index 10 (0-indexed), 11 blocks total
CROSSATTN_DIM = 1024
ADALN_LORA_DIM = 256

DIT_HP = CROP_SIZE // PATCH_SPATIAL # 10
DIT_WP = CROP_SIZE // PATCH_SPATIAL # 10

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


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CosmosDiTEarlyExit(nn.Module):
    """
    Loads the first `num_exit_blocks+1` blocks of Cosmos-Policy DiT.
    All parameters are frozen. No transformer_engine dependency.
    """

    def __init__(self, ckpt_path: str, num_exit_blocks: int = EARLY_EXIT_BLOCK,
                 device: str = "cpu"):
        super().__init__()
        self.num_exit_blocks = num_exit_blocks

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
            for _ in range(num_exit_blocks + 1)
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
            (B, T*Hp*Wp, 2048) — flattened patch tokens after `num_exit_blocks+1` blocks.
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


class DitPatchVectorizerAvgPool(nn.Module):
    """(B, Hp, Wp, D) → (B, D) via mean over spatial dims (global average pooling)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(1, 2))


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
