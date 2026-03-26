"""
WanVAE Encoder: self-contained encoder extracted from wan2pt1.py.

No dependency on megatron, transformer_engine, or cosmos_predict2 source code.
All components are copied verbatim from the original wan2pt1.py, with the
following removed:
  - distributed utilities (broadcast, get_rank, sync_model_states)
  - megatron / parallel_state imports
  - Decoder3d (not needed for encoding)
  - WanVAE_ full class (only encoder + conv1 needed)
  - scale / cache parameters from encode() (simplified for frozen inference)

Checkpoint: /mnt/wfm/ckpt/ckpt/pretrained/Cosmos-Predict2.5-2B/tokenizer.pth
Architecture (inferred from checkpoint keys):
  dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2,
  attn_scales=[], temperal_downsample=[True, True, False]

encode() input:  (B, 3, T, H, W) float32 in [0,1] or [-1,1]
               T=1 for single frame; H=W=240 → latent H=W=30
encode() output: (B, 16, T', H', W') where T'=T (no temporal compression for T=1)
                 spatial compression = 8x: 240 → 30

Usage:
    vae = WanVAEEncoder('/path/to/tokenizer.pth', device='cuda')
    latent = vae.encode(frames)        # (B, 16, 1, 30, 30) for 1-frame input
    mu = latent[:, :8]                 # (B, 8, 1, 30, 30) — mean channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["WanVAEEncoder", "WanVAECodec", "WanVaeLatentBridge"]

CACHE_T = 2


# ---------------------------------------------------------------------------
# Building blocks (copied from wan2pt1.py, all einops replaced with reshape)
# ---------------------------------------------------------------------------


class Upsample(nn.Upsample):
    """Nearest upsample with bfloat16-friendly forward (match cosmos wan2pt1)."""

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class CausalConv3d(nn.Conv3d):
    """Causal 3d convolution — pads time dimension on the left only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2], self.padding[2],
            self.padding[1], self.padding[1],
            2 * self.padding[0], 0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        # Spatial upsample (T=1 or no temporal cache): same 2D path for upsample2d / upsample3d
        if self.mode in ("upsample2d", "upsample3d") and feat_cache is None:
            x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            x_2d = self.resample(x_2d)
            c_out, h_new, w_new = x_2d.shape[1], x_2d.shape[2], x_2d.shape[3]
            x = x_2d.reshape(b, t, c_out, h_new, w_new).permute(0, 2, 1, 3, 4)
            return x

        # resample on spatial dims: reshape to (b*t, c, h, w)
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x_2d = self.resample(x_2d)
        t_new = t
        h_new, w_new = x_2d.shape[-2], x_2d.shape[-1]
        x = x_2d.reshape(b, t_new, c, h_new, w_new).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            # time_conv stride=2 requires T>1 or a cache frame prepended.
            # For T=1 single-frame inference (no cache), skip temporal compression.
            # The model was trained for multi-frame; for T=1 we keep T=1.
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
            # else: no cache → skip time_conv for T=1 (temporal dim unchanged)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """Single-head causal self-attention (spatial only)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        # reshape to (b*t, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.norm(x)
        q, k, v = (
            self.to_qkv(x)
            .reshape(b * t, 1, c * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        # reshape back to (b, c, t, h, w)
        x = x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return x + identity


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_downsample=(True, True, False),
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = list(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)
        self.temperal_downsample = list(temperal_downsample)

        dims = [dim * u for u in [1] + list(dim_mult)]
        scale = 1.0

        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x):
        # Simple non-cached forward (no streaming cache needed for training)
        x = self.conv1(x)
        for layer in self.downsamples:
            x = layer(x)
        for layer in self.middle:
            x = layer(x)
        for layer in self.head:
            x = layer(x)
        return x


class Decoder3d(nn.Module):
    """WAN VAE decoder (tokenizer.pth ``decoder.*``); T=1, feat_cache=None for visualization."""

    def __init__(
        self,
        dim=96,
        z_dim=16,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_upsample=(True, True, False),
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        dim_mult = tuple(dim_mult)
        dims = [dim * u for u in (dim_mult[-1],) + tuple(dim_mult[::-1])]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i in (1, 2, 3):
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)
        out_dim = dims[-1]
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

class WanVAEEncoder(nn.Module):
    """
    Self-contained WAN VAE encoder.

    Loads weights from tokenizer.pth without any cosmos_predict2 dependency.
    Architecture params auto-detected from checkpoint shapes.

    Args:
        ckpt_path: path to tokenizer.pth
        device:    'cpu' or 'cuda'

    encode() interface:
        Input:  (B, 3, T, H, W)  — pixel values in [-1, 1], float32
                For single-frame: T=1, H=W=240
        Output: (B, 16, T', H', W')  — VAE latent (mu=first 8ch, logvar=last 8ch)
                For T=1, H=W=240: output (B, 16, 1, 30, 30)

    To get DiT-compatible input (8 channels = mean):
        mu = encoder.encode(x)[:, :8]  # (B, 8, 1, 30, 30)
    """

    # Fixed architecture matching tokenizer.pth
    # Verified from checkpoint keys:
    #   encoder.conv1.weight (96, 3, ...) → dim=96
    #   encoder.head.2.weight (32, 384, ...) → encoder z_dim=32 (= latent_z_dim*2)
    #   downsamples.2  → downsample2d (no time_conv) → temperal_downsample[0]=False
    #   downsamples.5  → downsample3d (has time_conv) → temperal_downsample[1]=True
    #   downsamples.8  → downsample3d (has time_conv) → temperal_downsample[2]=True
    #   conv1.weight (32, 32, ...) → CausalConv3d(32, 32, 1)
    #   → mu = conv1_out[:, :16], logvar = conv1_out[:, 16:]
    DIM = 96
    ENC_Z_DIM = 32      # encoder output channels = latent_z_dim * 2 = 16 * 2
    LATENT_Z_DIM = 16   # actual latent z_dim (mu has 16ch, logvar has 16ch)
    DIM_MULT = (1, 2, 4, 4)
    NUM_RES_BLOCKS = 2
    ATTN_SCALES = ()
    TEMPORAL_DOWNSAMPLE = (False, True, True)   # verified from ckpt: ds2→ds5→ds8

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        super().__init__()

        # Build encoder (outputs ENC_Z_DIM=32 channels: [mu(16) || logvar(16)])
        self.encoder = Encoder3d(
            dim=self.DIM,
            z_dim=self.ENC_Z_DIM,
            dim_mult=self.DIM_MULT,
            num_res_blocks=self.NUM_RES_BLOCKS,
            attn_scales=self.ATTN_SCALES,
            temperal_downsample=self.TEMPORAL_DOWNSAMPLE,
            dropout=0.0,
        )

        # conv1 = CausalConv3d(32→32, 1×1×1), then .chunk(2) → mu(16), logvar(16)
        self.conv1 = CausalConv3d(self.ENC_Z_DIM, self.ENC_Z_DIM, 1)

        self._load_weights(ckpt_path, device)

        # Freeze — encoder used only for frozen GT feature extraction
        for p in self.parameters():
            p.requires_grad_(False)

    def _load_weights(self, ckpt_path: str, device: str):
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Load encoder weights
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=True)
        if missing:
            print(f"[WanVAEEncoder] Missing encoder keys: {missing}")
        if unexpected:
            print(f"[WanVAEEncoder] Unexpected encoder keys: {unexpected}")

        # Load conv1 weights
        conv1_sd = {k[len("conv1."):]: v for k, v in sd.items() if k.startswith("conv1.")}
        missing2, unexpected2 = self.conv1.load_state_dict(conv1_sd, strict=True)
        if missing2:
            print(f"[WanVAEEncoder] Missing conv1 keys: {missing2}")

        print(f"[WanVAEEncoder] Loaded weights from {ckpt_path}")

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, T, H, W), pixel values in [-1, 1].
               Activations are cast to ``conv1.weight.dtype`` so they match frozen WAN
               parameters (bf16 after parent ``.to(torch.bfloat16)``, or fp32).

        Returns:
            latent: (B, 32, T', H', W') — [mu(16ch) || logvar(16ch)] after conv1
        """
        w_dtype = self.conv1.weight.dtype
        x = x.to(w_dtype)
        out = self.encoder(x)
        out = self.conv1(out)
        return out


class WanVAECodec(nn.Module):
    """
    Full WAN tokenizer forward for **single-frame** recon: encode → mu (16ch) → conv2 → decoder → RGB.
    Uses ``tokenizer.pth`` keys ``encoder.*``, ``conv1.*`` (via :class:`WanVAEEncoder`), ``conv2.*``, ``decoder.*``.
    """

    _TEMPORAL_UPSAMPLE = tuple(reversed(WanVAEEncoder.TEMPORAL_DOWNSAMPLE))

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        super().__init__()
        self.encoder = WanVAEEncoder(ckpt_path, device=device)
        self.conv2 = CausalConv3d(
            WanVAEEncoder.LATENT_Z_DIM,
            WanVAEEncoder.LATENT_Z_DIM,
            1,
        )
        self.decoder = Decoder3d(
            dim=WanVAEEncoder.DIM,
            z_dim=WanVAEEncoder.LATENT_Z_DIM,
            dim_mult=WanVAEEncoder.DIM_MULT,
            num_res_blocks=WanVAEEncoder.NUM_RES_BLOCKS,
            attn_scales=WanVAEEncoder.ATTN_SCALES,
            temperal_upsample=self._TEMPORAL_UPSAMPLE,
            dropout=0.0,
        )
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        dec_sd = {k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")}
        dm, du = self.decoder.load_state_dict(dec_sd, strict=True)
        if dm:
            print(f"[WanVAECodec] decoder missing: {dm}")
        if du:
            print(f"[WanVAECodec] decoder unexpected: {du}")
        c2_sd = {k[len("conv2."):]: v for k, v in sd.items() if k.startswith("conv2.")}
        self.conv2.load_state_dict(c2_sd, strict=True)
        for p in self.conv2.parameters():
            p.requires_grad_(False)
        for p in self.decoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(x)

    @torch.no_grad()
    def decode_mu(self, mu: torch.Tensor) -> torch.Tensor:
        """mu: (B, 16, T, H, W) → RGB (B, 3, T, H', W')."""
        w_dtype = next(self.conv2.parameters()).dtype
        z = self.conv2(mu.to(w_dtype))
        return self.decoder(z, feat_cache=None, feat_idx=[0])

    @torch.no_grad()
    def recon(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, T, H, W) in [-1, 1] → reconstruction in [-1, 1]."""
        lat = self.encode_latent(x)
        mu = lat[:, : WanVAEEncoder.LATENT_Z_DIM]
        return self.decode_mu(mu)


class WanVaeLatentBridge(nn.Module):
    """
    Same role as ``ImageTokenizer.encode`` for the LLM gen branch: map VLM crops to a 16-channel
    spatial latent. Uses the Predict2.5 ``tokenizer.pth`` (WAN encoder), matching ``CosmosDiTEarlyExit``.

    Input:  (B, 3, H, W) in [-1, 1] (e.g. 256×256 from ``convert_to_cosmos_input``).
    Output: (B, 16, 30, 30) for 240×240 internal resize — same geometry as ``build_wan_vae_gt_features``.
    """

    def __init__(self, wan_encoder: WanVAEEncoder):
        super().__init__()
        self._wan = wan_encoder

    @torch.no_grad()
    def encode(self, input_tensor: torch.Tensor) -> torch.Tensor:
        original_dtype = input_tensor.dtype
        x = F.interpolate(
            input_tensor.float(),
            size=(240, 240),
            mode="bilinear",
            align_corners=False,
        )
        x = x.unsqueeze(2)
        latent = self._wan.encode(x)
        out = latent[:, :16, 0, :, :].to(original_dtype)
        return out

    def decode(self, input_latent: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "WAN tokenizer.pth bridge is encode-only. Pixel decode needs the legacy Cosmos .jit tokenizer."
        )


def build_wan_vae_gt_features(
    frames_BNCHW: torch.Tensor,
    vae: WanVAEEncoder,
    dit_model,
    num_frames: int = 4,
) -> torch.Tensor:
    """
    Compute GT DiT features using WAN VAE + DiT early exit.

    Pipeline:
      frames (B*N, 3, H, W) normalized to [-1,1]
      → resize to 240×240
      → unsqueeze T=1 → (B*N, 3, 1, 240, 240)
      → WAN VAE encode → (B*N, 16, 1, 30, 30)
      → take mean channels [:, :8] → (B*N, 8, 1, 30, 30)
      → DiT early exit → (B*N, 100, 2048)
      → GAP → (B*N, 2048)

    Args:
        frames_BNCHW: (B*num_frames, 3, H, W) — raw image tensors ([-1,1])
        vae: WanVAEEncoder (frozen)
        dit_model: CosmosDiTEarlyExit (frozen)
        num_frames: number of action frames per sample

    Returns:
        gt_features: (B*num_frames, 2048)
    """
    import torch.nn.functional as F_

    BN = frames_BNCHW.shape[0]
    device = frames_BNCHW.device
    dtype = frames_BNCHW.dtype

    # 1. Resize to 240×240
    x = F_.interpolate(frames_BNCHW, size=(240, 240), mode="bilinear", align_corners=False)
    # x: (BN, 3, 240, 240)

    # 2. Unsqueeze temporal dimension
    x = x.unsqueeze(2)   # (BN, 3, 1, 240, 240)

    # 3. WAN VAE encode → (BN, 16, 1, 30, 30)
    vae = vae.to(device)
    latent = vae.encode(x.float())  # always run VAE in float32

    # 4. Take mean channels (first 8)
    mu = latent[:, :8]   # (BN, 8, 1, 30, 30)
    mu = mu.to(dtype)

    # 5. DiT early exit → (BN, 100, 2048)
    features = dit_model(mu, timestep=0)

    # 6. GAP
    features = features.mean(dim=1)  # (BN, 2048)
    return features
