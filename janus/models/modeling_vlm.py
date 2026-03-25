# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import math
import os
import torch
from attrdict import AttrDict
from einops import rearrange
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.cache_utils import DynamicCache

from janus.models.clip_encoder import CLIPVisionTower
from janus.models.projector import MlpProjector
from janus.diffusion import ActionEmbedder, TimestepEmbedder, FinalLayer
from janus.uni3d import Uni3D
import torch.nn as nn

class vision_head(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = torch.nn.Linear(
            params.n_embed, params.image_token_embed
        )
        self.vision_activation = torch.nn.GELU()
        self.vision_head = torch.nn.Linear(
            params.image_token_embed, params.image_token_size
        )

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x


def model_name_to_cls(cls_name):
    if "MlpProjector" in cls_name:
        cls = MlpProjector

    elif "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower

    elif "VQ" in cls_name:
        from janus.models.vq_model import VQ_models

        cls = VQ_models[cls_name]
    elif "vision_head" in cls_name:
        cls = vision_head
    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls


class LatentDownsampleCrossAttn(nn.Module):
    """
    Compresses a VAE latent map (B*N, C, H, W) into 1 token per frame (B*N, 1, hidden_size).

    Pipeline:
      1. Reshape gt_latent to (B*N, C, H*W) — C key/value tokens of dim kv_dim=H*W
      2. pre_mlp  : 2-layer MLP applied to each kv token  (kv_dim → kv_dim)
      3. cross_attn: learnable query (1, kv_dim) attends to the C processed kv tokens
      4. post_act + post_mlp: GELU + 2-layer MLP  (kv_dim → hidden_size)

    Dimensions (default Cosmos CI8x8 + 256x256):
        vae_channels=16, vae_spatial=32  →  kv_dim=1024, hidden_size=2048
    """
    def __init__(
        self,
        vae_channels: int,
        vae_spatial: int,
        hidden_size: int,
        num_heads: int = 8,
    ):
        super().__init__()
        kv_dim = vae_spatial * vae_spatial          # 32*32 = 1024
        self.query_token = nn.Parameter(torch.zeros(1, 1, kv_dim))
        self.pre_mlp = nn.Sequential(
            nn.Linear(kv_dim, kv_dim),
            nn.GELU(),
            nn.Linear(kv_dim, kv_dim),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=kv_dim, num_heads=num_heads, batch_first=True
        )
        self.post_act = nn.GELU()
        self.post_mlp = nn.Sequential(
            nn.Linear(kv_dim, kv_dim),
            nn.GELU(),
            nn.Linear(kv_dim, hidden_size),
        )

    def forward(self, gt_latent: torch.Tensor) -> torch.Tensor:
        # gt_latent: (B*N, C, H, W)
        bn, c, h, w = gt_latent.shape
        kv = gt_latent.reshape(bn, c, h * w)              # (B*N, C, kv_dim)
        kv = self.pre_mlp(kv)                             # (B*N, C, kv_dim)
        q = self.query_token.expand(bn, -1, -1)           # (B*N, 1, kv_dim)
        out, _ = self.cross_attn(q, kv, kv)               # (B*N, 1, kv_dim)
        out = self.post_act(out)
        out = self.post_mlp(out)                           # (B*N, 1, hidden_size)
        return out


class LatentUpsampleCrossAttn(nn.Module):
    """
    Decodes 1 latent token per frame (B*N, 1, hidden_size) back to a VAE latent map
    (B*N, C, H, W).

    Pipeline:
      1. pre_mlp  : 2-layer MLP applied to the latent token  (hidden_size → hidden_size)
      2. cross_attn: C learnable query tokens (C, hidden_size) attend to the processed token
      3. post_act + post_mlp: GELU + 2-layer MLP  (hidden_size → H*W)
      4. Reshape to (B*N, C, H, W)

    Dimensions (default Cosmos CI8x8 + 256x256):
        vae_channels=16, vae_spatial=32  →  out_dim=1024, hidden_size=2048
    """
    def __init__(
        self,
        hidden_size: int,
        vae_channels: int,
        vae_spatial: int,
        num_heads: int = 16,
    ):
        super().__init__()
        n_q = vae_channels                              # 16
        out_dim = vae_spatial * vae_spatial             # 32*32 = 1024
        self.query_tokens = nn.Parameter(torch.zeros(1, n_q, hidden_size))
        self.pre_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        self.post_act = nn.GELU()
        self.post_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_dim),
        )
        self.n_q = n_q
        self.vae_spatial = vae_spatial

    def forward(self, latent_token: torch.Tensor) -> torch.Tensor:
        # latent_token: (B*N, 1, hidden_size)
        bn = latent_token.shape[0]
        kv = self.pre_mlp(latent_token)                              # (B*N, 1, hidden_size)
        q = self.query_tokens.expand(bn, -1, -1)                    # (B*N, n_q, hidden_size)
        out, _ = self.cross_attn(q, kv, kv)                         # (B*N, n_q, hidden_size)
        out = self.post_act(out)
        out = self.post_mlp(out)                                     # (B*N, n_q, out_dim)
        out = out.reshape(bn, self.n_q, self.vae_spatial, self.vae_spatial)
        return out                                                   # (B*N, C, H, W)


class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class AlignerConfig(PretrainedConfig):
    model_type = "aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenVisionConfig(PretrainedConfig):
    model_type = "gen_vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenAlignerConfig(PretrainedConfig):
    model_type = "gen_aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenHeadConfig(PretrainedConfig):
    model_type = "gen_head"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))

class MLPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, mlp_type: str = "gelu-mlp") -> None:
        super().__init__()
        if mlp_type == "gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)
    
    def initialize_weights(self):
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig

    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig

    language_config: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionConfig(**vision_config)

        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)

        gen_vision_config = kwargs.get("gen_vision_config", {})
        self.gen_vision_config = GenVisionConfig(**gen_vision_config)

        gen_aligner_config = kwargs.get("gen_aligner_config", {})
        self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)

        gen_head_config = kwargs.get("gen_head_config", {})
        self.gen_head_config = GenHeadConfig(**gen_head_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)


class MultiModalityPreTrainedModel(PreTrainedModel):
    config_class = MultiModalityConfig
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class MultiModalityCausalLM(MultiModalityPreTrainedModel):
    def __init__(self, config: MultiModalityConfig,
                action_dim = None,
                action_chunk = None,
                flow = False,
                use_latent = True,
                robot_state = False,
                use_pointcloud = False,
                fast_and_slow = False,
                fast_image_num = 1,
                vision_backend = 'cosmos_vae',
                cosmos_scale_factor = None,   # kept for backward compat, unused
                cosmos_side = None,           # spatial side of Cosmos VAE output (e.g. 32 for CI8x8 + 256x256)
                load_cosmos_tokenizer = True,
                latent_downsample_mode: str = "cross_attn",  # kept for compat, always cross_attn
            ):
        super().__init__(config)
        if cosmos_side is not None:
            self.config.cosmos_side = cosmos_side
        self.flow = flow
        self.use_pointcloud = use_pointcloud
        self.fast_and_slow = fast_and_slow
        self.fast_image_num = fast_image_num
        self.use_latent = use_latent
        self.robot_state = robot_state
        self.action_dim = action_dim
        self.action_chunk = action_chunk

        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)

        gen_vision_config = config.gen_vision_config
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
        self.gen_vision_model = gen_vision_cls()
        self.gen_vision_config = gen_vision_config

        gen_aligner_config = config.gen_aligner_config
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)

        gen_head_config = config.gen_head_config
        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)

        self.gen_embed = torch.nn.Embedding(
            gen_vision_config.params.image_token_size, gen_vision_config.params.n_embed
        )

        language_config = config.language_config
        language_config.torch_dtype = torch.bfloat16
        language_config.bf16 = True
        self.language_model = LlamaForCausalLM(language_config)

        # make config like a llm config
        for key, value in language_config.__dict__.items():
            if key not in self.config.__dict__:
                setattr(self.config, key, value)

        if self.flow:
            self.x_embedder = ActionEmbedder(action_size=action_dim, hidden_size=language_config.hidden_size)
            if self.robot_state:
                self.state_embedder = ActionEmbedder(action_size=action_dim, hidden_size=language_config.hidden_size)
            self.t_embedder = TimestepEmbedder(language_config.hidden_size)
            self.final_layer = FinalLayer(language_config.hidden_size, action_dim)

        if vision_backend == 'cosmos_vae':
            from janus.models.cosmos_tokenizer.image_lib import ImageTokenizer

            hidden_size = language_config.hidden_size
            vae_dim = 16
            # Spatial side of the Cosmos VAE latent (32 for CI8x8 tokenizer + 256x256 input).
            _cosmos_side = int(getattr(self.config, "cosmos_side", 32))
            self.config.cosmos_side = _cosmos_side

            cosmos_ckpt_dir = os.environ.get(
                "COSMOS_TOKENIZER_DIR",
                "/mnt/dataset/share/hwb/hf_cache/Cosmos-0.1-Tokenizer-CI8x8",
            )
            if load_cosmos_tokenizer:
                self.cosmos_tokenizer = ImageTokenizer(
                    checkpoint_enc=f"{cosmos_ckpt_dir}/encoder.jit",
                    checkpoint_dec=f"{cosmos_ckpt_dir}/decoder.jit",
                )
                for p in self.cosmos_tokenizer.parameters():
                    p.requires_grad = False
            else:
                self.cosmos_tokenizer = None

            # Cross-attention based compress / decompress:
            #   latent_compressor : (B*N, C, H, W) -> (B*N, 1, hidden_size)
            #   latent_decompressor: (B*N, 1, hidden_size) -> (B*N, C, H, W)
            self.latent_compressor = LatentDownsampleCrossAttn(
                vae_channels=vae_dim,
                vae_spatial=_cosmos_side,
                hidden_size=hidden_size,
            )
            self.latent_decompressor = LatentUpsampleCrossAttn(
                hidden_size=hidden_size,
                vae_channels=vae_dim,
                vae_spatial=_cosmos_side,
            )

        if self.use_latent and self.use_pointcloud:
            print("Using pointcloud embedder") 
            import timm
            from types import SimpleNamespace

            uni3d_args = SimpleNamespace(
                pc_model='eva02_base_patch14_448.mim_in22k_ft_in1k', # Uni3D Base ViT
                pretrained_pc='', 
                drop_path_rate=0.0,
                pc_feat_dim=768, 
                embed_dim=1024, 
                group_size=64,
                num_group=512,  
                pc_encoder_dim=512, 
                patch_dropout=0.0,
            )

            point_transformer = timm.create_model(uni3d_args.pc_model, checkpoint_path='', drop_path_rate=0.0)
            self.pointcloud_embedder = Uni3D(point_transformer, uni3d_args)

            self.projector_3d = MLPProjector(self.pointcloud_embedder.embed_dim, language_config.hidden_size)
            print("Pointcloud embedder initialized.")

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0) 
                nn.init.constant_(module.bias, 0)     

        self.apply(_basic_init)

    def load_encoder_to_pointcloud_embedder(self, ckpt_path):
        print(f"=> Loading Uni3D checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        if 'module' in checkpoint:
            state_dict = checkpoint['module']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint: 
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'): k = k[7:]
            if k.startswith('point_encoder.'): k = k[14:]
            if k == 'logit_scale': continue
            if 'encoder.first_conv.0.weight' in k:
                if v.shape[1] == 6:
                    print(f"Surgery on {k}: {v.shape} -> slicing first 3 channels (XYZ)")
                    v = v[:, :3, :]
            
            new_state_dict[k] = v
        missing, unexpected = self.pointcloud_embedder.load_state_dict(new_state_dict, strict=False)
        
        print(f"Successfully loaded parameters to self.pointcloud_embedder")
        if missing:
            print(f"! Missed ({len(missing)}): {missing[:3]}...")
        if unexpected:
            print(f"! Unexpected ({len(unexpected)}): {unexpected[:3]}...")

    def denoise_step(self, inputs_embeds, past_key_values, x_t, timestep):
        noisy_actions = self.x_embedder(x_t.to(torch.bfloat16))
        timesteps = self.t_embedder(timestep).unsqueeze(1)

        if past_key_values is None:
            inputs_embeds = torch.cat([
                inputs_embeds,
                timesteps,
                noisy_actions,
            ], dim=1)
            if not self.fast_and_slow:
                latent_indexes=torch.arange(0, inputs_embeds.shape[1]-3).to(inputs_embeds.device)
                action_indexes=torch.arange(inputs_embeds.shape[1]-3, inputs_embeds.shape[1]).to(inputs_embeds.device)
            else:
                action_len = 1 + 578 * self.fast_image_num + 1 + self.action_chunk
                latent_indexes=torch.arange(0, inputs_embeds.shape[1] - action_len).to(inputs_embeds.device)
                action_indexes=torch.arange(inputs_embeds.shape[1] - action_len, inputs_embeds.shape[1]).to(inputs_embeds.device)
        else:
            inputs_embeds = torch.cat([timesteps, noisy_actions], dim=1)
            past_key_values = tuple(
                (k[:, :, :-(timesteps.shape[1]+noisy_actions.shape[1]), :], v[:, :, :-(timesteps.shape[1]+noisy_actions.shape[1]), :]) for k, v in past_key_values
            )
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            latent_indexes=torch.arange(0, 0).to(inputs_embeds.device)
            action_indexes=torch.arange(0, inputs_embeds.shape[1]).to(inputs_embeds.device)

        outputs = self.language_model.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            latent_indexes=latent_indexes,
            action_indexes=action_indexes,
            use_latent=self.use_latent,
            return_dict=True,
            use_cache=True
        )
        hidden_states = outputs.last_hidden_state
        v_t = self.final_layer(hidden_states)[:, -noisy_actions.shape[1]:, :]
        return v_t, outputs.past_key_values

    def denoise_step_action(self, inputs_embeds, past_key_values, x_t, timestep):
        noisy_actions = self.x_embedder(x_t.to(torch.bfloat16))
        timesteps = self.t_embedder(timestep).unsqueeze(1)

        if past_key_values is None:
            inputs_embeds = torch.cat([
                inputs_embeds,
                timesteps,
                noisy_actions,
            ], dim=1)
            latent_indexes=torch.arange(0, 0).to(inputs_embeds.device)
            action_indexes=torch.arange(0, inputs_embeds.shape[1]).to(inputs_embeds.device)
        else:
            inputs_embeds = torch.cat([timesteps, noisy_actions], dim=1)
            past_key_values = tuple(
                (k[:, :, :-(timesteps.shape[1]+noisy_actions.shape[1]), :], v[:, :, :-(timesteps.shape[1]+noisy_actions.shape[1]), :]) for k, v in past_key_values
            )
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            latent_indexes=torch.arange(0, 0).to(inputs_embeds.device)
            action_indexes=torch.arange(0, inputs_embeds.shape[1]).to(inputs_embeds.device)

        outputs = self.language_model.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            latent_indexes=latent_indexes,
            action_indexes=action_indexes,
            use_latent=self.use_latent,
            return_dict=True,
            use_cache=True
        )
        hidden_states = outputs.last_hidden_state
        v_t = self.final_layer(hidden_states)[:, -noisy_actions.shape[1]:, :]
        return v_t, outputs.past_key_values

    def forward_flow(self, inputs_embeds, noise, num_steps=10):
        noisy_actions = self.x_embedder(noise.to(torch.bfloat16))
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.bfloat16, device=noisy_actions.device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.bfloat16, device=noisy_actions.device)
        past_key_values = None

        while time >= -dt / 2:
            expanded_time = time.expand(noisy_actions.shape[0])
            v_t, past_key_values = self.denoise_step(
                inputs_embeds,
                past_key_values,
                x_t,
                expanded_time,
            )
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t
    
    def initialize_weights(self):
        if self.flow:
            print("init flow components!!!")
            nn.init.normal_(self.x_embedder.mlp.fc1.weight, std=0.02)
            nn.init.normal_(self.x_embedder.mlp.fc2.weight, std=0.02)
            nn.init.constant_(self.x_embedder.mlp.fc1.bias, 0)
            nn.init.constant_(self.x_embedder.mlp.fc2.bias, 0)
            if self.robot_state:
                nn.init.normal_(self.state_embedder.mlp.fc1.weight, std=0.02)
                nn.init.normal_(self.state_embedder.mlp.fc2.weight, std=0.02)
                nn.init.constant_(self.state_embedder.mlp.fc1.bias, 0)
                nn.init.constant_(self.state_embedder.mlp.fc2.bias, 0)

            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            nn.init.constant_(self.t_embedder.mlp[0].bias, 0)
            nn.init.constant_(self.t_embedder.mlp[2].bias, 0)

            nn.init.normal_(self.final_layer.mlp.fc1.weight, std=0.02)
            nn.init.constant_(self.final_layer.mlp.fc1.bias, 0)
            nn.init.constant_(self.final_layer.mlp.fc2.weight, 0)
            nn.init.constant_(self.final_layer.mlp.fc2.bias, 0)

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        # [b x n, T2, D]
        images_embeds = self.aligner(self.vision_model(images))

        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        # [b, n, T2] -> [b, n x T2]
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # replace with the image embeddings
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds
    
    def prepare_inputs_embeds_gen(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.LongTensor, # should be image_ids
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """
        assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        
        # use vqgan as image encoder
        _, _, info = self.gen_vision_model.encode(images)
        image_ids = info[2].reshape(bs*n, -1)
        images_embeds = self.gen_aligner(self.gen_embed(image_ids))

        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        # [b, n, T2] -> [b, n x T2]
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # replace with the image embeddings
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds, image_ids

    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
        return self.gen_aligner(self.gen_embed(image_ids))


AutoConfig.register("vision", VisionConfig)
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_vision", GenVisionConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)

