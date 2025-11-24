from typing import Optional

import torch

from ..utils import logging


logger = logging.get_logger(__name__)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def build_causal_mask(q_length, kv_length, kv_offset=0, device=None):
    query_positions = torch.arange(q_length, device=device).view(1, 1, q_length, 1) + kv_offset
    key_positions = torch.arange(kv_length, device=device).view(1, 1, 1, kv_length)
    causal_mask = query_positions >= key_positions
    return causal_mask

def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask", None) is not None:
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

# # # ---------ch changed----------- # 
#     assert "use_latent" in kwargs
#     if kwargs['use_latent']:
#         if module.training:
#             q_length = query.shape[-2]
#             kv_length = key.shape[-2]
#             if attention_mask is None:
#                 attention_mask = build_causal_mask(q_length, kv_length, kv_offset=0, device=query.device)

#             assert 'latent_indexes' in kwargs
#             latent_indexes = kwargs['latent_indexes']

#             if len(latent_indexes) > 0:
#                 last_latent_idx = latent_indexes[-1].item() if isinstance(latent_indexes, torch.Tensor) else latent_indexes[-1]
#                 col_start_idx = max(0, last_latent_idx - 4 -1)
#                 row_start_idx = last_latent_idx + 1
#                 if row_start_idx < q_length and col_start_idx < kv_length:
#                     attention_mask[:, :, row_start_idx:, col_start_idx:last_latent_idx] = False
        
#         else:
#             q_length = query.shape[-2]
#             kv_length = key.shape[-2]
#             if attention_mask is None:
#                 attention_mask = build_causal_mask(q_length, kv_length, kv_offset=kv_length-q_length, device=query.device)
            
#             assert 'latent_indexes' in kwargs
#             latent_indexes = kwargs['latent_indexes']
            
#             if len(latent_indexes)==0:
#                 assert 'action_condition_len' in kwargs
#                 action_condition_len = kwargs['action_condition_len']
#                 attention_mask[:, :, :, action_condition_len:action_condition_len+4+1] = False

# # # ---------ch changed----------- # 

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
