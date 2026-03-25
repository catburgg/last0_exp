"""
Shim for ``accelerate`` + DeepSpeed: older import path uses ``transformers.deepspeed``.
This fork keeps the implementation in ``transformers.integrations.deepspeed`` (like upstream).
"""

from .integrations.deepspeed import HfDeepSpeedConfig, unset_hf_deepspeed_config

__all__ = ["HfDeepSpeedConfig", "unset_hf_deepspeed_config"]
