#!/usr/bin/env python3
"""
Inspect Cosmos DiT model structure to extract key configuration parameters.
"""
import torch
import sys

model_path = "/mnt/wfm/ckpt/ckpt/pretrained/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt"

print(f"Loading model from: {model_path}")
print("=" * 80)

try:
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    print(f"Checkpoint type: {type(checkpoint)}")
    print(f"Checkpoint keys: {list(checkpoint.keys())[:20]}")
    print("=" * 80)

    # Try to extract model structure
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model_state = checkpoint['model']
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
        else:
            model_state = checkpoint

        # Get all keys
        all_keys = list(model_state.keys())
        print(f"\nTotal parameters: {len(all_keys)}")
        print(f"\nFirst 30 keys:")
        for i, key in enumerate(all_keys[:30]):
