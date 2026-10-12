"""Torch-only extraction of N0-VTLA's OpenPI/Big Vision Gemma dimensions.

Copyright 2024 Big Vision Authors. Licensed under Apache-2.0 (see Apache-2.0.txt).
"""
from types import SimpleNamespace


def get_config(variant):
    specs = {"gemma_2b": (2048, 18, 16384, 8, 1, 256),
             "gemma_300m": (1024, 18, 4096, 8, 1, 256),
             "dummy": (64, 2, 128, 8, 1, 16)}
    if variant not in specs:
        raise ValueError(f"Unsupported N0 Gemma variant: {variant}")
    return SimpleNamespace(**dict(zip(("width", "depth", "mlp_dim", "num_heads", "num_kv_heads", "head_dim"), specs[variant])))
