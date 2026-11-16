# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""Read only the current rank's Qwen3-14B safetensors shards."""

import importlib
import json
from pathlib import Path
from typing import Any

import torch

EXPECTED_CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 40,
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "hidden_act": "silu",
}


def validate_config(config: dict[str, Any]) -> None:
    """Reject architectures unsupported by the fixed-shape kernels."""
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"Qwen3 TP requires {key}={expected!r}, got {config.get(key)!r}")
    for key in ("rope_scaling", "sliding_window"):
        if config.get(key) is not None:
            raise ValueError(f"Qwen3 TP requires {key}=None, got {config[key]!r}")
    if config.get("use_sliding_window", False):
        raise ValueError("Qwen3 TP requires use_sliding_window=False")


def shard_range(size: int, rank: int, world_size: int = 2) -> slice:
    """Return the contiguous partition owned by a TP rank."""
    if world_size not in (2, 4, 8) or not 0 <= rank < world_size or size <= 0 or size % world_size:
        raise ValueError(f"Invalid partition: rank={rank}, size={size}, world_size={world_size}")
    width = size // world_size
    return slice(rank * width, (rank + 1) * width)


class WeightLoader:
    """Slice HF [output, input] matrices before transfer, then transpose for PyPTO."""

    def __init__(self, model_dir: str | Path, rank: int, device: str, world_size: int = 2):
        self.root = Path(model_dir)
        self.rank = rank
        self.world_size = world_size
        shard_range(world_size, rank, world_size)
        self.device = device
        self.config = json.loads((self.root / "config.json").read_text())
        validate_config(self.config)
        self.weight_map = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]

    def read(self, name: str, axis: int | None = None) -> torch.Tensor:
        safe_open = importlib.import_module("safetensors").safe_open

        with safe_open(self.root / self.weight_map[name], framework="pt", device="cpu") as handle:
            view = handle.get_slice(name)
            if axis is None:
                value = view[:]
            else:
                selection = [slice(None)] * len(view.get_shape())
                selection[axis] = shard_range(view.get_shape()[axis], self.rank, self.world_size)
                value = view[tuple(selection)]
            return value.contiguous()

    def matrix(self, name: str, axis: int) -> torch.Tensor:
        return self.read(name, axis).T.contiguous().to(device=self.device, dtype=torch.bfloat16)

    def norm(self, name: str) -> torch.Tensor:
        return self.read(name).reshape(1, -1).to(device=self.device, dtype=torch.float32)

    def layer(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < 40:
            raise ValueError(f"Layer index must be in [0, 40), got {index}")
        prefix = f"model.layers.{index}."
        columns = {
            "wq": "self_attn.q_proj",
            "wk": "self_attn.k_proj",
            "wv": "self_attn.v_proj",
            "gate": "mlp.gate_proj",
            "up": "mlp.up_proj",
        }
        rows = {"wo": "self_attn.o_proj", "down": "mlp.down_proj"}
        result = {key: self.matrix(prefix + path + ".weight", 0) for key, path in columns.items()}
        result.update({key: self.matrix(prefix + path + ".weight", 1) for key, path in rows.items()})
        for key, path in {
            "input_norm": "input_layernorm",
            "post_norm": "post_attention_layernorm",
            "q_norm": "self_attn.q_norm",
            "k_norm": "self_attn.k_norm",
        }.items():
            result[key] = self.norm(prefix + path + ".weight")
        return result

    def vocabulary(self, name: str) -> torch.Tensor:
        value = self.read(name)
        padded = torch.zeros((152064, 5120), dtype=torch.bfloat16)
        padded[:151936].copy_(value)
        return padded.to(self.device)
