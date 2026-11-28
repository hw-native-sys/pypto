# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""Check TP2/TP4/TP8 matrix orientation against an unsplit linear layer."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

save_file = pytest.importorskip("safetensors.torch").save_file

SOURCE = Path(__file__).resolve().parents[4] / "examples/models/qwen3_tp/weights.py"
spec = importlib.util.spec_from_file_location("qwen3_tp_weights", SOURCE)
assert spec is not None and spec.loader is not None
weights = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weights)


@pytest.fixture
def model(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(weights.EXPECTED_CONFIG))
    torch.manual_seed(17)
    tensors = {"column": torch.randn(24, 8).bfloat16(), "row": torch.randn(8, 24).bfloat16()}
    save_file(tensors, tmp_path / "weights.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "weights.safetensors")})
    )
    return tmp_path, tensors


@pytest.mark.parametrize("world", [2, 4, 8])
def test_column_partition_concatenates_to_full_projection(model, world):
    root, tensors = model
    x = torch.randn(3, 8)
    outputs = [
        x @ weights.WeightLoader(root, rank, "cpu", world).matrix("column", 0).float()
        for rank in range(world)
    ]
    torch.testing.assert_close(torch.cat(outputs, dim=1), x @ tensors["column"].float().T)


@pytest.mark.parametrize("world", [2, 4, 8])
def test_row_partition_sums_to_full_projection(model, world):
    root, tensors = model
    x = torch.randn(3, 24)
    outputs = [
        x[:, weights.shard_range(24, rank, world)]
        @ weights.WeightLoader(root, rank, "cpu", world).matrix("row", 1).float()
        for rank in range(world)
    ]
    torch.testing.assert_close(torch.stack(outputs).sum(0), x @ tensors["row"].float().T)


@pytest.mark.parametrize(
    "key,value",
    [
        ("hidden_size", 4096),
        ("num_key_value_heads", 4),
        ("rope_scaling", {"type": "linear"}),
        ("attention_bias", True),
    ],
)
def test_reject_incompatible_architecture(key, value):
    with pytest.raises(ValueError, match=key):
        weights.validate_config({**weights.EXPECTED_CONFIG, key: value})


@pytest.mark.parametrize("size,rank", [(3, 0), (4, 2), (0, 0)])
def test_reject_invalid_partition(size, rank):
    with pytest.raises(ValueError):
        weights.shard_range(size, rank)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
