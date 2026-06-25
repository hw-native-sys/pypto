# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for ``tile.gatherb`` -> ``pto.tgatherb`` (byte-offset gather).

``gatherb`` reads ``dst[i, j] = *(src_base + offset[i, j])`` where ``offset`` is a
per-element **byte** offset into the source tile (UINT32). The destination is a
fresh tile shaped like ``offset`` with the source dtype.

The source tile is ``[16, 16]`` so each row is a whole number of 32-byte blocks
for every dtype tested (FP16=32B, FP32/INT32=64B) — no row padding — so the
flat byte offset of element ``k`` is exactly ``k * elem_size``. The golden is
derived straight from the offset tensor: ``out_flat[k] = src_flat[offset[k] /
elem_size]``, which is independent of the chosen permutation.

Two permutations exercise non-trivial addressing:
- ``reverse`` - gather the source in reverse flat order.
- ``roll7``   - gather rotated by 7 elements (wraps across rows).

Each dtype needs its own ``@pl.program`` (a program hardcodes its dtype at parse
time), and the op call must be a literal ``pl.tile.gatherb`` (the DSL parser
rejects aliases), so there is one program factory per dtype with a distinct
class name.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec

M = 16
N = 16

# dtype label -> (pl dtype, harness DataType, torch dtype, element byte size)
_DTYPES = {
    "fp16": (pl.FP16, DataType.FP16, torch.float16, 2),
    "fp32": (pl.FP32, DataType.FP32, torch.float32, 4),
    "int32": (pl.INT32, DataType.INT32, torch.int32, 4),
}


def _prog_fp16():
    @pl.program
    class GatherbFP16:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[M, N], pl.FP16],
            offset: pl.Tensor[[M, N], pl.UINT32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP16]],
        ) -> pl.Tensor[[M, N], pl.FP16]:
            t_src: pl.Tile[[M, N], pl.FP16] = pl.load(src, [0, 0], [M, N])
            t_off: pl.Tile[[M, N], pl.UINT32] = pl.load(offset, [0, 0], [M, N])
            out_tile: pl.Tile[[M, N], pl.FP16] = pl.tile.gatherb(t_src, t_off)
            out = pl.store(out_tile, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src: pl.Tensor[[M, N], pl.FP16],
            offset: pl.Tensor[[M, N], pl.UINT32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP16]],
        ) -> pl.Tensor[[M, N], pl.FP16]:
            out = self.kernel(src, offset, out)
            return out

    return GatherbFP16


def _prog_fp32():
    @pl.program
    class GatherbFP32:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[M, N], pl.FP32],
            offset: pl.Tensor[[M, N], pl.UINT32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            t_src: pl.Tile[[M, N], pl.FP32] = pl.load(src, [0, 0], [M, N])
            t_off: pl.Tile[[M, N], pl.UINT32] = pl.load(offset, [0, 0], [M, N])
            out_tile: pl.Tile[[M, N], pl.FP32] = pl.tile.gatherb(t_src, t_off)
            out = pl.store(out_tile, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src: pl.Tensor[[M, N], pl.FP32],
            offset: pl.Tensor[[M, N], pl.UINT32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            out = self.kernel(src, offset, out)
            return out

    return GatherbFP32


def _prog_int32():
    @pl.program
    class GatherbInt32:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[M, N], pl.INT32],
            offset: pl.Tensor[[M, N], pl.UINT32],
            out: pl.Out[pl.Tensor[[M, N], pl.INT32]],
        ) -> pl.Tensor[[M, N], pl.INT32]:
            t_src: pl.Tile[[M, N], pl.INT32] = pl.load(src, [0, 0], [M, N])
            t_off: pl.Tile[[M, N], pl.UINT32] = pl.load(offset, [0, 0], [M, N])
            out_tile: pl.Tile[[M, N], pl.INT32] = pl.tile.gatherb(t_src, t_off)
            out = pl.store(out_tile, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            src: pl.Tensor[[M, N], pl.INT32],
            offset: pl.Tensor[[M, N], pl.UINT32],
            out: pl.Out[pl.Tensor[[M, N], pl.INT32]],
        ) -> pl.Tensor[[M, N], pl.INT32]:
            out = self.kernel(src, offset, out)
            return out

    return GatherbInt32


_PROG_FACTORY = {"fp16": _prog_fp16, "fp32": _prog_fp32, "int32": _prog_int32}


def _flat_index(pattern: str) -> torch.Tensor:
    """Flat source element index gathered into each destination position."""
    k = torch.arange(M * N, dtype=torch.int64)
    if pattern == "reverse":
        return (M * N - 1) - k
    if pattern == "roll7":
        return (k + 7) % (M * N)
    raise ValueError(f"unknown pattern {pattern!r}")


def _src(torch_dtype: torch.dtype) -> torch.Tensor:
    """Distinct value per element so a wrong gather is detectable."""
    return torch.arange(M * N).reshape(M, N).to(torch_dtype).contiguous()


def _offset(pattern: str, elem_size: int) -> torch.Tensor:
    """Per-element byte offset (UINT32, carried as int32) for the gather pattern."""
    return (_flat_index(pattern) * elem_size).to(torch.int32).reshape(M, N).contiguous()


class GatherbTestCase(PTOTestCase):
    """tile.gatherb over a fixed flat permutation expressed as byte offsets."""

    __test__ = False

    def __init__(self, dtype_key: str, pattern: str, *, platform=None, config=None):
        super().__init__(config, platform=platform)
        self._dtype_key = dtype_key
        self._pattern = pattern
        _, _, _, self._elem_size = _DTYPES[dtype_key]

    def get_name(self) -> str:
        return f"gatherb_{self._dtype_key}_{self._pattern}"

    def define_tensors(self) -> list[TensorSpec]:
        _, hdtype, tdtype, elem = _DTYPES[self._dtype_key]
        return [
            TensorSpec("src", [M, N], hdtype, init_value=_src(tdtype)),
            TensorSpec("offset", [M, N], DataType.UINT32, init_value=_offset(self._pattern, elem)),
            TensorSpec("out", [M, N], hdtype, is_output=True),
        ]

    def get_program(self) -> Any:
        return _PROG_FACTORY[self._dtype_key]()

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        src_flat = tensors["src"].reshape(-1)
        idx = (tensors["offset"].reshape(-1).to(torch.int64)) // self._elem_size
        tensors["out"][:] = src_flat[idx].reshape(M, N)


_DTYPE_KEYS = ["fp16", "fp32", "int32"]
_PATTERNS = ["reverse", "roll7"]


class TestGatherb:
    """Test tile.gatherb across supported platforms, dtypes, and patterns."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("dtype_key", _DTYPE_KEYS)
    @pytest.mark.parametrize("pattern", _PATTERNS)
    def test_gatherb(self, test_runner, platform, dtype_key, pattern):
        result = test_runner.run(GatherbTestCase(dtype_key, pattern, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
