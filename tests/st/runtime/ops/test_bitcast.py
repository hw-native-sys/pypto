# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile.bitcast — zero-copy element-type reinterpretation.

``tile.bitcast`` emits no instruction: InitMemRef aliases the result onto the
source MemRef, so the result's ``pto.alloc_tile`` declares the target dtype at
the source address. These tests are what actually prove that alias is real on
hardware — a value conversion (``tile.cast``) or a lost alias would both give
visibly wrong numbers.

Coverage:
  * one-way reinterpretation (FP32 -> INT32, BF16 -> INT16), compared against
    ``torch.Tensor.view(dtype)`` — the exact bit-level reference;
  * a round trip through an integer op (BF16 -> INT16 -> ``not_`` -> BF16),
    which only produces the reference bits if both directions alias correctly.

``not_`` is used as the integer op because TNOT is the one integer bitwise tile
op that currently assembles and runs on a2a3 (see ``test_bitwise.py``).

Scope is a2a3 only (``@pytest.mark.platforms("a2a3")``).
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

_PL_DT = {
    DataType.FP32: pl.FP32,
    DataType.INT32: pl.INT32,
    DataType.BF16: pl.BF16,
    DataType.INT16: pl.INT16,
}

# (label, src dtype, dst dtype) — equal-width pairings accepted by strict bitcast.
_REINTERPRET_CFGS = [
    ("fp32_to_int32", DataType.FP32, DataType.INT32),
    ("bf16_to_int16", DataType.BF16, DataType.INT16),
]


def _spread(m: int, n: int) -> torch.Tensor:
    """Signed non-trivial float spread, so sign/exponent/mantissa bits all vary."""
    return (torch.arange(m * n, dtype=torch.float32).reshape(m, n) - (m * n) / 2) * 0.375


class BitcastReinterpretTestCase(PTOTestCase):
    """Load `src` dtype, bitcast to `dst` dtype, store — a pure bit reinterpretation."""

    __test__ = False

    def __init__(self, *, m=16, n=64, src_dtype=DataType.FP32, dst_dtype=DataType.INT32, config=None):
        super().__init__(config)
        self._m, self._n = m, n
        self._src_dtype, self._dst_dtype = src_dtype, dst_dtype

    def get_name(self) -> str:
        return f"tile_bitcast_{self._src_dtype.value}_to_{self._dst_dtype.value}_{self._m}x{self._n}"

    def define_tensors(self) -> list[TensorSpec]:
        m, n, src_torch = self._m, self._n, self._src_dtype.torch_dtype
        return [
            TensorSpec("a", [m, n], self._src_dtype, init_value=lambda: _spread(m, n).to(src_torch)),
            TensorSpec("out", [m, n], self._dst_dtype, is_output=True, init_value=torch.zeros),
        ]

    def get_program(self) -> Any:
        m, n = self._m, self._n
        src_dt, dst_dt = _PL_DT[self._src_dtype], _PL_DT[self._dst_dtype]

        @pl.program
        class BitcastReinterpretProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[m, n], src_dt], out: pl.InOut[pl.Tensor[[m, n], dst_dt]]
            ) -> pl.Tensor[[m, n], dst_dt]:
                a_tile = pl.load(a, [0, 0], [m, n])
                # `dtype=` kwarg form: the DSL parser resolves a dtype keyword
                # through TypeResolver, which a positional closure variable
                # cannot go through (same reason tile.cast uses target_type=).
                out = pl.store(pl.tile.bitcast(a_tile, dtype=dst_dt), [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[m, n], src_dt], out: pl.InOut[pl.Tensor[[m, n], dst_dt]]
            ) -> pl.Tensor[[m, n], dst_dt]:
                out = self.kernel(a, out)
                return out

        return BitcastReinterpretProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        # torch's own bit-level view is the reference: bitcast must reproduce it
        # exactly, which a value conversion never would.
        tensors["out"][:] = tensors["a"].view(self._dst_dtype.torch_dtype)


class BitcastRoundTripTestCase(PTOTestCase):
    """BF16 -> INT16 -> bitwise not_ -> BF16: exercises both bitcast directions."""

    __test__ = False

    def __init__(self, *, m=16, n=64, config=None):
        super().__init__(config)
        self._m, self._n = m, n

    def get_name(self) -> str:
        return f"tile_bitcast_roundtrip_bf16_{self._m}x{self._n}"

    def define_tensors(self) -> list[TensorSpec]:
        m, n = self._m, self._n
        return [
            TensorSpec("a", [m, n], DataType.BF16, init_value=lambda: _spread(m, n).to(torch.bfloat16)),
            TensorSpec("out", [m, n], DataType.BF16, is_output=True, init_value=torch.zeros),
        ]

    def get_program(self) -> Any:
        m, n = self._m, self._n

        @pl.program
        class BitcastRoundTripProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, a: pl.Tensor[[m, n], pl.BF16], out: pl.InOut[pl.Tensor[[m, n], pl.BF16]]
            ) -> pl.Tensor[[m, n], pl.BF16]:
                a_tile = pl.load(a, [0, 0], [m, n])
                bits = pl.tile.bitcast(a_tile, pl.INT16)
                flipped = pl.tile.not_(bits)
                out = pl.store(pl.tile.bitcast(flipped, pl.BF16), [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[m, n], pl.BF16], out: pl.InOut[pl.Tensor[[m, n], pl.BF16]]
            ) -> pl.Tensor[[m, n], pl.BF16]:
                out = self.kernel(a, out)
                return out

        return BitcastRoundTripProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        flipped = torch.bitwise_not(tensors["a"].view(torch.int16))
        tensors["out"][:] = flipped.view(torch.bfloat16)


class TestBitcast:
    """tile.bitcast on a2a3: one-way reinterpretation and a round trip."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize(
        "label,src_dtype,dst_dtype", _REINTERPRET_CFGS, ids=[c[0] for c in _REINTERPRET_CFGS]
    )
    def test_tile_bitcast_reinterpret(self, test_runner, label, src_dtype, dst_dtype):
        result = test_runner.run(BitcastReinterpretTestCase(src_dtype=src_dtype, dst_dtype=dst_dtype))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("m,n", [(16, 64), (32, 128)])
    def test_tile_bitcast_shapes(self, test_runner, m, n):
        result = test_runner.run(BitcastReinterpretTestCase(m=m, n=n))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_bitcast_roundtrip(self, test_runner):
        result = test_runner.run(BitcastRoundTripTestCase())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
