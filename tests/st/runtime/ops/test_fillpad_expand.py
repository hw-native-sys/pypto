# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test fillpad_expand operation (TFILLPAD_EXPAND).

Unlike ``fillpad`` (which keeps the same physical shape and only fills the
valid-region expansion), ``fillpad_expand`` produces a *larger* destination
tile: the source's valid region is copied into the top-left of the destination
and every other destination element is filled with ``pad_value``.

Golden (matches the pto-isa CPU/A2A3 reference): for every (i, j) in the
destination shape ``dst[i, j] = src[i, j]`` when ``i < valid_src_row`` and
``j < valid_src_col``, otherwise ``dst[i, j] = pad_value``.

Coverage:
- Row-only expansion, column-only expansion, and both-dimension expansion.
- Narrow ``valid_shape`` cases (the source's valid region is smaller than its
  physical shape) — exercises the valid-region copy path, including a
  non-32B-aligned valid column (50) and a ``validRow < 16`` tail path.
- Pad modes zero / max / min.
- Dtypes FP32, FP16, INT32.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

# =============================================================================
# Programs — one explicit @pl.program per scenario (distinct names, literal
# shapes/dtype, and a literal pl.tile.fillpad_expand call).
# =============================================================================


@pl.program
class FillpadExpandRowFP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        src: pl.Tile[[48, 64], pl.FP32] = pl.load(input_tensor, [0, 0], [48, 64])
        dst: pl.Tile[[64, 64], pl.FP32] = pl.tile.fillpad_expand(src, [64, 64], pad_value=pl.PadValue.zero)
        return pl.store(dst, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        return self.kernel(input_tensor, output)


@pl.program
class FillpadExpandColFP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[64, 48], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        src: pl.Tile[[64, 48], pl.FP32] = pl.load(input_tensor, [0, 0], [64, 48])
        dst: pl.Tile[[64, 64], pl.FP32] = pl.tile.fillpad_expand(src, [64, 64], pad_value=pl.PadValue.zero)
        return pl.store(dst, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[64, 48], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        return self.kernel(input_tensor, output)


@pl.program
class FillpadExpandBothMaxFP32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[48, 48], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        src: pl.Tile[[48, 48], pl.FP32] = pl.load(input_tensor, [0, 0], [48, 48])
        dst: pl.Tile[[64, 64], pl.FP32] = pl.tile.fillpad_expand(src, [64, 64], pad_value=pl.PadValue.max)
        return pl.store(dst, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[48, 48], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        return self.kernel(input_tensor, output)


@pl.program
class FillpadExpandNarrowValidFP32:
    """Source physical [48, 64] but only [40, 50] is valid (non-aligned col)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        src: pl.Tile[[48, 64], pl.FP32] = pl.load(input_tensor, [0, 0], [48, 64], valid_shapes=[40, 50])
        dst: pl.Tile[[64, 64], pl.FP32] = pl.tile.fillpad_expand(src, [64, 64], pad_value=pl.PadValue.zero)
        return pl.store(dst, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        return self.kernel(input_tensor, output)


@pl.program
class FillpadExpandNarrowValidBothMinFP32:
    """Narrow valid region AND expand in both dimensions, min pad."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
    ) -> pl.Tensor[[64, 128], pl.FP32]:
        src: pl.Tile[[64, 64], pl.FP32] = pl.load(input_tensor, [0, 0], [64, 64], valid_shapes=[40, 50])
        dst: pl.Tile[[64, 128], pl.FP32] = pl.tile.fillpad_expand(src, [64, 128], pad_value=pl.PadValue.min)
        return pl.store(dst, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
    ) -> pl.Tensor[[64, 128], pl.FP32]:
        return self.kernel(input_tensor, output)


@pl.program
class FillpadExpandTailValidFP32:
    """Small source (validRow < 16) to exercise the tail copy path."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[8, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
    ) -> pl.Tensor[[16, 32], pl.FP32]:
        src: pl.Tile[[8, 16], pl.FP32] = pl.load(input_tensor, [0, 0], [8, 16], valid_shapes=[8, 10])
        dst: pl.Tile[[16, 32], pl.FP32] = pl.tile.fillpad_expand(src, [16, 32], pad_value=pl.PadValue.zero)
        return pl.store(dst, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[8, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
    ) -> pl.Tensor[[16, 32], pl.FP32]:
        return self.kernel(input_tensor, output)


@pl.program
class FillpadExpandRowFP16:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP16],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP16]],
    ) -> pl.Tensor[[64, 64], pl.FP16]:
        src: pl.Tile[[48, 64], pl.FP16] = pl.load(input_tensor, [0, 0], [48, 64])
        dst: pl.Tile[[64, 64], pl.FP16] = pl.tile.fillpad_expand(src, [64, 64], pad_value=pl.PadValue.zero)
        return pl.store(dst, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP16],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP16]],
    ) -> pl.Tensor[[64, 64], pl.FP16]:
        return self.kernel(input_tensor, output)


@pl.program
class FillpadExpandNarrowValidFP16:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP16],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP16]],
    ) -> pl.Tensor[[64, 64], pl.FP16]:
        src: pl.Tile[[48, 64], pl.FP16] = pl.load(input_tensor, [0, 0], [48, 64], valid_shapes=[40, 50])
        dst: pl.Tile[[64, 64], pl.FP16] = pl.tile.fillpad_expand(src, [64, 64], pad_value=pl.PadValue.zero)
        return pl.store(dst, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.FP16],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP16]],
    ) -> pl.Tensor[[64, 64], pl.FP16]:
        return self.kernel(input_tensor, output)


@pl.program
class FillpadExpandRowINT32:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.INT32],
        output: pl.Out[pl.Tensor[[64, 64], pl.INT32]],
    ) -> pl.Tensor[[64, 64], pl.INT32]:
        src: pl.Tile[[48, 64], pl.INT32] = pl.load(input_tensor, [0, 0], [48, 64], valid_shapes=[40, 50])
        dst: pl.Tile[[64, 64], pl.INT32] = pl.tile.fillpad_expand(src, [64, 64], pad_value=pl.PadValue.zero)
        return pl.store(dst, [0, 0], output)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        input_tensor: pl.Tensor[[48, 64], pl.INT32],
        output: pl.Out[pl.Tensor[[64, 64], pl.INT32]],
    ) -> pl.Tensor[[64, 64], pl.INT32]:
        return self.kernel(input_tensor, output)


@pl.program
class FillpadExpandTensorFP32:
    """Tensor-level path: pl.fillpad_expand on a whole Tensor, lowered by
    ConvertTensorToTileOps to load + tile.fillpad_expand."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[48, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        result: pl.Tensor[[64, 64], pl.FP32] = pl.fillpad_expand(a, [64, 64], pad_value=pl.PadValue.zero)
        return pl.assemble(output, result, [0, 0])

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[48, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        return self.kernel(a, output)


# =============================================================================
# Test cases — generalized golden shared by every scenario.
# =============================================================================

_TORCH_DTYPE = {
    DataType.FP32: torch.float32,
    DataType.FP16: torch.float16,
    DataType.INT32: torch.int32,
}


def _pad_fill(pad: str, dtype: DataType) -> float:
    """Return the scalar that the destination padding region must equal."""
    if pad == "zero":
        return 0
    torch_dtype = _TORCH_DTYPE[dtype]
    if torch_dtype.is_floating_point:
        return float("inf") if pad == "max" else float("-inf")
    info = torch.iinfo(torch_dtype)
    return info.max if pad == "max" else info.min


def _make_init(src_shape, dtype):
    """No-arg init factory producing varied (non-zero) source data."""
    torch_dtype = _TORCH_DTYPE[dtype]
    if torch_dtype.is_floating_point:
        return lambda: torch.randn(src_shape)
    # Distinct, non-zero integers so an all-zero device bug cannot pass.
    return lambda: torch.arange(1, src_shape[0] * src_shape[1] + 1, dtype=torch.int32).reshape(src_shape)


class _FillpadExpandCase(PTOTestCase):
    """Base test case; subclasses set the scenario attributes below."""

    __test__ = False

    program: Any = None
    case_name: str = ""
    src_shape: list[int] = []
    valid_shape: list[int] = []
    dst_shape: list[int] = []
    dtype: DataType = DataType.FP32
    pad: str = "zero"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.dtype == DataType.FP16:
            self.config.rtol = 2e-2
            self.config.atol = 2e-2

    def get_name(self) -> str:
        return self.case_name

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "input_tensor", self.src_shape, self.dtype, init_value=_make_init(self.src_shape, self.dtype)
            ),
            TensorSpec("output", self.dst_shape, self.dtype, is_output=True),
        ]

    def get_program(self) -> Any:
        return self.program

    def compute_expected(self, tensors, params=None):
        vr, vc = self.valid_shape
        torch_dtype = _TORCH_DTYPE[self.dtype]
        expected = torch.full(self.dst_shape, _pad_fill(self.pad, self.dtype), dtype=torch_dtype)
        src = tensors["input_tensor"]
        expected[:vr, :vc] = src[:vr, :vc]
        tensors["output"][:] = expected


class FillpadExpandRowFP32Case(_FillpadExpandCase):
    program = FillpadExpandRowFP32
    case_name = "fillpad_expand_row_fp32"
    src_shape = [48, 64]
    valid_shape = [48, 64]
    dst_shape = [64, 64]
    dtype = DataType.FP32
    pad = "zero"


class FillpadExpandColFP32Case(_FillpadExpandCase):
    program = FillpadExpandColFP32
    case_name = "fillpad_expand_col_fp32"
    src_shape = [64, 48]
    valid_shape = [64, 48]
    dst_shape = [64, 64]
    dtype = DataType.FP32
    pad = "zero"


class FillpadExpandBothMaxFP32Case(_FillpadExpandCase):
    program = FillpadExpandBothMaxFP32
    case_name = "fillpad_expand_both_max_fp32"
    src_shape = [48, 48]
    valid_shape = [48, 48]
    dst_shape = [64, 64]
    dtype = DataType.FP32
    pad = "max"


class FillpadExpandNarrowValidFP32Case(_FillpadExpandCase):
    program = FillpadExpandNarrowValidFP32
    case_name = "fillpad_expand_narrow_valid_fp32"
    src_shape = [48, 64]
    valid_shape = [40, 50]
    dst_shape = [64, 64]
    dtype = DataType.FP32
    pad = "zero"


class FillpadExpandNarrowValidBothMinFP32Case(_FillpadExpandCase):
    program = FillpadExpandNarrowValidBothMinFP32
    case_name = "fillpad_expand_narrow_valid_both_min_fp32"
    src_shape = [64, 64]
    valid_shape = [40, 50]
    dst_shape = [64, 128]
    dtype = DataType.FP32
    pad = "min"


class FillpadExpandTailValidFP32Case(_FillpadExpandCase):
    program = FillpadExpandTailValidFP32
    case_name = "fillpad_expand_tail_valid_fp32"
    src_shape = [8, 16]
    valid_shape = [8, 10]
    dst_shape = [16, 32]
    dtype = DataType.FP32
    pad = "zero"


class FillpadExpandRowFP16Case(_FillpadExpandCase):
    program = FillpadExpandRowFP16
    case_name = "fillpad_expand_row_fp16"
    src_shape = [48, 64]
    valid_shape = [48, 64]
    dst_shape = [64, 64]
    dtype = DataType.FP16
    pad = "zero"


class FillpadExpandNarrowValidFP16Case(_FillpadExpandCase):
    program = FillpadExpandNarrowValidFP16
    case_name = "fillpad_expand_narrow_valid_fp16"
    src_shape = [48, 64]
    valid_shape = [40, 50]
    dst_shape = [64, 64]
    dtype = DataType.FP16
    pad = "zero"


class FillpadExpandRowINT32Case(_FillpadExpandCase):
    program = FillpadExpandRowINT32
    case_name = "fillpad_expand_row_int32"
    src_shape = [48, 64]
    valid_shape = [40, 50]
    dst_shape = [64, 64]
    dtype = DataType.INT32
    pad = "zero"


class FillpadExpandTensorFP32Case(_FillpadExpandCase):
    """Tensor-level entry point (pl.fillpad_expand on a Tensor)."""

    program = FillpadExpandTensorFP32
    case_name = "fillpad_expand_tensor_fp32"
    src_shape = [48, 64]
    valid_shape = [48, 64]
    dst_shape = [64, 64]
    dtype = DataType.FP32
    pad = "zero"


# =============================================================================
# Tests
# =============================================================================


class TestFillpadExpand:
    """fillpad_expand: copy a smaller source into a larger padded destination."""

    def test_row_expand_fp32(self, test_runner):
        result = test_runner.run(FillpadExpandRowFP32Case())
        assert result.passed, f"Test failed: {result.error}"

    def test_col_expand_fp32(self, test_runner):
        result = test_runner.run(FillpadExpandColFP32Case())
        assert result.passed, f"Test failed: {result.error}"

    def test_both_expand_max_fp32(self, test_runner):
        result = test_runner.run(FillpadExpandBothMaxFP32Case())
        assert result.passed, f"Test failed: {result.error}"

    def test_narrow_valid_fp32(self, test_runner):
        result = test_runner.run(FillpadExpandNarrowValidFP32Case())
        assert result.passed, f"Test failed: {result.error}"

    def test_narrow_valid_both_min_fp32(self, test_runner):
        result = test_runner.run(FillpadExpandNarrowValidBothMinFP32Case())
        assert result.passed, f"Test failed: {result.error}"

    def test_tail_valid_fp32(self, test_runner):
        result = test_runner.run(FillpadExpandTailValidFP32Case())
        assert result.passed, f"Test failed: {result.error}"

    def test_row_expand_fp16(self, test_runner):
        result = test_runner.run(FillpadExpandRowFP16Case())
        assert result.passed, f"Test failed: {result.error}"

    def test_narrow_valid_fp16(self, test_runner):
        result = test_runner.run(FillpadExpandNarrowValidFP16Case())
        assert result.passed, f"Test failed: {result.error}"

    def test_row_expand_int32(self, test_runner):
        result = test_runner.run(FillpadExpandRowINT32Case())
        assert result.passed, f"Test failed: {result.error}"

    def test_tensor_level_fp32(self, test_runner):
        result = test_runner.run(FillpadExpandTensorFP32Case())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
