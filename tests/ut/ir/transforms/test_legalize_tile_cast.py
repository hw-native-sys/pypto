# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the LegalizeTileCast pass."""

from __future__ import annotations

import pypto.language as pl
from pypto import backend, ir, passes
from pypto.backend import BackendType


class _CastTargetCollector(ir.IRVisitor):
    """Record each tile.cast's (src_dtype, target_type) in visitation order."""

    def __init__(self) -> None:
        super().__init__()
        self.pairs: list[tuple[str, str]] = []

    def visit_call(self, op: ir.Call) -> None:
        if op.op.name == "tile.cast":
            src_ty = op.args[0].type
            src = str(src_ty.dtype)
            dst = str(op.kwargs["target_type"])
            self.pairs.append((src, dst))
        super().visit_call(op)


def _cast_pairs(prog) -> list[tuple[str, str]]:
    c = _CastTargetCollector()
    c.visit_program(prog)
    return c.pairs


def _run(program, backend_type: BackendType):
    backend.reset_for_testing()
    backend.set_backend_type(backend_type)
    try:
        return passes.legalize_tile_cast()(program)
    finally:
        backend.reset_for_testing()


def test_legalize_tile_cast_pass_factory_exists():
    p = passes.legalize_tile_cast()
    assert p is not None
    assert p.get_name() == "LegalizeTileCast"


def test_a5_int32_to_fp16_becomes_fp32_bridge():
    """A5 has no native I32→FP16; expand to I32→FP32→FP16."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 16], pl.INT32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            t: pl.Tile[[16, 16], pl.INT32] = pl.load(x, [0, 0], [16, 16])
            c: pl.Tile[[16, 16], pl.FP16] = pl.tile.cast(t, target_type=pl.FP16, mode="round")
            out_t: pl.Tensor[[16, 16], pl.FP16] = pl.store(c, [0, 0], out)
            return out_t

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.INT32]) -> pl.Tensor[[16, 16], pl.FP16]:
            o: pl.Tensor[[16, 16], pl.FP16] = pl.create_tensor([16, 16], dtype=pl.FP16)
            return self.kernel(x, o)

    after = _run(Before, BackendType.Ascend950)
    pairs = _cast_pairs(after)
    assert pairs == [("int32", "fp32"), ("fp32", "fp16")], pairs


def test_a2a3_int32_to_fp16_stays_native():
    """A2A3 has a native I32→FP16 deq path — leave the single cast."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 16], pl.INT32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            t: pl.Tile[[16, 16], pl.INT32] = pl.load(x, [0, 0], [16, 16])
            c: pl.Tile[[16, 16], pl.FP16] = pl.tile.cast(t, target_type=pl.FP16, mode="round")
            out_t: pl.Tensor[[16, 16], pl.FP16] = pl.store(c, [0, 0], out)
            return out_t

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.INT32]) -> pl.Tensor[[16, 16], pl.FP16]:
            o: pl.Tensor[[16, 16], pl.FP16] = pl.create_tensor([16, 16], dtype=pl.FP16)
            return self.kernel(x, o)

    after = _run(Before, BackendType.Ascend910B)
    pairs = _cast_pairs(after)
    assert pairs == [("int32", "fp16")], pairs


def test_a5_fp16_to_bf16_via_fp32():
    """A5 has no native FP16→BF16; expand via FP32."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.BF16]],
        ) -> pl.Tensor[[16, 16], pl.BF16]:
            t: pl.Tile[[16, 16], pl.FP16] = pl.load(x, [0, 0], [16, 16])
            c: pl.Tile[[16, 16], pl.BF16] = pl.tile.cast(t, target_type=pl.BF16, mode="round")
            out_t: pl.Tensor[[16, 16], pl.BF16] = pl.store(c, [0, 0], out)
            return out_t

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP16]) -> pl.Tensor[[16, 16], pl.BF16]:
            o: pl.Tensor[[16, 16], pl.BF16] = pl.create_tensor([16, 16], dtype=pl.BF16)
            return self.kernel(x, o)

    after = _run(Before, BackendType.Ascend950)
    pairs = _cast_pairs(after)
    assert pairs == [("fp16", "fp32"), ("fp32", "bfloat16")], pairs


def test_native_fp32_to_fp16_unchanged_on_a5():
    """Already-native casts (and FIXPIPE-foldable ones) must not be rewritten."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            t: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
            c: pl.Tile[[16, 16], pl.FP16] = pl.tile.cast(t, target_type=pl.FP16, mode="rint")
            out_t: pl.Tensor[[16, 16], pl.FP16] = pl.store(c, [0, 0], out)
            return out_t

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP16]:
            o: pl.Tensor[[16, 16], pl.FP16] = pl.create_tensor([16, 16], dtype=pl.FP16)
            return self.kernel(x, o)

    after = _run(Before, BackendType.Ascend950)
    pairs = _cast_pairs(after)
    assert pairs == [("fp32", "fp16")], pairs
    ir.assert_structural_equal(after, Before)


def test_idempotent_on_already_bridged_chain():
    """Hand-written I32→FP32→FP16 stays unchanged (each hop already native)."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 16], pl.INT32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            t: pl.Tile[[16, 16], pl.INT32] = pl.load(x, [0, 0], [16, 16])
            m: pl.Tile[[16, 16], pl.FP32] = pl.tile.cast(t, target_type=pl.FP32, mode="round")
            c: pl.Tile[[16, 16], pl.FP16] = pl.tile.cast(m, target_type=pl.FP16, mode="round")
            out_t: pl.Tensor[[16, 16], pl.FP16] = pl.store(c, [0, 0], out)
            return out_t

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.INT32]) -> pl.Tensor[[16, 16], pl.FP16]:
            o: pl.Tensor[[16, 16], pl.FP16] = pl.create_tensor([16, 16], dtype=pl.FP16)
            return self.kernel(x, o)

    after = _run(Before, BackendType.Ascend950)
    assert _cast_pairs(after) == [("int32", "fp32"), ("fp32", "fp16")]
    ir.assert_structural_equal(after, Before)
