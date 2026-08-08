# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the LegalizeTileCast pass."""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType


class _CastTargetCollector(ir.IRVisitor):
    """Record each tile.cast's (src_dtype, target_type) in visitation order."""

    def __init__(self) -> None:
        super().__init__()
        self.pairs: list[tuple[str, str]] = []

    def visit_call(self, op: ir.Call) -> None:
        if op.op.name == ir.get_op("tile.cast").name:
            src_ty = op.args[0].type
            assert isinstance(src_ty, ir.TileType)
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


def test_a5_uint32_to_fp32_rejects_narrowing_bridge():
    """A narrowing bridge must never be chosen, even when it is the shortest path.

    A5 has no native UINT32 -> FP32. The only 2-hop routes go through INT16 /
    UINT16 / UINT8, all of which discard values a direct conversion would keep
    (40000 is exactly representable in FP32 but not in INT16), so the pass must
    refuse rather than silently emit a lossy chain.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 16], pl.UINT32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            t: pl.Tile[[16, 16], pl.UINT32] = pl.load(x, [0, 0], [16, 16])
            c: pl.Tile[[16, 16], pl.FP32] = pl.tile.cast(t, target_type=pl.FP32, mode="none")
            out_t: pl.Tensor[[16, 16], pl.FP32] = pl.store(c, [0, 0], out)
            return out_t

    with pytest.raises(ValueError) as excinfo:
        _run(Before, BackendType.Ascend950)
    assert "cast" in str(excinfo.value).lower()


def test_resolve_arch_without_configured_backend_is_a_noop():
    """With no backend configured the pass must do nothing, not raise.

    The native-cast table belongs to the BackendHandler, so with no backend
    there is nothing to legalize against. Both PassContext::GetBackendHandler()
    and BackendConfig::GetBackend() CHECK-fail when unconfigured, so the pass
    probes IsConfigured() first and returns the function untouched.

    FP32 -> INT8 is the discriminating pair: it is a 2-hop chain under *both*
    shipped profiles, so leaving it as a single cast can only mean no profile
    was applied at all.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.INT8]],
        ) -> pl.Tensor[[16, 16], pl.INT8]:
            t: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
            c: pl.Tile[[16, 16], pl.INT8] = pl.tile.cast(t, target_type=pl.INT8, mode="round")
            out_t: pl.Tensor[[16, 16], pl.INT8] = pl.store(c, [0, 0], out)
            return out_t

    backend.reset_for_testing()
    try:
        after = passes.legalize_tile_cast()(Before)
    finally:
        backend.reset_for_testing()
    assert _cast_pairs(after) == [("fp32", "int8")]
    ir.assert_structural_equal(after, Before)


@pytest.mark.parametrize(
    "backend_type,src,dst,expected",
    [
        # HF8 exists only on a5; a2a3 has no path and must reject.
        (BackendType.Ascend950, "HF8", "FP32", [("hf8", "fp32")]),
        (BackendType.Ascend910B, "HF8", "FP32", None),
        # INT4 exists only on a2a3; a5 has no path and must reject.
        (BackendType.Ascend910B, "FP16", "INT4", [("fp16", "int4")]),
        (BackendType.Ascend950, "FP16", "INT4", None),
    ],
)
def test_native_table_is_owned_by_the_backend_handler(backend_type, src, dst, expected):
    """Each arch's native-cast table lives on its own BackendHandler.

    These four cases are the ones that differ between the two shipped tables,
    so editing the wrong handler flips exactly this test.
    """
    src_dt, dst_dt = getattr(pl, src), getattr(pl, dst)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 16], src_dt],
            out: pl.Out[pl.Tensor[[16, 16], dst_dt]],
        ) -> pl.Tensor[[16, 16], dst_dt]:
            t: pl.Tile[[16, 16], src_dt] = pl.load(x, [0, 0], [16, 16])
            c: pl.Tile[[16, 16], dst_dt] = pl.tile.cast(t, target_type=dst_dt, mode="round")
            out_t: pl.Tensor[[16, 16], dst_dt] = pl.store(c, [0, 0], out)
            return out_t

    if expected is None:
        with pytest.raises(ValueError, match="no native cast path"):
            _run(Before, backend_type)
    else:
        assert _cast_pairs(_run(Before, backend_type)) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
