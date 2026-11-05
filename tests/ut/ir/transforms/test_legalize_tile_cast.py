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
from pypto import DataType, backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.printer import python_print
from pypto.language.parser.diagnostics import InvalidOperationError

_TILE_CAST = ir.get_op("tile.cast").name


class _CastTargetCollector(ir.IRVisitor):
    """Record each tile.cast's (src_dtype, target_type) in visitation order."""

    def __init__(self) -> None:
        super().__init__()
        self.pairs: list[tuple[str, str]] = []

    def visit_call(self, op: ir.Call) -> None:
        if op.op.name == _TILE_CAST:
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


def _assert_recorded_cast_roundtrip(program):
    printed = python_print(program, format=False)
    assert 'pl._dump.call("tile.cast"' in printed
    ir.assert_structural_equal(program, pl.parse_program(printed), enable_auto_mapping=True)


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


@pytest.mark.parametrize("dtype", [pl.fp4e2m1x2, pl.fp4e1m2x2])
def test_a5_packed_fp4_to_bf16_cast_is_silent_and_expands_last_dim(capfd, dtype):
    """Packed FP4→BF16 is a native hop; last axis expands 2:1."""
    from pypto import LogLevel, get_log_level, set_log_level  # noqa: PLC0415

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 32], dtype],
            out: pl.Out[pl.Tensor[[16, 64], pl.BF16]],
        ) -> pl.Tensor[[16, 64], pl.BF16]:
            t = pl.load(x, [0, 0], [16, 32])
            c = pl.cast(t, pl.BF16)
            return pl.store(c, [0, 0], out)

    prev = get_log_level()
    set_log_level(LogLevel.WARN)
    try:
        capfd.readouterr()
        after = _run(Before, BackendType.Ascend950)
    finally:
        set_log_level(prev)
    assert _cast_pairs(after) == [(str(dtype), "bfloat16")]
    err = capfd.readouterr().err
    assert "LegalizeTileCast" not in err
    assert "FP4→BF16→FP32→FP8" not in err

    class _ShapeCollector(ir.IRVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.cast_out_shape: tuple[int, int] | None = None

        def visit_call(self, op: ir.Call) -> None:
            if op.op.name == _TILE_CAST:
                ty = op.type
                assert isinstance(ty, ir.TileType)
                rows = ty.shape[0]
                cols = ty.shape[1]
                assert isinstance(rows, ir.ConstInt)
                assert isinstance(cols, ir.ConstInt)
                self.cast_out_shape = (rows.value, cols.value)
            super().visit_call(op)

    shapes = _ShapeCollector()
    shapes.visit_program(after)
    assert shapes.cast_out_shape == (16, 64)


def test_a5_uint8_ub_alias_to_bf16_stays_one_native_cast():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            packed: pl.Tensor[[32, 256], pl.UINT8],
            output: pl.Out[pl.Tensor[[32, 512], pl.BF16]],
        ) -> pl.Tensor[[32, 512], pl.BF16]:
            payload = pl.tile.load(packed, [0, 0], [32, 256])
            fp4 = pl.reinterpret_view(payload, pl.FP4E2M1X2)
            decoded = pl.cast(fp4, pl.BF16)
            return pl.tile.store(decoded, [0, 0], output)

    after = _run(Before, BackendType.Ascend950)
    ir.assert_structural_equal(after, Before)
    assert _cast_pairs(after) == [("fp4e2m1x2", "bfloat16")]


@pytest.mark.parametrize("source", [pl.fp4e2m1x2, pl.fp4e1m2x2])
@pytest.mark.parametrize("target", [pl.FP16, pl.FP32, pl.INT8, pl.UINT8, pl.INT16, pl.INT32, pl.INT64])
def test_a5_packed_fp4_decodes_through_lossless_float_bridges(source, target):
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, x: pl.Tensor[[4, 32], source], out: pl.Out[pl.Tensor[[4, 64], target]]):
            packed = pl.load(x, [0, 0], [4, 32])
            decoded = pl.cast(packed, target)
            return pl.store(decoded, [0, 0], out)

    after = _run(Before, BackendType.Ascend950)
    pairs = _cast_pairs(after)
    assert pairs[0] == (str(source), "bfloat16")
    assert pairs[-1][1] == str(target)
    # All intermediate floating-point formats represent every FP4 code exactly.
    assert all(dst in ("bfloat16", "fp16", "fp32") for _, dst in pairs[:-1])


@pytest.mark.parametrize("target", [pl.fp4e2m1x2, pl.fp4e1m2x2])
@pytest.mark.parametrize("source", [pl.FP16, pl.FP32, pl.INT16, pl.INT32])
def test_a5_packed_fp4_encoding_rejects_lossy_bf16_bridges(source, target):
    # For rint, E1M2 0.1251 and E2M1 0.2501 first round to BF16 midpoints,
    # then round down to zero, although a direct FP4 conversion rounds up.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, x: pl.Tensor[[4, 64], source]):
            values = pl.load(x, [0, 0], [4, 64])
            _encoded = pl.cast(values, target, mode="rint")

    with pytest.raises(ValueError, match="no lossless cast path.*BF16 intermediate can double-round"):
        _run(Before, BackendType.Ascend950)


@pytest.mark.parametrize("target", [pl.fp4e2m1x2, pl.fp4e1m2x2])
@pytest.mark.parametrize("source", [pl.FP8E4M3FN, pl.FP8E5M2, pl.HF8, pl.INT8, pl.UINT8])
def test_a5_packed_fp4_encoding_keeps_lossless_bf16_bridges(source, target):
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, x: pl.Tensor[[4, 64], source]):
            values = pl.load(x, [0, 0], [4, 64])
            _encoded = pl.cast(values, target, mode="rint")

    pairs = _cast_pairs(_run(Before, BackendType.Ascend950))
    assert pairs[0][0] == str(source)
    assert pairs[-1] == ("bfloat16", str(target))


@pytest.mark.parametrize(
    "dtype,target",
    [(pl.fp4e1m2x2, pl.FP8E4M3FN), (pl.fp4e2m1x2, pl.FP8E5M2), (pl.fp4e1m2x2, pl.FP8E5M2)],
)
def test_a5_packed_fp4_to_fp8_cast_emits_warning(capfd, dtype, target):
    """Packed FP4→FP8 is legalized with a warning about the three-hop bridge."""
    from pypto import LogLevel, get_log_level, set_log_level  # noqa: PLC0415

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 32], dtype],
            out: pl.Out[pl.Tensor[[16, 64], target]],
        ) -> pl.Tensor[[16, 64], target]:
            t = pl.load(x, [0, 0], [16, 32])
            c = pl.cast(t, target)
            return pl.store(c, [0, 0], out)

    prev = get_log_level()
    set_log_level(LogLevel.WARN)
    try:
        capfd.readouterr()
        after = _run(Before, BackendType.Ascend950)
    finally:
        set_log_level(prev)
    assert _cast_pairs(after) == [
        (str(dtype), "bfloat16"),
        ("bfloat16", "fp32"),
        ("fp32", str(target)),
    ]
    err = capfd.readouterr().err
    assert "LegalizeTileCast" in err
    assert "FP4→BF16→FP32→FP8" in err
    assert "LUT" in err


@pytest.mark.parametrize("dtype", [pl.fp4e2m1x2, pl.fp4e1m2x2])
def test_a5_packed_fp4_cast_expands_last_dim_valid_shape(dtype):
    """Packed FP4→FP8 legalization expands the last axis 2:1."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 32], dtype],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            t = pl.load(x, [0, 0], [16, 32], valid_shape=[8, 24])
            c = pl.cast(t, pl.FP8E4M3FN)
            return pl.store(c, [0, 0], out)

    after = _run(Before, BackendType.Ascend950)

    # E2M1 now decodes directly, including partial static validity. E1M2
    # retains the existing BF16/FP32 bridge; both must expand carrier columns.
    expected = (
        [(str(dtype), "fp8e4m3fn")]
        if dtype == pl.fp4e2m1x2
        else [(str(dtype), "bfloat16"), ("bfloat16", "fp32"), ("fp32", "fp8e4m3fn")]
    )
    assert _cast_pairs(after) == expected

    class _ValidShapeCollector(ir.IRVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.shapes: list[tuple[int, int]] = []

        def visit_call(self, op: ir.Call) -> None:
            if op.op.name == _TILE_CAST:
                tile_type = op.type
                assert isinstance(tile_type, ir.TileType)
                valid = tile_type.get_effective_tile_view().valid_shape
                rows, cols = valid
                assert isinstance(rows, ir.ConstInt)
                assert isinstance(cols, ir.ConstInt)
                self.shapes.append((rows.value, cols.value))
            super().visit_call(op)

    collector = _ValidShapeCollector()
    collector.visit_program(after)
    assert collector.shapes
    assert collector.shapes[0] == (8, 48)
    assert all(shape == (8, 48) for shape in collector.shapes)

    class _StrideCollector(ir.IRVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.strides: list[list[int]] = []

        def visit_call(self, op: ir.Call) -> None:
            if op.op.name == _TILE_CAST:
                tile_type = op.type
                assert isinstance(tile_type, ir.TileType)
                stride = tile_type.get_effective_tile_view().stride
                values: list[int] = []
                for dim in stride:
                    assert isinstance(dim, ir.ConstInt)
                    values.append(dim.value)
                self.strides.append(values)
            super().visit_call(op)

    strides = _StrideCollector()
    strides.visit_program(after)
    assert strides.strides
    # Packed physical [16, 32] stride [32, 1] unpacks to [16, 64] stride [64, 1].
    # valid_shape [8, 24] → [8, 48] is the sub-rectangle, not the row pitch.
    assert strides.strides[0] == [64, 1]
    assert all(stride == [64, 1] for stride in strides.strides)


@pytest.mark.parametrize("source,target", [(pl.fp4e2m1x2, pl.fp4e1m2x2), (pl.fp4e1m2x2, pl.fp4e2m1x2)])
def test_a5_packed_format_conversion_uses_bf16_without_changing_carrier_shape(source, target):
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, x: pl.Tensor[[16, 32], source], out: pl.Out[pl.Tensor[[16, 32], target]]):
            value = pl.load(x, [0, 0], [16, 32])
            result = pl.cast(value, target)
            return pl.store(result, [0, 0], out)

    after = _run(Before, BackendType.Ascend950)
    assert _cast_pairs(after) == [(str(source), "bfloat16"), ("bfloat16", str(target))]

    # Store type checking also establishes that the final width remains 32 carriers.
    class Shapes(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.cast_shapes = []

        def visit_call(self, op):
            if op.op.name == _TILE_CAST:
                tile_type = op.type
                assert isinstance(tile_type, ir.TileType)
                shape = []
                for extent in tile_type.shape:
                    assert isinstance(extent, ir.ConstInt)
                    shape.append(extent.value)
                self.cast_shapes.append(shape)
            super().visit_call(op)

    shapes = Shapes()
    shapes.visit_program(after)
    assert shapes.cast_shapes == [[16, 64], [16, 32]]


@pytest.mark.parametrize("dtype", [pl.fp4e2m1x2, pl.fp4e1m2x2])
@pytest.mark.parametrize("decode", [False, True])
def test_packed_fp4_cast_rejects_boxed_storage(dtype, decode):
    source_dtype = dtype if decode else pl.BF16
    target_dtype = pl.BF16 if decode else dtype
    span = ir.Span.unknown()
    view = ir.TileView(blayout=ir.TileLayout.row_major, slayout=ir.TileLayout.row_major)
    source = ir.Var("source", ir.TileType([32, 64], source_dtype, None, view), span)
    with pytest.raises(ValueError, match="requires none_box storage"):
        ir.op.tile.cast(source, target_dtype)


@pytest.mark.parametrize("dtype", [pl.fp4e2m1x2, pl.fp4e1m2x2])
def test_packed_fp4_cast_accepts_even_dynamic_valid_shape(dtype):
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 64], pl.BF16],
            out: pl.Out[pl.Tensor[[16, 32], dtype]],
            n: pl.Scalar[pl.INDEX],
        ):
            value = pl.load(x, [0, 0], [16, 64], valid_shape=[16, n * 2])
            result = pl.cast(value, dtype)
            return pl.store(result, [0, 0], out)

    after = _run(Before, BackendType.Ascend950)
    assert _cast_pairs(after) == [("bfloat16", str(dtype))]


@pytest.mark.parametrize("dtype", [pl.fp4e2m1x2, pl.fp4e1m2x2])
def test_packed_fp4_cast_rejects_unsupported_odd_rounding(dtype):
    with pytest.raises(InvalidOperationError, match="does not support mode=odd"):

        @pl.program
        class Bad:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[16, 64], pl.BF16]):
                value = pl.load(x, [0, 0], [16, 64])
                return pl.cast(value, dtype, mode="odd")


def test_cast_to_packed_fp4_rejects_dynamic_last_dim():
    """Wider → FP4E2M1X2 requires a static positive even last dim."""
    n = pl.dynamic("N")

    with pytest.raises(InvalidOperationError, match="static positive even last dimension"):

        @pl.program
        class Bad:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, n], pl.BF16],
                out: pl.Out[pl.Tensor[[16, n // 2], pl.FP4E2M1X2]],
            ) -> pl.Tensor[[16, n // 2], pl.FP4E2M1X2]:
                t = pl.load(x, [0, 0], [16, n])
                c = pl.cast(t, pl.FP4E2M1X2)
                return pl.store(c, [0, 0], out)


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

    after = _run(Before, BackendType.Ascend910)
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
        (BackendType.Ascend910, "HF8", "FP32", None),
        # INT4 exists only on a2a3; a5 has no path and must reject.
        (BackendType.Ascend910, "FP16", "INT4", [("fp16", "int4")]),
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


def _saturation_modes(program) -> list[int | None]:
    """Each tile.cast's saturation_mode in visitation order; None where left implicit.

    None is the *default* (saturating), not "unspecified": the IR records only a
    deviation from ``ir::kDefaultSaturationMode``.
    """

    class _Collector(ir.IRVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.modes: list[int | None] = []

        def visit_call(self, op: ir.Call) -> None:
            if op.op.name == _TILE_CAST:
                mode = op.kwargs.get("saturation_mode")
                assert mode is None or isinstance(mode, int)
                self.modes.append(mode)
            super().visit_call(op)

    collector = _Collector()
    collector.visit_program(program)
    return collector.modes


def test_an_opt_out_applies_only_to_the_final_legalized_hop():
    """A2/A3 FP32→INT8 expands via FP16; saturation names the destination, so it rides the last hop.

    Only the final hop reaches the dtype the author named, so an opt-out belongs
    there. The widening intermediate keeps the default: it converts into a dtype
    the author never named, and every value INT8 can represent passes through FP16
    exactly either way.
    """
    saturation_mode = "off"

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 64], pl.INT8]],
        ) -> pl.Tensor[[16, 64], pl.INT8]:
            t = pl.load(x, [0, 0], [16, 64])
            c = pl.cast(t, pl.INT8, mode="trunc", saturation_mode=saturation_mode)
            return pl.store(c, [0, 0], out)

    after = _run(Before, BackendType.Ascend910)
    assert _cast_pairs(after) == [("fp32", "fp16"), ("fp16", "int8")]
    assert _saturation_modes(after) == [None, 0], "intermediate defaulted, final opted out"


def test_native_cast_keeps_its_saturation_unchanged():
    """A single-hop cast is not rewritten, so its opt-out must survive untouched."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 64], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 64], pl.INT8]],
        ) -> pl.Tensor[[16, 64], pl.INT8]:
            t = pl.load(x, [0, 0], [16, 64])
            c = pl.cast(t, pl.INT8, mode="trunc", saturation_mode="off")
            return pl.store(c, [0, 0], out)

    after = _run(Before, BackendType.Ascend910)
    assert _cast_pairs(after) == [("fp16", "int8")]
    assert _saturation_modes(after) == [0]


def test_legalized_chain_keeps_a_caller_supplied_tmp_on_the_final_hop():
    """An explicit scratch operand belongs to the narrowing hop, which is the last one.

    The intermediate widening hop needs no scratch, and dropping the operand
    would silently discard the tile the author allocated for the cast.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 64], pl.INT8]],
        ) -> pl.Tensor[[16, 64], pl.INT8]:
            t = pl.load(x, [0, 0], [16, 64])
            tmp = pl.tile.create([1, 256], dtype=pl.INT8, target_memory=pl.Mem.Vec)
            c = pl.tile.cast(t, pl.INT8, mode="trunc", tmp=tmp)
            return pl.store(c, [0, 0], out)

    after = _run(Before, BackendType.Ascend910)
    assert _cast_pairs(after) == [("fp32", "fp16"), ("fp16", "int8")]

    class _ArityCollector(ir.IRVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.arities: list[int] = []

        def visit_call(self, op: ir.Call) -> None:
            if op.op.name == _TILE_CAST:
                self.arities.append(len(op.args))
            super().visit_call(op)

    collector = _ArityCollector()
    collector.visit_program(after)
    assert collector.arities == [1, 2]


@pytest.mark.parametrize("dtype", [pl.fp4e2m1x2, pl.fp4e1m2x2])
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("tile_cast", [False, True])
def test_packed_fp4_cast_dn_scales_row_axis_and_preserves_layout(dtype, decode, tile_cast):
    source_dtype = dtype if decode else pl.BF16
    target_dtype = pl.BF16 if decode else dtype
    span = ir.Span.unknown()
    rows = 32 if decode else 64
    expected_rows = 64 if decode else 32
    shape = [ir.ConstInt(value, pl.INDEX, span) for value in (rows, 32)]
    if tile_cast:
        view = ir.TileView(blayout=ir.TileLayout.col_major)
        source = ir.Var("source", ir.TileType(shape, source_dtype, None, view), span)
        result = ir.op.tile.cast(source, target_dtype).type
        assert isinstance(result, ir.TileType)
        result_view = result.get_effective_tile_view()
        assert result_view.blayout == ir.TileLayout.col_major
    else:
        stride = [ir.ConstInt(value, pl.INDEX, span) for value in (1, rows)]
        tensor_view = ir.TensorView(stride, ir.TensorLayout.DN)
        source = ir.Var("source", ir.TensorType(shape, source_dtype, None, tensor_view), span)
        result = ir.op.tensor.cast(source, target_dtype).type
        assert isinstance(result, ir.TensorType)
        result_view = result.tensor_view
        assert result_view is not None
        assert result_view.layout == ir.TensorLayout.DN
    for value, expected in zip(result.shape, (expected_rows, 32), strict=True):
        assert isinstance(value, ir.ConstInt)
        assert value.value == expected
    for value, expected in zip(result_view.stride, (1, expected_rows), strict=True):
        assert isinstance(value, ir.ConstInt)
        assert value.value == expected


@pytest.mark.parametrize("dtype", [pl.fp4e2m1x2, pl.fp4e1m2x2])
@pytest.mark.parametrize("valid", [[32, 32], [64, 16]])
@pytest.mark.parametrize("tile_cast", [False, True])
def test_packed_fp4_cast_dn_rejects_partial_or_padded_extent(dtype, valid, tile_cast):
    span = ir.Span.unknown()
    if tile_cast:
        view = ir.TileView(valid_shape=valid, blayout=ir.TileLayout.col_major)
        source = ir.Var("source", ir.TileType([64, 32], pl.BF16, None, view), span)
        with pytest.raises(ValueError, match="requires a full static 2D valid shape"):
            ir.op.tile.cast(source, dtype)
    else:
        view = ir.TensorView([1, 64], ir.TensorLayout.DN, valid_shape=valid)
        source = ir.Var("source", ir.TensorType([64, 32], pl.BF16, None, view), span)
        with pytest.raises(ValueError, match="requires a full static 2D valid shape"):
            ir.op.tensor.cast(source, dtype)


@pytest.mark.parametrize("dtype", [pl.fp4e2m1x2, pl.fp4e1m2x2])
def test_packed_fp4_cast_dn_rejects_strided_storage_and_wider_chains(dtype):
    span = ir.Span.unknown()
    strided = ir.TileView(stride=[1, 64], blayout=ir.TileLayout.col_major)
    source = ir.Var("source", ir.TileType([32, 32], dtype, None, strided), span)
    with pytest.raises(ValueError, match="requires canonical contiguous column-major strides"):
        ir.op.tile.cast(source, pl.BF16)
    dense = ir.TileView(blayout=ir.TileLayout.col_major)
    source = ir.Var("source", ir.TileType([32, 32], dtype, None, dense), span)
    with pytest.raises(ValueError, match="supports only BF16 and packed FP4 endpoints"):
        ir.op.tile.cast(source, pl.FP32)


@pytest.mark.parametrize("dtype", [DataType.FP4E2M1X2, DataType.FP4E1M2X2])
@pytest.mark.parametrize("loop_kind", ["for", "while"])
def test_legalize_packed_dn_cast_preserves_loop_backedge(dtype, loop_kind):
    """Already-packed DN casts keep their row-packed strides and carried storage."""
    span = ir.Span.unknown()
    source_view = ir.TileView(
        valid_shape=[64, 4], stride=[1, 64], blayout=ir.TileLayout.col_major, start_offset=None
    )
    source_type = ir.TileType([64, 4], DataType.BF16, None, source_view, ir.MemorySpace.Vec)
    source = ir.Var("source", source_type, span)
    cast = ir.create_op_call("tile.cast", [source], {"target_type": dtype, "mode": 1}, span)
    cast_type = cast.type
    assert isinstance(cast_type, ir.TileType)
    assert cast_type.tile_view is not None
    assert [dim.value for dim in cast_type.shape if isinstance(dim, ir.ConstInt)] == [32, 4]
    assert [dim.value for dim in cast_type.tile_view.stride if isinstance(dim, ir.ConstInt)] == [1, 32]

    storage = ir.MemRef("carried_storage", span)
    carry_type = ir.TileType([32, 4], dtype, storage, cast_type.tile_view, ir.MemorySpace.Vec)
    seed = ir.Var("seed", cast_type, span)
    argument = ir.IterArg("carried", carry_type, seed, span)
    alias = ir.Var("alias", carry_type, span)
    converted = ir.Var("converted", cast_type, span)
    result = ir.Var("result", carry_type, span)
    body = ir.SeqStmts(
        [
            ir.AssignStmt(alias, argument, span),
            ir.AssignStmt(converted, cast, span),
            ir.YieldStmt([converted], span),
        ],
        span,
    )
    if loop_kind == "for":
        loop = ir.ForStmt(
            ir.Var("i", ir.ScalarType(DataType.INDEX), span),
            ir.ConstInt(0, pl.INDEX, span),
            ir.ConstInt(2, pl.INDEX, span),
            ir.ConstInt(1, pl.INDEX, span),
            [argument],
            body,
            [result],
            span,
        )
    else:
        loop = ir.WhileStmt(ir.ConstBool(False, span), [argument], body, [result], span)
    program = ir.Program(
        [
            ir.Function(
                "main",
                [seed, source],
                [carry_type],
                ir.SeqStmts([loop, ir.ReturnStmt([result], span)], span),
                span,
                ir.FunctionType.Opaque,
            )
        ],
        "packed_dn_loop",
        span,
    )
    # Text loop headers cannot spell declaration-only carried storage; retain
    # property verification and inspect storage/alias identity directly below.
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        packed = _run(program, BackendType.Ascend950)
    ir.assert_structural_equal(packed, program)
    ir.verify_structural_bindings(packed)
    function = next(iter(packed.functions.values()))
    assert isinstance(function.body, ir.SeqStmts)
    packed_loop = function.body.stmts[0]
    assert isinstance(packed_loop, (ir.ForStmt, ir.WhileStmt))
    binder = packed_loop.iter_args[0]
    assert isinstance(binder.type, ir.TileType)
    assert binder.type.memref is storage
    assert isinstance(packed_loop.body, ir.SeqStmts)
    binding = packed_loop.body.stmts[0]
    assert isinstance(binding, ir.AssignStmt)
    assert binding.value is binder


@pytest.mark.parametrize("tile_surface", [False, True])
@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
)
def test_a5_e2m1_to_e4m3_is_one_silent_native_cast(capfd, tile_surface, planner):
    """Aligned ND E2M1 carriers decode without intermediate floating tiles."""
    from pypto import LogLevel, get_log_level, set_log_level  # noqa: PLC0415

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            source: pl.Tensor[[16, 32], pl.fp4e2m1x2],
            output: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            packed = pl.load(source, [0, 0], [16, 32])
            if tile_surface:
                decoded = pl.tile.cast(packed, pl.FP8E4M3FN)
            else:
                decoded = pl.cast(packed, pl.FP8E4M3FN)
            return pl.store(decoded, [0, 0], output)

    previous = get_log_level()
    set_log_level(LogLevel.WARN)
    try:
        capfd.readouterr()
        with passes.PassContext([], memory_planner=planner):
            # Resolve the compile-time surface choice before counting casts: the
            # parser otherwise retains both branches of the authored conditional.
            before = passes.simplify()(Before)
            after = _run(before, BackendType.Ascend950)
    finally:
        set_log_level(previous)
    assert _cast_pairs(after) == [("fp4e2m1x2", "fp8e4m3fn")]
    ir.assert_structural_equal(after, before)
    assert "FP4→BF16→FP32→FP8" not in capfd.readouterr().err
    with pytest.raises(ValueError, match="no native cast path"):
        _run(before, BackendType.Ascend910)


@pytest.mark.parametrize(
    "case", ["unaligned", "dynamic_valid", "dynamic_affine_valid", "narrowed_result", "row_limit", "strided"]
)
def test_e2m1_to_e4m3_unsuitable_descriptors_keep_float_bridge(case):
    """The dtype edge must not bypass direct TCVT's packed-axis restrictions."""
    span = ir.Span.unknown()
    rows, cols = (65536, 32) if case == "row_limit" else (16, 16 if case == "unaligned" else 32)
    valid_cols = ir.Var("valid_cols", ir.ScalarType(pl.INDEX), span)
    valid = [rows, valid_cols] if case == "dynamic_valid" else []
    if case == "dynamic_affine_valid":
        valid = [
            ir.ConstInt(rows, pl.INDEX, span),
            ir.Add(valid_cols, ir.ConstInt(1, pl.INDEX, span), pl.INDEX, span),
        ]
    stride = [cols + 32, 1] if case == "strided" else []
    source_type = ir.TileType([rows, cols], pl.fp4e2m1x2, None, ir.TileView(valid_shape=valid, stride=stride))
    source = ir.Var("source", source_type, span)
    cast = ir.op.tile.cast(source, pl.FP8E4M3FN)
    if case == "narrowed_result":
        result_type = ir.TileType(
            [rows, cols * 2], pl.FP8E4M3FN, None, ir.TileView(valid_shape=[8, cols * 2])
        )
        cast = ir.Call(cast.op, cast.args, cast.kwargs, result_type, span)
    decoded = ir.Var("decoded", cast.type, span)
    parameters = [valid_cols, source] if case in ("dynamic_valid", "dynamic_affine_valid") else [source]
    before = ir.Program(
        [
            ir.Function(
                "kernel",
                parameters,
                [cast.type],
                ir.SeqStmts([ir.AssignStmt(decoded, cast, span), ir.ReturnStmt([decoded], span)], span),
                span,
                ir.FunctionType.Opaque,
            )
        ],
        "fallback",
        span,
    )
    after = _run(before, BackendType.Ascend950)
    assert _cast_pairs(after) == [
        ("fp4e2m1x2", "bfloat16"),
        ("bfloat16", "fp32"),
        ("fp32", "fp8e4m3fn"),
    ]
    if case == "narrowed_result":
        _assert_recorded_cast_roundtrip(after)


@pytest.mark.parametrize("tile_surface", [False, True])
def test_e2m1_to_e4m3_type_inference_preserves_carrier_geometry(tile_surface):
    span = ir.Span.unknown()
    if tile_surface:
        source_type = ir.TileType(
            [16, 32], pl.fp4e2m1x2, None, ir.TileView(valid_shape=[8, 24], stride=[32, 1])
        )
        source = ir.Var("source", source_type, span)
        result = ir.op.tile.cast(source, pl.FP8E4M3FN).type
        assert isinstance(result, ir.TileType)
        view = result.get_effective_tile_view()
    else:
        source_type = ir.TensorType(
            [16, 32], pl.fp4e2m1x2, None, ir.TensorView([32, 1], ir.TensorLayout.ND, valid_shape=[8, 24])
        )
        source = ir.Var("source", source_type, span)
        result = ir.op.tensor.cast(source, pl.FP8E4M3FN).type
        assert isinstance(result, ir.TensorType)
        view = result.tensor_view
        assert view is not None
    assert [value.value for value in result.shape if isinstance(value, ir.ConstInt)] == [16, 64]
    assert [value.value for value in view.valid_shape if isinstance(value, ir.ConstInt)] == [8, 48]
    assert [value.value for value in view.stride if isinstance(value, ir.ConstInt)] == [64, 1]


@pytest.mark.parametrize("placed", [False, True])
def test_late_cast_bridge_preserves_placement_final_type_and_attrs(placed):
    """A split carrier row needs fresh Vec bridge storage after memory inference."""
    span = ir.Span.unknown()
    memory = ir.MemorySpace.Vec if placed else None
    source = ir.Var("source", ir.TileType([16, 16], pl.fp4e2m1x2, None, None, memory), span)
    tmp = ir.Var("tmp", ir.TileType([1, 64], pl.INT8, None, None, memory), span)
    # An absent view is the canonical implicit layout; legalization must
    # preserve it even when the final hop's inferred input has explicit stride.
    result_type = ir.TileType([16, 32], pl.FP8E4M3FN, None, None, memory)
    attrs = {"arg_directions": [ir.ArgDirection.Input, ir.ArgDirection.InOut], "test_metadata": "retained"}
    kwargs = {"target_type": pl.FP8E4M3FN, "mode": 1, "saturation_mode": 0}
    call = ir.Call(ir.get_op("tile.cast"), [source, tmp], kwargs, attrs, result_type, span)
    result = ir.Var("result", result_type, span)
    before = ir.Program(
        [
            ir.Function(
                "kernel",
                [source, tmp],
                [result_type],
                ir.SeqStmts([ir.AssignStmt(result, call, span), ir.ReturnStmt([result], span)], span),
                span,
                ir.FunctionType.AIV,
            )
        ],
        "late_bridge",
        span,
    )
    after = _run(before, BackendType.Ascend950)
    assert _cast_pairs(after) == [
        ("fp4e2m1x2", "bfloat16"),
        ("bfloat16", "fp32"),
        ("fp32", "fp8e4m3fn"),
    ]

    class Collector(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.assignments = []

        def visit_assign_stmt(self, op):
            if isinstance(op.value, ir.Call) and op.value.op.name == _TILE_CAST:
                self.assignments.append(op)
            super().visit_assign_stmt(op)

    collector = Collector()
    collector.visit_program(after)
    assert len(collector.assignments) == 3
    # A bridge hop lands in Vec either way: the pass copies a resolved placement
    # onto it, and ``tile.cast``'s registered output memory supplies Vec when the
    # final result is still unplaced (early legalization).
    bridge_memory = memory or ir.MemorySpace.Vec
    for assignment in collector.assignments[:-1]:
        assert isinstance(assignment.var.type, ir.TileType)
        assert isinstance(assignment.value.type, ir.TileType)
        assert assignment.var.type.memory_space == bridge_memory
        assert assignment.value.type.memory_space == bridge_memory
        assert assignment.var.type.memref is None and assignment.value.type.memref is None
        assert "test_metadata" not in assignment.value.attrs
        assert "arg_directions" not in assignment.value.attrs
    final = collector.assignments[-1]
    assert final.var is result
    assert final.value.type is result_type
    assert final.value.attrs == attrs
    assert final.value.kwargs == kwargs
    assert final.value.args[1] is tmp
    _assert_recorded_cast_roundtrip(after)


def test_native_direct_cast_empty_replay_is_not_expanded():
    """The late legality check leaves the all-zero replay for no-data emission."""
    span = ir.Span.unknown()
    view = ir.TileView(valid_shape=[0, 0])
    source = ir.Var("source", ir.TileType([16, 32], pl.fp4e2m1x2, None, view, ir.MemorySpace.Vec), span)
    result_type = ir.TileType([16, 64], pl.FP8E4M3FN, None, view, ir.MemorySpace.Vec)
    call = ir.Call(
        ir.get_op("tile.cast"), [source], {"target_type": pl.FP8E4M3FN, "mode": 2}, result_type, span
    )
    result = ir.Var("result", result_type, span)
    before = ir.Program(
        [
            ir.Function(
                "kernel",
                [source],
                [result_type],
                ir.SeqStmts([ir.AssignStmt(result, call, span), ir.ReturnStmt([result], span)], span),
                span,
                ir.FunctionType.AIV,
            )
        ],
        "empty_replay",
        span,
    )
    after = _run(before, BackendType.Ascend950)
    ir.assert_structural_equal(after, before)
    with pytest.raises(ValueError, match="no native cast path"):
        _run(before, BackendType.Ascend910)


def test_native_conditional_edge_is_not_a_lossy_bf16_to_fp8_shortcut():
    """Both BF16->FP4 and FP4->FP8 can be native, but their composition loses values."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, source: pl.Tensor[[16, 64], pl.BF16], output: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]]
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            value = pl.load(source, [0, 0], [16, 64])
            result = pl.cast(value, pl.FP8E4M3FN)
            return pl.store(result, [0, 0], output)

    after = _run(Before, BackendType.Ascend950)
    assert _cast_pairs(after) == [("bfloat16", "fp32"), ("fp32", "fp8e4m3fn")]


# ---------------------------------------------------------------------------
# NZ destination layout (blayout=col_major / slayout=row_major, fractal=512)
# ---------------------------------------------------------------------------


def _nz_cast_program(rows=16, packed_cols=32, dtype=pl.fp4e2m1x2, target=pl.FP8E4M3FN):
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            source: pl.Tensor[[rows, packed_cols], dtype],
            output: pl.Out[pl.Tensor[[rows, 2 * packed_cols], target]],
        ) -> pl.Tensor[[rows, 2 * packed_cols], target]:
            packed = pl.load(source, [0, 0], [rows, packed_cols])
            decoded = pl.tile.cast(
                packed,
                target,
                blayout=pl.TileLayout.col_major,
                slayout=pl.TileLayout.row_major,
            )
            return pl.store(decoded, [0, 0], output)

    return Before


def test_a5_nz_destination_cast_stays_one_native_cast():
    """An NZ destination is a native A5 tcvt, so legalization leaves it alone.

    This is the whole point of the attribute: PTOAS v0.72 converts packed FP4
    straight into a standard 512-byte NZ FP8 tile, so the cast must survive as
    one hop with its boxed result type intact rather than being expanded.
    """
    before = _nz_cast_program()
    after = _run(before, BackendType.Ascend950)
    assert _cast_pairs(after) == [("fp4e2m1x2", "fp8e4m3fn")]
    ir.assert_structural_equal(after, before)


def test_nz_destination_cast_survives_print_parse():
    """The boxed result and both layout kwargs round-trip through the printer.

    `TileLayout` kwargs already had printer, parser and serializer support --
    they are how `tile.move` spells the same thing -- so the cast needs no new
    round-trip machinery. What is asserted here is stronger than survival: the
    printer emits the ORDINARY DSL spelling (not the `pl._dump.call` escape
    hatch that a cast needs when its recorded type cannot be re-derived), which
    means re-deducing from the two kwargs reproduces the boxed result type
    exactly. If that ever stops holding, this fails rather than silently
    degrading to the dump form.
    """
    before = _nz_cast_program()
    after = _run(before, BackendType.Ascend950)
    printed = python_print(after, format=False)
    assert "blayout=pl.TileLayout.col_major" in printed
    assert "slayout=pl.TileLayout.row_major" in printed
    assert 'pl._dump.call("tile.cast"' not in printed
    ir.assert_structural_equal(after, pl.parse_program(printed), enable_auto_mapping=True)


def test_nz_destination_is_rejected_for_a_dtype_pair_without_a_native_edge():
    """An unsupported boxing must fail loudly, not assemble and misaddress.

    PTOAS's `tcvt` verifier inspects the destination layout ONLY for the packed
    FP4-to-FP8 pair; for every other pair it accepts a boxed destination without
    complaint. An unconstrained dtype edge in the backend table likewise states
    no layout opinion, so without this gate an FP32-to-FP16 NZ request would
    lower to a `pto.tcvt` whose destination the ISA never promised to write.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            source: pl.Tensor[[16, 32], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 32], pl.FP16]],
        ) -> pl.Tensor[[16, 32], pl.FP16]:
            loaded = pl.load(source, [0, 0], [16, 32])
            boxed = pl.tile.cast(
                loaded,
                pl.FP16,
                blayout=pl.TileLayout.col_major,
                slayout=pl.TileLayout.row_major,
            )
            return pl.store(boxed, [0, 0], output)

    with pytest.raises(ValueError, match="no native pto.tcvt that writes it"):
        _run(Before, BackendType.Ascend950)


def test_nz_destination_is_rejected_on_a2a3():
    """A2/A3 has no packed FP4 tcvt at all, so the NZ request fails there too."""
    before = _nz_cast_program()
    with pytest.raises(ValueError, match="no native pto.tcvt that writes it"):
        _run(before, BackendType.Ascend910)


def test_an_incidentally_boxed_cast_is_not_caught_by_the_nz_gate():
    """The gate keys on the REQUEST, not on the result being boxed.

    A cast that merely happens to carry a boxed tile (no layout kwargs) keeps
    the ordinary backend layout-repair path, so narrowing the gate to explicit
    requests is what keeps this working.
    """
    span = ir.Span.unknown()
    boxed_view = ir.TileView(blayout=ir.TileLayout.col_major, slayout=ir.TileLayout.row_major)
    source = ir.Var("source", ir.TileType([16, 32], pl.FP32, tile_view=boxed_view), span)
    call = ir.create_op_call("tile.cast", [source], {"target_type": pl.FP16, "mode": 2}, span)
    assert "blayout" not in dict(call.kwargs)
    result = call.type
    assert isinstance(result, ir.TileType)
    # Inherited, not requested: the gate must ignore it.
    assert result.tile_view is not None
    assert result.tile_view.slayout == ir.TileLayout.row_major


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
