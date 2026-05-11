# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the LowerCompositeOps pass.

The LowerCompositeOps pass decomposes composite tile ops into primitive
arithmetic tile ops. Today it covers ``tile.sin`` / ``tile.cos`` (Cody-Waite
range reduction + degree-9 odd Horner polynomial). The decomposition uses
only ``tile.muls``, ``tile.adds``, ``tile.add``, ``tile.sub``, ``tile.mul``
and ``tile.cast`` — no sin/cos remain after the pass.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.pypto_core import passes as _core_passes


@pytest.fixture(autouse=True)
def pass_verification_context():
    """Override the global roundtrip-verification fixture for this module.

    The parser stores ``ConstFloat.value_`` as a Python ``float`` (FP64) without
    snapping to the IR's ``FP32`` dtype, so the FP32-representable Cody-Waite
    constants emitted by ``LowerCompositeOps`` (e.g. ``1/pi`` = ``0.31830988732818603515625``)
    cannot round-trip bit-exactly through print → parse → ``assert_structural_equal``.
    The C++ field-based equality compares the raw ``double`` values, which differ
    after the lossy text trip even though both represent the same FP32 number.

    Falling back to ``BEFORE_AND_AFTER`` keeps property verification on while
    skipping the roundtrip check that depends on a print/parse-side fix.
    """
    instruments: list[_core_passes.PassInstrument] = [
        _core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)
    ]
    with _core_passes.PassContext(instruments):
        yield


# Primitive tile ops the decomposition is allowed to emit (besides framework
# infrastructure ops like tile.load / tile.store / tile.move that wrap the
# decomposed body).
_DECOMP_PRIMITIVES = {
    "tile.muls",
    "tile.adds",
    "tile.add",
    "tile.sub",
    "tile.mul",
    "tile.cast",
}


class _OpNameCollector(ir.IRVisitor):
    """Walk the IR and record the ``op.name`` of every Call encountered."""

    def __init__(self) -> None:
        super().__init__()
        self.op_names: list[str] = []

    def visit_call(self, op: ir.Call) -> None:
        self.op_names.append(op.op.name)
        super().visit_call(op)


def _collect_op_names(prog) -> list[str]:
    collector = _OpNameCollector()
    collector.visit_program(prog)
    return collector.op_names


def test_lower_composite_ops_pass_factory_exists():
    """The factory returns a Pass instance with the expected name."""
    p = passes.lower_composite_ops()
    assert p is not None
    assert p.get_name() == "LowerCompositeOps"


def test_lower_composite_ops_noop_on_no_trig():
    """Pass must leave programs without sin/cos unchanged."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            x_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
            y_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.exp(x_tile)
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
            return out_0

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
            r: pl.Tensor[[16, 16], pl.FP32] = self.main_incore_0(x, out_0)
            return r

    After = passes.lower_composite_ops()(Before)
    ir.assert_structural_equal(After, Before)


def test_sin_is_decomposed_to_primitives():
    """``tile.sin`` is removed and only allowed primitives appear in its place."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            x_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
            y_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.sin(x_tile)
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
            return out_0

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
            r: pl.Tensor[[16, 16], pl.FP32] = self.main_incore_0(x, out_0)
            return r

    after = passes.lower_composite_ops()(Prog)
    op_names = set(_collect_op_names(after))

    # The lowering must remove tile.sin entirely.
    assert "tile.sin" not in op_names

    # Every emitted decomposition op must come from the allowed primitive set.
    # Framework ops (tile.load/tile.store, tensor.create, the InCore main
    # function call) are filtered explicitly so an unexpected new op surfaces
    # as a test failure rather than being silently allowed.
    framework_ops = {"tile.load", "tile.store", "tensor.create", "main_incore_0"}
    leftover = op_names - _DECOMP_PRIMITIVES - framework_ops
    assert not leftover, f"Unexpected ops after lowering: {sorted(leftover)}"

    # Sanity: the decomposition actually emitted primitives (not just deleted
    # the call).
    assert _DECOMP_PRIMITIVES & op_names, "lowering produced no primitive ops"


def test_cos_is_decomposed_to_primitives():
    """``tile.cos`` is removed and only allowed primitives appear in its place."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            x_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
            y_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.cos(x_tile)
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
            return out_0

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
            r: pl.Tensor[[16, 16], pl.FP32] = self.main_incore_0(x, out_0)
            return r

    after = passes.lower_composite_ops()(Prog)
    op_names = set(_collect_op_names(after))

    assert "tile.cos" not in op_names

    framework_ops = {"tile.load", "tile.store", "tensor.create", "main_incore_0"}
    leftover = op_names - _DECOMP_PRIMITIVES - framework_ops
    assert not leftover, f"Unexpected ops after lowering: {sorted(leftover)}"
    assert _DECOMP_PRIMITIVES & op_names, "lowering produced no primitive ops"


def test_sin_lowering_is_idempotent():
    """Running the pass twice gives the same IR as running it once."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            x_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
            y_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.sin(x_tile)
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
            return out_0

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
            r: pl.Tensor[[16, 16], pl.FP32] = self.main_incore_0(x, out_0)
            return r

    once = passes.lower_composite_ops()(Prog)
    twice = passes.lower_composite_ops()(once)
    ir.assert_structural_equal(twice, once)


def test_cos_lowering_is_idempotent():
    """Running the pass twice on a cos program gives the same IR as once."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            x_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
            y_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.cos(x_tile)
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
            return out_0

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
            r: pl.Tensor[[16, 16], pl.FP32] = self.main_incore_0(x, out_0)
            return r

    once = passes.lower_composite_ops()(Prog)
    twice = passes.lower_composite_ops()(once)
    ir.assert_structural_equal(twice, once)


def test_both_sin_and_cos_in_same_function():
    """Verify sin and cos lowering don't interfere when both appear in one function."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            x_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
            a: pl.Tile[[16, 16], pl.FP32] = pl.tile.sin(x_tile)
            b: pl.Tile[[16, 16], pl.FP32] = pl.tile.cos(x_tile)
            y_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.add(a, b)
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
            return out_0

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
            r: pl.Tensor[[16, 16], pl.FP32] = self.main_incore_0(x, out_0)
            return r

    after = passes.lower_composite_ops()(Prog)
    op_names = set(_collect_op_names(after))

    # Both sin and cos must be removed by the lowering.
    assert "tile.sin" not in op_names
    assert "tile.cos" not in op_names

    # Every emitted op must be either an allowed primitive or framework op.
    framework_ops = {"tile.load", "tile.store", "tensor.create", "main_incore_0"}
    leftover = op_names - _DECOMP_PRIMITIVES - framework_ops
    assert not leftover, f"Unexpected ops after lowering: {sorted(leftover)}"

    # Sanity: the decomposition actually emitted primitives for both sin and cos.
    assert _DECOMP_PRIMITIVES & op_names, "lowering produced no primitive ops"


def test_sin_in_return_stmt_is_decomposed():
    """A ``tile.sin`` Call placed directly inside ``ReturnStmt::value_`` (i.e.
    not pre-bound to an AssignStmt — the shape pre-SSA / standalone callers can
    surface) must still be decomposed by the pass.

    SSA-form programs never produce this shape (every Call is bound to an
    AssignStmt), so the test constructs the IR programmatically via the IR
    builder API to exercise the ``VisitStmt_(ReturnStmtPtr)`` override.
    """
    span = ir.Span.unknown()
    tile_type = ir.TileType([16, 16], ir.DataType.FP32)

    x_param = ir.Var("x", tile_type, span)
    sin_call = ir.create_op_call("tile.sin", [x_param], {}, span)
    body = ir.ReturnStmt([sin_call], span)
    func = ir.Function("trig_return", [x_param], [tile_type], body, span, ir.FunctionType.InCore)
    prog = ir.Program([func], "test_program", span)

    after = passes.lower_composite_ops()(prog)
    op_names = set(_collect_op_names(after))

    # The trig op embedded directly in ReturnStmt must be lowered.
    assert "tile.sin" not in op_names

    # Decomposition primitives must appear in the lowered IR.
    assert _DECOMP_PRIMITIVES & op_names, "lowering produced no primitive ops"


def test_cos_in_return_stmt_is_decomposed():
    """Mirror of ``test_sin_in_return_stmt_is_decomposed`` for ``tile.cos``."""
    span = ir.Span.unknown()
    tile_type = ir.TileType([16, 16], ir.DataType.FP32)

    x_param = ir.Var("x", tile_type, span)
    cos_call = ir.create_op_call("tile.cos", [x_param], {}, span)
    body = ir.ReturnStmt([cos_call], span)
    func = ir.Function("trig_return", [x_param], [tile_type], body, span, ir.FunctionType.InCore)
    prog = ir.Program([func], "test_program", span)

    after = passes.lower_composite_ops()(prog)
    op_names = set(_collect_op_names(after))

    assert "tile.cos" not in op_names
    assert _DECOMP_PRIMITIVES & op_names, "lowering produced no primitive ops"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
