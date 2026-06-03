# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the AssignTypeSymmetry property verifier (#1285).

The verifier asserts that every ``AssignStmt(var, value)`` satisfies
``structural_equal(var.type, value.type)`` — covering dtype, shape, memref,
memory_space, and tile_view. It catches passes that mutate one side of an
assignment without keeping the other in sync (e.g. #1262, where
``InferTileMemorySpace`` wrote ``Mem.Acc`` onto a Var whose producing
``tile.full`` Call still declared ``Mem.Vec``).
"""

import pypto.language as pl
import pytest
from pypto import ir, passes

DataType = ir.DataType
_SPAN = ir.Span.unknown()


def _verify(program: ir.Program) -> list:
    """Run the AssignTypeSymmetry verifier and return its diagnostics."""
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.AssignTypeSymmetry)
    return passes.PropertyVerifierRegistry.verify(props, program)


def _one_assign_program(var_type: ir.Type, value_type: ir.Type) -> ir.Program:
    """Build a minimal InCore function whose body is a single ``dst = src``.

    ``src`` is a function parameter typed ``value_type``; ``dst`` is the LHS Var
    typed ``var_type``. Using a Var as the RHS value keeps the construction free
    of op-deducer side effects while still exercising ``value_->GetType()``.
    """
    src = ir.Var("src", value_type, _SPAN)
    dst = ir.Var("dst", var_type, _SPAN)
    body = ir.SeqStmts([ir.AssignStmt(dst, src, _SPAN)], _SPAN)
    func = ir.Function("main_incore_0", [src], [var_type], body, _SPAN, ir.FunctionType.InCore)
    return ir.Program([func], "AssignSymTest", _SPAN)


def _tile(shape, dtype=None, memref=None, tile_view=None, memory_space=None) -> ir.TileType:
    return ir.TileType(shape, dtype or DataType.FP32, memref, tile_view, memory_space or ir.MemorySpace.Vec)


# --------------------------------------------------------------------------- #
# Positive cases — symmetric assignments verify clean.
# --------------------------------------------------------------------------- #


def test_identical_tile_types_pass():
    t = _tile([16, 64], memory_space=ir.MemorySpace.Vec)
    # Two structurally-equal but distinct TileType objects (not the same pointer).
    t2 = _tile([16, 64], memory_space=ir.MemorySpace.Vec)
    assert len(_verify(_one_assign_program(t, t2))) == 0


def test_parsed_clean_program_passes():
    """A well-formed DSL program (var.type == deduced value.type) verifies clean."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
        ) -> pl.Tensor[[16, 128], pl.BF16]:
            x_tile: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
            )
            out_0: pl.Tensor[[16, 128], pl.BF16] = pl.store(x_tile, [0, 0], out_0)
            return out_0

    assert len(_verify(Program)) == 0


# --------------------------------------------------------------------------- #
# Negative cases — each field divergence is detected.
# --------------------------------------------------------------------------- #


def test_memory_space_mismatch_detected():
    """The #1262 pattern: LHS Var is Acc, RHS is Vec."""
    acc = _tile([16, 64], memory_space=ir.MemorySpace.Acc)
    vec = _tile([16, 64], memory_space=ir.MemorySpace.Vec)
    diags = _verify(_one_assign_program(acc, vec))
    assert len(diags) == 1
    assert diags[0].rule_name == "AssignTypeSymmetry"
    assert diags[0].severity == passes.DiagnosticSeverity.Error
    assert "memory_space" in diags[0].message


def test_dtype_mismatch_detected():
    fp32 = _tile([16, 64], dtype=DataType.FP32)
    bf16 = _tile([16, 64], dtype=DataType.BF16)
    diags = _verify(_one_assign_program(fp32, bf16))
    assert len(diags) == 1
    assert diags[0].rule_name == "AssignTypeSymmetry"


def test_shape_mismatch_detected():
    a = _tile([16, 64])
    b = _tile([16, 32])
    diags = _verify(_one_assign_program(a, b))
    assert len(diags) == 1
    assert diags[0].rule_name == "AssignTypeSymmetry"


def test_memref_difference_is_tolerated_by_design():
    """A MemRef difference is NOT flagged — by design.

    ``structural_equal`` (the IR's own type-equality contract, used by the
    roundtrip verifier) deliberately excludes ``memref_`` from type comparison:
    a MemRef is an allocation detail bound to the Var, not part of the value's
    structural type. The verifier reuses ``structural_equal``, so MemRef
    asymmetry — which legitimately arises after ``InitMemRef`` annotates Vars
    while transient producer results stay unbound — is intentionally out of
    scope. MemRef correctness is governed by ``HasMemRefs`` / ``AllocatedMemoryAddr``.
    """
    with_ref = _tile([16, 64], memref=ir.MemRef("mem_vec_0", 0, 256), memory_space=ir.MemorySpace.Vec)
    without_ref = _tile([16, 64], memref=None, memory_space=ir.MemorySpace.Vec)
    assert ir.structural_equal(with_ref, without_ref)  # pin the underlying contract
    assert len(_verify(_one_assign_program(with_ref, without_ref))) == 0


def test_tile_view_mismatch_detected():
    """An explicit non-implicit tile_view vs the canonical (None) form."""
    # Implicit Vec view is row_major / none_box, so col_major / row_major is
    # non-implicit and survives the TileType constructor's canonicalization.
    explicit_view = ir.TileView(
        valid_shape=[ir.ConstInt(16, DataType.INDEX, _SPAN), ir.ConstInt(64, DataType.INDEX, _SPAN)],
        blayout=ir.TileLayout.col_major,
        slayout=ir.TileLayout.row_major,
        fractal=512,
    )
    with_view = _tile([16, 64], tile_view=explicit_view, memory_space=ir.MemorySpace.Vec)
    assert with_view.tile_view is not None  # guard: view actually survived
    plain = _tile([16, 64], tile_view=None, memory_space=ir.MemorySpace.Vec)
    diags = _verify(_one_assign_program(with_view, plain))
    assert len(diags) == 1
    assert diags[0].rule_name == "AssignTypeSymmetry"


def test_tuple_type_assignment():
    """Tuple-typed assignment: symmetric passes, asymmetric is detected."""
    a = _tile([16, 64], memory_space=ir.MemorySpace.Vec)
    b = _tile([16, 64], memory_space=ir.MemorySpace.Vec)
    tup_ok_lhs = ir.TupleType([a, b])
    tup_ok_rhs = ir.TupleType([_tile([16, 64]), _tile([16, 64])])
    assert len(_verify(_one_assign_program(tup_ok_lhs, tup_ok_rhs))) == 0

    tup_bad_rhs = ir.TupleType([_tile([16, 64]), _tile([16, 64], memory_space=ir.MemorySpace.Acc)])
    diags = _verify(_one_assign_program(ir.TupleType([a, b]), tup_bad_rhs))
    assert len(diags) == 1
    assert diags[0].rule_name == "AssignTypeSymmetry"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
