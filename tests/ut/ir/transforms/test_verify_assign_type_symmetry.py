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
``structural_equal(var.type, value.type)`` — covering dtype, shape,
memory_space, and tile_view/tensor_view. ``memref`` is intentionally excluded
(``structural_equal`` treats it as an allocation detail; see
``test_memref_difference_is_tolerated_by_design``). It catches passes that
mutate one side of an assignment without keeping the other in sync (e.g. #1262,
where ``InferTileMemorySpace`` wrote ``Mem.Acc`` onto a Var whose producing
``tile.full`` Call still declared ``Mem.Vec``).
"""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.language.parser.diagnostics import ParserTypeError

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


def _walk_assigns(stmt: ir.Stmt) -> list:
    """Every AssignStmt reachable from ``stmt``, in source order."""
    found: list = []
    if isinstance(stmt, ir.AssignStmt):
        found.append(stmt)
    for child in (
        stmt.stmts
        if isinstance(stmt, ir.SeqStmts)
        else [stmt.body]
        if isinstance(stmt, (ir.ScopeStmt, ir.ForStmt, ir.WhileStmt))
        else []
    ):
        found.extend(_walk_assigns(child))
    return found


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


def test_annotated_scalar_literal_matches_annotation_dtype():
    """``acc: pl.Scalar[pl.INT64] = 0`` binds an INT64 constant, not an INDEX one.

    A bare int literal parses as the untyped placeholder ``ConstInt(v, INDEX)``.
    The annotation retypes the Var, so the parser must re-stamp the constant to
    match — otherwise the AssignStmt is asymmetric, and the mismatch resurfaces
    far downstream (Simplify propagates the constant into a loop's ``iter_arg``
    init, where TypeCheck compares it against the declared carry dtype).
    """

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, d: pl.Out[pl.Tensor[[4], pl.FP32]]) -> pl.Tensor[[4], pl.FP32]:
            acc: pl.Scalar[pl.INT64] = 0
            for _k in pl.range(4):
                acc = acc + 1
            return d

    assert len(_verify(Program)) == 0

    orch = next(f for f in Program.functions.values() if f.name == "orch")
    body = orch.body
    assert isinstance(body, ir.SeqStmts)
    seed = body.stmts[0]
    assert isinstance(seed, ir.AssignStmt)
    assert isinstance(seed.value, ir.ConstInt)
    seed_type = seed.value.type
    assert isinstance(seed_type, ir.ScalarType)
    assert seed_type.dtype == DataType.INT64


def test_annotated_scalar_index_expression_is_cast_to_the_annotation_dtype():
    """``v: pl.Scalar[pl.INT32] = <INDEX expr>`` binds an INT32 value, not an INDEX one.

    The literal case above is re-stamped; a non-constant RHS cannot be, because
    its dtype comes from its operands — scalar arithmetic normalizes them to
    INDEX, so even ``pl.cast(i, pl.INT32) + 1`` is INDEX-typed. The parser wraps
    it in the cast the annotation asks for.

    Left asymmetric this reaches PTOAS as invalid MLIR (the scalar emitter reads
    the Var's INT32, emits no cast, and feeds the ``index`` SSA value into an
    ``i32`` operand), reported at an SSA number that names neither the variable
    nor the user's line. See #2779.
    """

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def k(self, out: pl.Out[pl.Tensor[[1, 1], pl.INT32]]) -> pl.Tensor[[1, 1], pl.INT32]:
            for i in pl.range(4):
                v: pl.Scalar[pl.INT32] = pl.cast(i, pl.INT32) + 1
                pl.write(out, [0, 0], v)
            return out

    assert len(_verify(Program)) == 0

    k = next(f for f in Program.functions.values() if f.name == "k")
    assign = next(stmt for stmt in _walk_assigns(k.body) if stmt.var.name_hint.startswith("v"))
    assert isinstance(assign.value, ir.Cast)
    assert isinstance(assign.value.type, ir.ScalarType)
    assert assign.value.type.dtype == DataType.INT32
    # The wrapped expression keeps its own INDEX type — only the binding is cast.
    operand_type = assign.value.operand.type
    assert isinstance(operand_type, ir.ScalarType)
    assert operand_type.dtype == DataType.INDEX


@pytest.mark.parametrize("dtype", ["INT32", "INT64", "INDEX"])
@pytest.mark.parametrize("op", ["tile.get_block_idx", "tensor.get_block_idx"])
def test_annotated_index_call_preserves_inferred_type_and_roundtrips(dtype, op):
    """A call keeps its actual return type; the binding converts its result."""
    program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.InCore)
    def k(self, out: pl.Out[pl.Tensor[[1, 1], pl.INT32]]) -> pl.Tensor[[1, 1], pl.INT32]:
        v_index: pl.Scalar[pl.INT32] = 7
        v: pl.Scalar[pl.{dtype}] = pl.{op}()
        pl.write(out, [0, 0], pl.cast(v_index + v, pl.INT32))
        return out
""")
    func = program.get_function("k")
    assert func is not None
    assigns = _walk_assigns(func.body)
    call_assign = next(
        stmt
        for stmt in assigns
        if isinstance(stmt.value, ir.Call) and stmt.value.op.name == ir.get_op(op).name
    )
    assert isinstance(call_assign.value.type, ir.ScalarType)
    assert call_assign.value.type.dtype == DataType.INDEX
    binding = next(stmt for stmt in assigns if stmt.var.name_hint == "v")
    if dtype == "INDEX":
        assert isinstance(binding.value, ir.Call)
    else:
        assert isinstance(binding.value, ir.Cast)
        assert isinstance(binding.value.operand, ir.Var)
        assert binding.value.operand.same_as(call_assign.var)
    assert len(_verify(program)) == 0
    ir.assert_structural_equal(program, pl.parse_program(program.as_python()))


@pytest.mark.parametrize("rhs", ["pl.cast(i, pl.INT32) + 1", "pl.tile.get_block_idx()"])
def test_index_expression_rejects_float_annotation(rhs):
    """An annotation cannot implicitly convert INDEX arithmetic or calls to FP32."""
    with pytest.raises(ParserTypeError, match="dtype fp32 but expression has dtype index"):
        pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.InCore)
    def k(self, i: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.FP32]:
        v: pl.Scalar[pl.FP32] = {rhs}
        return v
""")


def test_annotated_scalar_index_annotation_needs_no_cast():
    """An INDEX annotation over an INDEX RHS is already symmetric — no cast added."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def k(self, out: pl.Out[pl.Tensor[[1, 1], pl.INT32]]) -> pl.Tensor[[1, 1], pl.INT32]:
            for i in pl.range(4):
                v: pl.Scalar[pl.INDEX] = i + 1
                pl.write(out, [0, 0], pl.cast(v, pl.INT32))
            return out

    assert len(_verify(Program)) == 0
    k = next(f for f in Program.functions.values() if f.name == "k")
    assign = next(stmt for stmt in _walk_assigns(k.body) if stmt.var.name_hint.startswith("v"))
    assert isinstance(assign.value, ir.Add)


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


# --------------------------------------------------------------------------- #
# Scope — only a Var with a single defining assignment is checked.
# --------------------------------------------------------------------------- #


def test_rebound_var_is_not_a_violation():
    """A Var defined by two AssignStmts has no single defining value.

    Before ``ConvertToSSA`` a source-level rebind reuses ONE ``Var`` node across
    several assignments, so that Var cannot equal every RHS type at once. The
    asymmetry is inherent to pre-SSA IR, not a defect, so the verifier checks
    only singly-defined Vars — which post-SSA is every Var.
    """
    vec = _tile([16, 64], memory_space=ir.MemorySpace.Vec)
    acc = _tile([16, 64], memory_space=ir.MemorySpace.Acc)
    src_vec = ir.Var("src_vec", vec, _SPAN)
    src_acc = ir.Var("src_acc", acc, _SPAN)
    dst = ir.Var("dst", vec, _SPAN)
    body = ir.SeqStmts([ir.AssignStmt(dst, src_vec, _SPAN), ir.AssignStmt(dst, src_acc, _SPAN)], _SPAN)
    func = ir.Function("main_incore_0", [src_vec, src_acc], [vec], body, _SPAN, ir.FunctionType.InCore)
    # The second assignment IS asymmetric in isolation ...
    assert not ir.structural_equal(dst.type, src_acc.type)
    # ... yet the program verifies clean, because `dst` is rebound.
    assert len(_verify(ir.Program([func], "RebindTest", _SPAN))) == 0


def test_dsl_rebind_tolerated_before_ssa_and_clean_after():
    """The same shape in real DSL: ``t`` is loaded, then rebound by ``pl.add``.

    ``pl.add`` deduces ``Mem.Vec`` while the Var minted at the first assignment
    carries no memory space, so the pre-SSA IR is asymmetric by construction.
    ``ConvertToSSA`` gives each definition its own Var and the asymmetry is
    gone — the invariant the verifier enforces is the post-SSA one.
    """

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV)
        def kern(
            self,
            x: pl.Tensor[[16, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ) -> pl.Tensor[[16, 64], pl.FP32]:
            t = pl.load(x, [0, 0], [16, 64])
            t = pl.add(t, t)
            return pl.store(t, [0, 0], out)

    rebinds = [
        stmt
        for stmt in _walk_assigns(next(f for f in Program.functions.values() if f.name == "kern").body)
        if not ir.structural_equal(stmt.var.type, stmt.value.type)
    ]
    assert len(rebinds) == 1  # pin the asymmetry this test is about

    assert len(_verify(Program)) == 0
    assert len(_verify(passes.convert_to_ssa()(Program))) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
