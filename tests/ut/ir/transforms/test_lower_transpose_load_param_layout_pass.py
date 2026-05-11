# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for LowerTransposeLoadParamLayout pass (RFC #1300 P6).

The pass promotes each InCore parameter loaded via ``tile.load(transpose=True)``
to canonical-form DN (RFC §3.3 + §4.2): the trailing shape pair is swapped,
the layout tag becomes DN, the body's ``tile.load`` call swaps its
``offsets`` / ``shapes`` / ``valid_shapes`` trailing pair and drops the
``transpose=True`` kwarg, and every non-InCore call site bridges its arg
through ``tensor.as_layout(arg, DN)``.

``tensor.as_layout`` is internal-only and not exposed via ``pypto.language``,
so we cannot write the post-pass IR as ``@pl.program``. Instead we drive the
pass with ``@pl.program`` ``Before`` programs and assert the resulting IR
shape programmatically.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


def _as_tensor_type(ty: ir.Type) -> ir.TensorType:
    """Narrow ``ty`` to ``TensorType`` for type-checker awareness."""
    assert isinstance(ty, ir.TensorType), f"expected TensorType, got {type(ty).__name__}"
    return ty


def _find_function(program, name):
    """Return the Function with the given name from a Program."""
    for _gv, func in program.functions.items():
        if func.name == name:
            return func
    raise AssertionError(f"function {name!r} not found in program")


def _iter_stmts(stmt):
    """Yield every statement under ``stmt`` (depth-first)."""
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            yield from _iter_stmts(s)
    else:
        yield stmt
        for attr in ("body", "then_body", "else_body"):
            inner = getattr(stmt, attr, None)
            if inner is not None:
                yield from _iter_stmts(inner)


def _find_tile_loads(func):
    """Return every ``tile.load`` Call expression in ``func.body``."""
    loads = []
    for stmt in _iter_stmts(func.body):
        value = getattr(stmt, "value", None)
        if isinstance(value, ir.Call) and value.op is not None and value.op.name == "tile.load":
            loads.append(value)
    return loads


def _find_calls_to(func, callee_name):
    """Return every Call in ``func.body`` whose op is GlobalVar(callee_name)."""
    calls = []
    for stmt in _iter_stmts(func.body):
        value = getattr(stmt, "value", None)
        if isinstance(value, ir.Call) and isinstance(value.op, ir.GlobalVar) and value.op.name == callee_name:
            calls.append(value)
    return calls


def _find_assign_rhs(func, var):
    """Return the RHS expression of the ``AssignStmt`` that defines ``var``."""
    for stmt in _iter_stmts(func.body):
        if isinstance(stmt, ir.AssignStmt) and stmt.var is var:
            return stmt.value
    raise AssertionError(f"no AssignStmt defines var {var.name_hint}")


def _shape_dims(ty):
    """Return ConstInt shape dims as ints (rejects symbolic dims for test fixtures)."""
    tensor_type = _as_tensor_type(ty)
    out = []
    for dim in tensor_type.shape:
        assert isinstance(dim, ir.ConstInt), f"non-constant dim {dim} in test fixture"
        out.append(dim.value)
    return out


def _transpose_kwarg(call):
    """Return the value of the ``transpose`` kwarg, or ``None`` if absent."""
    return call.kwargs.get("transpose")


class TestBTransposePromotesParam:
    """``C = A @ B^T`` with B loaded via ``transpose=True`` — param promoted to DN."""

    def test_btranspose_basic(self):
        M, K, N = 64, 128, 32

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[N, K], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [N, K], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32], b: pl.Tensor[[N, K], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        After = passes.lower_transpose_load_param_layout()(Before)

        incore = _find_function(After, "matmul_incore")
        b_type = _as_tensor_type(incore.params[1].type)
        assert _shape_dims(b_type) == [K, N], f"b param shape: {_shape_dims(b_type)}"
        assert b_type.tensor_view is not None
        assert b_type.tensor_view.layout == ir.TensorLayout.DN

        a_type = _as_tensor_type(incore.params[0].type)
        assert _shape_dims(a_type) == [M, K]
        assert a_type.tensor_view is None

        loads_by_src = {}
        for ld in _find_tile_loads(incore):
            assert isinstance(ld.args[0], ir.Var)
            loads_by_src[ld.args[0].name_hint] = ld

        load_b = loads_by_src["b"]
        shapes_arg = load_b.args[2]
        assert isinstance(shapes_arg, ir.MakeTuple)
        shape_vals = [el.value for el in shapes_arg.elements if isinstance(el, ir.ConstInt)]
        assert shape_vals == [K, N], f"tile.load(b) shapes: {shape_vals}"
        assert _transpose_kwarg(load_b) is False, "tile.load(b) transpose kwarg must be False after P6"

        load_a = loads_by_src["a"]
        shape_vals_a = [el.value for el in load_a.args[2].elements if isinstance(el, ir.ConstInt)]
        assert shape_vals_a == [M, K]

        orch = _find_function(After, "orchestrator")
        calls = _find_calls_to(orch, "matmul_incore")
        assert len(calls) == 1
        # `b` is bridged via an SSA AssignStmt: the call arg is a Var bound to
        # a separately-emitted ``tensor.as_layout(orig_b, DN)`` Call.
        b_arg = calls[0].args[1]
        assert isinstance(b_arg, ir.Var)
        b_def_rhs = _find_assign_rhs(orch, b_arg)
        assert isinstance(b_def_rhs, ir.Call) and b_def_rhs.op is not None
        assert b_def_rhs.op.name == "tensor.as_layout", (
            f"orch must wrap b in tensor.as_layout, got {b_def_rhs.op.name if b_def_rhs.op else None}"
        )
        bridged_t = _as_tensor_type(b_def_rhs.type)
        assert _shape_dims(bridged_t) == [K, N]
        assert bridged_t.tensor_view is not None
        assert bridged_t.tensor_view.layout == ir.TensorLayout.DN

    def test_btranspose_non_square(self):
        M, K, N = 128, 64, 32

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[N, K], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [N, K], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32], b: pl.Tensor[[N, K], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        After = passes.lower_transpose_load_param_layout()(Before)
        incore = _find_function(After, "matmul_incore")
        b_type = _as_tensor_type(incore.params[1].type)
        assert _shape_dims(b_type) == [K, N]
        assert b_type.tensor_view is not None
        assert b_type.tensor_view.layout == ir.TensorLayout.DN


class TestATransposePromotesParam:
    """``C = A^T @ B`` — A param promoted to canonical DN."""

    def test_atranspose_basic(self):
        M, K, N = 64, 128, 32

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[K, M], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [K, M], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[K, M], pl.FP32], b: pl.Tensor[[K, N], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        After = passes.lower_transpose_load_param_layout()(Before)
        incore = _find_function(After, "matmul_incore")
        a_t = _as_tensor_type(incore.params[0].type)
        assert _shape_dims(a_t) == [M, K]
        assert a_t.tensor_view is not None
        assert a_t.tensor_view.layout == ir.TensorLayout.DN

        b_t = _as_tensor_type(incore.params[1].type)
        assert _shape_dims(b_t) == [K, N]
        assert b_t.tensor_view is None

        loads = {ld.args[0].name_hint: ld for ld in _find_tile_loads(incore)}
        load_a = loads["a"]
        shape_vals = [el.value for el in load_a.args[2].elements if isinstance(el, ir.ConstInt)]
        assert shape_vals == [M, K]
        assert _transpose_kwarg(load_a) is False

        orch = _find_function(After, "orchestrator")
        call = _find_calls_to(orch, "matmul_incore")[0]
        # `a` is bridged via tensor.as_layout. After P6's SSA refactor (PR
        # review fix), the bridge is bound to a fresh Var by a preceding
        # AssignStmt, so the call arg is a Var, not the inline Call. Look up
        # the binding's RHS.
        a_arg = call.args[0]
        assert isinstance(a_arg, ir.Var)
        a_def_rhs = _find_assign_rhs(orch, a_arg)
        assert isinstance(a_def_rhs, ir.Call) and a_def_rhs.op is not None
        assert a_def_rhs.op.name == "tensor.as_layout"
        # `b` is not promoted, so its arg is the raw Var (no bridge).
        assert isinstance(call.args[1], ir.Var)


class TestABTransposePromotesBothParams:
    """``C = A^T @ B^T`` — both params promoted, both call args wrapped."""

    def test_abtranspose_basic(self):
        M, K, N = 64, 128, 32

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[K, M], pl.FP32],
                b: pl.Tensor[[N, K], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [K, M], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_b = pl.load(b, [0, 0], [N, K], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[K, M], pl.FP32], b: pl.Tensor[[N, K], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        After = passes.lower_transpose_load_param_layout()(Before)
        incore = _find_function(After, "matmul_incore")
        a_t = _as_tensor_type(incore.params[0].type)
        b_t = _as_tensor_type(incore.params[1].type)
        assert _shape_dims(a_t) == [M, K]
        assert a_t.tensor_view is not None and a_t.tensor_view.layout == ir.TensorLayout.DN
        assert _shape_dims(b_t) == [K, N]
        assert b_t.tensor_view is not None and b_t.tensor_view.layout == ir.TensorLayout.DN

        orch = _find_function(After, "orchestrator")
        call = _find_calls_to(orch, "matmul_incore")[0]
        # Both promoted args are bridged via SSA AssignStmts.
        for slot in (0, 1):
            arg = call.args[slot]
            assert isinstance(arg, ir.Var)
            rhs = _find_assign_rhs(orch, arg)
            assert isinstance(rhs, ir.Call) and rhs.op is not None
            assert rhs.op.name == "tensor.as_layout"


class TestNoOpCases:
    """Pass is a no-op when no parameter needs promotion."""

    def test_no_transpose_unchanged(self):
        M, K, N = 64, 128, 32

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32], b: pl.Tensor[[K, N], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        After = passes.lower_transpose_load_param_layout()(Before)
        ir.assert_structural_equal(After, Before)

    def test_already_dn_param_idempotent(self):
        """A param already carrying the DN tag short-circuits — IR unchanged.

        Mirrors the pre-P6 mid-state where the param has been DN-tagged but the
        body's tile.load still has ``transpose=True`` (idempotent re-run of the
        legacy pass form). The pass detects ``layout == DN`` on the param,
        ``continue``s past the promotion, and leaves the body untouched.
        """
        M, K, N = 64, 128, 32

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[N, K], pl.FP32, pl.DN],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [N, K], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32], b: pl.Tensor[[N, K], pl.FP32, pl.DN]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        After = passes.lower_transpose_load_param_layout()(Before)
        ir.assert_structural_equal(After, Before)


class TestMixedUseRejected:
    """A param loaded with both transpose=True and transpose=False is rejected."""

    def test_mixed_transpose_modes_rejected(self):
        M, K, N = 64, 128, 32

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[N, K], pl.FP32],
                b: pl.Tensor[[N, K], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [N, K], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_b = pl.load(a, [0, 0], [N, K], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[N, K], pl.FP32], b: pl.Tensor[[N, K], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        with pytest.raises(Exception, match="only one mode is supported per InCore parameter"):
            passes.lower_transpose_load_param_layout()(Before)


class TestPartialLoadPromotion:
    """A param with a partial-window transpose load: param shape swap is based on the
    full TensorType shape, not the load window."""

    def test_partial_load_square_tensor(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[64, 128], pl.BF16],
                key_cache: pl.Tensor[[128, 128], pl.BF16],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
                tile_k = pl.load(
                    key_cache, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                tile_a_l0 = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_k_l0 = pl.move(tile_k, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0, tile_k_l0)
                out_store = pl.store(tile_c, [0, 0], out)
                return out_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[64, 128], pl.BF16],
                key_cache: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
                out_result = self.kernel(a, key_cache, out)
                return out_result

        After = passes.lower_transpose_load_param_layout()(Before)
        incore = _find_function(After, "kernel")
        kc_t = _as_tensor_type(incore.params[1].type)
        assert _shape_dims(kc_t) == [128, 128]
        assert kc_t.tensor_view is not None
        assert kc_t.tensor_view.layout == ir.TensorLayout.DN

        loads = {ld.args[0].name_hint: ld for ld in _find_tile_loads(incore)}
        load_k = loads["key_cache"]
        shape_vals = [el.value for el in load_k.args[2].elements if isinstance(el, ir.ConstInt)]
        assert shape_vals == [128, 64]
        assert _transpose_kwarg(load_k) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
