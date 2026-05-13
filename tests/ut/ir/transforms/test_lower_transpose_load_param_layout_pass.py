# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for LowerTransposeLoadParamLayout pass (RFC #1300 P6).

The pass leaves InCore parameter signatures untouched and instead prepends a
``b_dn = tensor.as_layout(b, layout=DN)`` AssignStmt at the top of the InCore
body for each param ``b`` loaded via ``tile.load(transpose=True)``. Body uses
of ``b`` are substituted with ``b_dn`` (which has the canonical
``[..., b_dim, a_dim] DN`` view per RFC §3.3 + §4.2), and the matching
``tile.load`` calls have their ``offsets`` / ``shapes`` / ``valid_shapes``
trailing pair swapped while ``transpose=True`` is flipped to
``transpose=False``. Non-InCore (orch) call sites are not touched — they pass
their original ND args straight through to the kernel.

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


def _shape_dims(ty):
    """Return ConstInt shape dims as ints (rejects symbolic dims for test fixtures)."""
    tensor_type = _as_tensor_type(ty)
    out = []
    for dim in tensor_type.shape:
        assert isinstance(dim, ir.ConstInt), f"non-constant dim {dim} in test fixture"
        out.append(dim.value)
    return out


def _const_int_values(exprs):
    """Return [e.value for e in exprs] asserting each is a ConstInt — narrows
    the static type so type-checkers don't trip on ``Expr.value``."""
    out = []
    for e in exprs:
        assert isinstance(e, ir.ConstInt), f"expected ConstInt, got {type(e).__name__}"
        out.append(e.value)
    return out


def _transpose_kwarg(call):
    """Return the value of the ``transpose`` kwarg, or ``None`` if absent."""
    return call.kwargs.get("transpose")


def _find_as_layout_binding(func, input_var):
    """Find the body-prepended ``b_dn = tensor.as_layout(input_var, ...)``
    AssignStmt and return ``(lhs_var, rhs_call)``. Asserts that exactly one
    such binding exists in the body.
    """
    matches = []
    for stmt in _iter_stmts(func.body):
        if not isinstance(stmt, ir.AssignStmt):
            continue
        rhs = stmt.value
        if not isinstance(rhs, ir.Call) or rhs.op is None:
            continue
        if rhs.op.name != "tensor.as_layout":
            continue
        if not rhs.args or not isinstance(rhs.args[0], ir.Var):
            continue
        if rhs.args[0] is input_var:
            matches.append((stmt.var, rhs))
    assert len(matches) == 1, (
        f"expected exactly one tensor.as_layout binding for {input_var.name_hint}, found {len(matches)}"
    )
    return matches[0]


def _has_as_layout_for(func, input_var):
    """Return True iff ``func.body`` contains a tensor.as_layout binding whose
    first arg is ``input_var``."""
    for stmt in _iter_stmts(func.body):
        if not isinstance(stmt, ir.AssignStmt):
            continue
        rhs = stmt.value
        if (
            isinstance(rhs, ir.Call)
            and rhs.op is not None
            and rhs.op.name == "tensor.as_layout"
            and rhs.args
            and isinstance(rhs.args[0], ir.Var)
            and rhs.args[0] is input_var
        ):
            return True
    return False


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

        # Param signatures are untouched — ``b`` is still ``[N, K] ND``.
        b_param = incore.params[1]
        b_type = _as_tensor_type(b_param.type)
        assert _shape_dims(b_type) == [N, K], f"b param shape: {_shape_dims(b_type)}"
        assert b_type.tensor_view is None

        a_type = _as_tensor_type(incore.params[0].type)
        assert _shape_dims(a_type) == [M, K]
        assert a_type.tensor_view is None

        # The body has a prepended ``b_dn = tensor.as_layout(b, DN)`` binding
        # whose result carries the canonical ``[K, N] DN`` view.
        b_dn_var, b_dn_call = _find_as_layout_binding(incore, b_param)
        b_dn_type = _as_tensor_type(b_dn_var.type)
        assert _shape_dims(b_dn_type) == [K, N], f"b_dn shape: {_shape_dims(b_dn_type)}"
        assert b_dn_type.tensor_view is not None
        assert b_dn_type.tensor_view.layout == ir.TensorLayout.DN
        # ``layout`` kwarg on the as_layout call should be DN.
        assert b_dn_call.kwargs.get("layout") == ir.TensorLayout.DN

        # ``tile.load`` on the promoted slot now reads from ``b_dn``, has its
        # trailing pair swapped, and carries ``transpose=False``.
        loads_by_src = {}
        for ld in _find_tile_loads(incore):
            assert isinstance(ld.args[0], ir.Var)
            loads_by_src[ld.args[0].name_hint] = ld

        # The tile.load(b, transpose=True) was rewritten to load from b_dn.
        assert b_dn_var.name_hint in loads_by_src, (
            f"tile.load must read from the as_layout LHS, not raw param. "
            f"loaded srcs: {list(loads_by_src.keys())}"
        )
        load_b = loads_by_src[b_dn_var.name_hint]
        shapes_arg = load_b.args[2]
        assert isinstance(shapes_arg, ir.MakeTuple)
        shape_vals = [el.value for el in shapes_arg.elements if isinstance(el, ir.ConstInt)]
        assert shape_vals == [K, N], f"tile.load(b_dn) shapes: {shape_vals}"
        assert _transpose_kwarg(load_b) is False, "tile.load(b_dn) transpose kwarg must be False after P6"

        load_a = loads_by_src["a"]
        shape_vals_a = [el.value for el in load_a.args[2].elements if isinstance(el, ir.ConstInt)]
        assert shape_vals_a == [M, K]

        # Orch is untouched — its call site passes ``b`` (a wrapper param)
        # directly without any tensor.as_layout bridge.
        orch = _find_function(After, "orchestrator")
        calls = _find_calls_to(orch, "matmul_incore")
        assert len(calls) == 1
        b_arg = calls[0].args[1]
        assert isinstance(b_arg, ir.Var)
        assert b_arg is orch.params[1], "orch call arg must be the orch's own param (no bridge)"
        assert not _has_as_layout_for(orch, orch.params[1]), (
            "no tensor.as_layout bridge should be emitted in orch under the InCore-side design"
        )

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
        b_param = incore.params[1]
        b_type = _as_tensor_type(b_param.type)
        # Param is untouched.
        assert _shape_dims(b_type) == [N, K]
        assert b_type.tensor_view is None
        # Body prepends b_dn = as_layout(b, DN); LHS carries [K, N] DN.
        b_dn_var, _ = _find_as_layout_binding(incore, b_param)
        b_dn_type = _as_tensor_type(b_dn_var.type)
        assert _shape_dims(b_dn_type) == [K, N]
        assert b_dn_type.tensor_view is not None
        assert b_dn_type.tensor_view.layout == ir.TensorLayout.DN


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
        a_param = incore.params[0]
        a_t = _as_tensor_type(a_param.type)
        # Param is untouched.
        assert _shape_dims(a_t) == [K, M]
        assert a_t.tensor_view is None
        # Body prepends a_dn = as_layout(a, DN); LHS carries [M, K] DN.
        a_dn_var, _ = _find_as_layout_binding(incore, a_param)
        a_dn_type = _as_tensor_type(a_dn_var.type)
        assert _shape_dims(a_dn_type) == [M, K]
        assert a_dn_type.tensor_view is not None
        assert a_dn_type.tensor_view.layout == ir.TensorLayout.DN

        # ``b`` is not promoted (no transpose=True load), so no binding for it.
        b_param = incore.params[1]
        b_t = _as_tensor_type(b_param.type)
        assert _shape_dims(b_t) == [K, N]
        assert b_t.tensor_view is None
        assert not _has_as_layout_for(incore, b_param)

        # ``tile.load`` reads from a_dn (the binding's LHS), not from raw ``a``.
        loads = {ld.args[0].name_hint: ld for ld in _find_tile_loads(incore)}
        assert a_dn_var.name_hint in loads, f"expected load from a_dn, got srcs: {list(loads)}"
        load_a = loads[a_dn_var.name_hint]
        shape_vals = [el.value for el in load_a.args[2].elements if isinstance(el, ir.ConstInt)]
        assert shape_vals == [M, K]
        assert _transpose_kwarg(load_a) is False

        # Orch is untouched — call args are direct refs to wrapper params,
        # no tensor.as_layout bridges injected.
        orch = _find_function(After, "orchestrator")
        call = _find_calls_to(orch, "matmul_incore")[0]
        assert call.args[0] is orch.params[0]
        assert call.args[1] is orch.params[1]
        assert not _has_as_layout_for(orch, orch.params[0])


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
        a_param = incore.params[0]
        b_param = incore.params[1]
        # Both params are untouched.
        a_t = _as_tensor_type(a_param.type)
        b_t = _as_tensor_type(b_param.type)
        assert _shape_dims(a_t) == [K, M]
        assert a_t.tensor_view is None
        assert _shape_dims(b_t) == [N, K]
        assert b_t.tensor_view is None
        # Body prepends one as_layout binding per promoted param.
        a_dn_var, _ = _find_as_layout_binding(incore, a_param)
        b_dn_var, _ = _find_as_layout_binding(incore, b_param)
        a_dn_t = _as_tensor_type(a_dn_var.type)
        b_dn_t = _as_tensor_type(b_dn_var.type)
        assert _shape_dims(a_dn_t) == [M, K]
        assert a_dn_t.tensor_view is not None and a_dn_t.tensor_view.layout == ir.TensorLayout.DN
        assert _shape_dims(b_dn_t) == [K, N]
        assert b_dn_t.tensor_view is not None and b_dn_t.tensor_view.layout == ir.TensorLayout.DN

        # Orch is untouched — no bridges injected.
        orch = _find_function(After, "orchestrator")
        call = _find_calls_to(orch, "matmul_incore")[0]
        for slot in (0, 1):
            assert call.args[slot] is orch.params[slot]
            assert not _has_as_layout_for(orch, orch.params[slot])


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


class TestStridedParamFlipsCorrectly:
    """Regression for #1212 / #1213: when an InCore param's TensorView carries
    parent-derived strides (a sliced sub-view of a larger tensor — set up here
    via an explicit ``pl.TensorView(stride=...)`` annotation, the same shape
    ``SliceInputStridesOptimizer`` produces in the default pipeline), the body-
    prepended ``tensor.as_layout`` flip must propagate those strides through
    the §4.2 canonical-pair swap. The output DN view must carry the swapped
    parent stride, not the slice-shape-derived packed stride — otherwise PTOAS
    walks rows at the wrong stride and silently miscompiles."""

    def test_strided_nd_param_flips_to_strided_dn(self):
        """Parent buffer is ``[T, K_parent] ND`` with row stride ``K_parent``;
        a slice ``[T, K_slice]`` annotated with that parent stride must flip
        to ``[K_slice, T] DN`` with stride ``[1, K_parent]`` — preserving the
        parent's row stride at the trailing slot.
        """
        T, K_slice, K_parent = 16, 512, 16384

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[T, K_slice], pl.FP32],
                # Slice of a `[T, K_parent]` parent: strided-ND with the
                # parent's row stride preserved at the outer slot.
                b_slice: pl.Tensor[  # noqa: E501
                    [T, K_slice],
                    pl.FP32,
                    pl.TensorView(stride=[K_parent, 1], layout=pl.TensorLayout.ND),
                ],
                c: pl.Out[pl.Tensor[[T, T], pl.FP32]],
            ) -> pl.Tensor[[T, T], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [T, K_slice], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(
                    b_slice, [0, 0], [T, K_slice], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                tile_a_l0 = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0 = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0, tile_b_l0)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[T, K_slice], pl.FP32],
                b_slice: pl.Tensor[  # noqa: E501
                    [T, K_slice],
                    pl.FP32,
                    pl.TensorView(stride=[K_parent, 1], layout=pl.TensorLayout.ND),
                ],
            ) -> pl.Tensor[[T, T], pl.FP32]:
                c: pl.Tensor[[T, T], pl.FP32] = pl.create_tensor([T, T], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b_slice, c)
                return c_result

        After = passes.lower_transpose_load_param_layout()(Before)
        incore = _find_function(After, "matmul_incore")
        b_param = incore.params[1]
        b_type = _as_tensor_type(b_param.type)
        # Param is untouched: still [T, K_slice] ND with parent's stride.
        assert _shape_dims(b_type) == [T, K_slice]
        assert b_type.tensor_view is not None
        assert b_type.tensor_view.layout == ir.TensorLayout.ND
        assert _const_int_values(b_type.tensor_view.stride) == [K_parent, 1]

        # Body prepends b_dn = as_layout(b, DN). Critical: the DN view must
        # inherit the parent's stride, trailing-pair-swapped — NOT canonical
        # packed strides derived from the slice's logical shape.
        b_dn_var, _ = _find_as_layout_binding(incore, b_param)
        b_dn_type = _as_tensor_type(b_dn_var.type)
        assert _shape_dims(b_dn_type) == [K_slice, T]
        assert b_dn_type.tensor_view is not None
        assert b_dn_type.tensor_view.layout == ir.TensorLayout.DN
        assert _const_int_values(b_dn_type.tensor_view.stride) == [1, K_parent], (
            "DN view of a strided-ND parent must carry the parent's row stride "
            f"({K_parent}) at the trailing slot, not the slice-shape-derived "
            f"packed stride ({K_slice}). See #1212 / #1213."
        )


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
        kc_param = incore.params[1]
        kc_t = _as_tensor_type(kc_param.type)
        # Param is untouched.
        assert _shape_dims(kc_t) == [128, 128]
        assert kc_t.tensor_view is None
        # Body prepends key_cache_dn = as_layout(key_cache, DN). Shape stays
        # [128, 128] (square) but layout is DN — the trailing-pair swap on the
        # full TensorType shape happens to be identity here.
        kc_dn_var, _ = _find_as_layout_binding(incore, kc_param)
        kc_dn_t = _as_tensor_type(kc_dn_var.type)
        assert _shape_dims(kc_dn_t) == [128, 128]
        assert kc_dn_t.tensor_view is not None
        assert kc_dn_t.tensor_view.layout == ir.TensorLayout.DN

        # tile.load reads from the binding's LHS, with swapped load window
        # ([64, 128] -> [128, 64]) and transpose=False.
        loads = {ld.args[0].name_hint: ld for ld in _find_tile_loads(incore)}
        assert kc_dn_var.name_hint in loads
        load_k = loads[kc_dn_var.name_hint]
        shape_vals = [el.value for el in load_k.args[2].elements if isinstance(el, ir.ConstInt)]
        assert shape_vals == [128, 64]
        assert _transpose_kwarg(load_k) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
