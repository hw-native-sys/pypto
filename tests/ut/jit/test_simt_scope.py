# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile public SIMT authoring through the complete JIT pipeline."""

import ast
import json
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import CacheConfig, DataType, backend, ir, passes
from pypto.ir.compiled_program import CompiledProgram
from pypto.jit._specialize_body import BodyTransformer
from pypto.jit._specialize_types import TensorMeta
from pypto.runtime import ExecutionMode, RunConfig


@pl.jit.aiv
def scope_copy(
    src: pl.Tensor[[1, 32], pl.FP32],
    dst: pl.InOut[pl.Tensor[[1, 32], pl.FP32]],
    n: pl.Scalar[pl.INT32],
):
    with pl.simt_scope(dims=(32, 1, 1)):
        i = pl.simt.thread_idx(pl.simt.Axis.X)
        if i < n:
            value = pl.simt.load(src, (0, i))
            pl.simt.store(dst, (0, i), value)
    return dst


@pl.jit
def scope_entry(
    src: pl.Tensor[[1, 32], pl.FP32],
    dst: pl.InOut[pl.Tensor[[1, 32], pl.FP32]],
    n: pl.Scalar[pl.INT32],
):
    return scope_copy(src, dst, n)


def test_public_jit_scope_full_compile(tmp_path: Path):
    backend.reset_for_testing()
    try:
        compiled = scope_entry.compile(
            config=RunConfig(
                backend_type=backend.BackendType.Ascend950,
                execution_mode=ExecutionMode.ONBOARD,
                cache_config=CacheConfig(enabled=False),
                save_kernels_dir=str(tmp_path),
                dump_passes=True,
            ),
        )
        assert isinstance(compiled, CompiledProgram)
        helpers = list((tmp_path / "vpto").rglob("*.pto"))
        assert len(helpers) == 1
        text = helpers[0].read_text()
        assert "pto.simt_entry" in text
        assert "pto.simt_launch" in text
        assert "pto.ldg" in text and "pto.stg" in text
        manifest = json.loads(helpers[0].with_name("manifest.json").read_text())
        assert manifest["helpers"][0]["kind"] == "simt"
    finally:
        backend.reset_for_testing()


def test_jit_aiv_specialize_preserves_transient_scope():
    lowered = scope_entry.specialize()
    kernel = lowered.get_function("scope_copy")
    assert kernel is not None and kernel.func_type == ir.FunctionType.AIV
    scopes: list[ir.SimtScopeStmt] = []

    class Collector(ir.IRVisitor):
        def visit_simt_scope_stmt(self, op: ir.SimtScopeStmt) -> None:
            scopes.append(op)

    Collector().visit_program(lowered)
    assert len(scopes) == 1


@pl.jit.aiv
def shape_shadow_scope(
    src: pl.Tensor[[1, 32], pl.FP32],
    dst: pl.InOut[pl.Tensor[[1, 32], pl.FP32]],
    n: pl.Scalar[pl.INT32],
):
    with pl.simt_scope(dims=(32, 1, 1)):
        # JIT resolves this fixed Tensor annotation before visiting the body.
        n = src.shape[1]  # pyright: ignore[reportOptionalSubscript]
        pl.simt.store(dst, (0, 0), pl.cast(n, pl.FP32))
    with pl.simt_scope(dims=(32, 1, 1)):
        tid = pl.simt.thread_idx(pl.simt.Axis.X)
        if tid < n:
            value = pl.simt.load(src, (0, tid))
            pl.simt.store(dst, (0, tid), value)
    return dst


@pl.jit.aiv
def loaded_tile_scope(src: pl.Tensor[[1, 32], pl.FP32]):
    local = pl.tile.create([1, 32], pl.FP32)
    for i in pl.range(2):
        local = pl.tile.load(src, [0, 0], [1, 32])
        with pl.simt_scope(dims=(32, 1, 1)):
            pl.simt.store(local, (0, 0), pl.const(1.0, pl.FP32))


def test_jit_shape_shadow_keeps_runtime_guard():
    lowered = shape_shadow_scope.specialize()
    kernel = lowered.get_function("shape_shadow_scope")
    assert kernel is not None
    conditions: list[ir.Expr] = []

    class Collector(ir.IRVisitor):
        def visit_if_stmt(self, op: ir.IfStmt) -> None:
            conditions.append(op.condition)

    Collector().visit_program(lowered)
    assert len(conditions) == 1
    # The first scope folds n to 32 locally. The later guard must still use
    # the caller's runtime parameter, otherwise partial-tail bounds are lost.
    assert isinstance(conditions[0], ir.Lt)
    assert conditions[0].right is kernel.params[2]


def test_jit_loaded_tile_reassignment_can_be_captured():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    try:
        outlined = passes.outline_simt_scopes()(loaded_tile_scope.specialize())
        assert any(fn.func_type == ir.FunctionType.SIMT for fn in outlined.functions.values())
    finally:
        backend.reset_for_testing()


@pytest.mark.parametrize(
    "assignment",
    [
        "n = pl.simt.thread_idx(pl.simt.Axis.X)",
        "n: pl.Scalar[pl.INDEX] = pl.simt.thread_idx(pl.simt.Axis.X)",
        "m = n = pl.simt.thread_idx(pl.simt.Axis.X)",
        "n = n - 1",
    ],
)
def test_private_reassignment_invalidates_shape_fold(assignment):
    tree = ast.parse(f"""
n = src.shape[1]
with pl.simt_scope(dims=(32, 1, 1)):
    {assignment}
    pl.simt.store(dst, (0, n), pl.const(1.0, pl.FP32))
after = n
""")
    transformer = BodyTransformer(
        tensor_meta={"src": TensorMeta(shape=(1, 32), dtype=DataType.FP32)},
        dep_names=set(),
        param_names=["src", "dst"],
    )
    transformed = transformer.visit(tree)
    assert isinstance(transformed, ast.Module)
    scope = transformed.body[0]
    assert isinstance(scope, ast.With)
    store = scope.body[-1]
    assert isinstance(store, ast.Expr) and isinstance(store.value, ast.Call)
    offsets = store.value.args[1]
    assert isinstance(offsets, ast.Tuple)
    assert isinstance(offsets.elts[1], ast.Name) and offsets.elts[1].id == "n"
    after = transformed.body[1]
    assert isinstance(after, ast.Assign)
    assert isinstance(after.value, ast.Constant) and after.value.value == 32
    if assignment == "n = n - 1":
        initial, assigned = scope.body[:2]
        assert isinstance(initial, ast.Assign) and isinstance(initial.value, ast.Constant)
        assert initial.value.value == 32
        assert isinstance(assigned, ast.Assign) and isinstance(assigned.value, ast.BinOp)
        assert isinstance(assigned.value.left, ast.Name) and assigned.value.left.id == "n"


@pl.jit.aiv
def conditional_shape_rebind(src: pl.Tensor[[1, 32], pl.FP32], dst: pl.InOut[pl.Tensor[[1, 32], pl.FP32]]):
    # The fixed Tensor annotation supplies shape before JIT visits this body.
    n = src.shape[1]  # pyright: ignore[reportOptionalSubscript]
    with pl.simt_scope(dims=(32, 1, 1)):
        tid = pl.simt.thread_idx(pl.simt.Axis.X)
        if tid < 16:
            # Shape extents use INDEX; preserve the private binding's type.
            n = pl.cast(tid, pl.INDEX)
        if n < 32:
            pl.simt.store(dst, (0, n), pl.const(1.0, pl.FP32))
    with pl.simt_scope(dims=(32, 1, 1)):
        pl.simt.store(dst, (0, n - 32), pl.const(1.0, pl.FP32))
    return dst


def test_conditional_private_reassignment_materializes_live_in_shape():
    lowered = conditional_shape_rebind.specialize()
    conditions: list[ir.Expr] = []

    class Collector(ir.IRVisitor):
        def visit_if_stmt(self, op: ir.IfStmt) -> None:
            conditions.append(op.condition)
            self.visit_stmt(op.then_body)

    Collector().visit_program(lowered)
    assert len(conditions) == 2
    assert isinstance(conditions[1], ir.Lt)
    assert isinstance(conditions[1].left, ir.Var)
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    try:
        outlined = passes.outline_simt_scopes()(lowered)
        assert len([fn for fn in outlined.functions.values() if fn.func_type == ir.FunctionType.SIMT]) == 2
    finally:
        backend.reset_for_testing()


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("control", ["if condition:", "for i in pl.range(0):", "while condition:"])
def test_conditional_shape_reassignment_keeps_scalar_assignments(local, control):
    prefix = "" if local else "n = src.shape[1]\n"
    local_init = "n = src.shape[1]\n    " if local else ""
    tree = ast.parse(f"""
{prefix}with pl.simt_scope(dims=(32, 1, 1)):
    {local_init}{control}
        n = pl.cast(pl.simt.thread_idx(pl.simt.Axis.X), pl.INDEX)
    pl.simt.store(dst, (0, n), pl.const(1.0, pl.FP32))
""")
    transformed = BodyTransformer(
        tensor_meta={"src": TensorMeta(shape=(1, 32), dtype=DataType.FP32)},
        dep_names=set(),
        param_names=["src", "dst", "condition"],
    ).visit(tree)
    assert isinstance(transformed, ast.Module)
    scope = transformed.body[0]
    assert isinstance(scope, ast.With)
    initial = scope.body[0]
    assert isinstance(initial, ast.Assign) and isinstance(initial.value, ast.Constant)
    assert initial.value.value == 32
    store = scope.body[-1]
    assert isinstance(store, ast.Expr) and isinstance(store.value, ast.Call)
    offsets = store.value.args[1]
    assert isinstance(offsets, ast.Tuple)
    assert isinstance(offsets.elts[1], ast.Name) and offsets.elts[1].id == "n"


@pl.jit.aiv
def linear_shape_rebind(src: pl.Tensor[[1, 32], pl.FP32], dst: pl.InOut[pl.Tensor[[1, 32], pl.FP32]]):
    # The fixed Tensor annotation supplies shape before JIT visits this body.
    n = src.shape[1]  # pyright: ignore[reportOptionalSubscript]
    with pl.simt_scope(dims=(32, 1, 1)):
        n = pl.simt.thread_idx(pl.simt.Axis.X)
        pl.simt.store(dst, (0, n), pl.const(1.0, pl.FP32))
    return dst


def test_jit_private_reassignment_uses_thread_index():
    lowered = linear_shape_rebind.specialize()
    stores: list[ir.Call] = []

    class Collector(ir.IRVisitor):
        def visit_call(self, op: ir.Call) -> None:
            if op.op.name == ir.get_op("simt.store").name:
                stores.append(op)

    Collector().visit_program(lowered)
    assert len(stores) == 1
    offsets = stores[0].args[1]
    assert isinstance(offsets, ir.MakeTuple)
    assert isinstance(offsets.elements[1], ir.Var)


@pl.jit.aiv
def zero_loop_shape_rebind(src: pl.Tensor[[1, 32], pl.FP32], dst: pl.InOut[pl.Tensor[[1, 32], pl.FP32]]):
    # The fixed Tensor annotation supplies shape before JIT visits this body.
    n = src.shape[1]  # pyright: ignore[reportOptionalSubscript]
    with pl.simt_scope(dims=(32, 1, 1)):
        for i in pl.range(0):
            n = pl.const(0, pl.INDEX)
        if n < 32:
            pl.simt.store(dst, (0, n), pl.const(1.0, pl.FP32))
    return dst


def test_zero_iteration_private_loop_keeps_incoming_shape_value():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    try:
        outlined = passes.outline_simt_scopes()(zero_loop_shape_rebind.specialize())
        ssa = passes.convert_to_ssa()(outlined)
        properties = passes.IRPropertySet()
        properties.insert(passes.IRProperty.UseAfterDef)
        properties.insert(passes.IRProperty.SSAForm)
        assert not passes.PropertyVerifierRegistry.verify(properties, ssa)
    finally:
        backend.reset_for_testing()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
