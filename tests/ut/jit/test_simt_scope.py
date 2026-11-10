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
import re
import runpy
import subprocess
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import CacheConfig, DataType, backend, ir, passes
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.compiled_program import CompiledProgram
from pypto.jit import decorator
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


@pytest.mark.parametrize("use_ptoas", [False, True], ids=["diagnostic", "executable"])
def test_public_jit_scope_full_compile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_ptoas: bool):
    if use_ptoas:
        if find_ptoas_binary() is None:
            pytest.skip("PTOAS is unavailable; use --require-ptoas to make this an error")
    else:
        monkeypatch.setattr(decorator, "_ptoas_available", lambda: False)
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
        artifact_dir = "vpto" if use_ptoas else "kernels"
        helpers = list((tmp_path / artifact_dir).rglob("*.pto"))
        assert len(helpers) == 1
        text = helpers[0].read_text()
        assert "pto.simt_entry" in text
        assert "pto.simt_launch" in text
        assert "pto.ldg" in text and "pto.stg" in text
        if use_ptoas:
            manifest = json.loads(helpers[0].with_name("manifest.json").read_text())
            assert manifest["helpers"][0]["kind"] == "simt"
        else:
            assert not (tmp_path / "vpto").exists()
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


@pl.jit.aiv
def zero_while_shape_rebind(src: pl.Tensor[[1, 32], pl.FP32], dst: pl.InOut[pl.Tensor[[1, 32], pl.FP32]]):
    # The fixed Tensor annotation supplies shape before JIT visits this body.
    n = src.shape[1]  # pyright: ignore[reportOptionalSubscript]
    with pl.simt_scope(dims=(32, 1, 1)):
        while n < 0:
            n = pl.const(0, pl.INDEX)
        if n < 32:
            pl.simt.store(dst, (0, n), pl.const(1.0, pl.FP32))
    return dst


@pytest.mark.parametrize("kernel", [zero_loop_shape_rebind, zero_while_shape_rebind])
def test_zero_iteration_private_loop_keeps_incoming_shape_value(kernel):
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    try:
        outlined = passes.outline_simt_scopes()(kernel.specialize())
        ssa = passes.convert_to_ssa()(outlined)
        properties = passes.IRPropertySet()
        properties.insert(passes.IRProperty.UseAfterDef)
        properties.insert(passes.IRProperty.SSAForm)
        assert not passes.PropertyVerifierRegistry.verify(properties, ssa)
    finally:
        backend.reset_for_testing()


@pl.jit.aiv
def unpacked_shape_rebind(src: pl.Tensor[[1, 32], pl.FP32], dst: pl.InOut[pl.Tensor[[1, 32], pl.FP32]]):
    with pl.simt_scope(dims=(32, 1, 1)):
        # JIT resolves the fixed Tensor annotation before unpacking its shape.
        rows, n = src.shape  # pyright: ignore[reportOptionalIterable]
        tid = pl.simt.thread_idx(pl.simt.Axis.X)
        if tid < 16:
            n = pl.cast(tid, pl.INDEX)
        if n < 32:
            pl.simt.store(dst, (rows - 1, n), pl.const(1.0, pl.FP32))
    return dst


@pl.jit
def unpacked_shape_entry(src: pl.Tensor[[1, 32], pl.FP32], dst: pl.InOut[pl.Tensor[[1, 32], pl.FP32]]):
    return unpacked_shape_rebind(src, dst)


def test_conditional_shape_unpacking_compiles_as_scalars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(decorator, "_ptoas_available", lambda: False)
    backend.reset_for_testing()
    try:
        compiled = unpacked_shape_entry.compile(
            config=RunConfig(
                backend_type=backend.BackendType.Ascend950,
                execution_mode=ExecutionMode.ONBOARD,
                cache_config=CacheConfig(enabled=False),
                save_kernels_dir=str(tmp_path),
            )
        )
        assert isinstance(compiled, CompiledProgram)
        kernels = list((tmp_path / "kernels").rglob("*.pto"))
        assert len(kernels) == 1
        assert "pto.simt_entry" in kernels[0].read_text()
    finally:
        backend.reset_for_testing()


@pytest.mark.parametrize("assignment", ["n = src.shape[0]", "n, width = src.shape"])
@pytest.mark.parametrize("rebind", [False, True], ids=["first-binding", "renamed-binding"])
def test_shape_assignment_supersedes_scalar_binding(assignment, rebind):
    prior_write = "n = n + 2" if rebind else ""
    tree = ast.parse(f"""
with pl.simt_scope(dims=(1, 1, 1)):
    n = pl.simt.thread_idx(pl.simt.Axis.X)
    {prior_write}
    {assignment}
    pl.simt.store(dst, (0, n), pl.const(1.0, pl.FP32))
""")
    transformed = BodyTransformer(
        tensor_meta={"src": TensorMeta(shape=(1, 32), dtype=DataType.FP32)},
        dep_names=set(),
        param_names=["src", "dst"],
    ).visit(tree)
    assert isinstance(transformed, ast.Module)
    scope = transformed.body[0]
    assert isinstance(scope, ast.With)
    store = scope.body[-1]
    assert isinstance(store, ast.Expr) and isinstance(store.value, ast.Call)
    offsets = store.value.args[1]
    assert isinstance(offsets, ast.Tuple)
    index = offsets.elts[1]
    # Resolve the actual store operand without relying on generated alias names.
    # The last shape assignment must replace tid + 2, including after unpacking.
    if isinstance(index, ast.Name):
        definitions = {
            target.id: stmt.value
            for stmt in scope.body
            if isinstance(stmt, ast.Assign)
            for target in stmt.targets
            if isinstance(target, ast.Name)
        }
        index = definitions[index.id]
    assert isinstance(index, ast.Constant) and index.value == 1


@pl.jit.aiv
def renamed_shape_index(src: pl.Tensor[[1, 32], pl.FP32], dst: pl.InOut[pl.Tensor[[1, 32], pl.FP32]]):
    with pl.simt_scope(dims=(1, 1, 1)):
        n = pl.simt.thread_idx(pl.simt.Axis.X)
        n = n + 2
        # The fixed annotation supplies a known shape; this replaces the alias.
        n = src.shape[0]  # pyright: ignore[reportOptionalSubscript]
        pl.simt.store(dst, (0, n), pl.const(1.0, pl.FP32))
    return dst


@pl.jit.aiv
def renamed_shape_unpack(src: pl.Tensor[[1, 32], pl.FP32], dst: pl.InOut[pl.Tensor[[1, 32], pl.FP32]]):
    with pl.simt_scope(dims=(1, 1, 1)):
        n = pl.simt.thread_idx(pl.simt.Axis.X)
        n = n + 2
        # Tuple expansion must replace the current scalar alias as well.
        n, _width = src.shape  # pyright: ignore[reportOptionalIterable]
        pl.simt.store(dst, (0, n), pl.const(1.0, pl.FP32))
    return dst


@pl.jit
def renamed_shape_index_entry(src: pl.Tensor[[1, 32], pl.FP32], dst: pl.InOut[pl.Tensor[[1, 32], pl.FP32]]):
    return renamed_shape_index(src, dst)


@pl.jit
def renamed_shape_unpack_entry(src: pl.Tensor[[1, 32], pl.FP32], dst: pl.InOut[pl.Tensor[[1, 32], pl.FP32]]):
    return renamed_shape_unpack(src, dst)


@pytest.mark.parametrize("entry", [renamed_shape_index_entry, renamed_shape_unpack_entry])
def test_rebound_shape_store_index_full_compile(entry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(decorator, "_ptoas_available", lambda: False)
    backend.reset_for_testing()
    try:
        compiled = entry.compile(
            config=RunConfig(
                backend_type=backend.BackendType.Ascend950,
                execution_mode=ExecutionMode.ONBOARD,
                cache_config=CacheConfig(enabled=False),
                save_kernels_dir=str(tmp_path),
            )
        )
        assert isinstance(compiled, CompiledProgram) and compiled.program is not None
        kernels = list((tmp_path / "kernels").rglob("*.pto"))
        assert len(kernels) == 1 and "pto.stg" in kernels[0].read_text()
        simplified = passes.simplify()(
            passes.convert_to_ssa()(passes.outline_simt_scopes()(compiled.program))
        )
        stores: list[ir.Call] = []

        class Collector(ir.IRVisitor):
            def visit_call(self, op: ir.Call) -> None:
                if op.op.name == ir.get_op("simt.store").name:
                    stores.append(op)

        Collector().visit_program(simplified)
        assert len(stores) == 1
        indices = stores[0].args[1]
        assert isinstance(indices, ir.MakeTuple)
        index = indices.elements[1]
        assert isinstance(index, ir.ConstInt) and index.value == 1
    finally:
        backend.reset_for_testing()


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("entry_name", ["collective_entry", "while_collective_entry"])
def test_public_warp_collectives_native_compile(tmp_path, pytestconfig, entry_name):
    binary = find_ptoas_binary()
    if binary is None:
        pytest.skip("PTOAS is unavailable; use --require-ptoas to make this an error")
    source = Path(__file__).resolve().parents[2] / "st/runtime/vpto/test_simt_collectives_onboard.py"
    pytestconfig.addinivalue_line("markers", "platforms(*ids, reason): supported ST platforms")
    fixture = runpy.run_path(str(source))
    backend.reset_for_testing()
    try:
        compiled = fixture["compile_collectives"](tmp_path, entry=fixture[entry_name])
        assert isinstance(compiled, CompiledProgram)
        helpers = list((tmp_path / "vpto").rglob("*.pto"))
        assert len(helpers) == 1
        text = helpers[0].read_text()
        native_ops = (
            "vote_all",
            "vote_any",
            "vote_uni",
            "vote_ballot",
            "shuffle_idx",
            "shuffle_up",
            "shuffle_down",
            "shuffle_bfly",
            "redux_addi",
            "redux_mini",
            "redux_maxi",
            "redux_addf",
            "redux_minf",
            "redux_maxf",
        )
        is_while = entry_name == "while_collective_entry"
        for op in ("vote_any",) if is_while else native_ops:
            assert f"pto.{op} " in text
        if not is_while:
            assert " signed : i32" in text
        # Collectives must remain within the divergent branch after the full
        # PyPTO pass pipeline; their implicit active mask is part of the input.
        helper_body = text[text.index("pto.simt_entry") :]
        if is_while:
            assert (
                helper_body.index("scf.while")
                < helper_body.index("pto.vote_any")
                < helper_body.index("scf.condition")
            )
        else:
            assert helper_body.index("scf.if") < helper_body.index("pto.vote_all")
        output = tmp_path / "collectives.ll"
        result = subprocess.run(
            [
                binary,
                str(helpers[0]),
                "--pto-arch=a5",
                "--pto-backend=vpto",
                "--emit-vpto-llvm-ir",
                "-o",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        llvm = output.read_text()
        intrinsics = (
            "vote.all",
            "vote.any",
            "vote.uni",
            "vote.ballot",
            "redux.add.s32",
            "redux.min.s32",
            "redux.max.s32",
            "redux.add.f32",
            "redux.min.f32",
            "redux.max.f32",
        )
        for intrinsic in ("vote.any",) if is_while else intrinsics:
            assert f"@llvm.hivm.{intrinsic}" in llvm
        assert "asm " not in llvm
    finally:
        backend.reset_for_testing()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("entry_name", ["atomic_entry", "barrier_entry", "integer_entry", "floating_entry"])
def test_public_atomics_native_compile(tmp_path, pytestconfig, entry_name):
    binary = find_ptoas_binary()
    if binary is None:
        pytest.skip("PTOAS is unavailable; use --require-ptoas to make this an error")
    edges = entry_name in {"integer_entry", "floating_entry"}
    filename = "test_simt_atomic_edges_onboard.py" if edges else "test_simt_atomics_onboard.py"
    source = Path(__file__).resolve().parents[2] / "st/runtime/vpto" / filename
    pytestconfig.addinivalue_line("markers", "platforms(*ids, reason): supported ST platforms")
    fixture = runpy.run_path(str(source))
    backend.reset_for_testing()
    try:
        fixture["compile_case"](tmp_path, entry=fixture[entry_name])
        helpers = list((tmp_path / "vpto").rglob("*.pto"))
        assert len(helpers) == 1
        text = helpers[0].read_text()
        if entry_name == "atomic_entry":
            for op in ("add", "min", "max", "exch", "cas"):
                assert f"pto.atomic_{op} " in text
            # Four storage targets, each with a consumed and an ignored Add.
            assert text.count("pto.atomic_add ") == 8
            assert " signed :" in text and "!pto.ptr<i32, ub>" in text
        elif not edges:
            assert "pto.syncthreads" in text
        if not edges:
            assert "pto.threadfence" in text and "pto.threadfence_block" in text
        result = subprocess.run(
            [
                binary,
                str(helpers[0]),
                "--pto-arch=a5",
                "--pto-backend=vpto",
                "--emit-vpto-llvm-ir",
                "-o",
                str(tmp_path / "atomics.ll"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        llvm = (tmp_path / "atomics.ll").read_text()
        if not edges:
            assert "call void @llvm.hivm.fence.workitems()" in llvm
            assert "call void @llvm.hivm.fenceblock.workitems()" in llvm
        if entry_name == "barrier_entry":
            assert "call void @llvm.hivm.sync.workitems()" in llvm
        if entry_name == "atomic_entry":
            for space in ("G", "S"):
                for dtype in ("s32", "fp32"):
                    for op in ("ADD", "MIN", "MAX", "EXCH", "CAS"):
                        assert f"@llvm.hivm.atom.{op}.{space}.{dtype}(" in llvm
    finally:
        backend.reset_for_testing()


@pytest.mark.ptoas_compile
def test_atomic_nd_address_and_nested_calls_native(tmp_path, pytestconfig):
    binary = find_ptoas_binary()
    if binary is None:
        pytest.skip("PTOAS is unavailable; use --require-ptoas to make this an error")
    source = Path(__file__).resolve().parents[2] / "st/runtime/vpto/test_simt_atomic_regressions_onboard.py"
    pytestconfig.addinivalue_line("markers", "platforms(*ids, reason): supported ST platforms")
    fixture = runpy.run_path(str(source))
    backend.reset_for_testing()
    try:
        fixture["compile_case"](tmp_path)
        helpers = list((tmp_path / "vpto").rglob("*.pto"))
        assert len(helpers) == 1
        text = helpers[0].read_text()
        assert text.count("pto.atomic_add ") == 8
        # The while condition must still perform an atomic on each evaluation.
        condition = text[text.index("scf.while") : text.index("scf.condition")]
        assert "pto.atomic_add " in condition
        result = subprocess.run(
            [
                binary,
                str(helpers[0]),
                "--pto-arch=a5",
                "--pto-backend=vpto",
                "--emit-vpto-llvm-ir",
                "-o",
                str(tmp_path / "regression.ll"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        llvm = (tmp_path / "regression.ll").read_text()
        assert "i64 167" in llvm
        assert "i64 34" not in llvm
        assert llvm.count("call i32 @llvm.hivm.atom.ADD.") == 8
        # Include unselected tuple elements and verify order, not just count.
        exchanges = [line for line in llvm.splitlines() if "call i32 @llvm.hivm.atom.EXCH." in line]
        assert len(exchanges) == 7
        values = []
        for line in exchanges:
            match = re.search(r", i32 (\d+)[,)]", line)
            assert match is not None, line
            values.append(int(match.group(1)))
        assert values == [7, 1, 2, 1, 2, 1, 2]
    finally:
        backend.reset_for_testing()
