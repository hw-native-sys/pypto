# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Keep control-flow storage selections free of unnecessary same-space copies."""

import re
import subprocess

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _get_ptoas_flags
from pypto.runtime import ExecutionMode

_TILE_CREATE = ir.get_op("tile.create").name


@pl.jit.incore
def rotate(
    a: pl.Tensor[[512, 64], pl.FP16],
    b: pl.Tensor[[64, 16], pl.FP16],
    out: pl.Out[pl.Tensor[[512, 16], pl.FP32]],
):
    current = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
    weight = pl.load(b, [0, 0], [64, 16], target_memory=pl.MemorySpace.Mat)
    right = pl.tile.move(weight, target_memory=pl.MemorySpace.Right)
    for block in pl.range(8):
        if block + 1 < 8:
            following = pl.load(a, [(block + 1) * 64, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        else:
            following = current
        left = pl.tile.move(current, target_memory=pl.MemorySpace.Left)
        result = pl.tile.matmul(left, right)
        pl.store(result, [block * 64, 0], out)
        current = following
    return out


@pl.jit
def entry(
    a: pl.Tensor[[512, 64], pl.FP16],
    b: pl.Tensor[[64, 16], pl.FP16],
    out: pl.Out[pl.Tensor[[512, 16], pl.FP32]],
):
    with pl.spmd(1):
        out = rotate(a, b, out)
    return out


@pl.jit.incore
def mat_phi_core(
    x: pl.Tensor[[32, 16], pl.BF16],
    y: pl.Tensor[[16, 16], pl.BF16],
    flag: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
):
    a = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
    b = pl.load(x, [16, 0], [16, 16], target_memory=pl.Mem.Mat)
    rhs = pl.load(y, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
    if pl.read(flag, [0]) > 0:
        selected = pl.yield_(a)
    else:
        selected = pl.yield_(b)
    product = pl.matmul(selected, rhs, out_dtype=pl.FP32)
    pl.store(product, [0, 0], out)
    # Both inputs remain live after selection and must keep independent storage.
    keep_a = pl.matmul(a, rhs, out_dtype=pl.FP32)
    pl.store(keep_a, [16, 0], out)
    keep_b = pl.matmul(b, rhs, out_dtype=pl.FP32)
    pl.store(keep_b, [32, 0], out)
    return out


@pl.jit
def mat_phi(
    x: pl.Tensor[[32, 16], pl.BF16],
    y: pl.Tensor[[16, 16], pl.BF16],
    flag: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
):
    with pl.spmd(1):
        out = mat_phi_core(x, y, flag, out)
    return out


@pytest.mark.parametrize("entrypoint", [entry, mat_phi], ids=["3039", "3049"])
@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
)
@pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer"])
def test_live_mat_control_flow_has_no_mat_copy(
    tmp_path, ascend_backend, pass_verification_instruments, entrypoint, planner, enable_buffer_ir
):
    """Keep the public producer/consumer order and compile the resulting PTO."""
    binary = find_ptoas_binary()
    if binary is None:
        pytest.skip("PTOAS is unavailable")
    with passes.PassContext(
        pass_verification_instruments, memory_planner=planner, enable_buffer_ir=enable_buffer_ir
    ):
        ir.compile(
            entrypoint.specialize(),
            output_dir=str(tmp_path),
            backend_type=BackendType.Ascend910,
            execution_mode=ExecutionMode.ONBOARD,
            dump_passes=True,
            skip_ptoas=True,
        )
    kernels = list(tmp_path.rglob("*.pto"))
    assert kernels
    for kernel in kernels:
        for line in kernel.read_text().splitlines():
            if "pto.tmov" in line:
                assert len(re.findall(r"loc\s*=\s*mat(?:,|>)", line)) < 2, line
        result = subprocess.run(
            [binary, str(kernel), "-o", str(kernel.with_suffix(".checked.cpp")), *_get_ptoas_flags(planner)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr


@pl.program
class VecLiveInputs:
    """An outer value selected by an if must retain its own still-live storage."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[32, 16], pl.FP32],
        flag: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
    ) -> pl.Tensor[[48, 16], pl.FP32]:
        a = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
        b = pl.load(x, [16, 0], [16, 16], target_memory=pl.Mem.Vec)
        if flag > 0:
            selected = pl.yield_(a)
        else:
            selected = pl.yield_(b)
        pl.store(pl.add(selected, 3.0), [0, 0], out)
        pl.store(a, [16, 0], out)
        pl.store(b, [32, 0], out)
        return out


@pl.program
class VecLiveProducers:
    """Storage selection also applies when fresh values come from arithmetic."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[32, 16], pl.FP32],
        flag: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
    ) -> pl.Tensor[[48, 16], pl.FP32]:
        first = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
        second = pl.load(x, [16, 0], [16, 16], target_memory=pl.Mem.Vec)
        a = pl.add(first, 1.25)
        b = pl.mul(second, -2.0)
        if flag > 0:
            selected = pl.yield_(a)
        else:
            selected = pl.yield_(b)
        pl.store(pl.add(selected, 0.5), [0, 0], out)
        pl.store(a, [16, 0], out)
        pl.store(b, [32, 0], out)
        return out


@pl.program
class VecBranchLocalProducers:
    """Fresh producers inside exclusive arms still represent a storage choice."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[32, 16], pl.FP32],
        flag: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
    ) -> pl.Tensor[[48, 16], pl.FP32]:
        first = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
        second = pl.load(x, [16, 0], [16, 16], target_memory=pl.Mem.Vec)
        if flag > 0:
            a = pl.add(first, 1.25)
            selected = pl.yield_(a)
        else:
            b = pl.mul(second, -2.0)
            selected = pl.yield_(b)
        pl.store(pl.add(selected, 0.5), [0, 0], out)
        pl.store(first, [16, 0], out)
        pl.store(second, [32, 0], out)
        return out


@pl.program
class AccLiveProducers:
    """Both fresh accumulator products survive a later descriptor selection."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[32, 16], pl.FP16],
        y: pl.Tensor[[16, 16], pl.FP16],
        flag: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
    ) -> pl.Tensor[[48, 16], pl.FP32]:
        first = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        second = pl.load(x, [16, 0], [16, 16], target_memory=pl.Mem.Mat)
        rhs = pl.load(y, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        a = pl.matmul(first, rhs, out_dtype=pl.FP32)
        b = pl.matmul(second, rhs, out_dtype=pl.FP32)
        if flag > 0:
            selected = pl.yield_(a)
        else:
            selected = pl.yield_(b)
        pl.store(selected, [0, 0], out)
        pl.store(a, [16, 0], out)
        pl.store(b, [32, 0], out)
        return out


@pl.program
class VecNestedSelection:
    """A selected descriptor can itself be yielded by a surrounding if."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[48, 16], pl.FP32],
        outer_flag: pl.Scalar[pl.INDEX],
        inner_flag: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[64, 16], pl.FP32]],
    ) -> pl.Tensor[[64, 16], pl.FP32]:
        a = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
        b = pl.load(x, [16, 0], [16, 16], target_memory=pl.Mem.Vec)
        c = pl.load(x, [32, 0], [16, 16], target_memory=pl.Mem.Vec)
        if outer_flag > 0:
            if inner_flag > 0:
                selected_inner = pl.yield_(a)
            else:
                selected_inner = pl.yield_(b)
            selected = pl.yield_(selected_inner)
        else:
            selected = pl.yield_(c)
        pl.store(pl.add(selected, 0.5), [0, 0], out)
        pl.store(a, [16, 0], out)
        pl.store(b, [32, 0], out)
        pl.store(c, [48, 0], out)
        return out


@pl.program
class VecIndependentReturnShapes:
    """Each phi position has its own geometry; the two arms agree per position."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[32, 16], pl.FP32],
        flag: pl.Scalar[pl.INDEX],
        wide_out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
        narrow_out: pl.Out[pl.Tensor[[24, 16], pl.FP32]],
    ) -> tuple[pl.Tensor[[48, 16], pl.FP32], pl.Tensor[[24, 16], pl.FP32]]:
        a = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
        b = pl.load(x, [16, 0], [16, 16], target_memory=pl.Mem.Vec)
        c = pl.load(x, [0, 0], [8, 16], target_memory=pl.Mem.Vec)
        d = pl.load(x, [8, 0], [8, 16], target_memory=pl.Mem.Vec)
        if flag > 0:
            selected_wide, selected_narrow = pl.yield_(a, c)
        else:
            selected_wide, selected_narrow = pl.yield_(b, d)
        pl.store(pl.add(selected_wide, 0.5), [0, 0], wide_out)
        pl.store(a, [16, 0], wide_out)
        pl.store(b, [32, 0], wide_out)
        pl.store(pl.add(selected_narrow, 0.5), [0, 0], narrow_out)
        pl.store(c, [8, 0], narrow_out)
        pl.store(d, [16, 0], narrow_out)
        return wide_out, narrow_out


def _walk_statements(statement):
    yield statement
    if isinstance(statement, ir.SeqStmts):
        for child in statement.stmts:
            yield from _walk_statements(child)
    else:
        for attribute in ("body", "then_body", "else_body"):
            child = getattr(statement, attribute, None)
            if child is not None:
                yield from _walk_statements(child)


def _assert_index_selection_dump(tmp_path, expected_results, expected_shapes, memory_space):
    dumps = list(tmp_path.rglob("*_after_MaterializeSemanticAliases.py"))
    assert len(dumps) == 1, dumps
    program = pl.parse_program(dumps[0].read_text())
    statements = [
        statement for function in program.functions.values() for statement in _walk_statements(function.body)
    ]
    results = [
        result
        for statement in statements
        if isinstance(statement, ir.IfStmt)
        for result in statement.return_vars
        if result.name_hint.startswith("selected")
    ]
    assert len(results) == expected_results, dumps[0].read_text()
    assert all(
        isinstance(result.type, ir.ScalarType) and result.type.dtype == DataType.INDEX for result in results
    ), dumps[0].read_text()
    descriptors = [
        statement
        for statement in statements
        if isinstance(statement, ir.AssignStmt)
        and statement.var.name_hint.startswith("selected")
        and isinstance(statement.value, ir.Call)
        and statement.value.op.name == _TILE_CREATE
    ]
    assert len(descriptors) == expected_results, dumps[0].read_text()
    shapes: set[tuple[int, ...]] = set()
    for descriptor in descriptors:
        tile_type = descriptor.var.type
        assert isinstance(tile_type, ir.TileType)
        assert tile_type.memory_space == memory_space
        dimensions = []
        for dimension in tile_type.shape:
            assert isinstance(dimension, ir.ConstInt)
            dimensions.append(dimension.value)
        shapes.add(tuple(dimensions))
    assert shapes == expected_shapes


@pytest.mark.parametrize(
    ("program", "expected_results", "expected_shapes", "memory_space"),
    [
        (VecLiveInputs, 1, {(16, 16)}, ir.MemorySpace.Vec),
        (VecLiveProducers, 1, {(16, 16)}, ir.MemorySpace.Vec),
        (VecBranchLocalProducers, 1, {(16, 16)}, ir.MemorySpace.Vec),
        (AccLiveProducers, 1, {(16, 16)}, ir.MemorySpace.Acc),
        (VecNestedSelection, 2, {(16, 16)}, ir.MemorySpace.Vec),
        (VecIndependentReturnShapes, 2, {(16, 16), (8, 16)}, ir.MemorySpace.Vec),
    ],
    ids=[
        "live-loads",
        "live-arithmetic",
        "branch-local-arithmetic",
        "live-accumulators",
        "nested-if",
        "independent-return-shapes",
    ],
)
@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
)
@pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer"])
def test_live_vec_control_flow_uses_storage_selection(
    tmp_path,
    ascend_backend,
    pass_verification_instruments,
    program,
    expected_results,
    expected_shapes,
    memory_space,
    planner,
    enable_buffer_ir,
):
    """Storage-only phis need no same-space copy, whether legal or unsupported."""
    binary = find_ptoas_binary()
    if binary is None:
        pytest.skip("PTOAS is unavailable")
    with passes.PassContext(
        pass_verification_instruments, memory_planner=planner, enable_buffer_ir=enable_buffer_ir
    ):
        ir.compile(
            program,
            output_dir=str(tmp_path),
            backend_type=BackendType.Ascend910,
            execution_mode=ExecutionMode.ONBOARD,
            dump_passes=True,
            skip_ptoas=True,
        )
    _assert_index_selection_dump(tmp_path, expected_results, expected_shapes, memory_space)
    kernels = list(tmp_path.rglob("*.pto"))
    assert kernels
    space = memory_space.name.lower()
    for kernel in kernels:
        for line in kernel.read_text().splitlines():
            if "pto.tmov" in line:
                assert len(re.findall(rf"loc\s*=\s*{space}(?:,|>)", line)) < 2, line
        result = subprocess.run(
            [binary, str(kernel), "-o", str(kernel.with_suffix(".checked.cpp")), *_get_ptoas_flags(planner)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr


def test_explicit_mat_storage_is_not_overwritten_by_phi_fixup(ascend_backend):
    """A caller's explicit allocation cannot be migrated into a selection pool."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[32, 16], pl.FP16], flag: pl.Scalar[pl.INDEX]
        ) -> pl.Tensor[[32, 16], pl.FP16]:
            a: pl.Tile[[16, 16], pl.FP16, pl.MemRef("pinned_a"), pl.Mem.Mat] = pl.tile.load(
                x, [0, 0], [16, 16], target_memory=pl.Mem.Mat
            )
            b: pl.Tile[[16, 16], pl.FP16, pl.MemRef("pinned_b"), pl.Mem.Mat] = pl.tile.load(
                x, [16, 0], [16, 16], target_memory=pl.Mem.Mat
            )
            if flag > 0:
                chosen = pl.yield_(a)
            else:
                chosen = pl.yield_(b)
            _selected = pl.tile.move(chosen, target_memory=pl.Mem.Left)
            _keep_a = pl.tile.move(a, target_memory=pl.Mem.Left)
            _keep_b = pl.tile.move(b, target_memory=pl.Mem.Left)
            return x

    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=False):
        initialized = passes.init_mem_ref()(passes.convert_to_ssa()(Before))
        aliases = passes.materialize_semantic_aliases()(initialized)
        with pytest.raises(ValueError, match="tile.move memory-space pair is unsupported"):
            passes.memory_reuse()(aliases)


def test_mat_selection_with_writeback_keeps_its_storage_contract(ascend_backend):
    """A selected writeback target needs ownership handling beyond load/alias selection."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, pool: pl.Tensor[[32, 16], pl.FP16], flag: pl.Scalar[pl.INDEX]
        ) -> pl.Tensor[[32, 16], pl.FP16]:
            a = pl.tile.load(pool, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            b = pl.tile.load(pool, [16, 0], [16, 16], target_memory=pl.Mem.Mat)
            if flag > 0:
                chosen = pl.yield_(a)
            else:
                chosen = pl.yield_(b)
            updated = pl.tile.gather_row(chosen, pool, [0, 0], [0, 0], [16, 16])
            _selected = pl.tile.move(updated, target_memory=pl.Mem.Left)
            _keep_a = pl.tile.move(a, target_memory=pl.Mem.Left)
            _keep_b = pl.tile.move(b, target_memory=pl.Mem.Left)
            return pool

    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=False):
        initialized = passes.init_mem_ref()(passes.convert_to_ssa()(Before))
        aliases = passes.materialize_semantic_aliases()(initialized)
        with pytest.raises(ValueError, match="tile.move memory-space pair is unsupported"):
            passes.memory_reuse()(aliases)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
