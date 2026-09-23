# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cube tiles cross the Buffer boundary as explicit Mat/Left/Right/Acc storage and writes."""

from collections import Counter
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen

from .buffer_test_utils import statements

_PLANNERS = (passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS)
_BACKENDS = [BackendType.Ascend910B, BackendType.Ascend950]
pytestmark = pytest.mark.usefixtures("ascend_backend")


def _lower(program: ir.Program, planner: passes.MemoryPlanner) -> ir.Program:
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    restored = ir.deserialize(ir.serialize(lowered))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(lowered, restored, enable_auto_mapping=True)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.BufferIR)
    assert passes.PropertyVerifierRegistry.verify(properties, restored) == []
    return restored


def _call_names(program: ir.Program) -> Counter[str]:
    names: Counter[str] = Counter()
    for statement in statements(program):
        expr = statement.value if isinstance(statement, ir.AssignStmt) else getattr(statement, "expr", None)
        if isinstance(expr, ir.Call):
            names[expr.op.name] += 1
    return names


def _allocations(program: ir.Program) -> list[tuple[ir.BufferType, ir.Call]]:
    """(descriptor, allocation call) for every buffer.alloc, in source order."""
    result = []
    for statement in statements(program):
        if isinstance(statement, ir.AssignStmt) and isinstance(statement.value, ir.Call):
            if statement.value.op.name == ir.get_op("buffer.alloc").name:
                assert isinstance(statement.var.type, ir.BufferType)
                result.append((statement.var.type, statement.value))
    return result


def _compile_native(tmp_path: Path, program: ir.Program, backend: BackendType, addressed: bool) -> str:
    # PTOCodegen accepts device functions only; pl.at outlining leaves an orchestration parent.
    device = [function for function in program.functions.values() if ir.is_incore_type(function.func_type)]
    program = ir.Program(device, program.name, program.span)
    text = codegen.PTOCodegen().generate(program, emit_tile_addr=False, emit_source_loc=False)
    text = text if isinstance(text, str) else "".join(text.values())
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    source, output = tmp_path / "matrix.pto", tmp_path / "matrix.cpp"
    source.write_text(text)
    arch = "a2" if backend == BackendType.Ascend910B else "a5"
    _run_ptoas(
        str(source), str(output), [f"--pto-arch={arch}", f"--pto-level={'level3' if addressed else 'level2'}"]
    )
    return text


def _explicit_matmul(operand: str, accumulator: str) -> ir.Program:
    return pl.parse_program(f"""
@pl.program
class ExplicitMatmul:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 64], pl.{operand}], b: pl.Tensor[[64, 64], pl.{operand}],
               c: pl.Out[pl.Tensor[[64, 64], pl.{accumulator}]]) -> pl.Tensor[[64, 64], pl.{accumulator}]:
        a_mat = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        b_mat = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
        b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
        product = pl.matmul(a_left, b_right)
        total = pl.matmul_acc(product, a_left, b_right)
        result = pl.store(total, [0, 0], c)
        return result
""")


@pytest.mark.parametrize("ascend_backend", _BACKENDS, indirect=True)
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize(
    "operand,accumulator", [("FP16", "FP32"), ("BF16", "FP32"), ("FP32", "FP32"), ("INT8", "INT32")]
)
def test_cube_pipeline_lowers_to_explicit_matrix_storage_and_writes(
    tmp_path: Path, ascend_backend: BackendType, planner: passes.MemoryPlanner, operand: str, accumulator: str
) -> None:
    lowered = _lower(_explicit_matmul(operand, accumulator), planner)
    names = _call_names(lowered)
    assert names["buffer.load"] == 2 and names["buffer.copy"] == 2 and names["buffer.store"] == 1
    assert names["buffer.matmul"] == 1 and names["buffer.matmul_acc"] == 1
    assert not any(name.startswith("tile.") for name in names)
    spaces = Counter(descriptor.memory_space for descriptor, _ in _allocations(lowered))
    assert spaces == Counter(
        {ir.MemorySpace.Mat: 2, ir.MemorySpace.Left: 1, ir.MemorySpace.Right: 1, ir.MemorySpace.Acc: 1}
    )
    addressed = planner != passes.MemoryPlanner.PTOAS
    assert all((len(call.args) == 2) == addressed for _, call in _allocations(lowered))
    text = _compile_native(tmp_path, lowered, ascend_backend, addressed)
    assert text.count("pto.tmatmul ins(") == 1 and text.count("pto.tmatmul.acc ins(") == 1
    assert text.count("pto.tmov ins(") == 2 and "loc=acc" in text and "fractal=1024" in text


def _conditional_accumulation(init_cond: str) -> ir.Program:
    return pl.parse_program(f"""
@pl.program
class ConditionalAccumulation:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, lhs: pl.Tensor[[16, 64], pl.FP32], rhs: pl.Tensor[[64, 16], pl.FP32],
               output: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        acc: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Acc] = pl.tile.create(
            [16, 16], pl.FP32, target_memory=pl.MemorySpace.Acc)
        for k0 in pl.range(0, 64, 16):
            a: pl.Tile[[16, 16], pl.FP32] = pl.load(lhs, [0, k0], [16, 16], target_memory=pl.MemorySpace.Mat)
            b: pl.Tile[[16, 16], pl.FP32] = pl.load(rhs, [k0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
            acc = pl.matmul_acc(acc, a, b, init_cond={init_cond})
        return pl.store(acc, [0, 0], output)
""")


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize(
    "init_cond,initializing,accumulating,branches",
    [("True", 1, 0, 0), ("False", 0, 1, 0), ("(k0 == 0)", 1, 1, 1)],
)
def test_init_cond_selects_explicit_initializing_and_accumulating_writes(
    tmp_path: Path,
    planner: passes.MemoryPlanner,
    init_cond: str,
    initializing: int,
    accumulating: int,
    branches: int,
) -> None:
    lowered = _lower(_conditional_accumulation(init_cond), planner)
    names = _call_names(lowered)
    assert (names["buffer.matmul"], names["buffer.matmul_acc"]) == (initializing, accumulating)
    conditionals = [statement for statement in statements(lowered) if isinstance(statement, ir.IfStmt)]
    assert len(conditionals) == branches
    for branch in conditionals:
        assert branch.else_body is not None and not branch.return_vars
    text = _compile_native(tmp_path, lowered, BackendType.Ascend910B, planner != passes.MemoryPlanner.PTOAS)
    assert text.count("scf.if") == branches


@pl.program
class AutoTiledMatmul:
    @pl.function
    def kernel(
        self,
        a: pl.Tensor[[128, 512], pl.BF16],
        b: pl.Tensor[[512, 128], pl.BF16],
        out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="tiled"):
            out[:, :] = pl.matmul(a[:, :], b[:, :], out_dtype=pl.FP32)
        return out


@pytest.mark.parametrize("ascend_backend", _BACKENDS, indirect=True)
@pytest.mark.parametrize("planner", _PLANNERS)
def test_auto_tiled_matmul_extracts_l0_windows_from_mat(
    tmp_path: Path, ascend_backend: BackendType, planner: passes.MemoryPlanner
) -> None:
    lowered = _lower(AutoTiledMatmul, planner)
    names = _call_names(lowered)
    assert names["buffer.extract"] >= 2 and names["buffer.matmul"] + names["buffer.matmul_acc"] >= 2
    text = _compile_native(tmp_path, lowered, ascend_backend, planner != passes.MemoryPlanner.PTOAS)
    assert "pto.textract ins(" in text


@pl.program
class TransposedOperand:
    @pl.function
    def kernel(
        self,
        q: pl.Tensor[[16, 128], pl.BF16],
        k: pl.Tensor[[128, 128], pl.BF16],
        out: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
    ) -> pl.Tensor[[16, 128], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk"):
            out[:, :] = pl.matmul(q[:, :], k[:, :], b_trans=True, out_dtype=pl.FP32)
        return out


@pytest.mark.parametrize("planner", _PLANNERS)
def test_transposed_operand_relabels_its_mat_window_without_a_copy(
    tmp_path: Path, planner: passes.MemoryPlanner
) -> None:
    lowered = _lower(TransposedOperand, planner)
    addressed = planner != passes.MemoryPlanner.PTOAS
    mats = [(d, call) for d, call in _allocations(lowered) if d.memory_space == ir.MemorySpace.Mat]
    names = _call_names(lowered)
    if addressed:
        # The planner placed both descriptors: one allocation each, same address.
        address_args = [call.args[1] for _, call in mats]
        assert all(isinstance(arg, ir.ConstInt) for arg in address_args)
        addresses = {arg.value for arg in address_args if isinstance(arg, ir.ConstInt)}
        layouts = {(d.blayout, d.slayout) for d, _ in mats}
        assert len(mats) == 3 and len(layouts) == 2 and names["buffer.reshape"] == 0
        assert len(addresses) == 2
    else:
        assert len(mats) == 2 and names["buffer.reshape"] == 1
    text = _compile_native(tmp_path, lowered, BackendType.Ascend910B, addressed)
    assert text.count("pto.treshape") == (0 if addressed else 1)


def test_fixpipe_quantized_store_is_rejected_until_its_transfer_recipe_exists():
    program = pl.parse_program("""
@pl.program
class QuantizedStore:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 16], pl.INT8], b: pl.Tensor[[16, 16], pl.INT8],
               c: pl.Out[pl.Tensor[[16, 16], pl.FP16]]) -> pl.Tensor[[16, 16], pl.FP16]:
        a_mat = pl.load(a, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
        b_mat = pl.load(b, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
        product = pl.matmul(pl.move(a_mat, target_memory=pl.MemorySpace.Left),
                            pl.move(b_mat, target_memory=pl.MemorySpace.Right))
        result = pl.tile.store(product, [0, 0], c, pre_quant=0.5)
        return result
""")
    with pytest.raises(ValueError, match="pre_quant/pre_relu stores require a Buffer transfer recipe"):
        _lower(program, passes.MemoryPlanner.PYPTO)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
