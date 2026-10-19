# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cube tiles cross the Buffer boundary as explicit Mat/Left/Right/Acc storage and writes."""

import re
from collections import Counter
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from buffer_pipeline_utils import buffer_recipe_pipeline
from pypto import DataType, ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.debug import torch_codegen
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen

from .buffer_test_utils import statements

_PLANNERS = (passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS)
_BACKENDS = [BackendType.Ascend910, BackendType.Ascend950]
pytestmark = pytest.mark.usefixtures("ascend_backend")


def _lower(program: ir.Program, planner: passes.MemoryPlanner) -> ir.Program:
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        lowered = buffer_recipe_pipeline().run(program)
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
    arch = "a2" if backend == BackendType.Ascend910 else "a5"
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
    text = _compile_native(tmp_path, lowered, BackendType.Ascend910, planner != passes.MemoryPlanner.PTOAS)
    assert text.count("scf.if") == branches


def _wider_accumulator(init_cond: str, valid_columns: int = 24) -> ir.Program:
    return pl.parse_program(f"""
@pl.program
class WiderAccumulator:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, lhs: pl.Tensor[[16, 16], pl.FP16], seed: pl.Tensor[[16, 32], pl.FP16],
               partial: pl.Tensor[[16, 32], pl.FP16], flag: pl.Scalar[pl.INDEX],
               output: pl.Out[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        lhs_mat: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.load(
            lhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
        seed_mat: pl.Tile[[16, 32], pl.FP16, pl.MemorySpace.Mat] = pl.load(
            seed, [0, 0], [16, 32], target_memory=pl.MemorySpace.Mat)
        partial_mat: pl.Tile[
            [16, 32], pl.FP16, pl.MemorySpace.Mat, pl.TileView(valid_shape=[16, {valid_columns}])
        ] = pl.load(partial, [0, 0], [16, 32], valid_shape=[16, {valid_columns}],
                    target_memory=pl.MemorySpace.Mat)
        lhs_left = pl.move(lhs_mat, target_memory=pl.MemorySpace.Left)
        seed_right = pl.move(seed_mat, target_memory=pl.MemorySpace.Right)
        partial_right = pl.move(partial_mat, target_memory=pl.MemorySpace.Right)
        acc = pl.matmul(lhs_left, seed_right)
        result = pl.matmul_acc(acc, lhs_left, partial_right{init_cond})
        return pl.store(result, [0, 0], output)
""")


@pytest.mark.parametrize("ascend_backend", _BACKENDS, indirect=True)
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize(
    "init_cond,initializing,accumulating",
    [("", 1, 1), (", init_cond=True", 2, 0), (", init_cond=False", 1, 1), (", init_cond=(flag == 0)", 2, 1)],
)
def test_wider_accumulator_is_written_through_a_product_shaped_view(
    tmp_path: Path,
    ascend_backend: BackendType,
    planner: passes.MemoryPlanner,
    init_cond: str,
    initializing: int,
    accumulating: int,
) -> None:
    # Accumulator valid [16, 32], product [16, 24]: legal for tile.matmul_acc in
    # every form. PTOAS requires tmatmul[.acc] destinations to match the
    # product, so the write goes through a same-storage [16, 24] view.
    lowered = _lower(_wider_accumulator(init_cond), planner)
    names = _call_names(lowered)
    assert (names["buffer.matmul"], names["buffer.matmul_acc"]) == (initializing, accumulating)
    addressed = planner != passes.MemoryPlanner.PTOAS
    accumulators = [(d, call) for d, call in _allocations(lowered) if d.memory_space == ir.MemorySpace.Acc]
    assert sorted(list(d.valid_shape) for d, _ in accumulators) == (
        [[16, 24], [16, 32]] if addressed else [[16, 32]]
    )
    if addressed:
        assert len({str(call.args[1]) for _, call in accumulators}) == 1
    assert names["buffer.reshape"] == (0 if addressed else 1)
    text = _compile_native(tmp_path, lowered, ascend_backend, addressed)
    for line in text.splitlines():
        if "pto.tmatmul" in line:
            inputs, output = line.split("outs(", 1)
            # Every native destination matches its product: rhs is the last input.
            assert re.findall(r"v_col=(\d+)", inputs)[-1] == re.findall(r"v_col=(\d+)", output)[0], line


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
    text = _compile_native(tmp_path, lowered, BackendType.Ascend910, addressed)
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


_SPAN = ir.Span.unknown()


def _index(value: int) -> ir.ConstInt:
    return ir.ConstInt(value, DataType.INDEX, _SPAN)


class _PlannedMatrixKernel:
    """Hand-placed post-planning IR, for layouts the DSL pipeline never produces."""

    def __init__(self) -> None:
        self.output = ir.Var("output", ir.TensorType([16, 32], DataType.FP32), _SPAN)
        self.body: list[ir.Stmt] = []

    def allocation(self, name: str, space: ir.MemorySpace, capacity: int) -> ir.Var:
        base = ir.Var(name, ir.PtrType(), _SPAN)
        call = ir.Call(ir.get_op("tile.alloc"), [_index(space.value), _index(capacity)], base.type, _SPAN)
        self.body.append(ir.AssignStmt(base, call, _SPAN))
        return base

    def tile(
        self,
        name: str,
        base: ir.Var,
        offset: int,
        shape: tuple[int, int],
        dtype: DataType,
        space: ir.MemorySpace,
    ) -> ir.Var:
        size = shape[0] * shape[1] * dtype.get_bit() // 8
        memory = ir.MemRef(base, offset, size, _SPAN, is_pinned=False)
        var = ir.Var(name, ir.TileType([_index(d) for d in shape], dtype, memory, None, space), _SPAN)
        created = ir.Call(
            ir.get_op("tile.create"),
            [ir.MakeTuple([_index(d) for d in shape], _SPAN)],
            {"dtype": dtype, "target_memory": space},
            var.type,
            _SPAN,
        )
        self.body.append(ir.AssignStmt(var, created, _SPAN))
        return var

    def program(self) -> ir.Program:
        body = [*self.body, ir.ReturnStmt([self.output], _SPAN)]
        function = ir.Function(
            "kernel",
            [(self.output, ir.ParamDirection.Out)],
            [self.output.type],
            ir.SeqStmts(body, _SPAN),
            _SPAN,
            type=ir.FunctionType.InCore,
        )
        return ir.Program([function], "PlannedMatrix", _SPAN)


def _lower_planned(program: ir.Program, planner: passes.MemoryPlanner) -> ir.Program:
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        return passes.lower_tile_to_buffer()(program)


def test_addressless_matrix_root_rejects_a_second_window_at_another_offset():
    # The first member visited sits at offset 4096; an equal-size member at 0
    # is a different window and must not become a reshape of the first.
    kernel = _PlannedMatrixKernel()
    base = kernel.allocation("mat", ir.MemorySpace.Mat, 8192)
    kernel.tile("upper", base, 4096, (32, 64), DataType.FP16, ir.MemorySpace.Mat)
    kernel.tile("lower", base, 0, (32, 64), DataType.FP16, ir.MemorySpace.Mat)
    with pytest.raises(ValueError, match="can hold only one window"):
        _lower_planned(kernel.program(), passes.MemoryPlanner.PTOAS)


def test_matmul_kwargs_are_rejected_rather_than_dropped():
    kernel = _PlannedMatrixKernel()
    lhs = kernel.tile(
        "lhs",
        kernel.allocation("left", ir.MemorySpace.Left, 2048),
        0,
        (16, 64),
        DataType.FP16,
        ir.MemorySpace.Left,
    )
    rhs = kernel.tile(
        "rhs",
        kernel.allocation("right", ir.MemorySpace.Right, 4096),
        0,
        (64, 32),
        DataType.FP16,
        ir.MemorySpace.Right,
    )
    acc_base = kernel.allocation("acc", ir.MemorySpace.Acc, 2048)
    product = ir.Var(
        "product",
        ir.TileType(
            [_index(16), _index(32)],
            DataType.FP32,
            ir.MemRef(acc_base, 0, 2048, _SPAN, is_pinned=False),
            None,
            ir.MemorySpace.Acc,
        ),
        _SPAN,
    )
    call = ir.Call(ir.get_op("tile.matmul"), [lhs, rhs], {"unmodelled": 1}, product.type, _SPAN)
    kernel.body.append(ir.AssignStmt(product, call, _SPAN))
    with pytest.raises(ValueError, match="unmodelled"):
        _lower_planned(kernel.program(), passes.MemoryPlanner.PTOAS)


def _default_matrix(program, planner, enabled=None):
    options = {} if enabled is None else {"enable_buffer_ir": enabled}
    with passes.PassContext([], memory_planner=planner, **options):
        result = PassManager(OptimizationStrategy.Default).run_passes(program)
    kernels = [f for f in result.functions.values() if ir.is_incore_type(f.func_type)]
    assert kernels and all(
        f.ir_stage == (ir.FunctionIRStage.Functional if enabled is False else ir.FunctionIRStage.Buffer)
        for f in kernels
    )
    namespace = {}
    if enabled is not False:
        restored = ir.deserialize(ir.serialize(result))
        assert isinstance(restored, ir.Program)
        ir.assert_structural_equal(result, restored, enable_auto_mapping=True)
        exec(torch_codegen(restored), namespace)  # noqa: S102
    return namespace


def _matrix_data(rows, cols, dtype, modulus=7):
    values = torch.arange(rows * cols).reshape(rows, cols).remainder(modulus) - modulus // 2
    return values.to(dtype) if dtype == torch.int8 else values.to(dtype) / 4


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("enabled", [None, True, False])
@pytest.mark.parametrize(
    "operand,dtype",
    [("FP16", torch.float16), ("BF16", torch.bfloat16), ("FP32", torch.float32), ("INT8", torch.int8)],
)
def test_default_cube_debug_uses_explicit_writes(planner, enabled, operand, dtype):
    accumulator = "INT32" if dtype == torch.int8 else "FP32"
    namespace = _default_matrix(_explicit_matmul(operand, accumulator), planner, enabled)
    if enabled is False:
        return
    a, b = _matrix_data(64, 64, dtype), _matrix_data(64, 64, dtype, 5)
    acc_dtype = torch.int32 if dtype == torch.int8 else torch.float32
    output = torch.full((64, 64), -777, dtype=acc_dtype)
    namespace["kernel"](a, b, output)
    torch.testing.assert_close(output, 2 * (a.to(acc_dtype) @ b.to(acc_dtype)), rtol=0, atol=0)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("transpose", [False, True])
def test_default_cube_debug_extracts_and_relabels_shared_storage(planner, transpose):
    namespace = _default_matrix(TransposedOperand if transpose else AutoTiledMatmul, planner)
    a = _matrix_data(16 if transpose else 128, 128 if transpose else 512, torch.bfloat16)
    b = _matrix_data(128 if transpose else 512, 128, torch.bfloat16, 5)
    output = torch.full((a.shape[0], 128), -777.0)
    namespace["kernel"](a, b, output)
    torch.testing.assert_close(output, a.float() @ (b.float().T if transpose else b.float()), rtol=0, atol=0)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize(
    "init_cond", ["", ", init_cond=True", ", init_cond=False", ", init_cond=(flag == 0)"]
)
@pytest.mark.parametrize("valid_columns", [16, 24])
def test_default_cube_debug_product_view_preserves_untouched_boxes(planner, init_cond, valid_columns):
    # The device writes whole Acc boxes: only complete untouched boxes are asserted.
    original = _wider_accumulator(init_cond, valid_columns)
    namespace = _default_matrix(original, planner)
    lhs = _matrix_data(16, 16, torch.float16)
    seed = _matrix_data(16, 32, torch.float16, 5)
    partial = _matrix_data(16, 32, torch.float16, 3)
    first = lhs.float() @ seed.float()
    second = lhs.float() @ partial[:, :valid_columns].float()
    for flag in (0, 1):
        overwrite = init_cond == ", init_cond=True" or (init_cond.endswith("(flag == 0)") and flag == 0)
        output = torch.full((16, 32), -777.0)
        namespace["kernel"](lhs, seed, partial, flag, output)
        torch.testing.assert_close(
            output[:, :valid_columns],
            second if overwrite else first[:, :valid_columns] + second,
            rtol=0,
            atol=0,
        )

        if valid_columns == 16:
            torch.testing.assert_close(output[:, 16:], first[:, 16:], rtol=0, atol=0)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_default_cube_debug_runtime_initialization(planner):
    namespace = _default_matrix(_conditional_accumulation("(k0 == 0)"), planner)
    a, b = _matrix_data(16, 64, torch.float32), _matrix_data(64, 16, torch.float32, 5)
    output = torch.full((16, 16), -777.0)
    namespace["kernel"](a, b, output)
    torch.testing.assert_close(output, a @ b, rtol=0, atol=0)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_default_matrix_carry_rejects_storage_swaps_before_planning(planner):
    original = pl.parse_program("""
@pl.program
class MatrixSwap:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 16], pl.FP16], b: pl.Tensor[[16, 16], pl.FP16],
               out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        lhs = pl.load(a, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
        rhs = pl.load(b, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
        first = pl.matmul(lhs, rhs)
        second = pl.matmul(lhs, rhs)
        for i, (left, right) in pl.range(3, init_values=(first, second)):
            final_left, final_right = pl.yield_(right, left)
        return pl.store(final_left, [0, 0], out)
""")
    with passes.PassContext([], memory_planner=planner):
        manager = PassManager(OptimizationStrategy.Default)
        boundary = manager.get_pass_names().index("SelectBufferIRFunctions")
        for transform in manager.passes[: boundary + 1]:
            original = transform(original)
    kernel = original.get_function("kernel")
    assert kernel is not None and kernel.attrs["buffer_ir_selected"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
