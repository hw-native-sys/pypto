# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Direct FP4 casts retain runtime tile handles and revalidate split geometry."""

import re

import pypto.language as pl
import pytest
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes

_PLANNERS = [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]


@pytest.fixture(autouse=True)
def _a5_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def _emit(program: ir.Program, planner: passes.MemoryPlanner, compile_dir=None) -> str:
    """Use the production pipeline, including legalization after vector splitting."""
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    functions = [
        function for function in optimized.functions.values() if ir.is_incore_type(function.func_type)
    ]
    assert functions, "Expected an emitted InCore kernel"
    # Mixed kernels reference their peer's reserved buffers. Assemble both
    # sides in one module so PTOAS can resolve every peer_func symbol.
    kernels = ir.Program(functions, optimized.name, optimized.span)
    source = codegen.PTOCodegen().generate(kernels, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    if compile_dir is not None and find_ptoas_binary() is not None:
        pto = compile_dir / "kernels.pto"
        cpp = compile_dir / "kernels.cpp"
        pto.write_text(source)
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(str(pto), str(cpp), ["--enable-insert-sync", f"--pto-level={level}", "--pto-arch=a5"])
        assert cpp.is_file()
    return source


def _casts(mlir: str) -> list[str]:
    return [line for line in mlir.splitlines() if "pto.tcvt " in line]


def _assert_source_handle(mlir: str, expected: str) -> None:
    """Follow only descriptor reshapes: neither an allocation nor a copy preserves ownership."""
    casts = _casts(mlir)
    assert len(casts) == 1, mlir
    assert "f4E2M1x2" in casts[0] and "f8E4M3FN" in casts[0], casts[0]
    source = re.search(r"ins\((%[\w.$-]+)", casts[0])
    assert source is not None, casts[0]
    definitions = dict(re.findall(r"^\s*(%[\w.$-]+) = (.+)$", mlir, re.MULTILINE))
    handle = source.group(1)
    visited = set()
    while handle != expected:
        assert handle not in visited, f"Descriptor cycle: {mlir}"
        visited.add(handle)
        definition = definitions.get(handle, "")
        reshape = re.match(r"pto\.treshape (%[\w.$-]+)", definition)
        assert reshape is not None, f"Cast lost runtime source {expected}: {definition}\n{mlir}"
        handle = reshape.group(1)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
def test_rotating_packed_slot_is_the_direct_cast_source(planner, tmp_path):
    """The second iteration must read slot 1 rather than rebinding slot 0's static base."""
    slots = pl.MemRef(slots=2)

    @pl.program
    class Rotating:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            packed: pl.Tensor[[64, 32], pl.fp4e2m1x2],
            output: pl.Out[pl.Tensor[[64, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[64, 64], pl.FP8E4M3FN]:
            for iteration in pl.range(4):
                source: pl.Tile[[16, 32], pl.fp4e2m1x2, slots[iteration % 2], pl.Mem.Vec] = pl.load(
                    packed, [iteration * 16, 0], [16, 32]
                )
                decoded = pl.cast(source, pl.FP8E4M3FN)
                output = pl.store(decoded, [iteration * 16, 0], output)
            return output

    mlir = _emit(Rotating, planner, tmp_path)
    load = next(line for line in mlir.splitlines() if "pto.tload " in line)
    loaded_handle = re.search(r"outs\((%[\w.$-]+)", load)
    assert loaded_handle is not None, load
    assert "scf.for" in mlir and ("arith.remsi" in mlir or "arith.remui" in mlir), mlir
    _assert_source_handle(mlir, loaded_handle.group(1))


def _runtime_source_program(kind: str) -> ir.Program:
    """Build a post-planning AIV boundary without substituting a GM load.

    This isolates handle emission. TPOP deliberately omits the matching pipe
    producer, initialization and release, so this fixture is not assembled.
    MakeTpopCodegenPTO consumes only the split/result metadata at this boundary.
    """
    span = ir.Span.unknown()
    source_memref = None
    if kind == "parameter":
        # Real planned Tile parameters carry an allocation identity, while
        # their actual address still arrives from the caller at runtime.
        source_base = ir.Var("caller_allocation", ir.PtrType(), span)
        source_memref = ir.MemRef(source_base, ir.ConstInt(0, pl.INT64, span), 512, span)
    source_type = ir.TileType(
        [16, 32], pl.fp4e2m1x2, source_memref, ir.TileView(valid_shape=[16, 32]), ir.MemorySpace.Vec
    )
    source = ir.Var("source", source_type, span)
    allocation = ir.Var("output_allocation", ir.PtrType(), span)
    destination = ir.MemRef(allocation, ir.ConstInt(8192, pl.INT64, span), 1024, span)
    result_type = ir.TileType(
        [16, 64], pl.FP8E4M3FN, destination, ir.TileView(valid_shape=[16, 64]), ir.MemorySpace.Vec
    )
    result = ir.Var("result", result_type, span)
    params = [(source, ir.ParamDirection.In)] if kind == "parameter" else []
    statements = []
    if kind == "tpop":
        pop = ir.Call(ir.get_op("tile.tpop_from_aic"), [], {"split": 0}, source_type, span)
        statements.append(ir.AssignStmt(source, pop, span))
    cast = ir.Call(
        ir.get_op("tile.cast"), [source], {"target_type": pl.FP8E4M3FN, "mode": 1}, result_type, span
    )
    statements.append(ir.AssignStmt(result, cast, span))
    function = ir.Function("consumer", params, [], ir.SeqStmts(statements, span), span, ir.FunctionType.AIV)
    return ir.Program([function], "runtime_source", span)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("kind", ["parameter", "tpop"])
def test_runtime_owned_packed_source_preserves_its_handle(planner, kind):
    """A tile parameter and TPOP have no general-pool address the cast may reconstruct."""
    program = _runtime_source_program(kind)
    if kind == "parameter" and planner == passes.MemoryPlanner.PTOAS:
        # Caller-owned UB parameters already require level3 independently of
        # TCVT. Keep that ABI limitation separate from direct-cast eligibility.
        with pytest.raises(ValueError, match="Caller-owned Tile parameters require PyPTO memory planning"):
            codegen.PTOCodegen().generate(program, emit_tile_addr=False)
        return
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
        mlir = codegen.PTOCodegen().generate(program, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    if kind == "parameter":
        address = re.search(r"(%[\w.$-]+) = pto\.castptr %arg0", mlir)
        assert address is not None, mlir
        # Exactly one materialized descriptor binds the caller's actual UB
        # pointer. Any further constant-address FP4 descriptor would lose it.
        allocations = [line for line in mlir.splitlines() if "pto.alloc_tile" in line and "f4E2M1x2" in line]
        assert len(allocations) == 1, mlir
        assert f"addr = {address.group(1)} " in allocations[0], allocations[0]
        handle = re.match(r"\s*(%[\w.$-]+) =", allocations[0])
        assert handle is not None, allocations[0]
        expected = handle.group(1)
    else:
        pop = re.search(r"(%[\w.$-]+) = pto\.tpop_from_aic", mlir)
        assert pop is not None, mlir
        expected = pop.group(1)
    _assert_source_handle(mlir, expected)
    if kind == "tpop":
        assert not any("pto.alloc_tile" in line and "f4E2M1x2" in line for line in mlir.splitlines()), mlir


def _mixed_cast_program(packed_columns: int, split_mode, partial_rows: bool = False):
    """Keep a cube result live while independently splitting the packed vector cast."""
    valid_rows = 15 if partial_rows else 16

    @pl.program
    class Mixed:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": split_mode})
        def main(
            self,
            a: pl.Tensor[[16, 32], pl.BF16],
            b: pl.Tensor[[32, 64], pl.BF16],
            packed: pl.Tensor[[16, packed_columns], pl.fp4e2m1x2],
            matrix_output: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            output: pl.Out[pl.Tensor[[16, packed_columns * 2], pl.FP8E4M3FN]],
        ) -> tuple[pl.Tensor[[16, 64], pl.FP32], pl.Tensor[[16, packed_columns * 2], pl.FP8E4M3FN]]:
            lhs = pl.load(a, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
            rhs = pl.load(b, [0, 0], [32, 64], target_memory=pl.Mem.Mat)
            product = pl.matmul(
                pl.move(lhs, target_memory=pl.Mem.Left), pl.move(rhs, target_memory=pl.Mem.Right)
            )
            vector_product = pl.move(product, target_memory=pl.Mem.Vec)
            matrix_output = pl.store(vector_product, [0, 0], matrix_output)
            source = pl.load(packed, [0, 0], [16, packed_columns], valid_shape=[valid_rows, packed_columns])
            decoded = pl.cast(source, pl.FP8E4M3FN)
            output = pl.store(decoded, [0, 0], output)
            return matrix_output, output

    return Mixed


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize(("packed_columns", "expected_casts"), [(32, 3), (64, 1)])
def test_left_right_split_revalidates_packed_row_alignment(planner, packed_columns, expected_casts, tmp_path):
    """Halving 32 carrier columns invalidates direct TCVT; halving 64 preserves its contract."""
    mlir = _emit(_mixed_cast_program(packed_columns, pl.SplitMode.LEFT_RIGHT), planner, tmp_path)
    casts = _casts(mlir)
    assert len(casts) == expected_casts, mlir
    assert f"dtype=!pto.f4E2M1x2, rows=16, cols={packed_columns // 2}" in casts[0], casts[0]
    if expected_casts == 1:
        assert "dtype=f8E4M3FN, rows=16, cols=64" in casts[0], casts[0]
    else:
        assert "dtype=bf16" in casts[0], casts[0]
        assert "dtype=f32" in casts[1] and "dtype=f8E4M3FN" in casts[2], casts


@pytest.mark.ptoas_compile
def test_partial_up_down_split_relegalizes_dynamic_valid_rows(tmp_path):
    """A 15-row tile yields lane-dependent row validity after an 8+8 row split."""
    mlir = _emit(
        _mixed_cast_program(32, pl.SplitMode.UP_DOWN, partial_rows=True), passes.MemoryPlanner.PYPTO, tmp_path
    )
    casts = _casts(mlir)
    assert len(casts) == 3, mlir
    assert "f4E2M1x2" in casts[0] and "dtype=bf16" in casts[0], casts[0]
    assert "v_row=?" in casts[0], casts[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
