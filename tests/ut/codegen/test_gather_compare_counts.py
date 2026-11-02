# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Gather count capacity and valid extents survive tuple lowering (#3079)."""

import re
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import codegen, ir, passes
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _build_program(rows, count_dtype, *, partial=False, padded_output=False):
    count_capacity = ((rows + 7) // 8) * 8
    output_columns = count_capacity if padded_output else rows
    logical_rows = 7 if partial else rows

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[rows, 64], pl.FP32],
            indices_out: pl.Out[pl.Tensor[[rows, 64], pl.INT32]],
            counts_out: pl.Out[pl.Tensor[[1, output_columns], count_dtype]],
            scalar_out: pl.Out[pl.Tensor[[1], count_dtype]],
        ):
            source = pl.load(src, [0, 0], [rows, 64], valid_shape=[logical_rows, 64])
            workspace = pl.tile.create([1, rows * 512], pl.UINT8)
            # Tuple projection must retain the independently sized count tile.
            indices, counts = pl.tile.gather_compare(
                source, pl.const(0.5, pl.FP32), workspace, cmp_mode="gt", out_cols=64, count_dtype=count_dtype
            )
            value = pl.tile.read(counts, [0, logical_rows - 1])
            pl.write(scalar_out, [0], value)
            pl.store(indices, [0, 0], indices_out)
            pl.store(counts, [0, 0], counts_out)

    return Program


def _build_dynamic_valid_program():
    """Use a distinct signature so the dynamic extent is an actual IR parameter."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[9, 64], pl.FP32],
            n: pl.Scalar[pl.INDEX],
            indices_out: pl.Out[pl.Tensor[[9, 64], pl.INT32]],
            counts_out: pl.Out[pl.Tensor[[1, 9], pl.INT32]],
            scalar_out: pl.Out[pl.Tensor[[1], pl.INT32]],
        ):
            source = pl.load(src, [0, 0], [9, 64], valid_shape=[n, 64])
            workspace = pl.tile.create([1, 9 * 512], pl.UINT8)
            indices, counts = pl.tile.gather_compare(
                source, pl.const(0.5, pl.FP32), workspace, cmp_mode="gt", out_cols=64
            )
            value = pl.tile.read(counts, [0, 0])
            pl.write(scalar_out, [0], value)
            pl.store(indices, [0, 0], indices_out)
            pl.store(counts, [0, 0], counts_out)

    return Program


def _emit_pto(program, planner) -> str:
    # Run the production pipeline: this also covers tuple buffer allocation,
    # type reinference, memory reuse, and allocation-size propagation.
    with passes.PassContext([], memory_planner=planner):
        lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    functions = [
        func for func in lowered.functions.values() if func.func_type != pl.FunctionType.Orchestration
    ]
    assert len(functions) == 1
    kernel = ir.Program(functions, "GatherCompareCounts", lowered.span)
    return codegen.PTOCodegen().generate(kernel, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)


def _sole_line(mlir: str, fragment: str) -> str:
    lines = [line for line in mlir.splitlines() if fragment in line]
    assert len(lines) == 1, f"expected one {fragment!r} line, got {lines}:\n{mlir}"
    return lines[0]


def _count_destination(mlir: str) -> str:
    gather = _sole_line(mlir, "pto.tgather ins(")
    match = re.search(r"outs\(%[\w]+, (%[\w]+) :", gather)
    assert match, gather
    return match.group(1)


def _assert_count_storage_and_consumers(
    mlir: str, rows: int, valid_cols: str, *, with_read: bool = True
) -> str:
    count_name = _count_destination(mlir)
    allocation = _sole_line(mlir, f"{count_name} = pto.alloc_tile")
    assert f"rows=1, cols={((rows + 7) // 8) * 8}," in allocation, allocation
    assert "blayout=row_major" in allocation, allocation
    if with_read:
        read = _sole_line(mlir, "pto.tgetval ins(")
        assert f"pto.tgetval ins({count_name}," in read, read
        if valid_cols != "?":
            offset = re.search(r"pto.tgetval ins\(%[\w]+, (%[\w]+) :", read)
            assert offset, read
            definition = _sole_line(mlir, f"{offset.group(1)} = arith.constant")
            assert f"arith.constant {int(valid_cols) - 1} : index" in definition, definition
    store = _sole_line(mlir, f"pto.tstore ins({count_name} :")
    # Store the true count extent even when UB allocation has spare capacity
    # and the GM output has only logical capacity. Never store the UB padding.
    assert f"!pto.partition_tensor_view<1x{valid_cols}x" in store, store
    return allocation


@pytest.mark.parametrize("rows", [1, 2, 7, 8, 9, 16])
@pytest.mark.parametrize("count_dtype", [pl.INT32, pl.UINT32])
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_counts_have_aligned_storage_and_logical_store_extent(ascend_backend, rows, count_dtype, planner):
    """Both public count dtypes retain logical validity through the full pipeline."""
    mlir = _emit_pto(_build_program(rows, count_dtype), planner)
    _assert_count_storage_and_consumers(mlir, rows, str(rows))


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("rows", [1, 2, 7, 8, 9, 16])
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_aligned_count_storage_assembles_with_ptoas(ascend_backend, rows, planner, tmp_path: Path):
    """Nonmultiples of eight fail PTOAS allocation verification before the fix."""
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    # PTOAS currently requires counts to match the INT32 index destination's
    # dtype. UINT32 remains covered by type and codegen tests independently of
    # that backend restriction; physical count alignment is the same for both.
    mlir = _emit_pto(_build_program(rows, pl.INT32), planner)
    _assert_count_storage_and_consumers(mlir, rows, str(rows))
    source, output = tmp_path / "gather_counts.pto", tmp_path / "gather_counts.cpp"
    source.write_text(mlir)
    # Level 3 accepts PyPTO's explicit addresses; level 2 runs PTOAS PlanMemory.
    level = "level3" if planner == passes.MemoryPlanner.PYPTO else "level2"
    _run_ptoas(str(source), str(output), ["--pto-arch=a2", f"--pto-level={level}"])
    assert output.is_file()


def test_padding_caller_output_does_not_determine_internal_count_capacity(ascend_backend):
    """The issue's preallocated [1,8] output must still contain one valid count."""
    mlir = _emit_pto(_build_program(1, pl.INT32, padded_output=True), passes.MemoryPlanner.PYPTO)
    _assert_count_storage_and_consumers(mlir, 1, "1")


@pytest.mark.parametrize("dynamic_valid", [False, True])
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_partial_source_keeps_static_capacity_and_valid_count_extent(ascend_backend, dynamic_valid, planner):
    """Nine physical source rows can contain seven or runtime-selected valid rows."""
    program = _build_dynamic_valid_program() if dynamic_valid else _build_program(9, pl.INT32, partial=True)
    mlir = _emit_pto(program, planner)
    allocation = _assert_count_storage_and_consumers(mlir, 9, "?" if dynamic_valid else "7")
    valid_col = re.search(r"valid_col = (%[\w]+)", allocation)
    assert valid_col, allocation
    if dynamic_valid:
        # The function parameter must size count validity, while physical cols
        # remain the static capacity 16 required by memory planning.
        signature = _sole_line(mlir, "func.func @kernel(")
        parameter = re.search(r"(%[\w]+)\s*:\s*index", signature)
        assert parameter, signature
        assert valid_col.group(1) == parameter.group(1), allocation
    else:
        definition = _sole_line(mlir, f"{valid_col.group(1)} = arith.constant")
        assert "arith.constant 7 : index" in definition, definition


@pytest.mark.parametrize("rows", [1, 9])
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_tensor_gather_keeps_logical_count_shape_when_lowering_to_aligned_tiles(
    ascend_backend, rows, planner
):
    """Tensor-to-tile tuple conversion pads storage, not public tensor dimensions."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[rows, 64], pl.FP32],
            threshold: pl.Scalar[pl.FP32],
        ) -> tuple[pl.Tensor[[rows, 64], pl.INT32], pl.Tensor[[1, rows], pl.INT32]]:
            indices, counts = pl.tensor.gather(src, kvalue=threshold, cmp_mode="gt", out_cols=64)
            return indices, counts

    public_function = next(iter(Program.functions.values()))
    public_count_type = public_function.return_types[1]
    assert isinstance(public_count_type, ir.TensorType)
    assert len(public_count_type.shape) == 2
    for dim, expected in zip(public_count_type.shape, [1, rows]):
        assert isinstance(dim, ir.ConstInt)
        assert dim.value == expected
    mlir = _emit_pto(Program, planner)
    _assert_count_storage_and_consumers(mlir, rows, str(rows), with_read=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
