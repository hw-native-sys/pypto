# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Migrate existing compute contracts with live inputs and explicit scratch."""

import pypto.language as pl
import pytest
import torch
from buffer_pipeline_utils import buffer_recipe_pipeline
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.debug import torch_codegen
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.language.parser.diagnostics.exceptions import InvalidOperationError
from pypto.pypto_core import codegen

pytestmark = pytest.mark.usefixtures("ascend_backend")
_PLANNERS = (passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS)
_ROWS = (("min", "add"), ("prod", "mul"), ("sum", "max"), ("max", "min"))


def _lower(program, planner, enabled):
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enabled):
        return PassManager(OptimizationStrategy.Default).run_passes(program)


def _native(tmp_path, lowered, planner, enabled, arch="a2"):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    text = codegen.PTOCodegen().generate(
        lowered, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS, emit_source_loc=False
    )
    source = tmp_path / f"{'buffer' if enabled else 'tile'}.pto"
    source.write_text(text)
    output = source.with_suffix(".cpp")
    level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
    _run_ptoas(str(source), str(output), [f"--pto-arch={arch}", f"--pto-level={level}"])
    assert output.is_file()
    return text


def _row_program(reduction, broadcast, valid=(16, 64), dtype="FP32", extra_scratch=False):
    row, col = valid
    scalar = "cols: pl.Scalar[pl.INDEX], " if isinstance(col, str) else ""
    tmp = ", scratch" if extra_scratch else ""
    return pl.parse_program(f"""
@pl.program
class RowCompute:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, {scalar}a: pl.Tensor[[16, 64], pl.{dtype}],
               out: pl.InOut[pl.Tensor[[32, 64], pl.{dtype}]]) -> pl.Tensor[[32, 64], pl.{dtype}]:
        source = pl.load(a, [0, 0], [16, 64], valid_shape=[{row}, {col}])
        scratch = pl.create_tile([16, 64], dtype=pl.{dtype}, target_memory=pl.MemorySpace.Vec)
        column = pl.tile.row_{reduction}(source, scratch)
        result = pl.tile.row_expand_{broadcast}(source, column{tmp})
        out = pl.store(result, [0, 0], out)
        return pl.store(source, [16, 0], out)
""")


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("valid", [(16, 64), (13, 57)])
@pytest.mark.parametrize("reduction,broadcast", _ROWS)
def test_default_row_compute_matches_tile_and_native(tmp_path, planner, valid, reduction, broadcast):
    source = torch.rand((16, 64), generator=torch.Generator().manual_seed(28)) * 0.02 + 0.99
    rows, cols = valid
    active = source[:rows, :cols]
    column = {"min": torch.amin, "prod": torch.prod, "sum": torch.sum, "max": torch.amax}[reduction](
        active, dim=-1, keepdim=True
    )
    expected = torch.full((32, 64), -777.0)
    expected[:rows, :cols] = {"add": torch.add, "mul": torch.mul, "max": torch.maximum, "min": torch.minimum}[
        broadcast
    ](active, column)
    expected[16 : 16 + rows, :cols] = active
    for enabled in (False, True):
        lowered = _lower(_row_program(reduction, broadcast, valid), planner, enabled)
        kernel = lowered.get_function("kernel")
        assert kernel is not None
        assert (kernel.ir_stage == ir.FunctionIRStage.Buffer) == enabled
        restored = ir.deserialize(ir.serialize(lowered))
        assert isinstance(restored, ir.Program)
        ir.assert_structural_equal(lowered, restored, enable_auto_mapping=True)
        # Legacy Torch arithmetic drops Tile valid metadata and reduces padded
        # cells. Partial Tile behavior is checked by native/device tests; Buffer
        # replay carries explicit valid descriptors and must preserve sentinels.
        if enabled or valid == (16, 64):
            namespace = {}
            exec(torch_codegen(restored), namespace)  # noqa: S102
            output = torch.full_like(expected, -777.0)
            namespace["kernel"](source, output)
            torch.testing.assert_close(output, expected)
        # The second store observes the old input after scratch and destination
        # writes, while sentinel cells expose any enlarged valid rectangle.
        text = _native(tmp_path, restored, planner, enabled)
        assert text.count(f"pto.trow{reduction} ins(") == 1
        assert text.count(f"pto.trowexpand{broadcast} ins(") == 1


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("kind", ["fp16", "dynamic", "optional_scratch"])
def test_new_row_selection_preserves_unverified_forms(planner, kind):
    program = _row_program(
        "min",
        "add",
        (16, "cols" if kind == "dynamic" else 64),
        "FP16" if kind == "fp16" else "FP32",
        extra_scratch=kind == "optional_scratch",
    )
    kernel = _lower(program, planner, True).get_function("kernel")
    assert kernel is not None and kernel.ir_stage != ir.FunctionIRStage.Buffer


_COLUMNS = (("sum", "add"), ("max", "sub"), ("min", "mul"), ("prod", "div"), ("max", "max"), ("min", "min"))


def _column_program(reduction, broadcast, valid=(16, 64), dtype="FP32", extra_scratch=False, preserve=True):
    rows, cols = valid
    scalar = "rows: pl.Scalar[pl.INDEX], " if isinstance(rows, str) else ""
    scratch = (
        f"scratch = pl.create_tile([16, 64], dtype=pl.{dtype}, target_memory=pl.MemorySpace.Vec)"
        if extra_scratch
        else ""
    )
    extra = ", scratch" if extra_scratch else ""
    old = "out = pl.store(source, [16, 0], out)" if preserve else ""
    return pl.parse_program(f"""
@pl.program
class ColumnCompute:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, {scalar}a: pl.Tensor[[16, 64], pl.{dtype}],
               out: pl.InOut[pl.Tensor[[32, 64], pl.{dtype}]]) -> pl.Tensor[[32, 64], pl.{dtype}]:
        source = pl.load(a, [0, 0], [16, 64], valid_shape=[{rows}, {cols}])
        {scratch}
        row = pl.tile.col_{reduction}(source{extra})
        result = pl.tile.col_expand_{broadcast}(source, row)
        out = pl.store(result, [0, 0], out)
        {old}
        return out
""")


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("valid", [(16, 64), (13, 57)])
@pytest.mark.parametrize("reduction,broadcast", _COLUMNS)
@pytest.mark.parametrize("preserve", [False, True])
def test_default_column_compute_matches_tile_and_native(
    tmp_path, planner, valid, reduction, broadcast, preserve
):
    source = torch.rand((16, 64), generator=torch.Generator().manual_seed(29)) * 0.02 + 0.99
    rows, cols = valid
    active = source[:rows, :cols]
    row = {"min": torch.amin, "prod": torch.prod, "sum": torch.sum, "max": torch.amax}[reduction](
        active, dim=0, keepdim=True
    )
    expected = torch.full((32, 64), -777.0)
    expected[:rows, :cols] = {
        "add": torch.add,
        "sub": torch.sub,
        "mul": torch.mul,
        "div": torch.div,
        "max": torch.maximum,
        "min": torch.minimum,
    }[broadcast](active, row)
    if preserve:
        expected[16 : 16 + rows, :cols] = active
    for enabled in (False, True):
        lowered = _lower(_column_program(reduction, broadcast, valid, preserve=preserve), planner, enabled)
        kernel = lowered.get_function("kernel")
        assert kernel is not None and (kernel.ir_stage == ir.FunctionIRStage.Buffer) == enabled
        restored = ir.deserialize(ir.serialize(lowered))
        assert isinstance(restored, ir.Program)
        ir.assert_structural_equal(lowered, restored, enable_auto_mapping=True)
        # Buffer replay tracks valid rectangles. Legacy Tile replay does not;
        # partial Tile inputs are checked through native compilation and devices.
        if enabled or valid == (16, 64):
            namespace = {}
            exec(torch_codegen(restored), namespace)  # noqa: S102
            output = torch.full_like(expected, -777.0)
            namespace["kernel"](source, output)
            torch.testing.assert_close(output, expected)
        text = _native(tmp_path, restored, planner, enabled)
        assert text.count(f"pto.tcol{reduction} ins(") == 1
        assert text.count(f"pto.tcolexpand{broadcast} ins(") == 1


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("kind", ["fp16", "dynamic", "optional_scratch"])
def test_column_selection_preserves_unverified_forms(planner, kind):
    program = _column_program(
        "sum",
        "add",
        ("rows" if kind == "dynamic" else 16, 64),
        "FP16" if kind == "fp16" else "FP32",
        extra_scratch=kind == "optional_scratch",
    )
    kernel = _lower(program, planner, True).get_function("kernel")
    assert kernel is not None and kernel.ir_stage != ir.FunctionIRStage.Buffer


def _transpose_program(shape=(16, 32), dtype="FP32", partial=False):
    rows, cols = shape
    return pl.parse_program(f"""
@pl.program
class TransposeCompute:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[{rows}, {cols}], pl.{dtype}],
               out: pl.Out[pl.Tensor[[{cols}, {rows}], pl.{dtype}]],
               old: pl.Out[pl.Tensor[[{rows}, {cols}], pl.{dtype}]]) -> tuple[
                   pl.Tensor[[{cols}, {rows}], pl.{dtype}], pl.Tensor[[{rows}, {cols}], pl.{dtype}]]:
        source = pl.load(a, [0, 0], [{rows}, {cols}], valid_shape=[{rows - int(partial)}, {cols}])
        result = pl.tile.transpose(source, -2, -1)
        out = pl.store(result, [0, 0], out)
        old = pl.store(source, [0, 0], old)
        return out, old
""")


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("shape", [(16, 32), (32, 64), (24, 40)])
def test_default_transpose_reuses_planned_scratch_and_preserves_input(tmp_path, planner, shape):
    source = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape)
    for enabled in (False, True):
        lowered = _lower(_transpose_program(shape), planner, enabled)
        kernel = lowered.get_function("kernel")
        assert kernel is not None
        assert (kernel.ir_stage == ir.FunctionIRStage.Buffer) == enabled
        restored = ir.deserialize(ir.serialize(lowered))
        assert isinstance(restored, ir.Program)
        ir.assert_structural_equal(lowered, restored, enable_auto_mapping=True)
        namespace = {}
        exec(torch_codegen(restored), namespace)  # noqa: S102
        output, old = torch.zeros_like(source.T), torch.zeros_like(source)
        namespace["kernel"](source, output, old)
        torch.testing.assert_close(output, source.T)
        torch.testing.assert_close(old, source)
        text = _native(tmp_path, restored, planner, enabled)
        assert text.count("pto.ttrans ins(") == 1
        if enabled:
            printed = ir.python_print(restored)
            assert "buffer.transpose(" in printed
            assert "tile.transpose(" not in printed


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("kind", ["partial", "fp16"])
def test_transpose_selection_preserves_unverified_forms(planner, kind):
    program = _transpose_program(dtype="FP16" if kind == "fp16" else "FP32", partial=kind == "partial")
    kernel = _lower(program, planner, True).get_function("kernel")
    assert kernel is not None and kernel.ir_stage != ir.FunctionIRStage.Buffer


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("operation", [*_ROWS, "transpose"])
def test_explicit_compute_a5_native_keeps_default_boundary(tmp_path, ascend_backend, planner, operation):
    assert ascend_backend == BackendType.Ascend950
    program = _transpose_program() if operation == "transpose" else _row_program(*operation)
    default = _lower(program, planner, True)
    kernel = default.get_function("kernel")
    assert kernel is not None and kernel.ir_stage != ir.FunctionIRStage.Buffer
    # Explicit recipe validation exercises A5 native contracts without opening
    # default A5 selection before that platform's device rollout is complete.
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        lowered = buffer_recipe_pipeline().run(program)
    kernel = lowered.get_function("kernel")
    assert kernel is not None and kernel.ir_stage == ir.FunctionIRStage.Buffer
    _native(tmp_path, lowered, planner, True, arch="a5")


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("reduction,broadcast", _COLUMNS)
def test_explicit_column_a5_native_keeps_default_boundary(
    tmp_path, ascend_backend, planner, reduction, broadcast
):
    assert ascend_backend == BackendType.Ascend950
    program = _column_program(reduction, broadcast)
    kernel = _lower(program, planner, True).get_function("kernel")
    assert kernel is not None and kernel.ir_stage != ir.FunctionIRStage.Buffer
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        lowered = buffer_recipe_pipeline().run(program)
    kernel = lowered.get_function("kernel")
    assert kernel is not None and kernel.ir_stage == ir.FunctionIRStage.Buffer
    _native(tmp_path, lowered, planner, True, arch="a5")


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("reduction", ["sum", "max", "min", "prod"])
@pytest.mark.parametrize("valid", [(16, 64), (13, 57)])
def test_column_reduction_dead_source_uses_disjoint_destination(tmp_path, planner, reduction, valid):
    rows, cols = valid
    program = pl.parse_program(f"""
@pl.program
class ColumnReduction:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 64], pl.FP32],
               out: pl.InOut[pl.Tensor[[1, 64], pl.FP32]]) -> pl.Tensor[[1, 64], pl.FP32]:
        source = pl.load(a, [0, 0], [16, 64], valid_shape=[{rows}, {cols}])
        row = pl.tile.col_{reduction}(source)
        return pl.store(row, [0, 0], out)
""")
    # With no later source use, an unconstrained planner could reuse a prefix of
    # its allocation. Buffer requires disjoint reduction storage before lowering.
    lowered = _lower(program, planner, True)
    kernel = lowered.get_function("kernel")
    assert kernel is not None and kernel.ir_stage == ir.FunctionIRStage.Buffer
    source = torch.rand((16, 64), generator=torch.Generator().manual_seed(30)) * 0.02 + 0.99
    output = torch.full((1, 64), -777.0)
    expected = output.clone()
    expected[:, :cols] = {"sum": torch.sum, "max": torch.amax, "min": torch.amin, "prod": torch.prod}[
        reduction
    ](source[:rows, :cols], dim=0, keepdim=True)
    namespace = {}
    exec(torch_codegen(lowered), namespace)  # noqa: S102
    namespace["kernel"](source, output)
    torch.testing.assert_close(output, expected)
    _native(tmp_path, lowered, planner, True)


_INDEX_OPS = ("row_argmax", "row_argmin", "col_argmax", "col_argmin")


def _index_program(operation, valid=(16, 128), dtype="FP32", preserve=True, strided_output=False):
    rows, cols = valid
    row = operation.startswith("row_")
    shape = [32, 8 if strided_output else 1] if row else [2, 128]
    offsets = [8, 0] if row else [1, 0]
    scalar = "rows: pl.Scalar[pl.INDEX], " if isinstance(rows, str) else ""
    old_store = "old = pl.store(source, [0, 0], old)" if preserve else ""
    return pl.parse_program(f"""
@pl.program
class IndexReduction:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, {scalar}a: pl.Tensor[[16, 128], pl.{dtype}],
               out: pl.InOut[pl.Tensor[{shape}, pl.INT32]],
               old: pl.InOut[pl.Tensor[[16, 128], pl.{dtype}]]) -> pl.Tensor[{shape}, pl.INT32]:
        source = pl.load(a, [0, 0], [16, 128], valid_shape=[{rows}, {cols}])
        scratch = pl.create_tile([16, 128], dtype=pl.{dtype}, target_memory=pl.MemorySpace.Vec)
        indices = pl.tile.{operation}(source, scratch)
        out = pl.store(indices, {offsets}, out)
        {old_store}
        return out
""")


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("operation", _INDEX_OPS)
@pytest.mark.parametrize("valid", [(16, 128), (13, 72)])
@pytest.mark.parametrize("preserve", [False, True])
def test_index_reduction_default_buffer_matches_tile_and_native(
    tmp_path, planner, operation, valid, preserve
):
    # Distinct exactly representable inputs make every expected index unambiguous.
    # The 72-column partial domain crosses the native 64-element repeat boundary.
    source = torch.randperm(2048, generator=torch.Generator().manual_seed(37)).reshape(16, 128).float()
    rows, cols = valid
    row = operation.startswith("row_")
    reduction = torch.argmax if operation.endswith("max") else torch.argmin
    expected = torch.full((32, 1) if row else (2, 128), -777, dtype=torch.int32)
    indices = reduction(source[:rows, :cols], dim=1 if row else 0, keepdim=True).to(torch.int32)
    if row:
        expected[8 : 8 + rows] = indices
    else:
        expected[1:2, :cols] = indices
    for enabled in (False, True):
        lowered = _lower(_index_program(operation, valid, preserve=preserve), planner, enabled)
        kernel = lowered.get_function("kernel")
        assert kernel is not None and (kernel.ir_stage == ir.FunctionIRStage.Buffer) == enabled
        restored = ir.deserialize(ir.serialize(lowered))
        assert isinstance(restored, ir.Program)
        ir.assert_structural_equal(lowered, restored, enable_auto_mapping=True)
        if enabled or valid == (16, 128):
            namespace = {}
            exec(torch_codegen(restored), namespace)  # noqa: S102
            output = torch.full_like(expected, -777)
            old = torch.full_like(source, -777)
            namespace["kernel"](source, output, old)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            old_expected = torch.full_like(source, -777)
            if preserve:
                old_expected[:rows, :cols] = source[:rows, :cols]
            torch.testing.assert_close(old, old_expected, rtol=0, atol=0)
        text = _native(tmp_path, restored, planner, enabled)
        assert text.count(f"pto.t{operation.replace('_', '')} ins(") == 1


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("operation", _INDEX_OPS)
@pytest.mark.parametrize("kind", ["fp16", "dynamic"])
def test_index_reduction_retains_unverified_tile_forms(planner, operation, kind):
    valid = ("rows", 128) if kind == "dynamic" else (16, 128)
    program = _index_program(operation, valid, "FP16" if kind == "fp16" else "FP32")
    kernel = _lower(program, planner, True).get_function("kernel")
    assert kernel is not None and kernel.ir_stage != ir.FunctionIRStage.Buffer


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("operation", ["row_argmax", "row_argmin"])
def test_index_column_store_keeps_strided_matrix_destination_on_tile(planner, operation):
    kernel = _lower(_index_program(operation, strided_output=True), planner, True).get_function("kernel")
    assert kernel is not None and kernel.ir_stage != ir.FunctionIRStage.Buffer


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("operation", _INDEX_OPS)
def test_explicit_index_a5_native_preserves_default_boundary(tmp_path, ascend_backend, planner, operation):
    assert ascend_backend == BackendType.Ascend950
    program = _index_program(operation)
    kernel = _lower(program, planner, True).get_function("kernel")
    assert kernel is not None and kernel.ir_stage != ir.FunctionIRStage.Buffer
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        lowered = buffer_recipe_pipeline().run(program)
    _native(tmp_path, lowered, planner, True, arch="a5")


@pytest.mark.parametrize("operation", _INDEX_OPS)
def test_index_reduction_preserves_frontend_empty_domain_rejection(operation):
    # Empty index reductions are already invalid Tile IR, before selection.
    with pytest.raises(InvalidOperationError, match="reduction has no real data"):
        _index_program(operation, valid=(0, 128))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
