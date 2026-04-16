# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for CompiledProgram callable API."""

import os

import pytest
import torch
from pypto import DataType, ir
from pypto.ir.compiled_program import CompiledProgram, _extract_param_infos


def _make_program_with_orchestration(*, has_return: bool = False) -> ir.Program:
    """Build a minimal Program with an Orchestration function for testing.

    Creates a program with params: a (In), b (In), c (Out).
    """
    span = ir.Span.unknown()
    tensor_type = ir.TensorType([128, 128], DataType.FP32)

    a_var = ir.Var("a", tensor_type, span)
    b_var = ir.Var("b", tensor_type, span)
    c_var = ir.Var("c", tensor_type, span)

    params = [
        (a_var, ir.ParamDirection.In),
        (b_var, ir.ParamDirection.In),
        (c_var, ir.ParamDirection.Out),
    ]
    return_types = [tensor_type] if has_return else []
    body = ir.SeqStmts([], span)

    orch = ir.Function("orchestrator", params, return_types, body, span, ir.FunctionType.Orchestration)
    return ir.Program([orch], "TestProgram", span)


def _make_program_without_orchestration() -> ir.Program:
    """Build a Program with multiple InCore functions and no orchestration."""
    span = ir.Span.unknown()
    body = ir.SeqStmts([], span)
    incore1 = ir.Function("kernel1", [], [], body, span, ir.FunctionType.InCore)
    incore2 = ir.Function("kernel2", [], [], body, span, ir.FunctionType.InCore)
    return ir.Program([incore1, incore2], "NoOrchProgram", span)


def _make_single_function_program() -> ir.Program:
    """Build a Program with a single InCore function (fallback for no orchestration)."""
    span = ir.Span.unknown()
    tensor_type = ir.TensorType([64, 64], DataType.FP32)
    a_var = ir.Var("x", tensor_type, span)
    body = ir.SeqStmts([], span)
    incore = ir.Function("kernel", [(a_var, ir.ParamDirection.In)], [], body, span, ir.FunctionType.InCore)
    return ir.Program([incore], "SingleFnProgram", span)


def _make_program_with_inout() -> ir.Program:
    """Build a Program with an InOut parameter: a (In), acc (InOut), out (Out)."""
    span = ir.Span.unknown()
    tensor_type = ir.TensorType([128, 128], DataType.FP32)

    a_var = ir.Var("a", tensor_type, span)
    acc_var = ir.Var("acc", tensor_type, span)
    out_var = ir.Var("out", tensor_type, span)

    params = [
        (a_var, ir.ParamDirection.In),
        (acc_var, ir.ParamDirection.InOut),
        (out_var, ir.ParamDirection.Out),
    ]
    body = ir.SeqStmts([], span)
    orch = ir.Function("orchestrator", params, [], body, span, ir.FunctionType.Orchestration)
    return ir.Program([orch], "InOutProgram", span)


class TestCompiledProgramBackwardCompat:
    """Verify CompiledProgram behaves like a path string for backward compat."""

    def test_str_returns_output_dir(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))
        assert str(cp) == str(tmp_path.resolve())

    def test_fspath_returns_output_dir(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))
        assert os.fspath(cp) == str(tmp_path.resolve())

    def test_path_join_works(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))
        joined = os.path.join(cp, "kernels")
        assert joined == os.path.join(str(tmp_path.resolve()), "kernels")

    def test_eq_with_string(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))
        assert cp == str(tmp_path.resolve())

    def test_eq_with_compiled_program(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp1 = CompiledProgram(prog, str(tmp_path))
        cp2 = CompiledProgram(prog, str(tmp_path))
        assert cp1 == cp2

    def test_repr(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))
        r = repr(cp)
        assert "CompiledProgram" in r


class TestExtractParamInfos:
    """Verify metadata extraction from orchestration function."""

    def test_extracts_param_names_and_directions(self):
        prog = _make_program_with_orchestration()
        infos, out_idx, _ = _extract_param_infos(prog)

        assert len(infos) == 3
        assert infos[0].name == "a"
        assert infos[0].direction == ir.ParamDirection.In
        assert infos[1].name == "b"
        assert infos[1].direction == ir.ParamDirection.In
        assert infos[2].name == "c"
        assert infos[2].direction == ir.ParamDirection.Out

    def test_output_indices(self):
        prog = _make_program_with_orchestration()
        _, out_idx, _ = _extract_param_infos(prog)
        assert out_idx == [2]

    def test_shape_extraction(self):
        prog = _make_program_with_orchestration()
        infos, _, _ = _extract_param_infos(prog)
        assert infos[0].shape == [128, 128]
        assert infos[0].dtype == DataType.FP32

    def test_return_types(self):
        prog = _make_program_with_orchestration(has_return=True)
        _, _, ret_types = _extract_param_infos(prog)
        assert len(ret_types) == 1

    def test_no_orchestration_multi_func_raises(self):
        prog = _make_program_without_orchestration()
        with pytest.raises(ValueError, match="no Orchestration function"):
            _extract_param_infos(prog)

    def test_single_function_fallback(self):
        prog = _make_single_function_program()
        infos, _, _ = _extract_param_infos(prog)
        assert len(infos) == 1
        assert infos[0].name == "x"

    def test_inout_not_in_output_indices(self):
        """InOut params require caller-provided initial values; they must not
        be auto-allocated in return-style calls."""
        prog = _make_program_with_inout()
        infos, out_idx, _ = _extract_param_infos(prog)
        assert len(infos) == 3
        assert infos[0].direction == ir.ParamDirection.In
        assert infos[1].direction == ir.ParamDirection.InOut
        assert infos[2].direction == ir.ParamDirection.Out
        # Only pure Out (index 2) should be auto-allocated
        assert out_idx == [2]


class TestCompiledProgramMetadata:
    """Verify lazy metadata properties on CompiledProgram."""

    def test_param_names(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))
        assert cp.param_names == ["a", "b", "c"]

    def test_output_indices_property(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))
        assert cp.output_indices == [2]

    def test_has_return_false(self, tmp_path):
        prog = _make_program_with_orchestration(has_return=False)
        cp = CompiledProgram(prog, str(tmp_path))
        assert cp.has_return is False

    def test_has_return_true(self, tmp_path):
        prog = _make_program_with_orchestration(has_return=True)
        cp = CompiledProgram(prog, str(tmp_path))
        assert cp.has_return is True

    def test_properties_accessible(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))
        assert cp.output_dir == tmp_path.resolve()
        assert cp.program is prog
        assert cp.backend_type is not None


class TestCompiledProgramCall:
    """Verify __call__ argument validation (without device execution)."""

    def test_wrong_arg_count_raises(self, tmp_path):
        prog = _make_program_with_orchestration(has_return=False)
        cp = CompiledProgram(prog, str(tmp_path))
        a = torch.randn(128, 128)
        with pytest.raises(TypeError, match="expects 3"):
            cp(a)  # too few args

    def test_wrong_arg_count_with_return(self, tmp_path):
        prog = _make_program_with_orchestration(has_return=True)
        cp = CompiledProgram(prog, str(tmp_path))
        a = torch.randn(128, 128)
        # Program has 3 params (2 in + 1 out), with return.
        # Valid: 3 args (in-place) or 2 args (return style)
        with pytest.raises(TypeError, match="expects 3 .* or 2"):
            cp(a)  # 1 arg is neither 3 nor 2

    def test_no_orchestration_multi_func_call_raises(self, tmp_path):
        prog = _make_program_without_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))
        with pytest.raises(ValueError, match="no Orchestration"):
            cp(torch.randn(10))


class TestBuildFullArgs:
    """Verify output tensor allocation for return-style calls."""

    def test_allocates_output_tensors(self, tmp_path):
        prog = _make_program_with_orchestration(has_return=True)
        cp = CompiledProgram(prog, str(tmp_path))
        param_infos, output_indices, _ = cp._get_metadata()

        a = torch.randn(128, 128)
        b = torch.randn(128, 128)
        full_args = cp._build_full_args((a, b), param_infos, output_indices)

        assert len(full_args) == 3
        assert full_args[0] is a
        assert full_args[1] is b
        # Output tensor should be allocated with correct shape/dtype
        assert full_args[2].shape == (128, 128)
        assert full_args[2].dtype == torch.float32
        assert torch.all(full_args[2] == 0)


class TestCompileReturnsCompiledProgram:
    """Verify ir.compile() returns CompiledProgram."""

    def test_compile_return_type(self, tmp_path):
        """Call ir.compile() on a simple program and verify return type."""
        import pypto.language as pl  # noqa: PLC0415

        @pl.program
        class SimpleAdd:
            @pl.function(type=pl.FunctionType.InCore)
            def add_kernel(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ):
                tile_a = pl.tile.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_b = pl.tile.load(b, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.tile.add(tile_a, tile_b)
                pl.tile.store(tile_c, offsets=[0, 0], output_tensor=c)

        output_dir = str(tmp_path / "compiled")
        result = ir.compile(SimpleAdd, output_dir=output_dir, dump_passes=False, skip_ptoas=True)
        assert isinstance(result, CompiledProgram)
        # Backward compat: str() gives a path
        assert os.path.isdir(str(result))
        # output_dir property works
        assert result.output_dir.is_dir()
        # Metadata works on the original program
        assert result.param_names == ["a", "b", "c"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
