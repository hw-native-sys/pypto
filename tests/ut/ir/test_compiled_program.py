# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for CompiledProgram callable API."""

import contextlib
import ctypes
import json
import os
import sys
import types
import warnings
from unittest.mock import MagicMock, patch

import pytest
import torch
from pypto import DataType, backend, ir
from pypto.backend import BackendType
from pypto.ir.compiled_program import (
    _COMPILED_META_FILENAME,
    _COMPILED_META_SCHEMA,
    CompiledProgram,
    _build_full_args,
    _extract_param_infos,
)
from pypto.ir.distributed_compiled_program import (
    _DISTRIBUTED_META_FILENAME,
    DistributedCompiledProgram,
)
from pypto.runtime import DeviceTensor, RunConfig


@contextlib.contextmanager
def _fake_compile_and_assemble(return_value):
    """Stub ``pypto.runtime.device_runner._compile_and_assemble`` via a fake module.

    The real module imports ``simpler_setup`` (device-only), so it can't be loaded
    on host-only CI. Inserting a fake module into ``sys.modules`` lets the inner
    ``from pypto.runtime.device_runner import _compile_and_assemble`` resolve to the
    mock on every platform. Yields the mock for call assertions.
    """
    mock = MagicMock(return_value=return_value)
    fake = types.ModuleType("pypto.runtime.device_runner")
    setattr(fake, "_compile_and_assemble", mock)
    with patch.dict(sys.modules, {"pypto.runtime.device_runner": fake}):
        yield mock


@contextlib.contextmanager
def _fake_call_config(instance):
    """Stub ``pypto.runtime.task_interface.CallConfig`` via a fake module.

    ``task_interface`` imports the device-only ``simpler`` package, so it can't
    load on host CI. Fake it so ``_build_call_config``'s inner import binds to a
    ``CallConfig`` that returns ``instance``."""
    fake = types.ModuleType("pypto.runtime.task_interface")
    setattr(fake, "CallConfig", MagicMock(return_value=instance))
    with patch.dict(sys.modules, {"pypto.runtime.task_interface": fake}):
        yield


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


def _make_multi_orch_program() -> ir.Program:
    """Build a Program with two Orchestrations of deliberately different signatures.

    ``orch_a(a, b, c)`` mirrors :func:`_make_program_with_orchestration`;
    ``orch_b(x, y)`` is shorter, so a per-sub-build sidecar cannot accidentally
    pass by carrying the other orchestration's metadata.
    """
    span = ir.Span.unknown()
    tensor_type = ir.TensorType([128, 128], DataType.FP32)
    body = ir.SeqStmts([], span)

    orch_a = ir.Function(
        "orch_a",
        [
            (ir.Var("a", tensor_type, span), ir.ParamDirection.In),
            (ir.Var("b", tensor_type, span), ir.ParamDirection.In),
            (ir.Var("c", tensor_type, span), ir.ParamDirection.Out),
        ],
        [],
        body,
        span,
        ir.FunctionType.Orchestration,
    )
    orch_b = ir.Function(
        "orch_b",
        [
            (ir.Var("x", tensor_type, span), ir.ParamDirection.In),
            (ir.Var("y", tensor_type, span), ir.ParamDirection.Out),
        ],
        [],
        body,
        span,
        ir.FunctionType.Orchestration,
    )
    return ir.Program([orch_a, orch_b], "MultiOrchProgram", span)


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


def _make_program_with_scalar() -> ir.Program:
    """Build a Program with tensor and scalar params: a (In), n (Scalar INT64), c (Out)."""
    span = ir.Span.unknown()
    tensor_type = ir.TensorType([128, 128], DataType.FP32)
    scalar_type = ir.ScalarType(DataType.INT64)

    a_var = ir.Var("a", tensor_type, span)
    n_var = ir.Var("n", scalar_type, span)
    c_var = ir.Var("c", tensor_type, span)

    params = [
        (a_var, ir.ParamDirection.In),
        (n_var, ir.ParamDirection.In),
        (c_var, ir.ParamDirection.Out),
    ]
    body = ir.SeqStmts([], span)
    orch = ir.Function("orchestrator", params, [tensor_type], body, span, ir.FunctionType.Orchestration)
    return ir.Program([orch], "ScalarProgram", span)


def _make_fp4_output_program() -> ir.Program:
    span = ir.Span.unknown()
    logical_type = ir.TensorType([8, 16], DataType.FP4)
    src = ir.Var("src", logical_type, span)
    out = ir.Var("out", logical_type, span)
    params = [(src, ir.ParamDirection.In), (out, ir.ParamDirection.Out)]
    orch = ir.Function(
        "orchestrator",
        params,
        [logical_type],
        ir.SeqStmts([], span),
        span,
        ir.FunctionType.Orchestration,
    )
    return ir.Program([orch], "Fp4OutputProgram", span)


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

    def test_fp4_metadata_uses_torch_x2_carrier_shape(self):
        infos, out_idx, _ = _extract_param_infos(_make_fp4_output_program())
        assert [info.shape for info in infos] == [[8, 8], [8, 8]]
        assert all(info.dtype == DataType.FP4 for info in infos)
        assert out_idx == [1]

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

    def test_explicit_config_platform_overrides_compiled_platform(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path), platform="a2a3sim")
        args = (torch.zeros(128, 128), torch.zeros(128, 128), torch.zeros(128, 128))
        config = RunConfig(
            platform="a2a3",
            device_id=3,
            enable_pmu=2,
            aicpu_thread_num=7,
            ring_heap=512 * 1024 * 1024,
        )

        with patch("pypto.runtime.runner._execute_compiled") as mock_exec:
            cp(*args, config=config)

        assert mock_exec.call_args.kwargs["platform"] == "a2a3"
        assert mock_exec.call_args.kwargs["device_id"] == 3
        assert mock_exec.call_args.kwargs["dfx"].enable_pmu == 2
        assert mock_exec.call_args.kwargs["aicpu_thread_num"] == 7
        assert mock_exec.call_args.kwargs["config"] is config

    def test_dispatch_does_not_emit_the_execute_compiled_deprecation(self, tmp_path):
        """The artifact path goes to the implementation, not the deprecated wrapper.

        ``pypto.runtime.execute_compiled`` is deprecated and warns. It wraps the
        same implementation this dispatch uses, so routing back through it would
        make every supported call emit a deprecation the caller cannot act on.
        """
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path), platform="a2a3sim")
        args = (torch.zeros(128, 128), torch.zeros(128, 128), torch.zeros(128, 128))

        with patch("pypto.runtime.runner._execute_compiled"):
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                cp(*args)

    def test_no_config_uses_compiled_platform(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path), platform="a5sim")
        args = (torch.zeros(128, 128), torch.zeros(128, 128), torch.zeros(128, 128))

        with patch("pypto.runtime.runner._execute_compiled") as mock_exec:
            cp(*args)

        assert mock_exec.call_args.kwargs["platform"] == "a5sim"

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
        full_args = _build_full_args((a, b), param_infos, output_indices)

        assert len(full_args) == 3
        assert full_args[0] is a
        assert full_args[1] is b
        # Output tensor should be allocated with correct shape/dtype
        out = full_args[2]
        assert isinstance(out, torch.Tensor)
        assert out.shape == (128, 128)
        assert out.dtype == torch.float32
        assert torch.all(out == 0)

    @pytest.mark.skipif(
        not hasattr(torch, "float4_e2m1fn_x2"),
        reason="torch.float4_e2m1fn_x2 required",
    )
    def test_allocates_fp4_output_with_physical_x2_shape(self):
        infos, output_indices, _ = _extract_param_infos(_make_fp4_output_program())
        src = torch.zeros((8, 8), dtype=torch.float4_e2m1fn_x2)
        full_args = _build_full_args((src,), infos, output_indices)
        out = full_args[1]
        assert isinstance(out, torch.Tensor)
        assert out.shape == (8, 8)
        assert out.dtype == torch.float4_e2m1fn_x2


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

    def test_compile_platform_selects_codegen_backend(self, tmp_path):
        """platform='a5sim' should compile with the Ascend950 PTO backend."""
        import pypto.language as pl  # noqa: PLC0415

        backend.reset_for_testing()
        try:

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

            output_dir = str(tmp_path / "compiled_a5")
            result = ir.compile(
                SimpleAdd,
                output_dir=output_dir,
                dump_passes=False,
                skip_ptoas=True,
                platform="a5sim",
            )

            pto_files = list(result.output_dir.rglob("*.pto"))
            assert isinstance(result, CompiledProgram)
            assert result.backend_type == BackendType.Ascend950
            assert result.platform == "a5sim"
            assert pto_files
            assert 'pto.target_arch = "a5"' in pto_files[0].read_text()
        finally:
            backend.reset_for_testing()


class TestExtractParamInfosScalar:
    """Verify metadata extraction for scalar parameters."""

    def test_scalar_param_shape_is_none(self):
        prog = _make_program_with_scalar()
        infos, _, _ = _extract_param_infos(prog)
        assert infos[1].name == "n"
        assert infos[1].shape is None
        assert infos[1].dtype == DataType.INT64

    def test_scalar_param_not_in_output_indices(self):
        prog = _make_program_with_scalar()
        _, out_idx, _ = _extract_param_infos(prog)
        # Only index 2 (c, Out tensor) should be auto-allocatable
        assert out_idx == [2]


class TestCompiledProgramScalarCall:
    """Verify __call__ handles scalar parameters correctly."""

    def test_scalar_param_wraps_python_int(self, tmp_path):
        """Passing a Python int to a scalar param should wrap it as ctypes.c_int64."""
        prog = _make_program_with_scalar()
        cp = CompiledProgram(prog, str(tmp_path))

        a = torch.randn(128, 128)
        c = torch.zeros(128, 128)

        with patch("pypto.runtime.runner._execute_compiled") as mock_exec:
            cp(a, 5, c)

        coerced_args = mock_exec.call_args.args[1]  # second positional arg is the args list
        assert isinstance(coerced_args[0], torch.Tensor)
        assert isinstance(coerced_args[1], ctypes.c_int64)
        assert coerced_args[1].value == 5
        assert isinstance(coerced_args[2], torch.Tensor)

    def test_scalar_param_passes_through_ctypes(self, tmp_path):
        """Passing a ctypes scalar directly should pass through without re-wrapping."""
        prog = _make_program_with_scalar()
        cp = CompiledProgram(prog, str(tmp_path))

        a = torch.randn(128, 128)
        c = torch.zeros(128, 128)
        scalar = ctypes.c_int64(42)

        with patch("pypto.runtime.runner._execute_compiled") as mock_exec:
            cp(a, scalar, c)

        coerced_args = mock_exec.call_args.args[1]
        assert coerced_args[1] is scalar

    def test_scalar_param_rejects_wrong_ctypes(self, tmp_path):
        """Passing a ctypes scalar with mismatched dtype should raise TypeError."""
        prog = _make_program_with_scalar()  # n is INT64
        cp = CompiledProgram(prog, str(tmp_path))

        a = torch.randn(128, 128)
        c = torch.zeros(128, 128)

        with pytest.raises(TypeError, match="int64"):
            cp(a, ctypes.c_int32(5), c)  # wrong: c_int32 for INT64 param

    def test_scalar_param_rejects_tensor(self, tmp_path):
        """Passing a torch.Tensor for a scalar param should raise TypeError."""
        prog = _make_program_with_scalar()
        cp = CompiledProgram(prog, str(tmp_path))

        a = torch.randn(128, 128)
        c = torch.zeros(128, 128)

        with pytest.raises(TypeError, match="scalar"):
            cp(a, torch.tensor([5]), c)

    def test_tensor_param_rejects_scalar(self, tmp_path):
        """Passing a Python int for a tensor param should raise TypeError."""
        prog = _make_program_with_scalar()
        cp = CompiledProgram(prog, str(tmp_path))

        with pytest.raises(TypeError, match="tensor"):
            cp(5, 10, torch.zeros(128, 128))

    def test_return_style_with_scalar(self, tmp_path):
        """Return-style call with scalar: compiled(a, n) should allocate output."""
        prog = _make_program_with_scalar()
        cp = CompiledProgram(prog, str(tmp_path))

        a = torch.randn(128, 128)

        with patch("pypto.runtime.runner._execute_compiled") as mock_exec:
            result = cp(a, 7)

        # Should have called _execute_compiled with 3 args (a, scalar, allocated c)
        coerced_args = mock_exec.call_args.args[1]
        assert len(coerced_args) == 3
        assert isinstance(coerced_args[1], ctypes.c_int64)
        assert coerced_args[1].value == 7
        # Output should be returned
        assert isinstance(result, torch.Tensor)


class TestCompiledProgramDeviceTensor:
    """Verify __call__ accepts DeviceTensor in tensor parameter slots."""

    def test_device_tensor_in_input_slot(self, tmp_path):
        """A DeviceTensor passed for an In param is forwarded to _execute_compiled."""
        prog = _make_program_with_orchestration()  # a (In), b (In), c (Out)
        cp = CompiledProgram(prog, str(tmp_path))

        a = torch.randn(128, 128)
        b = DeviceTensor(0xB0000, (128, 128), torch.float32)
        c = torch.zeros(128, 128)

        with patch("pypto.runtime.runner._execute_compiled") as mock_exec:
            cp(a, b, c)

        coerced_args = mock_exec.call_args.args[1]
        assert isinstance(coerced_args[0], torch.Tensor)
        assert coerced_args[1] is b  # forwarded as-is
        assert isinstance(coerced_args[2], torch.Tensor)

    def test_all_device_tensors(self, tmp_path):
        """Every tensor slot can be a DeviceTensor."""
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))

        a = DeviceTensor(0x1000, (128, 128), torch.float32)
        b = DeviceTensor(0x2000, (128, 128), torch.float32)
        c = DeviceTensor(0x3000, (128, 128), torch.float32)

        with patch("pypto.runtime.runner._execute_compiled") as mock_exec:
            cp(a, b, c)

        coerced_args = mock_exec.call_args.args[1]
        assert coerced_args == [a, b, c]

    def test_unsupported_type_for_tensor_param(self, tmp_path):
        """Non-tensor / non-DeviceTensor in a tensor slot raises TypeError."""
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))

        with pytest.raises(TypeError, match="DeviceTensor"):
            cp("not a tensor", torch.zeros(128, 128), torch.zeros(128, 128))  # type: ignore[arg-type]

    def test_device_tensor_shape_mismatch_rejected_early(self, tmp_path):
        """DeviceTensor with wrong shape vs IR metadata fails before dispatch."""
        prog = _make_program_with_orchestration()  # all params are [128, 128]
        cp = CompiledProgram(prog, str(tmp_path))

        a = torch.zeros(128, 128)
        bad_b = DeviceTensor(0xB0000, (64, 128), torch.float32)  # wrong shape
        c = torch.zeros(128, 128)

        with pytest.raises(TypeError, match=r"expects shape \(128, 128\)"):
            cp(a, bad_b, c)

    def test_device_tensor_dtype_mismatch_rejected_early(self, tmp_path):
        """DeviceTensor with wrong dtype vs IR metadata fails before dispatch."""
        prog = _make_program_with_orchestration()  # all params are FP32
        cp = CompiledProgram(prog, str(tmp_path))

        a = torch.zeros(128, 128)
        bad_b = DeviceTensor(0xB0000, (128, 128), torch.float16)  # wrong dtype
        c = torch.zeros(128, 128)

        with pytest.raises(TypeError, match="expects dtype"):
            cp(a, bad_b, c)


class TestCompiledProgramExtraction:
    """Verify the extraction surface that lets users drive ``simpler.worker.Worker``
    directly: ``chip_callable`` / ``runtime_name`` / ``runtime_config`` properties,
    ``load()``, ``_build_orch_args()``, and ``_build_call_config()``.
    """

    def _patch_assemble(self, chip_callable_name: str = "fake_chip"):
        """Patch ``device_runner._compile_and_assemble`` and return the MagicMock.

        The patch target is the *source* module — inner-scope ``from ... import``
        statements bind to the patched name at import time.
        """
        cc = MagicMock(name=chip_callable_name)
        runtime_config = {"aicpu_thread_num": 2}
        return (
            cc,
            runtime_config,
            _fake_compile_and_assemble((cc, "host_build_graph", runtime_config)),
        )

    def test_chip_callable_triggers_assemble(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))

        cc, _, patcher = self._patch_assemble()
        with patcher as mock:
            assert cp.chip_callable is cc
            mock.assert_called_once_with(tmp_path.resolve(), cp.platform)

    def test_runtime_name_triggers_assemble(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))

        _, _, patcher = self._patch_assemble()
        with patcher:
            assert cp.runtime_name == "host_build_graph"

    def test_runtime_config_triggers_assemble(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))

        _, runtime_config, patcher = self._patch_assemble()
        with patcher:
            assert cp.runtime_config == runtime_config

    def test_properties_cache_across_calls(self, tmp_path):
        """All three properties together trigger exactly one _compile_and_assemble call."""
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))

        _, _, patcher = self._patch_assemble()
        with patcher as mock:
            _ = cp.chip_callable
            _ = cp.runtime_name
            _ = cp.runtime_config
            _ = cp.chip_callable  # repeat
            assert mock.call_count == 1

    def test_load_eagerly_assembles(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))

        _, _, patcher = self._patch_assemble()
        with patcher as mock:
            cp.load()
            mock.assert_called_once_with(tmp_path.resolve(), cp.platform)

    def test_load_after_first_access_is_noop(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))

        _, _, patcher = self._patch_assemble()
        with patcher as mock:
            _ = cp.chip_callable
            cp.load()
            assert mock.call_count == 1

    def test_build_orch_args_inplace_returns_full_list(self, tmp_path):
        """In-place call shape: all params passed, return_style is False."""
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))
        a = torch.zeros(128, 128)
        b = torch.zeros(128, 128)
        c = torch.zeros(128, 128)

        # Patch the runner-side helper so we don't hit simpler types.
        worker = MagicMock(name="worker")
        with patch("pypto.runtime.runner._coerced_to_orch_args") as oa_helper:
            oa_helper.return_value = "fake_orch_args"
            orch_args, coerced, return_style = cp._build_orch_args(a, b, c, worker=worker)

        assert orch_args == "fake_orch_args"
        assert coerced == [a, b, c]
        assert return_style is False
        oa_helper.assert_called_once_with([a, b, c], worker)

    def test_build_orch_args_return_style_allocates_output(self, tmp_path):
        """Return-style: only inputs passed; output auto-allocated at output_indices."""
        prog = _make_program_with_orchestration(has_return=True)
        cp = CompiledProgram(prog, str(tmp_path))
        a = torch.zeros(128, 128)
        b = torch.zeros(128, 128)

        worker = MagicMock(name="worker")
        with patch("pypto.runtime.runner._coerced_to_orch_args") as oa_helper:
            oa_helper.return_value = "fake_orch_args"
            orch_args, coerced, return_style = cp._build_orch_args(a, b, worker=worker)

        assert orch_args == "fake_orch_args"
        assert return_style is True
        assert len(coerced) == 3
        # Output slot is at index 2 (output_indices == [2]); a real torch.Tensor was allocated.
        out = coerced[cp.output_indices[0]]
        assert isinstance(out, torch.Tensor)
        assert out.shape == (128, 128)

    def test_build_orch_args_wrong_arg_count_raises(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))
        a = torch.zeros(128, 128)
        with pytest.raises(TypeError, match="expects 3"):
            cp._build_orch_args(a)

    def test_build_call_config_uses_runtime_config_default(self, tmp_path):
        """When config has no overrides, RUNTIME_CONFIG values feed CallConfig."""
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))

        _, _, patcher = self._patch_assemble()
        fake_call_config = MagicMock(name="CallConfig_instance")
        with (
            patcher,
            _fake_call_config(fake_call_config),
        ):
            from pypto.runtime import RunConfig  # noqa: PLC0415

            cfg = cp._build_call_config(RunConfig())

        assert cfg is fake_call_config
        assert fake_call_config.aicpu_thread_num == 2  # from runtime_config

    def test_build_call_config_explicit_overrides_runtime_config(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))

        _, _, patcher = self._patch_assemble()
        fake_call_config = MagicMock(name="CallConfig_instance")
        with (
            patcher,
            _fake_call_config(fake_call_config),
        ):
            from pypto.runtime import RunConfig  # noqa: PLC0415

            cp._build_call_config(RunConfig(aicpu_thread_num=8), aicpu_thread_num=4)

        assert fake_call_config.aicpu_thread_num == 4  # kwarg > RunConfig field > runtime_config

    def test_build_call_config_run_config_beats_runtime_config(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))

        _, _, patcher = self._patch_assemble()
        fake_call_config = MagicMock(name="CallConfig_instance")
        with (
            patcher,
            _fake_call_config(fake_call_config),
        ):
            from pypto.runtime import RunConfig  # noqa: PLC0415

            cp._build_call_config(RunConfig(aicpu_thread_num=16))

        assert fake_call_config.aicpu_thread_num == 16  # RunConfig wins over runtime_config's 2

    def test_build_call_config_copies_dfx_flags(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))

        _, _, patcher = self._patch_assemble()
        fake_call_config = MagicMock(name="CallConfig_instance")
        with (
            patcher,
            _fake_call_config(fake_call_config),
        ):
            from pypto.runtime import RunConfig  # noqa: PLC0415

            cp._build_call_config(
                RunConfig(
                    enable_chip_swimlane=True,
                    enable_dump_args=True,
                    enable_pmu=2,
                    enable_dep_gen=True,
                ),
                dfx_dir=tmp_path / "dfx",
            )

        assert fake_call_config.enable_chip_swimlane == 4  # True normalizes to the full level
        assert fake_call_config.enable_dump_args is True
        assert fake_call_config.enable_pmu == 2
        assert fake_call_config.enable_dep_gen is True
        assert fake_call_config.output_prefix == str(tmp_path / "dfx")

    def test_build_call_config_dfx_dir_omitted_when_none(self, tmp_path):
        """No dfx_dir → output_prefix must NOT be set on CallConfig."""
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))

        _, _, patcher = self._patch_assemble()
        fake_call_config = MagicMock(
            spec=[
                "aicpu_thread_num",
                "enable_chip_swimlane",
                "enable_dump_args",
                "enable_pmu",
                "enable_dep_gen",
            ]
        )
        with (
            patcher,
            _fake_call_config(fake_call_config),
        ):
            from pypto.runtime import RunConfig  # noqa: PLC0415

            cp._build_call_config(RunConfig())

        # spec doesn't include "output_prefix", so any attempted set would fail.
        # Reaching here means _build_call_config correctly skipped it.
        assert not hasattr(fake_call_config, "output_prefix")


class TestCompiledProgramExtractionMultiOrch:
    """Multi-orch programs must reject extraction on the parent and route through
    ``compiled[<name>]``.
    """

    def _make_multi_orch_layout(self, tmp_path):
        """Lay out next_levels/<name>/orchestration/ so CompiledProgram detects multi-orch."""
        for name in ("orch_a", "orch_b"):
            (tmp_path / "next_levels" / name / "orchestration").mkdir(parents=True)

    def test_chip_callable_rejected_on_multi_orch_parent(self, tmp_path):
        prog = _make_program_with_orchestration()
        self._make_multi_orch_layout(tmp_path)
        cp = CompiledProgram(prog, str(tmp_path))
        assert cp.orchestration_names  # sanity: multi-orch detected
        with pytest.raises(TypeError, match="Multi-orch"):
            _ = cp.chip_callable

    def test_build_orch_args_rejected_on_multi_orch_parent(self, tmp_path):
        prog = _make_program_with_orchestration()
        self._make_multi_orch_layout(tmp_path)
        cp = CompiledProgram(prog, str(tmp_path))
        a = torch.zeros(128, 128)
        b = torch.zeros(128, 128)
        c = torch.zeros(128, 128)
        with pytest.raises(TypeError, match="Multi-orch"):
            cp._build_orch_args(a, b, c)


class TestSubChipCallableExtraction:
    """Mirror of TestCompiledProgramExtraction for ``compiled[<name>]``."""

    def _make_subchip(self, tmp_path):
        from pypto.ir.compiled_program import _SubChipCallable  # noqa: PLC0415

        # Build a Function with the same params as _make_program_with_orchestration.
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
        body = ir.SeqStmts([], span)
        func = ir.Function("orch_a", params, [], body, span, ir.FunctionType.Orchestration)
        sub_dir = tmp_path / "next_levels" / "orch_a"
        sub_dir.mkdir(parents=True)
        return _SubChipCallable("orch_a", func, sub_dir, "a2a3sim")

    def test_chip_callable_triggers_assemble(self, tmp_path):
        sub = self._make_subchip(tmp_path)
        cc = MagicMock(name="sub_chip_callable")
        with _fake_compile_and_assemble((cc, "host_build_graph", {})) as mock:
            assert sub.chip_callable is cc
            mock.assert_called_once_with(sub.output_dir, sub.platform)

    def test_build_orch_args_routes_through_helper(self, tmp_path):
        sub = self._make_subchip(tmp_path)
        a = torch.zeros(128, 128)
        b = torch.zeros(128, 128)
        c = torch.zeros(128, 128)
        worker = MagicMock(name="worker")
        with patch("pypto.runtime.runner._coerced_to_orch_args") as oa_helper:
            oa_helper.return_value = "fake_orch_args"
            orch_args, coerced, return_style = sub._build_orch_args(a, b, c, worker=worker)

        assert orch_args == "fake_orch_args"
        assert coerced == [a, b, c]
        assert return_style is False
        oa_helper.assert_called_once_with([a, b, c], worker)

    def test_load_after_first_access_is_noop(self, tmp_path):
        sub = self._make_subchip(tmp_path)
        cc = MagicMock(name="sub_chip_callable")
        with _fake_compile_and_assemble((cc, "host_build_graph", {})) as mock:
            _ = sub.chip_callable  # cache warmed
            sub.load()
            assert mock.call_count == 1

    def test_explicit_config_platform_overrides_parent_platform(self, tmp_path):
        sub = self._make_subchip(tmp_path)
        args = (torch.zeros(128, 128), torch.zeros(128, 128), torch.zeros(128, 128))

        with patch("pypto.runtime.runner._execute_compiled") as mock_exec:
            sub(*args, config=RunConfig(platform="a2a3"))

        assert mock_exec.call_args.kwargs["platform"] == "a2a3"

    def test_no_config_uses_parent_platform(self, tmp_path):
        sub = self._make_subchip(tmp_path)
        args = (torch.zeros(128, 128), torch.zeros(128, 128), torch.zeros(128, 128))

        with patch("pypto.runtime.runner._execute_compiled") as mock_exec:
            sub(*args)

        assert mock_exec.call_args.kwargs["platform"] == "a2a3sim"


class TestValidateIr:
    """Verify CompiledProgram.validate_ir resolves passes_dump and delegates."""

    def test_missing_passes_dump_raises(self, tmp_path):
        prog = _make_program_with_orchestration()
        cp = CompiledProgram(prog, str(tmp_path))  # no passes_dump/ created
        with pytest.raises(FileNotFoundError, match="passes_dump"):
            cp.validate_ir({}, {})

    def test_delegates_to_validator(self, tmp_path):
        prog = _make_program_with_orchestration()
        passes_dump = tmp_path / "passes_dump"
        passes_dump.mkdir()
        cp = CompiledProgram(prog, str(tmp_path))

        tensors = {"a": torch.zeros(4)}
        expected = {"c": torch.ones(4)}
        with patch("pypto.debug.validate_pass_ir_codegen_results") as validator:
            cp.validate_ir(tensors, expected, rtol=1e-3, atol=1e-4)

        validator.assert_called_once_with(str(passes_dump), tensors, expected, rtol=1e-3, atol=1e-4)


class TestCompiledMetaAndFromDir:
    """``compiled_meta.json`` + ``CompiledProgram.from_dir`` — replay an
    already-compiled single-chip build without re-running the pypto compile,
    so ``benchmark()`` works against a ``runtime_dir`` replay (#2344).
    """

    def test_compile_persists_compiled_meta(self, tmp_path):
        """Constructing from live IR writes a compiled_meta.json sidecar."""
        prog = _make_program_with_orchestration(has_return=True)
        CompiledProgram(prog, str(tmp_path), platform="a2a3sim")

        meta = json.loads((tmp_path / _COMPILED_META_FILENAME).read_text())
        assert meta["schema"] == _COMPILED_META_SCHEMA
        assert [p["name"] for p in meta["params"]] == ["a", "b", "c"]
        assert [p["direction"] for p in meta["params"]] == ["In", "In", "Out"]
        assert [p["shape"] for p in meta["params"]] == [[128, 128]] * 3
        assert {p["dtype"] for p in meta["params"]} == {"fp32"}
        assert meta["num_return_types"] == 1
        assert meta["platform"] == "a2a3sim"
        assert meta["backend_type"] == "Ascend910B"

    def test_from_dir_round_trips_param_metadata(self, tmp_path):
        """from_dir reconstructs the same param metadata as the live compile."""
        prog = _make_program_with_orchestration(has_return=True)
        live = CompiledProgram(prog, str(tmp_path), platform="a2a3sim")
        reloaded = CompiledProgram.from_dir(tmp_path)

        def _key(cp):
            infos, _, _ = cp._get_metadata()
            return [(p.name, p.direction, p.shape, str(p.dtype)) for p in infos]

        assert _key(reloaded) == _key(live)
        assert reloaded.program is None  # reconstructed from disk, no live IR
        assert reloaded.platform == "a2a3sim"
        assert reloaded.backend_type == BackendType.Ascend910B
        assert reloaded.has_return is live.has_return

    def test_from_dir_output_indices_match_out_params(self, tmp_path):
        """output_indices are rederived from the persisted directions (c is the lone Out)."""
        CompiledProgram(_make_program_with_orchestration(), str(tmp_path))
        reloaded = CompiledProgram.from_dir(tmp_path)

        param_infos, output_indices, _ = reloaded._get_metadata()
        assert output_indices == [2]
        assert param_infos[2].direction == ir.ParamDirection.Out

    def test_from_dir_dispatches_via_runner(self, tmp_path):
        """A reconstructed program is callable and reaches _execute_compiled."""
        CompiledProgram(_make_program_with_orchestration(), str(tmp_path), platform="a2a3sim")
        reloaded = CompiledProgram.from_dir(tmp_path)
        a = torch.zeros(128, 128)
        b = torch.zeros(128, 128)
        c = torch.zeros(128, 128)

        with patch("pypto.runtime.runner._execute_compiled") as mock_exec:
            reloaded(a, b, c)

        mock_exec.assert_called_once()
        assert mock_exec.call_args.args[0] == tmp_path.resolve()
        assert list(mock_exec.call_args.args[1]) == [a, b, c]
        assert mock_exec.call_args.kwargs["platform"] == "a2a3sim"

    def test_from_dir_exposes_benchmark_surface(self, tmp_path):
        """Every member ``benchmark()``'s L2 branch touches works after from_dir.

        Regression test for #2344: ``benchmark`` needs ``platform`` /
        ``runtime_name`` / ``runtime_config`` / ``chip_callable`` plus the
        metadata-derived ``_build_orch_args`` / ``_build_call_config`` /
        ``output_indices``, which previously required a live ``Program``.
        """
        CompiledProgram(_make_program_with_orchestration(), str(tmp_path), platform="a2a3sim")
        reloaded = CompiledProgram.from_dir(tmp_path, platform="a2a3")

        cc = MagicMock(name="fake_chip")
        runtime_config = {"aicpu_thread_num": 2, "enable_sdma": True}
        call_config = MagicMock(name="call_config")
        worker = MagicMock(name="worker")
        args = [torch.zeros(128, 128) for _ in range(3)]

        with _fake_compile_and_assemble((cc, "host_build_graph", runtime_config)):
            assert reloaded.platform == "a2a3"
            assert reloaded.runtime_name == "host_build_graph"
            assert reloaded.runtime_config == runtime_config
            assert reloaded.chip_callable is cc
            assert reloaded.output_indices == [2]
            with patch("pypto.runtime.runner._coerced_to_orch_args") as oa_helper:
                oa_helper.return_value = "fake_orch_args"
                orch_args, coerced, return_style = reloaded._build_orch_args(*args, worker=worker)
                oa_helper.assert_called_once_with(args, worker)
            with _fake_call_config(call_config):
                assert reloaded._build_call_config(RunConfig()) is call_config

        assert orch_args == "fake_orch_args"
        assert coerced == args
        assert return_style is False

    def test_from_dir_does_not_clobber_debug_runner(self, tmp_path):
        """Reloading must preserve a hand-edited debug/run.py (the replay workflow)."""
        CompiledProgram(_make_program_with_orchestration(), str(tmp_path))
        run_py = tmp_path / "debug" / "run.py"
        # Asserted rather than skipped: a self-skip would silently disable this
        # regression check if debug-runner emission ever changed.
        assert run_py.exists(), "single-orch compile must emit debug/run.py"
        sentinel = "# hand-edited by the user — must survive from_dir\n"
        run_py.write_text(sentinel)

        CompiledProgram.from_dir(tmp_path)
        assert run_py.read_text() == sentinel

    def test_from_dir_does_not_rewrite_meta(self, tmp_path):
        """The reload path must not rewrite the sidecar it just read."""
        CompiledProgram(_make_program_with_orchestration(), str(tmp_path))
        meta_path = tmp_path / _COMPILED_META_FILENAME
        before = meta_path.stat().st_mtime_ns

        CompiledProgram.from_dir(tmp_path)
        assert meta_path.stat().st_mtime_ns == before

    def test_from_dir_overrides_platform_and_backend(self, tmp_path):
        """Explicit platform / backend_type override the persisted defaults.

        Both overrides differ from what was persisted (``a2a3sim`` / Ascend910B),
        so an ignored override cannot pass by coincidence.
        """
        CompiledProgram(_make_program_with_orchestration(), str(tmp_path), platform="a2a3sim")
        assert json.loads((tmp_path / _COMPILED_META_FILENAME).read_text())["backend_type"] == "Ascend910B"

        reloaded = CompiledProgram.from_dir(tmp_path, platform="a5sim", backend_type=BackendType.Ascend950)
        assert reloaded.platform == "a5sim"
        assert reloaded.backend_type == BackendType.Ascend950

        # Without overrides the persisted pair comes back unchanged.
        persisted = CompiledProgram.from_dir(tmp_path)
        assert persisted.platform == "a2a3sim"
        assert persisted.backend_type == BackendType.Ascend910B

    def test_from_dir_keeps_the_fp4_x2_carrier_shape(self, tmp_path):
        """Packed FP4 shapes are persisted already converted, so the reload must not re-halve.

        ``_extract_func_param_infos`` stores the runtime x2 carrier shape
        (logical ``[8, 16]`` -> ``[8, 8]``); applying ``_to_runtime_shape`` again
        on load would silently halve it a second time.
        """
        live = CompiledProgram(_make_fp4_output_program(), str(tmp_path))
        reloaded = CompiledProgram.from_dir(tmp_path)

        def _shapes(cp):
            infos, _, _ = cp._get_metadata()
            return [p.shape for p in infos]

        assert _shapes(live) == [[8, 8], [8, 8]]
        assert _shapes(reloaded) == _shapes(live)

    def test_from_dir_missing_meta_raises(self, tmp_path):
        """A directory without compiled_meta.json raises with a recompile hint."""
        with pytest.raises(FileNotFoundError, match=r"compiled_meta\.json"):
            CompiledProgram.from_dir(tmp_path)

    def test_from_dir_incompatible_schema_raises(self, tmp_path):
        """A compiled_meta.json written under a different schema version is rejected."""
        CompiledProgram(_make_program_with_orchestration(), str(tmp_path))
        meta_path = tmp_path / _COMPILED_META_FILENAME
        meta = json.loads(meta_path.read_text())
        meta["schema"] = meta["schema"] + 1  # simulate an incompatible future format
        meta_path.write_text(json.dumps(meta))

        with pytest.raises(ValueError, match="schema"):
            CompiledProgram.from_dir(tmp_path)

    def test_multi_orch_parent_writes_no_meta(self, tmp_path):
        """The multi-orch parent has no single canonical entry, so it gets no sidecar."""
        for name in ("orch_a", "orch_b"):
            (tmp_path / "next_levels" / name / "orchestration").mkdir(parents=True)
        cp = CompiledProgram(_make_multi_orch_program(), str(tmp_path))

        assert cp.orchestration_names  # sanity: multi-orch detected
        assert not (tmp_path / _COMPILED_META_FILENAME).exists()
        with pytest.raises(FileNotFoundError, match="multi-orch"):
            CompiledProgram.from_dir(tmp_path)

    def test_multi_orch_sub_builds_get_their_own_meta(self, tmp_path):
        """Each next_levels/<name>/ sub-build is independently reloadable.

        The parent's error message points users at the sub-builds, so each one
        must carry the signature of *its own* orchestration.
        """
        for name in ("orch_a", "orch_b"):
            (tmp_path / "next_levels" / name / "orchestration").mkdir(parents=True)
        CompiledProgram(_make_multi_orch_program(), str(tmp_path), platform="a2a3sim")

        # orch_a takes (a, b, c); orch_b takes only (x, y) — distinct signatures.
        reloaded_a = CompiledProgram.from_dir(tmp_path / "next_levels" / "orch_a")
        reloaded_b = CompiledProgram.from_dir(tmp_path / "next_levels" / "orch_b")
        assert reloaded_a.param_names == ["a", "b", "c"]
        assert reloaded_a.output_indices == [2]
        assert reloaded_b.param_names == ["x", "y"]
        assert reloaded_b.output_indices == [1]
        assert reloaded_a.platform == "a2a3sim"

    def test_malformed_meta_raises_value_error(self, tmp_path):
        """Every malformed payload surfaces as ValueError naming the file, not a raw KeyError."""
        meta_path = tmp_path / _COMPILED_META_FILENAME
        CompiledProgram(_make_program_with_orchestration(), str(tmp_path))
        good = json.loads(meta_path.read_text())

        def _param(**overrides):
            """Serialise ``good`` with its first param entry field-patched."""
            return json.dumps({**good, "params": [{**good["params"][0], **overrides}]})

        broken = {
            "not JSON": "{ this is not json",
            "top-level list": json.dumps([good]),
            "params not a list": json.dumps({**good, "params": "abc"}),
            "param entry not an object": json.dumps({**good, "params": ["abc"]}),
            "param missing a key": json.dumps({**good, "params": [{"name": "a"}]}),
            "non-string name": _param(name=7),
            "bad direction": _param(direction="Sideways"),
            # ``getattr(ParamDirection, "mro")`` resolves to a bound method — a
            # lax getattr() lookup would accept it as a direction.
            "direction names an attribute": _param(direction="mro"),
            "non-string direction": _param(direction=0),
            "shape not a list": _param(shape="invalid"),
            "shape holds a non-int": _param(shape=[128, "x"]),
            # bool is an int subclass; JSON ``true`` is never a dimension.
            "shape holds a bool": _param(shape=[128, True]),
            "bad dtype": _param(dtype="fp99"),
            "non-string dtype": _param(dtype=["fp32"]),
            "negative return count": json.dumps({**good, "num_return_types": -1}),
            "non-int return count": json.dumps({**good, "num_return_types": "two"}),
            "unknown backend": json.dumps({**good, "backend_type": "AscendNope"}),
            # Same trap as ``direction`` above: ``getattr(BackendType, "mro")``
            # resolves to a bound method a lax lookup would accept as a backend.
            "backend names an attribute": json.dumps({**good, "backend_type": "mro"}),
            "non-string backend": json.dumps({**good, "backend_type": 7}),
            "non-string platform": json.dumps({**good, "platform": 7}),
        }
        for label, payload in broken.items():
            meta_path.write_text(payload)
            with pytest.raises(ValueError, match=_COMPILED_META_FILENAME) as excinfo:
                CompiledProgram.from_dir(tmp_path)
            assert "ir.compile()" in str(excinfo.value), f"{label}: message lacks a recompile hint"

    def test_unreadable_meta_raises_value_error(self, tmp_path):
        """An OSError from reading the sidecar is reported like any malformed payload."""
        CompiledProgram(_make_program_with_orchestration(), str(tmp_path))
        meta_path = tmp_path / _COMPILED_META_FILENAME
        meta_path.unlink()
        meta_path.mkdir()  # exists(), but read_text() raises IsADirectoryError

        with pytest.raises(ValueError, match=_COMPILED_META_FILENAME) as excinfo:
            CompiledProgram.from_dir(tmp_path)
        assert "ir.compile()" in str(excinfo.value)

    def test_undecodable_meta_raises_value_error(self, tmp_path):
        """A non-UTF-8 sidecar is bad input, not an internal UnicodeDecodeError."""
        CompiledProgram(_make_program_with_orchestration(), str(tmp_path))
        (tmp_path / _COMPILED_META_FILENAME).write_bytes(b"\xff\xfe\x00binary")

        with pytest.raises(ValueError, match=_COMPILED_META_FILENAME):
            CompiledProgram.from_dir(tmp_path)

    def test_from_dir_ignores_stale_next_levels(self, tmp_path):
        """A sidecar is only written for a single-orch build, so it settles the layout.

        A ``next_levels/`` left by an earlier multi-orch compile into the same
        directory must not turn the reloaded program into a multi-orch parent:
        the artifacts the sidecar describes are the top-level ones.
        """
        CompiledProgram(_make_program_with_orchestration(), str(tmp_path), _sub_chip_names=[])
        for name in ("orch_a", "orch_b"):
            (tmp_path / "next_levels" / name / "orchestration").mkdir(parents=True)
        reloaded = CompiledProgram.from_dir(tmp_path)

        assert reloaded.orchestration_names == []
        args = (torch.zeros(128, 128), torch.zeros(128, 128), torch.zeros(128, 128))
        with patch("pypto.runtime.runner._execute_compiled") as mock_exec:
            reloaded(*args)
        assert mock_exec.call_args.args[0] == tmp_path.resolve()

    def test_subscript_without_live_ir_raises(self, tmp_path):
        """Scanned-layout dispatch (no declared layout) must not deref the absent IR."""
        for name in ("orch_a", "orch_b"):
            (tmp_path / "next_levels" / name / "orchestration").mkdir(parents=True)
        cp = CompiledProgram(None, str(tmp_path))  # no _sub_chip_names → on-disk scan

        assert cp.orchestration_names == ["orch_a", "orch_b"]
        with pytest.raises(RuntimeError, match="live IR"):
            _ = cp["orch_a"]

    def test_metadata_without_ir_or_sidecar_raises(self, tmp_path):
        """Constructing with neither live IR nor metadata fails loudly, not with None deref."""
        cp = CompiledProgram(None, str(tmp_path))
        with pytest.raises(RuntimeError, match="from_dir"):
            _ = cp.param_names


class TestCompiledMetaOutputDirReuse:
    """Recompiling into a reused ``output_dir``.

    ``ir.compile`` never clears ``output_dir`` (``makedirs(exist_ok=True)``), so
    a sidecar left by a previous compile of a *different* program shape would
    otherwise survive and drive the new artifacts with the old parameter ABI --
    a mismatch ``from_dir`` cannot detect. The same reuse also leaves the
    previous compile's ``next_levels/`` in place, so these tests equally pin
    that the *build layout* comes from the codegen that just ran rather than
    from whatever the directory happens to contain.
    """

    def test_recompile_refreshes_meta(self, tmp_path):
        """A second single-orch compile overwrites the first one's signature."""
        CompiledProgram(_make_program_with_orchestration(), str(tmp_path))
        CompiledProgram(_make_program_with_inout(), str(tmp_path))

        reloaded = CompiledProgram.from_dir(tmp_path)
        live = CompiledProgram(_make_program_with_inout(), str(tmp_path))
        assert reloaded.param_names == live.param_names

    def test_multi_orch_recompile_drops_stale_parent_meta(self, tmp_path):
        """Single-orch then multi-orch into one dir: the parent markers must go."""
        CompiledProgram(_make_program_with_orchestration(), str(tmp_path), _sub_chip_names=[])
        (tmp_path / "kernel_config.py").write_text("# emitted by the single-orch compile\n")
        assert (tmp_path / _COMPILED_META_FILENAME).exists()  # sanity: written by the first compile

        for name in ("orch_a", "orch_b"):
            (tmp_path / "next_levels" / name / "orchestration").mkdir(parents=True)
        CompiledProgram(_make_multi_orch_program(), str(tmp_path), _sub_chip_names=["orch_a", "orch_b"])

        assert not (tmp_path / _COMPILED_META_FILENAME).exists()
        # A multi-orch build writes nothing at the top level, so the previous
        # build's kernel_config.py would otherwise keep replay() pointed at it.
        assert not (tmp_path / "kernel_config.py").exists()
        # The parent must now point users at the sub-builds rather than hand out
        # the previous program's signature.
        with pytest.raises(FileNotFoundError, match="multi-orch"):
            CompiledProgram.from_dir(tmp_path)

    def test_unextractable_recompile_drops_stale_meta(self, tmp_path):
        """A program with no resolvable signature removes the stale sidecar instead of keeping it."""
        CompiledProgram(_make_program_with_orchestration(), str(tmp_path))
        CompiledProgram(_make_program_without_orchestration(), str(tmp_path))

        assert not (tmp_path / _COMPILED_META_FILENAME).exists()
        with pytest.raises(FileNotFoundError, match=r"compiled_meta\.json"):
            CompiledProgram.from_dir(tmp_path)

    def test_sub_build_without_matching_function_drops_meta(self, tmp_path):
        """A sub-build the IR has no function for loses its sidecar instead of keeping a stale one."""
        for name in ("orch_a", "orch_b"):
            (tmp_path / "next_levels" / name / "orchestration").mkdir(parents=True)
        names = ["orch_a", "orch_b"]
        CompiledProgram(_make_multi_orch_program(), str(tmp_path), _sub_chip_names=names)
        stale = tmp_path / "next_levels" / "orch_b" / _COMPILED_META_FILENAME
        assert stale.exists()  # sanity: written by the first compile

        # Same layout, but the IR no longer carries orch_b: its build dir must
        # not keep advertising the signature the previous compile recorded.
        orch_a = _make_multi_orch_program().get_function("orch_a")
        assert orch_a is not None
        only_a = ir.Program([orch_a], "OnlyOrchA", ir.Span.unknown())
        CompiledProgram(only_a, str(tmp_path), _sub_chip_names=names)
        assert not stale.exists()
        assert (tmp_path / "next_levels" / "orch_a" / _COMPILED_META_FILENAME).exists()

    def test_stale_next_levels_does_not_shadow_a_single_orch_build(self, tmp_path):
        """Multi-orch then single-orch into one dir: the new top-level build wins.

        The layout comes from the codegen that just ran, so the leftover
        ``next_levels/`` must not classify the second compile as multi-orch --
        that would hide its top-level artifacts behind the stale sub-builds,
        delete the sidecar it just wrote, and force dispatch through
        ``compiled[<name>]`` into the *old* build.
        """
        import pypto.language as pl  # noqa: PLC0415

        @pl.program
        class TwoOrch:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_add(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tf = pl.add(pl.load(a, [0, 0], [128, 128]), pl.load(b, [0, 0], [128, 128]))
                return pl.store(tf, [0, 0], f)

            @pl.function(type=pl.FunctionType.InCore)
            def tile_sub(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tf = pl.sub(pl.load(a, [0, 0], [128, 128]), pl.load(b, [0, 0], [128, 128]))
                return pl.store(tf, [0, 0], f)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_add(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                return self.tile_add(a, b, f)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_sub(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                return self.tile_sub(a, b, f)

        @pl.program
        class OneOrch:
            """Deliberately different signature (x, y, z) from either orch above."""

            @pl.function(type=pl.FunctionType.InCore)
            def mul_kernel(
                self,
                x: pl.Tensor[[64, 64], pl.FP32],
                y: pl.Tensor[[64, 64], pl.FP32],
                z: pl.Tensor[[64, 64], pl.FP32],
            ):
                tz = pl.tile.mul(pl.tile.load(x, [0, 0], [64, 64]), pl.tile.load(y, [0, 0], [64, 64]))
                pl.tile.store(tz, offsets=[0, 0], output_tensor=z)

        work_dir = tmp_path / "reused"
        multi = ir.compile(TwoOrch, output_dir=str(work_dir), dump_passes=False, skip_ptoas=True)
        assert isinstance(multi, CompiledProgram)  # L2, not the distributed wrapper
        assert multi.orchestration_names == ["orch_add", "orch_sub"]
        assert not (work_dir / _COMPILED_META_FILENAME).exists()

        single = ir.compile(OneOrch, output_dir=str(work_dir), dump_passes=False, skip_ptoas=True)
        assert isinstance(single, CompiledProgram)

        # 1) The second compile owns the top level: its own artifacts, a sidecar
        #    describing *its* signature, and no multi-orch dispatch surface.
        assert {p.stem for p in (work_dir / "kernels").rglob("*.pto")} == {"mul_kernel"}
        assert single.orchestration_names == []
        assert single.param_names == ["x", "y", "z"]
        meta = json.loads((work_dir / _COMPILED_META_FILENAME).read_text())
        assert [p["name"] for p in meta["params"]] == ["x", "y", "z"]

        # 2) Both call paths run the new top-level build, not a stale sub-build.
        reloaded = CompiledProgram.from_dir(work_dir)
        assert reloaded.orchestration_names == []
        assert reloaded.param_names == ["x", "y", "z"]
        args = (torch.zeros(64, 64), torch.zeros(64, 64), torch.zeros(64, 64))
        for compiled in (single, reloaded):
            with patch("pypto.runtime.runner._execute_compiled") as mock_exec:
                compiled(*args)
            assert mock_exec.call_args.args[0] == work_dir.resolve()

        # 3) The leftover sub-builds keep their own artifacts *and* their own
        #    sidecar, so reloading one replays that older build rather than
        #    describing the new one.
        stale = CompiledProgram.from_dir(work_dir / "next_levels" / "orch_add")
        assert stale.param_names == ["a", "b", "f"]

    def test_l2_then_l3_into_one_dir_drops_the_l2_markers(self, tmp_path):
        """An L2 build's markers must not survive an L3 compile into the same dir.

        ``replay()`` takes the L3 path only when there is **no** top-level
        ``kernel_config.py``, so an L2 leftover does not just age out — it makes
        the whole directory replay as the previous single-chip build.
        """
        CompiledProgram(_make_program_with_orchestration(), str(tmp_path), _sub_chip_names=[])
        (tmp_path / "kernel_config.py").write_text("# emitted by the L2 compile\n")
        assert (tmp_path / _COMPILED_META_FILENAME).exists()  # sanity

        DistributedCompiledProgram(_make_program_with_inout(), str(tmp_path))

        assert (tmp_path / _DISTRIBUTED_META_FILENAME).exists()
        assert not (tmp_path / _COMPILED_META_FILENAME).exists()
        assert not (tmp_path / "kernel_config.py").exists()  # replay() now resolves L3
        with pytest.raises(FileNotFoundError, match=r"compiled_meta\.json"):
            CompiledProgram.from_dir(tmp_path)

    def test_l3_then_l2_into_one_dir_drops_the_l3_markers(self, tmp_path):
        """The mirror direction: an L3 build's markers must not outlive an L2 compile."""
        DistributedCompiledProgram(_make_program_with_orchestration(), str(tmp_path))
        (tmp_path / "orchestration").mkdir(exist_ok=True)
        (tmp_path / "orchestration" / "host_orch.py").write_text("# emitted by the L3 compile\n")
        assert (tmp_path / _DISTRIBUTED_META_FILENAME).exists()  # sanity

        CompiledProgram(_make_program_with_inout(), str(tmp_path), _sub_chip_names=[])

        assert not (tmp_path / _DISTRIBUTED_META_FILENAME).exists()
        assert not (tmp_path / "orchestration" / "host_orch.py").exists()
        with pytest.raises(FileNotFoundError, match=r"distributed_meta\.json"):
            DistributedCompiledProgram.from_dir(tmp_path)
        # ...and the directory now reloads as the L2 program just compiled.
        assert CompiledProgram.from_dir(tmp_path).param_names == ["a", "acc", "out"]

    def test_no_temp_files_left_behind(self, tmp_path):
        """The atomic write leaves no ``.tmp`` residue next to the sidecar."""
        CompiledProgram(_make_program_with_orchestration(), str(tmp_path))
        assert not list(tmp_path.glob("*.tmp"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
