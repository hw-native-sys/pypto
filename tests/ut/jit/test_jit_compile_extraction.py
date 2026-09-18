# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for ``JITFunction.compile()`` — the public extraction surface that
returns the underlying :class:`CompiledProgram` so callers can drive worker
runtime APIs directly.

Closes hw-native-sys/pypto#1455.
"""

import ctypes
import importlib
import re
import warnings

import pypto.language as pl
import pytest
from pypto.compile_profiling import CompileProfiler
from pypto.ir import OptimizationStrategy, PassDumpLevel
from pypto.ir.compiled_program import CompiledProgram
from pypto.jit.decorator import jit
from pypto.language.parser.diagnostics.exceptions import ParserTypeError
from pypto.pypto_core import ir, passes
from pypto.runtime.runner import RunConfig


@pytest.fixture(autouse=True)
def _disable_ptoas_for_source_only_tests(monkeypatch, tmp_path):
    """Keep compile() coverage source-only on hosts with an unusable ptoas."""
    monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))


@jit.incore
def _add_incore(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    M, N = a.shape
    tile_a = pl.load(a, [0, 0], [M, N])
    tile_b = pl.load(b, [0, 0], [M, N])
    tile_c = pl.add(tile_a, tile_b)
    pl.store(tile_c, [0, 0], c)
    return c


@jit
def add_kernel(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    c = _add_incore(a, b, c)
    return c


class TestCompileReturnsCompiledProgram:
    """Verify ``kernel.compile(*sample_args)`` returns a usable CompiledProgram."""

    def test_compile_returns_compiled_program_instance(self):
        torch = pytest.importorskip("torch")

        a = torch.zeros(128, 128, dtype=torch.float32)
        b = torch.zeros(128, 128, dtype=torch.float32)
        c = torch.empty(128, 128, dtype=torch.float32)

        compiled = add_kernel.compile(a, b, c)
        assert isinstance(compiled, CompiledProgram)

    def test_compile_cache_hit_returns_same_instance(self):
        """Two compile() calls with the same specialisation must reuse the cache."""
        torch = pytest.importorskip("torch")

        a = torch.zeros(64, 64, dtype=torch.float32)
        b = torch.zeros(64, 64, dtype=torch.float32)
        c = torch.empty(64, 64, dtype=torch.float32)

        first = add_kernel.compile(a, b, c)
        second = add_kernel.compile(a, b, c)
        assert first is second

    def test_program_execution_then_compile_returns_cached_instance(self, monkeypatch):
        """Executing an explicitly compiled program preserves its compilation cache."""
        torch = pytest.importorskip("torch")

        a = torch.zeros(96, 96, dtype=torch.float32)
        b = torch.zeros(96, 96, dtype=torch.float32)
        c = torch.empty(96, 96, dtype=torch.float32)

        compiled = add_kernel.compile(a, b, c)
        monkeypatch.setattr(CompiledProgram, "__call__", lambda self, *_args, **_kwargs: "called")
        assert compiled(a, b, c) == "called"
        assert add_kernel.compile(a, b, c) is compiled

    def test_compile_cache_miss_on_different_shape(self):
        """Different shape causes a new compilation (distinct CompiledProgram)."""
        torch = pytest.importorskip("torch")

        a_a = torch.zeros(32, 32, dtype=torch.float32)
        b_a = torch.zeros(32, 32, dtype=torch.float32)
        c_a = torch.empty(32, 32, dtype=torch.float32)
        a_b = torch.zeros(48, 48, dtype=torch.float32)
        b_b = torch.zeros(48, 48, dtype=torch.float32)
        c_b = torch.empty(48, 48, dtype=torch.float32)

        compiled_a = add_kernel.compile(a_a, b_a, c_a)
        compiled_b = add_kernel.compile(a_b, b_b, c_b)
        assert compiled_a is not compiled_b

    def test_compile_keeps_outer_report_instrument(self, tmp_path):
        torch = pytest.importorskip("torch")
        # 32 columns, not 19: an unboxed FP32 tile is addressed in whole
        # 32-byte units, and this test is about the report instrument, not
        # the shape.
        x = torch.zeros(19, 32)

        with passes.PassContext([passes.ReportInstrument(str(tmp_path))]):
            compiled = add_kernel.compile(x, x, torch.empty_like(x))

        assert isinstance(compiled, CompiledProgram)
        assert (tmp_path / "perf_hints.log").is_file()


class TestLowerReturnsProgram:
    """Verify ``lower()`` specializes and runs passes without compiling."""

    def test_lower_returns_post_pass_program_without_compiling(self, monkeypatch):
        torch = pytest.importorskip("torch")
        a = torch.zeros(24, 24)
        b = torch.zeros(24, 24)
        c = torch.empty(24, 24)
        cache_before = dict(add_kernel._cache)

        def fail_compile(*_args, **_kwargs):
            pytest.fail("lower() entered codegen")

        monkeypatch.setattr(add_kernel, "_compile", fail_compile)
        program = add_kernel.lower(a, b, c)
        assert isinstance(program, ir.Program)
        assert add_kernel._cache == cache_before

    def test_lower_ignores_artifact_controls(self, tmp_path):
        torch = pytest.importorskip("torch")
        artifact_dir = tmp_path / "must_not_exist"
        config = RunConfig(
            save_kernels=True,
            save_kernels_dir=str(artifact_dir),
            dump_passes=True,
            compile_profiling=True,
            codegen_only=True,
            device_id=7,
        )
        x = torch.zeros(20, 20)
        program = add_kernel.lower(x, x, torch.empty_like(x), config=config)
        assert isinstance(program, ir.Program)
        assert not artifact_dir.exists()

    def test_lower_filters_outer_report_instrument_but_runs_callbacks(self, tmp_path):
        torch = pytest.importorskip("torch")
        seen_passes: list[str] = []
        report_instrument = passes.ReportInstrument(str(tmp_path))
        callback_instrument = passes.CallbackInstrument(
            before_pass=lambda pass_obj, _program: seen_passes.append(pass_obj.get_name()),
            name="observer",
        )
        x = torch.zeros(17, 17)

        with passes.PassContext([report_instrument, callback_instrument]):
            add_kernel.lower(x, x, torch.empty_like(x))

        assert seen_passes
        assert not (tmp_path / "perf_hints.log").exists()

    def test_lower_runs_the_configured_strategy_pipeline(self):
        torch = pytest.importorskip("torch")
        x = torch.zeros(16, 16)
        seen_default: list[str] = []
        default_instrument = passes.CallbackInstrument(
            before_pass=lambda pass_obj, _program: seen_default.append(pass_obj.get_name()),
            name="default",
        )
        with passes.PassContext([default_instrument]):
            add_kernel.lower(
                x,
                x,
                torch.empty_like(x),
                config=RunConfig(strategy=OptimizationStrategy.Default),
            )
        assert "ConvertTensorToTileOps" in seen_default

    def test_lower_supports_signature_and_keyword_modes(self):
        torch = pytest.importorskip("torch")

        @jit
        def copy(x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out = x
            return out

        assert isinstance(copy.lower(), ir.Program)
        x = torch.zeros(16, 16)
        assert isinstance(copy.lower(x=x, out=torch.empty_like(x)), ir.Program)

    def test_lower_signature_failure_guides_source_only_callers(self):
        with pytest.raises(TypeError) as exc_info:
            add_kernel.lower()

        message = str(exc_info.value)
        assert "lower(*sample_tensors)" in message
        assert "compile(*sample_tensors)" in message

    def test_lower_conflict_writes_no_artifacts(self, tmp_path):
        torch = pytest.importorskip("torch")
        artifact_dir = tmp_path / "must_not_exist"
        config = RunConfig(
            memory_planner=passes.MemoryPlanner.PTOAS,
            save_kernels_dir=str(artifact_dir),
            dump_passes=True,
            compile_profiling=True,
        )
        x = torch.zeros(16, 16)
        with passes.PassContext([]):
            with pytest.raises(RuntimeError, match=r"lower\(\).*memory_planner"):
                add_kernel.lower(x, x, torch.empty_like(x), config=config)
        assert not artifact_dir.exists()

    def test_lower_rewrites_specializer_names_in_pass_errors(self, monkeypatch):
        torch = pytest.importorskip("torch")
        x = torch.zeros(16, 16)

        def fail_pipeline(*_args, **_kwargs):
            raise ValueError("Pass rejected variable 'c_v1'")

        compile_module = importlib.import_module("pypto.ir.compile")
        monkeypatch.setattr(compile_module, "_run_pass_pipeline", fail_pipeline)
        with pytest.raises(ValueError, match="Pass rejected variable 'c'") as exc_info:
            add_kernel.lower(x, x, torch.empty_like(x))
        assert "c_v1" not in str(exc_info.value)


class TestCompileForwardsRunConfig:
    """``compile()`` consumes ``config=`` like ``__call__`` so the compiled
    artefact honours the same compile-side knobs (strategy, dump_passes, …)."""

    def test_compile_extracts_config_kwarg(self):
        """``config=`` must be consumed by JIT and not forwarded to the kernel."""
        torch = pytest.importorskip("torch")

        a = torch.zeros(16, 16, dtype=torch.float32)
        b = torch.zeros(16, 16, dtype=torch.float32)
        c = torch.empty(16, 16, dtype=torch.float32)

        # Passing config= should not raise a "unexpected keyword 'config'"
        # signature error from the decorated kernel.
        compiled = add_kernel.compile(a, b, c, config=RunConfig(platform="a2a3sim"))
        assert isinstance(compiled, CompiledProgram)


class TestCompileExposesExtractionSurface:
    """The returned CompiledProgram exposes the full extraction surface added
    in PR #1496 — the public runtime handles (chip_callable / runtime_name /
    runtime_config) plus the internal argument builders ``ChipWorker.run``
    marshals through — enabling worker integration as required by issue #1455.

    These tests only verify that the attributes are *defined on the class* —
    actually exercising _compile_and_assemble (which several of these properties
    invoke lazily on first access) requires simpler + a device, which unit
    tests don't have. ``hasattr(instance, ...)`` would trigger the property
    getter and import simpler, so check the class directly.
    """

    def test_compiled_program_has_extraction_attributes(self):
        torch = pytest.importorskip("torch")

        a = torch.zeros(8, 8, dtype=torch.float32)
        b = torch.zeros(8, 8, dtype=torch.float32)
        c = torch.empty(8, 8, dtype=torch.float32)

        compiled = add_kernel.compile(a, b, c)
        cls = type(compiled)
        # The properties + methods that ChipWorker.run / register rely on.
        # Checking ``cls`` instead of ``compiled`` avoids invoking lazy
        # property getters (chip_callable etc.) which call _compile_and_assemble.
        for name in (
            "chip_callable",
            "runtime_name",
            "runtime_config",
            "_build_orch_args",
            "_build_call_config",
            "output_dir",
            "platform",
            "output_indices",
        ):
            assert hasattr(cls, name), f"CompiledProgram missing {name!r}"


# Fully-annotated kernels for signature-mode compile() (issue #1996).
_SIG_M = pl.dynamic("M")


@jit.incore
def _sig_copy_incore(a: pl.Tensor[[_SIG_M, 128], pl.FP32], c: pl.Out[pl.Tensor[[_SIG_M, 128], pl.FP32]]):
    tile = pl.load(a, [0, 0], [128, 128])
    pl.store(tile, [0, 0], c)
    return c


@jit
def sig_kernel(a: pl.Tensor[[_SIG_M, 128], pl.FP32], c: pl.Out[pl.Tensor[[_SIG_M, 128], pl.FP32]]):
    c = _sig_copy_incore(a, c)
    return c


# Runtime (unspecialized) scalar parameters in signature mode (issue #2283).
# The scalar is consumed inside the incore dep, so these kernels also cover
# scalar propagation across the JIT call graph.
@jit.incore
def _rt_add_scalar_incore(
    a: pl.Tensor[[_SIG_M, 128], pl.FP32],
    n: pl.Scalar[pl.FP32],
    c: pl.Out[pl.Tensor[[_SIG_M, 128], pl.FP32]],
):
    tile = pl.load(a, [0, 0], [128, 128])
    shifted = pl.add(tile, n)
    pl.store(shifted, [0, 0], c)
    return c


@jit
def rt_scalar_kernel(
    a: pl.Tensor[[_SIG_M, 128], pl.FP32],
    n: pl.Scalar[pl.FP32],
    c: pl.Out[pl.Tensor[[_SIG_M, 128], pl.FP32]],
):
    c = _rt_add_scalar_incore(a, n, c)
    return c


@jit
def rt_scalar_default_kernel(
    a: pl.Tensor[[_SIG_M, 128], pl.FP32],
    c: pl.Out[pl.Tensor[[_SIG_M, 128], pl.FP32]],
    n: pl.Scalar[pl.FP32] = pl.RUNTIME,
):
    c = _rt_add_scalar_incore(a, n, c)
    return c


class TestCompileFromSignature:
    """``compile()`` with no positional args reads the shape/dtype contract
    straight from the kernel's own annotations — no throwaway ``torch.empty``
    dummies (issue #1996). Requires fully-annotated tensor params."""

    def test_compile_from_signature_returns_compiled_program(self):
        # No torch tensors involved — pure metadata + compile pipeline.
        compiled = sig_kernel.compile()
        assert isinstance(compiled, CompiledProgram)

    def test_signature_and_tensor_share_cache(self):
        """``compile()`` (signature) and ``compile(sample_tensors)`` produce the
        same cached artifact — dynamic dims collapse to None in the cache key."""
        torch = pytest.importorskip("torch")

        from_sig = sig_kernel.compile()
        t = torch.zeros(256, 128, dtype=torch.float32)
        from_tensor = sig_kernel.compile(t, t)
        assert from_tensor is from_sig

    def test_signature_meta_matches_tensor_meta(self):
        """Metadata derived from the signature equals metadata from a tensor of
        any concrete extent (dynamic dim marked, static dim/dtype identical)."""
        torch = pytest.importorskip("torch")

        _, _, meta_sig, _, cx, _ = sig_kernel._bind_args_from_signature({})
        t = torch.zeros(512, 128, dtype=torch.float32)
        _, _, meta_tensor, _, cx, _ = sig_kernel._bind_args((t, t), {})
        for name in ("a", "c"):
            assert meta_sig[name].dynamic_dim_indices() == meta_tensor[name].dynamic_dim_indices() == {0}
            assert meta_sig[name].static_shape()[1] == meta_tensor[name].static_shape()[1] == 128
            assert meta_sig[name].dtype == meta_tensor[name].dtype == pl.FP32

    def test_signature_program_equals_tensor_program(self):
        """Specializing from the signature yields the same IR as from tensors."""
        torch = pytest.importorskip("torch")

        _, _, tm_s, sd_s, cx, dyn_s = sig_kernel._bind_args_from_signature({})
        prog_sig = sig_kernel._compile_to_program(tm_s, sd_s, cx, dyn_s, pl)

        t = torch.zeros(64, 128, dtype=torch.float32)
        _, _, tm_t, sd_t, cx, dyn_t = sig_kernel._bind_args((t, t), {})
        prog_tensor = sig_kernel._compile_to_program(tm_t, sd_t, cx, dyn_t, pl)

        ir.assert_structural_equal(prog_sig, prog_tensor)

    def test_bare_tensor_annotation_raises(self):
        """A bare ``pl.Tensor`` param has no shape to read — clear error."""
        # add_kernel's params are bare ``pl.Tensor`` (no subscript).
        with pytest.raises(TypeError, match="bare 'pl.Tensor'"):
            add_kernel.compile()

    def test_scalar_param_needs_no_value(self):
        """A scalar parameter is a runtime value, so the signature needs none.

        Its value arrives at dispatch (issue #2751), so only the declared dtype
        is read here. A keyword is still accepted — it used to mean "specialize
        this value", so it warns rather than changing meaning silently.
        """

        s_m = pl.dynamic("SM")

        @jit
        def scalar_sig_kernel(
            a: pl.Tensor[[s_m, 64], pl.FP16],
            n: pl.Scalar[pl.INT32],
            c: pl.Out[pl.Tensor[[s_m, 64], pl.FP16]],
        ):
            c = a
            return c

        _, _, _, scalar_dtypes, cx, _ = scalar_sig_kernel._bind_args_from_signature({})
        assert scalar_dtypes == {"n": pl.INT32}

        with pytest.warns(DeprecationWarning, match="no longer folds that value"):
            _, _, _, kw_dtypes, cx, _ = scalar_sig_kernel._bind_args_from_signature({"n": 7})
        assert kw_dtypes == {"n": pl.INT32}

    def test_runtime_marker_still_accepted(self):
        """``pl.RUNTIME`` (issue #2283) is now what every scalar does by default.

        It stays accepted so existing signatures keep working; the resulting
        metadata is the dtype alone, exactly as when nothing is passed.
        """
        _, _, _, marked, cx, _ = rt_scalar_kernel._bind_args_from_signature({"n": pl.RUNTIME})
        _, _, _, unmarked, cx, _ = rt_scalar_kernel._bind_args_from_signature({})
        assert marked == unmarked == {"n": pl.FP32}

    def test_scalar_stays_symbolic_in_program_whatever_was_passed(self):
        """A scalar is symbolic end to end — in the entry *and* in the incore dep
        it is forwarded to — whether the caller marked it ``pl.RUNTIME``, passed
        a literal, or passed nothing (issue #2751).

        A literal used to be folded here, which is what made one artifact per
        value and left the declared parameter unused.
        """
        programs = []
        for kwargs in ({"n": pl.RUNTIME}, {"n": 7.0}, {}):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                _, _, tm, sd, cx, dyn = rt_scalar_kernel._bind_args_from_signature(kwargs)
            programs.append(str(rt_scalar_kernel._compile_to_program(tm, sd, cx, dyn, pl)))

        for prog in programs:
            # 'n' is a parameter of both the entry and the dep, and every use
            # forwards or consumes the symbol rather than a constant.
            assert prog.count("n: pl.Scalar[pl.FP32]") == 2
            assert "self._rt_add_scalar_incore(a, n, c)" in prog
            assert "pl.tile.adds(tile, n)" in prog
        assert programs[0] == programs[1] == programs[2]

    def test_runtime_scalar_forwards_dtype_to_dep(self):
        """A runtime scalar carries no value, but its dtype still reaches the dep
        it is forwarded to."""
        _, _, tm, sd, cx, dyn = rt_scalar_kernel._bind_args_from_signature({"n": pl.RUNTIME})
        contexts = rt_scalar_kernel._build_contexts(tm, sd, cx, dyn)
        dep_ctx = next(c for c in contexts if c.func_name == "_rt_add_scalar_incore")
        assert dep_ctx.scalar_dtypes == {"n": pl.FP32}

    def test_runtime_scalar_default_needs_no_keyword(self):
        """``pl.RUNTIME`` as the signature default makes the parameter runtime
        without the caller passing anything — through to the generated program
        (the specializer drops Python defaults, so the marker never leaks)."""
        _, _, tm, sd, cx, dyn = rt_scalar_default_kernel._bind_args_from_signature({})
        assert sd == {"n": pl.FP32}

        prog = str(rt_scalar_default_kernel._compile_to_program(tm, sd, cx, dyn, pl))
        assert "n: pl.Scalar[pl.FP32]" in prog
        assert "pl.RUNTIME" not in prog
        assert "pl.tile.adds(tile, n)" in prog

    def test_runtime_marker_rejected_on_the_dispatch_path(self):
        """``pl.RUNTIME`` is a compile-time marker. Binding real arguments — a
        dispatch call, or a sample-argument compile — must reject it by name
        instead of letting it reach the runtime as a bogus scalar. The signature
        default makes this reachable from a plain ``kernel(a, c)`` call."""
        torch = pytest.importorskip("torch")

        t = torch.zeros(256, 128, dtype=torch.float32)
        with pytest.raises(TypeError, match=r"'n' received pl\.RUNTIME"):
            rt_scalar_default_kernel._bind_args((t, t), {})
        with pytest.raises(TypeError, match=r"'n' received pl\.RUNTIME"):
            rt_scalar_kernel._bind_args((t, pl.RUNTIME, t), {})

    def test_scalar_value_never_splits_the_cache(self):
        """One artifact serves every scalar value (issue #2751).

        A literal used to compile its own artifact, so a caller that varied a
        token count or an offset paid a compilation per value.
        """
        from_runtime = rt_scalar_kernel.compile(n=pl.RUNTIME)
        assert isinstance(from_runtime, CompiledProgram)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert rt_scalar_kernel.compile(n=7.0) is from_runtime
            assert rt_scalar_kernel.compile(n=9.0) is from_runtime
        assert rt_scalar_kernel.compile() is from_runtime

    def test_unsupported_scalar_value_points_at_runtime_marker(self):
        """A value that is neither a literal nor ``pl.RUNTIME`` names both paths."""
        with pytest.raises(TypeError, match=r"pl\.RUNTIME"):
            rt_scalar_kernel._bind_args_from_signature({"n": ctypes.c_int32()})

    def test_keyword_tensor_samples_use_tensor_mode(self):
        """Passing sample tensors by keyword (no positional args) must still bind
        through the tensor path, not silently enter signature mode. add_kernel's
        params are bare ``pl.Tensor`` — signature mode would raise; tensor mode
        reads the sample shapes and compiles."""
        torch = pytest.importorskip("torch")

        t = torch.zeros(32, 32, dtype=torch.float32)
        # All tensors by keyword: tensor mode binds them; no bare-Tensor error.
        compiled = add_kernel.compile(a=t, b=t, c=t)
        assert isinstance(compiled, CompiledProgram)

    def test_closure_scope_future_annotations(self):
        """A closure-defined kernel under ``from __future__ import annotations``
        references a dynvar captured as a closure free var. Signature mode must
        resolve it (via globals + closure free-vars), not fail to parse the
        string annotation."""
        import importlib.util  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        fixture_path = Path(__file__).parent / "_sig_closure_fixture.py"
        spec = importlib.util.spec_from_file_location("_sig_closure_fixture", fixture_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        kernel = module.make_closure_kernel()
        _, _, tensor_meta, _, cx, _ = kernel._bind_args_from_signature({})
        assert tensor_meta["a"].dynamic_dim_indices() == {0}
        assert tensor_meta["a"].static_shape()[1] == 64
        assert tensor_meta["a"].dtype == pl.FP32


@jit
def _mx_kernel(a: pl.Tensor[[64, 128], pl.UINT8, pl.MX_A_ZZ], c: pl.Out[pl.Tensor[[64, 128], pl.UINT8]]):
    M, N = a.shape
    t = pl.load(a, [0, 0], [M, N], target_memory=pl.Mem.Mat)
    pl.store(t, [0, 0], c)
    return c


@jit
def _dn_kernel(a: pl.Tensor[[64, 128], pl.FP16, pl.DN], c: pl.Out[pl.Tensor[[64, 128], pl.FP16]]):
    M, N = a.shape
    t = pl.load(a, [0, 0], [M, N])
    pl.store(t, [0, 0], c)
    return c


class TestAnnotationLayoutReachesTheProgram:
    """The layout slot of a @pl.jit parameter annotation must reach the IR.

    JIT specialization regenerates each annotation from ``TensorMeta`` rather
    than reusing the user's source. ``TensorMeta`` carried only shape and dtype,
    so the slot was dropped: a layout silently became ND, and ``pl.DN`` — which
    the type resolver rejects — was never even seen, so it neither errored nor
    took effect.

    The bar is parity with ``@pl.function``: whatever the bare slot means there,
    it must mean here. Which layouts the *pipeline* then accepts is a separate,
    path-independent question — see :class:`TestNzOnTensorIsNotJitSpecific`.
    """

    def _entry_param_type(self, kernel):
        _, _, tm, sd, cx, dyn = kernel._bind_args_from_signature({})
        program = kernel._compile_to_program(tm, sd, cx, dyn, pl)
        return list(program.functions.values())[0].params[0].type

    def test_layout_reaches_the_param_type(self):
        """MX_A_ZZ is a layout the pipeline genuinely accepts on a TensorType."""
        param_type = self._entry_param_type(_mx_kernel)
        assert param_type.tensor_view is not None
        assert param_type.tensor_view.layout == ir.TensorLayout.MX_A_ZZ

    def test_unannotated_layout_stays_absent(self):
        """The plain two-slot form must not gain a view."""
        _, _, tm, sd, cx, dyn = _mx_kernel._bind_args_from_signature({})
        program = _mx_kernel._compile_to_program(tm, sd, cx, dyn, pl)
        out_param = list(program.functions.values())[0].params[1]
        assert out_param.type.tensor_view is None

    def test_dn_layout_is_rejected(self):
        """DN reaches the resolver now, so its rejection applies here too."""
        with pytest.raises(ParserTypeError, match=r"pl\.Tensor\[\.\.\., pl\.DN\] is not supported"):
            self._entry_param_type(_dn_kernel)

    def test_dn_rejection_points_at_the_user_source(self):
        """The span must name this test file, not the generated ``<jit:...>``."""
        with pytest.raises(ParserTypeError) as exc_info:
            self._entry_param_type(_dn_kernel)

        span = exc_info.value.span
        assert span is not None
        assert span["filename"].endswith("test_jit_compile_extraction.py")


@jit.incore
def _mx_dep(a: pl.Tensor[[64, 128], pl.UINT8, pl.MX_A_ZZ], c: pl.Out[pl.Tensor[[64, 128], pl.UINT8]]):
    M, N = a.shape
    t = pl.load(a, [0, 0], [M, N], target_memory=pl.Mem.Mat)
    pl.store(t, [0, 0], c)
    return c


@jit
def _calls_mx_dep(a: pl.Tensor[[64, 128], pl.UINT8], c: pl.Out[pl.Tensor[[64, 128], pl.UINT8]]):
    c = _mx_dep(a, c)
    return c


class TestDepDeclaredLayout:
    """A dep's own layout declaration has no caller-side counterpart.

    The caller's argument meta reflects the *caller's* annotation, so a dep
    declaring ``pl.Tensor[[...], pl.MX_A_ZZ]`` while the entry declares none would
    otherwise compile as ND — the same silent downgrade, one call deeper.
    """

    def test_dep_layout_survives_when_caller_declares_none(self):
        _, _, tm, sd, cx, dyn = _calls_mx_dep._bind_args_from_signature({})
        program = _calls_mx_dep._compile_to_program(tm, sd, cx, dyn, pl)
        views = [
            p.type.tensor_view
            for f in program.functions.values()
            for p in f.params
            if getattr(p.type, "tensor_view", None) is not None
        ]
        assert any(v.layout == ir.TensorLayout.MX_A_ZZ for v in views)


class TestUnsupportedLayoutSlot:
    """A ``pl.TensorView`` in the slot must be refused, never dropped.

    ``TensorMeta`` has nowhere to carry a view, and a dropped stride is silent
    wrong code. The DN rejection's own hint points users at this spelling, so
    the refusal has to be explicit rather than a quiet ND.
    """

    def test_tensorview_slot_raises(self):
        strided = pl.TensorView(stride=[256, 1], layout=ir.TensorLayout.ND)

        @jit
        def kernel(a: pl.Tensor[[64, 128], pl.FP16, strided], c: pl.Out[pl.Tensor[[64, 128], pl.FP16]]):
            M, N = a.shape
            t = pl.load(a, [0, 0], [M, N])
            pl.store(t, [0, 0], c)
            return c

        with pytest.raises(TypeError, match="does not yet support"):
            kernel._bind_args_from_signature({})


class TestNzOnTensorIsNotJitSpecific:
    """``pl.NZ`` in a *tensor* annotation reaches the param type on every path.

    NZ on a TensorType asserts that the GM bytes are already in PTO-native NZ
    fractal order; ``BlockNzTensorViews`` later rewrites the shape into the
    blocked rank-5 form pto-isa needs. What matters here is only that the
    annotation *survives specialization* — dropping it is what silently produced
    an ND buffer from an NZ annotation.

    ``@pl.function`` behaves identically, so @pl.jit carrying the layout through
    is parity. These tests deliberately stop at the param type: the blocking and
    its diagnostics are covered by
    tests/ut/ir/transforms/test_block_nz_tensor_views.py.
    """

    def test_nz_survives_specialization_unchanged(self):
        """Pre-pass, the annotation is carried verbatim — same as @pl.function."""

        @jit
        def kernel(a: pl.Tensor[[64, 128], pl.FP16, pl.NZ], c: pl.Out[pl.Tensor[[64, 128], pl.FP16]]):
            M, N = a.shape
            t = pl.load(a, [0, 0], [M, N])
            pl.store(t, [0, 0], c)
            return c

        _, _, tm, sd, cx, dyn = kernel._bind_args_from_signature({})
        assert tm["a"].layout == ir.TensorLayout.NZ
        program = kernel._compile_to_program(tm, sd, cx, dyn, pl)
        view = list(program.functions.values())[0].params[0].type.tensor_view
        assert view is not None and view.layout == ir.TensorLayout.NZ

    def test_pl_function_carries_nz_the_same_way(self):
        """The @pl.function path reaches the identical IR, confirming parity."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[64, 128], pl.FP16, pl.NZ],
                c: pl.Out[pl.Tensor[[64, 128], pl.FP16]],
            ):
                with pl.at(level=pl.Level.CORE_GROUP):
                    t = pl.load(a, [0, 0], [64, 128])
                    pl.store(t, [0, 0], c)
                return c

        param_type = list(Prog.functions.values())[0].params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert param_type.tensor_view is not None
        assert param_type.tensor_view.layout == ir.TensorLayout.NZ


def _request_kernel(x: pl.Tensor[[32, 32], pl.FP32]) -> pl.Tensor[[32, 32], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        result = pl.add(x, x)
    return result


@pytest.fixture
def kernel(monkeypatch):
    monkeypatch.delenv("PYPTO_COMPILE_PROFILING", raising=False)
    monkeypatch.setattr(CompileProfiler._local, "current", None, raising=False)
    return pl.jit(_request_kernel)


@pytest.fixture
def compile_calls(kernel, monkeypatch):
    calls = []

    def record_compile(*_args, **kwargs):
        artifact = object()
        calls.append((kwargs, artifact))
        return artifact

    monkeypatch.setattr(kernel, "_compile", record_compile)
    return calls


def test_default_and_runtime_only_requests_share_cache(kernel, compile_calls):
    cached = kernel.compile()
    for config in (
        RunConfig(),
        RunConfig(dump_passes=PassDumpLevel.NONE),
        RunConfig(device_id=3, codegen_only=True, save_kernels=True),
    ):
        assert kernel.compile(config=config) is cached
    with passes.PassContext([]):
        assert kernel.compile() is cached
    assert len(compile_calls) == 1
    kwargs = compile_calls[0][0]
    assert kwargs["platform"] == "a2a3sim"
    assert kwargs["dump_passes"] is PassDumpLevel.NONE


@pytest.mark.parametrize(
    "options",
    [
        {"dump_passes": True},
        {"dump_passes": PassDumpLevel.EXPLICIT},
        {"dump_ptoas_passes": True},
        {"compile_profiling": True},
        {"save_kernels_dir": "requested-output"},
        {"diagnostic_phase": passes.DiagnosticPhase.PRE_PIPELINE},
        {"disabled_diagnostics": passes.DiagnosticCheckSet()},
    ],
)
@pytest.mark.parametrize("warm", [False, True])
def test_diagnostic_requests_neither_lookup_nor_insert(kernel, compile_calls, monkeypatch, options, warm):
    cached = kernel.compile() if warm else None
    before = dict(kernel._cache)

    def fail_key():
        pytest.fail("diagnostic request constructed a cache key")

    with monkeypatch.context() as patch:
        patch.setattr(kernel, "_get_source_hash", fail_key)
        first = kernel.compile(config=RunConfig(**options))
        second = kernel.compile(config=RunConfig(**options))
    assert first is not second
    assert kernel._cache == before
    assert len(compile_calls) == (3 if warm else 2)
    if warm:
        assert kernel.compile() is cached


@pytest.mark.parametrize("env_name", ["PYPTO_COMPILE_PROFILING"])
def test_environment_request_bypasses_warm_cache(kernel, compile_calls, monkeypatch, tmp_path, env_name):
    kernel.compile()
    before = dict(kernel._cache)
    monkeypatch.setenv(env_name, "1" if env_name == "PYPTO_COMPILE_PROFILING" else str(tmp_path))
    first = kernel.compile()
    assert kernel.compile() is not first
    assert len(compile_calls) == 3
    assert kernel._cache == before


def test_build_directory_reuses_private_compilation_and_separates_roots(kernel, monkeypatch, tmp_path):
    # Persistence is explicitly disabled by the frontend fixture. Output roots
    # must still participate in object selection so a new root receives files.
    ordinary = kernel.compile()
    root = tmp_path / "requested"
    monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", str(root))
    first = kernel.compile()
    assert first is not ordinary and first.output_dir.parent == root
    assert kernel.compile() is first
    assert (first.output_dir / "compiled_meta.json").is_file()
    monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", str(tmp_path / "other"))
    second = kernel.compile()
    assert second is not first and second.output_dir.parent == tmp_path / "other"
    monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", str(root))
    assert kernel.compile() is first
    monkeypatch.delenv("PYPTO_PROG_BUILD_DIR")
    assert kernel.compile() is ordinary


def test_active_profiler_bypasses_warm_cache(kernel, compile_calls):
    cached = kernel.compile()
    with CompileProfiler():
        assert kernel.compile() is not cached
        assert kernel.compile() is not cached
    assert kernel.compile() is cached
    assert len(compile_calls) == 3


@pytest.mark.parametrize(
    "context_options",
    [
        {"verification_level": passes.VerificationLevel.ROUNDTRIP},
        {"diagnostic_phase": passes.DiagnosticPhase.POST_PASS},
        {"disabled_diagnostics": passes.DiagnosticCheckSet()},
    ],
)
def test_custom_pass_checks_bypass_warm_cache(kernel, compile_calls, context_options):
    cached = kernel.compile()
    with passes.PassContext([], **context_options):
        assert kernel.compile() is not cached
        assert kernel.compile() is not cached
    assert kernel.compile() is cached
    assert len(compile_calls) == 3


def test_pass_context_conflict_is_rejected_before_cache_lookup(kernel, compile_calls):
    config = RunConfig(memory_planner=passes.MemoryPlanner.PYPTO)
    kernel.compile(config=config)
    with passes.PassContext([]), pytest.raises(RuntimeError, match="memory_planner.*PassContext"):
        kernel.compile(config=config)
    assert len(compile_calls) == 1


def test_failed_diagnostic_compile_preserves_ordinary_entry(kernel, compile_calls, monkeypatch):
    cached = kernel.compile()

    def fail_compile(*_args, **_kwargs):
        raise ValueError("diagnostic compilation failed")

    with monkeypatch.context() as patch:
        patch.setattr(kernel, "_compile", fail_compile)
        with pytest.raises(ValueError, match="diagnostic compilation failed"):
            kernel.compile(config=RunConfig(dump_passes=True))
    assert kernel.compile() is cached
    assert len(compile_calls) == 1


def test_source_locations_use_captured_effective_option(kernel, compile_calls, monkeypatch):
    monkeypatch.setenv("PYPTO_EMIT_PTO_LOC", "0")
    without_locations = kernel.compile()
    monkeypatch.setenv("PYPTO_EMIT_PTO_LOC", "1")
    original_hash = kernel._get_source_hash

    def change_environment_after_resolution():
        monkeypatch.setenv("PYPTO_EMIT_PTO_LOC", "0")
        return original_hash()

    monkeypatch.setattr(kernel, "_get_source_hash", change_environment_after_resolution)
    assert kernel.compile() is not without_locations
    assert [kwargs["emit_source_loc"] for kwargs, _ in compile_calls] == [False, True]
    assert kernel.compile() is without_locations


def test_real_dumps_are_regenerated_after_ordinary_cache_hit(kernel, tmp_path):
    cached = kernel.compile()
    assert not (cached.output_dir / "passes_dump").exists()
    for name in ("first", "second"):
        output = tmp_path / name
        config = RunConfig(save_kernels_dir=str(output), dump_passes=True)
        compiled = kernel.compile(config=config)
        assert compiled.output_dir == output
        dumps = list((output / "passes_dump").glob("*.py"))
        assert dumps
        dump = dumps[0]
        expected = dump.read_bytes()
        dump.unlink()
        assert kernel.compile(config=config) is not compiled
        assert dump.read_bytes() == expected
    assert kernel.compile() is cached


def test_real_outer_instrument_runs_after_ordinary_cache_hit(kernel, tmp_path):
    cached = kernel.compile()
    calls = []
    callback = passes.CallbackInstrument(
        before_pass=lambda pass_obj, _program: calls.append(pass_obj.get_name()), name="observer"
    )
    with passes.PassContext([callback, passes.ReportInstrument(str(tmp_path))]):
        first = kernel.compile()
        assert first.output_dir != cached.output_dir
        assert calls
        calls.clear()
        second = kernel.compile()
        assert second.output_dir not in (cached.output_dir, first.output_dir)
        assert calls
    assert (tmp_path / "perf_hints.log").is_file()
    assert kernel.compile() is cached


def test_real_profile_contains_compile_stages_after_cache_hit(kernel):
    cached = kernel.compile()
    with CompileProfiler() as profiler:
        compiled = kernel.compile()
        assert compiled.output_dir != cached.output_dir
    assert profiler.to_dict()["stages"]
    assert kernel.compile() is cached


@pytest.mark.parametrize("fail", [False, True])
def test_disabling_environment_profiling_restores_warm_cache(kernel, monkeypatch, fail):
    """Real successful and failed compiles must not make profiling sticky."""
    cached = kernel.compile()
    monkeypatch.setenv("PYPTO_COMPILE_PROFILING", "1")
    if fail:

        def fail_parse(*_args, **_kwargs):
            raise ValueError("profiling compilation failed")

        with monkeypatch.context() as patch:
            patch.setattr(pl, "parse", fail_parse)
            with pytest.raises(ValueError, match="profiling compilation failed"):
                kernel.compile()
    else:
        assert kernel.compile() is not cached
    monkeypatch.delenv("PYPTO_COMPILE_PROFILING")
    assert kernel.compile() is cached
    assert CompileProfiler.current() is None


def test_warm_cache_hit_does_not_probe_toolchain(kernel, monkeypatch):
    cached = kernel.compile()

    def fail_discovery():
        pytest.fail("warm cache hit probed the toolchain")

    monkeypatch.setattr(importlib.import_module("pypto.jit.decorator"), "find_ptoas_binary", fail_discovery)
    assert kernel.compile() is cached


@jit.incore
def _constexpr_dep(
    x: pl.Tensor[[32, 32], pl.FP32],
    out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    scale: pl.Scalar[pl.FP32],
    BLOCK: pl.constexpr,
):
    tile = pl.load(x, [0, 0], [BLOCK, BLOCK])
    pl.store(pl.add(tile, scale), [0, 0], out)
    return out


@jit
def _constexpr_entry(
    x: pl.Tensor[[32, 32], pl.FP32],
    out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    scale: pl.Scalar[pl.FP32],
    BLOCK: pl.constexpr,
):
    return _constexpr_dep(x, out, scale, BLOCK)


_DEP_TILE = 8


@jit.incore
def _module_constant_dep(
    x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]], N: pl.constexpr
):
    pl.store(pl.load(x, [0, 0], [N, N]), [0, 0], out)
    return out


@jit.incore
def _literal_forms_dep(
    a: pl.Tensor[[32, 32], pl.FP32],
    o: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    BIAS: pl.constexpr,
    SHAPE: pl.constexpr,
    MEM: pl.constexpr,
):
    tile = pl.load(a, [0, 0], SHAPE, target_memory=MEM)
    pl.store(pl.add(tile, BIAS), [0, 0], o)
    return o


@jit
def _literal_forms_entry(a: pl.Tensor[[32, 32], pl.FP32], o: pl.Out[pl.Tensor[[32, 32], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        pass
    return _literal_forms_dep(a, o, -1, [16, 32], pl.Mem.Vec)


@jit
def _slice_write_literal(x: pl.Tensor[[128, 128], pl.FP32], out: pl.Out[pl.Tensor[[128, 128], pl.FP32]]):
    out[0:64, 0:128] = pl.add(x[0:64, 0:128], 1.0)
    return out


@jit
def _slice_write_constexpr(
    x: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    BLOCK: pl.constexpr,
):
    out[0:BLOCK, 0:128] = pl.add(x[0:BLOCK, 0:128], 1.0)
    return out


@jit
def _rebind_plain(
    x: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    BLOCK: pl.constexpr,
):
    BLOCK = 32
    out[0:BLOCK, 0:128] = pl.add(x[0:BLOCK, 0:128], 1.0)
    return out


@jit
def _rebind_augmented(
    x: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    BLOCK: pl.constexpr,
):
    BLOCK += 1
    out[0:BLOCK, 0:128] = pl.add(x[0:BLOCK, 0:128], 1.0)
    return out


@jit
def _rebind_unpacked(
    x: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    BLOCK: pl.constexpr,
):
    _unused, BLOCK = 1, 32
    out[0:BLOCK, 0:128] = pl.add(x[0:BLOCK, 0:128], 1.0)
    return out


@jit
def _rebind_loop_target(
    x: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    BLOCK: pl.constexpr,
):
    # PLR1704 is the point: this fixture exists to be refused by the specializer.
    for BLOCK in pl.range(2):  # noqa: PLR1704
        out[0:64, 0:128] = pl.add(x[0:64, 0:128], 1.0)
    return out


def _generated_functions(source: str) -> dict[str, str]:
    """Split generated ``@pl.program`` source into ``{function name: body text}``.

    Lets a test assert which *specific* generated function a constant folded
    into, rather than only that the constant appears somewhere in the program —
    the distinction that matters once one dep is emitted once per binding.
    """
    bodies: dict[str, str] = {}
    current: str | None = None
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") and "(" in stripped:
            current = stripped[len("def ") : stripped.index("(")]
            bodies[current] = ""
        elif current is not None:
            bodies[current] += line + "\n"
    return bodies


class TestConstexprThroughDeps:
    """A compile-time parameter keeps its meaning across a JIT call (issue #2759)."""

    @pytest.fixture
    def samples(self):
        torch = pytest.importorskip("torch")
        x = torch.zeros(32, 32, dtype=torch.float32)
        return x, torch.zeros_like(x)

    def test_forwarded_constant_folds_in_the_dep(self, samples):
        """The entry's constant reaches the dep body, and the call drops the arg.

        The dep no longer declares the parameter, so a forwarded argument would
        be an arity mismatch at the generated call site.
        """
        x, out = samples
        source = _constexpr_entry.specialize(x, out, 1.0, 16).as_python()

        assert "pl.tile.load(x, [0, 0], [16, 16]" in source
        assert "self._constexpr_dep(x, out, scale)" in source
        assert "BLOCK" not in source

        other = _constexpr_entry.specialize(x, out, 1.0, 32).as_python()
        assert "pl.tile.load(x, [0, 0], [32, 32]" in other

    def test_a_literal_at_the_dep_call_site_binds_it(self, samples):
        x, out = samples

        @jit
        def entry(x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                pass
            return _module_constant_dep(x, out, 4)

        assert "[4, 4]" in entry.specialize(x, out).as_python()

    def test_a_module_constant_at_the_dep_call_site_binds_it(self, samples):
        x, out = samples

        @jit
        def entry(x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                pass
            return _module_constant_dep(x, out, _DEP_TILE)

        assert "[8, 8]" in entry.specialize(x, out).as_python()

    def test_two_call_sites_with_different_constants_each_get_a_function(self, samples):
        """One generated function per *binding*, so both constants survive.

        The value is folded into the body, so a single generated function
        cannot serve both call sites. Each binding is emitted separately and
        each site is rewritten to reach its own.
        """
        x, out = samples

        @jit
        def entry(x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                pass
            out = _module_constant_dep(x, out, 4)
            return _module_constant_dep(x, out, 16)

        source = entry.specialize(x, out).as_python()
        bodies = _generated_functions(source)

        assert "[4, 4]" in bodies["_module_constant_dep"]
        assert "[16, 16]" in bodies["_module_constant_dep__2"]
        # Each site reaches its own compilation, in source order — the first
        # keeps the unsuffixed name.
        called = re.findall(r"self\.(_module_constant_dep(?:__\d+)?)\(", source)
        assert called == ["_module_constant_dep", "_module_constant_dep__2"]

    def test_call_sites_that_agree_still_share_one_function(self, samples):
        """Two sites at the same value must not split into two functions.

        The binding is the identity, not the call site, so agreeing sites
        collapse — otherwise every repeated call would duplicate the callee.
        """
        x, out = samples

        @jit
        def entry(x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                pass
            out = _module_constant_dep(x, out, 16)
            return _module_constant_dep(x, out, 16)

        source = entry.specialize(x, out).as_python()

        assert "def _module_constant_dep(" in source
        assert "def _module_constant_dep__2(" not in source

    def test_every_documented_literal_form_binds_at_a_dep_call_site(self, samples):
        """A dep must accept the value forms the entry path accepts.

        Only ``ast.Constant`` reached the renderer at first, so a negative
        number, a list and an enum member — ``UnaryOp`` / ``List`` /
        ``Attribute`` nodes — were rejected as having no compile-time value.
        """
        x, out = samples
        source = _literal_forms_entry.specialize(x, out).as_python()

        assert "[16, 32]" in source
        assert "Mem.Vec" in source
        assert "-1" in source

    def test_two_callers_of_one_dep_each_get_a_function(self, samples):
        """Divergence across *callers*, not just within one body, splits too.

        A diamond reaches the dep down two branches. Resolving from the
        first-recorded caller would fold its value and hand the other branch
        the wrong constant, so each branch gets its own compilation.
        """
        x, out = samples

        @jit.incore
        def shared(
            a: pl.Tensor[[32, 32], pl.FP32],
            o: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            N: pl.constexpr,
        ):
            pl.store(pl.load(a, [0, 0], [N, N]), [0, 0], o)
            return o

        @jit.inline
        def left(a: pl.Tensor[[32, 32], pl.FP32], o: pl.Out[pl.Tensor[[32, 32], pl.FP32]]):
            return shared(a, o, 16)

        @jit.inline
        def right(a: pl.Tensor[[32, 32], pl.FP32], o: pl.Out[pl.Tensor[[32, 32], pl.FP32]]):
            return shared(a, o, 32)

        @jit
        def diamond(a: pl.Tensor[[32, 32], pl.FP32], o: pl.Out[pl.Tensor[[32, 32], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                pass
            o = left(a, o)
            return right(a, o)

        bodies = _generated_functions(diamond.specialize(x, out).as_python())

        assert "[16, 16]" in bodies["shared"]
        assert "[32, 32]" in bodies["shared__2"]

    def test_splitting_survives_an_aliased_call_name(self, samples):
        """The alias and the split resolve the callee together, not in turn.

        ``dep_func_names`` maps a call name onto the generated function it
        reaches; the split needs a *different* function per site of that same
        name. Resolving either one alone gives the wrong callee.
        """
        x, out = samples
        aliased = _module_constant_dep

        @jit
        def entry(a: pl.Tensor[[32, 32], pl.FP32], o: pl.Out[pl.Tensor[[32, 32], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                pass
            o = aliased(a, o, 8)
            return aliased(a, o, 24)

        source = entry.specialize(x, out).as_python()
        bodies = _generated_functions(source)

        # Named after the callee, not the alias, and one per binding.
        assert "[8, 8]" in bodies["_module_constant_dep"]
        assert "[24, 24]" in bodies["_module_constant_dep__2"]
        called = re.findall(r"self\.(_module_constant_dep(?:__\d+)?)\(", source)
        assert called == ["_module_constant_dep", "_module_constant_dep__2"]

    def test_a_forwarded_constant_splits_the_whole_chain(self, samples):
        """Splitting a dep splits everything it forwards the value to.

        ``entry -> mid(N) -> leaf(N)`` at two values needs two ``mid``s *and*
        two ``leaf``s, with each ``mid`` calling its own ``leaf`` — one shared
        ``leaf`` would serve one of them the other's constant.
        """
        x, out = samples

        @jit.incore
        def leaf(
            a: pl.Tensor[[32, 32], pl.FP32],
            o: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            N: pl.constexpr,
        ):
            pl.store(pl.load(a, [0, 0], [N, N]), [0, 0], o)
            return o

        @jit.incore
        def mid(
            a: pl.Tensor[[32, 32], pl.FP32],
            o: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            N: pl.constexpr,
        ):
            return leaf(a, o, N)

        @jit
        def entry(a: pl.Tensor[[32, 32], pl.FP32], o: pl.Out[pl.Tensor[[32, 32], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                pass
            o = mid(a, o, 8)
            return mid(a, o, 24)

        bodies = _generated_functions(entry.specialize(x, out).as_python())

        assert "[8, 8]" in bodies["leaf"]
        assert "[24, 24]" in bodies["leaf__2"]
        assert "self.leaf(" in bodies["mid"]
        assert "self.leaf__2(" in bodies["mid__2"]

    def test_each_specialization_reads_its_own_call_site_metadata(self, samples):
        """A split dep must take its tensors from the call that produced it.

        Call-site arguments were resolved from the first call bearing the
        name. That was sound while one binding meant one function; once the
        second binding compiles separately it would fold its own constant over
        the *first* site's tensors — here a ``[32, 32]`` extent loaded from a
        ``[16, 16]`` parameter.
        """
        x, out = samples

        @jit.incore
        def sized(a: pl.Tensor, o: pl.Out[pl.Tensor], N: pl.constexpr):
            pl.store(pl.load(a, [0, 0], [N, N]), [0, 0], o)
            return o

        @jit
        def entry(a: pl.Tensor[[32, 32], pl.FP32], o: pl.Out[pl.Tensor[[32, 32], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                pass
            small = pl.create_tensor([16, 16], dtype=pl.FP32)
            big = pl.create_tensor([32, 32], dtype=pl.FP32)
            sized(small, small, 16)
            sized(big, big, 32)
            return o

        source = entry.specialize(x, out).as_python()
        signatures = {
            line.strip()[len("def ") : line.strip().index("(")]: line.strip()
            for line in source.splitlines()
            if line.strip().startswith("def sized")
        }

        assert "pl.Tensor[[16, 16], pl.FP32]" in signatures["sized"]
        assert "pl.Tensor[[32, 32], pl.FP32]" in signatures["sized__2"]

    def test_assigning_to_a_constexpr_parameter_is_rejected(self, samples):
        """A rebind cannot take effect, so it must not compile silently.

        Every load folds to the call-site value, so ``BLOCK = BLOCK // 2``
        left the later loads on the original constant and leaked the
        assignment into the body as a stray runtime local.
        """
        x, out = samples

        @jit
        def rebind(
            a: pl.Tensor[[32, 32], pl.FP32],
            o: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            BLOCK: pl.constexpr,
        ):
            BLOCK = BLOCK // 2
            with pl.at(level=pl.Level.CORE_GROUP):
                pl.store(pl.load(a, [0, 0], [BLOCK, BLOCK]), [0, 0], o)
            return o

        with pytest.raises(ValueError, match=r"'BLOCK' is a 'pl.constexpr' parameter"):
            rebind.specialize(x, out, 32)

    def test_a_constexpr_read_in_a_slice_assign_target_is_not_a_rebind(self):
        """Reading a constexpr inside a store target must stay legal.

        ``out[0:BLOCK, ...] = ...`` stores through a subscript whose *slice*
        loads ``BLOCK``. Walking the target for any occurrence rejected it,
        refusing the most natural use of a compile-time extent while the same
        statement written with a literal compiled.
        """
        torch = pytest.importorskip("torch")
        a = torch.zeros(128, 128, dtype=torch.float32)
        out = torch.zeros_like(a)

        literal = _slice_write_literal.specialize(a, out).as_python()
        folded = _slice_write_constexpr.specialize(a, out, 64).as_python()
        assert "64" in literal
        assert "64" in folded

    @pytest.mark.parametrize(
        "kernel",
        ["_rebind_plain", "_rebind_augmented", "_rebind_unpacked", "_rebind_loop_target"],
    )
    def test_every_binding_form_of_a_constexpr_is_rejected(self, kernel):
        """The Store-only rule must still catch each way a name can be bound."""
        torch = pytest.importorskip("torch")
        a = torch.zeros(128, 128, dtype=torch.float32)

        with pytest.raises(ValueError, match=r"'BLOCK' is a 'pl.constexpr' parameter"):
            globals()[kernel].specialize(a, torch.zeros_like(a), 64)

    def test_a_runtime_scalar_at_the_dep_call_site_is_rejected(self, samples):
        """A value that only exists at dispatch cannot fill a compile-time slot."""
        x, out = samples

        @jit
        def entry(
            x: pl.Tensor[[32, 32], pl.FP32],
            out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            n: pl.Scalar[pl.INT32],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                pass
            return _module_constant_dep(x, out, n)

        with pytest.raises(TypeError, match=r"no compile-time value"):
            entry.specialize(x, out, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
