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

import importlib

import pypto.language as pl
import pytest
from pypto.ir import OptimizationStrategy
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

    def test_call_then_compile_returns_cached_instance(self, monkeypatch):
        """``__call__`` and ``compile()`` share the same cached program."""
        torch = pytest.importorskip("torch")

        a = torch.zeros(96, 96, dtype=torch.float32)
        b = torch.zeros(96, 96, dtype=torch.float32)
        c = torch.empty(96, 96, dtype=torch.float32)

        compiled = add_kernel.compile(a, b, c)
        monkeypatch.setattr(CompiledProgram, "__call__", lambda self, *_args, **_kwargs: "called")
        assert add_kernel(a, b, c) == "called"
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
        x = torch.zeros(19, 19)

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
    in PR #1496 (chip_callable / build_orch_args / build_call_config),
    enabling worker integration as required by issue #1455.

    These tests only verify that the attributes are *defined on the class* —
    actually exercising compile_and_assemble (which several of these properties
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
        # property getters (chip_callable etc.) which call compile_and_assemble.
        for name in (
            "chip_callable",
            "runtime_name",
            "runtime_config",
            "build_orch_args",
            "build_call_config",
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

        _, _, meta_sig, _, _, _ = sig_kernel._bind_args_from_signature({})
        t = torch.zeros(512, 128, dtype=torch.float32)
        _, _, meta_tensor, _, _, _ = sig_kernel._bind_args((t, t), {})
        for name in ("a", "c"):
            assert meta_sig[name].dynamic_dim_indices() == meta_tensor[name].dynamic_dim_indices() == {0}
            assert meta_sig[name].static_shape()[1] == meta_tensor[name].static_shape()[1] == 128
            assert meta_sig[name].dtype == meta_tensor[name].dtype == pl.FP32

    def test_signature_program_equals_tensor_program(self):
        """Specializing from the signature yields the same IR as from tensors."""
        torch = pytest.importorskip("torch")

        _, _, tm_s, sv_s, sd_s, dyn_s = sig_kernel._bind_args_from_signature({})
        prog_sig = sig_kernel._compile_to_program(tm_s, sv_s, sd_s, dyn_s, pl)

        t = torch.zeros(64, 128, dtype=torch.float32)
        _, _, tm_t, sv_t, sd_t, dyn_t = sig_kernel._bind_args((t, t), {})
        prog_tensor = sig_kernel._compile_to_program(tm_t, sv_t, sd_t, dyn_t, pl)

        ir.assert_structural_equal(prog_sig, prog_tensor)

    def test_bare_tensor_annotation_raises(self):
        """A bare ``pl.Tensor`` param has no shape to read — clear error."""
        # add_kernel's params are bare ``pl.Tensor`` (no subscript).
        with pytest.raises(TypeError, match="bare 'pl.Tensor'"):
            add_kernel.compile()

    def test_scalar_param_needs_value(self):
        """Scalar params carry no value in the signature; must be supplied."""

        s_m = pl.dynamic("SM")

        @jit
        def scalar_sig_kernel(
            a: pl.Tensor[[s_m, 64], pl.FP16],
            n: pl.Scalar[pl.INT32],
            c: pl.Out[pl.Tensor[[s_m, 64], pl.FP16]],
        ):
            c = a
            return c

        with pytest.raises(TypeError, match="scalar parameter 'n'"):
            scalar_sig_kernel._bind_args_from_signature({})

        # Supplied via keyword: value flows into scalar_values.
        _, _, _, scalar_values, _, _ = scalar_sig_kernel._bind_args_from_signature({"n": 7})
        assert scalar_values == {"n": 7}

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
        _, _, tensor_meta, _, _, _ = kernel._bind_args_from_signature({})
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
        _, _, tm, sv, sd, dyn = kernel._bind_args_from_signature({})
        program = kernel._compile_to_program(tm, sv, sd, dyn, pl)
        return list(program.functions.values())[0].params[0].type

    def test_layout_reaches_the_param_type(self):
        """MX_A_ZZ is a layout the pipeline genuinely accepts on a TensorType."""
        param_type = self._entry_param_type(_mx_kernel)
        assert param_type.tensor_view is not None
        assert param_type.tensor_view.layout == ir.TensorLayout.MX_A_ZZ

    def test_unannotated_layout_stays_absent(self):
        """The plain two-slot form must not gain a view."""
        _, _, tm, sv, sd, dyn = _mx_kernel._bind_args_from_signature({})
        program = _mx_kernel._compile_to_program(tm, sv, sd, dyn, pl)
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
        _, _, tm, sv, sd, dyn = _calls_mx_dep._bind_args_from_signature({})
        program = _calls_mx_dep._compile_to_program(tm, sv, sd, dyn, pl)
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
    """``pl.NZ`` in a *tensor* annotation is rejected downstream on every path.

    NZ describes a tile layout; on a TensorType it survives parsing but ptoas
    later refuses it ("layout mismatch: user-specified layout=nz but
    inferred=nd"). That is true of ``@pl.function`` too, so @pl.jit carrying the
    layout through is parity, not a new defect — the alternative (dropping it)
    is what silently produced an ND buffer from an NZ annotation.

    This pins the shared behaviour so the propagation is not mistaken for a
    claim that NZ tensors compile.
    """

    def test_nz_survives_specialization_unchanged(self):
        """Pre-pass, the annotation is carried verbatim — same as @pl.function."""

        @jit
        def kernel(a: pl.Tensor[[64, 128], pl.FP16, pl.NZ], c: pl.Out[pl.Tensor[[64, 128], pl.FP16]]):
            M, N = a.shape
            t = pl.load(a, [0, 0], [M, N])
            pl.store(t, [0, 0], c)
            return c

        _, _, tm, sv, sd, dyn = kernel._bind_args_from_signature({})
        assert tm["a"].layout == ir.TensorLayout.NZ
        program = kernel._compile_to_program(tm, sv, sd, dyn, pl)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
