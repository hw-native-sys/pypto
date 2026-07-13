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

import pypto.language as pl
import pytest
from pypto.ir.compiled_program import CompiledProgram
from pypto.jit.decorator import jit
from pypto.pypto_core import ir
from pypto.runtime.runner import RunConfig


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

    def test_call_then_compile_returns_cached_instance(self):
        """``compile()`` after a regular call must return the cached CompiledProgram
        — the two entry points share a cache and never recompile for the same key.

        We cannot run ``__call__`` here without a device, but we can verify the
        cache invariant by populating it via ``compile_for_test`` (which uses
        the same key derivation) and checking ``compile`` is a hit.
        """
        torch = pytest.importorskip("torch")

        a = torch.zeros(96, 96, dtype=torch.float32)
        b = torch.zeros(96, 96, dtype=torch.float32)
        c = torch.empty(96, 96, dtype=torch.float32)

        # compile_for_test() also fills self._cache via _compile()
        add_kernel.compile_for_test(a, b, c)
        cache_len_before = len(add_kernel._cache)
        compiled = add_kernel.compile(a, b, c)
        assert isinstance(compiled, CompiledProgram)
        # No new entry — compile() hit the cache.
        assert len(add_kernel._cache) == cache_len_before

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
