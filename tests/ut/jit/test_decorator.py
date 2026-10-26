# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for @pl.jit decorator: decoration, cache hit/miss, and bind_dynamic."""

import ast
import importlib
import inspect
import re
import textwrap
import warnings
from copy import deepcopy
from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import backend
from pypto.ir import OptimizationStrategy, PassManager
from pypto.ir.compiled_program import CompiledProgram
from pypto.jit._arguments import (
    extract_tensor_meta,
    param_dtypes,
)
from pypto.jit._ast_analysis import (
    scan_dynamic_dims,
)
from pypto.jit._call_binding import (
    SlicedArg,
    arg_ref,
    build_param_mapping,
    compute_per_func_dyndim_maps,
    extract_call_args_for_dep,
)
from pypto.jit._tensor_meta import (
    _SYNTHESIZED_DYN_PREFIX,
    extract_local_tensor_metas,
    resolve_dep_call_metadata,
    scan_dep_io,
    synthesized_dyn_dim,
)
from pypto.jit.decorator import (
    JITFunction,
    _allocate_generated_names,
    _discover_dep_bindings,
    _discover_deps,
    _rewrite_jit_error,
    jit,
)
from pypto.jit.specializer import DynDim, Specializer, TensorMeta
from pypto.language.parser.diagnostics.exceptions import (
    InvalidOperationError,
    ParserTypeError,
    UnsupportedFeatureError,
)
from pypto.pypto_core import DataType, InternalError, ir
from pypto.runtime.runner import RunConfig

# ---------------------------------------------------------------------------
# Decoration tests (no torch needed)
# ---------------------------------------------------------------------------


class TestJitDecoration:
    def test_plain_jit_creates_jitfunction(self):
        @jit
        def my_kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        assert isinstance(my_kernel, JITFunction)

    def test_jit_preserves_name(self):
        @jit
        def my_kernel(a: pl.Tensor):
            return a

        assert my_kernel.__name__ == "my_kernel"

    def test_torch_fp32_without_expected_dtype_does_not_crash(self):
        """Unannotated FP32 must not hit DataType.__eq__(None) via ``in``."""
        torch = pytest.importorskip("torch")
        t = torch.empty((4, 8), dtype=torch.float32)
        meta = extract_tensor_meta(t)
        assert meta.dtype == DataType.FP32
        assert meta.static_shape() == (4, 8)

    def test_torch_index_annotation_accepts_int64_abi(self):
        """pl.INDEX is semantic; torch carries it as int64 — do not strict-mismatch."""
        torch = pytest.importorskip("torch")
        t = torch.empty((4,), dtype=torch.int64)
        # Even if a caller passes INDEX as expected_dtype, keep torch→INT64 ABI.
        meta = extract_tensor_meta(t, expected_dtype=DataType.INDEX)
        assert meta.dtype == DataType.INT64
        assert meta.static_shape() == (4,)

    def test_param_dtypes_only_records_packed_fp4_formats(self):
        """Only packed annotations select a numerical format for raw byte carriers."""

        def kernel(
            idx: pl.Tensor[[4], pl.INDEX],
            x: pl.Tensor[[8, 16], pl.FP32],
            packed: pl.Tensor[[8, 16], pl.FP4E2M1X2],
            e1m2: pl.Tensor[[8, 16], pl.fp4e1m2x2],
        ):
            return idx, x, packed, e1m2

        assert param_dtypes(kernel) == {
            "packed": DataType.FP4E2M1X2,
            "e1m2": DataType.FP4E1M2X2,
        }

    @pytest.mark.parametrize("dtype", [pl.fp4e2m1x2, pl.fp4e1m2x2])
    def test_packed_fp4_annotation_accepts_raw_byte_carriers(self, dtype):
        torch = pytest.importorskip("torch")
        data = torch.empty((8, 32), dtype=torch.uint8)
        meta = extract_tensor_meta(data, expected_dtype=dtype)
        assert meta.dtype == dtype
        assert meta.static_shape() == (8, 32)
        # Untyped bytes keep their type; only the explicit annotation selects FP4.
        assert extract_tensor_meta(data).dtype == DataType.UINT8

    def test_e1m2_rejects_torch_e2m1_format(self):
        torch = pytest.importorskip("torch")
        torch_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if torch_dtype is None:
            pytest.skip("torch.float4_e2m1fn_x2 required")
        with pytest.raises(TypeError, match="pass torch.uint8 carriers"):
            extract_tensor_meta(torch.empty((8, 32), dtype=torch_dtype), expected_dtype=pl.fp4e1m2x2)

    def test_torch_fp4_x2_inference_preserves_carrier_shape(self):
        """Native Torch x2 tensors use the same dtype and shape with or without an annotation."""
        torch = pytest.importorskip("torch")
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if fp4_dtype is None:
            pytest.skip("torch.float4_e2m1fn_x2 required")
        packed = torch.empty((128, 32), dtype=fp4_dtype)
        meta = extract_tensor_meta(packed)
        assert meta.dtype == DataType.FP4E2M1X2
        assert meta.static_shape() == (128, 32)

        meta_x2 = extract_tensor_meta(packed, expected_dtype=DataType.FP4E2M1X2)
        assert meta_x2.dtype == DataType.FP4E2M1X2
        assert meta_x2.static_shape() == (128, 32)

    def test_torch_fp4_x2_rejects_empty_packed_dimension(self):
        torch = pytest.importorskip("torch")
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if fp4_dtype is None:
            pytest.skip("torch.float4_e2m1fn_x2 required")
        packed = torch.empty((128, 0), dtype=fp4_dtype)
        with pytest.raises(TypeError, match="positive runtime x2 carrier last dimension"):
            extract_tensor_meta(packed)

    def test_jit_incore_creates_jitfunction(self):
        @jit.incore
        def sub_fn(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        assert isinstance(sub_fn, JITFunction)
        assert sub_fn._func_type == "incore"

    def test_jit_incore_with_level(self):
        @jit.incore(level=pl.Level.AIC)
        def aic_fn(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        assert isinstance(aic_fn, JITFunction)
        assert aic_fn._func_type == "incore"
        assert aic_fn._level == pl.Level.AIC

    def test_jit_entry_function_type(self):
        @jit
        def entry(a: pl.Tensor):
            return a

        assert entry._func_type == "orchestration"

    def test_jit_pl_access(self):
        """pl.jit should work the same as jit."""

        @pl.jit
        def kernel(a: pl.Tensor):
            return a

        assert isinstance(kernel, JITFunction)

    def test_jit_default_auto_scope_true(self):
        @jit
        def entry(a: pl.Tensor):
            return a

        assert entry._auto_scope is True

    def test_jit_auto_scope_false(self):
        @jit(auto_scope=False)
        def entry(a: pl.Tensor):
            return a

        assert isinstance(entry, JITFunction)
        assert entry._func_type == "orchestration"
        assert entry._auto_scope is False

    def test_jit_empty_parens_form(self):
        """@pl.jit() (bare parens) is equivalent to @pl.jit."""

        @jit()
        def entry(a: pl.Tensor):
            return a

        assert isinstance(entry, JITFunction)
        assert entry._auto_scope is True


class TestFp4DependencyContracts:
    """Packed dependencies preserve carrier shapes and numerical formats."""

    @pytest.mark.parametrize(
        "declared_dtype, caller_dtype",
        [
            (DataType.FP4E2M1X2, DataType.FP4E2M1X2),
            (DataType.FP4E1M2X2, DataType.FP4E1M2X2),
            (DataType.FP32, DataType.FP4E2M1X2),
        ],
    )
    @pytest.mark.parametrize("width", [31, DynDim(name="width", literal="width", static_bound=32)])
    def test_dep_metadata_keeps_carrier_shape(self, declared_dtype, caller_dtype, width):
        @jit.incore
        def dep(src: pl.Tensor[[16, 32], declared_dtype]):
            return src

        def caller(src: pl.Tensor):
            dep(src)

        rows = DynDim(name="rows", literal="rows", static_bound=16)
        original = TensorMeta(shape=(rows, width), dtype=caller_dtype, layout=ir.TensorLayout.NZ)
        metas, _ = resolve_dep_call_metadata(
            dep, caller, {"src": original}, {}, {}, discover_deps=_discover_dep_bindings
        )
        assert metas["src"] == original
        assert metas["src"].shape[0] is rows
        assert original.shape == (rows, width)
        assert original.dtype == caller_dtype

    @pytest.mark.parametrize("return_kind", ["identity", "allocated", "out"])
    @pytest.mark.parametrize("dtype", [DataType.FP4E2M1X2, DataType.FP4E1M2X2])
    def test_returned_metadata_keeps_carrier_shape(self, return_kind, dtype):
        @jit.inline
        def identity(src: pl.Tensor[[16, 32], dtype]):
            return src

        @jit.inline
        def allocated(src: pl.Tensor[[16, 32], dtype]):
            result = pl.create_tensor([16, 32], dtype=dtype)
            return result

        @jit.incore
        def out(src: pl.Out[pl.Tensor[[16, 32], dtype]]):
            return src

        producer = {"identity": identity, "allocated": allocated, "out": out}[return_kind]

        @jit.incore
        def consume(src: pl.Tensor):
            return src

        @jit
        def entry(src: pl.Tensor):
            packed = producer(src)
            consume(packed)
            return packed

        seed = {"src": TensorMeta(shape=(16, 32), dtype=dtype)}
        contexts = entry._build_contexts(seed, {}, {}, {})
        consumer = next(ctx for ctx in contexts if ctx.func_name == "consume")
        assert consumer.tensor_meta["src"] == TensorMeta(shape=(16, 32), dtype=dtype)
        assert seed["src"] == TensorMeta(shape=(16, 32), dtype=dtype)

    @pytest.mark.parametrize(
        "declared_dtype, caller_dtype",
        [
            (DataType.FP4E2M1X2, DataType.FP32),
            (DataType.FP4E1M2X2, DataType.FP32),
            (DataType.FP4E1M2X2, DataType.FP4E2M1X2),
            (DataType.FP4E2M1X2, DataType.FP4E1M2X2),
        ],
    )
    def test_packed_fp4_dependency_format_mismatch_is_rejected(self, declared_dtype, caller_dtype):
        @jit.incore
        def dep(src: pl.Tensor[[16, 32], declared_dtype]):
            return src

        def caller(src: pl.Tensor):
            dep(src)

        with pytest.raises(TypeError, match="cast explicitly to change formats"):
            resolve_dep_call_metadata(
                dep,
                caller,
                {"src": TensorMeta(shape=(16, 32), dtype=caller_dtype)},
                {},
                {},
                discover_deps=_discover_dep_bindings,
            )

    @pytest.mark.parametrize("dtype", [DataType.FP4E2M1X2, DataType.FP4E1M2X2])
    @pytest.mark.parametrize("dependency_kind", ["incore", "inline"])
    def test_fp4_dep_compiles_full_decoded_width(self, monkeypatch, tmp_path, dtype, dependency_kind):
        """Decoding 32 carriers produces 64 values without changing the caller ABI."""
        monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))

        @jit.incore
        def incore_decode(
            src: pl.Tensor[[16, 32], dtype],
            dst: pl.Out[pl.Tensor[[16, 64], pl.BF16]],
        ):
            tile = pl.load(src, [0, 0], [16, 32])
            decoded = pl.cast(tile, pl.BF16)
            pl.store(decoded, [0, 0], dst)
            return dst

        @jit.inline
        def inline_decode(
            src: pl.Tensor[[16, 32], dtype],
            dst: pl.Tensor[[16, 64], pl.BF16],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(src, [0, 0], [16, 32])
                decoded = pl.cast(tile, pl.BF16)
                pl.store(decoded, [0, 0], dst)
            return dst

        decode = inline_decode if dependency_kind == "inline" else incore_decode

        @jit
        def entry(src: pl.Tensor, dst: pl.Out[pl.Tensor]):
            decode(src, dst)
            return dst

        # Seed the same metadata the torch boundary produces. This keeps the
        # compile regression active even when Torch lacks its optional FP4 dtype.
        backend.reset_for_testing()
        compiled = entry._compile(
            {
                "src": TensorMeta(shape=(16, 32), dtype=dtype),
                "dst": TensorMeta(shape=(16, 64), dtype=DataType.BF16),
            },
            {},
            {},
            {},
            pl,
            backend_type=backend.BackendType.Ascend950,
            output_dir=str(tmp_path / "compiled"),
        )
        assert isinstance(compiled, CompiledProgram)
        assert compiled.program is not None
        # CompiledProgram retains the pre-pass input. Inspect an independently
        # lowered program as well, under the default verification instruments.
        parsed = entry._compile_to_program(
            {
                "src": TensorMeta(shape=(16, 32), dtype=dtype),
                "dst": TensorMeta(shape=(16, 64), dtype=DataType.BF16),
            },
            {},
            {},
            {},
            pl,
        )
        lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(parsed)
        entry_func = lowered.get_function("entry")
        assert entry_func is not None
        entry_type = entry_func.params[0].type
        assert isinstance(entry_type, ir.TensorType)

        def static_shape(shaped_type: ir.ShapedType) -> list[int]:
            result = []
            for dim in shaped_type.shape:
                assert isinstance(dim, ir.ConstInt)
                result.append(dim.value)
            return result

        assert entry_type.dtype == dtype
        assert static_shape(entry_type) == [16, 32]
        decoded_shapes: list[list[int]] = []

        class Casts(ir.IRVisitor):
            def visit_call(self, op: ir.Call) -> None:
                result_type = op.type
                if (
                    op.op.name == ir.get_op("tile.cast").name
                    and isinstance(result_type, ir.TileType)
                    and result_type.dtype == DataType.BF16
                ):
                    decoded_shapes.append(static_shape(result_type))
                super().visit_call(op)

        Casts().visit_program(lowered)
        assert decoded_shapes == [[16, 64]]


# ---------------------------------------------------------------------------
# @pl.jit.host decoration
# ---------------------------------------------------------------------------


class TestJitHostDecoration:
    """@pl.jit.host produces a HOST-Orchestrator JITFunction."""

    def test_jit_host_creates_jitfunction(self):
        @jit.host
        def host_orch(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        assert isinstance(host_orch, JITFunction)
        assert host_orch._func_type == "host"

    def test_jit_host_parens_form(self):
        @jit.host()
        def host_orch(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        assert isinstance(host_orch, JITFunction)
        assert host_orch._func_type == "host"

    def test_jit_host_rejects_level_kwarg(self):
        with pytest.raises(TypeError, match="does not accept a level= argument"):

            @jit.host(level=pl.Level.HOST)
            def host_orch(a: pl.Tensor):
                return a

    def test_jit_host_preserves_name(self):
        @jit.host
        def my_host(a: pl.Tensor):
            return a

        assert my_host.__name__ == "my_host"

    def test_jit_host_pl_access(self):
        @pl.jit.host
        def host_orch(a: pl.Tensor):
            return a

        assert isinstance(host_orch, JITFunction)
        assert host_orch._func_type == "host"

    def test_jit_host_accepts_auto_scope_false(self):
        @jit.host(auto_scope=False)
        def host_orch(a: pl.Tensor):
            return a

        assert isinstance(host_orch, JITFunction)
        assert host_orch._func_type == "host"
        assert host_orch._auto_scope is False

    def test_jit_incore_rejects_auto_scope_kwarg(self):
        with pytest.raises(TypeError, match="does not accept an auto_scope= argument"):

            @jit.incore(auto_scope=False)
            def sub_fn(a: pl.Tensor):
                return a

    def test_jit_inline_accepts_auto_scope_false(self):
        """Inline bodies are spliced into the caller, so hand-placed scopes land
        there — @pl.jit.inline must accept auto_scope=False (#1733)."""

        @jit.inline(auto_scope=False)
        def sub_fn(a: pl.Tensor):
            return a

        assert isinstance(sub_fn, JITFunction)
        assert sub_fn._func_type == "inline"
        assert sub_fn._auto_scope is False

    def test_jit_opaque_rejects_auto_scope_kwarg(self):
        with pytest.raises(TypeError, match="does not accept an auto_scope= argument"):

            @jit.opaque(auto_scope=False)
            def sub_fn(a: pl.Tensor):
                return a

    def test_jit_incore_rejects_auto_scope_true(self):
        """Sub-decorators reject auto_scope= even when explicitly True — the
        kwarg is not part of their API surface, so passing any value is an
        error, not just a non-True value."""
        with pytest.raises(TypeError, match="does not accept an auto_scope= argument"):

            @jit.incore(auto_scope=True)
            def sub_fn(a: pl.Tensor):
                return a


class TestHostDiscoversOrchestrationDep:
    """Host entries discover chip-level orchestrators as deps; other entries don't."""

    def test_host_discovers_orchestration_dep(self):
        @jit
        def chip_orch(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        @jit.host
        def host_orch(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return chip_orch(a, c)

        deps = host_orch._get_deps()
        assert len(deps) == 1
        assert deps[0]._func_type == "orchestration"
        assert deps[0].__name__ == "chip_orch"

    def test_orchestration_entry_does_not_discover_orchestration_dep(self):
        """A plain @pl.jit entry must still ignore @pl.jit deps — only
        sub-functions (incore/inline/opaque) are discovered. Otherwise two
        top-level kernels would silently fold into one program."""

        @jit
        def other_orch(a: pl.Tensor):
            return a

        @jit
        def entry(a: pl.Tensor):
            return other_orch(a)

        deps = entry._get_deps()
        assert deps == []

    def test_host_still_discovers_subfunction_deps(self):
        """Sub-function discovery is unchanged for host entries."""

        @jit.incore
        def sub(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        @jit.host
        def host_orch(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return sub(a, c)

        deps = host_orch._get_deps()
        assert len(deps) == 1
        assert deps[0]._func_type == "incore"

    def test_host_forwards_dep_auto_scope(self):
        """An @pl.jit(auto_scope=False) chip orchestrator discovered as a dep of
        an @pl.jit.host entry keeps its auto_scope=False when its
        SpecializeContext is built — the flag must be forwarded to the dep
        context, not defaulted to True."""
        torch = pytest.importorskip("torch")

        @jit(auto_scope=False)
        def chip_orch(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        @jit.host
        def host_orch(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return chip_orch(a, c)

        a = torch.empty(128, 128)
        c = torch.empty(128, 128)
        _, _, tensor_meta, scalar_dtypes, cx, per_func_dyn = host_orch._bind_args((a, c), {})
        contexts = host_orch._build_contexts(tensor_meta, scalar_dtypes, cx, per_func_dyn)
        dep_ctx = next(ctx for ctx in contexts if ctx.func_name == "chip_orch")
        assert dep_ctx.auto_scope is False

    def test_entry_forwards_inline_dep_auto_scope(self):
        """An @pl.jit.inline(auto_scope=False) dep of a plain @pl.jit entry keeps
        auto_scope=False in its SpecializeContext, while the entry's own flag
        stays at its default True (#1733)."""
        torch = pytest.importorskip("torch")

        @jit.inline(auto_scope=False)
        def inline_fn(a: pl.Tensor, c: pl.Tensor):
            return c

        @jit
        def entry(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return inline_fn(a, c)

        a = torch.empty(128, 128)
        c = torch.empty(128, 128)
        _, _, tensor_meta, scalar_dtypes, cx, per_func_dyn = entry._bind_args((a, c), {})
        contexts = entry._build_contexts(tensor_meta, scalar_dtypes, cx, per_func_dyn)
        dep_ctx = next(ctx for ctx in contexts if ctx.func_name == "inline_fn")
        assert dep_ctx.auto_scope is False
        entry_ctx = next(ctx for ctx in contexts if ctx.func_name == "entry")
        assert entry_ctx.auto_scope is True


# ---------------------------------------------------------------------------
# Tensor.bind_dynamic no-op
# ---------------------------------------------------------------------------


class TestScanDepIo:
    """scan_dep_io records both Out and InOut params as output-like (so meta
    propagates to captured dep results), in declaration order."""

    def test_includes_inout_params_in_declaration_order(self):
        @jit.inline
        def sub(a: pl.Tensor, cache: pl.InOut[pl.Tensor], out: pl.Out[pl.Tensor]):
            return out

        def caller(a, cache, out):
            out = sub(a, cache, out)
            return out

        io = scan_dep_io(caller, discover_deps=_discover_dep_bindings)
        assert "sub" in io
        param_names, output_params = io["sub"]
        assert param_names == ["a", "cache", "out"]
        # Both InOut (cache) and Out (out) are output-like; declaration order,
        # so the positional target<->param mapping stays aligned.
        assert output_params == ["cache", "out"]

    def test_out_only_dep_unchanged(self):
        @jit.inline
        def sub(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            return out

        def caller(a, out):
            out = sub(a, out)
            return out

        _, output_params = scan_dep_io(caller, discover_deps=_discover_dep_bindings)["sub"]
        assert output_params == ["out"]


class TestBindDynamic:
    def test_bind_dynamic_is_noop(self):
        """Tensor.bind_dynamic() must not raise at runtime."""
        # Annotation-only Tensor
        t = pl.Tensor[[128, 64], pl.FP32]
        M = pl.dynamic("M")
        t.bind_dynamic(0, M)  # should not raise

    def test_bind_dynamic_returns_none(self):
        t = pl.Tensor[[128, 64], pl.FP32]
        M = pl.dynamic("M")
        result = t.bind_dynamic(0, M)
        assert result is None


# ---------------------------------------------------------------------------
# Cache tests (torch-dependent, skipped if torch not available)
# ---------------------------------------------------------------------------


class TestJitCaching:
    """Cache behavior tests.

    These tests verify L1 cache hit/miss logic by inspecting the internal
    ``_cache`` dict.  They do NOT execute on device (no NPU required).
    """

    @pytest.fixture(autouse=True)
    def _disable_ptoas_for_source_only_tests(self, monkeypatch, tmp_path):
        """Keep cache compilation source-only on hosts with an unusable ptoas."""
        monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))

    def test_cache_hit_same_shape(self):
        """Second call with same shape returns cached program without recompilation."""
        torch = pytest.importorskip("torch")

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

        a = torch.randn(128, 128)
        b = torch.randn(128, 128)
        c = torch.empty(128, 128)

        add_kernel.compile(a, b, c)
        assert len(add_kernel._cache) == 1
        add_kernel.compile(a, b, c)
        assert len(add_kernel._cache) == 1  # no new entry — cache hit

    def test_cache_miss_different_shape(self):
        """Different shape causes new compilation."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def _add_incore2(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            M, N = a.shape
            tile_a = pl.load(a, [0, 0], [M, N])
            tile_b = pl.load(b, [0, 0], [M, N])
            tile_c = pl.add(tile_a, tile_b)
            pl.store(tile_c, [0, 0], c)
            return c

        @jit
        def add_kernel2(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = _add_incore2(a, b, c)
            return c

        a128 = torch.randn(128, 128)
        b128 = torch.randn(128, 128)
        c128 = torch.empty(128, 128)

        a64 = torch.randn(64, 64)
        b64 = torch.randn(64, 64)
        c64 = torch.empty(64, 64)

        add_kernel2.compile(a128, b128, c128)
        assert len(add_kernel2._cache) == 1
        add_kernel2.compile(a64, b64, c64)
        assert len(add_kernel2._cache) == 2  # different shape — cache miss

    def test_dynamic_dim_cache_hit_different_concrete_value(self):
        """With bind_dynamic, different M values should hit the same cache entry."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def _copy_incore_dyn(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            M = pl.dynamic("M")
            a.bind_dynamic(0, M)
            c.bind_dynamic(0, M)
            tile_a = pl.load(a, [0, 0], [128, 128])
            pl.store(tile_a, [0, 0], c)
            return c

        @jit
        def dyn_kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = _copy_incore_dyn(a, c)
            return c

        a256 = torch.randn(256, 128)
        c256 = torch.empty(256, 128)
        a512 = torch.randn(512, 128)
        c512 = torch.empty(512, 128)

        dyn_kernel.compile(a256, c256)
        assert len(dyn_kernel._cache) == 1
        dyn_kernel.compile(a512, c512)
        # Both M values → same cache entry (M is dynamic)
        assert len(dyn_kernel._cache) == 1

    def test_dynamic_dim_cache_miss_on_static_dim_change(self):
        """Changing a non-dynamic dim should miss the cache."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def _copy_incore_dyn2(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            M = pl.dynamic("M")
            a.bind_dynamic(0, M)
            c.bind_dynamic(0, M)
            tile_a = pl.load(a, [0, 0], [128, 128])
            pl.store(tile_a, [0, 0], c)
            return c

        @jit
        def dyn_kernel2(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = _copy_incore_dyn2(a, c)
            return c

        a128 = torch.randn(256, 128)
        c128 = torch.empty(256, 128)
        a256 = torch.randn(256, 256)
        c256 = torch.empty(256, 256)

        dyn_kernel2.compile(a128, c128)
        assert len(dyn_kernel2._cache) == 1
        dyn_kernel2.compile(a256, c256)
        # K changed (128 → 256), should be different compilations
        assert len(dyn_kernel2._cache) == 2

    def test_annotation_dynvar_scanned_without_bind_dynamic(self):
        """A pl.dynamic() var in the annotation marks the dim dynamic — no bind_dynamic.

        Mirrors @pl.program semantics so the same kernel works in either style.
        """
        M = pl.dynamic("M")

        @jit
        def ann_kernel(a: pl.Tensor[[M, 128], pl.FP32], c: pl.Out[pl.Tensor[[M, 128], pl.FP32]]):
            c = a
            return c

        dims = scan_dynamic_dims(ann_kernel._func, ann_kernel._param_names())
        assert ("a", 0) in dims
        assert ("c", 0) in dims
        # Static dim (128) must stay static.
        assert ("a", 1) not in dims

    def test_annotation_dynamic_dim_cache_hit_different_concrete_value(self):
        """Annotation-declared dynamic dim: different M values hit the same cache entry."""
        torch = pytest.importorskip("torch")

        M = pl.dynamic("M")

        @jit.incore
        def _copy_incore_ann(a: pl.Tensor[[M, 128], pl.FP32], c: pl.Out[pl.Tensor[[M, 128], pl.FP32]]):
            tile_a = pl.load(a, [0, 0], [128, 128])
            pl.store(tile_a, [0, 0], c)
            return c

        @jit
        def ann_dyn_kernel(a: pl.Tensor[[M, 128], pl.FP32], c: pl.Out[pl.Tensor[[M, 128], pl.FP32]]):
            c = _copy_incore_ann(a, c)
            return c

        a256 = torch.randn(256, 128)
        c256 = torch.empty(256, 128)
        a512 = torch.randn(512, 128)
        c512 = torch.empty(512, 128)

        ann_dyn_kernel.compile(a256, c256)
        assert len(ann_dyn_kernel._cache) == 1
        ann_dyn_kernel.compile(a512, c512)
        # Both M values → same cache entry (M is dynamic via annotation alone).
        assert len(ann_dyn_kernel._cache) == 1

    def test_returns_compiled_program(self):
        """JIT compilation should produce a CompiledProgram in the cache."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def _copy_incore_simple(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            M, N = a.shape
            tile_a = pl.load(a, [0, 0], [M, N])
            pl.store(tile_a, [0, 0], c)
            return c

        @jit
        def simple(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = _copy_incore_simple(a, c)
            return c

        a = torch.randn(64, 64)
        c = torch.empty(64, 64)
        simple.compile(a, c)
        cached_values = list(simple._cache.values())
        assert len(cached_values) == 1
        assert isinstance(cached_values[0], CompiledProgram)


class TestMultiFuncDepDiscovery:
    """Multi-function JIT: @pl.jit.incore deps are auto-discovered from entry function globals."""

    def test_dep_discovered_from_globals(self):
        """JITFunction.get_deps() finds @pl.jit.incore callees via lazy discovery."""

        # Define both at module scope so that inner is in entry's __globals__
        @jit.incore
        def _inner_dep(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        # Patch _inner_dep into a fresh function's globals to simulate module scope
        import types  # noqa: PLC0415

        def _entry_raw(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = _inner_dep(a, c)
            return c

        # Make _inner_dep visible in the function's globals
        new_globals = {**_entry_raw.__globals__, "_inner_dep": _inner_dep}
        entry_raw = types.FunctionType(
            _entry_raw.__code__,
            new_globals,
            _entry_raw.__name__,
            _entry_raw.__defaults__,
            _entry_raw.__closure__,
        )

        entry_fn = JITFunction(entry_raw, func_type="orchestration")
        deps = entry_fn._get_deps()
        dep_names = [d.__name__ for d in deps]
        assert "_inner_dep" in dep_names

    def test_incore_func_type_preserved(self):
        @jit.incore
        def sub(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        assert sub._func_type == "incore"

    def test_incore_level_preserved(self):
        @jit.incore(level=pl.Level.AIC)
        def aic_sub(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        assert aic_sub._level == pl.Level.AIC

    def test_non_jit_callees_not_in_deps(self):
        """Regular Python functions called from entry are not added as deps."""

        def plain_func(x):
            return x

        @jit
        def entry(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = plain_func(c)
            return c

        deps = _discover_deps(entry._func)
        assert len(deps) == 0


class TestAliasedDepCallName:
    """A dep reached under a name other than its own.

    ``from mod import kernel as kern`` (and the equivalent ``kern = kernel``
    rebinding) leaves the caller's AST calling ``kern``, while the callee's
    ``__name__`` stays ``kernel``. Every call-site lookup must key on the
    former; only the generated ``@pl.function`` keeps the latter.
    """

    @staticmethod
    def _aliased_entry():
        """Return ``(entry, dep)`` where ``entry`` calls ``dep`` via an alias.

        The dep's parameters are deliberately named differently from the
        caller's variables, so nothing resolves by coincidence of names.
        """

        @jit.incore
        def copy_incore(src: pl.Tensor, dst: pl.Out[pl.Tensor]) -> pl.Tensor:
            tile = pl.load(src, [0, 0], [64, 64])
            pl.store(tile, [0, 0], dst)
            return dst

        kern = copy_incore  # the alias an ``import ... as`` would bind

        @jit
        def entry(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            kern(a, c)
            return c

        return entry, copy_incore

    def test_discovery_reports_call_name_and_func_name(self):
        entry, dep = self._aliased_entry()
        bindings = _discover_dep_bindings(entry._func)
        assert [(b.call_name, b.dep.__name__) for b in bindings] == [("kern", "copy_incore")]
        assert bindings[0].dep is dep

    def test_discover_deps_view_drops_call_names(self):
        entry, dep = self._aliased_entry()
        assert _discover_deps(entry._func) == [dep]

    def test_scan_dep_io_keyed_by_call_name(self):
        """``_dep_out_metas`` looks the entry up by the AST's ``ast.Name``."""
        entry, _ = self._aliased_entry()
        io = scan_dep_io(entry._func, discover_deps=_discover_dep_bindings)
        assert "kern" in io
        assert "copy_incore" not in io
        param_names, output_params = io["kern"]
        assert param_names == ["src", "dst"]
        assert output_params == ["dst"]

    def test_dep_metadata_resolves_through_alias(self):
        """The reported failure: metadata came back empty, blaming the shapes."""
        entry, dep = self._aliased_entry()
        caller_meta = {
            "a": TensorMeta(shape=(64, 64), dtype=DataType.FP32),
            "c": TensorMeta(shape=(64, 64), dtype=DataType.FP32),
        }
        dep_meta, _ = resolve_dep_call_metadata(
            dep, entry._func, caller_meta, {}, {}, dep_call_name="kern", discover_deps=_discover_dep_bindings
        )
        # Positional call-site mapping, not the name-based fallback: the dep's
        # own parameter names appear nowhere in the caller.
        assert dep_meta["src"].shape == (64, 64)
        assert dep_meta["dst"].shape == (64, 64)

    def test_context_carries_call_name_and_generated_name(self):
        torch = pytest.importorskip("torch")
        entry, _ = self._aliased_entry()
        a = torch.empty(64, 64)
        c = torch.empty(64, 64)
        _pn, _, tmeta, sd, cx, pfd = entry._bind_args((a, c), {})
        contexts = entry._build_contexts(tmeta, sd, cx, pfd)

        dep_ctx = next(ctx for ctx in contexts if ctx.func_name == "copy_incore")
        assert dep_ctx.tensor_meta["src"].shape == (64, 64)

        entry_ctx = next(ctx for ctx in contexts if ctx.func_name == "entry")
        assert entry_ctx.dep_names == ["kern"]
        assert entry_ctx.dep_func_names == {"kern": "copy_incore"}

    def test_generated_source_calls_the_generated_name(self):
        """The body rewrite must target ``self.copy_incore``, not ``self.kern``."""
        torch = pytest.importorskip("torch")
        entry, _ = self._aliased_entry()
        a = torch.empty(64, 64)
        c = torch.empty(64, 64)
        _pn, _, tmeta, sd, cx, pfd = entry._bind_args((a, c), {})
        contexts = entry._build_contexts(tmeta, sd, cx, pfd)
        source = Specializer("_jit_entry", contexts).specialize()

        assert "self.copy_incore(a, c)" in source
        assert "kern" not in source
        # And it still parses — an unrewritten bare call does not.
        assert isinstance(pl.parse(source), ir.Program)

    def test_two_aliases_for_one_dep_both_rewrite(self):
        """Both call names map onto the single generated function."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def copy_incore(src: pl.Tensor, dst: pl.Out[pl.Tensor]) -> pl.Tensor:
            tile = pl.load(src, [0, 0], [64, 64])
            pl.store(tile, [0, 0], dst)
            return dst

        first = copy_incore
        second = copy_incore

        @jit
        def entry(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            first(a, c)
            second(a, c)
            return c

        a = torch.empty(64, 64)
        c = torch.empty(64, 64)
        _pn, _, tmeta, sd, cx, pfd = entry._bind_args((a, c), {})
        contexts = entry._build_contexts(tmeta, sd, cx, pfd)
        entry_ctx = next(ctx for ctx in contexts if ctx.func_name == "entry")
        assert entry_ctx.dep_func_names == {"first": "copy_incore", "second": "copy_incore"}

        source = Specializer("_jit_entry", contexts).specialize()
        assert source.count("self.copy_incore(a, c)") == 2
        # One generated function, called twice.
        assert source.count("def copy_incore(self") == 1

    def test_module_global_alias_from_import(self):
        """The reported shape: ``from mod import kernel as kern`` at module scope.

        Discovery reads the entry's ``__globals__``, so the alias must resolve
        there too — not only through the closure path the other cases use.
        """
        import importlib.util  # noqa: PLC0415
        import types  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        torch = pytest.importorskip("torch")

        fixture_path = Path(__file__).parent / "_alias_dep_fixture.py"
        spec = importlib.util.spec_from_file_location("_alias_dep_fixture", fixture_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def _entry_raw(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            # Bound below, as ``import ... as kern`` would, via the rebuilt globals.
            kern(a, c)  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
            return c

        # ``from _alias_dep_fixture import copy_incore as kern`` binds the dep
        # under ``kern`` in the importing module's globals.
        new_globals = {**_entry_raw.__globals__, "kern": module.copy_incore}
        entry = JITFunction(
            types.FunctionType(
                _entry_raw.__code__,
                new_globals,
                _entry_raw.__name__,
                _entry_raw.__defaults__,
                _entry_raw.__closure__,
            ),
            func_type="orchestration",
        )

        a = torch.empty(64, 64)
        c = torch.empty(64, 64)
        _pn, _, tmeta, sd, cx, pfd = entry._bind_args((a, c), {})
        contexts = entry._build_contexts(tmeta, sd, cx, pfd)

        dep_ctx = next(ctx for ctx in contexts if ctx.func_name == "copy_incore")
        assert dep_ctx.tensor_meta["src"].shape == (64, 64)
        source = Specializer("_jit_entry_raw", contexts).specialize()
        assert "self.copy_incore(a, c)" in source
        assert isinstance(pl.parse(source), ir.Program)

    def test_aliased_dep_out_metadata_propagates_to_capture(self):
        """``out = kern(a, buf)`` inherits the meta of the arg bound to ``dst``."""

        @jit.incore
        def copy_incore(src: pl.Tensor, dst: pl.Out[pl.Tensor]) -> pl.Tensor:
            tile = pl.load(src, [0, 0], [64, 64])
            pl.store(tile, [0, 0], dst)
            return dst

        kern = copy_incore

        @jit
        def entry(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            buf = pl.create_tensor([32, 16], dtype=pl.FP32)
            out = kern(a, buf)
            return out

        metas = extract_local_tensor_metas(
            entry._func,
            seed_meta={
                "a": TensorMeta(shape=(64, 64), dtype=DataType.FP32),
                "c": TensorMeta(shape=(64, 64), dtype=DataType.FP32),
            },
            discover_deps=_discover_dep_bindings,
        )
        # ``out`` captures the dep's Out param, which the call site bound to
        # ``buf`` — resolvable only if scan_dep_io keyed the dep by "kern".
        assert metas["out"].shape == (32, 16)


class TestDuplicateDepNames:
    """Two distinct deps that share a ``__name__``.

    Both emit into one ``@pl.program`` class, so the generated names must be
    made unique — otherwise the parser refuses the whole program with a bare
    ``Duplicate function name "helper"``, and the entry's two call sites both
    rewrite to the same ``self.helper``.
    """

    @staticmethod
    def _factory_entry():
        """``entry`` calling two same-named kernels built by one factory."""

        def make(rows):
            @jit.incore
            def helper(src: pl.Tensor, dst: pl.Out[pl.Tensor]) -> pl.Tensor:
                tile = pl.load(src, [0, 0], [rows, 64])
                pl.store(tile, [0, 0], dst)
                return dst

            return helper

        first, second = make(32), make(64)

        @jit
        def entry(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            first(a, c)
            second(a, c)
            return c

        return entry

    @staticmethod
    def _contexts_for(entry):
        torch = pytest.importorskip("torch")
        a = torch.empty(64, 64)
        c = torch.empty(64, 64)
        _pn, _, tmeta, sd, cx, pfd = entry._bind_args((a, c), {})
        return entry._build_contexts(tmeta, sd, cx, pfd)

    def test_same_named_deps_get_distinct_generated_names(self):
        contexts = self._contexts_for(self._factory_entry())
        assert [ctx.func_name for ctx in contexts] == ["helper", "helper__2", "entry"]
        # The uniquified context still finds its ``def`` in its own source.
        assert [ctx.source_def_name for ctx in contexts] == ["helper", "helper", "entry"]

    def test_each_call_site_targets_its_own_specialization(self):
        contexts = self._contexts_for(self._factory_entry())
        entry_ctx = next(ctx for ctx in contexts if ctx.func_name == "entry")
        assert entry_ctx.dep_func_names == {"first": "helper", "second": "helper__2"}

    def test_generated_program_parses_and_keeps_both_bodies(self):
        contexts = self._contexts_for(self._factory_entry())
        source = Specializer("_jit_entry", contexts).specialize()

        assert "def helper(self" in source
        assert "def helper__2(self" in source
        assert "self.helper(a, c)" in source
        assert "self.helper__2(a, c)" in source
        # Each specialization folded its own ``rows``; naming them apart is
        # what keeps both bodies in the program.
        assert "pl.load(src, [0, 0], [32, 64])" in source
        assert "pl.load(src, [0, 0], [64, 64])" in source

        program = pl.parse(source)
        assert isinstance(program, ir.Program)
        assert {f.name for f in program.functions} == {"helper", "helper__2", "entry"}

    def test_deps_from_two_modules_sharing_a_name(self):
        """The reported shape: two modules each defining the same kernel name.

        The fixture module is loaded twice under different module names, so the
        two ``copy_incore`` objects are genuinely distinct functions that agree
        on ``__name__`` — exactly what two ``expert_routed`` definitions give.
        """
        import importlib.util  # noqa: PLC0415
        import types  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        fixture_path = Path(__file__).parent / "_alias_dep_fixture.py"

        def _load(mod_name):
            spec = importlib.util.spec_from_file_location(mod_name, fixture_path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.copy_incore

        left, right = _load("_dup_name_left"), _load("_dup_name_right")
        assert left.__name__ == right.__name__ == "copy_incore"
        assert left._func is not right._func

        def _entry_raw(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            left(a, c)  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
            right(a, c)  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
            return c

        new_globals = {**_entry_raw.__globals__, "left": left, "right": right}
        entry = JITFunction(
            types.FunctionType(
                _entry_raw.__code__,
                new_globals,
                _entry_raw.__name__,
                _entry_raw.__defaults__,
                _entry_raw.__closure__,
            ),
            func_type="orchestration",
        )

        contexts = self._contexts_for(entry)
        assert [ctx.func_name for ctx in contexts] == [
            "copy_incore",
            "copy_incore__2",
            "_entry_raw",
        ]
        source = Specializer("_jit_entry_raw", contexts).specialize()
        assert "self.copy_incore(a, c)" in source
        assert "self.copy_incore__2(a, c)" in source
        assert isinstance(pl.parse(source), ir.Program)

    def test_dep_sharing_the_entry_name_yields_to_the_entry(self):
        """The entry keeps the name the user called; the dep is the one moved."""

        def make():
            @jit.incore
            def entry(src: pl.Tensor, dst: pl.Out[pl.Tensor]) -> pl.Tensor:
                tile = pl.load(src, [0, 0], [64, 64])
                pl.store(tile, [0, 0], dst)
                return dst

            return entry

        dep = make()

        @jit
        def entry(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            dep(a, c)
            return c

        contexts = self._contexts_for(entry)
        assert [ctx.func_name for ctx in contexts] == ["entry__2", "entry"]
        entry_ctx = next(ctx for ctx in contexts if ctx.func_name == "entry")
        assert entry_ctx.dep_func_names == {"dep": "entry__2"}
        assert isinstance(pl.parse(Specializer("_jit_entry", contexts).specialize()), ir.Program)

    def test_dep_layouts_cache_key_tracks_which_dep_declared_which_layout(self):
        """Swapping two same-named deps' layouts must change the cache key.

        ``dep_layouts`` is a *sorted* tuple, so it carries no position — keyed
        by ``__name__`` it collapsed, and the second call got the first call's
        artifact even though the generated signatures differ.
        """
        import importlib.util  # noqa: PLC0415
        import types  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        fixture_path = Path(__file__).parent / "_dup_layout_fixture.py"

        def _load(mod_name):
            """Load the fixture, returning ``(helper, its globals dict)``.

            A postponed annotation is resolved against the function's own
            globals, so rebinding ``LAYOUT`` there is what changes the layout
            the dep declares — the module's ``__dict__`` is that same mapping.
            """
            spec = importlib.util.spec_from_file_location(mod_name, fixture_path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.helper, module.__dict__

        (left, left_globals), (right, right_globals) = (
            _load("_dup_layout_left"),
            _load("_dup_layout_right"),
        )

        def _entry_raw(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            left(a, c)  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
            right(a, c)  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
            return c

        new_globals = {**_entry_raw.__globals__, "left": left, "right": right}
        entry = JITFunction(
            types.FunctionType(
                _entry_raw.__code__,
                new_globals,
                _entry_raw.__name__,
                _entry_raw.__defaults__,
                _entry_raw.__closure__,
            ),
            func_type="orchestration",
        )

        left_globals["LAYOUT"], right_globals["LAYOUT"] = ir.TensorLayout.NZ, ir.TensorLayout.ND
        first = entry._dep_declared_layouts()

        left_globals["LAYOUT"], right_globals["LAYOUT"] = ir.TensorLayout.ND, ir.TensorLayout.NZ
        second = entry._dep_declared_layouts()

        # Each triple names the generated function whose signature carries the
        # layout, so the swap is visible.
        assert first == (
            ("helper", "src", str(ir.TensorLayout.NZ)),
            ("helper__2", "src", str(ir.TensorLayout.ND)),
        )
        assert second == (
            ("helper", "src", str(ir.TensorLayout.ND)),
            ("helper__2", "src", str(ir.TensorLayout.NZ)),
        )
        assert first != second


class TestAllocateGeneratedNames:
    """Unit coverage for the generated-name allocator itself."""

    @staticmethod
    def _jit_named(name, *, func_type="incore", external_core_type=None):
        """A JITFunction over a trivial body — the allocator reads only names.

        Each call builds a fresh function object, so same-named holders are
        distinct keys, exactly as two factory-built kernels are.
        """

        def _f():
            pass

        _f.__name__ = name
        return JITFunction(_f, func_type=func_type, external_core_type=external_core_type)

    @staticmethod
    def _per_function(entry, deps):
        """Allocate one name per function — the caller shape used for layouts."""
        return _allocate_generated_names([(id(f._func), f) for f in [entry, *deps]])

    def test_entry_is_named_first(self):
        entry = self._jit_named("k")
        dep = self._jit_named("k")
        names = self._per_function(entry, [dep])
        assert names[id(entry._func)] == "k"
        assert names[id(dep._func)] == "k__2"

    def test_three_way_clash_counts_up(self):
        entry = self._jit_named("e")
        deps = [self._jit_named("k") for _ in range(3)]
        names = self._per_function(entry, deps)
        assert [names[id(d._func)] for d in deps] == ["k", "k__2", "k__3"]

    def test_suffix_shaped_user_name_does_not_collide(self):
        """A user function literally named ``k__2`` still gets its own slot."""
        entry = self._jit_named("e")
        deps = [self._jit_named("k__2"), self._jit_named("k"), self._jit_named("k")]
        names = self._per_function(entry, deps)
        generated = [names[id(d._func)] for d in deps]
        assert generated == ["k__2", "k", "k__3"]
        assert len(set(generated)) == len(generated)

    def test_mixed_extern_reserves_its_member_names(self):
        """A mixed extern occupies ``base``, ``base_aic`` and ``base_aiv``."""
        entry = self._jit_named("e")
        mixed = self._jit_named("k", func_type="extern", external_core_type="mixed")
        plain = self._jit_named("k_aic")
        names = self._per_function(entry, [mixed, plain])
        assert names[id(mixed._func)] == "k"
        assert names[id(plain._func)] == "k_aic__2"

    def test_one_function_reached_twice_keeps_one_name(self):
        """A diamond dep appears once in the map, not twice."""
        entry = self._jit_named("e")
        dep = self._jit_named("k")
        names = self._per_function(entry, [dep, dep])
        assert names[id(dep._func)] == "k"
        assert len(names) == 2

    def test_two_constexpr_bindings_of_one_function_get_two_names(self):
        """Keying by ``(function, binding)`` splits one function into two members.

        This is what lets a dep be called at two different ``pl.constexpr``
        values: each binding is its own generated ``@pl.function``.
        """
        entry = self._jit_named("e")
        dep = self._jit_named("k")
        names = _allocate_generated_names(
            [
                (id(entry._func), entry),
                ((id(dep._func), (("BLOCK", "16"),)), dep),
                ((id(dep._func), (("BLOCK", "32"),)), dep),
            ]
        )
        assert names[(id(dep._func), (("BLOCK", "16"),))] == "k"
        assert names[(id(dep._func), (("BLOCK", "32"),))] == "k__2"


class TestMultiFuncIntegration:
    """End-to-end multi-function @pl.jit compilation with @pl.jit.incore deps."""

    def test_multi_func_parseable(self):
        """@pl.jit with an @pl.jit.incore dep compiles successfully."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def copy_incore(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            M, N = x.shape
            tile_x = pl.load(x, [0, 0], [M, N])
            pl.store(tile_x, [0, 0], out)
            return out

        @jit
        def copy_entry(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = copy_incore(x, out)
            return out

        x = torch.randn(64, 64)
        out = torch.empty(64, 64)
        program = copy_entry.lower(x, out)
        assert isinstance(program, ir.Program)

    def test_multi_func_contains_both_functions(self):
        """Compiled program contains both the @jit.incore dep and the @jit entry functions."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def add_incore(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            M, N = a.shape
            tile_a = pl.load(a, [0, 0], [M, N])
            tile_b = pl.load(b, [0, 0], [M, N])
            tile_c = pl.add(tile_a, tile_b)
            pl.store(tile_c, [0, 0], c)
            return c

        @jit
        def add_entry(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = add_incore(a, b, c)
            return c

        a = torch.randn(32, 32)
        b = torch.randn(32, 32)
        c = torch.empty(32, 32)
        program = add_entry.lower(a, b, c)
        func_names = [f.name for f in program.functions.values()]
        assert "add_incore" in func_names
        assert "add_entry" in func_names

    def test_multi_func_cache_hit(self, monkeypatch, tmp_path):
        """Two multi-function JIT calls with same shapes reuse the cached program."""
        torch = pytest.importorskip("torch")
        monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))

        @jit.incore
        def relu_incore(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            M, N = x.shape
            t = pl.load(x, [0, 0], [M, N])
            r = pl.relu(t)
            pl.store(r, [0, 0], out)
            return out

        @jit
        def relu_entry(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = relu_incore(x, out)
            return out

        x = torch.randn(16, 16)
        out = torch.empty(16, 16)
        relu_entry.compile(x, out)
        assert len(relu_entry._cache) == 1
        relu_entry.compile(x, out)
        assert len(relu_entry._cache) == 1  # cache hit

    def test_multi_func_structural_equal_to_program(self):
        """Multi-function JIT output matches hand-written @pl.program structurally."""
        torch = pytest.importorskip("torch")

        # Hand-written equivalent
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def add_sub(
                self,
                a: pl.Tensor[[32, 32], pl.FP32],
                b: pl.Tensor[[32, 32], pl.FP32],
                c: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [32, 32])
                tile_b = pl.load(b, [0, 0], [32, 32])
                tile_c = pl.add(tile_a, tile_b)
                pl.store(tile_c, [0, 0], c)
                return c

            @pl.function(type=pl.FunctionType.Orchestration)
            def add_entry(
                self,
                a: pl.Tensor[[32, 32], pl.FP32],
                b: pl.Tensor[[32, 32], pl.FP32],
                c: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                c = self.add_sub(a, b, c)
                return c

        @jit.incore
        def add_sub(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            M, N = a.shape
            tile_a = pl.load(a, [0, 0], [M, N])
            tile_b = pl.load(b, [0, 0], [M, N])
            tile_c = pl.add(tile_a, tile_b)
            pl.store(tile_c, [0, 0], c)
            return c

        @jit
        def add_entry(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = add_sub(a, b, c)
            return c

        a = torch.randn(32, 32)
        b = torch.randn(32, 32)
        c = torch.empty(32, 32)
        got = add_entry.lower(a, b, c)
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected_post_pass = pm.run_passes(Expected)
        ir.assert_structural_equal(got, expected_post_pass)


# Module-level @pl.jit.incore kernel reused by the metadata-tracking tests below.
# Defined at module level (not inside a test method) so it can be imported by
# the unit test for ``extract_local_tensor_metas``; the integration tests
# below redefine their own deps inside the method to keep each test isolated.
@jit.incore
def _relu_kernel(x: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    M, N = x.shape
    t = pl.load(x, [0, 0], [M, N])
    r = pl.relu(t)
    pl.store(r, [0, 0], out)
    return out


def _slice_then_dep_body(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Plain (undecorated) function used only by the extract_local_tensor_metas unit test."""
    view = pl.slice(src, [16, 8], [0, 0])
    buf = pl.create_tensor([16, 8], dtype=pl.FP32)
    mid = _relu_kernel(view, buf)
    out = _relu_kernel(mid, out)
    return out


def _runtime_slice_body(src: pl.Tensor, cfg: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Plain (undecorated) function: a pl.slice whose width is a runtime scalar."""
    valid_len = pl.tensor.read(cfg, [0])
    view = pl.slice(src, [16, valid_len], [0, 0])
    out = _relu_kernel(view, out)
    return out


def _annotated_slice_body(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Plain (undecorated) function: pl.slice / pl.create_tensor / dep-call locals
    written with annotated assignments (``v: T = ...``), the common DSL style."""
    view: pl.Tensor = pl.slice(src, [16, 8], [0, 0])
    buf: pl.Tensor = pl.create_tensor([16, 8], dtype=pl.FP32)
    mid: pl.Tensor = _relu_kernel(view, buf)
    out = _relu_kernel(mid, out)
    return out


def _reshape_body(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Plain function for extract_local_tensor_metas unit test: reshape tracking."""
    x_flat = pl.reshape(src, [128, 128])  # noqa: F841 — tracked by JIT AST metadata extraction
    out_flat = pl.reshape(out, [128, 128])
    return out_flat


@jit.incore
def _callsite_metadata_kernel(x: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Dependency used by point-in-time call-site metadata tests."""
    return out


# --- Callee-allocated dep results -------------------------------------------
# A helper that allocates its own outputs and returns them, rather than writing
# into pl.Out params the caller allocated. Its results' shape/dtype live only in
# its body, so the caller's metadata pool has to descend into it.


@jit.inline
def _make_pair_inline(x: pl.Tensor):
    """Inline helper returning two tensors it created itself."""
    a = pl.create_tensor([1, 8], dtype=pl.FP32)
    b = pl.create_tensor([16, 8], dtype=pl.FP16)
    with pl.at(level=pl.Level.CORE_GROUP):
        a[:, :] = x[0:1, :]
        b[:, :] = pl.cast(x[0:16, :], pl.FP16)
    return a, b


def _tuple_dep_return_body(x: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Plain (undecorated) caller: unpacks a tuple of callee-allocated tensors."""
    a, b = _make_pair_inline(x)  # noqa: F841 — the metas under test
    return out


@jit.inline
def _widen_inline(src: pl.Tensor):
    """Inline helper whose own allocation is sized off its parameter."""
    cols = pl.tensor.dim(src, 1)
    wide = pl.create_tensor([2, cols], dtype=pl.FP32)
    return wide


def _param_sized_dep_body(x: pl.Tensor) -> pl.Tensor:
    """Plain (undecorated) caller: the callee sizes its result from the arg."""
    wide = _widen_inline(x)
    return wide


def _arity_mismatch_body(x: pl.Tensor) -> pl.Tensor:
    """Plain (undecorated) caller: one target against a two-element return."""
    only_one = _make_pair_inline(x)
    return only_one


@jit.inline
def _mixed_pair_inline(x: pl.Tensor, cfg: pl.Tensor):
    """Inline helper returning one typeable local and one the extractor declines.

    ``pl.reshape`` with a runtime extent is strict on purpose (a reshape's
    dims are constrained by the source's element count), so ``opaque`` has no
    meta on the callee side either.
    """
    opaque = pl.reshape(x, [2, pl.tensor.read(cfg, [0])])
    known = pl.create_tensor([4, 4], dtype=pl.FP16)
    return opaque, known


def _mixed_pair_body(x: pl.Tensor, cfg: pl.Tensor) -> pl.Tensor:
    """Plain (undecorated) caller: rebinds a known local through a helper that
    returns one typeable result and one it cannot type."""
    a = pl.create_tensor([16, 8], dtype=pl.FP32)
    a, b = _mixed_pair_inline(a, cfg)
    return b


# --- Runtime-sized local extents (synthesized DynDim) ------------------------
# Fixtures for the tests covering a local tensor whose extent is only known at
# runtime: one decorated dep, then plain (undecorated) caller bodies fed straight
# to ``extract_local_tensor_metas`` / ``resolve_dep_call_metadata``.


@jit.incore
def _synth_dim_kernel(t: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Dependency for the synthesized-dim overlay tests."""
    return out


# Module global (not a closure var) for the folding-precedence test.
_CLOSURE_FOLD_ROWS = 64


def _runtime_create_body(cfg: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """create_tensor whose leading extent is read out of a tensor at runtime."""
    n = pl.tensor.read(cfg, [0])
    tmp = pl.create_tensor([n, 8], dtype=pl.FP32)  # noqa: F841 — tracked by metadata extraction
    return out


def _dyn_arith_create_body(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """create_tensor sized by arithmetic over a DynDim — not statically foldable."""
    rows = pl.tensor.dim(src, 0)
    tmp = pl.create_tensor([rows * 2, 8], dtype=pl.FP32)  # noqa: F841 — tracked by extraction
    return out


def _runtime_window_body(out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """pld.window whose leading extent is the runtime world size."""
    buf = pld.alloc_window_buffer([2, 8], dtype=pl.INT32)
    win = pld.window(buf, [pld.world_size(), 8], dtype=pl.INT32)  # noqa: F841 — tracked
    return out


def _runtime_reshape_body(src: pl.Tensor, cfg: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """pl.reshape with a runtime extent — stays strict, no meta."""
    n = pl.tensor.read(cfg, [0])
    flat = pl.reshape(src, [n, 8])  # noqa: F841 — deliberately untracked
    return out


def _runtime_create_then_dep(cfg: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """A runtime-sized local crossing a dep call boundary (issue #2450's shape)."""
    n = pl.tensor.read(cfg, [0])
    tmp = pl.create_tensor([n, 8], dtype=pl.FP32)
    out = _synth_dim_kernel(tmp, out)
    return out


def _dyn_alias_create_then_dep(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """A local carrying a caller-derived (not synthesized) DynDim into a dep."""
    rows = pl.tensor.dim(src, 0)
    tmp = pl.create_tensor([rows, 8], dtype=pl.FP32)
    out = _synth_dim_kernel(tmp, out)
    return out


def _plain_rebind_callsite(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    src = pl.reshape(src, [4, 8])
    out = _callsite_metadata_kernel(src, out)
    src = pl.reshape(src, [2, 16])  # noqa: F841 — must not affect the earlier call
    return out


def _annotated_rebind_callsite(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    view: pl.Tensor = pl.reshape(src, [4, 8])
    out: pl.Tensor = _callsite_metadata_kernel(view, out)
    view = pl.reshape(view, [2, 16])  # noqa: F841 — must not affect the earlier call
    return out


def _chained_rebind_callsite(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    src = view = src[4:]
    out = _callsite_metadata_kernel(view, out)
    return out


def _direct_callsite(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    out = _callsite_metadata_kernel(src, out)
    return out


def _nested_rebind_callsite(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    with pl.scope():
        src = pl.reshape(src, [4, 8])
        out = _callsite_metadata_kernel(src, out)
        src = pl.reshape(src, [2, 16])  # noqa: F841 — after the nested call
    return out


def _subscript_slice_body(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Plain function for extract_local_tensor_metas unit test: subscript-slice
    sugar tracking (issue #1836). All locals are tracked by JIT AST metadata
    extraction, hence the noqa F841 on the unused names."""
    view = src[0:16, 0:8]  # static bounds → (16, 8)  # noqa: F841
    row = src[4, 0:8]  # scalar index drops dim 0 → (8,)  # noqa: F841
    partial = src[0:16]  # trailing implicit ``:`` keeps parent dim 1  # noqa: F841
    open_lo = src[4:]  # open upper bound → parent_dim - start = (28, 32)  # noqa: F841
    out_view = out[0:16, 0:8]  # noqa: F841
    return out


def _subscript_runtime_slice_body(src: pl.Tensor, cfg: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Plain function: a subscript slice whose upper bound is a runtime scalar."""
    valid_len = pl.tensor.read(cfg, [0])
    view = src[0:16, 0:valid_len]  # 2nd dim runtime → parent-dim fallback
    out = _relu_kernel(view, out)
    return out


class TestSliceAndDepReturnMetadata:
    """Regression tests: the JIT specializer must track tensor metadata for
    ``pl.slice`` views and ``@pl.jit.incore`` return values when they flow into
    subsequent kernels (KNOWN_ISSUES: "JIT specializer doesn't track pl.slice
    results or @pl.jit.incore return-value tensor metadata")."""

    def test_extract_local_tensor_metas_slice_and_dep_return(self):
        """``extract_local_tensor_metas`` infers metas for pl.slice views,
        pl.create_tensor locals, and @pl.jit.incore call results."""
        seed = {
            "src": TensorMeta(shape=(32, 32), dtype=DataType.FP32),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        metas = extract_local_tensor_metas(
            _slice_then_dep_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        # pl.slice view: shape from the literal list, dtype inherited from src.
        assert metas["view"] == TensorMeta(shape=(16, 8), dtype=DataType.FP32)
        # pl.create_tensor: unchanged behaviour.
        assert metas["buf"] == TensorMeta(shape=(16, 8), dtype=DataType.FP32)
        # @pl.jit.incore call result: inherits the dep's pl.Out param meta,
        # which here maps to ``buf``.
        assert metas["mid"] == TensorMeta(shape=(16, 8), dtype=DataType.FP32)

    def test_extract_local_tensor_metas_runtime_slice_uses_parent_dim(self):
        """A pl.slice dim that isn't a static int falls back to the parent
        tensor's static dim (the slice is bounded above by its parent)."""
        seed = {
            "src": TensorMeta(shape=(32, 64), dtype=DataType.FP16),
            "cfg": TensorMeta(shape=(1,), dtype=DataType.INT64),
            "out": TensorMeta(shape=(16, 64), dtype=DataType.FP16),
        }
        metas = extract_local_tensor_metas(
            _runtime_slice_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        # Runtime-scalar 2nd dim → falls back to src's dim 1 = 64; dtype from src.
        assert metas["view"] == TensorMeta(shape=(16, 64), dtype=DataType.FP16)

    def test_extract_local_tensor_metas_annotated_assignments(self):
        """Annotated assignments (``v: T = ...``) are tracked just like plain ones."""
        seed = {
            "src": TensorMeta(shape=(32, 32), dtype=DataType.FP32),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        metas = extract_local_tensor_metas(
            _annotated_slice_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        assert metas["view"] == TensorMeta(shape=(16, 8), dtype=DataType.FP32)
        assert metas["buf"] == TensorMeta(shape=(16, 8), dtype=DataType.FP32)
        assert metas["mid"] == TensorMeta(shape=(16, 8), dtype=DataType.FP32)

    def test_slice_view_flows_into_incore_dep(self):
        """A pl.slice view of an entry parameter can be passed into an @pl.jit.incore dep."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def copy_incore(x: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            M, N = x.shape
            t = pl.load(x, [0, 0], [M, N])
            pl.store(t, [0, 0], out)
            return out

        @jit
        def slice_entry(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            view = pl.slice(src, [16, 32], [0, 0])
            out = copy_incore(view, out)
            return out

        src = torch.randn(32, 32)
        out = torch.empty(16, 32)
        program = slice_entry.lower(src, out)
        func_names = [f.name for f in program.functions.values()]
        assert "copy_incore" in func_names
        assert "slice_entry" in func_names

    def test_dep_return_value_flows_into_next_dep(self):
        """The return value of one @pl.jit.incore dep can feed the next dep."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def relu_incore(x: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            M, N = x.shape
            t = pl.load(x, [0, 0], [M, N])
            r = pl.relu(t)
            pl.store(r, [0, 0], out)
            return out

        @jit
        def chain_entry(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            view = pl.slice(src, [16, 32], [0, 0])
            buf = pl.create_tensor([16, 32], dtype=pl.FP32)
            mid = relu_incore(view, buf)
            out = relu_incore(mid, out)
            return out

        src = torch.randn(32, 32)
        out = torch.empty(16, 32)
        program = chain_entry.lower(src, out)
        assert isinstance(program, ir.Program)

    def test_multi_value_dep_return_flows_into_next_dep(self):
        """A tuple-returning @pl.jit.incore dep's results inherit their Out params' metas."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def split_incore(
            x: pl.Tensor, lo: pl.Out[pl.Tensor], hi: pl.Out[pl.Tensor]
        ) -> tuple[pl.Tensor, pl.Tensor]:
            M, N = x.shape
            t = pl.load(x, [0, 0], [M, N])
            a = pl.relu(t)
            b = pl.abs(t)
            pl.store(a, [0, 0], lo)
            pl.store(b, [0, 0], hi)
            return lo, hi

        @jit.incore
        def relu_incore(x: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            M, N = x.shape
            t = pl.load(x, [0, 0], [M, N])
            r = pl.relu(t)
            pl.store(r, [0, 0], out)
            return out

        @jit
        def split_entry(src: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            view = pl.slice(src, [16, 32], [0, 0])
            lo_buf = pl.create_tensor([16, 32], dtype=pl.FP32)
            hi_buf = pl.create_tensor([16, 32], dtype=pl.FP32)
            a, b = split_incore(view, lo_buf, hi_buf)
            out = relu_incore(a, out)
            return out

        src = torch.randn(32, 32)
        out = torch.empty(16, 32)
        program = split_entry.lower(src, out)
        assert isinstance(program, ir.Program)

    def test_extract_local_tensor_metas_callee_allocated_tuple_return(self):
        """A dep with no ``Out`` params still resolves: its returned locals'
        metas are read out of the callee's own body."""
        seed = {
            "x": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        metas = extract_local_tensor_metas(
            _tuple_dep_return_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        # Each target takes the pl.create_tensor meta of the matching returned
        # name — including the dtype, which differs between the two.
        assert metas["a"] == TensorMeta(shape=(1, 8), dtype=DataType.FP32)
        assert metas["b"] == TensorMeta(shape=(16, 8), dtype=DataType.FP16)

    def test_callee_allocated_tuple_return_flows_into_next_dep(self, monkeypatch, tmp_path):
        """The reported failure: two tensors an inline helper created itself are
        unpacked and passed to a second inline helper, whose params then have no
        inferred metadata.

        Driven through ``compile()``, not ``lower()``: ``lower()`` stops after
        the passes, so a codegen precondition the metadata affects would not
        fire here. ``PTOAS_ROOT`` is pointed at nothing so the run stays
        source-only and needs no assembler (the
        ``test_jit_compile_extraction`` pattern).
        """
        torch = pytest.importorskip("torch")
        monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))

        @jit.inline
        def make_pair(x: pl.Tensor[[1, 8], pl.FP32]):
            a = pl.create_tensor([1, 8], dtype=pl.FP32)
            b = pl.create_tensor([1, 8], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP):
                a[:, :] = x[:, :]
                b[:, :] = pl.mul(x[:, :], 2.0)
            return a, b

        @jit.inline
        def consume_pair(a: pl.Tensor, b: pl.Tensor, out: pl.Tensor):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:, :] = pl.add(a[:, :], b[:, :])

        @jit
        def pair_entry(x: pl.Tensor[[1, 8], pl.FP32], out: pl.Out[pl.Tensor[[1, 8], pl.FP32]]):
            a, b = make_pair(x)
            consume_pair(a, b, out)

        compiled = pair_entry.compile(torch.randn(1, 8), torch.empty(1, 8))
        assert isinstance(compiled, CompiledProgram)

    def test_callee_allocated_single_return_flows_into_next_dep(self, monkeypatch, tmp_path):
        """The single-value shape of the same rule: ``buf = helper(x)`` where
        ``helper`` allocates ``buf`` itself. Compiled, not just lowered, for the
        reason given above."""
        torch = pytest.importorskip("torch")
        monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))

        @jit.inline
        def double_inline(x: pl.Tensor[[1, 8], pl.FP32]):
            scaled = pl.create_tensor([1, 8], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP):
                scaled[:, :] = pl.mul(x[:, :], 2.0)
            return scaled

        @jit.inline
        def copy_inline(src: pl.Tensor, out: pl.Tensor):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:, :] = src[:, :]

        @jit
        def single_entry(x: pl.Tensor[[1, 8], pl.FP32], out: pl.Out[pl.Tensor[[1, 8], pl.FP32]]):
            scaled = double_inline(x)
            copy_inline(scaled, out)

        compiled = single_entry.compile(torch.randn(1, 8), torch.empty(1, 8))
        assert isinstance(compiled, CompiledProgram)

    def test_callee_allocated_return_sized_from_its_param(self):
        """A callee whose allocation is sized off a parameter resolves through
        the call site: the seed metas descend with the recursion."""
        seed = {"x": TensorMeta(shape=(4, 64), dtype=DataType.FP16)}
        metas = extract_local_tensor_metas(
            _param_sized_dep_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        # ``pl.create_tensor(..., dtype=pl.FP32)`` sized by ``pl.tensor.dim(src, 1)``
        # of the caller's ``x``.
        assert metas["wide"] == TensorMeta(shape=(2, 64), dtype=DataType.FP32)

    def test_partially_resolved_tuple_clears_the_unresolved_target(self):
        """A target the callee rebinds to something untypeable must not keep the
        metadata it carried *before* the call — that shape describes a tensor
        the helper already replaced, and the next dep would silently receive it.
        Clearing it restores the clear ``_build_params`` error."""
        seed = {
            "x": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
            "cfg": TensorMeta(shape=(1,), dtype=DataType.INT64),
        }
        metas = extract_local_tensor_metas(
            _mixed_pair_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        # ``b`` still resolves from the callee's own pl.create_tensor ...
        assert metas["b"] == TensorMeta(shape=(4, 4), dtype=DataType.FP16)
        # ... while ``a`` drops its stale pre-call [16, 8] FP32 rather than
        # advertising it to whatever consumes ``a`` next.
        assert "a" not in metas

    def test_callee_return_arity_mismatch_resolves_nothing(self):
        """A target that does not line up with the callee's return list is
        declined rather than mis-paired — the clear ``_build_params`` error
        beats a wrong shape."""
        seed = {"x": TensorMeta(shape=(16, 8), dtype=DataType.FP32)}
        metas = extract_local_tensor_metas(
            _arity_mismatch_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        assert "only_one" not in metas

    def test_runtime_sized_slice_uses_static_parent_dim(self):
        """A pl.slice with a runtime-scalar width is advertised to the consuming
        kernel using the parent tensor's static dim, matching how hand-written
        @pl.program code annotates kernels that consume narrowed views (see
        examples/models/04_paged_attention.py)."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def softmax_incore(sij: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            M, N = sij.shape
            t = pl.load(sij, [0, 0], [M, N], target_memory=pl.MemorySpace.Vec)
            r = pl.relu(t)
            pl.store(r, [0, 0], out)
            return out

        @jit
        def attn_entry(big: pl.Tensor, cfg: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            valid_len = pl.tensor.read(cfg, [0])
            sij_valid = pl.slice(big, [16, valid_len], [0, 0])
            out = softmax_incore(sij_valid, out)
            return out

        big = torch.randn(16, 128)
        cfg = torch.zeros(1, dtype=torch.int64)
        out = torch.empty(16, 128)
        program = attn_entry.lower(big, cfg, out)
        assert isinstance(program, ir.Program)

    def test_dep_return_then_runtime_slice_then_dep(self):
        """The paged-attention shape: a dep return value is sliced to a runtime
        width, and that view feeds the next dep — both inferences must hold."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def fill_incore(x: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            M, N = x.shape
            t = pl.load(x, [0, 0], [M, N], target_memory=pl.MemorySpace.Vec)
            pl.store(t, [0, 0], out)
            return out

        @jit.incore
        def softmax_incore(sij: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            M, N = sij.shape
            t = pl.load(sij, [0, 0], [M, N], target_memory=pl.MemorySpace.Vec)
            r = pl.relu(t)
            pl.store(r, [0, 0], out)
            return out

        @jit
        def attn_entry(big: pl.Tensor, cfg: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            sij_buf = pl.create_tensor([16, 128], dtype=pl.FP32)
            sij = fill_incore(big, sij_buf)  # dep return → sij_buf's meta
            valid_len = pl.tensor.read(cfg, [0])
            sij_valid = pl.slice(sij, [16, valid_len], [0, 0])  # runtime slice of a dep return
            out = softmax_incore(sij_valid, out)
            return out

        big = torch.randn(16, 128)
        cfg = torch.zeros(1, dtype=torch.int64)
        out = torch.empty(16, 128)
        program = attn_entry.lower(big, cfg, out)
        assert isinstance(program, ir.Program)

    def test_runtime_create_tensor_dim_defers_to_the_shared_pipeline(self):
        """A runtime-sized ``pl.create_tensor`` dim no longer fails in the JIT
        frontend — it becomes a synthesized DynDim and the shared pass pipeline
        diagnoses whatever the program actually does with it.

        Here the kernel loads the WHOLE tensor as a tile, so the dynamic dim
        reaches the tile shape, which InitMemRef rejects. That is the same
        error a hand-written ``@pl.program`` gets for the same code — the point
        of this test is that ``@pl.jit`` no longer front-runs it with a
        frontend-only "missing inferred tensor metadata".
        """
        torch = pytest.importorskip("torch")

        @jit.incore
        def copy_incore(x: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            M, N = x.shape
            t = pl.load(x, [0, 0], [M, N])
            pl.store(t, [0, 0], out)
            return out

        @jit
        def bad_entry(cfg: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            n = pl.tensor.read(cfg, [0])
            buf = pl.create_tensor([16, n], dtype=pl.FP32)
            out = copy_incore(buf, out)
            return out

        cfg = torch.zeros(1, dtype=torch.int64)
        out = torch.empty(16, 32)
        with pytest.raises(InternalError, match="InitMemRef requires static shape") as excinfo:
            bad_entry.lower(cfg, out)
        assert "missing inferred tensor metadata" not in str(excinfo.value)

    def test_extract_local_tensor_metas_reshape(self):
        """``extract_local_tensor_metas`` infers metas for pl.reshape results."""
        seed = {
            "src": TensorMeta(shape=(2, 64, 128), dtype=DataType.BF16),
            "out": TensorMeta(shape=(2, 64, 128), dtype=DataType.BF16),
        }
        metas = extract_local_tensor_metas(
            _reshape_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        # reshape changes rank but keeps element count; dtype inherited from src.
        expected = TensorMeta(shape=(128, 128), dtype=DataType.BF16, layout=ir.TensorLayout.ND)
        assert metas["x_flat"] == expected
        assert metas["out_flat"] == expected

    @pytest.mark.parametrize("source_shape", [(256, 512), (2, 256, 256)])
    def test_whole_nz_flatten_has_nd_metadata(self, source_shape):
        """A complete physical NZ flatten keeps the storage but has a rank-1 ND type."""

        def body(src):
            flat = pl.reshape(src, [131072])
            qualified = pl.tensor.reshape(tensor=src, shape=[131072])
            return flat, qualified

        seed = {"src": TensorMeta(shape=source_shape, dtype=DataType.INT8, layout=ir.TensorLayout.NZ)}
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        expected = TensorMeta(shape=(131072,), dtype=DataType.INT8, layout=ir.TensorLayout.ND)
        assert metas["flat"] == expected
        assert metas["qualified"] == expected
        assert metas["src"] == seed["src"]

    @pytest.mark.parametrize(
        "source_layout", [None, ir.TensorLayout.ND, ir.TensorLayout.NZ], ids=["unannotated", "nd", "nz"]
    )
    def test_reshape_result_layout_is_not_overwritten_by_dep_annotation(self, source_layout):
        """A computed ND view is known metadata, not an omitted layout for a dep to fill."""

        @jit.incore
        def consume(flat: pl.Tensor[[512], pl.INT8, pl.NZ]):
            return flat

        def caller(src):
            flat = pl.reshape(src, [512])
            alias = flat
            return consume(alias)

        seed = {"src": TensorMeta(shape=(16, 32), dtype=DataType.INT8, layout=source_layout)}
        metas, scalar_dtypes = resolve_dep_call_metadata(
            consume, caller, seed, {}, {}, discover_deps=_discover_dep_bindings
        )
        # The helper's declaration must not turn raw one-dimensional storage
        # back into an NZ matrix when the caller metadata is specialized.
        assert metas["flat"] == TensorMeta(shape=(512,), dtype=DataType.INT8, layout=ir.TensorLayout.ND)
        assert scalar_dtypes == {}

    @pytest.mark.parametrize(
        "source_shape, target_shape, layout",
        [
            ((256, 512), (128, 1024), ir.TensorLayout.NZ),
            ((256, 512), (65536,), ir.TensorLayout.NZ),
            ((DynDim(name="M", literal="M", static_bound=256), 512), (131072,), ir.TensorLayout.NZ),
            ((256, 512), (DynDim(name="N", literal="N", static_bound=131072),), ir.TensorLayout.NZ),
            ((256, 512), (131072,), ir.TensorLayout.DN),
        ],
        ids=["rank-2", "wrong-size", "dynamic-source", "dynamic-target", "other-layout"],
    )
    def test_unproven_non_nd_reshape_has_no_metadata(self, source_shape, target_shape, layout):
        """Only the static whole-NZ case is safe; a dynamic upper bound is not its extent."""

        def body(src):
            flat = pl.reshape(src, target_shape)
            return flat

        seed = {"src": TensorMeta(shape=source_shape, dtype=DataType.INT8, layout=layout)}
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert "flat" not in metas

    def test_nz_flatten_to_dynamic_dim_alias_has_no_metadata(self):
        """A resolved symbolic target is still not proof of the physical storage size."""

        def body(src, shape_ref):
            flat = pl.reshape(src, [shape_ref.shape[0]])
            return flat

        seed = {
            "src": TensorMeta(shape=(256, 512), dtype=DataType.INT8, layout=ir.TensorLayout.NZ),
            "shape_ref": TensorMeta(
                shape=(DynDim(name="N", literal="N", static_bound=131072),), dtype=DataType.INT8
            ),
        }
        assert "flat" not in extract_local_tensor_metas(
            body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )

    @staticmethod
    def _resolve_callsite_metadata(caller):
        seed = {
            "src": TensorMeta(shape=(32,), dtype=DataType.FP32),
            "out": TensorMeta(shape=(4, 8), dtype=DataType.FP32),
        }
        tensor_meta, scalar_dtypes = resolve_dep_call_metadata(
            _callsite_metadata_kernel, caller, seed, {}, {}, discover_deps=_discover_dep_bindings
        )
        assert scalar_dtypes == {}
        return tensor_meta

    def test_dep_metadata_uses_plain_rebinding_at_callsite(self):
        """A plain assignment shadows the parameter only until the selected call."""
        metas = self._resolve_callsite_metadata(_plain_rebind_callsite)
        assert metas["x"] == TensorMeta(shape=(4, 8), dtype=DataType.FP32, layout=ir.TensorLayout.ND)
        assert metas["out"] == TensorMeta(shape=(4, 8), dtype=DataType.FP32)

    def test_dep_metadata_uses_annotated_rebinding_at_callsite(self):
        """Annotated and plain assignments expose the same point-in-time metadata."""
        metas = self._resolve_callsite_metadata(_annotated_rebind_callsite)
        assert metas["x"] == TensorMeta(shape=(4, 8), dtype=DataType.FP32, layout=ir.TensorLayout.ND)

    def test_dep_metadata_tracks_all_chained_assignment_targets(self):
        """Every target gets RHS metadata from the pre-assignment state."""
        metas = self._resolve_callsite_metadata(_chained_rebind_callsite)
        assert metas["x"] == TensorMeta(shape=(28,), dtype=DataType.FP32)

    def test_dep_metadata_direct_call_keeps_seed_metadata(self):
        """Direct dependency calls without local rebindings remain unchanged."""
        metas = self._resolve_callsite_metadata(_direct_callsite)
        assert metas["x"] == TensorMeta(shape=(32,), dtype=DataType.FP32)

    def test_dep_metadata_nested_scope_preserves_source_order(self):
        """Nested DSL scopes stop at their consumer after processing prior producers."""
        metas = self._resolve_callsite_metadata(_nested_rebind_callsite)
        assert metas["x"] == TensorMeta(shape=(4, 8), dtype=DataType.FP32, layout=ir.TensorLayout.ND)

    def test_extract_local_tensor_metas_subscript_slice(self):
        """``extract_local_tensor_metas`` tracks subscript-slice sugar
        ``v = src[a:b, ...]`` the same way it tracks ``pl.slice`` (issue #1836):
        static extents, scalar-index rank reduction, and trailing implicit ``:``."""
        seed = {
            "src": TensorMeta(shape=(32, 32), dtype=DataType.FP32),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        metas = extract_local_tensor_metas(
            _subscript_slice_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        # Two static slices → (stop - start) per dim; dtype inherited from src.
        assert metas["view"] == TensorMeta(shape=(16, 8), dtype=DataType.FP32)
        # Scalar index at dim 0 drops it (numpy-style rank reduction).
        assert metas["row"] == TensorMeta(shape=(8,), dtype=DataType.FP32)
        # Only dim 0 indexed → dim 1 is implicit ``:``, keeps the parent extent.
        assert metas["partial"] == TensorMeta(shape=(16, 32), dtype=DataType.FP32)
        # Open upper bound with a static lower bound → parent_dim - start on the
        # sliced dim (matching the parser), parent extent on the trailing dim.
        assert metas["open_lo"] == TensorMeta(shape=(28, 32), dtype=DataType.FP32)
        # Subscript of an Out parameter resolves just like an In parameter.
        assert metas["out_view"] == TensorMeta(shape=(16, 8), dtype=DataType.FP32)

    def test_extract_local_tensor_metas_subscript_runtime_bound_uses_parent_dim(self):
        """A subscript slice dim with a non-static upper bound falls back to the
        parent tensor's static dim (the slice is bounded above by its parent)."""
        seed = {
            "src": TensorMeta(shape=(32, 64), dtype=DataType.FP16),
            "cfg": TensorMeta(shape=(1,), dtype=DataType.INT64),
            "out": TensorMeta(shape=(16, 64), dtype=DataType.FP16),
        }
        metas = extract_local_tensor_metas(
            _subscript_runtime_slice_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        # Runtime-scalar 2nd bound → falls back to src's dim 1 = 64; dtype from src.
        assert metas["view"] == TensorMeta(shape=(16, 64), dtype=DataType.FP16)

    def test_extract_local_tensor_metas_subscript_propagates_dyndim(self):
        """A full ``:`` subscript over a ``DynDim`` parent dim flows the DynDim
        through transparently (matching ``pl.slice`` behaviour)."""
        from pypto.jit.specializer import DynDim  # noqa: PLC0415

        m_dim = DynDim(name="M", literal="M", static_bound=7)
        seed = {"src": TensorMeta(shape=(m_dim, 128), dtype=DataType.BF16)}

        def body(src):
            view = src[:, 0:64]  # dim 0 full ``:`` keeps DynDim; dim 1 → 64  # noqa: F841
            return view

        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["view"].shape == (m_dim, 64)
        assert metas["view"].dtype == DataType.BF16


# Module-level dynvar + constant for TestDynamicLocalTensorMetadata.
# Module-level so the generated @pl.program source sees them in the
# originating module's globals when it's parsed.
_M_1524 = pl.dynamic("M_1524")
_HIDDEN_1524 = 128


class TestDynamicLocalTensorMetadata:
    """Regression tests for issue #1524: `pl.create_tensor` whose shape is
    derived from `pl.tensor.dim` on a dynamic-bound parameter, or from a
    bind_dynamic'd DynVar, must inherit the matching :class:`DynDim` so the
    local can flow into a sub-function's annotation as `pl.Tensor[[M, ...], ...]`.
    """

    def test_dim_alias_propagates_dyndim_to_local(self):
        """`tokens = pl.tensor.dim(P, 0)` then `pl.create_tensor([tokens, K], ...)`
        stamps the parent's DynDim onto the local's shape."""
        from pypto.jit.specializer import DynDim  # noqa: PLC0415

        m_dim = DynDim(name="M", literal="M", static_bound=7)
        seed = {
            "hidden_states": TensorMeta(shape=(m_dim, 128), dtype=DataType.BF16),
            "out": TensorMeta(shape=(m_dim, 128), dtype=DataType.BF16),
        }

        def body(hidden_states, out):
            tokens = pl.tensor.dim(hidden_states, 0)
            current = pl.create_tensor([tokens, 128], dtype=pl.BF16)
            nxt = pl.create_tensor([tokens, 128], dtype=pl.BF16)
            return current, nxt

        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert "current" in metas
        assert "nxt" in metas
        assert metas["current"].shape == (m_dim, 128)
        assert metas["nxt"].shape == (m_dim, 128)

    def test_dim_alias_static_dim_stays_int(self):
        """Aliasing a static parent dim resolves to the plain int (no DynDim)."""
        from pypto.jit.specializer import DynDim  # noqa: PLC0415

        m_dim = DynDim(name="M", literal="M", static_bound=7)
        seed = {"x": TensorMeta(shape=(m_dim, 128), dtype=DataType.BF16)}

        def body(x):
            hidden = pl.tensor.dim(x, 1)  # static dim 1 = 128
            buf = pl.create_tensor([4, hidden], dtype=pl.FP32)
            return buf

        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["buf"].shape == (4, 128)

    def test_dynvar_in_create_tensor_substituted(self):
        """``pl.create_tensor([M, HIDDEN], ...)`` — M is a DynVar bound to a
        param dim. The body transformer rewrites the runtime ``M`` reference
        to ``pl.tensor.dim(P, k)`` so the generated IR doesn't leak the
        annotation-only DynVar past SSA conversion."""
        torch = pytest.importorskip("torch")

        @jit.inline
        def layer_dv(
            hidden_states: pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16],
            out: pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16],
        ) -> pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16]:
            hidden_states.bind_dynamic(0, _M_1524)
            out.bind_dynamic(0, _M_1524)
            return out

        @jit
        def fwd_dv(
            hidden_states: pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16],
            out: pl.Out[pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16]],
        ) -> pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16]:
            hidden_states.bind_dynamic(0, _M_1524)
            out.bind_dynamic(0, _M_1524)
            # Direct DynVar in the allocation shape — the issue's "alternative
            # workaround" form that previously failed with "Variable 'M' used
            # outside its defining scope".
            current = pl.create_tensor([_M_1524, _HIDDEN_1524], dtype=pl.BF16)
            nxt = pl.create_tensor([_M_1524, _HIDDEN_1524], dtype=pl.BF16)
            current = layer_dv(current, nxt)
            return current

        hidden = torch.empty(7, _HIDDEN_1524, dtype=torch.bfloat16)
        out = torch.empty(7, _HIDDEN_1524, dtype=torch.bfloat16)
        fwd_dv.lower(hidden, out)

    def test_shape_attribute_emits_anchor_not_dynvar(self):
        """``M, N = a.shape`` for a dynamic-bound param emits
        ``pl.tensor.dim(a, 0)`` rather than the bare DynVar — protects against
        the leak reported by Copilot / CodeRabbit (per-helper shape emission
        bypasses ``visit_Name``)."""
        torch = pytest.importorskip("torch")

        @jit
        def shape_unpack(
            a: pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16],
            out: pl.Out[pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16]],
        ) -> pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16]:
            a.bind_dynamic(0, _M_1524)
            out.bind_dynamic(0, _M_1524)
            M, _N = a.shape
            buf = pl.create_tensor([M, _HIDDEN_1524], dtype=pl.BF16)
            return buf

        a = torch.empty(5, _HIDDEN_1524, dtype=torch.bfloat16)
        out = torch.empty(5, _HIDDEN_1524, dtype=torch.bfloat16)
        shape_unpack.lower(a, out)

    def test_dim_alias_rebind_is_safe(self):
        """An alias rebound to a non-``pl.tensor.dim`` value must not stamp
        the parent's DynDim onto downstream shape resolution — regression for
        the flow-insensitive alias bug raised by CodeRabbit."""
        m_dim = DynDim(name="M", literal="M", static_bound=5)
        seed = {"x": TensorMeta(shape=(m_dim, 128), dtype=DataType.FP32)}

        def body(x):
            tokens = pl.tensor.dim(x, 0)
            tokens = tokens - 1  # rebound — alias must be dropped
            buf = pl.create_tensor([tokens, 128], dtype=pl.FP32)
            return buf

        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        # ``tokens`` is no longer a clean dim alias, so the dim cannot be
        # statically resolved: it gets a synthesized placeholder. What must NOT
        # happen is inheriting the parent's ``M`` — that is the flow-insensitive
        # bug this guards.
        dim = metas["buf"].shape[0]
        assert isinstance(dim, DynDim)
        assert dim.synthesized
        assert dim.name != m_dim.name

    def test_issue_1524_repro_compiles(self):
        """The exact failing pattern from issue #1524."""
        torch = pytest.importorskip("torch")

        # Both M_1524 and _HIDDEN_1524 are module-level — the generated
        # @pl.program source picks them up from the originating module's
        # globals when it's parsed.
        @jit.inline
        def layer_1524(
            hidden_states: pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16],
            out: pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16],
        ) -> pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16]:
            hidden_states.bind_dynamic(0, _M_1524)
            out.bind_dynamic(0, _M_1524)
            return out

        @jit
        def fwd_1524(
            hidden_states: pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16],
            out: pl.Out[pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16]],
        ) -> pl.Tensor[[_M_1524, _HIDDEN_1524], pl.BF16]:
            hidden_states.bind_dynamic(0, _M_1524)
            out.bind_dynamic(0, _M_1524)
            tokens = pl.tensor.dim(hidden_states, 0)
            current = pl.create_tensor([tokens, _HIDDEN_1524], dtype=pl.BF16)
            nxt = pl.create_tensor([tokens, _HIDDEN_1524], dtype=pl.BF16)
            current = layer_1524(current, nxt)
            return current

        hidden = torch.empty(7, _HIDDEN_1524, dtype=torch.bfloat16)
        out = torch.empty(7, _HIDDEN_1524, dtype=torch.bfloat16)
        # Should not raise — previously failed with
        # "missing inferred tensor metadata for parameter 'hidden_states'".
        fwd_1524.lower(hidden, out)

    def test_reshape_propagates_dyndim_via_dim_alias(self):
        """pl.reshape with a dim-aliased DynDim in shape propagates the DynDim.

        ``tokens = pl.tensor.dim(x, 2)`` extracts the 3rd dim's DynDim; using
        it in ``pl.reshape(x, [128, tokens])`` propagates the DynDim to the
        result's 2nd dim.
        """
        from pypto.jit.specializer import DynDim  # noqa: PLC0415

        t_dim = DynDim(name="T", literal="T", static_bound=128)
        seed = {"x": TensorMeta(shape=(2, 64, t_dim), dtype=DataType.BF16)}

        def body(x):
            tokens = pl.tensor.dim(x, 2)  # aliases x's dim 2 → DynDim
            flat = pl.reshape(x, [128, tokens])
            return flat

        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["flat"].shape == (128, t_dim)
        assert metas["flat"].dtype == DataType.BF16


# ---------------------------------------------------------------------------
# Per-rank sliced dispatch (chip_orch(x[r], ...)) and pld.window metadata.
# Module-level dynvar + functions so inspect.getsource / inspect.signature
# see real source and annotations (the host-orchestration distributed shape).
# ---------------------------------------------------------------------------
_M_SLICE = pl.dynamic("M_SLICE")


@jit.incore
def _sliced_chip(data: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Chip orchestrator dep reached via a per-rank ``_sliced_chip(x[r], ...)``
    dispatch from a host orchestrator body."""
    M, N = data.shape
    t = pl.load(data, [0, 0], [M, N])
    pl.store(t, [0, 0], out)
    return out


@jit.incore
def _dyn_sliced_chip(data: pl.Tensor[[_M_SLICE, 128], pl.FP32], out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Chip orchestrator dep whose leading dim is dynamic — used to verify the
    DynDim cascade does not flow through a per-rank sliced (dropped) dim."""
    M, N = data.shape
    t = pl.load(data, [0, 0], [M, N])
    pl.store(t, [0, 0], out)
    return out


def _per_rank_dispatch_body(inputs: pl.Tensor, outputs: pl.Out[pl.Tensor]) -> None:
    """Host orchestrator body: dispatch one chip orchestrator per rank by
    subscripting the leading (rank) dim — ``_sliced_chip(inputs[r], outputs[r])``."""
    for r in pl.range(2):
        _sliced_chip(inputs[r], outputs[r])


def _window_local_body(data_buf, signal_buf):
    """Host orchestrator body: per-rank window views over window buffers.
    The body is parsed from source only (never executed) — matching the
    existing ``_slice_then_dep_body`` style."""
    data = pld.window(data_buf, [1, 256], dtype=pl.FP32)
    signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
    return data, signal


# A dtype bound once at module scope, the way one constant selects the element
# type of a whole kernel variant. INT32 rather than FP32 so a result that merely
# fell back to a default cannot pass for a resolved one.
_WINDOW_DTYPE = pl.INT32


def _window_dtype_constant_body(buf):
    """pld.window whose dtype is a module-level constant, not a ``pl.<NAME>``."""
    win = pld.window(buf, [1, 256], dtype=_WINDOW_DTYPE)
    return win


def _window_dtype_shadowed_body(buf, _WINDOW_DTYPE):
    """A parameter shadows the module constant, so the name is a runtime local."""
    win = pld.window(buf, [1, 256], dtype=_WINDOW_DTYPE)
    return win


def _create_tensor_dtype_of_src_body(src):
    """create_tensor that takes its dtype from another tensor (``src``'s meta is seeded)."""
    t = pl.create_tensor([4, 8], dtype=src.dtype)
    return t


# Shape and layout constants for the operand-spelling fixtures below.
_OPERAND_ROWS = 4
_SHAPE_LIST = [4, 8]
_SHAPE_TUPLE = (4, 8)
_FLAT_SHAPE = (32,)
_NZ_LAYOUT = pl.NZ


def _shape_spellings_body(src):
    """Every spelling of one ``[4, 8]`` shape that the specializer folds to a
    literal; ``src``'s meta is seeded as ``(4, 8)``."""
    by_keyword = pl.create_tensor(shape=[4, 8], dtype=pl.FP32)
    by_list_constant = pl.create_tensor(_SHAPE_LIST, dtype=pl.FP32)
    by_tuple_constant = pl.create_tensor(_SHAPE_TUPLE, dtype=pl.FP32)
    by_tuple_literal = pl.create_tensor((4, 8), dtype=pl.FP32)
    by_src_shape = pl.create_tensor(src.shape, dtype=pl.FP32)
    rows, cols = src.shape
    by_unpacked_dims = pl.create_tensor([rows, cols], dtype=pl.FP32)
    width = src.shape[1]
    by_indexed_dim = pl.create_tensor([_OPERAND_ROWS, width], dtype=pl.FP32)
    by_inline_dims = pl.create_tensor([src.shape[0], pl.tensor.dim(src, 1)], dtype=pl.FP32)
    return (
        by_keyword,
        by_list_constant,
        by_tuple_constant,
        by_tuple_literal,
        by_src_shape,
        by_unpacked_dims,
        by_indexed_dim,
        by_inline_dims,
    )


def _shape_constant_views_body(src):
    """pl.slice / pl.reshape / pld.window with a shape constant or a ``shape=`` keyword."""
    view = pl.slice(src, _SHAPE_LIST, [0, 0])
    flat = pl.reshape(input=src, shape=_FLAT_SHAPE)
    win = pld.window(src, shape=_SHAPE_TUPLE, dtype=pl.INT32)
    return view, flat, win


def _shadowed_rows_body(_OPERAND_ROWS):
    """A runtime parameter shadows the module-level extent of the same name."""
    t = pl.create_tensor([_OPERAND_ROWS, 8], dtype=pl.FP32)
    return t


def _rebound_source_body(src):
    """Dims bound from ``src`` keep their extent after ``src`` itself is rebound."""
    rows, cols = src.shape
    width = src.shape[1]
    src = pl.reshape(src, [rows * cols])
    t = pl.create_tensor([rows, width], dtype=pl.FP32)
    return t


def _loop_rebinds_dim_body(src):
    """A loop target reusing a dim's name makes it a runtime value again."""
    rows, cols = src.shape
    for rows in pl.range(2):
        t = pl.create_tensor([rows, cols], dtype=pl.FP32)
    return t


def _constexpr_slice_body(src, shp):
    view = pl.slice(src, shp, [0, 0])
    return view


def _starred_unpack_body(src):
    """A starred unpack shifts positions, so ``cols`` names no single dim."""
    *lead, cols = src.shape
    t = pl.create_tensor([4, cols], dtype=pl.FP32)
    return lead, t


def _layout_body():
    nz = pl.create_tensor([16, 16], dtype=pl.FP16, layout=pl.NZ)
    by_constant = pl.create_tensor([16, 16], dtype=pl.FP16, layout=_NZ_LAYOUT)
    nd = pl.create_tensor([16, 16], dtype=pl.FP16, layout=pl.TensorLayout.ND)
    return nz, by_constant, nd


def _unresolved_layout_body(layout):
    t = pl.create_tensor([16, 16], dtype=pl.FP16, layout=layout)
    return t


def _constexpr_sized_body(rows, acc_dtype):
    """Sized and typed off ``pl.constexpr`` parameters (their binding is passed in)."""
    t = pl.create_tensor([rows, 8], acc_dtype)
    return t


_SCALAR_DTYPE = pl.INDEX


class TestArgRef:
    """Unit tests for ``arg_ref`` — caller-side reference classification."""

    @staticmethod
    def _ref(expr_src: str):
        return arg_ref(ast.parse(expr_src, mode="eval").body)

    def test_plain_name(self):
        assert self._ref("x") == "x"

    def test_single_integer_index_drops_one_dim(self):
        assert self._ref("x[r]") == SlicedArg("x", 1)

    def test_multi_integer_index_drops_each_dim(self):
        assert self._ref("x[r, 0]") == SlicedArg("x", 2)

    def test_slice_index_keeps_dim(self):
        # ``x[r:r+1]`` selects a range — the dim survives, so drop == 0 → None.
        assert self._ref("x[r:r+1]") is None

    def test_mixed_integer_and_slice_counts_only_integers(self):
        # One integer index (dropped) + one slice (kept) → drop == 1.
        assert self._ref("x[r, 0:2]") == SlicedArg("x", 1)

    def test_literal_returns_none(self):
        assert self._ref("3") is None

    def test_attribute_returns_none(self):
        assert self._ref("obj.attr") is None

    def test_subscript_of_non_name_returns_none(self):
        # ``f()[0]`` — base is a call, not a Name.
        assert self._ref("f()[0]") is None


class TestExtractCallArgsSlicedDispatch:
    """``extract_call_args_for_dep`` + ``build_param_mapping`` must carry a
    per-rank subscript (``x[r]``) through as a ``SlicedArg``."""

    def test_sliced_positional_args_extracted(self):
        call_args = extract_call_args_for_dep(_per_rank_dispatch_body, "_sliced_chip")
        assert call_args == [
            (None, SlicedArg("inputs", 1)),
            (None, SlicedArg("outputs", 1)),
        ]

    def test_param_mapping_pairs_sliced_args_by_position(self):
        call_args = extract_call_args_for_dep(_per_rank_dispatch_body, "_sliced_chip")
        assert call_args is not None
        mapping = build_param_mapping(["data", "out"], call_args)
        assert mapping == {
            "data": SlicedArg("inputs", 1),
            "out": SlicedArg("outputs", 1),
        }


class TestWindowLocalMetadata:
    """``extract_local_tensor_metas`` must infer metas for ``pld.window`` views
    so a host orchestrator's per-rank window locals propagate into the chip
    orchestrator's ``pld.DistributedTensor`` parameters."""

    def test_window_view_meta_inferred(self):
        metas = extract_local_tensor_metas(
            _window_local_body, seed_meta={}, discover_deps=_discover_dep_bindings
        )
        # Shape from the 2nd positional arg, dtype from the ``dtype=`` keyword.
        assert metas["data"] == TensorMeta(shape=(1, 256), dtype=DataType.FP32)
        assert metas["signal"] == TensorMeta(shape=(1, 1), dtype=DataType.INT32)

    def test_window_missing_dtype_untracked(self):
        def body(buf):
            data = pld.window(buf, [1, 256])  # no dtype= kw
            return data

        metas = extract_local_tensor_metas(body, seed_meta={}, discover_deps=_discover_dep_bindings)
        assert "data" not in metas

    def test_window_missing_shape_untracked(self):
        def body(buf):
            data = pld.window(buf, dtype=pl.FP32)  # no shape arg
            return data

        metas = extract_local_tensor_metas(body, seed_meta={}, discover_deps=_discover_dep_bindings)
        assert "data" not in metas


class TestPldTensorRebindPreservesMetadata:
    """A qualified (3+-segment) call -- ``pld.tensor.all_to_all_v(...)``,
    ``pl.tensor.slice(...)``, etc. -- is invisible to the one-level
    ``pl.<op>(...)`` branch in ``_update_local_tensor_meta`` (``fn.value`` is
    itself an ``ast.Attribute``, not a ``Name``), so before this fix it fell
    through to ``local.pop()`` instead of either computing real metadata or
    the managed collectives' documented rebind-target metadata (#2638).
    Dispatch is now an explicit allow-list keyed by the full dotted path
    (``_qualified_call_handlers``): an unlisted path still safely drops --
    it never silently fabricates or reuses stale metadata for an op this
    walker hasn't vetted. Each collective's handler reads its own
    documented rebind operand directly (``_same_operand_meta``), not the
    LHS name blindly, so a rebind to a *different* name than the operand
    still gets the operand's real metadata instead of leaking whatever the
    LHS name previously held."""

    def test_all_to_all_v_rebind_keeps_metadata(self):
        # Mirrors the real usage in
        # tests/st/distributed/collectives/test_l2_tensor_all_to_all_v.py:
        # ``data`` is rebound through the same collective it's passed into as
        # the staging window (real signature: all_to_all_v(input, target,
        # signal, send_counts, recv_counts, *, core_num=1) -- tensor_ops.py).
        def body(stage, data, signal, send_counts, recv_counts):
            data = pld.tensor.all_to_all_v(stage, data, signal, send_counts, recv_counts, core_num=1)
            return data

        seed = {
            "stage": TensorMeta(shape=(64, 64), dtype=DataType.FP32),
            "data": TensorMeta(shape=(64, 64), dtype=DataType.FP32),
            "signal": TensorMeta(shape=(4, 1), dtype=DataType.INT32),
            "send_counts": TensorMeta(shape=(4,), dtype=DataType.INT32),
            "recv_counts": TensorMeta(shape=(4, 1), dtype=DataType.INT32),
        }
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["data"] == seed["data"]

    def test_broadcast_rebind_keeps_metadata(self):
        # A second collective, confirming the allow-list is generic and not
        # accidentally specific to all_to_all_v's argument shape. Real DSL
        # signature: broadcast(target, signal, *, root).
        def body(target, signal):
            target = pld.tensor.broadcast(target, signal, root=0)
            return target

        seed = {
            "target": TensorMeta(shape=(64, 64), dtype=DataType.FP32),
            "signal": TensorMeta(shape=(4, 1), dtype=DataType.INT32),
        }
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["target"] == seed["target"]

    def test_fresh_name_rebind_gets_target_operand_metadata(self):
        """Binding a collective's result to a *fresh* name --
        ``result = f(..., data, ...)`` instead of ``data = f(..., data,
        ...)`` -- now gets ``data``'s (the real ``target`` operand's)
        metadata, computed directly rather than assumed absent. This used
        to be an accepted, documented limitation of a bare
        ``preserve_existing``; ``_same_operand_meta`` closes it as a side
        effect of reading the operand instead of the LHS name."""

        def body(stage, data, signal, send_counts, recv_counts):
            result = pld.tensor.all_to_all_v(stage, data, signal, send_counts, recv_counts, core_num=1)
            return result

        seed = {
            "stage": TensorMeta(shape=(64, 64), dtype=DataType.FP32),
            "data": TensorMeta(shape=(64, 64), dtype=DataType.FP32),
            "signal": TensorMeta(shape=(4, 1), dtype=DataType.INT32),
            "send_counts": TensorMeta(shape=(4,), dtype=DataType.INT32),
            "recv_counts": TensorMeta(shape=(4, 1), dtype=DataType.INT32),
        }
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["result"] == seed["data"]

    def test_rebind_to_a_different_stale_name_gets_operand_metadata_not_the_stale_one(self):
        """The bug a review pass caught in the first version of this fix:
        ``old = pld.tensor.broadcast(window, signal, root=0)`` where ``old``
        already holds *different* stale metadata than ``window``. A blind
        ``preserve_existing`` would have kept ``old``'s stale (32, 32) FP16
        entry; the correct answer is ``window``'s (64, 64) FP32 one, since
        that's the operand the collective actually rebinds."""

        def body(window, signal):
            old = pld.tensor.broadcast(window, signal, root=0)
            return old

        seed = {
            "window": TensorMeta(shape=(64, 64), dtype=DataType.FP32),
            "signal": TensorMeta(shape=(4, 1), dtype=DataType.INT32),
            "old": TensorMeta(shape=(32, 32), dtype=DataType.FP16),
        }
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["old"] == seed["window"]

    def test_pl_tensor_dim_is_not_mistaken_for_a_pld_rebind(self):
        """Guards the review trap: ``pl.tensor.dim`` is also a two-level-plus
        attribute call, but it returns a scalar and is already modeled by the
        separate dim-alias mechanism (``_extract_dim_alias`` /
        ``_update_dim_values``) -- it must stay absent from
        ``_qualified_call_handlers``, not be treated as an
        unlisted-but-safe-to-preserve tensor rebind."""

        def body(a):
            d = pl.tensor.dim(a, 0)
            return d

        seed = {"a": TensorMeta(shape=(64, 64), dtype=DataType.FP32)}
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert "d" not in metas

    def test_window_three_segment_form_gets_metadata(self):
        """``pld.tensor.window`` (3-segment) reuses the exact same
        ``_window_meta`` handler ``pld.window`` (2-segment) already has, so
        it computes REAL metadata now -- not just a safe drop. Seeding a
        stale, differently shaped/dtyped entry under the same name and
        confirming it's overwritten (not preserved) locks in that the
        qualified dispatch actually recomputes rather than reuses whatever
        was there before."""

        def body(buf):
            win = pld.tensor.window(buf, [128, 128], dtype=pl.FP32)
            return win

        seed = {
            "buf": TensorMeta(shape=(1,), dtype=DataType.INT64),
            "win": TensorMeta(shape=(64, 64), dtype=DataType.FP16),
        }
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["win"] == TensorMeta(shape=(128, 128), dtype=DataType.FP32)

    def test_pl_tensor_slice_three_segment_form_gets_metadata(self):
        """``pl.tensor.slice(...)`` is real, already-shipping DSL syntax
        (e.g. tests/st/runtime/framework_and_models/test_graph_execution.py),
        and was silently dropping metadata for the identical reason as
        #2638 before this fix generalized the walker past depth-1."""

        def body(src):
            view = pl.tensor.slice(src, [4, 8], [0, 0])
            return view

        seed = {"src": TensorMeta(shape=(4, 8), dtype=DataType.FP32)}
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["view"] == TensorMeta(shape=(4, 8), dtype=DataType.FP32)

    def test_pl_tensor_reshape_three_segment_form_gets_metadata(self):
        """``pl.tensor.reshape(...)`` is real, already-shipping DSL syntax
        (e.g. tests/ut/language/test_unified_ops.py), the same latent gap as
        ``pl.tensor.slice``."""

        def body(src):
            flat = pl.tensor.reshape(src, [128, 128])
            return flat

        seed = {"src": TensorMeta(shape=(2, 64, 128), dtype=DataType.BF16)}
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["flat"] == TensorMeta(shape=(128, 128), dtype=DataType.BF16, layout=ir.TensorLayout.ND)

    def test_alloc_window_buffer_does_not_preserve_stale_metadata(self):
        """``pld.tensor.alloc_window_buffer`` returns a ``Ptr``, not a
        tensor -- it is deliberately absent from ``_qualified_call_handlers``,
        so a rebind through it must still drop whatever stale ``TensorMeta``
        the name held rather than preserving it (the defect a blanket
        namespace-only match would reintroduce)."""

        def body():
            buf = pld.tensor.alloc_window_buffer(256, name="buf")
            return buf

        seed = {"buf": TensorMeta(shape=(64, 64), dtype=DataType.FP16)}
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert "buf" not in metas

    def test_window_three_segment_form_missing_dtype_untracked(self):
        """The identical "handler present but declines" case
        ``test_window_missing_dtype_untracked`` already covers for the
        2-segment spelling, now for the 3-segment one: ``_qualified_call_meta``
        must not conflate "handler ran and returned None" with "preserve
        existing"."""

        def body(buf):
            win = pld.tensor.window(buf, [1, 256])  # no dtype= kw
            return win

        metas = extract_local_tensor_metas(body, seed_meta={}, discover_deps=_discover_dep_bindings)
        assert "win" not in metas

    def test_all_to_all_v_rebind_inside_a_for_loop_keeps_metadata(self):
        """Exercises the recursive self-call in ``_walk_local_tensor_meta_stmts``,
        which threads ``qualified_call_handlers`` into nested bodies
        separately from the direct call -- a dropped or misordered parameter
        there would only break a rebind inside a block like this one, which
        none of the other (top-level-only) tests above would catch."""

        def body(stage, data, signal, send_counts, recv_counts, n):
            for _ in pl.range(n):
                data = pld.tensor.all_to_all_v(stage, data, signal, send_counts, recv_counts, core_num=1)
            return data

        seed = {
            "stage": TensorMeta(shape=(64, 64), dtype=DataType.FP32),
            "data": TensorMeta(shape=(64, 64), dtype=DataType.FP32),
            "signal": TensorMeta(shape=(4, 1), dtype=DataType.INT32),
            "send_counts": TensorMeta(shape=(4,), dtype=DataType.INT32),
            "recv_counts": TensorMeta(shape=(4, 1), dtype=DataType.INT32),
        }
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["data"] == seed["data"]

    def test_call_rooted_attribute_chain_is_not_mistaken_for_a_qualified_call(self):
        """``_flatten_dotted_call`` returns None for a callee that isn't a
        pure attribute/name chain -- e.g. one rooted at a Call rather than a
        Name, named in the helper's own docstring as an unresolvable shape.
        Must still safely drop rather than crash or misidentify the call."""

        def body(other):
            x = other().tensor.window([1, 1], dtype=pl.FP32)
            return x

        metas = extract_local_tensor_metas(body, seed_meta={}, discover_deps=_discover_dep_bindings)
        assert "x" not in metas

    def test_slice_with_drop_dims_declines_rather_than_advertise_the_wrong_rank(self):
        """A second review pass caught this: ``_slice_meta`` never read
        ``drop_dims``, so a rank-reducing slice was inferred at the
        pre-drop rank -- ``pl.tensor.slice(src, [1, 64], [0, 0],
        drop_dims=[0])`` would have been recorded as shape (1, 64) although
        the real result is (64,). Pre-existing gap in the shared handler
        (also reachable through the 2-segment ``pl.slice`` spelling before
        this fix), newly exercised through qualified dispatch. Must decline
        rather than advertise the wrong rank."""

        def body(src):
            view = pl.tensor.slice(src, [1, 64], [0, 0], drop_dims=[0])
            return view

        seed = {"src": TensorMeta(shape=(4, 64), dtype=DataType.FP32)}
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert "view" not in metas

    def test_slice_with_an_explicitly_empty_drop_dims_still_gets_metadata(self):
        """A third review pass caught this: an explicitly empty
        ``drop_dims=[]`` drops nothing, so it's a no-op that the
        drop_dims guard above must not blanket-decline -- only a
        *non-empty* ``drop_dims`` actually rank-reduces the result."""

        def body(src):
            view = pl.tensor.slice(src, [4, 64], [0, 0], drop_dims=[])
            return view

        seed = {"src": TensorMeta(shape=(4, 64), dtype=DataType.FP32)}
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["view"] == seed["src"]

    def test_slice_with_a_closure_constant_empty_drop_dims_still_gets_metadata(self):
        """A fourth review pass caught this: the empty-``drop_dims`` check
        above inspected the raw AST node, so ``drop_dims=EMPTY`` (a name
        bound to a closure or module-level ``[]`` constant) looked like an
        opaque, non-empty operand -- not a no-op -- and declined metadata
        that was previously inferred. Fold the operand before classifying
        it, the same way the ``shape`` operand already does."""

        EMPTY: list[int] = []

        def body(src):
            view = pl.tensor.slice(src, [4, 64], [0, 0], drop_dims=EMPTY)
            return view

        seed = {"src": TensorMeta(shape=(4, 64), dtype=DataType.FP32)}
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["view"] == seed["src"]

    def test_slice_with_a_name_bound_to_none_drop_dims_still_gets_metadata(self):
        """A fifth review pass: a global/closure name bound to ``None``
        (not ``[]``) is also a no-op -- ``drop_dims=NONE_DIMS`` must not be
        treated as non-empty just because it's a ``Name`` rather than a
        literal ``None``."""

        NONE_DIMS: list[int] | None = None

        def body(src):
            view = pl.tensor.slice(src, [4, 64], [0, 0], drop_dims=NONE_DIMS)
            return view

        seed = {"src": TensorMeta(shape=(4, 64), dtype=DataType.FP32)}
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["view"] == seed["src"]

    def test_slice_with_a_constexpr_empty_drop_dims_still_gets_metadata(self):
        """Same review pass: a ``pl.constexpr`` parameter bound to ``[]``
        needs ``scope.fold()`` first (its own, narrower mechanism) before
        the empty-list check can see it -- ``scope.value()`` alone only
        resolves ordinary closure/module names, not constexpr bindings."""

        def body(src, dd):
            view = pl.tensor.slice(src, [4, 64], [0, 0], drop_dims=dd)
            return view

        seed = {"src": TensorMeta(shape=(4, 64), dtype=DataType.FP32)}
        metas = extract_local_tensor_metas(
            body, seed_meta=seed, constexpr_values={"dd": "[]"}, discover_deps=_discover_dep_bindings
        )
        assert metas["view"] == seed["src"]

    def test_slice_with_an_unresolvable_local_drop_dims_declines_rather_than_guess(self):
        """A sixth review pass caught this: resolving a Name's bound value
        returns ``None`` both for a genuine ``drop_dims=None`` binding and
        for a local parameter forwarding ``drop_dims`` (unresolvable
        statically -- its real value only exists at runtime). Conflating
        the two would treat a real, non-empty runtime ``drop_dims`` as a
        no-op and advertise the wrong, pre-drop rank. A local/shadowed name
        must always decline, never be assumed empty."""

        def body(src, drop_dims):
            view = pl.tensor.slice(src, [1, 64], [0, 0], drop_dims=drop_dims)
            return view

        seed = {"src": TensorMeta(shape=(4, 64), dtype=DataType.FP32)}
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert "view" not in metas

    def test_slice_and_reshape_qualified_calls_accept_the_tensor_keyword(self):
        """The same review pass: ``pl.slice``/``pl.reshape``'s real first
        parameter is named ``tensor`` (tensor_ops.py), not ``input`` --
        ``_slice_meta``/``_reshape_meta`` were matching the wrong keyword
        name, invisible for a positional call (``_call_operand`` checks the
        positional slot first) but silently dropping metadata for an
        explicit-keyword one, on both the 2- and 3-segment spellings."""

        def body(src):
            sliced = pl.tensor.slice(tensor=src, shape=[4, 64], offset=[0, 0])
            flat = pl.tensor.reshape(tensor=src, shape=[256])
            return sliced, flat

        seed = {"src": TensorMeta(shape=(4, 64), dtype=DataType.FP32)}
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["sliced"] == seed["src"]
        assert metas["flat"] == TensorMeta(shape=(256,), dtype=DataType.FP32, layout=ir.TensorLayout.ND)


class TestDtypeOperandResolution:
    """A ``dtype=`` operand is resolved by value, not by its ``pl.<NAME>``
    spelling, so the inferred meta matches what the specializer folds into the
    generated source. Before, any other spelling silently dropped the local's
    meta and the first dep receiving it failed with "missing inferred tensor
    metadata" — naming that dep, not the line that built the tensor."""

    def test_window_dtype_module_constant(self):
        metas = extract_local_tensor_metas(
            _window_dtype_constant_body, seed_meta={}, discover_deps=_discover_dep_bindings
        )
        assert metas["win"] == TensorMeta(shape=(1, 256), dtype=DataType.INT32)

    def test_create_tensor_dtype_closure_constant(self):
        acc_dtype = pl.BF16

        def body():
            t = pl.create_tensor([4, 8], dtype=acc_dtype)
            return t

        metas = extract_local_tensor_metas(body, seed_meta={}, discover_deps=_discover_dep_bindings)
        assert metas["t"] == TensorMeta(shape=(4, 8), dtype=DataType.BF16)

    def test_create_tensor_positional_dtype(self):
        def body():
            t = pl.create_tensor([4, 8], pl.BF16)
            return t

        metas = extract_local_tensor_metas(body, seed_meta={}, discover_deps=_discover_dep_bindings)
        assert metas["t"] == TensorMeta(shape=(4, 8), dtype=DataType.BF16)

    def test_create_tensor_dtype_of_tracked_tensor(self):
        seed = {"src": TensorMeta(shape=(16, 16), dtype=DataType.INT8)}
        metas = extract_local_tensor_metas(
            _create_tensor_dtype_of_src_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        assert metas["t"] == TensorMeta(shape=(4, 8), dtype=DataType.INT8)

    def test_local_shadowing_dtype_constant_untracked(self):
        # The specializer does not fold a name the body binds, so neither may
        # the walker: the parameter, not the module constant, is what it names.
        metas = extract_local_tensor_metas(
            _window_dtype_shadowed_body, seed_meta={}, discover_deps=_discover_dep_bindings
        )
        assert "win" not in metas

    def test_host_window_dtype_constant_reaches_nested_deps(self):
        """The reported shape end to end: a host driver's window dtype comes
        from a module constant and must reach the chip orchestrator and the
        inline helper it calls."""

        @jit.inline
        def core(x: pl.Tensor[[8, 128], pl.INT32], win: pld.DistributedTensor[[8, 128], pl.INT32]):
            with pl.at(level=pl.Level.CORE_GROUP):
                win[0:8, 0:128] = pl.add(x[0:8, 0:128], x[0:8, 0:128])

        @jit
        def chip(
            x: pl.Tensor[[8, 128], pl.INT32],
            y: pl.Out[pl.Tensor[[8, 128], pl.INT32]],
            win: pld.DistributedTensor[[8, 128], pl.INT32],
        ):
            core(x, win)
            with pl.at(level=pl.Level.CORE_GROUP):
                y[0:8, 0:128] = win[0:8, 0:128]
            return y

        @jit.host
        def host(x: pl.Tensor[[2, 8, 128], pl.INT32], y: pl.Out[pl.Tensor[[2, 8, 128], pl.INT32]]):
            buf = pld.alloc_window_buffer([8, 128], dtype=_WINDOW_DTYPE)
            for r in pl.range(pld.world_size()):
                win = pld.window(buf, [8, 128], dtype=_WINDOW_DTYPE)
                chip(x[r], y[r], win, device=r)

        program = host.specialize()
        window_type = "win: pld.DistributedTensor[[8, 128], pl.INT32]"
        assert window_type in program.get_function("core").as_python()
        assert window_type in program.get_function("chip").as_python()


class TestShapeOperandResolution:
    """A shape operand resolves to the extent the specializer folds it to,
    whichever spelling names it: by keyword, a list or tuple constant, a tuple
    literal, ``src.shape`` whole, or dims taken from ``src.shape`` / ``pl.tensor.dim``.
    Before, all but the positional list literal dropped the local's meta, and
    a dim taken from ``src.shape`` became a synthesized dynamic dim although
    the extent was static."""

    def test_every_spelling_resolves_to_the_same_static_shape(self):
        seed = {"src": TensorMeta(shape=(4, 8), dtype=DataType.FP32)}
        metas = extract_local_tensor_metas(
            _shape_spellings_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        expected = TensorMeta(shape=(4, 8), dtype=DataType.FP32)
        for name in (
            "by_keyword",
            "by_list_constant",
            "by_tuple_constant",
            "by_tuple_literal",
            "by_src_shape",
            "by_unpacked_dims",
            "by_indexed_dim",
            "by_inline_dims",
        ):
            assert metas.get(name) == expected, name

    def test_unpacked_dynamic_dim_keeps_its_symbol(self):
        # A dim taken from ``src.shape`` is that dim — a declared DynDim flows
        # through by name instead of being replaced by a synthesized symbol.
        rows = DynDim(name="M", literal="M", static_bound=4)
        seed = {"src": TensorMeta(shape=(rows, 8), dtype=DataType.FP32)}
        metas = extract_local_tensor_metas(
            _shape_spellings_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        assert metas["by_unpacked_dims"].shape == (rows, 8)
        assert metas["by_src_shape"].shape == (rows, 8)

    def test_view_ops_accept_shape_constant_and_keywords(self):
        seed = {"src": TensorMeta(shape=(4, 8), dtype=DataType.FP16)}
        metas = extract_local_tensor_metas(
            _shape_constant_views_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        assert metas["view"] == TensorMeta(shape=(4, 8), dtype=DataType.FP16)
        assert metas["flat"] == TensorMeta(shape=(32,), dtype=DataType.FP16, layout=ir.TensorLayout.ND)
        assert metas["win"] == TensorMeta(shape=(4, 8), dtype=DataType.INT32)

    def test_local_shadowing_extent_constant_is_runtime(self):
        metas = extract_local_tensor_metas(
            _shadowed_rows_body, seed_meta={}, discover_deps=_discover_dep_bindings
        )
        leading = metas["t"].shape[0]
        assert isinstance(leading, DynDim) and leading.synthesized

    def test_bound_dim_survives_source_rebinding(self):
        # Resolved when bound, not when used: reshaping ``src`` afterwards must
        # not turn ``rows`` into the reshaped tensor's leading dim.
        seed = {"src": TensorMeta(shape=(4, 8), dtype=DataType.FP32)}
        metas = extract_local_tensor_metas(
            _rebound_source_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        assert metas["t"] == TensorMeta(shape=(4, 8), dtype=DataType.FP32)

    def test_loop_target_drops_bound_dim(self):
        seed = {"src": TensorMeta(shape=(4, 8), dtype=DataType.FP32)}
        metas = extract_local_tensor_metas(
            _loop_rebinds_dim_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        rows, cols = metas["t"].shape
        assert isinstance(rows, DynDim) and rows.synthesized
        assert cols == 8

    def test_spellings_reach_dep_static(self):
        @jit.inline
        def fill(t: pl.Tensor, x: pl.Tensor[[16, 128], pl.FP32]):
            with pl.at(level=pl.Level.CORE_GROUP):
                t[0:16, 0:128] = pl.add(x[0:16, 0:128], x[0:16, 0:128])

        @jit
        def entry(x: pl.Tensor[[16, 128], pl.FP32], y: pl.Out[pl.Tensor[[16, 128], pl.FP32]]):
            rows, cols = x.shape
            t = pl.create_tensor([rows, cols], dtype=pl.FP32)
            fill(t, x)
            with pl.at(level=pl.Level.CORE_GROUP):
                y[0:16, 0:128] = t[0:16, 0:128]
            return y

        fill_source = entry.specialize().get_function("fill").as_python()
        assert "t: pl.Tensor[[16, 128], pl.FP32]" in fill_source

    def test_starred_unpack_binds_no_dim(self):
        seed = {"src": TensorMeta(shape=(2, 4, 8), dtype=DataType.FP32)}
        metas = extract_local_tensor_metas(
            _starred_unpack_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        cols = metas["t"].shape[1]
        assert isinstance(cols, DynDim) and cols.synthesized


class TestLayoutOperandResolution:
    """``pl.create_tensor``'s ``layout`` reaches the local's meta. Before it was
    ignored, so an NZ local reached a dep declaring no layout as ND, silently."""

    def test_layout_resolved_by_value(self):
        metas = extract_local_tensor_metas(_layout_body, seed_meta={}, discover_deps=_discover_dep_bindings)
        nz = TensorMeta(shape=(16, 16), dtype=DataType.FP16, layout=ir.TensorLayout.NZ)
        assert metas["nz"] == nz
        assert metas["by_constant"] == nz
        # ND is what an omitted layout already means, so it is recorded as none.
        assert metas["nd"] == TensorMeta(shape=(16, 16), dtype=DataType.FP16)

    def test_unresolved_layout_untracked(self):
        # Claiming ND for a layout the walker cannot see would mis-declare deps.
        metas = extract_local_tensor_metas(
            _unresolved_layout_body, seed_meta={}, discover_deps=_discover_dep_bindings
        )
        assert "t" not in metas


class TestConstexprOperandResolution:
    """A shape or dtype operand naming a ``pl.constexpr`` parameter resolves to
    the folded value rather than to a synthesized runtime extent."""

    def test_constexpr_binding_folds_into_meta(self):
        metas = extract_local_tensor_metas(
            _constexpr_sized_body,
            seed_meta={},
            constexpr_values={"rows": "16", "acc_dtype": "pl.BF16"},
            discover_deps=_discover_dep_bindings,
        )
        assert metas["t"] == TensorMeta(shape=(16, 8), dtype=DataType.BF16)

    def test_constexpr_whole_shape_folds_into_slice(self):
        seed = {"src": TensorMeta(shape=(8, 8), dtype=DataType.FP16)}
        metas = extract_local_tensor_metas(
            _constexpr_slice_body,
            seed_meta=seed,
            constexpr_values={"shp": "[4, 8]"},
            discover_deps=_discover_dep_bindings,
        )
        assert metas["view"] == TensorMeta(shape=(4, 8), dtype=DataType.FP16)

    def test_callee_constexpr_binding_reaches_its_returned_local(self):
        # ``a`` is allocated inside ``make``, sized off ``make``'s own constexpr;
        # the descent that reads ``a`` back must see the binding this call gives.
        @jit.inline
        def make(x: pl.Tensor[[16, 128], pl.FP32], rows: pl.constexpr):
            a = pl.create_tensor([rows, 128], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP):
                a[0:16, 0:128] = x[0:16, 0:128]
            return a

        @jit.inline
        def fill(t: pl.Tensor, x: pl.Tensor[[16, 128], pl.FP32]):
            with pl.at(level=pl.Level.CORE_GROUP):
                t[0:16, 0:128] = pl.add(x[0:16, 0:128], x[0:16, 0:128])

        @jit
        def entry(x: pl.Tensor[[16, 128], pl.FP32], y: pl.Out[pl.Tensor[[16, 128], pl.FP32]]):
            a = make(x, 16)
            fill(a, x)
            with pl.at(level=pl.Level.CORE_GROUP):
                y[0:16, 0:128] = a[0:16, 0:128]
            return y

        fill_source = entry.specialize().get_function("fill").as_python()
        assert "t: pl.Tensor[[16, 128], pl.FP32]" in fill_source

    def test_constexpr_sized_local_reaches_dep_static(self):
        @jit.inline
        def fill(t: pl.Tensor, x: pl.Tensor[[16, 128], pl.FP32]):
            with pl.at(level=pl.Level.CORE_GROUP):
                t[0:16, 0:128] = pl.add(x[0:16, 0:128], x[0:16, 0:128])

        @jit
        def entry(
            x: pl.Tensor[[16, 128], pl.FP32],
            y: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            rows: pl.constexpr = 16,
        ):
            t = pl.create_tensor([rows, 128], dtype=pl.FP32)
            fill(t, x)
            with pl.at(level=pl.Level.CORE_GROUP):
                y[0:16, 0:128] = t[0:16, 0:128]
            return y

        fill_source = entry.specialize().get_function("fill").as_python()
        assert "t: pl.Tensor[[16, 128], pl.FP32]" in fill_source


def test_scalar_param_dtype_constant_is_rendered_by_value():
    """``pl.Scalar[<constant>]`` compiles under @pl.jit as it does in @pl.program:
    the generated module binds only ``pl`` / ``pld``, so the constant's name
    would be undefined there."""

    @jit
    def entry(
        x: pl.Tensor[[16, 128], pl.FP32],
        y: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        n: pl.Scalar[_SCALAR_DTYPE],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            y[0:16, 0:128] = x[0:16, 0:128]
        return y

    assert "n: pl.Scalar[pl.INDEX]" in entry.specialize().as_python()


class TestSlicedDispatchMetadata:
    """``resolve_dep_call_metadata`` must give a per-rank chip orchestrator dep
    the base tensor's meta with the subscripted leading dims removed, and the
    DynDim cascade must not flow through a dropped dim."""

    def test_sliced_arg_drops_leading_dim(self):
        # Host passes ``inputs[r]`` / ``outputs[r]`` — each drops the rank dim.
        seed = {
            "inputs": TensorMeta(shape=(2, 1, 256), dtype=DataType.FP32),
            "outputs": TensorMeta(shape=(2, 1, 256), dtype=DataType.FP32),
        }
        tensor_meta, _ = resolve_dep_call_metadata(
            _sliced_chip,
            _per_rank_dispatch_body,
            seed,
            {},
            {},
            caller_func_type="host",
            discover_deps=_discover_dep_bindings,
        )
        assert tensor_meta["data"] == TensorMeta(shape=(1, 256), dtype=DataType.FP32)
        assert tensor_meta["out"] == TensorMeta(shape=(1, 256), dtype=DataType.FP32)

    def test_sliced_arg_drop_at_or_past_rank_is_untracked(self):
        # Base is 1-D; dropping a leading dim leaves nothing meaningful, so the
        # guard ``drop < len(shape)`` skips it rather than producing an empty meta.
        seed = {
            "inputs": TensorMeta(shape=(2,), dtype=DataType.FP32),
            "outputs": TensorMeta(shape=(2,), dtype=DataType.FP32),
        }
        tensor_meta, _ = resolve_dep_call_metadata(
            _sliced_chip,
            _per_rank_dispatch_body,
            seed,
            {},
            {},
            caller_func_type="host",
            discover_deps=_discover_dep_bindings,
        )
        assert "data" not in tensor_meta
        assert "out" not in tensor_meta

    def test_dyndim_cascade_skips_sliced_arg(self):
        # The dep declares a dynamic leading dim; the host reaches it via
        # ``_dyn_sliced_chip(inputs[r], ...)``. The DynDim must NOT cascade onto
        # a ``SlicedArg`` key — doing so would corrupt the caller's dim map with
        # a non-string key (and is semantically wrong: the rank dim is dropped).
        call_args_cache: dict[tuple[int, str], list[tuple[str | None, str | SlicedArg | None]] | None] = {
            (id(_per_rank_dispatch_body), "_dyn_sliced_chip"): [
                (None, SlicedArg("inputs", 1)),
                (None, SlicedArg("outputs", 1)),
            ]
        }
        maps = compute_per_func_dyndim_maps(
            entry_func=_per_rank_dispatch_body,
            entry_param_names=["inputs", "outputs"],
            deps=[_dyn_sliced_chip],
            callers_by_dep_id={id(_dyn_sliced_chip._func): [(_per_rank_dispatch_body, "_dyn_sliced_chip")]},
            call_args_cache=call_args_cache,
        )
        host_map = maps[id(_per_rank_dispatch_body)]
        # Sanity: the dep itself carries the dynamic dim.
        assert maps[id(_dyn_sliced_chip._func)]["data"][0].literal == "M_SLICE"
        # The host map must only ever be keyed by parameter name strings.
        assert all(isinstance(key, str) for key in host_map)
        assert SlicedArg("inputs", 1) not in host_map
        assert SlicedArg("outputs", 1) not in host_map


class TestInlineFuncIntegration:
    """End-to-end @pl.jit.inline: dep body is spliced into entry by the IR pass."""

    def test_inline_dep_discovered(self):
        """@pl.jit.inline functions are picked up by dep discovery."""

        @jit.inline
        def helper(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        @jit
        def entry(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = helper(a, c)
            return c

        deps = _discover_deps(entry._func)
        assert len(deps) == 1
        assert deps[0]._func_type == "inline"

    def test_inline_compiled_program_drops_dep_function(self):
        """After compilation, the Inline dep is gone (spliced + removed).

        Inline bodies must include their own ``pl.at`` scope: unlike InCore,
        Inline doesn't provide an implicit scope. The body is spliced into the
        caller's lexical context as-is, then ``OutlineIncoreScopes`` extracts
        the spliced ``pl.at`` block normally.
        """
        torch = pytest.importorskip("torch")

        @jit.inline
        def add_inline(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            M, N = a.shape
            with pl.at(level=pl.Level.CORE_GROUP):
                tile_a = pl.load(a, [0, 0], [M, N])
                tile_b = pl.load(b, [0, 0], [M, N])
                tile_c = pl.add(tile_a, tile_b)
                pl.store(tile_c, [0, 0], c)
            return c

        @jit
        def add_entry(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = add_inline(a, b, c)
            return c

        a = torch.randn(32, 32)
        b = torch.randn(32, 32)
        c = torch.empty(32, 32)
        # lower() returns the post-pass IR (after PassManager.Default).
        # CompiledProgram.program is the *pre-pass* IR, so the cache entry would
        # still contain "add_inline"; we must inspect the post-pass return value.
        post_pass = add_entry.lower(a, b, c)
        func_names = [f.name for f in post_pass.functions.values()]
        assert "add_inline" not in func_names, (
            f"Inline function should have been spliced and removed, got {func_names}"
        )
        assert "add_entry" in func_names

    def test_inline_body_spliced_into_entry(self):
        """The inlined body's tile ops appear inside the (post-outline) entry.

        Detailed structural equality between JIT-with-inline and hand-written
        @pl.program is intentionally not asserted here: when the call site is
        `c = inline(a, b, c)`, the parser SSA-renames the LHS to `c_v1`, and
        the inline pass's substituted return Var does not match `c_v1` at Var
        identity — so a redundant `c_v1 = c` survives. A side-effect call
        (`inline(a, b, c)` — no LHS) avoids the rename but trips the
        InOutUseDiscipline verifier (must read the post-call return). Both
        equivalents are valid IRs, just structurally distinct from a fully
        hand-written equivalent.

        The pass-level tests in tests/ut/ir/transforms/test_inline_functions.py
        cover detailed structural correctness (Phase 0d).
        """
        torch = pytest.importorskip("torch")

        @jit.inline
        def add_inline(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            M, N = a.shape
            with pl.at(level=pl.Level.CORE_GROUP):
                tile_a = pl.load(a, [0, 0], [M, N])
                tile_b = pl.load(b, [0, 0], [M, N])
                tile_c = pl.add(tile_a, tile_b)
                pl.store(tile_c, [0, 0], c)
            return c

        @jit
        def add_entry(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = add_inline(a, b, c)
            return c

        a = torch.randn(32, 32)
        b = torch.randn(32, 32)
        c = torch.empty(32, 32)
        post_pass = add_entry.lower(a, b, c)
        # After OutlineIncoreScopes: entry is Orchestration, inline body became
        # an InCore-class function (AIV/AIC/InCore) named *_incore_*.
        func_names = [f.name for f in post_pass.functions.values()]
        assert "add_inline" not in func_names
        assert "add_entry" in func_names
        assert any("incore" in n for n in func_names), (
            f"Expected an *_incore_* outlined function from the spliced pl.at body, got {func_names}"
        )

    def test_nested_inline_dep_graph(self):
        """A @pl.jit.inline that calls another @pl.jit.inline: both deps must be
        discovered transitively in leaf-first topological order, with the call
        graph recording who calls whom (regression for issue #1302)."""

        @jit.inline
        def leaf(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            return out

        @jit.inline
        def mid(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = leaf(a, out)
            return out

        @jit
        def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = mid(a, out)
            return out

        deps_topo, callers_by_id, callees_by_id, _ = entry._get_dep_graph()
        assert [d.__name__ for d in deps_topo] == ["leaf", "mid"]
        # Callers are recorded with the name their source calls the dep by.
        assert callers_by_id[id(leaf._func)] == [(mid._func, "leaf")]
        assert callers_by_id[id(mid._func)] == [(entry._func, "mid")]
        assert callees_by_id[id(entry._func)] == ["mid"]
        assert callees_by_id[id(mid._func)] == ["leaf"]
        assert callees_by_id[id(leaf._func)] == []

    def test_nested_inline_compiles(self):
        """End-to-end repro from issue #1302: an entry that calls an inline
        that calls another inline must compile without raising and the post-pass
        IR must have spliced both inline bodies."""
        torch = pytest.importorskip("torch")

        @jit.inline
        def leaf(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [32, 32])
                pl.store(tile, [0, 0], out)
            return out

        @jit.inline
        def mid(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = leaf(a, out)
            return out

        @jit
        def entry_nested(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = mid(a, out)
            return out

        a = torch.randn(32, 32)
        out = torch.empty(32, 32)
        post_pass = entry_nested.lower(a, out)
        func_names = [f.name for f in post_pass.functions.values()]
        assert "leaf" not in func_names, f"leaf should be spliced, got {func_names}"
        assert "mid" not in func_names, f"mid should be spliced, got {func_names}"
        assert "entry_nested" in func_names
        assert any("incore" in n for n in func_names), (
            f"Expected an *_incore_* outlined function from the spliced pl.at body, got {func_names}"
        )

    def test_nested_inline_diamond(self):
        """Diamond: entry -> {a_helper, b_helper}, both call the same shared
        leaf. The shared leaf must be deduplicated in the dep graph."""
        torch = pytest.importorskip("torch")

        @jit.inline
        def shared(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [32, 32])
                pl.store(tile, [0, 0], out)
            return out

        @jit.inline
        def a_helper(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = shared(a, out)
            return out

        @jit.inline
        def b_helper(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = shared(a, out)
            return out

        @jit
        def entry_diamond(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = a_helper(a, out)
            out = b_helper(a, out)
            return out

        deps_topo, callers_by_id, _, _ = entry_diamond._get_dep_graph()
        assert len(deps_topo) == 3, f"expected dedup'd shared leaf + a_helper + b_helper, got {deps_topo}"
        assert {d.__name__ for d in deps_topo} == {"shared", "a_helper", "b_helper"}

        # Both diamond branches must record themselves as callers of the
        # shared leaf, so dyn-dim / dynvar propagation visits every branch
        # rather than just the first DFS path.
        shared_callers = callers_by_id[id(shared._func)]
        assert {c.__name__ for c, _ in shared_callers} == {"a_helper", "b_helper"}
        assert {name for _, name in shared_callers} == {"shared"}
        assert callers_by_id[id(a_helper._func)] == [(entry_diamond._func, "a_helper")]
        assert callers_by_id[id(b_helper._func)] == [(entry_diamond._func, "b_helper")]

        a = torch.randn(32, 32)
        out = torch.empty(32, 32)
        # Compilation must succeed without "Unsupported function call" errors,
        # and post-pass IR must have spliced both helpers and the shared leaf.
        post_pass = entry_diamond.lower(a, out)
        func_names = [f.name for f in post_pass.functions.values()]
        for spliced in ("shared", "a_helper", "b_helper"):
            assert spliced not in func_names, f"{spliced} should be spliced, got {func_names}"
        assert "entry_diamond" in func_names

    def test_mixed_positional_and_keyword_call(self):
        """Mixed ``dep(a, out=out)`` calls must preserve both positional and
        keyword bindings so tensor metadata propagates correctly to the dep
        (regression for CodeRabbit review on PR #1314)."""
        torch = pytest.importorskip("torch")

        @jit.inline
        def helper(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [32, 32])
                pl.store(tile, [0, 0], out)
            return out

        @jit
        def entry_mixed(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            # Positional ``a`` plus keyword ``out=out`` — the keyword binding
            # used to be silently dropped by extract_call_args_for_dep.
            out = helper(a, out=out)
            return out

        a = torch.randn(32, 32)
        out = torch.empty(32, 32)
        post_pass = entry_mixed.lower(a, out)
        func_names = [f.name for f in post_pass.functions.values()]
        assert "helper" not in func_names, f"helper should be spliced, got {func_names}"
        assert "entry_mixed" in func_names

    def test_inline_multi_return_tuple_unpack(self):
        """Multi-return @pl.jit.inline + tuple-unpack at call site (issue #1304).

        Pre-fix: ParserSyntaxError ('TupleGetItemExpr requires tuple to have
        TupleType, got TensorType') because the specializer emitted a single-
        tensor return annotation.
        """
        torch = pytest.importorskip("torch")

        @jit.inline
        def two_returns(
            a: pl.Tensor,
            o0: pl.Out[pl.Tensor],
            o1: pl.Out[pl.Tensor],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [32, 32])
                pl.store(tile, [0, 0], o0)
                pl.store(tile, [0, 0], o1)
            return o0, o1

        @jit
        def entry(
            a: pl.Tensor,
            o0: pl.Out[pl.Tensor],
            o1: pl.Out[pl.Tensor],
        ):
            y0, y1 = two_returns(a, o0, o1)
            return y0, y1

        a = torch.randn(32, 32)
        o0 = torch.empty(32, 32)
        o1 = torch.empty(32, 32)
        post_pass = entry.lower(a, o0, o1)
        func_names = [f.name for f in post_pass.functions.values()]
        assert "two_returns" not in func_names
        assert "entry" in func_names

    def test_inline_tensor_and_task_id_tuple_return(self):
        """Inline multi-return preserves an explicitly typed TASK_ID element."""
        torch = pytest.importorskip("torch")

        @jit.inline
        def produce(
            a: pl.Tensor,
            out: pl.Tensor,
        ) -> tuple[pl.Tensor[[32, 32], pl.FP32], pl.Scalar[pl.TASK_ID]]:
            with pl.at(level=pl.Level.CORE_GROUP) as producer_tid:
                tile = pl.load(a, [0, 0], [32, 32])
                out = pl.store(tile, [0, 0], out)
            return out, producer_tid

        @jit
        def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            result = produce(a, out)
            produced = result[0]
            producer_tid = result[1]
            with pl.at(level=pl.Level.CORE_GROUP, deps=[producer_tid]):
                tile = pl.load(produced, [0, 0], [32, 32])
                out = pl.store(tile, [0, 0], out)
            return out

        a = torch.randn(32, 32)
        out = torch.empty(32, 32)
        post_pass = entry.lower(a, out)
        func_names = [f.name for f in post_pass.functions.values()]
        assert "produce" not in func_names
        assert "entry" in func_names

    def test_inline_multi_return_direct_return(self):
        """Multi-return @pl.jit.inline + ``return inline_call(...)`` (issue #1304).

        Pre-fix: RuntimeError reporting the inline function as undefined, because
        InlineCallsMutator only handled AssignStmt/EvalStmt — ReturnStmt-shaped
        call sites slipped past splicing.
        """
        torch = pytest.importorskip("torch")

        @jit.inline
        def two_returns(
            a: pl.Tensor,
            o0: pl.Out[pl.Tensor],
            o1: pl.Out[pl.Tensor],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [32, 32])
                pl.store(tile, [0, 0], o0)
                pl.store(tile, [0, 0], o1)
            return o0, o1

        @jit
        def entry(
            a: pl.Tensor,
            o0: pl.Out[pl.Tensor],
            o1: pl.Out[pl.Tensor],
        ):
            return two_returns(a, o0, o1)

        a = torch.randn(32, 32)
        o0 = torch.empty(32, 32)
        o1 = torch.empty(32, 32)
        post_pass = entry.lower(a, o0, o1)
        func_names = [f.name for f in post_pass.functions.values()]
        assert "two_returns" not in func_names
        assert "entry" in func_names

    def test_inline_with_bare_tensor_no_pl_out(self):
        """@pl.jit.inline helpers with bare ``pl.Tensor`` params (no ``pl.Out``)
        compile cleanly without DeprecationWarning. The body is spliced
        identically — `pl.Out` is redundant on inline helpers because the
        splice happens before SSA.
        """
        torch = pytest.importorskip("torch")

        @jit.inline
        def two_returns_no_out(
            a: pl.Tensor,
            o0: pl.Tensor,  # bare — no pl.Out
            o1: pl.Tensor,  # bare — no pl.Out
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [32, 32])
                pl.store(tile, [0, 0], o0)
                pl.store(tile, [0, 0], o1)
            return o0, o1

        @jit
        def entry(
            a: pl.Tensor,
            o0: pl.Out[pl.Tensor],
            o1: pl.Out[pl.Tensor],
        ):
            y0, y1 = two_returns_no_out(a, o0, o1)
            return y0, y1

        a = torch.randn(32, 32)
        o0 = torch.empty(32, 32)
        o1 = torch.empty(32, 32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            post_pass = entry.lower(a, o0, o1)
        # Bare pl.Tensor inline params must not emit the deprecation warning.
        assert not any(issubclass(w.category, DeprecationWarning) for w in caught), (
            f"Unexpected DeprecationWarning for bare pl.Tensor inline params: "
            f"{[str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]}"
        )
        func_names = [f.name for f in post_pass.functions.values()]
        assert "two_returns_no_out" not in func_names
        assert "entry" in func_names

    def test_inline_with_reshape_compiles(self):
        """Regression: @pl.jit.inline + pl.reshape in caller compiles.

        Previously failed because ``extract_local_tensor_metas`` did not track
        ``pl.reshape``, so the specializer couldn't find metadata for the
        reshaped tensor passed to the inline dep.
        """
        torch = pytest.importorskip("torch")

        @jit.inline
        def copy_inline(
            x: pl.Tensor[[128, 128], pl.BF16],
            y: pl.Out[pl.Tensor[[128, 128], pl.BF16]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(x, [0, 0], [128, 128])
                pl.store(tile, [0, 0], y)
            return y

        @jit
        def reshape_caller(
            x: pl.Tensor[[2, 64, 128], pl.BF16],
            y: pl.Out[pl.Tensor[[2, 64, 128], pl.BF16]],
        ) -> pl.Tensor[[2, 64, 128], pl.BF16]:
            x_flat = pl.reshape(x, [128, 128])
            y = pl.reshape(y, [128, 128])
            y = copy_inline(x_flat, y)
            y = pl.reshape(y, [2, 64, 128])
            return y

        x = torch.randn(2, 64, 128, dtype=torch.bfloat16)
        y = torch.empty(2, 64, 128, dtype=torch.bfloat16)
        post_pass = reshape_caller.lower(x, y)
        func_names = [f.name for f in post_pass.functions.values()]
        assert "copy_inline" not in func_names, f"Inline should be spliced, got {func_names}"
        assert "reshape_caller" in func_names

    def test_inline_with_subscript_slice_compiles(self):
        """Regression for #1836: a subscript-slice view ``src[a:b]`` forwarded into
        an @pl.jit.inline dep compiles.

        Previously failed because ``extract_local_tensor_metas`` did not track the
        subscript-slice sugar (an ``ast.Subscript``, unlike ``pl.slice``'s Call),
        so the inline dep's param got no inferred metadata and ``_build_params``
        raised ``missing inferred tensor metadata``. Mirrors
        ``test_inline_with_reshape_compiles`` (the #1755 sibling) with the local
        view produced by subscript sugar instead of ``pl.reshape``.
        """
        torch = pytest.importorskip("torch")

        @jit.inline
        def copy_inline(
            x: pl.Tensor[[128, 128], pl.BF16],
            y: pl.Out[pl.Tensor[[128, 128], pl.BF16]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(x, [0, 0], [128, 128])
                pl.store(tile, [0, 0], y)
            return y

        @jit
        def subscript_caller(
            src: pl.Tensor[[256, 128], pl.BF16],
            y: pl.Out[pl.Tensor[[128, 128], pl.BF16]],
        ) -> pl.Tensor[[128, 128], pl.BF16]:
            x_view = src[0:128]  # subscript-slice sugar → (128, 128)
            y = copy_inline(x_view, y)
            return y

        src = torch.randn(256, 128, dtype=torch.bfloat16)
        y = torch.empty(128, 128, dtype=torch.bfloat16)
        post_pass = subscript_caller.lower(src, y)
        func_names = [f.name for f in post_pass.functions.values()]
        assert "copy_inline" not in func_names, f"Inline should be spliced, got {func_names}"
        assert "subscript_caller" in func_names

    def test_jit_inline_split_aiv_underscore_loop_var(self):
        """`for _ in pl.split_aiv(...)` is the documented lane-agnostic idiom and
        must compile inside an inline callee.

        The inliner alpha-renames callee locals by appending `_inline<N>`; with a
        loop variable named `_` that used to concatenate into `__inline<N>`, and
        the `__` there is the delimiter reserved by the IR auto-naming utility,
        so ConvertToSSA rejected a name the author never wrote.
        """
        torch = pytest.importorskip("torch")

        @jit.inline(auto_scope=False)
        def region(x: pl.Tensor, y: pl.Tensor):
            M, N = x.shape
            with pl.spmd(1, name_hint="usc") as _tid:
                _blk = pl.tile.get_block_idx()
                for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    y = pl.assemble(y, pl.mul(x[0:M, 0:N], 2.0), [0, 0])
            return y

        @jit
        def entry(x: pl.Tensor, y: pl.Out[pl.Tensor]):
            y = region(x, y)
            return y

        entry.compile(torch.zeros(16, 64), torch.zeros(16, 64))


class TestOpaqueFuncIntegration:
    """End-to-end @pl.jit.opaque: dep is emitted as a separate Opaque IR function."""

    def test_opaque_dep_discovered(self):
        """@pl.jit.opaque functions are picked up by dep discovery."""

        @jit.opaque
        def helper(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            return c

        @jit
        def entry(a: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = helper(a, c)
            return c

        deps = _discover_deps(entry._func)
        assert len(deps) == 1
        assert deps[0]._func_type == "opaque"

    def test_opaque_structural_equal_to_program(self):
        """JIT(@opaque + @jit) ≡ hand-written @pl.program with Opaque sub-function."""
        torch = pytest.importorskip("torch")

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.Opaque)
            def add_op(
                self,
                a: pl.Tensor[[32, 32], pl.FP32],
                b: pl.Tensor[[32, 32], pl.FP32],
                c: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    tile_a = pl.load(a, [0, 0], [32, 32])
                    tile_b = pl.load(b, [0, 0], [32, 32])
                    tile_c = pl.add(tile_a, tile_b)
                    pl.store(tile_c, [0, 0], c)
                return c

            @pl.function(type=pl.FunctionType.Orchestration)
            def add_entry(
                self,
                a: pl.Tensor[[32, 32], pl.FP32],
                b: pl.Tensor[[32, 32], pl.FP32],
                c: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                c = self.add_op(a, b, c)
                return c

        @jit.opaque
        def add_op(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            M, N = a.shape
            with pl.at(level=pl.Level.CORE_GROUP):
                tile_a = pl.load(a, [0, 0], [M, N])
                tile_b = pl.load(b, [0, 0], [M, N])
                tile_c = pl.add(tile_a, tile_b)
                pl.store(tile_c, [0, 0], c)
            return c

        @jit
        def add_entry(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            c = add_op(a, b, c)
            return c

        a = torch.randn(32, 32)
        b = torch.randn(32, 32)
        c = torch.empty(32, 32)
        got = add_entry.lower(a, b, c)
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected_post_pass = pm.run_passes(Expected)
        ir.assert_structural_equal(got, expected_post_pass)


class TestRoundTrip:
    """Round-trip: @pl.jit output must be structurally equal to a hand-written @pl.program."""

    def test_elementwise_add_128x128(self):
        torch = pytest.importorskip("torch")

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.Orchestration)
            def tile_add(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    tile_a = pl.load(a, [0, 0], [128, 128])
                    tile_b = pl.load(b, [0, 0], [128, 128])
                    tile_c = pl.add(tile_a, tile_b)
                    pl.store(tile_c, [0, 0], c)
                return c

        @jit
        def tile_add(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                M, N = a.shape
                tile_a = pl.load(a, [0, 0], [M, N])
                tile_b = pl.load(b, [0, 0], [M, N])
                tile_c = pl.add(tile_a, tile_b)
                pl.store(tile_c, [0, 0], c)
            return c

        a = torch.randn(128, 128)
        b = torch.randn(128, 128)
        c = torch.empty(128, 128)
        got = tile_add.lower(a, b, c)
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected_post_pass = pm.run_passes(Expected)
        ir.assert_structural_equal(got, expected_post_pass)


# ---------------------------------------------------------------------------
# Variable rebinding (Issue #1121)
# ---------------------------------------------------------------------------


class TestVariableRebinding:
    """Tests for Python-style variable rebinding in @pl.jit (Issue #1121)."""

    def test_rebind_same_type_compiles(self):
        """Rebinding a Tile variable to a new Tile value must compile without error."""

        @jit
        def kernel(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                t = pl.load(x, [0, 0], [128, 128])
                t = pl.mul(t, t)  # rebind: Tile → Tile (same type)
                pl.store(t, [0, 0], out)
            return out

        result = kernel._compile_to_program(
            tensor_meta={
                "x": TensorMeta((128, 128), DataType.FP32),
                "out": TensorMeta((128, 128), DataType.FP32),
            },
            scalar_dtypes={},
            constexpr_values={},
            per_func_dyn={id(kernel._func): {}},
            pl=pl,
        )
        assert result is not None

    def test_rebind_error_shows_original_name(self):
        """When a JIT compilation error occurs, error messages must show the
        user's original variable name, not the internal renamed alias."""
        rename_map = {"t_v1": "t", "x_v2": "x"}
        exc = ValueError("Variable 't_v1' has type Tile but expected Scalar")
        rewritten = _rewrite_jit_error(exc, rename_map)
        assert "t_v1" not in str(rewritten)
        assert "'t'" in str(rewritten)

    def test_no_rename_map_returns_original_exception(self):
        """With an empty rename map, the original exception object is returned."""
        exc = ValueError("some error")
        result = _rewrite_jit_error(exc, {})
        assert result is exc

    def test_rebind_longer_alias_replaced_first(self):
        """Longer aliases are replaced before shorter ones to avoid partial matches."""
        rename_map = {"t_v1": "t", "t_v10": "t"}
        exc = ValueError("'t_v10' and 't_v1' are both invalid")
        rewritten = _rewrite_jit_error(exc, rename_map)
        assert "t_v10" not in str(rewritten)
        assert "t_v1" not in str(rewritten)

    def test_rewrite_preserves_exception_fields(self):
        """copy.copy preserves extra fields (e.g. message) for ParserError-style exceptions."""
        exc = ParserTypeError("Variable 'x_v1' has wrong type", hint="use x instead")
        result = _rewrite_jit_error(exc, {"x_v1": "x"})
        assert "x_v1" not in str(result)
        assert "x" in str(result)
        # Extra fields are preserved via copy.copy
        assert isinstance(result, ParserTypeError)
        assert result.hint == "use x instead"  # type: ignore[attr-defined]

    def test_rewrite_non_standard_exception_falls_back(self):
        """Exceptions where copy.copy fails fall back to plain Exception."""

        class WeirdError(Exception):
            def __init__(self, code: int, msg: str) -> None:
                super().__init__(msg)
                self.code = code

        exc = WeirdError(42, "Variable 'x_v1' is invalid")
        result = _rewrite_jit_error(exc, {"x_v1": "x"})
        assert "x_v1" not in str(result)
        assert "x" in str(result)


# ---------------------------------------------------------------------------
# ir.compile() kwarg forwarding (Issue #1405)
# ---------------------------------------------------------------------------


class TestCompileKwargForwarding:
    """``ir.compile()`` kwargs are forwarded through ``JITFunction._compile``.

    Before this fix, ``_compile`` only forwarded ``skip_ptoas`` and
    ``platform`` — every other compile knob a user set on ``RunConfig``
    (``strategy``, ``dump_passes``, ...) was silently dropped on the JIT path.

    The mapping itself is ``RunConfig.compile_kwargs`` and is tested in
    ``tests/ut/runtime/test_run_config.py``; what these cases pin is that the
    JIT path uses it rather than a second copy of it.
    """

    def test_make_cache_key_splits_on_distributed_config(self):
        """distributed_config participates in the cache key (distinct device_ids ≠ collide)."""
        from pypto.ir import DistributedConfig  # noqa: PLC0415
        from pypto.jit.cache import make_cache_key  # noqa: PLC0415

        def key_for(distributed_config):
            return make_cache_key(
                source_hash="h",
                param_names=["x"],
                tensor_shapes={"x": (128, 128)},
                tensor_dtypes={"x": DataType.FP32},
                dynamic_dims=set(),
                platform="a2a3",
                strategy=OptimizationStrategy.Default,
                distributed_config=distributed_config,
            )

        key_none = key_for(None)
        key_01 = key_for(DistributedConfig(device_ids=[0, 1]))
        key_23 = key_for(DistributedConfig(device_ids=[2, 3]))
        key_01_again = key_for(DistributedConfig(device_ids=[0, 1]))

        # Distinct device_ids must not collide, and a distributed config must
        # not collide with the single-chip (None) default.
        assert len({key_none, key_01, key_23}) == 3
        # Equal configs yield equal keys, so a genuine re-call still hits the
        # cache; the key stays hashable (usable in a set / as a dict key).
        assert key_01 == key_01_again

    def test_make_cache_key_splits_on_auto_scope_deps_switch(self):
        """AUTO-scope dependency analysis changes codegen, so it splits cache."""
        from pypto.jit.cache import make_cache_key  # noqa: PLC0415

        def key_for(enabled):
            return make_cache_key(
                source_hash="h",
                param_names=["x"],
                tensor_shapes={"x": (128, 128)},
                tensor_dtypes={"x": DataType.FP32},
                dynamic_dims=set(),
                platform="a2a3",
                strategy=OptimizationStrategy.Default,
                analyze_auto_scopes_for_deps=enabled,
            )

        assert key_for(False) != key_for(True)

    def test_resolve_compiled_splits_cache_on_distributed_config(self, monkeypatch):
        """Two calls differing only in distributed_config compile distinct artifacts.

        Regression for the JIT cache key omitting ``distributed_config``: the
        config is baked into the ``DistributedCompiledProgram`` and drives
        per-rank dispatch, so reusing the first artifact for a second call with
        different ``device_ids`` would silently target the wrong ranks.
        """
        torch = pytest.importorskip("torch")
        from pypto.ir import DistributedConfig  # noqa: PLC0415

        @jit
        def cfg_kernel(a: pl.Tensor[[128, 128], pl.FP32], c: pl.Out[pl.Tensor[[128, 128], pl.FP32]]):
            c = a
            return c

        # Stub out the actual compile so the test stays device-free and only
        # exercises the cache-key / cache-miss logic in _resolve_compiled.
        compile_calls = {"n": 0}

        def fake_compile(*_args, **_kwargs):
            compile_calls["n"] += 1
            return f"compiled-{compile_calls['n']}"

        monkeypatch.setattr(cfg_kernel, "_compile", fake_compile)

        a = torch.randn(128, 128)
        c = torch.empty(128, 128)

        def resolve(device_ids, analyze_auto_scopes_for_deps=False):
            cfg = RunConfig(
                distributed_config=DistributedConfig(device_ids=device_ids),
                analyze_auto_scopes_for_deps=analyze_auto_scopes_for_deps,
            )
            return cfg_kernel._resolve_compiled((a, c), {"config": cfg})[0]

        first = resolve([0, 1])
        second = resolve([2, 3])  # different device_ids → cache miss
        third = resolve([0, 1])  # same as first → cache hit

        assert compile_calls["n"] == 2  # only two compiles, not three
        assert len(cfg_kernel._cache) == 2  # two distinct cached artifacts
        assert first != second  # not the same cached object
        assert third == first  # re-uses the first artifact

    def test_resolve_compiled_splits_cache_on_auto_scope_deps_switch(self, monkeypatch):
        """The same JIT call compiles separately when the AUTO-scope deps switch changes."""
        torch = pytest.importorskip("torch")

        @jit
        def cfg_kernel(a: pl.Tensor[[128, 128], pl.FP32], c: pl.Out[pl.Tensor[[128, 128], pl.FP32]]):
            c = a
            return c

        compile_calls = {"n": 0}

        def fake_compile(*_args, **_kwargs):
            compile_calls["n"] += 1
            return f"compiled-{compile_calls['n']}"

        monkeypatch.setattr(cfg_kernel, "_compile", fake_compile)

        a = torch.randn(128, 128)
        c = torch.empty(128, 128)

        def resolve(enabled):
            cfg = RunConfig(analyze_auto_scopes_for_deps=enabled)
            return cfg_kernel._resolve_compiled((a, c), {"config": cfg})[0]

        first = resolve(False)
        second = resolve(True)
        third = resolve(False)

        assert compile_calls["n"] == 2
        assert len(cfg_kernel._cache) == 2
        assert first != second
        assert third == first

    def test_resolve_compiled_bypasses_cache_on_ptoas_pass_dump(self, monkeypatch):
        """Enabling ptoas pass dumps cannot reuse an artifact compiled without them."""
        torch = pytest.importorskip("torch")

        @jit
        def cfg_kernel(a: pl.Tensor[[128, 128], pl.FP32], c: pl.Out[pl.Tensor[[128, 128], pl.FP32]]):
            c = a
            return c

        compile_calls = {"n": 0}

        def fake_compile(*_args, **_kwargs):
            compile_calls["n"] += 1
            return f"compiled-{compile_calls['n']}"

        monkeypatch.setattr(cfg_kernel, "_compile", fake_compile)

        a = torch.randn(128, 128)
        c = torch.empty(128, 128)

        def resolve(dump_ptoas_passes):
            cfg = RunConfig(dump_ptoas_passes=dump_ptoas_passes)
            return cfg_kernel._resolve_compiled((a, c), {"config": cfg})[0]

        first = resolve(False)
        second = resolve(True)
        repeated_dump = resolve(True)
        third = resolve(False)

        assert compile_calls["n"] == 3
        assert len(cfg_kernel._cache) == 1
        assert first != second
        assert second != repeated_dump
        assert third == first

    def test_compile_forwards_run_config_kwargs(self, monkeypatch):
        """_compile forwards ir_compile_kwargs verbatim to ir.compile()."""
        # `pypto.ir.compile` the attribute is the re-exported function, so
        # import the submodule explicitly to patch the name _compile reads.
        ir_compile_mod = importlib.import_module("pypto.ir.compile")

        @jit
        def fwd_kernel(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                t = pl.load(x, [0, 0], [128, 128])
                pl.store(t, [0, 0], out)
            return out

        captured: dict = {}

        def fake_compile(_program, **kwargs):
            captured.update(kwargs)
            return "fake-compiled-program"

        # _compile re-imports `compile` from pypto.ir.compile on each call,
        # so patching the module attribute intercepts the real compilation.
        monkeypatch.setattr(ir_compile_mod, "compile", fake_compile)

        from pypto.ir import DistributedConfig  # noqa: PLC0415

        dc = DistributedConfig(device_ids=[0, 1])
        cfg = RunConfig(
            strategy=OptimizationStrategy.Default,
            dump_passes=True,
            compile_profiling=True,
            distributed_config=dc,
            analyze_auto_scopes_for_deps=True,
        )
        result = fwd_kernel._compile(
            tensor_meta={
                "x": TensorMeta((128, 128), DataType.FP32),
                "out": TensorMeta((128, 128), DataType.FP32),
            },
            scalar_dtypes={},
            constexpr_values={},
            per_func_dyn={id(fwd_kernel._func): {}},
            pl=pl,
            **cfg.compile_kwargs(),
        )
        assert result == "fake-compiled-program"
        assert captured["strategy"] == OptimizationStrategy.Default
        assert captured["dump_passes"] is True
        assert captured["profiling"] is True
        assert captured["backend_type"] == cfg.backend_type
        assert captured["execution_mode"] is cfg.execution_mode
        assert "platform" not in captured
        assert captured["analyze_auto_scopes_for_deps"] is True
        assert "skip_ptoas" in captured
        # distributed_config reaches ir.compile() so a @pl.jit.host entry can
        # compile to a DistributedCompiledProgram and dispatch per-rank.
        assert captured["distributed_config"] is dc

    def test_compile_without_output_dir_allocates_one(self, monkeypatch, tmp_path):
        """Each JIT compilation owns an automatic directory when none is supplied."""
        ir_compile_mod = importlib.import_module("pypto.ir.compile")
        monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", str(tmp_path))

        @jit
        def plain_kernel(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                t = pl.load(x, [0, 0], [128, 128])
                pl.store(t, [0, 0], out)
            return out

        captured: dict = {}

        def fake_compile(_program, **kwargs):
            captured.update(kwargs)
            return "fake"

        monkeypatch.setattr(ir_compile_mod, "compile", fake_compile)

        plain_kernel._compile(
            tensor_meta={
                "x": TensorMeta((128, 128), DataType.FP32),
                "out": TensorMeta((128, 128), DataType.FP32),
            },
            scalar_dtypes={},
            constexpr_values={},
            per_func_dyn={id(plain_kernel._func): {}},
            pl=pl,
        )
        assert set(captured) == {"skip_ptoas", "output_dir"}
        output_dir = Path(captured["output_dir"])
        assert output_dir.is_dir()
        # Same helper as ir.compile(): <base>/<program name>_<YYYYmmdd_HHMMSS>_<random>.
        assert output_dir.parent == tmp_path
        assert re.fullmatch(r"_jit_plain_kernel_\d{8}_\d{6}_\w+", output_dir.name), output_dir.name


# ---------------------------------------------------------------------------
# Source provenance: diagnostics map back to the user's real .py (Issue #1612)
# ---------------------------------------------------------------------------


@jit
def _provenance_kernel(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(x, [0, 0], [128, 128])
        t = pl.mul(t, t)
        pl.store(t, [0, 0], out)
    return out


def _walk_spans(stmts):
    """Collect spans of all statements (recursing into nested bodies)."""
    out = []
    for s in stmts:
        span = getattr(s, "span", None)
        if span is not None:
            out.append(span)
        body = getattr(s, "body", None)
        if body is not None:
            try:
                out.extend(_walk_spans(list(body)))
            except TypeError:
                pass
    return out


class TestJitSourceProvenance:
    """JIT parse/compile diagnostics point at the user's real source (#1612).

    @pl.jit re-derives the kernel into a generated @pl.program string and
    reparses it, so spans used to land on a synthesized ``<string>`` source.
    A source map now remaps them back to the user's .py at statement
    granularity.
    """

    def test_diagnostic_filename_names_kernel(self):
        """The fallback synthetic filename names the kernel, not ``<string>``."""
        assert _provenance_kernel._diagnostic_filename == "<jit:_provenance_kernel>"

    def test_body_spans_point_at_real_file(self):
        """Every (user-written) body statement's span resolves to this file."""
        prog = _provenance_kernel._compile_to_program(
            tensor_meta={
                "x": TensorMeta((128, 128), DataType.FP32),
                "out": TensorMeta((128, 128), DataType.FP32),
            },
            scalar_dtypes={},
            constexpr_values={},
            per_func_dyn={id(_provenance_kernel._func): {}},
            pl=pl,
        )
        func = prog.get_function("_provenance_kernel")
        assert func is not None

        spans = _walk_spans(list(func.body))
        assert spans, "expected body statement spans"
        # The kernel body is entirely user-written (no synthesized statements),
        # so every body span must resolve to this real source file.
        assert all(s.filename == __file__ for s in spans), [s.filename for s in spans]

    def test_span_line_matches_the_real_source_line(self):
        """The remapped span line equals the statement's actual line in this file."""
        src_lines, start = inspect.getsourcelines(_provenance_kernel._func)
        with_offset = next(i for i, ln in enumerate(src_lines) if "with pl.at" in ln)
        expected_with_line = start + with_offset

        prog = _provenance_kernel._compile_to_program(
            tensor_meta={
                "x": TensorMeta((128, 128), DataType.FP32),
                "out": TensorMeta((128, 128), DataType.FP32),
            },
            scalar_dtypes={},
            constexpr_values={},
            per_func_dyn={id(_provenance_kernel._func): {}},
            pl=pl,
        )
        incore = list(prog.get_function("_provenance_kernel").body)[0]
        assert incore.span.filename == __file__
        assert incore.span.begin_line == expected_with_line


class TestClosureConstantFolding:
    """A ``@pl.jit`` function defined inside a factory (issue #2449).

    The generated ``@pl.program`` source is ``exec``'d in a fresh module holding
    only ``pl`` and ``pld``, so every free name in a body must be folded to a
    literal first. Folding used to read ``__globals__`` alone, where a closure
    free var never appears — so a factory constant survived verbatim and the
    parser raised ``Undefined variable``. Annotations always worked (they are
    rendered from ``tensor_meta``, not from the name), which made the gap
    invisible from outside.
    """

    @staticmethod
    def _specialize(entry, *args) -> str:
        _pn, _, tmeta, sd, cx, pfd = entry._bind_args(args, {})
        contexts = entry._build_contexts(tmeta, sd, cx, pfd)
        return Specializer(f"_jit_{entry.__name__}", contexts).specialize()

    @staticmethod
    def _build_factory_entry(rows: int):
        """Return a @pl.jit entry whose body references the factory's ``rows``."""

        @jit.incore
        def copy_incore(t: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            tile = pl.load(t, [0, 0], [64, 64])
            pl.store(tile, [0, 0], out)
            return out

        @jit
        def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            # ``rows`` is a closure free var, not a module global.
            tmp = pl.create_tensor([rows, 64], dtype=pl.FP32)
            return copy_incore(tmp, out)

        return entry

    def test_closure_constant_folds_into_generated_body(self):
        """A factory constant referenced in a body is inlined as a literal."""
        torch = pytest.importorskip("torch")

        entry = self._build_factory_entry(64)
        a = torch.zeros(64, 64, dtype=torch.float32)
        out = torch.zeros(64, 64, dtype=torch.float32)
        source = self._specialize(entry, a, out)

        assert "pl.create_tensor([64, 64]" in source
        # The free name must not survive — it is undefined in the generated module.
        assert not re.search(r"\brows\b", source)

    def test_closure_constant_specializes_per_value(self):
        """Two factory instantiations fold their own constant, not a shared one."""
        torch = pytest.importorskip("torch")

        a = torch.zeros(64, 64, dtype=torch.float32)
        out = torch.zeros(64, 64, dtype=torch.float32)
        assert "pl.create_tensor([32, 64]" in self._specialize(self._build_factory_entry(32), a, out)
        assert "pl.create_tensor([96, 64]" in self._specialize(self._build_factory_entry(96), a, out)

    @staticmethod
    def _build_mutable_factory_entry():
        """Return (entry, setter) sharing one closure cell, so the setter
        rebinds the *same* ``JITFunction``'s captured constant."""

        rows = 64

        @jit.incore
        def copy_incore(t: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            tile = pl.load(t, [0, 0], [64, 64])
            pl.store(tile, [0, 0], out)
            return out

        @jit
        def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            tmp = pl.create_tensor([rows, 64], dtype=pl.FP32)
            return copy_incore(tmp, out)

        def set_rows(value: int) -> None:
            nonlocal rows
            rows = value

        return entry, set_rows

    def test_rebound_closure_cell_changes_the_generated_source(self):
        """Rebinding the cell on one JITFunction changes what gets emitted.

        This is the premise of the cache-key test below: if the artifact did not
        depend on the cell, keying on it would be pointless.
        """
        torch = pytest.importorskip("torch")

        entry, set_rows = self._build_mutable_factory_entry()
        a = torch.zeros(64, 64, dtype=torch.float32)
        out = torch.zeros(64, 64, dtype=torch.float32)

        before = self._specialize(entry, a, out)
        set_rows(96)
        after = self._specialize(entry, a, out)

        assert "pl.create_tensor([64, 64]" in before
        assert "pl.create_tensor([96, 64]" in after

    def test_rebound_closure_cell_splits_the_cache(self):
        """A rebound cell must not silently reuse the previous artifact.

        A ``nonlocal`` rebind leaves the function text byte-identical, so the
        source dependency hash must capture the changed value.
        Two distinct factory instances would not catch this —
        they are different ``JITFunction`` objects with their own caches.
        """
        torch = pytest.importorskip("torch")

        entry, set_rows = self._build_mutable_factory_entry()
        a = torch.zeros(64, 64, dtype=torch.float32)
        out = torch.zeros(64, 64, dtype=torch.float32)

        del a, out
        source_hash_before = entry._get_source_hash()
        static_hash_before = entry._get_static_source_hash()
        set_rows(96)

        assert entry._get_static_source_hash() == static_hash_before
        assert entry._get_source_hash() != source_hash_before

    def test_rebound_closure_cell_recompiles_instead_of_reusing(self):
        """The observable consequence: ``compile()`` must not hand back the
        artifact built from the previous cell value.

        Kept separate from the key-component test so this assertion is reached
        on its own — a regression in the key wiring has to fail *here*, not be
        masked by an earlier assertion.
        """
        torch = pytest.importorskip("torch")

        entry, set_rows = self._build_mutable_factory_entry()
        a = torch.zeros(64, 64, dtype=torch.float32)
        out = torch.zeros(64, 64, dtype=torch.float32)

        compiled_before = entry.compile(a, out, config=RunConfig(platform="a2a3sim"))
        set_rows(128)
        compiled_after = entry.compile(a, out, config=RunConfig(platform="a2a3sim"))
        assert compiled_before is not compiled_after, "stale artifact reused after rebind"
        assert len(entry._cache) == 2
        # An unchanged cell must still hit the cache, or the key would be useless.
        assert entry.compile(a, out, config=RunConfig(platform="a2a3sim")) is compiled_after
        assert len(entry._cache) == 2

    def test_closure_source_hash_is_stable_and_typed(self):
        """The source hash distinguishes folded closure types and stays stable."""

        def make_entry(value: int | float | bool):
            @jit
            def entry(x):
                return pl.add(x, value)

            return entry

        entry = make_entry(1)
        base = entry._get_source_hash()
        assert base == entry._get_source_hash()
        assert len({base, *(make_entry(value)._get_source_hash() for value in (1.0, True))}) == 3

    def test_module_globals_still_fold(self):
        """Closure bindings are merged on top of globals, not instead of them."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def copy_incore(t: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            tile = pl.load(t, [0, 0], [64, 64])
            pl.store(tile, [0, 0], out)
            return out

        @jit
        def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            tmp = pl.create_tensor([_CLOSURE_FOLD_ROWS, 64], dtype=pl.FP32)
            return copy_incore(tmp, out)

        a = torch.zeros(64, 64, dtype=torch.float32)
        out = torch.zeros(64, 64, dtype=torch.float32)
        source = self._specialize(entry, a, out)
        assert "pl.create_tensor([64, 64]" in source
        assert not re.search(r"\b_CLOSURE_FOLD_ROWS\b", source)


_PAGES_3033 = pl.dynamic("PAGES_3033")
_DIM_ROW_AXIS_3033 = 0


@jit.incore
def _copy_key_3033(keys: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    value = pl.load(keys, [0, 0], [1, 128])
    pl.store(value, [0, 0], out)
    return out


@jit
def _descriptor_reshape_chain_3033(
    src: pl.Tensor[[_PAGES_3033, 4160], pl.INT8],
    out: pl.Out[pl.Tensor[[1, 128], pl.INT8]],
) -> pl.Tensor:
    pages = pl.tensor.dim(src, 0)
    width = pl.tensor.dim(src, 1)
    flat = pl.reshape(src, [pages * width])
    shifted = flat[64 : 64 + ((pages * width - 64) // 128) * 128]
    keys = pl.reshape(shifted, [(pages * width - 64) // 128, 128])
    return _copy_key_3033(keys, out)


@jit
def _descriptor_reshape_alias_chain_3033(
    src: pl.Tensor[[_PAGES_3033, 4160], pl.INT8],
    out: pl.Out[pl.Tensor[[1, 128], pl.INT8]],
) -> pl.Tensor:
    pages = pl.tensor.dim(src, 0)
    width = pl.tensor.dim(src, 1)
    total = pages * width
    rows = (total - 64) // 128
    alias = rows
    flat = pl.reshape(src, [total])
    shifted = flat[64 : 64 + alias * 128]
    keys = pl.reshape(shifted, [alias, 128])
    return _copy_key_3033(keys, out)


class TestRuntimeSizedLocalExtents:
    """A local tensor whose extent is only known at runtime (issue #2450).

    ``pld.window(buf, [pld.world_size(), 1], ...)`` and
    ``pl.create_tensor([pl.tensor.read(cfg, [0]), 128], ...)`` have no static
    extent. Before this behaviour the whole meta was dropped, and the moment the
    local was passed to a dep ``_build_params`` raised "missing inferred tensor
    metadata". Now the unresolved dim becomes a synthesized ``DynDim``, which the
    generated program declares via ``pl.dynamic`` and the kernel binds from the
    actual argument's descriptor — the same IR a hand-written ``@pl.program``
    produces for this pattern.
    """

    @staticmethod
    def _leading_dim(metas: dict[str, TensorMeta], name: str):
        assert name in metas, f"{name!r} has no meta: {sorted(metas)}"
        return metas[name].shape[0]

    def test_runtime_create_extent_synthesizes_dyn_dim(self):
        """A create_tensor dim read out of a tensor becomes a synthesized DynDim."""
        seed = {
            "cfg": TensorMeta(shape=(1,), dtype=DataType.INT64),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        metas = extract_local_tensor_metas(
            _runtime_create_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        dim = self._leading_dim(metas, "tmp")
        assert isinstance(dim, DynDim)
        assert dim.synthesized
        assert dim.name.startswith(_SYNTHESIZED_DYN_PREFIX)
        # Only the unresolved dim is synthesized; the literal one stays an int.
        assert metas["tmp"].shape[1] == 8
        assert metas["tmp"].dtype == DataType.FP32

    def test_runtime_window_extent_synthesizes_dyn_dim(self):
        """pld.window sized by pld.world_size() gets a synthesized DynDim."""
        seed = {"out": TensorMeta(shape=(16, 8), dtype=DataType.FP32)}
        metas = extract_local_tensor_metas(
            _runtime_window_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        dim = self._leading_dim(metas, "win")
        assert isinstance(dim, DynDim)
        assert dim.synthesized
        assert metas["win"].shape[1] == 8
        assert metas["win"].dtype == DataType.INT32

    def test_arithmetic_over_dyn_dim_synthesizes(self):
        """``M * 2`` is not statically foldable, so that dim is synthesized too."""
        seed = {
            "src": TensorMeta(shape=(DynDim(name="M", literal="M", static_bound=32), 8), dtype=DataType.FP32),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        metas = extract_local_tensor_metas(
            _dyn_arith_create_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        dim = self._leading_dim(metas, "tmp")
        assert isinstance(dim, DynDim)
        assert dim.synthesized

    def test_statically_resolvable_dims_are_not_synthesized(self):
        """Existing static resolution is untouched — no placeholder is invented."""
        seed = {
            "src": TensorMeta(shape=(32, 32), dtype=DataType.FP32),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        metas = extract_local_tensor_metas(
            _slice_then_dep_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        assert metas["buf"] == TensorMeta(shape=(16, 8), dtype=DataType.FP32)
        assert all(not isinstance(d, DynDim) for d in metas["buf"].shape)

    def test_reshape_stays_strict(self):
        """An arbitrary tensor.read extent is not descriptor-derived arithmetic,
        so reshape still declines its metadata."""
        seed = {
            "src": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
            "cfg": TensorMeta(shape=(1,), dtype=DataType.INT64),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        metas = extract_local_tensor_metas(
            _runtime_reshape_body, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        assert "flat" not in metas

    def test_issue_3033_descriptor_reshape_chain_specializes(self):
        """Flatten/slice/reshape reaches a bare-annotated dep without samples or a device."""
        program = _descriptor_reshape_chain_3033.specialize()
        assert isinstance(program, ir.Program)
        kernel = program.get_function("_copy_key_3033")
        assert kernel is not None
        param_type = kernel.params[0].type
        assert isinstance(param_type, ir.TensorType)
        shape = param_type.shape
        assert len(shape) == 2
        assert not isinstance(shape[0], ir.ConstInt)
        assert isinstance(shape[1], ir.ConstInt) and shape[1].value == 128
        entry = program.get_function("_descriptor_reshape_chain_3033")
        assert entry is not None
        source = entry.as_python()
        reshapes = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "reshape"
        ]
        assert len(reshapes) == 2
        assert any(isinstance(node, ast.Mult) for node in ast.walk(reshapes[0].args[1]))
        assert any(isinstance(node, ast.FloorDiv) for node in ast.walk(reshapes[1].args[1]))

    def test_descriptor_reshape_nested_integer_arithmetic(self):
        """Supported integer operators stay descriptor-derived through nesting and unary minus."""
        rows = DynDim(name="M", literal="M", static_bound=16)
        seed = {"src": TensorMeta(shape=(rows, 128), dtype=DataType.FP32)}

        def body(src):
            rows = src.shape[0]
            nested = pl.reshape(src, [((rows + 4) * 2 - -8) // 4])
            signed = pl.reshape(src, [-(-rows)])
            return nested, signed

        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        for name in ("nested", "signed"):
            dim = self._leading_dim(metas, name)
            assert isinstance(dim, DynDim) and dim.synthesized
            assert dim.name != rows.name

    @pytest.mark.parametrize(
        "source",
        [
            "transitive_shape_alias",
            "annotated_shape_alias",
            "rebound_shape_alias",
            "rebound_source_shape_alias",
        ],
    )
    def test_descriptor_reshape_local_integer_unary_plus_stays_unknown(self, source):
        """An inferred integer does not by itself prove unary plus can fold in generated source."""

        def transitive_shape_alias(src):
            width = src.shape[1]
            saved = width
            flat = pl.reshape(src, [+saved])
            return flat

        def annotated_shape_alias(src):
            width: int = src.shape[1]
            flat = pl.reshape(src, [+width])
            return flat

        def rebound_shape_alias(src):
            width = src.shape[1]
            width = 128
            flat = pl.reshape(src, [+width])
            return flat

        def rebound_source_shape_alias(src, other):
            src = other
            width = src.shape[1]
            flat = pl.reshape(src, [+width])
            return flat

        seed = {
            "src": TensorMeta(shape=(1, 128), dtype=DataType.INT8),
            "other": TensorMeta(shape=(1, 64), dtype=DataType.INT8),
        }
        bodies = {
            "transitive_shape_alias": transitive_shape_alias,
            "annotated_shape_alias": annotated_shape_alias,
            "rebound_shape_alias": rebound_shape_alias,
            "rebound_source_shape_alias": rebound_source_shape_alias,
        }
        metas = extract_local_tensor_metas(
            bodies[source], seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        assert "flat" not in metas

    @pytest.mark.parametrize(
        "rows_expr,total_expr,error",
        [
            ("src.shape[-2]", "rows * 128", "Only constant integer indices supported for tuple access"),
            ("pl.tensor.dim(src, 0)", "+rows", "Unsupported unary operator: UAdd"),
            ("pl.tensor.dim(src, 0)", "+(rows * 128)", "Unsupported unary operator: UAdd"),
            ("128", "+rows", "Unsupported unary operator: UAdd"),
            ("pl.tensor.dim(src, 1)", "+rows", "Unsupported unary operator: UAdd"),
            ("128", "+pl.tensor.dim(src, 1)", "Unsupported unary operator: UAdd"),
        ],
    )
    def test_descriptor_reshape_unsupported_shape_syntax_rejected(
        self, monkeypatch, rows_expr, total_expr, error
    ):
        """The actual parser still rejects unsupported shape indices and scalar unary plus."""

        def body(src: pl.Tensor[[_PAGES_3033, 128], pl.INT8]):
            rows = 128
            total = rows
            flat = pl.reshape(src, [total])
            return flat

        # Supply the unsupported DSL spelling through source text, keeping
        # Python's own Tensor/Scalar typing separate from parser diagnostics.
        get_lines = inspect.getsourcelines
        lines, line = get_lines(body)
        source = "".join(lines).replace("rows = 128", f"rows = {rows_expr}")
        source = source.replace("total = rows", f"total = {total_expr}")
        monkeypatch.setattr(
            inspect,
            "getsourcelines",
            lambda func: (source.splitlines(keepends=True), line) if func is body else get_lines(func),
        )
        entry = jit(body)
        seed = {
            "src": TensorMeta(
                shape=(DynDim(name="PAGES", literal="PAGES", static_bound=4), 128), dtype=DataType.INT8
            )
        }
        assert "flat" not in extract_local_tensor_metas(
            entry._func, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        with pytest.raises(UnsupportedFeatureError, match=error):
            entry.specialize()

    @pytest.mark.parametrize(
        "source",
        ["literal", "global_constant", "inline_shape", "nested_shape"],
    )
    def test_descriptor_reshape_compile_time_unary_plus_supported(self, monkeypatch, source):
        """Unary plus remains supported for expressions the parser can fold at compile time."""

        def body(src: pl.Tensor[[1, 128], pl.INT8]):
            rows = 128
            flat = pl.reshape(src, [+rows])
            return flat

        spellings = {
            "literal": ("rows = 128", "+128"),
            "global_constant": ("rows = 128", "+(_DIM_ROW_AXIS_3033 + 128)"),
            "inline_shape": ("rows = 128", "+src.shape[1]"),
            "nested_shape": ("rows = 128", "+(src.shape[1] + 1 - 1)"),
        }
        assignment, expression = spellings[source]
        get_lines = inspect.getsourcelines
        lines, line = get_lines(body)
        source_text = "".join(lines).replace("rows = 128", assignment).replace("[+rows]", f"[{expression}]")
        monkeypatch.setattr(
            inspect,
            "getsourcelines",
            lambda func: (source_text.splitlines(keepends=True), line) if func is body else get_lines(func),
        )
        entry = jit(body)
        seed = {"src": TensorMeta(shape=(1, 128), dtype=DataType.INT8)}
        metas = extract_local_tensor_metas(entry._func, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["flat"].shape == (128,)
        assert isinstance(entry.lower(), ir.Program)

    @pytest.mark.parametrize("source", ["shape_alias", "unpack", "future_write", "one_prior_write"])
    def test_descriptor_reshape_static_shape_unary_plus_cross_dep(self, monkeypatch, source):
        """Static shape aliases that the specializer inlines must continue to type a dependency."""

        def body(
            src: pl.Tensor[[1, 128], pl.INT8],
            out: pl.Out[pl.Tensor[[1, 128], pl.INT8]],
        ):
            rows = 128
            view = pl.reshape(src, [1, +rows])
            out = _copy_key_3033(view, out)
            return out

        get_lines = inspect.getsourcelines
        lines, line = get_lines(body)
        func_def = ast.parse(textwrap.dedent("".join(lines))).body[0]
        assert isinstance(func_def, ast.FunctionDef)
        assignments = {
            "shape_alias": "rows = src.shape[1]",
            "unpack": "height, rows = src.shape",
            "future_write": "rows = src.shape[1]",
            "one_prior_write": "rows = 1\nrows = src.shape[1]",
        }
        func_def.body[:1] = ast.parse(assignments[source]).body
        if source == "future_write":
            func_def.body.insert(-1, ast.parse("rows = 128").body[0])
        source_text = ast.unparse(func_def) + "\n"
        monkeypatch.setattr(
            inspect,
            "getsourcelines",
            lambda func: (source_text.splitlines(keepends=True), line) if func is body else get_lines(func),
        )
        entry = jit(body)
        seed = {
            "src": TensorMeta(shape=(1, 128), dtype=DataType.INT8),
            "out": TensorMeta(shape=(1, 128), dtype=DataType.INT8),
        }
        metas = extract_local_tensor_metas(entry._func, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["view"].shape == (1, 128)
        assert isinstance(entry.lower(), ir.Program)

    def test_descriptor_reshape_static_shape_unary_plus_nested_dep(self, monkeypatch):
        """An inlined shape alias inside a branch also reaches a real consumer."""

        def body(
            src: pl.Tensor[[1, 128], pl.INT8],
            out: pl.Out[pl.Tensor[[1, 128], pl.INT8]],
            flag: pl.Scalar[pl.INDEX],
        ):
            if flag > 0:
                rows = 128
                view = pl.reshape(src, [1, +rows])
                updated = _copy_key_3033(view, out)
                result = pl.yield_(updated)
            else:
                result = pl.yield_(out)
            return result

        get_lines = inspect.getsourcelines
        lines, line = get_lines(body)
        source = "".join(lines).replace("rows = 128", "rows = src.shape[1]")
        monkeypatch.setattr(
            inspect,
            "getsourcelines",
            lambda func: (source.splitlines(keepends=True), line) if func is body else get_lines(func),
        )
        entry = jit(body)
        seed = {
            "src": TensorMeta(shape=(1, 128), dtype=DataType.INT8),
            "out": TensorMeta(shape=(1, 128), dtype=DataType.INT8),
        }
        assert extract_local_tensor_metas(entry._func, seed_meta=seed, discover_deps=_discover_dep_bindings)[
            "view"
        ].shape == (1, 128)
        assert isinstance(entry.lower(), ir.Program)

    def test_descriptor_reshape_static_shape_unary_plus_active_rename_rejected(self, monkeypatch):
        """A shape assignment does not remove an earlier runtime SSA rename."""

        def body(src: pl.Tensor[[1, 128], pl.INT8]):
            rows = 1
            rows = 2
            rows = 128
            view = pl.reshape(src, [1, +rows])
            return view

        get_lines = inspect.getsourcelines
        lines, line = get_lines(body)
        source = "".join(lines).replace("rows = 128", "rows = src.shape[1]")
        monkeypatch.setattr(
            inspect,
            "getsourcelines",
            lambda func: (source.splitlines(keepends=True), line) if func is body else get_lines(func),
        )
        entry = jit(body)
        seed = {"src": TensorMeta(shape=(1, 128), dtype=DataType.INT8)}
        assert "view" not in extract_local_tensor_metas(
            entry._func, seed_meta=seed, discover_deps=_discover_dep_bindings
        )
        with pytest.raises(UnsupportedFeatureError, match="Unsupported unary operator: UAdd"):
            entry.specialize()

    def test_descriptor_reshape_arithmetic_alias_chain(self):
        """Transitive arithmetic aliases cross a dep boundary with no sample tensors."""
        program = _descriptor_reshape_alias_chain_3033.lower()
        assert isinstance(program, ir.Program)
        kernel = program.get_function("_copy_key_3033")
        assert kernel is not None
        param_type = kernel.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert len(param_type.shape) == 2
        assert not isinstance(param_type.shape[0], ir.ConstInt)
        assert isinstance(param_type.shape[1], ir.ConstInt) and param_type.shape[1].value == 128

    @pytest.mark.parametrize("entry", [_descriptor_reshape_chain_3033, _descriptor_reshape_alias_chain_3033])
    def test_descriptor_reshape_arithmetic_stays_dynamic(self, entry):
        """Aliased products retain dynamic shape information independently of sample bounds."""
        shapes = []
        for pages, width in ((8, 64), (16, 128)):
            seed = {
                "src": TensorMeta(
                    shape=(
                        DynDim(name="PAGES", literal="PAGES", static_bound=pages),
                        DynDim(name="WIDTH", literal="WIDTH", static_bound=width),
                    ),
                    dtype=DataType.INT8,
                ),
                "out": TensorMeta(shape=(1, 128), dtype=DataType.INT8),
            }
            metas = extract_local_tensor_metas(
                entry._func, seed_meta=seed, discover_deps=_discover_dep_bindings
            )
            flat_dim = self._leading_dim(metas, "flat")
            keys_dim = self._leading_dim(metas, "keys")
            assert isinstance(flat_dim, DynDim) and flat_dim.synthesized
            assert isinstance(keys_dim, DynDim) and keys_dim.synthesized
            assert flat_dim.name != keys_dim.name
            assert metas["keys"].shape[1] == 128
            shapes.append((metas["flat"].shape, metas["keys"].shape))
        assert shapes[0] == shapes[1]

    def test_descriptor_reshape_repeated_snapshot_shares_only_result_axes(self):
        """Repeated aliases share one shape symbol without merging independent snapshots or results."""
        rows = DynDim(name="ROWS", literal="ROWS", static_bound=4)
        seed = {"src": TensorMeta(shape=(rows, rows), dtype=DataType.FP32)}

        def body(src):
            n = src.shape[0] * 1
            alias = n
            square = pl.reshape(src, [n, alias])
            another = pl.reshape(src, [n, n])
            m = src.shape[1] * 1
            independent = pl.reshape(src, [n, m])
            saved = n
            n = src.shape[0] * 2
            rebound = pl.reshape(src, [saved, n])
            inline = pl.reshape(src, [src.shape[0] * 1, src.shape[1] * 1])
            work = pl.create_tensor([saved], dtype=pl.FP32)
            result = pl.reshape(work, [saved])
            return square, another, independent, rebound, inline, work, result

        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        square = metas["square"].shape
        another = metas["another"].shape
        assert isinstance(square[0], DynDim) and square[0].synthesized
        assert square[0] is square[1]
        assert isinstance(another[0], DynDim)
        assert another[0] is another[1]
        assert square[0] is not another[0]
        assert square[0].name != another[0].name
        for name in ("independent", "rebound", "inline"):
            first, second = metas[name].shape
            assert isinstance(first, DynDim) and isinstance(second, DynDim)
            assert first is not second
            assert first.name != second.name
        assert metas["work"].shape[0] != metas["result"].shape[0]

    def test_descriptor_reshape_repeated_snapshot_square_dep_specializes(self):
        """A square reshape retains axis equality for transpose/fmod inside a bare-annotated dep."""

        @jit.incore
        def square_dep(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            remainder = pl.fmod(x, pl.transpose(x, 0, 1))
            out = pl.assemble(out, remainder, [0, 0])
            return out

        @jit
        def entry(
            src: pl.Tensor[[_PAGES_3033, _PAGES_3033], pl.FP32],
            out: pl.Out[pl.Tensor[[_PAGES_3033, _PAGES_3033], pl.FP32]],
        ):
            n = pl.tensor.dim(src, 0) // 2
            block = pl.slice(src, [n, n], [0, 0])
            view = pl.reshape(block, [n, n])
            return square_dep(view, out)

        assert isinstance(entry.specialize(), ir.Program)

    def test_descriptor_reshape_repeated_snapshot_square_tile_dep(self):
        """The shared square type also survives verified lowering with fixed-size tile computation."""

        @jit.incore
        def square_tile_dep(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            value = pl.load(x, [0, 0], [2, 2])
            remainder = pl.fmod(value, pl.transpose(value, 0, 1))
            pl.store(remainder, [0, 0], out)
            return out

        @jit
        def entry(
            src: pl.Tensor[[_PAGES_3033, _PAGES_3033], pl.FP32],
            out: pl.Out[pl.Tensor[[2, 2], pl.FP32]],
        ):
            n = pl.tensor.dim(src, 0) // 2
            block = pl.slice(src, [n, n], [0, 0])
            view = pl.reshape(block, [n, n])
            return square_tile_dep(view, out)

        program = entry.lower()
        assert isinstance(program, ir.Program)
        kernel = program.get_function("square_tile_dep")
        assert kernel is not None
        param_type = kernel.params[0].type
        assert isinstance(param_type, ir.TensorType)
        first, second = param_type.shape
        assert isinstance(first, ir.Var) and isinstance(second, ir.Var)
        assert first.unique_id == second.unique_id

    def test_descriptor_reshape_alias_snapshots_scalar_rebindings(self):
        """An alias keeps its old expression when its source scalar is rebound or becomes unknown."""
        seed = {
            "src": TensorMeta(
                shape=(DynDim(name="PAGES", literal="PAGES", static_bound=4), 128), dtype=DataType.INT8
            ),
            "cfg": TensorMeta(shape=(1,), dtype=DataType.INT64),
        }

        def body(src, cfg):
            pages = pl.tensor.dim(src, 0)
            total = pages * 128
            saved = total
            total = total - 64
            rows = total // 128
            pages = pl.tensor.read(cfg, [0])
            total = pages
            original = pl.reshape(src, [saved])
            derived = pl.reshape(src, [rows, 128])
            unknown = pl.reshape(src, [total])
            return original, derived, unknown

        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        for name in ("original", "derived"):
            dim = self._leading_dim(metas, name)
            assert isinstance(dim, DynDim) and dim.synthesized
        assert "unknown" not in metas

    def test_descriptor_reshape_integer_aliases_use_assignment_values(self):
        """Integer aliases fold exactly and read the RHS before updating a rebound name."""
        seed = {"src": TensorMeta(shape=(4, 8), dtype=DataType.FP32)}

        def body(src):
            pages = src.shape[0]
            width = src.shape[1]
            total = pages * width
            saved = total
            total = half = total // 2
            pages = 2
            width = 4
            block = pages * width
            extent = ((saved + 16 - 8) // 2) - -12
            original = pl.reshape(src, [saved])
            rebound = pl.reshape(src, [half, 2])
            constants = pl.reshape(src, [block, 4])
            arithmetic = pl.reshape(src, [extent])
            return original, rebound, constants, arithmetic

        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["original"].shape == (32,)
        assert metas["rebound"].shape == (16, 2)
        assert metas["constants"].shape == (8, 4)
        assert metas["arithmetic"].shape == (32,)

    def test_descriptor_reshape_chained_aliases_cross_dep(self, monkeypatch, tmp_path):
        """Chained aliases and rebound shape snapshots compile across a dependency."""
        torch = pytest.importorskip("torch")
        monkeypatch.setenv("PTOAS_ROOT", str(tmp_path / "missing_ptoas"))

        @jit.incore
        def consume(first: pl.Tensor, second: pl.Tensor, out: pl.Out[pl.Tensor]):
            left = pl.load(first, [0, 0], [1, 32])
            right = pl.load(second, [0, 0], [1, 32])
            pl.store(pl.add(left, right), [0, 0], out)
            return out

        @jit
        def entry(
            src: pl.Tensor[[_PAGES_3033, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 32], pl.FP32]],
        ):
            PAGES = alias = pl.dynamic("PAGES")
            src.bind_dynamic(0, PAGES)
            total = alias * 128
            total = half = total // 2
            total = total * 2
            first = pl.reshape(src, [total // 32, 32])
            second = pl.reshape(src, [half // 16, 32])
            return consume(first, second, out)

        @jit
        def alias_first(src: pl.Tensor, out: pl.Out[pl.Tensor[[1, 32], pl.FP32]]):
            alias = PAGES = pl.dynamic("PAGES")
            src.bind_dynamic(0, PAGES)
            total = alias * 128
            view = pl.reshape(src, [total // 32, 32])
            return consume(view, view, out)

        for kernel in (entry, alias_first):
            compiled = kernel.compile(torch.empty(4, 128), torch.empty(1, 32))
            assert isinstance(compiled, CompiledProgram)

        def rebound(
            src: pl.Tensor[[1, 128], pl.FP32],
            other: pl.Tensor[[4, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 32], pl.FP32]],
        ):
            src = other
            rows, cols = pl.tensor.dim(src, 0), pl.tensor.dim(src, 1)
            width = pl.tensor.dim(src, 1)
            half = width // 2
            view = pl.reshape(src, [rows * cols // half, half])
            return consume(view, view, out)

        def via_add(
            src: pl.Tensor[[4, 64], pl.FP32],
            other: pl.Tensor[[4, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 32], pl.FP32]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                src = pl.add(src, other)
                rows, cols = pl.tensor.dim(src, 0), pl.tensor.dim(src, 1)
                _shape_view = pl.reshape(src, shape=[rows, cols])
                _buffer = pl.create_tensor([rows, cols], dtype=pl.FP32)
                view = pl.reshape(src, [rows * cols // 32, 32])
                out[:, :] = view[:1, :]
            return out

        def via_reshape(src: pl.Tensor, out: pl.Out[pl.Tensor[[1, 32], pl.FP32]]):
            src = pl.reshape(src, [256])
            length = pl.tensor.dim(src, 0)
            view = pl.reshape(src, [length // 32, 32])
            return consume(view, view, out)

        # Trace indexed shape syntax without Tensor.shape's optional Python type.
        get_lines = inspect.getsourcelines
        for func, args in (
            (rebound, (torch.empty(1, 128), torch.empty(4, 64), torch.empty(1, 32))),
            (via_add, (torch.empty(4, 64), torch.empty(4, 64), torch.empty(1, 32))),
            (via_reshape, (torch.empty(4, 64), torch.empty(1, 32))),
        ):
            lines, line = get_lines(func)
            source = "".join(lines).replace(
                "rows, cols = pl.tensor.dim(src, 0), pl.tensor.dim(src, 1)", "rows, cols = src.shape"
            )
            source = source.replace("length = pl.tensor.dim(src, 0)", "length, = src.shape")
            source = source.replace("pl.tensor.dim(src, 1)", "src.shape[1]")
            source = source.replace("pl.tensor.dim(src, 0)", "src.shape[0]")
            source = source.replace("shape=[rows, cols]", "shape=src.shape")
            source = source.replace("dtype=pl.FP32", "dtype=src.dtype")
            monkeypatch.setattr(
                inspect,
                "getsourcelines",
                lambda f, func=func, source=source, line=line: (
                    (source.splitlines(keepends=True), line) if f is func else get_lines(f)
                ),
            )
            compiled = jit(func).compile(*args)
            assert isinstance(compiled, CompiledProgram)

    def test_descriptor_reshape_alias_invalid_rebindings_are_forgotten(self):
        """Unknown rebindings and loop targets invalidate an earlier known arithmetic alias."""
        seed = {
            "src": TensorMeta(
                shape=(DynDim(name="symbol", literal="PAGES", static_bound=4), 128), dtype=DataType.INT8
            ),
            "cfg": TensorMeta(shape=(1,), dtype=DataType.INT64),
        }

        def body(src, cfg, runtime):
            pages = pl.tensor.dim(src, 0)
            symbol = pages
            total = pages * 128
            known = pl.reshape(src, [total])
            total = runtime
            scalar_sized = pl.reshape(src, [total])
            total = pages * 128
            total = pl.tensor.read(cfg, [0])
            read_sized = pl.reshape(src, [total])
            total = pages * 128
            for total in pl.range(2):
                loop_sized = pl.reshape(src, [total])
            symbol = pl.tensor.read(cfg, [0])
            symbol_sized = pl.reshape(src, [symbol])
            return known, scalar_sized, read_sized, loop_sized, symbol_sized

        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert "known" in metas
        assert not {"scalar_sized", "read_sized", "loop_sized", "symbol_sized"} & metas.keys()

    def test_descriptor_reshape_aliases_reject_invalid_arithmetic(self):
        """Invalid arithmetic cannot become descriptor-derived by copying its scalar alias."""
        seed = {
            "src": TensorMeta(
                shape=(DynDim(name="PAGES", literal="PAGES", static_bound=4), 128), dtype=DataType.INT8
            )
        }

        def body(src):
            pages = src.shape[0]
            zero = 2 - 2
            invalid = pages // zero
            alias = invalid
            zero_sized = pl.reshape(src, [alias])
            invalid = pages % 2
            alias = invalid + 1
            modulo_sized = pl.reshape(src, [alias])
            invalid = pages / 2
            alias = invalid
            true_div_sized = pl.reshape(src, [alias])
            invalid = pages**2
            alias = invalid
            power_sized = pl.reshape(src, [alias])
            return zero_sized, modulo_sized, true_div_sized, power_sized

        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert not {"zero_sized", "modulo_sized", "true_div_sized", "power_sized"} & metas.keys()

    def test_descriptor_reshape_alias_control_flow_stays_strict(self):
        """Control-flow writes must not advertise the last syntactic scalar value as certain."""
        seed = {"src": TensorMeta(shape=(64,), dtype=DataType.FP32)}

        def body(src, runtime):
            extent = 32
            if runtime:
                extent = 64
            after_branch = pl.reshape(src, [extent])
            extent = 32
            for _ in pl.range(2):
                before_write = pl.reshape(src, [extent])
                extent = 64
            after_loop = pl.reshape(src, [extent])
            return after_branch, before_write, after_loop

        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert not {"after_branch", "before_write", "after_loop"} & metas.keys()

    @pytest.mark.parametrize("scope_kind", ["if", "for"])
    @pytest.mark.parametrize("dynamic", [False, True])
    def test_descriptor_reshape_alias_unrelated_scope_keeps_callsite(self, scope_kind, dynamic):
        """A later scope's writes cannot invalidate a dimension at an earlier dependency call."""
        rows = DynDim(name="ROWS", literal="ROWS", static_bound=4) if dynamic else 4
        meta = TensorMeta(shape=(rows, 8), dtype=DataType.FP32)
        seed = {"src": meta, "out": meta}

        def conditional_body(src, out, flag, other):
            rows = src.shape[0]
            if flag:
                view = pl.reshape(src, [rows, 8])
                out = _callsite_metadata_kernel(view, out)
            if other:
                rows = 1
            return out

        def loop_body(src, out, flag, other):
            rows = src.shape[0]
            for _ in pl.range(2):
                view = pl.reshape(src, [rows, 8])
                out = _callsite_metadata_kernel(view, out)
            if other:
                rows = 1
            return out

        body = conditional_body if scope_kind == "if" else loop_body
        dep_metas, _ = resolve_dep_call_metadata(
            _callsite_metadata_kernel, body, seed, {}, {}, discover_deps=_discover_dep_bindings
        )
        assert dep_metas["x"] == TensorMeta(shape=meta.shape, dtype=meta.dtype, layout=ir.TensorLayout.ND)

    def test_descriptor_reshape_alias_unrelated_condition_specializes(self):
        """The review's earlier conditional dependency still specializes with dynamic dimensions."""

        @jit
        def entry(
            src: pl.Tensor[[_PAGES_3033, 8], pl.FP32],
            out: pl.Out[pl.Tensor[[_PAGES_3033, 8], pl.FP32]],
            flag: pl.Scalar[pl.BOOL],
            other: pl.Scalar[pl.BOOL],
        ):
            rows = pl.tensor.dim(src, 0)
            if flag:
                view = pl.reshape(src, [rows, 8])
                out = _callsite_metadata_kernel(view, out)
            if other:
                rows = 1
            return out

        program = entry.specialize()
        kernel = program.get_function("_callsite_metadata_kernel")
        assert kernel is not None
        param_type = kernel.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert not isinstance(param_type.shape[0], ir.ConstInt)

    @pytest.mark.parametrize("scope_kind", ["if", "for"])
    def test_descriptor_reshape_alias_scope_assignments_are_definite(self, scope_kind):
        """A loop's carried value is unknown before its write; definite body assignments remain usable."""
        seed = {"src": TensorMeta(shape=(64,), dtype=DataType.FP32)}

        def conditional_body(src, flag):
            extent = 32
            if flag:
                before_write = pl.reshape(src, [extent])
                extent = 64
                assigned = pl.reshape(src, [extent])
            return before_write, assigned

        def loop_body(src, flag):
            extent = 32
            for _ in pl.range(2):
                before_write = pl.reshape(src, [extent])
                extent = 64
                assigned = pl.reshape(src, [extent])
            return before_write, assigned

        body = conditional_body if scope_kind == "if" else loop_body
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        if scope_kind == "if":
            assert metas["before_write"].shape == (32,)
        else:
            assert "before_write" not in metas
        assert metas["assigned"].shape == (64,)

    @pytest.mark.parametrize(
        "arm_kind,unknown,dynamic",
        [("else", False, False), ("else", True, True), ("elif", False, True), ("elif", True, False)],
    )
    def test_descriptor_reshape_alias_sibling_arm_keeps_parent(self, monkeypatch, arm_kind, unknown, dynamic):
        """Each sibling reads the incoming scalar; conditional writes remain unknown after the join."""
        rows = DynDim(name="ROWS", literal="ROWS", static_bound=4) if dynamic else 4
        meta = TensorMeta(shape=(rows, 8), dtype=DataType.FP32)
        seed = {"src": meta, "out": meta}

        def body(src, out, flag, other, runtime):
            rows = src.shape[0]
            if flag:
                rows = runtime
            else:
                view = pl.reshape(src, [rows, 8])
                out = _callsite_metadata_kernel(view, out)
            after = pl.reshape(src, [rows, 8])
            return out, after

        ast_analysis = importlib.import_module("pypto.jit._ast_analysis")
        get_func_def = ast_analysis.get_func_def
        func_def = deepcopy(get_func_def(body))
        branch = func_def.body[1]
        assert isinstance(branch, ast.If)
        if not unknown:
            branch.body = ast.parse("rows = 1").body
        if arm_kind == "elif":
            branch.orelse = [ast.If(test=ast.Name(id="other", ctx=ast.Load()), body=branch.orelse, orelse=[])]
        monkeypatch.setattr(
            ast_analysis, "get_func_def", lambda f: func_def if f is body else get_func_def(f)
        )
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert "after" not in metas
        dep_metas, _ = resolve_dep_call_metadata(
            _callsite_metadata_kernel, body, seed, {}, {}, discover_deps=_discover_dep_bindings
        )
        assert dep_metas["x"] == TensorMeta(shape=meta.shape, dtype=meta.dtype, layout=ir.TensorLayout.ND)

    @pytest.mark.parametrize("arm_kind", ["else", "elif"])
    def test_descriptor_reshape_alias_sibling_arm_crosses_dep(self, arm_kind):
        """A sibling branch can pass a view of the original dynamic dimension into a bare callee."""

        @jit.incore
        def consume(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            value = pl.load(x, [0, 0], [1, 8])
            pl.store(value, [0, 0], out)
            return out

        @jit
        def entry_else(
            src: pl.Tensor[[_PAGES_3033, 8], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
            flag: pl.Scalar[pl.BOOL],
            other: pl.Scalar[pl.BOOL],
        ):
            rows = pl.tensor.dim(src, 0)
            if flag:
                rows = 1
                result = pl.yield_(out)
            else:
                view = pl.reshape(src, [rows, 8])
                updated = consume(view, out)
                result = pl.yield_(updated)
            return result

        @jit
        def entry_elif(
            src: pl.Tensor[[_PAGES_3033, 8], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
            flag: pl.Scalar[pl.BOOL],
            other: pl.Scalar[pl.BOOL],
        ):
            rows = pl.tensor.dim(src, 0)
            if flag:
                rows = 1
                result = pl.yield_(out)
            else:
                if other:
                    view = pl.reshape(src, [rows, 8])
                    updated = consume(view, out)
                    inner_result = pl.yield_(updated)
                else:
                    inner_result = pl.yield_(out)
                result = pl.yield_(inner_result)
            return result

        entry = entry_else if arm_kind == "else" else entry_elif
        program = entry.lower()
        assert isinstance(program, ir.Program)
        kernel = program.get_function("consume")
        assert kernel is not None
        param_type = kernel.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert not isinstance(param_type.shape[0], ir.ConstInt)

    @pytest.mark.parametrize("dynamic", [False, True])
    def test_descriptor_reshape_alias_sibling_arm_keeps_loop_carry_unknown(self, dynamic):
        """Recovering a sibling's parent value cannot recover a loop-carried value from before the loop."""
        rows = DynDim(name="ROWS", literal="ROWS", static_bound=4) if dynamic else 4
        seed = {"src": TensorMeta(shape=(rows, 8), dtype=DataType.FP32)}

        def body(src, flag):
            rows = src.shape[0]
            for _ in pl.range(2):
                if flag:
                    rows = 1
                else:
                    view = pl.reshape(src, [rows, 8])
            return view

        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert "view" not in metas

    @pytest.mark.parametrize(
        "arm_kind,unknown,dynamic",
        [("else", False, False), ("else", True, True), ("elif", False, True), ("elif", True, False)],
    )
    def test_descriptor_reshape_tensor_sibling_reads_parent(self, monkeypatch, arm_kind, unknown, dynamic):
        """Tensor writes in one arm cannot change dimension arithmetic in its sibling."""
        rows = DynDim(name="ROWS", literal="ROWS", static_bound=4) if dynamic else 4
        seed = {
            "a": TensorMeta(shape=(rows, 8), dtype=DataType.FP32),
            "b": TensorMeta(shape=(7, 16), dtype=DataType.FP32),
            "out": TensorMeta(shape=(1, 4), dtype=DataType.FP32),
        }

        def body(a, b, out, flag, other, runtime):
            current = a
            if flag:
                current = runtime
            else:
                extent = current.shape[0] * 2
                view = pl.reshape(current, [extent, current.shape[1] // 2])
                out = _callsite_metadata_kernel(view, out)
            return out

        ast_analysis = importlib.import_module("pypto.jit._ast_analysis")
        get_func_def = ast_analysis.get_func_def
        func_def = deepcopy(get_func_def(body))
        branch = func_def.body[1]
        assert isinstance(branch, ast.If)
        if not unknown:
            branch.body = ast.parse("current = b").body
        if arm_kind == "elif":
            branch.orelse = [ast.If(test=ast.Name(id="other", ctx=ast.Load()), body=branch.orelse, orelse=[])]
        monkeypatch.setattr(
            ast_analysis, "get_func_def", lambda f: func_def if f is body else get_func_def(f)
        )
        # Historical collection includes the branch-local view; the call snapshot must match it.
        collected = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        dep_metas, _ = resolve_dep_call_metadata(
            _callsite_metadata_kernel, body, seed, {}, {}, discover_deps=_discover_dep_bindings
        )
        assert dep_metas["x"] == collected["view"]
        if dynamic:
            extent = dep_metas["x"].shape[0]
            assert isinstance(extent, DynDim) and extent.synthesized
            assert dep_metas["x"].shape[1] == 4
        else:
            assert dep_metas["x"].shape == (8, 4)

    def test_descriptor_reshape_tensor_sibling_crosses_dep(self):
        """A compatible tensor rebind preserves dynamic sibling views through specialization."""

        @jit.incore
        def consume(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            value = pl.load(x, [0, 0], [1, 4])
            pl.store(value, [0, 0], out)
            return out

        @jit
        def entry(
            a: pl.Tensor[[_PAGES_3033, 8], pl.FP32],
            b: pl.Tensor[[_PAGES_3033, 8], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 4], pl.FP32]],
            flag: pl.Scalar[pl.BOOL],
        ):
            current = a
            if flag:
                current = b
            else:
                extent = pl.tensor.dim(current, 0) * 2
                view = pl.reshape(current, [extent, pl.tensor.dim(current, 1) // 2])
                consume(view, out)
            return out

        program = entry.lower()
        assert isinstance(program, ir.Program)
        kernel = program.get_function("consume")
        assert kernel is not None
        param_type = kernel.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert not isinstance(param_type.shape[0], ir.ConstInt)
        assert isinstance(param_type.shape[1], ir.ConstInt) and param_type.shape[1].value == 4

    @pytest.mark.parametrize("unknown", [False, True])
    def test_descriptor_reshape_tensor_conditional_rebind_is_unknown_after_join(self, unknown):
        """A later dependency cannot use either arm's tensor as a guaranteed post-if value."""
        seed = {
            "a": TensorMeta(shape=(4, 8), dtype=DataType.FP32),
            "b": TensorMeta(shape=(7, 16), dtype=DataType.FP32),
            "out": TensorMeta(shape=(1, 4), dtype=DataType.FP32),
        }

        def known_body(a, b, out, flag, runtime):
            current = a
            if flag:
                current = b
            out = _callsite_metadata_kernel(current, out)
            return out

        def unknown_body(a, b, out, flag, runtime):
            current = a
            if flag:
                current = runtime
            out = _callsite_metadata_kernel(current, out)
            return out

        body = unknown_body if unknown else known_body
        dep_metas, _ = resolve_dep_call_metadata(
            _callsite_metadata_kernel, body, seed, {}, {}, discover_deps=_discover_dep_bindings
        )
        assert "x" not in dep_metas

    @pytest.mark.parametrize("dynamic", [False, True])
    def test_descriptor_reshape_tensor_conditional_return_is_unknown(self, dynamic):
        """Helper return inference uses live state instead of the historical branch collection."""
        rows = DynDim(name="ROWS", literal="ROWS", static_bound=4) if dynamic else 4
        other_rows = DynDim(name="OTHER_ROWS", literal="OTHER_ROWS", static_bound=7) if dynamic else 7
        seed = {
            "a": TensorMeta(shape=(rows, 8), dtype=DataType.FP32),
            "b": TensorMeta(shape=(other_rows, 16), dtype=DataType.FP32),
            "out": TensorMeta(shape=(1, 4), dtype=DataType.FP32),
        }

        @jit.inline
        def choose(a: pl.Tensor, b: pl.Tensor, flag: pl.Scalar[pl.BOOL]) -> pl.Tensor:
            current = a
            if flag:
                current = b
            return current

        def body(a, b, out, flag):
            selected = choose(a, b, flag)
            out = _callsite_metadata_kernel(selected, out)
            return out

        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert "selected" not in metas
        dep_metas, _ = resolve_dep_call_metadata(
            _callsite_metadata_kernel, body, seed, {}, {}, discover_deps=_discover_dep_bindings
        )
        assert "x" not in dep_metas

    @pytest.mark.parametrize(
        "replacement", ["equal", "different", "unknown", "nd_left", "nd_right", "layout_conflict"]
    )
    def test_descriptor_reshape_tensor_sibling_return_join(self, replacement):
        """Only matching tensor types from both definite arms may reach a helper consumer."""
        rows = DynDim(name="ROWS", literal="ROWS", static_bound=4)
        meta = TensorMeta(shape=(rows, 8), dtype=DataType.FP32)
        seed = {"a": meta, "out": meta}
        nd_meta = TensorMeta(shape=meta.shape, dtype=meta.dtype, layout=ir.TensorLayout.ND)
        if replacement in ("nd_left", "layout_conflict"):
            seed["a"] = nd_meta
        if replacement != "unknown":
            seed["b"] = {
                "different": TensorMeta((7, 16), DataType.FP32),
                "nd_right": nd_meta,
                "layout_conflict": TensorMeta(meta.shape, meta.dtype, layout=ir.TensorLayout.NZ),
            }.get(replacement, meta)

        @jit.inline
        def choose(a: pl.Tensor, b: pl.Tensor, flag: pl.Scalar[pl.BOOL]) -> pl.Tensor:
            if flag:
                current = a
            else:
                current = b
            return current

        def body(a, b, out, flag):
            selected = choose(a, b, flag)
            out = _callsite_metadata_kernel(selected, out)
            return out

        dep_metas, _ = resolve_dep_call_metadata(
            _callsite_metadata_kernel, body, seed, {}, {}, discover_deps=_discover_dep_bindings
        )
        if replacement in ("equal", "nd_left", "nd_right"):
            assert dep_metas["x"] == (meta if replacement == "equal" else nd_meta)
            assert dep_metas["x"].shape[0] is rows
        else:
            assert "x" not in dep_metas
        if replacement == "layout_conflict":
            # An omitted layout is the default ND, not a wildcard for NZ.
            seed["a"] = meta
            dep_metas, _ = resolve_dep_call_metadata(
                _callsite_metadata_kernel, body, seed, {}, {}, discover_deps=_discover_dep_bindings
            )
            assert "x" not in dep_metas

    @pytest.mark.parametrize(
        "arm,nested,dynamic",
        [("body", False, False), ("orelse", False, True), ("body", True, True), ("orelse", True, False)],
    )
    def test_descriptor_reshape_tensor_restores_if_incoming_binding(self, monkeypatch, arm, nested, dynamic):
        """A temporary unknown followed by the incoming tensor remains known after the join."""
        rows = DynDim(name="ROWS", literal="ROWS", static_bound=4) if dynamic else 4
        original = TensorMeta(shape=(rows, 8), dtype=DataType.FP32)
        seed = {
            "a": original,
            "b": TensorMeta(shape=(7, 16), dtype=DataType.FP32),
            "out": TensorMeta(shape=(1, 4), dtype=DataType.FP32),
        }

        def body(a, b, out, flag, inner, runtime):
            current = a
            if flag:
                pass
            out = _callsite_metadata_kernel(current, out)
            return out

        ast_analysis = importlib.import_module("pypto.jit._ast_analysis")
        get_func_def = ast_analysis.get_func_def
        func_def = deepcopy(get_func_def(body))
        conditional = func_def.body[1]
        assert isinstance(conditional, ast.If)
        statements = "current = runtime(current)\ncurrent = a"
        if nested:
            statements = (
                "current = b\nif inner:\n    current = runtime(current)\n    current = b\ncurrent = a"
            )
        setattr(conditional, arm, ast.parse(statements).body)
        monkeypatch.setattr(
            ast_analysis, "get_func_def", lambda func: func_def if func is body else get_func_def(func)
        )
        metas = extract_local_tensor_metas(
            body, seed_meta=seed, visible_only=True, discover_deps=_discover_dep_bindings
        )
        assert metas["current"] == original
        dep_metas, _ = resolve_dep_call_metadata(
            _callsite_metadata_kernel, body, seed, {}, {}, discover_deps=_discover_dep_bindings
        )
        assert dep_metas["x"] == original

    @pytest.mark.parametrize("different", ["shape", "synthesized_identity"])
    def test_descriptor_reshape_tensor_restored_arm_conflict_stays_unknown(self, different):
        """Restoring one arm's parent cannot hide a conflicting value from the other arm."""
        original = TensorMeta(shape=(4, 8), dtype=DataType.FP32)
        replacement = TensorMeta(shape=(7, 16), dtype=DataType.FP32)
        if different == "synthesized_identity":
            original = TensorMeta(
                shape=(DynDim(name="shared", literal="shared", static_bound=0, synthesized=True), 8),
                dtype=DataType.FP32,
            )
            replacement = TensorMeta(
                shape=(DynDim(name="shared", literal="shared", static_bound=0, synthesized=True), 8),
                dtype=DataType.FP32,
            )

        def body(a, b, out, flag, runtime):
            current = a
            if flag:
                current = runtime(current)
                current = a
            else:
                current = b
            out = _callsite_metadata_kernel(current, out)
            return out

        seed = {"a": original, "b": replacement, "out": original}
        metas = extract_local_tensor_metas(
            body, seed_meta=seed, visible_only=True, discover_deps=_discover_dep_bindings
        )
        assert "current" not in metas
        dep_metas, _ = resolve_dep_call_metadata(
            _callsite_metadata_kernel, body, seed, {}, {}, discover_deps=_discover_dep_bindings
        )
        assert "x" not in dep_metas

    @pytest.mark.parametrize("dynamic", [False, True])
    def test_descriptor_reshape_tensor_nested_arms_restore_outer_incoming(self, dynamic):
        """Two inner arms can restore a tensor invalidated earlier in their outer arm."""
        rows = DynDim(name="ROWS", literal="ROWS", static_bound=4) if dynamic else 4
        original = TensorMeta(shape=(rows, 8), dtype=DataType.FP32)

        def body(a, out, outer, inner, runtime):
            current = a
            if outer:
                current = runtime(current)
                if inner:
                    current = a
                else:
                    current = a
            out = _callsite_metadata_kernel(current, out)
            return out

        seed = {"a": original, "out": original}
        metas = extract_local_tensor_metas(
            body, seed_meta=seed, visible_only=True, discover_deps=_discover_dep_bindings
        )
        assert metas["current"] == original
        dep_metas, _ = resolve_dep_call_metadata(
            _callsite_metadata_kernel, body, seed, {}, {}, discover_deps=_discover_dep_bindings
        )
        assert dep_metas["x"] == original

    @pytest.mark.parametrize("arm", ["body", "orelse"])
    def test_descriptor_reshape_tensor_restored_if_crosses_dep(self, monkeypatch, arm):
        """A real tensor operation followed by restoring the input still reaches its consumer."""

        @jit.incore
        def consume(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            value = pl.load(x, [0, 0], [1, 8])
            pl.store(value, [0, 0], out)
            return out

        def body(
            src: pl.Tensor[[4, 8], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
            flag: pl.Scalar[pl.BOOL],
        ):
            current = src
            if flag:
                pass
            return consume(current, out)

        get_lines = inspect.getsourcelines
        lines, line = get_lines(body)
        func_def = ast.parse(textwrap.dedent("".join(lines))).body[0]
        assert isinstance(func_def, ast.FunctionDef)
        conditional = func_def.body[1]
        assert isinstance(conditional, ast.If)
        setattr(conditional, arm, ast.parse("current = pl.tensor.add(current, current)\ncurrent = src").body)
        source = ast.unparse(func_def) + "\n"
        monkeypatch.setattr(
            inspect,
            "getsourcelines",
            lambda func: (source.splitlines(keepends=True), line) if func is body else get_lines(func),
        )
        assert isinstance(jit(body).lower(), ir.Program)

    @pytest.mark.parametrize("inner_definite", [False, True])
    def test_descriptor_reshape_tensor_nested_join_requires_definite_arms(self, inner_definite):
        """Equal outer-arm shapes merge only if the nested arm actually establishes its tensor."""
        original = TensorMeta(shape=(7, 16), dtype=DataType.FP32)
        replacement = TensorMeta(shape=(4, 8), dtype=DataType.FP32)
        seed = {"a": replacement, "b": original, "out": replacement}

        def partial_body(a, b, out, outer, inner):
            current = b
            if outer:
                if inner:
                    current = a
            else:
                current = a
            out = _callsite_metadata_kernel(current, out)
            return out

        def definite_body(a, b, out, outer, inner):
            current = b
            if outer:
                if inner:
                    current = a
                else:
                    current = a
            else:
                current = a
            out = _callsite_metadata_kernel(current, out)
            return out

        body = definite_body if inner_definite else partial_body
        metas = extract_local_tensor_metas(
            body, seed_meta=seed, visible_only=True, discover_deps=_discover_dep_bindings
        )
        dep_metas, _ = resolve_dep_call_metadata(
            _callsite_metadata_kernel, body, seed, {}, {}, discover_deps=_discover_dep_bindings
        )
        if inner_definite:
            assert metas["current"] == replacement
            assert dep_metas["x"] == replacement
        else:
            assert "current" not in metas
            assert "x" not in dep_metas

    @pytest.mark.parametrize("dynamic", [False, True])
    def test_descriptor_reshape_tensor_same_shape_out_loop_remains_visible(self, dynamic):
        """Same-shaped Out updates retain their incoming metadata through loops and branches."""
        rows = DynDim(name="ROWS", literal="ROWS", static_bound=4) if dynamic else 4
        meta = TensorMeta(shape=(rows, 8), dtype=DataType.FP32)

        def body(src, out, flag):
            for _ in pl.range(2):
                out = _callsite_metadata_kernel(src, out)
            if flag:
                out = _callsite_metadata_kernel(src, out)
            else:
                out = _callsite_metadata_kernel(src, out)
            return out

        metas = extract_local_tensor_metas(
            body,
            seed_meta={"src": meta, "out": meta},
            visible_only=True,
            discover_deps=_discover_dep_bindings,
        )
        assert metas["out"] == meta

    def test_descriptor_reshape_tensor_distinct_allocations_return_independent_symbols(self):
        """A helper's two runtime allocations remain independent when its tuple reaches the caller."""

        @jit.inline
        def allocate(first: pl.Scalar[pl.INDEX], second: pl.Scalar[pl.INDEX]):
            view = pl.create_tensor([first], dtype=pl.FP32)
            saved = view
            view = pl.create_tensor([second], dtype=pl.FP32)
            return saved, view

        def body(first, second):
            saved, rebound = allocate(first, second)
            return saved, rebound

        metas = extract_local_tensor_metas(body, visible_only=True, discover_deps=_discover_dep_bindings)
        saved = self._leading_dim(metas, "saved")
        rebound = self._leading_dim(metas, "rebound")
        assert isinstance(saved, DynDim) and saved.synthesized
        assert isinstance(rebound, DynDim) and rebound.synthesized
        assert saved.name != rebound.name

    def test_descriptor_reshape_alias_runtime_param_shadows_dim_symbol(self):
        """A runtime scalar parameter must not fall back to a same-named DynDim anchor."""
        seed = {
            "src": TensorMeta(
                shape=(DynDim(name="PAGES", literal="PAGES", static_bound=4), 128), dtype=DataType.INT8
            )
        }

        def body(src, PAGES):
            total = PAGES * 128
            flat = pl.reshape(src, [total])
            return flat

        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert "flat" not in metas

    def test_descriptor_reshape_static_dim_call_folds_axis_arithmetic(self):
        """A statically folded negative axis resolves to its actual static extent."""

        def body(src):
            rows = pl.tensor.dim(axis=_DIM_ROW_AXIS_3033 - 2, tensor=src)
            flat = pl.reshape(src, [rows * 128])
            return flat

        seed = {"src": TensorMeta((4, 128), DataType.FP32)}
        assert extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)[
            "flat"
        ].shape == (512,)

    @pytest.mark.parametrize(
        "expression",
        [
            "pl.tensor.dim(axis=0)",
            "pl.tensor.dim(src)",
            "pl.tensor.dim(src, 0, 1)",
            "pl.tensor.dim(src, 0, tensor=src)",
            "pl.tensor.dim(src, 0, axis=0)",
            "pl.tensor.dim(src, axis=0, axis=1)",
            "pl.tensor.dim(tensor=src, tensor=src, axis=0)",
            "pl.tensor.dim(src, 0, unknown=0)",
            "pl.tensor.dim(*src)",
            "pl.tensor.dim(**runtime)",
            "pl.tensor.dim(src, runtime)",
            "pl.tensor.dim(src, True)",
            "pl.tensor.dim(src, 0.0)",
            "pl.tensor.dim(src, 2)",
            "pl.tensor.dim(src, -3)",
            "src.shape[-2]",
            "src.shape[+0]",
            "src.shape[_DIM_ROW_AXIS_3033]",
            "src.shape[True]",
        ],
    )
    def test_descriptor_reshape_invalid_dim_expressions_stay_unknown(self, monkeypatch, expression):
        """Malformed dimension calls, invalid axes and unsupported subscripts cannot create metadata."""

        def body(src, runtime):
            rows = pl.tensor.dim(src, 0)
            total = rows * 128
            flat = pl.reshape(src, [total])
            return flat

        ast_analysis = importlib.import_module("pypto.jit._ast_analysis")
        get_func_def = ast_analysis.get_func_def
        func_def = deepcopy(get_func_def(body))
        assignment = func_def.body[0]
        assert isinstance(assignment, ast.Assign)
        assignment.value = ast.parse(expression, mode="eval").body
        monkeypatch.setattr(
            ast_analysis, "get_func_def", lambda func: func_def if func is body else get_func_def(func)
        )
        seed = {
            "src": TensorMeta(
                shape=(DynDim(name="ROWS", literal="ROWS", static_bound=4), 128), dtype=DataType.FP32
            )
        }
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert "flat" not in metas

    @pytest.mark.parametrize(
        "expression",
        [
            "pl.tensor.dim(src, axis=0)",
            "pl.tensor.dim(axis=0, tensor=src)",
            "pl.tensor.dim(src, _DIM_ROW_AXIS_3033)",
            "pl.tensor.dim(src, axis=-2)",
            "pl.tensor.dim(src, axis=_DIM_ROW_AXIS_3033 - 2)",
        ],
    )
    def test_descriptor_reshape_dim_spellings_cross_dep(self, monkeypatch, expression):
        """Supported dimension calls survive parsing and verified lowering across a helper."""

        def body(
            src: pl.Tensor[[_PAGES_3033, 128], pl.INT8],
            out: pl.Out[pl.Tensor[[1, 128], pl.INT8]],
        ):
            rows = pl.tensor.dim(src, 0)
            total = rows * 128
            flat = pl.reshape(src, [total])
            keys = pl.reshape(flat, [rows, 128])
            return _copy_key_3033(keys, out)

        get_lines = inspect.getsourcelines
        lines, line = get_lines(body)
        source = "".join(lines).replace("pl.tensor.dim(src, 0)", expression)
        monkeypatch.setattr(
            inspect,
            "getsourcelines",
            lambda f: (source.splitlines(keepends=True), line) if f is body else get_lines(f),
        )
        assert isinstance(jit(body).lower(), ir.Program)

    @pytest.mark.parametrize("symbol_name", ["PAGES", "N"])
    def test_descriptor_reshape_alias_local_dynamic_declaration(self, symbol_name):
        """A recognized dynamic declaration retains its anchor, including an aliased literal name."""
        dim = DynDim(name=symbol_name, literal="PAGES", static_bound=4)
        seed = {"src": TensorMeta(shape=(dim, 128), dtype=DataType.INT8)}

        def body(src):
            PAGES = pl.dynamic("PAGES")
            src.bind_dynamic(0, PAGES)
            total = PAGES * 128
            direct = pl.reshape(src, [PAGES, 128])
            aliased = pl.reshape(src, [total])
            return direct, aliased

        def renamed_body(src):
            N = pl.dynamic("PAGES")
            src.bind_dynamic(0, N)
            total = N * 128
            direct = pl.reshape(src, [N, 128])
            aliased = pl.reshape(src, [total])
            return direct, aliased

        metas = extract_local_tensor_metas(
            body if symbol_name == "PAGES" else renamed_body,
            seed_meta=seed,
            discover_deps=_discover_dep_bindings,
        )
        assert metas["direct"].shape == (dim, 128)
        derived = self._leading_dim(metas, "aliased")
        assert isinstance(derived, DynDim) and derived.synthesized
        assert derived.name != dim.name

    def test_descriptor_reshape_alias_augassign_invalidates_target_only(self):
        """Unsupported augmented assignment invalidates its target but preserves earlier snapshots."""
        seed = {
            "src": TensorMeta(
                shape=(DynDim(name="PAGES", literal="PAGES", static_bound=4), 128), dtype=DataType.INT8
            )
        }

        def body(src, runtime):
            extent = 16
            saved = extent
            extent += 16
            augmented = pl.reshape(src, [extent])
            snapshot = pl.reshape(src, [saved, 32])
            total = pl.tensor.dim(src, 0) * 128
            prior = total
            total -= 64
            derived = pl.reshape(src, [total])
            prior_view = pl.reshape(src, [prior])
            total += runtime
            unknown = pl.reshape(src, [total])
            return augmented, snapshot, derived, prior_view, unknown

        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert not {"augmented", "derived", "unknown"} & metas.keys()
        assert metas["snapshot"].shape == (16, 32)
        dim = self._leading_dim(metas, "prior_view")
        assert isinstance(dim, DynDim) and dim.synthesized

    def test_descriptor_reshape_alias_augassign_remains_unsupported(self):
        """Metadata tracking does not expand the DSL's accepted assignment syntax."""

        @jit
        def entry(
            src: pl.Tensor[[_PAGES_3033, 128], pl.INT8],
            out: pl.Out[pl.Tensor[[1, 128], pl.INT8]],
        ):
            total = pl.tensor.dim(src, 0) * 128
            total += 1
            return out

        with pytest.raises(UnsupportedFeatureError, match="Augmented assignment is not supported"):
            entry.specialize()

    @pytest.mark.parametrize("same_line", [False, True])
    def test_descriptor_reshape_alias_rebinding_has_distinct_descriptors(self, monkeypatch, same_line):
        """Separate reshape sites keep separate symbols when a name is rebound and aliased."""

        def body(a, b):
            ma = a.shape[0]
            mb = b.shape[0]
            first_extent = ma * 2
            second_extent = mb * 4
            view = pl.reshape(a, [first_extent, 64])
            saved = view
            view = pl.reshape(b, [second_extent, 32])
            return saved, view

        if same_line:
            ast_analysis = importlib.import_module("pypto.jit._ast_analysis")
            get_func_def = ast_analysis.get_func_def
            func_def = deepcopy(get_func_def(body))
            calls = [
                node
                for node in ast.walk(func_def)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "reshape"
            ]
            calls[1].lineno = calls[0].lineno
            calls[1].col_offset = calls[0].col_offset + 80
            monkeypatch.setattr(
                ast_analysis, "get_func_def", lambda func: func_def if func is body else get_func_def(func)
            )

        shapes = []
        for a_rows, b_rows in ((3, 5), (9, 7)):
            seed = {
                "a": TensorMeta(
                    shape=(DynDim(name="A", literal="A", static_bound=a_rows), 128), dtype=DataType.FP32
                ),
                "b": TensorMeta(
                    shape=(DynDim(name="B", literal="B", static_bound=b_rows), 128), dtype=DataType.FP32
                ),
            }
            metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
            saved_dim = self._leading_dim(metas, "saved")
            current_dim = self._leading_dim(metas, "view")
            assert isinstance(saved_dim, DynDim) and saved_dim.synthesized
            assert isinstance(current_dim, DynDim) and current_dim.synthesized
            assert saved_dim.name != current_dim.name
            assert metas["saved"].shape[1] == 64
            assert metas["view"].shape[1] == 32
            shapes.append((metas["saved"].shape, metas["view"].shape))
        assert shapes[0] == shapes[1]

    def test_descriptor_reshape_alias_rebinding_crosses_dep(self):
        """Two differently sized views of separate inputs do not share a callee dimension symbol."""

        @jit.incore
        def consume(first: pl.Tensor, second: pl.Tensor, out: pl.Out[pl.Tensor]):
            left = pl.load(first, [0, 0], [1, 32])
            right = pl.load(second, [0, 0], [1, 32])
            pl.store(pl.add(left, right), [0, 0], out)
            return out

        @jit
        def entry(
            a: pl.Tensor[[_PAGES_3033, 128], pl.FP32],
            b: pl.Tensor[[_M_SLICE, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 32], pl.FP32]],
        ):
            ma = pl.tensor.dim(a, 0)
            mb = pl.tensor.dim(b, 0)
            first_extent = ma * 2
            second_extent = mb * 4
            view = pl.reshape(a, [first_extent, 64])
            saved = view
            view = pl.reshape(b, [second_extent, 32])
            return consume(saved, view, out)

        program = entry.lower()
        kernel = program.get_function("consume")
        assert kernel is not None
        first_type = kernel.params[0].type
        second_type = kernel.params[1].type
        assert isinstance(first_type, ir.TensorType)
        assert isinstance(second_type, ir.TensorType)
        assert not first_type.shape[0].same_as(second_type.shape[0])

    def test_descriptor_reshape_alias_rebinding_preserves_direct_dim_identity(self):
        """Direct dimensions keep their source identity instead of acquiring a site symbol."""
        a_dim = DynDim(name="A", literal="A", static_bound=3)
        b_dim = DynDim(name="B", literal="B", static_bound=5)
        seed = {
            "a": TensorMeta(shape=(a_dim, 128), dtype=DataType.FP32),
            "b": TensorMeta(shape=(b_dim, 128), dtype=DataType.FP32),
        }

        def body(a, b):
            ma = a.shape[0]
            mb = b.shape[0]
            view = pl.reshape(a, [ma, 128])
            saved = view
            view = pl.reshape(b, [mb, 128])
            return saved, view

        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["saved"].shape[0] is a_dim
        assert metas["view"].shape[0] is b_dim

    def test_descriptor_reshape_alias_helper_return_calls_specialize_independently(self):
        """Chained aliases share a result; separate calls get independent symbols.

        Lowering helper-local dimension expressions additionally requires IR
        return-type closure support, outside this metadata regression.
        """

        @jit.inline
        def flatten(src: pl.Tensor):
            value = pl.reshape(src, [pl.tensor.dim(src, 0) * 128])
            return value

        @jit.incore
        def consume(first: pl.Tensor, second: pl.Tensor, out: pl.Out[pl.Tensor]):
            left = pl.load(first, [0], [128])
            right = pl.load(second, [0], [128])
            pl.store(pl.add(left, right), [0], out)
            return out

        @jit
        def entry(
            a: pl.Tensor[[_PAGES_3033, 128], pl.FP32],
            b: pl.Tensor[[_M_SLICE, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[128], pl.FP32]],
        ):
            first = first_alias = flatten(a)
            second = flatten(b)
            if pl.tensor.dim(a, 0) > 0:
                selected = first
            else:
                selected = first_alias
            return consume(selected, second, out)

        program = entry.specialize()
        kernel = program.get_function("consume")
        assert kernel is not None
        first_type = kernel.params[0].type
        second_type = kernel.params[1].type
        assert isinstance(first_type, ir.TensorType)
        assert isinstance(second_type, ir.TensorType)
        first_dim = first_type.shape[0]
        second_dim = second_type.shape[0]
        assert isinstance(first_dim, ir.Var)
        assert isinstance(second_dim, ir.Var)
        assert first_dim.name_hint != second_dim.name_hint
        assert not first_dim.same_as(second_dim)

    def test_descriptor_reshape_alias_helper_tuple_return_preserves_shared_alias(self):
        """A tuple's shared view stays shared within a call, while separate calls get fresh symbols."""

        @jit.inline
        def flatten_pair(src: pl.Tensor):
            pages = pl.tensor.dim(src, 0)
            value = pl.reshape(src, [pages * 128])
            return value, value

        def body(a, b):
            first, first_alias = flatten_pair(a)
            second, second_alias = flatten_pair(b)
            return first, first_alias, second, second_alias

        shapes = []
        for a_rows, b_rows in ((3, 5), (9, 7)):
            seed = {
                "a": TensorMeta(
                    shape=(DynDim(name="A", literal="A", static_bound=a_rows), 128), dtype=DataType.FP32
                ),
                "b": TensorMeta(
                    shape=(DynDim(name="B", literal="B", static_bound=b_rows), 128), dtype=DataType.FP32
                ),
            }
            metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
            assert metas["first"].shape == metas["first_alias"].shape
            assert metas["second"].shape == metas["second_alias"].shape
            first = self._leading_dim(metas, "first")
            second = self._leading_dim(metas, "second")
            assert isinstance(first, DynDim) and first.synthesized
            assert isinstance(second, DynDim) and second.synthesized
            assert first.name != second.name
            shapes.append((metas["first"].shape, metas["second"].shape))
        assert shapes[0] == shapes[1]

    @pytest.mark.parametrize("synthesized", [False, True])
    def test_descriptor_reshape_alias_helper_return_preserves_input_identity(self, synthesized):
        """Returning an input preserves its dimensions even when the caller synthesized them."""

        @jit.inline
        def passthrough(src: pl.Tensor):
            return src

        def body(a, b):
            first = passthrough(a)
            second = passthrough(b)
            return first, second

        a_dim = DynDim(name="A", literal="A", static_bound=3, synthesized=synthesized)
        b_dim = DynDim(name="B", literal="B", static_bound=5, synthesized=synthesized)
        seed = {
            "a": TensorMeta(shape=(a_dim, 128), dtype=DataType.FP32),
            "b": TensorMeta(shape=(b_dim, 128), dtype=DataType.FP32),
        }
        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert metas["first"].shape[0] is a_dim
        assert metas["second"].shape[0] is b_dim

    def test_descriptor_reshape_dep_uses_argument_descriptor(self):
        """The callee reads the reshaped argument's leading dim, rather than src's pages."""

        @jit.incore
        def copy_rows(keys: pl.Tensor, out: pl.Out[pl.Tensor]):
            rows, cols = keys.shape
            for row in pl.range(rows):
                value = pl.load(keys, [row, 0], [1, cols])
                pl.store(value, [row, 0], out)
            return out

        @jit
        def entry(
            src: pl.Tensor[[_PAGES_3033, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[_PAGES_3033, 128], pl.FP32]],
        ):
            pages = pl.tensor.dim(src, 0)
            keys = pl.reshape(src, [pages * 2, 64])
            result = pl.reshape(out, [pages * 2, 64])
            return copy_rows(keys, result)

        program = entry.specialize()
        kernel_source = program.get_function("copy_rows").as_python()
        assert "pl.tensor.dim(keys, 0)" in kernel_source
        assert "pl.tensor.dim(src, 0)" not in kernel_source

    def test_reshape_unknown_runtime_extent_stays_strict(self):
        """A descriptor dim mixed with a scalar or tensor.read must not invent metadata."""
        seed = {
            "src": TensorMeta(
                shape=(DynDim(name="M", literal="M", static_bound=16), 128), dtype=DataType.FP32
            ),
            "cfg": TensorMeta(shape=(1,), dtype=DataType.INT64),
        }

        def body(src, cfg, n):
            rows = src.shape[0]
            count = pl.tensor.read(cfg, [0])
            scalar_sized = pl.reshape(src, [rows * n])
            read_sized = pl.reshape(src, [rows + count])
            direct_read_sized = pl.reshape(src, [rows + pl.tensor.read(cfg, [0])])
            modulo_sized = pl.reshape(src, [rows % 2])
            true_div_sized = pl.reshape(src, [rows / 2])
            power_sized = pl.reshape(src, [rows**2])
            return scalar_sized, read_sized, direct_read_sized, modulo_sized, true_div_sized, power_sized

        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert (
            not {
                "scalar_sized",
                "read_sized",
                "direct_read_sized",
                "modulo_sized",
                "true_div_sized",
                "power_sized",
            }
            & metas.keys()
        )

    def test_reshape_zero_divisor_stays_strict(self):
        """Known invalid divisions remain unresolved even inside otherwise valid arithmetic."""
        seed = {
            "src": TensorMeta(
                shape=(DynDim(name="M", literal="M", static_bound=16), 128), dtype=DataType.FP32
            )
        }

        def body(src):
            rows = pl.tensor.dim(src, 0)
            direct = pl.reshape(src, [rows // 0])
            folded = pl.reshape(src, [rows // (2 - 2)])
            nested = pl.reshape(src, [(rows // 0) + 1])
            nested_denominator = pl.reshape(src, [rows // (1 // 0)])
            return direct, folded, nested, nested_denominator

        metas = extract_local_tensor_metas(body, seed_meta=seed, discover_deps=_discover_dep_bindings)
        assert not {"direct", "folded", "nested", "nested_denominator"} & metas.keys()

    def test_static_reshape_size_mismatch_still_rejected(self):
        """Metadata propagation leaves the IR's element-count validation in force."""

        @jit
        def invalid(src: pl.Tensor[[2, 128], pl.FP32]):
            return pl.reshape(src, [3, 128])

        with pytest.raises(InvalidOperationError, match="cannot reshape tensor of size 256"):
            invalid.specialize()

    def test_synthesized_dim_reaches_dep_parameter(self):
        """The dep parameter is typed instead of raising "missing inferred
        tensor metadata" — issue #2450's failure."""
        seed = {
            "cfg": TensorMeta(shape=(1,), dtype=DataType.INT64),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        metas, _ = resolve_dep_call_metadata(
            _synth_dim_kernel, _runtime_create_then_dep, seed, {}, {}, discover_deps=_discover_dep_bindings
        )
        dim = self._leading_dim(metas, "t")
        assert isinstance(dim, DynDim)
        assert dim.synthesized

    def test_dep_declared_symbol_replaces_synthesized_placeholder(self):
        """A dep that declares the dim ``pl.dynamic`` gets its own symbol back.

        The dep's body may reference that name (``for i in pl.range(NR)``);
        keeping the invented one would leave the reference unbound.
        """
        seed = {
            "cfg": TensorMeta(shape=(1,), dtype=DataType.INT64),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        dep_dyn_map = {"t": {0: DynDim(name="NR", literal="NR", static_bound=0)}}
        metas, _ = resolve_dep_call_metadata(
            _synth_dim_kernel,
            _runtime_create_then_dep,
            seed,
            {},
            dep_dyn_map,
            discover_deps=_discover_dep_bindings,
        )
        dim = self._leading_dim(metas, "t")
        assert isinstance(dim, DynDim)
        assert dim.name == "NR"
        assert not dim.synthesized

    def test_runtime_sized_local_lowers_end_to_end(self):
        """The headline claim: a runtime-sized local crossing a dep boundary
        reaches a real ``ir.Program``, with the synthesized symbol declared via
        ``pl.dynamic`` and left dynamic in the kernel's parameter type."""
        torch = pytest.importorskip("torch")

        @jit.incore
        def scale_incore(t: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            tile = pl.load(t, [0, 0], [64, 64])
            pl.store(pl.add(tile, tile), [0, 0], out)
            return out

        @jit
        def runtime_sized_entry(cfg: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            n = pl.tensor.read(cfg, [0])
            tmp = pl.create_tensor([n, 64], dtype=pl.FP32)
            return scale_incore(tmp, out)

        cfg = torch.zeros(1, dtype=torch.int64)
        out = torch.zeros(64, 64, dtype=torch.float32)
        program = runtime_sized_entry.lower(cfg, out)
        assert isinstance(program, ir.Program)

        # The kernel parameter kept a dynamic leading dim rather than being
        # frozen to the placeholder extent.
        kernel = program.get_function("scale_incore")
        assert kernel is not None
        param_type = kernel.params[0].type
        assert isinstance(param_type, ir.TensorType)
        leading = param_type.shape[0]
        assert not isinstance(leading, ir.ConstInt), f"leading dim froze to {leading}"

    def test_synthesized_symbol_is_qualified_by_owning_function(self):
        """Two functions each holding a runtime-sized local of the same name get
        distinct symbols, so a dumped program never shows one name on two
        unrelated tensors."""
        a = synthesized_dyn_dim("host_entry", "tmp", 0)
        b = synthesized_dyn_dim("mid", "tmp", 0)
        assert a.name != b.name
        assert a.name.startswith(_SYNTHESIZED_DYN_PREFIX)
        assert b.name.startswith(_SYNTHESIZED_DYN_PREFIX)

    def test_caller_derived_dyn_dim_outranks_dep_declaration(self):
        """A DynDim the caller actually derived is NOT replaced — its symbol is
        the one bound at the call site (pre-existing precedence)."""
        seed = {
            "src": TensorMeta(shape=(DynDim(name="M", literal="M", static_bound=32), 8), dtype=DataType.FP32),
            "out": TensorMeta(shape=(16, 8), dtype=DataType.FP32),
        }
        dep_dyn_map = {"t": {0: DynDim(name="NR", literal="NR", static_bound=0)}}
        metas, _ = resolve_dep_call_metadata(
            _synth_dim_kernel,
            _dyn_alias_create_then_dep,
            seed,
            {},
            dep_dyn_map,
            discover_deps=_discover_dep_bindings,
        )
        dim = self._leading_dim(metas, "t")
        assert isinstance(dim, DynDim)
        assert dim.name == "M"
        assert not dim.synthesized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
