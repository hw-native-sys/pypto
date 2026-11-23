# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Provider preparation through ordinary JIT, including all native topologies."""

import gc
import inspect
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pypto.language as pl
import pytest
from pypto import backend, passes
from pypto.backend.cannbotdsl_jit import CANNBotDSLBinding
from pypto.jit._source import capture_namespaces
from pypto.jit.external.compatibility import request_from_context
from pypto.jit.external.contracts import (
    Access,
    CompileTarget,
    Coverage,
    KernelContract,
    TensorEffect,
    WritebackBinding,
)
from pypto.jit.external.discovery import JITDeclaration, discover_jit
from pypto.jit.external.providers.cannbotdsl import CANNBotDSLProvider
from pypto.jit.external.validation import ContractError
from pypto.jit.specializer import TensorMeta, build_specialize_context
from pypto.runtime import ExecutionMode, RunConfig


class _Binding:
    def __init__(self, core, *, require_completion=True):
        self.core = core
        self.require_completion = require_completion
        self.requests = []

    def prepare(self, context, platform, runtime, output_dir):
        ctx = context
        assert platform == "a5"
        assert runtime == passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER
        request = request_from_context(ctx, CompileTarget("a5", "ccec", "test"))
        self.requests.append(request)
        path = Path(output_dir) / "callback.cpp"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"// export {len(self.requests)}\n")
        ctx.external_core_type = self.core
        ctx.external_aic_source = str(path) if self.core != "aiv" else None
        ctx.external_aiv_source = str(path) if self.core != "aic" else None
        ctx.external_dual_aiv_dispatch = self.core == "mixed"
        ctx.external_binary_policy = "a5-text-only" if self.require_completion else "ordinary"
        ctx.external_preflight_symbol = "test_native_preflight"
        ctx.external_preflight_source = "inline void test_native_preflight(const CoreTaskArgs&) {}"
        ctx.external_contract = KernelContract(
            request.parameters,
            (TensorEffect(0, Access.READ), TensorEffect(1, Access.WRITE, Coverage.FULL)),
            writebacks=(WritebackBinding(1),),
        )


class _Native:
    __hash__ = None

    def __init__(self, binding):
        self.binding = binding
        self.configured = False

    def __call__(self, x, out):
        raise AssertionError("discovery must not execute the native kernel")

    def __eq__(self, other):
        raise AssertionError("discovery must compare kernel identity")


@pytest.fixture
def native_api(monkeypatch):
    """Exercise the real adapter against the optional package's public protocol."""
    cb = ModuleType("cannbotdsl")
    cb.inspect_kernel = lambda kernel: (  # type: ignore[attr-defined]
        SimpleNamespace(signature=inspect.signature(kernel.__call__), configured=kernel.configured)
        if isinstance(kernel, _Native)
        else None
    )
    monkeypatch.setitem(sys.modules, "cannbotdsl", cb)
    monkeypatch.setattr(CANNBotDSLBinding, "prepare", lambda self, *args: self.kernel.binding.prepare(*args))


def _automatic_entry(native):
    alias = native

    @pl.jit
    def entry(
        x: pl.Tensor[[32], pl.FP32], tmp: pl.Tensor[[32], pl.FP32], out: pl.Out[pl.Tensor[[32], pl.FP32]]
    ):
        native(x, tmp)
        alias(tmp, out=out)
        return out

    return entry


@pytest.mark.parametrize("core", ["aic", "aiv", "mixed"])
def test_binding_prepares_owned_variants_without_jit_cache(monkeypatch, native_api, core):
    backend.reset_for_testing()
    monkeypatch.setattr("pypto.jit.decorator._ptoas_available", lambda: False)
    binding = _Binding(core)
    entry = _automatic_entry(_Native(binding))
    config = RunConfig(backend_type=backend.BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD)
    first = entry.compile(config=config)
    second = entry.compile(config=config)
    assert first is not second
    assert len(binding.requests) == 2
    assert not entry._cache
    with pytest.raises(RuntimeError, match="cannot request recompilation"):
        entry._resolve_compiled((), {"config": config}, allow_signature_mode=True, _lookup_only=True)
    # A second preparation cannot replace the first program's owned export.
    first_sources = [f for f in first.program.functions.values() if "external_source" in f.attrs]
    assert len(first_sources) == (2 if core == "mixed" else 1)
    for function in first_sources:
        assert function.attrs["external_binary_policy"] == "a5-text-only"
        assert Path(function.attrs["external_source"]).read_text() == "// export 1\n"
    output = Path(first.output_dir)
    config_source = (output / "kernel_config.py").read_text()
    assert config_source.count("a5-text-only") == len(first_sources)
    orchestration = "\n".join(path.read_text() for path in output.rglob("*.cpp"))
    assert orchestration.count(".set_allow_early_resolve(false)") >= 2
    if core == "mixed":
        assert "MixedKernels" in orchestration


@pytest.mark.parametrize("core", ["aic", "aiv", "mixed"])
@pytest.mark.parametrize("core_num,sync_start", [(2, False), (1, True)])
def test_binding_preflight_observes_explicit_launch(monkeypatch, native_api, core, core_num, sync_start):
    backend.reset_for_testing()
    monkeypatch.setattr("pypto.jit.decorator._ptoas_available", lambda: False)

    native = _Native(_Binding(core))

    @pl.jit
    def entry(x: pl.Tensor[[32], pl.FP32], out: pl.Out[pl.Tensor[[32], pl.FP32]]):
        pl.spmd_submit(native, x, out, core_num=core_num, sync_start=sync_start)
        return out

    config = RunConfig(backend_type=backend.BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD)
    compiled = entry.compile(config=config)
    source = "\n".join(path.read_text() for path in Path(compiled.output_dir).rglob("*.cpp"))
    # Native launch validation must see the explicit settings, not the default
    # one-block launch. These requests violate the real binding's contract.
    checker = source.index("test_native_preflight(params_t0);")
    assert source.index(f".launch_spec.set_block_num({core_num})") < checker
    if sync_start:
        assert source.index(".launch_spec.set_require_sync_start(true)") < checker
    assert checker < source.index("rt_submit_")


@pytest.mark.parametrize("core", ["aic", "aiv", "mixed"])
@pytest.mark.parametrize("producer_style", ["submit", "at"])
@pytest.mark.parametrize("require_completion", [False, True])
def test_binding_completion_overrides_explicit_hints(
    monkeypatch, native_api, core, producer_style, require_completion
):
    backend.reset_for_testing()
    monkeypatch.setattr("pypto.jit.decorator._ptoas_available", lambda: False)

    native = _Native(_Binding(core, require_completion=require_completion))

    @pl.jit.incore
    def producer(x: pl.Tensor, out: pl.Out[pl.Tensor]):
        return pl.store(pl.add(pl.load(x, [0, 0], [16, 16]), 1.0), [0, 0], out)

    @pl.jit
    def entry(x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
        tmp = pl.create_tensor([16, 16], dtype=pl.FP32)
        if producer_style == "submit":
            tmp, done = pl.submit(producer, x, tmp, allow_early_resolve=True)
        else:
            with pl.at(level=pl.Level.CORE_GROUP, allow_early_resolve=True):
                tmp = pl.store(pl.add(pl.load(x, [0, 0], [16, 16]), 1.0), [0, 0], tmp)
        pl.submit(native, tmp, out, allow_early_resolve=True)
        return out

    config = RunConfig(backend_type=backend.BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD)
    compiled = entry.compile(config=config)
    source = "\n".join(path.read_text() for path in Path(compiled.output_dir).rglob("*.cpp"))
    # Both the ordinary producer and native task opt in. Policy-bearing programs
    # must keep both tasks completion-based; ordinary programs retain the hints.
    assert source.count(".set_allow_early_resolve(true)") == (0 if require_completion else 2)
    assert source.count(".set_allow_early_resolve(false)") == (2 if require_completion else 0)


def test_discovered_kernel_requires_target(native_api):
    with pytest.raises(ValueError, match="explicit target"):
        _automatic_entry(_Native(_Binding("aiv"))).specialize()


def _context(shape):
    def native(x: pl.Tensor, out: pl.Tensor): ...

    return build_specialize_context(
        native,
        "native",
        "extern",
        None,
        {name: TensorMeta(shape, pl.FP32) for name in ("x", "out")},
        {},
        [],
    )


@pytest.mark.parametrize("shape", [(3,), (0,), ("n",)])
def test_cannbot_binding_rejects_unproven_cache_line_views(tmp_path, shape):
    binding = CANNBotDSLBinding(CANNBotDSLProvider(), object())
    with pytest.raises(ContractError, match="positive static|whole 64-byte"):
        binding.prepare(_context(shape), "a5", passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER, str(tmp_path))


@pytest.mark.parametrize(
    "platform,runtime",
    [
        ("a5sim", passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER),
        ("a2a3", passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER),
        ("a5", passes.RuntimeKind.HOST_BUILD_GRAPH),
    ],
)
def test_cannbot_binding_rejects_unsupported_target(tmp_path, platform, runtime):
    binding = CANNBotDSLBinding(CANNBotDSLProvider(), object())
    with pytest.raises(ContractError, match="requires platform a5"):
        binding.prepare(_context((32,)), platform, runtime, str(tmp_path))


def test_automatic_discovery_is_request_owned_and_identity_based(native_api):
    first = _Native(_Binding("aic"))
    second = _Native(_Binding("mixed"))
    with capture_namespaces():
        wrapper = discover_jit(first)
        assert wrapper is not None
        assert discover_jit(first) is wrapper
        assert discover_jit(second) is not wrapper
    with capture_namespaces():
        assert discover_jit(first) is not wrapper
    assert discover_jit(object()) is None


def test_automatic_discovery_does_not_import_optional_dsl(monkeypatch):
    monkeypatch.delitem(sys.modules, "cannbotdsl", raising=False)
    assert discover_jit(object()) is None
    assert "cannbotdsl" not in sys.modules


def test_automatic_discovery_rejects_configured_kernel(native_api):
    native = _Native(_Binding("aiv"))
    native.configured = True
    with pytest.raises(ContractError, match="unconfigured kernel"):
        discover_jit(native)


def test_automatic_discovery_refreshes_rebound_native(monkeypatch, native_api):
    backend.reset_for_testing()
    monkeypatch.setattr("pypto.jit.decorator._ptoas_available", lambda: False)
    first = _Binding("aic")
    second = _Binding("mixed")
    native = _Native(first)
    previous = weakref.ref(native)

    @pl.jit
    def entry(x: pl.Tensor[[32], pl.FP32], out: pl.Out[pl.Tensor[[32], pl.FP32]]):
        native(x, out)
        return out

    config = RunConfig(backend_type=backend.BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD)
    entry.compile(config=config)
    native = _Native(second)
    gc.collect()
    assert previous() is None
    compiled = entry.compile(config=config)
    assert len(first.requests) == len(second.requests) == 1
    sources = [f for f in compiled.program.functions.values() if "external_source" in f.attrs]
    assert len(sources) == 2


def test_discovery_releases_native_and_signature_with_wrapper(monkeypatch):
    class Provider:
        def discover(self, kernel):
            if not isinstance(kernel, _Native):
                return None
            return JITDeclaration(
                inspect.signature(kernel.__call__), CANNBotDSLBinding(CANNBotDSLProvider(), kernel)
            )

    monkeypatch.setattr("pypto.jit.external.discovery._providers", lambda: (Provider(),))
    native = _Native(_Binding("aiv"))
    reference = weakref.ref(native)
    wrapper = discover_jit(native)
    assert wrapper is not None
    filename = inspect.getsourcefile(wrapper._func)
    assert filename is not None
    path = Path(filename)
    del native
    gc.collect()
    assert reference() is not None
    assert path.exists()
    del wrapper
    gc.collect()
    assert reference() is None
    assert not path.exists()


def test_discovery_rejects_ambiguous_providers(monkeypatch):
    class Provider:
        def discover(self, kernel):
            return JITDeclaration(inspect.Signature(), _Binding("aiv"))

    monkeypatch.setattr("pypto.jit.external.discovery._providers", lambda: (Provider(), Provider()))
    with pytest.raises(ContractError, match="ambiguous providers"):
        discover_jit(object())


@pytest.mark.parametrize("signature", ["(x, out=None)", "(x, *, out)", "(*args)", "(self, out)"])
def test_discovery_rejects_unsupported_signatures(monkeypatch, signature):
    namespace = {}
    exec(f"def native{signature}: pass", namespace)

    class Provider:
        def discover(self, kernel):
            return JITDeclaration(inspect.signature(namespace["native"]), _Binding("aiv"))

    monkeypatch.setattr("pypto.jit.external.discovery._providers", lambda: (Provider(),))
    with pytest.raises(ContractError, match="explicit positional parameters"):
        discover_jit(object())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
