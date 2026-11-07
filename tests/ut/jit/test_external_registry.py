# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Provider discovery, immutable handoff and compatibility boundary tests."""

import gc
import hashlib
import importlib
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace

import pytest
from pypto.jit.external.compatibility import legacy_builder, make_request, request_from_context
from pypto.jit.external.contracts import (
    ABIParameter,
    ABIValue,
    Access,
    CompileTarget,
    Completion,
    Core,
    ExecutionRequirements,
    KernelContract,
    NativeABI,
    ParameterSpec,
    PreparationIdentity,
    PreparedKernel,
    SourceArtifact,
    SourceFile,
    TensorEffect,
    TensorSpec,
)
from pypto.jit.external.protocols import CapturedInputs
from pypto.jit.external.registry import KernelRef, ProviderRegistry
from pypto.jit.external.validation import ContractError
from pypto.jit.specializer import DynDim, SpecializeContext, TensorMeta
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import TensorLayout


class Kernel:
    def __init__(self, source: bytes | bytearray = b"// source"):
        self.source = source


class FakeProvider:
    def __init__(self, name="fake", complete=True):
        self.name = name
        self.complete = complete
        self.captures = 0
        self.compilations = 0

    def capture(self, kernel, request):
        self.captures += 1
        payload = bytes(kernel.source)
        return CapturedInputs(
            payload,
            PreparationIdentity(self.name, "1", "1", hashlib.sha256(payload).hexdigest(), self.complete),
        )

    def prepare(self, captured, request):
        self.compilations += 1
        tensors = [(i, p) for i, p in enumerate(request.parameters) if isinstance(p.type, TensorSpec)]
        abi = tuple(
            ABIParameter(i, ABIValue.DESCRIPTOR, 32, 8)
            if isinstance(p.type, TensorSpec)
            else ABIParameter(i, ABIValue.SCALAR, 8, 8)
            for i, p in enumerate(request.parameters)
        )
        return PreparedKernel(
            request,
            KernelContract(request.parameters, tuple(TensorEffect(i, Access.READ) for i, _ in tensors)),
            SourceArtifact(
                (SourceFile("entry.cpp", captured.identity.kernel_digest, captured.payload),),
                "entry.cpp",
                "kernel_entry",
            ),
            NativeABI("fake.descriptor", 1, abi),
            ExecutionRequirements((Core.AIV,), 1, Completion.ENTRY_RETURN),
            captured.identity,
        )


def _discover(registry: ProviderRegistry, kernel: object) -> KernelRef:
    ref = registry.discover(kernel)
    assert ref is not None
    return ref


@pytest.fixture
def kernel_request():
    return make_request(
        (ParameterSpec("x", TensorSpec((32,), DataType.FP32)),), CompileTarget("a5", "test-compiler", "1")
    )


@pytest.fixture
def setup_provider():
    registry = ProviderRegistry()
    provider = FakeProvider()
    registry.register("fake", matches=lambda obj: isinstance(obj, Kernel), load=lambda: provider)
    return registry, provider


def test_discovery_defers_loading_capture_and_compilation(kernel_request):
    registry = ProviderRegistry()
    loaded = []
    provider = FakeProvider()

    def load():
        loaded.append(True)
        return provider

    registry.register("fake", matches=lambda obj: isinstance(obj, Kernel), load=load)
    assert registry.discover(object()) is None
    ref = _discover(registry, Kernel())
    assert ref is not None and not loaded
    assert registry.discover(ref) is ref
    assert provider.captures == provider.compilations == 0
    captured = ref.capture(kernel_request)
    assert loaded == [True] and provider.captures == 1 and provider.compilations == 0
    key = captured.preparation_key
    assert key is not None and len(key) == 64
    assert captured.prepare().request == kernel_request
    assert provider.compilations == 1


def test_duplicate_matches_require_explicit_binding(kernel_request):
    registry = ProviderRegistry()
    for name in ("first", "second"):
        registry.register(name, matches=lambda _: True, load=lambda name=name: FakeProvider(name))
    with pytest.raises(ContractError, match="ambiguous providers.*first.*second"):
        _discover(registry, Kernel())
    assert (
        registry.bind(Kernel(), provider="second").capture(kernel_request).prepare().identity.provider
        == "second"
    )
    with pytest.raises(ContractError, match="already registered"):
        registry.register("first", matches=lambda _: True, load=FakeProvider)
    with pytest.raises(ContractError, match="unknown provider"):
        registry.bind(Kernel(), provider="missing")


def test_reference_owns_kernel_but_registry_does_not(kernel_request, setup_provider):
    registry, _ = setup_provider
    kernel = Kernel()
    weak = weakref.ref(kernel)
    ref = _discover(registry, kernel)
    del kernel
    gc.collect()
    assert weak() is not None
    captured = ref.capture(kernel_request)
    del ref
    gc.collect()
    assert weak() is None
    # Captured bytes remain usable after the original provider input disappears.
    assert captured.prepare().artifact.files[0].content == b"// source"


def test_unhashable_kernel_and_registration_replacement(kernel_request, setup_provider):
    registry, provider = setup_provider
    ref = registry.bind({"unhashable": True}, provider="fake")
    assert ref.provider_name == "fake"
    original = registry.bind(Kernel(), provider="fake")
    registry.unregister("fake")
    assert registry.discover(Kernel()) is None
    registry.register("fake", matches=lambda _: True, load=lambda: FakeProvider(complete=False))
    assert original.capture(kernel_request).preparation_key is not None
    assert provider.captures == 1
    assert _discover(registry, Kernel()).capture(kernel_request).preparation_key is None


def test_snapshot_and_request_are_isolated(kernel_request, setup_provider):
    registry, _ = setup_provider
    source = bytearray(b"original")
    kernel = Kernel(source)
    captured = _discover(registry, kernel).capture(kernel_request)
    source[:] = b"modified"
    assert captured.prepare().artifact.files[0].content == b"original"
    with pytest.raises(FrozenInstanceError):
        setattr(captured.inputs, "payload", b"mutated")
    params = list(kernel_request.parameters)
    options = {"fast_math": True}
    copied = make_request(params, kernel_request.target, options=options)
    params.clear()
    options["fast_math"] = False
    assert copied.parameters == kernel_request.parameters and copied.options == (("fast_math", True),)


@pytest.mark.parametrize("parameters", [set(), frozenset(), {}])
def test_request_builder_rejects_unordered_parameters(kernel_request, parameters):
    with pytest.raises(ContractError, match="ordered parameter sequence"):
        make_request(parameters, kernel_request.target)


def test_incomplete_identity_never_reuses_preparation(kernel_request):
    registry = ProviderRegistry()
    provider = FakeProvider(complete=False)
    registry.register("fake", matches=lambda _: True, load=lambda: provider)
    captured = _discover(registry, Kernel()).capture(kernel_request)
    assert captured.preparation_key is None
    captured.prepare()
    captured.prepare()
    assert provider.compilations == 2


@pytest.mark.parametrize(
    "change",
    [
        lambda capture: replace(capture, payload=bytearray(capture.payload)),
        lambda capture: replace(capture, payload=b"mismatched"),
        lambda capture: replace(capture, identity=replace(capture.identity, provider="other")),
    ],
)
def test_invalid_capture_rejected_before_preparation(kernel_request, change):
    class BadProvider(FakeProvider):
        def capture(self, kernel, request):
            return change(super().capture(kernel, request))

    registry = ProviderRegistry()
    provider = BadProvider()
    registry.register("fake", matches=lambda _: True, load=lambda: provider)
    with pytest.raises(ContractError):
        _discover(registry, Kernel()).capture(kernel_request)
    assert provider.compilations == 0


@pytest.mark.parametrize(
    "change",
    [
        lambda prepared: replace(
            prepared, request=replace(prepared.request, target=CompileTarget("a2", "cc", "1"))
        ),
        lambda prepared: replace(prepared, request=replace(prepared.request, options=(("flag", 1),))),
        lambda prepared: replace(prepared, identity=replace(prepared.identity, adapter_version="2")),
    ],
)
def test_provider_cannot_change_captured_request_or_identity(kernel_request, change):
    class BadProvider(FakeProvider):
        def prepare(self, captured, request):
            return change(super().prepare(captured, request))

    registry = ProviderRegistry()
    registry.register("fake", matches=lambda _: True, load=BadProvider)
    # True == 1 in Python must not hide a changed compile option.
    captured = _discover(registry, Kernel()).capture(replace(kernel_request, options=(("flag", True),)))
    with pytest.raises(ContractError, match="changed the captured"):
        captured.prepare()


def test_invalid_request_does_not_load_provider(kernel_request):
    registry = ProviderRegistry()
    loaded = []

    def load():
        loaded.append(True)
        return FakeProvider()

    registry.register("fake", matches=lambda _: True, load=load)
    with pytest.raises(ContractError):
        _discover(registry, Kernel()).capture(
            replace(kernel_request, parameters=list(kernel_request.parameters))
        )
    assert loaded == []


def test_concurrent_capture_loads_provider_once(kernel_request):
    registry = ProviderRegistry()
    loaded = []

    def load():
        loaded.append(True)
        return FakeProvider()

    registry.register("fake", matches=lambda _: True, load=load)
    ref = _discover(registry, Kernel())
    with ThreadPoolExecutor(max_workers=4) as pool:
        captures = list(pool.map(lambda _: ref.capture(kernel_request), range(8)))
    assert loaded == [True]
    assert len({id(capture) for capture in captures}) == 8
    assert len({capture.preparation_key for capture in captures}) == 1


def _context():
    return SpecializeContext(
        "kernel",
        "def kernel(x, y, count, tile): pass",
        "extern",
        None,
        ["x", "y", "count", "tile"],
        {
            "x": TensorMeta((DynDim("M", "M", 32), 16), DataType.FP32, TensorLayout.ND),
            "y": TensorMeta((DynDim("M", "M", 64), 16), DataType.FP32, TensorLayout.ND),
        },
        {"count": DataType.INT64},
        constexpr_values={"tile": "16"},
    )


def test_context_conversion_preserves_symbols_and_isolates_mutation(kernel_request):
    context = _context()
    normalized = request_from_context(context, kernel_request.target)
    context.tensor_meta["x"].shape = (999, 999)
    context.constexpr_values["tile"] = "32"
    first, second = normalized.parameters[0].type, normalized.parameters[1].type
    assert isinstance(first, TensorSpec) and isinstance(second, TensorSpec)
    assert first.shape == second.shape == ("d0", 16)
    assert first.strides == ()
    assert first.offset is None
    assert normalized.constants == (("tile", 16),)


def test_inexact_extent_does_not_become_static_constraint(kernel_request):
    context = _context()
    context.tensor_meta["x"] = TensorMeta(
        (32, 16), DataType.FP32, TensorLayout.ND, inexact_axes=frozenset({0})
    )
    spec = request_from_context(context, kernel_request.target).parameters[0].type
    assert isinstance(spec, TensorSpec)
    assert spec.shape == ("extent_0_0", 16)


@pytest.mark.parametrize(
    "change",
    [
        lambda context: context.constexpr_values.update(tile="__import__('os')"),
        lambda context: context.constexpr_values.update(tile="[1, 2]"),
        lambda context: context.scalar_dtypes.update(x=DataType.INT64),
        lambda context: context.param_names.append("missing"),
        lambda context: setattr(context.tensor_meta["x"], "layout", None),
    ],
)
def test_context_rejects_unresolved_or_mutable_inputs(kernel_request, change):
    context = _context()
    change(context)
    with pytest.raises(ContractError):
        request_from_context(context, kernel_request.target)


def test_legacy_builder_uses_immutable_boundary(kernel_request, setup_provider):
    registry, provider = setup_provider
    lowered = []
    options = {"fast_math": True}
    build = legacy_builder(
        _discover(registry, Kernel()),
        kernel_request.target,
        lambda prepared: lowered.append(prepared) or prepared.artifact,
        options=options,
    )
    options["fast_math"] = False
    assert provider.captures == provider.compilations == 0
    result = build(_context())
    assert result.files[0].content == b"// source"
    assert lowered[0].request.options == (("fast_math", True),)
    assert provider.captures == provider.compilations == 1


def test_no_optional_dsl_import_during_discovery(monkeypatch):
    original = importlib.import_module

    def guarded(name, *args, **kwargs):
        assert not name.startswith(("cannbotdsl", "tilelang"))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", guarded)
    registry = ProviderRegistry()

    def load():
        importlib.import_module("cannbotdsl")
        return FakeProvider("optional")

    registry.register("optional", matches=lambda obj: False, load=load)
    assert registry.discover(object()) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
