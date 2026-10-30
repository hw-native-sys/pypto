# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cross-request signature reuse and publication consistency."""

import inspect
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from pypto.jit import _source


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setattr(_source, "_SIGNATURES", {})


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="signature reuse excludes lazy annotations")
def test_request_and_cross_request_reuse():
    def f(x: int, /, y: str) -> int:
        return x

    first = _source.function_signature(f)
    assert _source.function_signature(f) is first
    with _source.capture_namespaces():
        assert _source.function_signature(f) is first
        f.__annotations__["x"] = str
        assert _source.function_signature(f) is first
    assert _source.function_signature(f).parameters["x"].annotation is str
    f.__annotations__["return"] = str
    assert _source.function_signature(f).return_annotation is str
    f.__annotations__["extra"] = int
    second = _source.function_signature(f)
    del f.__annotations__["extra"]
    assert _source.function_signature(f) is not second
    with pytest.raises(TypeError):
        _source.function_signature(f).bind(x=1, y="a")


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="signature reuse excludes lazy annotations")
def test_annotation_contents_remain_live_without_equality():
    class Annotation:
        __hash__ = None
        value = 0

        def __eq__(self, other):
            raise AssertionError("user equality must not run")

    annotation = Annotation()

    def f(x):
        pass

    setattr(f, "__annotations__", {"x": annotation})
    first = _source.function_signature(f)
    annotation.value = 2
    assert _source.function_signature(f) is first
    assert first.parameters["x"].annotation.value == 2
    replacement = Annotation()
    f.__annotations__["x"] = replacement
    assert _source.function_signature(f).parameters["x"].annotation is replacement


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="signature reuse excludes lazy annotations")
def test_defaults_added_removed_and_keyword_mutation():
    def f(x):
        pass

    first = _source.function_signature(f)
    default = []
    f.__defaults__ = (default,)
    bound = _source.function_signature(f).bind()
    bound.apply_defaults()
    assert bound.arguments["x"] is default
    default.append(1)
    assert bound.arguments["x"] == [1]
    f.__defaults__ = None
    assert _source.function_signature(f) is first

    def keyword(*, x=1):
        pass

    assert _source.function_signature(keyword).parameters["x"].default == 1
    assert keyword.__kwdefaults__ is not None
    keyword.__kwdefaults__["x"] = 2
    assert _source.function_signature(keyword).parameters["x"].default == 2
    assert keyword not in _source._SIGNATURES


@pytest.mark.parametrize("attribute", ["__wrapped__", "__signature__", "__text_signature__"])
def test_special_protocol_fallback(attribute):
    def f(x):
        pass

    def replacement(y):
        pass

    _source.function_signature(f)
    values = {
        "__wrapped__": replacement,
        "__signature__": inspect.signature(replacement),
        "__text_signature__": "(y)",
    }
    setattr(f, attribute, values[attribute])
    assert str(_source.function_signature(f)) == str(inspect.signature(f))
    delattr(f, attribute)
    assert list(_source.function_signature(f).parameters) == ["x"]


@pytest.mark.parametrize("mutation", ["annotation", "code", "signature", "parameter"])
def test_aba_does_not_publish_mismatched_signature(monkeypatch, mutation):
    def f(x: int):
        pass

    def replacement(y, *, z):
        pass

    class CustomSignature(inspect.Signature):
        def bind(self, *args, **kwargs):
            raise AssertionError("custom bind leaked")

    class CustomParameter(inspect.Parameter):
        pass

    original = inspect.signature

    def interrupted(func):
        if mutation == "annotation":
            before = f.__annotations__
            f.__annotations__ = {"x": str}
            result = original(f)
            f.__annotations__ = before
        elif mutation == "code":
            before = f.__code__
            f.__code__ = replacement.__code__
            result = original(f)
            f.__code__ = before
        else:
            signature_type = CustomSignature if mutation == "signature" else inspect.Signature
            parameter_type = CustomParameter if mutation == "parameter" else inspect.Parameter
            setattr(
                f,
                "__signature__",
                signature_type(
                    [parameter_type("x", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int)]
                ),
            )
            result = original(f)
            delattr(f, "__signature__")
        return result

    monkeypatch.setattr(inspect, "signature", interrupted)
    _source.function_signature(f)
    assert f not in _source._SIGNATURES
    monkeypatch.setattr(inspect, "signature", original)
    assert _source.function_signature(f).parameters["x"].annotation is int
    assert _source.function_signature(f).bind(1).arguments == {"x": 1}


def test_failure_does_not_publish(monkeypatch):
    def f(x):
        pass

    original = inspect.signature

    def failure(func):
        raise ValueError("injected failure")

    monkeypatch.setattr(inspect, "signature", failure)
    with pytest.raises(ValueError, match="injected failure"):
        _source.function_signature(f)
    assert f not in _source._SIGNATURES
    monkeypatch.setattr(inspect, "signature", original)
    assert list(_source.function_signature(f).parameters) == ["x"]


def test_concurrent_readers_and_bounded_cache():
    def f(x):
        pass

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: str(_source.function_signature(f)), range(100)))
    assert set(results) == {"(x)"}


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="Python 3.14 uses inspect fallback")
def test_cache_clears_at_capacity_and_rebuilds_correct_signatures():
    # Clear-all is a bounded policy, not LRU. Eviction must only lose reuse.
    _source._SIGNATURES.clear()
    functions = [lambda x: x for _ in range(513)]
    for func in functions[:512]:
        assert str(_source.function_signature(func)) == "(x)"
    assert len(_source._SIGNATURES) == 512
    _source.function_signature(functions[-1])
    assert set(_source._SIGNATURES) == {functions[-1]}
    for func in functions[:512]:
        assert _source.function_signature(func) == inspect.signature(func)
    assert set(_source._SIGNATURES) == {functions[511]}


def test_lazy_annotation_python_falls_back(monkeypatch):
    def f(x):
        pass

    original = inspect.signature
    calls = []

    def counted(func):
        calls.append(func)
        return original(func)

    monkeypatch.setattr(_source, "sys", SimpleNamespace(version_info=(3, 14)))
    monkeypatch.setattr(inspect, "signature", counted)
    _source.function_signature(f)
    _source.function_signature(f)
    assert calls == [f, f]
    assert f not in _source._SIGNATURES


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="signature reuse excludes lazy annotations")
def test_code_replacement_after_hit():
    def f(x):
        pass

    def replacement(y, /, z):
        pass

    _source.function_signature(f)
    f.__code__ = replacement.__code__
    result = _source.function_signature(f)
    assert list(result.parameters) == ["y", "z"]
    assert result.parameters["y"].kind is inspect.Parameter.POSITIONAL_ONLY


def test_unsupported_shapes_and_annotation_containers():
    class Annotations(dict):
        pass

    class Key(str):
        pass

    def f(x):
        pass

    for annotations in (Annotations(x=int), {Key("x"): int}):
        setattr(f, "__annotations__", annotations)
        assert str(_source.function_signature(f)) == str(inspect.signature(f))
        assert f not in _source._SIGNATURES

    def variadic(*args, **kwargs):
        pass

    class Callable:
        def __call__(self, x):
            pass

    for func in (variadic, Callable(), len):
        assert str(_source.function_signature(func)) == str(inspect.signature(func))
        assert func not in _source._SIGNATURES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
