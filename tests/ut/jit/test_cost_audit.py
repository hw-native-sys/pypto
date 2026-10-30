# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Differential checks for diagnostic-only CSA counterfactuals."""

import ast
import importlib.util
import sys
from collections import UserDict
from pathlib import Path

import pytest
from pypto.jit import _call_binding, specializer

spec = importlib.util.spec_from_file_location(
    "_csa_cost_audit", Path(__file__).parents[2] / "benchmarks/hbg_csa_cost_audit.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def candidate():
    experiment = module.Candidates()
    try:
        yield experiment
    finally:
        experiment.restore()


def test_text_values_and_type_separation(candidate):
    values = (
        None,
        True,
        False,
        0,
        1,
        -1,
        2**64 - 1,
        2**64,
        "a'\n\\b",
        "",
        0.0,
        -0.0,
        float("nan"),
        float("inf"),
        [1, 2],
        (1, 2),
    )
    for value in values:
        assert candidate.render(value) == specializer.constant_source(value)
        assert candidate.render(value) == specializer.constant_source(value)
    assert candidate.render(True) != candidate.render(1)
    assert candidate.render(False) != candidate.render(0)
    data = [1]
    assert candidate.render(data) == "[1]"
    data[0] = 2
    assert candidate.render(data) == "[2]"


def test_name_reads_and_unbound_stay_current(candidate):
    class Namespace(UserDict):
        def __init__(self):
            super().__init__({"value": 16})
            self.events = []

        def __contains__(self, key):
            self.events.append(("contains", key))
            return key in self.data

        def __getitem__(self, key):
            self.events.append(("getitem", key))
            return self.data[key]

    namespace = Namespace()
    for value in (16, 32, None, specializer._UNBOUND):
        namespace.data["value"] = value
        namespace.events.clear()
        expected = specializer.free_name_source("value", namespace)
        events = namespace.events.copy()
        namespace.events.clear()
        assert candidate.free_name("value", namespace) == expected
        assert namespace.events == events
    assert candidate.free_name("missing", namespace) is None


def test_text_failure_not_cached(monkeypatch):
    original = specializer.constant_source
    calls = []

    def renderer(value):
        calls.append(value)
        if len(calls) == 1:
            raise ValueError("diagnostic rendering failure")
        return original(value)

    monkeypatch.setattr(specializer, "constant_source", renderer)
    candidate = module.Candidates()
    with pytest.raises(ValueError, match="diagnostic rendering failure"):
        candidate.render(16)
    assert candidate.literal_text.cache_info().currsize == 0
    assert candidate.render(16) == "16"
    assert candidate.render(16) == "16"
    assert len(calls) == 2


def test_subclasses_and_float_keep_renderer(monkeypatch, candidate):
    class DynamicInt(int):
        pass

    values = [DynamicInt(1), 1.0, -0.0]
    for value in values:
        assert not module.eligible(value)
    original = specializer._render_free_value
    calls = []

    def counted(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(specializer, "_render_free_value", counted)
    for value in values:
        candidate.render(value)
        candidate.render(value)
    assert len(calls) == 6


@pytest.mark.parametrize("expression", ["16", "-16", "[16, 32]", "cfg.value", "x + 1", "missing", "BLOCK"])
def test_callsite_text_keeps_current_folding(candidate, expression):
    class Config:
        value = 16

    arg = ast.parse(expression, mode="eval").body
    original = _call_binding._fold_call_site_constant
    for stage in ("callsite_text", "combined"):
        for value in (16, 32):
            Config.value = value
            namespace = {"cfg": Config, "x": value, "BLOCK": 99}
            for forwarded in ({}, {"BLOCK": str(value)}):
                for locals_ in (frozenset(), frozenset({"x", "BLOCK"})):
                    expected = original(arg, forwarded, namespace, locals_)
                    candidate.install(stage)
                    try:
                        assert candidate.fold(arg, forwarded, namespace, locals_) == expected
                    finally:
                        candidate.restore()
        candidate.restore()


def test_callsite_keeps_namespace_exceptions(candidate):
    class Namespace(UserDict):
        def items(self):
            raise RuntimeError("namespace failure")

    candidate.install("callsite_text")
    with pytest.raises(RuntimeError, match="namespace failure"):
        candidate.fold(ast.Constant(16), {}, Namespace(), frozenset())


@pytest.mark.skipif(not hasattr(sys, "set_int_max_str_digits"), reason="No integer digit-limit API")
def test_digit_limit_changes_are_observed(candidate):
    original_limit = sys.get_int_max_str_digits()
    arg = ast.Constant(10**1000)
    candidate.install("callsite_text")
    try:
        sys.set_int_max_str_digits(0)
        assert candidate.fold(arg, {}, {}, frozenset()) is not None
        assert candidate.render(arg.value) is not None
        sys.set_int_max_str_digits(640)
        assert candidate.fold(arg, {}, {}, frozenset()) is None
        with pytest.raises(ValueError):
            candidate.render(arg.value)
    finally:
        sys.set_int_max_str_digits(original_limit)


def test_callsite_recursion_limit_changes_select_a_miss(candidate):
    original_limit = sys.getrecursionlimit()
    arg = ast.parse("-" * 40 + "1", mode="eval").body
    candidate.install("callsite_text")
    try:
        sys.setrecursionlimit(1000)
        candidate.fold(arg, {}, {}, frozenset())
        misses = candidate.call_text.cache_info().misses
        sys.setrecursionlimit(100)
        candidate.fold(arg, {}, {}, frozenset())
        assert candidate.call_text.cache_info().misses == misses + 1
    finally:
        sys.setrecursionlimit(original_limit)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
