# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check literal algorithms against the existing canonical source writer."""

import ast
import importlib
import random
import sys
from pathlib import Path

import pytest
from pypto.jit import _source, specializer


@pytest.fixture
def design(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "benchmarks"))
    instance = importlib.import_module("hbg_csa_literal_design").DesignCandidates()
    try:
        yield instance
    finally:
        instance.restore()


@pytest.mark.parametrize(
    "stage", ["source_only", "callsite_only", "global_text", "request_text", "direct_ast", "direct_value"]
)
def test_literal_algorithms_match_original(design, stage):
    rng = random.Random(7)
    values = [None, True, False, 0, 1, -1, 2**256, "", "\0\n\\'\"", "\ud800", "中文"]
    values += [rng.getrandbits(256) - 2**255 for _ in range(100)]
    values += ["".join(chr(rng.randrange(0x110000)) for _ in range(20)) for _ in range(100)]
    expected = [specializer.constant_source(value) for value in values]
    design.install(stage)
    with _source.capture_namespaces():
        assert [design.render(value) for value in values] == expected
        assert [design.render(value) for value in values] == expected


def test_type_checks_do_not_invoke_metaclass_equality(design):
    class EqualToEveryType(type):
        __hash__ = type.__hash__

        def __eq__(cls, other):
            raise AssertionError("Exact-type guard must use identity")

    class Opaque(metaclass=EqualToEveryType):
        pass

    value = Opaque()
    expected = specializer.constant_source(value)
    for stage in design.stages:
        design.install(stage)
        assert not importlib.import_module("hbg_csa_literal_design").plain_literal(value)
        assert not importlib.import_module("hbg_csa_cost_audit").eligible(value)
        assert design.render(value) == expected


@pytest.mark.parametrize("stage", ["direct_ast", "direct_value", "request_text"])
def test_fallback_values_remain_live(design, stage):
    mutable = [1, 2]
    values = [0.0, -0.0, float("inf"), float("nan"), mutable, (1,), b"unsupported"]
    expected = [specializer.constant_source(value) for value in values]
    design.install(stage)
    with _source.capture_namespaces():
        assert [design.render(value) for value in values] == expected
        mutable[0] = 3
        assert design.render(mutable) == "[3, 2]"


def test_request_cache_lifetime_and_binding_changes(design):
    design.install("request_text")
    assert _source._ACTIVE_NAMESPACES.get() is None
    with _source.capture_namespaces():
        snapshot = _source._ACTIVE_NAMESPACES.get()
        assert snapshot is not None
        assert design.free_name("n", {"n": 16}) == "16"
        assert design.free_name("n", {"n": 32}) == "32"
        assert design.free_name("n", {"n": None}) == "None"
        assert design.free_name("n", {}) is None
        assert design.free_name("n", {"n": specializer._UNBOUND}) is None
        assert len(snapshot.values) == 3
    assert _source._ACTIVE_NAMESPACES.get() is None
    with _source.capture_namespaces():
        next_snapshot = _source._ACTIVE_NAMESPACES.get()
        assert next_snapshot is not None
        assert not next_snapshot.values


@pytest.mark.skipif(not hasattr(sys, "set_int_max_str_digits"), reason="No digit-limit API")
@pytest.mark.parametrize("stage", ["direct_ast", "direct_value"])
def test_direct_rendering_preserves_integer_limit_errors(design, stage):
    previous = sys.get_int_max_str_digits()
    value = 10**1000
    design.install(stage)
    try:
        sys.set_int_max_str_digits(0)
        assert design.render(value) == ast.unparse(ast.Constant(value))
        sys.set_int_max_str_digits(640)
        with pytest.raises(ValueError):
            design.render(value)
        with pytest.raises(ValueError):
            ast.unparse(ast.Constant(value))
    finally:
        sys.set_int_max_str_digits(previous)


def test_ablation_patch_boundaries_and_restore(design):
    slots = [
        (owner, name, getattr(owner, name))
        for owner, name in (
            (design.specializer, "constant_source"),
            (design.specializer, "free_name_source"),
            (design.binding, "free_name_source"),
            (design.decorator, "free_name_source"),
        )
    ]
    for stage in design.stages:
        design.install(stage)
        control = stage in ("current_a", "current_b")
        for owner, name, original in slots:
            changed = not control and stage != (
                "source_only" if name == "constant_source" else "callsite_only"
            )
            assert (getattr(owner, name) is not original) == changed
        design.restore()
        assert all(getattr(owner, name) is original for owner, name, original in slots)


def test_runner_factory_restored_after_failure(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "benchmarks"))
    module = importlib.import_module("hbg_csa_literal_design")
    original = module.audit.Candidates

    def fail():
        assert module.audit.Candidates is module.DesignCandidates
        raise RuntimeError("diagnostic failure")

    monkeypatch.setattr(module.audit, "main", fail)
    with pytest.raises(RuntimeError, match="diagnostic failure"):
        module.main()
    assert module.audit.Candidates is original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
