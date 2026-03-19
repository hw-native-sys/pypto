# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for Level and Role enums (Step 01) and Function level/role fields (Step 02)."""

import pypto.language as pl
import pytest
from pypto.pypto_core import ir

# ─── Step 01: Level enum ──────────────────────────────────────────────────────


def test_level_primary_values_distinct():
    """Primary level values are all distinct."""
    primary = [
        ir.Level.AIV,
        ir.Level.AIC,
        ir.Level.CORE_GROUP,
        ir.Level.CHIP_DIE,
        ir.Level.CHIP,
        ir.Level.HOST,
        ir.Level.CLUSTER_0,
        ir.Level.CLUSTER_1,
        ir.Level.CLUSTER_2,
        ir.Level.GLOBAL,
    ]
    assert len(set(primary)) == 10


def test_level_aliases():
    """Aliases resolve to the same Linqu level number as their primary."""
    ll = ir.level_to_linqu_level
    assert ll(ir.Level.L2CACHE) == ll(ir.Level.CHIP_DIE)
    assert ll(ir.Level.PROCESSOR) == ll(ir.Level.CHIP)
    assert ll(ir.Level.UMA) == ll(ir.Level.CHIP)
    assert ll(ir.Level.NODE) == ll(ir.Level.HOST)
    assert ll(ir.Level.POD) == ll(ir.Level.CLUSTER_0)
    assert ll(ir.Level.CLOS1) == ll(ir.Level.CLUSTER_1)
    assert ll(ir.Level.CLOS2) == ll(ir.Level.CLUSTER_2)


def test_level_to_linqu_level():
    """level_to_linqu_level maps correctly."""
    assert ir.level_to_linqu_level(ir.Level.AIV) == 0
    assert ir.level_to_linqu_level(ir.Level.AIC) == 0
    assert ir.level_to_linqu_level(ir.Level.CORE_GROUP) == 0
    assert ir.level_to_linqu_level(ir.Level.CHIP_DIE) == 1
    assert ir.level_to_linqu_level(ir.Level.CHIP) == 2
    assert ir.level_to_linqu_level(ir.Level.HOST) == 3
    assert ir.level_to_linqu_level(ir.Level.CLUSTER_0) == 4
    assert ir.level_to_linqu_level(ir.Level.CLUSTER_1) == 5
    assert ir.level_to_linqu_level(ir.Level.CLUSTER_2) == 6
    assert ir.level_to_linqu_level(ir.Level.GLOBAL) == 7


# ─── Step 01: Role enum ──────────────────────────────────────────────────────


def test_role_values():
    """Role enum has exactly two distinct values."""
    assert ir.Role.Orchestrator != ir.Role.Worker


# ─── Step 01: DSL exports ────────────────────────────────────────────────────


def test_level_accessible_from_pl():
    """Level and Role are accessible via pl namespace."""
    assert pl.Level.HOST == ir.Level.HOST
    assert ir.level_to_linqu_level(pl.Level.POD) == ir.level_to_linqu_level(ir.Level.CLUSTER_0)
    assert pl.Role.Worker == ir.Role.Worker
    assert pl.Role.Orchestrator == ir.Role.Orchestrator


# ─── Step 02: Function with level/role ────────────────────────────────────────


def _make_empty_func(name, **kwargs):
    body = ir.SeqStmts([], ir.Span("test", 1, 0))
    span = ir.Span("test", 1, 0)
    return ir.Function(name, [], [], body, span, **kwargs)


def test_function_default_level_role():
    """Default Function has no level or role."""
    f = _make_empty_func("foo")
    assert f.level is None
    assert f.role is None
    assert f.func_type == ir.FunctionType.Opaque


def test_function_with_level():
    """Function can carry a hierarchy level."""
    f = _make_empty_func("orch", type=ir.FunctionType.Orchestration, level=ir.Level.HOST)
    assert f.level == ir.Level.HOST
    assert f.func_type == ir.FunctionType.Orchestration


def test_function_with_level_and_role():
    """Function can carry both level and role."""
    f = _make_empty_func("worker", level=ir.Level.HOST, role=ir.Role.Worker)
    assert f.level == ir.Level.HOST
    assert f.role == ir.Role.Worker


def test_function_with_level_alias():
    """Function level set via alias resolves correctly."""
    f = _make_empty_func("orch", level=ir.Level.POD)
    assert f.level == ir.Level.CLUSTER_0


def test_function_backward_compat():
    """Existing code creating Function(type=InCore) still works."""
    f = _make_empty_func("kernel", type=ir.FunctionType.InCore)
    assert f.func_type == ir.FunctionType.InCore
    assert f.level is None
    assert f.role is None


def test_structural_equal_with_level():
    """structural_equal considers level and role fields."""
    f1 = _make_empty_func("a", level=ir.Level.HOST, role=ir.Role.Worker)
    f2 = _make_empty_func("a", level=ir.Level.HOST, role=ir.Role.Worker)
    f3 = _make_empty_func("a", level=ir.Level.POD, role=ir.Role.Worker)
    ir.assert_structural_equal(f1, f2)
    with pytest.raises(Exception):
        ir.assert_structural_equal(f1, f3)


def test_structural_equal_none_vs_set():
    """structural_equal distinguishes None from set level."""
    f_none = _make_empty_func("a")
    f_set = _make_empty_func("a", level=ir.Level.HOST)
    with pytest.raises(Exception):
        ir.assert_structural_equal(f_none, f_set)


def test_function_printer_shows_level_role():
    """Python printer includes level/role in decorator."""
    f = _make_empty_func(
        "worker",
        type=ir.FunctionType.Orchestration,
        level=ir.Level.HOST,
        role=ir.Role.Worker,
    )
    printed = str(f)
    assert "Level.HOST" in printed
    assert "Role.Worker" in printed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
