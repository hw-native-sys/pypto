# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for Call.arg_directions: bindings, construction, structural eq/hash, and serialization."""

import pytest
from pypto import DataType, ir
from pypto.ir import directions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _span() -> ir.Span:
    return ir.Span.unknown()


def _scalar_var(name: str) -> ir.Var:
    return ir.Var(name, ir.ScalarType(DataType.INT64), _span())


def _make_op() -> ir.Op:
    return ir.Op("test.kernel")


def _make_type() -> ir.Type:
    return ir.UnknownType()


# ---------------------------------------------------------------------------
# Enum bindings
# ---------------------------------------------------------------------------


class TestArgDirectionEnum:
    """ArgDirection enum exposes the runtime task-submission semantics."""

    def test_all_six_members_present(self):
        members = {d.name for d in ir.ArgDirection}
        assert members == {"Input", "Output", "InOut", "OutputExisting", "NoDep", "Scalar"}

    def test_values_are_stable(self):
        # Wire format relies on these integer codes.
        assert ir.ArgDirection.Input.value == 0
        assert ir.ArgDirection.Output.value == 1
        assert ir.ArgDirection.InOut.value == 2
        assert ir.ArgDirection.OutputExisting.value == 3
        assert ir.ArgDirection.NoDep.value == 4
        assert ir.ArgDirection.Scalar.value == 5


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestCallArgDirectionsConstruction:
    """Call construction with the new arg_directions field."""

    def test_legacy_constructors_default_to_empty(self):
        op = _make_op()
        c1 = ir.Call(op, [], _span())
        c2 = ir.Call(op, [_scalar_var("x")], _make_type(), _span())
        c3 = ir.Call(op, [_scalar_var("x")], {}, _span())
        c4 = ir.Call(op, [_scalar_var("x")], {}, _make_type(), _span())
        for c in (c1, c2, c3, c4):
            assert list(c.arg_directions) == []

    def test_explicit_directions_round_trip(self):
        op = _make_op()
        x, y, z = _scalar_var("x"), _scalar_var("y"), _scalar_var("z")
        dirs = [
            ir.ArgDirection.Input,
            ir.ArgDirection.Output,
            ir.ArgDirection.InOut,
        ]
        call = ir.Call(op, [x, y, z], dirs, {}, _make_type(), _span())
        assert [d for d in call.arg_directions] == dirs

    def test_explicit_empty_directions_allowed(self):
        # An explicit empty list is equivalent to "legacy / not yet derived".
        op = _make_op()
        call = ir.Call(op, [_scalar_var("x")], [], {}, _make_type(), _span())
        assert list(call.arg_directions) == []

    def test_size_mismatch_raises(self):
        op = _make_op()
        with pytest.raises(Exception):  # noqa: PT011  TypeError raised from C++
            ir.Call(
                op,
                [_scalar_var("x"), _scalar_var("y")],
                [ir.ArgDirection.Input],  # length 1, args length 2
                {},
                _make_type(),
                _span(),
            )

    def test_arg_directions_is_read_only(self):
        op = _make_op()
        call = ir.Call(
            op,
            [_scalar_var("x")],
            [ir.ArgDirection.Output],
            {},
            _make_type(),
            _span(),
        )
        with pytest.raises(AttributeError):
            call.arg_directions = []  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Structural equality and hashing
# ---------------------------------------------------------------------------


class TestCallArgDirectionsStructural:
    """arg_directions participates in structural_hash / structural_equal."""

    def _make_pair(self, dirs_a, dirs_b):
        op = _make_op()
        x = _scalar_var("x")
        a = ir.Call(op, [x], list(dirs_a), {}, _make_type(), _span())
        b = ir.Call(op, [x], list(dirs_b), {}, _make_type(), _span())
        return a, b

    def test_equal_when_directions_match(self):
        a, b = self._make_pair(
            [ir.ArgDirection.InOut],
            [ir.ArgDirection.InOut],
        )
        assert ir.structural_equal(a, b, enable_auto_mapping=True)
        assert ir.structural_hash(a) == ir.structural_hash(b)

    def test_unequal_when_directions_differ(self):
        a, b = self._make_pair(
            [ir.ArgDirection.Input],
            [ir.ArgDirection.Output],
        )
        assert not ir.structural_equal(a, b, enable_auto_mapping=True)

    def test_legacy_empty_vs_explicit_input_unequal(self):
        # Empty (legacy) and explicitly Input should be distinguishable.
        a, b = self._make_pair([], [ir.ArgDirection.Input])
        assert not ir.structural_equal(a, b, enable_auto_mapping=True)


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestCallArgDirectionsSerialization:
    """arg_directions survives serialize/deserialize round-trip."""

    def test_round_trip_preserves_directions(self):
        op = _make_op()
        x, y = _scalar_var("x"), _scalar_var("y")
        dirs = [ir.ArgDirection.Input, ir.ArgDirection.Output]
        call = ir.Call(op, [x, y], dirs, {}, _make_type(), _span())

        data = ir.serialize(call)
        restored = ir.deserialize(data)

        ir.assert_structural_equal(call, restored, enable_auto_mapping=True)
        assert isinstance(restored, ir.Call)
        assert [d for d in restored.arg_directions] == dirs

    def test_round_trip_empty_directions(self):
        op = _make_op()
        x = _scalar_var("x")
        call = ir.Call(op, [x], _span())  # legacy constructor → empty

        data = ir.serialize(call)
        restored = ir.deserialize(data)

        ir.assert_structural_equal(call, restored, enable_auto_mapping=True)
        assert isinstance(restored, ir.Call)
        assert list(restored.arg_directions) == []

    def test_round_trip_all_six_kinds(self):
        op = _make_op()
        vars_ = [_scalar_var(f"v{i}") for i in range(6)]
        dirs = [
            ir.ArgDirection.Input,
            ir.ArgDirection.Output,
            ir.ArgDirection.InOut,
            ir.ArgDirection.OutputExisting,
            ir.ArgDirection.NoDep,
            ir.ArgDirection.Scalar,
        ]
        call = ir.Call(op, vars_, dirs, {}, _make_type(), _span())

        data = ir.serialize(call)
        restored = ir.deserialize(data)

        ir.assert_structural_equal(call, restored, enable_auto_mapping=True)
        assert isinstance(restored, ir.Call)
        assert [d for d in restored.arg_directions] == dirs


class TestDirectionHelpers:
    """``ir.input/output/...`` are stable aliases of ``ArgDirection`` values."""

    def test_aliases_match_enum(self):
        assert ir.input is ir.ArgDirection.Input
        assert ir.output is ir.ArgDirection.Output
        assert ir.output_existing is ir.ArgDirection.OutputExisting
        assert ir.inout is ir.ArgDirection.InOut
        assert ir.no_dep is ir.ArgDirection.NoDep
        assert ir.scalar_dir is ir.ArgDirection.Scalar

    def test_directions_module_reexports(self):
        assert directions.input is ir.input
        assert directions.output is ir.output
        assert directions.inout is ir.inout
        assert directions.no_dep is ir.no_dep
        assert directions.output_existing is ir.output_existing
        assert directions.scalar is ir.scalar_dir

    def test_make_call_with_explicit_directions(self):
        op = _make_op()
        x, y = _scalar_var("x"), _scalar_var("y")
        call = ir.make_call(op, [x, y], directions=[ir.input, ir.inout])
        assert [d for d in call.arg_directions] == [ir.ArgDirection.Input, ir.ArgDirection.InOut]

    def test_make_call_without_directions_is_legacy(self):
        op = _make_op()
        call = ir.make_call(op, [_scalar_var("x")])
        assert list(call.arg_directions) == []

    def test_make_call_size_mismatch_raises(self):
        op = _make_op()
        with pytest.raises(ValueError, match="must match args length"):
            ir.make_call(op, [_scalar_var("x"), _scalar_var("y")], directions=[ir.input])
