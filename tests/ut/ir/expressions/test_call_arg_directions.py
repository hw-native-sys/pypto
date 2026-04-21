# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for Call.arg_directions / Call.attrs.

Covers Python bindings, construction overloads, structural equality / hashing,
and serialization round-trip of the ``arg_directions`` value carried under
``Call.attrs['arg_directions']``.
"""

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


def _attrs_with_dirs(dirs):
    return {"arg_directions": list(dirs)}


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
    """Call construction with the new attrs-based arg_directions storage."""

    def test_legacy_constructors_default_to_empty(self):
        op = _make_op()
        c1 = ir.Call(op, [], _span())
        c2 = ir.Call(op, [_scalar_var("x")], _make_type(), _span())
        c3 = ir.Call(op, [_scalar_var("x")], {}, _span())
        c4 = ir.Call(op, [_scalar_var("x")], {}, _make_type(), _span())
        for c in (c1, c2, c3, c4):
            assert list(c.arg_directions) == []
            assert c.attrs == {}

    def test_explicit_directions_round_trip(self):
        op = _make_op()
        x, y, z = _scalar_var("x"), _scalar_var("y"), _scalar_var("z")
        dirs = [
            ir.ArgDirection.Input,
            ir.ArgDirection.Output,
            ir.ArgDirection.InOut,
        ]
        call = ir.Call(op, [x, y, z], {}, _attrs_with_dirs(dirs), _make_type(), _span())
        assert [d for d in call.arg_directions] == dirs
        assert "arg_directions" in call.attrs
        assert list(call.attrs["arg_directions"]) == dirs

    def test_attrs_none_is_equivalent_to_empty(self):
        op = _make_op()
        call = ir.Call(op, [_scalar_var("x")], {}, None, _make_type(), _span())
        assert list(call.arg_directions) == []
        assert call.attrs == {}

    def test_size_mismatch_raises(self):
        op = _make_op()
        with pytest.raises(TypeError, match=r"attrs\['arg_directions'\] size .* must match args size"):
            ir.Call(
                op,
                [_scalar_var("x"), _scalar_var("y")],
                {},
                _attrs_with_dirs([ir.ArgDirection.Input]),  # length 1, args length 2
                _make_type(),
                _span(),
            )

    def test_arg_directions_is_read_only(self):
        op = _make_op()
        call = ir.Call(
            op,
            [_scalar_var("x")],
            {},
            _attrs_with_dirs([ir.ArgDirection.Output]),
            _make_type(),
            _span(),
        )
        with pytest.raises(AttributeError):
            call.arg_directions = []  # type: ignore[misc]

    def test_attrs_is_read_only(self):
        op = _make_op()
        call = ir.Call(
            op,
            [_scalar_var("x")],
            {},
            _attrs_with_dirs([ir.ArgDirection.Output]),
            _make_type(),
            _span(),
        )
        with pytest.raises(AttributeError):
            call.attrs = {}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Structural equality and hashing
# ---------------------------------------------------------------------------


class TestCallArgDirectionsStructural:
    """arg_directions stored under attrs participates in structural_hash / structural_equal."""

    def _make_pair(self, dirs_a, dirs_b):
        op = _make_op()
        x = _scalar_var("x")
        a = ir.Call(op, [x], {}, _attrs_with_dirs(dirs_a), _make_type(), _span())
        b = ir.Call(op, [x], {}, _attrs_with_dirs(dirs_b), _make_type(), _span())
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

    def test_legacy_no_attrs_vs_explicit_input_unequal(self):
        # No attrs (legacy) and explicitly Input should be distinguishable.
        op = _make_op()
        x = _scalar_var("x")
        a = ir.Call(op, [x], {}, _make_type(), _span())
        b = ir.Call(op, [x], {}, _attrs_with_dirs([ir.ArgDirection.Input]), _make_type(), _span())
        assert not ir.structural_equal(a, b, enable_auto_mapping=True)


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestCallArgDirectionsSerialization:
    """arg_directions stored under attrs survives serialize/deserialize round-trip."""

    def test_round_trip_preserves_directions(self):
        op = _make_op()
        x, y = _scalar_var("x"), _scalar_var("y")
        dirs = [ir.ArgDirection.Input, ir.ArgDirection.Output]
        call = ir.Call(op, [x, y], {}, _attrs_with_dirs(dirs), _make_type(), _span())

        data = ir.serialize(call)
        restored = ir.deserialize(data)

        ir.assert_structural_equal(call, restored, enable_auto_mapping=True)
        assert isinstance(restored, ir.Call)
        assert [d for d in restored.arg_directions] == dirs

    def test_round_trip_no_directions(self):
        op = _make_op()
        x = _scalar_var("x")
        call = ir.Call(op, [x], _span())  # legacy constructor → no attrs

        data = ir.serialize(call)
        restored = ir.deserialize(data)

        ir.assert_structural_equal(call, restored, enable_auto_mapping=True)
        assert isinstance(restored, ir.Call)
        assert list(restored.arg_directions) == []
        assert restored.attrs == {}

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
        call = ir.Call(op, vars_, {}, _attrs_with_dirs(dirs), _make_type(), _span())

        data = ir.serialize(call)
        restored = ir.deserialize(data)

        ir.assert_structural_equal(call, restored, enable_auto_mapping=True)
        assert isinstance(restored, ir.Call)
        assert [d for d in restored.arg_directions] == dirs


class TestLegacyArgDirectionsCompat:
    """Backward compatibility: deserialize legacy .pir payloads with top-level arg_directions."""

    def test_legacy_top_level_arg_directions_lifted_into_attrs(self):
        msgpack = pytest.importorskip("msgpack")

        op = _make_op()
        x, y = _scalar_var("x"), _scalar_var("y")
        dirs = [ir.ArgDirection.Input, ir.ArgDirection.Output]
        call = ir.Call(op, [x, y], {}, _attrs_with_dirs(dirs), _make_type(), _span())

        # Round-trip through msgpack to mutate the wire format: strip the new
        # `attrs` field and inject a legacy top-level `arg_directions` array.
        payload = bytes(ir.serialize(call))
        root = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        assert root["type"] == "Call"
        fields = root["fields"]
        # Drop the new attrs container so only the legacy field remains.
        fields.pop("attrs", None)
        fields["arg_directions"] = [int(d.value) for d in dirs]
        legacy_payload = msgpack.packb(root, use_bin_type=True)

        restored = ir.deserialize(legacy_payload)

        assert isinstance(restored, ir.Call)
        assert [d for d in restored.arg_directions] == dirs
        assert "arg_directions" in restored.attrs
        assert list(restored.attrs["arg_directions"]) == dirs


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
        assert "arg_directions" in call.attrs

    def test_make_call_without_directions_is_legacy(self):
        op = _make_op()
        call = ir.make_call(op, [_scalar_var("x")])
        assert list(call.arg_directions) == []
        assert call.attrs == {}

    def test_make_call_size_mismatch_raises(self):
        op = _make_op()
        with pytest.raises(ValueError, match="must match args length"):
            ir.make_call(op, [_scalar_var("x"), _scalar_var("y")], directions=[ir.input])
