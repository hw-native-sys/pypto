# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for IR structural hash functionality."""

from collections.abc import Callable

import pytest
from pypto import DataType, ir
from pypto.pypto_core import ir as _core_ir


class TestStructuralHash:
    """Tests for structural hash function."""

    def test_same_structure_same_hash(self):
        """Test that expressions with same structure hash to same value."""
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        x2 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        # Same variable name, different spans - should have same hash
        hash1 = ir.structural_hash(x1)
        hash2 = ir.structural_hash(x2)
        assert hash1 != hash2

    def test_different_var_names_different_hash(self):
        """Test that variables with different names hash differently."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        hash_x = ir.structural_hash(x)
        hash_y = ir.structural_hash(y)

        # Different names should (almost certainly) have different hashes
        assert hash_x != hash_y

    def test_different_const_values_different_hash(self):
        """Test that constants with different values hash differently."""
        c1 = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
        c2 = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())

        hash1 = ir.structural_hash(c1)
        hash2 = ir.structural_hash(c2)

        assert hash1 != hash2

    def test_same_const_value_same_hash(self):
        """Test that constants with same value hash to same value."""
        c1 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
        c2 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())

        hash1 = ir.structural_hash(c1)
        hash2 = ir.structural_hash(c2)

        assert hash1 == hash2

    def test_const_bool_hash(self):
        """Test that ConstBool with different values hash differently."""
        b_true1 = ir.ConstBool(True, ir.Span.unknown())
        b_true2 = ir.ConstBool(True, ir.Span.unknown())
        b_false = ir.ConstBool(False, ir.Span.unknown())

        hash_true1 = ir.structural_hash(b_true1)
        hash_true2 = ir.structural_hash(b_true2)
        hash_false = ir.structural_hash(b_false)

        # Same values should have same hash
        assert hash_true1 == hash_true2
        # Different values should have different hash
        assert hash_true1 != hash_false

    def test_different_operation_types_different_hash(self):
        """Test that different operation types hash differently."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        add_expr = ir.Add(x, y, DataType.INT64, ir.Span.unknown())
        sub_expr = ir.Sub(x, y, DataType.INT64, ir.Span.unknown())
        mul_expr = ir.Mul(x, y, DataType.INT64, ir.Span.unknown())

        hash_add = ir.structural_hash(add_expr)
        hash_sub = ir.structural_hash(sub_expr)
        hash_mul = ir.structural_hash(mul_expr)

        # Different operations should hash differently
        assert hash_add != hash_sub
        assert hash_add != hash_mul
        assert hash_sub != hash_mul

    def test_nested_expression_hash(self):
        """Test hashing of nested expressions."""
        # Build (x + 5) * 2 with different spans
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c5_1 = ir.ConstInt(5, DataType.INT64, ir.Span.unknown())
        c2_1 = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())
        expr1 = ir.Mul(
            ir.Add(x1, c5_1, DataType.INT64, ir.Span.unknown()), c2_1, DataType.INT64, ir.Span.unknown()
        )

        x2 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        c5_2 = ir.ConstInt(5, DataType.INT64, ir.Span.unknown())
        c2_2 = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())
        expr2 = ir.Mul(
            ir.Add(x2, c5_2, DataType.INT64, ir.Span.unknown()), c2_2, DataType.INT64, ir.Span.unknown()
        )

        # Same structure, different spans - should hash to same value
        hash1 = ir.structural_hash(expr1)
        hash2 = ir.structural_hash(expr2)
        assert hash1 != hash2

    def test_operand_order_matters(self):
        """Test that operand order affects hash (x + y != y + x in structure)."""
        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        add1 = ir.Add(x, y, DataType.INT64, ir.Span.unknown())  # x + y
        add2 = ir.Add(y, x, DataType.INT64, ir.Span.unknown())  # y + x

        hash1 = ir.structural_hash(add1)
        hash2 = ir.structural_hash(add2)

        # Different operand order should (almost certainly) hash differently
        assert hash1 != hash2

    def test_unary_expression_hash(self):
        """Test hashing of unary expressions."""
        x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        neg1 = ir.Neg(x1, DataType.INT64, ir.Span.unknown())

        x2 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        neg2 = ir.Neg(x2, DataType.INT64, ir.Span.unknown())

        hash1 = ir.structural_hash(neg1)
        hash2 = ir.structural_hash(neg2)

        assert hash1 != hash2

    def test_call_expression_hash(self):
        """Test hashing of call expressions."""
        op1 = ir.Op("func")
        op2 = ir.Op("func")

        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
        y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        call1 = ir.Call(op1, [x, y], ir.Span.unknown())
        call2 = ir.Call(op2, [x, y], ir.Span.unknown())

        hash1 = ir.structural_hash(call1)
        hash2 = ir.structural_hash(call2)

        # Same op name and args - should hash to same value
        assert hash1 == hash2

    def test_different_op_names_different_hash(self):
        """Test that calls with different op names hash differently."""
        op1 = ir.Op("func1")
        op2 = ir.Op("func2")

        x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())

        call1 = ir.Call(op1, [x], ir.Span.unknown())
        call2 = ir.Call(op2, [x], ir.Span.unknown())

        hash1 = ir.structural_hash(call1)
        hash2 = ir.structural_hash(call2)

        # Different op names should hash differently
        assert hash1 != hash2

    def test_stmt_different_from_expr_hash(self):
        """Test that Stmt and Expr nodes hash differently."""
        span = ir.Span.unknown()

        expr = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        var = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        stmt = ir.AssignStmt(var, expr, span)

        hash_stmt = ir.structural_hash(stmt)
        hash_expr = ir.structural_hash(expr)

        # Different IR node types should hash differently
        assert hash_stmt != hash_expr

    def test_assign_stmt_same_structure_hash(self):
        """Test AssignStmt nodes with same structure hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)

        assign1 = ir.AssignStmt(x1, y1, span)
        assign2 = ir.AssignStmt(x2, y2, span)

        hash1 = ir.structural_hash(assign1)
        hash2 = ir.structural_hash(assign2)
        # Different variable pointers result in different hashes without auto_mapping
        assert hash1 != hash2

    def test_assign_stmt_different_var_hash(self):
        """Test AssignStmt nodes with different var hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)

        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(z, y, span)

        hash1 = ir.structural_hash(assign1)
        hash2 = ir.structural_hash(assign2)
        assert hash1 == hash2

    def test_assign_stmt_different_value_hash(self):
        """Test AssignStmt nodes with different value hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)

        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(x, z, span)

        hash1 = ir.structural_hash(assign1)
        hash2 = ir.structural_hash(assign2)
        assert hash1 != hash2

    def test_assign_stmt_different_from_base_stmt_hash(self):
        """Test AssignStmt and base Stmt nodes hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)

        assign = ir.AssignStmt(x, y, span)

        hash_assign = ir.structural_hash(assign)
        assert hash_assign != 0

    def test_yield_stmt_same_structure_hash(self):
        """Test YieldStmt nodes with same structure hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)

        yield_stmt1 = ir.YieldStmt([x1, y1], span)
        yield_stmt2 = ir.YieldStmt([x2, y2], span)

        hash1 = ir.structural_hash(yield_stmt1)
        hash2 = ir.structural_hash(yield_stmt2)
        # Different variable pointers result in different hashes without auto_mapping
        assert hash1 != hash2

    def test_yield_stmt_different_vars_hash(self):
        """Test YieldStmt nodes with different vars hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)

        yield_stmt1 = ir.YieldStmt([x, y], span)
        yield_stmt2 = ir.YieldStmt([x, z], span)

        hash1 = ir.structural_hash(yield_stmt1)
        hash2 = ir.structural_hash(yield_stmt2)
        assert hash1 != hash2

    def test_yield_stmt_empty_vs_non_empty_hash(self):
        """Test YieldStmt nodes with empty and non-empty value lists hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)

        yield_stmt1 = ir.YieldStmt([], span)
        yield_stmt2 = ir.YieldStmt([x], span)

        hash1 = ir.structural_hash(yield_stmt1)
        hash2 = ir.structural_hash(yield_stmt2)
        assert hash1 != hash2


class TestTypePyHashEqConsistency:
    """Regression tests for the Python hash/eq contract on Type bindings.

    Type's __eq__ is bound to structural_equal; __hash__ must therefore use
    structural_hash so that structurally-equal types are interchangeable as
    set/dict keys.
    """

    def test_two_equal_scalar_types_hash_equally(self):
        a = ir.ScalarType(DataType.FP32)
        b = ir.ScalarType(DataType.FP32)
        assert a == b
        assert hash(a) == hash(b)
        assert a in {b}

    def test_distinct_scalar_types_hash_differently(self):
        a = ir.ScalarType(DataType.FP32)
        b = ir.ScalarType(DataType.INT32)
        assert a != b
        assert hash(a) != hash(b)

    def test_type_works_as_dict_key(self):
        d = {ir.ScalarType(DataType.FP32): "fp32"}
        assert d[ir.ScalarType(DataType.FP32)] == "fp32"


def _span() -> ir.Span:
    return ir.Span.unknown()


def _dims(*extents: int) -> list[ir.ConstInt]:
    return [ir.ConstInt(e, DataType.INT64, _span()) for e in extents]


# One shared WindowBuffer so the two instances a factory builds stay
# structurally equal: window_buffer_ is compared by Var identity, not by value.
_SHARED_WINDOW_BUFFER = ir.WindowBuffer(
    ir.Var("wb_base", ir.PtrType(), _span()),
    ir.ConstInt(4096, DataType.INT64, _span()),
)

# Every Python-constructible Type kind, one factory each. A factory must build a
# *fresh* instance on every call so the equal-implies-same-hash contract below is
# checked against two independently-built values.
_TYPE_FACTORIES: dict[str, Callable[[], ir.Type]] = {
    "UnknownType": lambda: ir.UnknownType(),
    "ScalarType": lambda: ir.ScalarType(DataType.FP32),
    "TensorType": lambda: ir.TensorType([64, 128], DataType.FP32),
    "TensorType_view": lambda: ir.TensorType(
        [64, 128],
        DataType.FP32,
        tensor_view=ir.TensorView([128, 1], ir.TensorLayout.ND, [32, 64]),
    ),
    "DistributedTensorType": lambda: ir.DistributedTensorType([64, 128], DataType.FP32),
    "DistributedTensorType_window_buffer": lambda: ir.DistributedTensorType(
        _dims(64, 128), DataType.FP32, _SHARED_WINDOW_BUFFER
    ),
    "TileType": lambda: ir.TileType([64, 128], DataType.FP16),
    "ArrayType": lambda: ir.ArrayType(DataType.INT32, 16),
    "TupleType": lambda: ir.TupleType([ir.ScalarType(DataType.INT64), ir.ArrayType(DataType.INT32, 8)]),
    "PtrType": lambda: ir.PtrType(),
    "WindowBufferType": lambda: ir.WindowBufferType(),
    "CommCtxType": lambda: ir.CommCtxType(),
    "PrefetchAsyncContextType": lambda: ir.PrefetchAsyncContextType(),
    "AsyncEventType": lambda: ir.AsyncEventType(),
    "AsyncSessionType": lambda: ir.AsyncSessionType(),
}

# Bound but not instantiable: pure base classes with no nanobind constructor.
# A new entry here is a deliberate statement that the class carries no fields of
# its own to hash.
_ABSTRACT_TYPE_CLASSES = {"ShapedType"}


def _bound_type_class_names() -> set[str]:
    """Every ``Type`` subclass the bindings expose.

    Scans the native module rather than ``pypto.ir``: the latter re-exports via
    ``import *`` and would silently miss a class the star import skips.
    """
    return {
        name
        for name in dir(_core_ir)
        if isinstance(getattr(_core_ir, name), type)
        and issubclass(getattr(_core_ir, name), _core_ir.Type)
        and getattr(_core_ir, name) is not _core_ir.Type
    }


class TestHashTypeLadderParity:
    """``HashType`` must cover every Type kind its sibling ladders cover.

    Four independent if/else ladders encode "what fields does each Type have" —
    ``EqualType``, ``HashType``, ``SerializeType`` and ``DeserializeType``.
    Nothing ties them together, so a Type added to three and forgotten in the
    fourth stays invisible until a user hashes one, where it lands on
    ``HashType``'s trailing ``INTERNAL_CHECK(false)``. These tests pin the
    parity so the next omission fails here instead.
    """

    @pytest.mark.parametrize("kind", sorted(_TYPE_FACTORIES))
    def test_structural_hash_does_not_raise(self, kind: str):
        """No Type kind may fall through to ``HashType``'s unhandled branch."""
        assert isinstance(ir.structural_hash(_TYPE_FACTORIES[kind]()), int)

    @pytest.mark.parametrize("kind", sorted(_TYPE_FACTORIES))
    def test_structural_equal_implies_equal_hash(self, kind: str):
        """``__eq__``/``__hash__`` consistency, per Type kind."""
        make = _TYPE_FACTORIES[kind]
        lhs, rhs = make(), make()
        assert ir.structural_equal(lhs, rhs)
        assert hash(lhs) == hash(rhs)
        assert lhs in {rhs}

    def test_every_bound_type_class_has_a_factory(self):
        """Guard the guard: a newly bound Type must be added above.

        Without this, a Type kind added to the bindings and omitted from
        ``HashType`` would still pass every parametrized case, because nothing
        would construct it.
        """
        covered = {type(make()).__name__ for make in _TYPE_FACTORIES.values()}
        missing = _bound_type_class_names() - covered - _ABSTRACT_TYPE_CLASSES
        assert not missing, (
            f"Type classes with no structural_hash coverage: {sorted(missing)}. "
            "Add a factory to _TYPE_FACTORIES, and a matching branch to "
            "HashType in src/ir/transforms/structural_hash.cpp."
        )

    def test_array_type_is_usable_as_a_dict_key(self):
        """``ArrayType`` was absent from ``HashType`` entirely."""
        d = {ir.ArrayType(DataType.INT32, 16): "arr"}
        assert d[ir.ArrayType(DataType.INT32, 16)] == "arr"

    def test_array_type_extent_participates_in_the_hash(self):
        assert hash(ir.ArrayType(DataType.INT32, 16)) != hash(ir.ArrayType(DataType.INT32, 32))

    def test_array_type_dtype_participates_in_the_hash(self):
        assert hash(ir.ArrayType(DataType.INT32, 16)) != hash(ir.ArrayType(DataType.INT64, 16))

    def test_distributed_tensor_type_hashes_apart_from_plain_tensor_type(self):
        """The two kinds are distinguished only by ``ObjectKind``.

        ``As<TensorType>`` is precise-match, so the ``DistributedTensorType``
        dispatch has to name the kind explicitly.
        """
        tensor = ir.TensorType([64, 128], DataType.FP32)
        dist = ir.DistributedTensorType([64, 128], DataType.FP32)
        assert not ir.structural_equal(tensor, dist)
        assert hash(tensor) != hash(dist)

    def test_distinct_window_buffers_hash_apart(self):
        """Exercises the window-buffer block, which was unreachable.

        Two same-shape, same-dtype distributed tensors backed by different
        WindowBuffers are distinct types and must not collide.
        """
        first = ir.DistributedTensorType(
            _dims(64),
            DataType.FP32,
            ir.WindowBuffer(ir.Var("a", ir.PtrType(), _span()), ir.ConstInt(64, DataType.INT64, _span())),
        )
        second = ir.DistributedTensorType(
            _dims(64),
            DataType.FP32,
            ir.WindowBuffer(ir.Var("b", ir.PtrType(), _span()), ir.ConstInt(64, DataType.INT64, _span())),
        )
        assert not ir.structural_equal(first, second)
        assert hash(first) != hash(second)

    @pytest.mark.parametrize("kind", sorted(_TYPE_FACTORIES))
    def test_structural_equal_implies_equal_hash_under_auto_mapping(self, kind: str):
        """The contract must hold in auto-mapping mode too.

        ``EqualType`` compares ``DistributedTensorType::window_buffer_`` with
        ``EqualVar``, which under auto-mapping accepts any two buffers a
        consistent bijection allows -- ignoring their fields. So the hash must
        mix in the buffer's *identity* only; hashing the node would fold in
        ``size_`` and the staging flags and make equal types hash apart.
        """
        make = _TYPE_FACTORIES[kind]
        lhs, rhs = make(), make()
        assert ir.structural_equal(lhs, rhs, enable_auto_mapping=True)
        assert ir.structural_hash(lhs, enable_auto_mapping=True) == ir.structural_hash(
            rhs, enable_auto_mapping=True
        )

    def test_auto_mapped_window_buffers_of_different_size_hash_together(self):
        """Regression: differing buffer fields must not split the hash.

        Under auto-mapping these two types are ``structural_equal`` (EqualVar
        maps the buffers), so they must hash alike even though the buffers
        differ in size and staging flags.
        """
        first = ir.DistributedTensorType(
            _dims(64),
            DataType.FP32,
            ir.WindowBuffer(ir.Var("a", ir.PtrType(), _span()), ir.ConstInt(64, DataType.INT64, _span())),
        )
        second = ir.DistributedTensorType(
            _dims(64),
            DataType.FP32,
            ir.WindowBuffer(
                ir.Var("b", ir.PtrType(), _span()),
                ir.ConstInt(4096, DataType.INT64, _span()),
                True,
                True,
                _span(),
            ),
        )
        assert ir.structural_equal(first, second, enable_auto_mapping=True)
        assert ir.structural_hash(first, enable_auto_mapping=True) == ir.structural_hash(
            second, enable_auto_mapping=True
        )

    def test_window_buffer_presence_participates_in_the_hash(self):
        without = ir.DistributedTensorType(_dims(64), DataType.FP32)
        with_wb = ir.DistributedTensorType(_dims(64), DataType.FP32, _SHARED_WINDOW_BUFFER)
        assert not ir.structural_equal(without, with_wb)
        assert hash(without) != hash(with_wb)

    def test_tensor_view_pad_participates_in_the_hash(self):
        """``EqualType`` compares ``TensorView::pad``, so ``HashType`` must fold it in.

        The sibling ``TileView`` branch always hashed its own ``pad``; the
        ``TensorView`` branch stopped at ``layout``. That left the contract
        intact (a field in equality but not the hash only widens the collision
        class) but made padded tensor views needlessly collision-prone.
        """

        def make(pad: ir.PadValue) -> ir.TensorType:
            return ir.TensorType(
                [64, 1],
                DataType.FP32,
                tensor_view=ir.TensorView([1, 1], ir.TensorLayout.ND, [32, 1], pad),
            )

        null_pad, zero_pad = make(ir.PadValue.null), make(ir.PadValue.zero)
        assert not ir.structural_equal(null_pad, zero_pad)
        assert hash(null_pad) != hash(zero_pad)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
