# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The explicit result-storage binding for allocating producers.

``ir.bind_result_storage(call, destinations)`` states where a Functional-stage
producer places its results: it validates the producer, returns an immutable
Call whose Tile result types carry the requested MemRefs, and never mutates
the original. These tests pin the contract at the IR level — single and
multiple destinations, slot selection, and every rejection — plus the two
round-trip properties the printing surface must keep: a bound allocation
declaration survives printing, and a typed-constant MemRef offset parses.
"""

import pytest
from pypto.ir import IRBuilder
from pypto.ir.printer import python_print
from pypto.language.parser.text_parser import parse_program
from pypto.pypto_core import DataType, ir

_EXP = ir.get_op("tile.exp").name
_ROW_EXPAND = ir.get_op("tile.row_expand").name


def _span() -> ir.Span:
    return ir.Span.unknown()


def _vec_tile(memref: ir.MemRef | None = None) -> ir.TileType:
    return ir.TileType([64, 64], DataType.FP32, memref, None, ir.MemorySpace.Vec)


def _exp_producer() -> ir.Call:
    """A detached elementwise producer with a resolved Vec output space."""
    x = ir.Var("x", _vec_tile(), _span())
    return ir.create_op_call(_EXP, [x], _span())


class TestResultStorageBecomesPartOfTheType:
    def test_binding_places_the_declaration_in_the_produced_type(self):
        call = _exp_producer()
        pool = ir.MemRef("scratch")
        bound = ir.bind_result_storage(call, [pool])

        assert bound is not call
        memory = bound.type.memref
        assert memory is not None
        assert memory.is_pinned_
        assert memory.base_.name_hint == "scratch"
        assert memory.slot_count_ == 1
        # The value type is preserved: shape, dtype, view and memory space.
        assert bound.type.shape == call.type.shape
        assert bound.type.dtype == call.type.dtype
        assert bound.type.memory_space == call.type.memory_space
        # Immutability: the original call is untouched.
        assert call.type.memref is None

    def test_a_none_destination_preserves_the_result(self):
        call = _exp_producer()
        bound = ir.bind_result_storage(call, [None])
        assert bound is not call
        assert bound.type.memref is None

    def test_one_destination_is_required_per_result(self):
        call = _exp_producer()
        with pytest.raises(ValueError, match="one destination per result"):
            ir.bind_result_storage(call, [None, None])

    def test_a_multi_slot_allocation_requires_slot_selection(self):
        call = _exp_producer()
        pool = ir.MemRef("pool", slots=2)
        with pytest.raises(ValueError, match="Select a slot before binding"):
            ir.bind_result_storage(call, [pool])

        selected = ir.bind_result_storage(call, [pool[1]])
        memory = selected.type.memref
        assert memory is not None
        assert memory.slot_index_ is not None
        assert ir.MemRef.same_allocation(pool, memory)


class TestRejections:
    def test_rebinding_an_already_placed_producer_is_rejected(self):
        call = _exp_producer()
        bound = ir.bind_result_storage(call, [ir.MemRef("scratch")])
        with pytest.raises(ValueError, match="already has an explicit storage binding"):
            ir.bind_result_storage(bound, [ir.MemRef("other")])

    def test_a_memoryless_producer_has_no_bindable_storage(self):
        size = ir.ConstInt(64, DataType.INDEX, _span())
        window = ir.create_op_call("pld.tensor.alloc_window_buffer", [size], {"name": "w"}, _span())
        with pytest.raises(ValueError, match="requires a Tile producer"):
            ir.bind_result_storage(window, [ir.MemRef("scratch")])

    def test_a_destination_may_not_alias_a_forbidden_operand(self):
        # tile.row_expand reads its second operand while writing its output, so
        # the registry forbids the output from landing on that operand's buffer.
        buffer = ir.MemRef("buf")
        source = ir.Var("src", _vec_tile(), _span())
        column = ir.Var("col", ir.TileType([64, 1], DataType.FP32, buffer, None, ir.MemorySpace.Vec), _span())
        producer = ir.create_op_call(_ROW_EXPAND, [source, column], _span())
        with pytest.raises(ValueError, match="reads input argument 1 while writing its output"):
            ir.bind_result_storage(producer, [buffer])

        distinct = ir.bind_result_storage(producer, [ir.MemRef("dst")])
        assert distinct.type.memref is not None
        assert distinct.type.memref.base_.name_hint == "dst"


class _BindExp(ir.IRMutator):
    """Bind every elementwise producer in a body to a slot of ``pool``."""

    def __init__(self, pool: ir.MemRef) -> None:
        super().__init__()
        self._pool = pool

    def visit_assign_stmt(self, op):
        op = super().visit_assign_stmt(op)
        if isinstance(op.value, ir.Call) and op.value.op.name == _EXP:
            bound = ir.bind_result_storage(op.value, [self._pool[0]])
            var = ir.Var(op.var.name_hint, bound.type, op.span)
            return ir.AssignStmt(var, bound, op.span)
        return op


class TestPrintingRetainsTheBinding:
    def test_the_printed_annotation_carries_the_declaration_and_slot(self):
        source = """import pypto.language as pl

@pl.program
class P:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 64], pl.FP32], o: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
        t = pl.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
        e = pl.exp(t)
        o = pl.store(e, [0, 0], o)
"""
        program = parse_program(source)
        pool = ir.MemRef("pool", slots=2)
        functions = []
        for function in program.functions.values():
            if function.body is None:
                functions.append(function)
                continue
            body = _BindExp(pool).visit_stmt(function.body)
            functions.append(
                ir.Function(
                    function.name,
                    function.params,
                    function.return_types,
                    body,
                    function.span,
                    function.func_type,
                )
            )
        printed = python_print(ir.Program(functions, program.name, program.span))
        assert 'pl.MemRef("pool", slots=2)[0]' in printed, printed

    def test_a_typed_constant_memref_offset_round_trips(self):
        """A printed MemRef offset arrives as ``pl.const(value, dtype)``.

        The printer emits typed constants for MemRef byte offsets; the
        annotation resolver must rebuild them with their declared dtype (a
        bare eval of a ``pl.const`` call has no Python value), or a printed
        program with a shifted allocation stops reparsing.
        """
        source = """import pypto.language as pl

@pl.program
class P:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16, 128], pl.FP32], out: pl.Out[pl.Tensor[[16, 128], pl.FP32]]):
        p: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        t: pl.Tile[[16, 128], pl.FP32, pl.MemRef(p, pl.const(64, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(
            x, [0, 0], [16, 128]
        )
        out = pl.tile.store(t, [0, 0], out)
        return out
"""
        program = parse_program(source)
        function = next(iter(program.functions.values()))
        tile_param = None

        class _Find(ir.IRVisitor):
            def visit_assign_stmt(self, op):
                nonlocal tile_param
                if isinstance(op.var.type, ir.TileType) and op.var.type.memref is not None:
                    tile_param = op.var.type
                super().visit_assign_stmt(op)

        _Find().visit_function(function)
        assert tile_param is not None, "the annotated tile did not resolve a MemRef"
        offset = tile_param.memref.byte_offset_
        assert isinstance(offset, ir.ConstInt)
        assert offset.dtype == DataType.INT64
        assert offset.value == 64


class TestStorageBindsThroughTheDirectBuilder:
    def test_a_bound_producer_builds_into_a_function(self):
        """The bound call flows through ordinary IR construction unchanged."""

        span = _span()
        pool = ir.MemRef("pool", slots=2)
        ib = IRBuilder()
        with ib.function("f", span, ir.FunctionType.InCore) as fn:
            a = ib.func_arg("a", ir.TensorType([64, 64], DataType.FP32), span, ir.ParamDirection.In)
            ib.return_type(ir.TensorType([64, 64], DataType.FP32))

            def _pair(values: tuple[int, int]) -> ir.MakeTuple:
                return ir.MakeTuple([ir.ConstInt(v, DataType.INDEX, span) for v in values], span)

            t = ib.var("t", _vec_tile())
            load = ir.create_op_call(
                ir.get_op("tile.load").name,
                [a, _pair((0, 0)), _pair((64, 64)), _pair((64, 64))],
                span,
            )
            ib.assign(t, load, span)
            bound = ir.bind_result_storage(_exp_producer(), [pool[0]])
            e = ib.var("e", bound.type)
            ib.assign(e, bound, span)
            o = ib.var("o", ir.TensorType([64, 64], DataType.FP32), span)
            store = ir.create_op_call(
                ir.get_op("tile.store").name,
                [e, _pair((0, 0)), o],
                span,
            )
            r = ib.var("r", ir.TensorType([64, 64], DataType.FP32), span)
            ib.assign(r, store, span)
            ib.return_stmt([r])
        func = fn.get_result()
        assert func is not None
        printed = python_print(ir.Program([func], "p", span))
        assert 'pl.MemRef("pool", slots=2)[0]' in printed, printed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
