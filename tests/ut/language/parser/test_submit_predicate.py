# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Parser / IR tests for ``pl.spmd_submit(..., predicate=pl.dispatch_pred(...))``.

``pl.dispatch_pred(tensor, [indices], op, target)`` attaches a dispatch predicate
to a ``ir.Submit``: the scheduler evaluates ``tensor[indices] <op> target`` at the
dispatch point and retires the task inline (never dispatched to a core) when the
comparison is false, while still settling fanin/fanout. The predicate is stored on
first-class ``Submit.predicate_operand`` / ``predicate_indices`` / ``predicate_op``
/ ``predicate_target`` fields.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics.exceptions import ParserSyntaxError, ParserTypeError


def _flatten(stmt):
    if isinstance(stmt, ir.SeqStmts):
        out = []
        for s in stmt.stmts:
            out.extend(_flatten(s))
        return out
    if isinstance(stmt, ir.RuntimeScopeStmt):
        return _flatten(stmt.body)
    return [stmt]


def _main_submits(prog):
    fn = prog.get_function("main")
    assert fn is not None
    return [s.value for s in _flatten(fn.body) if isinstance(s, ir.AssignStmt) and isinstance(s.value, ir.Submit)]


def _program(predicate_src: str):
    """Build a two-kernel program whose expert submit carries ``predicate_src``.

    ``predicate_src`` is spliced verbatim as the ``predicate=`` argument text.
    """
    src = f'''
@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def expert(self, x: pl.Tensor[[512, 128], pl.FP32], out: pl.Out[pl.Tensor[[512, 128], pl.FP32]]) -> pl.Tensor[[512, 128], pl.FP32]:
        t = pl.load(x, [0, 0], [128, 128])
        out = pl.store(t, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def gate(self, g: pl.Out[pl.Tensor[[512, 128], pl.INT32]]) -> pl.Tensor[[512, 128], pl.INT32]:
        t = pl.load(g, [0, 0], [128, 128])
        g = pl.store(t, [0, 0], g)
        return g

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[512, 128], pl.FP32], out: pl.Out[pl.Tensor[[512, 128], pl.FP32]], rc: pl.Out[pl.Tensor[[512, 128], pl.INT32]]) -> pl.Tensor[[512, 128], pl.FP32]:
        with pl.manual_scope():
            rc, g_tid = pl.spmd_submit(self.gate, rc, core_num=1)
            out, _ = pl.spmd_submit(self.expert, x, out, core_num=1, deps=[g_tid], predicate={predicate_src})
        return out
'''
    return pl.parse_program(src)


def test_predicate_populates_submit_fields():
    prog = _program('pl.dispatch_pred(rc, [0, 0], ">", 0)')
    submits = _main_submits(prog)
    # gate submit has no predicate; expert submit carries the predicate.
    gate_sub, expert_sub = submits[0], submits[1]
    assert gate_sub.predicate_op == 0
    assert gate_sub.predicate_operand is None
    assert expert_sub.predicate_op == int(ir.DispatchPredicateOp.Gt.value)
    assert expert_sub.predicate_target == 0
    assert isinstance(expert_sub.predicate_operand, ir.Var)
    assert len(expert_sub.predicate_indices) == 2
    assert all(isinstance(i, ir.ConstInt) for i in expert_sub.predicate_indices)
    # Contract surface: the operand producer (gate) must be reachable via deps.
    assert len(expert_sub.deps) == 1


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("==", ir.DispatchPredicateOp.Eq),
        ("!=", ir.DispatchPredicateOp.Ne),
        (">", ir.DispatchPredicateOp.Gt),
        ("<", ir.DispatchPredicateOp.Lt),
        (">=", ir.DispatchPredicateOp.Ge),
        ("<=", ir.DispatchPredicateOp.Le),
    ],
)
def test_all_comparison_spellings(spelling, expected):
    prog = _program(f'pl.dispatch_pred(rc, [0, 0], "{spelling}", 3)')
    expert_sub = _main_submits(prog)[1]
    assert expert_sub.predicate_op == int(expected.value)
    assert expert_sub.predicate_target == 3


def test_negative_target_literal():
    prog = _program('pl.dispatch_pred(rc, [0, 0], ">=", -5)')
    assert _main_submits(prog)[1].predicate_target == -5


def test_bad_op_spelling_rejected():
    with pytest.raises(ParserSyntaxError, match="not supported"):
        _program('pl.dispatch_pred(rc, [0, 0], "=<", 0)')


def test_non_tensor_operand_rejected():
    # ``g_tid`` is a Scalar[TASK_ID], not a tensor.
    with pytest.raises(ParserTypeError, match="operand must be a tensor"):
        _program('pl.dispatch_pred(g_tid, [0], ">", 0)')


def test_indices_must_be_list_literal():
    with pytest.raises(ParserSyntaxError, match="indices must be a list"):
        _program('pl.dispatch_pred(rc, 0, ">", 0)')


def test_non_literal_target_rejected():
    with pytest.raises(ParserSyntaxError, match="target must be an integer literal"):
        _program('pl.dispatch_pred(rc, [0, 0], ">", 1.5)')


def test_predicate_must_be_dispatch_pred_call():
    with pytest.raises(ParserSyntaxError, match="must be a pl.dispatch_pred"):
        _program("rc")


def test_no_predicate_leaves_fields_default():
    prog = _program('pl.dispatch_pred(rc, [0, 0], ">", 0)')
    gate_sub = _main_submits(prog)[0]
    assert not gate_sub.predicate_op  # 0 == no predicate
    assert gate_sub.predicate_indices == []


def test_print_parse_round_trip():
    from pypto.ir import python_print

    prog = _program('pl.dispatch_pred(rc, [0, 0], ">=", 2)')
    printed = python_print(prog)
    # The predicate surfaces on the submit line for the round trip.
    assert "predicate=pl.dispatch_pred(" in printed
    reparsed = pl.parse_program(printed)
    ir.assert_structural_equal(reparsed, prog)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
