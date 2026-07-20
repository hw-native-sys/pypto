# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Parser / IR tests for ``pl.spmd_submit(..., predicate=(tensor[i] > 0))``.

``predicate=`` attaches a dispatch predicate to a ``ir.Submit``: the scheduler
evaluates ``tensor[indices] <op> target`` at the dispatch point and retires the
task inline (never dispatched to a core) when the comparison is false, while
still settling fanin/fanout. The predicate is stored on first-class
the first-class ``Submit.predicate`` field as an ordinary comparison Expr —
``Gt(Cast(tensor.read(rc, [0, 0])), 0)`` — reusing the IR's existing comparison
nodes and ``tensor.read`` rather than a bespoke encoding. Decomposition into the
runtime's ``operand OP target`` triple is orchestration-codegen's job.

The comparison is matched syntactically, never evaluated — in this position
``rc[0, 0] > 0`` is a declarative spec, not a ``tensor.read`` plus a compare.
Only ``tensor[indices] OP int-literal`` is expressible, mirroring the runtime's
single-comparison predicate.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.ir import python_print
from pypto.language.parser.diagnostics.exceptions import (
    ParserSyntaxError,
    ParserTypeError,
    UnsupportedFeatureError,
)


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
    stmts = _flatten(fn.body)
    return [s.value for s in stmts if isinstance(s, ir.AssignStmt) and isinstance(s.value, ir.Submit)]


_FP32_T = "pl.Tensor[[512, 128], pl.FP32]"
_INT32_T = "pl.Tensor[[512, 128], pl.INT32]"


def _pred_read(p) -> ir.Call:
    """Return the tensor.read Call inside a predicate Expr (either operand order)."""

    def strip_cast(e):
        while isinstance(e, ir.Cast):
            e = e.operand
        return e

    def is_read(e):
        return isinstance(e, ir.Call) and e.op.name == "tensor.read"

    lhs, rhs = strip_cast(p.left), strip_cast(p.right)
    read = lhs if is_read(lhs) else rhs
    assert isinstance(read, ir.Call)
    return read


def _pred_indices(p) -> list:
    """Per-axis indices of the predicate's tensor.read.

    ``tensor.read`` takes ``(tensor, indices)`` where ``indices`` is a MakeTuple
    for a multi-axis read and a bare expression for a single axis.
    """
    read = _pred_read(p)
    idx = read.args[1]
    return list(idx.elements) if isinstance(idx, ir.MakeTuple) else [idx]


def _pred_const(p) -> ir.ConstInt:
    """Return the ConstInt side of a predicate Expr."""

    def strip_cast(e):
        while isinstance(e, ir.Cast):
            e = e.operand
        return e

    lhs, rhs = strip_cast(p.left), strip_cast(p.right)
    konst = lhs if isinstance(lhs, ir.ConstInt) else rhs
    assert isinstance(konst, ir.ConstInt)
    return konst


def _program(predicate_src: str, deps_src: str = "[g_tid]", rc_dtype: str = "pl.INT32"):
    """Build a two-kernel program whose expert submit carries ``predicate_src``.

    ``predicate_src`` is spliced verbatim as the ``predicate=`` argument text,
    ``deps_src`` as the ``deps=`` argument text. The expert submit's predicate
    reads ``rc``, which the gate submit (bound to ``g_tid``) produces — so the
    default ``deps_src`` satisfies the "producer must be in deps=" contract.
    """
    src = f"""
@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def expert(self, x: {_FP32_T}, out: pl.Out[{_FP32_T}]) -> {_FP32_T}:
        t = pl.load(x, [0, 0], [128, 128])
        out = pl.store(t, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def gate(self, g: pl.Out[pl.Tensor[[512, 128], {rc_dtype}]]) -> pl.Tensor[[512, 128], {rc_dtype}]:
        t = pl.load(g, [0, 0], [128, 128])
        g = pl.store(t, [0, 0], g)
        return g

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: {_FP32_T},
        out: pl.Out[{_FP32_T}],
        rc: pl.Out[pl.Tensor[[512, 128], {rc_dtype}]],
        rc_in: {_INT32_T},
    ) -> {_FP32_T}:
        with pl.manual_scope():
            rc, g_tid = pl.spmd_submit(self.gate, rc, core_num=1)
            out, _ = pl.spmd_submit(
                self.expert, x, out, core_num=1, deps={deps_src}, predicate={predicate_src}
            )
        return out
"""
    return pl.parse_program(src)


def test_predicate_populates_submit_fields():
    prog = _program("rc[0, 0] > 0")
    submits = _main_submits(prog)
    # gate submit has no predicate; expert submit carries the predicate.
    gate_sub, expert_sub = submits[0], submits[1]
    assert gate_sub.predicate is None
    # The predicate is a plain comparison Expr over a tensor.read — no bespoke encoding.
    pred = expert_sub.predicate
    assert isinstance(pred, ir.Gt)
    read = _pred_read(pred)
    assert read.op.name == "tensor.read"
    assert isinstance(read.args[0], ir.Var)  # operand tensor
    assert len(_pred_indices(pred)) == 2  # one index per rank-2 axis
    assert _pred_const(pred).value == 0
    # Contract surface: the operand producer (gate) must be reachable via deps.
    assert len(expert_sub.deps) == 1


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("==", ir.Eq),
        ("!=", ir.Ne),
        (">", ir.Gt),
        ("<", ir.Lt),
        (">=", ir.Ge),
        ("<=", ir.Le),
    ],
)
def test_all_comparison_spellings(spelling, expected):
    prog = _program(f"rc[0, 0] {spelling} 3")
    expert_sub = _main_submits(prog)[1]
    assert isinstance(expert_sub.predicate, expected)
    assert _pred_const(expert_sub.predicate).value == 3


def test_negative_target_literal():
    prog = _program("rc[0, 0] >= -5")
    assert _pred_const(_main_submits(prog)[1].predicate).value == -5


def test_explicitly_positive_target_literal():
    prog = _program("rc[0, 0] >= +5")
    assert _pred_const(_main_submits(prog)[1].predicate).value == 5


def test_unsupported_comparison_operator_rejected():
    # ``in`` is a Compare op the DSL has no IR node for.
    with pytest.raises(UnsupportedFeatureError, match="Unsupported comparison"):
        _program("rc[0, 0] in x")


def test_chained_comparison_rejected():
    # The runtime evaluates exactly one comparison.
    with pytest.raises(ParserSyntaxError, match="Only simple comparisons supported"):
        _program("0 < rc[0, 0] < 8")


def test_reversed_operand_order_is_normalized():
    # ``0 < rc[e]`` means the same as ``rc[e] > 0``; the tensor must end up as
    # the operand and the operator is flipped to match.
    prog = _program("0 < rc[0, 0]")
    expert_sub = _main_submits(prog)[1]
    # `0 < rc[0,0]` keeps its written `Lt` kind in the IR; orchestration codegen
    # flips it to the runtime's `operand OP target` orientation (GT).
    assert isinstance(expert_sub.predicate, ir.Lt)
    assert _pred_read(expert_sub.predicate).op.name == "tensor.read"
    assert _pred_const(expert_sub.predicate).value == 0


def test_reversed_operand_order_flips_asymmetric_op():
    # ``5 >= rc[e]`` normalizes to ``rc[e] <= 5``.
    prog = _program("5 >= rc[0, 0]")
    expert_sub = _main_submits(prog)[1]
    # `5 >= rc[0,0]` stays `Ge` in the IR; codegen flips it to LE.
    assert isinstance(expert_sub.predicate, ir.Ge)
    assert _pred_const(expert_sub.predicate).value == 5


def test_non_tensor_operand_rejected():
    # ``g_tid`` is a Scalar[TASK_ID], not a tensor — rejected by the DSL's own
    # subscript typing, before any predicate-specific check.
    with pytest.raises(ParserTypeError, match="Subscript requires Tuple, Tensor, Tile, or Array"):
        _program("g_tid[0] > 0")


def test_bare_tensor_operand_rejected():
    # The predicate must locate one *element*; comparing a whole tensor is not a
    # scalar comparison, so ordinary expression typing rejects it.
    with pytest.raises(ParserSyntaxError, match="must be ScalarExpr or Var with ScalarType"):
        _program("rc > 0")


def test_tensor_index_rejected():
    # ``x`` is a tensor param — it would otherwise render into the runtime
    # predicate index array as ``ext_x``, emitting invalid C++.
    with pytest.raises(ParserSyntaxError, match="index element 0 must be ScalarType"):
        _program("rc[x, 0] > 0")


def test_index_count_must_match_operand_rank():
    # ``rc`` is rank-2; a single index yields a rank-1 view, not an element, so
    # the comparison is not scalar-vs-scalar.
    with pytest.raises(ParserSyntaxError, match="must be ScalarExpr or Var with ScalarType"):
        _program("rc[0] > 0")


def test_predicate_operand_producer_must_be_in_deps():
    # ``rc`` is produced by the gate submit (``g_tid``). Dropping it from deps=
    # would let the scheduler evaluate the predicate against stale data.
    with pytest.raises(ParserSyntaxError, match="not in deps="):
        _program("rc[0, 0] > 0", deps_src="[]")


def test_predicate_operand_without_tracked_producer_allowed():
    # ``rc_in`` is a function parameter, not a submit result — nothing to prove,
    # so the deps= contract check stays out of the way.
    prog = _program("rc_in[0, 0] > 0", deps_src="[]")
    assert isinstance(_main_submits(prog)[1].predicate, ir.Gt)


def test_non_literal_target_rejected():
    # int32 element vs a float literal — rejected by ordinary operand typing.
    with pytest.raises(ParserSyntaxError, match="requires same numeric dtype category"):
        _program("rc[0, 0] > 1.5")


def test_non_literal_tensor_rhs_rejected():
    # ``t[i] > u[j]`` — the runtime compares against a constant, not another
    # tensor element. Here the two elements also differ in dtype (int32 vs fp32).
    with pytest.raises(ParserSyntaxError, match="requires same numeric dtype category"):
        _program("rc[0, 0] > x[0, 0]")


def test_predicate_must_be_a_comparison():
    with pytest.raises(ParserSyntaxError, match="must be a single comparison"):
        _program("rc")


def test_no_predicate_leaves_fields_default():
    prog = _program("rc[0, 0] > 0")
    gate_sub = _main_submits(prog)[0]
    assert gate_sub.predicate is None


def test_print_parse_round_trip():
    prog = _program("rc[0, 0] >= 2")
    printed = python_print(prog)
    # The predicate surfaces on the submit line as the comparison expression.
    assert "predicate=(" in printed
    assert ">= 2)" in printed
    reparsed = pl.parse_program(printed)
    ir.assert_structural_equal(reparsed, prog)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Runtime-ABI safety: operand dtype and index range
# ---------------------------------------------------------------------------
#
# DispatchPredicate::pass() reads `elem_size` bytes and **sign-extends** to
# int64 before comparing. An unsigned operand with the top bit set therefore
# compares as negative and silently inverts the dispatch decision — a UINT32
# row_count of 3_000_000_000 reads back as -1_294_967_296, so `row_count[e] > 0`
# is false and the expert is skipped with no diagnostic at all. Sub-byte dtypes
# have no addressable single-element read.


@pytest.mark.parametrize("dtype", ["pl.UINT8", "pl.UINT16", "pl.UINT32", "pl.UINT64", "pl.INT4"])
def test_unsigned_and_subbyte_operand_rejected(dtype):
    with pytest.raises(ParserTypeError, match="signed 8/16/32/64-bit integer"):
        _program("rc[0, 0] > 0", rc_dtype=dtype)


@pytest.mark.parametrize("dtype", ["pl.INT8", "pl.INT16", "pl.INT32", "pl.INT64"])
def test_signed_integer_operands_accepted(dtype):
    prog = _program("rc[0, 0] > 0", rc_dtype=dtype)
    assert isinstance(_main_submits(prog)[1].predicate, ir.Gt)


def test_negative_constant_index_rejected():
    # L0PredicateOperand::indices is uint32_t — a negative index wraps to a huge
    # value and yields an out-of-bounds GM address read at the dispatch point.
    with pytest.raises(ParserTypeError, match="index must be non-negative"):
        _program("rc[-1, 0] > 0")


def test_rebound_taskid_is_not_accepted_as_producer_dep():
    """A rebound TaskId name must not certify an unrelated producer.

    Strict SSA reuses one ``ir.Var`` object across same-named rebindings, so
    object identity alone would match here even though ``deps=[tid]`` now refers
    to the *second* gate submit and the predicate's producer is the first.
    """
    src = """
@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def gate(
        self, g: pl.Out[pl.Tensor[[512, 128], pl.INT32]]
    ) -> pl.Tensor[[512, 128], pl.INT32]:
        t = pl.load(g, [0, 0], [128, 128])
        g = pl.store(t, [0, 0], g)
        return g

    @pl.function(type=pl.FunctionType.InCore)
    def expert(
        self, x: pl.Tensor[[512, 128], pl.FP32], out: pl.Out[pl.Tensor[[512, 128], pl.FP32]]
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        t = pl.load(x, [0, 0], [128, 128])
        out = pl.store(t, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        rc: pl.Out[pl.Tensor[[512, 128], pl.INT32]],
        fb: pl.Out[pl.Tensor[[512, 128], pl.INT32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        with pl.manual_scope():
            rc, tid = pl.spmd_submit(self.gate, rc, core_num=1)
            fb, tid = pl.spmd_submit(self.gate, fb, core_num=1)
            out, _ = pl.spmd_submit(
                self.expert, x, out, core_num=1, deps=[tid], predicate=(rc[0, 0] > 0)
            )
        return out
"""
    with pytest.raises(ParserSyntaxError, match="not in deps="):
        pl.parse_program(src)


# ---------------------------------------------------------------------------
# Scope form — ``with pl.spmd(..., predicate=...)``
# ---------------------------------------------------------------------------
#
# The scope form shares every validation helper with pl.spmd_submit (the shape
# checks above are not re-run per form), but reaches the IR by a different
# route: the predicate rides on ``SpmdScopeStmt.attrs`` as an Expr until
# OutlineSpmdScopes moves it onto ``Submit.predicate``. These tests cover the
# attr landing, all three spmd spellings, the print round-trip, and the two
# rejections specific to a scope (cluster nesting, producer-not-in-deps).

_SCOPE_HEAD = f"""
@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def expert(self, x: {_FP32_T}, out: pl.Out[{_FP32_T}]) -> {_FP32_T}:
        t = pl.load(x, [0, 0], [128, 128])
        out = pl.store(t, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def gate(self, g: pl.Out[{_INT32_T}]) -> {_INT32_T}:
        t = pl.load(g, [0, 0], [128, 128])
        g = pl.store(t, [0, 0], g)
        return g
"""


def _scope_program(predicate_src: str = "rc[0, 0] > 0", deps_src: str = "deps=[g_tid], "):
    """Two spmd scopes; the second carries ``predicate_src`` (as-tid form)."""
    return pl.parse_program(
        _SCOPE_HEAD
        + f"""
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, x: {_FP32_T}, out: pl.Out[{_FP32_T}], rc: pl.Out[{_INT32_T}]
    ) -> {_FP32_T}:
        with pl.spmd(1) as g_tid:
            rc = self.gate(rc)
        with pl.spmd(1, {deps_src}predicate=({predicate_src})) as t:
            out = self.expert(x, out)
        return out
"""
    )


def _spmd_scopes(prog):
    """All SpmdScopeStmts in ``main``, in source order."""
    found = []

    def walk(stmt):
        if isinstance(stmt, ir.SpmdScopeStmt):
            found.append(stmt)
        for field in ("stmts", "body"):
            value = getattr(stmt, field, None)
            if isinstance(value, list):
                for child in value:
                    walk(child)
            elif value is not None:
                walk(value)

    walk(prog.get_function("main").body)
    return found


def test_scope_predicate_lands_on_scope_attrs():
    """The predicate is stored on the scope as the comparison Expr itself."""
    prog = _scope_program("rc[0, 0] > 0")
    gate_scope, expert_scope = _spmd_scopes(prog)
    assert "predicate" not in dict(gate_scope.attrs.items())
    predicate = dict(expert_scope.attrs.items())["predicate"]
    assert isinstance(predicate, ir.Gt)
    assert _pred_const(predicate).value == 0
    # Reuses tensor.read rather than a bespoke encoding.
    assert _pred_read(predicate).op.name == "tensor.read"
    assert [c.value for c in _pred_indices(predicate)] == [0, 0]


def test_scope_predicate_canonical_attr_order():
    """deps -> task_id_var -> predicate; the print round-trip relies on it."""
    prog = _scope_program()
    _, expert_scope = _spmd_scopes(prog)
    assert [k for k, _ in expert_scope.attrs.items()] == [
        "manual_dep_edges",
        "task_id_var",
        "predicate",
    ]


def test_scope_predicate_print_parse_round_trip():
    prog = _scope_program("rc[0, 0] >= 2")
    printed = python_print(prog)
    assert "predicate=(" in printed
    reparsed = pl.parse_program(printed)
    ir.assert_structural_equal(reparsed, prog)


def test_scope_predicate_plain_with_form():
    """No ``as tid``: the predicate still lands (the outliner synthesises a tid).

    ``rc`` is a plain parameter here, so it has no tracked producer and the
    deps contract check passes without a ``deps=`` (unavailable on this form).
    """
    prog = pl.parse_program(
        _SCOPE_HEAD
        + f"""
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, x: {_FP32_T}, out: pl.Out[{_FP32_T}], rc: {_INT32_T}
    ) -> {_FP32_T}:
        with pl.spmd(1, predicate=(rc[0, 0] > 0)):
            out = self.expert(x, out)
        return out
"""
    )
    (scope,) = _spmd_scopes(prog)
    assert isinstance(dict(scope.attrs.items())["predicate"], ir.Gt)


def test_scope_predicate_for_form():
    """``for i in pl.spmd(n, predicate=...)`` records the predicate too."""
    prog = pl.parse_program(
        _SCOPE_HEAD
        + f"""
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, x: {_FP32_T}, out: pl.Out[{_FP32_T}], rc: {_INT32_T}
    ) -> {_FP32_T}:
        for i in pl.spmd(4, predicate=(rc[0, 0] > 0)):
            t = pl.load(x, [i * 128, 0], [128, 128])
            out = pl.store(t, [i * 128, 0], out)
        return out
"""
    )
    (scope,) = _spmd_scopes(prog)
    assert isinstance(dict(scope.attrs.items())["predicate"], ir.Gt)


def test_scope_predicate_operand_producer_must_be_in_deps():
    """Same contract as the call form: omitting deps= is caught."""
    with pytest.raises(ParserSyntaxError, match="produced by the task"):
        _scope_program("rc[0, 0] > 0", deps_src="")


def test_scope_predicate_rejected_inside_cluster():
    """A cluster-nested pl.spmd never becomes a Submit, so the predicate is lost."""
    with pytest.raises(ParserSyntaxError, match="cannot be nested inside `pl.cluster"):
        pl.parse_program(
            _SCOPE_HEAD
            + f"""
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, x: {_FP32_T}, out: pl.Out[{_FP32_T}], rc: {_INT32_T}
    ) -> {_FP32_T}:
        with pl.cluster():
            with pl.spmd(1, predicate=(rc[0, 0] > 0)):
                out = self.expert(x, out)
        return out
"""
        )


def test_scope_predicate_shape_validation_is_shared():
    """The call form's shape rules apply verbatim — spot-check two."""
    with pytest.raises(ParserSyntaxError, match="must compare one tensor element"):
        _scope_program("rc[0, 0] % 8 == 0")
    with pytest.raises(ParserSyntaxError, match="Only simple comparisons"):
        _scope_program("0 < rc[0, 0] < 8")


def test_scope_without_predicate_has_no_attr():
    prog = pl.parse_program(
        _SCOPE_HEAD
        + f"""
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: {_FP32_T}, out: pl.Out[{_FP32_T}]) -> {_FP32_T}:
        with pl.spmd(1):
            out = self.expert(x, out)
        return out
"""
    )
    (scope,) = _spmd_scopes(prog)
    assert "predicate" not in dict(scope.attrs.items())


def test_scope_producer_is_tracked_for_the_deps_contract():
    """A scope-produced operand IS tracked, so ``deps=`` is genuinely enforced.

    The rejection test above only proves *something* raised; this pins the
    mechanism: naming the producing scope in ``deps=`` accepts, and the same
    program without it rejects. Before scope-producer tracking existed both
    spellings parsed, silently certifying a stale-read predicate.
    """
    # Producer named in deps= -> accepted.
    prog = _scope_program("rc[0, 0] > 0", deps_src="deps=[g_tid], ")
    _, expert_scope = _spmd_scopes(prog)
    assert isinstance(dict(expert_scope.attrs.items())["predicate"], ir.Gt)
    # Same program, producer omitted -> rejected, naming the producing task.
    with pytest.raises(ParserSyntaxError, match="produced by the task"):
        _scope_program("rc[0, 0] > 0", deps_src="")


def test_scope_producer_tracking_survives_tid_rebinding():
    """A rebound ``tid`` name must not certify an unrelated scope.

    Strict SSA reuses one ``ir.Var`` object across same-named rebindings, so
    identity alone would match; the per-Var generation counter is what makes
    this reject.
    """
    with pytest.raises(ParserSyntaxError, match="produced by the task"):
        pl.parse_program(
            _SCOPE_HEAD
            + f"""
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, x: {_FP32_T}, out: pl.Out[{_FP32_T}], rc: pl.Out[{_INT32_T}],
        rc2: pl.Out[{_INT32_T}]
    ) -> {_FP32_T}:
        with pl.spmd(1) as tid:
            rc = self.gate(rc)
        with pl.spmd(1) as tid:
            rc2 = self.gate(rc2)
        with pl.spmd(1, deps=[tid], predicate=(rc[0, 0] > 0)) as t:
            out = self.expert(x, out)
        return out
"""
        )
