# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public L4 ctx propagation with the default verification/roundtrip instruments.

Native Expr-valued attribute fixtures live in test_materialize_dist_tensor_ctx;
these DSL programs deliberately retain the shared default PassContext fixture.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


def _function(program: ir.Program, name: str) -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None
    return program.functions[gvar]


def _statements(stmt: ir.Stmt):
    yield stmt
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            yield from _statements(child)
    elif isinstance(stmt, ir.IfStmt):
        yield from _statements(stmt.then_body)
        if stmt.else_body is not None:
            yield from _statements(stmt.else_body)
    elif isinstance(stmt, (ir.ScopeStmt, ir.ForStmt, ir.WhileStmt)):
        yield from _statements(stmt.body)


def _calls(func: ir.Function, name: str) -> list[ir.Call]:
    result = []
    for stmt in _statements(func.body):
        expr = stmt.value if isinstance(stmt, ir.AssignStmt) else None
        if isinstance(stmt, ir.EvalStmt):
            expr = stmt.expr
        if isinstance(expr, ir.Call) and expr.op.name == name:
            result.append(expr)
    return result


def _pipeline(program: ir.Program) -> ir.Program:
    for transform in (
        passes.convert_to_ssa,
        passes.synthesize_allreduce_signals,
        passes.materialize_comm_domain_scopes,
        passes.lower_host_tensor_collectives,
        passes.derive_call_directions,
        passes.materialize_dist_tensor_ctx,
    ):
        program = transform()(program)
    return program


@pytest.mark.parametrize("explicit_query", [False, True])
@pytest.mark.parametrize("roundtrip", ["text", "serde"])
def test_c2b_l4_local_and_inherited_ctx_with_shuffled_definitions(explicit_query, roundtrip):
    query = "        ctx = pld.system.get_comm_ctx(inherited)\n        rank = pld.system.rank(ctx)\n"
    before = pl.parse_program(
        """
@pl.program
class Before:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def m_chip(self, inherited: pld.DistributedTensor[[4], pl.FP32],
               local: pld.DistributedTensor[[4], pl.FP32]):
        inherited_ctx = pld.system.get_comm_ctx(inherited)
        local_ctx = pld.system.get_comm_ctx(local)
        return inherited, local
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def a_host(self, inherited: pld.DistributedTensor[[4], pl.FP32]):
        intra_buf = pld.alloc_window_buffer(16)
        local = pld.window(intra_buf, [4], dtype=pl.FP32)
        self.m_chip(inherited, local, worker=0)
        return inherited
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def z_root(self):
        inter_buf = pld.alloc_window_buffer(16)
        inherited = pld.window(inter_buf, [4], dtype=pl.FP32)
"""
        + (query if explicit_query else "")
        + "        self.a_host(inherited, worker=0)\n        return 0\n"
    )
    after = _pipeline(before)
    restored = (
        pl.parse_program(ir.python_print(after, format=False))
        if roundtrip == "text"
        else ir.deserialize(ir.serialize(after))
    )
    ir.assert_structural_equal(after, restored)
    root, host, chip = [_function(restored, name) for name in ("z_root", "a_host", "m_chip")]
    root_scope = next(stmt for stmt in _statements(root.body) if isinstance(stmt, ir.CommDomainScopeStmt))
    host_scope = next(stmt for stmt in _statements(host.body) if isinstance(stmt, ir.CommDomainScopeStmt))
    inherited_slot, local_slot = root_scope.slots[0], host_scope.slots[0]
    assert inherited_slot is not local_slot
    assert host.params[0].type.window_buffer is chip.params[0].type.window_buffer is inherited_slot
    assert chip.params[1].type.window_buffer is local_slot
    root_call, host_call = _calls(root, "a_host")[0], _calls(host, "m_chip")[0]
    assert len(root_call.args) == 2
    assert len(host_call.args) == 4
    assert host_call.args[2] is host.params[-1]
    assert all(isinstance(arg.type, ir.CommCtxType) for arg in host_call.args[-2:])
    assert list(host_call.arg_directions)[-2:] == [ir.ArgDirection.Scalar] * 2
    query_name = ir.get_op("pld.system.get_comm_ctx").name
    assert len(_calls(root, query_name)) == (2 if explicit_query else 1)
    assert len(_calls(host, query_name)) == 1
    assert _calls(chip, query_name) == []
    props = passes.IRPropertySet()
    for prop in (
        passes.IRProperty.DistTensorCtxMaterialized,
        passes.IRProperty.AssignTypeSymmetry,
        passes.IRProperty.TypeChecked,
    ):
        props.insert(prop)
    assert passes.PropertyVerifierRegistry.verify(props, restored) == []


@pytest.mark.parametrize("lineage", ["view", "writeback", "for", "while", "if"])
def test_c2b_l4_ctx_follows_inherited_lineage(lineage):
    bodies = {
        "view": "        alias = data\n        result = pl.slice(alias, [4], [0])\n",
        "writeback": (
            "        view = pl.slice(data, [4], [0])\n        result = pl.assemble(data, view, [0])\n"
        ),
        "for": (
            "        for _, (carry,) in pl.range(2, init_values=(data,)):\n"
            "            result = pl.yield_(carry)\n"
        ),
        "while": (
            "        for (carry,) in pl.while_(init_values=(data,)):\n"
            "            pl.cond(flag)\n            result = pl.yield_(carry)\n"
        ),
        "if": (
            "        if flag:\n            result = pl.yield_(data)\n"
            "        else:\n            result = pl.yield_(data)\n"
        ),
    }
    before = pl.parse_program(
        """
@pl.program
class Before:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, data: pld.DistributedTensor[[4], pl.FP32]):
        return data
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(self, data: pld.DistributedTensor[[4], pl.FP32], flag: pl.Scalar[pl.BOOL]):
"""
        + bodies[lineage]
        + """
        self.chip(result, worker=0)
        return result
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def root(self, flag: pl.Scalar[pl.BOOL]):
        buf = pld.alloc_window_buffer(16)
        data = pld.window(buf, [4], dtype=pl.FP32)
        self.host(data, flag, worker=0)
        return 0
"""
    )
    after = _pipeline(before)
    host = _function(after, "host")
    call = _calls(host, "chip")[0]
    assert len(call.args) == 2
    assert call.args[-1] is host.params[-1]
    assert _calls(host, ir.get_op("pld.system.get_comm_ctx").name) == []
    assert call.args[0].type.window_buffer is host.params[0].type.window_buffer


def test_c2b_l4_query_only_program_without_ctx_plans():
    """The ctx fast path permits an L4 query without testing domain creation.

    There is no dispatch to cover this window with a communication domain; this
    case only exercises ctx materialization and its independent property, using
    the shared default verification/roundtrip instruments.
    """
    before = pl.parse_program("""
@pl.program
class Before:
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def root(self):
        buf = pld.alloc_window_buffer(16)
        data = pld.window(buf, [4], dtype=pl.FP32)
        ctx = pld.system.get_comm_ctx(data)
        rank = pld.system.rank(ctx)
        return rank
""")
    after = passes.materialize_dist_tensor_ctx()(passes.convert_to_ssa()(before))
    root = _function(after, "root")
    assert list(root.params) == []
    assert len(_calls(root, ir.get_op("pld.system.get_comm_ctx").name)) == 1
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.DistTensorCtxMaterialized)
    assert passes.PropertyVerifierRegistry.verify(props, after) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
