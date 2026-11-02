# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Region validation must cover manually constructed IR as well as DSL."""

import pypto.language as pl
import pytest
from pypto import DataType, backend, codegen, ir, passes

SPAN = ir.Span("reg_test.py", 3, 1, 3, 10)


def scalar(name):
    return ir.Var(name, ir.ScalarType(DataType.INDEX), SPAN)


def integer(value):
    return ir.ConstInt(value, DataType.INDEX, SPAN)


def region(body):
    return ir.RegScopeStmt("", body, SPAN)


def sequence(*stmts):
    return ir.SeqStmts(list(stmts), SPAN)


def register(name="r"):
    return ir.Var(name, ir.RegType(64, DataType.BOOL, ir.RegKind.MaskReg), SPAN)


def definition(value):
    return ir.AssignStmt(value, ir.create_op_call("reg.full_mask", [], {"lanes": 64}, SPAN), SPAN)


def verify(body, func_type=ir.FunctionType.InCore, final=False, params=()):
    function = ir.Function("kernel", list(params), [], body, SPAN, type=func_type)
    program = ir.Program([function], "RegTest", SPAN)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.MixedKernelExpanded if final else passes.IRProperty.RegScopeValid)
    return passes.PropertyVerifierRegistry.verify(properties, program)


@pytest.mark.parametrize("kind", [ir.FunctionType.InCore, ir.FunctionType.AIV])
def test_legal_region_early(kind):
    r = register()
    assert verify(region(definition(r)), kind) == []


@pytest.mark.parametrize("wrapper", ["direct", "if", "for"])
def test_while_inside_register_scope_is_rejected(wrapper):
    backend.set_backend_type(backend.BackendType.Ascend950)
    body = ir.WhileStmt(ir.ConstBool(False, SPAN), [], sequence(), [], SPAN)
    if wrapper == "if":
        body = ir.IfStmt(ir.ConstBool(True, SPAN), body, None, [], SPAN)
    elif wrapper == "for":
        body = ir.ForStmt(scalar("i"), integer(0), integer(1), integer(1), [], body, [], SPAN)
    diagnostics = verify(region(body))
    assert len(diagnostics) == 1
    assert "WhileStmt is not supported inside RegScope" in diagnostics[0].message
    function = ir.Function("kernel", [], [], region(body), SPAN, type=ir.FunctionType.AIV)
    with pytest.raises(ValueError, match="WhileStmt.*RegScope"):
        codegen.PTOCodegen().generate(ir.Program([function], "RegTest", SPAN))


def test_while_outside_register_scope_is_allowed():
    body = ir.WhileStmt(ir.ConstBool(False, SPAN), [], region(definition(register())), [], SPAN)
    assert verify(body) == []


@pytest.mark.parametrize(
    "kind",
    [
        ir.FunctionType.AIC,
        ir.FunctionType.Group,
        ir.FunctionType.Spmd,
        ir.FunctionType.Orchestration,
        ir.FunctionType.Graph,
        ir.FunctionType.Opaque,
    ],
)
def test_invalid_function_context(kind):
    assert any("context" in d.message for d in verify(region(definition(register())), kind))


@pytest.mark.parametrize(
    "kind", [ir.FunctionType.Opaque, ir.FunctionType.Orchestration, ir.FunctionType.Graph]
)
@pytest.mark.parametrize("graph_ancestor", [False, True])
@pytest.mark.parametrize("tuple_depth", [0, 1, 2])
def test_explicit_incore_register_region_survives_outlining(kind, graph_ancestor, tuple_depth):
    # The device boundary is already supported by outlining for these owners;
    # RegScope validation must not impose a narrower owner whitelist.
    value = register()
    stmts = [definition(value)]
    for depth in range(tuple_depth):
        packed = ir.Var(f"packed_{depth}", ir.TupleType([value.type]), SPAN)
        stmts.append(ir.AssignStmt(packed, ir.MakeTuple([value], SPAN), SPAN))
        value = packed
    local_body = stmts[0] if len(stmts) == 1 else sequence(*stmts)
    body = ir.InCoreScopeStmt(body=region(local_body), span=SPAN)
    if graph_ancestor:
        body = ir.GraphScopeStmt("graph", body=sequence(body, ir.EvalStmt(integer(0), SPAN)), span=SPAN)
    assert verify(body, kind) == []
    # Keep the scope terminal while retaining a second statement so the
    # outliner must use its fallback output analysis. Reg locals must not be
    # exported merely because no following variable use exists.
    function = ir.Function("kernel", [], [], sequence(body, ir.EvalStmt(integer(0), SPAN)), SPAN, type=kind)
    program = ir.Program([function], "RegTest", SPAN)
    outlined = passes.outline_incore_scopes()(program)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.RegScopeValid)
    assert passes.PropertyVerifierRegistry.verify(properties, outlined) == []
    kernels = [f for f in outlined.functions.values() if f.func_type == ir.FunctionType.InCore]
    assert len(kernels) == 1
    assert isinstance(kernels[0].body, ir.RegScopeStmt)


@pytest.mark.parametrize("tuple_depth", [1, 2])
@pytest.mark.parametrize("inside_region", [False, True])
def test_terminal_scalar_tuple_is_not_a_fallback_output(tuple_depth, inside_region):
    value = integer(64)
    stmts = []
    for depth in range(tuple_depth):
        packed = ir.Var(f"packed_{depth}", ir.TupleType([value.type]), SPAN)
        stmts.append(ir.AssignStmt(packed, ir.MakeTuple([value], SPAN), SPAN))
        value = packed
    local = stmts[0] if len(stmts) == 1 else sequence(*stmts)
    scope_body = region(local) if inside_region else local
    outer_stmts = [scope_body] if inside_region else stmts
    outer_stmts.append(ir.EvalStmt(integer(0), SPAN))
    function = ir.Function(
        "kernel",
        [],
        [],
        ir.InCoreScopeStmt(body=sequence(*outer_stmts), span=SPAN),
        SPAN,
        type=ir.FunctionType.Orchestration,
    )
    program = ir.Program([function], "RegTest", SPAN)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.RegScopeValid)
    assert passes.PropertyVerifierRegistry.verify(properties, program) == []
    # Main excludes all unused tuple temporaries from fallback outputs, whether
    # they are defined inside RegScope or directly in the enclosing InCore scope.
    outlined = passes.outline_incore_scopes()(program)
    assert passes.PropertyVerifierRegistry.verify(properties, outlined) == []
    kernel = next(f for f in outlined.functions.values() if f.func_type == ir.FunctionType.InCore)
    assert kernel.return_types == []


def test_terminal_register_region_scalar_tuple_unpacking():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self):
            with pl.at(level=pl.Level.CORE_GROUP):
                with pl.reg_scope():
                    _lo, _hi = (0, 64)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def main_incore_0(self):
            with pl.reg_scope():
                _lo, _hi = (0, 64)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self):
            self.main_incore_0()

    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.RegScopeValid)
    assert passes.PropertyVerifierRegistry.verify(properties, Before) == []
    after = passes.outline_incore_scopes()(Before)
    # Tuple unpacking creates a scalar-containing _tuple_tmp. Outlining must
    # leave it local instead of introducing a return outside RegScope.
    assert passes.PropertyVerifierRegistry.verify(properties, after) == []
    ir.assert_structural_equal(after, Expected)


def test_terminal_register_region_preserves_tile_output():
    # A store's Tile alias is a legal output even though its Reg operand must
    # remain local to the outlined RegScope.
    tile_type = ir.TileType([64], DataType.FP32, memory_space=ir.MemorySpace.Vec)
    tile = ir.Var("tile", tile_type, SPAN)
    value = ir.Var("value", ir.RegType(64, DataType.FP32, ir.RegKind.VecReg), SPAN)
    out = ir.Var("out", tile_type, SPAN)
    load = ir.create_op_call("reg.load", [tile, integer(0)], {"lanes": 64}, SPAN)
    store = ir.create_op_call("reg.store", [tile, integer(0), value], {}, SPAN)
    local_body = region(sequence(ir.AssignStmt(value, load, SPAN), ir.AssignStmt(out, store, SPAN)))
    body = sequence(ir.InCoreScopeStmt(body=local_body, span=SPAN), ir.EvalStmt(integer(0), SPAN))
    function = ir.Function("kernel", [tile], [], body, SPAN, type=ir.FunctionType.Orchestration)
    outlined = passes.outline_incore_scopes()(ir.Program([function], "RegTest", SPAN))
    kernel = next(f for f in outlined.functions.values() if f.func_type == ir.FunctionType.InCore)
    assert len(kernel.return_types) == 1
    assert isinstance(kernel.return_types[0], ir.TileType)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.RegScopeValid)
    assert passes.PropertyVerifierRegistry.verify(properties, outlined) == []


def test_incore_context_is_restored_after_device_boundary():
    device = ir.InCoreScopeStmt(body=region(definition(register("device"))), span=SPAN)
    # A legal device region must not authorize a following orchestration region.
    body = sequence(device, region(definition(register("outside"))))
    diagnostics = verify(body, ir.FunctionType.Orchestration)
    assert len(diagnostics) == 1
    assert "context" in diagnostics[0].message


def test_orchestration_block_is_restored_after_device_boundary():
    device = ir.InCoreScopeStmt(body=region(definition(register("device"))), span=SPAN)
    body = ir.GraphScopeStmt(
        "graph", body=sequence(device, region(definition(register("outside")))), span=SPAN
    )
    diagnostics = verify(body, ir.FunctionType.Opaque)
    assert len(diagnostics) == 1
    assert "context" in diagnostics[0].message


def test_graph_inside_register_region_remains_illegal():
    body = region(ir.GraphScopeStmt("graph", body=sequence(), span=SPAN))
    assert any("Orchestration scopes cannot appear" in d.message for d in verify(body))


def test_final_placement_requires_aiv():
    body = region(definition(register()))
    assert verify(body, ir.FunctionType.AIV, final=True) == []
    for kind in (ir.FunctionType.InCore, ir.FunctionType.AIC):
        assert any("AIV" in d.message for d in verify(body, kind, final=True))


def test_wrong_target():
    backend.set_backend_type(backend.BackendType.Ascend910)
    assert any("A5" in d.message for d in verify(region(definition(register()))))


def test_reg_and_scalar_escape_and_identity_rebinding():
    r = register()
    assert any("escape" in d.message for d in verify(sequence(region(definition(r)), ir.EvalStmt(r, SPAN))))
    s = scalar("local")
    body = sequence(region(ir.AssignStmt(s, integer(1), SPAN)), ir.EvalStmt(s, SPAN))
    assert any("escape" in d.message for d in verify(body))
    # A new outer Scalar with the same spelling does not reuse the local value.
    outer = scalar("local")
    body = sequence(
        region(ir.AssignStmt(s, integer(1), SPAN)),
        ir.AssignStmt(outer, integer(2), SPAN),
        ir.EvalStmt(outer, SPAN),
    )
    assert verify(body) == []


def test_outer_scalar_reassignment_rejected():
    outer = scalar("outer")
    assert any(
        "ownership" in d.message
        for d in verify(region(ir.AssignStmt(outer, integer(1), SPAN)), params=[outer])
    )


def test_nesting_and_unscoped_call():
    assert any("nest" in d.message for d in verify(region(region(definition(register())))))
    assert any("requires RegScope" in d.message for d in verify(definition(register())))


def test_loop_yield_cannot_export_region_value():
    r = register()
    i = scalar("i")
    result = register("result")
    init = register("initial")
    arg = ir.IterArg("carry", r.type, init, SPAN)
    body = region(sequence(definition(r), ir.YieldStmt([r], SPAN)))
    loop = ir.ForStmt(i, integer(0), integer(4), integer(1), [arg], body, [result], SPAN)
    assert any("escape" in d.message for d in verify(loop))


def test_builder_constructs_store_as_assignment():
    ib = ir.IRBuilder()
    with ib.function("kernel", type=ir.FunctionType.InCore) as f:
        src = f.param("src", ir.TileType([128], DataType.FP32, memory_space=ir.MemorySpace.Vec))
        with ib.reg_scope():
            value = ib.emit_reg("value", "reg.load", [src, integer(0)], {"lanes": 64})
            out = ib.emit_reg_store("out", src, value)
        ib.emit(ir.EvalStmt(out, SPAN))
    func = f.get_result()
    assert isinstance(func.body, ir.SeqStmts)
    scope = func.body.stmts[0]
    assert isinstance(scope, ir.RegScopeStmt)
    assert isinstance(scope.body, ir.SeqStmts)
    store = scope.body.stmts[1]
    assert isinstance(store, ir.AssignStmt)
    assert isinstance(store.value, ir.Call)
    assert store.value.op.name == ir.get_op("reg.store").name
    ir.assert_structural_equal(store.var.type, src.type)


def test_expand_mixed_kernel_places_register_region_on_aiv():
    backend.set_backend_type(backend.BackendType.Ascend950)
    body = region(definition(register()))
    function = ir.Function("kernel", [], [], body, SPAN, type=ir.FunctionType.InCore)
    program = ir.Program([function], "RegTest", SPAN)
    expanded = passes.expand_mixed_kernel()(program)
    functions = list(expanded.functions.values())
    assert len(functions) == 1
    assert functions[0].func_type == ir.FunctionType.AIV
    assert isinstance(functions[0].body, ir.RegScopeStmt)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.MixedKernelExpanded)
    assert passes.PropertyVerifierRegistry.verify(properties, expanded) == []


def test_register_only_scope_emits_without_tile_capture():
    backend.set_backend_type(backend.BackendType.Ascend950)
    function = ir.Function("kernel", [], [], region(definition(register())), SPAN, type=ir.FunctionType.AIV)
    program = ir.Program([function], "RegTest", SPAN)
    mlir = codegen.PTOCodegen().generate(program)
    assert "pto.vecscope" in mlir
    assert 'pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<64xpred>' in mlir


def test_local_scalar_if_results_stay_inside_region():
    local, result = scalar("local"), scalar("result")
    branch = ir.IfStmt(
        ir.ConstBool(True, SPAN), ir.YieldStmt([local], SPAN), ir.YieldStmt([local], SPAN), [result], SPAN
    )
    body = region(sequence(ir.AssignStmt(local, integer(1), SPAN), branch, ir.EvalStmt(result, SPAN)))
    assert verify(body) == []


def test_local_scalar_cannot_be_stored_in_external_array():
    local = scalar("local")
    array_type = ir.ArrayType(DataType.INDEX, integer(4))
    external = ir.Var("external", array_type, SPAN)
    result = ir.Var("updated", array_type, SPAN)
    update = ir.create_op_call("array.update_element", [external, integer(0), local], {}, SPAN)
    body = region(sequence(ir.AssignStmt(local, integer(7), SPAN), ir.AssignStmt(result, update, SPAN)))
    assert any("container" in d.message for d in verify(body, params=[external]))


def test_readonly_external_scalar_is_implicitly_captured():
    external = scalar("external")
    local = scalar("local")
    body = region(ir.AssignStmt(local, external, SPAN))
    assert verify(body, params=[external]) == []


@pytest.mark.parametrize("nested", [False, True])
def test_tuple_cannot_hide_register_function_boundary(nested):
    type_ = register().type
    if nested:
        type_ = ir.TupleType([type_])
    param = ir.Var("argument", ir.TupleType([type_]), SPAN)
    assert any("function boundary" in d.message for d in verify(sequence(), params=[param]))


def test_tuple_cannot_hide_local_scalar_escape():
    local = scalar("local")
    packed = ir.Var("packed", ir.TupleType([local.type]), SPAN)
    body = sequence(
        region(
            sequence(
                ir.AssignStmt(local, integer(1), SPAN),
                ir.AssignStmt(packed, ir.MakeTuple([local], SPAN), SPAN),
            )
        ),
        ir.EvalStmt(packed, SPAN),
    )
    assert any("escape" in d.message for d in verify(body))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
