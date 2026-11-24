# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Early VF outlining preserves captures, alias returns and caller storage."""

import textwrap

import pytest
from pypto import DataType, backend, codegen, ir, passes
from pypto.ir.pass_manager import PassManager
from pypto.language.parser import parse_program
from pypto.language.parser.diagnostics.exceptions import InvalidOperationError

_TILE_ADD = ir.get_op("tile.add").name
_REG_MEMORY = {ir.get_op("reg.load").name, ir.get_op("reg.store").name}


@pytest.fixture(autouse=True)
def a5_backend():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def program(body: str, kind: str = "InCore", *, initialized: bool = True) -> ir.Program:
    src = (
        "pl.tile.full([128], dtype=pl.FP32, value=1.0)"
        if initialized
        else "pl.tile.create([128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)"
    )
    return parse_program(f"""
@pl.program
class Kernel:
    @pl.function(type=pl.FunctionType.{kind})
    def kernel(self, n: pl.Scalar[pl.INDEX]):
        src = {src}
        dst = pl.tile.full([128], dtype=pl.FP32, value=0.0)
{textwrap.indent(textwrap.dedent(body), "        ")}
""")


COPY = """
with pl.reg_scope():
    x = pl.reg.load(src, lanes=64)
    mask = pl.reg.active_mask(n, lanes=64)
    out = pl.reg.store(dst, value=x, mask=mask)
z = pl.tile.add(out, out)
"""


def helpers(prog: ir.Program) -> list[ir.Function]:
    return [f for f in prog.functions.values() if f.func_type == ir.FunctionType.VF]


def aiv_payload_stmts(aiv: ir.Function) -> list[ir.Stmt]:
    """Check native shared initialization, then inspect the primary payload."""
    assert isinstance(aiv.body, ir.SeqStmts)
    body = aiv.body.stmts
    if not aiv.attrs.get("dual_aiv_dispatch", False):
        return body
    (lane,) = (
        s
        for s in body
        if isinstance(s, ir.AssignStmt)
        and isinstance(s.value, ir.Call)
        and s.value.op.name == ir.get_op("tile.get_subblock_idx").name
    )
    (branch,) = (s for s in body if isinstance(s, ir.IfStmt))
    assert isinstance(branch.condition, ir.Eq) and branch.condition.left is lane.var
    assert isinstance(branch.condition.right, ir.ConstInt) and branch.condition.right.value == 0
    (setup,) = (
        s
        for s in body
        if isinstance(s, ir.EvalStmt)
        and isinstance(s.expr, ir.Call)
        and s.expr.op.name == ir.get_op("system.aiv_initialize_pipe").name
    )
    assert body.index(setup) < body.index(branch)
    inactive = branch.else_body
    if isinstance(inactive, ir.SeqStmts):
        assert len(inactive.stmts) == 1
        inactive = inactive.stmts[0]
    assert isinstance(inactive, ir.ReturnStmt)
    assert all(any(value is param for param in aiv.params) for value in inactive.value)
    assert isinstance(branch.then_body, ir.SeqStmts)
    return branch.then_body.stmts


def test_early_pass_order():
    names = PassManager.get_strategy().pass_names
    assert names.index("OutlineRegScopes") == names.index("OutlineSimtScopes") + 1
    assert names.index("OutlineRegScopes") < names.index("InlineFunctions")
    assert names.index("OutlineRegScopes") < names.index("ConvertToSSA")
    assert names.index("OutlineRegScopes") < names.index("ExpandMixedKernel")


def test_captures_directions_alias_return_and_idempotence():
    before = program(COPY)
    after = passes.outline_reg_scopes()(before)
    (helper,) = helpers(after)
    assert [p.name_hint.split("__ssa_v")[0] for p in helper.params] == ["src", "n", "dst"]
    assert helper.param_directions == [ir.ParamDirection.In, ir.ParamDirection.In, ir.ParamDirection.InOut]
    assert helper.level is None and helper.role is None
    assert not ir.is_incore_type(helper.func_type)
    assert ir.is_device_function_type(helper.func_type)
    assert isinstance(helper.body, ir.SeqStmts)
    assert isinstance(helper.body.stmts[0], ir.RegScopeStmt)
    ret = helper.body.stmts[-1]
    assert isinstance(ret, ir.ReturnStmt)
    assert ret.value[0] is helper.params[2]
    caller = after.get_function("kernel")
    assert caller is not None
    assert "with pl.reg_scope()" not in ir.python_print(caller)
    assert ir.structural_equal(after, passes.outline_reg_scopes()(after))
    ssa = passes.convert_to_ssa()(after)
    assert len(helpers(ssa)) == 1
    assert ir.structural_equal(ssa, parse_program(ir.python_print(ssa)))


def test_two_scopes_keep_distinct_functions_and_storage_updates():
    after = passes.outline_reg_scopes()(
        program(
            COPY
            + COPY.replace("src", "out")
            .replace("x", "x2")
            .replace("mask =", "mask2 =")
            .replace("mask=mask", "mask=mask2")
        )
    )
    assert len(helpers(after)) == 2
    assert len({f.name for f in helpers(after)}) == 2
    assert len(helpers(passes.convert_to_ssa()(after))) == 2


def test_unused_store_result_still_calls_void_helper():
    source = COPY.split("z =")[0]
    after = passes.outline_reg_scopes()(program(source))
    (helper,) = helpers(after)
    assert not helper.return_types
    assert helper.param_directions[-1] == ir.ParamDirection.InOut
    assert "self.kernel_vf_0" in ir.python_print(after)


def test_pipeline_places_call_on_aiv_and_reuses_return_storage():
    with passes.PassContext([]):
        result = PassManager.get_strategy().run_passes(program(COPY))
    caller = result.get_function("kernel")
    assert caller is not None
    assert caller.func_type == ir.FunctionType.AIV
    assert len(helpers(result)) == 1
    # Alias returns and the downstream add must retain the caller's allocation.
    assert isinstance(caller.body, ir.SeqStmts)
    vf_call = next(
        stmt
        for stmt in caller.body.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and isinstance(stmt.value.op, ir.GlobalVar)
    )
    assert isinstance(vf_call.value, ir.Call)
    dst = vf_call.value.args[-1]
    assert isinstance(dst.type, ir.TileType) and isinstance(vf_call.var.type, ir.TileType)
    assert dst.type.memref is not None
    assert vf_call.var.type.memref is dst.type.memref
    add = next(
        stmt.value
        for stmt in caller.body.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == _TILE_ADD
    )
    assert all(isinstance(arg.type, ir.TileType) and arg.type.memref is dst.type.memref for arg in add.args)
    bundle = codegen.PTOCodegen().generate_bundle(result, emit_source_loc=False)
    assert len(bundle.helpers) == 1
    assert bundle.helpers[0].parameter_effects.count("readwrite") == 1
    assert "pto.vecscope" not in bundle.caller_source
    assert "pto.vecscope" in bundle.helpers[0].source
    assert "pto.vmi.vload" in bundle.helpers[0].source


def test_multiple_alias_returns_are_explicit():
    body = """
with pl.reg_scope():
    x = pl.reg.load(src, lanes=64)
    first = pl.reg.store(dst, value=x)
    second = pl.reg.store(first, value=x)
z = pl.tile.add(first, second)
"""
    after = passes.outline_reg_scopes()(program(body))
    (helper,) = helpers(after)
    assert isinstance(helper.body, ir.SeqStmts)
    ret = helper.body.stmts[-1]
    assert isinstance(ret, ir.ReturnStmt)
    assert len(ret.value) == 2
    assert ret.value[0] is ret.value[1]
    with passes.PassContext([]):
        result = PassManager.get_strategy().run_passes(program(body))
    assert len(codegen.PTOCodegen().generate_bundle(result).helpers) == 1


@pytest.mark.parametrize("initialized", [False, True])
def test_initialization_is_checked_at_the_call_site(initialized):
    body = """
with pl.reg_scope():
    x = pl.reg.load(src, lanes=64)
    out = pl.reg.store(dst, value=x)
"""
    source = program(body, initialized=initialized)
    with passes.PassContext([]):
        if initialized:
            PassManager.get_strategy().run_passes(source)
        else:
            with pytest.raises(ValueError, match="uninitialized"):
                PassManager.get_strategy().run_passes(source)


def test_hardware_loop_register_values_stay_inside_helper():
    body = """
with pl.reg_scope():
    for i in pl.hardware_loop(n):
        x = pl.reg.load(src, offset=i * 64, lanes=64)
        out = pl.reg.store(dst, offset=i * 64, value=x)
"""
    after = passes.outline_reg_scopes()(program(body))
    (helper,) = helpers(after)
    assert "hardware_loop" in ir.python_print(helper)
    assert all(not isinstance(param.type, ir.RegType) for param in helper.params)
    with passes.PassContext([]):
        result = PassManager.get_strategy().run_passes(program(body))
    bundle = codegen.PTOCodegen().generate_bundle(result)
    assert "scf.for" in bundle.helpers[0].source


def test_vf_enum_roundtrip():
    after = passes.outline_reg_scopes()(program(COPY))
    restored = parse_program(ir.python_print(after))
    assert ir.structural_equal(after, restored)


@pytest.mark.parametrize("read_lanes", [64, 128])
def test_partial_store_does_not_initialize_whole_inout_argument(read_lanes):
    source = program(f"""
scratch = pl.tile.create([128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
with pl.reg_scope():
    x = pl.reg.load(src, lanes=64)
    ignored = pl.reg.store(scratch, value=x)
with pl.reg_scope():
    y = pl.reg.load(scratch, lanes={read_lanes})
    ignored_again = pl.reg.store(dst, value=y)
""")
    with passes.PassContext([]):
        if read_lanes == 64:
            result = PassManager.get_strategy().run_passes(source)
            assert len(codegen.PTOCodegen().generate_bundle(result).helpers) == 2
        else:
            with pytest.raises(ValueError, match="uninitialized"):
                PassManager.get_strategy().run_passes(source)


def test_pre_ssa_rebinding_preserves_original_storage():
    source = program(COPY.replace("out =", "dst =").replace("tile.add(out, out)", "tile.add(dst, dst)"))
    with passes.PassContext([]):
        result = PassManager.get_strategy().run_passes(source)
    bundle = codegen.PTOCodegen().generate_bundle(result)
    assert len(bundle.helpers) == 1
    assert bundle.helpers[0].parameter_effects.count("readwrite") == 1


def test_outlined_program_serialization():
    outlined = passes.outline_reg_scopes()(program(COPY))
    ir.assert_structural_equal(outlined, ir.deserialize(ir.serialize(outlined)))
    assert ir.structural_hash(outlined) == ir.structural_hash(ir.deserialize(ir.serialize(outlined)))


MIXED_SOURCE = """
@pl.program
class Mixed:
    @pl.function(type=pl.FunctionType.InCore)
    def main(self, x: pl.Tensor[[16, 128], pl.BF16],
             y: pl.Tensor[[128, 128], pl.BF16],
             output: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
        a = pl.tile.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
        left = pl.tile.move(a, target_memory=pl.MemorySpace.Left)
        b = pl.tile.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
        right = pl.tile.move(b, target_memory=pl.MemorySpace.Right)
        acc = pl.tile.matmul(left, right)
        vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec,
                     pl.TileView(valid_shape=[16, 128], stride=[128, 1])] = pl.tile.move(
            acc, target_memory=pl.MemorySpace.Vec,
            blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)
        with pl.reg_scope():
            value = pl.reg.load(vec, lanes=64)
            updated = pl.reg.store(vec, value=value)
        output = pl.tile.store(updated, [0, 0], output)
        return output
"""


@pytest.mark.parametrize("local_copy", [False, True])
def test_mixed_kernel_keeps_vf_on_aiv_and_passes_pipe_storage(local_copy):
    source_text = MIXED_SOURCE
    if local_copy:
        source_text = (
            source_text.replace(
                "        with pl.reg_scope():",
                "        local: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, "
                "pl.TileView(valid_shape=[16, 128], stride=[128, 1])] = pl.tile.add(vec, vec)\n"
                "        with pl.reg_scope():",
            )
            .replace("pl.reg.load(vec", "pl.reg.load(local")
            .replace("pl.reg.store(vec", "pl.reg.store(local")
        )
    source = parse_program(source_text)
    with passes.PassContext([]):
        result = PassManager.get_strategy().run_passes(source)
    (helper,) = helpers(result)
    callers = [
        f for f in result.functions.values() if f.func_type in (ir.FunctionType.AIC, ir.FunctionType.AIV)
    ]
    assert {f.func_type for f in callers} == {ir.FunctionType.AIC, ir.FunctionType.AIV}
    for caller in callers:
        has_call = helper.name in ir.python_print(caller)
        assert has_call == (caller.func_type == ir.FunctionType.AIV)
    aiv = next(f for f in callers if f.func_type == ir.FunctionType.AIV)
    text = ir.python_print(aiv)
    if not local_copy:
        assert text.index("tile.store") < text.index("system.tfree_to_aic")
    unit = ir.Program([aiv, helper], "DeviceUnit", result.span)
    bundle = codegen.PTOCodegen().generate_bundle(unit, emit_source_loc=False)
    (artifact,) = bundle.helpers
    assert artifact.parameter_effects == ["readwrite"]
    assert artifact.capture_base_indices == [0]
    grouped = codegen.PTOCodegen().generate_bundle(
        ir.Program([*callers, helper], "GroupedDeviceUnit", result.span), emit_source_loc=False
    )
    (artifact,) = grouped.helpers
    aic = next(f for f in callers if f.func_type == ir.FunctionType.AIC)
    assert grouped.entry_helpers == {aic.name: [], aiv.name: [artifact.symbol]}
    if not local_copy:
        pto = grouped.caller_source
        assert pto.index("pto.tpop_from_aic") < pto.index(f"func.call @{artifact.adapter_symbol}")
        assert (
            pto.index(f"func.call @{artifact.adapter_symbol}")
            < pto.index("pto.tstore")
            < pto.index("pto.tfree_from_aic")
        )


def mixed_epilogue(body: str) -> ir.Program:
    prefix = MIXED_SOURCE.split("        with pl.reg_scope():")[0]
    prefix = prefix.replace("def main(self, x:", "def main(self, n: pl.Scalar[pl.INDEX], x:")
    return parse_program(
        prefix + textwrap.indent(textwrap.dedent(body), "        ") + "        return output\n"
    )


def lower_through(source: ir.Program, pass_name: str | None) -> ir.Program:
    if pass_name is None:
        with passes.PassContext([]):
            return PassManager.get_strategy().run_passes(source)
    with passes.PassContext([], enable_buffer_ir=False):
        for transform in PassManager.get_strategy().passes:
            source = transform(source)
            if transform.get_name() == pass_name:
                return source
    raise AssertionError(f"{pass_name} missing from the default strategy")


@pytest.mark.parametrize(
    "pass_name", ["ExpandMixedKernel", "InitMemRef", pytest.param(None, id="full-default")]
)
def test_vf_loop_result_keeps_outer_receive_until_store(pass_name):
    # The runtime bound permits zero trips. Both the initial Tile and the VF
    # writeback on every backedge must denote the receive consumed after the loop.
    result = lower_through(
        mixed_epilogue("""
        for _, (carried,) in pl.range(n, init_values=(vec,)):
            with pl.reg_scope():
                value = pl.reg.load(carried, lanes=64)
                updated = pl.reg.store(carried, value=value)
            result = pl.yield_(updated)
        output = pl.tile.store(result, [0, 0], output)
    """),
        pass_name,
    )
    (vf,) = helpers(result)
    aiv = next(f for f in result.functions.values() if f.func_type == ir.FunctionType.AIV)
    assert isinstance(aiv.body, ir.SeqStmts)
    body = aiv_payload_stmts(aiv)
    (loop,) = (s for s in body if isinstance(s, ir.ForStmt))
    assert vf.name in ir.python_print(loop)
    (pop,) = (
        s
        for s in body
        if isinstance(s, ir.AssignStmt)
        and isinstance(s.value, ir.Call)
        and s.value.op.name == ir.get_op("tile.tpop_from_aic").name
    )
    (store,) = (
        s
        for s in body
        if isinstance(s, ir.AssignStmt)
        and isinstance(s.value, ir.Call)
        and s.value.op.name == ir.get_op("tile.store").name
    )
    (free,) = (
        s
        for s in body
        if isinstance(s, ir.EvalStmt)
        and isinstance(s.expr, ir.Call)
        and s.expr.op.name == ir.get_op("system.tfree_to_aic").name
    )
    assert isinstance(store.value, ir.Call) and store.value.args[0] is loop.return_vars[0]
    assert isinstance(free.expr, ir.Call) and free.expr.args[0] is pop.var
    assert body.index(store) < body.index(free)
    if pass_name != "ExpandMixedKernel":
        assert isinstance(loop.return_vars[0].type, ir.TileType)
        assert loop.return_vars[0].type.memref is None
        properties = passes.IRPropertySet()
        properties.insert(passes.IRProperty.HasMemRefs)
        passes.verify_properties(properties, result, "outlined VF FIFO loop result")


def test_vf_loop_rejects_replacing_a_borrow_with_local_storage():
    source = mixed_epilogue("""
        for _, (carried,) in pl.range(n, init_values=(vec,)):
            with pl.reg_scope():
                value = pl.reg.load(carried, lanes=64)
                updated = pl.reg.store(carried, value=value)
            local = pl.tile.add(updated, updated)
            result = pl.yield_(local)
        output = pl.tile.store(result, [0, 0], output)
    """)
    with pytest.raises(ValueError, match="including zero iterations"):
        lower_through(source, "ExpandMixedKernel")


def test_vf_tuple_projection_keeps_received_slot_across_branch():
    # Same-shaped outputs are returned by a single VF. Only the receive is
    # borrowed, and one branch names it through the second projection.
    result = lower_through(
        mixed_epilogue("""
        local: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec,
                       pl.TileView(valid_shape=[16, 128], stride=[128, 1])] = pl.tile.add(vec, vec)
        with pl.reg_scope():
            value = pl.reg.load(local, lanes=64)
            updated_local = pl.reg.store(local, value=value)
            updated_vec = pl.reg.store(vec, value=value)
        if n > 0:
            selected = pl.yield_(updated_vec)
        else:
            selected = pl.yield_(vec)
        output = pl.tile.store(updated_local, [0, 0], output)
        output = pl.tile.store(selected, [0, 0], output)
    """),
        "InitMemRef",
    )
    (vf,) = helpers(result)
    assert len(vf.return_types) == 2
    aiv = next(f for f in result.functions.values() if f.func_type == ir.FunctionType.AIV)
    assert isinstance(aiv.body, ir.SeqStmts)
    # The allocated sibling must keep its own buffer, and the selected receive
    # must survive until its last store without gaining a new allocation.
    stores = [
        s.value
        for s in aiv_payload_stmts(aiv)
        if isinstance(s, ir.AssignStmt)
        and isinstance(s.value, ir.Call)
        and s.value.op.name == ir.get_op("tile.store").name
    ]
    assert len(stores) == 2
    local, received = (call.args[0].type for call in stores)
    assert isinstance(local, ir.TileType) and local.memref is not None
    assert isinstance(received, ir.TileType) and received.memref is None
    emitted = ir.python_print(aiv)
    assert emitted.rindex("tile.store") < emitted.index("system.tfree_to_aic")
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.HasMemRefs)
    passes.verify_properties(properties, result, "outlined VF projection and same-slot join")


def test_register_loop_carries_are_local_and_retyped():
    source = program("""
with pl.reg_scope():
    seed = pl.reg.load(src, lanes=64)
    for i, (acc,) in pl.hardware_loop(n, init_values=(seed,)):
        value = pl.reg.add(acc, seed)
        result = pl.yield_(value)
    out = pl.reg.store(dst, value=result)
z = pl.tile.add(out, out)
""")
    with passes.PassContext([]):
        result = PassManager.get_strategy().run_passes(source)
    assert all(not isinstance(p.type, ir.RegType) for f in helpers(result) for p in f.params)
    bundle = codegen.PTOCodegen().generate_bundle(result)
    assert "iter_args" in bundle.helpers[0].source


def test_tile_view_symbols_are_captured_with_helper_owned_identity():
    source = parse_program("""
@pl.program
class DynamicView:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, offset: pl.Scalar[pl.INDEX],
               view: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec,
                             pl.TileView(valid_shape=[64], stride=[1], start_offset=offset)]):
        with pl.reg_scope():
            value = pl.reg.load(view, lanes=64)
""")
    outlined = passes.outline_reg_scopes()(source)
    (helper,) = helpers(outlined)
    assert [param.name_hint.split("__ssa_v")[0] for param in helper.params] == ["offset", "view"]
    view_type = helper.params[1].type
    assert isinstance(view_type, ir.TileType)
    assert view_type.tile_view is not None
    assert view_type.tile_view.start_offset is helper.params[0]
    caller = source.get_function("kernel")
    assert caller is not None
    assert helper.params[0] is not caller.params[0]
    ir.assert_structural_equal(outlined, parse_program(ir.python_print(outlined)))


def test_helper_call_cannot_run_on_aic():
    outlined = passes.outline_reg_scopes()(program(COPY))
    caller = outlined.get_function("kernel")
    assert caller is not None
    aic = ir.Function(
        caller.name,
        caller.params,
        caller.return_types,
        caller.body,
        caller.span,
        ir.FunctionType.AIC,
    )
    invalid = ir.Program([aic, *helpers(outlined)], "InvalidPlacement", outlined.span)
    with pytest.raises(ValueError, match="VF calls require"):
        passes.outline_reg_scopes()(invalid)


def test_pipeline_keeps_verifiable_helper_boundaries():
    result = PassManager.get_strategy().run_passes(program(COPY))
    assert len(helpers(result)) == 1


def test_earlier_store_result_survives_later_capture_rebinding():
    source = program("""
with pl.reg_scope():
    value = pl.reg.load(src, lanes=64)
    first = pl.reg.store(dst, value=value)
    dst = pl.reg.store(src, value=value)
z = pl.tile.add(first, first)
""")
    outlined = passes.outline_reg_scopes()(source)
    (helper,) = helpers(outlined)
    assert helper.param_directions == [ir.ParamDirection.InOut, ir.ParamDirection.InOut]
    assert isinstance(helper.body, ir.SeqStmts)
    returned = helper.body.stmts[-1]
    assert isinstance(returned, ir.ReturnStmt)
    assert returned.value[0] is helper.params[1]
    with passes.PassContext([]):
        lowered = PassManager.get_strategy().run_passes(source)
    assert codegen.PTOCodegen().generate_bundle(lowered).helpers[0].parameter_effects == [
        "readwrite",
        "readwrite",
    ]


@pytest.mark.parametrize("flag", [-1, 1])
def test_constant_branch_selects_tile_alias_before_vf_contract(flag):
    source = program(f"""
flag: pl.Scalar[pl.INDEX] = {flag}
with pl.reg_scope():
    value = pl.reg.load(src, lanes=64)
    if flag > 0:
        out = pl.reg.store(dst, value=value)
    else:
        doubled = pl.reg.add(value, value)
        out = pl.reg.store(src, value=doubled)
z = pl.tile.add(out, out)
""")
    outlined = passes.outline_reg_scopes()(source)
    (helper,) = helpers(outlined)
    assert isinstance(helper.body, ir.SeqStmts)
    returned = helper.body.stmts[-1]
    assert isinstance(returned, ir.ReturnStmt)
    selected = next(p for p in helper.params if p.name_hint.startswith("dst" if flag > 0 else "src"))
    # Only the selected storage belongs in the alias contract. The dead arm
    # must not force its different buffer to match the selected return.
    assert returned.value[0] is selected
    result = PassManager.get_strategy().run_passes(source)
    (emitted,) = codegen.PTOCodegen().generate_bundle(result).helpers
    assert emitted.source.count("pto.vmi.vstore") == 1
    assert emitted.source.count("pto.vmi.vadd") == int(flag < 0)


@pytest.mark.parametrize("wrapped", [False, True])
def test_inline_owned_unroll_specializes_each_caller_bound(wrapped):
    target = "wrapper" if wrapped else "copy"
    source = parse_program(f"""
@pl.program
class InlineBound:
    @pl.function(type=pl.FunctionType.Inline)
    def copy(self, count: pl.Scalar[pl.INDEX]):
        src = pl.tile.full([128], dtype=pl.FP32, value=1.0)
        dst = pl.tile.full([128], dtype=pl.FP32, value=0.0)
        with pl.reg_scope():
            seed = pl.reg.load(src, lanes=64)
            value = seed
            for i in pl.unroll(2):
                value = pl.reg.add(value, seed)
            out = pl.reg.store(dst, value=value)
    @pl.function(type=pl.FunctionType.Inline)
    def wrapper(self, count: pl.Scalar[pl.INDEX]):
        self.copy(count)
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self):
        self.{target}(1)
        self.{target}(2)
""")
    function = source.get_function("copy")
    assert function is not None
    count = function.params[0]

    class ParameterBound(ir.IRMutator):
        def visit_for_stmt(self, op: ir.ForStmt) -> ir.Stmt:
            return ir.ForStmt(
                op.loop_var,
                op.start,
                count,
                op.step,
                op.iter_args,
                op.body,
                op.return_vars,
                op.span,
                op.kind,
            )

    # This direct-IR parameter bound is accepted by ordinary inlining, but
    # cannot be unrolled in an unspecialized Inline declaration.
    source = ir.Program(
        [ParameterBound().visit_function(f) if f.name == "copy" else f for f in source.functions.values()],
        source.name,
        source.span,
    )
    with passes.PassContext([]):
        outlined = passes.outline_reg_scopes()(source)
        assert len(helpers(outlined)) == 2
        result = PassManager.get_strategy().run_passes(source)
    emitted = codegen.PTOCodegen().generate_bundle(result).helpers
    assert sorted(h.source.count("pto.vmi.vadd") for h in emitted) == [1, 2]


def test_program_without_register_scopes_keeps_ordinary_inline_stage():
    source = parse_program("""
@pl.program
class NoRegion:
    @pl.function(type=pl.FunctionType.Inline)
    def count(self) -> pl.Scalar[pl.INDEX]:
        return 1
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self):
        value = self.count()
""")
    ir.assert_structural_equal(passes.outline_reg_scopes()(source), source)


@pytest.mark.parametrize("flag", [-1, 1])
def test_inline_constant_producer_selects_vf_tile_alias(flag):
    source = parse_program(f"""
@pl.program
class InlineFlag:
    @pl.function(type=pl.FunctionType.Inline)
    def get_flag(self) -> pl.Scalar[pl.INDEX]:
        return {flag}
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self):
        src = pl.tile.full([128], dtype=pl.FP32, value=1.0)
        dst = pl.tile.full([128], dtype=pl.FP32, value=0.0)
        flag = self.get_flag()
        with pl.reg_scope():
            value = pl.reg.load(src, lanes=64)
            if flag > 0:
                out = pl.reg.store(dst, value=value)
            else:
                doubled = pl.reg.add(value, value)
                out = pl.reg.store(src, value=doubled)
        observed = pl.tile.add(out, out)
""")
    result = PassManager.get_strategy().run_passes(source)
    (emitted,) = codegen.PTOCodegen().generate_bundle(result).helpers
    assert emitted.source.count("pto.vmi.vstore") == 1
    assert emitted.source.count("pto.vmi.vadd") == int(flag < 0)


@pytest.mark.parametrize("dynamic", [False, True])
def test_inline_scalar_out_write_reaches_vf_mask(dynamic):
    actual = "n" if dynamic else "64"
    source = parse_program(f"""
@pl.program
class ScalarOut:
    @pl.function(type=pl.FunctionType.Inline)
    def set_count(
        self, count: pl.Out[pl.Scalar[pl.INDEX]], value: pl.Scalar[pl.INDEX]
    ) -> pl.Scalar[pl.INDEX]:
        count = value
        return count
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, n: pl.Scalar[pl.INDEX]):
        src = pl.tile.full([128], dtype=pl.FP32, value=1.0)
        dst = pl.tile.full([128], dtype=pl.FP32, value=0.0)
        count: pl.Scalar[pl.INDEX] = 0
        updated_count = self.set_count(count, {actual})
        with pl.reg_scope():
            value = pl.reg.load(src, lanes=64)
            mask = pl.reg.active_mask(updated_count, lanes=64)
            out = pl.reg.store(dst, value=value, mask=mask)
        observed = pl.tile.add(out, out)
""")
    result = PassManager.get_strategy().run_passes(source)
    (helper,) = helpers(result)
    count_position = next(i for i, p in enumerate(helper.params) if p.name_hint.startswith("updated_count"))
    arguments = []

    class Calls(ir.IRVisitor):
        def visit_call(self, op: ir.Call) -> None:
            if isinstance(op.op, ir.GlobalVar) and op.op.name == helper.name:
                arguments.append(op.args[count_position])
            super().visit_call(op)

    Calls().visit_program(result)
    assert len(arguments) == 1
    # InOutUseValid requires reading the returned handle. The mask must use
    # that updated value, including when the output is supplied dynamically.
    if dynamic:
        assert not isinstance(arguments[0], ir.ConstInt)
    else:
        assert isinstance(arguments[0], ir.ConstInt) and arguments[0].value == 64
    assert len(codegen.PTOCodegen().generate_bundle(result).helpers) == 1


def test_branch_cannot_choose_between_different_tile_storage():
    source = program("""
with pl.reg_scope():
    value = pl.reg.load(src, lanes=64)
    if n > 0:
        out = pl.reg.store(dst, value=value)
    else:
        out = pl.reg.store(src, value=value)
z = pl.tile.add(out, out)
""")
    with pytest.raises(ValueError, match="shared external view"):
        passes.outline_reg_scopes()(source)


@pytest.mark.parametrize("control", ["for i in pl.range(n)", "if n > 0"])
def test_conditional_rebinding_keeps_incoming_tile_storage(control):
    source = program(f"""
with pl.reg_scope():
    {control}:
        value = pl.reg.load(src, lanes=64)
        dst = pl.reg.store(src, value=value)
z = pl.tile.add(dst, dst)
""")
    # Skipping the body leaves dst on its original buffer. Returning src
    # unconditionally would silently change the caller's storage identity.
    with pytest.raises(ValueError, match="shared external view"):
        passes.outline_reg_scopes()(source)


def test_inline_scalar_call_survives_early_outlining():
    source = ir.python_print(program(COPY)).replace("reg.active_mask(n,", "reg.active_mask(self.count(n),")
    source = source.replace(
        "class Kernel:",
        """class Kernel:
    @pl.function(type=pl.FunctionType.Inline)
    def count(self, n: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
        return n + 1
""",
    )
    outlined = passes.outline_reg_scopes()(parse_program(source))
    assert "count(" not in ir.python_print(helpers(outlined)[0])
    result = PassManager.get_strategy().run_passes(parse_program(source))
    assert "count(" not in ir.python_print(helpers(result)[0])
    assert len(codegen.PTOCodegen().generate_bundle(result).helpers) == 1


def test_unrolled_caller_offsets_fold_after_argument_binding():
    source = program("""
for i in pl.unroll(2):
    with pl.reg_scope():
        value = pl.reg.load(src, offset=i * 64, lanes=64)
        out = pl.reg.store(dst, offset=i * 64, value=value)
""")
    result = PassManager.get_strategy().run_passes(source)
    # Each call must preserve its distinct byte offset after early substitution.
    bundle = codegen.PTOCodegen().generate_bundle(result)
    assert len(bundle.helpers) == 2
    assert all("pto.vmi.vload" in helper.source for helper in bundle.helpers)
    offsets: list[list[int]] = []

    class MemoryOffsets(ir.IRVisitor):
        def visit_call(self, op: ir.Call) -> None:
            if op.op.name in _REG_MEMORY:
                offset = op.args[1]
                assert isinstance(offset, ir.ConstInt)
                offsets[-1].append(offset.value)
            super().visit_call(op)

    for helper in helpers(result):
        offsets.append([])
        MemoryOffsets().visit_stmt(helper.body)
    assert sorted(offsets) == [[0, 0], [64, 64]]
    assert bundle.helpers[0].source != bundle.helpers[1].source
    assert bundle.entry_helpers["kernel"] == [h.symbol for h in bundle.helpers]
    for helper in bundle.helpers:
        assert f"func.call @{helper.adapter_symbol}(" in bundle.caller_source


def test_unrolled_register_accumulator_precedes_helper_ssa():
    source = program("""
with pl.reg_scope():
    seed = pl.reg.load(src, lanes=64)
    acc = seed
    for i in pl.unroll(2):
        acc = pl.reg.add(acc, seed)
    out = pl.reg.store(dst, value=acc)
z = pl.tile.add(out, out)
""")
    outlined = passes.outline_reg_scopes()(source)
    assert "unroll" not in ir.python_print(helpers(outlined)[0])
    result = PassManager.get_strategy().run_passes(source)
    bundle = codegen.PTOCodegen().generate_bundle(result)
    assert bundle.helpers[0].source.count("pto.vmi.vadd") == 2


def test_register_scope_requires_vec_capture_before_outlining():
    # Unplaced Tiles are not valid register operands, even before outlining.
    # This guards the prerequisite shared by the parser and RegScopeValid.
    with pytest.raises(InvalidOperationError, match="requires Vec UB memory"):
        parse_program("""
@pl.program
class Unplaced:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[128], pl.FP32]):
        src = pl.tile.load(x, [0], [128])
        with pl.reg_scope():
            value = pl.reg.load(src, lanes=64)
""")


class CopyVFTuples(ir.IRMutator):
    def __init__(self):
        super().__init__()
        self.copies: dict[ir.Var, ir.Var] = {}

    def visit_assign_stmt(self, op: ir.AssignStmt) -> ir.Stmt:
        if isinstance(op.value, ir.Call) and isinstance(op.value.op, ir.GlobalVar):
            assert isinstance(op.var.type, ir.TupleType)
            first_copy = ir.Var("first_vf_copy", op.var.type, op.span)
            copy = ir.Var("copied_vf_results", op.var.type, op.span)
            self.copies[op.var] = copy
            return ir.SeqStmts(
                [op, ir.AssignStmt(first_copy, op.var, op.span), ir.AssignStmt(copy, first_copy, op.span)],
                op.span,
            )
        if (
            isinstance(op.value, ir.TupleGetItemExpr)
            and isinstance(op.value.tuple, ir.Var)
            and op.value.tuple in self.copies
        ):
            value = ir.TupleGetItemExpr(self.copies[op.value.tuple], op.value.index, op.span)
            return ir.AssignStmt(op.var, value, op.span)
        return super().visit_assign_stmt(op)


def test_codegen_preserves_vf_tuple_copy_projections():
    source = program("""
with pl.reg_scope():
    value = pl.reg.load(src, lanes=64)
    first = pl.reg.store(dst, value=value)
    second = pl.reg.store(src, value=value)
z = pl.tile.add(first, second)
""")
    result = PassManager.get_strategy().run_passes(source)

    expected = codegen.PTOCodegen().generate_bundle(result, emit_source_loc=False)
    copied = CopyVFTuples().visit_program(result)
    actual = codegen.PTOCodegen().generate_bundle(copied, emit_source_loc=False)
    assert actual.caller_source == expected.caller_source


def test_pipeline_reordering_preserves_vf_storage_effects():
    source = program("""
for i in pl.pipeline(2, stage=1):
    with pl.reg_scope():
        value = pl.reg.load(src, lanes=64)
        out = pl.reg.store(dst, value=value)
    observed = pl.tile.read(dst, [0])
""")
    outlined = passes.convert_to_ssa()(passes.outline_reg_scopes()(source))
    after = passes.canonicalize_io_order()(outlined)
    caller = after.get_function("kernel")
    assert caller is not None
    text = ir.python_print(caller)
    assert text.index("kernel_vf_0") < text.index("tile.read")


def test_nested_vf_tuple_projection_keeps_pipe_storage_live():
    source = (
        MIXED_SOURCE.replace("def main(self, x:", "def main(self, n: pl.Scalar[pl.INDEX], x:")
        .replace(
            "            updated = pl.reg.store(vec, value=value)",
            "            updated = pl.reg.store(vec, value=value)\n"
            "            second = pl.reg.store(vec, value=value)",
        )
        .replace(
            "        output = pl.tile.store(updated, [0, 0], output)",
            "        if n > 0:\n"
            "            summed = pl.tile.add(updated, second)\n"
            "            output = pl.tile.store(summed, [0, 0], output)",
        )
    )
    outlined = passes.outline_reg_scopes()(parse_program(source))

    class DelayProjections(ir.IRMutator):
        def __init__(self):
            super().__init__()
            self.projections: list[ir.Stmt] = []

        def visit_assign_stmt(self, op: ir.AssignStmt) -> ir.Stmt:
            if isinstance(op.value, ir.TupleGetItemExpr):
                self.projections.append(op)
                return ir.SeqStmts([], op.span)
            return super().visit_assign_stmt(op)

        def visit_if_stmt(self, op: ir.IfStmt) -> ir.Stmt:
            statements = op.then_body.stmts if isinstance(op.then_body, ir.SeqStmts) else [op.then_body]
            body = ir.SeqStmts([*self.projections, *statements], op.span)
            return ir.IfStmt(op.condition, body, op.else_body, op.return_vars, op.span)

    # Authored VF calls may defer projection until a branch. Outlining normally
    # projects immediately, which previously hid the too-early tfree.
    nested = DelayProjections().visit_program(outlined)
    with passes.PassContext([]):
        result = PassManager.get_strategy().run_passes(nested)
    aiv = next(f for f in result.functions.values() if f.func_type == ir.FunctionType.AIV)
    text = ir.python_print(aiv)
    assert text.index("tile.store") < text.index("system.tfree_to_aic")


def test_unused_vf_capture_does_not_leave_stale_codegen_aliases():
    source = program("for i in pl.unroll(16):\n" + textwrap.indent(COPY, "    "))
    result = PassManager.get_strategy().run_passes(source)
    helper_names = {f.name for f in helpers(result)}

    class AddUnusedCapture(ir.IRMutator):
        def visit_function(self, func: ir.Function) -> ir.Function:
            if func.func_type == ir.FunctionType.VF:
                unused = ir.Var("unused_capture", func.params[0].type, func.span)
                return ir.Function(
                    func.name,
                    [*zip(func.params, func.param_directions), (unused, ir.ParamDirection.In)],
                    func.return_types,
                    func.body,
                    func.span,
                    func.func_type,
                )
            return super().visit_function(func)

        def visit_call(self, op: ir.Call) -> ir.Expr:
            if op.op.name in helper_names:
                attrs = dict(op.attrs)
                attrs["arg_directions"] = [*attrs["arg_directions"], ir.ArgDirection.Input]
                # A fresh scalar identity deliberately exercises LookupVarName's
                # supported name fallback after earlier borrowed Vars expired.
                args = [
                    ir.Var(arg.name_hint, arg.type, arg.span)
                    if isinstance(arg, ir.Var) and isinstance(arg.type, ir.ScalarType)
                    else arg
                    for arg in op.args
                ]
                return ir.Call(op.op, [*args, op.args[0]], op.kwargs, attrs, op.type, op.span)
            return super().visit_call(op)

    # The unused borrowed Var is absent from the retained scope. Repeated calls
    # exercise allocator reuse; extra captures must never alter storage names.
    extended = AddUnusedCapture().visit_program(result)
    emitter = codegen.PTOCodegen()
    expected = emitter.generate_bundle(result, emit_source_loc=False)
    actual = emitter.generate_bundle(extended, emit_source_loc=False)
    assert actual.caller_source == expected.caller_source
    assert [h.source for h in actual.helpers] == [h.source for h in expected.helpers]


@pytest.mark.parametrize("argument", ["tensor", "unplaced_tile"])
@pytest.mark.parametrize("owner", ["InCore", "Inline"])
def test_inline_unused_non_vec_arguments_are_removed_before_capture(argument, owner):
    annotation = "pl.Tensor[[128], pl.FP32]" if argument == "tensor" else "pl.Tile[[128], pl.FP32]"
    actual = "x" if argument == "tensor" else "extra"
    source_text = f"""
@pl.program
class InlineCapture:
    @pl.function(type=pl.FunctionType.Inline)
    def count(self, unused: {annotation}) -> pl.Scalar[pl.INDEX]:
        return 64
    @pl.function(type=pl.FunctionType.{owner})
    def kernel(self, x: pl.Tensor[[128], pl.FP32]):
        extra = pl.tile.load(x, [0], [128])
        outside_n = self.count({actual})
        src = pl.tile.full([128], dtype=pl.FP32, value=1.0)
        dst = pl.tile.full([128], dtype=pl.FP32, value=0.0)
        with pl.reg_scope():
            n = self.count({actual})
            mask = pl.reg.active_mask(n, lanes=64)
            value = pl.reg.load(src, lanes=64)
            out = pl.reg.store(dst, value=value, mask=mask)
        result = pl.tile.add(out, out)
"""
    if owner == "Inline":
        source_text += """
    @pl.function(type=pl.FunctionType.InCore)
    def entry(self, x: pl.Tensor[[128], pl.FP32]):
        self.kernel(x)
"""
    source = parse_program(source_text)
    outlined = passes.outline_reg_scopes()(source)
    assert all(isinstance(p.type, (ir.TileType, ir.ScalarType)) for p in helpers(outlined)[0].params)
    assert "count(" not in ir.python_print(helpers(outlined)[0])
    # All Inline calls are expanded before constant propagation: even calls
    # outside a region may supply constants or mutate scalar output bindings.
    assert all(f.func_type != ir.FunctionType.Inline for f in outlined.functions.values())
    assert "self.count(" not in ir.python_print(outlined)
    result = PassManager.get_strategy().run_passes(source)
    assert len(codegen.PTOCodegen().generate_bundle(result).helpers) == 1


@pytest.mark.parametrize("flag", [1, -1])
@pytest.mark.parametrize("has_else", [False, True])
def test_vf_binding_eliminates_proven_branches_and_preserves_stores(flag, has_else):
    body = f"""
flag: pl.Scalar[pl.INDEX] = {flag}
with pl.reg_scope():
    value = pl.reg.load(src, lanes=64)
    if flag > 0:
        out = pl.reg.store(dst, value=value)
"""
    if has_else:
        body += """
    else:
        doubled = pl.reg.add(value, value)
        out = pl.reg.store(dst, value=doubled)
z = pl.tile.add(out, out)
"""
    else:
        body += "z = pl.tile.add(dst, dst)\n"
    source = program(body)
    outlined = passes.outline_reg_scopes()(source)
    # Caller constants now select the branch before fixing the VF contract.
    assert "if " not in ir.python_print(helpers(outlined)[0])
    result = PassManager.get_strategy().run_passes(source)
    bundle = codegen.PTOCodegen().generate_bundle(result)
    (helper,) = bundle.helpers
    # Stores are effects even if their SSA result is unused. A constant false
    # branch drops its store, whereas an else branch keeps its store and phi.
    assert helper.source.count("pto.vmi.vstore") == int(flag > 0 or has_else)
    assert helper.source.count("pto.vmi.vadd") == int(flag < 0 and has_else)


def test_vf_binding_keeps_runtime_dependent_branch_restriction():
    source = program("""
with pl.reg_scope():
    value = pl.reg.load(src, lanes=64)
    if n > 0:
        unused = pl.reg.store(dst, value=value)
z = pl.tile.add(dst, dst)
""")
    result = PassManager.get_strategy().run_passes(source)
    with pytest.raises(ValueError, match="supports straight-line RegScope"):
        codegen.PTOCodegen().generate_bundle(result)


@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("loop_kind", ["range", "hardware_loop"])
def test_vf_binding_folds_trivial_loops_without_losing_stores_or_policy(count, loop_kind):
    source = program(f"""
count: pl.Scalar[pl.INDEX] = {count}
with pl.reg_scope():
    for i in pl.{loop_kind}(count):
        value = pl.reg.load(src, offset=i * 64, lanes=64)
        unused = pl.reg.store(dst, offset=i * 64, value=value)
    observed = pl.reg.load(dst, lanes=64)
z = pl.tile.add(dst, dst)
""")
    outlined = passes.outline_reg_scopes()(source)
    assert ("for " in ir.python_print(helpers(outlined)[0])) == (loop_kind == "hardware_loop" and count == 1)
    result = PassManager.get_strategy().run_passes(source)
    (helper,) = codegen.PTOCodegen().generate_bundle(result).helpers
    # A dead loop has no stores; one Auto iteration must execute its store once.
    # Require keeps a nonempty hardware loop even when its trip count is one.
    assert helper.source.count("pto.vmi.vstore") == count
    assert helper.source.count("pto.vmi.vload") == count + 1
    assert ("scf.for" in helper.source) == (loop_kind == "hardware_loop" and count == 1)


@pytest.mark.parametrize("count", [0, 1])
def test_vf_binding_preserves_zero_and_single_trip_register_carries(count):
    source = program(f"""
count: pl.Scalar[pl.INDEX] = {count}
with pl.reg_scope():
    seed = pl.reg.load(src, lanes=64)
    for i, (acc,) in pl.range(count, init_values=(seed,)):
        value = pl.reg.add(acc, seed)
        carried = pl.yield_(value)
    out = pl.reg.store(dst, value=carried)
z = pl.tile.add(out, out)
""")
    result = PassManager.get_strategy().run_passes(source)
    (helper,) = codegen.PTOCodegen().generate_bundle(result).helpers
    # Zero trips forwards the initial register; one trip forwards the yielded
    # addition. Neither may leave a dangling carry or a residual Auto loop.
    assert "scf.for" not in helper.source
    assert helper.source.count("pto.vmi.vadd") == count
    assert helper.source.count("pto.vmi.vstore") == 1


@pytest.mark.parametrize("bound", ["2", "n"])
def test_vf_binding_keeps_nontrivial_auto_loop_restriction(bound):
    source = program(f"""
count: pl.Scalar[pl.INDEX] = {bound}
with pl.reg_scope():
    for i in pl.range(count):
        value = pl.reg.load(src, lanes=64)
        unused = pl.reg.store(dst, value=value)
z = pl.tile.add(dst, dst)
""")
    result = PassManager.get_strategy().run_passes(source)
    with pytest.raises(ValueError, match="hardware loop lowering requires canonical integer loops"):
        codegen.PTOCodegen().generate_bundle(result)


def test_outer_unroll_substitutes_helper_inner_bound():
    source = program("""
for outer in pl.unroll(1, 3):
    with pl.reg_scope():
        seed = pl.reg.load(src, lanes=64)
        value = seed
        for inner in pl.unroll(2):
            value = pl.reg.add(value, seed)
        out = pl.reg.store(dst, value=value)
""")

    class InnerBound(ir.IRMutator):
        def __init__(self, outer: ir.Var):
            super().__init__()
            self.outer = outer
            self.changed = False

        def visit_for_stmt(self, op: ir.ForStmt) -> ir.Stmt:
            assert op.loop_var.name_hint == "inner"
            self.changed = True
            return ir.ForStmt(
                op.loop_var,
                op.start,
                self.outer,
                op.step,
                op.iter_args,
                op.body,
                op.return_vars,
                op.span,
                op.kind,
            )

    class DependentBound(ir.IRMutator):
        def visit_for_stmt(self, op: ir.ForStmt) -> ir.Stmt:
            assert op.loop_var.name_hint == "outer"
            inner = InnerBound(op.loop_var)
            body = inner.visit_stmt(op.body)
            assert inner.changed
            return ir.ForStmt(
                op.loop_var,
                op.start,
                op.stop,
                op.step,
                op.iter_args,
                body,
                op.return_vars,
                op.span,
                op.kind,
            )

    # Direct IR supports dependent unroll bounds even though the parser does not.
    # Disable printer/parser instruments for this parser-inexpressible input.
    with passes.PassContext([]):
        result = PassManager.get_strategy().run_passes(DependentBound().visit_program(source))
    bundle = codegen.PTOCodegen().generate_bundle(result)
    assert sorted(h.source.count("pto.vmi.vadd") for h in bundle.helpers) == [1, 2]


def test_rank_three_vf_signature_matches_body_and_caller():
    source = parse_program("""
@pl.program
class RankThree:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[2, 3, 64], pl.FP32]):
        src: pl.Tile[
            [2, 3, 64], pl.FP32, pl.MemorySpace.Vec,
            pl.TileView(valid_shape=[2, 3, 64], stride=[192, 64, 1])
        ] = pl.tile.load(x, [0, 0, 0], [2, 3, 64], target_memory=pl.MemorySpace.Vec)
        with pl.reg_scope():
            value = pl.reg.load(src, lanes=64)
            out = pl.reg.store(src, value=value)
        pl.tile.store(out, [0, 0, 0], x)
""")
    result = PassManager.get_strategy().run_passes(source)
    helper = helpers(result)[0]
    tiles = [p.type for p in helper.params if isinstance(p.type, ir.TileType)]
    assert tiles
    for tile in tiles:
        assert all(isinstance(d, ir.ConstInt) for d in tile.shape)
        assert [d.value for d in tile.shape if isinstance(d, ir.ConstInt)] == [6, 64]
    assert all(isinstance(t, ir.TileType) and len(t.shape) == 2 for t in helper.return_types)
    bundle = codegen.PTOCodegen().generate_bundle(result)
    assert len(bundle.helpers) == 1
    # VF borrows local Tiles; treating its GlobalVar as an opaque GM write
    # inserts a full barrier and breaks automatic MTE2/V/MTE3 synchronization.
    assert "pto.cmo.cacheinvalid" not in bundle.caller_source
    assert "pto.fence.barrier_all" not in bundle.caller_source


@pytest.mark.parametrize("pipe", [False, True])
def test_vf_tuple_copy_before_init_memref_preserves_captured_storage(pipe):
    if pipe:
        text = MIXED_SOURCE.replace(
            "            updated = pl.reg.store(vec, value=value)",
            "            updated = pl.reg.store(vec, value=value)\n"
            "            second = pl.reg.store(vec, value=value)",
        ).replace("pl.tile.store(updated,", "pl.tile.store(pl.tile.add(updated, second),")
        source = parse_program(text)
    else:
        source = program("""
with pl.reg_scope():
    value = pl.reg.load(src, lanes=64)
    first = pl.reg.store(dst, value=value)
    second = pl.reg.store(src, value=value)
z = pl.tile.add(first, second)
""")
    result = source
    with passes.PassContext([], enable_buffer_ir=False):
        for transform in PassManager.get_strategy().passes:
            if transform.get_name() == "InitMemRef":
                result = CopyVFTuples().visit_program(result)
                assert "copied_vf_results" in ir.python_print(result)
            result = transform(result)
            if transform.get_name() == "InitMemRef":
                break

    aliases: dict[ir.Var, list[ir.Expr]] = {}
    projected: list[ir.TileType] = []

    class CheckAliases(ir.IRVisitor):
        def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
            if isinstance(op.value, ir.Call) and isinstance(op.value.op, ir.GlobalVar):
                callee = result.get_function(op.value.op.name)
                if callee is not None and callee.func_type == ir.FunctionType.VF:
                    assert isinstance(callee.body, ir.SeqStmts)
                    ret = callee.body.stmts[-1]
                    assert isinstance(ret, ir.ReturnStmt)
                    aliases[op.var] = [
                        op.value.args[next(i for i, param in enumerate(callee.params) if param is v)]
                        for v in ret.value
                    ]
            elif isinstance(op.value, ir.Var) and op.value in aliases:
                aliases[op.var] = aliases[op.value]
            elif (
                isinstance(op.value, ir.TupleGetItemExpr)
                and isinstance(op.value.tuple, ir.Var)
                and op.value.tuple in aliases
            ):
                captured = aliases[op.value.tuple][op.value.index]
                assert isinstance(op.var.type, ir.TileType) and isinstance(captured.type, ir.TileType)
                assert op.var.type.memref is captured.type.memref
                projected.append(op.var.type)
            super().visit_assign_stmt(op)

    for function in result.functions.values():
        CheckAliases().visit_stmt(function.body)
    assert len(projected) == 2
    assert all((t.memref is None) == pipe for t in projected)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.HasMemRefs)
    passes.verify_properties(properties, result, "VF tuple copy regression")


def test_vf_tuple_allows_repeated_projection():
    source = program("""
with pl.reg_scope():
    value = pl.reg.load(src, lanes=64)
    first = pl.reg.store(dst, value=value)
    second = pl.reg.store(src, value=value)
z = pl.tile.add(first, second)
""")
    lowered = PassManager.get_strategy().run_passes(source)

    class RepeatProjection(ir.IRMutator):
        def visit_assign_stmt(self, op: ir.AssignStmt) -> ir.Stmt:
            if isinstance(op.value, ir.TupleGetItemExpr):
                duplicate = ir.Var("duplicate_" + op.var.name_hint, op.var.type, op.span)
                return ir.SeqStmts([op, ir.AssignStmt(duplicate, op.value, op.span)], op.span)
            return super().visit_assign_stmt(op)

    bundle = codegen.PTOCodegen().generate_bundle(RepeatProjection().visit_program(lowered))
    assert len(bundle.helpers) == 1


@pytest.mark.parametrize("region", ["branch", "loop"])
@pytest.mark.parametrize("different_storage", [False, True])
def test_vf_tuple_control_flow_result_keeps_pipe_storage_live(region, different_storage):
    text = MIXED_SOURCE.replace("def main(self, x:", "def main(self, n: pl.Scalar[pl.INDEX], x:")
    text = text.replace(
        "            updated = pl.reg.store(vec, value=value)",
        "            updated = pl.reg.store(vec, value=value)\n"
        "            second = pl.reg.store(vec, value=value)",
    )
    text = text.replace(
        "        output = pl.tile.store(updated, [0, 0], output)",
        "        output = pl.tile.store(updated, [0, 0], output)\n"
        "        output = pl.tile.store(second, [0, 0], output)",
    )
    source = passes.outline_reg_scopes()(parse_program(text))
    caller = source.get_function("main")
    assert caller is not None
    condition = ir.Gt(
        caller.params[0], ir.ConstInt(0, DataType.INDEX, caller.span), DataType.BOOL, caller.span
    )

    class CarryTuple(ir.IRMutator):
        def visit_assign_stmt(self, op: ir.AssignStmt) -> ir.Stmt:
            if isinstance(op.value, ir.Call) and isinstance(op.value.op, ir.GlobalVar):
                left = ir.Var("left_pair", op.var.type, op.span)
                right = ir.Var("right_pair", op.var.type, op.span)
                second_call = op.value
                right_stmts: list[ir.Stmt] = []
                if different_storage:
                    (arg,) = op.value.args
                    local = ir.Var("independent", arg.type, op.span)
                    add = ir.Call(ir.get_op("tile.add"), [arg, arg], arg.type, op.span)
                    right_stmts.append(ir.AssignStmt(local, add, op.span))
                    second_call = ir.Call(op.value.op, [local], op.var.type, op.span)
                right_stmts.extend(
                    [ir.AssignStmt(right, second_call, op.span), ir.YieldStmt([right], op.span)]
                )
                right_body = ir.SeqStmts(right_stmts, op.span)
                if region == "loop":
                    # Zero trips select the first call, a nonempty loop selects
                    # its last call. Project the whole tuple only after the join.
                    assert caller is not None
                    seed = ir.AssignStmt(left, op.value, op.span)
                    carry = ir.IterArg("pair", op.var.type, left, op.span)
                    index = ir.Var("i", caller.params[0].type, op.span)
                    zero = ir.ConstInt(0, DataType.INDEX, op.span)
                    one = ir.ConstInt(1, DataType.INDEX, op.span)
                    loop = ir.ForStmt(
                        index, zero, caller.params[0], one, [carry], right_body, [op.var], op.span
                    )
                    return ir.SeqStmts([seed, loop], op.span)
                then = ir.SeqStmts(
                    [ir.AssignStmt(left, op.value, op.span), ir.YieldStmt([left], op.span)], op.span
                )
                return ir.IfStmt(condition, then, right_body, [op.var], op.span)
            return super().visit_assign_stmt(op)

    source = CarryTuple().visit_program(source)
    if different_storage:
        message = "same received slot on every path" if region == "branch" else "including zero iterations"
        with pytest.raises(ValueError, match=message):
            lower_through(source, "ExpandMixedKernel")
        return
    result = lower_through(source, "InitMemRef")
    aiv = next(f for f in result.functions.values() if f.func_type == ir.FunctionType.AIV)
    emitted = ir.python_print(aiv)
    assert "left_pair" in emitted and "right_pair" in emitted
    assert emitted.rindex("tile.store") < emitted.index("system.tfree_to_aic")
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.HasMemRefs)
    passes.verify_properties(properties, result, "VF tuple branch result")
    assert isinstance(aiv.body, ir.SeqStmts)
    projected = [
        s.var.type
        for s in aiv_payload_stmts(aiv)
        if isinstance(s, ir.AssignStmt) and isinstance(s.value, ir.TupleGetItemExpr)
    ]
    assert len(projected) == 2
    assert all(isinstance(tile, ir.TileType) and tile.memref is None for tile in projected)


def test_vf_identity_result_inherits_mat_memory():
    source = parse_program("""
@pl.program
class Identity:
    @pl.function(type=pl.FunctionType.VF)
    def identity(
        self, x: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Mat]
    ) -> pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Mat]:
        with pl.reg_scope():
            n = pl.cast(1, pl.INDEX)
        return x
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, x: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Mat]):
        alias = self.identity(x)
        moved = pl.tile.move(alias, target_memory=pl.MemorySpace.Left)
""")
    result = passes.infer_tile_memory_space()(source)
    caller = result.get_function("kernel")
    assert caller is not None and isinstance(caller.body, ir.SeqStmts)
    assignment = caller.body.stmts[0]
    assert isinstance(assignment, ir.AssignStmt) and isinstance(assignment.var.type, ir.TileType)
    assert assignment.var.type.memory_space == ir.MemorySpace.Mat


@pytest.mark.parametrize("return_alias", [False, True])
def test_squeeze_stream_preserves_write_capture_and_flush_alias(return_alias):
    flush = (
        "written = pl.reg.squeeze_store_flush(curr_ureg, dst)"
        if return_alias
        else "_flushed = pl.reg.squeeze_store_flush(curr_ureg, dst)"
    )
    use = "z = pl.tile.add(written, written)" if return_alias else "z = pl.tile.add(dst, dst)"
    before = parse_program(f"""
@pl.program
class Kernel:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self):
        src = pl.tile.full([256], dtype=pl.UINT32, value=17)
        dst = pl.tile.full([320], dtype=pl.UINT32, value=123)
        with pl.reg_scope():
            init_ureg = pl.reg.squeeze_store_init(dst, offset=7)
            for i, (curr_ureg,) in pl.range(0, 4, init_values=(init_ureg,)):
                value = pl.reg.load(src, offset=i * 64, lanes=64)
                keep = pl.reg.active_mask(9, lanes=64)
                next_ureg = pl.reg.squeeze_store_step(curr_ureg, dst, value, keep)
                curr_ureg = pl.yield_(next_ureg)
            {flush}
        {use}
""")
    after = passes.outline_reg_scopes()(before)
    (helper,) = helpers(after)
    # SSA normalization may rename captures. Verify memory directions rather
    # than requiring source spellings or one particular capture ordering.
    assert len(helper.params) == 2
    assert helper.param_directions.count(ir.ParamDirection.In) == 1
    assert helper.param_directions.count(ir.ParamDirection.InOut) == 1
    destination = next(
        param
        for param, direction in zip(helper.params, helper.param_directions)
        if direction == ir.ParamDirection.InOut
    )
    assert all(isinstance(param.type, ir.TileType) for param in helper.params)
    if return_alias:
        assert isinstance(helper.body, ir.SeqStmts)
        result = helper.body.stmts[-1]
        assert isinstance(result, ir.ReturnStmt)
        assert result.value[0] is destination
    ir.assert_structural_equal(after, passes.outline_reg_scopes()(after))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
