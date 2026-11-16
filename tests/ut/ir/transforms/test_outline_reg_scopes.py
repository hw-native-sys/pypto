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
from pypto import backend, codegen, ir, passes
from pypto.ir.pass_manager import PassManager
from pypto.language.parser import parse_program


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


@pytest.mark.parametrize("local_copy", [False, True])
def test_mixed_kernel_keeps_vf_on_aiv_and_passes_pipe_storage(local_copy):
    source_text = """
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
    if local_copy:
        bundle = codegen.PTOCodegen().generate_bundle(unit, emit_source_loc=False)
        assert len(bundle.helpers) == 1
        assert bundle.helpers[0].parameter_effects == ["readwrite"]
        # Upstream grouped execution must retain the outlined helper's owner:
        # the AIC peer has no VF helper, while the AIV member calls this one.
        grouped_unit = ir.Program([*callers, helper], "GroupedDeviceUnit", result.span)
        grouped = codegen.PTOCodegen().generate_bundle(grouped_unit, emit_source_loc=False)
        assert len(grouped.helpers) == 1
        assert grouped.entry_helpers[aiv.name] == [grouped.helpers[0].symbol]
        aic = next(f for f in callers if f.func_type == ir.FunctionType.AIC)
        assert grouped.entry_helpers[aic.name] == []
    else:
        # The existing VF ABI requires static allocated UB addresses; outlining
        # preserves that restriction while retaining pipe lifetime in the IR.
        with pytest.raises(ValueError, match="statically planned local UB"):
            codegen.PTOCodegen().generate_bundle(unit, emit_source_loc=False)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
