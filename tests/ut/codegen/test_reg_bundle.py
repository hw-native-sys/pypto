# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Separate VF execution bundles preserve caller storage and diagnostic codegen."""

import json
import re
import subprocess

import pytest
from pypto import DataType, backend, codegen, ir, passes
from pypto.backend import pto_backend
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _generate_pto_bundle
from pypto.ir.builder import IRBuilder
from pypto.ir.op import tile
from test_reg_codegen import SPAN, build_kernel


@pytest.fixture(autouse=True)
def a5_backend():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def test_bundle_preserves_call_order_storage_and_diagnostic_output():
    generator = codegen.PTOCodegen()
    program = build_kernel()
    bundle = generator.generate_bundle(program, emit_source_loc=False)
    assert len(bundle.helpers) == 1
    helper = bundle.helpers[0]
    assert helper.parameter_types == ["!pto.ptr<f32, ub>"]
    assert helper.parameter_effects == ["read", "readwrite"]
    assert all(t.startswith("!pto.tile_buf<") for t in helper.caller_parameter_types)
    assert helper.adapter_symbol == helper.symbol + "_tile_adapter"
    caller = bundle.caller_source
    assert "pto.vecscope" not in caller
    assert "pto.vmi." not in caller
    assert caller.count("pto.tile_buf_addr") == 1
    assert f"func.func private @{helper.symbol}(" in caller
    main, adapter = caller.split(f"func.func private @{helper.adapter_symbol}(")
    call = main.index(f"func.call @{helper.adapter_symbol}")
    assert call < main.index("pto.tadd")
    assert "pto.tile_buf_addr" not in main
    assert "pto.barrier" not in caller
    assert 'pto.ptodsl.subkernel_helper = "simd"' in adapter
    assert 'pto.tileop.effects = ["read", "readwrite"]' in adapter
    assert f"func.call @{helper.symbol}(%ptr0)" in adapter
    assert "pto.tile_buf" not in helper.source
    assert "pto.alloc_tile" not in helper.source
    assert "pto.vmi.vadd" in helper.source
    assert "pto.vmi.vstore" in helper.source

    diagnostic = generator.generate(program, emit_source_loc=False)
    assert "pto.vecscope" in diagnostic
    assert "__pypto_vf_" not in diagnostic
    again = generator.generate_bundle(program, emit_source_loc=False)
    assert again.caller_source == caller
    assert again.helpers[0].source == helper.source


@pytest.mark.parametrize(
    "dtype,element,scalar_type",
    [(DataType.FP32, "f32", "f32"), (DataType.INT32, "si32", "i32"), (DataType.UINT32, "ui32", "ui32")],
)
def test_captures_scalar_once_and_collapses_store_versions(dtype, element, scalar_type):
    builder = IRBuilder()
    with builder.function("scalar_capture", span=SPAN, type=ir.FunctionType.AIV) as f:
        scalar = f.param("scale", ir.ScalarType(dtype))
        dst = builder.let(
            "dst",
            tile.full([128], dtype=dtype, value=0),
            type=ir.TileType([128], dtype, memory_space=ir.MemorySpace.Vec),
        )
        with builder.reg_scope(SPAN):
            x = builder.emit_reg("x", "reg.broadcast", [scalar], {"lanes": 64, "dtype": dtype}, SPAN)
            y = builder.emit_reg("y", "reg.broadcast", [scalar], {"lanes": 64, "dtype": dtype}, SPAN)
            first = builder.emit_reg_store("first", dst, x, span=SPAN)
            builder.emit_reg_store("second", first, y, span=SPAN)
    with passes.PassContext([]):
        program = passes.init_mem_ref()(ir.Program([f.get_result()], "Captures", SPAN))
    bundle = codegen.PTOCodegen().generate_bundle(program, emit_source_loc=False)
    assert bundle.helpers[0].parameter_types == [f"!pto.ptr<{element}, ub>", scalar_type]
    assert bundle.helpers[0].parameter_effects == ["readwrite", "none"]
    assert bundle.helpers[0].caller_parameter_types[1] == scalar_type
    assert bundle.caller_source.count("pto.tile_buf_addr") == 1
    assert bundle.helpers[0].source.count("pto.vmi.vstore") == 2


def test_index_capture_keeps_local_cast_inside_helper():
    builder = IRBuilder()
    with builder.function("index_capture", span=SPAN, type=ir.FunctionType.AIV) as f:
        count = f.param("count", ir.ScalarType(DataType.INDEX))
        with builder.reg_scope(SPAN):
            local = builder.let("count_i32", ir.cast(count, DataType.INT32, SPAN))
            builder.emit_reg("x", "reg.broadcast", [local], {"lanes": 64, "dtype": DataType.INT32}, SPAN)
    bundle = codegen.PTOCodegen().generate_bundle(ir.Program([f.get_result()], "IndexCapture", SPAN))
    assert bundle.helpers[0].parameter_types == ["index"]
    assert bundle.helpers[0].caller_parameter_types == ["index"]
    assert bundle.helpers[0].parameter_effects == ["none"]
    assert "pto.tile_buf_addr" not in bundle.caller_source
    assert "arith.index_cast" in bundle.helpers[0].source
    assert "arith.index_cast" not in bundle.caller_source


def test_same_named_scalar_captures_keep_distinct_identity():
    builder = IRBuilder()
    with builder.function("capture_identity", span=SPAN, type=ir.FunctionType.AIV) as f:
        first = f.param("scale", ir.ScalarType(DataType.FP32))
        second = f.param("scale", ir.ScalarType(DataType.FP32))
        with builder.reg_scope(SPAN):
            x = builder.emit_reg("x", "reg.broadcast", [second], {"lanes": 64, "dtype": DataType.FP32}, SPAN)
            y = builder.emit_reg("y", "reg.broadcast", [first], {"lanes": 64, "dtype": DataType.FP32}, SPAN)
            builder.emit_reg("z", "reg.add", [x, y], span=SPAN)
    bundle = codegen.PTOCodegen().generate_bundle(ir.Program([f.get_result()], "CaptureIdentity", SPAN))
    helper = bundle.helpers[0]
    assert helper.parameter_types == ["f32", "f32"]
    broadcasts = [line for line in helper.source.splitlines() if "pto.vmi.vbr" in line]
    assert len(broadcasts) == 2
    assert "%arg0" in broadcasts[0] and "%arg1" in broadcasts[1]


@pytest.mark.parametrize("dtype", [DataType.FP16, DataType.BF16])
def test_unsupported_scalar_capture_reports_source(dtype):
    builder = IRBuilder()
    with builder.function("unsupported_capture", span=SPAN, type=ir.FunctionType.AIV) as f:
        scalar = f.param("scale", ir.ScalarType(dtype), span=SPAN)
        with builder.reg_scope(SPAN):
            builder.emit_reg("x", "reg.broadcast", [scalar], {"lanes": 64, "dtype": dtype}, SPAN)
    with pytest.raises(ValueError, match="VF scalar capture supports") as error:
        codegen.PTOCodegen().generate_bundle(ir.Program([f.get_result()], "UnsupportedCapture", SPAN))
    assert "register_kernel.py" in str(error.value)


@pytest.mark.parametrize("control_flow", ["for", "while", "if"])
def test_control_flow_in_helper_is_explicitly_rejected(control_flow):
    builder = IRBuilder()
    generator = codegen.PTOCodegen()
    # The IR verifier rejects while before codegen; other control flow remains
    # legal diagnostic IR but is outside executable helper admission.
    message = "WhileStmt.*RegScope" if control_flow == "while" else "straight-line RegScope"
    with pytest.raises(ValueError, match=message):
        with builder.function("control_flow", span=SPAN, type=ir.FunctionType.AIV) as f:
            with builder.reg_scope(SPAN):
                if control_flow == "for":
                    induction = ir.Var("i", ir.ScalarType(DataType.INDEX), SPAN)
                    context = builder.for_loop(induction, 0, 2, 1, span=SPAN)
                elif control_flow == "while":
                    context = builder.while_loop(ir.ConstBool(True, SPAN), span=SPAN)
                else:
                    context = builder.if_stmt(ir.ConstBool(True, SPAN), span=SPAN)
                with context:
                    builder.emit_reg("mask", "reg.full_mask", [], {"lanes": 64}, SPAN)
        generator.generate_bundle(ir.Program([f.get_result()], "ControlFlow", SPAN))
    assert len(generator.generate_bundle(build_kernel()).helpers) == 1


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PTOAS, passes.MemoryPlanner.DSA_RP])
def test_execution_driver_rejects_unsupported_planners(planner):
    with pytest.raises(ValueError, match="PyPTO Tile memory planner"):
        _generate_pto_bundle(build_kernel(), skip_ptoas=False, memory_planner=planner, emit_source_loc=False)


def test_execution_driver_rejects_grouped_function():
    with pytest.raises(ValueError, match="grouped functions"):
        _generate_pto_bundle(
            build_kernel(),
            skip_ptoas=False,
            memory_planner=passes.MemoryPlanner.PYPTO,
            emit_source_loc=False,
            is_group=True,
        )


def test_bundle_rejects_peer_functions():
    program = build_kernel()
    main = program.get_function("register_kernel")
    assert main is not None
    peer = ir.Function("peer", [], [], ir.SeqStmts([], SPAN), SPAN, type=ir.FunctionType.AIV)
    with pytest.raises(ValueError, match="grouped/peer functions"):
        codegen.PTOCodegen().generate_bundle(ir.Program([main, peer], "Peers", SPAN))


@pytest.mark.parametrize("function_type", [ir.FunctionType.AIC, ir.FunctionType.InCore])
def test_bundle_rejects_non_aiv_function(function_type):
    scope = ir.RegScopeStmt("vf", ir.SeqStmts([], SPAN), SPAN)
    function = ir.Function("kernel", [], [], scope, SPAN, type=function_type)
    with pytest.raises(ValueError, match="RegScope.*requires.*AIV"):
        codegen.PTOCodegen().generate_bundle(ir.Program([function], "NonAIV", SPAN))


def test_bundle_rejects_non_a5_target():
    program = build_kernel()
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend910)
    with pytest.raises(ValueError, match="A5"):
        codegen.PTOCodegen().generate_bundle(program)


def test_bundle_rejects_nested_tile_expression_as_memory_capture():
    # Raw IR can use a Tile-valued expression as a reg.load operand. Extraction
    # requires a named external view and must diagnose it before resolving aliases.
    program = build_kernel()
    function = program.get_function("register_kernel")
    assert function is not None and isinstance(function.body, ir.SeqStmts)
    statements = list(function.body.stmts)
    scope_index = next(i for i, stmt in enumerate(statements) if isinstance(stmt, ir.RegScopeStmt))
    scope = statements[scope_index]
    assert isinstance(scope, ir.RegScopeStmt) and isinstance(scope.body, ir.SeqStmts)
    body = list(scope.body.stmts)
    load = body[0]
    assert isinstance(load, ir.AssignStmt) and isinstance(load.value, ir.Call)
    args = list(load.value.args)
    view = tile.reshape(args[0], [1, 128])
    # Keep the proven allocation, as a serialized typed view would. Otherwise
    # initialization validation rejects the missing MemRef before extraction.
    args[0] = ir.Call(view.op, view.args, view.kwargs, args[0].type, SPAN)
    call = ir.Call(load.value.op, args, load.value.kwargs, load.value.type, SPAN)
    body[0] = ir.AssignStmt(load.var, call, SPAN)
    statements[scope_index] = ir.RegScopeStmt("vf", ir.SeqStmts(body, SPAN), SPAN)
    function = ir.Function(
        function.name, [], [], ir.SeqStmts(statements, SPAN), SPAN, type=ir.FunctionType.AIV
    )
    with pytest.raises(ValueError, match="VF memory capture requires an external Tile variable"):
        codegen.PTOCodegen().generate_bundle(ir.Program([function], "NestedTile", SPAN))


def test_unsupported_planner_is_explicit_and_generator_recovers():
    generator = codegen.PTOCodegen()
    with pytest.raises(ValueError, match="memory planner"):
        generator.generate_bundle(build_kernel(), emit_tile_addr=False)
    assert len(generator.generate_bundle(build_kernel()).helpers) == 1


def test_register_bundle_after_tile_flattening_and_address_assignment():
    # The standard pipeline used to drop RegScope in flattening and retain the
    # old reg.store result MemRef after assigning a nonzero destination address.
    with passes.PassContext([]):
        program = build_kernel()
        program = passes.flatten_tile_nd_to_2d()(program)
        program = passes.memory_reuse()(program)
        program = passes.allocate_memory_addr()(program)
    bundle = codegen.PTOCodegen().generate_bundle(program)
    assert bundle.helpers
    assert "pto.vmi.vstore" in bundle.helpers[0].source


def test_multiple_explicit_scopes_have_distinct_symbols_and_order():
    builder = IRBuilder()
    with builder.function("two_scopes", span=SPAN, type=ir.FunctionType.AIV) as f:
        value = f.param("value", ir.ScalarType(DataType.INT32))
        dst = builder.let(
            "dst",
            tile.full([64], dtype=DataType.INT32, value=0),
            type=ir.TileType([64], DataType.INT32, memory_space=ir.MemorySpace.Vec),
        )
        for i in range(2):
            with builder.reg_scope(SPAN):
                reg = builder.emit_reg(
                    f"x{i}", "reg.broadcast", [value], {"lanes": 64, "dtype": DataType.INT32}, SPAN
                )
                dst = builder.emit_reg_store(f"out{i}", dst, reg, span=SPAN)
    with passes.PassContext([]):
        program = passes.init_mem_ref()(ir.Program([f.get_result()], "TwoScopes", SPAN))
    bundle = codegen.PTOCodegen().generate_bundle(program, emit_source_loc=False)
    assert len(bundle.helpers) == 2
    first, second = bundle.helpers
    assert first.symbol != second.symbol
    assert bundle.caller_source.index(f"func.call @{first.adapter_symbol}") < bundle.caller_source.index(
        f"func.call @{second.adapter_symbol}"
    )
    for helper in bundle.helpers:
        assert helper.parameter_types == ["!pto.ptr<si32, ub>", "i32"]
        assert helper.parameter_effects == ["readwrite", "none"]
        assert "builtin.unrealized_conversion_cast" in helper.source
        assert "i32 to si32" in helper.source
        assert "si32 -> !pto.vmi.vreg<64xsi32>" in helper.source


def test_load_store_load_aliases_share_one_readwrite_capture():
    builder = IRBuilder()
    with builder.function("alias_effect", span=SPAN, type=ir.FunctionType.AIV) as f:
        dst = builder.let(
            "dst",
            tile.full([128], dtype=DataType.FP32, value=0),
            type=ir.TileType([128], DataType.FP32, memory_space=ir.MemorySpace.Vec),
        )
        with builder.reg_scope(SPAN):
            offset = ir.ConstInt(0, DataType.INDEX, SPAN)
            first = builder.emit_reg("first", "reg.load", [dst, offset], {"lanes": 64}, SPAN)
            updated = builder.emit_reg_store("updated", dst, first, span=SPAN)
            builder.emit_reg("second", "reg.load", [updated, offset], {"lanes": 64}, SPAN)
    with passes.PassContext([]):
        program = passes.init_mem_ref()(ir.Program([f.get_result()], "AliasEffect", SPAN))
    bundle = codegen.PTOCodegen().generate_bundle(program, emit_source_loc=False)
    helper = bundle.helpers[0]
    assert helper.parameter_types == ["!pto.ptr<f32, ub>"]
    assert helper.parameter_effects == ["readwrite"]
    assert bundle.caller_source.count("pto.tile_buf_addr") == 1
    assert helper.source.count("pto.vmi.vload %arg0[") == 2
    assert "pto.vmi.vstore" in helper.source


def test_bundle_rejects_dynamic_register_offset():
    offset = ir.Var("offset", ir.ScalarType(DataType.INDEX), SPAN)
    with pytest.raises(ValueError, match="Cannot prove register offset bounds"):
        codegen.PTOCodegen().generate_bundle(build_kernel(offset=offset), emit_source_loc=False)


def test_capture_offsets_exclude_tile_view_origin():
    """The adapter extracts MemRef bases; VMI adds each view origin once."""
    builder = IRBuilder()
    with builder.function("view_origin", span=SPAN, type=ir.FunctionType.AIV) as f:
        first = builder.let(
            "first",
            tile.full([128], dtype=DataType.FP32, value=1),
            type=ir.TileType(
                [128],
                DataType.FP32,
                ir.MemRef("first_storage", 128, 512),
                tile_view=ir.TileView([128], [1], ir.ConstInt(32, DataType.INDEX, SPAN)),
                memory_space=ir.MemorySpace.Vec,
            ),
        )
        second = builder.let(
            "second",
            tile.full([128], dtype=DataType.FP32, value=0),
            type=ir.TileType(
                [128],
                DataType.FP32,
                ir.MemRef("second_storage", 768, 512),
                memory_space=ir.MemorySpace.Vec,
            ),
        )
        with builder.reg_scope(SPAN):
            value = builder.emit_reg(
                "value", "reg.load", [first, ir.ConstInt(0, DataType.INDEX, SPAN)], {"lanes": 64}, SPAN
            )
            builder.emit_reg_store("updated", second, value, span=SPAN)
    helper = codegen.PTOCodegen().generate_bundle(ir.Program([f.get_result()], "Origins", SPAN)).helpers[0]
    assert helper.capture_byte_offsets == [0, 640]
    assert helper.parameter_indices == [0]
    assert "arith.constant 160 : index" in helper.source
    assert "arith.constant 32 : index" in helper.source
    assert helper.source.count("arith.addi") == 1


def test_external_plain_alias_shares_one_capture():
    builder = IRBuilder()
    with builder.function("plain_alias", span=SPAN, type=ir.FunctionType.AIV) as f:
        original = builder.let(
            "original",
            tile.full([128], dtype=DataType.FP32, value=1),
            type=ir.TileType([128], DataType.FP32, memory_space=ir.MemorySpace.Vec),
        )
        alias = builder.let("alias", original, type=original.type)
        with builder.reg_scope(SPAN):
            value = builder.emit_reg(
                "value", "reg.load", [alias, ir.ConstInt(0, DataType.INDEX, SPAN)], {"lanes": 64}, SPAN
            )
            builder.emit_reg_store("updated", original, value, span=SPAN)
    with passes.PassContext([]):
        program = passes.init_mem_ref()(ir.Program([f.get_result()], "Aliases", SPAN))
    helper = codegen.PTOCodegen().generate_bundle(program).helpers[0]
    assert helper.parameter_effects == ["readwrite"]
    assert helper.capture_byte_offsets == [0]


@pytest.mark.parametrize("offset", [0, 32, 64])
@pytest.mark.parametrize("reverse", [False, True])
def test_distinct_captured_views_preserve_shared_base(offset, reverse):
    """The adapter retains both effects; the helper reconstructs static aliases."""
    builder = IRBuilder()
    with builder.function("overlapping_views", span=SPAN, type=ir.FunctionType.AIV) as f:
        base = builder.let(
            "base",
            tile.full([1, 128], dtype=DataType.FP32, value=0),
            type=ir.TileType([1, 128], DataType.FP32, memory_space=ir.MemorySpace.Vec),
        )
        first_slice = builder.let("first_slice", tile.slice(base, [1, 64], [0, offset if reverse else 0]))
        second_slice = builder.let("second_slice", tile.slice(base, [1, 64], [0, 0 if reverse else offset]))
        first = builder.let("first", tile.reshape(first_slice, [64]))
        second = builder.let("second", tile.reshape(second_slice, [64]))
        with builder.reg_scope(SPAN):
            value = builder.emit_reg(
                "value",
                "reg.broadcast",
                [ir.ConstFloat(5.0, DataType.FP32, SPAN)],
                {"lanes": 64, "dtype": DataType.FP32},
                SPAN,
            )
            builder.emit_reg_store("updated", first, value, span=SPAN)
            builder.emit_reg(
                "loaded",
                "reg.load",
                [second, ir.ConstInt(0, DataType.INDEX, SPAN)],
                {"lanes": 64},
                SPAN,
            )
    with passes.PassContext([]):
        program = passes.init_mem_ref()(ir.Program([f.get_result()], "SharedViews", SPAN))
        program = passes.flatten_tile_nd_to_2d()(program)
    helper = codegen.PTOCodegen().generate_bundle(program, emit_source_loc=False).helpers[0]
    assert helper.parameter_types == ["!pto.ptr<f32, ub>"]
    assert helper.capture_byte_offsets == ([offset * 4, 0] if reverse else [0, offset * 4])
    assert helper.parameter_indices == ([1] if reverse and offset else [0])
    assert ("pto.addptr" in helper.source) == (offset != 0)
    assert helper.parameter_effects == ["readwrite", "read"]


def test_bundle_manifest_preserves_both_interfaces_and_effects(tmp_path, monkeypatch):
    program = build_kernel(masked=True)
    bundle = codegen.PTOCodegen().generate_bundle(program, emit_source_loc=False)
    function = program.get_function("register_kernel")
    assert function is not None
    unit = pto_backend._CodegenUnit(
        name="register_kernel",
        pto_code=bundle.caller_source,
        funcs=[function],
        is_group=False,
        stage_record=pto_backend.StageRecord(name="test", start=0),
        helpers=tuple(bundle.helpers),
    )
    monkeypatch.setattr(pto_backend, "_emit_single_function_output", lambda *args: None)
    result = pto_backend._emit_unit(unit, str(tmp_path), skip_ptoas=True)
    assert result.error is None
    manifest = json.loads(result.files["vpto/register_kernel/manifest.json"])
    assert manifest["schema_version"] == 3
    assert manifest["effect_scope"] == "whole-tile"
    assert manifest["synchronization"] == "ptoas-insert-sync"
    entry = manifest["helpers"][0]
    helper = bundle.helpers[0]
    assert entry["adapter_symbol"] == helper.adapter_symbol
    assert entry["caller_parameter_types"] == helper.caller_parameter_types
    assert entry["parameter_types"] == helper.parameter_types
    assert entry["parameter_indices"] == helper.parameter_indices
    assert entry["capture_byte_offsets"] == helper.capture_byte_offsets
    assert entry["parameter_effects"] == ["read", "readwrite"]


def test_execution_rejects_old_ptoas_before_emitting_tile_adapter(tmp_path, monkeypatch):
    program = build_kernel()
    bundle = codegen.PTOCodegen().generate_bundle(program)
    function = program.get_function("register_kernel")
    assert function is not None
    unit = pto_backend._CodegenUnit(
        name="register_kernel",
        pto_code=bundle.caller_source,
        funcs=[function],
        is_group=False,
        stage_record=pto_backend.StageRecord(name="test", start=0),
        helpers=tuple(bundle.helpers),
    )
    monkeypatch.setattr(pto_backend, "_find_ptoas_binary", lambda: "old-ptoas")

    def reject(binary):
        raise RuntimeError("VF Tile adapter synchronization requires PTOAS >= v0.70")

    monkeypatch.setattr(pto_backend, "_check_ptoas_version", reject)
    result = pto_backend._emit_unit(unit, str(tmp_path), skip_ptoas=False)
    assert isinstance(result.error, RuntimeError)
    assert "v0.70" in str(result.error)
    assert result.files == {}


def test_mixed_element_memory_captures_are_explicitly_rejected():
    builder = IRBuilder()
    with builder.function("mixed", span=SPAN, type=ir.FunctionType.AIV) as f:
        first = builder.let(
            "first",
            tile.full([64], dtype=DataType.FP32, value=1),
            type=ir.TileType([64], DataType.FP32, memory_space=ir.MemorySpace.Vec),
        )
        second = builder.let(
            "second",
            tile.full([64], dtype=DataType.INT32, value=2),
            type=ir.TileType([64], DataType.INT32, memory_space=ir.MemorySpace.Vec),
        )
        with builder.reg_scope(SPAN):
            for name, view in (("x", first), ("y", second)):
                builder.emit_reg(
                    "loaded_" + name,
                    "reg.load",
                    [view, ir.ConstInt(0, DataType.INDEX, SPAN)],
                    {"lanes": 64},
                    SPAN,
                )
    with passes.PassContext([]):
        program = passes.init_mem_ref()(ir.Program([f.get_result()], "Mixed", SPAN))
    with pytest.raises(ValueError, match="identical element types"):
        codegen.PTOCodegen().generate_bundle(program, emit_source_loc=False)


def test_dynamic_physical_capture_address_is_explicitly_rejected():
    builder = IRBuilder()
    with builder.function("dynamic_address", span=SPAN, type=ir.FunctionType.AIV) as f:
        address = f.param("address", ir.ScalarType(DataType.INDEX))
        allocation = ir.MemRef("allocation", 0, 256)
        memref = ir.MemRef(allocation.base_, address, 256, SPAN)
        value = builder.let(
            "value",
            tile.full([64], dtype=DataType.FP32, value=1),
            type=ir.TileType([64], DataType.FP32, memref, memory_space=ir.MemorySpace.Vec),
        )
        with builder.reg_scope(SPAN):
            builder.emit_reg(
                "loaded", "reg.load", [value, ir.ConstInt(0, DataType.INDEX, SPAN)], {"lanes": 64}, SPAN
            )
    with pytest.raises(ValueError, match="Cannot prove register offset bounds"):
        codegen.PTOCodegen().generate_bundle(
            ir.Program([f.get_result()], "DynamicAddress", SPAN), emit_source_loc=False
        )


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("local_name", ["arg0", "arg1", "capture1"])
def test_helper_synthetic_names_do_not_collide(tmp_path, local_name):
    builder = IRBuilder()
    with builder.function("names", span=SPAN, type=ir.FunctionType.AIV) as f:
        scale = f.param("scale", ir.ScalarType(DataType.FP32))
        views = [
            builder.let(
                name,
                tile.full([64], dtype=DataType.FP32, value=0),
                type=ir.TileType(
                    [64],
                    DataType.FP32,
                    ir.MemRef(name, offset, 256),
                    memory_space=ir.MemorySpace.Vec,
                ),
            )
            for name, offset in (("src", 0), ("dst", 256))
        ]
        with builder.reg_scope(SPAN):
            loaded = builder.emit_reg(
                local_name,
                "reg.load",
                [views[0], ir.ConstInt(0, DataType.INDEX, SPAN)],
                {"lanes": 64},
                SPAN,
            )
            factor = builder.emit_reg(
                "factor", "reg.broadcast", [scale], {"lanes": 64, "dtype": DataType.FP32}, SPAN
            )
            value = builder.emit_reg("sum", "reg.add", [loaded, factor], span=SPAN)
            builder.emit_reg_store("updated", views[1], value, span=SPAN)
    helper = codegen.PTOCodegen().generate_bundle(ir.Program([f.get_result()], "Names", SPAN)).helpers[0]
    definitions = re.findall(r"^\s*(%[\w.$]+) =", helper.source, re.MULTILINE)
    definitions += re.findall(r"(%arg\d+):", helper.source)
    assert len(definitions) == len(set(definitions)), helper.source
    compiler = find_ptoas_binary()
    if compiler is None:
        pytest.skip("PTOAS is unavailable")
    source = tmp_path / "helper.pto"
    source.write_text(helper.source)
    result = subprocess.run(
        [
            compiler,
            str(source),
            "--pto-arch=a5",
            "--pto-level=level3",
            "--emit-vpto-llvm-ir",
            "-o",
            str(tmp_path / "helper.ll"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("capture_role", ["source", "destination"])
@pytest.mark.parametrize("view_kind", ["direct", "alias", "reshape"])
@pytest.mark.parametrize("same_source", [False, True])
def test_bundle_checks_runtime_selected_capture_storage(capture_role, view_kind, same_source):
    """A representative offset cannot relate selected and fixed capture pointers."""
    builder = IRBuilder()
    with builder.function("selected_capture", span=SPAN, type=ir.FunctionType.AIV) as f:
        condition = f.param("condition", ir.ScalarType(DataType.BOOL), SPAN)
        tile_type = ir.TileType(
            [1, 128],
            DataType.FP32,
            tile_view=ir.TileView([1, 128], [128, 1], 0),
            memory_space=ir.MemorySpace.Vec,
        )
        first = builder.let("first", tile.full([1, 128], dtype=DataType.FP32, value=1), type=tile_type)
        second = builder.let("second", tile.full([1, 128], dtype=DataType.FP32, value=2), type=tile_type)
        fixed = builder.let("fixed", tile.full([1, 128], dtype=DataType.FP32, value=0), type=tile_type)
        with builder.if_stmt(condition, SPAN) as branch:
            branch.return_var("selected", tile_type, SPAN)
            builder.emit(ir.YieldStmt([first], SPAN))
            branch.else_(SPAN)
            builder.emit(ir.YieldStmt([first if same_source else second], SPAN))
        selected = branch.output(0)
        if view_kind == "alias":
            selected = builder.let("selected_alias", selected, type=selected.type)
        elif view_kind == "reshape":
            # Register accesses require an explicit contiguous View for 2-D
            # operands. Preserve the distinct reshape descriptor while making
            # the fixture legal before executable-capture admission runs.
            selected = builder.let(
                "selected_view",
                tile.reshape(selected, [2, 64]),
                type=ir.TileType(
                    [2, 64],
                    DataType.FP32,
                    tile_view=ir.TileView([2, 64], [64, 1], 0),
                    memory_space=ir.MemorySpace.Vec,
                ),
            )
        source = selected if capture_role == "source" else fixed
        destination = selected if capture_role == "destination" else fixed
        with builder.reg_scope(SPAN):
            value = builder.emit_reg(
                "value", "reg.load", [source, ir.ConstInt(0, DataType.INDEX, SPAN)], {"lanes": 64}, SPAN
            )
            builder.emit_reg_store("updated", destination, value, span=SPAN)
    with passes.PassContext([], enable_buffer_ir=False):
        program = passes.init_mem_ref()(ir.Program([f.get_result()], "SelectedCapture", SPAN))
        program = passes.allocate_memory_addr()(program)
    generator = codegen.PTOCodegen()
    if same_source:
        # Identical yielded SSAs retain their actual local allocation; an If
        # alone must not exclude an otherwise supported fixed capture.
        assert len(generator.generate_bundle(program).helpers) == 1
    else:
        with pytest.raises(ValueError, match="runtime-selected Tiles") as error:
            generator.generate_bundle(program)
        assert "register_kernel.py" in str(error.value)
        assert len(generator.generate_bundle(build_kernel()).helpers) == 1


@pytest.mark.parametrize("view_kind", ["direct", "slice", "reshape", "alias"])
def test_bundle_rejects_parameter_backed_captures_and_views(view_kind):
    builder = IRBuilder()
    with builder.function("incoming", span=SPAN, type=ir.FunctionType.AIV) as f:
        src = f.param(
            "src",
            ir.TileType(
                [2, 64],
                DataType.FP32,
                tile_view=ir.TileView([2, 64], [64, 1], 0),
                memory_space=ir.MemorySpace.Vec,
            ),
        )
        dst = f.param(
            "dst",
            ir.TileType(
                [2, 64],
                DataType.FP32,
                tile_view=ir.TileView([2, 64], [64, 1], 0),
                memory_space=ir.MemorySpace.Vec,
            ),
        )
        if view_kind == "slice":
            src = builder.let(
                "view",
                tile.slice(src, [1, 64], [1, 0]),
                type=ir.TileType(
                    [1, 64],
                    DataType.FP32,
                    tile_view=ir.TileView([1, 64], [64, 1], 0),
                    memory_space=ir.MemorySpace.Vec,
                ),
            )
        elif view_kind == "reshape":
            src = builder.let("view", tile.reshape(src, [128]))
        elif view_kind == "alias":
            src = builder.let("view", src, type=src.type)
        with builder.reg_scope(SPAN):
            value = builder.emit_reg(
                "value", "reg.load", [src, ir.ConstInt(0, DataType.INDEX, SPAN)], {"lanes": 64}, SPAN
            )
            builder.emit_reg_store("updated", dst, value, span=SPAN)
    with passes.PassContext([]):
        program = passes.init_mem_ref()(ir.Program([f.get_result()], "Incoming", SPAN))
    generator = codegen.PTOCodegen()
    with pytest.raises(ValueError, match="parameter-backed Tiles"):
        generator.generate_bundle(program)
    # Reusing the emitter after rejection must not taint a local allocation.
    assert len(generator.generate_bundle(build_kernel()).helpers) == 1


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("reshape", [False, True])
def test_adapter_preserves_static_view_operand_type(tmp_path, reshape):
    builder = IRBuilder()
    with builder.function("static_view", span=SPAN, type=ir.FunctionType.AIV) as f:
        base = builder.let(
            "base",
            tile.full([2, 64], dtype=DataType.FP32, value=0),
            type=ir.TileType(
                [2, 64],
                DataType.FP32,
                tile_view=ir.TileView([2, 64], [64, 1], 0),
                memory_space=ir.MemorySpace.Vec,
            ),
        )
        view = builder.let(
            "view",
            tile.slice(base, [1, 64], [1, 0]),
            type=ir.TileType(
                [1, 64],
                DataType.FP32,
                tile_view=ir.TileView([1, 64], [64, 1], 0),
                memory_space=ir.MemorySpace.Vec,
            ),
        )
        if reshape:
            view = builder.let("reshaped", tile.reshape(view, [64]))
        with builder.reg_scope(SPAN):
            value = builder.emit_reg(
                "value", "reg.load", [view, ir.ConstInt(0, DataType.INDEX, SPAN)], {"lanes": 64}, SPAN
            )
            builder.emit_reg_store("updated", view, value, span=SPAN)
    with passes.PassContext([]):
        program = passes.init_mem_ref()(ir.Program([f.get_result()], "StaticView", SPAN))
        program = passes.flatten_tile_nd_to_2d()(program)
        program = passes.allocate_memory_addr()(program)
    bundle = codegen.PTOCodegen().generate_bundle(program, emit_source_loc=False)
    helper = bundle.helpers[0]
    # slice emits static valid dimensions; reshape currently emits a dynamic
    # descriptor. Both calls must use their defining SSA type and pass PTOAS.
    validity = "v_row=?, v_col=?" if reshape else "v_row=1, v_col=64"
    assert validity in helper.caller_parameter_types[0]
    compiler = find_ptoas_binary()
    if compiler is None:
        pytest.skip("PTOAS is unavailable")
    source = tmp_path / "caller.pto"
    source.write_text(bundle.caller_source)
    result = subprocess.run(
        [
            compiler,
            str(source),
            "--pto-arch=a5",
            "--pto-level=level3",
            "--enable-insert-sync",
            "-o",
            str(tmp_path / "caller.cpp"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
