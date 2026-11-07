# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""SIMT frontend execution boundaries, capture identity and implicit outputs."""

import textwrap

import pypto.language as pl
import pytest
from pypto import DataType, Error, backend, ir, passes
from pypto.language.parser import parse_program
from pypto.language.parser.diagnostics import ParserError


@pytest.fixture(autouse=True)
def a5_backend():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def program(body: str, *, direction: str = "InOut", kind: str = "AIV") -> ir.Program:
    source = f"""
@pl.program
class Kernel:
    @pl.function(type=pl.FunctionType.{kind})
    def kernel(self, src: pl.Tensor[[1, 32], pl.FP32],
               dst: pl.{direction}[pl.Tensor[[1, 32], pl.FP32]], n: pl.Scalar[pl.INT32]):
{textwrap.indent(textwrap.dedent(body), "        ")}
"""
    return parse_program(source)


COPY = """
with pl.simt_scope(dims=(32, 1, 1)):
    tid = pl.simt.thread_idx(pl.simt.Axis.X)
    if tid < n:
        value = pl.simt.load(src, (0, tid))
        pl.simt.store(dst, (0, tid), value)
return dst
"""


def function(prog: ir.Program, name: str) -> ir.Function:
    result = prog.get_function(name)
    assert result is not None
    return result


def launches(prog: ir.Program) -> list[ir.SimtLaunchStmt]:
    result: list[ir.SimtLaunchStmt] = []

    class Collector(ir.IRVisitor):
        def visit_simt_launch_stmt(self, op: ir.SimtLaunchStmt) -> None:
            result.append(op)

    Collector().visit_program(prog)
    return result


def test_outline_capture_order_directions_and_implicit_result():
    before = program(COPY)
    after = passes.outline_simt_scopes()(before)
    launch = launches(after)[0]
    helper = function(after, launch.callee.name)
    names = []
    for arg in launch.args:
        assert isinstance(arg, ir.Var)
        names.append(arg.name_hint)
    assert names == ["n", "src", "dst"]
    assert helper.param_directions == [ir.ParamDirection.In, ir.ParamDirection.In, ir.ParamDirection.InOut]
    assert launch.result_param_indices == [2]
    assert launch.results[0] is launch.args[2]
    assert helper.func_type == ir.FunctionType.SIMT
    assert not helper.return_types
    after = passes.convert_to_ssa()(after)
    launch = launches(after)[0]
    assert launch.results[0] is not launch.args[2]
    body = function(after, "kernel").body
    assert isinstance(body, ir.SeqStmts)
    ret = body.stmts[-1]
    assert isinstance(ret, ir.ReturnStmt)
    assert ret.value[0] is launch.results[0]


def test_region_roundtrip_serialization_hash_clone():
    before = program(COPY)
    printed = ir.python_print(before)
    assert "with pl.simt_scope(dims=(" in printed
    restored = parse_program(printed)
    ir.assert_structural_equal(before, restored)
    assert ir.structural_hash(before) == ir.structural_hash(restored)
    ir.assert_structural_equal(before, ir.deserialize(ir.serialize(before)))
    body = function(before, "kernel").body
    cloned, _ = ir.deep_clone(body)
    ir.assert_structural_equal(body, cloned, enable_auto_mapping=True)


def test_scalar_shadow_does_not_escape():
    before = program("""
with pl.simt_scope(dims=(32, 1, 1)):
    n = n + pl.const(1, pl.INT32)
    pl.simt.store(dst, (0, 0), pl.cast(n, pl.FP32))
return n
""")
    ir.assert_structural_equal(before, parse_program(ir.python_print(before)))
    ir.assert_structural_equal(before, ir.deserialize(ir.serialize(before)))
    original = function(before, "kernel")
    cloned, _ = ir.deep_clone(original.body)
    ir.assert_structural_equal(original.body, cloned, enable_auto_mapping=True)
    assert isinstance(cloned, ir.SeqStmts)
    returned = cloned.stmts[-1]
    assert isinstance(returned, ir.ReturnStmt)
    assert returned.value[0] is original.params[2]
    after = passes.convert_to_ssa()(passes.outline_simt_scopes()(before))
    kernel = function(after, "kernel")
    assert isinstance(kernel.body, ir.SeqStmts)
    ret = kernel.body.stmts[-1]
    assert isinstance(ret, ir.ReturnStmt)
    assert ret.value[0] is kernel.params[2]
    helper = function(after, launches(after)[0].callee.name)
    assert helper.params[0] is not kernel.params[2]


@pytest.mark.parametrize("dims", ["(0, 1, 1)", "(True, 1, 1)", "(1.0, 1, 1)", "(1, 1)", "(n, 1, 1)"])
def test_invalid_scope_dimensions(dims):
    with pytest.raises((Error, ValueError, ParserError), match="SIMT dimensions"):
        program(COPY.replace("(32, 1, 1)", dims))


@pytest.mark.parametrize("axis", ["0", '"x"', "pl.MemorySpace.Vec", "n"])
def test_invalid_axis(axis):
    with pytest.raises((Error, ValueError, ParserError), match="typed|resolve"):
        program(COPY.replace("pl.simt.Axis.X", axis))


@pytest.mark.parametrize("kind", ["InCore", "AIC", "SIMT", "Orchestration"])
def test_scope_requires_explicit_aiv(kind):
    with pytest.raises((Error, ValueError, ParserError), match="explicit AIV"):
        program(COPY, kind=kind)


def test_readonly_write_rejected():
    with pytest.raises((Error, ValueError, ParserError), match="enclosing In parameter"):
        passes.outline_simt_scopes()(program(COPY.replace("pl.simt.store(dst", "pl.simt.store(src")))


def test_uninitialized_output_rejected():
    with pytest.raises((Error, ValueError, ParserError), match="initialized storage"):
        passes.outline_simt_scopes()(program(COPY, direction="Out"))


@pytest.mark.parametrize("body", ["return", "return n"])
def test_return_cannot_cross_scope(body):
    with pytest.raises((Error, ValueError, ParserError), match="return.*SIMT region"):
        passes.outline_simt_scopes()(program(f"with pl.simt_scope(dims=(32, 1, 1)):\n    {body}"))


def test_new_scalar_is_not_visible_after_scope():
    with pytest.raises((Error, ValueError, ParserError), match="tid"):
        program(COPY.replace("return dst", "return tid"))


def test_scope_rejects_resource_knobs_and_nested_scope():
    for extra in (", grid_dims=(1, 1, 1)", ", simt_max_regs=64"):
        with pytest.raises((Error, ValueError, ParserError), match="requires only dims"):
            program(COPY.replace("dims=(32, 1, 1)", "dims=(32, 1, 1)" + extra))
    with pytest.raises((Error, ValueError, ParserError), match="Nested simt_scope"):
        program("with pl.simt_scope(dims=(32, 1, 1)):\n" + textwrap.indent(COPY, "    "))


def test_scope_keeps_outer_control_flow_placement():
    before = program("""
for i in pl.range(2):
    if n > 0:
        with pl.simt_scope(dims=(32, 1, 1)):
            tid = pl.simt.thread_idx(pl.simt.Axis.X)
            pl.simt.store(dst, (0, tid), pl.cast(i, pl.FP32))
return dst
""")
    after = passes.convert_to_ssa()(passes.outline_simt_scopes()(before))
    assert len(launches(after)) == 1
    selected = passes.IRPropertySet()
    selected.insert(passes.IRProperty.SSAForm)
    selected.insert(passes.IRProperty.UseAfterDef)
    assert not passes.PropertyVerifierRegistry.verify(selected, after)


def test_axis_public_alias_and_builder():
    assert pl.simt.Axis is ir.SimtAxis
    builder = ir.IRBuilder()
    span = ir.Span.unknown()
    dims = [ir.ConstInt(x, DataType.INT32, span) for x in (32, 1, 1)]
    with builder.function("kernel", type=ir.FunctionType.AIV) as fn:
        with builder.simt_scope(dims):
            builder.let("lane", pl.simt.lane_id().unwrap())
    assert isinstance(fn.get_result().body, ir.SimtScopeStmt)
    assert not isinstance(fn.get_result().body, ir.ScopeStmt)


@pytest.mark.parametrize("threads", [1, 256, 257, 512, 513, 1024, 1025, 2048])
def test_scope_resource_boundaries(threads):
    after = passes.outline_simt_scopes()(program(COPY.replace("(32, 1, 1)", f"({threads}, 1, 1)")))
    assert ir.infer_simt_resources(launches(after)[0]).max_threads == threads


def test_readonly_view_cannot_gain_write_permission():
    before = program("""
view = pl.tensor.slice(src, [1, 16], [0, 0])
with pl.simt_scope(dims=(16, 1, 1)):
    pl.simt.store(view, (0, 0), pl.const(1.0, pl.FP32))
""")
    with pytest.raises((Error, ValueError), match="enclosing In parameter"):
        passes.outline_simt_scopes()(before)


def test_out_capture_after_full_initialization():
    before = program(
        """
value = pl.tile.load(src, [0, 0], [1, 32])
dst = pl.tile.store(value, [0, 0], dst)
with pl.simt_scope(dims=(32, 1, 1)):
    pl.simt.store(dst, (0, 0), pl.const(1.0, pl.FP32))
return dst
""",
        direction="Out",
    )
    assert len(launches(passes.outline_simt_scopes()(before))) == 1


def test_partial_initialization_is_not_enough():
    before = program(
        """
value = pl.tile.load(src, [0, 0], [1, 16])
dst = pl.tile.store(value, [0, 0], dst)
with pl.simt_scope(dims=(32, 1, 1)):
    pl.simt.store(dst, (0, 0), pl.const(1.0, pl.FP32))
""",
        direction="Out",
    )
    with pytest.raises((Error, ValueError), match="initialized storage"):
        passes.outline_simt_scopes()(before)


def test_scope_inside_register_region_rejected():
    with pytest.raises(ParserError, match="inside reg_scope"):
        program("with pl.reg_scope():\n" + textwrap.indent(COPY, "    "))


def test_storage_rebinding_inside_region_rejected():
    before = program("""
with pl.simt_scope(dims=(32, 1, 1)):
    dst = src
""")
    with pytest.raises((Error, ValueError), match="rebind storage"):
        passes.outline_simt_scopes()(before)


def test_empty_tile_requires_initialization():
    before = program("""
empty = pl.tile.create([1, 32], pl.FP32)
with pl.simt_scope(dims=(32, 1, 1)):
    pl.simt.store(empty, (0, 0), pl.const(1.0, pl.FP32))
""")
    with pytest.raises((Error, ValueError), match="initialized storage"):
        passes.outline_simt_scopes()(before)


def test_ssa_requires_outlining_first():
    with pytest.raises((Error, ValueError), match="OutlineSimtScopes before ConvertToSSA"):
        passes.convert_to_ssa()(program(COPY))


def test_outline_matches_canonical_helper_launch():
    before = program(COPY)
    expected = parse_program("""
@pl.program
class Kernel:
    @pl.function(type=pl.FunctionType.SIMT)
    def kernel_simt_0(self, n: pl.Scalar[pl.INT32], src: pl.Tensor[[1, 32], pl.FP32],
                      dst: pl.InOut[pl.Tensor[[1, 32], pl.FP32]]):
        tid = pl.simt.thread_idx(pl.simt.Axis.X)
        if tid < n:
            value = pl.simt.load(src, (0, tid))
            pl.simt.store(dst, (0, tid), value)

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, src: pl.Tensor[[1, 32], pl.FP32],
               dst: pl.InOut[pl.Tensor[[1, 32], pl.FP32]], n: pl.Scalar[pl.INT32]):
        dst = pl.simt.launch(self.kernel_simt_0, n, src, dst,
                             block_dims=(32, 1, 1), result_param_indices=(2,))
        return dst
""")
    ir.assert_structural_equal(passes.outline_simt_scopes()(before), expected)


def test_initializing_subview_does_not_initialize_entire_output():
    before = program(
        """
view = pl.tensor.slice(dst, [1, 16], [0, 0])
value = pl.tile.load(src, [0, 0], [1, 16])
view = pl.tile.store(value, [0, 0], view)
with pl.simt_scope(dims=(32, 1, 1)):
    pl.simt.store(dst, (0, 0), pl.const(1.0, pl.FP32))
""",
        direction="Out",
    )
    with pytest.raises((Error, ValueError), match="initialized storage"):
        passes.outline_simt_scopes()(before)


def test_conditional_initialization_does_not_escape_branch():
    before = program(
        """
if n > 0:
    value = pl.tile.load(src, [0, 0], [1, 32])
    dst = pl.tile.store(value, [0, 0], dst)
with pl.simt_scope(dims=(32, 1, 1)):
    pl.simt.store(dst, (0, 0), pl.const(1.0, pl.FP32))
""",
        direction="Out",
    )
    with pytest.raises((Error, ValueError), match="initialized storage"):
        passes.outline_simt_scopes()(before)


def test_multi_output_and_repeated_stores_have_single_results():
    before = program("""
local = pl.tile.load(src, [0, 0], [1, 32])
old = local
with pl.simt_scope(dims=(32, 1, 1)):
    pl.simt.store(dst, (0, 0), pl.const(1.0, pl.FP32))
    if n > 0:
        pl.simt.store(local, (0, 0), pl.const(2.0, pl.FP32))
        pl.simt.store(dst, (0, 1), pl.const(3.0, pl.FP32))
return old, local, dst
""")
    after = passes.convert_to_ssa()(passes.outline_simt_scopes()(before))
    launch = launches(after)[0]
    assert launch.result_param_indices == [0, 2]
    body = function(after, "kernel").body
    assert isinstance(body, ir.SeqStmts)
    ret = body.stmts[-1]
    assert isinstance(ret, ir.ReturnStmt)
    assert ret.value[1] is launch.results[1]
    assert ret.value[2] is launch.results[0]
    assert ret.value[0] is not launch.results[1]


def test_raw_ir_private_value_escape_rejected():
    span = ir.Span.unknown()
    value = ir.Var("private", ir.ScalarType(DataType.INT32), span)
    one = ir.ConstInt(1, DataType.INT32, span)
    region = ir.SimtScopeStmt([one, one, one], ir.AssignStmt(value, one, span), span)
    leaked = ir.Var("leaked", value.type, span)
    body = ir.SeqStmts([region, ir.AssignStmt(leaked, value, span)], span)
    kernel = ir.Function("kernel", [], [], body, span, ir.FunctionType.AIV)
    with pytest.raises((Error, ValueError), match="thread-private scalar"):
        passes.outline_simt_scopes()(ir.Program([kernel], "Escape", span))


def test_readonly_tuple_alias_cannot_gain_permission():
    before = program("""
pair = (src, dst)
aliased = pair[0]
with pl.simt_scope(dims=(32, 1, 1)):
    pl.simt.store(aliased, (0, 0), pl.const(1.0, pl.FP32))
""")
    with pytest.raises((Error, ValueError), match="enclosing In parameter"):
        passes.outline_simt_scopes()(before)


def test_conditional_scalar_shadow_keeps_captured_initial_value():
    before = program("""
with pl.simt_scope(dims=(32, 1, 1)):
    tid = pl.simt.thread_idx(pl.simt.Axis.X)
    if tid > 0:
        n = n + pl.const(1, pl.INT32)
    pl.simt.store(dst, (0, tid), pl.cast(n, pl.FP32))
return dst
""")
    after = passes.convert_to_ssa()(passes.outline_simt_scopes()(before))
    selected = passes.IRPropertySet()
    selected.insert(passes.IRProperty.SSAForm)
    selected.insert(passes.IRProperty.UseAfterDef)
    assert not passes.PropertyVerifierRegistry.verify(selected, after)


def test_raw_ir_yield_cannot_cross_region_boundary():
    span = ir.Span.unknown()
    one = ir.ConstInt(1, DataType.INDEX, span)
    region = ir.SimtScopeStmt([one, one, one], ir.YieldStmt([], span), span)
    kernel = ir.Function("kernel", [], [], region, span, ir.FunctionType.AIV)
    with pytest.raises((Error, ValueError), match="yield cannot cross"):
        passes.outline_simt_scopes()(ir.Program([kernel], "YieldEscape", span))


@pytest.mark.parametrize(
    "kind,dtype", [(ir.RegKind.VecReg, DataType.FP32), (ir.RegKind.MaskReg, DataType.BOOL)]
)
def test_vmi_register_capture_rejected(kind, dtype):
    span = ir.Span.unknown()
    captured = ir.Var("register", ir.RegType(64, dtype, kind), span)
    one = ir.ConstInt(1, DataType.INDEX, span)
    region = ir.SimtScopeStmt([one, one, one], ir.EvalStmt(captured, span), span)
    kernel = ir.Function("kernel", [captured], [], region, span, ir.FunctionType.AIV)
    # Outlining must enforce this boundary even without generic pass checks.
    with passes.PassContext([], passes.VerificationLevel.NONE):
        with pytest.raises((Error, ValueError), match="SIMT parameters must be scalar, Tile or Tensor"):
            passes.outline_simt_scopes()(ir.Program([kernel], "VmiCapture", span))


@pytest.mark.parametrize("write_to", ["a", "tmp"])
def test_possible_alias_does_not_share_initialization(write_to):
    # A branch can make tmp refer to b, but storing a (or tmp) does not prove
    # that the distinct Out buffer b was initialized on every path.
    before = parse_program(f"""
@pl.program
class Kernel:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, src: pl.Tensor[[1, 32], pl.FP32],
               a: pl.InOut[pl.Tensor[[1, 32], pl.FP32]],
               b: pl.Out[pl.Tensor[[1, 32], pl.FP32]], n: pl.Scalar[pl.INT32]):
        tmp = a
        if n > 0:
            tmp = b
        value = pl.tile.load(src, [0, 0], [1, 32])
        {write_to} = pl.tile.store(value, [0, 0], {write_to})
        with pl.simt_scope(dims=(32, 1, 1)):
            pl.simt.store(b, (0, 0), pl.const(1.0, pl.FP32))
""")
    with pytest.raises((Error, ValueError), match="initialized storage"):
        passes.outline_simt_scopes()(before)


def test_definite_alias_store_initializes_original_output():
    before = program(
        """
alias = dst
value = pl.tile.load(src, [0, 0], [1, 32])
alias = pl.tile.store(value, [0, 0], alias)
with pl.simt_scope(dims=(32, 1, 1)):
    pl.simt.store(dst, (0, 0), pl.const(1.0, pl.FP32))
""",
        direction="Out",
    )
    assert len(launches(passes.outline_simt_scopes()(before))) == 1


@pytest.mark.parametrize("capture,accepted", [("local", True), ("old", False)])
def test_loaded_tile_reassignment_preserves_old_initialization(capture, accepted):
    before = program(f"""
local = pl.tile.create([1, 32], pl.FP32)
old = local
for i in pl.range(2):
    local = pl.tile.load(src, [0, 0], [1, 32])
    with pl.simt_scope(dims=(32, 1, 1)):
        pl.simt.store({capture}, (0, 0), pl.const(1.0, pl.FP32))
""")
    if accepted:
        assert len(launches(passes.outline_simt_scopes()(before))) == 1
    else:
        with pytest.raises((Error, ValueError), match="initialized storage"):
            passes.outline_simt_scopes()(before)


def test_conditional_tile_reallocation_invalidates_initialization():
    # A conditional allocation must not revive the previously loaded value's
    # initialization at the join, even though both definitions share a Var.
    before = program("""
local = pl.tile.load(src, [0, 0], [1, 32])
if n > 0:
    local = pl.tile.create([1, 32], pl.FP32)
with pl.simt_scope(dims=(32, 1, 1)):
    pl.simt.store(local, (0, 0), pl.const(1.0, pl.FP32))
""")
    with pytest.raises((Error, ValueError), match="initialized storage"):
        passes.outline_simt_scopes()(before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
