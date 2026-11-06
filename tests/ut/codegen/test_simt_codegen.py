# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""SIMT pointer ABI, physical addressing and module reachability from explicit IR."""

import re
import textwrap

import pytest
from pypto import DataType, backend, codegen, ir, passes
from pypto.backend import pto_backend
from pypto.ir.op import simt
from pypto.language.parser import parse_program

SPAN = ir.Span.unknown()


@pytest.fixture(autouse=True)
def a5_backend():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def make_copy(*, stride: int | ir.Var = 32, dtype=DataType.FP32, dims=(32, 2, 1), repeats=1, ub_source=False):
    """Logical rows have 17 elements; physical rows deliberately have padding."""
    plain = (
        ir.TileType([2, 32], dtype, memory_space=ir.MemorySpace.Vec)
        if ub_source
        else ir.TensorType([2, 17], dtype)
    )
    a = ir.Var("a", plain, SPAN)
    b = ir.Var("b", ir.TensorType([2, 32], dtype) if ub_source else plain, SPAN)
    n = ir.Var("n", ir.ScalarType(DataType.INT32), SPAN)
    x, y = (ir.Var(name, ir.ScalarType(DataType.INT32), SPAN) for name in ("x", "y"))
    value = ir.Var("value", ir.ScalarType(dtype), SPAN)
    helper = ir.Function(
        "copy",
        [a, (b, ir.ParamDirection.InOut), n],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(x, simt.thread_idx(ir.SimtAxis.X), SPAN),
                ir.AssignStmt(y, simt.thread_idx(ir.SimtAxis.Y), SPAN),
                ir.IfStmt(
                    ir.Lt(x, n, DataType.BOOL, SPAN),
                    ir.SeqStmts(
                        [
                            ir.AssignStmt(value, simt.load(a, [y, x]), SPAN),
                            ir.EvalStmt(simt.store(b, [y, x], value), SPAN),
                        ],
                        SPAN,
                    ),
                    None,
                    [],
                    SPAN,
                ),
            ],
            SPAN,
        ),
        SPAN,
        ir.FunctionType.SIMT,
    )
    view = ir.TensorView(stride=[stride, 1], layout=ir.TensorLayout.ND)
    actual = (
        ir.TileType(
            [2, 32],
            dtype,
            memory_space=ir.MemorySpace.Vec,
            tile_view=ir.TileView(
                valid_shape=[2, 17], stride=[stride, 1], start_offset=ir.ConstInt(8, DataType.INDEX, SPAN)
            ),
        )
        if ub_source
        else ir.TensorType([2, 17], dtype, tensor_view=view)
    )
    src = ir.Var("src", actual, SPAN)
    dst = ir.Var("dst", ir.TensorType([2, 32], dtype, tensor_view=view) if ub_source else actual, SPAN)
    valid = ir.Var("valid", n.type, SPAN)
    parameters = [src, (dst, ir.ParamDirection.InOut), valid]
    if isinstance(stride, ir.Var):
        parameters.append(stride)
    statements = []
    for i in range(repeats):
        result = ir.Var(f"updated{i}", dst.type, SPAN)
        statements.append(
            ir.SimtLaunchStmt(
                ir.GlobalVar("copy"),
                [ir.ConstInt(d, DataType.INT32, SPAN) for d in dims],
                [src, dst, valid],
                [result],
                [1],
                SPAN,
            )
        )
        dst = result
    kernel = ir.Function(
        "kernel",
        parameters,
        [dst.type],
        ir.SeqStmts([*statements, ir.ReturnStmt([dst], SPAN)], SPAN),
        SPAN,
        ir.FunctionType.AIV,
    )
    return passes.specialize_simt_helpers()(ir.Program([helper, kernel], "Copy", SPAN))


def test_gm_bundle_preserves_native_abi_and_physical_stride():
    bundle = codegen.PTOCodegen().generate_bundle(make_copy(), emit_source_loc=False)
    assert len(bundle.helpers) == 1
    helper = bundle.helpers[0]
    assert helper.kind == "simt"
    assert helper.parameter_types == ["!pto.ptr<f32, gm>", "!pto.ptr<f32, gm>", "i32"]
    assert helper.parameter_effects == ["read", "readwrite", "none"]
    assert "pto.get_tid_y" in helper.source
    assert "arith.constant 32 : index" in helper.source
    assert "arith.constant 17 : index" not in helper.source
    assert "pto.ldg" in helper.source and "pto.stg" in helper.source
    assert "pto.tload" not in helper.source and ", ub>" not in helper.source
    assert "pto.simt_max_threads = 64 : i32" in helper.source
    assert 'pto.ptodsl.subkernel_helper = "simd"' in bundle.caller_source
    assert 'pto.tileop.effects = ["read", "readwrite", "none"]' in bundle.caller_source


def test_dynamic_stride_is_captured_by_value():
    stride = ir.Var("row_stride", ir.ScalarType(DataType.INDEX), SPAN)
    # Explicit IR permits metadata to name a later parameter. The text parser
    # resolves annotations in declaration order, so use binary roundtrip here.
    with passes.PassContext([]):
        program = make_copy(stride=stride)
    ir.assert_structural_equal(program, ir.deserialize(ir.serialize(program)), enable_auto_mapping=True)
    bundle = codegen.PTOCodegen().generate_bundle(program, emit_source_loc=False)
    helper = bundle.helpers[0]
    assert helper.parameter_types[-1] == "index"
    assert "arith.muli" in helper.source and ", %arg3 : index" in helper.source
    assert helper.parameter_effects[-1] == "none"


def test_ub_view_uses_physical_stride_and_origin_once():
    program = passes.init_mem_ref()(make_copy(ub_source=True))
    bundle = codegen.PTOCodegen().generate_bundle(program, emit_source_loc=False)
    helper = bundle.helpers[0]
    assert helper.parameter_types[:2] == ["!pto.ptr<f32, ub>", "!pto.ptr<f32, gm>"]
    assert "pto.tile_buf_addr" in bundle.caller_source
    assert "pto.load" in helper.source and "pto.stg" in helper.source
    assert "pto.ldg" not in helper.source
    assert "arith.constant 32 : index" in helper.source
    assert "arith.constant 17 : index" not in helper.source
    assert "arith.constant 8 : index" in helper.source
    # The adapter passes the allocation base; only the native body adds origin.
    assert "pto.addptr" not in bundle.caller_source


def test_repeated_launches_share_one_helper_and_preserve_results():
    program = make_copy(repeats=2)
    generator = codegen.PTOCodegen()
    bundle = generator.generate_bundle(program, emit_source_loc=False)
    assert len(bundle.helpers) == 1
    assert bundle.caller_source.count("func.call @" + bundle.helpers[0].adapter_symbol) == 2
    combined = generator.generate(program, emit_source_loc=False)
    assert combined.count("pto.simt_launch @") == 2
    assert combined.count("attributes {pto.simt_entry,") == 1
    assert "__pypto_simt_" not in combined


@pytest.mark.parametrize(
    "threads,registers",
    [(1, 128), (256, 128), (257, 64), (512, 64), (513, 32), (1024, 32), (1025, 16), (2048, 16)],
)
def test_specialized_resource_annotations(threads, registers):
    helper = codegen.PTOCodegen().generate_bundle(make_copy(dims=(threads, 1, 1))).helpers[0]
    assert f"pto.simt_max_threads = {threads} : i32" in helper.source
    assert f"pto.simt_max_regs = {registers} : i32" in helper.source


def test_python_backend_keeps_only_reachable_helpers(tmp_path):
    files = pto_backend.generate(make_copy(repeats=2), str(tmp_path), skip_ptoas=True)
    kernels = [source for name, source in files.items() if name.endswith(".pto")]
    assert len(kernels) == 1
    assert kernels[0].count("pto.simt_launch @") == 2
    assert "func.func @copy_simt_0" in kernels[0]
    assert "func.func @copy(" not in kernels[0]
    assert len([name for name in files if name.endswith(".pto")]) == 1


def test_unused_simt_definition_does_not_select_vpto():
    program = make_copy()
    empty = ir.Function("empty", [], [], ir.SeqStmts([], SPAN), SPAN, ir.FunctionType.AIV)
    helper = program.get_function("copy")
    assert helper is not None
    source = codegen.PTOCodegen().generate(ir.Program([helper, empty], "Unused", SPAN))
    assert "pto.backend" not in source
    assert "pto.simt_entry" not in source


def test_unsupported_unsigned_arithmetic_fails_before_native_compilation():
    with pytest.raises(ValueError, match="SIMT.*currently supports"):
        codegen.PTOCodegen().generate_bundle(make_copy(dtype=DataType.UINT32))


def scalar_program(body, dtype="INT32"):
    program = parse_program(f"""
@pl.program
class Scalars:
    @pl.function(type=pl.FunctionType.SIMT)
    def helper(self, dst: pl.InOut[pl.Tensor[[1, 128], pl.{dtype}]], n: pl.Scalar[pl.INT32]):
{textwrap.indent(textwrap.dedent(body).strip(), "        ")}

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, dst: pl.InOut[pl.Tensor[[1, 128], pl.{dtype}]], n: pl.Scalar[pl.INT32]):
        updated = pl.simt.launch(self.helper, dst, n, block_dims=(128, 1, 1), result_param_indices=(0,))
        return updated
""")
    return passes.specialize_simt_helpers()(passes.convert_to_ssa()(program))


@pytest.mark.parametrize(
    "query,operation",
    [
        (f"{query}(axis=pl.SimtAxis.{axis.upper()})", f"{operation}_{axis}")
        for query, operation in [
            ("thread_idx", "get_tid"),
            ("block_idx", "get_block_idx"),
            ("block_dim", "get_block_dim"),
            ("grid_dim", "get_grid_dim"),
        ]
        for axis in ("x", "y", "z")
    ]
    + [("lane_id()", "get_laneid")],
)
def test_queries(query, operation):
    program = scalar_program(f"value = pl.simt.{query}\npl.simt.store(dst, [0, 0], value)")
    source = codegen.PTOCodegen().generate_bundle(program).helpers[0].source
    assert f"pto.{operation} : i32" in source


def test_signed_division_modulo_and_structured_loops():
    program = scalar_program("""
        x = pl.simt.thread_idx(axis=pl.SimtAxis.X)
        total = pl.const(0, pl.INT32)
        for i in pl.range(3):
            total = total + n // pl.const(3, pl.INT32)
        j = pl.const(0, pl.INT32)
        while j < 2:
            total = total + n % pl.const(3, pl.INT32)
            j = j + pl.const(1, pl.INT32)
        pl.simt.store(dst, [0, x], total)
    """)
    source = codegen.PTOCodegen().generate_bundle(program).helpers[0].source
    assert "scf.for" in source and "scf.while" in source
    assert "iter_args" in source and "scf.yield" in source
    assert "arith.floordivsi" in source and "arith.remsi" in source
    assert "arith.select" in source  # Negative remainders need floor correction.


def test_nan_inequality_and_gm_cache_policies():
    program = scalar_program(
        """
        x = pl.simt.thread_idx(axis=pl.SimtAxis.X)
        value = pl.simt.load(dst, [0, x], l1cache="uncache", l2cache="nmpref")
        if value != value:
            pl.simt.store(dst, [0, x], pl.const(0.0, pl.FP32), l2cache="napi")
    """,
        dtype="FP32",
    )
    source = codegen.PTOCodegen().generate_bundle(program).helpers[0].source
    assert "arith.cmpf une" in source
    assert "l1cache(uncache) l2cache(nmpref)" in source
    assert "l2cache(napi)" in source


def test_early_return_has_an_actionable_error():
    program = scalar_program("""
        if n == 0:
            return
        pl.simt.store(dst, [0, 0], n)
    """)
    with pytest.raises(ValueError, match="early return; predicate the body"):
        codegen.PTOCodegen().generate_bundle(program)


@pytest.mark.parametrize("dtype,predicate", [("INT32", "arith.cmpi ne"), ("FP32", "arith.cmpf une")])
def test_numeric_bool_conversion_compares_with_zero(dtype, predicate):
    program = scalar_program(
        """
        value = pl.simt.load(dst, [0, 0])
        truth = pl.cast(value, pl.BOOL)
        if truth:
            pl.simt.store(dst, [0, 1], value)
    """,
        dtype=dtype,
    )
    source = codegen.PTOCodegen().generate_bundle(program).helpers[0].source
    assert predicate in source
    assert "arith.trunci" not in source and "arith.fptosi" not in source


@pytest.mark.parametrize("dtype,exponent", [("FP32", "2.0"), ("INT32", "2")])
def test_runtime_scalar_power_is_rejected(dtype, exponent):
    # A nonconstant base survives Simplify. The generic visitor used to leave
    # the exponent as the result, silently storing 2 instead of the square.
    program = scalar_program(
        f"""
        value = pl.simt.load(dst, [0, 0])
        squared = value ** pl.const({exponent}, pl.{dtype})
        pl.simt.store(dst, [0, 0], squared)
        """,
        dtype=dtype,
    )
    with pytest.raises(ValueError, match="SIMT scalar exponentiation.*explicit multiplication"):
        codegen.PTOCodegen().generate_bundle(program)


@pytest.mark.parametrize(
    "shape,layout,expected",
    [
        ([3, 4], ir.TensorLayout.DN, [1, 3]),
        ([2, 3, 4], ir.TensorLayout.DN, [12, 1, 3]),
        ([2, 3, 4, 5], ir.TensorLayout.DN, [60, 20, 1, 4]),
        ([2, 3, 4], ir.TensorLayout.ND, [12, 4, 1]),
    ],
)
def test_tensor_fallback_strides_keep_batch_axes_outermost(shape, layout, expected):
    # Deliberately bypass MaterializeTensorStrides: generate_bundle accepts
    # specialized IR directly and must agree with the caller's tensor view.
    tensor = ir.TensorType(shape, DataType.FP32, tensor_view=ir.TensorView(layout=layout))
    src = ir.Var("src", tensor, SPAN)
    indices = [ir.Var(f"i{i}", ir.ScalarType(DataType.INDEX), SPAN) for i in range(len(shape))]
    helper = ir.Function(
        "read", [src, *indices], [], ir.EvalStmt(simt.load(src, indices), SPAN), SPAN, ir.FunctionType.SIMT
    )
    args = [ir.Var("arg", tensor, SPAN), *[ir.Var(v.name_hint, v.type, SPAN) for v in indices]]
    launch = ir.SimtLaunchStmt(
        ir.GlobalVar("read"), [ir.ConstInt(1, DataType.INT32, SPAN)] * 3, [*args], [], [], SPAN
    )
    caller = ir.Function("kernel", args, [], launch, SPAN, ir.FunctionType.AIV)
    program = passes.specialize_simt_helpers()(ir.Program([helper, caller], "Strides", SPAN))
    source = codegen.PTOCodegen().generate_bundle(program).helpers[0].source
    constants = {
        int(value): name for name, value in re.findall(r"(%\w+) = arith.constant (\d+) : index", source)
    }
    for axis, stride in enumerate(expected):
        assert f"arith.muli %arg{axis + 1}, {constants[stride]} : index" in source


@pytest.mark.parametrize("offset", [0, 8])
@pytest.mark.parametrize("alias_depth", [0, 1, 2])
@pytest.mark.parametrize("local_tile,emit_tile_addr", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("materialized", [False, True])
def test_borrowed_ub_slice_requires_materialization(
    offset, alias_depth, local_tile, emit_tile_addr, materialized
):
    # A [4, 32] window of [4, 64] retains row pitch 64. Its specialized
    # TileType has no stride, so deriving pitch 32 from shape would misread it.
    # Local aliases get separate alloc_tile handles with emit_tile_addr=True;
    # unlike parameter aliases, they must be traced through IR provenance.
    aliases = ["window", *[f"borrowed{i}" for i in range(alias_depth)]]
    alias_stmt = "\n        ".join(f"{right} = {left}" for left, right in zip(aliases, aliases[1:]))
    source_param = (
        "input: pl.Tensor[[4, 64], pl.FP32]" if local_tile else "src: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec]"
    )
    source_stmt = "src = pl.tile.load(input, [0, 0], [4, 64], target_memory=pl.Mem.Vec)" if local_tile else ""
    window = (
        f"pl.tile.extract(src, 0, {offset}, [4, 32], target_memory=pl.Mem.Vec)"
        if materialized
        else f"pl.tile.slice(src, [4, 32], [0, {offset}])"
    )
    program = parse_program(f"""
@pl.program
class Sliced:
    @pl.function(type=pl.FunctionType.SIMT)
    def read(self, src: pl.Tile[[4, 32], pl.FP32, pl.Mem.Vec],
             dst: pl.InOut[pl.Tensor[[1, 1], pl.FP32]]):
        value = pl.simt.load(src, [1, 0])
        pl.simt.store(dst, [0, 0], value)

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, {source_param},
               dst: pl.InOut[pl.Tensor[[1, 1], pl.FP32]]):
        {source_stmt}
        window = {window}
        {alias_stmt}
        result = pl.simt.launch(self.read, {aliases[-1]}, dst,
                               block_dims=(1, 1, 1), result_param_indices=(1,))
        return result
""")
    program = passes.init_mem_ref()(passes.specialize_simt_helpers()(passes.convert_to_ssa()(program)))
    if local_tile and emit_tile_addr:
        program = passes.allocate_memory_addr()(passes.flatten_tile_nd_to_2d()(program))
    if materialized:
        bundle = codegen.PTOCodegen().generate_bundle(program, emit_tile_addr=emit_tile_addr)
        assert "pto.textract" in bundle.caller_source
        assert "pto.load" in bundle.helpers[0].source
    else:
        with pytest.raises(ValueError, match="SIMT cannot borrow a Tile view.*tile.extract"):
            codegen.PTOCodegen().generate_bundle(program, emit_tile_addr=emit_tile_addr)


@pytest.mark.parametrize("view_op", ["reshape", "reinterpret_view"])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("emit_tile_addr", [False, True])
@pytest.mark.parametrize("offset", [0, 8])
def test_borrowed_reshaped_ub_slice_is_rejected(view_op, alias, emit_tile_addr, offset):
    # Fixed-address reshape/reinterpret can reuse an alloc_tile without emitting
    # treshape. The view still has the sliced parent's pitch: logical [0, 32]
    # names parent [1, offset], not parent [0, offset + 32].
    dtype = "FP32" if view_op == "reshape" else "INT32"
    view = (
        "pl.tile.reshape(window, [2, 64])"
        if view_op == "reshape"
        else "pl.tile.reinterpret_view(window, pl.INT32, shape=[2, 64])"
    )
    alias_stmt = "borrowed = view" if alias else ""
    argument = "borrowed" if alias else "view"
    program = parse_program(f"""
@pl.program
class Reshaped:
    @pl.function(type=pl.FunctionType.SIMT)
    def read(self, src: pl.Tile[[2, 64], pl.{dtype}, pl.Mem.Vec],
             dst: pl.InOut[pl.Tensor[[1, 1], pl.{dtype}]]):
        value = pl.simt.load(src, [0, 32])
        pl.simt.store(dst, [0, 0], value)

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, input: pl.Tensor[[4, 64], pl.FP32],
               dst: pl.InOut[pl.Tensor[[1, 1], pl.{dtype}]]):
        src = pl.tile.load(input, [0, 0], [4, 64], target_memory=pl.Mem.Vec)
        window = pl.tile.slice(src, [4, 32], [0, {offset}])
        view = {view}
        {alias_stmt}
        result = pl.simt.launch(self.read, {argument}, dst,
                               block_dims=(1, 1, 1), result_param_indices=(1,))
        return result
""")
    program = passes.init_mem_ref()(passes.specialize_simt_helpers()(passes.convert_to_ssa()(program)))
    if emit_tile_addr:
        program = passes.allocate_memory_addr()(passes.flatten_tile_nd_to_2d()(program))
    with pytest.raises(ValueError, match="SIMT cannot borrow a Tile view.*tile.extract"):
        codegen.PTOCodegen().generate_bundle(program, emit_tile_addr=emit_tile_addr)


def prepare_control_flow_storage(program, reuse):
    program = passes.convert_to_ssa()(program)
    program = passes.specialize_simt_helpers()(program)
    program = passes.init_mem_ref()(program)
    program = passes.flatten_tile_nd_to_2d()(program)
    if reuse:
        program = passes.memory_reuse()(program)
    return passes.allocate_memory_addr()(program)


@pytest.mark.parametrize("view_branch", ["none", "then", "else", "both"])
@pytest.mark.parametrize("forward", ["direct", "nested", "loop"])
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("offset", [0, 8])
def test_conditional_tile_views_require_materialization(view_branch, forward, reuse, offset):
    # The addressed phi may share the then branch's strided storage while
    # having a fresh, unmarked SSA handle. Either branch can introduce a view.
    def window(source, branch):
        if view_branch in (branch, "both"):
            return f"pl.tile.slice({source}, [4, 32], [0, {offset}])"
        return f"pl.tile.extract({source}, 0, {offset}, [4, 32], target_memory=pl.Mem.Vec)"

    forwarding = {
        "direct": "forwarded = selected",
        "nested": """
        if n > 1:
            forwarded = pl.yield_(selected)
        else:
            forwarded = pl.yield_(selected)
        """,
        "loop": """
        for i, (carry,) in pl.range(n, init_values=(selected,)):
            forwarded = pl.yield_(carry)
        """,
    }[forward]
    forwarding = textwrap.indent(textwrap.dedent(forwarding).strip(), "        ")
    program = parse_program(f"""
@pl.program
class Conditional:
    @pl.function(type=pl.FunctionType.SIMT)
    def read(self, src: pl.Tile[[4, 32], pl.FP32, pl.Mem.Vec],
             dst: pl.InOut[pl.Tensor[[1, 1], pl.FP32]]):
        value = pl.simt.load(src, [1, 0])
        pl.simt.store(dst, [0, 0], value)

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, a: pl.Tensor[[4, 64], pl.FP32], b: pl.Tensor[[4, 64], pl.FP32],
               dst: pl.InOut[pl.Tensor[[1, 1], pl.FP32]], n: pl.Scalar[pl.INT32]):
        left = pl.tile.load(a, [0, 0], [4, 64], target_memory=pl.Mem.Vec)
        right = pl.tile.load(b, [0, 0], [4, 64], target_memory=pl.Mem.Vec)
        if n > 0:
            then_window = {window("left", "then")}
            selected = pl.yield_(then_window)
        else:
            else_window = {window("right", "else")}
            selected = pl.yield_(else_window)
{forwarding}
        result = pl.simt.launch(self.read, forwarded, dst,
                               block_dims=(1, 1, 1), result_param_indices=(1,))
        return result
""")
    if reuse and offset == 0 and view_branch == "then":
        # YieldFixup already rejects this mixed view/materialized destination
        # before codegen. Keep that earlier fail-closed boundary covered too.
        with pytest.raises(ValueError, match="cannot reconcile ambiguous views of one destination"):
            prepare_control_flow_storage(program, reuse)
        return
    program = prepare_control_flow_storage(program, reuse)
    if view_branch == "none":
        bundle = codegen.PTOCodegen().generate_bundle(program)
        assert "scf.if" in bundle.caller_source
        assert "pto.textract" in bundle.caller_source
        assert "pto.load" in bundle.helpers[0].source
    else:
        with pytest.raises(ValueError, match="SIMT cannot borrow a Tile view.*tile.extract"):
            codegen.PTOCodegen().generate_bundle(program)


@pytest.mark.parametrize("loop", ["for", "while"])
@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("reuse", [False, True])
def test_loop_backedge_tile_view_requires_materialization(loop, materialized, reuse):
    # The init value is dense. A later iteration can replace it with a view,
    # so checking only the iter_arg's init chain cannot prove the result safe.
    window = (
        "pl.tile.extract(src, 0, 8, [4, 32], target_memory=pl.Mem.Vec)"
        if materialized
        else "pl.tile.slice(src, [4, 32], [0, 8])"
    )
    loop_body = (
        f"for i in pl.range(n):\n    carried = {window}"
        if loop == "for"
        else (
            f"i = pl.const(0, pl.INT32)\nwhile i < n:\n    carried = {window}\n"
            "    i = i + pl.const(1, pl.INT32)"
        )
    )
    program = parse_program(f"""
@pl.program
class Backedge:
    @pl.function(type=pl.FunctionType.SIMT)
    def read(self, src: pl.Tile[[4, 32], pl.FP32, pl.Mem.Vec],
             dst: pl.InOut[pl.Tensor[[1, 1], pl.FP32]]):
        value = pl.simt.load(src, [1, 0])
        pl.simt.store(dst, [0, 0], value)

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, a: pl.Tensor[[4, 64], pl.FP32],
               dst: pl.InOut[pl.Tensor[[1, 1], pl.FP32]], n: pl.Scalar[pl.INT32]):
        src = pl.tile.load(a, [0, 0], [4, 64], target_memory=pl.Mem.Vec)
        carried = pl.tile.extract(src, 0, 0, [4, 32], target_memory=pl.Mem.Vec)
{textwrap.indent(loop_body, "        ")}
        result = pl.simt.launch(self.read, carried, dst,
                               block_dims=(1, 1, 1), result_param_indices=(1,))
        return result
""")
    program = prepare_control_flow_storage(program, reuse)
    if materialized:
        bundle = codegen.PTOCodegen().generate_bundle(program)
        assert f"scf.{loop}" in bundle.caller_source
        assert "pto.load" in bundle.helpers[0].source
    else:
        with pytest.raises(ValueError, match="SIMT cannot borrow a Tile view.*tile.extract"):
            codegen.PTOCodegen().generate_bundle(program)


@pytest.mark.parametrize("dtype", ["FP32", "INT32"])
@pytest.mark.parametrize("emit_tile_addr", [False, True])
@pytest.mark.parametrize("target_view", [False, True])
@pytest.mark.parametrize("source_view", [False, True])
@pytest.mark.parametrize("conditional", [False, True])
def test_assemble_inherits_destination_provenance(
    dtype, emit_tile_addr, target_view, source_view, conditional
):
    target = (
        "pl.tile.slice(parent, [4, 32], [0, 8])"
        if target_view
        else "pl.tile.extract(parent, 0, 8, [4, 32], target_memory=pl.Mem.Vec)"
    )
    source = (
        "pl.tile.slice(patch_parent, [2, 32], [0, 8])"
        if source_view
        else "pl.tile.extract(patch_parent, 0, 8, [2, 32], target_memory=pl.Mem.Vec)"
    )
    forwarding = (
        "if n > 0:\n    selected = pl.yield_(updated)\nelse:\n    selected = pl.yield_(updated)"
        if conditional
        else "selected = updated"
    )
    program = parse_program(f"""
@pl.program
class Assembled:
    @pl.function(type=pl.FunctionType.SIMT)
    def read(self, src: pl.Tile[[4, 32], pl.{dtype}, pl.Mem.Vec],
             dst: pl.InOut[pl.Tensor[[1, 1], pl.{dtype}]]):
        value = pl.simt.load(src, [1, 0])
        pl.simt.store(dst, [0, 0], value)

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, a: pl.Tensor[[4, 64], pl.{dtype}], b: pl.Tensor[[4, 64], pl.{dtype}],
               dst: pl.InOut[pl.Tensor[[1, 1], pl.{dtype}]], n: pl.Scalar[pl.INT32]):
        parent = pl.tile.load(a, [0, 0], [4, 64], target_memory=pl.Mem.Vec)
        patch_parent = pl.tile.load(b, [0, 0], [4, 64], target_memory=pl.Mem.Vec)
        target = {target}
        source = {source}
        updated = pl.tile.assemble(target, source, [1, 0])
{textwrap.indent(forwarding, "        ")}
        result = pl.simt.launch(self.read, selected, dst,
                               block_dims=(1, 1, 1), result_param_indices=(1,))
        return result
""")
    program = prepare_control_flow_storage(program, reuse=True)
    if target_view:
        with pytest.raises(ValueError, match="SIMT cannot borrow a Tile view.*tile.extract"):
            codegen.PTOCodegen().generate_bundle(program, emit_tile_addr=emit_tile_addr)
    else:
        bundle = codegen.PTOCodegen().generate_bundle(program, emit_tile_addr=emit_tile_addr)
        if dtype == "FP32":
            assert "pto.tinsert" in bundle.caller_source
        else:
            assert "pto.subview" in bundle.caller_source
            assert "pto.tmov" in bundle.caller_source
        assert "pto.load" in bundle.helpers[0].source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
