# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fixpipe schemas, cross-peer verification, and real PTOAS compilation."""

import subprocess
from textwrap import indent

import pypto.language as pl
import pytest
from pypto import DataType, Error, backend, codegen, ir, passes
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.op import system_ops, tile_ops
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.language.parser.diagnostics import ParserSyntaxError


@pytest.fixture(autouse=True)
def _backend():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def _epilogue(layout="nz2nd", quant="no_convert", relu="no_relu"):
    return {"layout": layout, "quant": quant, "relu": relu}


def _program(
    quant="no_convert",
    relu="no_relu",
    slot_size=2048,
    state=True,
    state_id=0,
    peer_quant=None,
    split=0,
    layout="nz2nd",
    consumer_dtype=None,
    input_dtype_override=None,
):
    buffer_size = slot_size * 2
    epilogue = _epilogue(layout=layout, quant=quant, relu=relu)
    peer_epilogue = _epilogue(layout=layout, quant=peer_quant or quant, relu=relu)
    input_dtype = input_dtype_override or (pl.INT8 if quant.startswith("deq") else pl.BF16)
    scalar_state = state and quant in ("deqf16_scalar", "qf322f16_pre_scalar")
    closure_scale = quant == "qf322f16_pre_scalar"
    setter = (
        "pl.set_quant_scalar(scalar_config.scale" if closure_scale else "pl.system.set_quant_scalar(scale=0.5"
    )
    vector_state = state and quant == "deqf16_vec"
    output_dtype = consumer_dtype or (
        pl.FP32 if quant == "no_convert" else (pl.BF16 if quant == "f32_bf16" else pl.FP16)
    )

    src = "BF16" if input_dtype == pl.BF16 else str(input_dtype).upper()
    dst = "BF16" if output_dtype == pl.BF16 else str(output_dtype).upper()
    signature = (
        f"a: pl.Tensor[[32, 32], pl.{src}], b: pl.Tensor[[16, 32], pl.{src}], "
        f"out: pl.Out[pl.Tensor[[32, 16], pl.{dst}]], control_words: pl.Tensor[[1, 32], pl.UINT64]"
    )
    return pl.parse_program(f"""
from types import SimpleNamespace
scalar_config = SimpleNamespace(scale=0.5)
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIC)
    def producer(self, {signature}):
        peer = pl.import_peer_buffer(name="c2v", peer_func="consumer")
        pl.aic_initialize_pipe(peer, dir_mask=1, slot_size={slot_size}, slot_num=2,
                               acc_push_epilogue={epilogue!r})
        lhs = pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
        acc = pl.matmul(lhs, pl.tile.transpose_view(rhs))
        if {vector_state}:
            packed: pl.Tile[[1, 32], pl.UINT64, pl.Mem.Mat,
                            pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)
                            ] = pl.load(control_words, [0, 0], [1, 32], target_memory=pl.Mem.Mat)
            controls = pl.tile.move(packed, target_memory=pl.Mem.Scaling)
            pl.system.set_quant_vector(controls, id={state_id})
        if {scalar_state}:
            {setter}, id={state_id})
        pl.tpush_to_aiv(acc, split={split})

    @pl.function(type=pl.FunctionType.AIV)
    def consumer(self, {signature}):
        buf = pl.reserve_buffer(name="c2v", size={buffer_size})
        pl.aiv_initialize_pipe(buf, dir_mask=1, slot_size={slot_size}, slot_num=2,
                               acc_push_epilogue={peer_epilogue!r})
        tile = pl.tpop_from_aic(shape=[32, 16], dtype=pl.{dst}, split={split})
        pl.store(tile, [0, 0], out)
        pl.tfree_to_aic(tile, split=1)
""")


class _PipeBufferAliases(ir.IRMutator):
    def __init__(self, producer_depth: int, consumer_depth: int, offset: int = 0):
        super().__init__()
        self._offset = offset
        self._depths = {
            ir.get_op("system.aic_initialize_pipe").name: producer_depth,
            ir.get_op("system.aiv_initialize_pipe").name: consumer_depth,
        }

    def visit_eval_stmt(self, op: ir.EvalStmt) -> ir.Stmt:
        call = op.expr
        if not isinstance(call, ir.Call) or not (depth := self._depths.get(call.op.name, 0)):
            return super().visit_eval_stmt(op)
        value = call.args[0]
        if self._offset:
            value = ir.Add(value, ir.ConstInt(self._offset, DataType.INT32, op.span), DataType.INT32, op.span)
        statements: list[ir.Stmt] = []
        for index in range(depth):
            alias = ir.Var(f"c2v_buffer_alias_{index}", value.type, op.span)
            statements.append(ir.AssignStmt(alias, value, op.span))
            value = alias
        aliased = ir.Call(call.op, [value, *call.args[1:]], dict(call.kwargs), call.type, call.span)
        statements.append(ir.EvalStmt(aliased, op.span))
        return ir.SeqStmts(statements, op.span)


def _generate(program, buffer_ir=False, verification_level=passes.VerificationLevel.BASIC):
    with passes.PassContext([], enable_buffer_ir=buffer_ir, verification_level=verification_level):
        lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
        core_functions = [func for func in lowered.functions.values() if ir.is_incore_type(func.func_type)]
        modules = codegen.PTOCodegen().generate(ir.Program(core_functions, "Fixpipe", lowered.span))
    return lowered, modules if isinstance(modules, str) else "".join(modules.values())


def _assemble(mlir, tmp_path, expected_error=None):
    compiler = find_ptoas_binary()
    if not compiler:
        pytest.skip("real PTOAS compiler is not installed")
    source = tmp_path / "kernel.pto"
    source.write_text(mlir)
    output = tmp_path / "kernel.cpp"
    result = subprocess.run(
        [compiler, "--pto-arch=a5", "--pto-level=level3", str(source), "-o", str(output)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if expected_error is not None:
        assert result.returncode != 0, "PTOAS unexpectedly accepted an unsupported contract"
        assert expected_error in result.stderr, result.stderr
        return result.stderr
    assert result.returncode == 0, result.stderr
    assert output.stat().st_size > 0
    return output.read_text()


@pytest.mark.parametrize(
    "quant", ["no_convert", "f32_f16", "f32_bf16", "deqf16_scalar", "deqf16_vec", "qf322f16_pre_scalar"]
)
@pytest.mark.parametrize("relu", ["no_relu", "normal_relu"])
@pytest.mark.ptoas_compile
def test_explicit_fixpipe_compiles(quant, relu, tmp_path):
    slot_size = 2048 if quant == "no_convert" else 1024
    _, mlir = _generate(_program(quant, relu, slot_size=slot_size))
    cpp = _assemble(mlir, tmp_path)
    assert mlir.count("#pto.acc_push_epilogue<") == 2
    assert "FixpipeParams<" in cpp
    if quant.endswith("scalar"):
        assert "SET_QUANT_SCALAR" in cpp
    if quant.endswith("vec"):
        assert "SET_QUANT_VECTOR" in cpp
        assert "TileType::Scaling, uint64_t" in cpp


@pytest.mark.parametrize("buffer_ir", [False, True])
@pytest.mark.parametrize("side", ["producer", "consumer", "both"])
@pytest.mark.parametrize("depth", [1, 4])
@pytest.mark.ptoas_compile
def test_pipe_buffer_aliases_compile(buffer_ir, side, depth, tmp_path):
    aliases = _PipeBufferAliases(depth if side != "consumer" else 0, depth if side != "producer" else 0)
    lowered, mlir = _generate(aliases.visit_program(_program()), buffer_ir)
    consumer = lowered.get_function("consumer")
    assert consumer is not None and consumer.attrs["dual_aiv_dispatch"] is True
    assert mlir.index("pto.aiv_initialize_pipe") < mlir.index("scf.if")
    _assemble(mlir, tmp_path)


def test_buffer_arithmetic_is_not_an_identity_alias():
    program = _PipeBufferAliases(4, 4, offset=16).visit_program(_program())
    with pytest.raises(Error, match="must trace to reserve_buffer/import_peer_buffer"):
        _generate(program)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"quant": "deqf16_scalar", "state": False}, "preceding set_quant"),
        ({"quant": "deqf16_vec", "state": False}, "preceding set_quant"),
        ({"quant": "deqf16_scalar", "state_id": 1}, "dominating initialize_pipe"),
        ({"split": 1}, "split=0"),
        ({"peer_quant": "f32_f16", "_aliases": True}, "contract mismatch"),
        ({"slot_size": 128, "_aliases": True}, "exceeds slot_size"),
        ({"consumer_dtype": pl.FP16}, "converted dtype mismatch"),
        ({"_a2a3": True}, "currently require Ascend950"),
        ({"layout": "nz2dn"}, "consumer layout"),
        ({"layout": "nz2nz", "quant": "f32_f16"}, "NZ2NZ currently supports no_convert"),
        ({"layout": "nz2nz", "consumer_dtype": pl.INT32, "input_dtype_override": pl.INT8}, "FP32 source"),
    ],
)
def test_invalid_fixpipe_contract(kwargs, message):
    options = kwargs.copy()
    aliases = options.pop("_aliases", False)
    if options.pop("_a2a3", False):
        backend.reset_for_testing()
        backend.set_backend_type(backend.BackendType.Ascend910)
    program = _program(**options)
    with pytest.raises(Error, match=message):
        _generate(_PipeBufferAliases(4, 4).visit_program(program) if aliases else program)


def test_fixpipe_pop_requires_a_matching_push():
    source = ir.python_print(_program())
    without_push = "\n".join(line for line in source.splitlines() if ".tpush_to_aiv(" not in line)
    with pytest.raises(Error, match="matching TPUSH and TPOP endpoints"):
        _generate(pl.parse_program(without_push))


@pytest.mark.parametrize(
    ("guarded_pop", "binding"),
    [(False, None), (True, None), (False, True), (False, False), (False, "member")],
)
@pytest.mark.ptoas_compile
def test_fixpipe_endpoint_counts(guarded_pop, binding, tmp_path):
    source = ir.python_print(_program()).replace("pl.UINT64]):", "pl.UINT64], enabled: pl.Scalar[pl.BOOL]):")
    source = source.replace(
        "        pl.tile.tpush_to_aiv(", "        if enabled:\n            pl.tile.tpush_to_aiv("
    )
    if guarded_pop:
        before, body = source.split("        tile: ", 1)
        consumer, producer = ("        tile: " + body).split("    @pl.function", 1)
        source = before + "        if enabled:\n            pass\n        else:\n" + indent(consumer, "    ")
        source += "    @pl.function" + producer
    literal_member = binding == "member"
    if guarded_pop or binding is not None:
        signature = next(line for line in source.splitlines() if line.startswith("    def producer("))
        group = signature.replace("producer(", "group(self, ")
        if binding is not None:
            group = group.replace(", enabled: pl.Scalar[pl.BOOL]", "")
            if not literal_member:
                group = group.replace("group(self, ", "group(self, enabled: pl.Scalar[pl.BOOL], ")
        source += "\n    @pl.function(type=pl.FunctionType.Group)\n" + group
        arg = "True" if literal_member else "enabled"
        source += f"\n        self.producer(a, b, out, control_words, {arg})\n"
        source += f"        self.consumer(a, b, out, control_words, {arg})\n"
    if binding is not None and not literal_member:
        caller = signature.replace("producer(", "main(self, ").replace(", enabled: pl.Scalar[pl.BOOL]", "")
        source += "\n    @pl.function(type=pl.FunctionType.Spmd)\n" + group.replace("group(", "wrapper(")
        source += "\n        self.group(enabled, a, b, out, control_words)\n"
        source += "\n    @pl.function(type=pl.FunctionType.Orchestration)\n" + caller
        source += f"\n        self.wrapper({binding}, a, b, out, control_words)\n"
    program = pl.parse_program(source)
    if binding is True or literal_member:
        properties = passes.IRPropertySet()
        properties.insert(passes.IRProperty.MixedKernelExpanded)
        passes.verify_properties(properties, passes.simplify()(program), "test")
        _assemble(_generate(program)[1], tmp_path)
    else:
        with pytest.raises(Error, match="endpoint counts differ across control-flow paths"):
            _generate(program)


@pytest.mark.parametrize("schedule", ["overflow", "reverse", "sequential", "nested"])
@pytest.mark.ptoas_compile
def test_fixpipe_retained_slots_and_fifo_order(schedule, tmp_path):
    slots = 1 if schedule in {"overflow", "sequential"} else 2
    source = ir.python_print(_program()).replace("slot_num=2", f"slot_num={slots}")
    push = next(line for line in source.splitlines() if ".tpush_to_aiv(" in line)
    pop = next(line for line in source.splitlines() if ".tpop_from_aic(" in line)
    free = next(line for line in source.splitlines() if ".tfree_to_aic(" in line)
    second, release = pop.replace("tile:", "second:", 1), free.replace("(tile,", "(second,")
    sequence = {"reverse": [second, release, free], "sequential": [free, second, release]}
    sequence["nested"] = [
        "        for j in pl.range(2):",
        indent(second, "    "),
        indent(release, "    "),
        free,
    ]
    source = source.replace(push, "\n".join([push] * (3 if schedule == "nested" else 2)))
    source = source.replace(free, "\n".join(sequence.get(schedule, [second, free, release])))
    if schedule == "sequential":
        _assemble(_generate(pl.parse_program(source))[1], tmp_path)
    else:
        with pytest.raises(
            Error, match="outstanding TPOP count" if schedule == "overflow" else "oldest outstanding"
        ):
            _generate(pl.parse_program(source))


@pytest.mark.parametrize(("operands", "message"), [("[]", "exactly one operand"), ("[1]", "FP32 scalar")])
def test_raw_scalar_setter_does_not_configure_the_pipe(operands, message):
    source = ir.python_print(_program(quant="deqf16_scalar")).replace(
        "pl.system.set_quant_scalar(0.5, id=0)",
        f'pl._dump.call("system.set_quant_scalar", {operands}, pl.Scalar[pl.INT32], kwargs={{"id": 0}})',
    )
    with pytest.raises(Error, match=message):
        _generate(pl.parse_program(source))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"acc_push_epilogue": {"layout": "nz2dn"}}, "acc_push_epilogue"),
        ({"acc_push_epilogue": _epilogue(layout="dn2nz")}, "acc_push_epilogue"),
        ({"acc_push_epilogue": _epilogue(quant="unknown")}, "acc_push_epilogue"),
        ({"acc_push_epilogue": _epilogue(relu="relu")}, "acc_push_epilogue"),
        ({"dir_mask": 3}, "C2V only"),
        ({"nosplit": False}, "nosplit=true"),
    ],
)
def test_init_rejects_invalid_schema_or_transport(options, message):
    kwargs = {"dir_mask": 1, "slot_size": 2048, "acc_push_epilogue": _epilogue()}
    with pytest.raises(ValueError, match=message):
        system_ops.aic_initialize_pipe(**(kwargs | options))


@pytest.mark.parametrize("initialize", [system_ops.aic_initialize_pipe, system_ops.aiv_initialize_pipe])
@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"acc_push_epilogue": _epilogue(), "c2v_dtype": DataType.FP32}, "cannot be combined"),
        ({"acc_push_epilogue": _epilogue(), "pre_relu": True}, "cannot be combined"),
        ({"c2v_dtype": DataType.FP32, "nosplit": False}, "requires nosplit=true"),
    ],
)
def test_pipe_init_option_contract(initialize, options, message):
    with pytest.raises(ValueError, match=message):
        initialize(dir_mask=1, slot_size=2048, **options)


def test_epilogue_value_round_trip():
    program = _program()
    ir.assert_structural_equal(program, ir.deserialize(ir.serialize(program)))
    ir.assert_structural_equal(program, pl.parse_program(ir.python_print(program)))

    call = ir.Call(
        ir.get_op("system.aic_initialize_pipe"),
        [],
        {"acc_push_epilogue": {}},
        ir.ScalarType(pl.INT32),
        ir.Span.unknown(),
    )
    with pytest.raises(ValueError, match="Invalid serialized AccPushEpilogue"):
        ir.deserialize(ir.serialize(call))


@pytest.mark.parametrize("use_relu", [False, True])
def test_epilogue_rejects_shadowed_runtime_parameter(use_relu):
    with pytest.raises(ParserSyntaxError, match="acc_push_epilogue must be a compile-time record"):
        pl.parse(f"""
use_relu = {use_relu}
@pl.function(type=pl.FunctionType.AIC)
def producer(use_relu: pl.Scalar[pl.BOOL]):
    pl.aic_initialize_pipe(dir_mask=1, slot_size=2048, acc_push_epilogue={{
        "layout": "nz2nd", "quant": "no_convert", "relu": "normal_relu" if use_relu else "no_relu",
    }})
""")


@pytest.mark.parametrize("recorded", [False, True])
def test_parse_epilogue_dict_metadata(recorded):
    metadata = {"acc_push_epilogue": _epilogue()}
    call = (
        f'pl._dump.call("system.aic_initialize_pipe", [], pl.Scalar[pl.INT32], '
        f"kwargs={metadata!r}, attrs={metadata!r})"
        if recorded
        else f"pl.aic_initialize_pipe(dir_mask=1, slot_size=2048, attrs={metadata!r})"
    )
    func = pl.parse(f"""
@pl.function(type=pl.FunctionType.AIC)
def producer():
    {call}
""")
    assert isinstance(func, ir.Function)
    stmt = func.body.stmts[0] if isinstance(func.body, ir.SeqStmts) else func.body
    assert isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call)
    assert stmt.expr.attrs["acc_push_epilogue"] == _epilogue()
    if recorded:
        assert stmt.expr.kwargs["acc_push_epilogue"] == _epilogue()


@pytest.mark.parametrize(
    ("dtype", "space", "valid"),
    [
        (DataType.FP32, ir.MemorySpace.Scaling, False),
        (DataType.UINT64, ir.MemorySpace.Vec, False),
        (DataType.UINT64, ir.MemorySpace.Scaling, True),
    ],
)
def test_vector_quant_controls_dtype_space_and_pipe_id(dtype, space, valid):
    tile = ir.Var("controls", ir.TileType([1, 32], dtype, memory_space=space), ir.Span.unknown())
    if valid:
        assert system_ops.set_quant_vector(tile, id=3).kwargs["id"] == 3
    else:
        with pytest.raises(ValueError, match="packed 64-bit controls in loc=scaling"):
            system_ops.set_quant_vector(tile)


@pytest.mark.parametrize(
    ("valid_shape", "setter", "layout", "message"),
    [
        *(
            (shape, setter, {}, "valid_shape to cover the full")
            for shape in ([1, 16], [0, 32])
            for setter in (False, True)
        ),
        ([1, 32], False, {"blayout": ir.TileLayout.col_major}, "row-major unboxed destination"),
        ([1, 32], False, {"slayout": ir.TileLayout.row_major}, "row-major unboxed destination"),
    ],
)
def test_packed_control_extent_and_layout(valid_shape, setter, layout, message):
    controls = ir.Var(
        "controls",
        ir.TileType(
            [1, 32],
            DataType.UINT64,
            memory_space=ir.MemorySpace.Scaling if setter else ir.MemorySpace.Mat,
            tile_view=ir.TileView(
                valid_shape=valid_shape, blayout=ir.TileLayout.row_major, slayout=ir.TileLayout.none_box
            ),
        ),
        ir.Span.unknown(),
    )
    with pytest.raises(ValueError, match=message):
        if setter:
            system_ops.set_quant_vector(controls)
        else:
            tile_ops.move(controls, target_memory=ir.MemorySpace.Scaling, **layout)


@pytest.mark.parametrize(
    ("legacy_first", "extra", "message"),
    [
        (False, None, "exactly one initialize_pipe"),
        (True, None, "exactly one initialize_pipe"),
        (False, {"c2v_dtype": pl.FP32}, "cannot be combined"),
        (False, {"pre_relu": True}, "cannot be combined"),
        (False, {"local_slot_num": 2}, "local_slot_num is only supported on A2/A3"),
    ],
)
def test_raw_fixpipe_initializer_contract(legacy_first, extra, message):
    class MutateInit(ir.IRMutator):
        def visit_eval_stmt(self, op: ir.EvalStmt) -> ir.Stmt:
            call = op.expr
            if not isinstance(call, ir.Call) or "acc_push_epilogue" not in call.kwargs:
                return super().visit_eval_stmt(op)
            kwargs = dict(call.kwargs) | (extra or {})
            if extra is None:
                del kwargs["acc_push_epilogue"]
            changed = ir.EvalStmt(ir.Call(call.op, call.args, kwargs, call.type, call.span), op.span)
            if extra is not None:
                return changed
            return ir.SeqStmts([changed, op] if legacy_first else [op, changed], op.span)

    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.MixedKernelExpanded)
    with pytest.raises(Error, match=message):
        passes.verify_properties(properties, MutateInit().visit_program(_program()), "test")


@pytest.mark.parametrize("buffer_ir", [False, True])
def test_scaling_allocations_remain_distinct_after_setter(buffer_ir):
    body = "".join(
        f"""
source{i}: pl.Tile[[1, 32], pl.UINT64, pl.Mem.Mat,
                  pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)
                  ] = pl.load(packed, [0, 0], [1, 32], target_memory=pl.Mem.Mat)
controls{i} = pl.tile.move(source{i}, target_memory=pl.Mem.Scaling)
pl.system.set_quant_vector(controls{i})
"""
        for i in range(2)
    )
    controls = pl.parse_program(f"""
@pl.program
class Controls:
    @pl.function(type=pl.FunctionType.AIC)
    def kernel(packed: pl.Tensor[[1, 32], pl.UINT64]):
{indent(body, "        ")}
""")
    with passes.PassContext([], enable_buffer_ir=buffer_ir):
        normalized = passes.init_mem_ref()(controls)
        lowered = passes.memory_reuse()(passes.materialize_semantic_aliases()(normalized))
    allocations: list[ir.MemRef] = []

    class Collector(ir.IRVisitor):
        def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
            if isinstance(op.var.type, ir.TileType) and op.var.type.memory_space == ir.MemorySpace.Scaling:
                assert op.var.type.memref is not None
                allocations.append(op.var.type.memref)
            super().visit_assign_stmt(op)

    Collector().visit_stmt(next(iter(lowered.functions.values())).body)
    assert len(allocations) == 2
    assert not ir.MemRef.same_allocation(allocations[0], allocations[1])


def _automatic_move(rows=(32,), *, loop=False, nz_int32=False):
    dtype, out_dtype = ("INT8", "INT32") if nz_int32 else ("BF16", "FP32")
    out_shape = [32, 16] if loop or nz_int32 else [16, sum(rows)]
    scatter = "row_major" if nz_int32 else "none_box"
    blocks = []
    for i, n in enumerate(rows):
        stored = f"tile{i}" if nz_int32 else f"pl.tile.transpose_view(tile{i})"
        if loop:
            stored = f"pl.tile.transpose({stored}, 0, 1)"
        blocks.append(f"""
lhs{i} = pl.load(a, [0, 0], [{n}, 32], target_memory=pl.Mem.Mat)
rhs = pl.load(b, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
acc{i} = pl.matmul(lhs{i}, rhs)
tile{i} = pl.tile.move(acc{i}, target_memory=pl.Mem.Vec,
                       blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.{scatter})
pl.store({stored}, [0, {sum(rows[:i])}], out)
""")
    body = "".join(blocks)
    if loop:
        body = "for iteration in pl.range(2):\n" + indent(body, "    ")
    return pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(a: pl.Tensor[[{max(rows)}, 32], pl.{dtype}], b: pl.Tensor[[32, 16], pl.{dtype}],
               out: pl.Out[pl.Tensor[{out_shape}, pl.{out_dtype}]]):
{indent(body, "        ")}
""")


@pytest.mark.parametrize("buffer_ir", [False, True])
@pytest.mark.ptoas_compile
def test_automatic_column_major_producer_and_consumer_compile(buffer_ir, tmp_path):
    _, mlir = _generate(_automatic_move(), buffer_ir)
    cpp = _assemble(mlir, tmp_path)
    assert mlir.count("layout = nz2dn, quant = no_convert, relu = no_relu") == 2
    assert "LayoutMode_t::NZ2DN" in cpp


@pytest.mark.ptoas_compile
def test_transpose_over_popped_view_in_loop_compiles(tmp_path):
    _, mlir = _generate(_automatic_move(loop=True))
    _assemble(mlir, tmp_path)


def test_automatic_nz2nz_requires_fp32():
    with pytest.raises(ValueError, match="Automatic C2V NZ2NZ requires an FP32 accumulator.*int32"):
        _generate(_automatic_move(nz_int32=True))


@pytest.mark.parametrize("buffer_ir", [False, True])
@pytest.mark.ptoas_compile
def test_varying_column_major_payload_shapes_require_separate_ids(buffer_ir, tmp_path):
    program = _automatic_move((32, 64))
    with pytest.raises(Error, match="one physical payload shape/converted dtype"):
        _generate(program, buffer_ir)
    _, mlir = _generate(program, buffer_ir, passes.VerificationLevel.NONE)
    _assemble(mlir, tmp_path, expected_error="to use the same tile type")


@pytest.mark.parametrize("quant", ["deqf16_scalar", "deqf16_vec"])
@pytest.mark.parametrize("buffer_ir", [False, True])
@pytest.mark.parametrize("round_trip", [False, True])
@pytest.mark.ptoas_compile
def test_quantized_pipeline_compiles(quant, buffer_ir, round_trip, tmp_path):
    source = ir.python_print(_program(quant=quant, slot_size=1024))
    consumer, producer = source.rsplit("    @pl.function", 1)
    c_init, c_body = consumer.split("        tile:", 1)
    p_init, p_body = producer.split("        lhs:", 1)
    c_body = "        tile:" + c_body
    p_body = "        lhs:" + p_body.replace(
        "set_quant_scalar(0.5", "set_quant_scalar(pl.cast(pl.cast(i + 1, pl.INT32), pl.FP32) * 0.5"
    )
    if round_trip:
        c_init += "        reply = pl.import_peer_buffer(name='reply', peer_func='producer')\n"
        c_init += "        pl.aiv_initialize_pipe(0, reply, dir_mask=2, slot_size=1024, slot_num=2, id=1)\n"
        p_init += "        reply = pl.reserve_buffer(name='reply', size=2048)\n"
        p_init += "        pl.aic_initialize_pipe(0, reply, dir_mask=2, slot_size=1024, slot_num=2, id=1)\n"
        c_body = c_body.replace(
            "        pl.system.tfree_to_aic",
            "        pl.tpush_to_aic(tile, split=0, id=1)\n        pl.system.tfree_to_aic",
        )
        p_body += "\n        reply_tile = pl.tpop_from_aiv(shape=[32, 16], dtype=pl.FP16, split=0, id=1)"
        p_body += "\n        pl.tfree_to_aiv(reply_tile, id=1)\n"
    loop = "        for i in pl.pipeline(4, stage=2):\n"
    source = (
        c_init + loop + indent(c_body, "    ") + "    @pl.function" + p_init + loop + indent(p_body, "    ")
    )
    _, mlir = _generate(pl.parse_program(source), buffer_ir)
    _assemble(mlir, tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
