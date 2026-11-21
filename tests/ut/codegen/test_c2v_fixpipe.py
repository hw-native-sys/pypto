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

import pypto.language as pl
import pytest
from pypto import DataType, Error, backend, codegen, ir, passes
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.op import system_ops, tile_ops
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


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
    vector_state = state and quant == "deqf16_vec"
    output_dtype = consumer_dtype or (
        pl.FP32 if quant == "no_convert" else (pl.BF16 if quant == "f32_bf16" else pl.FP16)
    )

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIC)
        def producer(
            self,
            a: pl.Tensor[[32, 32], input_dtype],
            b: pl.Tensor[[16, 32], input_dtype],
            out: pl.Out[pl.Tensor[[32, 16], output_dtype]],
            control_words: pl.Tensor[[1, 32], pl.UINT64],
        ):
            peer = pl.import_peer_buffer(name="c2v", peer_func="consumer")
            pl.aic_initialize_pipe(
                peer, dir_mask=1, slot_size=slot_size, slot_num=2, acc_push_epilogue=epilogue
            )
            lhs = pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
            rhs = pl.load(b, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
            acc = pl.matmul(lhs, pl.tile.transpose_view(rhs))
            if vector_state:
                packed: pl.Tile[
                    [1, 32],
                    pl.UINT64,
                    pl.Mem.Mat,
                    pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box),
                ] = pl.load(control_words, [0, 0], [1, 32], target_memory=pl.Mem.Mat)
                controls = pl.tile.move(packed, target_memory=pl.Mem.Scaling)
                pl.system.set_quant_vector(controls, id=state_id)
            if scalar_state:
                pl.system.set_quant_scalar(pl.const(0.5, pl.FP32), id=state_id)
            pl.tpush_to_aiv(acc, split=split)

        @pl.function(type=pl.FunctionType.AIV)
        def consumer(
            self,
            a: pl.Tensor[[32, 32], input_dtype],
            b: pl.Tensor[[16, 32], input_dtype],
            out: pl.Out[pl.Tensor[[32, 16], output_dtype]],
            control_words: pl.Tensor[[1, 32], pl.UINT64],
        ):
            buf = pl.reserve_buffer(name="c2v", size=buffer_size)
            pl.aiv_initialize_pipe(
                buf, dir_mask=1, slot_size=slot_size, slot_num=2, acc_push_epilogue=peer_epilogue
            )
            tile = pl.tpop_from_aic(shape=[32, 16], dtype=output_dtype, split=split)
            pl.store(tile, [0, 0], out)
            pl.tfree_to_aic(tile)

    return Program


class _PipeBufferAliases(ir.IRMutator):
    """Retain ordinary scalar SSA aliases through the actual compiler pipeline."""

    def __init__(self, producer_depth: int, consumer_depth: int, offset: int = 0):
        super().__init__()
        self._offset = offset
        # Validate names once so renamed operators cannot silently skip aliases.
        self._depths = {
            ir.get_op("system.aic_initialize_pipe").name: producer_depth,
            ir.get_op("system.aiv_initialize_pipe").name: consumer_depth,
        }

    def visit_eval_stmt(self, op: ir.EvalStmt) -> ir.Stmt:
        call = op.expr
        depth = self._depths.get(call.op.name, 0) if isinstance(call, ir.Call) else 0
        if not depth:
            return super().visit_eval_stmt(op)
        assert isinstance(call, ir.Call)
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
        core_functions = [
            func
            for func in lowered.functions.values()
            if func.func_type in (pl.FunctionType.InCore, pl.FunctionType.AIC, pl.FunctionType.AIV)
        ]
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
    _, mlir = _generate(_program(quant, relu))
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
    # Both imported and reserved addresses may have shared, transitive aliases.
    # Compile all the way to PTOAS so SSA alias emission is checked as well.
    aliases = _PipeBufferAliases(depth if side != "consumer" else 0, depth if side != "producer" else 0)
    _, mlir = _generate(aliases.visit_program(_program()), buffer_ir)
    _assemble(mlir, tmp_path)


def test_buffer_arithmetic_is_not_an_identity_alias():
    # Offsetting a reserved address does not name the same FIFO allocation.
    program = _PipeBufferAliases(4, 4, offset=16).visit_program(_program())
    with pytest.raises(Error, match="must trace to reserve_buffer/import_peer_buffer"):
        _generate(program)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"peer_quant": "f32_f16"}, "contract mismatch"),
        ({"slot_size": 128}, "exceeds slot_size"),
    ],
)
def test_buffer_aliases_preserve_contract_errors(kwargs, message):
    program = _PipeBufferAliases(4, 4).visit_program(_program(**kwargs))
    with pytest.raises(Error, match=message):
        _generate(program)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"quant": "deqf16_scalar", "state": False}, "preceding set_quant"),
        ({"quant": "deqf16_vec", "state": False}, "preceding set_quant"),
        ({"quant": "deqf16_scalar", "state_id": 1}, "dominating initialize_pipe"),
        ({"peer_quant": "f32_f16"}, "contract mismatch"),
        ({"slot_size": 128}, "exceeds slot_size"),
        ({"split": 1}, "split=0"),
        ({"consumer_dtype": pl.FP16}, "converted dtype mismatch"),
        ({"layout": "nz2dn"}, "consumer layout"),
        ({"layout": "nz2nz", "quant": "f32_f16"}, "NZ2NZ currently supports no_convert"),
        (
            {"layout": "nz2nz", "consumer_dtype": pl.INT32, "input_dtype_override": pl.INT8},
            "requires an FP32 source",
        ),
    ],
)
def test_invalid_fixpipe_contract(kwargs, message):
    with pytest.raises(Error, match=message):
        _generate(_program(**kwargs))


@pytest.mark.parametrize(
    "epilogue",
    [{"layout": "nz2dn"}, _epilogue(layout="dn2nz"), _epilogue(quant="unknown"), _epilogue(relu="relu")],
)
def test_schema_rejects_unknown_or_incomplete_fields(epilogue):
    with pytest.raises(ValueError, match="acc_push_epilogue"):
        system_ops.aic_initialize_pipe(dir_mask=1, slot_size=2048, acc_push_epilogue=epilogue)


@pytest.mark.parametrize(
    ("kwargs", "message"), [({"dir_mask": 3}, "C2V only"), ({"nosplit": False}, "nosplit=true")]
)
def test_schema_rejects_unsupported_transport(kwargs, message):
    with pytest.raises(ValueError, match=message):
        system_ops.aic_initialize_pipe(
            slot_size=2048, acc_push_epilogue=_epilogue(), **({"dir_mask": 1} | kwargs)
        )


@pytest.mark.parametrize("initialize", [system_ops.aic_initialize_pipe, system_ops.aiv_initialize_pipe])
@pytest.mark.parametrize("legacy", [{"c2v_dtype": DataType.FP32}, {"pre_relu": True}])
def test_pipe_contract_forms_cannot_be_combined(initialize, legacy):
    with pytest.raises(ValueError, match="cannot be combined"):
        initialize(dir_mask=1, slot_size=2048, acc_push_epilogue=_epilogue(), **legacy)


def test_legacy_pipe_dtype_requires_nosplit():
    with pytest.raises(ValueError, match="requires nosplit=true"):
        system_ops.aic_initialize_pipe(dir_mask=1, slot_size=2048, c2v_dtype=DataType.FP32, nosplit=False)


@pytest.mark.ptoas_compile
def test_converted_slot_uses_result_bytes(tmp_path):
    # FP32 source occupies 2048 bytes; the FP16 transport occupies only 1024.
    _, mlir = _generate(_program("f32_f16", slot_size=1024))
    _assemble(mlir, tmp_path)


def test_epilogue_value_round_trip():
    program = _program()
    restored = ir.deserialize(ir.serialize(program))
    ir.assert_structural_equal(program, restored)
    text = ir.python_print(program)
    reparsed = pl.parse_program(text)
    ir.assert_structural_equal(program, reparsed)


@pytest.mark.parametrize(
    ("dtype", "space"),
    [
        (DataType.FP32, ir.MemorySpace.Scaling),
        (DataType.UINT64, ir.MemorySpace.Vec),
    ],
)
def test_vector_quant_rejects_float_scale_or_wrong_location(dtype, space):
    tile = ir.Var("controls", ir.TileType([1, 32], dtype, memory_space=space), ir.Span.unknown())
    with pytest.raises(ValueError, match="packed 64-bit controls in loc=scaling"):
        system_ops.set_quant_vector(tile)


def test_vector_quant_packed_controls():
    tile = ir.Var(
        "controls",
        ir.TileType([1, 32], DataType.UINT64, memory_space=ir.MemorySpace.Scaling),
        ir.Span.unknown(),
    )
    call = system_ops.set_quant_vector(tile, id=3)
    assert call.kwargs["id"] == 3


@pytest.mark.parametrize("valid_shape", [[1, 16], [0, 32]])
@pytest.mark.parametrize("setter", [False, True])
def test_packed_controls_require_full_valid_extent(valid_shape, setter):
    controls = ir.Var(
        "controls",
        ir.TileType(
            [1, 32],
            DataType.UINT64,
            memory_space=ir.MemorySpace.Scaling if setter else ir.MemorySpace.Mat,
            tile_view=ir.TileView(valid_shape=valid_shape),
        ),
        ir.Span.unknown(),
    )
    with pytest.raises(ValueError, match="valid_shape to cover the full"):
        if setter:
            system_ops.set_quant_vector(controls)
        else:
            tile_ops.move(controls, target_memory=ir.MemorySpace.Scaling)


@pytest.mark.parametrize("legacy_first", [False, True])
def test_legacy_init_cannot_hide_duplicate_fixpipe_init(legacy_first):
    class DuplicateInit(ir.IRMutator):
        def visit_eval_stmt(self, op: ir.EvalStmt) -> ir.Stmt:
            call = op.expr
            if isinstance(call, ir.Call) and "acc_push_epilogue" in call.kwargs:
                kwargs = dict(call.kwargs)
                del kwargs["acc_push_epilogue"]
                legacy = ir.Call(call.op, call.args, kwargs, call.type, call.span)
                statements: list[ir.Stmt] = [op, ir.EvalStmt(legacy, op.span)]
                return ir.SeqStmts(statements[::-1] if legacy_first else statements, op.span)
            return super().visit_eval_stmt(op)

    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.MixedKernelExpanded)
    with pytest.raises(Error, match="exactly one initialize_pipe"):
        passes.verify_properties(properties, DuplicateInit().visit_program(_program()), "test")


@pytest.mark.parametrize("buffer_ir", [False, True])
def test_scaling_allocations_remain_distinct_after_setter(buffer_ir):
    @pl.program
    class Controls:
        @pl.function(type=pl.FunctionType.AIC)
        def kernel(self, packed: pl.Tensor[[1, 32], pl.UINT64]):
            source0: pl.Tile[
                [1, 32],
                pl.UINT64,
                pl.Mem.Mat,
                pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box),
            ] = pl.load(packed, [0, 0], [1, 32], target_memory=pl.Mem.Mat)
            controls0 = pl.tile.move(source0, target_memory=pl.Mem.Scaling)
            pl.system.set_quant_vector(controls0)
            source1: pl.Tile[
                [1, 32],
                pl.UINT64,
                pl.Mem.Mat,
                pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box),
            ] = pl.load(packed, [0, 0], [1, 32], target_memory=pl.Mem.Mat)
            controls1 = pl.tile.move(source1, target_memory=pl.Mem.Scaling)
            pl.system.set_quant_vector(controls1)

    with passes.PassContext([], enable_buffer_ir=buffer_ir):
        lowered = passes.memory_reuse()(
            passes.materialize_semantic_aliases()(passes.init_mem_ref()(Controls))
        )
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


@pl.program
class ColumnMajorMove:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[32, 32], pl.BF16],
        b: pl.Tensor[[32, 16], pl.BF16],
        out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
    ):
        lhs = pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
        acc = pl.matmul(lhs, rhs)
        dn = pl.tile.move(
            acc, target_memory=pl.Mem.Vec, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box
        )
        pl.store(pl.tile.transpose_view(dn), [0, 0], out)


@pytest.mark.parametrize("buffer_ir", [False, True])
@pytest.mark.ptoas_compile
def test_automatic_column_major_producer_and_consumer_compile(buffer_ir, tmp_path):
    _, mlir = _generate(ColumnMajorMove, buffer_ir)
    cpp = _assemble(mlir, tmp_path)
    assert mlir.count("layout = nz2dn, quant = no_convert, relu = no_relu") == 2
    assert "LayoutMode_t::NZ2DN" in cpp


@pl.program
class ColumnMajorLoopTranspose:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[32, 32], pl.BF16],
        b: pl.Tensor[[32, 16], pl.BF16],
        out: pl.Out[pl.Tensor[[32, 16], pl.FP32]],
    ):
        for iteration in pl.range(2):
            lhs = pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
            rhs = pl.load(b, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
            acc = pl.matmul(lhs, rhs)
            dn = pl.tile.move(
                acc, target_memory=pl.Mem.Vec, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box
            )
            normal = pl.tile.transpose(pl.tile.transpose_view(dn), 0, 1)
            pl.store(normal, [0, 0], out)


@pytest.mark.ptoas_compile
def test_transpose_over_popped_view_in_loop_compiles(tmp_path):
    _, mlir = _generate(ColumnMajorLoopTranspose)
    _assemble(mlir, tmp_path)


@pl.program
class VaryingColumnMajorMoves:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[64, 32], pl.BF16],
        b: pl.Tensor[[32, 16], pl.BF16],
        out: pl.Out[pl.Tensor[[16, 96], pl.FP32]],
    ):
        lhs0 = pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
        rhs0 = pl.load(b, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
        acc0 = pl.matmul(lhs0, rhs0)
        dn0 = pl.tile.move(
            acc0, target_memory=pl.Mem.Vec, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box
        )
        pl.store(pl.tile.transpose_view(dn0), [0, 0], out)
        lhs1 = pl.load(a, [0, 0], [64, 32], target_memory=pl.Mem.Mat)
        rhs1 = pl.load(b, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
        acc1 = pl.matmul(lhs1, rhs1)
        dn1 = pl.tile.move(
            acc1, target_memory=pl.Mem.Vec, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box
        )
        pl.store(pl.tile.transpose_view(dn1), [0, 32], out)


@pytest.mark.parametrize("buffer_ir", [False, True])
@pytest.mark.ptoas_compile
def test_varying_column_major_payload_shapes_require_separate_ids(buffer_ir, tmp_path):
    with pytest.raises(Error, match="one physical payload shape/converted dtype"):
        _generate(VaryingColumnMajorMoves, buffer_ir)
    # Intentionally bypass PyPTO verification only to construct a negative PTOAS
    # fixture. The real compiler must independently reject the same boundary.
    _, mlir = _generate(VaryingColumnMajorMoves, buffer_ir, passes.VerificationLevel.NONE)
    _assemble(mlir, tmp_path, expected_error="to use the same tile type")


@pytest.mark.parametrize("quant", ["deqf16_scalar", "deqf16_vec"])
@pytest.mark.parametrize("buffer_ir", [False, True])
@pytest.mark.parametrize("round_trip", [False, True])
@pytest.mark.ptoas_compile
def test_quantized_pipeline_compiles(quant, buffer_ir, round_trip, tmp_path):
    epilogue = _epilogue(quant=quant)
    vector = quant == "deqf16_vec"

    @pl.program
    class QuantizedPipeline:
        @pl.function(type=pl.FunctionType.AIC)
        def producer(
            self,
            a: pl.Tensor[[128, 32], pl.INT8],
            b: pl.Tensor[[16, 32], pl.INT8],
            out: pl.Out[pl.Tensor[[128, 16], pl.FP16]],
            words: pl.Tensor[[1, 32], pl.UINT64],
        ):
            forward = pl.import_peer_buffer(name="forward", peer_func="consumer")
            if round_trip:
                reply = pl.reserve_buffer(name="reply", size=2048)
                pl.aic_initialize_pipe(0, reply, dir_mask=2, slot_size=1024, slot_num=2, id=1)
            pl.aic_initialize_pipe(
                forward, dir_mask=1, slot_size=1024, slot_num=2, acc_push_epilogue=epilogue
            )
            packed: pl.Tile[
                [1, 32],
                pl.UINT64,
                pl.Mem.Mat,
                pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box),
            ] = pl.load(words, [0, 0], [1, 32], target_memory=pl.Mem.Mat)
            controls = pl.tile.move(packed, target_memory=pl.Mem.Scaling)
            for i in pl.pipeline(4, stage=2):
                lhs = pl.load(a, [i * 32, 0], [32, 32], target_memory=pl.Mem.Mat)
                rhs = pl.load(b, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
                acc = pl.matmul(lhs, pl.tile.transpose_view(rhs))
                if vector:
                    pl.set_quant_vector(controls)
                else:
                    pl.set_quant_scalar(pl.cast(pl.cast(i + 1, pl.INT32), pl.FP32) * 0.5)
                pl.tpush_to_aiv(acc, split=0)
                if round_trip:
                    returned = pl.tpop_from_aiv(shape=[32, 16], dtype=pl.FP16, split=0, id=1)
                    pl.tfree_to_aiv(returned, id=1)

        @pl.function(type=pl.FunctionType.AIV)
        def consumer(
            self,
            a: pl.Tensor[[128, 32], pl.INT8],
            b: pl.Tensor[[16, 32], pl.INT8],
            out: pl.Out[pl.Tensor[[128, 16], pl.FP16]],
            words: pl.Tensor[[1, 32], pl.UINT64],
        ):
            forward = pl.reserve_buffer(name="forward", size=2048)
            if round_trip:
                reply = pl.import_peer_buffer(name="reply", peer_func="producer")
                pl.aiv_initialize_pipe(0, reply, dir_mask=2, slot_size=1024, slot_num=2, id=1)
            pl.aiv_initialize_pipe(
                forward, dir_mask=1, slot_size=1024, slot_num=2, acc_push_epilogue=epilogue
            )
            for i in pl.pipeline(4, stage=2):
                tile = pl.tpop_from_aic(shape=[32, 16], dtype=pl.FP16, split=0)
                pl.store(tile, [i * 32, 0], out)
                if round_trip:
                    pl.tpush_to_aic(tile, split=0, id=1)
                pl.tfree_to_aic(tile)

    _, mlir = _generate(QuantizedPipeline, buffer_ir)
    _assemble(mlir, tmp_path)


def test_fixpipe_rejects_unvalidated_a2a3_target():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend910)
    with pytest.raises(Error, match="currently require Ascend950"):
        _generate(_program())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
