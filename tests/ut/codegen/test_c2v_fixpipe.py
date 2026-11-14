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
from pypto.ir.op import system_ops
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


def _generate(program, buffer_ir=False):
    with passes.PassContext([], enable_buffer_ir=buffer_ir):
        lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
        core_functions = [
            func
            for func in lowered.functions.values()
            if func.func_type in (pl.FunctionType.InCore, pl.FunctionType.AIC, pl.FunctionType.AIV)
        ]
        modules = codegen.PTOCodegen().generate(ir.Program(core_functions, "Fixpipe", lowered.span))
    return lowered, modules if isinstance(modules, str) else "".join(modules.values())


def _assemble(mlir, tmp_path):
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
    assert result.returncode == 0, result.stderr
    assert output.stat().st_size > 0
    return output.read_text()


@pytest.mark.parametrize(
    "quant", ["no_convert", "f32_f16", "f32_bf16", "deqf16_scalar", "deqf16_vec", "qf322f16_pre_scalar"]
)
@pytest.mark.parametrize("relu", ["no_relu", "normal_relu"])
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


def test_transpose_over_popped_view_in_loop_compiles(tmp_path):
    _, mlir = _generate(ColumnMajorLoopTranspose)
    _assemble(mlir, tmp_path)


def test_fixpipe_rejects_unvalidated_a2a3_target():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend910B)
    with pytest.raises(Error, match="currently require Ascend950"):
        _generate(_program())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
