# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Automatic C2V narrowing must agree on transport type, storage and lifetime."""

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir, passes
from pypto.backend import pto_backend
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def statements(body):
    yield body
    if isinstance(body, ir.SeqStmts):
        for item in body.stmts:
            yield from statements(item)
    elif isinstance(body, ir.ForStmt):
        yield from statements(body.body)


def calls(function):
    for stmt in statements(function.body):
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
            yield stmt.value
        elif isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call):
            yield stmt.expr


def tile_dtype(expr):
    tile_type = expr.type
    assert isinstance(tile_type, ir.TileType)
    return tile_type.dtype


def make_program(mode="rint", relu=True, widening=True, extra_reader=False, bidirectional=False, cols=128):
    # pl.program validates both arms of an IR conditional. Construct each
    # concrete dtype chain before parsing rather than merging BF16/FP32 arms.
    left = (
        "raw = pl.tile.full([64, 128], dtype=pl.BF16, value=1.0)\n"
        "left = pl.tile.move(raw, target_memory=pl.Mem.Left)"
        if bidirectional
        else "a_mat = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)\n"
        "left = pl.move(a_mat, target_memory=pl.Mem.Left)"
    )
    body = left.splitlines() + [
        f"b_mat = pl.load(b, [0, 0], [128, {cols}], target_memory=pl.Mem.Mat)",
        "right = pl.move(b_mat, target_memory=pl.Mem.Right)",
        "dot = pl.matmul(left, right)",
        f"rounded = pl.cast(dot, pl.BF16, mode={mode!r})",
    ]
    value = "rounded"
    if widening:
        body.append("widened = pl.cast(rounded, pl.FP32)")
        value = "widened"
    if relu:
        body.append(f"activated = pl.maximum({value}, 0.0)")
        value = "activated"
    if extra_reader:
        body.append(f"extra = pl.add({value}, dot)")
        value = "extra"
    if widening or extra_reader:
        body.append(f'final = pl.cast({value}, pl.BF16, mode="rint")')
        value = "final"
    body.append(f"pl.store({value}, [0, 0], out)")
    source = f"""@pl.program
class Program:
    @pl.function(type=pl.FunctionType.InCore)
    def main(self, a: pl.Tensor[[64, 128], pl.BF16],
             b: pl.Tensor[[128, {cols}], pl.BF16],
             out: pl.Out[pl.Tensor[[64, {cols}], pl.BF16]]):
"""
    parsed = pl.parse(source + "\n".join("        " + statement for statement in body))
    assert isinstance(parsed, ir.Program)
    return parsed


def expand(program):
    return passes.expand_mixed_kernel()(passes.infer_tile_memory_space()(passes.convert_to_ssa()(program)))


def by_name(function, name):
    expected = ir.get_op(name).name
    return [call for call in calls(function) if call.op.name == expected]


@pytest.mark.parametrize("relu", [False, True])
@pytest.mark.parametrize("widening", [False, True])
def test_bf16_transport_and_optional_relu(relu, widening):
    backend.set_backend_type(backend.BackendType.Ascend950)
    after = expand(make_program(relu=relu, widening=widening))
    cube = after.get_function("main_aic")
    vector = after.get_function("main_aiv")
    push = by_name(cube, "tile.tpush_to_aiv")[0]
    pop = by_name(vector, "tile.tpop_from_aic")[0]
    assert tile_dtype(push.args[0]) == pl.FP32
    assert tile_dtype(pop) == pl.BF16
    assert push.kwargs["fixpipe_bf16"]
    assert push.kwargs["fixpipe_relu"] == relu
    for fn, init in [(cube, "system.aic_initialize_pipe"), (vector, "system.aiv_initialize_pipe")]:
        config = by_name(fn, init)[0]
        assert config.kwargs["slot_size"] == 64 * 128 * 2
        assert config.kwargs["fixpipe_bf16"]
        assert config.kwargs["fixpipe_relu"] == relu
    assert not by_name(vector, "tile.maximums")
    assert len(by_name(vector, "system.tfree_to_aic")) == 1


@pytest.mark.parametrize("mode", ["round", "floor", "ceil", "trunc"])
def test_other_rounding_modes_keep_fp32_transport(mode):
    backend.set_backend_type(backend.BackendType.Ascend950)
    after = expand(make_program(mode=mode))
    vector = after.get_function("main_aiv")
    assert tile_dtype(by_name(vector, "tile.tpop_from_aic")[0]) == pl.FP32
    assert by_name(vector, "tile.maximums")


def test_fp32_reader_keeps_fp32_transport():
    backend.set_backend_type(backend.BackendType.Ascend950)
    after = expand(make_program(extra_reader=True))
    assert tile_dtype(by_name(after.get_function("main_aiv"), "tile.tpop_from_aic")[0]) == pl.FP32


def test_bidirectional_pipe_keeps_separate_typed_rings():
    backend.set_backend_type(backend.BackendType.Ascend950)
    after = expand(make_program(bidirectional=True, cols=256))
    cube = after.get_function("main_aic")
    vector = after.get_function("main_aiv")
    for fn, init in [(cube, "system.aic_initialize_pipe"), (vector, "system.aiv_initialize_pipe")]:
        configs = by_name(fn, init)
        assert len(configs) == 2
        assert {c.kwargs.get("id", 0): c.kwargs["dir_mask"] for c in configs} == {0: 1, 1: 2}
        assert {c.kwargs.get("id", 0): c.kwargs["slot_size"] for c in configs} == {
            0: 64 * 256 * 2,
            1: 64 * 128 * 2,
        }
    assert by_name(vector, "tile.tpush_to_aic")[0].kwargs["id"] == 1
    assert by_name(cube, "tile.tpop_from_aiv")[0].kwargs["id"] == 1
    assert by_name(cube, "system.tfree_to_aiv")[0].kwargs["id"] == 1


def test_a2a3_keeps_existing_transport():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend910B)
    after = expand(make_program())
    assert tile_dtype(by_name(after.get_function("main_aiv"), "tile.tpop_from_aic")[0]) == pl.FP32


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("relu", [False, True])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_codegen_carries_fixpipe_configuration(tmp_path, relu, bidirectional):
    backend.set_backend_type(backend.BackendType.Ascend950)
    with passes.PassContext([], enable_buffer_ir=False):
        after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(
            make_program(relu=relu, bidirectional=bidirectional)
        )
    kernels = [function for function in after.functions.values() if ir.is_incore_type(function.func_type)]
    text = codegen.PTOCodegen().generate(ir.Program(kernels, "kernels", after.span))
    assert f"quant = f32_bf16, relu = {'normal_relu' if relu else 'no_relu'}" in text
    assert "nosplit = true" in text
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    cpp = pto_backend._compile_pto_module(text, "main_aic", str(tmp_path))
    assert "F322BF16" in cpp


@pytest.mark.parametrize("saturation_mode", ["on", "off"])
def test_explicit_saturation_keeps_vector_cast(saturation_mode):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[64, 128], pl.BF16],
            b: pl.Tensor[[128, 128], pl.BF16],
            out: pl.Out[pl.Tensor[[64, 128], pl.BF16]],
        ):
            a_mat = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
            left = pl.move(a_mat, target_memory=pl.Mem.Left)
            b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
            right = pl.move(b_mat, target_memory=pl.Mem.Right)
            dot = pl.matmul(left, right)
            rounded = pl.cast(dot, pl.BF16, mode="rint", saturation_mode=saturation_mode)
            pl.store(rounded, [0, 0], out)

    after = expand(Program)
    vector = after.get_function("main_aiv")
    assert tile_dtype(by_name(vector, "tile.tpop_from_aic")[0]) == pl.FP32
    assert by_name(vector, "tile.cast")


@pytest.mark.parametrize("second_mode,second_relu", [("floor", True), ("rint", False)])
def test_one_pipe_requires_uniform_conversion_and_relu(second_mode, second_relu):
    # One pipe config controls every C2V push. An incompatible cast must keep
    # both payloads FP32; a missing ReLU must keep every ReLU on Vector.
    activation = "value1 = pl.maximum(rounded1, 0.0)" if second_relu else "value1 = rounded1"
    before = pl.parse(f"""@pl.program
class Program:
    @pl.function(type=pl.FunctionType.InCore)
    def main(self, a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16],
             out0: pl.Out[pl.Tensor[[64, 128], pl.BF16]], out1: pl.Out[pl.Tensor[[64, 128], pl.BF16]]):
        am = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        bm = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        left = pl.move(am, target_memory=pl.Mem.Left)
        right = pl.move(bm, target_memory=pl.Mem.Right)
        dot0 = pl.matmul(left, right)
        rounded0 = pl.cast(dot0, pl.BF16, mode="rint")
        value0 = pl.maximum(rounded0, 0.0)
        pl.store(value0, [0, 0], out0)
        dot1 = pl.matmul(left, right)
        rounded1 = pl.cast(dot1, pl.BF16, mode={second_mode!r})
        {activation}
        pl.store(value1, [0, 0], out1)
""")
    after = expand(before)
    vector = after.get_function("main_aiv")
    pops = by_name(vector, "tile.tpop_from_aic")
    assert len(pops) == 2
    narrowed = second_mode == "rint"
    assert all(tile_dtype(pop) == (pl.BF16 if narrowed else pl.FP32) for pop in pops)
    init = by_name(vector, "system.aiv_initialize_pipe")[0]
    assert init.kwargs.get("fixpipe_bf16", False) == narrowed
    assert not init.kwargs.get("fixpipe_relu", False)
    assert len(by_name(vector, "tile.maximums")) == (1 if narrowed else 2)


@pytest.mark.parametrize(
    "target,dir_mask,bf16,relu,message",
    [
        (backend.BackendType.Ascend950, 1, False, True, "requires fixpipe_bf16"),
        (backend.BackendType.Ascend950, 3, True, False, "A5 C2V-only pipe"),
        (backend.BackendType.Ascend910, 1, True, False, "A5 C2V-only pipe"),
    ],
)
def test_invalid_fixpipe_configuration_is_rejected(target, dir_mask, bf16, relu, message):
    backend.reset_for_testing()
    backend.set_backend_type(target)

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIC)
        def main(self):
            pl.aic_initialize_pipe(dir_mask=dir_mask, slot_size=1024, fixpipe_bf16=bf16, fixpipe_relu=relu)

    with pytest.raises(ValueError, match=message):
        codegen.PTOCodegen().generate(Program)


def test_bf16_row_stride_alignment_keeps_fp32_transport():
    # Eight FP32 columns have a legal 32-byte C2V stride; narrowing that FIFO
    # to BF16 would produce a 16-byte stride, rejected by A5 TMovCcToUb.
    after = expand(make_program(cols=8))
    vector = after.get_function("main_aiv")
    assert tile_dtype(by_name(vector, "tile.tpop_from_aic")[0]) == pl.FP32
    assert not by_name(vector, "system.aiv_initialize_pipe")[0].kwargs.get("fixpipe_bf16", False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
