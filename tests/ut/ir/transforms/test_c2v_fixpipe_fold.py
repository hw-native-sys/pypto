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
import torch
from pypto import backend, codegen, ir, passes
from pypto.backend import pto_backend
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.debug import torch_codegen
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


def make_expected(
    *,
    mode="rint",
    relu=True,
    widening=True,
    extra_reader=False,
    bidirectional=False,
    cols=128,
    folded=True,
    saturation_mode=None,
    dual_aiv=False,
):
    """Author both split bodies and the group, including every setup and free."""
    signature = (
        f"a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, {cols}], pl.BF16], "
        f"out: pl.Out[pl.Tensor[[64, {cols}], pl.BF16]]"
    )
    wire = "BF16" if folded else "FP32"
    slot = 64 * cols * (2 if folded else 4)
    split_rings = folded and bidirectional
    mask = 1 if split_rings or not bidirectional else 3
    epilogue = f", fixpipe_bf16=True, fixpipe_relu={relu}" if folded else ""
    zero = "pl.const(0, pl.INT32)"
    cube = []
    vector = ['pl.func_attr({"dual_aiv_dispatch": True})'] if dual_aiv else []
    cube += [
        "c2v_import: pl.Scalar[pl.INT32] = "
        'pl.system.import_peer_buffer(name="main_c2v_slot_buffer", peer_func="main_aiv")',
        f"pl.system.aic_initialize_pipe(c2v_import, {zero}, dir_mask={mask}, slot_size={slot}, "
        f"slot_num=2{epilogue})",
    ]
    vector += [
        f'c2v: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(name="main_c2v_slot_buffer", '
        f"size={slot * 2}, base=-1)",
        f"pl.system.aiv_initialize_pipe(c2v, {zero}, dir_mask={mask}, slot_size={slot}, "
        f"slot_num=2{epilogue})",
    ]
    if split_rings:
        cube += [
            'v2c: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(name="main_v2c_slot_buffer", '
            "size=32768, base=-1)",
            f"pl.system.aic_initialize_pipe({zero}, v2c, dir_mask=2, slot_size=16384, slot_num=2, id=1)",
        ]
        vector += [
            "v2c_import: pl.Scalar[pl.INT32] = "
            'pl.system.import_peer_buffer(name="main_v2c_slot_buffer", peer_func="main_aic")',
            f"pl.system.aiv_initialize_pipe({zero}, v2c_import, dir_mask=2, slot_size=16384, "
            f"slot_num=2, id=1)",
        ]
    if bidirectional:
        pipe_id = ", id=1" if split_rings else ""
        vector += [
            "raw: pl.Tile[[64, 128], pl.BF16, pl.Mem.Vec] = pl.tile.full([64, 128], "
            "dtype=pl.BF16, value=pl.const(1.0, pl.BF16))",
            "raw_fractal: pl.Tile[[64, 128], pl.BF16, pl.Mem.Vec, "
            "pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = "
            "pl.tile.move(raw, target_memory=pl.Mem.Vec, blayout=pl.TileLayout.col_major, "
            "slayout=pl.TileLayout.row_major)",
            f"pl.tile.tpush_to_aic(raw_fractal, split=0{pipe_id})",
        ]
        cube += [
            f"raw_recv: pl.Tile[[64, 128], pl.BF16, pl.Mem.Mat] = pl.tile.tpop_from_aiv(split=0{pipe_id})",
            "left: pl.Tile[[64, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(raw_recv, "
            "target_memory=pl.Mem.Left, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)",
            f"pl.system.tfree_to_aiv(raw_recv{pipe_id})",
        ]
    else:
        cube += [
            "a_mat: pl.Tile[[64, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(a, [0, 0], [64, "
            "128], [64, 128], target_memory=pl.Mem.Mat)",
            "left: pl.Tile[[64, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(a_mat, target_memory=pl.Mem.Left)",
        ]
    cube += [
        f"b_mat: pl.Tile[[128, {cols}], pl.BF16, pl.Mem.Mat] = pl.tile.load(b, [0, 0], [128, "
        f"{cols}], [128, {cols}], target_memory=pl.Mem.Mat)",
        f"right: pl.Tile[[128, {cols}], pl.BF16, pl.Mem.Right] = pl.tile.move(b_mat, "
        f"target_memory=pl.Mem.Right)",
        f"dot: pl.Tile[[64, {cols}], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(left, right)",
        f"pl.tile.tpush_to_aiv(dot, split=0{epilogue})",
    ]
    pop = "rounded" if folded else "received"
    if folded and relu and not widening:
        pop = "activated"
    vector += [f"{pop}: pl.Tile[[64, {cols}], pl.{wire}, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=0)"]
    if not folded:
        sat = f", saturation_mode={saturation_mode!r}" if saturation_mode else ""
        vector += [f"rounded = pl.tile.cast(received, target_type=pl.BF16, mode={mode!r}{sat})"]
        if not extra_reader:
            vector += ["pl.system.tfree_to_aic(received)"]
    value = pop if folded else "rounded"
    if widening:
        vector += [f'widened = pl.tile.cast({value}, target_type=pl.FP32, mode="round")']
        value = "widened"
        if folded:
            vector += [f"pl.system.tfree_to_aic({pop})"]
    if relu:
        if folded and widening:
            vector += ["activated = widened"]
        elif not folded:
            vector += [f"activated = pl.tile.maximums({value}, 0.0)"]
        value = "activated"
    if extra_reader:
        vector += [f"extra = pl.tile.add({value}, received)", "pl.system.tfree_to_aic(received)"]
        value = "extra"
    if widening or extra_reader:
        vector += [f'final = pl.tile.cast({value}, target_type=pl.BF16, mode="rint")']
        value = "final"
    vector += [f"pl.tile.store({value}, [0, 0], out)"]
    if folded and not widening:
        vector += [f"pl.system.tfree_to_aic({pop})"]
    lines = ["@pl.program", "class Program:"]
    for name, kind, level, body in [
        ("main_aic", "AIC", "AIC", cube),
        ("main_aiv", "AIV", "AIV", vector),
        ("main", "Group", "CORE_GROUP", ["self.main_aic(a, b, out)", "self.main_aiv(a, b, out)"]),
    ]:
        lines += [
            f"    @pl.function(type=pl.FunctionType.{kind}, level=pl.Level.{level}, role=pl.Role.SubWorker)",
            f"    def {name}(self, {signature}):",
        ]
        lines += ["        " + statement for statement in body]
    return pl.parse("\n".join(lines))


def expand(program):
    return passes.expand_mixed_kernel()(passes.infer_tile_memory_space()(passes.convert_to_ssa()(program)))


def by_name(function, name):
    expected = ir.get_op(name).name
    return [call for call in calls(function) if call.op.name == expected]


@pytest.mark.parametrize("relu", [False, True])
@pytest.mark.parametrize("widening", [False, True])
def test_bf16_transport_and_optional_relu(relu, widening):
    backend.set_backend_type(backend.BackendType.Ascend950)
    Before = make_program(relu=relu, widening=widening)
    Expected = make_expected(relu=relu, widening=widening)
    after = expand(Before)
    ir.assert_structural_equal(after, Expected)
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
    Before = make_program(mode=mode)
    Expected = make_expected(mode=mode, folded=False)
    after = expand(Before)
    ir.assert_structural_equal(after, Expected)
    vector = after.get_function("main_aiv")
    assert tile_dtype(by_name(vector, "tile.tpop_from_aic")[0]) == pl.FP32
    assert by_name(vector, "tile.maximums")


def test_fp32_reader_keeps_fp32_transport():
    backend.set_backend_type(backend.BackendType.Ascend950)
    Before = make_program(extra_reader=True)
    Expected = make_expected(extra_reader=True, folded=False)
    after = expand(Before)
    ir.assert_structural_equal(after, Expected)
    assert tile_dtype(by_name(after.get_function("main_aiv"), "tile.tpop_from_aic")[0]) == pl.FP32


def test_bidirectional_pipe_keeps_separate_typed_rings():
    backend.set_backend_type(backend.BackendType.Ascend950)
    Before = make_program(bidirectional=True, cols=256)
    Expected = make_expected(bidirectional=True, cols=256)
    after = expand(Before)
    ir.assert_structural_equal(after, Expected)
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
    Before = make_program()
    Expected = make_expected(folded=False, dual_aiv=True)
    after = expand(Before)
    ir.assert_structural_equal(after, Expected)
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

    Expected = make_expected(relu=False, widening=False, folded=False, saturation_mode=saturation_mode)
    after = expand(Program)
    ir.assert_structural_equal(after, Expected)
    vector = after.get_function("main_aiv")
    assert tile_dtype(by_name(vector, "tile.tpop_from_aic")[0]) == pl.FP32
    assert by_name(vector, "tile.cast")


def make_uniform_pipe_expected(second_mode, second_relu):
    folded = second_mode == "rint"
    size = 16384 if folded else 32768
    flags = ", fixpipe_bf16=True, fixpipe_relu=False" if folded else ""
    signature = (
        "a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16], "
        "out0: pl.Out[pl.Tensor[[64, 128], pl.BF16]], out1: pl.Out[pl.Tensor[[64, 128], pl.BF16]]"
    )
    cube = [
        "buf: pl.Scalar[pl.INT32] = "
        'pl.system.import_peer_buffer(name="main_c2v_slot_buffer", peer_func="main_aiv")',
        f"pl.system.aic_initialize_pipe(buf, pl.const(0, pl.INT32), dir_mask=1, "
        f"slot_size={size}, slot_num=2{flags})",
        "am = pl.tile.load(a, [0, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat)",
        "bm = pl.tile.load(b, [0, 0], [128, 128], [128, 128], target_memory=pl.Mem.Mat)",
        "left = pl.tile.move(am, target_memory=pl.Mem.Left)",
        "right = pl.tile.move(bm, target_memory=pl.Mem.Right)",
    ]
    vector = [
        f'buf: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(name="main_c2v_slot_buffer", '
        f"size={size * 2}, base=-1)",
        f"pl.system.aiv_initialize_pipe(buf, pl.const(0, pl.INT32), dir_mask=1, "
        f"slot_size={size}, slot_num=2{flags})",
    ]
    for i, mode in enumerate(["rint", second_mode]):
        cube += [f"dot{i} = pl.tile.matmul(left, right)", f"pl.tile.tpush_to_aiv(dot{i}, split=0{flags})"]
        pop = f"rounded{i}" if folded else f"recv{i}"
        dtype = "BF16" if folded else "FP32"
        vector += [f"{pop}: pl.Tile[[64, 128], pl.{dtype}, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=0)"]
        if not folded:
            vector += [
                f"rounded{i} = pl.tile.cast({pop}, target_type=pl.BF16, mode={mode!r})",
                f"pl.system.tfree_to_aic({pop})",
            ]
        value = f"pl.tile.maximums(rounded{i}, 0.0)" if i == 0 or second_relu else f"rounded{i}"
        vector += [f"value{i} = {value}"]
        if folded and i == 0:
            vector += [f"pl.system.tfree_to_aic({pop})"]
        vector += [f"pl.tile.store(value{i}, [0, 0], out{i})"]
        if folded and i == 1:
            vector += [f"pl.system.tfree_to_aic({pop})"]
    lines = ["@pl.program", "class Program:"]
    for name, kind, level, body in [
        ("main_aic", "AIC", "AIC", cube),
        ("main_aiv", "AIV", "AIV", vector),
        (
            "main",
            "Group",
            "CORE_GROUP",
            ["self.main_aic(a, b, out0, out1)", "self.main_aiv(a, b, out0, out1)"],
        ),
    ]:
        lines += [
            f"    @pl.function(type=pl.FunctionType.{kind}, level=pl.Level.{level}, role=pl.Role.SubWorker)",
            f"    def {name}(self, {signature}):",
        ]
        lines += ["        " + statement for statement in body]
    return pl.parse("\n".join(lines))


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
    Expected = make_uniform_pipe_expected(second_mode, second_relu)
    after = expand(before)
    ir.assert_structural_equal(after, Expected)
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
    Before = make_program(cols=8)
    Expected = make_expected(cols=8, folded=False)
    after = expand(Before)
    ir.assert_structural_equal(after, Expected)
    vector = after.get_function("main_aiv")
    assert tile_dtype(by_name(vector, "tile.tpop_from_aic")[0]) == pl.FP32
    assert not by_name(vector, "system.aiv_initialize_pipe")[0].kwargs.get("fixpipe_bf16", False)


def make_lifetime_case(kind, *, expected=False, slots: int | None = 2):
    attrs = [f'pl.func_attr({{"slot_num": {slots}}})'] if slots is not None else []
    slots = slots if slots is not None else 2
    signature = (
        "a: pl.Tensor[[16, 16], pl.BF16], b: pl.Tensor[[16, 16], pl.BF16], "
        "flag: pl.Scalar[pl.BOOL], out: pl.Out[pl.Tensor[[48, 16], pl.BF16]]"
    )
    folded = kind == "loop" or (kind == "capacity" and slots >= 3)
    count = 3 if kind == "capacity" else 1
    cube_loads = [
        "am = pl.tile.load(a, [0, 0], [16, 16], [16, 16], target_memory=pl.Mem.Mat)",
        "left = pl.tile.move(am, target_memory=pl.Mem.Left)",
        "bm = pl.tile.load(b, [0, 0], [16, 16], [16, 16], target_memory=pl.Mem.Mat)",
        "right = pl.tile.move(bm, target_memory=pl.Mem.Right)",
    ]
    body = [] if expected else list(cube_loads)
    for i in range(count):
        if expected:
            dtype = "BF16" if folded else "FP32"
            name = f"r{i}" if folded else f"v{i}"
            body += [f"{name}: pl.Tile[[16, 16], pl.{dtype}, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=0)"]
        else:
            body += [
                f"x{i} = pl.tile.matmul(left, right)",
                f"v{i} = pl.tile.move(x{i}, target_memory=pl.Mem.Vec, "
                f"blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)",
            ]
        if kind == "branch":
            body += [
                "if flag:",
                '    r0 = pl.tile.cast(v0, target_type=pl.BF16, mode="rint")',
                "    pl.tile.store(r0, [0, 0], out)",
            ]
        elif not (expected and folded):
            body += [f'r{i} = pl.tile.cast(v{i}, target_type=pl.BF16, mode="rint")']
        if expected and not folded:
            body += [f"pl.system.tfree_to_aic(v{i})"]
    if kind == "loop":
        body += ['wide = pl.tile.cast(r0, target_type=pl.FP32, mode="round")']
        if expected:
            body += ["pl.system.tfree_to_aic(r0)"]
        body += [
            "active = pl.tile.maximums(wide, 0.0)",
            "for i, (carry,) in pl.range(1, init_values=(wide,)):",
            "    result = pl.yield_(carry)",
            'negative = pl.tile.cast(result, target_type=pl.BF16, mode="rint")',
            'positive = pl.tile.cast(active, target_type=pl.BF16, mode="rint")',
            "pl.tile.store(negative, [0, 0], out)",
            "pl.tile.store(positive, [16, 0], out)",
        ]
    elif kind == "capacity":
        for i in range(count):
            body += [f"pl.tile.store(r{i}, [{i * 16}, 0], out)"]
            if expected and folded:
                body += [f"pl.system.tfree_to_aic(r{i})"]
    lines = ["@pl.program", "class Program:"]
    if not expected:
        functions = [("main", "InCore", None, attrs + body)]
    else:
        size = 512 if folded else 1024
        epilogue = ", fixpipe_bf16=True, fixpipe_relu=False" if folded else ""
        cube = [
            "buf: pl.Scalar[pl.INT32] = "
            'pl.system.import_peer_buffer(name="main_c2v_slot_buffer", peer_func="main_aiv")',
            f"pl.system.aic_initialize_pipe(buf, pl.const(0, pl.INT32), dir_mask=1, "
            f"slot_size={size}, slot_num={slots}{epilogue})",
        ]
        cube += cube_loads
        for i in range(count):
            cube += [f"x{i} = pl.tile.matmul(left, right)", f"pl.tile.tpush_to_aiv(x{i}, split=0{epilogue})"]
        cube += ["for i in pl.range(1):", "    pass"] if kind == "loop" else []
        vector = [
            f'buf: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(name="main_c2v_slot_buffer", '
            f"size={size * slots}, base=-1)",
            f"pl.system.aiv_initialize_pipe(buf, pl.const(0, pl.INT32), dir_mask=1, "
            f"slot_size={size}, slot_num={slots}{epilogue})",
        ] + body
        functions = [
            ("main_aic", "AIC", "AIC", cube),
            ("main_aiv", "AIV", "AIV", vector),
            (
                "main",
                "Group",
                "CORE_GROUP",
                ["self.main_aic(a, b, flag, out)", "self.main_aiv(a, b, flag, out)"],
            ),
        ]
    for name, function_kind, level, statements_ in functions:
        placement = f", level=pl.Level.{level}, role=pl.Role.SubWorker" if level else ""
        lines += [
            f"    @pl.function(type=pl.FunctionType.{function_kind}{placement})",
            f"    def {name}(self, {signature}):",
        ]
        function_attrs = attrs if expected else []
        lines += ["        " + line for line in function_attrs + statements_]
    return pl.parse("\n".join(lines))


@pytest.mark.parametrize(
    "kind,slots", [("loop", 2), ("branch", 2), ("capacity", None), ("capacity", 2), ("capacity", 3)]
)
def test_fold_preserves_initializer_branch_and_fifo_lifetimes(kind, slots):
    Before = make_lifetime_case(kind, slots=slots)
    Expected = make_lifetime_case(kind, expected=True, slots=slots)
    after = expand(Before)
    ir.assert_structural_equal(after, Expected)
    # Execute the expanded group with shape/dtype checks. Negative inputs make
    # an incorrectly hoisted ReLU observable in the loop initializer's output.
    ns = {}
    exec(torch_codegen(after, check_shapes=True), ns)  # noqa: S102
    a = torch.full((16, 16), -2.0, dtype=torch.bfloat16)
    b = torch.eye(16, dtype=torch.bfloat16)
    output = torch.zeros((48, 16), dtype=torch.bfloat16)
    ns["main"](a, b, True, output)
    expected = torch.zeros_like(output)
    if kind == "capacity":
        expected[:] = a.repeat(3, 1)
    else:
        expected[:16] = a
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_different_pop_shapes_keep_fp32_pipe():
    # One Fixpipe initializer cannot serve differently shaped BF16 pop types.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self, out: pl.Out[pl.Tensor[[32, 16], pl.BF16]]):
            a0 = pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            v0 = pl.tile.move(
                a0, target_memory=pl.Mem.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box
            )
            b0 = pl.tile.cast(v0, target_type=pl.BF16, mode="rint")
            pl.tile.store(b0, [0, 0], out)
            a1 = pl.tile.create([32, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            v1 = pl.tile.move(
                a1, target_memory=pl.Mem.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box
            )
            b1 = pl.tile.cast(v1, target_type=pl.BF16, mode="rint")
            pl.tile.store(b1, [0, 0], out)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
        def main_aic(self, out: pl.Out[pl.Tensor[[32, 16], pl.BF16]]):
            buf = pl.system.import_peer_buffer(name="main_c2v_slot_buffer", peer_func="main_aiv")
            pl.system.aic_initialize_pipe(buf, pl.const(0, pl.INT32), dir_mask=1, slot_size=2048, slot_num=2)
            a0 = pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            pl.tile.tpush_to_aiv(a0, split=0)
            a1 = pl.tile.create([32, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            pl.tile.tpush_to_aiv(a1, split=0)

        @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
        def main_aiv(self, out: pl.Out[pl.Tensor[[32, 16], pl.BF16]]):
            buf = pl.system.reserve_buffer(name="main_c2v_slot_buffer", size=4096, base=-1)
            pl.system.aiv_initialize_pipe(buf, pl.const(0, pl.INT32), dir_mask=1, slot_size=2048, slot_num=2)
            v0: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=0)
            b0 = pl.tile.cast(v0, target_type=pl.BF16, mode="rint")
            pl.system.tfree_to_aic(v0)
            pl.tile.store(b0, [0, 0], out)
            v1: pl.Tile[[32, 16], pl.FP32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=0)
            b1 = pl.tile.cast(v1, target_type=pl.BF16, mode="rint")
            pl.system.tfree_to_aic(v1)
            pl.tile.store(b1, [0, 0], out)

        @pl.function(type=pl.FunctionType.Group, level=pl.Level.CORE_GROUP, role=pl.Role.SubWorker)
        def main(self, out: pl.Out[pl.Tensor[[32, 16], pl.BF16]]):
            self.main_aic(out)
            self.main_aiv(out)

    ir.assert_structural_equal(expand(Before), Expected)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("buffer_ir", [False, True])
@pytest.mark.parametrize("reinterpret", [False, True])
def test_metadata_view_through_alias_keeps_local_cast(tmp_path, buffer_ir, reinterpret):
    # A tpop result cannot replace the local cast allocation used by these views.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self, rows: pl.Scalar[pl.INDEX], out: pl.Out[pl.Tensor[[16, 16], pl.BF16]]):
            acc = pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            vec = pl.tile.move(
                acc, target_memory=pl.Mem.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box
            )
            rounded = pl.tile.cast(vec, target_type=pl.BF16, mode="rint")
            alias = rounded
            if reinterpret:
                words = pl.tile.reinterpret_view(alias, pl.INT16)
                values = pl.tile.reinterpret_view(words, pl.BF16)
                pl.tile.store(values, [0, 0], out)
            else:
                partial = pl.tile.set_validshape(alias, rows, 16)
                pl.tile.store(partial, [0, 0], out)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
        def main_aic(self, rows: pl.Scalar[pl.INDEX], out: pl.Out[pl.Tensor[[16, 16], pl.BF16]]):
            buf = pl.system.import_peer_buffer(name="main_c2v_slot_buffer", peer_func="main_aiv")
            pl.system.aic_initialize_pipe(buf, pl.const(0, pl.INT32), dir_mask=1, slot_size=1024, slot_num=2)
            acc = pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            pl.tile.tpush_to_aiv(acc, split=0)

        @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
        def main_aiv(self, rows: pl.Scalar[pl.INDEX], out: pl.Out[pl.Tensor[[16, 16], pl.BF16]]):
            buf = pl.system.reserve_buffer(name="main_c2v_slot_buffer", size=2048, base=-1)
            pl.system.aiv_initialize_pipe(buf, pl.const(0, pl.INT32), dir_mask=1, slot_size=1024, slot_num=2)
            vec: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=0)
            rounded = pl.tile.cast(vec, target_type=pl.BF16, mode="rint")
            pl.system.tfree_to_aic(vec)
            alias = rounded
            if reinterpret:
                words = pl.tile.reinterpret_view(alias, pl.INT16)
                values = pl.tile.reinterpret_view(words, pl.BF16)
                pl.tile.store(values, [0, 0], out)
            else:
                partial = pl.tile.set_validshape(alias, rows, 16)
                pl.tile.store(partial, [0, 0], out)

        @pl.function(type=pl.FunctionType.Group, level=pl.Level.CORE_GROUP, role=pl.Role.SubWorker)
        def main(self, rows: pl.Scalar[pl.INDEX], out: pl.Out[pl.Tensor[[16, 16], pl.BF16]]):
            self.main_aic(rows, out)
            self.main_aiv(rows, out)

    ir.assert_structural_equal(expand(Before), Expected)
    with passes.PassContext([], enable_buffer_ir=buffer_ir):
        after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Before)
    kernels = [function for function in after.functions.values() if ir.is_incore_type(function.func_type)]
    text = codegen.PTOCodegen().generate(ir.Program(kernels, "kernels", after.span))
    assert "acc_push_epilogue" not in text
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    pto_backend._compile_pto_module(text, "main_aic", str(tmp_path))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
