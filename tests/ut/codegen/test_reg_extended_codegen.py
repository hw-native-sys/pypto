# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Extended register APIs through parsing, passes and VMI lowering."""

import subprocess

import pypto.language as pl
import pytest
from pypto import DataType, backend, codegen, ir, passes
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.builder import IRBuilder
from pypto.ir.op import tile
from pypto.ir.printer import python_print
from pypto.language.parser import ParserError
from pypto.runtime._vpto import PTOAS_FLAGS

SPAN = ir.Span.unknown()
CASES = [
    ("expdif", "FP32", "y = pl.reg.expdif(x, x, m)"),
    ("expdif", "FP16", "y = pl.reg.expdif(x, x, m)"),
    ("axpy", "FP32", "y = pl.reg.axpy(x, x, 0.5, m)"),
    ("axpy", "FP16", "y = pl.reg.axpy(x, x, 0.5, m)"),
    ("mula", "FP32", "y = pl.reg.mula(x, x, x, m)"),
    ("mula", "INT32", "y = pl.reg.mula(x, x, x)"),
    ("lrelu", "FP32", "y = pl.reg.lrelu(x, 0.25, m)"),
    ("prelu", "FP32", "y = pl.reg.prelu(x, x, m)"),
    ("mull", "INT32", "y, second = pl.reg.mull(x, x, m)"),
    ("mull", "UINT32", "y, second = pl.reg.mull(x, x, m)"),
    ("addc", "UINT32", "y, carry = pl.reg.addc(x, x, m)"),
    ("subc", "UINT32", "y, carry = pl.reg.subc(x, x, m)"),
    ("addcs", "INT32", "y, carry = pl.reg.addcs(x, x, m, m)"),
    ("subcs", "INT32", "y, carry = pl.reg.subcs(x, x, m, m)"),
    ("gather", "FP32", "y = pl.reg.gather(src, indices, m)"),
    ("gather", "FP16", "y = pl.reg.gather(src, indices, m)"),
    ("gather", "INT8", "y = pl.reg.gather(src, indices, m)"),
    ("scatter", "FP32", "out = pl.reg.scatter(dst, indices, x, m)"),
    ("scatter", "INT8", "out = pl.reg.scatter(dst, indices, x, m)"),
    ("intlv", "FP32", "y, second = pl.reg.intlv(x, x, m)"),
    ("dintlv", "FP32", "y, second = pl.reg.dintlv(x, x, m)"),
    *[
        (name, dtype, f"y, second = pl.reg.{name}(x, x, m)")
        for name in ("intlv", "dintlv")
        for dtype in ("FP16", "BF16", "INT32", "UINT32")
    ],
]


@pytest.fixture(autouse=True)
def a5_backend():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def frontend(name, dtype, statement, tensor_io=False):
    output_dtype = "FP32" if name == "expdif" else "INT16" if name == "gather" and dtype == "INT8" else dtype
    index_dtype = "UINT16" if dtype in ("FP16", "INT8") else "UINT32"
    lines = ["@pl.function(type=pl.FunctionType.AIV)"]
    if tensor_io:
        lines.append(
            f"def kernel(input: pl.Tensor[[1, 256], pl.{dtype}], "
            f"output: pl.InOut[pl.Tensor[[1, 256], pl.{output_dtype}]]):"
        )
        for local, param, element in (("src", "input", dtype), ("dst", "output", output_dtype)):
            lines.append(
                f"    {local}: pl.Tile[[1, 256], pl.{element}, pl.MemorySpace.Vec, "
                "pl.TileView(valid_shape=[1, 256], stride=[256, 1])] = "
                f"pl.tile.load({param}, [0, 0], [1, 256], [1, 256], target_memory=pl.MemorySpace.Vec)"
            )
    else:
        lines.extend(
            [
                "def kernel():",
                f"    src = pl.tile.full([256], pl.{dtype}, 1)",
                f"    dst = pl.tile.full([256], pl.{output_dtype}, 0)",
            ]
        )
    lines.extend(
        [
            "    with pl.reg_scope():",
            "        x = pl.reg.load(src, lanes=64)",
            "        m = pl.reg.active_mask(32, lanes=64)",
            f"        indices = pl.reg.index_gen(0, lanes=64, dtype=pl.{index_dtype})",
            f"        {statement}",
        ]
    )
    if "carry" in statement:
        lines.append("        y = pl.reg.select(carry, y, x)")
    if "second" in statement:
        lines.append("        dst = pl.reg.store(dst, offset=64, value=second, mask=m)")
    if name != "scatter":
        lines.append("        out = pl.reg.store(dst, value=y, mask=m)")
    lines.append(
        "    _stored = pl.tile.store(out, [0, 0], output)"
        if tensor_io
        else "    _consumer = pl.tile.add(out, out)"
    )
    function = pl.parse("\n".join(lines))
    assert isinstance(function, ir.Function)
    return function


def lower(function):
    with passes.PassContext([]):
        program = ir.Program([function], "ExtendedRegisters", SPAN)
        program = passes.convert_to_ssa()(program)
        program = passes.flatten_tile_nd_to_2d()(program)
        program = passes.init_mem_ref()(program)
        return passes.allocate_memory_addr()(program)


@pytest.mark.parametrize("name,dtype,statement", CASES)
def test_roundtrip_and_codegen(name, dtype, statement):
    function = frontend(name, dtype, statement)
    printed = python_print(function, format=False)
    ir.assert_structural_equal(function, pl.parse(printed), enable_auto_mapping=True)
    source = codegen.PTOCodegen().generate(lower(function), emit_source_loc=False)
    # Tuple projections must not duplicate the hardware operation.
    assert source.count(f"pto.vmi.v{name}") == 1
    line = next(line for line in source.splitlines() if f"pto.vmi.v{name}" in line)
    if "second" in statement or "carry" in statement:
        assert "," in line.split(" = ", 1)[0]
        assert "," in line.split(" -> ", 1)[1]
    if name in ("addc", "subc", "addcs", "subcs"):
        assert "pmode" not in line
        assert "xpred>" in line.split(" -> ", 1)[1]
    if name in ("intlv", "dintlv"):
        # PTOAS drops the rearrangement mask; both results need explicit zeroing.
        assert source.count("pto.vmi.vsel") == 2
        assert "interleave_zero" in source
        if dtype == "BF16":
            assert source.count("pto.vmi.vinterpret_cast") == 4
    if name == "expdif":
        assert line.endswith("!pto.vmi.vreg<64xf32>")
    if name == "scatter":
        assert " = " not in line and " -> " not in line
        assert "pto.tadd" in source.split("pto.vmi.vscatter", 1)[1]
    if name == "mull" and dtype == "INT32":
        assert "si32" not in line and "xi32>" in line
        assert source.count("pto.vmi.vinterpret_cast") == 4


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("name,dtype,statement", CASES)
def test_extended_ops_compile_with_ptoas(name, dtype, statement, tmp_path):
    compiler = find_ptoas_binary()
    if compiler is None:
        pytest.skip("PTOAS is unavailable")
    source = codegen.PTOCodegen().generate(lower(frontend(name, dtype, statement)), emit_source_loc=False)
    path = tmp_path / "extended.pto"
    path.write_text(source)
    result = subprocess.run(
        [
            compiler,
            str(path),
            "--pto-arch=a5",
            "--pto-level=level3",
            "--emit-vpto",
            "-o",
            str(tmp_path / "extended.vpto"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "name,dtype,statement",
    [case for case in CASES if case[0] in ("mull", "scatter", "gather") and case[1] != "INT8"],
)
def test_vf_helper_capture_and_effects(name, dtype, statement):
    bundle = codegen.PTOCodegen().generate_bundle(
        lower(frontend(name, dtype, statement)), emit_source_loc=False
    )
    assert len(bundle.helpers) == 1
    helper = bundle.helpers[0]
    assert f"pto.vmi.v{name}" in helper.source
    assert "readwrite" in helper.parameter_effects


def test_paired_result_cannot_escape_reg_scope():
    with pytest.raises((ValueError, ParserError), match="escape|outside|RegScope"):
        pl.parse("""
@pl.function(type=pl.FunctionType.AIV)
def kernel():
    with pl.reg_scope():
        x = pl.reg.broadcast(1, lanes=64, dtype=pl.UINT32)
        m = pl.reg.full_mask(lanes=64)
        pair = pl.reg.mull(x, x, m)
    escaped = pair[0]
""")


@pytest.mark.parametrize("op_name", ["mull", "addc", "subc", "addcs", "subcs", "intlv", "dintlv"])
@pytest.mark.parametrize("index", [0, 1])
def test_direct_paired_call_projection_rejected_before_codegen(op_name, index):
    args = "x, x, m, m" if op_name in ("addcs", "subcs") else "x, x, m"
    # Direct indexing leaves the call nested inside TupleGetItemExpr. Reject it
    # during verification rather than reaching an unbound codegen SSA lookup.
    with pytest.raises((ValueError, ParserError), match="bind the paired call before indexing"):
        pl.parse(f"""
@pl.function(type=pl.FunctionType.AIV)
def kernel():
    with pl.reg_scope():
        x = pl.reg.broadcast(2, lanes=64, dtype=pl.UINT32)
        m = pl.reg.full_mask(lanes=64)
        component = pl.reg.{op_name}({args})[{index}]
""")


def test_unused_pair_projection_and_alias():
    function = pl.parse("""
@pl.function(type=pl.FunctionType.AIV)
def kernel():
    dst = pl.tile.full([128], pl.UINT32, 0)
    with pl.reg_scope():
        x = pl.reg.broadcast(2, lanes=64, dtype=pl.UINT32)
        m = pl.reg.full_mask(lanes=64)
        pair = pl.reg.mull(x, x, m)
        alias = pair
        lo = alias[0]
        out = pl.reg.store(dst, value=lo)
""")
    source = codegen.PTOCodegen().generate(lower(function), emit_source_loc=False)
    assert source.count("pto.vmi.vmull ") == 1
    assert "pto.vmi.vstore" in source


@pytest.mark.parametrize("op_name", ["expdif", "mull", "gather"])
def test_builder_defaults_roundtrip(op_name):
    builder = IRBuilder()
    dtype = DataType.UINT32 if op_name == "mull" else DataType.FP32
    with builder.function("kernel", type=ir.FunctionType.AIV) as function:
        src = builder.let("src", tile.full([128], dtype=dtype, value=1))
        with builder.reg_scope():
            x = builder.emit_reg("x", "reg.load", [src, ir.ConstInt(0, DataType.INDEX, SPAN)], {"lanes": 64})
            m = builder.emit_reg("m", "reg.full_mask", [], {"lanes": 64})
            indices = builder.emit_reg(
                "indices",
                "reg.index_gen",
                [ir.ConstInt(0, DataType.UINT32, SPAN)],
                {"lanes": 64, "dtype": DataType.UINT32},
            )
            args = [src, indices, m] if op_name == "gather" else [x, x, m]
            builder.emit_reg("result", f"reg.{op_name}", args)
    original = function.get_result()
    ir.assert_structural_equal(
        original, pl.parse(python_print(original, format=False)), enable_auto_mapping=True
    )


@pytest.mark.ptoas_compile
def test_indexed_view_origin_and_scatter_alias_compile(tmp_path):
    builder = IRBuilder()
    with builder.function("kernel", type=ir.FunctionType.AIV) as function:
        view = ir.TileView([128], [1], ir.ConstInt(32, DataType.INDEX, SPAN))
        src = builder.let(
            "src",
            tile.full([128], dtype=DataType.FP32, value=1.0),
            type=ir.TileType([128], DataType.FP32, tile_view=view, memory_space=ir.MemorySpace.Vec),
        )
        with builder.reg_scope():
            indices = builder.emit_reg(
                "indices",
                "reg.index_gen",
                [ir.ConstInt(0, DataType.UINT32, SPAN)],
                {"lanes": 64, "dtype": DataType.UINT32},
            )
            m = builder.emit_reg("m", "reg.full_mask", [], {"lanes": 64})
            value = builder.emit_reg("value", "reg.gather", [src, indices, m])
            updated = builder.emit_reg("updated", "reg.scatter", [src, indices, value, m])
            builder.emit_reg("reread", "reg.gather", [updated, indices, m])
    with passes.PassContext([]):
        program = passes.init_mem_ref()(ir.Program([function.get_result()], "View", SPAN))
        program = passes.allocate_memory_addr()(program)
    source = codegen.PTOCodegen().generate(program, emit_source_loc=False)
    assert source.count("pto.tile_buf_addr") == 1
    assert source.count("pto.addptr") == 3  # Each access adds its view origin once.
    # The pinned PTOAS handles unknown indexed dependencies, including aliases.
    assert 'pto.mem_bar "VV_ALL"' not in source
    compiler = find_ptoas_binary()
    if compiler is None:
        pytest.skip("PTOAS is unavailable")
    path = tmp_path / "view.pto"
    path.write_text(source)
    result = subprocess.run(
        [
            compiler,
            str(path),
            "--pto-arch=a5",
            "--pto-level=level3",
            "--emit-vpto",
            "-o",
            str(tmp_path / "view.vpto"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name,dtype,statement", CASES)
def test_extended_ops_survive_default_pipeline(name, dtype, statement, tmp_path):
    # Tensor IO keeps the operation observable through the complete default
    # pipeline, including DCE, memory reuse and alias propagation.
    kernel = frontend(name, dtype, statement, tensor_io=True)
    with passes.PassContext([]):
        ir.compile(
            ir.Program([kernel], "ExtendedDefault", SPAN),
            output_dir=str(tmp_path),
            backend_type=backend.BackendType.Ascend950,
            skip_ptoas=True,
        )
    generated = (tmp_path / "kernels/aiv/kernel.pto").read_text()
    assert f"pto.vmi.v{name}" in generated


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("name,dtype,statement", CASES)
def test_extended_vf_helpers_compile_to_llvm(name, dtype, statement, tmp_path):
    compiler = find_ptoas_binary()
    if compiler is None:
        pytest.skip("PTOAS is unavailable")
    bundle = codegen.PTOCodegen().generate_bundle(
        lower(frontend(name, dtype, statement)), emit_source_loc=False
    )
    path = tmp_path / "helper.pto"
    path.write_text(bundle.helpers[0].source)
    result = subprocess.run(
        [
            compiler,
            str(path),
            *PTOAS_FLAGS,
            "-o",
            str(tmp_path / "helper.ll"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.ptoas_compile
def test_signed_vf_adapter_compiles_with_ptoas(tmp_path):
    compiler = find_ptoas_binary()
    if compiler is None:
        pytest.skip("PTOAS is unavailable")
    bundle = codegen.PTOCodegen().generate_bundle(
        lower(frontend("mull", "INT32", "y, second = pl.reg.mull(x, x, m)")), emit_source_loc=False
    )
    # A signed VMI pointer cannot be extracted directly from signless Tile storage.
    assert "pto.castptr %storage" in bundle.caller_source
    path = tmp_path / "caller.pto"
    path.write_text(bundle.caller_source)
    result = subprocess.run(
        [
            compiler,
            str(path),
            "--pto-arch=a5",
            "--pto-level=level3",
            "--pto-backend=emitc",
            "-o",
            str(tmp_path / "caller.cpp"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("bins", [128, 256])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("reverse_captures", [False, True])
def test_histogram_group_mask_and_reinterpret_codegen(bins, dynamic, reverse_captures, tmp_path):
    extra = (
        "dm = pl.reg.create_group_mask(n, num_groups=4, group_size=16)\n"
        "        zero = pl.reg.broadcast(0, lanes=64, dtype=pl.UINT32)\n"
        "        words = pl.reg.select(dm, words, zero)"
        if dynamic
        else ""
    )
    loads = ["x = pl.reg.load(src, lanes=256)", f"acc = pl.reg.load(dst, offset=768, lanes={bins})"]
    if reverse_captures:
        loads.reverse()
    load_statements = "\n        ".join(loads)
    function = pl.parse(f"""
@pl.function(type=pl.FunctionType.AIV)
def kernel(n: pl.Scalar[pl.INDEX]):
    src = pl.tile.full([256], pl.UINT8, 1)
    dst = pl.tile.full([1024], pl.UINT16, 3)
    with pl.reg_scope():
        {load_statements}
        m = pl.reg.create_group_mask(7, num_groups=16, group_size=16)
        h = pl.reg.chist(acc, x, m)
        d = pl.reg.dhist(acc, x, m)
        words = pl.reg.interpret_cast(x, dtype=pl.UINT32)
        {extra}
        halves = pl.reg.interpret_cast(words, dtype=pl.UINT16)
        out = pl.reg.store(dst, value=h)
        out = pl.reg.store(out, offset=256, value=d)
        out = pl.reg.store(out, offset=512, value=halves)
    _consumer = pl.tile.add(out, out)
""")
    ir.assert_structural_equal(
        function, pl.parse(python_print(function, format=False)), enable_auto_mapping=True
    )
    bundle = codegen.PTOCodegen().generate_bundle(lower(function), emit_source_loc=False)
    helper = bundle.helpers[0]
    assert helper.parameter_types[0] == "!pto.ptr<ui8, ub>"
    assert helper.parameter_indices[0] == (1 if reverse_captures else 0)
    assert "pto.castptr" in helper.source
    assert helper.source.count("pto.vmi.create_group_mask") == (2 if dynamic else 1)
    for name in ("vchist", "vdhist"):
        assert helper.source.count(f"pto.vmi.{name}") == 1
    assert "num_groups = 16, group_size = 16" in helper.source
    assert "!pto.vmi.vreg<64xui32>" in helper.source
    assert helper.source.count("pto.vmi.vinterpret_cast") == 2
    compiler = find_ptoas_binary()
    if compiler is None:
        pytest.skip("PTOAS is unavailable")
    source = tmp_path / "histogram.pto"
    llvm = tmp_path / "histogram.ll"
    source.write_text(helper.source)
    result = subprocess.run(
        [compiler, str(source), *PTOAS_FLAGS, "-o", str(llvm)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "llvm.hivm.chistv2" in llvm.read_text()
    assert "llvm.hivm.dhistv2" in llvm.read_text()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
