# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile-time regressions for GM allocation identity and layout contracts."""

import ast
import json
from pathlib import Path
from textwrap import dedent, indent

import pypto.language as pl
import pytest
from pypto import backend, ir


def _matmul(name: str = "mm", rows: int = 64) -> str:
    return f"""
@pl.function(type=pl.FunctionType.InCore)
def {name}(self, a: pl.Tensor[[{rows},64], pl.FP32], w: pl.Tensor[[64,64], pl.FP32],
           dst: pl.Out[pl.Tensor[[{rows},64], pl.FP32]]) -> pl.Tensor[[{rows},64], pl.FP32]:
    ta = pl.load(a, [0,0], [{rows},64], target_memory=pl.Mem.Mat)
    tw = pl.load(w, [0,0], [64,64], target_memory=pl.Mem.Mat)
    dst = pl.store(pl.matmul(ta, tw), [0,0], dst)
    return dst
"""


_ENTRY = """
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64,64], pl.FP32], w: pl.Tensor[[64,64], pl.FP32],
         dst: pl.Out[pl.Tensor[[64,64], pl.FP32]]) -> pl.Tensor[[64,64], pl.FP32]:
"""


def _compile(output: Path, functions: str) -> str:
    program = pl.parse("@pl.program\nclass Kernel:\n" + indent(functions, "    "), filename=__file__)
    assert isinstance(program, ir.Program)
    ir.compile(
        program,
        strategy=ir.OptimizationStrategy.AutoLayout,
        backend_type=backend.BackendType.Ascend910,
        skip_ptoas=True,
        dump_passes=ir.PassDumpLevel.EXPLICIT,
        output_dir=str(output),
    )
    return next((output / "passes_dump").glob("*_after_AutoLayout.py")).read_text()


@pytest.mark.parametrize("local", ["x", "fresh"])
def test_rebound_external_and_fresh_allocations(tmp_path, ascend_backend, local):
    # Repeated calls make promoting the external input profitable if ownership leaks.
    body = f"""\
    for i in pl.range(64):
        dst = self.mm(x, w, dst)
    {local} = pl.tensor.create([64,64], dtype=pl.FP32)
    {local} = self.mm(w, w, {local})
    dst = self.mm({local}, w, dst)
    return dst
"""
    dump = _compile(tmp_path, _matmul() + _ENTRY + body)
    meta = json.loads((tmp_path / "compiled_meta.json").read_text())
    external = next(p for p in meta["lowered_params"] if p["name"] == "x")
    assert (external["layout"], external["shape"]) == ("ND", [64, 64])
    assert "layout=pl.TensorLayout.NZ" in next(line for line in dump.splitlines() if "tensor.create(" in line)


@pytest.mark.parametrize("result", ["scratch", "produced"])
def test_buffer_returned_through_loop(tmp_path, ascend_backend, result):
    # The store returns the same Out buffer on every iteration.
    producer = """
@pl.function(type=pl.FunctionType.InCore)
def produce(self, x: pl.Tensor[[64,64], pl.FP16], w: pl.Tensor[[64,64], pl.FP16],
            mid: pl.Out[pl.Tensor[[64,64], pl.FP32]]) -> pl.Tensor[[64,64], pl.FP32]:
    for j in pl.range(4):
        tx = pl.load(x, [0,0], [64,64], target_memory=pl.Mem.Mat)
        tw = pl.load(w, [0,j*16], [64,16], target_memory=pl.Mem.Mat)
        mid = pl.store(pl.matmul(tx, tw), [0,j*16], mid)
    return mid

@pl.function(type=pl.FunctionType.InCore)
def consume(self, data: pl.Tensor[[64,64], pl.FP32],
            dst: pl.Out[pl.Tensor[[64,64], pl.FP32]]) -> pl.Tensor[[64,64], pl.FP32]:
    tile = pl.load(data, [0,0], [64,64], target_memory=pl.Mem.Vec)
    dst = pl.store(tile, [0,0], dst)
    return dst
"""
    entry = _ENTRY.replace("pl.FP32", "pl.FP16", 2)
    _compile(
        tmp_path,
        producer
        + entry
        + f"""\
    scratch = pl.tensor.create([64,64], dtype=pl.FP32)
    {result} = self.produce(x, w, scratch)
    dst = self.consume({result}, dst)
    return dst
""",
    )


def test_carried_alternative_keeps_external_contract(tmp_path, ascend_backend):
    dump = _compile(
        tmp_path,
        _matmul()
        + _ENTRY
        + """\
    temp = pl.tensor.create([64,64], dtype=pl.FP32)
    temp = self.mm(w, w, temp)
    for i, (current,) in pl.range(4, init_values=(temp,)):
        dst = self.mm(w, current, dst)
        final = pl.yield_(x)
    return dst
""",
    )
    # The carried tensor uses temp on iteration zero and external x thereafter.
    assert "TensorLayout.NZ" not in dump
    assert "TensorLayout.DN" not in dump


def test_opaque_kernel_pins_buffer_layout(tmp_path, ascend_backend):
    fixture = Path(__file__).resolve().parents[3] / "st/runtime/external_kernel/kernels/aiv/spmd_write.cpp"
    external = dedent(f"""
        @pl.function(type=pl.FunctionType.AIV, external_source={str(fixture)!r})
        def external_write(self, data: pl.InOut[pl.Tensor[[64,64], pl.FP32]],
                           base_cl: pl.Scalar[pl.INDEX]) -> pl.Tensor[[64,64], pl.FP32]: ...
    """)
    dump = _compile(
        tmp_path,
        _matmul()
        + external
        + _ENTRY
        + """\
    temp = pl.tensor.create([64,64], dtype=pl.FP32)
    temp = self.mm(x, w, temp)
    temp = self.external_write(temp, 0)
    dst = self.mm(temp, w, dst)
    return dst
""",
    )
    assert "TensorLayout.NZ" not in dump
    assert "TensorLayout.DN" not in dump


def test_tuple_positions_keep_their_returned_buffers(tmp_path, ascend_backend):
    # Return the two Out parameters in reverse order. Only the first tuple
    # element is the compiler-owned tensor consumed at the next matmul.
    pair = """
@pl.function(type=pl.FunctionType.InCore)
def pair(self, a: pl.Tensor[[64,64], pl.FP32], w: pl.Tensor[[64,64], pl.FP32],
         public: pl.Out[pl.Tensor[[64,64], pl.FP32]],
         internal: pl.Out[pl.Tensor[[64,64], pl.FP32]]) -> pl.Tuple[
             pl.Tensor[[64,64], pl.FP32], pl.Tensor[[64,64], pl.FP32]]:
    for i in pl.range(2):
        ta = pl.load(a, [0,0], [64,64], target_memory=pl.Mem.Mat)
        tw = pl.load(w, [0,0], [64,64], target_memory=pl.Mem.Mat)
        tile = pl.matmul(ta, tw)
        public = pl.store(tile, [0,0], public)
        internal = pl.store(tile, [0,0], internal)
    return internal, public
"""
    dump = _compile(
        tmp_path,
        _matmul()
        + pair
        + _ENTRY
        + """\
    temp = pl.tensor.create([64,64], dtype=pl.FP32)
    produced, public = self.pair(x, w, dst, temp)
    public = self.mm(produced, w, public)
    return public
""",
    )
    assert "layout=pl.TensorLayout.NZ" in next(line for line in dump.splitlines() if "tensor.create(" in line)
    meta = json.loads((tmp_path / "compiled_meta.json").read_text())
    assert all(p["layout"] == "ND" for p in meta["lowered_params"])


def test_mixed_nz_blocks_each_allocation_once(tmp_path, ascend_backend):
    # Only the read_existing helper uses the preblocked parameter. The mm chain
    # is free to promote its separate local allocation.
    existing = """
@pl.function(type=pl.FunctionType.InCore)
def read_existing(self, x: pl.Tensor[[64,64], pl.FP32], ext: pl.Tensor[[64,64], pl.FP32, pl.NZ],
                  dst: pl.Out[pl.Tensor[[64,64], pl.FP32]]) -> pl.Tensor[[64,64], pl.FP32]:
    a = pl.load(x, [0,0], [64,64], target_memory=pl.Mem.Mat)
    b = pl.load(ext, [0,16], [64,16], target_memory=pl.Mem.Mat)
    return pl.store(pl.matmul(a, b), [0,16], dst)
"""
    entry = _ENTRY.replace("def main(self, ", "def main(self, ext: pl.Tensor[[64,64], pl.FP32, pl.NZ], ")
    dump = _compile(
        tmp_path,
        _matmul()
        + existing
        + entry
        + """\
    dst = self.read_existing(x, ext, dst)
    mid = pl.tensor.create([64,64], dtype=pl.FP32)
    mid = self.mm(x, w, mid)
    return self.mm(mid, w, dst)
""",
    )
    assert "layout=pl.TensorLayout.NZ" in next(line for line in dump.splitlines() if "tensor.create(" in line)
    meta = json.loads((tmp_path / "compiled_meta.json").read_text())
    lowered = next(p for p in meta["lowered_params"] if p["name"] == "ext")
    assert (lowered["layout"], lowered["shape"]) == ("NZ", [1, 8, 4, 16, 8])
    final = sorted((tmp_path / "passes_dump").glob("*_after_BlockNzTensorViews.py"))[-1].read_text()
    offsets = [
        ast.literal_eval(call.args[1])
        for call in ast.walk(ast.parse(final))
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "load"
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id.startswith("ext__")
    ]
    assert offsets == [[0, 2, 0, 0, 0]]


def test_aligned_specialization_keeps_function_attr_in_formal_scope(tmp_path, ascend_backend):
    offset_kernel = """
@pl.function(type=pl.FunctionType.InCore)
def offset_mm(self, w: pl.Tensor[[128,64], pl.FP32], b: pl.Tensor[[64,64], pl.FP32],
              dst: pl.Out[pl.Tensor[[64,64], pl.FP32]],
              p: pl.Scalar[pl.INDEX]) -> pl.Tensor[[64,64], pl.FP32]:
    pl.func_attr({"source_offset": p})
    ta = pl.load(w, [p,0], [64,64], target_memory=pl.Mem.Mat)
    tb = pl.load(b, [0,0], [64,64], target_memory=pl.Mem.Mat)
    return pl.store(pl.matmul(ta, tb), [0,0], dst)
"""
    entry = _ENTRY.replace("x: pl.Tensor[[64,64]", "x: pl.Tensor[[128,64]")
    dump = _compile(
        tmp_path,
        _matmul("wide", 128)
        + offset_kernel
        + entry
        + """\
    mid = pl.tensor.create([128,64], dtype=pl.FP32)
    mid = self.wide(x, w, mid)
    dst = self.offset_mm(x, w, dst, 16)
    return self.offset_mm(mid, w, dst, 16)
""",
    )
    assert "p_aligned" in dump
    attrs = [line for line in dump.splitlines() if "source_offset" in line]
    assert len(attrs) == 2
    assert all("p_aligned" not in line for line in attrs)


@pytest.mark.parametrize("rows", [128, 256])
def test_slice_keeps_its_backing_vec_allocation_live(tmp_path, ascend_backend, rows):
    program = f"""
@pl.function(type=pl.FunctionType.InCore)
def write_view(self, x: pl.Tensor[[512,64], pl.FP32],
               mid: pl.Out[pl.Tensor[[{rows},128], pl.FP32]]) -> pl.Tensor[[{rows},128], pl.FP32]:
    backing = pl.load(x, [0,0], [512,64], target_memory=pl.Mem.Vec)
    view = pl.tile.slice(backing, [{rows},64], [0,0])
    return pl.store(view, [0,0], mid)

@pl.function(type=pl.FunctionType.InCore)
def copy(self, mid: pl.Tensor[[{rows},128], pl.FP32],
         out: pl.Out[pl.Tensor[[{rows},64], pl.FP32]]) -> pl.Tensor[[{rows},64], pl.FP32]:
    tile = pl.load(mid, [0,0], [{rows},64], target_memory=pl.Mem.Vec)
    return pl.store(tile, [0,0], out)

@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[512,64], pl.FP32],
         out: pl.Out[pl.Tensor[[{rows},64], pl.FP32]]) -> pl.Tensor[[{rows},64], pl.FP32]:
    mid = pl.tensor.create([{rows},128], dtype=pl.FP32)
    mid = self.write_view(x, mid)
    return self.copy(mid, out)
"""
    # Either view fits on its own. Its destination and scratch coexist with
    # the original 128 KiB backing on a target with 184 KiB of Vec.
    _compile(tmp_path, program)
