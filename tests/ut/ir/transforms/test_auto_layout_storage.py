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
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import generate
from pypto.pypto_core import passes


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


def _compile(
    output: Path,
    functions: str,
    strategy: ir.OptimizationStrategy = ir.OptimizationStrategy.AutoLayout,
    *,
    input_layout_policy: ir.InputLayoutPolicy | None = None,
) -> str:
    program = pl.parse("@pl.program\nclass Kernel:\n" + indent(functions, "    "), filename=__file__)
    assert isinstance(program, ir.Program)
    ir.compile(
        program,
        strategy=strategy,
        input_layout_policy=input_layout_policy,
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


@pytest.mark.parametrize("layout", ["NZ", "DN"])
def test_reselect_private_intermediate_preserves_caller_state(tmp_path, ascend_backend, layout):
    plain = "pl.Tensor[[64,64], pl.FP32]"
    view = "pl.NZ" if layout == "NZ" else "pl.TensorView(stride=[1,64], layout=pl.TensorLayout.DN)"
    intermediate = f"pl.Tensor[[64,64], pl.FP32, {view}]"
    # The helper's Out formal is bound to private storage. The entry's InOut
    # formal is the caller's allocation throughout the complete call chain.
    source = f"""
@pl.function(type=pl.FunctionType.InCore)
def produce(self, x: {plain}, scratch: pl.Out[{intermediate}]) -> {intermediate}:
    tile = pl.load(x, [0,0], [64,64], target_memory=pl.Mem.Vec)
    return pl.store(tile, [0,0], scratch)
@pl.function(type=pl.FunctionType.InCore)
def consume(self, scratch: {intermediate}, state: pl.InOut[{plain}]) -> {plain}:
    tile = pl.load(scratch, [0,0], [64,64], target_memory=pl.Mem.Vec)
    old = pl.load(state, [0,0], [64,64], target_memory=pl.Mem.Vec)
    return pl.store(pl.tile.add(tile, old), [0,0], state)
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: {plain}, state: pl.InOut[{plain}]) -> {plain}:
    scratch = pl.tensor.create([64,64], dtype=pl.FP32, layout=pl.TensorLayout.{layout})
    alias = self.produce(x, scratch)
    return self.consume(alias, state)
"""
    dump = _compile(tmp_path, source, ir.OptimizationStrategy.AutoLayoutIntermediates)
    allocation = next(line for line in dump.splitlines() if "tensor.create(" in line)
    assert "layout=pl.TensorLayout.ND" in allocation
    meta = json.loads((tmp_path / "compiled_meta.json").read_text())
    assert meta["params"] == meta["lowered_params"]
    assert [(p["name"], p["direction"], p["layout"]) for p in meta["params"]] == [
        ("x", "In", "ND"),
        ("state", "InOut", "ND"),
    ]


def test_returned_allocation_keeps_entry_layout(tmp_path, ascend_backend):
    packed = "pl.Tensor[[64,64], pl.FP32, pl.NZ]"
    source = f"""
@pl.function(type=pl.FunctionType.InCore)
def fill(self, x: pl.Tensor[[64,64], pl.FP32], dst: pl.Out[{packed}]) -> {packed}:
    lhs = pl.load(x, [0,0], [64,64], target_memory=pl.Mem.Mat)
    rhs = pl.load(x, [0,0], [64,64], target_memory=pl.Mem.Mat)
    return pl.store(pl.matmul(lhs, rhs), [0,0], dst)
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64,64], pl.FP32]) -> {packed}:
    scratch = pl.tensor.create([64,64], dtype=pl.FP32, layout=pl.TensorLayout.NZ)
    final = self.fill(x, scratch)
    return final
"""
    dump = _compile(tmp_path, source, ir.OptimizationStrategy.AutoLayoutIntermediates)
    entry = next(line for line in dump.splitlines() if "def main(" in line)
    assert "layout=pl.TensorLayout.NZ, nz_blocked=True" in entry.split(" -> ")[1]
    scratch = next(line for line in dump.splitlines() if line.strip().startswith("scratch__ssa_v0:"))
    assert "layout=pl.TensorLayout.NZ, nz_blocked=True" in scratch


@pytest.mark.parametrize("allowed", ["x", "w"])
def test_only_named_entry_input_can_be_packed(tmp_path, ascend_backend, allowed):
    # The entry's w feeds helper a, and x feeds helper w. Reuse the site in a
    # loop so packing either operand pays off if the loader can supply it.
    body = """
    alias = x
    for i in pl.range(64):
        dst = self.mm(w, alias, dst)
    return dst
"""
    _compile(
        tmp_path,
        _matmul() + _ENTRY + body,
        input_layout_policy=ir.InputLayoutPolicy((allowed,), ("x", "w"), 100),
    )
    params = json.loads((tmp_path / "compiled_meta.json").read_text())["lowered_params"]
    assert {p["name"]: p["layout"] for p in params} == {
        "x": "NZ" if allowed == "x" else "ND",
        "w": "NZ" if allowed == "w" else "ND",
        "dst": "ND",
    }


@pytest.mark.parametrize("name", ["a", "dst"])
def test_loader_permission_requires_read_only_entry_input(tmp_path, ascend_backend, name):
    with pytest.raises(ValueError, match="NZ (entry )?input"):
        _compile(
            tmp_path,
            _matmul() + _ENTRY + "    return self.mm(x, w, dst)\n",
            input_layout_policy=ir.InputLayoutPolicy((name,)),
        )


def test_steady_state_requires_per_forward_horizon():
    with pytest.raises(ValueError, match="steady_state requires expected_uses=1"):
        ir.InputLayoutPolicy(("w",), ("w",), expected_uses=2, steady_state=True)
    with pytest.raises(ValueError, match="steady_state requires expected_uses=1"):
        passes.auto_layout(nz_inputs=["w"], reusable_tensors=["w"], expected_uses=2, steady_state=True)


@pytest.mark.parametrize(("steady_state", "layout"), [(False, "ND"), (True, "NZ")])
def test_steady_state_waives_only_reusable_input_packing(tmp_path, ascend_backend, steady_state, layout):
    _compile(
        tmp_path,
        _matmul() + _ENTRY + "    return self.mm(x, w, dst)\n",
        input_layout_policy=ir.InputLayoutPolicy(("x", "w"), ("w",), steady_state=steady_state),
    )
    meta = json.loads((tmp_path / "compiled_meta.json").read_text())
    assert {p["name"]: p["layout"] for p in meta["params"]} == {"x": "ND", "w": "ND", "dst": "ND"}
    assert {p["name"]: p["layout"] for p in meta["lowered_params"]} == {"x": "ND", "w": layout, "dst": "ND"}


_HOST_CHIP = """
@pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
def chip(self, a: pl.Tensor[[64,64], pl.FP32], b: pl.Tensor[[64,64], pl.FP32],
         dst: pl.Out[pl.Tensor[[64,64], pl.FP32]]) -> pl.Tensor[[64,64], pl.FP32]:
    for i in pl.range(3):
        dst = self.mm(a, b, dst)
    return dst
"""

_HOST_ENTRY = """
@pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
def main(self, x: pl.Tensor[[2,64,64], pl.FP32], w: pl.Tensor[[2,64,64], pl.FP32],
         dst: pl.Out[pl.Tensor[[2,64,64], pl.FP32]]):
    for r in pl.range(2):
        shard = x[r]
        self.chip(w[r], shard, dst[r], device=r)
"""


def _host_program(functions: str) -> ir.Program:
    program = pl.parse("@pl.program\nclass Kernel:\n" + indent(functions, "    "), filename=__file__)
    assert isinstance(program, ir.Program)
    return program


def _lower_host(output: Path, program: ir.Program, policy: ir.InputLayoutPolicy):
    dumps = output / "passes_dump"
    lowered = ir.PassManager.get_strategy(
        ir.OptimizationStrategy.AutoLayout, input_layout_policy=policy
    ).run_passes(program, dump_ir=ir.PassDumpLevel.EXPLICIT, output_dir=str(dumps))
    return (
        lowered,
        next(dumps.glob("*_after_InferTileMemorySpace.py")).read_text(),
        next(dumps.glob("*_after_AutoLayout.py")).read_text(),
    )


def _chip_has_nz(dump: str, name: str = "chip") -> tuple[bool, bool, bool]:
    chip = next(
        node for node in ast.walk(ast.parse(dump)) if isinstance(node, ast.FunctionDef) and node.name == name
    )
    a, b, dst = chip.args.args[1:]

    def has_nz(param: ast.arg) -> bool:
        return param.annotation is not None and "TensorLayout.NZ" in ast.unparse(param.annotation)

    return (has_nz(a), has_nz(b), has_nz(dst))


def _host_layout_proof(program: ir.Program) -> str:
    host = program.get_function("main")
    assert host is not None
    proof = host.attrs["auto_layout_host_inputs"]
    assert isinstance(proof, str)
    return proof


@pytest.mark.parametrize(("uses", "layout"), [(1, "ND"), (100, "NZ")])
def test_whole_host_weight_crosses_implicit_single_chip_dispatch(tmp_path, ascend_backend, uses, layout):
    entry = """
@pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
def main(self, x: pl.Tensor[[64,64], pl.FP32], w: pl.Tensor[[64,64], pl.FP32],
         dst: pl.Tensor[[64,64], pl.FP32]):
    weight = w
    self.chip(x, weight, dst)
"""
    compiled = ir.compile(
        _host_program(_matmul() + _HOST_CHIP + entry),
        strategy=ir.OptimizationStrategy.AutoLayout,
        input_layout_policy=ir.InputLayoutPolicy(("w",), ("w",), uses),
        output_dir=str(tmp_path),
        skip_ptoas=True,
        distributed_config=ir.DistributedConfig(device_ids=[5], num_sub_workers=0),
    )
    assert isinstance(compiled, ir.DistributedCompiledProgram)
    descriptor = compiled.input_layouts[1]
    assert descriptor.layout == layout
    assert descriptor.logical_shape == (64, 64)
    metadata = json.loads((tmp_path / "distributed_meta.json").read_text())
    assert [param["direction"] for param in metadata["params"]] == ["In", "In", "Out"]
    # Device affinity is logical rank zero, independently of the sole physical
    # device ID, and is needed only for the selected NZ crossing.
    orchestrator = (tmp_path / "orchestration/host_orch.py").read_text()
    dispatch = next(
        line for line in orchestrator.splitlines() if '_submit_chip(orch, callables["chip"]' in line
    )
    assert dispatch.endswith(", 0)" if layout == "NZ" else ", None)")


@pytest.mark.parametrize("allowed", ["x", "w"])
@pytest.mark.parametrize(("uses", "steady_state"), [(1, False), (100, False), (1, True)])
def test_named_host_input_layout(tmp_path, ascend_backend, allowed, uses, steady_state):
    entry = _HOST_ENTRY.replace("    for r", "    bank = x\n    for r").replace(
        "shard = x[r]", "shard = bank[r]"
    )
    lowered, before, after = _lower_host(
        tmp_path,
        _host_program(_matmul() + _HOST_CHIP + entry),
        ir.InputLayoutPolicy((allowed,), ("x", "w"), uses, steady_state=steady_state),
    )
    # HOST dispatch coordinates and types must stay in the logical ABI even
    # when the device interprets one input's bytes in NZ order.
    marker = "@pl.function(level=pl.Level.HOST"
    before_host = [line for line in before.split(marker)[1].splitlines() if "pl.func_attr" not in line]
    after_host = [line for line in after.split(marker)[1].splitlines() if "pl.func_attr" not in line]
    assert before_host == after_host
    packed = uses > 1 or steady_state
    expected = (allowed == "w" and packed, allowed == "x" and packed, False)
    assert _chip_has_nz(after) == expected
    slot = 0 if allowed == "x" else 1
    assert _host_layout_proof(lowered) == f"v1;{slot}={'NZ' if packed else 'ND'}"
    if find_ptoas_binary() is not None:
        files = generate(lowered, str(tmp_path), skip_ptoas=False)
        assert any(path.startswith("next_levels/chip/kernels/") and path.endswith(".cpp") for path in files)
        pto = (tmp_path / "next_levels/chip/ptoas/mm.pto").read_text()
        for i, is_nz in enumerate(expected):
            view = next(line for line in pto.splitlines() if f"pto.make_tensor_view %arg{i}," in line)
            layout = "nz" if is_nz else "nd"
            assert f"layout = #pto.layout<{layout}>" in view


@pytest.mark.parametrize("forged_proof", [False, True])
def test_distributed_named_input_guard_preserves_artifacts(tmp_path, ascend_backend, forged_proof):
    entry = _HOST_ENTRY.replace("x: pl.Tensor[[2,64,64]", "x: pl.Tensor[[2,128,64]")
    entry = entry.replace("shard = x[r]", "shard = x[r, :64, :]")
    program = _host_program(_matmul() + _HOST_CHIP + entry)
    compiled = ir.compile(program, output_dir=str(tmp_path), skip_ptoas=True)
    assert isinstance(compiled, ir.DistributedCompiledProgram)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert (tmp_path / "distributed_meta.json") in before
    if forged_proof:
        entry = entry.replace(
            "    for r", '    pl.func_attr({"auto_layout_host_inputs": "v1;0=NZ"})\n    for r'
        )
        program = _host_program(_matmul() + _HOST_CHIP + entry)
    with pytest.raises(ValueError, match="Distributed input layout"):
        ir.compile(
            program,
            strategy=ir.OptimizationStrategy.AutoLayout,
            input_layout_policy=ir.InputLayoutPolicy(("x",), ("x",), 100),
            output_dir=str(tmp_path),
            skip_ptoas=True,
        )
    assert before == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    reloaded = ir.DistributedCompiledProgram.from_dir(str(tmp_path))
    assert reloaded.execution_capabilities == compiled.execution_capabilities


@pytest.mark.parametrize("uses", [1, 100])
def test_host_permission_reopens_one_authored_nz_operand(tmp_path, ascend_backend, uses):
    dense = "pl.Tensor[[64,64], pl.FP32]"
    nz = "pl.Tensor[[64,64], pl.FP32, pl.NZ]"
    chip = _HOST_CHIP.replace(f"a: {dense}", f"a: {nz}").replace(f"b: {dense}", f"b: {nz}")
    mm = _matmul().replace(f"a: {dense}", f"a: {nz}").replace(f"w: {dense}", f"w: {nz}")
    _, before, after = _lower_host(
        tmp_path, _host_program(mm + chip + _HOST_ENTRY), ir.InputLayoutPolicy(("x",), ("x",), uses)
    )
    assert _chip_has_nz(before) == (True, True, False)
    assert _chip_has_nz(after) == (True, uses > 1, False)


@pytest.mark.parametrize(
    "restriction",
    [
        "partial_matrix",
        "rank_window",
        "second_slice",
        "opaque_use",
        "write_alias",
        "bank_dispatch",
        "reshape",
    ],
)
def test_host_permission_requires_compatible_readers(tmp_path, ascend_backend, restriction):
    auxiliary = ""
    chip = _HOST_CHIP
    entry = _HOST_ENTRY
    if restriction == "partial_matrix":
        entry = entry.replace("x: pl.Tensor[[2,64,64]", "x: pl.Tensor[[2,128,64]")
        entry = entry.replace("shard = x[r]", "shard = x[r, :64, :]")
    elif restriction == "rank_window":
        entry = entry.replace("shard = x[r]", "window = x[0:1]\n        shard = window[0]")
    elif restriction == "second_slice":
        entry = entry.replace("shard = x[r]", "whole = x[r]\n        shard = whole[0:64, 0:64]")
    elif restriction in {"opaque_use", "reshape"}:
        fixtures = Path(__file__).resolve().parents[3] / "st/runtime/external_kernel/kernels/aiv"
        fixture = fixtures / "spmd_write.cpp"
        shape = "64,64" if restriction == "opaque_use" else "4096"
        actual = "b" if restriction == "opaque_use" else "pl.tensor.reshape(b, [4096])"
        auxiliary = f"""
@pl.function(type=pl.FunctionType.AIV, external_source={str(fixture)!r})
def inspect(self, weight: pl.Tensor[[{shape}], pl.FP32]): ...
"""
        chip = chip.replace("    for i", f"    self.inspect({actual})\n    for i")
    elif restriction == "write_alias":
        auxiliary = """
@pl.function(type=pl.FunctionType.InCore)
def overwrite(self, weight: pl.InOut[pl.Tensor[[64,64], pl.FP32]]) -> pl.Tensor[[64,64], pl.FP32]:
    tile = pl.tile.full([64,64], dtype=pl.FP32, value=1.0)
    return pl.store(tile, [0,0], weight)
"""
        chip = chip.replace("    for i", "    alias = b\n    b = self.overwrite(alias)\n    for i")
    else:
        auxiliary = """
@pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
def inspect_bank(self, bank: pl.Tensor[[2,64,64], pl.FP32]):
    pass
"""
        entry += "    self.inspect_bank(x, device=0)\n"

    lowered, _, after = _lower_host(
        tmp_path,
        _host_program(_matmul() + auxiliary + chip + entry),
        ir.InputLayoutPolicy(("x",), ("x",), 100),
    )
    assert _chip_has_nz(after) == (False, False, False)
    assert _host_layout_proof(lowered) == "v1;0=UNSUPPORTED"


@pytest.mark.parametrize("devices", [1, 3])
def test_offered_rank_extent_must_match_device_count(tmp_path, ascend_backend, devices):
    program = _host_program(_matmul() + _HOST_CHIP + _HOST_ENTRY)
    with pytest.raises(ValueError, match="Distributed input layout for 'x' is unsupported"):
        ir.compile(
            program,
            strategy=ir.OptimizationStrategy.AutoLayout,
            input_layout_policy=ir.InputLayoutPolicy(("x",), ("x",), steady_state=True),
            output_dir=str(tmp_path),
            skip_ptoas=True,
            distributed_config=ir.DistributedConfig(device_ids=list(range(devices)), num_sub_workers=0),
        )
    assert not (tmp_path / "distributed_meta.json").exists()


def test_host_policy_rejects_packed_fp4_bank(tmp_path, ascend_backend):
    program = _host_program("""
@pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
def main(self, weight: pl.Tensor[[2,64,64], pl.FP4E2M1X2]):
    pass
""")
    with pytest.raises(ValueError, match="cannot select a layout for a packed FP4 bank"):
        _lower_host(tmp_path, program, ir.InputLayoutPolicy(("weight",)))


@pytest.mark.parametrize("uses", [1, 100])
def test_host_named_input_layout_with_rank_and_layer_axes(tmp_path, ascend_backend, uses):
    dense = "pl.Tensor[[64,64], pl.FP32]"
    nz = "pl.Tensor[[64,64], pl.FP32, pl.NZ]"
    layer_bank = "pl.Tensor[[3,64,64], pl.FP32, pl.NZ]"
    chip = _HOST_CHIP.replace(f"a: {dense}", f"a: {layer_bank}").replace(f"b: {dense}", f"b: {layer_bank}")
    chip = chip.replace("self.mm(a, b, dst)", "self.mm(a[i], b[i], dst)")
    mm = _matmul().replace(f"a: {dense}", f"a: {nz}").replace(f"w: {dense}", f"w: {nz}")
    entry = _HOST_ENTRY.replace("x: pl.Tensor[[2,64,64]", "x: pl.Tensor[[2,3,64,64]")
    entry = entry.replace("w: pl.Tensor[[2,64,64]", "w: pl.Tensor[[2,3,64,64]")
    program = _host_program(mm + chip + entry)
    policy = ir.InputLayoutPolicy(("x",), ("x",), uses)
    lowered, _, after = _lower_host(tmp_path, program, policy)
    assert _chip_has_nz(after) == (True, uses > 1, False)
    restored = ir.deserialize(ir.serialize(lowered))
    assert isinstance(restored, ir.Program)
    selected = "NZ" if uses > 1 else "ND"
    assert _host_layout_proof(restored) == f"v1;0={selected}"
    artifact_dir = tmp_path / "compiled"
    compiled = ir.compile(
        program,
        strategy=ir.OptimizationStrategy.AutoLayout,
        input_layout_policy=policy,
        output_dir=str(artifact_dir),
        skip_ptoas=True,
        distributed_config=ir.DistributedConfig(device_ids=[0, 1], num_sub_workers=0),
    )
    assert isinstance(compiled, ir.DistributedCompiledProgram)
    assert set(compiled.input_layouts) == {0}
    descriptor = compiled.input_layouts[0]
    assert descriptor.layout == selected
    assert descriptor.logical_shape == (2, 3, 64, 64)
    assert descriptor.physical_shape == ((6, 8, 4, 16, 8) if selected == "NZ" else (2, 3, 64, 64))
    reloaded = ir.DistributedCompiledProgram.from_dir(str(artifact_dir))
    assert reloaded.input_layouts == compiled.input_layouts


def test_host_layout_proof_rejects_conflicting_authored_dispatches(tmp_path, ascend_backend):
    dense = "pl.Tensor[[64,64], pl.FP32]"
    nz = "pl.Tensor[[64,64], pl.FP32, pl.NZ]"
    chip_nz = _HOST_CHIP.replace("def chip(", "def chip_nz(").replace("self.mm(", "self.mm_nz(")
    chip_nz = chip_nz.replace(f"b: {dense}", f"b: {nz}")
    matmul_nz = _matmul("mm_nz").replace(f"w: {dense}", f"w: {nz}")
    out = "pl.Out[pl.Tensor[[2,64,64], pl.FP32]]"
    entry = _HOST_ENTRY.replace(f"dst: {out}", f"dst: {out}, dst_nz: {out}")
    entry += "        self.chip_nz(w[r], shard, dst_nz[r], device=r)\n"
    program = _host_program(_matmul() + matmul_nz + _HOST_CHIP + chip_nz + entry)
    policy = ir.InputLayoutPolicy(("w", "x"), ("w", "x"), 100)
    lowered, _, after = _lower_host(tmp_path, program, policy)
    assert _host_layout_proof(lowered) == "v1;0=UNSUPPORTED;1=NZ"
    assert _chip_has_nz(after) == (True, False, False)
    assert _chip_has_nz(after, "chip_nz") == (True, True, False)
    artifact = tmp_path / "compiled"
    with pytest.raises(ValueError, match="Distributed input layout for 'x' is unsupported"):
        ir.compile(
            program,
            strategy=ir.OptimizationStrategy.AutoLayout,
            input_layout_policy=policy,
            output_dir=str(artifact),
            skip_ptoas=True,
        )
    assert not (artifact / "distributed_meta.json").exists()
