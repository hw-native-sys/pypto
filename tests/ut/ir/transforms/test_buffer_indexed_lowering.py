# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Whole-kernel indexed conversion and preservation through all memory planners."""

import pypto.language as pl
import pytest
import torch
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.debug import torch_codegen
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen

pytestmark = pytest.mark.usefixtures("ascend_backend")
_PLANNERS = (passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS)


def _lower(program, planner, enabled=None):
    options = {} if enabled is None else {"enable_buffer_ir": enabled}
    with passes.PassContext([], memory_planner=planner, **options):
        return PassManager(OptimizationStrategy.Default).run_passes(program)


def _kernel(program: ir.Program) -> ir.Function:
    kernel = program.get_function("kernel")
    assert kernel is not None
    return kernel


def _gather_program(dtype="FP32", valid=(4, 32), scratch_cols=32):
    return pl.parse_program(f"""
@pl.program
class Gather:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[8, 64], pl.{dtype}], idx: pl.Tensor[[4, 32], pl.INT32],
               out: pl.InOut[pl.Tensor[[12, 64], pl.{dtype}]]) -> pl.Tensor[[12, 64], pl.{dtype}]:
        source = pl.load(a, [0, 0], [8, 64])
        indices = pl.load(idx, [0, 0], [4, 32], valid_shape=[{valid[0]}, {valid[1]}])
        scratch = pl.create_tile([4, {scratch_cols}], dtype=pl.INT32)
        gathered = pl.tile.gather(source, indices, scratch)
        out = pl.store(gathered, [0, 0], out)
        return pl.store(source, [4, 0], out)
""")


def _scatter_program(valid=(4, 32), base="zero", preserve=True):
    initialization = (
        "pl.tile.full([8, 64], pl.FP32, 0.0)" if base == "zero" else "pl.load(a, [0, 0], [8, 64])"
    )
    suffix = (
        """
        ones = pl.tile.full([4, 32], pl.FP32, 1.0)
        active_ones = pl.tile.set_validshape(ones, ROWS, COLS)
        mask_zero = pl.tile.full([8, 64], pl.FP32, 0.0)
        written = pl.tile.scatter(mask_zero, active_ones, indices)
        predicate = pl.tile.cmps(written, 0.0, cmp_type=1)
        scratch = pl.create_tile([1, 16], dtype=pl.UINT32)
        result = pl.tile.sel(predicate, scattered, old, scratch)
""".replace("ROWS", str(valid[0])).replace("COLS", str(valid[1]))
        if preserve
        else "\n        result = scattered\n"
    )
    return pl.parse_program(f"""
@pl.program
class Scatter:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[8, 64], pl.FP32], values: pl.Tensor[[4, 32], pl.FP32],
               idx: pl.Tensor[[4, 32], pl.INT32], out: pl.InOut[pl.Tensor[[20, 64], pl.FP32]]
               ) -> pl.Tensor[[20, 64], pl.FP32]:
        old = pl.load(a, [0, 0], [8, 64])
        source = pl.load(values, [0, 0], [4, 32], valid_shape=[{valid[0]}, {valid[1]}])
        indices = pl.load(idx, [0, 0], [4, 32], valid_shape=[{valid[0]}, {valid[1]}])
        zero = {initialization}
        scattered = pl.tile.scatter(zero, source, indices)
{suffix}
        out = pl.store(result, [0, 0], out)
        out = pl.store(old, [8, 0], out)
        return pl.store(source, [16, 0], out)
""")


def _native(program, tmp_path, planner):
    restored = ir.deserialize(ir.serialize(program))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(program, restored, enable_auto_mapping=True)
    text = codegen.PTOCodegen().generate(restored, emit_tile_addr=False, emit_source_loc=False)
    if find_ptoas_binary() is not None:
        source = tmp_path / "indexed.pto"
        source.write_text(text)
        level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
        _run_ptoas(str(source), str(tmp_path / "indexed.cpp"), ["--pto-arch=a2", f"--pto-level={level}"])
    return text


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("dtype", ["FP16", "FP32", "INT32"])
def test_gather_default_roundtrip_native_and_live_table(tmp_path, planner, dtype):
    program = _lower(_gather_program(dtype), planner)
    assert _kernel(program).ir_stage == ir.FunctionIRStage.Buffer
    assert "pto.tgather" in _native(program, tmp_path, planner)
    torch_dtype = {"FP16": torch.float16, "FP32": torch.float32, "INT32": torch.int32}[dtype]
    source = torch.arange(512).reshape(8, 64).to(torch_dtype)
    indices = ((torch.arange(128).reshape(4, 32) * 37) % 512).to(torch.int32)
    out = torch.full((12, 64), -777, dtype=torch_dtype)
    expected = out.clone()
    expected[:4, :32] = source.reshape(-1)[indices.long()]
    expected[4:] = source
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    namespace["kernel"](source, indices, out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("valid", [(4, 32), (3, 29)])
def test_scatter_default_preserves_sentinels_and_last_duplicate(tmp_path, planner, valid):
    program = _lower(_scatter_program(valid), planner)
    assert _kernel(program).ir_stage == ir.FunctionIRStage.Buffer
    assert "pto.tscatter" in _native(program, tmp_path, planner)
    original = -torch.arange(1, 513).reshape(8, 64).float()
    values = torch.arange(1, 129).reshape(4, 32).float()
    indices = ((torch.arange(128).reshape(4, 32) // 2 * 7) % 512).to(torch.int32)
    out = torch.full((20, 64), -777.0)
    expected = out.clone()
    expected[:8] = original
    expected[8:16] = original
    expected[16 : 16 + valid[0], : valid[1]] = values[: valid[0], : valid[1]]
    for r in range(valid[0]):
        for c in range(valid[1]):
            expected[:8].view(-1)[indices[r, c]] = values[r, c]
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    namespace["kernel"](original, values, indices, out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("kind", ["partial_gather", "scratch", "nonzero_scatter", "off"])
def test_indexed_rollout_keeps_unproven_forms_on_tile(planner, kind):
    if kind == "nonzero_scatter":
        program = _scatter_program(base="loaded", preserve=False)
    else:
        program = _gather_program(
            valid=(3, 29) if kind == "partial_gather" else (4, 32),
            scratch_cols=64 if kind == "scratch" else 32,
        )
    lowered = _lower(program, planner, False if kind == "off" else None)
    assert _kernel(lowered).ir_stage == ir.FunctionIRStage.Functional


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("planner", _PLANNERS)
def test_indexed_rollout_keeps_a5_on_tile(ascend_backend, planner):
    assert ascend_backend == BackendType.Ascend950
    assert _kernel(_lower(_gather_program(), planner)).ir_stage == ir.FunctionIRStage.Functional


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("alias", ["old_read", "view", "outside_loop"])
def test_scatter_zero_base_must_have_no_alias_or_loop_reuse(planner, alias):
    # Reading the zero base again, exporting a view, or reusing a fill across
    # a backedge invalidates the proof that native zero-fill preserves it.
    zero = "zero = pl.tile.full([8, 64], pl.FP32, 0.0)"
    operation = "scattered = pl.tile.scatter(zero, source, indices)"
    tail = "out = pl.store(zero, [8, 0], out)" if alias == "old_read" else ""
    if alias == "view":
        operation = (
            "alias = pl.tile.reshape(zero, [8, 64])\n"
            "        scattered = pl.tile.scatter(alias, source, indices)"
        )
        tail = "out = pl.store(zero, [8, 0], out)"
    if alias == "outside_loop":
        operation = (
            "for i in pl.range(2):\n"
            "            scattered = pl.tile.scatter(zero, source, indices)\n"
            "            out = pl.store(scattered, [0, 0], out)"
        )
        tail = "return out"
    else:
        tail += "\n        return pl.store(scattered, [0, 0], out)"
    program = pl.parse_program(f"""
@pl.program
class ZeroBase:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, values: pl.Tensor[[4, 32], pl.FP32], idx: pl.Tensor[[4, 32], pl.INT32],
               out: pl.InOut[pl.Tensor[[16, 64], pl.FP32]]) -> pl.Tensor[[16, 64], pl.FP32]:
        source = pl.load(values, [0, 0], [4, 32])
        indices = pl.load(idx, [0, 0], [4, 32])
        {zero}
        {operation}
        {tail}
""")
    assert _kernel(_lower(program, planner)).ir_stage == ir.FunctionIRStage.Functional


@pytest.mark.parametrize("planner", _PLANNERS)
def test_tensor_scatter_preservation_chain_is_selected(tmp_path, planner):
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            base: pl.Tensor[[1, 64], pl.FP32],
            index: pl.Tensor[[1, 32], pl.INT32],
            values: pl.Tensor[[1, 32], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ) -> pl.Tensor[[1, 64], pl.FP32]:
            result = pl.tensor.scatter(base, dim=-1, index=index, src=values)
            return pl.assemble(out, result, [0, 0])

    program = _lower(Before, planner)
    assert _kernel(program).ir_stage == ir.FunctionIRStage.Buffer
    text = _native(program, tmp_path, planner)
    assert "pto.tscatter" in text and "pto.tsel" in text
    base = -torch.arange(1, 65).reshape(1, 64).float()
    index = (torch.arange(32).reshape(1, 32) // 2 * 3).to(torch.int32)
    values = torch.arange(1, 33).reshape(1, 32).float()
    expected = base.clone()
    for col in range(32):
        expected[0, index[0, col]] = values[0, col]
    out = torch.empty_like(base)
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    namespace["kernel"](base, index, values, out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["gather", "scatter"])
@pytest.mark.parametrize("bad_index", [-1, 512])
def test_indexed_replay_checks_bounds_before_writing_output(kind, bad_index):
    logical = _gather_program() if kind == "gather" else _scatter_program(preserve=False)
    program = _lower(logical, passes.MemoryPlanner.PYPTO)
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    table = torch.ones((8, 64))
    indices = torch.zeros((4, 32), dtype=torch.int32)
    indices[0, 0] = bad_index
    out = torch.full((12 if kind == "gather" else 20, 64), -777.0)
    args = (table, indices, out) if kind == "gather" else (table, torch.ones((4, 32)), indices, out)
    with pytest.raises(ValueError, match=f"buffer.{kind} index is outside"):
        namespace["kernel"](*args)
    assert torch.all(out == -777)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
