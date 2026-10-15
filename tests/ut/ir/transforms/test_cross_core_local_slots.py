# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end capacity and backend contracts for automatic local pipe depth."""

from pathlib import Path

import pytest
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.language.parser.text_parser import parse_program


@pytest.fixture(autouse=True)
def _reset_backend():
    backend.reset_for_testing()
    yield
    backend.reset_for_testing()


def _program(slot_num: int, local_slot_num: int | None = None) -> ir.Program:
    local = "" if local_slot_num is None else f", local_slot_num={local_slot_num}"
    return parse_program(
        "@pl.program\nclass Ring:\n"
        "    @pl.function(type=pl.FunctionType.Orchestration)\n"
        "    def main(self, a: pl.Tensor[[256, 128], pl.BF16], "
        "b: pl.Tensor[[128, 256], pl.BF16], "
        "out: pl.Out[pl.Tensor[[256, 256], pl.FP32]]) -> pl.Tensor[[256, 256], pl.FP32]:\n"
        f"        for blk in pl.spmd(4, optimizations=[pl.cross_core_slot(slot_num={slot_num}{local})]):\n"
        "            r0 = blk * 64\n"
        "            acc = pl.matmul(a[r0:r0+64, 0:128], b[0:128, 0:256], out_dtype=pl.FP32)\n"
        "            out[r0:r0+64, 0:256] = pl.add(acc, 1.0)\n"
        "        return out\n"
    )


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_local_depth_unblocks_capacity_without_shrinking_gm_ring(tmp_path: Path, depth: int):
    """The same tile fits with local depth 1 while its deeper transport ring remains."""
    with pytest.raises(ValueError, match="Vec buffer usage"):
        ir.compile(_program(depth), output_dir=str(tmp_path / "full"), skip_ptoas=True)
    output = tmp_path / "local"
    ir.compile(_program(depth, 1), output_dir=str(output), skip_ptoas=True, dump_passes=True)
    pto = "\n".join(p.read_text() for p in (output / "kernels").glob("*.pto"))
    assert pto.count(f"slot_num = {depth}, local_slot_num = 1") == 2
    assert 'name = "main_incore_0_c2v_slot_buffer", size = 65536' in pto
    # The injected IR tensor is a placeholder. Orchestration codegen sizes it
    # in float32 elements per block, multiplying by the four SPMD blocks.
    orchestration = (output / "orchestration" / "main.cpp").read_text()
    assert f"static_cast<uint32_t>(({65536 * depth // 4}) * (4))" in orchestration


@pytest.mark.parametrize("local_depth", [1, 4])
def test_a5_rejects_explicit_local_depth(tmp_path: Path, local_depth: int):
    """Even equal depths cannot emit this A2/A3-only PTOAS attribute on A5."""
    with pytest.raises(ValueError, match="local_slot_num.*only supported on A2/A3"):
        ir.compile(
            _program(4, local_depth),
            output_dir=str(tmp_path),
            backend_type=BackendType.Ascend950,
            skip_ptoas=True,
        )


@pytest.mark.parametrize("local_depth", [1, 2, 4])
def test_local_depth_covers_retained_pops_on_bidirectional_pipe(tmp_path: Path, local_depth: int):
    """A deep GM ring cannot prevent two live Mat pops aliasing a one-slot local FIFO."""
    before = parse_program(
        "@pl.program\nclass Retained:\n"
        "    @pl.function(type=pl.FunctionType.InCore)\n"
        "    def main(self, a: pl.Tensor[[32, 32], pl.FP16], b: pl.Tensor[[32, 32], pl.FP16], "
        "out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]) -> pl.Tensor[[32, 32], pl.FP32]:\n"
        "        pl.func_attr({'split': pl.SplitMode.UP_DOWN, 'slot_num': 4, "
        f"'local_slot_num': {local_depth}}})\n"
        "        av = pl.add(pl.load(a, [0, 0], [32, 32]), 1.0)\n"
        "        bv = pl.add(pl.load(b, [0, 0], [32, 32]), 1.0)\n"
        "        am = pl.move(av, target_memory=pl.Mem.Mat)\n"
        "        bm = pl.move(bv, target_memory=pl.Mem.Mat)\n"
        "        al = pl.move(am, target_memory=pl.Mem.Left)\n"
        "        br = pl.move(bm, target_memory=pl.Mem.Right)\n"
        "        result = pl.add(pl.matmul(al, br), 1.0)\n"
        "        out = pl.store(result, [0, 0], out)\n"
        "        return out\n"
    )
    if local_depth == 1:
        with pytest.raises(ValueError, match="local_slot_num=1 is insufficient for 2"):
            ir.compile(before, output_dir=str(tmp_path), skip_ptoas=True)
    else:
        ir.compile(before, output_dir=str(tmp_path), skip_ptoas=True)
        pto = "\n".join(p.read_text() for p in (tmp_path / "kernels").glob("*.pto"))
        assert pto.count(f"slot_num = 4, local_slot_num = {local_depth}") == 2
        reserves = [line for line in pto.splitlines() if "pto.reserve_buffer" in line]
        assert len(reserves) == 2
        assert all(f"size = {4096 * local_depth}" in line for line in reserves)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
