# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Issues #2902 and #2992: Vec operands must reuse their gathered Mat storage."""

import re
import runpy
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from pypto import passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.runtime import ExecutionMode, RunConfig


@pytest.mark.parametrize("mode", ["mixed", "left_right", "staged"])
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910, BackendType.Ascend950], indirect=True)
def test_mixed_example_fits_safe_vec_budget(tmp_path, ascend_backend, planner, mode):
    """Compile the real example, including both splits and the staged control."""
    example = runpy.run_path(
        str(Path(__file__).resolve().parents[3] / "examples/advanced/03_mixed_kernel.py")
    )
    m, n, k = example["M"], example["N"], example["K"]
    args = (
        torch.empty(m, k, dtype=torch.float16),
        torch.empty(k, n, dtype=torch.float16),
        torch.empty(m, n, dtype=torch.float32),
        torch.empty(m, n, dtype=torch.float32),
    )
    # A two-slot ring plus the independently allocated bias and output needs
    # 192 KiB on Ascend910. The safety guard must remain enabled; retune only
    # the example's ring. Compile() exercises allocation and native codegen,
    # without dispatching to hardware or requiring a simulator installation.
    # The UT harness already supplies a PassContext; select the planner on a
    # nested context rather than supplying conflicting RunConfig overrides.
    with passes.PassContext([], memory_planner=planner):
        example["_MODES"][mode].compile(
            *args,
            config=RunConfig(
                backend_type=ascend_backend,
                execution_mode=ExecutionMode.SIM,
                codegen_only=True,
                save_kernels=True,
                save_kernels_dir=str(tmp_path),
            ),
        )
    files = list(tmp_path.rglob("*.pto"))
    assert files
    if mode != "staged":
        # Check the emitted pipe contract, not just the authored optimization.
        pto = "\n".join(path.read_text() for path in files)
        pipe_lines = [line for line in pto.splitlines() if "initialize_pipe" in line]
        assert pipe_lines
        assert all(re.search(r"\bslot_num\s*=\s*1\b", line) for line in pipe_lines)
    if find_ptoas_binary() is not None:
        assert list((tmp_path / "kernels").rglob("*.cpp"))


def _assert_single_mat_handoff(pto: str, min_consumers: int = 1) -> str:
    """Check direct Mat consumption and release after the last FIFO tile use."""
    lines = pto.splitlines()
    pops = [(i, line) for i, line in enumerate(lines) if "pto.tpop_from_aiv" in line]
    assert len(pops) == 1
    pop_idx, pop = pops[0]
    popped = re.search(r"(%[\w]+) = pto.tpop_from_aiv", pop)
    assert popped is not None, pop
    consumers = [
        i
        for i, line in enumerate(lines)
        if f"pto.textract ins({popped[1]}," in line or f"pto.tmov ins({popped[1]} :" in line
    ]
    assert len(consumers) >= min_consumers, pto
    frees = [i for i, line in enumerate(lines) if "pto.tfree_from_aiv" in line]
    assert len(frees) == 1
    assert pop_idx < min(consumers) <= max(consumers) < frees[0]
    for line in pto.splitlines():
        if "pto.tmov " in line:
            assert line.count("loc=mat") < 2, line
    return pop


@pytest.mark.parametrize("b_trans", [False, True])
@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910, BackendType.Ascend950], indirect=True)
def test_auto_split_mat_staging(tmp_path, ascend_backend, b_trans):
    """AutoTile's K loop consumes the gathered operand without a Mat->Mat move."""
    vn, vk = (256, 128) if b_trans else (128, 256)

    @pl.jit
    def attention(
        q: pl.Tensor[[256, 256], pl.BF16],
        k: pl.Tensor[[256, 256], pl.BF16],
        v: pl.Tensor[[256, 256], pl.BF16],
        out: pl.Out[pl.Tensor[[256, 256], pl.FP32]],
    ):
        for blk in pl.spmd(
            4,
            name_hint="attention",
            optimizations=[pl.cross_core_slot(slot_num=1), pl.split(pl.SplitMode.UP_DOWN)],
        ):
            r0 = blk * 64
            scores = pl.matmul(q[r0 : r0 + 64, 0:256], k[0:128, 0:256], b_trans=True, out_dtype=pl.FP32)
            prob = pl.cast(pl.exp(scores), target_type=pl.BF16, mode="rint")
            product = pl.matmul(prob, v[0:vn, 0:vk], b_trans=b_trans, out_dtype=pl.FP32)
            out[r0 : r0 + 64, 0:256] = pl.add(product, 1.0)
        return out

    attention.compile(
        config=RunConfig(
            platform="a5" if ascend_backend == BackendType.Ascend950 else "a2a3",
            codegen_only=True,
            save_kernels=True,
            save_kernels_dir=str(tmp_path),
        )
    )
    files = list(tmp_path.rglob("*.pto"))
    assert files
    pto = "\n".join(path.read_text() for path in files)
    _assert_single_mat_handoff(pto, min_consumers=2 if ascend_backend == BackendType.Ascend910 else 1)
    if find_ptoas_binary() is not None:
        assert list((tmp_path / "kernels").rglob("*.cpp"))


@pytest.mark.parametrize("split_mode", [pl.SplitMode.NONE, pl.SplitMode.UP_DOWN])
@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910, BackendType.Ascend950], indirect=True)
def test_explicit_vec_to_mat_qk_pv(tmp_path, ascend_backend, split_mode):
    """An authored NZ Vec->Mat boundary needs no copy, even without K tiling (#2992)."""

    @pl.jit(auto_scope=False)
    def entry(
        query: pl.Tensor[[16, 16], pl.BF16],
        keys: pl.Tensor[[16, 16], pl.BF16],
        values: pl.Tensor[[16, 16], pl.BF16],
        output: pl.Out[pl.Tensor[[16, 16], pl.BF16]],
    ):
        with pl.spmd(
            1, name_hint="qk_pv", optimizations=[pl.split(split_mode), pl.cross_core_slot(slot_num=1)]
        ):
            block = pl.tile.get_block_idx()
            q = pl.tile.load(query, [block * 16, 0], [16, 16], target_memory=pl.Mem.Mat)
            k = pl.tile.load(keys, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            v = pl.tile.load(values, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            qk = pl.tile.matmul(q, pl.tile.transpose_view(k))
            scores = pl.tile.move(
                qk, pl.Mem.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box
            )
            probabilities = pl.exp(scores)
            weights = pl.tile.move(
                pl.cast(probabilities, pl.BF16, mode="rint"),
                pl.Mem.Mat,
                blayout=pl.TileLayout.col_major,
                slayout=pl.TileLayout.row_major,
            )
            pv = pl.tile.matmul(weights, v)
            result = pl.tile.move(
                pv, pl.Mem.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box
            )
            pl.tile.store(pl.cast(result, pl.BF16, mode="rint"), [block * 16, 0], output)
        return output

    entry.compile(
        config=RunConfig(
            platform="a5" if ascend_backend == BackendType.Ascend950 else "a2a3",
            codegen_only=True,
            save_kernels=True,
            save_kernels_dir=str(tmp_path),
        )
    )
    files = list(tmp_path.rglob("*.pto"))
    assert files
    pto = "\n".join(path.read_text() for path in files)
    pop = _assert_single_mat_handoff(pto)
    split_code = 0 if split_mode == pl.SplitMode.NONE else 1
    assert f"split = {split_code}" in pop
    assert "loc=mat, dtype=bf16, rows=16, cols=16" in pop
    assert "blayout=col_major, slayout=row_major" in pop
    if find_ptoas_binary() is not None:
        assert list((tmp_path / "kernels").rglob("*.cpp"))


@pytest.mark.parametrize("slot_num", [1, 2])
@pytest.mark.parametrize("place_first", [False, True])
@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910, BackendType.Ascend950], indirect=True)
def test_two_retained_mat_operands_require_two_fifo_slots(tmp_path, ascend_backend, slot_num, place_first):
    """Two gathers before either placement cannot share a one-slot ring."""

    @pl.jit
    def two_operands(
        a: pl.Tensor[[32, 32], pl.FP16],
        b: pl.Tensor[[32, 32], pl.FP16],
        out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ):
        for _ in pl.spmd(
            1,
            optimizations=[pl.cross_core_slot(slot_num=slot_num), pl.split(pl.SplitMode.UP_DOWN)],
        ):
            av = pl.add(pl.load(a, [0, 0], [32, 32]), 1.0)
            bv = pl.add(pl.load(b, [0, 0], [32, 32]), 1.0)
            am = pl.move(av, target_memory=pl.Mem.Mat)
            bm = pl.move(bv, target_memory=pl.Mem.Mat)
            al = pl.move(am, target_memory=pl.Mem.Left)
            br = pl.move(bm, target_memory=pl.Mem.Right)
            out = pl.store(pl.matmul(al, br), [0, 0], out)
        return out

    @pl.jit
    def early_placement(
        a: pl.Tensor[[32, 32], pl.FP16],
        b: pl.Tensor[[32, 32], pl.FP16],
        out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ):
        for _ in pl.spmd(
            1,
            optimizations=[pl.cross_core_slot(slot_num=slot_num), pl.split(pl.SplitMode.UP_DOWN)],
        ):
            # Repeated and nested handoffs need one slot, not one per gather.
            for _iteration in pl.range(2):
                av = pl.add(pl.load(a, [0, 0], [32, 32]), 1.0)
                bv = pl.add(pl.load(b, [0, 0], [32, 32]), 1.0)
                am = pl.move(av, target_memory=pl.Mem.Mat)
                al = pl.move(am, target_memory=pl.Mem.Left)
                bm = pl.move(bv, target_memory=pl.Mem.Mat)
                br = pl.move(bm, target_memory=pl.Mem.Right)
                out = pl.store(pl.matmul(al, br), [0, 0], out)
        return out

    config = RunConfig(
        platform="a5" if ascend_backend == BackendType.Ascend950 else "a2a3",
        codegen_only=True,
        save_kernels=True,
        save_kernels_dir=str(tmp_path),
    )
    if slot_num == 1 and not place_first:
        with pytest.raises(ValueError, match="slot_num=1 is insufficient for 2 simultaneously retained"):
            two_operands.compile(config=config)
        assert not list(tmp_path.rglob("*.pto"))
    else:
        kernel = early_placement if place_first else two_operands
        kernel.compile(config=config)
        pto = "\n".join(path.read_text() for path in tmp_path.rglob("*.pto"))
        lines = pto.splitlines()
        pops = [i for i, line in enumerate(lines) if "pto.tpop_from_aiv" in line]
        frees = [i for i, line in enumerate(lines) if "pto.tfree_from_aiv" in line]
        assert len(pops) == len(frees) == 2
        assert (frees[0] < pops[1]) if place_first else (pops[1] < frees[0])
        assert f"slot_num = {slot_num}" in pto or f"slot_num={slot_num}" in pto


@pytest.mark.parametrize("third_pop, slot_num", [(False, 2), (True, 2), (True, 3)])
@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910, BackendType.Ascend950], indirect=True)
def test_reverse_consumption_preserves_fifo_prefix(tmp_path, ascend_backend, third_pop, slot_num):
    """Finishing the second tile first must not release the first tile's slot."""

    @pl.jit
    def reverse_two(
        a: pl.Tensor[[32, 32], pl.FP16],
        b: pl.Tensor[[32, 32], pl.FP16],
        out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ):
        for _ in pl.spmd(
            1,
            optimizations=[pl.cross_core_slot(slot_num=slot_num), pl.split(pl.SplitMode.UP_DOWN)],
        ):
            av = pl.add(pl.load(a, [0, 0], [32, 32]), 1.0)
            bv = pl.add(pl.load(b, [0, 0], [32, 32]), 1.0)
            am = pl.move(av, target_memory=pl.Mem.Mat)
            bm = pl.move(bv, target_memory=pl.Mem.Mat)
            br = pl.move(bm, target_memory=pl.Mem.Right)
            al = pl.move(am, target_memory=pl.Mem.Left)
            out = pl.store(pl.matmul(al, br), [0, 0], out)
        return out

    @pl.jit
    def reverse_three(
        a: pl.Tensor[[32, 32], pl.FP16],
        b: pl.Tensor[[32, 32], pl.FP16],
        c: pl.Tensor[[32, 32], pl.FP16],
        out_a: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
        out_c: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ):
        for _ in pl.spmd(
            1,
            optimizations=[pl.cross_core_slot(slot_num=slot_num), pl.split(pl.SplitMode.UP_DOWN)],
        ):
            av = pl.add(pl.load(a, [0, 0], [32, 32]), 1.0)
            bv = pl.add(pl.load(b, [0, 0], [32, 32]), 1.0)
            cv = pl.add(pl.load(c, [0, 0], [32, 32]), 1.0)
            am = pl.move(av, target_memory=pl.Mem.Mat)
            bm = pl.move(bv, target_memory=pl.Mem.Mat)
            br = pl.move(bm, target_memory=pl.Mem.Right)
            cm = pl.move(cv, target_memory=pl.Mem.Mat)
            al = pl.move(am, target_memory=pl.Mem.Left)
            cl = pl.move(cm, target_memory=pl.Mem.Left)
            out_a = pl.store(pl.matmul(al, br), [0, 0], out_a)
            out_c = pl.store(pl.matmul(cl, br), [0, 0], out_c)
        return out_a, out_c

    kernel = reverse_three if third_pop else reverse_two
    config = RunConfig(
        platform="a5" if ascend_backend == BackendType.Ascend950 else "a2a3",
        codegen_only=True,
        save_kernels=True,
        save_kernels_dir=str(tmp_path),
    )
    if third_pop and slot_num == 2:
        with pytest.raises(ValueError, match="slot_num=2 is insufficient for 3 simultaneously retained"):
            kernel.compile(config=config)
        assert not list(tmp_path.rglob("*.pto"))
        return

    kernel.compile(config=config)
    pto = "\n".join(path.read_text() for path in tmp_path.rglob("*.pto"))
    lines = pto.splitlines()
    pops = [i for i, line in enumerate(lines) if "pto.tpop_from_aiv" in line]
    frees = [i for i, line in enumerate(lines) if "pto.tfree_from_aiv" in line]
    assert len(pops) == len(frees) == (3 if third_pop else 2)
    first_tile = re.search(r"(%[\w]+) = pto.tpop_from_aiv", lines[pops[0]])
    assert first_tile is not None
    first_consumers = [i for i, line in enumerate(lines) if f"pto.tmov ins({first_tile[1]} :" in line]
    assert first_consumers
    assert max(first_consumers) < frees[0] < frees[1]
    assert pops[-1] < frees[0]
    if find_ptoas_binary() is not None:
        assert list((tmp_path / "kernels").rglob("*.cpp"))


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910, BackendType.Ascend950], indirect=True)
def test_nested_handoff_cannot_release_an_enclosing_fifo_tile(tmp_path, ascend_backend):
    """A nested last-use finalizer must not free past an enclosing live pop."""

    @pl.jit
    def nested_reverse(
        a: pl.Tensor[[32, 32], pl.FP16],
        b: pl.Tensor[[32, 32], pl.FP16],
        inner_out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
        outer_out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ):
        for _ in pl.spmd(
            1,
            optimizations=[pl.cross_core_slot(slot_num=3), pl.split(pl.SplitMode.UP_DOWN)],
        ):
            base_l = pl.move(pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat), pl.Mem.Left)
            base_r = pl.move(pl.load(b, [0, 0], [32, 32], target_memory=pl.Mem.Mat), pl.Mem.Right)
            av = pl.add(pl.load(a, [0, 0], [32, 32]), 1.0)
            am = pl.move(av, target_memory=pl.Mem.Mat)
            for _iteration in pl.range(2):
                bv = pl.add(pl.load(b, [0, 0], [32, 32]), 1.0)
                bm = pl.move(bv, target_memory=pl.Mem.Mat)
                br = pl.move(bm, target_memory=pl.Mem.Right)
                inner_out = pl.store(pl.matmul(base_l, br), [0, 0], inner_out)
            al = pl.move(am, target_memory=pl.Mem.Left)
            outer_out = pl.store(pl.matmul(al, base_r), [0, 0], outer_out)
        return inner_out, outer_out

    with pytest.raises(ValueError, match="requires FIFO release order"):
        nested_reverse.compile(
            config=RunConfig(
                platform="a5" if ascend_backend == BackendType.Ascend950 else "a2a3",
                codegen_only=True,
                save_kernels=True,
                save_kernels_dir=str(tmp_path),
            )
        )
    assert not list(tmp_path.rglob("*.pto"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
