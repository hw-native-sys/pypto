# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Issue #2902: auto-split Vec operands must reuse their gathered Mat storage."""

import re

import pypto.language as pl
import pytest
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.runtime import RunConfig


@pytest.mark.parametrize("b_trans", [False, True])
@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910B, BackendType.Ascend950], indirect=True)
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
    # There is exactly one Vec->cube crossing. Its consumers must read the popped
    # Mat directly, and the FIFO release must follow the final use. A5's larger
    # operand capacity permits a whole-tile placement rather than K extracts.
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
    assert len(consumers) >= (2 if ascend_backend == BackendType.Ascend910B else 1), pto
    frees = [i for i, line in enumerate(lines) if "pto.tfree_from_aiv" in line]
    assert len(frees) == 1
    assert pop_idx < min(consumers) <= max(consumers) < frees[0]
    for line in pto.splitlines():
        if "pto.tmov " in line:
            assert line.count("loc=mat") < 2, line
    if find_ptoas_binary() is not None:
        assert list((tmp_path / "kernels").rglob("*.cpp"))


@pytest.mark.parametrize("slot_num", [1, 2])
@pytest.mark.parametrize("place_first", [False, True])
@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910B, BackendType.Ascend950], indirect=True)
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
@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910B, BackendType.Ascend950], indirect=True)
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


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910B, BackendType.Ascend950], indirect=True)
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
