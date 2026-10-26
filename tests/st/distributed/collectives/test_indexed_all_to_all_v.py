# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 URMA indexed exchange: full rows, segmented doorbells, and window reuse."""

import os

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig


def build_indexed_program(nr: int, capacity: int, width: int, cores: int, payload_dtype: str = "fp32"):
    """Build both rails with identical staging and valid-prefix consumption."""
    dtype = pl.INT8 if payload_dtype == "int8" else pl.FP32
    element_bytes = dtype.get_byte()
    padding_value = 123.0 if payload_dtype == "fp32" else 123
    aligned = width * element_bytes % 32 == 0
    nmax = capacity + 3
    total = nr * capacity

    @pl.jit.incore
    def stage_step(
        inp: pl.Tensor[[nmax, width], dtype],
        ids: pl.Tensor[[nr, capacity], pl.INT32],
        cnt: pl.Tensor[[nr, 1], pl.INT32],
        x: pl.InOut[pld.DistributedTensor[[nmax, width], dtype]],
        index: pl.InOut[pld.DistributedTensor[[nr, capacity], pl.INT32]],
        counts: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        target: pl.InOut[pld.DistributedTensor[[total, width], dtype]],
    ) -> tuple[
        pld.DistributedTensor[[nmax, width], dtype],
        pld.DistributedTensor[[nr, capacity], pl.INT32],
        pld.DistributedTensor[[nr, 1], pl.INT32],
        pld.DistributedTensor[[total, width], dtype],
    ]:
        for i in pl.range(nmax):
            if aligned:
                row = pl.load(inp, [i, 0], [1, width])
                x = pl.store(row, [i, 0], x)
            else:
                for j in pl.range(width):
                    pl.write(x, [i, j], pl.read(inp, [i, j]))
        for d in pl.range(nr):
            pl.write(counts, [d, 0], pl.read(cnt, [d, 0]))
            for j in pl.range(capacity):
                pl.write(index, [d, j], pl.read(ids, [d, j]))
        for i in pl.range(total):
            if aligned:
                padding = pl.tile.full([1, width], dtype=dtype, value=123)
                target = pl.store(padding, [i, 0], target)
            else:
                for j in pl.range(width):
                    pl.write(target, [i, j], pl.cast(padding_value, dtype))
        return x, index, counts, target

    @pl.jit.incore
    def consume_step(
        target: pld.DistributedTensor[[total, width], dtype],
        recv: pl.Tensor[[nr, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[total, width], dtype]],
        rc: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
    ) -> tuple[pl.Tensor[[total, width], dtype], pl.Tensor[[nr, 1], pl.INT32]]:
        for i in pl.range(total):
            if aligned:
                row = pl.load(target, [i, 0], [1, width])
                out = pl.store(row, [i, 0], out)
            else:
                for j in pl.range(width):
                    pl.write(out, [i, j], pl.read(target, [i, j]))
        for s in pl.range(nr):
            pl.write(rc, [s, 0], pl.read(recv, [s, 0]))
        return out, rc

    @pl.jit
    def stage_orch(
        inp: pl.Tensor[[nmax, width], dtype],
        ids: pl.Tensor[[nr, capacity], pl.INT32],
        cnt: pl.Tensor[[nr, 1], pl.INT32],
        x: pl.InOut[pld.DistributedTensor[[nmax, width], dtype]],
        index: pl.InOut[pld.DistributedTensor[[nr, capacity], pl.INT32]],
        counts: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        target: pl.InOut[pld.DistributedTensor[[total, width], dtype]],
    ) -> tuple[
        pld.DistributedTensor[[nmax, width], dtype],
        pld.DistributedTensor[[nr, capacity], pl.INT32],
        pld.DistributedTensor[[nr, 1], pl.INT32],
        pld.DistributedTensor[[total, width], dtype],
    ]:
        return stage_step(inp, ids, cnt, x, index, counts, target)

    @pl.jit
    def pipeline(
        x: pl.Tensor[[nmax, width], dtype],
        index: pl.Tensor[[nr, capacity], pl.INT32],
        counts: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        target: pl.InOut[pld.DistributedTensor[[total, width], dtype]],
        signal: pl.InOut[pld.DistributedTensor[[nr, cores], pl.INT32]],
        out: pl.Out[pl.Tensor[[total, width], dtype]],
        rc: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
    ) -> tuple[pl.Tensor[[total, width], dtype], pl.Tensor[[nr, 1], pl.INT32]]:
        recv: pl.Tensor[[nr, 1], pl.INT32] = pl.create_tensor([nr, 1], dtype=pl.INT32)
        target = pld.tensor.indexed_all_to_all_v(x, target, signal, counts, recv, index, core_num=cores)
        return consume_step(target, recv, out, rc)

    @pl.jit.host
    def main(
        inputs: pl.Tensor[[nr, nmax, width], dtype],
        indices: pl.Tensor[[nr, nr, capacity], pl.INT32],
        send: pl.Tensor[[nr, nr, 1], pl.INT32],
        outputs: pl.Out[pl.Tensor[[nr, total, width], dtype]],
        received: pl.Out[pl.Tensor[[nr, nr, 1], pl.INT32]],
    ) -> tuple[pl.Tensor[[nr, total, width], dtype], pl.Tensor[[nr, nr, 1], pl.INT32]]:
        xb = pld.alloc_window_buffer(nmax * width * element_bytes)
        ib = pld.alloc_window_buffer(nr * capacity * 4)
        cb = pld.alloc_window_buffer(nr * 4)
        tb = pld.alloc_window_buffer(total * width * element_bytes)
        sb = pld.alloc_window_buffer(nr * cores * 4)
        for r in pl.range(nr):
            sx = pld.window(xb, [nmax, width], dtype=dtype)
            si = pld.window(ib, [nr, capacity], dtype=pl.INT32)
            sc = pld.window(cb, [nr, 1], dtype=pl.INT32)
            st = pld.window(tb, [total, width], dtype=dtype)
            stage_orch(inputs[r], indices[r], send[r], sx, si, sc, st, device=r)
        for r in pl.range(nr):
            pc = pld.window(cb, [nr, 1], dtype=pl.INT32)
            pt = pld.window(tb, [total, width], dtype=dtype)
            ps = pld.window(sb, [nr, cores], dtype=pl.INT32)
            pipeline(inputs[r], indices[r], pc, pt, ps, outputs[r], received[r], device=r)
        return outputs, received

    return main


def make_inputs(nr: int, capacity: int, width: int, mode: str, payload_dtype: str = "fp32"):
    nmax = capacity + 3
    torch_dtype = torch.int8 if payload_dtype == "int8" else torch.float32
    inputs = (torch.arange(nr * nmax * width) % 97).to(torch_dtype).reshape(nr, nmax, width)
    indices = torch.full((nr, nr, capacity), -1, dtype=torch.int32)
    counts = torch.zeros((nr, nr, 1), dtype=torch.int32)
    golden = torch.full((nr, nr * capacity, width), 123, dtype=torch_dtype)
    for s in range(nr):
        for d in range(nr):
            n = capacity if mode == "full" else ((s + d) % 3) * capacity // 2
            if mode == "self":
                n = capacity if s == d else 0
            elif mode == "zero":
                n = 0
            counts[s, d, 0] = n
            ids = (torch.arange(n, dtype=torch.int32) * 7 + d + s) % nmax
            if n > 1:
                ids[-1] = ids[0]  # Explicit duplicate: preserve both slots.
            indices[s, d, :n] = ids
            golden[d, s * capacity : s * capacity + n] = inputs[s, ids.long()]
    return inputs, indices, counts, golden


@pytest.mark.parametrize("width", [64, 17], ids=["aligned", "ragged"])
@pytest.mark.parametrize("payload_dtype", ["fp32", "int8"])
@pytest.mark.parametrize("cores", [1, 2, 3, 5], ids=lambda c: f"c{c}")
@pytest.mark.parametrize(
    "capacity,mode", [(1, "full"), (13, "mixed"), (25, "full"), (769, "full"), (13, "self"), (13, "zero")]
)
def test_indexed_exchange(test_config, device_ids, cores, capacity, mode, payload_dtype, width):
    if test_config.platform != "a5":
        pytest.skip("indexed_all_to_all_v supports A5 URMA only")
    if len(device_ids) < 2:
        pytest.skip("requires two A5 devices")
    nr = 2
    inputs, indices, counts, golden = make_inputs(nr, capacity, width, mode, payload_dtype)
    output = torch.empty_like(golden)
    recv = torch.empty((nr, nr, 1), dtype=torch.int32)
    args = [inputs, indices, counts, output, recv]
    for tensor in args:
        tensor.share_memory_()
    compiled = build_indexed_program(nr, capacity, width, cores, payload_dtype).compile(
        *args,
        config=RunConfig(
            platform="a5", distributed_config=DistributedConfig(device_ids=device_ids[:nr], num_sub_workers=0)
        ),
    )
    # Reuse live signals/queue state without clearing persistent windows.
    with compiled.prepare(persistent=True, reset_persistent_windows=False) as worker:
        registered = worker.register(compiled)
        for iteration in range(3):
            inputs.add_(iteration)
            registered(*args, config=RunConfig(platform="a5"))
            expected = golden.clone()
            expected[golden != 123] += iteration * (iteration + 1) // 2
            assert torch.equal(recv, counts.transpose(0, 1)), (iteration, recv, counts)
            assert torch.equal(output, expected), (
                iteration,
                (output != expected).any(dim=-1).nonzero().tolist(),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5", "--device", os.environ.get("TASK_DEVICE", "0,1")])
