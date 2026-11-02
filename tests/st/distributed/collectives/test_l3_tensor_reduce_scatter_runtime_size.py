# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: ``pld.tensor.reduce_scatter`` over a *runtime* SIZE.

The sibling ``test_l3_tensor_reduce_scatter_chunked.py`` compiles one program per
compile-time extent. This file compiles **once** with ``SIZE`` dynamic and drives
that artifact at three runtime extents:

* ``1001`` -- below one VEC chunk and not 32-byte aligned;
* ``3 * CHUNK`` -- an exact multi-chunk extent;
* ``3 * CHUNK + 1`` -- a one-element unaligned tail.

Each invocation calls ``reduce_scatter`` twice on the same signal, re-staging the
window between calls. That exercises the runtime loop bounds, the ragged tail,
the runtime-computed signal-reset count, and signal reuse in one artifact.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig

# FP32 VEC chunk emitted by the composite lowering (16 KiB tile).
CHUNK = 4096

# The row extent is resolvable only at run time. Annotations read this
# module-level symbol; a body that *uses* it in an expression must re-bind it
# locally (see ``reduce_step`` / ``host_orch``).
SIZE = pl.dynamic("SIZE")

# Capacity of the window buffer, in elements per row. Any SIZE <= this fits.
SIZE_MAX = 3 * CHUNK + 1

# Below one chunk and unaligned; exact three chunks; three chunks plus a
# one-element unaligned tail.
RUNTIME_SIZES = (1001, 3 * CHUNK, 3 * CHUNK + 1)


def _expected_reduce_scatter(inputs: torch.Tensor) -> torch.Tensor:
    """Per-rank golden: element-wise sum of chunk r across all ranks.

    Both calls compute this same value because the program re-stages the window
    from ``inp`` between them.
    """
    return inputs.sum(dim=0)


def _make_rank_inputs(n_ranks: int, size: int) -> torch.Tensor:
    """Distinct per-rank tensors holding ``n_ranks`` contiguous chunks of ``size``.

    Values are integer-valued and bounded, so every partial sum stays inside
    FP32's exact-integer range and the comparison can be bit-exact regardless of
    the accumulation order.
    """
    rows = [
        torch.arange(r * 100.0, r * 100.0 + n_ranks * size, dtype=torch.float32).reshape(n_ranks, size)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


def _build_runtime_size_program(n_ranks: int):
    """Build the runtime-SIZE reduce_scatter/reuse program.

    The window is allocated at ``SIZE_MAX`` but created at ``[nr, SIZE]``, so the
    collective's extent follows the runtime symbol. The input layout is
    ``[rank, chunk, size]``; this avoids a dynamic product in an annotation.
    """
    nr = n_ranks

    @pl.jit.incore
    def reduce_step(
        inp: pl.Tensor[[nr, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[2, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[2, SIZE], pl.FP32]:
        # A dynamic symbol is resolvable in a body only when the body binds it
        # locally; the annotations above read the module-level constant.
        SIZE = pl.dynamic("SIZE")
        # Stage-in: this rank writes chunk j of its input to window row j.
        for j in pl.range(nr):
            for c in pl.range(0, SIZE, CHUNK):
                staged = pl.load(inp, [j, c], [1, CHUNK], [1, pl.min(CHUNK, SIZE - c)])
                pl.store(staged, [j, c], data)

        # First collective: after it, rank r's own row holds chunk r reduced
        # across every rank.
        data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
        for c in pl.range(0, SIZE, CHUNK):
            first = pl.load(data, [my_rank, c], [1, CHUNK], [1, pl.min(CHUNK, SIZE - c)])
            pl.store(first, [0, c], out)

        # Restore the window so the second call sees exactly the first call's
        # input. Safe against the collective's WAR barrier: that barrier already
        # guaranteed every peer finished reading row r.
        for j in pl.range(nr):
            for c in pl.range(0, SIZE, CHUNK):
                restaged = pl.load(inp, [j, c], [1, CHUNK], [1, pl.min(CHUNK, SIZE - c)])
                pl.store(restaged, [j, c], data)

        # Second collective on the same signal. It can only complete if the
        # first call's epilogue reset the runtime-computed generation count, and
        # it must reproduce the first result.
        data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
        for c in pl.range(0, SIZE, CHUNK):
            second = pl.load(data, [my_rank, c], [1, CHUNK], [1, pl.min(CHUNK, SIZE - c)])
            pl.store(second, [1, c], out)
        return out

    @pl.jit
    def chip_orch(
        inp: pl.Tensor[[nr, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[2, SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[2, SIZE], pl.FP32]:
        return reduce_step(inp, out, data, signal, my_rank)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[nr, nr, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[nr, 2, SIZE], pl.FP32]],
    ) -> pl.Tensor[[nr, 2, SIZE], pl.FP32]:
        SIZE = pl.dynamic("SIZE")
        # Capacity allocation: any runtime extent <= SIZE_MAX fits this buffer.
        data_buf = pld.alloc_window_buffer(nr * SIZE_MAX * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
            sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
            chip_orch(inputs[r], outputs[r], data, sig, r, device=r)
        return outputs

    return host_orch


class TestL3TensorReduceScatterRuntimeSize:
    """Runtime-extent chunked reduce_scatter with signal reuse.

    One compiled artifact is exercised at three different extents. The program is
    compiled with no tensors at all -- the shape contract comes from the
    signature and a dynamic dim needs no value -- then run at each extent. If
    SIZE were baked to a constant at compile time, the second and third runs
    would fail.
    """

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_reduce_scatter_runtime_size(self, test_config, device_ids, n_ranks):
        if len(device_ids) < n_ranks:
            pytest.skip(f"reduce-scatter P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_runtime_size_program(n_ranks)
        compiled = program.compile(
            config=RunConfig(
                platform=test_config.platform,
                distributed_config=DistributedConfig(
                    device_ids=device_ids[:n_ranks],
                    num_sub_workers=0,
                ),
            ),
        )

        for size in RUNTIME_SIZES:
            inputs = _make_rank_inputs(n_ranks, size)
            outputs = torch.zeros((n_ranks, 2, size), dtype=torch.float32)

            compiled(inputs, outputs, config=RunConfig(platform=test_config.platform))

            expected = _expected_reduce_scatter(inputs)
            for call in range(2):
                got = outputs[:, call, :]
                assert torch.equal(got, expected), (
                    f"runtime-SIZE reduce-scatter SIZE={size} P={n_ranks} "
                    f"call {call + 1} mismatch: max diff = "
                    f"{(got - expected).abs().max().item()}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
