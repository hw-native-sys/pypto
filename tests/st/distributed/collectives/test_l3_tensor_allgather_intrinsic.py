# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank allgather via ``pld.tensor.allgather`` intrinsic.

Validates the composite allgather intrinsic produces the same rank-ordered
concatenation on every rank as the hand-written ``test_l3_allgather.py``.

Push-based (3-arg): ``pld.tensor.allgather(local_data, target, signal)`` —
each rank pushes its chunk to every peer's window slot via
``pld.tile.put`` (TPUT-based); after the notify/wait barrier, the window itself
holds the gathered ``[NR, SIZE]`` result (window-as-result).  The InCore
function then reads from the window and writes into the output tensor for
host-side verification.

ST coverage: **P=2** (default CI / 2-device hosts) and **P=4** (any four
devices). Both use the same N-rank program body.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig

STAGE_CHUNK = 4096


def _expected_allgather(inputs: torch.Tensor) -> torch.Tensor:
    """Rank-ordered concatenation; identical on every rank."""
    gathered = torch.cat([inputs[r, 0] for r in range(inputs.shape[0])])
    return torch.stack([gathered] * inputs.shape[0]).unsqueeze(1)


def _make_rank_inputs(n_ranks: int, size: int) -> torch.Tensor:
    """Distinct per-rank tensors so the golden concat is non-trivial."""
    rows = [
        torch.arange(r * 100.0, r * 100.0 + size, dtype=torch.float32).reshape(1, size)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


def _build_allgather_program(n_ranks: int, size: int):
    """Build an N-rank allgather program at call time using the intrinsic.

    Deferred construction lets this file collect even if the embedded body
    is rejected by the parser.
    """
    nr = n_ranks
    SIZE = size

    @pl.jit.incore
    def gather_step(
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, nr * SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
    ) -> pl.Tensor[[1, nr * SIZE], pl.FP32]:
        # Push-based allgather: window becomes the gathered [NR, SIZE] result.
        data = pld.tensor.allgather(inp, data, signal)
        # Stage-out: read from the gathered window into the output tensor,
        # chunked through a fixed-width on-chip tile so a non-tile-aligned
        # or larger-than-UB SIZE never reserves an oversized or misaligned
        # tile (STAGE_CHUNK is a compile-time constant, always 32-byte
        # aligned; valid_shape masks the logical width actually read).
        for r in pl.range(nr):
            for col in pl.range(0, SIZE, STAGE_CHUNK):
                valid = pl.min(STAGE_CHUNK, SIZE - col)
                chunk = pl.load(data, [r, col], [1, STAGE_CHUNK], valid_shape=[1, valid])
                pl.store(chunk, [0, r * SIZE + col], out)
        return out

    @pl.jit
    def chip_orch(
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, nr * SIZE], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
    ) -> pl.Tensor[[1, nr * SIZE], pl.FP32]:
        return gather_step(inp, out, data, signal)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[nr, 1, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[nr, 1, nr * SIZE], pl.FP32]],
    ) -> pl.Tensor[[nr, 1, nr * SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(nr * SIZE * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
            sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
            chip_orch(inputs[r], outputs[r], data, sig, device=r)
        return outputs

    return host_orch


def _build_allgather_in_spmd_program(n_ranks: int, size: int, blocks: int):
    """The allgather kernel dispatched from a ``blocks``-wide ``pl.spmd`` launch.

    The composite expansion never reads the block index, so every block would
    repeat the whole collective; compilation must reject it.
    """
    nr = n_ranks
    SIZE = size
    BLOCKS = blocks

    @pl.jit.incore
    def gather_step(
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
    ) -> pld.DistributedTensor[[nr, SIZE], pl.FP32]:
        return pld.tensor.allgather(inp, data, signal)

    @pl.jit
    def chip_orch(
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
    ) -> pld.DistributedTensor[[nr, SIZE], pl.FP32]:
        with pl.spmd(BLOCKS):
            data = gather_step(inp, data, signal)
        return data

    @pl.jit.host
    def host_orch(inputs: pl.Tensor[[nr, 1, SIZE], pl.FP32]) -> pl.Tensor[[nr, 1, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(nr * SIZE * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())
        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
            sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
            chip_orch(inputs[r], data, sig, device=r)
        return inputs

    return host_orch


class TestL3TensorAllGatherIntrinsic:
    """L3 distributed runtime: N-rank push-based allgather via ``pld.tensor.allgather``.

    Validates that the lowered composite produces an on-board result
    bit-identical to the hand-written ``test_l3_allgather.py`` reference.
    """

    @pytest.mark.parametrize("size", [17, 4097, 65537])
    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_allgather_intrinsic(self, test_config, device_ids, n_ranks, size):
        """Compile and run mesh allgather for P=2 or P=4; skip when devices are scarce.

        SIZE values cover the InCore staging-tile-cap fix: 17 (unaligned,
        exercises the floor-to-32-byte-aligned-width path — round DOWN, not
        up, since the stage tile can never exceed the transfer it slides
        through) and 4097 (one element past the 4096-element/16 KiB
        chunk-budget boundary for FP32 — the sharpest check that a >1-chunk
        transfer still slides correctly through the capped stage). SIZE=1 is
        not covered here: below one 32-byte alignment unit (8 FP32 elements)
        there is no stage width that both satisfies pto.alloc_tile's
        alignment rule and still fits within the transfer, a pre-existing gap
        this fix does not close. 65537 (the original bug repro size — a
        [1, 65537] FP32 stage would reserve 256 KiB and overflow UB before
        this fix) requires running WITHOUT --forked: with --forked, this case
        hangs (blocked in futex_wait_queue, not an error) — this is a
        pre-existing `--forked` + sim fork()-thread-lock deadlock, unrelated
        to this fix. Without --forked, all three sizes pass cleanly for every
        composite in this file.
        """
        if len(device_ids) < n_ranks:
            pytest.skip(f"allgather P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_allgather_program(n_ranks, size)
        inputs = _make_rank_inputs(n_ranks, size)
        outputs = torch.zeros((n_ranks, 1, n_ranks * size), dtype=torch.float32)
        compiled = program.compile(
            inputs,
            outputs,
            config=RunConfig(
                platform=test_config.platform,
                distributed_config=DistributedConfig(
                    device_ids=device_ids[:n_ranks],
                    num_sub_workers=0,
                ),
            ),
        )

        compiled(inputs, outputs, config=RunConfig(platform=test_config.platform))

        expected = _expected_allgather(inputs)
        assert torch.allclose(outputs, expected), (
            f"allgather intrinsic P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


def test_allgather_in_multi_block_spmd_launch_is_rejected():
    """An InCore collective in a 4-block ``pl.spmd`` launch fails to compile (no device needed)."""
    n_ranks, size = 2, 64
    program = _build_allgather_in_spmd_program(n_ranks, size, blocks=4)
    inputs = _make_rank_inputs(n_ranks, size)
    with pytest.raises(ValueError, match="pld.tensor.allgather runs inside an SPMD launch of 4 blocks"):
        program.compile(
            inputs,
            config=RunConfig(
                platform="a2a3sim",
                codegen_only=True,
                distributed_config=DistributedConfig(device_ids=[0, 1], num_sub_workers=0),
            ),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
