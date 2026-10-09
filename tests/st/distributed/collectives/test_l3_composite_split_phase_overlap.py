# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 NPU ST: ``defer=True`` allgather overlap via ``device_wall``.

Interleaved fused vs deferred+independent-compute on real A2/A3 hardware.
Simulator platforms are skipped — sim does not model device timing.

The deferred program issues ``allgather(..., defer=True)`` then a compute task
that does **not** depend on the gather TaskId, then a consume that does. The
fused program runs gather+compute serially in one InCore body. On a quiet box
the deferred median ``device_wall`` should beat the fused serial baseline; on a
shared box the gate only fails when defer is clearly slower (>10% after
trimming high outliers).
"""

from __future__ import annotations

import statistics
import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir import DistributedConfig
from pypto.runtime import benchmark

SIZE = 256
COMPUTE_ITERS = 64  # Large enough that fused serial gather+compute dominates noise.
N_RANKS = 2
WARMUP = 2
ROUNDS = 12


def _make_inputs() -> torch.Tensor:
    rows = [
        torch.arange(r * 100.0, r * 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE)
        for r in range(N_RANKS)
    ]
    return torch.stack(rows).share_memory_()


def _build_fused(n_ranks: int = N_RANKS):
    nr = n_ranks
    iters = COMPUTE_ITERS

    @pl.program
    class FusedGatherThenCompute:
        @pl.function(type=pl.FunctionType.InCore)
        def body(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, nr * SIZE], pl.FP32]],
            scratch: pl.InOut[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[1, nr * SIZE], pl.FP32]:
            data = pld.tensor.allgather(inp, data, signal)
            acc = pl.load(inp, [0, 0], [1, SIZE])
            for _ in pl.range(iters):
                acc = pl.add(acc, acc)
            pl.store(acc, [0, 0], scratch)
            for r in pl.range(nr):
                chunk = pl.load(data, [r, 0], [1, SIZE])
                pl.store(chunk, [0, r * SIZE], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, nr * SIZE], pl.FP32]],
            scratch: pl.InOut[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[1, nr * SIZE], pl.FP32]:
            return self.body(inp, out, scratch, data, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, 1, nr * SIZE], pl.FP32]],
            scratches: pl.InOut[pl.Tensor[[nr, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, 1, nr * SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(nr * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())
            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], scratches[r], data, sig, device=r)
            return outputs

    return FusedGatherThenCompute


def _build_deferred_overlap(n_ranks: int = N_RANKS):
    nr = n_ranks
    iters = COMPUTE_ITERS

    @pl.program
    class DeferredGatherOverlapCompute:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, nr * SIZE], pl.FP32]],
            scratch: pl.InOut[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[1, nr * SIZE], pl.FP32]:
            with pl.manual_scope():
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="ag_defer") as ag_tid:
                    pld.tensor.allgather(inp, data, signal, defer=True)
                # Independent of ag_tid — may run while the deferred wait is pending.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="local_compute"):
                    acc = pl.load(inp, [0, 0], [1, SIZE])
                    for _ in pl.range(iters):
                        acc = pl.add(acc, acc)
                    pl.store(acc, [0, 0], scratch)
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="ag_consume",
                    deps=[ag_tid],
                ):
                    for r in pl.range(nr):
                        chunk = pl.load(data, [r, 0], [1, SIZE])
                        pl.store(chunk, [0, r * SIZE], out)
            return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, 1, nr * SIZE], pl.FP32]],
            scratches: pl.InOut[pl.Tensor[[nr, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, 1, nr * SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(nr * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())
            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], scratches[r], data, sig, device=r)
            return outputs

    return DeferredGatherOverlapCompute


def _compile(program, test_config, device_ids, output_dir):
    return ir.compile(
        program,
        output_dir=str(output_dir),
        platform=test_config.platform,
        distributed_config=DistributedConfig(
            device_ids=device_ids[:N_RANKS],
            num_sub_workers=0,
        ),
    )


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def _trimmed_median(xs: list[float], *, drop_high: int = 2) -> float:
    """Median after dropping the highest outliers (shared-box device_wall spikes)."""
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) > drop_high + 1:
        ys = ys[: len(ys) - drop_high]
    return float(statistics.median(ys))


@pytest.mark.platforms("a2a3")
def test_allgather_defer_overlap_device_wall(test_config, device_ids, tmp_path):
    if len(device_ids) < N_RANKS:
        pytest.skip(f"overlap ST needs {N_RANKS} devices, got {device_ids}")

    fused = _compile(_build_fused(), test_config, device_ids, tmp_path / "fused")
    deferred = _compile(_build_deferred_overlap(), test_config, device_ids, tmp_path / "defer")

    inputs = _make_inputs()
    outs_f = torch.zeros((N_RANKS, 1, N_RANKS * SIZE), dtype=torch.float32).share_memory_()
    outs_d = torch.zeros((N_RANKS, 1, N_RANKS * SIZE), dtype=torch.float32).share_memory_()
    scratch_f = torch.zeros((N_RANKS, 1, SIZE), dtype=torch.float32).share_memory_()
    scratch_d = torch.zeros((N_RANKS, 1, SIZE), dtype=torch.float32).share_memory_()

    # Correctness first (one-shot).
    fused(inputs, outs_f, scratch_f, config=test_config)
    deferred(inputs, outs_d, scratch_d, config=test_config)
    assert torch.allclose(outs_f, outs_d), (
        f"fused vs defer gather mismatch: max diff={(outs_f - outs_d).abs().max().item()}"
    )

    # Interleaved A/B/A/B device_wall (interleaved A/B sampling).
    samples_fused: list[float] = []
    samples_defer: list[float] = []
    for _ in range(ROUNDS):
        for label, compiled, outs, scratch, bucket in (
            ("fused", fused, outs_f, scratch_f, samples_fused),
            ("defer", deferred, outs_d, scratch_d, samples_defer),
        ):
            stats = benchmark(
                compiled,
                [inputs, outs, scratch],
                rounds=1,
                warmup=0 if bucket else WARMUP,
                config=test_config,
                persistent=True,
                reset_persistent_windows=False,
            )
            if not stats.device_wall_us:
                pytest.skip("benchmark returned no device_wall_us (profiling markers missing)")
            bucket.extend(stats.device_wall_us)

    med_f = _trimmed_median(samples_fused)
    med_d = _trimmed_median(samples_defer)
    speedup = (med_f / med_d) if med_d else float("nan")
    print(
        f"[defer-overlap] fused_trimmed_median_us={med_f:.1f} "
        f"defer_trimmed_median_us={med_d:.1f} speedup={speedup:.4f}x "
        f"raw_fused_median_us={_median(samples_fused):.1f} "
        f"raw_defer_median_us={_median(samples_defer):.1f}",
        flush=True,
    )
    # Shared-box device_wall is noisy: require a clear regression to fail, not a
    # strict win every session. Ten-run surveys on this host saw ~1.15× mean
    # speedup with intermittent near-ties; treat within 10% as inconclusive.
    assert med_d <= med_f * 1.10, (
        f"defer=True clearly slower than fused serial device_wall "
        f"(defer_trimmed_median={med_d:.1f}us fused_trimmed_median={med_f:.1f}us "
        f"speedup={speedup:.3f}x fused_samples={samples_fused} "
        f"defer_samples={samples_defer}); "
        "split-phase API shows a clear regression on this box"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
