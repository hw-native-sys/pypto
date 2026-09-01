# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""Compose broadcast + allreduce + allgather in one tiny kernel — the capstone.

Concepts introduced:
  - the three collectives from steps 12-15 working together in ONE kernel,
    the way a real model uses them: weights are broadcast, activations are
    reduced, and results are gathered
  - one signal per collective — a teaching choice: the InCore composites'
    self-clearing credit protocol makes back-to-back reuse of a single
    `[nr, 1]` window safe (verified at P=2 and P=4); separate signals keep
    each barrier visible in the IR diff
  - the pipeline: broadcast root's weights -> allreduce the per-rank slices
    (mesh) -> allgather the per-rank slices -> scale the gathered matrix by
    the broadcast weight locally

The golden checks both stages: ``allred[r] == sum_k x[k]`` on every rank, and
``gathered[r] == concat(x[0], ..., x[P-1]) * w`` — the allgather result scaled
by the broadcast weight, which also proves the weight reached every rank.

Cross-links (see the walkthrough): this is the picotron ``model.py`` idea in
miniature. The same pattern scales to real models — AllGather-GEMM
(pypto-lib #869) is an allgather built from step 13's pattern; distributed MoE
dispatch/combine is an all-to-all built from step 15's pattern.

Run + walkthrough: see docs/en/user/distributed/21-putting_it_together.md
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto import ir
from pypto.ir import DistributedConfig

SIZE = 64
ROOT_RANK = 0


def build_compose(nr: int):
    """Build the composition program for a rank count ``nr``.

    Unlike steps 12-15 there is no mode flag here, and unlike steps 09/10 no
    **tile shape** depends on the rank count — so nothing in this step
    actually requires a compile-time ``nr``. Verified: replacing it with a
    ``pl.dynamic`` rank count plus ``pld.nranks(ctx)`` compiles and passes the
    golden, exactly as step 08 does. The factory is kept only so the three
    ``[nr, 1]`` signals and the ``[nr, SIZE]`` gather target read as literal
    shapes; it could be dropped for a module-level program.
    """

    @pl.program
    class PuttingItTogether:
        @pl.function(type=pl.FunctionType.InCore)
        def compose_step(
            self,
            x: pl.Tensor[[1, SIZE], pl.FP32],
            w_in: pl.Tensor[[1, SIZE], pl.FP32],
            allred: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            gathered: pl.Out[pl.Tensor[[1, nr * SIZE], pl.FP32]],
            w_data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            ar_data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            ag_data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            sig_bcast: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            sig_ar: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            sig_ag: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[1, nr * SIZE], pl.FP32]:
            """Chip kernel: broadcast weights, allreduce, allgather, scale locally."""
            ctx = pld.get_comm_ctx(w_data)
            my_rank = pld.rank(ctx)
            nranks = pld.nranks(ctx)

            # 1 — Broadcast: root stages its weights, every rank gets them.
            if my_rank == ROOT_RANK:
                local_w = pl.load(w_in, [0, 0], [1, SIZE])
                w_data = pl.store(local_w, [0, 0], w_data)
            w_data = pld.tensor.broadcast(w_data, sig_bcast, root=ROOT_RANK)
            w = pl.load(w_data, [0, 0], [1, SIZE])

            # 2 — Allreduce: every rank ends with the element-wise sum of the
            # per-rank slices (the mesh mode from step 11).
            local_x = pl.load(x, [0, 0], [1, SIZE])
            ar_data = pl.store(local_x, [0, 0], ar_data)
            ar_data = pld.tensor.allreduce(ar_data, sig_ar, op=pld.ReduceOp.Sum, mode="mesh")
            total = pl.load(ar_data, [0, 0], [1, SIZE])
            allred = pl.store(total, [0, 0], allred)

            # 3 — Allgather: every rank ends with all ranks' raw slices (the
            # push-based form from step 13; its source is a plain tensor).
            ag_data = pld.tensor.allgather(x, ag_data, sig_ag)

            # 4 — Local compute: scale the gathered matrix by the shared
            # broadcast weight — a learned per-feature weight over the
            # gathered hidden states.
            for src in pl.range(nranks):
                chunk = pl.load(ag_data, [src, 0], [1, SIZE])
                chunk = pl.mul(chunk, w)
                gathered = pl.store(chunk, [0, src * SIZE], gathered)
            return gathered

        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(
            self,
            x: pl.Tensor[[1, SIZE], pl.FP32],
            w_in: pl.Tensor[[1, SIZE], pl.FP32],
            allred: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            gathered: pl.Out[pl.Tensor[[1, nr * SIZE], pl.FP32]],
            w_data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            ar_data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            ag_data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            sig_bcast: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            sig_ar: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            sig_ag: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[1, nr * SIZE], pl.FP32]:
            """Per-device orchestration: one incore call, on this device."""
            return self.compose_step(
                x,
                w_in,
                allred,
                gathered,
                w_data,
                ar_data,
                ag_data,
                sig_bcast,
                sig_ar,
                sig_ag,
            )

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, 1, SIZE], pl.FP32],
            weights: pl.Tensor[[1, SIZE], pl.FP32],
            allred_out: pl.Out[pl.Tensor[[nr, 1, SIZE], pl.FP32]],
            gathered_out: pl.Out[pl.Tensor[[nr, 1, nr * SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, 1, nr * SIZE], pl.FP32]:
            """Host orchestrator: one shared set of windows, one dispatch per rank."""
            w_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
            ar_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
            ag_buf = pld.alloc_window_buffer([nr, SIZE], dtype=pl.FP32)
            sig_bcast_buf = pld.alloc_window_buffer([nr, 1], dtype=pl.INT32)
            sig_ar_buf = pld.alloc_window_buffer([nr, 1], dtype=pl.INT32)
            sig_ag_buf = pld.alloc_window_buffer([nr, 1], dtype=pl.INT32)

            for r in pl.range(pld.world_size()):
                w_data = pld.window(w_buf, [1, SIZE], dtype=pl.FP32)
                ar_data = pld.window(ar_buf, [1, SIZE], dtype=pl.FP32)
                ag_data = pld.window(ag_buf, [nr, SIZE], dtype=pl.FP32)
                sig_bcast = pld.window(sig_bcast_buf, [nr, 1], dtype=pl.INT32)
                sig_ar = pld.window(sig_ar_buf, [nr, 1], dtype=pl.INT32)
                sig_ag = pld.window(sig_ag_buf, [nr, 1], dtype=pl.INT32)
                self.per_rank(
                    inputs[r],
                    weights,
                    allred_out[r],
                    gathered_out[r],
                    w_data,
                    ar_data,
                    ag_data,
                    sig_bcast,
                    sig_ar,
                    sig_ag,
                    device=r,
                )
            return gathered_out

    return PuttingItTogether


def expected_compose(inputs: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Golden: allred[r] = sum_k x[k]; gathered[r] = concat_k(x[k] * w)."""
    allred = inputs.sum(dim=0, keepdim=True).expand(inputs.shape[0], 1, -1).contiguous()
    concat_all = torch.cat([inputs[k] * weights for k in range(inputs.shape[0])], dim=1)
    gathered = concat_all.expand(inputs.shape[0], -1, -1).contiguous()
    return allred, gathered


def main() -> int:
    parser = argparse.ArgumentParser(description="16_putting_it_together")
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a2a3sim",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="0,1",
        help="comma-separated device ids -- any count >= 2; root is rank 0",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) < 2:
        raise SystemExit(f"need at least 2 devices, got {device_ids}")

    nr = len(device_ids)
    program = build_compose(nr)

    # Distinct per-rank data so the reduction and gather goldens are non-trivial.
    rows = [
        torch.arange(r * 100.0, r * 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE) for r in range(nr)
    ]
    inputs = torch.stack(rows)
    weights = torch.arange(1.0, 1.0 + SIZE, dtype=torch.float32).reshape(1, SIZE)
    allred_out = torch.zeros((nr, 1, SIZE), dtype=torch.float32)
    gathered_out = torch.zeros((nr, 1, nr * SIZE), dtype=torch.float32)

    compiled = ir.compile(
        program,
        platform=args.platform,
        distributed_config=DistributedConfig(
            device_ids=device_ids,
            num_sub_workers=0,
        ),
    )
    if args.compile_only:
        print(f"compile_only done: {compiled.output_dir}")
        return 0

    compiled(inputs, weights, allred_out, gathered_out)

    exp_allred, exp_gathered = expected_compose(inputs, weights)
    assert torch.allclose(allred_out, exp_allred, rtol=1e-5, atol=1e-5), (
        f"compose allreduce P={nr} mismatch: max diff = {(allred_out - exp_allred).abs().max().item()}"
    )
    assert torch.allclose(gathered_out, exp_gathered, rtol=1e-5, atol=1e-5), (
        f"compose allgather P={nr} mismatch: max diff = {(gathered_out - exp_gathered).abs().max().item()}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
