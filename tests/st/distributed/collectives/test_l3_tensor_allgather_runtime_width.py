# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: composite allgather over a *runtime* per-rank extent.

The sibling ``test_l3_tensor_allgather_intrinsic.py`` covers static transfers,
including ones well past one staging tile (SIZE=65537 is 16 chunks + 1), so
``pld.tile.put`` auto-chunking is proven for a **compile-time** extent.

What no test covers is the same path with an extent only known at run time. The
composite sizes its transfer from ``target.shape[1]`` and caps its bounce stage
at one 16 KiB chunk, delegating chunking to ``pld.tile.put``; the lowering emits
a single ``tput`` per peer with no chunk loop of its own, so whether a runtime
extent past one stage slides correctly is a property of the generated code's
*runtime* behaviour, not of its shape.

BF16 at D=5120 is 10 KiB per row, so ``w >= 2`` already exceeds one 16 KiB stage
and ``w = 5`` exceeds it by more than 3x.

The per-rank row count is the runtime symbol ``W``: it is supplied by the shape
of the tensors the host actually passes, so nothing about the transfer extent is
baked in at compile time.

Coverage: P=2 (default CI hosts) and P=4, ``a2a3sim``.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig

D = 5120  # token width; 10 KiB per BF16 row
W_MAX = 5  # capacity, in rows per rank
STAGE_CHUNK = 2048  # capacity-bounded stage-out read width, in elements

# The transfer extent, in elements. The annotations read this module-level
# symbol; a body that *uses* it in an expression must re-bind it locally (see
# ``gather_step``), which is how the DSL resolves a dynamic symbol in a body.
RUNTIME_N = pl.dynamic("RUNTIME_N")


def _make_rank_inputs(n_ranks: int, w: int) -> torch.Tensor:
    """Rank ``r``, row ``i`` holds ``r + i/1000``.

    Scaled per rank so a mis-slid or stale chunk shows up in the golden instead
    of hiding behind symmetry.
    """
    rows = [torch.arange(w * D, dtype=torch.float32) / 1000.0 + r for r in range(n_ranks)]
    return torch.stack(rows).unsqueeze(1)  # [nr, 1, w*D]


def _expected_allgather(inputs: torch.Tensor) -> torch.Tensor:
    """Every rank ends holding the same gathered ``[nr, N]`` matrix.

    Returns the ``[nr, nr, N]`` output shape — rank index first, then the
    gathered row — so the comparison cannot pass by broadcasting one rank's
    result across all of them.
    """
    n_ranks = inputs.shape[0]
    gathered = torch.cat([inputs[r, 0] for r in range(n_ranks)])
    matrix = gathered.reshape(n_ranks, inputs.shape[2])
    return matrix.unsqueeze(0).expand(n_ranks, *matrix.shape).contiguous()


def _build_runtime_width_program(n_ranks: int):
    """Allgather whose transfer extent is the runtime symbol ``RUNTIME_N``.

    Built with the ``@pl.jit`` family -- the surface users write, and the one
    whose specialisation path the test should therefore exercise. The window
    buffer is allocated at capacity, but the window itself is created at
    ``[nr, RUNTIME_N]`` so the composite's extent follows the runtime value.
    """
    nr = n_ranks

    @pl.jit.incore
    def gather_step(
        inp: pl.Tensor[[1, RUNTIME_N], pl.BF16],
        out: pl.Out[pl.Tensor[[nr, RUNTIME_N], pl.BF16]],
        data: pl.InOut[pld.DistributedTensor[[nr, RUNTIME_N], pl.BF16]],
        signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
    ) -> pl.Tensor[[nr, RUNTIME_N], pl.BF16]:
        # A dynamic symbol is resolvable in a body only when the body binds it
        # locally; the annotations above read the module-level constant.
        RUNTIME_N = pl.dynamic("RUNTIME_N")
        # The transfer under test: one push per peer over a runtime extent.
        data = pld.tensor.allgather(inp, data, signal)
        # Stage-out copies the gathered [nr, RUNTIME_N] window into the output;
        # the loop bound is the runtime extent, which is the point of the test.
        for r in pl.range(nr):
            for col in pl.range(0, RUNTIME_N, STAGE_CHUNK):
                valid = pl.min(STAGE_CHUNK, RUNTIME_N - col)
                chunk = pl.load(data, [r, col], [1, STAGE_CHUNK], valid_shape=[1, valid])
                pl.store(chunk, [r, col], out)
        return out

    @pl.jit
    def chip_orch(
        inp: pl.Tensor[[1, RUNTIME_N], pl.BF16],
        out: pl.Out[pl.Tensor[[nr, RUNTIME_N], pl.BF16]],
        data: pl.InOut[pld.DistributedTensor[[nr, RUNTIME_N], pl.BF16]],
        signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
    ) -> pl.Tensor[[nr, RUNTIME_N], pl.BF16]:
        return gather_step(inp, out, data, signal)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[nr, 1, RUNTIME_N], pl.BF16],
        outputs: pl.Out[pl.Tensor[[nr, nr, RUNTIME_N], pl.BF16]],
    ) -> pl.Tensor[[nr, nr, RUNTIME_N], pl.BF16]:
        RUNTIME_N = pl.dynamic("RUNTIME_N")
        # Capacity allocation: any extent <= W_MAX * D fits the same buffer.
        data_buf = pld.alloc_window_buffer(nr * W_MAX * D * pl.BF16.get_byte())
        signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [nr, RUNTIME_N], dtype=pl.BF16)
            sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
            chip_orch(inputs[r], outputs[r], data, sig, device=r)
        return outputs

    return host_orch


class TestL3TensorAllGatherRuntimeWidth:
    """Runtime-extent composite allgather.

    One compiled artifact is exercised at three different extents. That is the
    whole assertion: the program is compiled with no tensors at all -- the shape
    contract comes from the signature and a dynamic dim needs no value, so the
    artifact is extent-independent by construction -- and then run at each
    extent. If ``W`` were baked to a constant at compile time, the second and
    third runs would fail.
    """

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_allgather_runtime_width(self, test_config, device_ids, n_ranks):
        if len(device_ids) < n_ranks:
            pytest.skip(f"allgather P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_runtime_width_program(n_ranks)
        compiled = program.compile(
            config=RunConfig(
                platform=test_config.platform,
                distributed_config=DistributedConfig(
                    device_ids=device_ids[:n_ranks],
                    num_sub_workers=0,
                ),
            ),
        )

        # w=1 is under one 16 KiB stage; w=2 is 20 KiB and w=5 is 50 KiB, so
        # both cross the staging-tile cap and must be chunked at run time.
        for w in (1, 2, 5):
            inputs = _make_rank_inputs(n_ranks, w).bfloat16()
            outputs = torch.zeros((n_ranks, n_ranks, w * D), dtype=torch.bfloat16)

            compiled(inputs, outputs, config=RunConfig(platform=test_config.platform))

            expected = _expected_allgather(inputs)
            # Pure data movement, so the result must be bit-exact: a tolerance
            # here would hide a partially-written or mis-slid chunk.
            assert torch.equal(outputs, expected), (
                f"runtime-width allgather P={n_ranks} w={w}: max diff = "
                f"{(outputs.float() - expected.float()).abs().max().item()}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
