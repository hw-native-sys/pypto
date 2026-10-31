# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A declared ``CachePolicy.BYPASS`` read returns the same bytes on device.

The bypass is an *address* change on a2a3: the load is issued against the
uncached alias of the page, ``addr + get_l2_cache_offset(args)``. That makes the
declaration the one kind of performance hint that can return wrong data — a
wrong offset reads unrelated memory rather than running slower, and on this box
the driver reports a nonzero one (``0x80000000000``), so the aliased path is
what actually executes here.

Only a device run can check that: the simulator has no alias and reports zero,
where ``addr + 0`` is the ordinary address and any offset plumbing would pass.
Both surfaces are covered, because they reach codegen by different routes — the
per-load kwarg annotates one ``tile.load``, while the scope declaration is
resolved to the outlined kernel's parameters and applied by
``ConvertTensorToTileOps``.

Inputs are small integers so the FP32 accumulation is exact and the comparison
runs at ``rtol=atol=0``: a load that read the wrong address would have to
reproduce the product by accident to pass.

DEFAULT and BYPASS run through both declaration surfaces and memory planners.
The A5 explicit-hint matrix lives in ``cross_core/test_cross_core.py``, which
is included in the A5 onboard CI selection without changing the workflow.
"""

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto.pypto_core.passes import MemoryPlanner

M, K, N = 64, 256, 128


def _inputs(transposed_weight: bool):
    """Small integers, so the FP32 accumulation is exact at rtol=atol=0."""
    generator = torch.Generator().manual_seed(11)
    x = torch.randint(-4, 5, (M, K), generator=generator).to(torch.float16)
    w_shape = (N, K) if transposed_weight else (K, N)
    w = torch.randint(-4, 5, w_shape, generator=generator).to(torch.float16)
    return x, w, torch.zeros((M, N), dtype=torch.float32)


def _golden_transposed(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.matmul(tensors["x"].to(torch.float32), tensors["w"].to(torch.float32).T)


def _golden(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.matmul(tensors["x"].to(torch.float32), tensors["w"].to(torch.float32))


def _cache_policy_case(cache, planner, scope):
    """Keep both legacy declaration paths, with DEFAULT as the reference."""

    @pl.jit.incore
    def load_kernel(
        x: pl.Tensor[[M, K], pl.FP16],
        w: pl.Tensor[[N, K], pl.FP16],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        xt = pl.load(x, [0, 0], [M, K], target_memory=pl.Mem.Mat)
        wt = pl.load(w, [0, 0], [N, K], target_memory=pl.Mem.Mat, cache=cache)
        return pl.store(pl.matmul(xt, pl.tile.transpose_view(wt)), [0, 0], out)

    @pl.jit
    def per_load(
        x: pl.Tensor[[M, K], pl.FP16],
        w: pl.Tensor[[N, K], pl.FP16],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        return load_kernel(x, w, out)

    @pl.jit
    def scope_bypass(
        x: pl.Tensor[[M, K], pl.FP16],
        w: pl.Tensor[[K, N], pl.FP16],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
            pl.set_cache_policy(w, pl.CachePolicy.BYPASS)
            c: pl.Tensor[[M, N], pl.FP32] = pl.matmul(x, w, out_dtype=pl.FP32)
            out = pl.assemble(out, c, [0, 0])
        return out

    @pl.jit
    def scope_default(
        x: pl.Tensor[[M, K], pl.FP16],
        w: pl.Tensor[[K, N], pl.FP16],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
            pl.set_cache_policy(w, pl.CachePolicy.DEFAULT)
            c: pl.Tensor[[M, N], pl.FP32] = pl.matmul(x, w, out_dtype=pl.FP32)
            out = pl.assemble(out, c, [0, 0])
        return out

    scope_kernel = scope_bypass if cache == pl.CachePolicy.BYPASS else scope_default
    surface = "scope" if scope else "per_load"
    return st.case(
        scope_kernel if scope else per_load,
        *_inputs(transposed_weight=not scope),
        name=f"cache_{cache.name}_{surface}_{planner.name}",
        golden=_golden if scope else _golden_transposed,
        memory_planner=planner,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.platforms("a2a3", "a2a3sim")
@st.cases(
    *[
        _cache_policy_case(cache, planner, scope)
        for cache in (pl.CachePolicy.DEFAULT, pl.CachePolicy.BYPASS)
        for planner in (MemoryPlanner.PYPTO, MemoryPlanner.PTOAS)
        for scope in (False, True)
    ]
)
def test_cache_policy_reads_the_same_bytes(case_run):
    """The runtime-offset BYPASS path preserves the same bytes as DEFAULT."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
