# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end codegen guard for a task-parallel ``pl.split_aiv(2, mode=NONE)``
region nested inside a ``pl.pipeline`` loop.

A NONE region keeps its tiles full-width (no halving) and dispatches both AIV
lanes via ``aiv_id``. This test compiles such a kernel all the way through PTO
codegen (``RunConfig(codegen_only=True)``, no device) and asserts a ``.pto`` is
emitted — guarding both that the lowering survives to codegen and that the nested
region + in-region vector MemRef (the pipeline accumulator) is registered for
MLIR emission (a regression that previously crashed ``pto_codegen`` with
"no MLIR mapping for MemRef").
"""

import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import pypto.language as pl  # noqa: E402
from pypto.runtime import RunConfig  # noqa: E402

T, N = 128, 128
DUMP_DIR = Path(__file__).resolve().parents[4] / "build_output" / "split_aiv_none_codegen"


@pl.jit
def split_aiv_none_pipe(
    a: pl.Tensor[[T, N], pl.FP32],
    c: pl.Out[pl.Tensor[[T, N], pl.FP32]],
):
    """Both AIV lanes accumulate a full [64, N] tile over a pipeline, then store
    to disjoint halves selected by aiv_id (task-parallel, no halving)."""
    for blk in pl.spmd(2):  # noqa: B007 - block dispatch context
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            base = aiv_id * 64
            acc = pl.load(a, [base, 0], [64, N])
            for sb in pl.pipeline(1, 4, stage=2):  # noqa: B007 - pipeline index
                t = pl.load(a, [base, 0], [64, N])
                acc = pl.add(acc, t)
            c = pl.store(acc, [base, 0], c)
    return c


def test_none_region_in_pipeline_compiles_to_pto():
    """The nested NONE region compiles through PTO codegen and emits a .pto."""
    if DUMP_DIR.exists():
        shutil.rmtree(DUMP_DIR)

    cfg = RunConfig(
        platform="a2a3",
        codegen_only=True,
        save_kernels=True,
        save_kernels_dir=str(DUMP_DIR),
    )
    split_aiv_none_pipe(torch.randn(T, N), torch.empty(T, N), config=cfg)

    ptos = sorted(DUMP_DIR.rglob("*.pto"))
    assert ptos, f"codegen emitted no .pto under {DUMP_DIR} for a NONE split_aiv region"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
