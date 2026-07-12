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
lanes via ``aiv_id``. These tests compile such a kernel all the way through PTO
codegen (``RunConfig(codegen_only=True)``, no device) and assert:

* a ``.pto`` is emitted — guarding both that the lowering survives to codegen and
  that the nested region + in-region vector MemRef (the pipeline accumulator) is
  registered for MLIR emission (a regression that previously crashed
  ``pto_codegen`` with "no MLIR mapping for MemRef");
* the orchestration submit actually launches **both** AIV lanes (issue #2006 — it
  used to emit a single-AIV ``rt_submit_aiv_task``, silently dropping one lane's
  share of the output).
"""

import re
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


@pytest.fixture(scope="module")
def compiled() -> Exception | None:
    """Compile the kernel once (codegen only) into DUMP_DIR; return any late error.

    PTO codegen writes its artifacts before the downstream kernel-compilation step
    (simpler_setup), which is absent in the codegen-only CI env. Capture any such
    post-codegen failure — the guards below are on the emitted files, which
    materialize first.
    """
    if DUMP_DIR.exists():
        shutil.rmtree(DUMP_DIR)

    cfg = RunConfig(
        platform="a2a3",
        codegen_only=True,
        save_kernels=True,
        save_kernels_dir=str(DUMP_DIR),
    )
    try:
        split_aiv_none_pipe(torch.randn(T, N), torch.empty(T, N), config=cfg)
    except Exception as e:  # noqa: BLE001 - see docstring
        return e
    return None


def test_none_region_in_pipeline_compiles_to_pto(compiled):
    """The nested NONE region compiles through PTO codegen and emits a .pto."""
    ptos = sorted(DUMP_DIR.rglob("*.pto"))
    assert ptos, (
        f"codegen emitted no .pto under {DUMP_DIR} for a NONE split_aiv region; "
        f"compile raised before .pto materialized: {compiled!r}"
    )


def test_none_region_dispatches_both_aiv_lanes(compiled):
    """The orchestration must launch the kernel on BOTH AIV lanes.

    A NONE region is task-parallel: each lane runs the full body on the disjoint
    half selected by ``aiv_id``. ``rt_submit_aiv_task`` fills only the AIV0 slot, so
    the runtime schedules a single-AIV task — the second lane never launches, and
    the lone lane that does reads a ``get_sub_block_id()`` the runtime leaves
    undefined for single-AIV tasks. Half the output is then silently never written
    (issue #2006). The submit must be a two-lane ``MixedKernels`` (AIV0 + AIV1, no
    AIC), which the runtime schedules as a full cluster with sub_block_id 0 / 1.
    """
    orch = sorted(DUMP_DIR.rglob("orchestration/*.cpp"))
    assert orch, (
        f"codegen emitted no orchestration .cpp under {DUMP_DIR}; "
        f"compile raised before it materialized: {compiled!r}"
    )
    code = orch[0].read_text()

    mixed = re.search(r"MixedKernels \w+ = \{INVALID_KERNEL_ID, (\d+), \1\};", code)
    assert mixed, code
    assert "rt_submit_task(" in code, code

    # The single-AIV submit would run one lane per block and drop the other. Scope the
    # negative to *this* kernel id: a plain vector kernel in the same orchestration is
    # entitled to rt_submit_aiv_task (it keeps its independent-AIV parallelism).
    kernel_id = mixed.group(1)
    single_aiv_submits = [
        call
        for call in re.findall(r"rt_submit_aiv_task\([^;]*\)", code)
        if re.search(rf"\b{kernel_id}\b", call)
    ]
    assert not single_aiv_submits, single_aiv_submits


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
