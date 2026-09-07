# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end guard for the ``st.cases`` declaration path.

A case declared with ``@st.cases(...)`` has to survive the whole route:
collection reads it out of ``callspec.params``, the pre-compile pool builds its
IR from the ``@pl.jit`` entry, the golden runs in this process, and the device
step is batched like any other case.

The discovery half is what this file actually guards, and it is guarded by
assertion rather than by inspection: a case the pipeline picked up has a
published artifact directory, and one that fell through to the per-case inline
path does not.  Before ``st.cases`` existed, a ``@pl.jit`` test could not be
discovered at all — it silently took the inline path — so a regression here
would otherwise show up only as CI getting slower.

Runs card-free under ``--codegen-only``; on a device it is one small kernel in
the batched pool.
"""

import pypto.language as pl
import pytest
import torch

from harness import st

M = 16
N = 16


@pl.jit.incore
def _abs_kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    tile_a = pl.load(a, [0, 0], [M, N])
    return pl.store(pl.tile.abs(tile_a), [0, 0], out)


@pl.jit
def _abs_entry(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    out = _abs_kernel(a, out)
    return out


torch.manual_seed(0)
_A = torch.randn(M, N, dtype=torch.float32)


@st.cases(
    st.case(
        _abs_entry,
        _A,
        torch.zeros(M, N, dtype=torch.float32),
        name="harness_pipeline_abs",
        golden=lambda tensors: torch.abs(tensors["a"]),
    ),
)
def test_declared_case_runs(case_run, request):
    """The declared case compiles, runs, and matches its golden."""
    case_run.assert_passed()

    # Discovery check: with the pre-compile pool active, a discovered case owns
    # a published artifact directory. ``None`` here means collection did not see
    # the declaration and the case fell back to inline compilation.
    if request.config.getoption("--precompile-workers") is not None:
        assert case_run.work_dir is not None, (
            "case was not picked up by the pre-compile pipeline — "
            "pytest_collection_finish did not read it from callspec.params"
        )
        assert (case_run.work_dir / "golden.py").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
