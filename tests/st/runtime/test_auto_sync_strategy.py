# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 device goldens for automatic synchronization and buffer reuse.

Run under the caller's build/device locks with a diagnostic timeout:
``timeout 180 python -m pytest tests/st/runtime/test_auto_sync_strategy.py
--platform=a5 --device=0 -v``. No CPU simulator path is exercised here.
"""

import pypto.language as pl
import pytest
import torch
from pypto import passes
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig, ensure_pto_isa_root


@pl.jit
def tagged_reuse(
    a: pl.Tensor[[256, 32], pl.FP32],
    out: pl.Out[pl.Tensor[[256, 32], pl.FP32]],
) -> pl.Tensor[[256, 32], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        for row in pl.range(0, 256, 32):
            x = pl.load(a, [row, 0], [32, 32])
            alias = pl.reshape(x, [16, 64])
            y = pl.add(alias, 1.0)
            result = pl.reshape(y, [32, 32])
            pl.store(result, [row, 0], out)
    return out


@pytest.mark.parametrize("strategy", list(passes.AutoSyncStrategy.__members__.values()))
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_tagged_loop_reuse_on_a5(test_config, tmp_path, strategy, planner):
    """Eight tile generations and four dispatch tags must never read stale storage."""
    if test_config.platform != "a5":
        pytest.skip("Automatic Buffer-ID device validation requires --platform=a5")
    ensure_pto_isa_root()
    config = RunConfig(
        backend_type=BackendType.Ascend950,
        execution_mode=ExecutionMode.ONBOARD,
        device_id=test_config.device_id,
        save_kernels_dir=str(tmp_path / f"{strategy.name}_{planner.name}"),
        dump_passes=False,
    )
    with passes.PassContext([], sync_strategy=strategy, memory_planner=planner):
        compiled = tagged_reuse.compile(config=config)
    for tag in range(4):
        # Distinct, exactly representable integers label every tile and dispatch.
        a = torch.arange(256 * 32, dtype=torch.float32).reshape(256, 32) + tag * 65536
        out = torch.full_like(a, -777)
        expected = a + 1.0
        compiled(a, out, config=config)
        torch.testing.assert_close(
            out,
            expected,
            rtol=0,
            atol=0,
            msg=f"stale tile data: strategy={strategy.name}, planner={planner.name}, dispatch={tag}",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
