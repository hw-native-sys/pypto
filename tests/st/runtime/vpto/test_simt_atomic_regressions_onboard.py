# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public JIT regressions for ND atomics and nested observable expressions."""

import json
import os
import runpy
import sys
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from pypto import CacheConfig, runtime
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig

pytestmark = pytest.mark.platforms("a5", reason="Native SIMT atomics require A5")
ROOT = Path(__file__).resolve().parents[4]


@pl.jit.aiv
def regression_kernel(
    src: pl.Tensor[[2, 3, 32], pl.INT32],
    dst: pl.InOut[pl.Tensor[[2, 3, 32], pl.INT32]],
    buf: pl.InOut[pl.Tensor[[6], pl.INT32]],
    old: pl.InOut[pl.Tensor[[6], pl.INT32]],
):
    tile = pl.load(src, (0, 0, 0), (2, 3, 32))
    with pl.simt_scope(dims=(1, 1, 1)):
        one = pl.const(1, pl.INT32)
        a = pl.simt.atomic_add(tile, (1, 2, 7), one)
        b = pl.simt.atomic_min(tile, (1, 2, 7), pl.const(5, pl.INT32))
        c = pl.simt.atomic_max(tile, (1, 2, 7), pl.const(9, pl.INT32))
        d = pl.simt.atomic_exch(tile, (1, 2, 7), pl.const(7, pl.INT32))
        e = pl.simt.atomic_cas(tile, (1, 2, 7), pl.const(7, pl.INT32), pl.const(20, pl.INT32))
        pl.simt.store(old, (0,), a)
        pl.simt.store(old, (1,), b)
        pl.simt.store(old, (2,), c)
        pl.simt.store(old, (3,), d)
        pl.simt.store(old, (4,), e)
        delta = pl.simt.atomic_add(buf, (0,), one) - pl.simt.atomic_add(buf, (0,), one)
        while pl.simt.atomic_add(buf, (1,), one) < pl.const(2, pl.INT32):
            pl.simt.atomic_add(buf, (2,), one)
        if pl.simt.atomic_add(buf, (3,), one) * pl.const(0, pl.INT32) == pl.const(0, pl.INT32):
            pl.simt.atomic_add(buf, (4,), one)
        discarded = pl.simt.atomic_add(buf, (5,), one) * pl.const(0, pl.INT32)
        pl.simt.store(old, (5,), delta + discarded)
        # Copy the complete Tile through scalar addressing so the regression
        # checks untouched elements without requiring an ND tile.store path.
        for i in pl.range(192):
            value = pl.simt.load(tile, (i // 96, (i // 32) % 3, i % 32))
            pl.simt.store(dst, (i // 96, (i // 32) % 3, i % 32), value)
    return dst, buf, old


@pl.jit
def regression_entry(
    src: pl.Tensor[[2, 3, 32], pl.INT32],
    dst: pl.InOut[pl.Tensor[[2, 3, 32], pl.INT32]],
    buf: pl.InOut[pl.Tensor[[6], pl.INT32]],
    old: pl.InOut[pl.Tensor[[6], pl.INT32]],
):
    return regression_kernel(src, dst, buf, old)


def compile_case(directory: Path):
    return regression_entry.compile(
        config=RunConfig(
            backend_type=BackendType.Ascend950,
            execution_mode=ExecutionMode.ONBOARD,
            cache_config=CacheConfig(enabled=False),
            save_kernels_dir=str(directory),
            dump_passes=True,
        )
    )


def acceptance(output: Path, device: int | None):
    compiled = compile_case(output)
    compiled.load()
    checks = 0
    if device is not None:
        config = RunConfig(
            backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=device
        )
        with runtime.ChipWorker(config=config, runtime=compiled.runtime_name) as worker:
            kernel = worker.register(compiled)
            src = torch.full((2, 3, 32), -777, dtype=torch.int32)
            src[1, 2, 7] = 10
            dst = torch.full_like(src, -999)
            buf = torch.zeros(6, dtype=torch.int32)
            old = torch.full((6,), -777, dtype=torch.int32)
            kernel(src, dst, buf, old)
            expected = src.clone()
            expected[1, 2, 7] = 20
            torch.testing.assert_close(dst, expected, rtol=0, atol=0)
            torch.testing.assert_close(old, torch.tensor([10, 11, 5, 9, 7, -1], dtype=torch.int32))
            torch.testing.assert_close(buf, torch.tensor([2, 3, 2, 1, 1, 1], dtype=torch.int32))
            checks = 3
    (output / "result.json").write_text(json.dumps({"checks": checks}))


@pytest.mark.parametrize("onboard", [False, True], ids=["build", "persistent"])
def test_simt_atomic_regressions(tmp_path, onboard):
    gate = "PYPTO_RUN_VPTO_ONBOARD" if onboard else "PYPTO_RUN_VPTO_VF_VALIDATION"
    if os.environ.get(gate) != "1":
        pytest.skip(f"Opt in with {gate}=1")
    command = [sys.executable, Path(__file__).resolve(), "--worker", tmp_path]
    if onboard:
        command.append(os.environ["PYPTO_VPTO_DEVICE_ID"])
    support = runpy.run_path(str(ROOT / "tests/ut/backend/vpto_vf_validation/test_validation.py"))
    support["run"](
        tmp_path, "atomic_regressions", command, timeout=int(os.environ.get("PYPTO_VPTO_TIMEOUT", "600"))
    )
    assert json.loads((tmp_path / "result.json").read_text())["checks"] == (3 if onboard else 0)


if __name__ == "__main__":
    if len(sys.argv) in (3, 4) and sys.argv[1] == "--worker":
        acceptance(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) == 4 else None)
    else:
        raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
