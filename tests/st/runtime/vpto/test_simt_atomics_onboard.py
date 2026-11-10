# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Atomic old values, contention and UB barrier visibility on the public JIT path."""

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

pytestmark = pytest.mark.platforms("a5", reason="SIMT atomics require A5")
ROOT = Path(__file__).resolve().parents[4]


@pl.jit.aiv
def atomic_kernel(
    integers: pl.InOut[pl.Tensor[[6, 32], pl.INT32]],
    floats: pl.InOut[pl.Tensor[[6, 32], pl.FP32]],
    old_i: pl.InOut[pl.Tensor[[5, 32], pl.INT32]],
    old_f: pl.InOut[pl.Tensor[[5, 32], pl.FP32]],
    old_ub: pl.InOut[pl.Tensor[[5, 32], pl.INT32]],
    ub_out: pl.Out[pl.Tensor[[6, 32], pl.INT32]],
    old_ubf: pl.InOut[pl.Tensor[[5, 32], pl.FP32]],
    ubf_out: pl.Out[pl.Tensor[[6, 32], pl.FP32]],
):
    ub = pl.load(integers, (0, 0), (6, 32))
    ubf = pl.load(floats, (0, 0), (6, 32))
    with pl.simt_scope(dims=(32, 1, 1)):
        tid = pl.simt.thread_idx(pl.simt.Axis.X)
        vi = tid + pl.const(1, pl.INT32)
        vf = pl.cast(vi, pl.FP32)
        previous = pl.simt.atomic_add(integers, (0, 0), vi)
        pl.simt.store(old_i, (0, tid), previous)
        previous = pl.simt.atomic_min(integers, (1, 0), vi)
        pl.simt.store(old_i, (1, tid), previous)
        previous = pl.simt.atomic_max(integers, (2, 0), vi)
        pl.simt.store(old_i, (2, tid), previous)
        previous = pl.simt.atomic_exch(integers, (3, 0), vi)
        pl.simt.store(old_i, (3, tid), previous)
        previous = pl.simt.atomic_cas(integers, (4, 0), pl.const(0, pl.INT32), vi)
        pl.simt.store(old_i, (4, tid), previous)
        pl.simt.atomic_add(integers, (5, 0), pl.const(1, pl.INT32))
        previous = pl.simt.atomic_add(floats, (0, 0), vf)
        pl.simt.store(old_f, (0, tid), previous)
        previous = pl.simt.atomic_min(floats, (1, 0), vf)
        pl.simt.store(old_f, (1, tid), previous)
        previous = pl.simt.atomic_max(floats, (2, 0), vf)
        pl.simt.store(old_f, (2, tid), previous)
        previous = pl.simt.atomic_exch(floats, (3, 0), vf)
        pl.simt.store(old_f, (3, tid), previous)
        previous = pl.simt.atomic_cas(floats, (4, 0), pl.const(0.0, pl.FP32), vf)
        pl.simt.store(old_f, (4, tid), previous)
        pl.simt.atomic_add(floats, (5, 0), pl.const(1.0, pl.FP32))
        previous = pl.simt.atomic_add(ub, (0, 0), vi)
        pl.simt.store(old_ub, (0, tid), previous)
        previous = pl.simt.atomic_min(ub, (1, 0), vi)
        pl.simt.store(old_ub, (1, tid), previous)
        previous = pl.simt.atomic_max(ub, (2, 0), vi)
        pl.simt.store(old_ub, (2, tid), previous)
        previous = pl.simt.atomic_exch(ub, (3, 0), vi)
        pl.simt.store(old_ub, (3, tid), previous)
        previous = pl.simt.atomic_cas(ub, (4, 0), pl.const(0, pl.INT32), vi)
        pl.simt.store(old_ub, (4, tid), previous)
        pl.simt.atomic_add(ub, (5, 0), pl.const(1, pl.INT32))
        previous = pl.simt.atomic_add(ubf, (0, 0), vf)
        pl.simt.store(old_ubf, (0, tid), previous)
        previous = pl.simt.atomic_min(ubf, (1, 0), vf)
        pl.simt.store(old_ubf, (1, tid), previous)
        previous = pl.simt.atomic_max(ubf, (2, 0), vf)
        pl.simt.store(old_ubf, (2, tid), previous)
        previous = pl.simt.atomic_exch(ubf, (3, 0), vf)
        pl.simt.store(old_ubf, (3, tid), previous)
        previous = pl.simt.atomic_cas(ubf, (4, 0), pl.const(0.0, pl.FP32), vf)
        pl.simt.store(old_ubf, (4, tid), previous)
        pl.simt.atomic_add(ubf, (5, 0), pl.const(1.0, pl.FP32))
        pl.simt.threadfence()
        pl.simt.threadfence_block()
    pl.store(ub, (0, 0), ub_out)
    pl.store(ubf, (0, 0), ubf_out)
    return integers, floats, old_i, old_f, old_ub, ub_out, old_ubf, ubf_out


@pl.jit
def atomic_entry(
    integers: pl.InOut[pl.Tensor[[6, 32], pl.INT32]],
    floats: pl.InOut[pl.Tensor[[6, 32], pl.FP32]],
    old_i: pl.InOut[pl.Tensor[[5, 32], pl.INT32]],
    old_f: pl.InOut[pl.Tensor[[5, 32], pl.FP32]],
    old_ub: pl.InOut[pl.Tensor[[5, 32], pl.INT32]],
    ub_out: pl.Out[pl.Tensor[[6, 32], pl.INT32]],
    old_ubf: pl.InOut[pl.Tensor[[5, 32], pl.FP32]],
    ubf_out: pl.Out[pl.Tensor[[6, 32], pl.FP32]],
):
    return atomic_kernel(integers, floats, old_i, old_f, old_ub, ub_out, old_ubf, ubf_out)


@pl.jit.aiv
def barrier_kernel(
    src: pl.Tensor[[1, 128], pl.INT32],
    dst: pl.InOut[pl.Tensor[[1, 128], pl.INT32]],
    copied: pl.Out[pl.Tensor[[1, 128], pl.INT32]],
    published: pl.InOut[pl.Tensor[[1, 128], pl.INT32]],
):
    tile = pl.load(src, (0, 0), (1, 128))
    with pl.simt_scope(dims=(128, 1, 1)):
        tid = pl.simt.thread_idx(pl.simt.Axis.X)
        pl.simt.store(tile, (0, tid), tid)
        pl.simt.store(published, (0, tid), tid)
        pl.simt.threadfence()
        pl.simt.threadfence_block()
        pl.simt.syncthreads()
        # Read a different warp's write only after the block rendezvous.
        neighbor = pl.simt.load(tile, (0, (tid + 32) % 128))
        gm_neighbor = pl.simt.load(published, (0, (tid + 32) % 128), l1cache="uncache")
        pl.simt.store(dst, (0, tid), neighbor + gm_neighbor)
        pl.simt.threadfence()
    # Outer MTE visibility is the launch protocol's responsibility.
    pl.store(tile, (0, 0), copied)
    return dst, copied, published


@pl.jit
def barrier_entry(
    src: pl.Tensor[[1, 128], pl.INT32],
    dst: pl.InOut[pl.Tensor[[1, 128], pl.INT32]],
    copied: pl.Out[pl.Tensor[[1, 128], pl.INT32]],
    published: pl.InOut[pl.Tensor[[1, 128], pl.INT32]],
):
    return barrier_kernel(src, dst, copied, published)


def compile_case(directory: Path, *, entry=atomic_entry):
    return entry.compile(
        config=RunConfig(
            backend_type=BackendType.Ascend950,
            execution_mode=ExecutionMode.ONBOARD,
            cache_config=CacheConfig(enabled=False),
            save_kernels_dir=str(directory),
            dump_passes=True,
        )
    )


def check_atomics(state, old):
    assert state[0, 0].item() == 528
    assert state[1, 0].item() == 1
    assert state[2, 0].item() == 32
    assert state[5, 0].item() == 32
    # Add returns the serialized prefix sums; the first old value is zero.
    assert old[0].min().item() == 0
    assert old[0].unique().numel() == 32
    assert bool(((old[1] >= 1) & (old[1] <= 100)).all())
    assert bool(((old[2] >= -100) & (old[2] <= 32)).all())
    # Exchange serializes a permutation, with one final value left in storage.
    values = torch.cat((old[3], state[3, :1])).sort().values
    expected = torch.tensor([-1, *range(1, 33)], dtype=state.dtype)
    torch.testing.assert_close(values, expected, rtol=0, atol=0)
    # Exactly one CAS succeeds; failures all return the winning replacement.
    winner = state[4, 0].item()
    assert 1 <= winner <= 32
    assert (old[4] == 0).sum().item() == 1
    assert bool(((old[4] == 0) | (old[4] == winner)).all())
    assert bool((state[:, 1:] == -777).all())


def acceptance(output: Path, device: int | None):
    compiled = compile_case(output)
    compiled.load()
    barrier = compile_case(output / "barrier", entry=barrier_entry)
    barrier.load()
    checks = 0
    if device is not None:
        config = RunConfig(
            backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=device
        )
        with runtime.ChipWorker(config=config, runtime=compiled.runtime_name) as worker:
            atomic = worker.register(compiled)
            sync = worker.register(barrier)
            for _ in range(3):
                integers = torch.full((6, 32), -777, dtype=torch.int32)
                integers[:, 0] = torch.tensor([0, 100, -100, -1, 0, 0])
                floats = integers.float()
                old_i = torch.full((5, 32), -777, dtype=torch.int32)
                old_f = old_i.float()
                old_ub = old_i.clone()
                ub_out = torch.empty_like(integers)
                old_ubf, ubf_out = old_f.clone(), floats.clone()
                atomic(integers, floats, old_i, old_f, old_ub, ub_out, old_ubf, ubf_out)
                for state, old in [(integers, old_i), (floats, old_f), (ub_out, old_ub), (ubf_out, old_ubf)]:
                    check_atomics(state, old)
                    checks += 1
                src = torch.full((1, 128), -777, dtype=torch.int32)
                dst, copied = src.clone(), src.clone()
                published = src.clone()
                sync(src, dst, copied, published)
                torch.testing.assert_close(
                    dst.flatten(), 2 * ((torch.arange(128, dtype=torch.int32) + 32) % 128)
                )
                torch.testing.assert_close(copied.flatten(), torch.arange(128, dtype=torch.int32))
                checks += 1
    (output / "result.json").write_text(json.dumps({"checks": checks}))


@pytest.mark.parametrize("onboard", [False, True], ids=["build", "persistent"])
def test_simt_atomics(tmp_path, onboard):
    gate = "PYPTO_RUN_VPTO_ONBOARD" if onboard else "PYPTO_RUN_VPTO_VF_VALIDATION"
    if os.environ.get(gate) != "1":
        pytest.skip(f"Opt in with {gate}=1")
    command = [sys.executable, Path(__file__).resolve(), "--worker", tmp_path]
    if onboard:
        command.append(os.environ["PYPTO_VPTO_DEVICE_ID"])
    support = runpy.run_path(str(ROOT / "tests/ut/backend/vpto_vf_validation/test_validation.py"))
    support["run"](tmp_path, "atomics", command, timeout=int(os.environ.get("PYPTO_VPTO_TIMEOUT", "600")))
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["checks"] == (15 if onboard else 0)


if __name__ == "__main__":
    if len(sys.argv) in (3, 4) and sys.argv[1] == "--worker":
        acceptance(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) == 4 else None)
    else:
        raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
