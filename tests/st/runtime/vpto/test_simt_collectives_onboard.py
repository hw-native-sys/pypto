# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Warp collectives through public JIT, native compilation and opt-in A5 execution."""

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
from pypto.ir import CompiledProgram
from pypto.runtime import ExecutionMode, RunConfig

pytestmark = pytest.mark.platforms("a5", reason="SIMT collectives require A5")
ROOT = Path(__file__).resolve().parents[4]


@pl.jit.aiv
def collective_kernel(
    integers: pl.InOut[pl.Tensor[[11, 64], pl.INT32]],
    floats: pl.InOut[pl.Tensor[[7, 64], pl.FP32]],
    active: pl.Scalar[pl.INT32],
):
    with pl.simt_scope(dims=(64, 1, 1)):
        tid = pl.simt.thread_idx(pl.simt.Axis.X)
        lane = pl.simt.lane_id()
        value = pl.cast(tid - 40, pl.INT32)
        fp = pl.cast(value, pl.FP32)
        if lane < active:
            pred = value > 0
            pl.simt.store(integers, (0, tid), pl.cast(pl.simt.vote_all(pred), pl.INT32))
            pl.simt.store(integers, (1, tid), pl.cast(pl.simt.vote_any(pred), pl.INT32))
            pl.simt.store(integers, (2, tid), pl.cast(pl.simt.vote_uni(pred), pl.INT32))
            pl.simt.store(integers, (3, tid), pl.simt.vote_ballot(pred))
            pl.simt.store(integers, (4, tid), pl.simt.shuffle_idx(value, pl.cast(lane // 16 * 16, pl.INT32)))
            pl.simt.store(integers, (5, tid), pl.simt.shuffle_up(value, 1, width=16))
            pl.simt.store(integers, (6, tid), pl.simt.shuffle_down(value, 1, width=16))
            pl.simt.store(integers, (7, tid), pl.simt.shuffle_bfly(value, 1))
            pl.simt.store(integers, (8, tid), pl.simt.redux_add(value))
            pl.simt.store(integers, (9, tid), pl.simt.redux_min(value))
            pl.simt.store(integers, (10, tid), pl.simt.redux_max(value))
            pl.simt.store(floats, (0, tid), pl.simt.shuffle_idx(fp, 0, width=16))
            pl.simt.store(floats, (1, tid), pl.simt.shuffle_up(fp, 1))
            pl.simt.store(floats, (2, tid), pl.simt.shuffle_down(fp, 1))
            pl.simt.store(floats, (3, tid), pl.simt.shuffle_bfly(fp, 1, width=16))
            pl.simt.store(floats, (4, tid), pl.simt.redux_add(fp))
            pl.simt.store(floats, (5, tid), pl.simt.redux_min(fp))
            pl.simt.store(floats, (6, tid), pl.simt.redux_max(fp))
    return integers, floats


@pl.jit
def collective_entry(
    integers: pl.InOut[pl.Tensor[[11, 64], pl.INT32]],
    floats: pl.InOut[pl.Tensor[[7, 64], pl.FP32]],
    active: pl.Scalar[pl.INT32],
):
    return collective_kernel(integers, floats, active)


@pl.jit.aiv
def while_collective_kernel(dst: pl.InOut[pl.Tensor[[64], pl.INT32]], active: pl.Scalar[pl.INT32]):
    with pl.simt_scope(dims=(64, 1, 1)):
        tid = pl.simt.thread_idx(pl.simt.Axis.X)
        lane = pl.simt.lane_id()
        if lane < active:
            n = lane
            # The collective must read the current loop-carried n on each check.
            # Even lane zero keeps iterating until the highest active lane finishes.
            while pl.simt.vote_any(n > 0):
                n = pl.cast(n - 1, pl.INT32)
            pl.simt.store(dst, (tid,), n)
    return dst


@pl.jit
def while_collective_entry(dst: pl.InOut[pl.Tensor[[64], pl.INT32]], active: pl.Scalar[pl.INT32]):
    return while_collective_kernel(dst, active)


def compile_collectives(directory: Path, *, entry=collective_entry):
    return entry.compile(
        config=RunConfig(
            backend_type=BackendType.Ascend950,
            execution_mode=ExecutionMode.ONBOARD,
            cache_config=CacheConfig(enabled=False),
            save_kernels_dir=str(directory),
            dump_passes=True,
        )
    )


def check_results(integers, floats, active):
    # Ignore only shuffle destinations whose source was inactive or outside the
    # subgroup. Every unwritten lane must retain its original sentinel.
    expected_i = torch.full_like(integers, -777)
    expected_f = torch.full_like(floats, -777)
    valid_i = torch.ones_like(integers, dtype=torch.bool)
    valid_f = torch.ones_like(floats, dtype=torch.bool)
    for warp in range(2):
        values = [warp * 32 + lane - 40 for lane in range(active)]
        predicates = [value > 0 for value in values]
        bits = sum(1 << lane for lane, pred in enumerate(predicates) if pred)
        if bits >= 1 << 31:
            bits -= 1 << 32
        for lane in range(active):
            tid = warp * 32 + lane
            expected_i[:4, tid] = torch.tensor(
                [
                    all(predicates),
                    any(predicates),
                    len(set(predicates)) == 1,
                    bits,
                ],
                dtype=torch.int32,
            )
            expected_i[8:, tid] = torch.tensor([sum(values), min(values), max(values)])
            expected_f[4:, tid] = torch.tensor([sum(values), min(values), max(values)])
            selections = [
                (expected_i, valid_i, 4, lane // 16 * 16, 32),
                (expected_i, valid_i, 5, lane - 1, 16),
                (expected_i, valid_i, 6, lane + 1, 16),
                (expected_i, valid_i, 7, lane ^ 1, 32),
                (expected_f, valid_f, 0, lane // 16 * 16, 16),
                (expected_f, valid_f, 1, lane - 1, 32),
                (expected_f, valid_f, 2, lane + 1, 32),
                (expected_f, valid_f, 3, lane ^ 1, 16),
            ]
            for expected, valid, row, source, width in selections:
                if 0 <= source < active and source // width == lane // width:
                    expected[row, tid] = values[source]
                else:
                    valid[row, tid] = False
    torch.testing.assert_close(integers[valid_i], expected_i[valid_i], rtol=0, atol=0)
    torch.testing.assert_close(floats[valid_f], expected_f[valid_f], rtol=0, atol=0)


def compile_compaction(directory: Path, use_ballot: bool):
    example = runpy.run_path(str(ROOT / "examples/advanced/04_simt_compaction.py"))
    return example["compact"].compile(
        USE_BALLOT=use_ballot,
        config=RunConfig(
            backend_type=BackendType.Ascend950,
            execution_mode=ExecutionMode.ONBOARD,
            cache_config=CacheConfig(enabled=False),
            save_kernels_dir=str(directory),
            dump_passes=True,
        ),
    )


def compaction_inputs():
    lanes = torch.arange(64)
    generator = torch.Generator().manual_seed(3404)
    return (
        torch.zeros(64, dtype=torch.int32),
        torch.ones(64, dtype=torch.int32),
        (lanes % 2).to(torch.int32),
        # Both sign bits and both warp boundaries must survive ballot counting.
        ((lanes == 0) | (lanes == 31) | (lanes == 32) | (lanes == 63)).to(torch.int32),
        (lanes < 32).to(torch.int32),
        (lanes >= 32).to(torch.int32),
        # Selection means nonzero, including negative values, not flags == 1.
        torch.randint(-2, 3, (64,), generator=generator, dtype=torch.int32),
    )


def check_compaction(flags, indices, counts, ranks, valid):
    selected = (flags != 0) & (torch.arange(64) < valid)
    expected_indices = torch.full_like(indices, -777)
    expected_counts = selected.reshape(2, 32).sum(dim=1).to(torch.int32)
    expected_ranks = (
        (selected.reshape(2, 32).cumsum(dim=1) - selected.reshape(2, 32).to(torch.int64))
        .reshape(64)
        .to(torch.int32)
    )
    for warp in range(2):
        ids = torch.arange(warp * 32, (warp + 1) * 32)
        chosen = ids[selected[ids]].to(torch.int32)
        expected_indices[warp, : len(chosen)] = chosen
    torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(counts, expected_counts, rtol=0, atol=0)
    torch.testing.assert_close(ranks, expected_ranks, rtol=0, atol=0)


def compaction_acceptance(output: Path, device: int | None):
    checks = 0
    timings = {}
    artifact_file = output / "artifacts.json"
    # Prepare before acquiring a device; replay preserves the exact compiled
    # programs and avoids holding a shared card through native compilation.
    artifacts = json.loads(artifact_file.read_text()) if device is not None else {}
    for use_ballot in (True, False):
        method = "ballot" if use_ballot else "scan"
        compiled = (
            CompiledProgram.from_dir(artifacts[method])
            if device is not None
            else compile_compaction(output / method, use_ballot)
        )
        compiled.load()
        if device is None:
            artifacts[method] = str(compiled.output_dir.resolve())
            continue
        config = RunConfig(
            backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=device
        )
        with runtime.ChipWorker(config=config, runtime=compiled.runtime_name) as worker:
            handle = worker.register(compiled)
            for flags in compaction_inputs():
                original = flags.clone()
                for valid in (0, 1, 17, 31, 32, 33, 47, 63, 64):
                    indices = torch.full((2, 32), -777, dtype=torch.int32)
                    counts = torch.full((2,), -777, dtype=torch.int32)
                    ranks = torch.full((64,), -777, dtype=torch.int32)
                    handle(flags, indices, counts, ranks, valid)
                    check_compaction(flags, indices, counts, ranks, valid)
                    torch.testing.assert_close(flags, original, rtol=0, atol=0)
                    checks += 1
        rounds = int(os.environ.get("PYPTO_SIMT_COMPACTION_BENCH_ROUNDS", "0"))
        if rounds:
            # Reuse the same sparse input for both methods. Device wall includes
            # orchestration/launch, and must not be described as kernel latency.
            flags = compaction_inputs()[-1]
            stats = runtime.benchmark(
                compiled, (flags, indices, counts, ranks, 64), rounds=rounds, warmup=3, config=config
            )
            timings[method] = {"device_wall_us": stats.samples, "median_us": stats.device_us_median}
    if device is None:
        artifact_file.write_text(json.dumps(artifacts))
    (output / "result.json").write_text(json.dumps({"checks": checks, "timings": timings}))


def acceptance(output: Path, device: int | None):
    compiled = compile_collectives(output)
    compiled.load()
    loop = compile_collectives(output / "while", entry=while_collective_entry)
    loop.load()
    checks = 0
    while_checks = 0
    if device is not None:
        config = RunConfig(
            backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=device
        )
        with runtime.ChipWorker(config=config, runtime=compiled.runtime_name) as worker:
            handle = worker.register(compiled)
            loop_handle = worker.register(loop)
            for active in (0, 1, 15, 16, 17, 31, 32):
                integers = torch.full((11, 64), -777, dtype=torch.int32)
                floats = torch.full((7, 64), -777, dtype=torch.float32)
                handle(integers, floats, active)
                check_results(integers, floats, active)
                checks += 1
                dst = torch.full((64,), -777, dtype=torch.int32)
                loop_handle(dst, active)
                expected = torch.full_like(dst, -777)
                for warp in range(2):
                    expected[warp * 32 : warp * 32 + active] = torch.arange(active) - (active - 1)
                torch.testing.assert_close(dst, expected, rtol=0, atol=0)
                while_checks += 1
    (output / "result.json").write_text(json.dumps({"checks": checks, "while_checks": while_checks}))


@pytest.mark.parametrize("onboard", [False, True], ids=["build", "persistent"])
def test_simt_collectives(tmp_path, onboard):
    gate = "PYPTO_RUN_VPTO_ONBOARD" if onboard else "PYPTO_RUN_VPTO_VF_VALIDATION"
    if os.environ.get(gate) != "1":
        pytest.skip(f"Opt in with {gate}=1")
    command = [sys.executable, Path(__file__).resolve(), "--worker", tmp_path]
    if onboard:
        command.append(os.environ["PYPTO_VPTO_DEVICE_ID"])
    support = runpy.run_path(str(ROOT / "tests/ut/backend/vpto_vf_validation/test_validation.py"))
    support["run"](tmp_path, "collectives", command, timeout=int(os.environ.get("PYPTO_VPTO_TIMEOUT", "600")))
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["checks"] == (7 if onboard else 0)
    assert result["while_checks"] == (7 if onboard else 0)


@pytest.mark.parametrize("onboard", [False, True], ids=["build", "persistent"])
def test_simt_compaction(tmp_path, onboard):
    gate = "PYPTO_RUN_VPTO_ONBOARD" if onboard else "PYPTO_RUN_VPTO_VF_VALIDATION"
    if os.environ.get(gate) != "1":
        pytest.skip(f"Opt in with {gate}=1")
    support = runpy.run_path(str(ROOT / "tests/ut/backend/vpto_vf_validation/test_validation.py"))
    command = [sys.executable, Path(__file__).resolve(), "--compaction-worker", tmp_path]
    timeout = int(os.environ.get("PYPTO_VPTO_TIMEOUT", "600"))
    support["run"](tmp_path, "compaction-build", command, timeout=timeout)
    if onboard:
        command.append(os.environ["PYPTO_VPTO_DEVICE_ID"])
        support["run"](tmp_path, "compaction-onboard", command, timeout=timeout)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["checks"] == (126 if onboard else 0)


if __name__ == "__main__":
    if len(sys.argv) in (3, 4) and sys.argv[1] == "--compaction-worker":
        compaction_acceptance(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) == 4 else None)
    elif len(sys.argv) in (3, 4) and sys.argv[1] == "--worker":
        acceptance(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) == 4 else None)
    else:
        raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
