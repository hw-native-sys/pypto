# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public SIMT scopes through JIT, production linking and persistent A5 execution."""

import json
import os
import re
import runpy
import sys
import tempfile
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from pypto import CacheConfig
from pypto.backend import BackendType
from pypto.ir.compiled_program import CompiledProgram
from pypto.jit._artifact_manifest import BuildKind
from pypto.runtime import ChipWorker, ExecutionMode, RunConfig
from pypto.runtime._artifact_sources import read_kernel_config, validate_generated_sources
from pypto.runtime.runtime_pin import check_runtime_pin

pytestmark = pytest.mark.platforms("a5", reason="SIMT lowering requires A5")
ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/ut/backend/vpto_vf_validation"
LENGTHS = (0, 1, 31, 32, 33, 127, 128)


@pl.jit.aiv
def copy_kernel(
    src: pl.Tensor[[1, 128], pl.FP32],
    dst: pl.InOut[pl.Tensor[[1, 128], pl.FP32]],
    snapshot: pl.Out[pl.Tensor[[1, 128], pl.FP32]],
    n: pl.Scalar[pl.INT32],
):
    with pl.simt_scope(dims=(128, 1, 1)):
        x = pl.simt.thread_idx(pl.simt.Axis.X)
        if x < n:
            value = pl.simt.load(src, [0, x])
            pl.simt.store(dst, [0, x], value)
    value = pl.load(dst, [0, 0], [1, 128])
    result = pl.store(value, [0, 0], snapshot)
    return dst, result


@pl.jit
def copy_entry(
    src: pl.Tensor[[1, 192], pl.FP32],
    dst: pl.InOut[pl.Tensor[[1, 192], pl.FP32]],
    snapshot: pl.InOut[pl.Tensor[[1, 192], pl.FP32]],
    n: pl.Scalar[pl.INT32],
):
    a = pl.slice(src, [1, 128], [0, 32])
    b = pl.slice(dst, [1, 128], [0, 32])
    c = pl.slice(snapshot, [1, 128], [0, 32])
    updated, copied = copy_kernel(a, b, c, n)
    dst = pl.assemble(dst, updated, [0, 32])
    snapshot = pl.assemble(snapshot, copied, [0, 32])
    return dst, snapshot


@pl.jit.aiv
def gather_kernel(
    src: pl.Tensor[[4, 32], pl.FP32],
    indices: pl.Tensor[[4, 32], pl.INT32],
    old: pl.Tensor[[4, 32], pl.FP32],
    dst: pl.Out[pl.Tensor[[4, 32], pl.FP32]],
    snapshot: pl.Out[pl.Tensor[[4, 32], pl.FP32]],
    n: pl.Scalar[pl.INT32],
):
    loaded = pl.load(src, [0, 0], [4, 32])
    values = pl.add(loaded, 1.0)
    offsets = pl.load(indices, [0, 0], [4, 32])
    previous = pl.load(old, [0, 0], [4, 32])
    saved_value = previous
    with pl.simt_scope(dims=(32, 4, 1)):
        x = pl.simt.thread_idx(pl.simt.Axis.X)
        y = pl.simt.thread_idx(axis=pl.simt.Axis.Y)
        linear = y * 32 + x
        if linear < n:
            index = pl.simt.load(offsets, [y, x])
            value = pl.simt.load(values, [index // 32, index % 32])
            pl.simt.store(previous, [y, x], value)
    result = pl.store(previous, [0, 0], dst)
    saved = pl.store(saved_value, [0, 0], snapshot)
    return result, saved


@pl.jit
def gather_entry(
    src: pl.Tensor[[4, 32], pl.FP32],
    indices: pl.Tensor[[4, 32], pl.INT32],
    old: pl.Tensor[[4, 32], pl.FP32],
    dst: pl.Out[pl.Tensor[[4, 32], pl.FP32]],
    snapshot: pl.Out[pl.Tensor[[4, 32], pl.FP32]],
    n: pl.Scalar[pl.INT32],
):
    result, saved = gather_kernel(src, indices, old, dst, snapshot, n)
    return result, saved


def compile_scope(directory: Path, case: str) -> CompiledProgram:
    entry = copy_entry if case == "gm" else gather_entry
    return entry.compile(
        config=RunConfig(
            backend_type=BackendType.Ascend950,
            execution_mode=ExecutionMode.ONBOARD,
            cache_config=CacheConfig(enabled=False),
            save_kernels_dir=str(directory),
            dump_passes=True,
        ),
    )


def _variants(output: Path, case: str):
    pin = check_runtime_pin(checkout_root=ROOT)
    assert pin.matches and pin.installed == pin.pinned, pin
    directory = output / "original"
    compiled = compile_scope(directory, case)
    assert isinstance(compiled, CompiledProgram)
    helpers = list((directory / "vpto").rglob("*.pto"))
    assert len(helpers) == 1
    manifest = json.loads(helpers[0].with_name("manifest.json").read_text())
    assert manifest["helpers"][0]["kind"] == "simt"
    assert manifest["effect_scope"] == "whole-storage"
    kernels = read_kernel_config(directory / "kernel_config.py").KERNELS
    assert len(kernels) == 1 and kernels[0]["core_type"] == "aiv"
    native = helpers[0].read_text()
    assert "pto.simt_entry" in native and "pto.simt_launch" in native
    assert ("pto.ldg" if case == "gm" else "pto.load") in native
    launch_targets = re.findall(r"pto\.simt_launch @([\w]+)<<<", native)
    assert len(launch_targets) == 1
    record = runpy.run_path(str(ROOT / "examples/runtime/vf_emitc_vpto.py"))["_record_compilation"]
    with record(output) as calls:
        compiled.load()
        assert len(calls) == 1
        validation = runpy.run_path(str(FIXTURE / "simt_validation.py"))
        validation["check_simt_launch_size"](output / "compilation-1", launch_targets[0])
        caller = (output / "compilation-1/caller.cpp").read_text()
        pipes = ("PIPE_V, PIPE_MTE2",) if case == "gm" else ("PIPE_MTE2, PIPE_V", "PIPE_V, PIPE_MTE3")
        for pair in pipes:
            assert f"set_flag({pair}," in caller and f"wait_flag({pair}," in caller
        yield "cold", compiled
        warm = CompiledProgram.from_dir(directory)
        warm.load()
        assert len(calls) == 1
        yield "warm", warm
        relocated = output / "relocated"
        directory.rename(relocated)
        validate_generated_sources(relocated, BuildKind.SINGLE_CHIP)
        moved = CompiledProgram.from_dir(relocated)
        moved.load()
        assert len(calls) == 1
        yield "relocated", moved


def _numerics(worker, compiled, case, stage):
    handle = worker.register(compiled)
    checks = 0
    for repeat in range(3):
        for n in LENGTHS:
            if case == "gm":
                src = torch.arange(192, dtype=torch.float32).reshape(1, 192) * 0.25 + repeat
                dst = torch.full_like(src, -777 - repeat)
                snapshot = torch.full_like(src, -999)
                expected = dst.clone()
                expected[:, 32 : 32 + n] = src[:, 32 : 32 + n]
                saved = snapshot.clone()
                saved[:, 32:160] = expected[:, 32:160]
                handle(src, dst, snapshot, n)
            else:
                src = torch.arange(128, dtype=torch.float32).reshape(4, 32) * 0.25 + repeat
                indices = ((torch.arange(128, dtype=torch.int32) * 37 + repeat) % 128).reshape(4, 32)
                old = torch.full_like(src, -777 - repeat)
                dst, snapshot = torch.zeros_like(src), torch.zeros_like(src)
                expected = old.clone()
                expected.view(-1)[:n] = src.view(-1)[indices.view(-1)[:n].long()] + 1
                saved = old.clone()
                handle(src, indices, old, dst, snapshot, n)
                torch.testing.assert_close(old, saved, rtol=0, atol=0)
            torch.testing.assert_close(dst, expected, rtol=0, atol=0)
            torch.testing.assert_close(snapshot, saved, rtol=0, atol=0)
            checks += 1
            print(f"PASS SIMT IR {case} {stage} repeat={repeat} n={n}", flush=True)
    return checks


def _acceptance(root: Path, case: str, device: int | None):
    output = Path(tempfile.mkdtemp(prefix=f"simt-ir-{case}-", dir=root))
    variants = _variants(output, case)
    checks = 0
    if device is None:
        for stage, _ in variants:
            print(f"PASS SIMT IR {case} build {stage}", flush=True)
    else:
        stage, compiled = next(variants)
        config = RunConfig(
            backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=device
        )
        with ChipWorker(config=config, runtime=compiled.runtime_name) as worker:
            checks += _numerics(worker, compiled, case, stage)
            for stage, compiled in variants:
                checks += _numerics(worker, compiled, case, stage)
    (root / f"{case}-result.json").write_text(json.dumps({"checks": checks, "directory": str(output)}))


@pytest.mark.parametrize("onboard", [False, True], ids=["build", "persistent"])
@pytest.mark.parametrize("case", ["gm", "ub"])
def test_simt_ir(case, onboard):
    gate = "PYPTO_RUN_VPTO_ONBOARD" if onboard else "PYPTO_RUN_VPTO_VF_VALIDATION"
    if os.environ.get(gate) != "1":
        pytest.skip(f"Opt in with {gate}=1")
    device = int(os.environ["PYPTO_VPTO_DEVICE_ID"]) if onboard else None
    output = Path(os.environ.get("PYPTO_VPTO_RUNTIME_OUTPUT", ROOT / "build/simt-ir-runtime")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, Path(__file__).resolve(), "--worker", output, case]
    if device is not None:
        command.append(str(device))
    support = runpy.run_path(str(FIXTURE / "test_validation.py"))
    support["run"](
        output, f"simt-ir-{case}", command, timeout=int(os.environ.get("PYPTO_VPTO_TIMEOUT", "600"))
    )
    report = json.loads((output / f"{case}-result.json").read_text())
    assert report["checks"] == (63 if onboard else 0)


if __name__ == "__main__":
    if len(sys.argv) in (4, 5) and sys.argv[1] == "--worker":
        _acceptance(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]) if len(sys.argv) == 5 else None)
    else:
        raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
