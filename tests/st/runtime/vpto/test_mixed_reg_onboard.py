# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Production mixed Group VF packaging, native cache and persistent A5 execution.

The BF16 cast materializes local UB after receiving the Cube result. Direct
FIFO capture remains outside this test's supported subset. Run through
task-submit with PYPTO_RUN_VPTO_ONBOARD=1, or use --build for compiler evidence.
"""

import json
import os
import re
import runpy
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
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

ROOT = Path(__file__).resolve().parents[4]
SUPPORT = ROOT / "tests/ut/backend/vpto_vf_validation/test_validation.py"
pytestmark = pytest.mark.platforms("a5", reason="Grouped VF helpers require A5")


@pl.jit
def mixed_bf16(
    a: pl.Tensor[[32, 32], pl.BF16],
    b: pl.Tensor[[32, 32], pl.BF16],
    dst: pl.Out[pl.Tensor[[32, 32], pl.BF16]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mixed_reg"):
        lhs = pl.load(a, [0, 0], [32, 32])
        rhs = pl.load(b, [0, 0], [32, 32])
        product = pl.matmul(lhs, rhs)
        rounded = pl.cast(product, pl.BF16, mode="rint")
        vector = pl.tile.move(rounded, target_memory=pl.MemorySpace.Vec)
        flat = pl.reshape(vector, [1024])
        with pl.reg_scope():
            value = pl.reg.load(flat, lanes=64)
            flat = pl.reg.store(flat, value=pl.reg.add(value, value))
        dst = pl.store(pl.reshape(flat, [32, 32]), [0, 0], dst)
    return dst


@pl.jit
def mixed_bf16_dual(
    a: pl.Tensor[[32, 32], pl.BF16],
    b: pl.Tensor[[32, 32], pl.BF16],
    dst: pl.Out[pl.Tensor[[32, 32], pl.BF16]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mixed_reg_dual"):
        lhs = pl.load(a, [0, 0], [32, 32])
        rhs = pl.load(b, [0, 0], [32, 32])
        product = pl.matmul(lhs, rhs)
        for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            received = pl.aiv_shard(product)
            rounded = pl.cast(received, pl.BF16, mode="rint")
            flat = pl.reshape(rounded, [512])
            with pl.reg_scope():
                value = pl.reg.load(flat, lanes=64)
                flat = pl.reg.store(flat, value=pl.reg.add(value, value))
            # Two helpers exercise per-entry lists and ordering on both lanes.
            with pl.reg_scope():
                value = pl.reg.load(flat, offset=64, lanes=64)
                flat = pl.reg.store(flat, offset=64, value=pl.reg.mul(value, value))
            dst = pl.store(pl.reshape(flat, [16, 32]), [lane * 16, 0], dst)
    return dst


def _build_variants(output: Path, dual: bool):
    from pypto.runtime import device_runner  # noqa: PLC0415
    from pypto.runtime.kernel_compiler import KernelCompiler  # noqa: PLC0415

    assert os.environ.get("PYPTO_SKIP_RUNTIME_PIN_CHECK", "0") == "0"
    pin = check_runtime_pin(checkout_root=ROOT)
    assert pin.matches and pin.pinned == pin.installed, pin
    output.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=output))
    directory = run_dir / "original"
    kernel = mixed_bf16_dual if dual else mixed_bf16
    compiled = kernel.compile(
        config=RunConfig(
            backend_type=BackendType.Ascend950,
            execution_mode=ExecutionMode.ONBOARD,
            cache_config=CacheConfig(enabled=False),
            save_kernels_dir=str(directory),
            dump_passes=True,
        )
    )
    config = read_kernel_config(directory / "kernel_config.py")
    assert len(config.KERNELS) == 2
    cube = next(k for k in config.KERNELS if k["core_type"] == "aic")
    vector = next(k for k in config.KERNELS if k["core_type"] == "aiv")
    assert not cube.get("vpto_sources")
    helpers = vector["vpto_sources"]
    assert len(helpers) == (2 if dual else 1)
    manifest = json.loads(Path(helpers[0]).with_name("manifest.json").read_text())
    assert manifest["schema_version"] == 4
    assert manifest["entry_helpers"][cube["code_name"]] == []
    assert manifest["entry_helpers"][vector["code_name"]] == [Path(p).stem for p in helpers]
    support = runpy.run_path(str(SUPPORT))
    original_compile = KernelCompiler.compile_incore
    calls = []

    def observe_compile(self, *args, **kwargs):
        core = kwargs["core_type"]
        assert bool(kwargs.get("vpto_sources")) == (core == "aiv")
        binary = original_compile(self, *args, **kwargs)
        elf = run_dir / f"{len(calls)}-{core}.elf"
        elf.write_bytes(binary)
        symbols = support["run"](run_dir, f"{len(calls)}-symbols", ["readelf", "-Ws", elf])
        relocs = support["run"](run_dir, f"{len(calls)}-relocs", ["readelf", "-r", elf])
        assert not re.search(r"\bUND[ \t]+\S", symbols), symbols
        assert "There are no relocations" in relocs
        assert ("__pypto_vf_" in symbols) == (core == "aiv")
        calls.append(core)
        return binary

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(KernelCompiler, "compile_incore", observe_compile)
        patch.setattr(device_runner, "ThreadPoolExecutor", lambda max_workers: ThreadPoolExecutor(1))
        compiled.load()
        assert sorted(calls) == ["aic", "aiv"]
        yield "cold", compiled
        warm = CompiledProgram.from_dir(directory)
        warm.load()
        assert len(calls) == 2, "Warm load must reuse both native binaries"
        yield "warm", warm
        # Deliberate cache-invalidation probe only. Cold/warm execute the
        # original compiler output; changed/relocated must now zero 64 lanes.
        helper = Path(helpers[0])
        source = helper.read_text()
        assert source.count("pto.vmi.vadd") == 1
        helper.write_text(source.replace("pto.vmi.vadd", "pto.vmi.vsub"))
        changed = CompiledProgram.from_dir(directory)
        changed.load()
        assert calls.count("aic") == 1 and calls.count("aiv") == 2
        yield "changed", changed
        relocated = run_dir / "relocated"
        directory.rename(relocated)
        validate_generated_sources(relocated, BuildKind.SINGLE_CHIP)
        moved = CompiledProgram.from_dir(relocated)
        moved.load()
        assert len(calls) == 3, "Relocation must preserve the valid binary cache"
        yield "relocated", moved
    report = {"dual": dual, "compile_calls": calls, "runtime": pin.installed, "run_directory": str(run_dir)}
    (output / "build-result.json").write_text(json.dumps(report, indent=2) + "\n")


def _acceptance(output: Path, device: int, dual: bool) -> None:
    config = RunConfig(
        backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=device
    )
    variants = _build_variants(output, dual)
    first = next(variants)
    index = torch.arange(1024, dtype=torch.float32).reshape(32, 32)
    checks = []

    def check(worker, stage, compiled):
        handle = worker.register(compiled)
        for repeat in range(8):
            a = ((index.remainder(7) - 3 + repeat % 3) * 0.125).to(torch.bfloat16)
            b = ((index.remainder(11) - 5 - repeat % 2) * 0.0625).to(torch.bfloat16)
            expected = (a.float() @ b.float()).to(torch.bfloat16)
            chunks = expected.reshape(2 if dual else 1, -1)
            chunks[:, :64] = 0 if stage in ("changed", "relocated") else chunks[:, :64] * 2
            if dual:
                chunks[:, 64:128] = chunks[:, 64:128] * chunks[:, 64:128]
            actual = torch.full_like(a, -123)
            handle(a, b, actual)
            # Check all 1024 values, including untouched regions on both lanes.
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            checks.append({"stage": stage, "repeat": repeat, "elements": 1024})

    with ChipWorker(config=config, runtime=first[1].runtime_name) as worker:
        check(worker, *first)
        for stage, compiled in variants:
            check(worker, stage, compiled)
    assert len(checks) == 32
    (output / "result.json").write_text(
        json.dumps({"device_id": device, "dual": dual, "hardware_executed": True, "checks": checks}, indent=2)
        + "\n"
    )


@pytest.mark.parametrize("dual", [False, True], ids=["single-lane", "dual-lane"])
def test_mixed_reg_persistent_onboard(dual):
    if os.environ.get("PYPTO_RUN_VPTO_ONBOARD") != "1":
        pytest.skip("Opt in through task-submit with PYPTO_RUN_VPTO_ONBOARD=1")
    device = int(os.environ["PYPTO_VPTO_DEVICE_ID"])
    timeout = int(os.environ.get("PYPTO_VPTO_TIMEOUT", "600"))
    output = Path(os.environ["PYPTO_VPTO_RUNTIME_OUTPUT"]).resolve()
    runpy.run_path(str(SUPPORT))["run"](
        output,
        "mixed-persistent-runtime",
        [sys.executable, Path(__file__).resolve(), "--worker", output, device, int(dual)],
        timeout=timeout,
    )
    assert len(json.loads((output / "result.json").read_text())["checks"]) == 32


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        _acceptance(Path(sys.argv[2]), int(sys.argv[3]), bool(int(sys.argv[4])))
    elif len(sys.argv) == 4 and sys.argv[1] == "--build":
        for stage, _ in _build_variants(Path(sys.argv[2]), bool(int(sys.argv[3]))):
            print(f"PASS grouped VF native build: {stage}", flush=True)
    else:
        raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
