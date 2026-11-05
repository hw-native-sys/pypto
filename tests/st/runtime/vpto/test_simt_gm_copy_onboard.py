# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""P0 native SIMT GM copy through production device linking and persistent A5.

The private source artifact substitutes a native PTO fixture for the scheduled
copy kernel. General SimtLaunchStmt lowering remains a separate milestone.
"""

import json
import os
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
from pypto.runtime._artifact_sources import (
    package_generated_sources,
    read_kernel_config,
    validate_generated_sources,
)
from pypto.runtime.elf_parser import extract_text_section
from pypto.runtime.runtime_pin import check_runtime_pin

pytestmark = pytest.mark.platforms("a5", reason="Native SIMT requires the A5 AIV backend")
ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/ut/backend/vpto_vf_validation"
LENGTHS = (0, 1, 31, 32, 33, 127, 128)
REPEATS = 3


def _prepare(directory: Path) -> CompiledProgram:
    @pl.jit.incore
    def body(
        src: pl.Tensor[[1, 128], pl.FP32],
        dst: pl.InOut[pl.Tensor[[1, 128], pl.FP32]],
        n: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 128], pl.FP32]:
        extent = pl.cast(n, pl.INDEX)
        value = pl.load(src, [0, 0], [1, 128], valid_shape=[1, extent])
        pl.store(value, [0, 0], dst)
        return dst

    @pl.jit
    def dispatch(
        src: pl.Tensor[[1, 192], pl.FP32],
        dst: pl.InOut[pl.Tensor[[1, 192], pl.FP32]],
        n: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 192], pl.FP32]:
        a = pl.slice(src, [1, 128], [0, 32])
        b = pl.slice(dst, [1, 128], [0, 32])
        result = body(a, b, n)
        dst = pl.assemble(dst, result, [0, 32])
        return dst

    compiled = dispatch.compile(
        config=RunConfig(
            backend_type=BackendType.Ascend950,
            execution_mode=ExecutionMode.ONBOARD,
            cache_config=CacheConfig(enabled=False),
            save_kernels_dir=str(directory),
        ),
    )
    kernels = read_kernel_config(directory / "kernel_config.py").KERNELS
    assert len(kernels) == 1 and kernels[0]["core_type"] == "aiv"
    kernel = kernels[0]
    assert [direction.name for direction in kernel["signature"]] == ["IN", "INOUT"]
    assert "args[2]" in Path(kernel["source"]).read_text()
    native = runpy.run_path(str(FIXTURE / "simt_validation.py"))
    Path(kernel["source"]).write_text(native["make_wrapper"]())
    with (directory / "kernel_config.py").open("a") as stream:
        stream.write(f"\nKERNELS[0]['vpto_sources'] = {[str(FIXTURE / 'simt_gm_copy.pto')]!r}\n")
    package_generated_sources(directory, BuildKind.SINGLE_CHIP)
    return compiled


def _variants(output: Path):
    assert os.environ.get("PYPTO_SKIP_RUNTIME_PIN_CHECK", "0") == "0"
    pin = check_runtime_pin(checkout_root=ROOT)
    assert pin.matches and pin.installed == pin.pinned, pin
    directory = output / "original"
    compiled = _prepare(directory)
    record = runpy.run_path(str(ROOT / "examples/runtime/vf_emitc_vpto.py"))["_record_compilation"]
    native = runpy.run_path(str(FIXTURE / "simt_validation.py"))
    with record(output) as calls:
        compiled.load()
        assert len(calls) == 1
        native["check_native_image"](output / "compilation-1")
        binaries = list((directory / "cache").glob("incore_*.bin"))
        assert len(binaries) == 1
        text = extract_text_section((output / "compilation-1/kernel.elf").read_bytes())
        assert binaries[0].read_bytes() == text, "CoreCallable must consume the linked SIMT .text"
        yield "cold", compiled
        warm = CompiledProgram.from_dir(directory)
        warm.load()
        assert len(calls) == 1, "Warm load must reuse the SIMT binary cache"
        yield "warm", warm
        relocated = output / "relocated"
        directory.rename(relocated)
        validate_generated_sources(relocated, BuildKind.SINGLE_CHIP)
        moved = CompiledProgram.from_dir(relocated)
        moved.load()
        assert len(calls) == 1, "Relocation must preserve the helper and binary cache"
        yield "relocated", moved
    (output / "build-result.json").write_text(
        json.dumps({"runtime": pin.installed, "compile_calls": len(calls), "stages": 3}, indent=2)
    )


def _numerics(worker: ChipWorker, compiled: CompiledProgram, stage: str) -> int:
    handle = worker.register(compiled)
    indices = torch.arange(192, dtype=torch.float32).reshape(1, 192)
    for repeat in range(REPEATS):
        src = indices * 0.25 - 4 + repeat
        for n in LENGTHS:
            dst = torch.full_like(src, -777 - repeat)
            expected = dst.clone()
            expected[:, 32 : 32 + n] = src[:, 32 : 32 + n]
            handle(src, dst, n)
            # Includes both guards and all inactive positions, including n=0.
            torch.testing.assert_close(dst, expected, rtol=0, atol=0)
            print(f"PASS SIMT stage={stage} repeat={repeat} n={n} offset=32 guards=32", flush=True)
    return REPEATS * len(LENGTHS)


def _acceptance(root: Path, device: int | None) -> None:
    output = Path(tempfile.mkdtemp(prefix="simt-", dir=root))
    variants = _variants(output)
    checks = 0
    if device is None:
        for stage, _ in variants:
            print(f"PASS SIMT build stage={stage}; hardware not executed", flush=True)
    else:
        stage, compiled = next(variants)
        config = RunConfig(
            backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=device
        )
        with ChipWorker(config=config, runtime=compiled.runtime_name) as worker:
            checks += _numerics(worker, compiled, stage)
            for stage, compiled in variants:
                checks += _numerics(worker, compiled, stage)
    report = {
        "device_id": device,
        "hardware_executed": device is not None,
        "numerical_checks": checks,
        "run_directory": str(output),
    }
    (root / "result.json").write_text(json.dumps(report, indent=2))


@pytest.mark.parametrize("onboard", [False, True], ids=["build", "persistent"])
def test_simt_gm_copy(onboard: bool) -> None:
    gate = "PYPTO_RUN_VPTO_ONBOARD" if onboard else "PYPTO_RUN_VPTO_VF_VALIDATION"
    if os.environ.get(gate) != "1":
        pytest.skip(f"Opt in with {gate}=1")
    device = int(os.environ["PYPTO_VPTO_DEVICE_ID"]) if onboard else None
    if onboard:
        assert device is not None and device >= 0
        for key in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
            assert not any(
                token in os.environ.get(key, "").lower() for token in ("simulator", "camodel", "msopprof")
            )
    output = Path(os.environ.get("PYPTO_VPTO_RUNTIME_OUTPUT", ROOT / "build/simt-runtime")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, Path(__file__).resolve(), "--worker", output]
    if device is not None:
        command.append(str(device))
    support = runpy.run_path(str(FIXTURE / "test_validation.py"))
    support["run"](output, "simt-runtime", command, timeout=int(os.environ.get("PYPTO_VPTO_TIMEOUT", "600")))
    result = json.loads((output / "result.json").read_text())
    assert result["hardware_executed"] == onboard
    assert result["numerical_checks"] == (3 * REPEATS * len(LENGTHS) if onboard else 0)


if __name__ == "__main__":
    if len(sys.argv) in (3, 4) and sys.argv[1] == "--worker":
        _acceptance(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) == 4 else None)
    else:
        raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
