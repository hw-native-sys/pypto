# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Mixed Cube -> dual-AIV RegScope through generated bundles and persistent Worker.

CPU preparation and device replay are separate, so the device allocation never
contains a compiler invocation. Three iterations reuse the C2V slots and each
dispatch changes the inputs; both AIV lanes must produce their own forty rows.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from pypto import CacheConfig
from pypto.backend import BackendType
from pypto.ir.compiled_program import CompiledProgram
from pypto.runtime import ChipWorker, ExecutionMode, RunConfig
from pypto.runtime._artifact_sources import read_kernel_config
from pypto.runtime.kernel_compiler import KernelCompiler

pytestmark = pytest.mark.platforms("a5", reason="Mixed register helpers require the A5 backend")


@pl.jit(auto_scope=False)
def mixed_epilogue(
    lhs: pl.Tensor[[80, 32], pl.BF16],
    rhs: pl.Tensor[[32, 128], pl.BF16],
    out: pl.InOut[pl.Tensor[[240, 128], pl.FP32]],
    bias: pl.Scalar[pl.FP32],
):
    with pl.spmd(1, name_hint="mixed_reg"):
        worker = pl.tile.get_block_idx()
        for iteration in pl.range(3):
            a = pl.tile.load(lhs, [worker * 80, 0], [80, 32], target_memory=pl.MemorySpace.Mat)
            b = pl.tile.load(rhs, [0, 0], [32, 128], target_memory=pl.MemorySpace.Mat)
            acc = pl.tile.matmul(a, b)
            for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                part = pl.aiv_shard(acc)
                flat = pl.reshape(part, [40 * 128])
                with pl.reg_scope():
                    mask = pl.reg.full_mask(lanes=64)
                    for chunk in pl.hardware_loop(40 * 128 // 64):
                        value = pl.reg.load(flat, chunk * 64, lanes=64)
                        changed = pl.reg.adds(value, bias, mask)
                        _written = pl.reg.store(flat, chunk * 64, changed)
                # A returned store alias must retain both the popped address
                # and its lifetime through the later GM transfer.
                with pl.reg_scope():
                    first = pl.reg.load(flat, 0, lanes=64)
                    stored = pl.reg.store(flat, 0, first)
                result = pl.reshape(stored, [40, 128])
                row = iteration * 80 + lane * 40
                out = pl.tile.store(result, [row, 0], out)
    return out


def _files(directory: Path) -> dict[str, str]:
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.suffix in {".cpp", ".pto", ".bin", ".so"}
    }


def prepare(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    compiled = mixed_epilogue.compile(
        config=RunConfig(
            backend_type=BackendType.Ascend950,
            execution_mode=ExecutionMode.ONBOARD,
            cache_config=CacheConfig(enabled=False),
            save_kernels_dir=str(directory),
            dump_passes=True,
        )
    )
    kernels = read_kernel_config(directory / "kernel_config.py").KERNELS
    assert {k["core_type"] for k in kernels} == {"aic", "aiv"}
    for kernel in kernels:
        assert bool(kernel.get("vpto_sources")) == (kernel["core_type"] == "aiv"), kernel
    compiled.load()
    # Prepare deterministic integral operands whose products and sums are exact
    # in FP32, so transport and scheduling are checked without a tolerance.
    lhs = torch.arange(80 * 32).reshape(80, 32).remainder(7).sub(3).to(torch.bfloat16)
    rhs = torch.arange(32 * 128).reshape(32, 128).remainder(5).sub(2).to(torch.bfloat16)
    torch.save({"lhs": lhs, "rhs": rhs}, directory / "inputs.pt")
    (directory / "seal.json").write_text(json.dumps(_files(directory), indent=2) + "\n")


def replay(directory: Path, device: int) -> None:
    sealed = json.loads((directory / "seal.json").read_text())
    assert _files(directory) == sealed
    inputs = torch.load(directory / "inputs.pt", weights_only=True)

    def no_compile(*args, **kwargs):
        raise AssertionError("Prepared mixed RegScope artifact missed the compiler cache")

    config = RunConfig(
        backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=device
    )
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(KernelCompiler, "compile_incore", no_compile)
        guard.setattr(KernelCompiler, "compile_orchestration", no_compile)
        compiled = CompiledProgram.from_dir(directory)
        compiled.load()
        with ChipWorker(config=config, runtime=compiled.runtime_name) as worker:
            handle = worker.register(compiled)
            for repeat in range(8):
                lhs = inputs["lhs"] + repeat
                rhs = inputs["rhs"] - repeat
                bias = float(repeat + 1)
                expected = (lhs.float() @ rhs.float() + bias).repeat(3, 1)
                output = torch.full((240, 128), -777.0)
                handle(lhs, rhs, output, bias)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert _files(directory) == sealed
    (directory / "device-result.json").write_text(
        json.dumps({"device": device, "dispatches": 8, "iterations": 3, "exact": True}) + "\n"
    )


def test_mixed_reg_scope_persistent_onboard():
    artifact = os.environ.get("PYPTO_MIXED_REG_ARTIFACT")
    if artifact is None or os.environ.get("PYPTO_RUN_VPTO_ONBOARD") != "1":
        pytest.skip("CPU-prepare PYPTO_MIXED_REG_ARTIFACT and opt in with PYPTO_RUN_VPTO_ONBOARD=1")
    replay(Path(artifact), int(os.environ["PYPTO_VPTO_DEVICE_ID"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--device", type=int)
    options, rest = parser.parse_known_args()
    if options.prepare:
        prepare(options.prepare.resolve())
    elif options.replay:
        assert options.device is not None
        replay(options.replay.resolve(), options.device)
    else:
        raise SystemExit(pytest.main([__file__, "-v", *rest]))
