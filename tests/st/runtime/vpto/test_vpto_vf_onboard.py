# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Opt-in production .text -> CoreCallable -> persistent Worker A5 acceptance.

Run with --platform=a5, PYPTO_RUN_VPTO_ONBOARD=1,
PYPTO_VPTO_DEVICE_ID=<verified idle card>, and PYPTO_VPTO_TIMEOUT=<seconds>.
The subprocess bounds compilation, device initialization, dispatch, and teardown.
No simulator fallback or runtime-pin exemption is permitted.
"""

import hashlib
import importlib
import json
import os
import runpy
import shlex
import shutil
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
from pypto.runtime.runtime_pin import check_runtime_pin

pytestmark = pytest.mark.platforms("a5", reason="VPTO helpers require the A5 AIV backend")

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/ut/backend/vpto_vf_validation"
LENGTHS = (0, 1, 63, 64, 65, 127, 128, 1023, 1024)


def _prepare(directory: Path):
    @pl.jit.incore
    def body(
        lhs: pl.Tensor[[1, 1024], pl.FP32],
        rhs: pl.Tensor[[1, 1024], pl.FP32],
        out: pl.InOut[pl.Tensor[[1, 1024], pl.FP32]],
        n: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 1024], pl.FP32]:
        extent = pl.cast(n, pl.INDEX)
        a = pl.load(lhs, [0, 0], [1, 1024], valid_shape=[1, extent])
        b = pl.load(rhs, [0, 0], [1, 1024], valid_shape=[1, extent])
        pl.store(pl.add(a, b), [0, 0], out)
        return out

    @pl.jit
    def dispatch(
        lhs: pl.Tensor[[1, 1088], pl.FP32],
        rhs: pl.Tensor[[1, 1088], pl.FP32],
        out: pl.InOut[pl.Tensor[[1, 1088], pl.FP32]],
        n: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 1088], pl.FP32]:
        a = pl.slice(lhs, [1, 1024], [0, 32])
        b = pl.slice(rhs, [1, 1024], [0, 32])
        c = pl.slice(out, [1, 1024], [0, 32])
        result = body(a, b, c, n)
        out = pl.assemble(out, result, [0, 32])
        return out

    compiled = dispatch.compile(
        config=RunConfig(
            backend_type=BackendType.Ascend950,
            execution_mode=ExecutionMode.ONBOARD,
            cache_config=CacheConfig(enabled=False),
            save_kernels_dir=str(directory),
        ),
    )
    # Modify only this private generated tree before packaging or runtime loading.
    config = read_kernel_config(directory / "kernel_config.py")
    assert len(config.KERNELS) == 1, config.KERNELS
    kernel = config.KERNELS[0]
    assert kernel["core_type"] == "aiv"
    # Runtime signatures list tensor directions only; scalar n occupies args[3].
    assert [direction.name for direction in kernel["signature"]] == ["IN", "IN", "INOUT"]
    original_wrapper = Path(kernel["source"]).read_text()
    assert "args[3]" in original_wrapper and "int32_t" in original_wrapper
    support = runpy.run_path(str(FIXTURE / "test_validation.py"))
    ptoas = shutil.which("ptoas")
    assert ptoas, "PTOAS must be on PATH"
    support["run"](
        directory,
        "emit-caller",
        [
            ptoas,
            FIXTURE / "caller.pto",
            "--pto-arch=a5",
            "--pto-backend=emitc",
            "--pto-level=level3",
            "-o",
            directory / "caller.cpp",
        ],
    )
    Path(kernel["source"]).write_text(support["make_wrapper"]((directory / "caller.cpp").read_text()))
    # Appending to the generated Python keeps its ArgDirection objects and path resolution intact.
    with (directory / "kernel_config.py").open("a") as stream:
        stream.write(f"\nKERNELS[0]['vpto_sources'] = {[str(FIXTURE / 'helper.pto')]!r}\n")
    package_generated_sources(directory, BuildKind.SINGLE_CHIP)
    return compiled


def _numerics(worker, compiled, *, subtract: bool, stage: str) -> None:
    handle = worker.register(compiled)
    indices = torch.arange(1088, dtype=torch.float32).reshape(1, 1088)
    for repeat in range(3):
        lhs = indices.remainder(31) * 0.25 - 4 + repeat
        rhs = indices.remainder(17) * 0.5 + 2 - 2 * repeat
        for n in LENGTHS:
            output = torch.full_like(lhs, -777)
            expected = output.clone()
            values = lhs - rhs if subtract else lhs + rhs
            expected[:, 32 : 32 + n] = values[:, 32 : 32 + n]
            # Reuse one Worker and registration across every length and input change.
            handle(lhs, rhs, output, n)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            print(f"PASS stage={stage} repeat={repeat} n={n} offset=32 guards=32", flush=True)


def _build_variants(output: Path, prepare=_prepare):
    # CPU-only collection/import checks do not require the optional runtime SDK.
    compiler_class = importlib.import_module("pypto.runtime.kernel_compiler").KernelCompiler

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evidence_root = output
    # Preserve every invocation, including direct --build/--worker runs that
    # bypass pytest fixtures. A prior relocated artifact must never be replaced.
    output = Path(tempfile.mkdtemp(prefix="run-", dir=evidence_root))
    assert os.environ.get("PYPTO_SKIP_RUNTIME_PIN_CHECK", "0") == "0"
    pin = check_runtime_pin(checkout_root=ROOT)
    assert pin.matches and pin.pinned == pin.installed, pin
    print(f"runtime={pin.installed} extension={pin.extension_path}", flush=True)
    directory = output / "original"
    compiled = prepare(directory)
    original_compile = compiler_class.compile_incore
    original_run = compiler_class._run
    calls = []

    def counted_compile(self, *args, **kwargs):
        # Observe real production compilation; no replacement binary or runtime stub.
        assert kwargs.get("vpto_sources"), "Helper lost before production compilation"
        evidence = output / f"compilation-{len(calls) + 1}"
        evidence.mkdir(parents=True, exist_ok=True)
        inputs = {"caller.cpp": Path(args[0])}
        inputs.update({f"helper-{i}.pto": Path(p) for i, p in enumerate(kwargs["vpto_sources"])})
        hashes = {}
        # Snapshot before the cache-invalidation mutation and artifact move.
        # These are the actual inputs, not sources reconstructed after the run.
        for name, source in inputs.items():
            data = source.read_bytes()
            (evidence / name).write_bytes(data)
            hashes[name] = {"source": str(source), "sha256": hashlib.sha256(data).hexdigest()}
        (evidence / "inputs.json").write_text(json.dumps(hashes, indent=2) + "\n")
        binary = original_compile(self, *args, **kwargs)
        calls.append(str(args[0]))
        return binary

    def recorded_run(self, command, path, label):
        # Copy outputs from this compilation only, before its temporary directory
        # is cleaned. The real compiler still produces every returned byte.
        evidence = output / f"compilation-{len(calls) + 1}"
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / f"{path.name}.command.txt").write_text(shlex.join(command) + "\n")
        binary = original_run(self, command, path, label)
        (evidence / path.name).write_bytes(binary)
        return binary

    with pytest.MonkeyPatch.context() as observer:
        observer.setattr(compiler_class, "compile_incore", counted_compile)
        observer.setattr(compiler_class, "_run", recorded_run)
        compiled.load()
        assert len(calls) == 1
        binaries = list((directory / "cache").glob("incore_*.bin"))
        assert len(binaries) == 1, binaries
        before = binaries[0].read_bytes()
        assert before and not before.startswith(b"\x7fELF"), "CoreCallable must receive extracted .text"
        (output / "cold.text.bin").write_bytes(before)
        yield "cold", compiled
        warm = CompiledProgram.from_dir(directory)
        warm.load()
        assert len(calls) == 1, "Warm load compiled instead of using the binary cache"
        yield "warm", warm
        helper = Path(read_kernel_config(directory / "kernel_config.py").KERNELS[0]["vpto_sources"][0])
        text = helper.read_text()
        opcode = "pto.vmi.vadd" if "pto.vmi.vadd" in text else "pto.vadd"
        assert opcode in text
        helper.write_text(text.replace(opcode, opcode.replace("vadd", "vsub")))
        changed = CompiledProgram.from_dir(directory)
        changed.load()
        assert len(calls) == 2, "Changing helper contents must invalidate the binary cache"
        after_files = list((directory / "cache").glob("incore_*.bin"))
        assert any(path.read_bytes() != before for path in after_files)
        yield "changed", changed
        # Move, rather than copy: stale absolute paths cannot keep working by accident.
        relocated = output / "relocated"
        directory.rename(relocated)
        validate_generated_sources(relocated, BuildKind.SINGLE_CHIP)
        moved = CompiledProgram.from_dir(relocated)
        moved.load()
        assert len(calls) == 2, "Relocated artifact should retain its valid binary cache"
        yield "relocated", moved
    report = json.dumps(
        {
            "runtime": pin.installed,
            "compile_calls": len(calls),
            "stages": ["cold", "warm", "changed", "relocated"],
            "evidence": "compiler_cache_and_relocation",
            "run_directory": str(output),
        },
        indent=2,
    )
    (output / "build-result.json").write_text(report)
    (evidence_root / "build-result.json").write_text(report)


def _acceptance(output: Path, device_id: int, prepare=_prepare) -> None:
    config = RunConfig(
        backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=device_id
    )
    variants = _build_variants(output, prepare)
    # Build before opening a device; one persistent worker serves all four stages.
    stage, compiled = next(variants)
    with ChipWorker(config=config, runtime=compiled.runtime_name) as worker:
        _numerics(worker, compiled, subtract=False, stage=stage)
        for stage, compiled in variants:
            _numerics(worker, compiled, subtract=stage in ("changed", "relocated"), stage=stage)
    (output / "result.json").write_text(
        json.dumps(
            {
                "device_id": device_id,
                "numerical_checks": 108,
                "hardware_executed": True,
            },
            indent=2,
        )
    )


def test_production_build_cache_relocation() -> None:
    if os.environ.get("PYPTO_RUN_VPTO_VF_VALIDATION") != "1":
        pytest.skip("Opt in with PYPTO_RUN_VPTO_VF_VALIDATION=1")
    output = Path(os.environ.get("PYPTO_VPTO_BUILD_OUTPUT", ROOT / "build/vpto-investigation/production"))
    output.mkdir(parents=True, exist_ok=True)
    support = runpy.run_path(str(FIXTURE / "test_validation.py"))
    support["run"](
        output.resolve(),
        "production-build",
        [
            sys.executable,
            Path(__file__).resolve(),
            "--build",
            str(output.resolve()),
        ],
        timeout=int(os.environ.get("PYPTO_VPTO_TIMEOUT", "600")),
    )
    assert json.loads((output / "build-result.json").read_text())["compile_calls"] == 2


def test_persistent_vpto_onboard() -> None:
    if os.environ.get("PYPTO_RUN_VPTO_ONBOARD") != "1":
        pytest.skip("Opt in with PYPTO_RUN_VPTO_ONBOARD=1 and PYPTO_VPTO_DEVICE_ID")
    device_id = int(os.environ["PYPTO_VPTO_DEVICE_ID"])
    timeout = int(os.environ.get("PYPTO_VPTO_TIMEOUT", "600"))
    assert device_id >= 0 and timeout > 0
    assert os.environ.get("PYPTO_SKIP_RUNTIME_PIN_CHECK", "0") == "0"
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
        value = os.environ.get(key, "").lower()
        assert not any(token in value for token in ("simulator", "camodel", "msopprof")), (key, value)
    output = Path(os.environ.get("PYPTO_VPTO_RUNTIME_OUTPUT", ROOT / "build/vpto-investigation/runtime"))
    output.mkdir(parents=True, exist_ok=True)
    support = runpy.run_path(str(FIXTURE / "test_validation.py"))
    support["run"](
        output.resolve(),
        "persistent-runtime",
        [
            sys.executable,
            Path(__file__).resolve(),
            "--worker",
            str(output.resolve()),
            str(device_id),
        ],
        timeout=timeout,
    )
    assert json.loads((output / "result.json").read_text())["numerical_checks"] == 108


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _acceptance(Path(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) == 3 and sys.argv[1] == "--build":
        for name, _ in _build_variants(Path(sys.argv[2])):
            print(f"PASS production build stage={name}; hardware not executed", flush=True)
    else:
        raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
