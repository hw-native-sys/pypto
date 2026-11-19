# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check native masked and omitted-mask interleave on A5.

Build with --build OUTPUT under task-submit --no-device, then execute with
--worker OUTPUT DEVICE under task-submit --device auto. Both outputs are stored
without the governing mask so ignored masks cannot hide behind masked stores.
Default lengths cover physical registers and larger vectors. Pass --lanes 8 to
run the short-vector hardware regression for PTOAS #1659 separately.
"""

import argparse
import importlib
import json
import os
import runpy
import subprocess
import sys
import traceback
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pypto.language as pl
import pytest
import torch
from pypto import CacheConfig
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.compiled_program import CompiledProgram
from pypto.runtime import ChipWorker, ExecutionMode, RunConfig
from pypto.runtime._artifact_sources import read_kernel_config
from pypto.runtime.runtime_pin import check_runtime_pin

pytestmark = pytest.mark.platforms("a5", reason="Interleave qualification requires A5")
ROOT = Path(__file__).resolve().parents[4]
DTYPES = ("INT8", "UINT8", "INT16", "UINT16", "INT32", "UINT32", "FP16", "BF16", "FP32")
VECTOR_LENGTHS = (8, 64, 128, 256)
MODES = ("masked", "omitted")
OFFSETS = (256, 640, 1024, 1408)


def _storage(dtype):
    bits = 8 if dtype.endswith("8") else 16 if dtype.endswith("16") else 32
    return getattr(pl, f"UINT{bits}"), np.dtype(f"uint{bits}")


def _program(dtype_name, lanes, omitted):
    storage, _ = _storage(dtype_name)
    dtype = getattr(pl, dtype_name)

    @pl.jit.incore
    def body(
        src: pl.Tensor[[1, 1024], storage],
        out: pl.InOut[pl.Tensor[[1, 2048], storage]],
        n: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 2048], storage]:
        a = pl.reshape(pl.load(src, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec), [1024])
        c = pl.reshape(pl.load(out, [0, 0], [1, 2048], target_memory=pl.MemorySpace.Vec), [2048])
        if not omitted:
            count = pl.cast(n, pl.INDEX)
        with pl.reg_scope():
            first = pl.reg.load(a, lanes=lanes)
            second = pl.reg.load(a, offset=512, lanes=lanes)
            x = pl.reg.interpret_cast(first, dtype=dtype)
            y = pl.reg.interpret_cast(second, dtype=dtype)
            if omitted:
                low, high = pl.reg.intlv(x, y)
                even, odd = pl.reg.dintlv(x, y)
            else:
                mask = pl.reg.active_mask(count, lanes=lanes)
                low, high = pl.reg.intlv(x, y, mask)
                even, odd = pl.reg.dintlv(x, y, mask)
            low_bits = pl.reg.interpret_cast(low, dtype=storage)
            high_bits = pl.reg.interpret_cast(high, dtype=storage)
            even_bits = pl.reg.interpret_cast(even, dtype=storage)
            odd_bits = pl.reg.interpret_cast(odd, dtype=storage)
            c = pl.reg.store(c, offset=256, value=low_bits)
            c = pl.reg.store(c, offset=640, value=high_bits)
            c = pl.reg.store(c, offset=1024, value=even_bits)
            c = pl.reg.store(c, offset=1408, value=odd_bits)
        pl.store(pl.reshape(c, [1, 2048]), [0, 0], out)
        return out

    @pl.jit
    def dispatch(
        src: pl.Tensor[[1, 1024], storage],
        out: pl.InOut[pl.Tensor[[1, 2048], storage]],
        n: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 2048], storage]:
        return body(src, out, n)

    return dispatch


def _environment():
    assert os.environ.get("PYPTO_SKIP_RUNTIME_PIN_CHECK", "0") == "0"
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
        assert not any(s in os.environ.get(key, "").lower() for s in ("simulator", "camodel", "msopprof"))
    pin = check_runtime_pin(checkout_root=ROOT)
    assert pin.matches and pin.pinned == pin.installed, pin
    return pin


def _inputs(dtype, repeat):
    _, storage = _storage(dtype)
    # Independent deterministic streams distinguish the two sources and result
    # halves. All nonzero payloads expose inactive lanes left unmasked.
    rng = np.random.default_rng(1732 + repeat)
    data = rng.integers(1, np.iinfo(storage).max, size=1024, dtype=storage)
    if dtype == "BF16":
        # Include signed zeros, subnormals, infinities and several NaN payloads.
        special = np.array([0, 0x8000, 1, 0x807F, 0x7F80, 0xFF80, 0x7FC1, 0x7F81], dtype=storage)
        data[:8] = np.roll(special, repeat)
        data[512:520] = np.roll(special[::-1], repeat)
    return data


def _golden(data, lanes, active):
    lhs, rhs = data[:lanes], data[512 : 512 + lanes]
    interleaved = np.stack((lhs, rhs), axis=1).reshape(-1)
    joined = np.concatenate((lhs, rhs))
    outputs = (interleaved[:lanes], interleaved[lanes:], joined[::2], joined[1::2])
    expected = np.full(2048, 0x5A, dtype=data.dtype)
    for offset, values in zip(OFFSETS, outputs):
        expected[offset : offset + lanes] = 0
        expected[offset : offset + active] = values[:active]
    return expected


def _build(output, dtypes, lengths, modes):
    pin = _environment()
    compiler = find_ptoas_binary()
    assert compiler is not None
    version = subprocess.check_output([compiler, "--version"], text=True).strip()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"runtime_commit": pin.installed, "ptoas": version, "cases": [], "compile_failures": []}
    recorder = runpy.run_path(str(ROOT / "examples/runtime/vf_emitc_vpto.py"))["_record_compilation"]
    for dtype in dtypes:
        _, storage = _storage(dtype)
        # Short-vector hardware qualification is opt-in through --lanes.
        selected_lengths = (
            lengths
            if lengths is not None
            else [lanes for lanes in VECTOR_LENGTHS if lanes >= 256 // storage.itemsize]
        )
        for lanes in selected_lengths:
            for mode in modes:
                name = f"{dtype.lower()}_{lanes}_{mode}"
                directory = output / name
                entry = {"name": name, "dtype": dtype, "lanes": lanes, "mode": mode}
                try:
                    compiled = _program(dtype, lanes, mode == "omitted").compile(
                        config=RunConfig(
                            backend_type=BackendType.Ascend950,
                            execution_mode=ExecutionMode.ONBOARD,
                            cache_config=CacheConfig(enabled=False),
                            save_kernels_dir=str(directory),
                        )
                    )
                    config = read_kernel_config(directory / "kernel_config.py")
                    helpers = [Path(p) for kernel in config.KERNELS for p in kernel["vpto_sources"]]
                    assert len(helpers) == 1
                    for helper in helpers:
                        source = helper.read_text()
                        assert source.count("pto.vmi.vintlv ") == 1
                        assert source.count("pto.vmi.vdintlv ") == 1
                        assert "pto.vmi.vsel " not in source
                        assert "interleave_zero" not in source
                        # Only the two input and four output bit transports remain.
                        assert source.count("pto.vmi.vinterpret_cast ") == 6
                    with recorder(directory):
                        # Persist binaries for compilation-free device execution.
                        compiled = CompiledProgram.from_dir(directory)
                        compiled.load()
                    entry.update(directory=str(directory.resolve()), runtime=compiled.runtime_name)
                    manifest["cases"].append(entry)
                    for repeat in range(2):
                        np.save(directory / f"input-{repeat}.npy", _inputs(dtype, repeat))
                    print(f"BUILT {name}", flush=True)
                except Exception as error:
                    manifest["compile_failures"].append(
                        {**entry, "error": str(error), "traceback": traceback.format_exc()}
                    )
                    print(f"COMPILE FAIL {name}: {error}", flush=True)
                (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    assert not manifest["compile_failures"], "Compilation failures are recorded in manifest.json"


def _run(output, device):
    pin = _environment()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["runtime_commit"] == pin.installed
    assert manifest["cases"], "No compiled cases to execute"
    config = RunConfig(
        backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=device
    )
    checks = []
    compiler = importlib.import_module("pypto.runtime.kernel_compiler").KernelCompiler

    def reject_compile(*args, **kwargs):
        raise AssertionError("Device execution must reuse prepared binaries; rebuild outside allocation")

    with (
        pytest.MonkeyPatch.context() as compiler_guard,
        ExitStack() as workers,
        (output / "checks.jsonl").open("w") as evidence,
    ):
        compiler_guard.setattr(compiler, "compile_incore", reject_compile)
        worker = None
        for index, entry in enumerate(manifest["cases"]):
            # ChipWorker retains registrations until close; handle.unregister()
            # alone does not release the runtime's 64 callable slots.
            if index % 24 == 0:
                workers.close()
                worker = workers.enter_context(ChipWorker(config=config, runtime=entry["runtime"]))
            assert worker is not None
            directory = Path(entry["directory"])
            handle = worker.register(CompiledProgram.from_dir(directory))
            lanes = entry["lanes"]
            counts = (lanes,) if entry["mode"] == "omitted" else (0, 1, lanes // 2, lanes - 1, lanes)
            for repeat in range(2):
                data = np.load(directory / f"input-{repeat}.npy")
                src = torch.from_numpy(data.reshape(1, 1024).copy())
                for active in counts:
                    expected = _golden(data, lanes, active)
                    actual = torch.from_numpy(np.full((1, 2048), 0x5A, dtype=data.dtype))
                    handle(src, actual, active)
                    # Compare bits, including BF16 NaNs and signed zero. Report
                    # each output independently, plus the untouched guard regions.
                    observed = actual.numpy().reshape(-1)
                    for label, start, size in zip(("low", "high", "even", "odd"), OFFSETS, (lanes,) * 4):
                        bad = np.flatnonzero(observed[start : start + size] != expected[start : start + size])
                        checks.append(
                            {
                                "case": entry["name"],
                                "repeat": repeat,
                                "active": active,
                                "output": label,
                                "mismatches": bad.tolist(),
                                "actual": observed[start + bad[:8]].tolist(),
                                "expected": expected[start + bad[:8]].tolist(),
                            }
                        )
                        evidence.write(json.dumps(checks[-1]) + "\n")
                    guard = np.ones(2048, dtype=bool)
                    for start in OFFSETS:
                        guard[start : start + lanes] = False
                    assert np.array_equal(observed[guard], expected[guard]), entry["name"]
            evidence.flush()
            print(f"EXECUTED {entry['name']}", flush=True)
    failures = [c for c in checks if c["mismatches"]]
    (output / "result.json").write_text(
        json.dumps(
            {
                "device": device,
                "hardware_executed": True,
                "incore_compilation_during_run": False,
                "ptoas": manifest["ptoas"],
                "runtime_commit": pin.installed,
                "compile_failures": manifest["compile_failures"],
                "checks": checks,
                "failures": len(failures),
            },
            indent=2,
        )
        + "\n"
    )
    assert not failures, f"{len(failures)}/{len(checks)} output checks failed; see result.json"
    assert not manifest["compile_failures"], "Hardware checks cover only the successfully compiled cases"


def test_reg_interleave_onboard(vpto_onboard_session):
    if vpto_onboard_session is None:
        pytest.skip("Build first, then opt in under task-submit with PYPTO_RUN_VPTO_ONBOARD=1")
    _run(vpto_onboard_session, int(os.environ["PYPTO_VPTO_DEVICE_ID"]))


if __name__ == "__main__":
    if "--build" in sys.argv or "--worker" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--build", type=Path)
        parser.add_argument("--worker", nargs=2, metavar=("OUTPUT", "DEVICE"))
        parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=DTYPES)
        parser.add_argument(
            "--lanes",
            nargs="+",
            type=int,
            help="Explicit lengths, including small-VL reproductions; defaults to physical width and above",
        )
        parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
        args = parser.parse_args()
        if args.build:
            _build(args.build.resolve(), args.dtypes, args.lanes, args.modes)
        else:
            _run(Path(args.worker[0]).resolve(), int(args.worker[1]))
    else:
        pytest.main([__file__, "-v", *sys.argv[1:]])
