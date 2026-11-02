# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 numerical acceptance for all fourteen extended register operations.

Build with --build OUTPUT in a task-submit --no-device job. Execute the saved
artifacts with --worker OUTPUT DEVICE in a task-submit --device auto job, using
TASK_DEVICE. Compilation never reserves a device; execution never skips pin
validation or substitutes simulator code. Inputs and outputs are CPU tensors
transferred by the production ChipWorker, as in test_reg_scope_onboard.py.
"""

import json
import os
import runpy
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pypto.language as pl
import pytest
import torch
from pypto import CacheConfig
from pypto.backend import BackendType
from pypto.ir.compiled_program import CompiledProgram
from pypto.runtime import ChipWorker, ExecutionMode, RunConfig
from pypto.runtime._artifact_sources import read_kernel_config
from pypto.runtime.runtime_pin import check_runtime_pin

pytestmark = pytest.mark.platforms("a5", reason="Extended VMI operations require A5 hardware")
ROOT = Path(__file__).resolve().parents[4]
LENGTHS = (0, 1, 31, 32, 37, 63, 64)
CASES = ("float32", "integer_u32", "integer_i32", "indexed_f32", "indexed_u32", "indexed_f16")
FLOAT_OPS = ("expdif", "axpy", "mula", "lrelu", "prelu")
INTEGER_OPS = ("mull", "addc", "subc", "addcs", "subcs", "intlv", "dintlv")


def _program(case):
    dtype = {
        "integer_u32": pl.UINT32,
        "integer_i32": pl.INT32,
        "indexed_u32": pl.UINT32,
        "indexed_f16": pl.FP16,
    }.get(case, pl.FP32)
    index_dtype = pl.UINT16 if case == "indexed_f16" else pl.UINT32
    if case == "float32":

        @pl.jit.incore
        def body(
            src: pl.Tensor[[1, 256], dtype],
            out: pl.InOut[pl.Tensor[[1, 1024], dtype]],
            n: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, 1024], dtype]:
            a = pl.reshape(pl.load(src, [0, 0], [1, 256], target_memory=pl.MemorySpace.Vec), [256])
            c = pl.reshape(pl.load(out, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec), [1024])
            with pl.reg_scope():
                x = pl.reg.load(a, offset=0, lanes=64)
                y = pl.reg.load(a, offset=64, lanes=64)
                z = pl.reg.load(a, offset=128, lanes=64)
                idx = pl.reg.index_gen(0, lanes=64, dtype=pl.INT32)
                limit = pl.reg.broadcast(n, lanes=64, dtype=pl.INT32)
                seed = pl.reg.full_mask(lanes=64)
                mask = pl.reg.compare(idx, limit, seed, predicate="lt")
                e = pl.reg.expdif(x, y, mask)
                ax = pl.reg.axpy(x, y, 0.75, mask)
                ma = pl.reg.mula(z, x, y, mask)
                lr = pl.reg.lrelu(x, 0.125, mask)
                pr = pl.reg.prelu(x, y, mask)
                c = pl.reg.store(c, offset=64, value=e)
                c = pl.reg.store(c, offset=128, value=ax)
                c = pl.reg.store(c, offset=192, value=ma)
                c = pl.reg.store(c, offset=256, value=lr)
                c = pl.reg.store(c, offset=320, value=pr)
            c = pl.reshape(c, [1, 1024])
            pl.store(c, [0, 0], out)
            return out
    elif case.startswith("integer_"):

        @pl.jit.incore
        def body(
            src: pl.Tensor[[1, 256], dtype],
            out: pl.InOut[pl.Tensor[[1, 1024], dtype]],
            n: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, 1024], dtype]:
            a = pl.reshape(pl.load(src, [0, 0], [1, 256], target_memory=pl.MemorySpace.Vec), [256])
            c = pl.reshape(pl.load(out, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec), [1024])
            with pl.reg_scope():
                x = pl.reg.load(a, offset=0, lanes=64)
                y = pl.reg.load(a, offset=64, lanes=64)
                ci = pl.reg.load(a, offset=128, lanes=64)
                idx = pl.reg.index_gen(0, lanes=64, dtype=pl.INT32)
                limit = pl.reg.broadcast(n, lanes=64, dtype=pl.INT32)
                seed = pl.reg.full_mask(lanes=64)
                mask = pl.reg.compare(idx, limit, seed, predicate="lt")
                zero = pl.reg.broadcast(0, lanes=64, dtype=dtype)
                one = pl.reg.broadcast(1, lanes=64, dtype=dtype)
                carry_in = pl.reg.compare(ci, zero, seed, predicate="ne")
                lo, hi = pl.reg.mull(x, y, mask)
                add, ac = pl.reg.addc(x, y, mask)
                sub, sc = pl.reg.subc(x, y, mask)
                adds, asc = pl.reg.addcs(x, y, carry_in, mask)
                subs, ssc = pl.reg.subcs(x, y, carry_in, mask)
                il, ih = pl.reg.intlv(x, y, mask)
                de, do = pl.reg.dintlv(x, y, mask)
                acv = pl.reg.select(ac, one, zero)
                scv = pl.reg.select(sc, one, zero)
                ascv = pl.reg.select(asc, one, zero)
                sscv = pl.reg.select(ssc, one, zero)
                c = pl.reg.store(c, offset=64, value=lo)
                c = pl.reg.store(c, offset=128, value=hi)
                c = pl.reg.store(c, offset=192, value=add)
                c = pl.reg.store(c, offset=256, value=acv)
                c = pl.reg.store(c, offset=320, value=sub)
                c = pl.reg.store(c, offset=384, value=scv)
                c = pl.reg.store(c, offset=448, value=adds)
                c = pl.reg.store(c, offset=512, value=ascv)
                c = pl.reg.store(c, offset=576, value=subs)
                c = pl.reg.store(c, offset=640, value=sscv)
                c = pl.reg.store(c, offset=704, value=il)
                c = pl.reg.store(c, offset=768, value=ih)
                c = pl.reg.store(c, offset=832, value=de)
                c = pl.reg.store(c, offset=896, value=do)
            c = pl.reshape(c, [1, 1024])
            pl.store(c, [0, 0], out)
            return out
    else:

        @pl.jit.incore
        def body(
            src: pl.Tensor[[1, 256], dtype],
            out: pl.InOut[pl.Tensor[[1, 1024], dtype]],
            n: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, 1024], dtype]:
            a = pl.reshape(pl.load(src, [0, 0], [1, 256], target_memory=pl.MemorySpace.Vec), [256])
            c = pl.reshape(pl.load(out, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec), [1024])
            with pl.reg_scope():
                idx = pl.reg.index_gen(0, lanes=64, dtype=pl.INT32)
                limit = pl.reg.broadcast(n, lanes=64, dtype=pl.INT32)
                seed = pl.reg.full_mask(lanes=64)
                mask = pl.reg.compare(idx, limit, seed, predicate="lt")
                offsets = pl.reg.index_gen(0, lanes=64, dtype=index_dtype)
                three = pl.reg.broadcast(3, lanes=64, dtype=index_dtype)
                last = pl.reg.broadcast(255, lanes=64, dtype=index_dtype)
                scaled = pl.reg.mul(offsets, three)
                reverse = pl.reg.sub(last, scaled)
                values = pl.reg.gather(a, reverse, mask)
                x = pl.reg.load(a, lanes=64)
                c = pl.reg.scatter(c, reverse, x, mask)
                reread = pl.reg.gather(c, reverse, mask)
                repeated = pl.reg.broadcast(7, lanes=64, dtype=index_dtype)
                repeats = pl.reg.gather(a, repeated, mask)
                c = pl.reg.store(c, offset=320, value=values)
                c = pl.reg.store(c, offset=448, value=reread)
                c = pl.reg.store(c, offset=576, value=repeats)
            c = pl.reshape(c, [1, 1024])
            pl.store(c, [0, 0], out)
            return out

    @pl.jit
    def dispatch(
        src: pl.Tensor[[1, 256], dtype], out: pl.InOut[pl.Tensor[[1, 1024], dtype]], n: pl.Scalar[pl.INT32]
    ) -> pl.Tensor[[1, 1024], dtype]:
        return body(src, out, n)

    return dispatch


def _environment():
    assert os.environ.get("PYPTO_SKIP_RUNTIME_PIN_CHECK", "0") == "0"
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
        assert not any(
            token in os.environ.get(key, "").lower() for token in ("simulator", "camodel", "msopprof")
        )
    pin = check_runtime_pin(checkout_root=ROOT)
    assert pin.matches and pin.pinned == pin.installed, pin
    return pin


def _build(output):
    pin = _environment()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"runtime_commit": pin.installed, "cases": []}
    recorder = runpy.run_path(str(ROOT / "examples/runtime/vf_emitc_vpto.py"))["_record_compilation"]
    for case in CASES:
        directory = output / case
        compiled = _program(case).compile(
            config=RunConfig(
                backend_type=BackendType.Ascend950,
                execution_mode=ExecutionMode.ONBOARD,
                cache_config=CacheConfig(enabled=False),
                save_kernels_dir=str(directory),
                dump_passes=True,
            )
        )
        config = read_kernel_config(directory / "kernel_config.py")
        helpers = [Path(p).read_text() for kernel in config.KERNELS for p in kernel["vpto_sources"]]
        ops = (
            FLOAT_OPS
            if case == "float32"
            else INTEGER_OPS
            if case.startswith("integer_")
            else ("gather", "scatter")
        )
        for op in ops:
            assert any(f"pto.vmi.v{op}" in helper for helper in helpers), (case, op)
        evidence = Path(tempfile.mkdtemp(prefix="compilation-", dir=directory))
        with recorder(evidence):
            compiled.load()  # Production PTOAS -> LLVM -> Bisheng -> linked .text, no device.
        manifest["cases"].append(
            {
                "name": case,
                "directory": str(directory.resolve()),
                "runtime": compiled.runtime_name,
                "ops": list(ops),
            }
        )
        print(f"BUILT {case}: {', '.join(ops)}", flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _inputs(case, repeat):
    if case.startswith("integer_"):
        patterns = np.array(
            [0, 1, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 123456789, 0xFFFFFFFE, 987654321], dtype=np.uint32
        )
        data = np.resize(np.roll(patterns, repeat), 256)
        data[64:128] = np.resize(np.roll(patterns, 3 + repeat), 64)
        data[128:192] = (np.arange(64) + repeat) % 2
        return data.view(np.int32) if case == "integer_i32" else data
    if case == "indexed_u32":
        return np.arange(256, dtype=np.uint32) * 104729 + repeat * 31
    dtype = np.float16 if case == "indexed_f16" else np.float32
    values = ((np.arange(256) % 29) * 0.125 - 1.75 + repeat * 0.25).astype(dtype)
    if case == "float32":
        values[64:128] = (np.arange(64) % 11) * 0.125 + 0.5
    return values


def _golden(case, data, n):
    sentinel = 0xDEADBEEF if data.dtype == np.uint32 else -777
    expected = np.full(1024, sentinel, dtype=data.dtype)
    x, y, z = (data[start : start + 64] for start in (0, 64, 128))
    if case == "float32":
        xd, yd, zd = x.astype(np.float64), y.astype(np.float64), z.astype(np.float64)
        results = [
            np.exp(xd - yd),
            xd * 0.75 + yd,
            zd + xd * yd,
            np.where(xd > 0, xd, xd * 0.125),
            np.maximum(xd, 0) + yd * np.minimum(xd, 0),
        ]
    elif case.startswith("integer_"):
        ux, uy = x.view(np.uint32).astype(object), y.view(np.uint32).astype(object)
        ci = z.astype(object)
        product = x.astype(object) * y.astype(object)
        summed, difference = ux + uy, ux - uy
        summed_c, difference_c = ux + uy + ci, ux - uy - (1 - ci)
        interleaved = np.column_stack((x, y)).reshape(-1)
        combined = np.concatenate((x, y))
        raw = [
            product,
            product >> 32,
            summed,
            summed > 0xFFFFFFFF,
            difference,
            ux >= uy,
            summed_c,
            summed_c > 0xFFFFFFFF,
            difference_c,
            ux >= uy + (1 - ci),
            interleaved[:64],
            interleaved[64:],
            combined[::2],
            combined[1::2],
        ]
        results = [
            np.array([int(v) & 0xFFFFFFFF for v in result], dtype=np.uint32).view(data.dtype)
            for result in raw
        ]
    else:
        indices = 255 - np.arange(64) * 3
        expected[indices[:n]] = x[:n]
        for start, values in ((320, data[indices]), (448, x), (576, np.full(64, data[7], dtype=data.dtype))):
            expected[start : start + 64] = 0
            expected[start : start + n] = values[:n]
        return expected
    for index, result in enumerate(results, 1):
        start = index * 64
        expected[start : start + 64] = 0
        expected[start : start + n] = result[:n]
    return expected


def _run(output, device):
    pin = _environment()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["runtime_commit"] == pin.installed
    assert {case["name"] for case in manifest["cases"]} == set(CASES)
    config = RunConfig(
        backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=device
    )
    checks = []
    failures = []
    with ChipWorker(config=config, runtime=manifest["cases"][0]["runtime"]) as worker:
        for entry in manifest["cases"]:
            case = entry["name"]
            compiled = CompiledProgram.from_dir(entry["directory"])
            handle = worker.register(compiled)
            for repeat in range(2):
                data = _inputs(case, repeat)
                src = torch.from_numpy(data.reshape(1, 256).copy())
                for n in LENGTHS:
                    expected = _golden(case, data, n)
                    sentinel = 0xDEADBEEF if data.dtype == np.uint32 else -777
                    actual = torch.from_numpy(np.full((1, 1024), sentinel, dtype=data.dtype))
                    handle(src, actual, n)
                    try:
                        if case == "float32":
                            np.testing.assert_allclose(
                                actual.numpy()[0],
                                expected,
                                rtol=3e-5,
                                atol=2e-6,
                                err_msg=f"{case}, repeat={repeat}, n={n}",
                            )
                        else:
                            np.testing.assert_array_equal(
                                actual.numpy()[0], expected, err_msg=f"{case}, repeat={repeat}, n={n}"
                            )
                    except AssertionError as error:
                        mismatches = (
                            np.flatnonzero(~np.isclose(actual.numpy()[0], expected, rtol=3e-5, atol=2e-6))
                            if case == "float32"
                            else np.flatnonzero(actual.numpy()[0] != expected)
                        )
                        failures.append(
                            {
                                "case": case,
                                "repeat": repeat,
                                "active_lanes": n,
                                "error": str(error),
                                "offsets": mismatches[:16].tolist(),
                                "actual": actual.numpy()[0][mismatches[:16]].tolist(),
                                "expected": expected[mismatches[:16]].tolist(),
                            }
                        )
                        print(
                            f"FAIL {case} repeat={repeat} n={n} offsets={mismatches[:8].tolist()}", flush=True
                        )
                    else:
                        print(
                            f"PASS {case} repeat={repeat} n={n} (results, inactive lanes, guards)", flush=True
                        )
                    checks.append({"case": case, "repeat": repeat, "active_lanes": n, "ops": entry["ops"]})
    (output / "result.json").write_text(
        json.dumps(
            {
                "device_id": device,
                "hardware_executed": True,
                "runtime_commit": pin.installed,
                "numerical_checks": len(checks),
                "checks": checks,
                "failures": failures,
                "passed_checks": len(checks) - len(failures),
            },
            indent=2,
        )
        + "\n"
    )

    (output / f"result-{time.time_ns()}.json").write_bytes((output / "result.json").read_bytes())
    assert not failures, f"{len(failures)} of {len(checks)} hardware numerical cases failed; see result.json"


def test_reg_extended_onboard(vpto_onboard_session: Path | None):
    if os.environ.get("PYPTO_RUN_VPTO_ONBOARD") != "1":
        pytest.skip("Build first, then opt in with PYPTO_RUN_VPTO_ONBOARD=1 under task-submit")
    # The autouse fixture redirects the per-test output environment. This
    # precompiled acceptance consumes the original session artifact directory.
    assert vpto_onboard_session is not None
    _run(vpto_onboard_session, int(os.environ["PYPTO_VPTO_DEVICE_ID"]))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--build":
        _build(Path(sys.argv[2]).resolve())
    elif len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _run(Path(sys.argv[2]).resolve(), int(sys.argv[3]))
    else:
        pytest.main([__file__, "-v", *sys.argv[1:]])
