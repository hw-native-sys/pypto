# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Production Tile adapters: persistent A5 execution and inter-helper dependencies.

Every caller comes from normal JIT compilation. No generated source is rewritten.
Opt in on a reserved device with PYPTO_RUN_VPTO_ONBOARD=1, an explicit
PYPTO_VPTO_DEVICE_ID and a bounded PYPTO_VPTO_TIMEOUT.
"""

import json
import os
import runpy
import sys
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from pypto import CacheConfig
from pypto.backend import BackendType
from pypto.runtime import ChipWorker, ExecutionMode, RunConfig
from pypto.runtime._artifact_sources import read_kernel_config
from pypto.runtime.runtime_pin import check_runtime_pin
from test_vpto_vf_onboard import FIXTURE, ROOT

pytestmark = pytest.mark.platforms("a5", reason="VPTO helpers require the A5 AIV backend")

LENGTHS = (0, 1, 63, 64, 65, 127, 128, 1023, 1024)
# Workers run from evidence directories; examples is not an installed package.
_record_compilation = runpy.run_path(str(ROOT / "examples/runtime/vf_emitc_vpto.py"))["_record_compilation"]


def _environment() -> tuple[int, int, Path]:
    if os.environ.get("PYPTO_RUN_VPTO_ONBOARD") != "1":
        pytest.skip("Opt in with PYPTO_RUN_VPTO_ONBOARD=1 on a reserved device")
    device = int(os.environ["PYPTO_VPTO_DEVICE_ID"])
    timeout = int(os.environ.get("PYPTO_VPTO_TIMEOUT", "600"))
    assert device >= 0 and timeout > 0
    assert os.environ.get("PYPTO_SKIP_RUNTIME_PIN_CHECK", "0") == "0"
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
        assert not any(
            word in os.environ.get(key, "").lower() for word in ("simulator", "camodel", "msopprof")
        )
    output = Path(os.environ.get("PYPTO_VPTO_RUNTIME_OUTPUT", ROOT / "build/vf-tile-sync/onboard"))
    output.mkdir(parents=True, exist_ok=True)
    return device, timeout, output.resolve()


def test_tile_adapter_persistent_onboard():
    device, timeout, output = _environment()
    example = ROOT / "examples/runtime/vf_emitc_vpto.py"
    runpy.run_path(str(FIXTURE / "test_validation.py"))["run"](
        output,
        "production-example",
        [sys.executable, example, "--device", device, "--output", output / "example", "--timeout", timeout],
        timeout=timeout + 30,
    )
    result = json.loads((output / "example/result.json").read_text())
    assert result["numerical_checks"] == 108
    assert result["production_tile_adapter"]
    assert result["compile_calls"] == 2


def _prepare_multiple(directory: Path):
    @pl.jit.incore
    def body(
        lhs: pl.Tensor[[1, 1024], pl.FP32],
        rhs: pl.Tensor[[1, 1024], pl.FP32],
        out: pl.InOut[pl.Tensor[[1, 1024], pl.FP32]],
        n: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 1024], pl.FP32]:
        a = pl.load(lhs, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec)
        b = pl.load(rhs, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec)
        c = pl.load(out, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec)
        a = pl.reshape(a, [1024])
        b = pl.reshape(b, [1024])
        c = pl.reshape(c, [1024])
        with pl.reg_scope():
            for chunk in pl.unroll(16):
                x = pl.reg.load(a, offset=chunk * 64, lanes=64)
                y = pl.reg.load(b, offset=chunk * 64, lanes=64)
                base = pl.cast(chunk * 64, pl.INT32)
                index = pl.reg.index_gen(base, lanes=64, dtype=pl.INT32)
                limit = pl.reg.broadcast(n, lanes=64, dtype=pl.INT32)
                seed = pl.reg.full_mask(lanes=64)
                mask = pl.reg.compare(index, limit, seed, predicate="lt")
                value = pl.reg.add(x, y)
                c = pl.reg.store(c, offset=chunk * 64, value=value, mask=mask)
        # Exercise VF -> ordinary Tile operation -> VF on the first helper's
        # store-version alias. The second scope writes the same output storage.
        doubled = pl.add(c, c)
        with pl.reg_scope():
            for chunk2 in pl.unroll(16):
                x2 = pl.reg.load(doubled, offset=chunk2 * 64, lanes=64)
                y2 = pl.reg.load(b, offset=chunk2 * 64, lanes=64)
                base2 = pl.cast(chunk2 * 64, pl.INT32)
                index2 = pl.reg.index_gen(base2, lanes=64, dtype=pl.INT32)
                limit2 = pl.reg.broadcast(n, lanes=64, dtype=pl.INT32)
                seed2 = pl.reg.full_mask(lanes=64)
                mask2 = pl.reg.compare(index2, limit2, seed2, predicate="lt")
                value2 = pl.reg.sub(x2, y2)
                c = pl.reg.store(c, offset=chunk2 * 64, value=value2, mask=mask2)
        c = pl.reshape(c, [1, 1024])
        pl.store(c, [0, 0], out)
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

    return dispatch.compile(
        config=RunConfig(
            backend_type=BackendType.Ascend950,
            execution_mode=ExecutionMode.ONBOARD,
            cache_config=CacheConfig(enabled=False),
            save_kernels_dir=str(directory),
            dump_passes=True,
        )
    )


def _multiple_worker(output: Path, device: int) -> None:
    pin = check_runtime_pin(checkout_root=ROOT)
    assert pin.matches and pin.pinned == pin.installed, pin
    (output / "test_tile_helper_sync.py").write_bytes(Path(__file__).read_bytes())
    directory = output / "multiple"
    compiled = _prepare_multiple(directory)
    kernel = read_kernel_config(directory / "kernel_config.py").KERNELS[0]
    assert len(kernel["vpto_sources"]) == 2
    manifest = json.loads((directory / "vpto/body/manifest.json").read_text())
    assert all(h["parameter_effects"] == ["read", "read", "readwrite", "none"] for h in manifest["helpers"])
    caller = (directory / "ptoas/body.pto").read_text()
    assert "pto.barrier" not in caller
    first, second = manifest["helpers"]
    assert caller.index(f"func.call @{first['adapter_symbol']}") < caller.index("pto.tadd")
    assert caller.index("pto.tadd") < caller.index(f"func.call @{second['adapter_symbol']}")
    config = RunConfig(
        backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=device
    )
    with _record_compilation(output / "multiple-objects") as calls:
        compiled.load()
        assert len(calls) == 1
    checks = 0
    with ChipWorker(config=config, runtime=compiled.runtime_name) as worker:
        handle = worker.register(compiled)
        indices = torch.arange(1088, dtype=torch.float32).reshape(1, 1088)
        for repeat in range(3):
            lhs = indices.remainder(31) * 0.25 - 4 + repeat
            rhs = indices.remainder(17) * 0.5 + 2 - 2 * repeat
            for n in LENGTHS:
                out = torch.full_like(lhs, -777)
                expected = out.clone()
                expected[:, 32 : 32 + n] = (2 * lhs + rhs)[:, 32 : 32 + n]
                handle(lhs, rhs, out, n)
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                checks += 1
                print(f"PASS multi-helper repeat={repeat} n={n} offset=32", flush=True)
    assert checks == 27
    (output / "multiple-result.json").write_text(
        json.dumps(
            {
                "device_id": device,
                "hardware_executed": True,
                "numerical_checks": checks,
                "helpers": 2,
                "sequence": "VF -> Tile add -> VF",
                "runtime_pin_matches": pin.matches,
                "runtime_off_pin": pin.off_pin,
                "generated_source_rewritten": False,
            },
            indent=2,
        )
        + "\n"
    )


def test_multiple_helpers_and_tile_operation_persistent_onboard():
    device, timeout, output = _environment()
    runpy.run_path(str(FIXTURE / "test_validation.py"))["run"](
        output,
        "multiple-worker",
        [sys.executable, Path(__file__).resolve(), "--multiple", output, device],
        timeout=timeout,
    )
    assert json.loads((output / "multiple-result.json").read_text())["numerical_checks"] == 27


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--multiple":
        _multiple_worker(Path(sys.argv[2]), int(sys.argv[3]))
    else:
        raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
