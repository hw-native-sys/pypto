# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end DSL RegScope -> production .text -> persistent A5 Worker.

Unlike the independent link control, this test never replaces generated source
or inserts vpto_sources. The JIT must generate and package every helper itself.
"""

import json
import os
import runpy
import sys
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import CacheConfig
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig
from pypto.runtime._artifact_sources import read_kernel_config
from test_vpto_vf_onboard import FIXTURE, ROOT, _acceptance, _build_variants

pytestmark = pytest.mark.platforms("a5", reason="VPTO helpers require the A5 AIV backend")


def _prepare(directory: Path):
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
        # Static unrolling gives one helper fixed bounds while n remains a
        # runtime scalar. Masked stores preserve the initialized output tail.
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

    compiled = dispatch.compile(
        config=RunConfig(
            backend_type=BackendType.Ascend950,
            execution_mode=ExecutionMode.ONBOARD,
            cache_config=CacheConfig(enabled=False),
            save_kernels_dir=str(directory),
            dump_passes=True,
        ),
    )
    kernel = read_kernel_config(directory / "kernel_config.py").KERNELS[0]
    assert kernel["core_type"] == "aiv"
    assert kernel["vpto_sources"], "Normal JIT lost the structured helper bundle"
    assert all("pto.vmi." in Path(p).read_text() for p in kernel["vpto_sources"])
    wrapper = Path(kernel["source"]).read_text()
    assert "__pypto_vf_" in wrapper
    assert "args[3]" in wrapper
    return compiled


def test_reg_scope_persistent_onboard():
    if os.environ.get("PYPTO_RUN_VPTO_ONBOARD") != "1":
        pytest.skip("Opt in with PYPTO_RUN_VPTO_ONBOARD=1 and PYPTO_VPTO_DEVICE_ID")
    device = int(os.environ["PYPTO_VPTO_DEVICE_ID"])
    timeout = int(os.environ.get("PYPTO_VPTO_TIMEOUT", "600"))
    assert device >= 0 and timeout > 0
    assert os.environ.get("PYPTO_SKIP_RUNTIME_PIN_CHECK", "0") == "0"
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
        assert not any(
            token in os.environ.get(key, "").lower() for token in ("simulator", "camodel", "msopprof")
        )
    output = Path(os.environ.get("PYPTO_VPTO_RUNTIME_OUTPUT", ROOT / "build/vf-pr5-onboard"))
    output.mkdir(parents=True, exist_ok=True)
    runpy.run_path(str(FIXTURE / "test_validation.py"))["run"](
        output.resolve(),
        "dsl-persistent-runtime",
        [sys.executable, Path(__file__).resolve(), "--worker", output.resolve(), device],
        timeout=timeout,
    )
    assert json.loads((output / "result.json").read_text())["numerical_checks"] == 108


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _acceptance(Path(sys.argv[2]), int(sys.argv[3]), _prepare)
    elif len(sys.argv) == 3 and sys.argv[1] == "--build":
        for name, _ in _build_variants(Path(sys.argv[2]), _prepare):
            print(f"PASS DSL production build stage={name}; hardware not executed", flush=True)
    else:
        raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
