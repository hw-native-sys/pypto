# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cross-process cache reuse with real compiler binaries and a preloaded allocator.

Set PYPTO_TEST_PRELOAD_LIBRARY to a jemalloc shared library. No NPU execution
is needed: warmup builds and loads binaries without creating a Worker.
"""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import CacheConfig, cache_stats, passes
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig

_RESULT_PREFIX = "CACHE_ENV_RESULT "


@pl.jit
def add_scalar(x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.add(pl.load(x, [0, 0], [16, 16]), 3.0), [0, 0], out)
    return out


def _cache_worker(root: Path, phase: str, token: str) -> None:
    from pypto.jit._toolchain import capture_toolchain  # noqa: PLC0415
    from pypto.runtime.kernel_compiler import KernelCompiler  # noqa: PLC0415

    library = os.environ.get("PYPTO_TEST_PRELOAD_LIBRARY")
    loaded = library is not None and str(Path(library).resolve()) in Path("/proc/self/maps").read_text()
    if os.environ.get("LD_PRELOAD"):
        assert loaded, "The test allocator must really be loaded at process startup"
    config = RunConfig(
        backend_type=BackendType.Ascend910B,
        execution_mode=ExecutionMode.ONBOARD,
        cache_config=CacheConfig(enabled=True, root=root, extra_fingerprint=token),
    )
    if phase == "hit":

        def forbidden(*args, **kwargs):
            raise AssertionError("A READY cache hit invoked a compiler")

        KernelCompiler.compile_incore = forbidden
        KernelCompiler.compile_orchestration = forbidden

    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        compiled = add_scalar.warmup(config=config)
        first = cache_stats()
        assert first.bypasses == 0, first.last_bypass_reason
        assert first.forced_rebuilds == 0
        if phase == "hit":
            assert first.ready_hits == 1
            assert first.generation_builds == first.binary_builds == 0
        else:
            assert first.generation_builds == first.binary_builds == 1
        for _ in range(3):
            assert add_scalar.warmup(config=config) is compiled
    identity = capture_toolchain("a2a3", "host_build_graph")
    assert identity.usable, identity.failures
    result = {
        "stats": asdict(first),
        "identity": identity.digest,
        "preload_loaded": loaded,
        "artifact": str(compiled._artifact_runtime.handle.directory),
        "run_base": str(compiled._artifact_runtime.run_directory.parent),
    }
    print(f"{_RESULT_PREFIX}{json.dumps(result, sort_keys=True)}")


def _run_worker(root: Path, phase: str, token: str, library: Path, preload: bool) -> dict:
    environment = dict(os.environ)
    environment["PYPTO_TEST_PRELOAD_LIBRARY"] = str(library)
    environment["PYPTO_PROG_BUILD_DIR"] = str(root.parent / f"private-{phase}")
    environment["PYPTO_EMIT_DEBUG_RUNNER"] = "0"
    environment["PYPTO_REBUILD_FROM_PTO"] = "0"
    environment["PYTHONHASHSEED"] = "0"
    if preload:
        environment["LD_PRELOAD"] = f"{library}:"
    else:
        environment.pop("LD_PRELOAD", None)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--cache-worker", str(root), phase, token],
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    line = next(line for line in result.stdout.splitlines() if line.startswith(_RESULT_PREFIX))
    return json.loads(line.removeprefix(_RESULT_PREFIX))


@pytest.mark.parametrize("preload_warmup", [False, True])
def test_preloaded_service_reuses_binaries_across_processes(tmp_path, request, preload_warmup):
    if request.config.getoption("--codegen-only"):
        pytest.skip("Persistent binary reuse needs the target compiler and SDK")
    selected = os.environ.get("PYPTO_TEST_PRELOAD_LIBRARY")
    if not selected:
        pytest.skip("Set PYPTO_TEST_PRELOAD_LIBRARY to the jemalloc shared library")
    library = Path(selected).resolve(strict=True)
    root = tmp_path / "cache"
    built = _run_worker(root, "build", "deployment-1", library, preload_warmup)
    restored = _run_worker(root, "hit", "deployment-1", library, True)
    assert restored["preload_loaded"]
    assert restored["identity"] == built["identity"]
    assert restored["artifact"] == built["artifact"]
    assert Path(restored["run_base"]) == tmp_path / "private-hit"
    changed = _run_worker(root, "build", "deployment-2", library, True)
    # The deployment fingerprint changes the artifact key, not the toolchain identity.
    assert changed["identity"] == built["identity"]
    assert changed["artifact"] != built["artifact"]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cache-worker":
        _cache_worker(Path(sys.argv[2]), *sys.argv[3:])
    else:
        pytest.main([__file__, "-v"])
