# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Two-rank eager compute -> SHMEM -> eager compute integration.

First prepare examples/distributed/08_gloo_shmem_collectives.py --prepare-only
on the CPU. Set PYPTO_SHMEM_EAGER_ARTIFACTS to that directory and launch this
file with torchrun --standalone --nproc_per_node=2 -m pytest -n 0.
Each rank owns its pytest process; xdist must not duplicate rank identities.
"""

import os
import runpy
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2" or not os.environ.get("PYPTO_SHMEM_EAGER_ARTIFACTS"),
    reason="requires torchrun with two allocated NPUs and CPU-prepared artifacts",
)
def test_existing_eager_worker_can_compute_communicate_and_compute(monkeypatch):
    path = Path(__file__).resolve().parents[4] / "examples/distributed/08_gloo_shmem_collectives.py"
    monkeypatch.syspath_prepend(str(path.parent))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--artifacts",
            os.environ["PYPTO_SHMEM_EAGER_ARTIFACTS"],
            "--iterations",
            "3",
        ],
    )
    # A regression that constructs ChipWorker must fail before touching native
    # state, even if it tries to borrow the torch context again.
    from pypto.jit.decorator import JITFunction  # noqa: PLC0415
    from pypto.runtime import ChipWorker, device_runner  # noqa: PLC0415
    from pypto.runtime.kernel_compiler import KernelCompiler  # noqa: PLC0415

    def forbidden_program_worker(*args, **kwargs):
        pytest.fail("SHMEM in an eager process must reuse the KERNEL Worker")

    def forbidden_compilation(*args, **kwargs):
        pytest.fail("Device stage requires CPU-prepared binaries; rerun --prepare-only")

    monkeypatch.setattr(ChipWorker, "init", forbidden_program_worker)
    monkeypatch.setattr(KernelCompiler, "compile_incore", forbidden_compilation)
    monkeypatch.setattr(KernelCompiler, "compile_orchestration", forbidden_compilation)
    monkeypatch.setattr(device_runner, "_compile_single_kernel", forbidden_compilation)
    monkeypatch.setattr(device_runner, "_compile_single_orchestration", forbidden_compilation)
    monkeypatch.setattr(JITFunction, "_compile", forbidden_compilation)
    # Use the same module and absolute source identity as the CPU CLI.
    with pytest.raises(SystemExit) as completed:
        runpy.run_path(str(path), run_name="__main__")
    assert completed.value.code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
