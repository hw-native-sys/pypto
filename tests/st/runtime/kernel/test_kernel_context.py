# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real kernel Worker init/prepare/close, isolated from program-mode processes."""

import multiprocessing
import traceback
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import CacheConfig
from pypto.pypto_core import passes
from pypto.runtime.runner import RunConfig


@pl.jit
def first_kernel(x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.add(pl.load(x, [0, 0], [16, 16]), 1.0), [0, 0], out)
    return out


@pl.jit
def second_kernel(x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.mul(pl.load(x, [0, 0], [16, 16]), 2.0), [0, 0], out)
    return out


def _native_case(case, platform, device_id, directory, queue):
    try:
        import torch  # noqa: PLC0415
        import torch_npu  # noqa: PLC0415
        from pypto.runtime.kernel.abi import KernelConfig  # noqa: PLC0415
        from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415
        from simpler.task_interface import CallConfig, ChipWorker  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        torch_npu.npu.set_device(device_id)
        borrowed = torch.empty(1, device=f"npu:{device_id}")
        state = get_process_kernel_state()
        config = KernelConfig(platform, "tensormap_and_ringbuffer", device_id)
        try:
            if case == "unsupported":
                with pytest.raises(RuntimeError, match="does not support kernel mode"):
                    state.ensure_worker(KernelConfig(platform, "host_build_graph", device_id))
            else:
                worker = state.ensure_worker(config)
                assert worker is state.ensure_worker(config)
                if case == "prepare":
                    from pypto.runtime.worker import ChipWorker as ProgramWorker  # noqa: PLC0415

                    build_config = RunConfig(
                        platform=platform, device_id=device_id, cache_config=CacheConfig(enabled=False)
                    )
                    registrations = []
                    for index, op in enumerate((first_kernel, second_kernel)):
                        # Explicit directories make these independent fresh DSL builds.
                        import os  # noqa: PLC0415

                        os.environ["PYPTO_PROG_BUILD_DIR"] = str(Path(directory) / str(index))
                        with passes.PassContext([], runtime=passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER):
                            artifact = op._resolve_kernel_artifact(
                                (), {"config": build_config}, allow_signature_mode=True
                            )
                        registration = state.ensure_callable(artifact, config)
                        assert registration is state.ensure_callable(artifact, config)
                        registration.require_live()
                        registrations.append(registration)
                    assert registrations[0].handle != registrations[1].handle
                    assert registrations[0].owner is registrations[1].owner is state
                    assert set(worker.worker._kernel_callables) == {r.handle for r in registrations}
                    assert not worker.worker._callable_registry
                    program = ProgramWorker(config=build_config, auto_init=False)
                    with pytest.raises(RuntimeError, match="already claimed kernel"):
                        program.init()
                else:
                    # Native rejection is independent of PyPTO's singleton guard.
                    other = ChipWorker()
                    bins = RuntimeBuilder(platform=platform).get_binaries(config.runtime, build=False)
                    try:
                        with pytest.raises(RuntimeError, match="kernel_mode_init failed"):
                            other.kernel_init(device_id, bins, CallConfig())
                    finally:
                        other.finalize()
                    assert worker.worker.kernel_mode_supported
        finally:
            state.close()
        # Teardown has not reset the framework-owned device or released its storage.
        borrowed.fill_(7)
        assert borrowed.item() == 7
        queue.put(None)
    except BaseException:
        queue.put(traceback.format_exc())


def _run_isolated(case, platform, device_id, directory):
    """Keep blocking IPC out of the ST harness's collection-time assignment evaluator."""
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    child = ctx.Process(
        target=_native_case,
        args=(case, platform, device_id, directory, queue),
    )
    child.start()
    try:
        result = queue.get(timeout=120)
        child.join(timeout=15)
        assert result is None, result
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=15)
        queue.close()


@pytest.mark.parametrize("case", ["prepare", "duplicate", "unsupported"])
def test_real_kernel_context(test_config, tmp_path, case):
    if test_config.codegen_only or test_config.platform != "a2a3":
        pytest.skip("This lifecycle smoke test requires an A2/A3 NPU and built runtime binaries")
    npu = pytest.importorskip("torch_npu")
    if not npu.npu.is_available():
        pytest.skip("torch_npu reports no available NPU")
    _run_isolated(case, test_config.platform, test_config.device_id, str(tmp_path))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
