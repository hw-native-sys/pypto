# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public JIT eager execution and explicit program dispatch in isolated processes."""

import importlib
import multiprocessing
import os
import traceback
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import CacheConfig, configure_cache
from pypto.runtime import RunConfig


@pl.jit
def accumulate(
    x: pl.Tensor[[16, 16], pl.FP32], step: pl.Scalar[pl.FP32], acc: pl.InOut[pl.Tensor[[16, 16], pl.FP32]]
):
    with pl.at(level=pl.Level.CORE_GROUP):
        value = pl.add(pl.load(acc, [0, 0], [16, 16]), pl.mul(pl.load(x, [0, 0], [16, 16]), step))
        pl.store(value, [0, 0], acc)
    return acc


@pl.jit
def add_constant(
    x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]], value: pl.constexpr = 1
):
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.add(pl.load(x, [0, 0], [16, 16]), value), [0, 0], out)
    return out


def _run(case, device, directory):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415
    from pypto.jit.decorator import JITFunction  # noqa: PLC0415
    from pypto.runtime.kernel.abi import _NativeWorker  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415
    from pypto.torch import init  # noqa: PLC0415

    os.chdir(directory)
    os.environ.pop("PYPTO_PROG_BUILD_DIR", None)
    torch_npu.npu.set_device(device)
    cache = CacheConfig(enabled=False, root=Path(directory))
    config = RunConfig(platform="a2a3", device_id=device, cache_config=cache)
    if case == "program":
        x, acc = torch.full((16, 16), 2.0), torch.zeros(16, 16)
        program = accumulate.compile(x, 1.0, acc, config=config)
        assert get_process_kernel_state()._worker is None
        assert program(x, 3.0, acc, config=config) is None
        torch.testing.assert_close(acc, torch.full_like(acc, 6.0))
        with pytest.raises(TypeError, match="Out/InOut"):
            program(x, 3.0, config=config)
        return

    counts = dict(compile=0, frontend=0, init=0, prepare=0)
    compiler = importlib.import_module("pypto.ir.compile")

    def counted(name, original):
        def wrapped(*args, **kwargs):
            counts[name] += 1
            return original(*args, **kwargs)

        return wrapped

    def forbidden_compile(*args, **kwargs):
        raise AssertionError("eager execution used explicit program compilation")

    # Kernel calls carry no execution information; the process binds it once.
    configure_cache(cache)
    state = get_process_kernel_state()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(compiler, "_compile_impl", counted("compile", compiler._compile_impl))
        patch.setattr(
            JITFunction, "_compile_to_program", counted("frontend", JITFunction._compile_to_program)
        )
        patch.setattr(JITFunction, "compile", forbidden_compile)
        patch.setattr(_NativeWorker, "init", counted("init", _NativeWorker.init))
        patch.setattr(_NativeWorker, "prepare", counted("prepare", _NativeWorker.prepare))
        try:
            probe = torch.zeros((16, 16), device=f"npu:{device}")
            with pytest.raises(RuntimeError, match=r"call pypto\.torch\.init"):
                accumulate(probe, 1.0, probe)
            with pytest.raises(TypeError, match="CompileOptions"):
                accumulate(probe, 1.0, probe, config=config)
            assert state._worker is None and counts == dict(compile=0, frontend=0, init=0, prepare=0)
            init()
            assert counts == dict(compile=0, frontend=0, init=1, prepare=0)
            stream = torch_npu.npu.Stream(device=device)
            with torch_npu.npu.stream(stream):
                x = torch.full((16, 16), 2.0, device=f"npu:{device}")
                acc = torch.zeros_like(x)
                expected = 0.0
                for step in (1.0, 2.0, 3.0):
                    assert accumulate(x, step, acc) is acc
                    expected += 2 * step
                    torch.testing.assert_close(acc.cpu(), torch.full((16, 16), expected))
                assert counts == dict(compile=1, frontend=1, init=1, prepare=1)
                out = torch.empty_like(x)
                for value in (1, 1, 2):
                    assert add_constant(x, out, value=value) is out
                    torch.testing.assert_close((out + 3).cpu(), torch.full((16, 16), float(5 + value)))
                assert counts == dict(compile=3, frontend=3, init=1, prepare=3)
                assert len(state._registrations) == 3
            state.drain()
        finally:
            # 07 owns automatic framework shutdown; 05 tests this internal boundary.
            state.close()


def _child(case, device, directory, result):
    try:
        _run(case, device, directory)
        result.put(None)
    except BaseException:
        result.put(traceback.format_exc())


def _isolated(case, device, directory):
    context = multiprocessing.get_context("spawn")
    result = context.Queue()
    process = context.Process(target=_child, args=(case, device, directory, result))
    process.start()
    try:
        error = result.get(timeout=240)
        process.join(timeout=20)
        assert error is None, error
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=20)
        result.close()


@pytest.mark.parametrize("case,queue_enabled", [("eager", 0), ("eager", 1), ("program", 1)])
def test_jit_eager(test_config, tmp_path, monkeypatch, case, queue_enabled):
    if test_config.codegen_only or test_config.platform != "a2a3":
        pytest.skip("Requires an A2/A3 NPU and the optional torch adapter")
    pytest.importorskip("torch_npu")
    monkeypatch.setenv("TASK_QUEUE_ENABLE", str(queue_enabled))
    _isolated(case, test_config.device_id, str(tmp_path))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
