# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Registered torch operators share JIT execution and survive compiler tracing."""

import os
import subprocess
import sys

import pytest


def _run(case, device, directory):
    import importlib  # noqa: PLC0415

    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415
    from pypto import CacheConfig  # noqa: PLC0415
    from pypto.jit.decorator import JITFunction  # noqa: PLC0415
    from pypto.runtime import RunConfig  # noqa: PLC0415
    from pypto.runtime.kernel.abi import _NativeWorker  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415
    from pypto.torch import register  # noqa: PLC0415
    from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: PLC0415

    from tests.st.runtime.kernel.test_jit_eager import accumulate, add_constant  # noqa: PLC0415

    os.chdir(directory)
    os.environ.pop("PYPTO_PROG_BUILD_DIR", None)
    torch_npu.npu.set_device(device)
    config = RunConfig(platform="a2a3", device_id=device, cache_config=CacheConfig(enabled=False))
    state = get_process_kernel_state()
    counts = dict(compile=0, init=0, prepare=0)
    compiler = importlib.import_module("pypto.ir.compile")

    def counted(name, original):
        def wrapped(*args, **kwargs):
            counts[name] += 1
            return original(*args, **kwargs)

        return wrapped

    def forbidden_compile(*args, **kwargs):
        raise AssertionError("registered operator used explicit program compilation")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(compiler, "_compile_impl", counted("compile", compiler._compile_impl))
        patch.setattr(JITFunction, "compile", forbidden_compile)
        patch.setattr(_NativeWorker, "init", counted("init", _NativeWorker.init))
        patch.setattr(_NativeWorker, "prepare", counted("prepare", _NativeWorker.prepare))
        update = register(accumulate, "pypto_ops_st::update", config=config)
        add = register(add_constant, "pypto_ops_st::add", constexpr={"value": 4}, config=config)
        assert update is register(accumulate, "pypto_ops_st::update", config=config)
        assert state._worker is None
        meta_x = torch.empty(16, 16, device="meta")
        meta_out = torch.empty_like(meta_x)
        assert update(meta_x, 2.0, meta_out) is meta_out
        assert add(meta_x, meta_out) is meta_out
        with FakeTensorMode():
            fake_x = torch.empty(16, 16, device=f"npu:{device}")
            fake_out = torch.empty_like(fake_x)
            assert update(fake_x, 2.0, fake_out) is fake_out
        assert counts == dict(compile=0, init=0, prepare=0)
        assert state._worker is None

        stream = torch_npu.npu.Stream(device=device)
        with torch_npu.npu.stream(stream):
            bad = torch.zeros((32, 16), device=f"npu:{device}")
            with pytest.raises(RuntimeError, match="incompatible shape"):
                update(bad, 1.0, bad)
            assert counts == dict(compile=0, init=0, prepare=0)
            x = torch.full((16, 16), 2.0, device=f"npu:{device}")
            acc = torch.zeros_like(x)
            assert accumulate(x, 1.0, acc, config=config) is acc
            worker = state._worker
            assert update(x=x, step=2.0, acc=acc) is acc
            assert torch.ops.pypto_ops_st.update(x, 3.0, acc) is acc
            torch.testing.assert_close(acc.cpu(), torch.full((16, 16), 12.0))
            assert counts == dict(compile=1, init=1, prepare=1)
            out = torch.empty_like(x)
            assert add_constant(x, out, value=4, config=config) is out
            assert add(x, out) is out
            torch.testing.assert_close(out.cpu(), torch.full((16, 16), 6.0))
            assert counts == dict(compile=2, init=1, prepare=2)
            assert state._worker is worker and len(state._registrations) == 2

            if case in ("compile", "capture"):

                def call(x, acc):
                    result = update(x + 1, 2.0, acc)
                    return result, result + 3

                compiled = torch.compile(call, backend="aot_eager", fullgraph=True)
                for expected in (18.0, 24.0, 30.0):
                    result, following = compiled(x, acc)
                    assert result is acc
                    torch.testing.assert_close(acc.cpu(), torch.full((16, 16), expected))
                    torch.testing.assert_close(following.cpu(), torch.full((16, 16), expected + 3))
                assert counts == dict(compile=2, init=1, prepare=2)
                assert state._worker is worker
                if case == "capture":
                    from tests.st.runtime.kernel.test_capture import _replay  # noqa: PLC0415

                    torch_npu.npu.synchronize()
                    acc.zero_()
                    graph = torch_npu.npu.NPUGraph()
                    with torch_npu.npu.graph(graph):
                        result, following = compiled(x, acc)
                    assert result is acc
                    expected = 0.0
                    for value in (1.0, 4.0, 2.0):
                        x.fill_(value)
                        _replay(graph)
                        expected += (value + 1) * 2
                        torch.testing.assert_close(acc.cpu(), torch.full((16, 16), expected))
                        torch.testing.assert_close(following.cpu(), torch.full((16, 16), expected + 3))
                    assert counts == dict(compile=2, init=1, prepare=2)
                    assert state._worker is worker
                    # Keep the compiled graph alive through ordinary process exit.
                    globals()["retained_compiled_graph"] = graph
        # Ordinary process exit exercises 07; no caller-owned close or drain.


def _isolated(case, device, directory, queue_enabled):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.st.runtime.kernel.test_torch_ops import _run; "
            "import sys; _run(sys.argv[1], int(sys.argv[2]), sys.argv[3])",
            case,
            str(device),
            str(directory),
        ],
        env=dict(os.environ, TASK_QUEUE_ENABLE=str(queue_enabled)),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PyPTO kernel shutdown did not complete" not in result.stderr


@pytest.mark.parametrize("case", ["eager", "compile", "capture"])
@pytest.mark.parametrize("queue_enabled", [0, 1])
def test_torch_ops(test_config, tmp_path, case, queue_enabled):
    if test_config.codegen_only or test_config.platform != "a2a3":
        pytest.skip("Requires an A2/A3 NPU and the optional torch adapter")
    pytest.importorskip("torch_npu")
    _isolated(case, test_config.device_id, tmp_path, queue_enabled)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
