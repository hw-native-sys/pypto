# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Direct JIT capture acceptance in isolated NPU processes."""

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run(device, directory, case):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415
    from pypto import CacheConfig  # noqa: PLC0415
    from pypto.runtime import RunConfig  # noqa: PLC0415
    from pypto.runtime.kernel.abi import _NativeWorker  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415

    from tests.st.runtime.kernel.test_jit_eager import accumulate, add_constant  # noqa: PLC0415

    os.chdir(directory)
    os.environ.pop("PYPTO_PROG_BUILD_DIR", None)
    torch_npu.npu.set_device(device)
    config = RunConfig(
        platform="a2a3",
        device_id=device,
        cache_config=CacheConfig(enabled=case == "persistent", root=Path(directory) / "cache"),
    )
    x = torch.full((16, 16), 2.0, device=f"npu:{device}")
    out = torch.zeros_like(x)
    following = torch.empty_like(out)
    state = get_process_kernel_state()
    counts = dict(init=0, prepare=0)

    def counted(name, original):
        def wrapped(*args, **kwargs):
            counts[name] += 1
            return original(*args, **kwargs)

        return wrapped

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_NativeWorker, "init", counted("init", _NativeWorker.init))
        patch.setattr(_NativeWorker, "prepare", counted("prepare", _NativeWorker.prepare))
        rejected = case in ("cold", "generated", "binary", "second-cold", "new-variant")
        if case in ("generated", "binary"):
            artifact = accumulate._resolve_kernel_artifact((x, 3.0, out), dict(config=config))
            assert state._worker is None and artifact._loaded is None
            if case == "binary":
                artifact.load()
                assert state._worker is None
        elif case != "cold":
            accumulate(x, 3.0, out, config=config)
            if case in (
                "multi",
                "graphs",
                "streams",
                "recreate",
                "gc",
                "owners",
                "shutdown",
                "new-variant",
                "persistent",
            ):
                add_constant(out, following, value=4, config=config)
            torch_npu.npu.synchronize()
            out.zero_()
        warmed = counts.copy()
        graph = torch_npu.npu.NPUGraph()
        scalar = ctypes.c_float(3.0)
        with torch_npu.npu.graph(graph):
            if rejected:
                # Catch inside capture so the framework can finish a valid graph.
                with pytest.raises(RuntimeError, match="requires warmup outside capture"):
                    if case in ("second-cold", "new-variant"):
                        add_constant(out, following, value=5 if case == "new-variant" else 4, config=config)
                    else:
                        accumulate(x, scalar, out, config=config)
                following.copy_(out)
            else:
                assert accumulate(x, scalar, out, config=config) is out
                if case != "single":
                    add_constant(out, following, value=4, config=config)
        assert counts == warmed
        if rejected:
            graph.reset()
            return
        assert warmed == dict(init=1, prepare=1 if case == "single" else 2)
        scalar.value = 99.0
        tensors = [x, out, following]
        del x, out, following
        graphs = [graph]
        del graph
        graph = _replay_case(case, graphs, tensors, config, torch_npu)
        assert counts == warmed
        globals()["retained_graph"] = graph


def _replay_case(case, graphs, tensors, config, torch_npu):
    import gc  # noqa: PLC0415
    import weakref  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from tests.st.runtime.kernel.test_jit_eager import add_constant  # noqa: PLC0415

    graph = graphs.pop()
    x, out, following = tensors
    tensors.clear()
    device = config.device_id
    replay_stream = torch_npu.npu.Stream(device=device)
    expected = 0.0
    with torch_npu.npu.stream(replay_stream):
        for value in range(2, 7):
            x.fill_(value)
            graph.replay()
            expected += value * 3.0
            torch.testing.assert_close(out.cpu(), torch.full((16, 16), expected))
            if case != "single":
                torch.testing.assert_close(following.cpu(), torch.full((16, 16), expected + 4))
    if case in ("graphs", "streams"):
        second = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(second):
            add_constant(following, out, value=4, config=config)
        current = torch_npu.npu.current_stream()
        for _ in range(5):
            if case == "streams":
                replay_stream.wait_stream(current)
                with torch_npu.npu.stream(replay_stream):
                    graph.replay()
                current.wait_stream(replay_stream)
            else:
                graph.replay()
            second.replay()
            expected += 6 * 3 + 8
        torch_npu.npu.synchronize()
        torch.testing.assert_close(out.cpu(), torch.full((16, 16), expected))
        globals()["second_graph"] = second
    elif case in ("recreate", "gc"):
        if case == "recreate":
            graph.reset()
        else:
            reference = weakref.ref(graph)
            del graph
            gc.collect()
            assert reference() is None
        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph):
            add_constant(out, following, value=4, config=config)
        graph.replay()
        torch.testing.assert_close(following.cpu(), torch.full((16, 16), expected + 4))
    elif case == "owners":
        input_pointer = x.data_ptr()
        del x
        gc.collect()
        pressure = [torch.full_like(out, 999) for _ in range(32)]
        graph.replay()
        expected += 6 * 3
        torch.testing.assert_close(out.cpu(), torch.full((16, 16), expected))
        torch.testing.assert_close(following.cpu(), torch.full((16, 16), expected + 4))
        assert all(tensor.data_ptr() != input_pointer for tensor in pressure)
    elif case == "shutdown":
        with torch_npu.npu.stream(replay_stream):
            for _ in range(20):
                graph.replay()
        # Ordinary process exit must drain pending replay before Worker close.
    return graph


@pytest.mark.parametrize(
    "case",
    [
        "cold",
        "generated",
        "binary",
        "second-cold",
        "new-variant",
        "single",
        "multi",
        "graphs",
        "streams",
        "persistent",
        "owners",
        "recreate",
        "gc",
        "shutdown",
    ],
)
@pytest.mark.parametrize("queue_enabled", [0, 1])
def test_capture(test_config, tmp_path, case, queue_enabled):
    if test_config.codegen_only or test_config.platform != "a2a3":
        pytest.skip("Requires an A2/A3 NPU")
    pytest.importorskip("torch_npu")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.st.runtime.kernel.test_capture import _run; "
            "import sys; _run(int(sys.argv[1]), sys.argv[2], sys.argv[3])",
            str(test_config.device_id),
            str(tmp_path),
            case,
        ],
        env=dict(os.environ, TASK_QUEUE_ENABLE=str(queue_enabled)),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PyPTO kernel shutdown did not complete" not in result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
