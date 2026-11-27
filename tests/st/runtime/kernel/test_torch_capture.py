# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Native Torch graph capture after eager warmup, with no graph method hooks."""

import ctypes
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from pypto.torch import init

from tests.st.runtime.kernel.test_torch_eager import eager_add_bias


def _reference_input() -> torch.Tensor:
    return torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64) / 64


def _run_capture_case(case: str, device: int, runtime: str, reference: torch.Tensor) -> None:
    torch_npu = importlib.import_module("torch_npu")
    if not torch_npu.npu.is_available():
        sys.exit(77)
    npu = torch_npu.npu
    npu.set_device(device)
    graph_type = npu.NPUGraph
    original_methods = tuple(
        getattr(graph_type, name) for name in ("capture_begin", "capture_end", "replay", "reset")
    )
    init(device=device, runtime=runtime)
    stream = npu.Stream(device=device)
    replay_stream = npu.Stream(device=device)
    graphs = []
    frames = []
    with npu.stream(stream):
        x = reference.to(device=f"npu:{device}")
        warm = torch.empty_like(x)
        if case == "cold":
            graph = graph_type()
            with npu.graph(graph, stream=stream):
                with pytest.raises(RuntimeError, match="not prepared"):
                    eager_add_bias(x, 1.5, warm)
                torch.add(x, x, out=warm)
            graph.replay()
            stream.synchronize()
            torch.testing.assert_close(warm.cpu(), reference * 2, rtol=0, atol=0)
            graph.reset()
            return
        eager_add_bias(x, 1.5, warm)
        stream.synchronize()
        if case == "joined":
            graph = graph_type()
            with npu.graph(graph, stream=stream):
                eager_add_bias(x, 1.5, warm)
                replay_stream.wait_stream(stream)
                with npu.stream(replay_stream), pytest.raises(ValueError, match="single NPU stream"):
                    eager_add_bias(x, 2.0, warm)
                stream.wait_stream(replay_stream)
            graph.replay()
            stream.synchronize()
            torch.testing.assert_close(warm.cpu(), reference + 1.5, rtol=0, atol=0)
            graph.reset()
            eager_add_bias(x, 2.0, warm)
            stream.synchronize()
            torch.testing.assert_close(warm.cpu(), reference + 2.0, rtol=0, atol=0)
            return
        for _ in range(3):
            pre, mid, out, result = [torch.full_like(x, float("nan")) for _ in range(4)]
            graph = graph_type()
            scalar = ctypes.c_float(1.5)
            with npu.graph(graph, stream=stream):
                torch.add(x, 0.25, out=pre)
                assert eager_add_bias(pre, scalar, mid) is mid
                assert eager_add_bias(mid, -0.5, out) is out
                torch.mul(out, 2, out=result)
            scalar.value = 99  # Replay uses the capture-time scalar snapshot.
            graphs.append(graph)
            frames.append((pre, mid, out, result))
        # End/reset/replay remain the original framework methods.
        assert original_methods == tuple(
            getattr(graph_type, name) for name in ("capture_begin", "capture_end", "replay", "reset")
        )
        stream.synchronize()
    with npu.stream(replay_stream):
        for offset in (0, 3, -2):
            x.copy_(reference + offset)
            for graph in graphs:
                graph.replay()
            replay_stream.synchronize()
            for *_, result in frames:
                torch.testing.assert_close(result.cpu(), (reference + offset + 1.25) * 2, rtol=0, atol=0)
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415

    state = get_process_kernel_state()
    captured = [ticket for ticket in state._submissions if ticket.capture is not None]
    assert len(captured) == 6
    eager_add_bias(x, 2, warm)  # Live graph owners survive subsequent ordinary calls.
    assert all(not ticket.retired for ticket in captured)
    npu.synchronize()
    graphs.pop(0).reset()
    eager_add_bias(x, 2, warm)  # Reclaim one model while two remain live.
    assert all(ticket.retired for ticket in captured[:2])
    assert all(not ticket.retired for ticket in captured[2:])
    npu.synchronize()
    with npu.stream(replay_stream):
        x.fill_(7)
        for graph in graphs:
            graph.replay()
        replay_stream.synchronize()
    for *_, result in frames[1:]:
        torch.testing.assert_close(result.cpu(), torch.full_like(reference, 16.5), rtol=0, atol=0)
    for graph in graphs:
        graph.reset()
    # The next ordinary op reclaims every remaining model before its enqueue.
    eager_add_bias(x, 2, warm)
    assert all(ticket.retired for ticket in captured)
    npu.synchronize()
    torch.testing.assert_close(warm.cpu(), torch.full_like(reference, 9), rtol=0, atol=0)


@pytest.mark.parametrize("task_queue", ["0", "1"])
@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
@pytest.mark.parametrize("case", ["cold", "multi", "joined"])
def test_torch_capture_execution(test_config, runtime, case, task_queue):
    if test_config.codegen_only or test_config.platform != "a2a3":
        pytest.skip("Torch capture execution requires an onboard a2a3 NPU")
    for dependency in ("torch_npu", "pypto._torch_npu"):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f"Torch capture execution requires {dependency}")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; ns = runpy.run_path(sys.argv[1]); "
            "ns['_run_capture_case'](sys.argv[2], int(sys.argv[3]), sys.argv[4], ns['_reference_input']())",
            str(Path(__file__).resolve()),
            case,
            str(test_config.device_id),
            runtime,
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(sys.path),
            "TASK_QUEUE_ENABLE": task_queue,
            "ASCEND_LAUNCH_BLOCKING": "0",
            "PYTHONOPTIMIZE": "0",
        },
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode == 77:
        pytest.skip("torch_npu reports no available NPU in the child process")
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
