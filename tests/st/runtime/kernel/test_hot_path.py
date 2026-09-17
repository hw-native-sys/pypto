# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reproducible kernel host timings and counters; timings have no pass threshold."""

import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

SAMPLES = 1024
BATCH = 16


def _summary(samples):
    """Report nearest-rank host-return latency in microseconds."""
    values = sorted(samples)
    return {
        "samples": len(values),
        "p50_us": values[math.ceil(len(values) * 0.5) - 1] / 1000,
        "p99_us": values[math.ceil(len(values) * 0.99) - 1] / 1000,
    }


def _measure(call, state, native, counts):
    """Bound pending work between timing batches; exclude warmup and drain time."""
    state.drain()
    native._test_reset_counters()
    before = counts.copy()
    samples = []
    for index in range(SAMPLES):
        start = time.perf_counter_ns()
        call()
        samples.append(time.perf_counter_ns() - start)
        if (index + 1) % BATCH == 0:
            state.drain()
    return dict(
        _summary(samples),
        **native._test_counters(),
        **{name: value - before[name] for name, value in counts.items()},
    )


def _backlog(call, background, stream, state, native):
    """Hold the host queue so another stream has exactly N initial pending tickets."""
    import _torch_npu_test  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415

    # Reading npu_stream can flush the framework queue. Resolve it before
    # blocking that queue; the stream object stays alive throughout the test.
    stream_pointer = stream.npu_stream
    result = {}
    for pending in (0, 16, 256):
        samples = []
        polls = 0
        for _ in range(SAMPLES // BATCH):
            state.drain()
            gate = _torch_npu_test.block_queue()
            try:
                gate.wait_entered()
                with torch_npu.npu.stream(stream):
                    for _ in range(pending):
                        background()
                # Simpler requires the old caller tail to complete before a
                # stream switch. This callback runs after gate release, outside
                # host timings, while all measured calls remain queued.
                _torch_npu_test.queue_stream_fence(stream_pointer)
                assert len(state._submissions) == pending
                assert all(not ticket.done() for ticket in state._submissions)
                native._test_reset_counters()
                for _ in range(BATCH):
                    start = time.perf_counter_ns()
                    call()
                    samples.append(time.perf_counter_ns() - start)
                stats = native._test_counters()
                assert stats["done"] == BATCH * pending + BATCH * (BATCH - 1) // 2
                assert stats["device_sync"] == 0
                polls += stats["done"]
            finally:
                gate.release()
            state.drain()
        result[str(pending)] = dict(_summary(samples), done=polls, initial_pending=pending, batch_size=BATCH)
    return result


def _threaded(call, streams, device, state):
    """Measure producer batches with the completion boundary Simpler requires."""
    import torch_npu  # noqa: PLC0415

    result = {}
    for workers in (1, 2):
        state.drain()
        handoff = threading.Lock()
        started = []
        barrier = threading.Barrier(workers + 1, action=lambda: started.append(time.perf_counter_ns()))

        def produce(index):
            torch_npu.npu.set_device(device)
            with torch_npu.npu.stream(streams[index]):
                barrier.wait(timeout=30)
                for _ in range(SAMPLES // BATCH):
                    with handoff:
                        for _ in range(BATCH):
                            call(index)
                        # The pinned SDK rejects a different caller stream
                        # before this one completes. Include handoff cost.
                        state.drain()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(produce, index) for index in range(workers)]
            barrier.wait(timeout=30)
            start = started[0]
            for future in futures:
                future.result(timeout=180)
            host_end = time.perf_counter_ns()
        state.drain()
        end = time.perf_counter_ns()
        result[str(workers)] = {
            "calls": workers * SAMPLES,
            "batch_size": BATCH,
            "stream_handoff": "serialized_batch_and_drain",
            "producer_calls_per_second": workers * SAMPLES * 1e9 / (host_end - start),
            "completed_calls_per_second": workers * SAMPLES * 1e9 / (end - start),
        }
    return result


def _run(device, directory, queue_enabled):
    """Run all measurements in one isolated process and write JSON for JUnit."""
    import importlib  # noqa: PLC0415
    import platform  # noqa: PLC0415

    import simpler.callable_identity as sdk_identity  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415
    from pypto import CacheConfig, configure_cache  # noqa: PLC0415
    from pypto.runtime import _kernel_artifact  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415
    from pypto.torch import init, register  # noqa: PLC0415
    from pypto.torch.interop import CallSignature  # noqa: PLC0415
    from pypto.torch.launch import _load_native  # noqa: PLC0415
    from pypto.torch.registration import RegistrationSignature  # noqa: PLC0415

    from tests.st.runtime.kernel.test_jit_eager import add_constant  # noqa: PLC0415

    os.chdir(directory)
    os.environ.pop("PYPTO_PROG_BUILD_DIR", None)
    torch_npu.npu.set_device(device)
    configure_cache(CacheConfig(enabled=False))
    init(device=device)
    native = _load_native()
    state = get_process_kernel_state()
    counts = dict(descriptor=0, describe=0, registration_validate=0, cache_key=0)
    results = {
        "environment": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "chip": torch_npu.npu.get_device_name(device),
            "device": device,
            "machine": platform.machine(),
            "queue_enabled": queue_enabled,
            "adapter_test_counters": True,
        }
    }

    def counted(name, original):
        def wrapped(*args, **kwargs):
            counts[name] += 1
            return original(*args, **kwargs)

        return wrapped

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            sdk_identity,
            "build_chip_callable_descriptor",
            counted("descriptor", sdk_identity.build_chip_callable_descriptor),
        )
        patch.setattr(CallSignature, "describe_call", counted("describe", CallSignature.describe_call))
        patch.setattr(
            RegistrationSignature,
            "_validate",
            counted("registration_validate", RegistrationSignature._validate),
        )
        decorator = importlib.import_module("pypto.jit.decorator")
        patch.setattr(decorator, "make_cache_key", counted("cache_key", decorator.make_cache_key))
        op = register(add_constant, "pypto_hot_path::add", constexpr={"value": 4})
        x = torch.ones((16, 16), device=f"npu:{device}")
        outputs = [torch.empty_like(x) for _ in range(2)]
        streams = [torch_npu.npu.Stream(device=device) for _ in range(2)]
        torch_npu.npu.synchronize()
        entries = {
            "jit": lambda index=0: add_constant(x, outputs[index], value=4),
            "torch_ops": lambda index=0: op(x, outputs[index]),
        }
        with torch_npu.npu.stream(streams[0]):
            for entry in entries.values():
                entry()
            state.drain()
            assert counts["descriptor"] == 1
            for name, call in entries.items():
                results[name] = {}
                # The control restores the old per-call hash cost only. Same build,
                # operator, counters and batching; no timing-based pass assertion.
                for variant in ("uncached_control", "cached"):
                    with pytest.MonkeyPatch.context() as control:
                        if variant == "uncached_control":
                            control.setattr(
                                _kernel_artifact.KernelArtifact,
                                "identity",
                                lambda self: _kernel_artifact.callable_identity(self.load(), self.kernel_abi),
                            )
                        measured = _measure(call, state, native, counts)
                    assert measured["descriptor"] == (SAMPLES if variant == "uncached_control" else 0)
                    assert measured["describe"] == measured["cache_key"] == SAMPLES
                    assert measured["registration_validate"] == (SAMPLES if name == "torch_ops" else 0)
                    assert measured["device_sync"] == 0
                    results[name][variant] = measured
                    Path(directory, "hot-path.json").write_text(json.dumps(results, indent=2) + "\n")
                if queue_enabled:
                    results[name]["blocked_queue_backlog"] = _backlog(
                        call, lambda: call(1), streams[1], state, native
                    )
                results[name]["producers"] = _threaded(call, streams, device, state)
            # Count the current per-ticket graph cleanup without changing its safety contract.
            graph = torch_npu.npu.NPUGraph()
            before = counts["descriptor"]
            with torch_npu.npu.graph(graph, stream=streams[0]):
                for _ in range(4):
                    entries["jit"]()
            graph.replay()
            torch_npu.npu.synchronize()
            graph.reset()
            native._test_reset_counters()
            entries["jit"]()
            assert counts["descriptor"] == before
            assert native._test_counters()["device_sync"] == 4
            results["destroyed_graph"] = native._test_counters()
            state.drain()
        for output in outputs:
            torch.testing.assert_close(output.cpu(), torch.full((16, 16), 5.0))
        state.close()
    Path(directory, "hot-path.json").write_text(json.dumps(results, indent=2) + "\n")


@pytest.mark.parametrize("queue_enabled", [0, 1])
def test_kernel_hot_path(test_config, tmp_path, queue_enabled, record_property):
    """Persist timings/counters as JUnit properties without hardware speed thresholds."""
    if test_config.codegen_only or test_config.platform != "a2a3":
        pytest.skip("Requires an A2/A3 NPU and the optional torch adapter")
    pytest.importorskip("torch_npu")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.st.runtime.kernel.test_hot_path import _run; "
            "import sys; _run(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]))",
            str(test_config.device_id),
            str(tmp_path),
            str(queue_enabled),
        ],
        env=dict(os.environ, TASK_QUEUE_ENABLE=str(queue_enabled)),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PyPTO kernel shutdown did not complete" not in result.stderr
    report = json.loads((tmp_path / "hot-path.json").read_text())
    for name, value in report.items():
        record_property(name, json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
