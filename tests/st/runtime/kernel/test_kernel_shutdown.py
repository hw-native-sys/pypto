# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Normal interpreter exit must finalize kernel resources before torch_npu teardown."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _child(case, device, directory):
    import atexit  # noqa: PLC0415
    import threading  # noqa: PLC0415
    import time  # noqa: PLC0415

    events = []
    report = Path(directory) / "shutdown.json"
    state = None
    worker = None

    # Registered before torch_npu: LIFO execution observes the completed framework
    # hook. No explicit close/drain appears in the child's ordinary execution.
    def report_exit():
        report.write_text(
            json.dumps(
                {
                    "events": events,
                    "state": state.state.value if state is not None else "unused",
                    "retained": state._worker is not None if state is not None else False,
                    "owner_alive": worker._owner.thread.is_alive() if worker is not None else False,
                }
            )
        )

    atexit.register(report_exit)
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415
    from pypto import CacheConfig  # noqa: PLC0415
    from pypto.runtime import RunConfig  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415
    from simpler.task_interface import ChipWorker  # noqa: PLC0415

    from tests.st.runtime.kernel.test_jit_eager import accumulate, add_constant  # noqa: PLC0415

    os.chdir(directory)
    os.environ.pop("PYPTO_PROG_BUILD_DIR", None)
    torch_npu.npu.set_device(device)
    state = get_process_kernel_state()
    original_shutdown = torch_npu._C._npu_shutdown

    def finalize_framework(success):
        events.append(["framework-finalize", state.state.value])
        return original_shutdown(success)

    torch_npu._C._npu_shutdown = finalize_framework
    original_init = ChipWorker.kernel_init
    original_finalize = ChipWorker.finalize

    def initialize(native_worker, *args, **kwargs):
        events.append(["init", threading.get_ident()])
        original_init(native_worker, *args, **kwargs)
        if case == "partial":
            raise RuntimeError("injected partial init failure")

    def finalize(native_worker):
        events.append(["finalize", threading.get_ident()])
        if case == "failure":
            raise RuntimeError("injected close failure")
        original_finalize(native_worker)

    ChipWorker.kernel_init = initialize
    ChipWorker.finalize = finalize
    if case == "unused":
        return
    config = RunConfig(platform="a2a3", device_id=device, cache_config=CacheConfig(enabled=False))
    outputs = []
    failures = []

    def execute():
        try:
            torch_npu.npu.set_device(device)
            stream = torch_npu.npu.Stream(device=device)
            with torch_npu.npu.stream(stream):
                x = torch.full((16, 16), 2.0, device=f"npu:{device}")
                out = torch.zeros_like(x)
                accumulate(x, 3.0, out, config=config)
                outputs.append((out, 6.0))
                other = torch.empty_like(x)
                add_constant(x, other, value=4, config=config)
                outputs.append((other, 6.0))
                if case == "delayed":
                    import _torch_npu_test  # noqa: PLC0415

                    torch_npu.npu.synchronize()
                    gate = _torch_npu_test.block_queue()
                    gate.wait_entered()

                    def release():
                        time.sleep(1)
                        events.append(["gate-release"])
                        gate.release()

                    threading.Thread(target=release, daemon=True).start()
                    accumulate(x, 1.0, out, config=config)
                    outputs[0] = (out, 8.0)
        except RuntimeError as exc:
            if case != "partial" or "injected partial init failure" not in str(exc):
                failures.append(str(exc))

    if case == "thread":
        caller = threading.Thread(target=execute)
        caller.start()
        caller.join()
        assert not caller.is_alive()
    else:
        execute()
    assert not failures, failures
    worker = state._worker
    assert worker is not None
    original_close = state.close

    def close():
        events.append(["close-start"])
        original_close()
        # Framework context must remain usable until the kernel close completes.
        for output, expected in outputs:
            torch.testing.assert_close(output.cpu(), torch.full((16, 16), expected))
        assert torch.ones(1, device=f"npu:{device}").item() == 1
        events.append(["close-done"])

    state.close = close
    if case == "repeat":
        assert torch_npu._C._npu_shutdown_synchronize()


def _run_child(case, device, directory, queue_enabled):
    env = dict(os.environ, TASK_QUEUE_ENABLE=str(queue_enabled))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.st.runtime.kernel.test_kernel_shutdown import _child; "
            "import sys; _child(sys.argv[1], int(sys.argv[2]), sys.argv[3])",
            case,
            str(device),
            str(directory),
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((directory / "shutdown.json").read_text())
    events = report["events"]
    names = [event[0] for event in events]
    if case == "unused":
        assert names == ["framework-finalize"]
        assert report["state"] == "uninitialized" and not report["retained"]
        return
    assert names.count("init") == names.count("finalize") == names.count("close-start") == 1
    assert next(e[1] for e in events if e[0] == "init") == next(e[1] for e in events if e[0] == "finalize")
    if case == "failure":
        assert report["state"] == "failed" and report["retained"]
        assert "resources retained until process exit" in result.stderr
        assert "close-done" not in names
    else:
        assert report["state"] == "closed" and not report["retained"] and not report["owner_alive"]
        assert names.index("close-done") < names.index("framework-finalize")
        assert "shutdown did not complete" not in result.stderr
    if case == "delayed":
        assert names.index("close-start") < names.index("gate-release") < names.index("finalize")


@pytest.mark.parametrize(
    "case,queue_enabled",
    [
        ("normal", 0),
        ("normal", 1),
        ("thread", 1),
        ("delayed", 1),
        ("unused", 1),
        ("partial", 1),
        ("repeat", 1),
        ("failure", 1),
    ],
)
def test_kernel_shutdown(test_config, tmp_path, case, queue_enabled):
    if test_config.codegen_only or test_config.platform != "a2a3":
        pytest.skip("Requires an A2/A3 NPU and the optional torch adapter")
    pytest.importorskip("torch_npu")
    _run_child(case, test_config.device_id, tmp_path, queue_enabled)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
