# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Kernel DFX windows, including mixed torch/PyPTO execution and queued graph replay."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run(device: int, directory: str, mode: str, diagnostics: str) -> None:
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415
    from pypto import CacheConfig, configure_cache  # noqa: PLC0415
    from pypto.torch import begin_dfx, end_dfx, init  # noqa: PLC0415

    from tests.st.runtime.kernel.test_jit_eager import add_constant  # noqa: PLC0415

    os.chdir(directory)
    torch_npu.npu.set_device(device)
    configure_cache(CacheConfig(enabled=True, root=Path(directory) / "cache"))
    output = Path(directory) / "dfx"
    swimlane, deps = diagnostics != "deps", diagnostics != "swimlane"
    init(enable_chip_swimlane=4 if swimlane else 0, enable_dep_gen=deps, output_dir=output)
    stream = torch_npu.npu.Stream(device=device)
    mixed = mode.startswith("mixed_")
    launches = 2 if mixed else 1
    graph = None
    with torch_npu.npu.stream(stream):
        x = torch.full((16, 16), 2.0, device=f"npu:{device}")
        out = torch.empty_like(x)
        intermediate = torch.empty_like(x)
        staged = torch.empty_like(x)

        def run_ops() -> None:
            if mixed:
                # Each op consumes the preceding op's output, with no host waits.
                torch.add(x, 3, out=staged)
                add_constant(staged, intermediate, value=4)
                torch.mul(intermediate, 2, out=staged)
                add_constant(staged, out, value=4)
                torch.sub(out, 1, out=out)
            else:
                add_constant(x, out, value=4)

        run_ops()
        stream.synchronize()
        if mode in ("replay", "mixed_replay"):
            graph = torch_npu.npu.NPUGraph()
            with torch_npu.npu.graph(graph, stream=stream):
                with pytest.raises(RuntimeError, match="outside graph capture"):
                    begin_dfx()
                run_ops()
        with pytest.raises(RuntimeError, match="No kernel DFX"):
            end_dfx()
        for index in range(2):
            if index:
                # Work between windows must also stay out of the next capture.
                if graph is None:
                    run_ops()
                else:
                    graph.replay()
            if mixed:
                # A changed input rules out reusing the warmup/capture result.
                x.fill_(2 + index)
            begin_dfx()
            with pytest.raises(RuntimeError, match="already open"):
                begin_dfx()
            if graph is None:
                run_ops()
            else:
                graph.replay()
            # No caller synchronization: end_dfx must drain the host queue too.
            end_dfx()
            expected = float(21 + 2 * index) if mixed else 6.0
            torch.testing.assert_close(out.cpu(), torch.full((16, 16), expected))
            window = output if index == 0 else output / f"window_{index}"
            records, topology = window / "chip_swimlane_records.json", window / "deps.json"
            assert records.is_file() == swimlane
            assert topology.is_file() == deps
            if swimlane:
                captured = json.loads(records.read_text())
                assert captured["chip_swimlane_level"] == 4
                assert len(captured["metadata"]["run_boundaries"]) == launches
                assert captured["metadata"]["dropped_run_boundaries"] == 0
                # Runtime DFX records PyPTO tasks, not the intervening torch ops.
                assert len(captured["aicore_tasks"]) == launches
                assert all(0 < row[3] <= row[4] for row in captured["aicore_tasks"])
                assert len(captured["scheduler_tasks"]["records"]) == launches
            if deps:
                assert len(json.loads(topology.read_text())["tasks"]) == launches
            if swimlane and deps:
                merged = window / "merged_swimlane.json"
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "simpler_setup.tools.swimlane_converter",
                        str(records),
                        "-o",
                        str(merged),
                    ],
                    check=True,
                    timeout=60,
                )
                assert json.loads(merged.read_text())["traceEvents"]
    if graph is not None:
        graph.reset()


@pytest.mark.parametrize("queue_enabled", [0, 1])
@pytest.mark.parametrize(
    "mode,diagnostics",
    [
        ("eager", "both"),
        ("replay", "both"),
        ("eager", "swimlane"),
        ("eager", "deps"),
        ("mixed_eager", "both"),
        ("mixed_replay", "both"),
    ],
)
def test_kernel_dfx(test_config, tmp_path, queue_enabled, mode, diagnostics):
    if test_config.codegen_only or test_config.platform != "a2a3":
        pytest.skip("Requires an A2/A3 NPU and the optional torch adapter")
    pytest.importorskip("torch_npu")
    env = dict(os.environ, TASK_QUEUE_ENABLE=str(queue_enabled))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.st.runtime.kernel.test_dfx",
            str(test_config.device_id),
            str(tmp_path),
            mode,
            diagnostics,
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1].isdigit():
        _run(int(sys.argv[1]), *sys.argv[2:])
    else:
        pytest.main([__file__, "-v"])
