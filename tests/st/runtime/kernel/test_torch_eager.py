# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Numerical eager execution on real Torch NPU streams, isolated per process.

Run with --platform=a2a3 and a built optional pypto._torch_npu adapter.
The codegen smoke test also runs without an NPU; execution cases skip there.
"""

import importlib.util
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pypto.language as pl
import pypto.torch
import pytest
import torch
from pypto.pypto_core import passes

_RUNTIME_KINDS = {
    "tensormap_and_ringbuffer": passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER,
    "host_build_graph": passes.RuntimeKind.HOST_BUILD_GRAPH,
}


@pl.jit
def eager_add_bias(x: pl.Tensor, bias: pl.Scalar[pl.FP32], out: pl.Out[pl.Tensor]):
    """Use a runtime scalar so repeated calls exercise argument snapshots."""
    with pl.at(level=pl.Level.CORE_GROUP):
        value = pl.load(x, [0, 0], [32, 64])
        result = pl.add(value, bias)
        pl.store(result, [0, 0], out)
    return out


def _single_call(x, reference):
    out = torch.full_like(x, float("nan"))
    result = eager_add_bias(x, 1.25, out)
    assert result is out
    return [out], [reference + 1.25]


def _consecutive_calls(x, reference):
    # Queue a dependency chain without host reads or intermediate synchronization.
    # Distinct poisoned outputs catch skipped submissions and stale scalar values.
    outputs = [torch.full_like(x, float("nan")) for _ in range(8)]
    expected = []
    value = x
    for index, out in enumerate(outputs):
        bias = (index + 1) / 8.0
        value = eager_add_bias(value, bias, out)
        assert value is out
        reference = reference + bias
        expected.append(reference)
    return outputs, expected


def _interleaved_calls(x, reference):
    outputs = [torch.full_like(x, float("nan")) for _ in range(4)]
    actual, expected = [], []
    value = x
    for bias, out in zip((0.5, -1.25, 2.0, -0.75), outputs, strict=True):
        # These Torch operations dispatch to torch_npu on the selected stream.
        # Exercise Torch -> PyPTO -> Torch dependencies in both directions,
        # including newly allocated Torch inputs whose Python references expire.
        produced = torch.add(value, 0.25)
        result = eager_add_bias(produced, bias, out)
        assert result is out
        value = torch.relu(result * 0.5 - 0.125)
        kernel_reference = reference + 0.25 + bias
        reference = torch.relu(kernel_reference * 0.5 - 0.125)
        actual.extend((out, value))
        expected.extend((kernel_reference, reference))
    return actual, expected


_CASES = {
    "single": _single_call,
    "consecutive": _consecutive_calls,
    "interleaved": _interleaved_calls,
}


def _run_eager_case(case: str, device: int, runtime: str, stream_kind: str) -> None:
    # Initialize only in the fresh child: other STs must not claim its mode.
    torch_npu = importlib.import_module("torch_npu")

    if not torch_npu.npu.is_available():
        sys.exit(77)  # Report missing hardware to pytest without initializing it in the parent.
    torch_npu.npu.set_device(device)
    pypto.torch.init(device=device, runtime=runtime)
    stream = torch_npu.npu.current_stream() if stream_kind == "default" else torch_npu.npu.Stream()
    with torch_npu.npu.stream(stream):
        # Binary fractions make the CPU reference exact, with mixed signs and
        # varying elements so a no-op or wrong ordering cannot accidentally pass.
        reference = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64) / 64.0 - 8.0
        x = reference.to(device=f"npu:{device}")
        actual, expected = _CASES[case](x, reference)
        # This is the only explicit synchronization in each numerical scenario.
        stream.synchronize()
        for output, golden in zip(actual, expected, strict=True):
            torch.testing.assert_close(output.cpu(), golden, rtol=0, atol=0)
    # Normal child exit also exercises the framework-ordered Worker shutdown.


@pytest.fixture
def eager_npu(test_config):
    if test_config.codegen_only or test_config.platform != "a2a3":
        pytest.skip("Eager numerical execution requires an onboard a2a3 NPU")
    # Keep the parent device-free; even probing availability belongs in the child.
    for dependency in ("torch_npu", "pypto._torch_npu"):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f"Eager numerical execution requires {dependency}")
    return test_config.device_id


@pytest.mark.parametrize("runtime", list(_RUNTIME_KINDS))
@pytest.mark.parametrize("stream_kind", ["default", "nondefault"])
@pytest.mark.parametrize("case", list(_CASES))
def test_torch_eager_execution(eager_npu, runtime, stream_kind, case):
    # exec a fresh interpreter rather than fork an already initialized runtime.
    script = (
        "import runpy, sys; "
        "namespace = runpy.run_path(sys.argv[1]); "
        "namespace['_run_eager_case'](sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).resolve()),
            case,
            str(eager_npu),
            runtime,
            stream_kind,
        ],
        env={
            **os.environ,
            # Preserve paths supplied by the ST harness, including the runtime submodule.
            "PYTHONPATH": os.pathsep.join(sys.path),
            "TASK_QUEUE_ENABLE": "1",
            "ASCEND_LAUNCH_BLOCKING": "0",
            "PYTHONOPTIMIZE": "0",  # Keep child assertions enabled outside pytest's rewriting.
        },
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode == 77:
        pytest.skip("torch_npu reports no available NPU in the child process")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("runtime", list(_RUNTIME_KINDS))
def test_eager_kernel_codegen(test_config, runtime):
    """Fully compile the ST kernel for both runtimes without claiming an execution mode."""
    x = torch.empty((32, 64), dtype=torch.float32)
    out = torch.empty_like(x)
    with passes.PassContext([], runtime=_RUNTIME_KINDS[runtime]):
        artifact = eager_add_bias.compile(x, 1.25, out, config=replace(test_config, codegen_only=True))
    assert list(artifact.output_dir.glob("orchestration/*.cpp"))
    assert list(artifact.output_dir.glob("ptoas/*.pto"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
