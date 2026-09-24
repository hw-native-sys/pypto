# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO-to-Simpler HBG Device tensors, Host scalars and access-site rejection."""

import os
import subprocess
import sys

import pypto.language as pl
import pytest


@pl.jit
def add(x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.add(pl.load(x, [0, 0], [16, 16]), 1.0), [0, 0], out)
    return out


@pl.jit
def repeat_add(
    x: pl.Tensor[[16, 16], pl.FP32],
    count: pl.Scalar[pl.INT32],
    out: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
):
    for _ in pl.range(count):
        with pl.at(level=pl.Level.CORE_GROUP):
            pl.store(
                pl.add(pl.load(out, [0, 0], [16, 16]), pl.load(x, [0, 0], [16, 16])),
                [0, 0],
                out,
            )
    return out


@pl.jit
def read_on_host(
    x: pl.Tensor[[16, 16], pl.FP32],
    control: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
):
    count: pl.Scalar[pl.INT32] = pl.tensor.read(control, [0])
    for _ in pl.range(count):
        with pl.at(level=pl.Level.CORE_GROUP):
            pl.store(pl.add(pl.load(x, [0, 0], [16, 16]), 1.0), [0, 0], out)
    return out


def _run(case, device, directory):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415
    from pypto import CacheConfig, configure_cache  # noqa: PLC0415
    from pypto.ir import _kernel_compile  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415
    from pypto.torch import init  # noqa: PLC0415

    os.chdir(directory)
    os.environ.pop("PYPTO_PROG_BUILD_DIR", None)
    configure_cache(CacheConfig(enabled=False))
    torch_npu.npu.set_device(device)
    init(runtime="host_build_graph")
    state = get_process_kernel_state()
    x = torch.full((16, 16), 2.0, device=f"npu:{device}")
    out = torch.zeros_like(x)
    if case == "runtime_read":
        control = torch.ones((1,), dtype=torch.int32, device=x.device)
        torch_npu.npu.synchronize()
        # Deliberately bypass the compiler guard to exercise Simpler's independent
        # access-site check with a real PyPTO-generated orchestration and adapter.
        error = "simpler_kernel_mode_launch failed|working operator name is PyPTOKernel"
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_kernel_compile, "validate_hbg_kernel_orchestration", lambda program: None)
            with pytest.raises(RuntimeError, match=error):
                read_on_host(x, control, out)
                state.drain()
        with pytest.raises(RuntimeError, match=error):
            state.close()
        assert state._worker is None and not state._submissions
        return
    try:
        if case == "add":
            assert add(x, out) is out
            state.drain()
            torch.testing.assert_close(out.cpu(), torch.full((16, 16), 3.0))
        else:
            for count in (1, 3, 2):
                out.zero_()
                assert repeat_add(x, count, out) is out
                state.drain()
                torch.testing.assert_close(out.cpu(), torch.full((16, 16), 2.0 * count))
            # Different scalar values change the Host graph, not the callable.
            assert len(state._registrations) == 1
    finally:
        state.close()


@pytest.mark.parametrize("case", ["add", "scalar", "runtime_read"])
@pytest.mark.parametrize("queue_enabled", [0, 1])
def test_hbg_contract(test_config, tmp_path, case, queue_enabled):
    if test_config.codegen_only or test_config.platform != "a2a3":
        pytest.skip("Requires an A2/A3 NPU and the optional torch adapter")
    pytest.importorskip("torch_npu")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.st.runtime.kernel.test_hbg_contract import _run; "
            "import sys; _run(sys.argv[1], int(sys.argv[2]), sys.argv[3])",
            case,
            str(test_config.device_id),
            str(tmp_path),
        ],
        env=dict(os.environ, TASK_QUEUE_ENABLE=str(queue_enabled)),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if case == "runtime_read":
        assert "HBG kernel Host orchestration cannot read Device tensor data" in result.stdout + result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
