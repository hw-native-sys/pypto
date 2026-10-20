# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Full 1C2V MIX prefetch must preserve tiles live across SDMA submit/wait."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pypto.language as pl
import pytest


@pl.jit.incore
def cube_copy(
    a: pl.Tensor[[384, 16], pl.BF16],
    b: pl.Tensor[[16, 16], pl.BF16],
    out: pl.InOut[pl.Tensor[[384, 16], pl.FP32]],
):
    worker = pl.tile.get_block_idx()
    left = pl.load(a, [worker * 16, 0], [16, 16])
    right = pl.load(b, [0, 0], [16, 16])
    value = pl.matmul(left, right, out_dtype=pl.FP32)
    out = pl.store(value, [worker * 16, 0], out)
    return out


@pl.jit.incore
def vector_prefetch(
    x: pl.Tensor[[48, 1024], pl.FP32],
    prefetch_src: pl.Tensor[[4096], pl.INT8],
    issuer_mask: pl.Tensor[[48], pl.INT32],
    out: pl.InOut[pl.Tensor[[48, 1024], pl.FP32]],
    done: pl.InOut[pl.Tensor[[768], pl.INT32]],
    post_count: pl.constexpr = 1,
):
    pl.func_attr({"split_aiv": True})
    worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
    # Keep business data live while the context uses its hidden UB scratch.
    value = pl.load(x, [worker, 0], [1, 1024])
    complete = pl.const(2, pl.INT32)
    if pl.read(issuer_mask, [worker]) != 0:
        ctx = pl.prefetch.make_context()
        for _ in pl.range(post_count - 1):
            pl.prefetch.async_prefetch(prefetch_src, ctx)
        event = pl.prefetch.async_prefetch(prefetch_src, ctx)
        session = pl.prefetch.session(ctx)
        if pl.prefetch.wait(event, session):
            complete = pl.const(2, pl.INT32)
        else:
            complete = pl.const(1, pl.INT32)
    if complete == 2:
        out = pl.store(pl.add(value, 1.0), [worker, 0], out)
    pl.write(done, [worker * 16], complete)
    pl.system.cacheinvalid(done, [1], [worker * 16])
    pl.system.fence()
    return out, done


@pl.jit.inline(auto_scope=False)
def mixed(
    a: pl.Tensor,
    b: pl.Tensor,
    x: pl.Tensor,
    prefetch_src: pl.Tensor,
    issuer_mask: pl.Tensor,
    cube_out: pl.InOut[pl.Tensor],
    vector_out: pl.InOut[pl.Tensor],
    done: pl.InOut[pl.Tensor],
    post_count: pl.constexpr = 1,
):
    with pl.cluster(name_hint="sdma_mix_smoke"):
        cube_out = cube_copy(a, b, cube_out)
        vector_out, done = vector_prefetch(x, prefetch_src, issuer_mask, vector_out, done, post_count)
    return cube_out, vector_out, done


@pl.jit
def entry(
    a: pl.Tensor[[384, 16], pl.BF16],
    b: pl.Tensor[[16, 16], pl.BF16],
    x: pl.Tensor[[48, 1024], pl.FP32],
    prefetch_src: pl.Tensor[[4096], pl.INT8],
    issuer_mask: pl.Tensor[[48], pl.INT32],
    cube_out: pl.InOut[pl.Tensor[[384, 16], pl.FP32]],
    vector_out: pl.InOut[pl.Tensor[[48, 1024], pl.FP32]],
    done: pl.InOut[pl.Tensor[[768], pl.INT32]],
    post_count: pl.constexpr = 1,
):
    with pl.spmd(24, sync_start=True):
        cube_out, vector_out, done = mixed(
            a, b, x, prefetch_src, issuer_mask, cube_out, vector_out, done, post_count
        )
    return cube_out, vector_out, done


def _run(device, directory, runtime, scenario, issuers, rounds=100, post_count=1):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415
    from pypto import CacheConfig, configure_cache  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415
    from pypto.torch import init  # noqa: PLC0415

    os.chdir(directory)
    os.environ.pop("PYPTO_PROG_BUILD_DIR", None)
    configure_cache(CacheConfig(enabled=False))
    torch_npu.npu.set_device(device)
    init(runtime=runtime, enable_sdma=True)
    state = get_process_kernel_state()
    a = torch.zeros((384, 16), dtype=torch.bfloat16, device=f"npu:{device}")
    b = torch.eye(16, dtype=torch.bfloat16).to(a.device)
    x = torch.zeros((48, 1024), dtype=torch.float32, device=a.device)
    source = torch.zeros(4096, dtype=torch.int8, device=a.device)
    mask_cpu = torch.ones(48, dtype=torch.int32)
    if issuers in ("none", "single"):
        mask_cpu.zero_()
        if issuers == "single":
            mask_cpu[32] = 1
    mask = mask_cpu.to(a.device)
    cube_out = torch.full((384, 16), -999.0, dtype=torch.float32, device=a.device)
    vector_out = torch.full_like(x, -999)
    done = torch.zeros(768, dtype=torch.int32, device=a.device)
    values = (a, b, x, source, mask, cube_out, vector_out, done)
    graph = None
    try:
        entry(*values, post_count=post_count)
        torch_npu.npu.synchronize()
        assert torch.equal(vector_out.cpu(), torch.ones((48, 1024))), (
            f"warmup failed; completion={done.cpu()[::16].tolist()}"
        )
        registration = next(iter(state._registrations.values()))
        assert registration.artifact.requires_sdma
        orchestration = "\n".join(
            path.read_text() for path in (registration.artifact.output_dir / "orchestration").glob("*.cpp")
        )
        assert "MixedKernels mixed_0 = {0, 1, 1}" in orchestration
        if scenario == "replay":
            graph = torch_npu.npu.NPUGraph()
            with torch_npu.npu.graph(graph):
                entry(*values, post_count=post_count)
        expected_done = torch.zeros(768, dtype=torch.int32)
        expected_done[::16] = 2
        for iteration in range(rounds):
            expected_a = ((torch.arange(384 * 16).reshape(384, 16) + iteration * 3) % 13).bfloat16()
            expected_x = (torch.arange(48 * 1024).reshape(48, 1024) + iteration * 100000).float()
            a.copy_(expected_a)
            x.copy_(expected_x)
            cube_out.fill_(-999)
            vector_out.fill_(-999)
            done.zero_()
            if graph is None:
                entry(*values, post_count=post_count)
            else:
                graph.replay()
            torch_npu.npu.synchronize()
            assert torch.equal(cube_out.cpu(), expected_a.float()), f"cube round={iteration}"
            actual_vector = vector_out.cpu()
            if not torch.equal(actual_vector, expected_x + 1):
                bad = (actual_vector != expected_x + 1).nonzero()
                Path("failure.json").write_text(
                    json.dumps(
                        {
                            "iteration": iteration,
                            "positions": bad.tolist(),
                            "actual": actual_vector[bad[:, 0], bad[:, 1]].tolist(),
                            "expected": (expected_x + 1)[bad[:, 0], bad[:, 1]].tolist(),
                            "done": done.cpu().reshape(48, 16).tolist(),
                        }
                    )
                )
            assert torch.equal(actual_vector, expected_x + 1), (
                f"vector round={iteration}; mismatches="
                f"{(actual_vector != expected_x + 1).sum(dim=1).tolist()}; "
                f"completion={done.cpu()[::16].tolist()}; details in failure.json"
            )
            assert torch.equal(done.cpu(), expected_done), f"completion/guard round={iteration}"
        Path("result.json").write_text(
            json.dumps(
                dict(
                    runtime=runtime, scenario=scenario, issuers=issuers, rounds=rounds, post_count=post_count
                )
            )
        )
    finally:
        if graph is not None:
            graph.reset()
        state.close()


@pytest.mark.platforms("a2a3")
@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
@pytest.mark.parametrize("scenario", ["eager", "replay"])
@pytest.mark.parametrize("issuers,post_count", [("none", 1), ("single", 1), ("all", 1), ("all", 65)])
def test_kernel_mix_prefetch_preserves_live_tiles(
    test_config, tmp_path, runtime, scenario, issuers, post_count
):
    if test_config.codegen_only or test_config.platform != "a2a3":
        pytest.skip("Requires an A2/A3 NPU and the native torch adapter")
    pytest.importorskip("torch_npu")
    logs = tmp_path / "ascend"
    logs.mkdir()
    env = dict(os.environ, ASCEND_PROCESS_LOG_PATH=str(logs))
    with (tmp_path / "run.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from tests.st.runtime.kernel.test_mix_prefetch import _run; "
                "import sys; _run(int(sys.argv[1]), *sys.argv[2:6], post_count=int(sys.argv[6]))",
                str(test_config.device_id),
                str(tmp_path),
                runtime,
                scenario,
                issuers,
                str(post_count),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )
    output = (tmp_path / "run.log").read_text()
    assert result.returncode == 0, output
    assert "PyPTO kernel shutdown did not complete" not in output
    assert json.loads((tmp_path / "result.json").read_text())["rounds"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
