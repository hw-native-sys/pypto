# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate PyPTO SHMEM from vLLM Ascend TP2 workers with prepared Qwen3 artifacts.

Launch on two allocated Ascend NPUs. Point --model at the Qwen3-14B checkpoint,
--weights at the CPU-prepared TP2 shards, and --reference at standalone PyPTO
logits saved by the matching Qwen3 test. This exercises the vLLM multiprocess
worker and its process groups through RPC with PyPTO serving the fixed request.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from pypto.jit.decorator import JITFunction
from pypto.runtime import ChipWorker, device_runner
from pypto.runtime.kernel_compiler import KernelCompiler
from vllm.distributed.parallel_state import get_tp_group
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm_ascend.worker.worker import NPUWorker

from examples.models.qwen3_tp.runner import Runner


def _prebuilt_only(*args, **kwargs):
    raise RuntimeError("The device stage requires CPU-prepared KERNEL binaries")


class PyPTOProbeWorker(NPUWorker):
    """Reuse vLLM's device, executor and process groups for a PyPTO request."""

    def init_device(self) -> None:
        # The fixed RPC workload uses PyPTO's own model runner. Keep
        # NPUWorker's device and group setup without allocating the native
        # model runner's independent staging buffers and sampler.
        self.device = self._init_device()

    def load_model(self) -> None:
        """Load the prepared PyPTO weights when the validation RPC starts."""

    @torch.inference_mode()
    def validate_pypto_qwen(self, artifacts: str, weights: str, reference: str) -> dict:
        group = get_tp_group().cpu_group
        rank = dist.get_rank(group)
        result = torch.load(reference, map_location="cpu", weights_only=True)
        if dist.get_backend() != "hccl" or dist.get_backend(group) != "gloo":
            raise RuntimeError("Expected vLLM Ascend HCCL WORLD and the Gloo TP CPU group")
        if result["world_size"] != dist.get_world_size(group):
            raise ValueError("The logits reference must match the executor's TP size")

        # Keep CPU preparation outside the device job and guard against any
        # accidental fallback to the PROGRAM worker.
        ChipWorker.init = _prebuilt_only
        JITFunction._compile = _prebuilt_only
        KernelCompiler.compile_incore = _prebuilt_only
        KernelCompiler.compile_orchestration = _prebuilt_only
        device_runner._compile_single_kernel = _prebuilt_only
        device_runner._compile_single_orchestration = _prebuilt_only
        runner = Runner(
            Path(artifacts), Path(weights), rank, self.local_rank, capture=False, gloo_group=group
        )
        try:
            runner.load()
            generated = []
            for step, expected in enumerate(result["logits"]):
                logits = runner.forward(result["prompt_ids"] if step == 0 else generated[-1:])
                torch.testing.assert_close(logits, expected, atol=0, rtol=0)
                generated.append(int(logits.argmax()))
            if generated != result["generated_ids"]:
                raise AssertionError(f"Rank {rank} produced unexpected tokens {generated}")
            dist.barrier(group=group)
            return {
                "rank": rank,
                "gloo_group": group.group_name,
                "peers": len(runner.window.peer_bases),
                "layers": len(runner.layers),
                "exact_logits": len(generated),
                "generated_ids": generated,
            }
        finally:
            runner.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    config = EngineArgs(
        model=str(args.model),
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        worker_cls=f"{Path(__file__).stem}.PyPTOProbeWorker",
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=1,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        async_scheduling=False,
    ).create_engine_config()
    executor = MultiprocExecutor(config)
    try:
        result = executor.collective_rpc(
            "validate_pypto_qwen",
            timeout=180,
            args=(str(args.artifacts), str(args.weights), str(args.reference)),
        )
        if sorted(rank["rank"] for rank in result) != [0, 1]:
            raise AssertionError(f"Expected both worker results, got {result}")
        print(json.dumps(result))
    finally:
        executor.shutdown()


if __name__ == "__main__":
    main()
