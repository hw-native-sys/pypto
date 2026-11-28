# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise vLLM Ascend LLM.generate on two NPUs with a prepared PyPTO Qwen3."""

import argparse
import json
from pathlib import Path

import torch
from pypto.jit.decorator import JITFunction
from pypto.runtime import ChipWorker, device_runner
from pypto.runtime.kernel_compiler import KernelCompiler
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from examples.models.qwen3_tp.runner import Runner
from examples.models.qwen3_tp.vllm_ascend_worker import Qwen3PyPTOAscendWorker


def _prebuilt_only(*args, **kwargs):
    raise RuntimeError("The device stage requires CPU-prepared KERNEL binaries")


class ValidationWorker(Qwen3PyPTOAscendWorker):
    """Check the logits from every scheduler step in both TP worker processes."""

    def load_model(self) -> None:
        # A fallback that builds binaries or claims PROGRAM mode invalidates
        # this device-only test, which consumes an already-prepared bundle.
        ChipWorker.init = _prebuilt_only
        JITFunction._compile = _prebuilt_only
        KernelCompiler.compile_incore = _prebuilt_only
        KernelCompiler.compile_orchestration = _prebuilt_only
        device_runner._compile_single_kernel = _prebuilt_only
        device_runner._compile_single_orchestration = _prebuilt_only
        super().load_model()

    def set_reference(self, path: str) -> None:
        reference = torch.load(path, map_location="cpu", weights_only=True)
        if reference["world_size"] != 2 or not reference["logits"]:
            raise ValueError("The reference must contain TP2 logits")
        self._expected_logits = reference["logits"]
        self._exact_steps = 0

    def _forward(self, runner: Runner, tokens: list[int]) -> torch.Tensor:
        logits = super()._forward(runner, tokens)
        expected = self._expected_logits[self._exact_steps % len(self._expected_logits)]
        torch.testing.assert_close(logits, expected, atol=0, rtol=0)
        self._exact_steps += 1
        return logits

    def validation_result(self) -> dict[str, int]:
        runner = self._runner
        assert runner is not None
        return {
            "rank": self.rank,
            "peers": len(runner.window.peer_bases),
            "layers": len(runner.layers),
            "exact_logits": self._exact_steps,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    expected = reference["generated_ids"]
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        worker_cls=f"{Path(__file__).stem}.ValidationWorker",
        additional_config={
            "pypto_qwen3_tp": {
                "artifacts": str(args.artifacts.resolve()),
                "weights": str(args.weights.resolve()),
            }
        },
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=1,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        async_scheduling=False,
        block_size=128,
    )
    try:
        llm.collective_rpc("set_reference", args=(str(args.reference.resolve()),))
        prompt = TokensPrompt(prompt_token_ids=reference["prompt_ids"])
        params = SamplingParams(temperature=0, max_tokens=len(expected), ignore_eos=True)
        requests = llm.generate([prompt, prompt], params, use_tqdm=False)
        actual = [list(request.outputs[0].token_ids) for request in requests]
        assert actual == [expected, expected], f"Unexpected output tokens {actual}"
        result = llm.collective_rpc("validation_result")
        assert sorted(row["rank"] for row in result) == [0, 1]
        assert all(
            row["peers"] == 2 and row["layers"] == 40 and row["exact_logits"] == 2 * len(expected)
            for row in result
        )
        print(json.dumps({"requests": actual, "workers": result}), flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
