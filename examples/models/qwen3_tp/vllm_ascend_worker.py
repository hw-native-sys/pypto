# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fixed-shape Qwen3-14B model execution for the vLLM Ascend V1 executor."""

from pathlib import Path

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_tp_group
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes
from vllm_ascend.worker.worker import NPUWorker

from . import kernels as k
from .runner import Runner


def _validate_sampling(params: SamplingParams | None) -> None:
    if params is None:
        raise ValueError("PyPTO worker requires generation sampling parameters")
    allowed = {
        "temperature": 0.0,
        "n": 1,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "min_tokens": 0,
        "logprobs": None,
        "prompt_logprobs": None,
        "logprob_token_ids": None,
        "structured_outputs": None,
        "logit_bias": None,
        "allowed_token_ids": None,
        "thinking_token_budget": None,
    }
    unsupported = [name for name, value in allowed.items() if getattr(params, name) != value]
    if params.bad_words:
        unsupported.append("bad_words")
    if unsupported:
        raise ValueError(f"PyPTO worker supports greedy generation with default modifiers: {unsupported}")


class Qwen3PyPTOAscendWorker(NPUWorker):
    """Run vLLM's scheduled text requests with one private PyPTO KV cache per rank."""

    def init_device(self) -> None:
        config = self.vllm_config
        model = config.model_config
        scheduler = config.scheduler_config
        parallel = config.parallel_config
        if (
            parallel.tensor_parallel_size != 2
            or parallel.world_size != 2
            or k.TP_WORLD != 2
            or scheduler.max_num_seqs != 1
            or scheduler.async_scheduling
            or scheduler.enable_chunked_prefill
            or config.cache_config.enable_prefix_caching
            or config.speculative_config is not None
            or config.kv_transfer_config is not None
            or config.lora_config is not None
            or not model.enforce_eager
            or not 1 <= model.max_model_len <= k.MAX_CACHE
        ):
            raise ValueError(
                "PyPTO worker requires eager TP2 with one sequence, no speculative decoding, "
                "prefix caching, chunked prefill, KV transfer, LoRA or async scheduling"
            )
        hf = model.hf_config
        if (
            hf.model_type != "qwen3"
            or hf.num_hidden_layers != k.NUM_LAYERS
            or hf.hidden_size != k.HIDDEN
            or model.dtype != torch.bfloat16
            or model.logits_processors
        ):
            raise ValueError("Expected Qwen3-14B BF16 geometry and default logits processing")
        self._runner: Runner | None = None
        self._request_id: str | None = None
        self._previous_token: int | None = None
        # Reuse Ascend's device setup and WORLD/TP construction. The PyPTO
        # runner supplies the model state and all KV allocations for this worker.
        self.device = self._init_device()

    def load_model(self) -> None:
        additional = self.vllm_config.additional_config
        paths = additional.get("pypto_qwen3_tp") if isinstance(additional, dict) else None
        if not isinstance(paths, dict) or not all(
            isinstance(paths.get(name), str) and paths[name] for name in ("artifacts", "weights")
        ):
            raise ValueError("Set additional_config.pypto_qwen3_tp artifacts and weights directories")
        group = get_tp_group().cpu_group
        if dist.get_backend() != "hccl" or dist.get_backend(group) != "gloo":
            raise RuntimeError("PyPTO worker requires HCCL WORLD and vLLM's Gloo TP CPU group")
        runner = Runner(
            Path(paths["artifacts"]),
            Path(paths["weights"]),
            dist.get_rank(group),
            self.local_rank,
            capture=False,
            gloo_group=group,
        )
        try:
            runner.load()
        except BaseException:
            runner.close()
            raise
        self._runner = runner

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Keep vLLM's scheduler bookkeeping, with KV storage owned by PyPTO."""
        return {}

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        if kv_cache_config.kv_cache_groups:
            raise ValueError("PyPTO worker manages its own KV cache")

    def compile_or_warm_up_model(self) -> CompilationTimes:
        return CompilationTimes(0.0, 0.0)

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return ("generate",)

    def reset_mm_cache(self) -> None:
        """The worker accepts text tokens only."""

    def _forward(self, runner: Runner, tokens: list[int]) -> torch.Tensor:
        return runner.forward(tokens)

    @torch.inference_mode()
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        runner = self._runner
        if runner is None:
            raise RuntimeError("PyPTO model is not loaded")
        if self._request_id in scheduler_output.finished_req_ids:
            self._request_id = None
            self._previous_token = None

        scheduled = scheduler_output.num_scheduled_tokens
        if not scheduled:
            return EMPTY_MODEL_RUNNER_OUTPUT
        if len(scheduled) != 1 or scheduler_output.scheduled_spec_decode_tokens:
            raise ValueError("Expected one scheduled request without speculative tokens")
        request_id, count = next(iter(scheduled.items()))
        new = scheduler_output.scheduled_new_reqs
        cached = scheduler_output.scheduled_cached_reqs
        if new:
            request = new[0]
            prompt = request.prompt_token_ids
            if (
                len(new) != 1
                or cached.req_ids
                or request.req_id != request_id
                or self._request_id is not None
                or request.num_computed_tokens != 0
                or request.mm_features
                or request.prompt_embeds is not None
                or prompt is None
                or not 1 <= len(prompt) == count <= k.TOK
            ):
                raise ValueError(f"PyPTO worker requires one complete text prompt of at most {k.TOK} tokens")
            _validate_sampling(request.sampling_params)
            runner.reset()
            self._request_id = request_id
            tokens = prompt
        else:
            if (
                cached.req_ids != [request_id]
                or cached.resumed_req_ids
                or cached.num_computed_tokens != [runner.position]
                or self._request_id != request_id
                or self._previous_token is None
                or count != 1
            ):
                raise ValueError("PyPTO worker requires one-token decode for the active request")
            tokens = [self._previous_token]
        # The final head is replicated and each rank computes the same logits.
        # All ranks keep the deterministic greedy token for the next decode.
        token = int(self._forward(runner, tokens).argmax())
        self._previous_token = token
        return ModelRunnerOutput(
            req_ids=[request_id], req_id_to_index={request_id: 0}, sampled_token_ids=[[token]]
        )

    def shutdown(self) -> None:
        runner = getattr(self, "_runner", None)
        if runner is not None:
            runner.close()
            self._runner = None
