# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""Qwen3-14B tensor-parallel inference on the existing KERNEL worker.

Artifacts and rank-local weights must be prepared on CPU. Both eager and graph
replay use device barriers. A graph is cached per (position, token count), so
scalar bindings and tensor addresses remain fixed across repeated requests.
"""

import hashlib
import json
from pathlib import Path

import pypto.torch
import torch
import torch.distributed as dist
from pypto.runtime.kernel.context import get_process_kernel_state
from pypto.runtime.shmem_gloo import acquire_gloo_shmem_window

from examples.distributed.gloo_shmem_utils import load_eager_kernel, run_eager_kernel
from examples.models.qwen3_tp import kernels as k


class Runner:
    """Own model storage and dispatch only PyPTO kernels for model arithmetic."""

    def __init__(self, artifacts: Path, weights: Path, rank: int, local_rank: int, *, capture: bool) -> None:
        manifest = json.loads((artifacts / "manifest.json").read_text())
        digest = hashlib.sha256(Path(k.__file__).read_bytes()).hexdigest()
        if manifest["world_size"] != k.TP_WORLD or manifest["source_sha256"] != digest:
            raise ValueError("Prepared artifacts do not match TP size or kernel source")
        self.programs = {
            name: load_eager_kernel(artifacts / name)
            for name in manifest["entries"]
            if not capture or name not in k.EAGER_LAYER_ENTRIES
        }
        self.npu = getattr(torch, "npu")
        self.device = f"npu:{local_rank}"
        self.weights_path = weights / f"rank{rank}.pt"
        self.capture = capture
        self.fuse_attention = True
        runtime = next(iter(self.programs.values())).runtime_name
        self.group_eager_layers = (
            not capture
            and runtime == "host_build_graph"
            and k.TP_WORLD == 2
            and k.EAGER_LAYER_ENTRIES.issubset(self.programs)
        )
        # Each captured invocation retains its runtime arena until graph destruction.
        # These entries submit at most five tasks and use caller-owned tensors only;
        # the default per-ring heap would waste a GiB for every recorded invocation.
        pypto.torch.init(
            device=local_rank,
            runtime=runtime,
            ring_task_window=16,
            ring_heap=1 << 20,
            ring_dep_pool=64,
        )
        self.state = get_process_kernel_state()
        for artifact in self.programs.values():
            self.state.ensure_callable(artifact, self.state.require_config())
        self.window = acquire_gloo_shmem_window(
            rank=rank,
            world_size=k.TP_WORLD,
            device=self.device,
            slot_names=("partial", "signal"),
            slot_nbytes=(k.TOK * k.HIDDEN * 4, k.TP_WORLD * 4),
        )
        self.shared = self.window.as_torch_tensor("partial", (k.TOK, k.HIDDEN), torch.float32)
        self.signal = self.window.as_torch_tensor("signal", (k.TP_WORLD, 1), torch.int32)
        self.signal.zero_()
        # A peer may notify this slot immediately after initialization.
        self.npu.synchronize(self.signal.device)
        dist.barrier()
        self.layers: list[dict[str, torch.Tensor]] = []
        self.position = 0
        self.graphs = {}
        self.stream = self.npu.Stream(device=local_rank)

    def allocate(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=self.device)

    def load(self) -> None:
        weights = torch.load(self.weights_path, map_location="cpu", weights_only=True, mmap=True)
        if weights["world_size"] != k.TP_WORLD:
            raise ValueError("Prepared weights do not match TP size")
        self.embed = weights["embed"].to(self.device)
        self.head = weights["head"].to(self.device)
        self.final_norm = weights["final_norm"].to(self.device)
        self.layers = [
            {name: value.to(self.device) for name, value in layer.items()} for layer in weights["layers"]
        ]
        self.hidden = self.allocate((k.TOK, k.HIDDEN), torch.bfloat16)
        self.xn = torch.zeros_like(self.hidden)
        self.residual = torch.zeros_like(self.hidden)
        self.partial = self.allocate((k.TOK, k.HIDDEN))
        self.reduced = torch.zeros_like(self.partial)
        self.q = self.allocate((k.TOK, k.HIDDEN_TP))
        self.key = self.allocate((k.TOK, k.KV_HIDDEN_TP))
        self.value = torch.zeros_like(self.key)
        self.q_bf = self.allocate(tuple(self.q.shape), torch.bfloat16)
        self.k_bf = self.allocate(tuple(self.key.shape), torch.bfloat16)
        self.v_bf = torch.zeros_like(self.k_bf)
        self.context = torch.zeros_like(self.q)
        self.k_cache = self.allocate((k.CACHE_ROWS, k.KV_HIDDEN_TP), torch.bfloat16)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.k_ctx = self.allocate((k.MAX_CACHE, k.KV_HIDDEN_TP))
        self.v_ctx = torch.zeros_like(self.k_ctx)
        self.gate = self.allocate((k.TOK, k.INTER_TP))
        self.up = torch.zeros_like(self.gate)
        self.act = self.allocate(tuple(self.gate.shape), torch.bfloat16)
        self.logits = self.allocate((k.TOK, k.VOCAB))
        self.cos = weights["cos"].to(self.device)
        self.sin = weights["sin"].to(self.device)
        self.ids = self.allocate((k.TOK,), torch.int32)
        self.attention = {
            prefill: (
                self.allocate((k.ATTN_ROWS, width)),
                self.allocate((k.ATTN_ROWS, width)),
                self.allocate((k.TOK, width)),
            )
            for prefill, width in ((True, k.TOK), (False, k.MAX_CACHE))
        }

    def run(self, name: str, *values: torch.Tensor | int) -> None:
        run_eager_kernel(self.programs[name], *values)

    def allreduce(self) -> None:
        run_eager_kernel(self.programs["publish_chip"], self.partial, self.shared)
        run_eager_kernel(self.programs["barrier_chip"], self.signal)
        run_eager_kernel(self.programs["allreduce_chip"], self.shared, self.reduced)
        run_eager_kernel(self.programs["barrier_chip"], self.signal)

    def close(self) -> None:
        self.npu.synchronize()
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.state.drain()
        self.state.close()

    def reset(self) -> None:
        """Begin an independent sequence while retaining weights and compiled programs."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.hidden.zero_()
        self.position = 0

    def forward(self, token_ids: list[int]) -> torch.Tensor:
        n_tok = len(token_ids)
        prefill = self.position == 0
        if not 1 <= n_tok <= k.TOK or (not prefill and n_tok != 1):
            raise ValueError(f"Expected prefill 1..32 tokens or decode 1 token, got {n_tok}")
        if self.position + n_tok > k.MAX_CACHE or any(t < 0 or t >= 151936 for t in token_ids):
            raise ValueError("Token ID or cache capacity exceeded")
        ids = torch.zeros(k.TOK, dtype=torch.int32)
        ids[:n_tok] = torch.tensor(token_ids, dtype=torch.int32)
        self.ids.copy_(ids)
        width = k.TOK if prefill else k.MAX_CACHE
        columns = torch.arange(width)[None, :]
        allowed = (
            (columns <= torch.arange(k.TOK)[:, None]) & (columns < n_tok)
            if prefill
            else (columns < self.position + 1).expand(k.TOK, -1)
        )
        self.attention[prefill][2].copy_(torch.where(allowed, 0.0, -1e9))
        self.npu.synchronize()
        if self.capture:
            key = (self.position, n_tok)
            if key not in self.graphs:
                graph = self.npu.NPUGraph()
                self.graphs[key] = graph
                with self.npu.graph(graph, stream=self.stream):
                    self._compute(n_tok, prefill)
                # Recording can finish at different times on different ranks.
                # Do not let an early replay wait in a device collective while
                # its peer is still recording hundreds of model nodes.
                dist.barrier()
            with self.npu.stream(self.stream):
                self.graphs[key].replay()
        else:
            self._compute(n_tok, prefill)
        self.npu.synchronize()
        # Live graphs retain all tickets; replay creates no new submissions.
        # Reclaim them once in close(), after resetting every graph.
        if not self.capture:
            self.state.drain()
        self.position += n_tok
        return self.logits[n_tok - 1, :151936].cpu()

    def _compute(self, n_tok: int, prefill: bool) -> None:
        self.run("tp_embed_step_chip", self.ids, self.embed, self.hidden, n_tok)
        if self.group_eager_layers and self.fuse_attention:
            self._compute_grouped_layers(n_tok, prefill)
        else:
            self._compute_layers(n_tok, prefill)
        self.run("rms_hidden_step_chip", self.hidden, self.final_norm, self.xn)
        self.run("tp_lm_head_step_chip", self.xn, self.head, self.logits)

    def _compute_layers(self, n_tok: int, prefill: bool) -> None:
        scores, probs, mask = self.attention[prefill]
        mode = "prefill" if prefill else "decode"
        for i, w in enumerate(self.layers):
            self.run("tp_hidden_rms_step_chip", self.hidden, w["input_norm"], self.xn)
            self.run(
                "tp_qkv_gemm_step_chip", self.xn, w["wq"], w["wk"], w["wv"], self.q, self.key, self.value
            )
            self.run(
                "tp_qkv_post_step_chip",
                self.q,
                self.key,
                self.value,
                w["q_norm"],
                w["k_norm"],
                self.cos,
                self.sin,
                self.k_cache,
                self.v_cache,
                self.q_bf,
                self.k_bf,
                self.v_bf,
                i,
                self.position,
                n_tok,
            )
            if self.fuse_attention:
                rows = slice(i * k.MAX_CACHE, (i + 1) * k.MAX_CACHE)
                self.run(
                    f"tp_{mode}_attention_batch_chip",
                    self.q_bf,
                    self.k_bf if prefill else self.k_cache[rows],
                    self.v_bf if prefill else self.v_cache[rows],
                    self.q,
                    self.key if prefill else self.k_ctx,
                    self.value if prefill else self.v_ctx,
                    scores,
                    mask,
                    probs,
                    self.context,
                )
            else:
                if prefill:
                    self.run(
                        "tp_prefill_attention_prepare_step_chip",
                        self.q_bf,
                        self.k_bf,
                        self.v_bf,
                        self.q,
                        self.key,
                        self.value,
                    )
                    keys, values = self.key, self.value
                else:
                    rows = slice(i * k.MAX_CACHE, (i + 1) * k.MAX_CACHE)
                    self.run(
                        "tp_decode_attention_prepare_step_chip",
                        self.q_bf,
                        self.k_cache[rows],
                        self.v_cache[rows],
                        self.q,
                        self.k_ctx,
                        self.v_ctx,
                    )
                    keys, values = self.k_ctx, self.v_ctx
                self.run(f"tp_{mode}_attention_scores_step_chip", self.q, keys, scores)
                self.run(f"tp_{mode}_attention_softmax_step_chip", scores, mask, probs)
                self.run(f"tp_{mode}_attention_context_step_chip", probs, values, self.context)
                self.run("tp_attention_cast_step_chip", self.context, self.q_bf)
            self.run("tp_o_gemm_step_chip", self.q_bf, w["wo"], self.partial)
            self.allreduce()
            self.run(
                "tp_ffn_step_chip",
                self.hidden,
                self.reduced,
                w["post_norm"],
                self.residual,
                self.xn,
                w["gate"],
                w["up"],
                self.gate,
                self.up,
                self.act,
                w["down"],
                self.partial,
            )
            self.allreduce()
            self.run("tp_residual_tail_chip", self.residual, self.reduced, self.hidden)

    def _compute_grouped_layers(self, n_tok: int, prefill: bool) -> None:
        scores, probs, mask = self.attention[prefill]
        for i, w in enumerate(self.layers):
            qkv = (
                self.hidden,
                w["input_norm"],
                self.xn,
                w["wq"],
                w["wk"],
                w["wv"],
                self.q,
                self.key,
                self.value,
                w["q_norm"],
                w["k_norm"],
                self.cos,
                self.sin,
                self.k_cache,
                self.v_cache,
                self.q_bf,
                self.k_bf,
                self.v_bf,
            )
            if prefill:
                run_eager_kernel(
                    self.programs["tp_prefill_attention_reduce_chip"],
                    *qkv,
                    scores,
                    mask,
                    probs,
                    self.context,
                    w["wo"],
                    self.partial,
                    self.shared,
                    self.signal,
                    self.reduced,
                    i,
                    self.position,
                    n_tok,
                )
            else:
                # Submit the full KV write before passing a read-only layer view.
                self.run("tp_qkv_batch_chip", *qkv, i, self.position, n_tok)
                rows = slice(i * k.MAX_CACHE, (i + 1) * k.MAX_CACHE)
                run_eager_kernel(
                    self.programs["tp_decode_attention_reduce_chip"],
                    self.q_bf,
                    self.k_cache[rows],
                    self.v_cache[rows],
                    self.q,
                    self.k_ctx,
                    self.v_ctx,
                    scores,
                    mask,
                    probs,
                    self.context,
                    w["wo"],
                    self.partial,
                    self.shared,
                    self.signal,
                    self.reduced,
                )
            run_eager_kernel(
                self.programs["tp_ffn_reduce_chip"],
                self.hidden,
                self.reduced,
                w["post_norm"],
                self.residual,
                self.xn,
                w["gate"],
                w["up"],
                self.gate,
                self.up,
                self.act,
                w["down"],
                self.partial,
                self.shared,
                self.signal,
            )
