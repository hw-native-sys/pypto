# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""Standalone, serial Qwen3-14B TP2/TP4/TP8 prefill/decode using direct L2 dispatch.

Launch from the repository root with its root and python directories on
PYTHONPATH: torchrun --standalone --nproc_per_node=2 -m
examples.models.qwen3_tp.runner --model-dir /path/to/Qwen3-14B
"""

import argparse
import hashlib
import importlib
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import cast

import torch
from pypto.ir import CompiledProgram
from pypto.runtime import ChipWorker, DeviceTensor, RunConfig
from pypto.runtime.shmem_gloo import acquire_gloo_shmem_window

from examples.distributed.gloo_shmem_utils import DispatchPlan, run_kernel
from examples.models.qwen3_tp import kernels as k
from examples.models.qwen3_tp.weights import WeightLoader


def log(message: str) -> None:
    print(f"[rank {os.environ.get('RANK', '?')}] {message}", flush=True)


class Runner:
    """Own model storage and dispatch only PyPTO kernels for model arithmetic."""

    def __init__(
        self,
        model_dir: str,
        rank: int,
        local_rank: int,
        *,
        trace_layers: bool = False,
        cache_dispatch_plan: bool = False,
        fuse_swiglu_down: bool = False,
        fuse_ffn: bool = False,
        artifacts: Path | None = None,
    ):
        self.fuse_ffn = fuse_ffn
        self.fuse_swiglu_down = fuse_swiglu_down
        self.cache_dispatch_plan = cache_dispatch_plan
        self.dispatch_plans: dict[str, DispatchPlan] = {}
        self.prepared: dict[str, str] | None = None
        if artifacts is not None:
            manifest = json.loads((artifacts / "manifest.json").read_text())
            digest = hashlib.sha256(Path(k.__file__).read_bytes()).hexdigest()
            if manifest["world_size"] != k.TP_WORLD or manifest["source_sha256"] != digest:
                raise ValueError("Prepared artifacts do not match TP size or kernel source")
            self.prepared = manifest["entries"]
        self.device = f"npu:{local_rank}"
        self.dist = importlib.import_module("torch.distributed")
        self.loader = WeightLoader(model_dir, rank, self.device, k.TP_WORLD)
        self.programs: dict[str, CompiledProgram] = {}
        self.window = acquire_gloo_shmem_window(
            rank=rank,
            world_size=k.TP_WORLD,
            device=self.device,
            slot_names=("partial",),
            slot_nbytes=(k.TOK * k.HIDDEN * 4,),
        )
        self.shared = self.window.as_device_tensor("partial", (k.TOK, k.HIDDEN), torch.float32)
        self.chip = ChipWorker(config=RunConfig(platform="a2a3", device_id=local_rank), auto_init=False)
        self.chip.init(borrowed_context_generation=1)
        self.layers: list[dict[str, torch.Tensor]] = []
        self.position = 0
        self.trace_layers = trace_layers and rank == 0
        self.trace: dict[str, torch.Tensor] = {}

    def allocate(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=self.device)

    def load(self) -> None:
        self.embed = self.loader.vocabulary("model.embed_tokens.weight")
        self.head = self.loader.vocabulary("lm_head.weight")
        self.final_norm = self.loader.norm("model.norm.weight")
        for i in range(k.NUM_LAYERS):
            self.layers.append(self.loader.layer(i))
            if i % 10 == 9:
                log(f"loaded {i + 1}/40 layers")
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
        # Position tables and masks are host-created metadata, not model operators.
        inv = 1.0 / (1e6 ** (torch.arange(0, k.HEAD_DIM, 2).float() / k.HEAD_DIM))
        angles = torch.arange(k.MAX_SEQ).float()[:, None] * inv[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        self.cos = angles.cos().to(self.device)
        self.sin = angles.sin().to(self.device)

    def run(self, name: str, *values: torch.Tensor | DeviceTensor | int) -> None:
        if name not in self.programs:
            log(f"compile {name}")
            meta = [
                torch.empty(v.shape, dtype=v.dtype, device="meta")
                if isinstance(v, (torch.Tensor, DeviceTensor))
                else v
                for v in values
            ]
            self.compile_serial(name, *meta)
        self.dispatch(name, *values)

    def dispatch(self, name: str, *values: torch.Tensor | DeviceTensor | int) -> None:
        compiled = self.programs[name]
        if self.cache_dispatch_plan:
            plan = self.dispatch_plans.get(name)
            if plan is None or plan.compiled is not compiled:
                plan = DispatchPlan(compiled)
                self.dispatch_plans[name] = plan
            run_kernel(self.chip, compiled, *values, plan=plan)
        else:
            run_kernel(self.chip, compiled, *values)

    def compile_serial(self, name: str, *meta: torch.Tensor | int) -> None:
        """Compile one rank at a time within the shared host worker budget."""
        for rank in range(k.TP_WORLD):
            if self.dist.get_rank() == rank:
                if self.prepared is None:
                    compiled = getattr(k, name).compile(
                        *meta, config=RunConfig(platform="a2a3", save_kernels=True)
                    )
                else:
                    compiled = CompiledProgram.from_dir(self.prepared[name])
                    _ = compiled.chip_callable
                self.programs[name] = compiled
            self.dist.barrier()

    def allreduce(self) -> None:
        # Both barriers are required: publish visibility, then protection from reuse.
        ctx = self.window.device_ctx_ptr
        if "publish_chip" not in self.programs:
            meta = torch.empty((k.TOK, k.HIDDEN), dtype=torch.float32, device="meta")
            for name in ("publish_chip", "allreduce_chip"):
                self.compile_serial(name, meta, meta)
        self.dispatch("publish_chip", self.partial, self.shared, ctx)
        self.dist.barrier()
        self.dispatch("allreduce_chip", self.shared, self.reduced, ctx)
        self.dist.barrier()

    def reset(self) -> None:
        """Begin an independent sequence while retaining weights and compiled programs."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.hidden.zero_()
        self.position = 0
        self.trace = {}

    def snapshot(self, layer: int, stage: str, value: torch.Tensor, row: int) -> None:
        """Optionally capture the last valid token for CPU numerical diagnosis."""
        if self.trace_layers:
            self.trace[f"{layer}.{stage}"] = value[row].detach().cpu().clone()

    def forward(self, token_ids: list[int]) -> torch.Tensor:
        n_tok = len(token_ids)
        self.trace = {}
        prefill = self.position == 0
        if not 1 <= n_tok <= k.TOK or (not prefill and n_tok != 1):
            raise ValueError(f"Expected prefill 1..32 tokens or decode 1 token, got {n_tok}")
        if self.position + n_tok > k.MAX_CACHE or any(t < 0 or t >= 151936 for t in token_ids):
            raise ValueError("Token ID or cache capacity exceeded")
        ids = torch.zeros(k.TOK, dtype=torch.int32)
        ids[:n_tok] = torch.tensor(token_ids, dtype=torch.int32)
        self.run("tp_embed_step_chip", ids.to(self.device), self.embed, self.hidden, n_tok)
        width = k.TOK if prefill else k.MAX_CACHE
        scores = self.allocate((k.ATTN_ROWS, width))
        probs = torch.zeros_like(scores)
        columns = torch.arange(width)[None, :]
        if prefill:
            allowed = (columns <= torch.arange(k.TOK)[:, None]) & (columns < n_tok)
        else:
            allowed = (columns < self.position + 1).expand(k.TOK, -1)
        mask = torch.where(allowed, 0.0, -1e9).to(self.device)
        mode = "prefill" if prefill else "decode"
        for i, w in enumerate(self.layers):
            self.snapshot(i, "input", self.hidden, n_tok - 1)
            self.run("tp_hidden_rms_step_chip", self.hidden, w["input_norm"], self.xn)
            self.snapshot(i, "input_norm", self.xn, n_tok - 1)
            self.run(
                "tp_qkv_gemm_step_chip", self.xn, w["wq"], w["wk"], w["wv"], self.q, self.key, self.value
            )
            self.snapshot(i, "q_proj", self.q, n_tok - 1)
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
            self.snapshot(i, "q_rope", self.q_bf, n_tok - 1)
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
            self.snapshot(i, "o_proj", self.reduced, n_tok - 1)
            if self.fuse_ffn:
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
            else:
                self.run(
                    "tp_ffn_residual_rms_step_chip",
                    self.hidden,
                    self.reduced,
                    w["post_norm"],
                    self.residual,
                    self.xn,
                )
                self.run("tp_ffn_gate_up_step_chip", self.xn, w["gate"], w["up"], self.gate, self.up)
                if self.fuse_swiglu_down:
                    self.run(
                        "tp_ffn_swiglu_down_step_chip",
                        self.gate,
                        self.up,
                        self.act,
                        w["down"],
                        self.partial,
                    )
                else:
                    self.run("tp_ffn_swiglu_step_chip", self.gate, self.up, self.act)
                    self.run("tp_ffn_down_step_chip", self.act, w["down"], self.partial)
            self.snapshot(i, "post_norm", self.xn, n_tok - 1)
            self.snapshot(i, "gate_proj", self.gate, n_tok - 1)
            self.snapshot(i, "up_proj", self.up, n_tok - 1)
            self.snapshot(i, "swiglu", self.act, n_tok - 1)
            self.allreduce()
            self.snapshot(i, "down_proj", self.reduced, n_tok - 1)
            self.run("tp_residual_tail_chip", self.residual, self.reduced, self.hidden)
            self.snapshot(i, "output", self.hidden, n_tok - 1)
            if i % 10 == 9:
                log(f"{mode}: {i + 1}/40 layers")
        self.run("rms_hidden_step_chip", self.hidden, self.final_norm, self.xn)
        self.run("tp_lm_head_step_chip", self.xn, self.head, self.logits)
        self.position += n_tok
        return self.logits[n_tok - 1, :151936].cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, help="Save CPU logits for independent reference checks")
    parser.add_argument("--prompts-file", type=Path, help="JSON array of independent prompt strings")
    parser.add_argument("--trace-layers", action="store_true", help="Save last-token stage snapshots")
    fusion = parser.add_mutually_exclusive_group()
    fusion.add_argument("--fuse-swiglu-down", action="store_true")
    fusion.add_argument("--fuse-ffn", action="store_true")
    parser.add_argument("--cache-dispatch-plan", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--artifacts", type=Path, help="Reuse CPU-prepared benchmark artifacts")
    args = parser.parse_args()
    if not 2 <= args.steps <= 96:
        parser.error("--steps must be in [2, 96] to cover prefill and decode")
    AutoTokenizer = importlib.import_module("transformers").AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    raw_prompts: object = json.loads(args.prompts_file.read_text()) if args.prompts_file else [args.prompt]
    if not isinstance(raw_prompts, list) or not raw_prompts:
        parser.error("--prompts-file must contain a nonempty JSON array of strings")
    prompts = cast(list[object], raw_prompts)
    if not all(isinstance(prompt, str) for prompt in prompts):
        parser.error("--prompts-file must contain a nonempty JSON array of strings")
    token_sets = [tokenizer.encode(prompt) for prompt in prompts]
    if any(not 1 <= len(tokens) <= k.TOK for tokens in token_sets):
        parser.error("Every prompt must tokenize to 1..32 tokens")
    if args.trace_layers and args.output_dir is None:
        parser.error("--trace-layers requires --output-dir")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) not in (2, 4, 8):
        parser.error("This runner requires 2, 4 or 8 ranks")
    torch.set_num_threads(1)
    importlib.import_module("torch_npu")
    getattr(torch, "npu").set_device(local_rank)
    dist = importlib.import_module("torch.distributed")
    dist.init_process_group("gloo", timeout=timedelta(minutes=30))
    runner = Runner(
        args.model_dir,
        rank,
        local_rank,
        trace_layers=args.trace_layers,
        cache_dispatch_plan=args.cache_dispatch_plan,
        fuse_swiglu_down=args.fuse_swiglu_down,
        fuse_ffn=args.fuse_ffn,
        artifacts=args.artifacts,
    )
    try:
        runner.load()
        for case, tokens in enumerate(token_sets):
            runner.reset()
            log(f"case={case} prompt_tokens={len(tokens)} steps={args.steps}")
            generated: list[int] = []
            logits_history: list[torch.Tensor] = []
            traces: list[dict[str, torch.Tensor]] = []
            for step in range(args.steps):
                logits = runner.forward(tokens if step == 0 else [generated[-1]])
                if not torch.isfinite(logits).all():
                    raise AssertionError(f"Non-finite logits at generation step {step}")
                peers = [torch.empty_like(logits) for _ in range(k.TP_WORLD)]
                dist.all_gather(peers, logits)
                for peer_logits in peers[1:]:
                    torch.testing.assert_close(peers[0], peer_logits, rtol=1e-4, atol=1e-4)
                logits_history.append(logits)
                if rank == 0 and args.trace_layers:
                    traces.append(runner.trace)
                generated.append(int(logits.argmax()))
                log(f"step={step} token={generated[-1]} text={tokenizer.decode(generated)!r}")
            if rank == 0 and args.output_dir is not None:
                output_dir = args.output_dir / f"case_{case}" if args.prompts_file else args.output_dir
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "world_size": k.TP_WORLD,
                        "cache_dispatch_plan": args.cache_dispatch_plan,
                        "fuse_swiglu_down": args.fuse_swiglu_down,
                        "fuse_ffn": args.fuse_ffn,
                        "prompt_ids": tokens,
                        "generated_ids": generated,
                        "logits": logits_history,
                        "traces": traces,
                    },
                    output_dir / "outputs.pt",
                )
        log("QWEN3_TP_RANK_AGREEMENT_PASS (independent reference validation still required)")
    finally:
        runner.chip.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
