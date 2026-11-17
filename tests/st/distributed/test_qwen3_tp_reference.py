# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Independent HF logits and optional layer/stage diagnostics for saved TP runs.

PYPTO_QWEN3_OUTPUT accepts outputs.pt or a directory containing case_*/outputs.pt.
PYPTO_QWEN3_REFERENCE_DTYPE selects fp32 (default) or bf16; both use the same
acceptance limits. PYPTO_QWEN3_REPORT optionally saves all diagnostic rows as JSON,
including when numerical assertions fail. Run with --noconftest on CPU only.
"""

import importlib
import json
import os
from pathlib import Path
from types import MethodType

import pytest
import torch


def metrics(actual, expected):
    actual, expected = actual.float().flatten(), expected.float().flatten()
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    denominator = expected.square().mean().sqrt().clamp_min(1e-12)
    return {
        "relative_rmse": float((actual - expected).square().mean().sqrt() / denominator),
        "cosine": float(torch.nn.functional.cosine_similarity(actual, expected, dim=0)),
    }


def attach_stage_hooks(model, captured, world_size=2):
    """Capture independent HF intermediates, selecting rank zero's contiguous shards."""
    handles = []
    rope = {}
    apply_rotary_pos_emb = importlib.import_module(
        "transformers.models.qwen3.modeling_qwen3"
    ).apply_rotary_pos_emb

    def save_rope(module, args, output):
        rope["cos"], rope["sin"] = output

    handles.append(model.model.rotary_emb.register_forward_hook(save_rope))

    def hook(layer, stage, width=None, before=False):
        def save(module, args, output=None):
            value = args[0] if before else output
            if isinstance(value, tuple):
                value = value[0]
            value = value[0, -1].flatten()
            captured[f"{layer}.{stage}"] = value[:width].detach().clone()

        return save

    def q_rope_hook(layer):
        def save(module, args, output):
            q = output[:, -1:, : 40 // world_size, :].transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, q, rope["cos"][:, -1:], rope["sin"][:, -1:])
            captured[f"{layer}.q_rope"] = q.flatten().detach().clone()

        return save

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.input_layernorm.register_forward_pre_hook(hook(i, "input", before=True)))
        handles.append(layer.register_forward_hook(hook(i, "output")))
        handles.append(layer.self_attn.q_norm.register_forward_hook(q_rope_hook(i)))
        handles.append(
            layer.mlp.down_proj.register_forward_pre_hook(hook(i, "swiglu", 17408 // world_size, True))
        )
        for stage, module, width in (
            ("input_norm", layer.input_layernorm, None),
            ("post_norm", layer.post_attention_layernorm, None),
            ("q_proj", layer.self_attn.q_proj, 5120 // world_size),
            ("o_proj", layer.self_attn.o_proj, None),
            ("gate_proj", layer.mlp.gate_proj, 17408 // world_size),
            ("up_proj", layer.mlp.up_proj, 17408 // world_size),
            ("down_proj", layer.mlp.down_proj, None),
        ):
            handles.append(module.register_forward_hook(hook(i, stage, width)))
    return handles


def local_rounding_checks(trace, model, position):
    """Hold actual kernel inputs fixed to separate rounding from accumulated layer drift."""
    rows = []
    for i, layer in enumerate(model.model.layers):
        prefix = f"{i}."
        x = trace[prefix + "input"].float()
        normalized = x * torch.rsqrt(x.square().mean() + 1e-6)
        weight = layer.input_layernorm.weight.detach().float()
        norm_fused = (normalized * weight).bfloat16()
        norm_eager = normalized.bfloat16() * weight.bfloat16()
        gate, up = trace[prefix + "gate_proj"].float(), trace[prefix + "up_proj"].float()
        act_fused = (torch.nn.functional.silu(gate) * up).bfloat16()
        act_eager = torch.nn.functional.silu(gate.bfloat16()) * up.bfloat16()
        q = trace[prefix + "q_proj"].float().reshape(-1, 128)
        gamma = layer.self_attn.q_norm.weight.detach().float()
        inv_freq = 1.0 / (1e6 ** (torch.arange(0, 128, 2).float() / 128))
        angles = torch.cat((position * inv_freq, position * inv_freq))
        cos, sin = angles.cos(), angles.sin()
        norm = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6) * gamma
        rotated = torch.cat((-norm[:, 64:], norm[:, :64]), dim=-1)
        rope_fused = (norm * cos + rotated * sin).bfloat16()
        qb = q.bfloat16().float()
        norm_b = (qb * torch.rsqrt(qb.square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
        norm_b = norm_b * gamma.bfloat16()
        rotated_b = torch.cat((-norm_b[:, 64:], norm_b[:, :64]), dim=-1)
        rope_eager = norm_b * cos.bfloat16() + rotated_b * sin.bfloat16()
        for stage, fused, eager in (
            ("input_norm", norm_fused, norm_eager),
            ("q_rope", rope_fused, rope_eager),
            ("swiglu", act_fused, act_eager),
        ):
            actual = trace[prefix + stage]
            rows.append(
                {
                    "layer": i,
                    "stage": stage,
                    "fused": metrics(actual, fused),
                    "eager_bf16": metrics(actual, eager),
                }
            )
    return rows


def install_fused_storage_contract(model, monkeypatch):
    """CPU counterfactual with PyPTO's rounding boundaries, keeping full HF matrices.

    This diagnoses the BF16-storage/FP32-intermediate contract. It is separate
    from the unmodified FP32/BF16 eager acceptance references.
    """
    module = importlib.import_module("transformers.models.qwen3.modeling_qwen3")
    original_rope = module.apply_rotary_pos_emb

    def rounded(value):
        return value.bfloat16().float()

    def rope(*args, **kwargs):
        q, key = original_rope(*args, **kwargs)
        return rounded(q), rounded(key)

    def output_hook(module, args, output):
        return rounded(output)

    def input_hook(module, args):
        return (rounded(args[0]), *args[1:])

    def layer_forward(layer, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states, _ = layer.self_attn(hidden_states=hidden_states, **kwargs)
        hidden_states = rounded(residual + hidden_states)
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        return rounded(residual + hidden_states)

    monkeypatch.setattr(module, "apply_rotary_pos_emb", rope)
    for layer in model.model.layers:
        monkeypatch.setattr(layer, "forward", MethodType(layer_forward, layer))
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            norm.register_forward_hook(output_hook)
        layer.self_attn.v_proj.register_forward_hook(output_hook)
        layer.self_attn.o_proj.register_forward_pre_hook(input_hook)
        layer.mlp.down_proj.register_forward_pre_hook(input_hook)
    model.model.norm.register_forward_hook(output_hook)


@pytest.mark.skipif(
    not os.environ.get("PYPTO_QWEN3_MODEL") or not os.environ.get("PYPTO_QWEN3_OUTPUT"),
    reason="Requires real Qwen3-14B weights and saved TP outputs",
)
def test_prefill_and_decode_against_hugging_face(monkeypatch):
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(2)
    # Transformers 5 otherwise starts a separate four-worker weight loader.
    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "1")
    source = Path(os.environ["PYPTO_QWEN3_OUTPUT"])
    paths = sorted(source.glob("case_*/outputs.pt")) if source.is_dir() else [source]
    assert paths, f"No output artifacts found in {source}"
    precision = os.environ.get("PYPTO_QWEN3_REFERENCE_DTYPE", "fp32")
    assert precision in ("fp32", "bf16", "fused")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        os.environ["PYPTO_QWEN3_MODEL"],
        dtype=torch.bfloat16 if precision == "bf16" else torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    if precision == "fused":
        install_fused_storage_contract(model, monkeypatch)
    captured = {}
    world_size = torch.load(paths[0], map_location="cpu", weights_only=True).get("world_size", 2)
    handles = attach_stage_hooks(model, captured, world_size)
    report = {"reference_dtype": precision, "logits": [], "stages": [], "local_rounding": []}
    failures = []
    try:
        with torch.inference_mode():
            for path in paths:
                artifact = torch.load(path, map_location="cpu", weights_only=True)
                assert artifact.get("world_size", 2) == world_size
                assert len(artifact["logits"]) >= 2, "Acceptance requires both prefill and decode"
                cache = None
                position = 0
                for step, actual in enumerate(artifact["logits"]):
                    ids = artifact["prompt_ids"] if step == 0 else [artifact["generated_ids"][step - 1]]
                    captured.clear()
                    result = model(input_ids=torch.tensor([ids]), past_key_values=cache, use_cache=True)
                    cache = result.past_key_values
                    expected = result.logits[0, -1].float()
                    row = {
                        "case": path.parent.name,
                        "step": step,
                        **metrics(actual, expected),
                        "actual_token": int(actual.argmax()),
                        "expected_token": int(expected.argmax()),
                    }
                    report["logits"].append(row)
                    print(json.dumps(row), flush=True)
                    if (
                        row["relative_rmse"] >= 0.03
                        or row["cosine"] <= 0.999
                        or row["actual_token"] != row["expected_token"]
                    ):
                        failures.append(row)
                    if artifact.get("traces"):
                        trace = artifact["traces"][step]
                        assert trace.keys() == captured.keys(), (
                            trace.keys() - captured.keys(),
                            captured.keys() - trace.keys(),
                        )
                        for key, value in trace.items():
                            layer, stage = key.split(".", 1)
                            report["stages"].append(
                                {
                                    "case": path.parent.name,
                                    "step": step,
                                    "layer": int(layer),
                                    "stage": stage,
                                    **metrics(value, captured[key]),
                                }
                            )
                        for local in local_rounding_checks(trace, model, position + len(ids) - 1):
                            report["local_rounding"].append({"case": path.parent.name, "step": step, **local})
                    position += len(ids)
    finally:
        for handle in handles:
            handle.remove()
        if report_path := os.environ.get("PYPTO_QWEN3_REPORT"):
            Path(report_path).write_text(json.dumps(report, indent=2))
    assert not failures, f"{precision} numerical mismatches: {failures}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--noconftest"])
