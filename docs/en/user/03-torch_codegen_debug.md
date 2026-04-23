# Torch Codegen Debug Guide

This guide summarizes the practical `torch_codegen` workflows used in:

- `tests/st/codegen/test_torch_codegen_qwen3_decode_scope3_mixed.py`

Use these workflows to debug numerical correctness at different compilation stages.

## Prerequisites

```python
import torch
from pypto.debug.torch_codegen import torch_codegen, validate_pass_ir_codegen_results
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.backend import BackendType, reset_for_testing, set_backend_type
```

In examples below, assume you already have:

- A program object (for example from `@pl.program`)
- Input tensors (`dict[str, torch.Tensor]`)
- Golden output from a reference function

## Build Inputs and Golden Output

If you do not already have test tensors and a golden reference, use a helper like this:

```python
import torch


def build_tensors(batch: int, hidden_size: int, intermediate_size: int) -> dict[str, torch.Tensor]:
    attn_out = (torch.randn([batch, hidden_size], dtype=torch.float32) / (hidden_size**0.5)).to(
        torch.bfloat16
    )
    hidden_states = (torch.randn([batch, hidden_size], dtype=torch.float32) / (hidden_size**0.5)).to(
        torch.bfloat16
    )
    wo = (torch.randn([hidden_size, hidden_size], dtype=torch.float32) / (hidden_size**0.5)).to(
        torch.bfloat16
    )
    post_rms_weight = torch.randn([1, hidden_size], dtype=torch.float32) / (hidden_size**0.5)
    w_gate = (
        torch.randn([hidden_size, intermediate_size], dtype=torch.float32) / (intermediate_size**0.5)
    ).to(torch.bfloat16)
    w_up = (torch.randn([hidden_size, intermediate_size], dtype=torch.float32) / (intermediate_size**0.5)).to(
        torch.bfloat16
    )
    w_down = (torch.randn([intermediate_size, hidden_size], dtype=torch.float32) / (hidden_size**0.5)).to(
        torch.bfloat16
    )
    out = torch.zeros([batch, hidden_size], dtype=torch.bfloat16)

    return {
        "attn_out": attn_out,
        "hidden_states": hidden_states,
        "wo": wo,
        "post_rms_weight": post_rms_weight,
        "w_gate": w_gate,
        "w_up": w_up,
        "w_down": w_down,
        "out": out,
    }


def golden(tensors: dict[str, torch.Tensor]) -> None:
    eps = 1e-6
    o_proj = torch.matmul(tensors["attn_out"].float(), tensors["wo"].float())
    resid1 = o_proj + tensors["hidden_states"].float()
    variance = resid1.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    normed_bf16 = (resid1 * inv_rms * tensors["post_rms_weight"]).bfloat16()
    gate = torch.matmul(normed_bf16.float(), tensors["w_gate"].float())
    up = torch.matmul(normed_bf16.float(), tensors["w_up"].float())
    mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
    down = torch.matmul(mlp_bf16.float(), tensors["w_down"].float())
    tensors["out"][:] = (down + resid1).bfloat16()


batch, hidden_size, intermediate_size = 16, 512, 1024
torch.manual_seed(42)
tensors = build_tensors(batch, hidden_size, intermediate_size)
golden_tensors = {k: v.clone() for k, v in tensors.items()}
golden(golden_tensors)
golden_out = golden_tensors["out"]
```

## 1. Codegen Directly from Program IR

Use this mode to verify frontend program IR without pass expansion.

```python
code = torch_codegen(program, check_shapes=True)

ns = {}
exec(code, ns)  # noqa: S102

out = tensors["out"].clone()
ns["scope3"](
    tensors["attn_out"],
    tensors["hidden_states"],
    tensors["wo"],
    tensors["post_rms_weight"],
    tensors["w_gate"],
    tensors["w_up"],
    tensors["w_down"],
    out,
)

assert torch.allclose(out, golden_out, rtol=5e-2, atol=5e-2)
```

When to use:

- Fast check that generated PyTorch code matches your original program behavior.

## 2. Codegen After `PassManager(Default)`

Use this mode to verify pass-expanded IR (including mixed-kernel and cross-core forms).

```python
reset_for_testing()
set_backend_type(BackendType.Ascend910B)  # or BackendType.Ascend950
try:
    transformed = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    code = torch_codegen(transformed, check_shapes=True)
finally:
    reset_for_testing()

assert "_cross_core_rt.push_to_" in code
assert "_cross_core_rt.pop_from_" in code

ns = {}
exec(code, ns)  # noqa: S102
out = tensors["out"].clone()
ns["scope3"](
    tensors["attn_out"],
    tensors["hidden_states"],
    tensors["wo"],
    tensors["post_rms_weight"],
    tensors["w_gate"],
    tensors["w_up"],
    tensors["w_down"],
    out,
)
assert torch.allclose(out, golden_out, rtol=5e-2, atol=5e-2)
```

When to use:

- Validate behavior after the full default pass pipeline.
- Validate backend-dependent transformation behavior.

## 3. Validate Every Dumped Pass IR

Use `validate_pass_ir_codegen_results` to execute `torch_codegen` on each dumped IR file and compare outputs.

```python
expected = {"out": golden_out}

validate_pass_ir_codegen_results(
    "build_output/qwen3_decode_scope3_mixed/passes_dump/",
    tensors,
    expected,
)
```

`validate_pass_ir_codegen_results` behavior:

- Accepts a pass dump directory or a single `.py` IR file path.
- Parses each IR file with `pl.loads`.
- Generates runnable PyTorch code with `torch_codegen(..., check_shapes=True)`.
- Executes the selected entry function.
- Compares tensors by key from `expected` (currently `dict[str, torch.Tensor]` mode).

Typical output format:

```text
==================== 19_after_ExpandMixedKernel ====================
validate tensor: 'out', max_abs_diff: 1.234567e-03, pass: True
```

If a pass result is wrong, it raises with pass file context and max diff.

## Which Workflow to Choose

1. Use direct program codegen first to confirm baseline correctness.
2. Use default-pass codegen to check transformed IR behavior on target backend.
3. Use pass-dump validation to locate the exact pass where behavior diverges.
