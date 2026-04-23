# Torch Codegen 调试指南

本文档基于以下测试用例：

- `tests/st/codegen/test_torch_codegen_qwen3_decode_scope3_mixed.py`

用于在不同编译阶段验证 `torch_codegen` 的数值正确性。

## 前置准备

```python
import torch
from pypto.debug.torch_codegen import torch_codegen, validate_pass_ir_codegen_results
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.backend import BackendType, reset_for_testing, set_backend_type
```

下文默认你已经具备：

- 程序对象（例如来自 `@pl.program`）
- 输入张量（`dict[str, torch.Tensor]`）
- 参考实现生成的 golden 输出

## 构造输入与 golden 输出

如果你还没有准备好输入张量和 golden 参考，可以直接使用如下辅助代码：

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

## 1. 直接对 Program IR 做代码生成

该模式用于验证“未经过 pass 展开”的程序 IR。

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

适用场景：

- 快速确认生成的 PyTorch 代码与原始程序语义一致。

## 2. `PassManager(Default)` 展开后再代码生成

该模式用于验证 pass 展开后的 IR（包含 mixed-kernel / cross-core 形态）。

```python
reset_for_testing()
set_backend_type(BackendType.Ascend910B)  # 或 BackendType.Ascend950
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

适用场景：

- 验证默认 pass pipeline 后的行为是否正确。
- 验证后端相关（backend-dependent）转换是否符合预期。

## 3. 对每个 pass dump IR 逐一校验

使用 `validate_pass_ir_codegen_results` 对 dump 出来的每个 IR 文件执行 `torch_codegen` 并做结果对比。

```python
expected = {"out": golden_out}

validate_pass_ir_codegen_results(
    "build_output/qwen3_decode_scope3_mixed/passes_dump/",
    tensors,
    expected,
)
```

`validate_pass_ir_codegen_results` 的行为：

- 输入可以是 pass dump 目录，也可以是单个 `.py` IR 文件。
- 逐文件用 `pl.loads` 解析 IR。
- 用 `torch_codegen(..., check_shapes=True)` 生成可执行代码。
- 选择入口函数并执行。
- 按 `expected` 的 key 做张量对比（当前为 `dict[str, torch.Tensor]` 模式）。

典型输出格式：

```text
==================== 19_after_ExpandMixedKernel ====================
validate tensor: 'out', max_abs_diff: 1.234567e-03, pass: True
```

若某个 pass 的结果不符合预期，会抛出带 pass 文件上下文和 diff 信息的异常。

## 如何选择这三种方式

1. 先用“直接 program codegen”建立基线正确性。
2. 再用“default pass 后 codegen”验证展开后的真实执行路径。
3. 最后用“逐 pass dump 校验”定位具体从哪个 pass 开始出现偏差。
