# InjectGMPipeBuffer Pass

为通过 GM 路由槽位数据的后端（当前为 Ascend910B）注入 `__gm_pipe_buffer` 工作区参数。该 Pass 紧随 `ExpandMixedKernel` 运行。

## 概述

在 Ascend910B 上，跨核 `tpush`/`tpop` 通过共享 GM 缓冲区中转，而不是直连的核间通道。在 `ExpandMixedKernel` 已经把 mixed InCore 函数拆分为 AIC/AIV 并 prepend `aic_initialize_pipe` / `aiv_initialize_pipe` 之后，本 Pass：

1. 找出每个函数体中包含 `aic_initialize_pipe` 或 `aiv_initialize_pipe` 的函数。
2. 给上述每个函数追加一个新的 Out-tensor 参数 `__gm_pipe_buffer`。
3. 沿调用图向上传播参数：调用了已新增参数函数的调用者，也会拿到该参数（让工作区从 Orchestration 一路下沉到 AIC/AIV）。
4. 在 Orchestration 函数停止——它**不会**收到该参数，而是改为在每个调用点注入 `tensor.create`，在主机侧本地物化工作区并传入。

本 Pass 受 `BackendHandler::RequiresGMPipeBuffer()` **后端门控**。不需要 GM 路由 pipe 的后端（例如带直连核间通道的 Ascend950）会把它视为 no-op。

**前置要求**：

- 输入 IR 已完成 AIC/AIV 拆分并带有跨核 pipe setup（先运行 `ExpandMixedKernel`）。
- 后端报告 `RequiresGMPipeBuffer() == true`，否则本 Pass 为 no-op。

**何时使用**：在 `ExpandMixedKernel` 之后，针对 Ascend910B（或任意上报 `RequiresGMPipeBuffer` 的后端）运行。默认 tile pipeline 已经把它放在正确的位置。

> **说明**：本 Pass 是从 `ExpandMixedKernel` 中拆分出来的，让原先的 Pass 聚焦于 AIC/AIV 拆分逻辑，把 GM 工作区这一关注点收敛到一个由后端门控的独立 transform。

## API

| C++ | Python | 层级 |
| --- | ------ | ---- |
| `pass::InjectGMPipeBuffer()` | `passes.inject_gm_pipe_buffer()` | Program 级 |

**Python 用法**：

```python
from pypto.pypto_core import passes

inject_pass = passes.inject_gm_pipe_buffer()
program = inject_pass(program)
```

## 算法

```text
阶段 1 — 发现种子函数：
  遍历每个函数。如果（递归扁平化后的）函数体中包含 aic_initialize_pipe
  或 aiv_initialize_pipe 操作，则将其加入种子集合。

阶段 2 — 给种子函数追加参数并向上传播：
  使用工作队列，初始包含所有种子函数。每次弹出函数 F：
    - 给 F 的参数列表追加 __gm_pipe_buffer（Out-tensor）。
    - 对于程序中调用 F 的每个调用者 C：
        - 若 C 是 Orchestration：记入 orch_needs_tensor_create。
        - 否则：改写 C 中的调用点，将 C 自身的 __gm_pipe_buffer 参数转发，
          并且如果 C 还未被改写过，则将 C 加入工作队列。

阶段 3 — 在 Orchestration 调用点注入 tensor.create：
  对 orch_needs_tensor_create 中的每个 Orchestration 函数，prepend 一条
  物化工作区的 tensor.create，并改写相关调用以传递该工作区。
```

## 示例

**Before**（`ExpandMixedKernel` 之后，Ascend910B）：

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIC)
    def compute_aic(self, x, y, out_0):
        # ... aic_initialize_pipe(...) 和 Cube ops ...

    @pl.function(type=pl.FunctionType.AIV)
    def compute_aiv(self, x, y, out_0):
        # ... aiv_initialize_pipe(...) 和 Vector ops ...

    @pl.function(type=pl.FunctionType.Group)
    def compute(self, x, y, out_0):
        self.compute_aic(x, y, out_0)
        return self.compute_aiv(x, y, out_0)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, y):
        out_0 = pl.create_tensor([16, 128], dtype=pl.FP32)
        return self.compute(x, y, out_0)
```

**After**：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.AIC)
    def compute_aic(self, x, y, out_0, __gm_pipe_buffer):
        # ... aic_initialize_pipe(...) 引用 __gm_pipe_buffer ...

    @pl.function(type=pl.FunctionType.AIV)
    def compute_aiv(self, x, y, out_0, __gm_pipe_buffer):
        # ... aiv_initialize_pipe(...) 引用 __gm_pipe_buffer ...

    @pl.function(type=pl.FunctionType.Group)
    def compute(self, x, y, out_0, __gm_pipe_buffer):
        self.compute_aic(x, y, out_0, __gm_pipe_buffer)
        return self.compute_aiv(x, y, out_0, __gm_pipe_buffer)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, y):
        out_0 = pl.create_tensor([16, 128], dtype=pl.FP32)
        # 由本 Pass 注入：工作区大小按 FP32 元素数计算（ceil(required_bytes / 4)）。
        # 当前契约使用 FP32：tensor.create 的 shape 即元素数，配合 FP32 即可
        # 得到 4 * elements 字节的底层缓冲。
        __gm_pipe_buffer = pl.create_tensor([math.ceil(required_bytes / 4)], dtype=pl.FP32)
        return self.compute(x, y, out_0, __gm_pipe_buffer)
```

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass InjectGMPipeBuffer();
```

**实现**：`src/ir/transforms/inject_gm_pipe_buffer_pass.cpp`

- `HasInitializePipeOps` —— 递归扫描 `aic_initialize_pipe` / `aiv_initialize_pipe`（通过 `op_predicates::IsInitializePipe`）
- `AddGMSlotBufferParam` —— 追加 Out-tensor 参数
- `RewriteCallsForGMBuffer` —— 改写调用者的调用点
- `CreateGMPipeBufferTensorCreate` —— 构造 Orchestration 侧的 `tensor.create`
- `RewriteCallsWithPerCallGMBuffer` —— 驱动 Orchestration 侧的改写：插入 `tensor.create` 并按调用点转发工作区

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("inject_gm_pipe_buffer", &pass::InjectGMPipeBuffer,
           "Inject __gm_pipe_buffer workspace parameter for GM-routed cross-core pipes");
```

**测试**：通过 `tests/ut/ir/transforms/test_expand_mixed_kernel_a2a3.py`（Ascend910B 流水线）传递性覆盖。

## Pass 属性

| 属性 | 取值 |
| ---- | ---- |
| Required | SSAForm、MixedKernelExpanded、NormalizedStmtStructure |
| Produced | SSAForm、MixedKernelExpanded、NormalizedStmtStructure |
| Invalidated | — |

本 Pass 保留它所需要的所有属性（非 910B 后端为 no-op；910B 上是同形改写）。

## 设计取舍

| 决策 | 理由 |
| ---- | ---- |
| 与 `ExpandMixedKernel` 解耦 | 让原 Pass 专注于 AIC/AIV 函数体构建，并把 GM 工作区这一关注点收敛到一个可单独禁用的 Pass |
| 通过 `BackendHandler::RequiresGMPipeBuffer()` 进行后端门控 | 后端可按需启用，无需在 Pass 代码中散落 `if (backend == "910B")` 之类的分支（参见 `pass-context-config.md`） |
| 通过 `initialize_pipe` 操作识别种子，而非按函数名匹配 | 对函数重命名不敏感，也避免误改不需要 setup 的函数 |
| 在 Orchestration 停止传播 | Orchestration 是主机侧调度层，由它来物化工作区并下发给设备侧被调函数最为自然 |
| 通过 `tensor.create` 在调用点物化工作区 | 保持 Orchestration 对外签名稳定，让 codegen 直接发出常规的主机侧分配 |
