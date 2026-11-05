# 集成手写 C++ Kernel

PyPTO 可以从 PyPTO 编写的 orchestration（编排）中直接调用一个**已有的手写 C++
InCore kernel（核内核函数）**，而不经过 PyPTO 的 tile 代码生成。你在 DSL 里声明一个
*仅有签名的函数头*；orchestration 像调用普通 InCore kernel 那样调用它，但编译器会
跳过其函数体的代码生成，转而把所引用的 `.cpp` 编译为该 kernel。

适用场景：你已有一个调优过的 AICore kernel（例如定制的 attention kernel），希望用
PyPTO orchestration 驱动它，复用 PyPTO 的任务调度、依赖分析与运行时派发。

## 契约（Contract）

从运行时的角度看，外部 kernel 就是一个普通 InCore kernel，因此手写源码必须满足与
PyPTO 生成 kernel 相同的 ABI：

- 导出唯一的 `extern "C" void kernel_entry(__gm__ int64_t* args)` 入口（一个入口
  `.cpp` 一个 kernel；运行时按 `func_id` 派发，符号名固定）。这与 PyPTO 自身生成
  kernel 导出的入口完全一致。
- 声明的**参数顺序与方向**（`pl.Out` / `pl.InOut`）必须与 kernel 读取参数的方式一致
  —— orchestration 会依据声明构建任务负载（`add_input` / `add_inout` /
  `add_output`）。

函数体可以是裸的 `...`，也可以是单条 `return out` / `return first, second`，
返回已声明的 `Out`/`InOut` Tensor 参数。该 return 仅声明输出别名，不包含设备计算。
显式写回关系使动态 SPMD 的空任务分支能够保留调用者预分配的输出。
CANNBotDSL 适配器自动生成此契约。

### 多文件 kernel

`external_source` 指向唯一的入口 `.cpp`（导出 `kernel_entry` 的那个）。该文件可以
`#include` 任意数量的兄弟文件 —— PyPTO **在原始路径引用它**（不做拷贝），因此相对
include 会相对原始目录树解析。形如下面这样的 kernel：

```text
my_kernel/
  aic/entry.cpp          # external_source; #include "../kernel/impl.cce"
  kernel/impl.cce        #                  #include "../tiling/params.h"
  tiling/params.h
```

无需改动即可使用 —— 把 `external_source` 指向 `aic/entry.cpp`，整条 `../kernel/` /
`../tiling/` include 链会在编译时被拉入。位于运行时 include 路径上的头文件（如
`tensor.h`、`intrinsic.h`、`pto/pto-inst.hpp`）照常解析。

## `@pl.program` 写法

用 `@pl.function(type=AIC/AIV, external_source=...)` + 空体声明 kernel。混合的
**AIC + AIV** kernel（一次 `MixedKernels` 派发）表示为一个 `pl.FunctionType.Group`，
其成员为一个 AIC 与一个 AIV：

```python
from pathlib import Path
import pypto.language as pl

KDIR = Path(__file__).parent / "kernels"

@pl.program
class PagedAttention:
    @pl.function(type=pl.FunctionType.AIC, external_source=KDIR / "aic/pa.cpp")
    def PA_AIC(self, query: pl.Tensor[[B, H, D], pl.FP16], ...,
               out: pl.Out[pl.Tensor[[B, H, D], pl.FP16]], ...
               ) -> pl.Tensor[[B, H, D], pl.FP16]:
        ...                                   # 函数体在 aic/pa.cpp 中

    @pl.function(type=pl.FunctionType.AIV, external_source=KDIR / "aic/pa.cpp")
    def PA_AIV(self, ...same signature...) -> ...:
        ...

    @pl.function(type=pl.FunctionType.Group)
    def PA(self, ...same signature...) -> ...:
        r = self.PA_AIC(...)                  # 定义 group 成员
        self.PA_AIV(...)
        return r

    @pl.function(type=pl.FunctionType.Orchestration)
    def entry(self, query, ..., out):
        # 在此构建 tiling / workspace 张量（host 侧），然后派发：
        out = self.PA(query, ..., out)        # -> MixedKernels 派发
        return out
```

`external_source` 接受绝对路径，或相对于定义该 program 的文件的相对路径。AIC 与 AIV
成员可指向**同一**源文件（按核各编译一次），也可指向不同文件。

单核 kernel：只声明一个 `AIC` 或 `AIV` 函数，并在 orchestration 中直接调用（无需
group）。

## `@pl.jit.extern` 写法

在 `@pl.jit` 下，用 `@pl.jit.extern` 声明 kernel。`core_type="mixed"` 会自动展开为上面
的 AIC + AIV + Group 形式：

```python
@pl.jit.extern(core_type="mixed",
               aic_source="kernels/aic/pa.cpp",
               aiv_source="kernels/aiv/pa.cpp",
               include_dirs=["kernels/include", "third_party/attention/include"],
               dual_aiv_dispatch=True)
def pa(query: pl.Tensor[[B, H, D], pl.FP16], ...,
       out: pl.Out[pl.Tensor[[B, H, D], pl.FP16]], ...
       ) -> pl.Tensor[[B, H, D], pl.FP16]: ...

@pl.jit
def decode(query: pl.Tensor, ..., out: pl.Out[pl.Tensor]):
    out = pa(query, ..., out)                 # 依赖自动发现
    return out
```

单核写法：`@pl.jit.extern(core_type="aic"|"aiv", source="k.cpp")`。

对于无法通过入口源码相对引用找到的头文件，使用 `include_dirs=[...]`。与 `source`
相同，每个目录既可以是绝对路径，也可以相对于定义 `@pl.jit.extern` stub 的 Python
文件。解析后的目录会按给定顺序传给 CCEC；被引用头文件的内容和解析后的 include
路径元数据都会参与 JIT cache key。

默认情况下，mixed 外部 kernel 派发一个 AIC lane 和一个 AIV lane。手写 kernel 需要一个
AIC lane 加**两个** AIV 子 lane 时，设置 `dual_aiv_dispatch=True`。此时每个逻辑 block
会启动两次 AIV 入口；AIV 实现必须根据 sub-block id 切分工作，否则会产生重复写或写竞争。
该参数只允许与 `core_type="mixed"` 一起使用。A2/A3 external wrapper 必须通过
`intrinsic.h` 提供的 `get_sub_block_id(args)` 读取 runtime lane id；CCE 自身的 lane
accessor 没有接入 PyPTO 逻辑任务 payload，两个派发都可能读到 lane 0。

直接使用 `@pl.program` 声明时，在 AIV 成员上设置相同的派发约定：

```python
@pl.function(type=pl.FunctionType.AIV, external_source="kernels/aiv/pa.cpp",
             attrs={"dual_aiv_dispatch": True})
def pa_aiv(self, query: pl.Tensor, out: pl.Out[pl.Tensor]): ...
```

路径相对于定义该 kernel 的文件解析。修改所引用的 `.cpp` 会改变 JIT 缓存键，因此即使
Python stub 未变，kernel 变更也会触发重新编译。

## 编译器行为

- **Passes**：仅有函数头的函数原样穿过 pass 流水线（空体对 tile passes 是 no-op）；它
  仅豁免 `ReturnParamsExplicit` 属性 —— 该属性要求存在 `ReturnStmt`，而函数头本就没有。
- **Orchestration 代码生成**：为该 kernel 分配 `func_id`，并像 DSL kernel 一样生成派发
  —— 单核 AIC/AIV 的 `rt_submit_*_task`，或 group 的 `MixedKernels{aic_id, aiv_id, ...}`
  - `rt_submit_task`。
- **Backend**：对外部 kernel 跳过 ptoas，并在生成的 `kernel_config.py` manifest 中
  像生成 kernel 一样列出它（`func_id`、`core_type`、`signature`），但 `source` 指向
  原始手写 `.cpp`（在原始路径引用，使其兄弟文件仍可达），而非拷贝到产物目录下。

## 约束

- `external_source` 仅在 `FunctionType.AIC` / `FunctionType.AIV` 上有效。
- 函数体必须是裸的 `...`（仅签名）。
- 一个 `Group` 必须全为外部 kernel 或全为 DSL kernel —— 不允许在同一 group 中混用外部
  与 DSL 成员。

## 直接调用 Python DSL kernel

Python DSL 可以实现惰性的 `__pypto_jit__()` 协议，使用户只保留原始算子表达。
PyPTO 也会直接识别 CANNBotDSL 的 `cb.kernel` 对象，无需生产者提供 PyPTO hook。
安装配套 CANNBotDSL 和 PyPTO 适配器后，A5 示例为：

```python
import os

import cannbotdsl as cb
import pypto.language as pl
import torch
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig


@cb.kernel
def add_one(x: cb.Tensor, out: cb.Tensor):
    for i in cb.range(x.shape[1]):
        out[0, i] = x[0, i] + 1.0


@pl.jit
def program(x: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
    add_one(x, out)
    return out


def main():
    # Torch 仅用于 CPU 缓冲区准备和结果校验。
    x = torch.arange(32, dtype=torch.float32, device="cpu").reshape(1, 32)
    out = torch.empty_like(x)
    config = RunConfig(
        backend_type=BackendType.Ascend950,
        execution_mode=ExecutionMode.ONBOARD,
        device_id=int(os.environ.get("TASK_DEVICE", "0")),
    )

    program(x, out, config=config)

    torch.testing.assert_close(out, x + 1.0, rtol=0, atol=0)
    print("PASS: program -> CANNBotDSL add_one")
    print(out)


if __name__ == "__main__":
    main()
```

这是完整的 **program 模式**示例：`program(x, out, config=config)` 在 A5 上编译并执行，
断言检查复制回 CPU 的输出。Torch 仅用于 CPU 数据准备和结果校验。
用户不需要写 `TensorSpec`、导出调用、ABI wrapper 或 `jit.extern` 声明。
运行脚本前需要安装两个配套改动并激活 A5 环境。默认使用设备 0，
通过设备队列运行时读取 `TASK_DEVICE`。本功能不增加 Torch kernel 模式，
也不允许在 InCore 核内派发其他 kernel。

生产者钩子返回稳定的 `JITFunction(func, func_type="extern", external_builder=...)`。
构建器接收本次请求的 `SpecializeContext`，其中包含推导的 Tensor/Scalar 元数据，
并返回 `pypto.jit.foreign.ExternalKernelSpec`，提供源码路径、include 目录、core type、
Tensor 读写方向和可选的 backend/runtime 要求。生成的完整源码文件必须在编译和缓存期间
保持可用。PyPTO 不强制依赖生产者包。

框架将独立调用语句改写为隐式接收每个写入 Tensor，保留 SSA 和循环携带依赖。
支持多输出与关键字参数；输出实参必须是不同的 Tensor 变量。Python callable 仍为 void，
不能将调用用作值。支持从芯片 orchestration 及其 inline helper 调用；
推导签名不同的调用会生成独立特化，包括通过嵌套 helper 发起的调用。

无返回值的 inline helper 直接把 Tensor 形参传给 foreign writer 时，框架也会向其
调用者传播写回。有显式返回值的 helper 保持原有返回契约。若 helper 重新绑定了被写入
的形参，或通过局部别名/view 写入，请显式返回更新后的 Tensor，并在调用处接回结果。

普通范围边界的 `pl.range`、`pl.parallel` 循环变量，在循环体内未被重新赋值时，
会在外部调用处推导为 `pl.INDEX`。生产者可以直接接收这些行号或页号，
无需读取硬件启动时的 block ID。

构建器在查缓存前运行；源码、局部 include 和读写/core 元数据参与两层缓存身份，
本次请求内复用生成结果。生产者可以采用按内容寻址的存储；JIT 依赖对象不保存调用者的
Tensor、地址或本次请求的可变源码路径。

CANNBotDSL 适配器位于 `pypto.jit.cannbotdsl`。识别 kernel、转换 Tensor/Scalar 元数据、
通过生产者现有编译器 API 完成 tracing/lowering、生成 runtime ABI 包装以及配置外部任务，
均由 PyPTO 负责。适配器消费生成的 SourceBundle 和原生设备 ABI，包括物理参数顺序和
Tensor 读写信息。CANNBotDSL 负责通用编译和设备代码生成，此接入无需它提供
`export_device` 入口或 `cannbotdsl.interop.pypto` 模块。
CANNBotDSL 是可选依赖，仅在使用此适配器时导入。

通过嵌套 inline helper 调用时，Tensor 形状、dtype、layout 或 Scalar 类型不同的调用
会保留各自的外部 kernel 特化；签名相同的调用复用同一特化。这也适用于生产者 kernel
为同一个 Python 对象的情况。

适配器将 PyPTO FP8E8M0 直接映射到 CANNBotDSL 的 `float8_e8m0` 存储类型。
打包的 MX scale 布局需使用显式、等大小的 ND 存储视图；此操作只改元数据，保留原始
scale 字节，不重新编码。CANNBotDSL 接入目前支持 A5/TMR、最内层连续的
ND Tensor、显式 runtime Scalar、AIV/AIC/MIX 和基本动态形状。
`cb.get_block_idx()` 和 `cb.get_block_num()` 对应当前 simpler 任务的逻辑 SPMD
编号及数量。PyPTO 在提供方编译器折叠占位启动维度之前，将查询替换为隐藏运行时参数，
用户签名无需增加参数。MIX 向量代码中的 `cb.get_subblock_id()` 为 0 或 1，
`cb.get_subblock_dim()` 为 2；AIC 和独立 AIV 分别得到 0 和 1。

`cb.global_sync_all()` 和 mode-0 block 屏障要求全核同时驻留，并设置
`sync_start=True`。AIC/MIX 使用 `pl.system.available_cluster_count()`，
独立 AIV 使用 `pl.system.available_aiv_count()`。假定已有名为 `mixed_kernel`
的 MIX `@cb.kernel`：

```python
@pl.jit
def program(x: pl.Tensor, scratch: pl.Tensor, out: pl.Out[pl.Tensor]):
    count = pl.system.available_cluster_count()
    with pl.spmd(count, sync_start=True):
        mixed_kernel(x, scratch, out)
    return out


# CPU 缓冲区及 config 的准备方式见上面的完整示例。
program(x, scratch, out, config=config)
```

PyPTO 通过 `external_hard_syncall` 保留外部源码的屏障约束，复用已有占用校验，
并在任务提交前核对实际运行时核数。部分核启动或缺少 `sync_start` 会被拒绝。
其他提供方可用 `ExternalKernelSpec.hard_syncall` 声明 `"aic_only"`、`"aiv_only"`
或 `"mix"`，其中 MIX 必须启动两条向量通道。所有参与核必须按相同顺序到达屏障；
启动检查无法证明外部代码内部的控制流一致。跨卡通信、SIMT、隐式 GM 工作空间和
提供方管理的 debug buffer 仍不支持。

可运行的 A5 用例位于 `tests/st/runtime/external_kernel/test_cannbotdsl.py`，
覆盖部分核 SPMD 上下文映射，以及全核 AIV/MIX 的多轮邻核数据交换。用例使用
提供方的 Buffer 路径；CANNBotDSL 0.8.1 的 Buffer 校验可能拒绝纯标量核中的
`global_sync_all()`，PyPTO 保留该错误，不关闭提供方校验。
SDK 安装若缺少可选的 CPU-debug 条件头文件，可能使持久缓存保守跳过；
默认的进程内缓存仍可使用。
