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

声明只携带签名，函数体为一个裸的 `...`。

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
               aiv_source="kernels/aic/pa.cpp")
def pa(query: pl.Tensor[[B, H, D], pl.FP16], ...,
       out: pl.Out[pl.Tensor[[B, H, D], pl.FP16]], ...
       ) -> pl.Tensor[[B, H, D], pl.FP16]: ...

@pl.jit
def decode(query: pl.Tensor, ..., out: pl.Out[pl.Tensor]):
    out = pa(query, ..., out)                 # 依赖自动发现
    return out
```

单核写法：`@pl.jit.extern(core_type="aic"|"aiv", source="k.cpp")`。

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
