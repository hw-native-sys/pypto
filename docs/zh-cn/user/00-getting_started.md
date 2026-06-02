# PyPTO 入门指南

## 什么是 PyPTO？

PyPTO 是一个基于 Python 的 Ascend NPU 内核编程框架。使用 `pypto.language` 模块编写计算内核，PyPTO 将其编译为优化的设备代码。

```python
import pypto.language as pl
from pypto import ir
```

所有内核代码使用 `pl` 命名空间。`ir` 模块提供编译和 IR 工具。

## Hello World：向量加法（张量级别）

最简单的内核操作 **Tensor（张量）** —— DDR 内存中的高级数组。PyPTO 自动处理数据搬运和内存分配。

```python
import pypto.language as pl
from pypto import ir

@pl.function
def vector_add(
    a: pl.Tensor[[64], pl.FP32],
    b: pl.Tensor[[64], pl.FP32],
) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
    return result
```

**逐行说明：**

| 行 | 功能 |
| -- | ---- |
| `@pl.function` | 将 Python 函数体解析为 PyPTO IR |
| `a: pl.Tensor[[64], pl.FP32]` | 输入：一维张量，64 个元素，32 位浮点 |
| `pl.add(a, b)` | 逐元素加法（分发到张量加法） |
| `return result` | 函数返回一个张量 |

装饰器执行后，`vector_add` 是一个 `ir.Function` 对象 —— 不是 Python 可调用函数。打印 IR：

```python
print(vector_add.as_python())
```

## Tile 内核：Load-Compute-Store

要进行硬件级控制，使用 **Tile（数据块）** —— 片上内存缓冲区。显式地从 DDR 加载数据、在片上计算、然后将结果存回。

```python
@pl.function
def vector_add_tile(
    a: pl.Tensor[[64], pl.FP32],
    b: pl.Tensor[[64], pl.FP32],
    output: pl.Out[pl.Tensor[[64], pl.FP32]],
) -> pl.Tensor[[64], pl.FP32]:
    # 从 DDR 加载到片上（Vec 内存）
    a_tile: pl.Tile[[64], pl.FP32] = pl.load(a, [0], [64])
    b_tile: pl.Tile[[64], pl.FP32] = pl.load(b, [0], [64])

    # 片上计算
    result: pl.Tile[[64], pl.FP32] = pl.add(a_tile, b_tile)

    # 存回 DDR
    out: pl.Tensor[[64], pl.FP32] = pl.store(result, [0], output)
    return out
```

**与张量版本的关键区别：**

| 概念 | 张量级别 | Tile 级别 |
| ---- | -------- | --------- |
| 数据位置 | DDR（自动） | 显式 load/store |
| 类型 | `pl.Tensor` | `pl.Tile`（片上） |
| 输出参数 | 返回值 | `pl.Out[pl.Tensor[...]]` |
| 内存控制 | 编译器决定 | 用户决定 |

**`pl.load(tensor, offsets, shapes)`** 从 DDR Tensor 拷贝一个区域到片上 Tile。

**`pl.store(tile, offsets, output_tensor)`** 将 Tile 拷贝回 DDR。

## 循环与累加

使用 `pl.range()` 进行循环。通过 `init_values` 实现循环携带值（累加器）：

```python
@pl.function
def sum_elements(
    a: pl.Tensor[[64], pl.FP32],
) -> pl.Tensor[[1], pl.FP32]:
    zero: pl.Tensor[[1], pl.FP32] = pl.create_tensor([1], dtype=pl.FP32)

    for i, (acc,) in pl.range(64, init_values=(zero,)):
        elem: pl.Tensor[[1], pl.FP32] = pl.slice(a, [1], [i])
        new_acc: pl.Tensor[[1], pl.FP32] = pl.add(acc, elem)
        acc_out: pl.Tensor[[1], pl.FP32] = pl.yield_(new_acc)

    return acc_out
```

**`init_values` 工作原理：**

1. `init_values=(zero,)` —— 累加器的初始值
2. `for i, (acc,)` —— `i` 是循环变量，`acc` 是当前累加器
3. `pl.yield_(new_acc)` —— 将 `new_acc` 作为下一次迭代的累加器
4. 循环结束后，`acc_out` 保存最终值

简单循环（无累加器）：

```python
for i in pl.range(10):
    # i 从 0 到 9
    ...

for i in pl.range(0, 100, 2):
    # i 从 0 到 98，步长 2
    ...
```

## 多函数程序

使用 `@pl.program` 将多个相互调用的函数组合在一起：

```python
@pl.program
class VectorAddProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_add(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        a_tile: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
        b_tile: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [0, 0], [128, 128])
        result: pl.Tile[[128, 128], pl.FP32] = pl.add(a_tile, b_tile)
        out: pl.Tensor[[128, 128], pl.FP32] = pl.store(
            result, [0, 0], output
        )
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        c: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor(
            [128, 128], dtype=pl.FP32
        )
        c = self.kernel_add(a, b, c)
        return c
```

**关键概念：**

| 概念 | 说明 |
| ---- | ---- |
| `@pl.program` | 装饰类 → 成为 `ir.Program` |
| `self` | 必需的第一个参数；从 IR 中去除 |
| `self.kernel_add(...)` | 程序内跨函数调用 |
| `FunctionType.InCore` | 在 AICore 上运行（计算内核） |
| `FunctionType.Orchestration` | 在主机端运行（任务图协调器） |

**函数类型（FunctionType）：**

- **`Opaque`**（默认）—— 无特定执行上下文
- **`InCore`** —— AICore 计算内核；使用 load/store 进行数据搬运
- **`Orchestration`** —— 主机端函数，创建张量并调度 InCore 任务

## 编译

编译程序以生成设备代码：

```python
from pypto.backend import BackendType

output_dir = ir.compile(
    VectorAddProgram,
    strategy=ir.OptimizationStrategy.Default,
    dump_passes=True,
    backend_type=BackendType.Ascend910B,
)
print(f"Generated code in: {output_dir}")
```

**`ir.compile()` 参数：**

| 参数 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `program` | （必需） | 要编译的 `ir.Program` |
| `output_dir` | `None` → `<base>/<name>_<timestamp>` | 代码生成、报告及（开启 dump 时）各 pass IR 的输出目录。`<base>` 取自 `PYPTO_PROG_BUILD_DIR` 环境变量，未设置时为 `build_output` |
| `strategy` | `OptimizationStrategy.Default` | Pass 流水线预设（`Default` 或 `DebugTileOptimization`） |
| `dump_passes` | `True` | 为 `True` 时，在每个 pass 后将 IR 快照写入 `output_dir/passes_dump/` |
| `backend_type` | `BackendType.Ascend910B` | Pass 与代码生成的目标硬件（`Ascend910B` 或 `Ascend950`） |
| `skip_ptoas` | `False` | 为 `True` 时跳过 ptoas，只生成原始 `.pto`（MLIR），不生成已编译的 C++ 包装代码 |
| `verification_level` | `None` | 可选的 `ir.VerificationLevel` 覆盖；`None` 表示使用默认（或环境变量 `PYPTO_VERIFY_LEVEL`） |

`DebugTileOptimization` 只是用于观察 PTO tile 流水线的调试捷径。除非你正在
专门排查策略选择或 pass 顺序，否则应优先使用 `Default`。

**不编译直接查看 IR：**

```python
# 打印单个函数
print(vector_add.as_python())

# 打印整个程序
print(VectorAddProgram.as_python())

# 省略中间类型标注（简洁模式）
print(vector_add.as_python(concise=True))
```

## 在 worker 上复用权重（DeviceTensor）

当同一个大张量被多次内核调用复用 —— 例如前向计算每个 batch 都要用到的权重矩阵 ——
每次都重新上传会浪费带宽。`ChipWorker.alloc_tensor` 在 device 上分配一块常驻内存，并返回
一个 `DeviceTensor` 句柄；`CompiledProgram` 接受它替代 `torch.Tensor` 入参。runtime
把这块 buffer 视为已经驻留在 device 上，对该入参跳过 H2D 与 D2H 拷贝。

```python
import torch
from pypto import ir
from pypto.runtime import ChipWorker, RunConfig

compiled = ir.compile(MyKernel)

with ChipWorker(config=RunConfig(platform="a2a3sim")) as w:
    weight = w.alloc_tensor((1024, 4096), torch.float16, init=host_weight)
    for batch in batches:
        out = torch.empty(batch.shape[0], 4096, dtype=torch.float16)
        compiled(batch, weight, out)
    w.free_tensor(weight)
```

### 注意事项

- `DeviceTensor` 永远不会被拷回 host。如果内核写入了它，需要在同一个 ChipWorker
  实例上显式调用 `w.copy_from(host_ptr, t.data_ptr, t.nbytes)` 读回结果。
- 必须在 ChipWorker 关闭之前用 `w.free_tensor(t)` 释放句柄，否则该内存会泄漏到
  ChipWorker 生命周期结束。
- 只有分配它的那个 ChipWorker 实例可以使用该 buffer。

### 显式 dispatch（`worker.run`、`worker.register`）

上面的 `with ChipWorker(): compiled(...)` 隐式模式依赖 `ContextVar` 发现：块内任何
`compiled(...)` 调用都会找到当前活跃的 worker 并复用它。这对脚本写法很方便，但 worker
对象本身被藏起来了 —— 库代码需要把 worker 传来传去，或者常驻服务想预注册多个 kernel
时，应该显式地驱动 dispatch：

```python
worker = ChipWorker(config=RunConfig(platform="a2a3sim"))
try:
    out = worker.run(compiled, a, b)                 # 单次
    handle = worker.register(compiled)               # 预注册
    for _ in range(1000):                            # 热循环，无 cid lookup
        handle(a, b, out)
finally:
    worker.close()                                   # cid + DeviceTensor 统一释放
```

`worker.register(compiled)` 立即触发 `compile_and_assemble` + simpler `register`，
配置错误会在这里抛出而不是到第一次 dispatch 才暴露。返回的 `RegistrationHandle` 是
可调用的、支持 `with handle:` 作用域清理，也有 `handle.unregister()` 用于显式提前
关闭。对同一个 `compiled.chip_callable` 多次 `register` 返回的是同一个 cid 的别名；
真正的 simpler 反注册在 `worker.close()` 里集中做。

`@pl.jit` 内核走同样的流程，先经过 `JITFunction.compile()`：

```python
@pl.jit
def add_kernel(a, b, out): ...

compiled = add_kernel.compile(sample_a, sample_b, sample_out)
handle = worker.register(compiled)
for batch in stream:
    handle(batch.a, batch.b, batch.out)
```

完整三种使用模式（推理服务、训练循环、register/dispatch 开销验证）见
`examples/runtime/explicit_dispatch.py`。

### 分布式（L3+）程序

`ir.compile` 对 L3+ 分布式程序返回的 `DistributedCompiledProgram` 与 `CompiledProgram`
一样接受 `DeviceTensor` 入参：用 worker 常驻 buffer 替代 `torch.Tensor`，runtime 即对该
参数跳过 H2D/D2H。这是在 generate 循环的多次 dispatch 之间保持大块静态权重常驻的推荐做法。

```python
import torch
from pypto.runtime import DeviceTensor

compiled = ir.compile(MyDistributedProgram)   # 返回 DistributedCompiledProgram
weight = DeviceTensor(dev_ptr, (1024, 4096), torch.float16)   # 调用方自管 buffer
compiled(x, weight, out)                       # weight：无 H2D/D2H 拷贝
```

#### 跨多次 dispatch 复用 setup（`prepare()`）

`compiled(*args)` 每次调用都会跑完整的分布式 setup（逐 chip 装配、构造 simpler Worker 并 fork）。
对反复 dispatch 同一程序的常驻服务（如 generate 循环），可调用一次 `compiled.prepare()` 得到
一个 `DistributedWorker` 句柄：setup 只做一次，多次 dispatch 复用同一个 worker。

per-call 的 IO buffer（输入**和**输出）是**在 `prepare()` 之前分配的共享内存 host 张量**，
原地复用 —— fork 出的 chip worker 通过继承的映射读写它们，所以输出直接从该张量读回。大块静态
权重则用 `rt.alloc_tensor` 一次性上传到 worker 常驻的 `DeviceTensor`（其 `init` 源同样必须是
`prepare()` 之前共享的张量），混合传入。非共享的 host 张量（或 `prepare()` 之后才分配的）会被拒绝
—— chip worker 看不到它。

```python
compiled = ir.compile(MyDistributedProgram)

# 共享内存 host buffer —— 必须在 prepare() 之前分配
host_x = torch.zeros((seq, 4096), dtype=torch.float16).share_memory_()
host_out = torch.zeros((seq, 4096), dtype=torch.float16).share_memory_()
host_weight = load_weight().share_memory_()

with compiled.prepare() as rt:                  # setup 只跑一次
    weight = rt.alloc_tensor(host_weight.shape, host_weight.dtype, init=host_weight)
    for step in generate_steps:
        host_x.copy_(next_input(step))          # 原地刷新输入
        rt(host_x, weight, host_out)            # host shm IO + 常驻权重
        consume(host_out)                       # 直接读输出
    rt.free_tensor(weight)
# 退出时自动 rt.close()
```

## 下一步

- **[语言指南](01-language_guide.md)** —— 类型、操作、控制流、内存和编译的完整参考
- **[操作参考](02-operation_reference.md)** —— 所有 `pl.*` 操作的查找表
