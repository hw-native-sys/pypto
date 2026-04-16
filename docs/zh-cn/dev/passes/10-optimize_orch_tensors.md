# OptimizeOrchTensors Pass

优化编排函数与 InCore 函数之间的张量缓冲区使用，消除冗余分配并改善数据流。

## 概述

`ConvertTensorToTileOps` 之后，编排函数在每个 InCore 调用点分配输出张量（`tensor.create`），即使在循环内同一缓冲区可以复用。本 pass 应用三个优化模式来减少分配并改善缓冲区布局信息。

**前置条件**：

- 输入 IR 必须已完成 InCore 作用域提取和 tile 转换（需先运行 `ConvertTensorToTileOps`）

**使用时机**：在 `ConvertTensorToTileOps` 之后、`FlattenTileNdTo2D` 之前运行。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::OptimizeOrchTensors()` | `passes.optimize_orch_tensors()` | Program 级 |

**Python 用法**：

```python
from pypto.pypto_core import passes

opt_pass = passes.optimize_orch_tensors()
program_opt = opt_pass(program)
```

## 优化模式

本 pass 按顺序应用四个模式。每个模式可以看到前一个模式的结果。

### 模式 1：迭代参数复用（IterArgReuseOptimizer）

**问题**：在 `for`/`while` 循环内，每次迭代都通过 `tensor.create` 分配新的输出张量，即使 InCore 结果作为 iter-arg 反馈到下一次迭代。

**方案**：将 `Out` 参数合并到对应的 `In` 参数（提升为 `InOut`），移除 `tensor.create`，并重定向 `tile.store` 写入复用的缓冲区。

**优化前**：

```python
for i in pl.range(N, init_values=[init_buf]):
    out: pl.Tensor = pl.tensor.create(shape, dtype=pl.FP32)  # 冗余分配
    result: pl.Tensor = self.incore_fn(iter_arg, out)          # In + Out 参数
    pl.yield_(result)
```

**优化后**：

```python
for i in pl.range(N, init_values=[init_buf]):
    result: pl.Tensor = self.incore_fn(iter_arg)  # InOut 参数（复用 iter-arg 缓冲区）
    pl.yield_(result)
```

### 模式 2：Assemble 父张量步长（AssembleParentStridesOptimizer）

**问题**：当编排函数通过 `tensor.assemble` 将 InCore 结果分散到更大的张量时，InCore 函数的 `tile.store` 不知道父张量的步长，可能导致次优的内存布局。

**方案**：分析编排函数中的 `tensor.assemble(parent, incore_result, offset)` 模式，将父张量的形状作为 `TensorView` 步长附加到 InCore 函数的 `Out` 参数类型上，使 `tile.store` 能使用正确的内存布局。

### 模式 4：切片输入步长（SliceInputStridesOptimizer）

**问题**：当编排函数将切片张量（`tensor.slice`）作为 `In` 参数传递给 InCore 函数时，InCore 函数的参数使用连续步长（从自身形状计算），而非父张量的步长。当切片是父张量的非连续视图时，这会导致错误的内存访问。

**方案**：分析编排函数中的 `tensor.slice(parent, size, offset)` 模式。当切片结果作为 `In` 参数传递给 InCore 调用时，将父张量形状推导出的步长通过 `TensorView` 附加到 InCore 函数的 `In` 参数类型上，使 `tile.load` 能使用正确的内存布局。

### 模式 3：Assemble 循环重写（AssembleLoopRewriter）

**问题**：InCore 函数包含一个通过 `tile.assemble` 将结果累积到 iter-arg 的 `for` 循环，然后存储最终结果。`tile.assemble` 每次迭代都创建中间 tile 副本。

**方案**：将循环体重写为直接使用 `tile.store`（写入 `Out` 参数），用 `Out` 参数初始化 iter-arg 代替 `tile.create`。

## 示例（模式 1）

**优化前**：

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def compute(self, x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        x_tile = pl.load(x, (0,), (64,))
        y_tile = pl.tile.add(x_tile, x_tile)
        ret = pl.store(y_tile, (0,), out_0)
        return ret

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, buf: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        for i in pl.range(10, init_values=[buf]):
            out_0 = pl.tensor.create((64,), dtype=pl.FP32)
            result = self.compute(iter_arg, out_0)
            pl.yield_(result)
        return loop_result
```

**优化后**（模式 1 将 Out 合并到 In，提升为 InOut）：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def compute(self, x: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        x_tile = pl.load(x, (0,), (64,))
        y_tile = pl.tile.add(x_tile, x_tile)
        ret = pl.store(y_tile, (0,), x)
        return ret

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, buf: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        for i in pl.range(10, init_values=[buf]):
            result = self.compute(iter_arg)
            pl.yield_(result)
        return loop_result
```

`tensor.create` 被消除；iter-arg 缓冲区跨迭代复用。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**实现**：`src/ir/transforms/optimize_orch_tensors_pass.cpp`

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_optimize_orch_tensors.py`

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| Required | SplitIncoreOrch, IncoreTileOps |
| Produced | SplitIncoreOrch, IncoreTileOps |
| Invalidated | — |

## 关键组件

| 组件 | 作用 |
| ---- | ---- |
| `IterArgReuseOptimizer` | 模式 1 — 合并 Out 参数到 In 参数以复用循环携带缓冲区 |
| `AssembleParentStridesOptimizer` | 模式 2 — 通过 TensorView 附加父张量步长 |
| `SliceInputStridesOptimizer` | 模式 4 — 通过 TensorView 为切片输入的 In 参数附加父张量步长 |
| `AssembleLoopRewriter` | 模式 3 — 将 tile.assemble 循环重写为 tile.store 循环 |
| `BuildOutParamReturnMappings` | 共享辅助函数 — 通过 tile.store 映射 Out 参数到返回索引 |
| `ComputeRowMajorStrides` | 共享辅助函数 — 从形状计算行主序步长 |

## 作用范围

| 函数类型 | 操作 |
| -------- | ---- |
| InCore | 参数/函数体重写（模式 1、3、4） |
| Orchestration / Opaque | 调用点重写（模式 1、2） |
| Group | 不变 |
