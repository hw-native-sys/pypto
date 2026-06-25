# OptimizeOrchTensors Pass

优化编排函数与 InCore 函数之间的张量缓冲区使用，消除冗余分配、改善布局信息，并把静态可证明的局部张量窗口在 orchestration callsite 显式化。

## 概述

`ConvertTensorToTileOps` 之后，编排函数在每个 InCore 调用点分配输出张量（`tensor.create`），即使在循环内同一缓冲区可以复用。本 pass 应用五个优化模式来减少分配、改善缓冲区布局信息，并在 orchestration callsite 显式化可静态证明的局部张量窗口。

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

本 pass 按顺序应用五个模式。每个模式可以看到前一个模式的结果。

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

### 模式 3：Assemble 循环重写（AssembleLoopRewriter）

**问题**：InCore 函数包含一个通过 `tile.assemble` 将结果累积到 iter-arg 的 `for` 循环，然后存储最终结果。`tile.assemble` 每次迭代都创建中间 tile 副本。

**方案**：将循环体重写为直接使用 `tile.store`（写入 `Out` 参数），用 `Out` 参数初始化 iter-arg 代替 `tile.create`。

### 模式 4：切片输入步长（SliceInputStridesOptimizer）

**问题**：当编排函数将切片张量（`tensor.slice`）作为 `In` 参数传递给 InCore 函数时，InCore 函数的参数使用连续步长（从自身形状计算），而非父张量的步长。当切片是父张量的非连续视图时，这会导致错误的内存访问。

**方案**：分析编排函数中的 `tensor.slice(parent, size, offset)` 模式。当切片结果作为 `In` 参数传递给 InCore 调用时，将父张量形状推导出的步长通过 `TensorView` 附加到 InCore 函数的 `In` 参数类型上，使 `tile.load` 能使用正确的内存布局。

### 模式 5：静态窗口外提（OutWindowExternalizer）

**问题**：某些 outlined callee 实际只写入大 `Out` 张量中的一个静态可证明局部窗口，或只消费大 `In` 张量中的一个静态可证明局部窗口，但调用点仍传入整块张量。后续依赖分析会把它视为整块缓冲区访问，从而引入不必要的串行化。

**方案**：为 callee 克隆出 `__windowed` 版本，收窄被改写的张量参数类型，并局部化内部 offset。然后在 orchestration callsite 物化局部 slice。输出窗口使用 `slice + __windowed call + assemble`：

```python
out_window = pl.tensor.slice(out, shape, offset)
out_window_next = self.kernel__windowed(..., out_window)
out = pl.tensor.assemble(out, out_window_next, offset)
```

输入窗口使用相同的 callsite 局部 slice 物化，但不需要 assemble：

```python
in_window = pl.tensor.slice(inp, shape, offset)
result = self.consumer__windowed(in_window, ...)
```

如果待物化 slice 的 parent 是循环返回 alias，pass 会对 `ForStmt` 和 `WhileStmt`
都把这个 parent 改写为该循环 codegen 可见的 init tensor，避免生成的
orchestration C++ 在作用域外引用 loop-return SSA 名字。循环体内部的
loop-carried iter-arg 不会被这样折叠。

本 pass 有意保持保守的 window eligibility。它不会按 `topk` 等算子名字做特判；只有 callee 函数体能证明满足下面的访问模式时，才会 window 化。

支持的改写形态：

- `FinalStore`：callee 返回一次写入局部窗口的最终 `tile.store(...)` 结果
- `AggregateWindowLoop`：callee 在循环中携带一个或多个 `Out`，并写入静态可证明的聚合窗口，例如 outlined `kv_proj` 分组形态
- `PureInputWindowConsumer`：有数据返回的 callee 中，某个 `In` 张量参数只通过同一个局部输入窗口被使用
- `AggregateInputWindowLoop`：与 `AggregateWindowLoop` 输出改写配套使用；某个 `In` 张量参数只通过内部 loop 的局部 `tile.load`/`tensor.slice` 窗口读取，并且这些 offset 能沿同一个内部 loop 展开为一个静态可证明的 parent-shaped region，例如 qk norm 的 q/k 输入

输出窗口 eligibility：

- 写入必须是静态可证明的局部 `tile.store` 窗口或聚合窗口循环
- window shape 和 offset 必须足够静态，能够物化为 `tensor.slice`
- offset 必须是该 pass 可接受的外层循环变量仿射表达式
- multi-`Out` 改写采用逐输出策略：每个 `Out` 参数独立评估 window eligibility，不满足条件的输出保持 full-tensor baseline 参数
- 如果同一 callsite 中多个被 externalize 的 `Out` 参数解析到同一个 parent tensor，该 callsite 保持 full-tensor；Pattern 5 不尝试把多个 `tensor.assemble` 串成同一个 parent state
- 顺序循环 sibling 只有在每个被改写 `Out` 都能证明跨 sibling iteration 不重叠时才改写
- 同一 scope 内写入同一 parent 或 alias parent tensor 的 sibling writer，只要每个 writer 自身满足静态 output-window eligibility，仍然可以 externalize；但如果同一个 parent 还存在无法 externalize 成 output window 的 sibling full writer（`Out` 或 `InOut`），则写同一 parent 的其他 writer 也保持 full-tensor，避免这个非 window writer 掩盖只被部分初始化的区域
- 对剩余 windowed writer，写写/写读顺序交给 runtime TensorMap 对实际 submit 的 window descriptor 做 overlap 建边
- sibling-writer alias 收集会递归进入嵌套 `SeqStmts`、`ForStmt`、`WhileStmt` 和 `IfStmt` body，因此 loop return、tuple projection 这类 tensor alias 会先折叠到 codegen 可见的 parent，再生成 call-site slice
- 后续 full-parent read 不会关闭输出 window；callsite 暴露真实窗口张量之后，正确性依赖 runtime TensorMap overlap dependence

输入窗口 eligibility：

- 参数必须是 `In` 张量
- callee 内部对该参数的每一次引用都必须匹配同一个局部窗口
- 支持的引用只有 `tile.load` 和 `tensor.slice`
- 拒绝 transpose load
- `tile.load` 的 read shape 必须等于候选 window shape
- 所有匹配引用必须具有相同 window shape 和 offset
- 如果存在任何 unsupported ref，则整个输入参数保持 full-tensor
- pure input-window 的 shape 和 callee-local offset 表达式只能引用 callee 参数；callsite 替换后这些参数可以携带外层 loop-affine 值，windowed callee 内部再相对 `[0, ...]` 读取
- 对 `PureInputWindowConsumer`，如果匹配出的窗口其实是 zero offset 的 full shape，则跳过，因为 slice 不能暴露更窄依赖
- 对 `PureInputWindowConsumer`，如果 callee 没有数据返回，则保持 full-tensor；这类 consumer 可能是 side-effect 或 fence task，full input 本身用于表达更宽的依赖
- input-only 的 `Submit` callsite 保持 full-tensor；在 `manual_scope` 中，即使 callee body 只读局部窗口，full input 也可能有意表达更宽的依赖
- 如果同一个 callee 同时满足 output-window 改写，已经证明成立的 pure input window 会被保留，并在同一个 callsite 一起物化
- 对 `AggregateInputWindowLoop`，所有引用必须位于同一个静态 `ForStmt` 内，至少一个 offset 维度必须随该 loop 变化，并且聚合窗口必须等于输入 parent shape；权重子窗口这类 partial aggregate read 仍保持 full-tensor

非目标与依赖模型：

- pass 不添加 explicit dependency edges
- pass 不重新引入 later full-parent-read guard
- pass 不预生成全局 window descriptor 数组
- pass 不拆分 SPMD launch，也不 externalize per-block SPMD window
- unsupported consumer，包括 full-tensor reader，保持 baseline/full-tensor input
- `DeriveCallDirections` 保持现有 sound 的顺序 `Out -> InOut` 规则；Pattern 5 只是在该 pass 运行前显式化可证明的局部窗口

### Windowize Opt-in 使用方法

Pattern 5（window externalization）**默认关闭**。要给某个 kernel 启用，在 kernel 装饰器上添加 `windowize=True`：

```python
@pypto.kernel(windowize=True)
def my_kernel(...):
    ...
```

只有显式标注 `windowize=True` 的 InCore function 才能进入 Pattern 5 的候选。这是一个 per-kernel 的二值开关，没有全局开关或方向级覆盖参数。

**何时启用**：观察模型的 swimlane 时间线。如果发现 sibling orchestration tasks 之间有串行化——一个 task 等另一个虽然它们操作同一 buffer 的不同区域——串行化往往是因为依赖分析看到的是全张量而非实际访问的局部窗口。通过显式化局部窗口（`slice → windowed callee → assemble` 模式），编译器能表达更窄的依赖关系，减少或消除串行，提高流水线利用率。

泳道图中典型的线索：

- 同 core 上的相邻 task 本可重叠但实际没有
- task 执行时间远短于它们之间的间隔
- TensorMap auto-dependency edges 出现在操作同一 buffer 不同区域的 tasks 之间

**风险和权衡**：

- **增大 orchestrator 开销**：任务依赖粒度更细 → TensorMap 自动建依赖的边数增多 → 编排编译阶段开销增大。
- **增大 scheduler 开销**：tasks 的 complete 不同时 → dispatch 和 complete 变碎（更多单独 dispatch 调用、更频繁的 complete 通知）→ 运行时调度开销增大。

如果这些开销主导了性能收益——表现在泳道图中空闲时间增加，超过了节省的串行化开销——则**不建议**对该 kernel 开启 `windowize=True`。每个标注应通过对比开关前后的 swimlane 效果来单独评估。

**首次使用**：从局部窗口访问模式清晰稳定的小集合 kernel 开始（例如 transformer prefill 中的 rmsnorm、q_proj、kv_proj）。验证生成的 orchestration C++ 和 swimlane 效果后再增加更多标注。每个 kernel 单独评估；在一个模型上收益的 kernel 在另一个模型上可能回退。

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
| `OutWindowExternalizer` | 模式 5 — 将 eligible 的局部 Out 写和 eligible In-window consumer 改写为显式 callsite slice |
| `BuildOutParamReturnMappings` | 共享辅助函数 — 通过 tile.store 映射 Out 参数到返回索引 |
| `ComputeRowMajorStrides` | 共享辅助函数 — 从形状计算行主序步长 |

## 作用范围

| 函数类型 | 操作 |
| -------- | ---- |
| InCore / outlined non-builtin callee | 参数/函数体重写（模式 1、3、4、5） |
| Orchestration / Opaque | 调用点重写（模式 1、2、5） |
| Group | 不变 |
