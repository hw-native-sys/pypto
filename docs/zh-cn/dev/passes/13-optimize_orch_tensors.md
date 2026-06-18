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

本 pass 接受单个 `window_policy` 参数（默认 `"auto"`）：

```python
passes.optimize_orch_tensors(window_policy="auto")
```

`window_policy` 控制窗口外部化策略：

| 值 | Output 行为 | Input 行为 | 说明 |
|----|------------|-----------|------|
| `"auto"` | ABI-safe 单 piece window | ABI-safe + carrier-based dynamic | 默认；proof-based 安全 |
| `"all"` | 所有静态合法 piece（含多 piece） | 同 `auto` + carrier-based dynamic reader | 放宽 output 过滤 |
| `"off"` | 不做 windowing | 不做 windowing | 全局 kill switch，覆盖 kernel attrs |

**Per-kernel 覆盖**（通过 function attrs）：

```python
@pl.function(
    type=pl.FunctionType.InCore,
    attrs={
        "window_outputs": "coalesce",  # off | auto | all | coalesce
        "window_inputs":  "off",       # off | auto
    }
)
def my_kernel(...): ...
```

当全局 `window_policy != "off"` 时，kernel attrs 覆盖全局默认值。`"off"` 是全局 kill switch，不可被覆盖。

例如，在 Qwen prefill 风格的 dynamic-indexed KV-cache writer/reader 链中：

- `"auto"`：只保留局部简单 window。Dynamic KV-cache writer 和 reader 保持 full-tensor，不引入 carrier/remat。
- `"all"`：重写已证明正确的 writer window 为 exact pieces。无 carrier 的 dynamic reader 保持 full parent tensor（无 private dynamic reader）。

parent shape 含 dynamic dim 的 output window 会保守处理。对 dynamic parent 的
static-offset/static-size 静态 partial window 会保持 full-tensor，因为同一个编译图可能用更小的
dynamic extent 运行。动态 offset 的 window，例如运行时 slot 上的 KV-cache 写入，
只要静态证明和策略门槛允许，仍可以改写。

调试时，`PYPTO_WINDOW_EXTERNALIZE_INCLUDE` 和
`PYPTO_WINDOW_EXTERNALIZE_EXCLUDE` 可以按 callee 或参数名过滤候选。
`PYPTO_WINDOW_EXTERNALIZE_LOG=1` 会打印 `auto` 的 accept/reject 决策。

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

静态 eligibility 之后，默认 `auto` 策略还会再做一层保守 cost gate。它会拒绝增加 rewritten tensor 参数数量的候选，并默认拒绝 multi-piece output rewrite；dynamic-indexed input 只有在能复用 output-derived carrier 时才会 window 化，否则保持 full-parent。这样默认流水线不会为了局部 window 精度而换来更大的 dispatch 签名、更多 callsite orchestration，或收益依赖 workload 的私有 dynamic-window scan。需要手动研究 output 候选时，可以使用 `window_policy="all"`；input 侧 v5 仍只有 `off` / `auto` 两档。

支持的改写形态：

- `FinalStore`：callee 返回一次写入局部窗口的最终 `tile.store(...)` 结果
- `AggregateWindowLoop`：callee 在循环中携带一个或多个 `Out`，并写入静态可证明的聚合窗口，例如 outlined `kv_proj` 分组形态
- `PureInputWindowConsumer`：有数据返回的 callee 中，某个 `In` 张量参数只通过同一个局部输入窗口被使用
- `AggregateInputWindowLoop`：与 `AggregateWindowLoop` 输出改写配套使用；某个 `In` 张量参数只通过内部 loop 的局部 `tile.load`/`tensor.slice` 窗口读取，并且这些 offset 能沿同一个内部 loop 展开为一个静态可证明的 parent-shaped region，例如 qk norm 的 q/k 输入
- `RuntimeCurrentAggregator`：当任何函数有 `window_outputs="coalesce"`（per-kernel attr 或全局 coalesce）时，coalesced producer loop 返回的张量如果会喂给后续重复 full-tensor `In` reader，会插入一个 runtime 可见的 current marker。匹配由 attrs 驱动，不依赖模型特定的 callee 名或变量名。

输出窗口 eligibility：

- 写入必须是静态可证明的局部 `tile.store` 窗口或聚合窗口循环
- window shape 和 offset 必须足够静态，能够物化为 `tensor.slice`
- offset 必须是该 pass 可接受的外层循环变量仿射表达式
- multi-`Out` 改写采用全有或全无策略
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
- dynamic-indexed reader window 需要可证明的 min/max scan。`auto` 会拒绝这类候选。`all` 下，有 carrier 的 eligible writer-reader chain 走 carrier/remat 路径；无 carrier 的 dynamic reader 保持 full parent tensor（无 private dynamic reader）。

非目标与依赖模型：

- pass 不添加 explicit dependency edges
- pass 不重新引入 later full-parent-read guard
- pass 不预生成全局 window descriptor 数组
- pass 不拆分 SPMD launch，也不 externalize per-block SPMD window
- unsupported consumer，包括 full-tensor reader，保持 baseline/full-tensor input
- 部分 prefill full-tensor reader 如果属于已识别的 coalesced fan-in producer loop，仍可能获得 runtime-current marker；这不会缩小 reader 区域，但能让 runtime 依赖追踪匹配一个 current tensor，而不是反复扫描大量 producer window
- `DeriveCallDirections` 保持现有 sound 的顺序 `Out -> InOut` 规则；Pattern 5 只是在该 pass 运行前显式化可证明的局部窗口

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
