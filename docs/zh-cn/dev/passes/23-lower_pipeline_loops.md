# LowerPipelineLoops Pass

在 tile 层级展开 `pl.pipeline(N, stage=F)` 循环：将循环体复制 `F` 份以启用 ping-pong 缓冲，同时保留外层顺序循环。

## 概述

`pl.unroll(N)` 在 SSA 之前的 slot #1 完整展开循环为 `N` 份副本。用户使用它通常并非需要 `N` 份副本，而是希望获得不同的 tile MemRef —— 否则 `MemoryReuse` 会把生命周期相邻的 tile 合并为同一缓冲区，导致 ping-pong 失效。

`LowerPipelineLoops` 提供更精细的开关：在 tile 层级把循环体复制 `F` 份（典型值 2–4），保留外层 `N/F` 次顺序迭代。每个副本获得独立的定义变量（保持 SSA），各自操作独立的 tile，下游 `MemoryReuse` 无法将其合并。

`pl.pipeline(...)` 在内部生成 `ForStmt(kind=ForKind::Pipeline, attrs={"pipeline_stages": F})`。结构性不变量 `kind == Pipeline ⇔ pipeline_stages 属性存在`（双向；由 `PipelineLoopValid` 验证器强制）保证 kind 与属性始终成对存在。`LowerPipelineLoops` 在 `F > 1` 时触发：复制循环体并把属性下调为 `1` 作为降级后的标记位，保留 `ForKind::Pipeline` 让下游 `CanonicalizeIOOrder` 继续作用。再次运行 `LowerPipelineLoops` 看到 `factor == 1` 即跳过（自然幂等）。

**前置条件**: SSAForm、SplitIncoreOrch、IncoreTileOps、TileOps2D、TileMemoryInferred、NormalizedStmtStructure。

**流水线位置**: 位于 [`NormalizeReturnOrder`](22-normalize_return_order.md) 之后、`CanonicalizeIOOrder` 与 `InitMemRef` 之前（slot 20.5）。此时 tile 结构决策已完成；同时早于 `CanonicalizeIOOrder`/`InitMemRef`/`MemoryReuse`，使其看到每个副本独立的 tile 变量。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::LowerPipelineLoops()` | `passes.lower_pipeline_loops()` | 函数级 |

```python
from pypto import passes
result = passes.lower_pipeline_loops()(program)
```

## DSL 语法

```python
# 每次外层迭代复制循环体 4 次；外层循环 16 次，步长为 4。
for i in pl.pipeline(64, stage=4):
    tile_x = pl.tile.load(input_a, [i * 128], [128])
    pl.tile.store(tile_x, [i * 128], output)
```

## 行为

对于 `attrs_["pipeline_stages"] = F`（`F > 1`）的循环：

- **主循环**：步长为 `F*step`，循环体为 `F` 份副本组成的 `SeqStmts`，kind 仍为 `ForKind::Pipeline`，属性下调为 `1`（降级后的标记位）。kind 与属性成对保留以维持 `PipelineLoopValid` 不变量，使 IR 在 print/parse 往返中保持一致（输出形式为 `pl.pipeline(..., stage=1)`）。
- **克隆细节**：每份副本通过 `DeepClone(body, {loop_var → new_var + k * step}, clone_def_vars=true)` 生成。每个副本拥有新鲜的定义变量，既保持 SSA，又给 `MemoryReuse` 提供独立的 tile 身份。

`stage=1` 是无操作触发：本 Pass 保留循环原样（kind 与属性都不动），仅递归进入循环体处理嵌套 pipeline。`CanonicalizeIOOrder` 随后基于该标记位完成 IO 重排并降级 kind / 移除属性。用户手写的 `pl.pipeline(stage=1)` 与 `factor>1` 路径输出后的循环走相同流程 —— 都需要 IO 重排但无需进一步复制；这也使再次运行 `LowerPipelineLoops` 自然幂等。

根据 `start` / `stop` 是否为编译期常量，分为两种降级模式，区别仅在主循环的 `stop` 与余数处理方式。

### 静态边界 —— `start`、`stop`、`step` 均为编译期整数

迭代次数 `T = (stop - start) / step`：

- 主循环终点为 `start + (T // F) * F * step`。
- 若 `T % F != 0`，再发射一段**裸 `SeqStmts`**：`T % F` 份克隆体，偏移为 `start + (T // F) * F * step + j * step`（`j ∈ [0, T%F)`），直接扁平化到外层作用域。余数已知，无需运行时分派，也无需任何包装结构。
- 当源循环存在 `iter_args` 时，尾部克隆后附加 `AssignStmt` 将源循环的 `return_vars` 绑定到尾部最终 yield 表达式，保证下游引用仍然有效。

### 动态边界 —— `start` / `stop` 为运行时 Expr（`step` 仍为静态且为正）

- 计算总迭代数 `trip_iters = ceil_div(stop - start, step)`。`step == 1` 时退化为 `stop - start`，Pass 直接发射简化形式。
- 令 `main_iters = trip_iters / factor`（向下取整），并把 `main_end = start + main_iters * (factor * step)` 以 `AssignStmt` 绑定为 SSA 变量 `unroll_main_end`。
- 主循环 `for i in range(start, main_end, F*step)`。
- 以 SSA 变量 `unroll_rem` 绑定 `rem_iters = trip_iters - main_iters * factor`（`step == 1` 时等价于 `stop - main_end`，Pass 直接发射该简化形式）。通过级联 IfStmt 根据迭代数分派：

  ```text
  if rem_iters == 1:    <1 份克隆>
  else if rem_iters == 2: <2 份克隆>
  else if rem_iters == 3: <3 份克隆>
  # ...
  else if rem_iters == F-1: <F-1 份克隆>
  # rem_iters == 0 不匹配任何分支，跳过尾部。
  ```

  每个分支 body 为 `k` 份克隆体组成的裸 `SeqStmts`（若源循环存在 `iter_args` 则追加一条 `YieldStmt`）。外层 `IfStmt` 携带 `return_vars`：最外层即原循环的 `return_vars`，内层级联分支使用新鲜变量，通过一系列 `YieldStmt` 向上传递。SSA 依然干净：每个分支自包含，任何条件定义的变量都不会逃出其 IfStmt。

## 约束

| 约束 | 原因 |
| ---- | ---- |
| `step` 必须为编译期整数常量 | 主循环步长及各副本偏移均依赖 `factor * step` 为整数 |
| 动态边界要求 `step > 0` | 动态 trip 计算公式假设正步长；负步长需使用静态边界 |
| `stage=` 与 `chunk=` 在 `pl.pipeline` 中互斥 | 二者优化方向不同，组合使用语义模糊且无明显场景 |
| `stage=` 仅支持 `pl.pipeline()` | 该特性作用域限定于 `pl.pipeline()`；`pl.range()` / `pl.parallel()` / `pl.unroll()` 语义不同 |

## 示例

### 静态 —— 迭代次数已知（`N=10`、`F=4`）

```python
# 变换前
for i in pl.pipeline(0, 10, 1, stage=4):
    tile_x = pl.tile.load(input_a, [i * 128], [128])
    pl.tile.store(tile_x, [i * 128], output)

# 变换后：主循环覆盖 [0, 8)，kind=Pipeline（标记位）、属性下调为 stage=1；
# 尾部克隆直接扁平化到外层作用域
for i in pl.pipeline(0, 8, 4, stage=1):
    tile_x_0 = pl.tile.load(input_a, [i * 128], [128]); pl.tile.store(tile_x_0, [i * 128], output)
    tile_x_1 = pl.tile.load(input_a, [(i + 1) * 128], [128]); pl.tile.store(tile_x_1, [(i + 1) * 128], output)
    tile_x_2 = pl.tile.load(input_a, [(i + 2) * 128], [128]); pl.tile.store(tile_x_2, [(i + 2) * 128], output)
    tile_x_3 = pl.tile.load(input_a, [(i + 3) * 128], [128]); pl.tile.store(tile_x_3, [(i + 3) * 128], output)

tile_x_4 = pl.tile.load(input_a, [8 * 128], [128]); pl.tile.store(tile_x_4, [8 * 128], output)
tile_x_5 = pl.tile.load(input_a, [9 * 128], [128]); pl.tile.store(tile_x_5, [9 * 128], output)
```

### 动态 —— 运行时 `n`

```python
# 变换前
for i in pl.pipeline(0, n, 1, stage=4):
    tile_x = pl.tile.load(input_a, [i * 128], [128])
    pl.tile.store(tile_x, [i * 128], output)

# 变换后
unroll_main_end: pl.Scalar[pl.INDEX] = ((n - 0) // 4) * 4 + 0
for i in pl.pipeline(0, unroll_main_end, 4, stage=1):  # 降级后的标记位
    <4 份克隆体，与静态示例相同>

unroll_rem: pl.Scalar[pl.INDEX] = n - unroll_main_end
if unroll_rem == 1:
    tile_x_t0 = pl.tile.load(input_a, [unroll_main_end * 128], [128])
    pl.tile.store(tile_x_t0, [unroll_main_end * 128], output)
else:
    if unroll_rem == 2:
        <偏移 unroll_main_end + 0、+1 的 2 份克隆体>
    else:
        if unroll_rem == 3:
            <偏移 unroll_main_end + 0、+1、+2 的 3 份克隆体>
```

本 Pass 之后，`CanonicalizeIOOrder` 作用于全程序的每一个 `SeqStmts`，将 load 上拉、store 下沉，使各副本的输入 tile 同时活跃，从而 `MemoryReuse` 不能合并它们。主循环与尾部克隆都能从 ping-pong 缓冲中受益。

## 相关

- [`CanonicalizeIOOrder`](24-canonicalize_io_order.md) —— 下一个 Pass，对 `ForKind::Pipeline` 作用域内的 `SeqStmts` 做 IO 顺序规范化
- [`UnrollLoops`](02-unroll_loops.md) —— slot #1 的全展开 Pass，仍是 `pl.unroll(N)` 的主要降级路径
