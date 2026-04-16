# PartialUnrollTileLoops Pass

在 tile 层级展开 `pl.range(N, unroll=F)` 循环：将循环体复制 `F` 份以启用 ping-pong 缓冲，同时保留外层顺序循环。

## 概述

`pl.unroll(N)` 在 SSA 之前的 slot #1 完整展开循环为 `N` 份副本。用户使用它通常并非需要 `N` 份副本，而是希望获得不同的 tile MemRef —— 否则 `MemoryReuse` 会把生命周期相邻的 tile 合并为同一缓冲区，导致 ping-pong 失效。

`PartialUnrollTileLoops` 提供更精细的开关：在 tile 层级把循环体复制 `F` 份（典型值 2–4），保留外层 `N/F` 次顺序迭代。每个副本获得独立的定义变量（保持 SSA），各自操作独立的 tile，下游 `MemoryReuse` 无法将其合并。

**前置条件**: SSAForm、SplitIncoreOrch、IncoreTileOps、TileOps2D、TileMemoryInferred、NormalizedStmtStructure。

**流水线位置**: 位于 `NormalizeReturnOrder` 之后、`InitMemRef` 之前（slot 20.5）。此时 tile 结构决策已完成；同时早于 `InitMemRef`/`MemoryReuse`，使其看到每个副本独立的 tile 变量。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::PartialUnrollTileLoops()` | `passes.partial_unroll_tile_loops()` | 函数级 |

```python
from pypto import passes
result = passes.partial_unroll_tile_loops()(program)
```

## DSL 语法

```python
# 每次外层迭代复制循环体 4 次；外层循环 16 次，步长为 4。
for i in pl.range(64, unroll=4):
    tile_x = pl.tile.load(input_a, [i * 128], [128])
    pl.tile.store(tile_x, [i * 128], output)
```

## 行为

对于 `attrs_["unroll_factor"] = F` 的循环：

- **主循环**：步长为 `F*step`，循环体为 `F` 份副本组成的 `SeqStmts`；外层循环带 `attrs_["unroll_replicated"] = F` 标记。
- **克隆细节**：每份副本通过 `DeepClone(body, {loop_var → new_var + k * step}, clone_def_vars=true)` 生成。每个副本拥有新鲜的定义变量，既保持 SSA，又给 `MemoryReuse` 提供独立的 tile 身份。

根据 `start` / `stop` 是否为编译期常量，分为两种降级模式，区别仅在主循环的 `stop` 与余数处理方式。

### 静态边界 —— `start`、`stop`、`step` 均为编译期整数

迭代次数 `T = (stop - start) / step`：

- 主循环终点为 `start + (T // F) * F * step`。
- 若 `T % F != 0`，再发射一个**尾部分支**：trip-1 `ForStmt`（标记 `unroll_replicated = T % F`），内含 `T % F` 份克隆体，偏移为 `start + (T // F) * F * step + j * step`，`j ∈ [0, T%F)`。余数已知，不需要运行时分派。

### 动态边界 —— `start` / `stop` 为运行时 Expr（`step` 仍为静态且为正）

- 计算总迭代数 `trip_iters = ceil_div(stop - start, step)`。`step == 1` 时退化为 `stop - start`，Pass 直接发射简化形式。
- 令 `main_iters = trip_iters / factor`（向下取整），并把 `main_end = start + main_iters * (factor * step)` 以 `AssignStmt` 绑定为 SSA 变量 `unroll_main_end`。
- 主循环 `for i in range(start, main_end, F*step)`。
- 以 SSA 变量 `unroll_rem` 绑定 `rem_iters = trip_iters - main_iters * factor`（`step == 1` 时等价于 `stop - main_end`，Pass 直接发射该简化形式）。通过级联 IfStmt 根据迭代数分派：

  ```text
  if rem_iters == 1:    <1 份克隆>                         # 最外层
  else if rem_iters == 2: <2 份克隆，unroll_replicated=2>
  else if rem_iters == 3: <3 份克隆，unroll_replicated=3>
  # ...
  else if rem_iters == F-1: <F-1 份克隆>
  # rem_iters == 0 不匹配任何分支，跳过尾部。
  ```

  每个分支的 body 为一个 trip-1 `ForStmt`，带 `unroll_replicated = k` 标记，因此 `ReorderUnrolledIO` 以与主循环相同的方式对每个分支内部进行重排。SSA 依然干净：每个分支自包含，任何条件定义的变量都不会逃出其 IfStmt。

## 约束

| 约束 | 原因 |
| ---- | ---- |
| `step` 必须为编译期整数常量 | 主循环步长及各副本偏移均依赖 `factor * step` 为整数 |
| 动态边界要求 `step > 0` | 动态 trip 计算公式假设正步长；负步长需使用静态边界 |
| `unroll` 与 `chunk` 在 `pl.range` 中互斥 | 二者优化方向不同，组合使用语义模糊且无明显场景 |
| `unroll=` 仅支持 `pl.range()` | 该特性作用域限定于 `pl.range()`；`pl.parallel()` / `pl.unroll()` 语义不同 |

### 循环携带状态（`iter_args`）

支持 `iter_args` / `init_values`。循环携带状态按顺序穿过 `F` 个副本：第 `k` 个副本以上一副本的 `YieldStmt` 表达式作为其 iter-arg 的替换值，仅最后一个副本保留 `YieldStmt` 以传递给主循环下一次外层迭代。若存在尾部分支，主循环的 `return_vars` 使用新的 SSA 名字，作为尾部分支 `iter_args` 的初值；尾部分支则继承原外层循环的 `return_vars`，确保下游引用有效。

动态边界下，级联中的每个 `IfStmt` 都携带与 iter-arg 类型一致的 `return_vars`，每个分支均以 `YieldStmt` 结尾。最内层 `else` 直接 yield 主循环的 `return_vars` —— 即 `rem == 0` 时的空操作回退。

## 示例

### 静态 —— 迭代次数已知（`N=10`、`F=4`）

```python
# 变换前
for i in pl.range(0, 10, 1, attrs={"unroll_factor": 4}):
    tile_x = pl.tile.load(input_a, [i * 128], [128])
    pl.tile.store(tile_x, [i * 128], output)

# 变换后：主循环覆盖 [0, 8)，单个尾部分支处理剩余 2 次迭代
for i in pl.range(0, 8, 4, attrs={"unroll_replicated": 4}):
    tile_x_0 = pl.tile.load(input_a, [i * 128], [128]); pl.tile.store(tile_x_0, [i * 128], output)
    tile_x_1 = pl.tile.load(input_a, [(i + 1) * 128], [128]); pl.tile.store(tile_x_1, [(i + 1) * 128], output)
    tile_x_2 = pl.tile.load(input_a, [(i + 2) * 128], [128]); pl.tile.store(tile_x_2, [(i + 2) * 128], output)
    tile_x_3 = pl.tile.load(input_a, [(i + 3) * 128], [128]); pl.tile.store(tile_x_3, [(i + 3) * 128], output)

for _tail_iter_2 in pl.range(0, 1, 1, attrs={"unroll_replicated": 2}):
    tile_x_4 = pl.tile.load(input_a, [8 * 128], [128]); pl.tile.store(tile_x_4, [8 * 128], output)
    tile_x_5 = pl.tile.load(input_a, [9 * 128], [128]); pl.tile.store(tile_x_5, [9 * 128], output)
```

### 静态带 `iter_args` —— 循环累加（`N=10`、`F=4`）

```python
# 变换前
for i, (a,) in pl.range(0, 10, 1, init_values=(s0,), attrs={"unroll_factor": 4}):
    b = a + i
    r = pl.yield_(b)

# 变换后：主循环按 a → b → b_1 → b_2 → b_3 串接，尾部分支以 r_main 作为 a_tail 的初值
for i, (a,) in pl.range(0, 8, 4, init_values=(s0,), attrs={"unroll_replicated": 4}):
    b = a + i
    b_1 = b + (i + 1)
    b_2 = b_1 + (i + 2)
    b_3 = b_2 + (i + 3)
    r_main = pl.yield_(b_3)

for _tail_iter_2, (a,) in pl.range(0, 1, 1, init_values=(r_main,), attrs={"unroll_replicated": 2}):
    b_4 = a + 8
    b_5 = b_4 + 9
    r = pl.yield_(b_5)
```

### 动态 —— 运行时 `n`

```python
# 变换前
for i in pl.range(0, n, 1, attrs={"unroll_factor": 4}):
    tile_x = pl.tile.load(input_a, [i * 128], [128])
    pl.tile.store(tile_x, [i * 128], output)

# 变换后
unroll_main_end: pl.Scalar[pl.INDEX] = ((n - 0) // 4) * 4 + 0
for i in pl.range(0, unroll_main_end, 4, attrs={"unroll_replicated": 4}):
    <4 份克隆体，与静态示例相同>

unroll_rem: pl.Scalar[pl.INDEX] = n - unroll_main_end
if unroll_rem == 1:
    for _tail_iter_1 in pl.range(0, 1, 1, attrs={"unroll_replicated": 1}):
        tile_x_t0 = pl.tile.load(input_a, [unroll_main_end * 128], [128])
        pl.tile.store(tile_x_t0, [unroll_main_end * 128], output)
else:
    if unroll_rem == 2:
        for _tail_iter_2 in pl.range(0, 1, 1, attrs={"unroll_replicated": 2}):
            <偏移 unroll_main_end + 0、+1 的 2 份克隆体>
    else:
        if unroll_rem == 3:
            for _tail_iter_3 in pl.range(0, 1, 1, attrs={"unroll_replicated": 3}):
                <偏移 unroll_main_end + 0、+1、+2 的 3 份克隆体>
```

主循环与每个尾部分支都带 `unroll_replicated` 标记，`ReorderUnrolledIO` 以一致方式将 load 上拉、store 下沉，使各副本的输入 tile 同时活跃，从而 `MemoryReuse` 不能合并它们。主干与尾部都能从 ping-pong 缓冲中受益。

## 相关

- [`ReorderUnrolledIO`](21-reorder_unrolled_io.md) —— 消费 `unroll_replicated` 标记
- [`UnrollLoops`](01-unroll_loops.md) —— slot #1 的全展开 Pass，仍是 `pl.unroll(N)` 的主要降级路径
- RFC #1025 —— 设计文档
