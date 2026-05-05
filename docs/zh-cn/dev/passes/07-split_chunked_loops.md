# SplitChunkedLoops Pass

将带有 `chunk` 的循环按两种策略之一拆分为嵌套的外层/内层循环。

## 概述

此 Pass 将使用 `chunk=C` 创建的 for 循环转换为嵌套循环：外层循环遍历分块索引，内层循环在每个分块内迭代。支持两种生成策略：

- **`guarded`**（默认）— 发射一个长度为 `ceil(T/C)` 的外层循环和一个长度为 `C` 的内层循环，并用 `if (idx < stop)`（负步长时为 `idx > stop`）包裹循环体。越界迭代变为空操作。只发射一个 kernel。
- **`leading_full`** — 发射一个长度为 `T/C` 的满块循环加一个长度为 `T % C` 的独立余数循环。发射两个并列循环。

两种策略都在 SSA 转换之后运行，并将 `iter_args` 传播到生成的循环中。

**前置条件**: `TypeChecked`、`SSAForm`。

**使用时机**: 在默认流水线中自动运行，位于 `FlattenCallExpr` 之后、`InterchangeChunkLoops` 之前。在 `with pl.auto_incore():` 作用域内的 `pl.range()`、`pl.parallel()`、`pl.unroll()` 上使用 `chunk=`。`auto_incore` 之外的分块循环不会被拆分。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::SplitChunkedLoops()` | `passes.split_chunked_loops()` | 函数级 |

```python
from pypto import passes
result = passes.split_chunked_loops()(program)
```

## DSL 语法

分块循环必须包裹在 `with pl.auto_incore():` 中：

```python
with pl.auto_incore():
    # 默认 (guarded)：单 kernel + if-guard
    for i in pl.range(10, chunk=5):
        x = pl.add(x, 1.0)

    # 显式 guarded（与默认等价）
    for i in pl.parallel(n, chunk=4, chunk_policy="guarded"):
        x = pl.add(x, 1.0)

    # 显式 leading_full：余数剥离为独立循环
    for i in pl.range(7, chunk=5, chunk_policy="leading_full"):
        x = pl.add(x, 1.0)

    # 两种策略都支持 iter_args
    for i, (s,) in pl.range(10, init_values=(x,), chunk=5):
        s = pl.add(s, 1.0)
        s = pl.yield_(s)
```

## 策略选择

| 场景 | 偏好 `guarded` | 偏好 `leading_full` |
| ---- | -------------- | ------------------- |
| 动态 bound（`stop` 非编译期常量） | ✅ —— 单 kernel 保留跨边界的 loop-carried 状态 | ❌ —— 余数 kernel 的 iter_args 只能以 input-only 拷贝方式传入，破坏跨迭代累积 |
| 静态 bound 且可整除 | guard 稍显冗余 | ✅ —— 无 guard、无余数 |
| 希望 `pl.auto_incore()` 下 kernel 数量最少 | ✅ | 每个分块循环会生成 2 个 kernel |
| 希望热点循环内部不存在掩码迭代 | ❌ | ✅ —— 满块无条件执行 |

`guarded` 被设为默认，原因在于：(1) 动态 bound 下能保留 `add_inout()` 累积；(2) 避免 `pl.auto_incore()` 下 kernel 数量翻倍。

## 约束

| 约束 | 原因 |
| ---- | ---- |
| `step`、`chunk` 必须为整数常量 | 编译期需要确定值 |
| `chunk` 必须为正整数 | 非正数的分块大小无效 |
| `step` 可以为负（下降循环） | `guarded` 会根据步长符号选择判据 |
| `start`、`stop` 在 `guarded` 下可以是动态表达式 | 迭代次数取 `max(abs(stop - start), 0) / abs(step)` |
| 分块循环必须在 `pl.auto_incore()` 内 | 仅 `auto_incore` 作用域内的循环会被拆分 |
| `chunk` 可以与 `init_values` 同时使用 | 两种策略都会将 iter_args 串联到生成的循环 |

## 算法

记 `T = ceil(max(|stop - start|, 0) / |step|)`，`C = chunk`。

### `guarded`（默认）

1. `n_total = ceil(T / C)`。静态 bound 直接计算，动态 bound 用 `(T + C - 1) // C`。
2. 发射外层循环 `for out_var in [0, n_total)` 与内层循环 `for in_var in [0, C)`。
3. 计算 `idx = start + (out_var * C + in_var) * step`，并替换到循环体里。
4. 将访问后的循环体包裹进 `IfStmt`，条件为：
   - `idx < stop`（当 `step > 0`）
   - `idx > stop`（当 `step < 0`）
5. **无 iter_args** —— IfStmt 无 else 分支；被跳过的迭代为空操作。
6. **有 iter_args** —— IfStmt 的 `return_vars` 作为 phi：then 分支保留用户循环体的末尾 `YieldStmt`（更新后的值），else 分支 yield 未变的 inner iter_args。内层循环的末尾 `YieldStmt` 引用 IfStmt 的 phi 变量，从而在生效与被跳过的迭代之间都能串联循环携带状态。

### `leading_full`

1. `n_full = T // C`，`n_rem = T % C`。
2. 发射外层 `for out_var in [0, n_full)` 与内层 `for in_var in [0, C)`，`idx = start + (out_var * C + in_var) * step`；若 `n_full == 0` 则跳过。
3. 若 `n_rem > 0`，发射余数循环 `for rem_var in [0, n_rem)`，`idx = start + (n_full * C + rem_var) * step`。其 `init_values` 链接自外层循环的 `return_vars`（如果没有满块循环，则链接自原始 init 值）。
4. 将原始 `return_vars` 重映射到最终循环的 `return_vars`。

两种路径都在内层与外层/余数循环上保留原始的 `ForKind`（Sequential、Parallel、Unroll）。

## 自动命名缩写

打印出来的 IR 使用紧凑的自动命名格式 `base__qualifier_role_vN`。缩写 qualifier：

| 缩写 | 含义 | 发射时机 |
| ---- | ---- | -------- |
| `co` | chunk_outer | 两种策略 |
| `ci` | chunk_inner | 两种策略 |
| `cr` | chunk_rem（余数） | 仅 `leading_full` |
| `cg` | chunk_guard（IfStmt phi） | 仅带 iter_args 的 `guarded` |

示例：`i__co_idx_v0`（外层索引）、`x__ci_iter_v1`（内层 iter_arg）、`x__cr_rv_v1`（余数 return var）、`x__cg_rv_v1`（IfStmt phi 变量）。

## 示例

### `guarded`，可整除（`chunk=5`，trip_count=10）

**之后**：

```python
for i__co_idx_v0, (x__co_iter_v1,) in pl.range(2, init_values=(x__ssa_v0,)):
    for i__ci_idx_v0, (x__ci_iter_v1,) in pl.range(5, init_values=(x__co_iter_v1,)):
        if i__co_idx_v0 * 5 + i__ci_idx_v0 < 10:
            x__ssa_v3 = pl.tensor.add(x__ci_iter_v1, 1.0)
            x__cg_rv_v1 = pl.yield_(x__ssa_v3)
        else:
            x__cg_rv_v1 = pl.yield_(x__ci_iter_v1)
        x__ci_rv_v1 = pl.yield_(x__cg_rv_v1)
    x__co_rv_v1 = pl.yield_(x__ci_rv_v1)
return x__co_rv_v1
```

### `guarded`，动态 bound（`chunk=4`，`stop=n`）

**之后**（单 kernel，`n_total = (n + 3) // 4`）：

```python
for i__co_idx_v0, (x__co_iter_v1,) in pl.range((n + 3) // 4, init_values=(x__ssa_v0,)):
    for i__ci_idx_v0, (x__ci_iter_v1,) in pl.range(4, init_values=(x__co_iter_v1,)):
        if i__co_idx_v0 * 4 + i__ci_idx_v0 < n:
            x__ssa_v3 = pl.tensor.add(x__ci_iter_v1, 1.0)
            x__cg_rv_v1 = pl.yield_(x__ssa_v3)
        else:
            x__cg_rv_v1 = pl.yield_(x__ci_iter_v1)
        x__ci_rv_v1 = pl.yield_(x__cg_rv_v1)
    x__co_rv_v1 = pl.yield_(x__ci_rv_v1)
return x__co_rv_v1
```

### `leading_full`，不可整除（`chunk=5`，trip_count=7）

**之后**（两个并列循环）：

```python
for i__co_idx_v0, (x__co_iter_v1,) in pl.range(1, init_values=(x__ssa_v0,)):
    for i__ci_idx_v0, (x__ci_iter_v1,) in pl.range(5, init_values=(x__co_iter_v1,)):
        x__ssa_v3 = pl.tensor.add(x__ci_iter_v1, 1.0)
        x__ci_rv_v1 = pl.yield_(x__ssa_v3)
    x__co_rv_v1 = pl.yield_(x__ci_rv_v1)
for i__cr_idx_v0, (x__cr_iter_v1,) in pl.range(2, init_values=(x__co_rv_v1,)):
    x__ssa_v4 = pl.tensor.add(x__cr_iter_v1, 1.0)
    x__cr_rv_v1 = pl.yield_(x__ssa_v4)
return x__cr_rv_v1
```

## LoopOrigin 标记

| LoopOrigin | 说明 | 发射时机 |
| ---------- | ---- | -------- |
| `Original` | 普通用户循环（默认） | — |
| `ChunkOuter` | 遍历分块索引的外层循环 | 两种策略 |
| `ChunkInner` | 在分块内迭代的内层循环 | 两种策略 |
| `ChunkRemainder` | 处理剩余迭代的余数循环 | 仅 `leading_full` |

通过 `for_stmt.attrs.get("loop_origin")`（Python）或 `for_stmt->GetAttr<LoopOrigin>("loop_origin")`（C++）访问。

## 流水线位置

```text
UnrollLoops → ConvertToSSA → FlattenCallExpr → SplitChunkedLoops → InterchangeChunkLoops → OutlineIncoreScopes → ...
```

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| Required | `TypeChecked`、`SSAForm` |
| Produced | `TypeChecked`、`SSAForm` |
| Invalidated | （无） |
