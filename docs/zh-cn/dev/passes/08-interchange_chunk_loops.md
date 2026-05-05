# InterchangeChunkLoops Pass

重新排列嵌套的 ChunkOuter/ChunkInner 循环对并插入 `InCore` 作用域，为下游提取做准备。

## 概述

在 `SplitChunkedLoops` 将分块循环拆分为嵌套的 `ChunkOuter→ChunkInner` 对之后，嵌套分块循环的结构为：

```text
i_out[ChunkOuter] → i_in[ChunkInner,Parallel] → j_out[ChunkOuter] → j_in[ChunkInner,Parallel] → body
```

此 Pass 重新排列，使所有外层循环在顶部，并将内层循环 + 循环体包裹在 `InCoreScopeStmt` 中：

```text
i_out[ChunkOuter] → j_out[ChunkOuter] → InCore{ i_in[ChunkInner] → j_in[ChunkInner] → body }
```

**前置条件**: TypeChecked、SSAForm 属性。

**使用时机**: 在默认流水线中自动运行，位于 `SplitChunkedLoops` 之后、`OutlineIncoreScopes` 之前。仅处理 `pl.auto_incore()` 作用域内的循环。此 Pass 会消费（移除）`AutoInCore` 作用域。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::InterchangeChunkLoops()` | `passes.interchange_chunk_loops()` | 函数级 |

**Python 用法**:

```python
from pypto import passes

result = passes.interchange_chunk_loops()(program)
```

## 约束

| 约束 | 行为 |
| ---- | ---- |
| 仅 SSA | 在 `SplitChunkedLoops` 之后运行（需要 `SSAForm`） |
| 仅并行交换 | 仅当所有 ChunkInner 循环具有 `ForKind::Parallel` 时才交换 |
| 顺序分块循环 | 不交换，但如果在 `auto_incore` 内则包裹在 InCore 中 |
| 已有 InCore | 如果链体已包含 `InCoreScopeStmt`，则跳过 |
| 需要 `auto_incore` 作用域 | 仅处理 `AutoInCoreScopeStmt` 内的循环；该作用域会被消费 |

## 算法

1. **收集链** — 从 `ChunkOuter` ForStmt 开始，遍历嵌套的 ForStmt 体。构建 `(ForStmt, LoopOrigin)` 条目列表。在遇到非 ForStmt、`Original` 循环或 `ScopeStmt` 时停止。

2. **守卫检查** — 验证所有 ChunkInner 循环为 Parallel。检查最内层循环体中无已有 InCore 作用域。

3. **分离** — 将链分为 `outers`（ChunkOuter）和 `inners`（ChunkInner）。

4. **重建**（由内到外构建）：
   - 访问最内层循环体
   - 将 inners 包裹在循环体外（保持顺序），重新连接 iter_args
   - 包裹在 `InCoreScopeStmt` 中
   - 将 outers 包裹在 InCore 外（保持顺序），重新连接 iter_args 和 yields

5. **处理余数** — `ChunkRemainder` 循环：递归进入循环体。将独立的并行余数子循环包裹在 InCore 中。

## 自动命名缩写

下面示例里的变量名使用了 `base__qualifier_role_vN` 这一紧凑格式，其中 qualifier 有若干缩写：

| 缩写 | 含义 |
| ---- | ---- |
| `co` | `chunk_outer` |
| `ci` | `chunk_inner` |
| `cr` | `chunk_rem` / 余数分块 |
| `lN` | interchange 之后的第 `N` 层循环 |

示例：

- `x__co_iter_v1`：交换前的外层分块 iter_arg
- `x__co_l0_iter_v1`：交换后第 0 层循环上传递的 iter_arg
- `x__co_l2_rv_v1`：从重排后第 2 层循环流出的 return var

像 `iter`、`rv`、`idx`、`ssa` 这样的 role 不再继续缩写，以便变量用途仍然一眼可见。

## 示例

**之前**（SplitChunkedLoops 之后，全并行）：

```python
for i__co_idx_v0, (x__co_iter_v1,) in pl.range(2, init_values=(x__ssa_v0,)):  # ChunkOuter
    for i__ci_idx_v0, (x__ci_iter_v1,) in pl.parallel(
        4, init_values=(x__co_iter_v1,)
    ):  # ChunkInner
        for j__co_idx_v0, (y__co_iter_v1,) in pl.range(
            3, init_values=(x__ci_iter_v1,)
        ):  # ChunkOuter
            for j__ci_idx_v0, (y__ci_iter_v1,) in pl.parallel(
                4, init_values=(y__co_iter_v1,)
            ):  # ChunkInner
                z = pl.add(y__ci_iter_v1, 1.0)
                y__ci_rv_v1 = pl.yield_(z)
            y__co_rv_v1 = pl.yield_(y__ci_rv_v1)
        x__ci_rv_v1 = pl.yield_(y__co_rv_v1)
    x__co_rv_v1 = pl.yield_(x__ci_rv_v1)
return x__co_rv_v1
```

**之后**（InterchangeChunkLoops）：

```python
for i__co_idx_v0, (x__co_l0_iter_v1,) in pl.range(
    2, init_values=(x__ssa_v0,)
):  # ChunkOuter
    for j__co_idx_v0, (x__co_l1_iter_v1,) in pl.range(
        3, init_values=(x__co_l0_iter_v1,)
    ):  # ChunkOuter
        with pl.incore():                                               # 插入 InCore
            for i__ci_idx_v0, (x__co_l2_iter_v1,) in pl.parallel(
                4, init_values=(x__co_l1_iter_v1,)
            ):  # ChunkInner
                for j__ci_idx_v0, (x__co_l3_iter_v1,) in pl.parallel(
                    4, init_values=(x__co_l2_iter_v1,)
                ):  # ChunkInner
                    z = pl.add(x__co_l3_iter_v1, 1.0)
                    x__co_l3_rv_v1 = pl.yield_(z)
                x__co_l2_rv_v1 = pl.yield_(x__co_l3_rv_v1)
        x__co_l1_rv_v1 = pl.yield_(x__co_l2_rv_v1)
    x__co_l0_rv_v1 = pl.yield_(x__co_l1_rv_v1)
return x__co_l0_rv_v1
```

## 余数处理

对于不整除的迭代次数，余数循环会被包裹在 InCore 中：

```python
for i_rem, (...) in pl.parallel(2, init_values=(...)):   # ChunkRemainder
    for j_out, (...) in pl.range(3, init_values=(...)):   # 已应用交换
        with pl.incore():
            for j_in, (...) in pl.parallel(4, init_values=(...)):
                body
    with pl.incore():                                            # 余数已包裹
        for j_rem, (...) in pl.parallel(2, init_values=(...)):
            body
```

## 非分块语句处理

当 `auto_incore` 被消费时，未被分块交换处理的语句（独立张量算子、非分块循环、未通过并行守卫检查的顺序分块循环）会被包裹在 `InCoreScopeStmt` 中，以确保它们被 `OutlineIncoreScopes` 提取到 InCore 函数中。

连续的非 InCore 语句会被分组到单个 `InCoreScopeStmt` 中。控制流语句（`YieldStmt`、`ReturnStmt`）和纯标量赋值（例如索引运算 `offset = ob * 32`）不会被包裹——它们留在编排作用域中。

**示例** — 独立算子 + 并行分块：

```python
# 之前（在 auto_incore 内部，SplitChunkedLoops 之后）
with pl.auto_incore():
    x = pl.add(x, 1.0)                           # 独立算子
    for i_out in pl.range(2):                     # ChunkOuter（并行内层）
        for i_in in pl.parallel(4):
            x = pl.add(x, 2.0)

# InterchangeChunkLoops 之后
with pl.incore():                                 # 独立算子已包裹
    x = pl.add(x, 1.0)
for i_out in pl.range(2):                         # 已交换的分块
    with pl.incore():
        for i_in in pl.parallel(4):
            x = pl.add(x, 2.0)
```

**示例** — 顺序分块（未通过交换守卫检查）：

```python
# 之前
with pl.auto_incore():
    for i_out in pl.range(2):                     # ChunkOuter（顺序内层）
        for i_in in pl.range(4):                  # ChunkInner，Sequential → 未通过守卫
            x = pl.add(x, 1.0)

# 之后 — 整个链被包裹在 InCore 中
with pl.incore():
    for i_out in pl.range(2):
        for i_in in pl.range(4):
            x = pl.add(x, 1.0)
```

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
