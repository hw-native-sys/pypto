# InterchangeChunkLoops Pass

重新排列嵌套的 ChunkOuter/ChunkInner 循环对并插入 `InCore` 作用域，为下游提取做准备。

## 概述

在 `SplitChunkedLoops` 将分块循环拆分为嵌套的 `ChunkOuter→ChunkInner` 对之后，嵌套分块循环的结构为：

```text
i_out[ChunkOuter] → i_in[ChunkInner,Parallel] → j_out[ChunkOuter] → j_in[ChunkInner,Parallel] → body
```

此 Pass 重新排列，使所有外层循环在顶部，并将内层循环 + 循环体包裹在 `ScopeStmt(InCore)` 中：

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
| 顺序分块循环 | 保持原样（不交换，不插入 InCore） |
| 已有 InCore | 如果链体已包含 `ScopeStmt(InCore)`，则跳过 |
| 需要 `auto_incore` 作用域 | 仅处理 `ScopeStmt(AutoInCore)` 内的循环；该作用域会被消费 |

## 算法

1. **收集链** — 从 `ChunkOuter` ForStmt 开始，遍历嵌套的 ForStmt 体。构建 `(ForStmt, LoopOrigin)` 条目列表。在遇到非 ForStmt、`Original` 循环或 `ScopeStmt` 时停止。

2. **守卫检查** — 验证所有 ChunkInner 循环为 Parallel。检查最内层循环体中无已有 InCore 作用域。

3. **分离** — 将链分为 `outers`（ChunkOuter）和 `inners`（ChunkInner）。

4. **重建**（由内到外构建）：
   - 访问最内层循环体
   - 将 inners 包裹在循环体外（保持顺序），重新连接 iter_args
   - 包裹在 `ScopeStmt(ScopeKind::InCore)` 中
   - 将 outers 包裹在 InCore 外（保持顺序），重新连接 iter_args 和 yields

5. **处理余数** — `ChunkRemainder` 循环：递归进入循环体。将独立的并行余数子循环包裹在 InCore 中。

## 示例

**之前**（SplitChunkedLoops 之后，全并行）：

```python
for i_out, (x_outer,) in pl.range(2, init_values=(x_0,)):        # ChunkOuter
    for i_in, (x_ia,) in pl.parallel(4, init_values=(x_outer,)):   # ChunkInner
        for j_out, (y_outer,) in pl.range(3, init_values=(x_ia,)):  # ChunkOuter
            for j_in, (y_ia,) in pl.parallel(4, init_values=(y_outer,)):  # ChunkInner
                z = pl.add(y_ia, 1.0)
                y_ia_rv = pl.yield_(z)
            y_outer_rv = pl.yield_(y_ia_rv)
        x_ia_rv = pl.yield_(y_outer_rv)
    x_outer_rv = pl.yield_(x_ia_rv)
return x_outer_rv
```

**之后**（InterchangeChunkLoops）：

```python
for i_out, (x_l0,) in pl.range(2, init_values=(x_0,)):        # ChunkOuter
    for j_out, (x_l1,) in pl.range(3, init_values=(x_l0,)):    # ChunkOuter
        with pl.incore():                                               # 插入 InCore
            for i_in, (x_l2,) in pl.parallel(4, init_values=(x_l1,)):  # ChunkInner
                for j_in, (x_l3,) in pl.parallel(4, init_values=(x_l2,)):  # ChunkInner
                    z = pl.add(x_l3, 1.0)
                    x_l3_rv = pl.yield_(z)
                x_l2_rv = pl.yield_(x_l3_rv)
        x_l1_rv = pl.yield_(x_l2_rv)
    x_l0_rv = pl.yield_(x_l1_rv)
return x_l0_rv
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
