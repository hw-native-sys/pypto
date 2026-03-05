# SplitChunkedLoops Pass

将带有 `chunk` 的循环拆分为嵌套的外层/内层循环，实现分块迭代。

## 概述

此 Pass 将使用 `chunk=C` 创建的 for 循环转换为嵌套循环：外层循环遍历分块索引，内层循环在每个分块内迭代。当迭代总次数不能被分块大小整除时，会附加一个余数循环 (remainder loop)。在 SSA 转换之后运行，将 `iter_args` 传播到生成的嵌套循环中。

**前置条件**: TypeChecked、SSAForm 属性。

**使用时机**: 在默认流水线中自动运行，位于 `FlattenCallExpr` 之后、`RunVerifier` 之前。在 `with pl.auto_incore():` 作用域内的 `pl.range()`、`pl.parallel()` 或 `pl.unroll()` 上使用 `chunk=` 来将循环拆分为分块。`auto_incore` 之外的分块循环不会被拆分。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::SplitChunkedLoops()` | `passes.split_chunked_loops()` | 函数级 |

**Python 用法**:

```python
from pypto import passes

result = passes.split_chunked_loops()(program)
```

## DSL 语法

分块循环必须包裹在 `with pl.auto_incore():` 中才会被拆分：

```python
with pl.auto_incore():
    # 分块顺序循环：10 次迭代，每块 5 次
    for i in pl.range(10, chunk=5):
        x = pl.add(x, 1.0)

    # 分块并行循环：内层循环并行，外层顺序
    for i in pl.parallel(8, chunk=4):
        x = pl.add(x, 1.0)

    # 分块展开循环：内层循环展开，外层顺序
    for i in pl.unroll(12, chunk=4):
        x = pl.add(x, 1.0)
```

`auto_incore` 之外的分块循环保持原样（不被拆分）。

## 约束

| 约束 | 原因 |
| ---- | ---- |
| `start`、`stop`、`step`、`chunk` 必须为整数常量 | 编译时需要确定值 |
| `chunk` 必须为正整数 | 非正数的分块大小无效 |
| DSL 中 `chunk` 不能与 `init_values` 一起使用 | 分块循环不支持用户指定的 iter_args |
| 分块循环必须在 `pl.auto_incore()` 内 | 仅 `auto_incore` 作用域内的循环会被拆分 |

## 算法

给定 SSA 形式的分块循环：

```text
for i, (x_iter=x_0,) in range(start, stop, step, chunk=C) -> (x_rv,):
    x_1 = add(x_iter, 1.0)
    yield(x_1)
```

1. 计算 `trip_count = ceil((stop - start) / step)`
2. `num_full_chunks = trip_count // C`，`remainder = trip_count % C`
3. 生成外层循环，iter_args 从原始初始值初始化
4. 生成内层循环，iter_args 从外层 iter_args 提供，循环体替换：`i = start + (i_out * C + i_in) * step`
5. 内层循环 yield 到内层 return_vars；外层循环 yield 内层 return_vars 到外层 return_vars
6. 若 `remainder > 0`，生成余数循环，iter_args 从外层 return_vars 链接
7. 将原始 return_vars 重映射到最终循环的 return_vars

内层循环保留原始的 `ForKind`（Sequential、Parallel 或 Unroll），外层循环始终为 Sequential。

## 示例

**之前**（打印的 SSA IR 形式；非合法 DSL 源码，因为 DSL 中 `chunk` + `init_values` 不能同时使用）：

```python
@pl.program
class Before:
    @pl.function
    def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        for i_0, (x_iter_1,) in pl.range(10, init_values=(x_0,), chunk=5):
            x_3 = pl.tensor.add(x_iter_1, 1.0)
            x_2 = pl.yield_(x_3)
        return x_2
```

**之后**（嵌套循环，整除情况）：

```python
@pl.program
class After:
    @pl.function
    def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        for i_0_out, (x_iter_1_outer,) in pl.range(2, init_values=(x_0,)):
            for i_0_in, (x_iter_1_inner,) in pl.range(5, init_values=(x_iter_1_outer,)):
                x_3 = pl.tensor.add(x_iter_1_inner, 1.0)
                x_iter_1_inner_rv = pl.yield_(x_3)
            x_iter_1_outer_rv = pl.yield_(x_iter_1_inner_rv)
        return x_iter_1_outer_rv
```

**带余数**（`chunk=5`，trip_count=7）：

```python
# 生成：outer(0,1) * inner(0,5) + remainder(0,2)
for i_0_out, (x_iter_1_outer,) in pl.range(1, init_values=(x_0,)):
    for i_0_in, (x_iter_1_inner,) in pl.range(5, init_values=(x_iter_1_outer,)):
        x_3 = pl.tensor.add(x_iter_1_inner, 1.0)
        x_iter_1_inner_rv = pl.yield_(x_3)
    x_iter_1_outer_rv = pl.yield_(x_iter_1_inner_rv)
for i_0_rem, (x_iter_1_rem,) in pl.range(2, init_values=(x_iter_1_outer_rv,)):
    x_3 = pl.tensor.add(x_iter_1_rem, 1.0)
    x_iter_1_rem_rv = pl.yield_(x_3)
return x_iter_1_rem_rv
```

## LoopOrigin 标记

每个生成的循环都带有 `LoopOrigin` 注解，标识其产生方式：

| LoopOrigin | 说明 |
| ---------- | ---- |
| `Original` | 普通循环（默认，非拆分生成） |
| `ChunkOuter` | 遍历分块索引的外层循环 |
| `ChunkInner` | 在分块内迭代的内层循环 |
| `ChunkRemainder` | 处理剩余迭代的余数循环 |

通过 `for_stmt.loop_origin`（Python）或 `for_stmt->loop_origin_`（C++）访问。下游 Pass 可使用此标记区分生成的循环与用户编写的循环。

## 流水线位置

```text
UnrollLoops → ConvertToSSA → FlattenCallExpr → SplitChunkedLoops → RunVerifier → ...
```

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| Required | `TypeChecked`、`SSAForm` |
| Produced | `TypeChecked`、`SSAForm` |
| Invalidated | （无） |
