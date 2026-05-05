# UnrollLoops Pass

在编译时展开 `ForKind::Unroll` 循环，将循环体内联到每次迭代中。

## 概述

此 Pass 静态展开通过 `pl.unroll()` 创建的 for 循环，将其替换为循环体的多份副本，其中循环变量被替换为每次迭代的常量值。

**前置条件**: TypeChecked 属性。

**使用场景**: 在默认流水线（pipeline）中自动运行，位于 `ConvertToSSA` 之前。当循环次数为编译时常量并且您希望为每次迭代复制循环体时，使用 `pl.unroll()`。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::UnrollLoops()` | `passes.unroll_loops()` | 函数级（Function-level） |

**Python 用法**:

```python
from pypto import passes

# 创建并应用 Pass
result = passes.unroll_loops()(program)
```

## DSL 语法

```python
# 基本展开：循环体复制 4 次，i = 0, 1, 2, 3
for i in pl.unroll(4):
    x = pl.add(x, i)

# 带 start/stop/step：循环体复制，i = 0, 2, 4
for i in pl.unroll(0, 6, 2):
    x = pl.add(x, i)
```

## 约束条件

| 约束 | 原因 |
| ---- | ---- |
| `start`、`stop`、`step` 必须为整数常量 | 编译时需要确定值 |
| `step` 不能为零 | 防止无限循环 |
| `init_values` 不能与 `pl.unroll()` 一起使用 | 展开的循环不支持循环携带状态 |

## 示例

**展开前**（包含展开循环）:

```python
@pl.program
class Before:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        for i in pl.unroll(3):
            x = pl.add(x, i)
        return x
```

**展开后**（循环已展开）:

```python
@pl.program
class After:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        x = pl.add(x, 0)
        x = pl.add(x, 1)
        x = pl.add(x, 2)
        return x
```

## 流水线位置

UnrollLoops 在 `Default` 和 `DebugTileOptimization` 中都只**运行一次**，位于控制流结构化之前：

```text
UnrollLoops → CtrlFlowTransform → ConvertToSSA → FlattenCallExpr → SplitChunkedLoops → InterchangeChunkLoops → OutlineIncoreScopes → ...
```

UnrollLoops 展开非分块的 `pl.unroll()` 循环（跳过分块展开循环，保留 `chunk` 供后续 `SplitChunkedLoops` 处理）。

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| 前置要求（Required） | `TypeChecked` |
| 产生（Produced） | `TypeChecked` |
| 失效（Invalidated） | （无） |
