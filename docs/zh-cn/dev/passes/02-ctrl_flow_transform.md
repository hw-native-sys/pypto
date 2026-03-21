# CtrlFlowTransform Pass（控制流结构化）

将 `break` 和 `continue` 语句转换为等价的结构化控制流（if-else + while 循环），使下游 Pass 和代码生成可以在没有非结构化跳转的 IR 上工作。

## 概述

PTO codegen 生成 MLIR 格式的 SCF（结构化控制流），不直接支持 `break` 和 `continue`。该 Pass 通过将循环重写为等价的结构化形式来消除两者。

**适用范围**: 仅对 InCore 类函数生效（InCore、AIC、AIV）。Orchestration/Host 函数会被跳过，因为它们可以原生支持 `break`/`continue`。

**所需属性**: `TypeChecked`

**产生属性**: `TypeChecked`, `StructuredCtrlFlow`

**使用时机**: 在默认流水线中自动运行，位于 `UnrollLoops` 之后、`ConvertToSSA` 之前。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::CtrlFlowTransform()` | `passes.ctrl_flow_transform()` | Function 级别 |

**Python 用法**:

```python
from pypto.pypto_core import passes

result = passes.ctrl_flow_transform()(program)
```

## 算法

该 Pass 按顺序运行两个阶段：

| 阶段 | 消除目标 | 策略 |
| ---- | -------- | ---- |
| 1 | `continue` | 将剩余循环体重构为 `if-else` |
| 2 | `break` | 将 `ForStmt` 转换为带 break 标志的 `WhileStmt` |

**阶段 1 必须在阶段 2 之前运行**，因为 `continue` 消除保留循环类型（ForStmt/WhileStmt），而 `break` 消除会将 ForStmt 转换为 WhileStmt。先运行阶段 1 使变换更简单。

### 阶段 1：Continue 消除

`continue` 之后的剩余语句被移入 `else` 分支。循环类型保持不变。

**变换前**:

```python
for i in pl.range(n):
    A
    if cond:
        continue
    B
    C
```

**变换后**:

```python
for i in pl.range(n):
    A
    if cond:
        pass  # nothing
    else:
        B
        C
```

WhileStmt 的处理方式相同。多个 `continue` 语句通过重复应用处理（从最内层开始）。

### 阶段 2：Break 消除

包含 `break` 的 `ForStmt` 被转换为带有辅助 break 标志变量的 `WhileStmt`。

**变换前**:

```python
for i in pl.range(start, stop, step):
    A
    if cond:
        break
    B
    C
```

**变换后**:

```python
i = start
__break_N: bool = False
while i < stop and not __break_N:       # i > stop for negative step
    A
    if cond:
        __break_N = True
    else:
        B
        C
    if not __break_N:
        i = i + step
```

关键细节：

| 方面 | 行为 |
| ---- | ---- |
| Break 标志命名 | `__break_N`，其中 N 为唯一计数器 |
| While 条件 | 正步长：`And(i < stop, Not(__break_N))`；负步长：`And(i > stop, Not(__break_N))` |
| 迭代器推进 | 由 `if not __break_N` 保护，防止在 break 点之后继续推进 |
| WhileStmt 的 break | 相同模式，但无需 for-to-while 转换 |

### 同时包含 Break 和 Continue

当循环同时包含 `break` 和 `continue` 时，阶段 1 先消除所有 `continue` 语句，然后阶段 2 在已变换的 IR 上消除 `break` 语句。

**变换前**:

```python
for i in pl.range(n):
    A
    if cond1:
        continue
    B
    if cond2:
        break
    C
```

**阶段 1 后**（continue 已消除）:

```python
for i in pl.range(n):
    A
    if cond1:
        pass
    else:
        B
        if cond2:
            break
        C
```

**阶段 2 后**（break 已消除）:

```python
i = 0
__break_0: bool = False
while i < n and not __break_0:
    A
    if cond1:
        pass
    else:
        B
        if cond2:
            __break_0 = True
        else:
            C
    if not __break_0:
        i = i + 1
```

## 流水线位置

CtrlFlowTransform 在 UnrollLoops 之后、ConvertToSSA 之前运行：

```text
UnrollLoops -> CtrlFlowTransform -> ConvertToSSA -> FlattenCallExpr -> SplitChunkedLoops -> ...
```

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| 所需 (Required) | `TypeChecked` |
| 产生 (Produced) | `TypeChecked`, `StructuredCtrlFlow` |
| 失效 (Invalidated) | (none) |
