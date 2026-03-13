# LowerBreakContinue Pass

将 InCore/AIC/AIV 函数中的 `BreakStmt` 和 `ContinueStmt` 下降（Lower）为等价的结构化控制流。

## 概述

PTO 和 CCE 代码生成后端未实现 `BreakStmt`/`ContinueStmt` 的访问器。该 Pass 在代码生成前将其重写：

- **Continue**：重构为带有 phi 节点 `return_vars` 的 `IfStmt`。Continue 路径 yield 当前 iter_arg 值；正常路径 yield 原始值。由单个顶层 `YieldStmt` 使用 phi 结果。
- **Break**：将 `ForStmt` 转换为带有 `__brk_flag` IterArg 的 `WhileStmt`。Break 路径将标志设置为 `True`；while 条件检查 `not __brk_flag`。
- **两者兼有**：先转换为 `WhileStmt`（break 要求），然后在 while 体内通过 phi 节点方式处理 continue。

**前置条件**：

- 输入 IR 必须为 SSA 形式（SSAForm 必需且保持）
- InCore 作用域必须已提取（SplitIncoreOrch 必需且保持）

**使用时机**：在 InferTileMemorySpace 之后、InitMemRef 之前运行。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::LowerBreakContinue()` | `passes.lower_break_continue()` | 函数级 |

**工厂函数**：

```cpp
Pass LowerBreakContinue();
```

**Python 用法**：

```python
from pypto import passes

lowered = passes.lower_break_continue()(program)
```

## 属性

| 属性 | 必需 | 产生 | 失效 |
| ---- | ---- | ---- | ---- |
| SSAForm | 是 | 是 | — |
| SplitIncoreOrch | 是 | 是 | — |

## 算法

### Continue 下降（ForStmt）

```python
# 变换前：
for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
    if i < 5:
        continue
    y = pl.add(x_iter, x_iter)
    x_iter = pl.yield_(y)

# 变换后（phi 节点方式）：
for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
    if i < 5:
        __phi_0 = pl.yield_(x_iter)       # IfStmt 分支：yield 当前值
    else:
        y = pl.add(x_iter, x_iter)
        __phi_0 = pl.yield_(y)            # IfStmt 分支：yield 计算值
    x_iter = pl.yield_(__phi_0)           # 循环顶层 yield 使用 phi
```

### Break 下降（ForStmt → WhileStmt）

```python
# 变换前：
for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
    if i > 5:
        break
    y = pl.add(x_iter, x_iter)
    x_iter = pl.yield_(y)

# 变换后：
for (__brk_flag, __lv, x_iter) in pl.while_(init_values=(False, 0, x_0)):
    pl.cond(__lv < 10 and not __brk_flag)
    if __lv > 5:
        __phi_0, __phi_1, __phi_2 = pl.yield_(True, __lv + 1, x_iter)
    else:
        y = pl.add(x_iter, x_iter)
        __phi_0, __phi_1, __phi_2 = pl.yield_(False, __lv + 1, y)
    __brk_flag, __lv, x_iter = pl.yield_(__phi_0, __phi_1, __phi_2)
```

## 作用范围

- 仅针对 `IsInCoreType()` 函数（InCore、AIC、AIV）
- Orchestration 函数保持不变
- 自底向上处理嵌套循环 — 每个循环的 break/continue 在该循环层级独立处理
- 处理同一循环中的多个 continue 和嵌套的 break+continue

## 流水线位置

```text
... → InferTileMemorySpace → LowerBreakContinue → InitMemRef → ...
```
