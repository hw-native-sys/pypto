# FlattenCallExpr Pass

将嵌套的调用表达式 (Expression) 展平为三地址码形式。

## 概述

此 Pass 通过将调用表达式提取到临时变量中，确保调用表达式不会出现在嵌套上下文中。它强制执行三地址码约束：

1. 调用参数不能是调用
2. If 条件不能是调用
3. For 循环范围（start/stop/step）不能是调用
4. 二元/一元表达式操作数不能是调用

**需要**：TypeChecked、SSAForm 属性 (Property)（通常由前序 Pass 产生；如需在执行前校验 required/produced，请在 `PassContext` 中启用 `VerificationInstrument`）。

**使用时机**：通常在类型检查 Pass 之后、代码生成 (CodeGen) 之前运行此 Pass，以简化下游分析和代码生成；此顺序是约定而非自动强制的要求。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::FlattenCallExpr()` | `passes.flatten_call_expr()` | 函数级 |

**工厂函数**：

```cpp
Pass FlattenCallExpr();
```

**Python 用法**：

```python
from pypto.pypto_core import passes

flatten_pass = passes.flatten_call_expr()
program_flat = flatten_pass(program)
```

## 算法

1. **检测嵌套调用**：识别嵌套上下文中的调用表达式
2. **提取到临时变量**：创建临时变量（命名为 `t__tmp_v0`、`t__tmp_v1` 等）
3. **插入 AssignStmt**：在原始语句 (Statement) 之前添加赋值语句
4. **替换为变量**：将嵌套调用替换为临时变量引用
5. **处理控制流**：对于 if/for 语句，将提取出的临时语句直接插入到外层 `SeqStmts` 中该控制流语句之前

**提取位置**：

- AssignStmt/EvalStmt 之前：直接插入在前面
- 在 IfStmt/ForStmt 之前：作为外层 `SeqStmts` 中的同级语句插入
- ScopeStmt 内部（`pl.at()`）：临时变量始终插入在 scope body **内部**，保持执行上下文边界

## 示例

### 嵌套调用参数

**变换前**：

```python
c = foo(bar(a))  # bar(a) is nested in foo's arguments
```

**变换后**：

```python
t__tmp_v0 = bar(a)
c = foo(t__tmp_v0)
```

### If 条件中的嵌套调用

**变换前**：

```python
if is_valid(compute(x)):
    y = 1
```

**变换后**：

```python
t__tmp_v0 = compute(x)
t__tmp_v1 = is_valid(t__tmp_v0)
if t__tmp_v1:
    y = 1
```

### 多个嵌套调用

**变换前**：

```python
result = add(mul(a, b), div(c, d))
```

**变换后**：

```python
t__tmp_v0 = mul(a, b)
t__tmp_v1 = div(c, d)
result = add(t__tmp_v0, t__tmp_v1)
```

### 二元表达式中的嵌套

**变换前**：

```python
x = compute(a) + compute(b)
```

**变换后**：

```python
t__tmp_v0 = compute(a)
t__tmp_v1 = compute(b)
x = t__tmp_v0 + t__tmp_v1
```

### Scope 块内的嵌套调用

**变换前**：

```python
with pl.at(level=pl.Level.CORE_GROUP):
    result = assemble(target, cast(x, BF16), offsets)
```

**变换后**：

```python
with pl.at(level=pl.Level.CORE_GROUP):
    t__tmp_v0 = cast(x, BF16)
    result = assemble(target, t__tmp_v0, offsets)
```

临时变量保留在 `pl.at()` 块内。如果没有这种 scope 感知机制，`t__tmp_v0 = cast(...)` 会被提升到 scope 外部，导致代码生成时出现 "Misplaced tensor op" 错误。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass FlattenCallExpr();
```

**实现文件**：`src/ir/transforms/flatten_call_expr.cpp`

- 使用 IRMutator 遍历表达式
- 维护临时变量计数器
- 收集提取的赋值
- 使用展平后的表达式重建语句

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("flatten_call_expr", &pass::FlattenCallExpr, "Flatten nested calls");
```

**测试**：`tests/ut/ir/transforms/test_flatten_call_expr_pass.py`

- 测试调用参数提取
- 测试 if 条件提取
- 测试 for 范围提取
- 测试二元/一元表达式提取
- 测试多个嵌套调用
- 测试 scope 感知提取（`pl.at()` 块）

## 错误类型

此 Pass 可以通过 `NestedCallErrorType` 检测并报告嵌套调用违规：

- `CALL_IN_CALL_ARGS`：调用参数中的调用
- `CALL_IN_IF_CONDITION`：if 条件中的调用
- `CALL_IN_FOR_RANGE`：for 范围中的调用
- `CALL_IN_BINARY_EXPR`：二元表达式中的调用
- `CALL_IN_UNARY_EXPR`：一元表达式中的调用
