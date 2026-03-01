# FlattenCallExpr Pass

将嵌套的调用表达式 (Expression) 展平为三地址码形式。

## 概述

此 Pass 通过将调用表达式提取到临时变量中，确保调用表达式不会出现在嵌套上下文中。它强制执行三地址码约束：

1. 调用参数不能是调用
2. If 条件不能是调用
3. For 循环范围（start/stop/step）不能是调用
4. 二元/一元表达式操作数不能是调用

**需要**：TypeChecked 属性 (Property)（运行 `RunVerifier` 或确保类型 (Type) 检查已通过）。

**使用时机**：在类型检查之后、代码生成 (CodeGen) 之前运行此 Pass，以简化下游分析和代码生成。

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
2. **提取到临时变量**：创建临时变量（命名为 `_t0`、`_t1` 等）
3. **插入 AssignStmt**：在原始语句 (Statement) 之前添加赋值语句
4. **替换为变量**：将嵌套调用替换为临时变量引用
5. **处理控制流**：对于 if/for 语句，插入到最后一个 OpStmts 或创建新的 OpStmts

**提取位置**：

- AssignStmt/EvalStmt 之前：直接插入在前面
- IfStmt/ForStmt 之前：插入到前面 SeqStmts 中的最后一个 OpStmts，或创建新的 OpStmts

## 示例

### 嵌套调用参数

**变换前**：

```python
c = foo(bar(a))  # bar(a) is nested in foo's arguments
```

**变换后**：

```python
_t0 = bar(a)
c = foo(_t0)
```

### If 条件中的嵌套调用

**变换前**：

```python
if is_valid(compute(x)):
    y = 1
```

**变换后**：

```python
_t0 = compute(x)
_t1 = is_valid(_t0)
if _t1:
    y = 1
```

### 多个嵌套调用

**变换前**：

```python
result = add(mul(a, b), div(c, d))
```

**变换后**：

```python
_t0 = mul(a, b)
_t1 = div(c, d)
result = add(_t0, _t1)
```

### 二元表达式中的嵌套

**变换前**：

```python
x = compute(a) + compute(b)
```

**变换后**：

```python
_t0 = compute(a)
_t1 = compute(b)
x = _t0 + _t1
```

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

## 错误类型

此 Pass 可以通过 `NestedCallErrorType` 检测并报告嵌套调用违规：

- `CALL_IN_CALL_ARGS`：调用参数中的调用
- `CALL_IN_IF_CONDITION`：if 条件中的调用
- `CALL_IN_FOR_RANGE`：for 范围中的调用
- `CALL_IN_BINARY_EXPR`：二元表达式中的调用
- `CALL_IN_UNARY_EXPR`：一元表达式中的调用
