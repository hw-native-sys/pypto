# ConvertToSSA Pass

将非 SSA IR 转换为静态单赋值 (SSA) 形式，包含变量重命名、phi 节点和 iter_args。

## 概述

此 Pass 将具有对同一变量多次赋值的中间表示 (IR) 转换为 SSA 形式，使每个变量恰好被赋值一次。它处理以下情况：

- **直线代码**：对同一变量的多次赋值
- **If 语句 (Statement)**：在一个或两个分支中修改的变量
- **For 循环**：在循环体内修改的变量
- **混合 SSA/非 SSA**：保留现有的 SSA 结构，同时转换非 SSA 部分

**需要**：TypeChecked 属性 (Property)（运行 `RunVerifier` 或确保类型 (Type) 检查已通过）。

**使用时机**：在任何需要 SSA 形式的优化或分析之前运行此 Pass（如 OutlineIncoreScopes、内存优化 Pass）。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::ConvertToSSA()` | `passes.convert_to_ssa()` | 函数级 |

**工厂函数**：

```cpp
Pass ConvertToSSA();
```

**Python 用法**：

```python
from pypto.pypto_core import passes

ssa_pass = passes.convert_to_ssa()
program_ssa = ssa_pass(program)
```

## 算法

1. **变量重命名**：为每次赋值添加版本后缀（x -> x_0, x_1, x_2）
2. **If 的 Phi 节点**：为在 if 分支中修改的变量添加 phi 节点（return_vars + YieldStmt）
3. **循环的 Iter_args**：将循环中修改的变量转换为 iter_args + return_vars 模式，带有 YieldStmt
4. **作用域跟踪**：跨嵌套作用域跟踪变量定义
5. **保留**：保持现有 SSA 构造不变

**关键变换**：

- `x = 1; x = 2` -> `x_0 = 1; x_1 = 2`
- 具有分歧赋值的 If -> 在两个分支中添加 return_vars 和 YieldStmt
- 具有循环携带依赖的 For 循环 -> 添加 iter_args/return_vars/YieldStmt

## 示例

### 直线代码

**变换前**：

```python
x = 1
y = x + 2
x = 3  # Multiple assignment
z = x + 4
```

**变换后**：

```python
x_0 = 1
y = x_0 + 2
x_1 = 3
z = x_1 + 4
```

### If 语句

**变换前**：

```python
x = 1
if condition:
    x = 2  # Modified in then branch
z = x + 3  # Uses x after if
```

**变换后**：

```python
x_0 = 1
if condition:
    x_1 = 2
    yield (x_1,)  # Yield modified variable
else:
    yield (x_0,)  # Yield original variable
return_vars = (x_2,)  # Phi node
z = x_2 + 3
```

### For 循环

**变换前**：

```python
sum = 0
for i in range(10):
    sum = sum + i  # Loop-carried dependency
```

**变换后**：

```python
sum_0 = 0
for i in range(10):
    iter_args = (sum_1,)
    init_values = (sum_0,)
    # Loop body
    sum_2 = sum_1 + i
    yield (sum_2,)
return_vars = (sum_3,)
```

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass ConvertToSSA();
```

**实现文件**：`src/ir/transforms/convert_to_ssa.cpp`

- 使用 IRMutator 模式遍历和变换 IR
- 维护变量重命名的版本映射
- 插入 YieldStmt 并管理 return_vars/iter_args

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("convert_to_ssa", &pass::ConvertToSSA, "Convert to SSA form");
```

**测试**：`tests/ut/ir/transforms/test_convert_to_ssa_pass.py`

- 测试直线代码重命名
- 测试 if 语句 phi 节点
- 测试 for 循环 iter_args
- 测试嵌套作用域
- 测试混合 SSA/非 SSA
