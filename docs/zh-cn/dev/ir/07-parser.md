# DSL 函数的 IR 解析器 (Parser)

## 概述

IR 解析器 (Parser) 使用装饰器（`@pl.function`、`@pl.program`）将 Python DSL 代码转换为 PyPTO 中间表示 (IR)。它强制执行静态单赋值 (SSA) 属性 (Property)、跟踪源码位置，并支持嵌套控制流。

**关键组件**：装饰器 → AST 解析器 → IR 构建器 (Builder) → 作用域管理器 (SSA) → ir.Function

参见 [IR 构建器](06-builder.md) 了解手动 IR 构建，以及 [Python IR 语法](../language/00-python_syntax.md) 了解完整语法。

## 用法

### 基本函数

```python
import pypto
import pypto.language as pl

@pl.function
def simple_add(
    x: pl.Tensor[[64, 128], pl.FP16],
    y: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    result: pl.Tensor[[64, 128], pl.FP16] = pl.add(x, y)
    return result

# simple_add is now an ir.Function object
assert isinstance(simple_add, pypto.ir.Function)
```

### 类型 (Type) 标注

所有参数和局部变量需要类型标注：

```python
x: pl.Tensor[[64, 128], pl.FP16]  # Recommended subscript syntax
x: pl.Tensor((64, 128), pl.FP16)  # Legacy call syntax (also accepted)
```

两种语法等价；打印器始终输出下标记法。

### 带迭代参数的 For 循环

使用 `pl.range()` 和元组解包实现循环携带值（iter_args）：

```python
for i, (sum_val,) in pl.range(10, init_values=(sum_init,)):
    new_sum: pl.Tensor[[1], pl.INT32] = pl.add(sum_val, i)
    sum_out = pl.yield_(new_sum)  # Use pl.yield_ (not yield)
```

**语法**：`loop_var, (iter_arg1, ...)` - iter_args 的数量必须与 init_values 匹配。

### Yield 与 If 语句 (Statement)

使用 `pl.yield_()` 从嵌套作用域返回值：

```python
# Single/multiple value yield
result = pl.yield_(expr)
v1, v2, v3 = pl.yield_(expr1, expr2, expr3)

# If statements create phi nodes
if x > 0:
    positive: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    result = pl.yield_(positive)
else:
    negative: pl.Tensor[[64], pl.FP32] = pl.mul(x, -1.0)
    result = pl.yield_(negative)
```

## 基于文本的解析

从字符串或文件解析 DSL 代码，用于动态代码生成：

| 函数 | 用途 | 示例 |
| ---- | ---- | ---- |
| `pl.parse(code)` | 从字符串解析（自动检测函数/程序） | `result = pl.parse("@pl.function\ndef f(x): ...")` |
| `pl.loads(path)` | 从文件加载（自动检测函数/程序） | `result = pl.loads('kernel.py')` |

**特性**：

- **自动检测**：自动检测代码是否包含 `@pl.function` 或 `@pl.program`
- 根据检测结果返回 `ir.Function` 或 `ir.Program`
- 每次解析仅限单个函数/程序（否则抛出 `ValueError`）
- 生成与装饰器相同的 `ir.Function`/`ir.Program` 对象
- 参见 `examples/ir_parser/parse_from_text.py` 获取示例

**已弃用的别名**（仍然支持）：

- `pl.parse_program(code)` → 请改用 `pl.parse(code)`
- `pl.loads_program(path)` → 请改用 `pl.loads(path)`

## SSA 属性

解析器强制执行 SSA：

**单次赋值**：每个变量在每个作用域中只赋值一次

```python
# ✓ Valid
y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)

# ✗ Invalid - SSA violation
y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
y = pl.mul(x, 2.0)  # Error: y already defined
```

**作用域隔离**：内部作用域的变量必须通过 yield 传出

```python
# ✗ Invalid - temp not yielded
for i, (sum_val,) in pl.range(10, init_values=(x,)):
    temp: pl.Tensor[[64], pl.FP32] = pl.add(sum_val, i)
return temp  # Error: temp not in outer scope

# ✓ Valid - explicit yield
for i, (sum_val,) in pl.range(10, init_values=(x,)):
    temp: pl.Tensor[[64], pl.FP32] = pl.add(sum_val, i)
    result = pl.yield_(temp)
return result  # OK
```

**迭代参数**：通过 phi 节点在每次迭代中创建新的 SSA 值。

## Span 跟踪与操作

**Span 跟踪**：保留源码位置以提供更好的错误消息

- 每个 IR 节点包含带有文件名、行/列范围的 `Span`
- 支持调试、错误报告和源码到 IR 的映射

**支持的操作**：

| 分类 | 示例 |
| ---- | ---- |
| **张量操作** | `pl.{add, mul, sub, div, matmul, cast, view, ...}` |
| **二元表达式 (Expression)** | `a + b`, `a - b`, `a * b`, `a / b`, `i == 0`, `x < 10` |
| **字面量** | `42` → `ConstInt`, `3.14` → `ConstFloat` |

参见 [Python IR 语法](../language/00-python_syntax.md) 获取完整操作列表。

## 完整示例

嵌套控制流示例：

```python
@pl.function
def flash_attn_simplified(
    q: pl.Tensor[[64, 128], pl.FP16],
    k: pl.Tensor[[1024, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP32]:
    attn_init: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)

    for i, (attn,) in pl.range(16, init_values=(attn_init,)):
        k_block: pl.Tensor[[64, 128], pl.FP16] = pl.view(k, [64, 128], [i * 64, 0])
        scores: pl.Tensor[[64, 128], pl.FP16] = pl.matmul(q, k_block, b_trans=True)

        if i == 0:
            new_attn: pl.Tensor[[64, 128], pl.FP32] = pl.cast(scores, target_type=pl.FP32)
            result = pl.yield_(new_attn)
        else:
            updated: pl.Tensor[[64, 128], pl.FP32] = pl.add(attn, scores)
            result = pl.yield_(updated)

        final = pl.yield_(result)

    return final
```

## 使用 @pl.program 的多函数程序

定义包含多个可相互调用的函数的程序：

```python
@pl.program
class MathOps:
    @pl.function
    def square(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        result: pl.Tensor[[1], pl.INT32] = pl.mul(x, x)
        return result

    @pl.function
    def sum_of_squares(self, a: pl.Tensor[[1], pl.INT32], b: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        a_squared: pl.Tensor[[1], pl.INT32] = self.square(a)  # Cross-function call
        b_squared: pl.Tensor[[1], pl.INT32] = self.square(b)
        result: pl.Tensor[[1], pl.INT32] = pl.add(a_squared, b_squared)
        return result
```

**关键规则**：

- 使用 `@pl.program` 的基于类的语法
- 方法需要 `self` 参数（自动从 IR 中剥离）
- 跨函数调用使用 `self.method_name()` → 解析为 `GlobalVar` 引用
- 两阶段解析：先收集 `GlobalVar`，再解析函数体（支持前向引用）
- 访问函数：`program.get_function("name")`
- 文本解析：`pl.parse(code)`、`pl.loads(path)`（自动检测程序/函数）
- 打印：`pypto.ir.python_print(program)` 生成有效的 `@pl.program` 类

**示例**：参见 `examples/ir_parser/program_example.py` 和 `examples/ir_builder/program_builder_example.py`

## 限制与测试

**当前限制**：

- if 条件中仅支持标量比较（不支持张量）
- `@pl.function` 内不支持嵌套函数定义
- 有限的 Python 子集（函数内不支持类、装饰器）
- 所有作用域输出都需要显式 yield
- 所有变量都需要类型标注

**测试**：运行 `pytest tests/ut/language/parser/` 获取完整的解析器测试。

## 另请参阅

- [Python IR 语法](../language/00-python_syntax.md) - 完整语法规范
- [IR 构建器](06-builder.md) - 手动 IR 构建 API
- [IR 概述](00-overview.md) - 核心 IR 概念
