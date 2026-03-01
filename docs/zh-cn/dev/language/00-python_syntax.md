# Python IR 语法规范

## 概述

PyPTO IR 的 Python 风格语法:

- **完整**: 包含重构 IR 所需的全部信息
- **可解析 (Parser)**: 可解析回 IR (参见 [IR 解析器](../ir/07-parser.md))
- **Pythonic**: 遵循 Python 风格, 通过大部分代码检查工具
- **静态单赋值 (SSA) 风格**: 使用 SSA, 配合 `pl.yield_()` 和 `pl.range()`

## 模块结构

```python
# pypto.program: program_name
import pypto.language as pl
```

对于未命名程序: `# pypto.program`

**注意:** 模块前缀可配置 (默认 `pl`, 旧版 `ir`, 支持自定义)。

## 类型系统

### 标量类型

```python
x: pl.INT64
y: pl.FP32
z: pl.BOOL
```

可用类型:

| 类别 | 类型 |
| ---- | ---- |
| **整数** | `INT4`, `INT8`, `INT16`, `INT32`, `INT64` |
| **无符号整数** | `UINT4`, `UINT8`, `UINT16`, `UINT32`, `UINT64` |
| **浮点数** | `FP4`, `FP8`, `FP16`, `FP32` |
| **Brain Float** | `BF16` |
| **Hisilicon** | `HF4`, `HF8` |
| **布尔值** | `BOOL` |

### 张量 (Tensor) 和 Tile 类型

```python
# Tensor (subscript notation)
a: pl.Tensor[[4, 8], pl.FP32]      # Fixed shape
b: pl.Tensor[[n, m], pl.INT64]     # Symbolic shape

# Tile (block in unified buffer)
t: pl.Tile[[16, 16], pl.FP16]
```

### 内存引用 (MemRef)

```python
# Create MemRef
addr_expr = pl.ConstInt(0x1000, pl.INT64, span)
memref = pl.MemRef(pl.MemorySpace.DDR, addr_expr, 1024)

# Memory spaces: DDR, Vec, Mat, Left, Right, Acc

# Tensor with memref
tensor: pl.Tensor[[64, 128], pl.FP32]  # memref=pl.MemRef(pl.MemorySpace.DDR, addr, 8192)
```

### Tile 视图 (TileView)

```python
# Create TileView
valid_shape = [pl.ConstInt(16, pl.INT64, span)] * 2
stride = [pl.ConstInt(1, pl.INT64, span), pl.ConstInt(16, pl.INT64, span)]
start_offset = pl.ConstInt(0, pl.INT64, span)
tile_view = pl.TileView(valid_shape=valid_shape, stride=stride, start_offset=start_offset)

# Tile with memref and tile_view
tile: pl.Tile(
    (16, 16), pl.FP16,
    memref=pl.MemRef(pl.MemorySpace.Left, addr, 512),
    tile_view=pl.TileView(valid_shape=..., stride=..., start_offset=...)
)
```

## 表达式 (Expression)

### 变量和常量

```python
x              # Variable reference
tensor_a       # Tensor variable
42             # Integer literal
3.14           # Float literal
```

**闭包变量:** 在 DSL 作用域中未找到的名称会从外层 Python 作用域解析。支持的类型: `int`, `float`, `bool`, `list`, `tuple` 以及 IR 表达式。

```python
OFFSET = [0, 0]
TILE_SHAPE = [64, 64]

@pl.function
def func(t: pl.Tensor[[128, 128], pl.FP32], out: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
    a: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, OFFSET, TILE_SHAPE)  # closure vars as positional args
    ...
```

### 二元操作

| Python 操作符 | PyPTO IR | 类别 |
| ------------- | -------- | ---- |
| `+` | Add | 算术 |
| `-` | Sub | 算术 |
| `*` | Mul | 算术 |
| `//` | FloorDiv | 算术 |
| `%` | FloorMod | 算术 |
| `/` | FloatDiv | 算术 |
| `**` | Pow | 算术 |
| `==`, `!=`, `<`, `<=`, `>`, `>=` | Eq, Ne, Lt, Le, Gt, Ge | 比较 |
| `and`, `or` | And, Or | 逻辑 |
| `^` | Xor | 逻辑 |
| `&` | BitAnd | 位运算 |
| `\|` | BitOr | 位运算 |
| `<<`, `>>` | BitShiftLeft, BitShiftRight | 位运算 |

### 一元操作和函数

```python
-x              # Neg
~x              # BitNot
not x           # Not
abs(x)          # Abs
min(a, b)       # Min
max(a, b)       # Max
```

### 函数/操作调用

```python
# Explicit namespace
pl.tensor.add(a, b)                  # Tensor addition
pl.block.load(t, [0, 0], [64, 64])      # Block load

# Unified dispatch (auto-selects tensor/block based on input type)
pl.add(a, b)                          # Tensor or Tile — dispatched automatically
pl.mul(tile, 2.0)                     # Tile + scalar -> block.muls
pl.exp(tile)                          # Tile -> block.exp

# Promoted ops (single-module ops accessible at pl.*)
pl.load(t, [0, 0], [64, 64])            # Promoted from block
pl.create_tensor([64], dtype=pl.FP32)       # Promoted from tensor
```

## 语句 (Statement)

### 赋值

```python
x: pl.INT64 = expr
y: pl.Tensor[[4], pl.FP32] = tensor_op(a)
```

### If 语句 (SSA 风格)

```python
# If with both branches
if condition:
    y1 = pl.yield_(value1)
else:
    y1 = pl.yield_(value2)

# Multiple return values (no inline type annotations)
if condition:
    y1, y2 = pl.yield_(value1, value2)
else:
    y1, y2 = pl.yield_(value3, value4)
```

**要点:**

- `pl.yield_()` 赋值给 SSA phi 节点
- yield 中定义的变量在 if 之后可访问
- 两个分支必须 yield 相同的变量
- 元组解包时不能使用内联类型标注

### For 循环 (带 iter_args 的 SSA 风格)

```python
# Simple loop
for i in pl.range(start, stop, step):
    body_statements

# Loop with iter_args (loop-carried values)
sum_init: pl.INT64 = 0
for i, (sum,) in pl.range(0, n, 1, init_values=(sum_init,)):
    sum = pl.yield_(sum + i)
sum_final = sum

# Parallel for loop
for i in pl.parallel(start, stop, step):
    body_statements
```

**要点:** 循环携带值使用 `pl.range()` 或 `pl.parallel()` 的 `init_values`, 元组解包 `(sum,)` 声明 iter_args, `pl.yield_()` 为下一次迭代更新值, 循环结束后 iter_args 包含最终值。`pl.parallel()` 生成 `ForKind.Parallel` 循环, `pl.range()` 生成 `ForKind.Sequential` (默认)。

### Yield 语句

```python
yield            # No values
yield x          # Single value
yield x, y       # Multiple values
```

### 语句序列

```python
stmt1            # Natural Python sequencing
stmt2
stmt3
```

## 函数

```python
# Single return type
def function_name(param1: pl.INT64, param2: pl.FP32) -> pl.INT64:
    x: pl.INT64 = param1 + 1
    return x

# Multiple return types
def function_name(x: pl.INT64) -> tuple[pl.INT64, pl.INT64]:
    y: pl.INT64 = x + 1
    z: pl.INT64 = x * 2
    return y, z

# No return types
def function_name(x: pl.INT64):
    y: pl.INT64 = x + 1

# With function type
@pl.function(type=pl.FunctionType.Orchestration)
def orchestrator(n: pl.INT64) -> pl.INT64:
    return n + 1

@pl.function(type=pl.FunctionType.InCore)
def aicore_kernel(x: pl.INT64) -> pl.INT64:
    return x * 2
```

### 函数类型

| 类型 | 用途 | 描述 |
| ---- | ---- | ---- |
| `pl.FunctionType.Opaque` | 默认 | 未指定的函数类型 |
| `pl.FunctionType.Orchestration` | Host/AICPU | 控制流和依赖分析 |
| `pl.FunctionType.InCore` | AICore | 特定 AICore 上的子图 |

未指定类型时, 函数默认为 `Opaque`。

### 参数方向

参数可以使用包装类型指定 `In` (默认)、`Out` 或 `InOut` 方向:

```python
@pl.function(type=pl.FunctionType.InCore)
def kernel(
    qi: pl.Tensor[[16, 128], pl.BF16],                   # In (default)
    output: pl.InOut[pl.Tensor[[16, 128], pl.FP32]],      # InOut
    result: pl.Out[pl.Tensor[[16, 128], pl.FP32]],        # Out
    scale: pl.Scalar[pl.FP32],                             # In (default)
) -> pl.Tensor[[16, 128], pl.FP32]:
    ...
```

| 方向 | 包装类型 | 描述 |
| ---- | -------- | ---- |
| `In` | 无 (默认) | 只读输入参数 |
| `Out` | `pl.Out[type]` | 只写输出参数 |
| `InOut` | `pl.InOut[type]` | 读写输入/输出参数 |

**约束:** `Scalar` 参数不能使用 `InOut` 方向 (会抛出 `ParserTypeError`)。

## 完整示例

### 张量操作 (带 iter_args 的循环)

```python
# pypto.program: my_program
import pypto.language as pl

def loop_sum(n: pl.INT64) -> pl.INT64:
    sum_init: pl.INT64 = 0
    for i, (sum,) in pl.range(0, n, 1, init_values=(sum_init,)):
        sum = pl.yield_(sum + i)
    return sum
```

### 块操作 (基于 Tile 的计算)

```python
import pypto.language as pl

@pl.program
class BlockExample:
    @pl.function
    def tile_add(
        self,
        input_a: pl.Tensor[[64, 64], pl.FP32],
        input_b: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
        tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(input_b, [0, 0], [64, 64])
        tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)
        result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], [64, 64], output)
        return result
```

## SSA 风格控制流

`pl.yield_()` 为 if/for 语句创建 SSA phi 节点:

```python
# If: phi node at merge point
if condition:
    y1 = pl.yield_(x + 1)
else:
    y1 = pl.yield_(x + 2)
# y1 = phi(x + 1, x + 2)

# For: loop-carried values via iter_args
sum_init: pl.INT64 = 0
for i, (sum,) in pl.range(0, 10, 1, init_values=(sum_init,)):
    sum = pl.yield_(sum + i)
sum_final: pl.INT64 = sum  # captures final value
```

## 可配置模块前缀

打印器支持可配置的模块前缀 (`pl`, `pi`, `ir` 或自定义):

```python
print(ir.python_print(stmt))          # "x: pl.INT64 = a + b" (default)
print(ir.python_print(stmt, "ir"))    # "x: ir.INT64 = a + b"
```

## 参考资料

- [IR 概述](../ir/00-overview.md) - 核心 IR 结构
- [IR 解析器 (Parser)](../ir/07-parser.md) - 将 Python 语法解析回 IR
- [操作符注册](../ir/05-operators.md) - 操作系统和类型推断
