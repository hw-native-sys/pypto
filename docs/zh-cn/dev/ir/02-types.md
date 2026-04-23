# PyPTO IR 类型与示例

本文档介绍类型 (Type) 系统并提供实用的使用示例。

## 类型系统

### ScalarType

表示原始标量类型。

```python
from pypto import DataType, ir

int_type = ir.ScalarType(DataType.INT64)
float_type = ir.ScalarType(DataType.FP32)
```

**支持的 DataType：** INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FP16, FP32, FP64, BOOL, INDEX

> **注意：** `INDEX` 是用于索引计算（循环变量、维度、偏移量、步长）的独立整数类型。它拥有自己的类型代码和字符串表示（`"index"`）。虽然语义上与 `INT64` 类似，但 `INDEX != INT64` —— 它们是不同的类型。在代码生成中，INDEX 和 INT64 之间的隐式类型转换会被抑制。

### TensorType

带可选内存引用 (MemRef) 的多维张量 (Tensor)。

```python
span = ir.Span.unknown()

# Tensor with shape [10, 20]
shape = [ir.ConstInt(10, DataType.INT64, span), ir.ConstInt(20, DataType.INT64, span)]
tensor_type = ir.TensorType(shape, DataType.FP32)

# Tensor with MemRef
memref = ir.MemRef(ir.ConstInt(0x1000, DataType.INT64, span), 800, 0)
tensor_with_memref = ir.TensorType(shape, DataType.FP32, memref)
```

`TensorType.memory_space` 始终是 `ir.Mem.DDR`。`MemRef` 只保存地址、大小和
id；内存空间不再存储在 `MemRef` 本身上。

### 带 TensorView 的 TensorType

带有布局和步长信息的张量，用于优化内存访问。

```python
# Create tensor with tensor view (stride/valid_shape accept int or Expr)
tensor_view = ir.TensorView(stride=[1, 128], layout=ir.TensorLayout.ND)
tensor_with_view = ir.TensorType([128, 256], DataType.FP32, memref=None, tensor_view=tensor_view)

# With valid_shape
tensor_view = ir.TensorView(stride=[1, 128], layout=ir.TensorLayout.ND, valid_shape=[64, 128])

# With pad mode for out-of-valid-shape accesses (symmetric with TileView)
tensor_view = ir.TensorView(
    stride=[1, 128], layout=ir.TensorLayout.ND, valid_shape=[64, 128], pad=ir.PadValue.zero
)

# Different layouts
nd_view = ir.TensorView(stride=[1, 128], layout=ir.TensorLayout.ND)  # ND layout
dn_view = ir.TensorView(stride=[1, 128], layout=ir.TensorLayout.DN)  # DN layout
nz_view = ir.TensorView(stride=[1, 128], layout=ir.TensorLayout.NZ)  # NZ layout

# Expr values also accepted (e.g., symbolic dimensions)
stride = [ir.ConstInt(1, DataType.INT64, span), ir.ConstInt(128, DataType.INT64, span)]
tensor_view = ir.TensorView(stride=stride, layout=ir.TensorLayout.ND)

# Tensor with both MemRef and TensorView
memref = ir.MemRef(ir.ConstInt(0x2000, DataType.INT64, span), 16384, 1)
tensor_with_both = ir.TensorType([128, 256], DataType.FP16, memref=memref, tensor_view=tensor_view)
```

**TensorLayout 值：**

- `ND`：ND 布局
- `DN`：DN 布局
- `NZ`：NZ 布局

**TensorView 字段：**

- `stride`：每个维度的步长
- `layout`：`TensorLayout.ND` / `DN` / `NZ`
- `valid_shape`：可选的有效区域维度（为空表示使用完整 shape）
- `pad`：`PadValue.null`（默认）/ `zero` / `max` / `min`，用于访问超出
  `valid_shape` 部分时的填充模式。与 `TileView.pad` 对称；
  `tensor.slice(..., pad_value=PadValue.zero)` 会写入该字段。

### TileType

专用张量类型，带可选内存和视图信息，用于硬件优化操作。

```python
# Basic 16x16 tile
shape = [ir.ConstInt(16, DataType.INT64, span)] * 2
tile_type = ir.TileType(shape, DataType.FP16)

# 3D tile (supported at IR level)
shape_3d = [ir.ConstInt(4, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span)]
tile_type_3d = ir.TileType(shape_3d, DataType.FP16)

# Tile with MemRef and TileView
memref = ir.MemRef(ir.ConstInt(0, DataType.INT64, span), 512, 0)

tile_view = ir.TileView()
tile_view.valid_shape = [ir.ConstInt(16, DataType.INT64, span)] * 2
tile_view.stride = [ir.ConstInt(1, DataType.INT64, span), ir.ConstInt(16, DataType.INT64, span)]
tile_view.start_offset = ir.ConstInt(0, DataType.INT64, span)

tile_with_view = ir.TileType(shape, DataType.FP16, memref, tile_view, ir.Mem.Left)
```

`TileType.memory_space` 才是 Tile 放置位置的唯一来源。如果 `TileType`
携带 `MemRef`, 请在 `TileType` 自身上显式提供 tile 内存空间。

对于 Python DSL 类型标注，省略的 `TileView` 语法会被规范化为一个隐式
TileView：它由 tile shape 以及（如果存在）tile memory space 推导得到。
像 `pl.TileView()` 这样的冗余显式默认写法，会与省略写法被视为语义等价，
并且在 printer 输出时可能统一成规范形式。

### TupleType

异构类型元组。

```python
# Scalar tuple: (int, float)
scalar_tuple = ir.TupleType([
    ir.ScalarType(DataType.INT64),
    ir.ScalarType(DataType.FP32)
])

# Nested tuple
nested = ir.TupleType([
    ir.TupleType([ir.ScalarType(DataType.INT64)]),
    ir.ScalarType(DataType.FP32)
])
```

### PipeType

硬件执行流水线或同步屏障。

```python
pipe_s = ir.PipeType(ir.PipeType.S)    # Scalar pipe
pipe_v = ir.PipeType(ir.PipeType.V)    # Vector pipe
pipe_m = ir.PipeType(ir.PipeType.M)    # Matrix pipe
pipe_all = ir.PipeType(ir.PipeType.ALL) # All pipes
```

### UnknownType

未知或待推断类型的占位符。

```python
unknown = ir.UnknownType()
```

### DSL 中的 MemRef 类型注解

MemRef 可以在 `@pl.program` / `@pl.function` DSL 代码中作为位置参数指定在类型注解中：

```python
import pypto.language as pl

@pl.program
class MyProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32]):
        # Tile with MemRef and explicit tile memory space
        tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(0, 16384, 0), pl.Mem.Vec] = pl.tile.load(
            x, offsets=[0, 0], shapes=[64, 64]
        )

        # Tensor with MemRef (3-arg: shape, dtype, memref)
        y: pl.Tensor[[64, 64], pl.FP32, pl.MemRef(0, 16384, 1)] = pl.add(x, 1.0)

        # Tensor with layout and MemRef (4-arg: shape, dtype, layout, memref)
        z: pl.Tensor[[64, 64], pl.FP32, pl.NZ, pl.MemRef(0, 16384, 2)] = pl.add(x, 1.0)
```

**`pl.MemRef(addr, size, id)` 参数：**

| 参数 | 类型 | 说明 |
| ---- | ---- | ---- |
| `addr` | `int` | 基地址偏移 |
| `size` | `int` | 内存分配大小（字节） |
| `id` | `int` | 内存缓冲区标识符 |

`TensorType` 注解默认位于 `DDR`。为了兼容旧代码，解析器仍接受
`pl.MemRef(pl.Mem.DDR, addr, size, id)`，但新代码应优先使用 3 参数形式。

**消歧义（3 参数 Tensor）：** 解析器会自动区分 `pl.MemRef(...)` 和
`pl.NZ`/`pl.DN`/`pl.ND` 布局枚举。

**Tile 规则：** 如果在 `pl.Tile[...]` 注解中使用 `pl.MemRef(...)`，必须再
单独提供一个 `pl.Mem.*` 参数来声明 tile 的内存空间。

### MemorySpace 枚举（别名：`Mem`）

| 值 | 说明 |
| -- | ---- |
| `DDR` | 主存储器（片外） |
| `Vec` | 向量/统一缓冲区（片上） |
| `Mat` | 矩阵/L1 缓冲区 |
| `Left` | 左矩阵操作数缓冲区 |
| `Right` | 右矩阵操作数缓冲区 |
| `Acc` | 累加器缓冲区 |

## Python 使用示例

### 示例 1：构建表达式

```python
from pypto import DataType, ir

span = ir.Span.unknown()
dtype = DataType.INT64

# Variables and constants
x = ir.Var("x", ir.ScalarType(dtype), span)
y = ir.Var("y", ir.ScalarType(dtype), span)
one = ir.ConstInt(1, dtype, span)
two = ir.ConstInt(2, dtype, span)

# Build: ((x + 1) * (y - 2)) / (x + y)
x_plus_1 = ir.Add(x, one, dtype, span)
y_minus_2 = ir.Sub(y, two, dtype, span)
numerator = ir.Mul(x_plus_1, y_minus_2, dtype, span)
denominator = ir.Add(x, y, dtype, span)
result = ir.FloatDiv(numerator, denominator, dtype, span)
```

### 示例 2：控制流（绝对值）

```python
# if (x >= 0) then { result = x } else { result = -x }
x = ir.Var("x", ir.ScalarType(dtype), span)
result = ir.Var("result", ir.ScalarType(dtype), span)
zero = ir.ConstInt(0, dtype, span)

condition = ir.Ge(x, zero, dtype, span)
then_assign = ir.AssignStmt(result, x, span)
else_assign = ir.AssignStmt(result, ir.Neg(x, dtype, span), span)

abs_stmt = ir.IfStmt(condition, then_assign, else_assign, [result], span)
```

### 示例 3：带累加的循环

```python
# for i, (sum,) in pl.range(n, init_values=(0,)):
#     sum = pl.yield_(sum + i)

n = ir.Var("n", ir.ScalarType(dtype), span)
i = ir.Var("i", ir.ScalarType(dtype), span)
zero = ir.ConstInt(0, dtype, span)
one = ir.ConstInt(1, dtype, span)

sum_iter = ir.IterArg("sum", ir.ScalarType(dtype), zero, span)
add_expr = ir.Add(sum_iter, i, dtype, span)
yield_stmt = ir.YieldStmt([add_expr], span)
sum_final = ir.Var("sum_final", ir.ScalarType(dtype), span)

loop = ir.ForStmt(i, zero, n, one, [sum_iter], yield_stmt, [sum_final], span)
```

### 示例 4：带运算符调用的函数

```python
# def matmul(a, b) -> tensor:
#     result = tensor.matmul(a, b, out_dtype=FP32)

shape_m = ir.ConstInt(128, DataType.INT64, span)
shape_k = ir.ConstInt(64, DataType.INT64, span)
shape_n = ir.ConstInt(256, DataType.INT64, span)

a = ir.Var("a", ir.TensorType([shape_m, shape_k], DataType.FP16), span)
b = ir.Var("b", ir.TensorType([shape_k, shape_n], DataType.FP16), span)

matmul_call = ir.op.tensor.matmul(a, b, out_dtype=DataType.FP32)
result = ir.Var("result", ir.TensorType([shape_m, shape_n], DataType.FP32), span)
body = ir.AssignStmt(result, matmul_call, span)

return_types = [ir.TensorType([shape_m, shape_n], DataType.FP32)]
func = ir.Function("matmul", [a, b], return_types, body, span)
```

### 示例 5：包含多个函数的程序

```python
# Helper: square(x) -> int { return x * x }
x = ir.Var("x", ir.ScalarType(dtype), span)
square_result = ir.Var("result", ir.ScalarType(dtype), span)
square_body = ir.AssignStmt(square_result, ir.Mul(x, x, dtype, span), span)
square_func = ir.Function("square", [x], [ir.ScalarType(dtype)], square_body, span)

# Main: sum_squares(a, b) -> int { return square(a) + square(b) }
a = ir.Var("a", ir.ScalarType(dtype), span)
b = ir.Var("b", ir.ScalarType(dtype), span)

program = ir.Program([square_func], "math", span)
square_gvar = program.get_global_var("square")

call_a = ir.Call(square_gvar, [a], span)
call_b = ir.Call(square_gvar, [b], span)
sum_expr = ir.Add(call_a, call_b, dtype, span)

main_result = ir.Var("result", ir.ScalarType(dtype), span)
main_body = ir.AssignStmt(main_result, sum_expr, span)
main_func = ir.Function("sum_squares", [a, b], [ir.ScalarType(dtype)], main_body, span)

program = ir.Program([square_func, main_func], "math", span)
```

### 示例 6：使用 TileType 的内存布局

```python
# 32x32 tile in Left memory with custom stride
shape = [ir.ConstInt(32, DataType.INT64, span)] * 2
memref = ir.MemRef(ir.ConstInt(0, DataType.INT64, span), 2048, 0)

tile_view = ir.TileView()
tile_view.valid_shape = shape
tile_view.stride = [ir.ConstInt(1, DataType.INT64, span), ir.ConstInt(32, DataType.INT64, span)]
tile_view.start_offset = ir.ConstInt(0, DataType.INT64, span)

tile_type = ir.TileType(shape, DataType.FP16, memref, tile_view, ir.Mem.Left)
```

## 类型系统总结

| 类型 | 维度 | 内存信息 | 使用场景 |
| ---- | ---- | -------- | -------- |
| **ScalarType** | 0 | - | 单个值 |
| **TensorType** | N（任意） | 可选 MemRef | 通用张量 |
| **TileType** | N（任意）* | 可选 MemRef + TileView | 硬件优化 Tile |
| **TupleType** | - | - | 多返回值 |
| **PipeType** | - | - | 硬件同步 |
| **UnknownType** | - | - | 类型推断占位符 |

## 常用模式

**创建常量：**

```python
i32 = ir.ConstInt(42, DataType.INT32, span)
f32 = ir.ConstFloat(3.14, DataType.FP32, span)
```

**创建运算符：**

```python
# High-level API (recommended)
call = ir.op.tensor.matmul(a, b, out_dtype=DataType.FP32)

# Generic operator with kwargs
call = ir.create_op_call("tensor.matmul", [a, b], {"out_dtype": DataType.FP32}, span)
```

**语句序列：**

```python
seq = ir.SeqStmts([stmt1, stmt2, stmt3], span)
```

## 类型检查与转换

```python
# Check expression types
if isinstance(expr, ir.Var):
    print(expr.name_)

# Check type objects
if isinstance(type_obj, ir.TileType):
    # Access tile-specific properties
    shape = type_obj.shape
```

## 相关文档

- [IR 概述](00-overview.md) - 核心概念与设计原则
- [IR 节点层次结构](01-hierarchy.md) - 完整节点类型参考
- [结构比较](03-structural_comparison.md) - 相等性和哈希工具

## 总结

PyPTO 的类型系统提供：

- **标量类型** 用于原始值
- **张量/Tile 类型** 用于带内存布局的多维数据
- **元组类型** 用于异构集合
- **流水线类型** 用于硬件同步

IR 构建 API 支持：

- 通过共享指针创建不可变节点
- 带编译时检查的类型安全操作
- 通过 MemRef 和 TileView 实现硬件感知的内存管理
- 通过 GlobalVar 实现程序内函数调用
- 通过 IterArg 实现循环携带依赖
