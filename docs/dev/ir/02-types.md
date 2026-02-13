# PyPTO IR Types and Examples

This document covers the type system and provides practical usage examples.

## Type System

### ScalarType

Represents primitive scalar types.

```python
from pypto import DataType, ir

int_type = ir.ScalarType(DataType.INT64)
float_type = ir.ScalarType(DataType.FP32)
```

**Supported DataTypes:** INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FP16, FP32, FP64, BOOL

### TensorType

Multi-dimensional tensor with optional memory reference.

```python
# Tensor with shape [10, 20]
shape = [ir.ConstInt(10, DataType.INT64, span), ir.ConstInt(20, DataType.INT64, span)]
tensor_type = ir.TensorType(shape, DataType.FP32)

# Tensor with MemRef
memref = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0x1000, DataType.INT64, span), 800)
tensor_with_memref = ir.TensorType(shape, DataType.FP32, memref)
```

### TensorType with TensorView

Tensor with layout and stride information for optimized memory access.

```python
# Create tensor with tensor view
shape = [ir.ConstInt(128, DataType.INT64, span), ir.ConstInt(256, DataType.INT64, span)]
stride = [ir.ConstInt(1, DataType.INT64, span), ir.ConstInt(128, DataType.INT64, span)]

tensor_view = ir.TensorView(stride, ir.TensorLayout.ND)
tensor_with_view = ir.TensorType(shape, DataType.FP32, memref=None, tensor_view=tensor_view)

# Different layouts
nd_view = ir.TensorView(stride, ir.TensorLayout.ND)  # ND layout
dn_view = ir.TensorView(stride, ir.TensorLayout.DN)  # DN layout
nz_view = ir.TensorView(stride, ir.TensorLayout.NZ)  # NZ layout

# Tensor with both MemRef and TensorView
memref = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0x2000, DataType.INT64, span), 16384)
tensor_with_both = ir.TensorType(shape, DataType.FP16, memref=memref, tensor_view=tensor_view)
```

**TensorLayout values:**
- `ND`: ND layout
- `DN`: DN layout
- `NZ`: NZ layout

### TileType

Specialized tensor with optional memory and view information for hardware-optimized operations.

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
memref = ir.MemRef(ir.MemorySpace.L0A, ir.ConstInt(0, DataType.INT64, span), 512)

tile_view = ir.TileView()
tile_view.valid_shape = [ir.ConstInt(16, DataType.INT64, span)] * 2
tile_view.stride = [ir.ConstInt(1, DataType.INT64, span), ir.ConstInt(16, DataType.INT64, span)]
tile_view.start_offset = ir.ConstInt(0, DataType.INT64, span)

tile_with_view = ir.TileType(shape, DataType.FP16, memref, tile_view)
```

### TupleType

Heterogeneous tuple of types.

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

Hardware execution pipelines or synchronization barriers.

```python
pipe_s = ir.PipeType(ir.PipeType.S)    # Scalar pipe
pipe_v = ir.PipeType(ir.PipeType.V)    # Vector pipe
pipe_m = ir.PipeType(ir.PipeType.M)    # Matrix pipe
pipe_all = ir.PipeType(ir.PipeType.ALL) # All pipes
```

### UnknownType

Placeholder for unknown or inferred types.

```python
unknown = ir.UnknownType()
```

### MemorySpace Enum

| Value | Description |
|-------|-------------|
| `DDR` | Main memory (off-chip) |
| `UB` | Unified Buffer (on-chip shared memory) |
| `L1` | L1 cache |
| `L0A` | L0A buffer (matrix A) |
| `L0B` | L0B buffer (matrix B) |
| `L0C` | L0C buffer (matrix C/result) |

## Python Usage Examples

### Example 1: Building Expressions

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

### Example 2: Control Flow (Absolute Value)

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

### Example 3: Loop with Accumulation

```python
# for i, (sum,) in pl.range(0, n, 1, init_values=(0,)):
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

### Example 4: Function with Operator Calls

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

### Example 5: Program with Multiple Functions

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

### Example 6: Memory Layout with TileType

```python
# 32x32 tile in L0A memory with custom stride
shape = [ir.ConstInt(32, DataType.INT64, span)] * 2
memref = ir.MemRef(ir.MemorySpace.L0A, ir.ConstInt(0, DataType.INT64, span), 2048)

tile_view = ir.TileView()
tile_view.valid_shape = shape
tile_view.stride = [ir.ConstInt(1, DataType.INT64, span), ir.ConstInt(32, DataType.INT64, span)]
tile_view.start_offset = ir.ConstInt(0, DataType.INT64, span)

tile_type = ir.TileType(shape, DataType.FP16, memref, tile_view)
```

## Type System Summary

| Type | Dimensions | Memory Info | Use Case |
|------|------------|-------------|----------|
| **ScalarType** | 0 | - | Single values |
| **TensorType** | N (any) | Optional MemRef | General tensors |
| **TileType** | N (any)* | Optional MemRef + TileView | Hardware-optimized tiles |
| **TupleType** | - | - | Multiple return values |
| **PipeType** | - | - | Hardware synchronization |
| **UnknownType** | - | - | Type inference placeholder |

## Common Patterns

**Creating constants:**
```python
i32 = ir.ConstInt(42, DataType.INT32, span)
f32 = ir.ConstFloat(3.14, DataType.FP32, span)
```

**Creating operators:**
```python
# High-level API (recommended)
call = ir.op.tensor.matmul(a, b, out_dtype=DataType.FP32)

# Generic operator with kwargs
call = ir.create_op_call("tensor.matmul", [a, b], {"out_dtype": DataType.FP32}, span)
```

**Statement sequences:**
```python
seq = ir.SeqStmts([stmt1, stmt2, stmt3], span)
```

## Type Checking and Casting

```python
# Check expression types
if isinstance(expr, ir.Var):
    print(expr.name_)

# Check type objects
if isinstance(type_obj, ir.TileType):
    # Access tile-specific properties
    shape = type_obj.shape
```

## Related Documentation

- [IR Overview](00-overview.md) - Core concepts and design principles
- [IR Node Hierarchy](01-ir_hierarchy.md) - Complete node type reference
- [Structural Comparison](03-structural_comparison.md) - Equality and hashing utilities

## Summary

PyPTO's type system provides:
- **Scalar types** for primitive values
- **Tensor/Tile types** for multi-dimensional data with memory layout
- **Tuple types** for heterogeneous collections
- **Pipe types** for hardware synchronization

The IR construction API supports:
- Immutable node creation with shared pointers
- Type-safe operations with compile-time checking
- Hardware-aware memory management via MemRef and TileView
- Intra-program function calls via GlobalVar
- Loop-carried dependencies via IterArg
