# Operator Registration System

Type-safe operator definitions with automatic type deduction using a fluent API.

## Operator Categories

| Category | Types | Use Case | See Also |
|----------|-------|----------|----------|
| **TensorOp** | TensorType | N-D tensor operations with broadcasting | [06-operator_organization.md](06-operator_organization.md) |
| **BlockOp** | TileType | Hardware-optimized block operations (max 2D) | [06-operator_organization.md](06-operator_organization.md) |
| **SyncOp** | UnknownType/PipeType | Pipeline barriers and synchronization | [SyncOp Operations](#syncop-operations) |

**Key Features**: Fluent API, automatic type deduction, kwargs for metadata, NumPy-style broadcasting, type promotion, dynamic dimensions (`kDynamicDim`)

## Type System

```cpp
// TensorType: N-dimensional tensors
TensorType(DataType::FP32, {dim1, dim2, dim3, ...})

// TileType: Max 2D for hardware optimization
TileType(DataType::FP16, {dim1, dim2})  // 2D OK
TileType(DataType::FP16, {dim1})        // 1D OK
TileType(DataType::FP16, {d1, d2, d3})  // Error: too many dimensions

// Dynamic dimensions (pypto/core/common.h)
constexpr int64_t kDynamicDim = -1;
auto dynamic_dim = make_int(kDynamicDim);
```

## REGISTER_OP Fluent API

| Method | Purpose | Example |
|--------|---------|---------|
| `set_op_category(str)` | Operator category | `.set_op_category("TensorOp")` |
| `set_description(str)` | Human-readable description | `.set_description("Element-wise add")` |
| `add_argument(name, desc)` | Positional Expr argument | `.add_argument("lhs", "Left tensor")` |
| `no_argument()` | No arguments (sync ops) | `.no_argument()` |
| `set_attr<T>(name)` | Kwarg schema (T: bool, int, DataType, etc.) | `.set_attr<bool>("a_trans")` |
| `set_pipe(PipeType)` | Hardware pipeline type | `.set_pipe(PipeType::S)` |
| `f_deduce_type(fn)` | Type deduction function | `.f_deduce_type(DeduceAddType)` |

**Type Deduction Signature:**
```cpp
std::function<TypePtr(const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs)>
```

## C++ Registration Examples

### Simple Elementwise Operator

```cpp
// src/ir/op/tensor_ops/elementwise.cpp
REGISTER_OP("tensor.add")
    .set_op_category("TensorOp")
    .add_argument("lhs", "Left tensor")
    .add_argument("rhs", "Right tensor")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 2);
      auto t1 = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
      auto t2 = std::dynamic_pointer_cast<const TensorType>(args[1]->GetType());
      auto dtype = PromoteDataTypes(t1->dtype_, t2->dtype_);
      auto shape = BroadcastShapes(t1->shape_, t2->shape_);
      return std::make_shared<TensorType>(shape.shape, *dtype);
    });
```

### Operator with Kwargs

```cpp
// src/ir/op/tensor_ops/matmul.cpp
TypePtr DeduceMatMul(const std::vector<ExprPtr>& args,
                     const std::vector<std::pair<std::string, std::any>>& kwargs) {
  auto lhs = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
  auto rhs = std::dynamic_pointer_cast<const TensorType>(args[1]->GetType());

  auto get = [&](const std::string& k, bool d) {
    auto it = kwargs.find(k);
    return (it != kwargs.end()) ? std::any_cast<bool>(it->second) : d;
  };

  auto it = kwargs.find("out_dtype");
  DataType dtype = (it != kwargs.end()) ? static_cast<DataType>(std::any_cast<int>(it->second))
                                        : *PromoteDataTypes(lhs->dtype_, rhs->dtype_);

  bool a_t = get("a_trans", false), b_t = get("b_trans", false);
  ExprPtr m = a_t ? lhs->shape_[1] : lhs->shape_[0];
  ExprPtr n = b_t ? rhs->shape_[0] : rhs->shape_[1];
  return std::make_shared<TensorType>(std::vector<ExprPtr>{m, n}, dtype);
}

REGISTER_OP("tensor.matmul")
    .set_op_category("TensorOp")
    .add_argument("lhs", "Left matrix")
    .add_argument("rhs", "Right matrix")
    .set_attr<DataType>("out_dtype")
    .set_attr<bool>("a_trans")
    .set_attr<bool>("b_trans")
    .f_deduce_type(DeduceMatMul);
```

## Python Usage

### Creating and Using Operators

```python
from pypto.pypto_core import DataType, ir
from pypto.ir import op

span = ir.Span.unknown()
dim4, dim8 = ir.ConstInt(4, DataType.INT32, span), ir.ConstInt(8, DataType.INT32, span)

# Create tensors
tensor_a = ir.Var("a", ir.TensorType([dim4, dim8], DataType.FP32), span)
tensor_b = ir.Var("b", ir.TensorType([dim8], DataType.FP32), span)

# Simple operators
result = op.tensor.add(tensor_a, tensor_b)  # Broadcasting: [4,8] + [8] → [4,8]

# Operators with kwargs
a = ir.Var("a", ir.TensorType([dim64, dim128], DataType.FP16), span)
b = ir.Var("b", ir.TensorType([dim128, dim64], DataType.FP16), span)
matmul = op.tensor.matmul(a, b, out_dtype=DataType.FP32, a_trans=True)
# Prints: tensor.matmul(a, b, a_trans=True, out_dtype=51)

# Query registry
assert ir.is_op_registered("tensor.add")
op_instance = ir.get_op("tensor.add")
```

## Kwargs (Keyword Arguments)

Call expressions separate Expr arguments from metadata parameters using kwargs.

### Kwargs vs Args vs Attributes

| | **Args** | **Kwargs** | **Op Attributes** |
|---|----------|------------|-------------------|
| **Type** | `ExprPtr` | `std::any` | Type-erased |
| **Scope** | Per-Call | Per-Call | Global |
| **Use** | Tensors, dims, offsets | `out_dtype`, flags, modes | Device, category |
| **Access** | `call.args_` | `call.kwargs_` | `op.get_attr()` |

### C++ - Reading Kwargs

```cpp
TypePtr DeduceCastType(const std::vector<ExprPtr>& args,
                       const std::vector<std::pair<std::string, std::any>>& kwargs) {
  auto input = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());

  // Required kwarg
  auto it = kwargs.find("target_type");
  CHECK(it != kwargs.end()) << "tensor.cast requires 'target_type'";
  DataType target = static_cast<DataType>(std::any_cast<int>(it->second));

  // Optional with default
  int mode = 0;
  auto mode_it = kwargs.find("mode");
  if (mode_it != kwargs.end()) mode = std::any_cast<int>(mode_it->second);

  return std::make_shared<TensorType>(input->shape_, target);
}
```

### Python - Using Kwargs

```python
result = op.tensor.matmul(a, b, out_dtype=DataType.FP32, a_trans=True)
print(result.kwargs)  # {'out_dtype': 51, 'a_trans': True}
```

### Defining Kwarg Schema

Use `set_attr<T>()` to define allowed kwargs. Types: `bool`, `int`, `std::string`, `double`, `DataType`.

```cpp
REGISTER_OP("tensor.matmul")
    .set_attr<DataType>("out_dtype")
    .set_attr<bool>("a_trans")
    .f_deduce_type(DeduceMatMulType);

// Python: query schema
op = ir.get_op("tensor.matmul")
keys = op.get_attr_keys()  # ['out_dtype', 'a_trans']
```

## Broadcasting and Type Promotion

### NumPy-style Broadcasting

Dimensions aligned right to left:
```
[4, 8] + [4, 8] → [4, 8]  # Exact match
[4, 8] + [8]    → [4, 8]  # Missing left dimension = 1
[4, 1] + [8]    → [4, 8]  # Size 1 broadcasts
[1, 8] + [4, 8] → [4, 8]  # Size 1 broadcasts
[4, 8] + [5]    → Error   # 8 ≠ 5
```

### Type Promotion

Standard numeric rules: float > int, larger > smaller, signed > unsigned (same size).

```
INT32 + INT32 → INT32
INT32 + FP32  → FP32   (float precedence)
INT32 + INT64 → INT64  (larger size)
UINT32 + INT32 → INT32 (signed precedence)
```

## SyncOp Operations

Hardware synchronization and barriers. Return `UnknownType`. Use with `ib.emit()`.

| Operation | Description | Kwargs |
|-----------|-------------|--------|
| `system.bar_all` | Global barrier | None |
| `system.bar_v` | Vector barrier | None |
| `system.bar_m` | Matrix barrier | None |
| `system.sync_src` | Set sync flag | `set_pipe`, `wait_pipe`, `event_id` |
| `system.sync_dst` | Wait sync flag | `set_pipe`, `wait_pipe`, `event_id` |

**Python Example:**
```python
from pypto.ir.op import system
ib.emit(system.bar_all())
ib.emit(system.sync_src(set_pipe=2, wait_pipe=4, event_id=0))
```

**C++ Registration (`src/ir/op/sync_ops/sync.cpp`):**
```cpp
REGISTER_OP("system.bar_all")
    .set_op_category("SyncOp")
    .set_pipe(PipeType::S)
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

REGISTER_OP("system.sync_src")
    .set_op_category("SyncOp")
    .set_pipe(PipeType::S)
    .set_attr<int>("set_pipe")
    .set_attr<int>("wait_pipe")
    .set_attr<int>("event_id")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);
```

## Adding New Operations

1. **Choose category file**: `src/ir/op/tensor_ops/elementwise.cpp`, `matmul.cpp`, `reduction.cpp`, or `src/ir/op/block_ops/memory.cpp`, `unary.cpp`

2. **Implement type deduction**:
   ```cpp
   TypePtr DeduceType(const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
     CHECK(args.size() == 2) << "op requires 2 arguments";
     // Validate types, read kwargs, compute output type
     return result_type;
   }
   ```

3. **Register**:
   ```cpp
   REGISTER_OP("tensor.matmul")
       .set_op_category("TensorOp")
       .add_argument("lhs", "Left tensor")
       .add_argument("rhs", "Right tensor")
       .set_attr<DataType>("out_dtype")
       .f_deduce_type(DeduceType);
   ```

4. **Python wrapper** (`python/pypto/ir/op/tensor_ops.py`):
   ```python
   def matmul(lhs: Expr, rhs: Expr, out_dtype=None, a_trans=False) -> Call:
       kwargs = {}
       if out_dtype: kwargs["out_dtype"] = out_dtype.code() if isinstance(out_dtype, DataType) else out_dtype
       if a_trans: kwargs["a_trans"] = a_trans
       return _ir_core.create_op_call("tensor.matmul", [lhs, rhs], kwargs, Span.unknown())
   ```

5. **Add tests** in `tests/ut/ir/` and update `CMakeLists.txt` if needed

## References

- Common constants: `include/pypto/core/common.h`
- Type definitions: `include/pypto/ir/type.h`
- Operator registry: `include/pypto/ir/op_registry.h`
- Type inference utilities: `include/pypto/ir/type_inference.h`
- Type inference implementation: `src/ir/op/type_inference.cpp`
- Operator registry implementation: `src/ir/op_registry.cpp`
- Tensor operator implementations: `src/ir/op/tensor_ops/`
- Block operator implementations: `src/ir/op/block_ops/`
- Sync operator implementations: `src/ir/op/sync_ops/`
