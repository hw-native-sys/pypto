# IR 构建器 (Builder)

IR 构建器 (Builder) 提供了一套便捷的 API，用于使用上下文管理器（Python）或 Begin/End 模式（C++）增量构建 PyPTO 中间表示 (IR)。它管理上下文栈、验证构建过程并跟踪源码位置。

## 概述

### 主要特性

- **上下文管理**：基于栈的跟踪确保正确嵌套
- **自动 Span 跟踪**（Python）：使用 `inspect` 模块
- **显式 Span 参数**（C++）：所有方法接受 span 参数
- **验证**：检查上下文使用和结构的正确性
- **嵌套构造**：支持函数中的循环、循环中的 if 等

## Python API

使用上下文管理器（`with` 语句 (Statement)）提供简洁的接口。

### 基本函数

```python
from pypto import ir, DataType
from pypto.ir import IRBuilder

ib = IRBuilder()

with ib.function("add") as f:
    x = f.param("x", ir.ScalarType(DataType.INT64))
    y = f.param("y", ir.ScalarType(DataType.INT64))
    f.return_type(ir.ScalarType(DataType.INT64))

    result = ib.var("result", ir.ScalarType(DataType.INT64))
    add_expr = ir.Add(x, y, DataType.INT64, ir.Span.unknown())
    ib.assign(result, add_expr)

func = f.get_result()

# With function type
with ib.function("orchestrator", type=ir.FunctionType.Orchestration) as f:
    n = f.param("n", ir.ScalarType(DataType.INT64))
    f.return_type(ir.ScalarType(DataType.INT64))
    # ... function body

func_orch = f.get_result()
```

### 带迭代参数的 For 循环

```python
with ib.function("sum_to_n") as f:
    n = f.param("n", ir.ScalarType(DataType.INT64))
    f.return_type(ir.ScalarType(DataType.INT64))

    i = ib.var("i", ir.ScalarType(DataType.INT64))
    start = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
    step = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
    init_val = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())

    with ib.for_loop(i, start, n, step) as loop:
        sum_iter = loop.iter_arg("sum", init_val)
        sum_final = loop.return_var("sum_final")

        add_expr = ir.Add(sum_iter, i, DataType.INT64, ir.Span.unknown())
        ib.emit(ir.YieldStmt([add_expr], ir.Span.unknown()))

    result = loop.output()  # Get first return variable

func = f.get_result()
```

### If 语句

```python
with ib.function("max") as f:
    x = f.param("x", ir.ScalarType(DataType.INT64))
    y = f.param("y", ir.ScalarType(DataType.INT64))
    f.return_type(ir.ScalarType(DataType.INT64))

    condition = ir.Gt(x, y, DataType.INT64, ir.Span.unknown())

    with ib.if_stmt(condition) as if_builder:
        if_builder.return_var("phi_result", ir.ScalarType(DataType.INT64))
        ib.emit(ir.YieldStmt([x], ir.Span.unknown()))

        if_builder.else_()
        ib.emit(ir.YieldStmt([y], ir.Span.unknown()))

    result = if_builder.output()

func = f.get_result()
```

### 访问返回变量

`ForLoopBuilder` 和 `IfStmtBuilder` 都提供了便捷的访问方法：

| 方法 | 描述 | 示例 |
| ---- | ---- | ---- |
| `output(index=0)` | 按索引获取单个返回变量 | `result = loop.output()` |
| `outputs()` | 获取所有返回变量的列表 | `sum_result, prod_result = loop.outputs()` |

### Return 语句

```python
with ib.function("add_and_return") as f:
    x = f.param("x", ir.ScalarType(DataType.INT64))
    y = f.param("y", ir.ScalarType(DataType.INT64))
    f.return_type(ir.ScalarType(DataType.INT64))

    add_expr = ir.Add(x, y, DataType.INT64, ir.Span.unknown())
    ib.return_stmt(add_expr)  # Single value
    # ib.return_stmt([x, y])  # Multiple values
    # ib.return_stmt()        # Empty return

func = f.get_result()
```

## C++ API

使用 Begin/End 方法和显式 span 参数。

### 示例：带 For 循环的函数

```cpp
#include "pypto/ir/builder.h"
using namespace pypto::ir;

IRBuilder ib;
auto here = [](int line) { return Span(__FILE__, line, 0); };

// Begin function (with optional type parameter)
ib.BeginFunction("sum_to_n", here(__LINE__), FunctionType::Opaque);
auto n = ib.FuncArg("n", std::make_shared<ScalarType>(DataType::INT64), here(__LINE__));
ib.ReturnType(std::make_shared<ScalarType>(DataType::INT64));

// Begin for loop
auto i = ib.Var("i", std::make_shared<ScalarType>(DataType::INT64), here(__LINE__));
auto start = std::make_shared<ConstInt>(0, DataType::INT64, here(__LINE__));
auto step = std::make_shared<ConstInt>(1, DataType::INT64, here(__LINE__));
ib.BeginForLoop(i, start, n, step, here(__LINE__));

// Add iter_arg and return_var
auto init_val = std::make_shared<ConstInt>(0, DataType::INT64, here(__LINE__));
auto sum_iter = std::make_shared<IterArg>("sum", std::make_shared<ScalarType>(DataType::INT64),
                                          init_val, here(__LINE__));
ib.AddIterArg(sum_iter);
ib.AddReturnVar(ib.Var("sum_final", std::make_shared<ScalarType>(DataType::INT64), here(__LINE__)));

// Loop body
auto add_expr = std::make_shared<Add>(sum_iter, i, DataType::INT64, here(__LINE__));
ib.Emit(std::make_shared<YieldStmt>(std::vector<ExprPtr>{add_expr}, here(__LINE__)));

// End loop and function
ib.EndForLoop(here(__LINE__));
ib.Return(std::vector<ExprPtr>{...}, here(__LINE__));
auto func = ib.EndFunction(here(__LINE__));
```

**与 Python API 的主要差异：**

- 使用 `BeginFunction`/`EndFunction` 代替 `with` 语句
- 使用 `BeginForLoop`/`EndForLoop` 处理循环
- 使用 `BeginIf`/`EndIf` 处理 if 语句
- 所有方法需要显式的 `Span` 参数

## 上下文栈与验证

### 验证规则

| 规则 | 描述 |
| ---- | ---- |
| **不允许嵌套函数** | 不能在另一个函数内调用 `BeginFunction` |
| **上下文匹配** | 必须使用正确的 End 方法结束上下文 |
| **迭代参数匹配返回变量** | For 循环中两者数量必须相等 |
| **正确嵌套** | 循环/if 必须在函数或有效上下文内 |

### 错误消息

```python
with ib.function("outer") as f:
    with ib.function("inner") as f2:  # Error!
        pass
# RuntimeError: Cannot begin function 'inner': already inside function 'outer' at file.py:10
```

### 上下文状态查询

```python
ib.in_function()  # True if inside function context
ib.in_loop()      # True if inside for loop context
ib.in_if()        # True if inside if statement context
```

```cpp
ib.InFunction()  // true if inside function
ib.InLoop()      // true if inside loop
ib.InIf()        // true if inside if
```

## 类型创建辅助方法（Python）

用于创建带有内存引用 (MemRef) 和 TileView 的类型的便捷方法。

### MemRef

```python
# Create memory reference
memref = ib.memref(
    memory_space=ir.MemorySpace.DDR,
    addr=0x1000,  # Can be int or Expr
    size=1024
)

# With symbolic address
base_addr = ib.var("base_addr", ir.ScalarType(DataType.INT64))
memref = ib.memref(ir.MemorySpace.Vec, base_addr, 2048)
```

### TileView

```python
# Integer dimensions
tile_view = ib.tile_view(
    valid_shape=[16, 16],
    stride=[1, 16],
    start_offset=0
)

# Symbolic dimensions
n = ib.var("n", ir.ScalarType(DataType.INT64))
tile_view = ib.tile_view([n, n], [1, n], 0)
```

### TensorType 和 TileType

```python
# Simple types
tensor_t = ib.tensor_type([64, 128], DataType.FP32)
tile_t = ib.tile_type([16, 16], DataType.FP16)

# With memory reference
memref = ib.memref(ir.MemorySpace.DDR, 0x1000, 8192)
tensor_t = ib.tensor_type([64, 128], DataType.FP32, memref=memref)

# Complete tile with memref and tile_view
memref = ib.memref(ir.MemorySpace.Left, 0, 512)
tile_view = ib.tile_view([16, 16], [1, 16], 0)
tile_t = ib.tile_type([16, 16], DataType.FP16, memref=memref, tile_view=tile_view)
```

### 完整示例

```python
ib = IRBuilder()

with ib.function("matmul_tile") as f:
    # Create tile types with memory references
    memref_a = ib.memref(ir.MemorySpace.Left, 0, 512)
    tile_t_a = ib.tile_type([16, 16], DataType.FP16, memref=memref_a)

    memref_b = ib.memref(ir.MemorySpace.Right, 0, 512)
    tile_t_b = ib.tile_type([16, 16], DataType.FP16, memref=memref_b)

    a = f.param("a", tile_t_a)
    b = f.param("b", tile_t_b)

    memref_c = ib.memref(ir.MemorySpace.Acc, 0, 512)
    tile_view_c = ib.tile_view([16, 16], [1, 16], 0)
    tile_t_c = ib.tile_type([16, 16], DataType.FP32, memref=memref_c, tile_view=tile_view_c)
    f.return_type(tile_t_c)

func = f.get_result()
```

## 设计原则

1. **显式 Span**：所有 IR 节点需要源码位置。Python 自动捕获；C++ 需要显式参数。
2. **不可变 IR**：Builder 创建不可变的 IR 节点。
3. **渐进构建**：逐语句增量构建 IR。
4. **上下文安全**：Builder 验证正确的嵌套和闭合。
5. **静态单赋值 (SSA) 风格**：For 循环使用迭代参数实现 SSA 风格的循环携带值。

## 测试

参见 `tests/ut/ir/test_builder.py` 和 `tests/ut/ir/test_flash_attention_builder.py` 获取完整示例。

## 实现细节

### 文件

- `include/pypto/ir/builder.h` - C++ 头文件
- `src/ir/builder.cpp` - C++ 实现
- `python/pypto/ir/builder.py` - Python 封装
- `python/bindings/modules/ir_builder.cpp` - Python 绑定

### 关键类

| 类 | 用途 |
| -- | ---- |
| **IRBuilder** | 带上下文栈的主构建器 |
| **BuildContext** | 上下文基类 |
| **FunctionContext** | 函数构建上下文 |
| **ForLoopContext** | For 循环构建上下文 |
| **IfStmtContext** | If 语句构建上下文 |
| **FunctionBuilder** | Python 函数辅助类 |
| **ForLoopBuilder** | Python 循环辅助类 |
| **IfStmtBuilder** | Python if 辅助类 |
