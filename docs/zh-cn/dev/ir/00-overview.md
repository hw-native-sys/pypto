# PyPTO 中间表示 (IR) 概述

## 概述

PyPTO 的中间表示 (IR) 是一种基于树的不可变数据结构，用于在编译过程中表示程序。IR 是程序变换、优化和代码生成 (CodeGen) 的基础。

**核心设计原则：**

1. **不可变性**：所有 IR 节点一旦构造即不可变
2. **树结构**：形成有向无环图 (DAG)，节点可在多个父节点之间共享
3. **共享指针**：所有节点通过 `std::shared_ptr<const T>` 管理
4. **引用相等性**：默认 `==` 比较指针地址；使用 `structural_equal()` 进行结构比较

## 核心概念

### 源位置追踪

每个 IR 节点都包含一个 `Span` 对象，用于追踪其源位置：

```python
from pypto import ir

# Create a span for source location tracking
span = ir.Span("example.py", 10, 5, 10, 20)
print(span.filename)      # "example.py"
print(span.begin_line)    # 10

# Create unknown span when location unavailable
unknown_span = ir.Span.unknown()
```

### 字段描述符与反射

IR 节点使用反射系统进行通用遍历。每个节点定义三种类型 (Type) 的字段：

| 字段类型 | 用途 | 示例用法 |
| -------- | ---- | -------- |
| **IgnoreField** | 遍历时忽略 | `Span`（源位置） |
| **DefField** | 引入新绑定的定义字段 | 循环变量、赋值目标 |
| **UsualField** | 正常遍历的常规字段 | 表达式 (Expression) 操作数、语句 (Statement) 主体 |

```cpp
// Example: AssignStmt field descriptors
static constexpr auto GetFieldDescriptors() {
  return std::tuple_cat(
    Stmt::GetFieldDescriptors(),
    std::make_tuple(
      reflection::DefField(&AssignStmt::var_, "var"),      // Definition
      reflection::UsualField(&AssignStmt::value_, "value") // Normal field
    )
  );
}
```

## Kind 机制的类型识别

PyPTO IR 使用高效的 **基于 Kind 的类型识别机制**，避免 C++ RTTI（`dynamic_cast`）的开销。提供 O(1) 的类型检查和转换，零运行时开销。

### ObjectKind 枚举

所有 IR 节点类型在统一枚举中表示：

| 类别 | Kinds |
| ---- | ----- |
| **基类** | IRNode, Expr, Stmt, Type |
| **表达式** | Var, IterArg, Call, TupleGetItemExpr, ConstInt, ConstFloat, ConstBool |
| **二元运算** | Add, Sub, Mul, FloorDiv, FloorMod, FloatDiv, Min, Max, Pow, Eq, Ne, Lt, Le, Gt, Ge, And, Or, Xor, BitAnd, BitOr, BitXor, BitShiftLeft, BitShiftRight |
| **一元运算** | Abs, Neg, Not, BitNot, Cast |
| **语句** | AssignStmt, IfStmt, YieldStmt, ReturnStmt, ForStmt, SeqStmts, OpStmts, EvalStmt |
| **类型** | UnknownType, ScalarType, ShapedType, TensorType, TileType, TupleType, PipeType |
| **其他** | Function, Program, Op, GlobalVar |

### GetKind() 虚方法

每个 IR 节点都实现 `GetKind()` 方法：

```cpp
class IRNode {
 public:
  [[nodiscard]] virtual ObjectKind GetKind() const = 0;
};

class Var : public Expr {
 public:
  [[nodiscard]] ObjectKind GetKind() const override {
    return ObjectKind::Var;
  }
};
```

### 使用 `IsA<T>()` 进行类型检查

使用 `IsA<T>()` 检查节点是否为特定类型：

```cpp
#include "pypto/ir/kind_traits.h"

ExprPtr expr = ...;

// Check if expr is a Var
if (IsA<Var>(expr)) {
  // expr is a Var
}

// Check if expr is a ConstInt
if (IsA<ConstInt>(expr)) {
  // expr is a ConstInt
}

// Works with TypePtr too
TypePtr type = expr->GetType();
if (IsA<TileType>(type)) {
  // type is a TileType
}
```

### 使用 `As<T>()` 进行类型转换

使用 `As<T>()` 安全地将节点转换为具体类型：

```cpp
#include "pypto/ir/kind_traits.h"

ExprPtr expr = ...;

// Cast to Var (returns nullptr if not a Var)
if (auto var = As<Var>(expr)) {
  std::cout << "Variable name: " << var->name_ << std::endl;
}

// Cast ConstInt
if (auto const_int = As<ConstInt>(expr)) {
  std::cout << "Integer value: " << const_int->value_ << std::endl;
}

// Type casting
TypePtr type = expr->GetType();
if (auto tile_type = As<TileType>(type)) {
  // Access tile-specific properties
  auto shape = tile_type->GetShape();
}
```

**核心优势：**

- **O(1) 性能**：单次虚函数调用，无需多次 `dynamic_cast` 尝试
- **类型安全**：转换失败返回 `nullptr`，不抛出异常
- **简洁语法**：`IsA<T>()` 和 `As<T>()` 比 `dynamic_pointer_cast` 更具可读性
- **零开销**：编译器在很多情况下可以优化掉虚调用

## IRNode - 基类

```cpp
class IRNode {
  Span span_;                           // Source location (IgnoreField)
  virtual ObjectKind GetKind() const;   // Returns node's kind for O(1) type checking
  virtual std::string TypeName() const; // Returns node type name (for debugging)
};
```

所有 IR 节点继承自 `IRNode`，且必须实现：

- `GetKind()`：返回节点的 `ObjectKind`，用于类型识别
- `TypeName()`：返回人类可读的类型名称（如 "Var"、"AssignStmt"）

## 表达式基类

```cpp
class Expr : public IRNode {
  TypePtr type_;  // Result type of the expression
};
```

所有表达式都会产生一个带有关联类型的值。

## 语句基类

```cpp
class Stmt : public IRNode {
  // Statements represent actions but don't produce values
};
```

语句表示程序行为，如赋值、控制流和循环。

## 类型基类

```cpp
class Type : public IRNode {
  // Base class for all type representations
};
```

类型描述 IR 中数据的结构和属性 (Property)。

## Python 使用模式

```python
from pypto import DataType, ir

# Create basic IR nodes
span = ir.Span.unknown()
dtype = DataType.INT64

# Variables
x = ir.Var("x", ir.ScalarType(dtype), span)
y = ir.Var("y", ir.ScalarType(dtype), span)

# Constants
one = ir.ConstInt(1, dtype, span)
pi = ir.ConstFloat(3.14, DataType.FP32, span)
flag = ir.ConstBool(True, span)

# Expressions
sum_expr = ir.Add(x, one, dtype, span)
product = ir.Mul(x, y, dtype, span)

# Statements
assign = ir.AssignStmt(x, sum_expr, span)
```

## 设计哲学

**不可变性的优势：**

- 跨变换的线程安全共享
- 结构共享减少内存使用
- 更安全的程序语义推理

**Kind 机制的优势：**

- 无需 RTTI 开销的快速类型检查
- 支持高效的访问者模式
- 支持通用变换和分析

**反射系统的优势：**

- 无需代码重复的通用树遍历
- 结构相等性和哈希计算
- 美化打印和序列化

## 相关文档

- [IR 节点层次结构](01-hierarchy.md) - 完整节点类型参考
- [IR 类型与示例](02-types.md) - 类型系统与使用示例
- [结构比较](03-structural_comparison.md) - 相等性和哈希工具

## 总结

PyPTO IR 提供：

- **不可变树结构** 用于安全变换
- **高效类型识别** 通过 Kind 机制实现 O(1) 性能
- **基于反射的遍历** 支持访问者、变换器和结构比较
- **Python 友好的 API** 用于 IR 构建
- **源位置追踪** 用于错误报告
- **三层字段系统**（Ignore、Def、Usual）用于灵活遍历
