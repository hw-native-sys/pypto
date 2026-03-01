# PyPTO IR Overview

## Overview

PyPTO's Intermediate Representation (IR) is a tree-based, immutable data structure used to represent programs during compilation. The IR serves as the foundation for program transformation, optimization, and code generation.

**Key Design Principles:**

1. **Immutability**: All IR nodes are immutable once constructed
2. **Tree Structure**: Forms a DAG where nodes can be shared across multiple parents
3. **Shared Pointers**: All nodes managed through `std::shared_ptr<const T>`
4. **Reference Equality**: Default `==` compares pointer addresses; use `structural_equal()` for structural comparison

## Core Concepts

### Source Location Tracking

Every IR node contains a `Span` object tracking its source location:

```python
from pypto import ir

# Create a span for source location tracking
span = ir.Span("example.py", 10, 5, 10, 20)
print(span.filename)      # "example.py"
print(span.begin_line)    # 10

# Create unknown span when location unavailable
unknown_span = ir.Span.unknown()
```

### Field Descriptors and Reflection

IR nodes use a reflection system for generic traversal. Each node defines three types of fields:

| Field Type | Purpose | Example Usage |
| ---------- | ------- | ------------- |
| **IgnoreField** | Ignored during traversal | `Span` (source location) |
| **DefField** | Definition fields introducing new bindings | Loop variables, assignment targets |
| **UsualField** | Regular fields traversed normally | Expression operands, statement bodies |

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

## Type Identification with Kind Mechanism

PyPTO IR uses an efficient **Kind-based type identification mechanism** to avoid the overhead of C++ RTTI (`dynamic_cast`). This provides O(1) type checking and casting with zero runtime overhead.

### ObjectKind Enumeration

All IR node types are represented in a unified enumeration:

| Category | Kinds |
| -------- | ----- |
| **Base** | IRNode, Expr, Stmt, Type |
| **Expressions** | Var, IterArg, Call, TupleGetItemExpr, ConstInt, ConstFloat, ConstBool |
| **Binary Ops** | Add, Sub, Mul, FloorDiv, FloorMod, FloatDiv, Min, Max, Pow, Eq, Ne, Lt, Le, Gt, Ge, And, Or, Xor, BitAnd, BitOr, BitXor, BitShiftLeft, BitShiftRight |
| **Unary Ops** | Abs, Neg, Not, BitNot, Cast |
| **Statements** | AssignStmt, IfStmt, YieldStmt, ReturnStmt, ForStmt, SeqStmts, OpStmts, EvalStmt |
| **Types** | UnknownType, ScalarType, ShapedType, TensorType, TileType, TupleType, PipeType |
| **Other** | Function, Program, Op, GlobalVar |

### GetKind() Virtual Method

Every IR node implements the `GetKind()` method:

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

### Type Checking with `IsA<T>()`

Use `IsA<T>()` to check if a node is of a specific type:

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

### Type Casting with `As<T>()`

Use `As<T>()` to safely cast nodes to their concrete types:

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

**Key Benefits:**

- **O(1) Performance**: Single virtual function call vs. multiple `dynamic_cast` attempts
- **Type Safe**: Returns `nullptr` on failed cast, no exceptions
- **Clean Syntax**: `IsA<T>()` and `As<T>()` are more readable than `dynamic_pointer_cast`
- **Zero Overhead**: Compiler can optimize away virtual calls in many cases

## IRNode - Base Class

```cpp
class IRNode {
  Span span_;                           // Source location (IgnoreField)
  virtual ObjectKind GetKind() const;   // Returns node's kind for O(1) type checking
  virtual std::string TypeName() const; // Returns node type name (for debugging)
};
```

All IR nodes inherit from `IRNode` and must implement:

- `GetKind()`: Returns the node's `ObjectKind` for type identification
- `TypeName()`: Returns a human-readable type name (e.g., "Var", "AssignStmt")

## Expression Base Class

```cpp
class Expr : public IRNode {
  TypePtr type_;  // Result type of the expression
};
```

All expressions produce a value with an associated type.

## Statement Base Class

```cpp
class Stmt : public IRNode {
  // Statements represent actions but don't produce values
};
```

Statements represent program actions like assignments, control flow, and loops.

## Type Base Class

```cpp
class Type : public IRNode {
  // Base class for all type representations
};
```

Types describe the structure and properties of data in the IR.

## Python Usage Pattern

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

## Design Philosophy

**Immutability Benefits:**

- Thread-safe sharing across transformations
- Structural sharing reduces memory usage
- Safer reasoning about program semantics

**Kind Mechanism Benefits:**

- Fast type checks without RTTI overhead
- Enables efficient visitor patterns
- Supports generic transformations and analyses

**Reflection System Benefits:**

- Generic tree traversal without code duplication
- Structural equality and hashing
- Pretty printing and serialization

## Related Documentation

- [IR Node Hierarchy](01-ir_hierarchy.md) - Complete node type reference
- [IR Types and Examples](02-types.md) - Type system and usage examples
- [Structural Comparison](03-structural_comparison.md) - Equality and hashing utilities

## Summary

PyPTO IR provides:

- **Immutable tree structure** for safe transformations
- **Efficient type identification** via Kind mechanism with O(1) performance
- **Reflection-based traversal** enabling visitors, mutators, and structural comparison
- **Python-friendly API** for IR construction
- **Source location tracking** for error reporting
- **Three-tier field system** (Ignore, Def, Usual) for flexible traversal
