# PyPTO IR Definition

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [BNF Grammar](#bnf-grammar)
- [IR Node Hierarchy](#ir-node-hierarchy)
- [Type System](#type-system)
- [Python Usage Examples](#python-usage-examples)

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

1. **IgnoreField**: Ignored during traversal (e.g., `Span`)
2. **DefField**: Definition fields introducing new bindings (e.g., loop variables, assignment targets)
3. **UsualField**: Regular fields traversed normally

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

## BNF Grammar

The PyPTO IR can be described using the following BNF grammar:

```bnf
<program>    ::= <function> | <stmt>

<function>   ::= "def" identifier "(" [ <param_list> ] ")" [ "->" <type_list> ] ":" <stmt>

<param_list> ::= <var> { "," <var> }

<type_list>  ::= <type> { "," <type> }

<stmt>       ::= <assign_stmt>
               | <if_stmt>
               | <for_stmt>
               | <yield_stmt>
               | <seq_stmts>
               | <op_stmts>

<assign_stmt> ::= <var> "=" <expr>

<if_stmt>    ::= "if" <expr> ":" <stmt_list>
                 [ "else" ":" <stmt_list> ]
                 [ "return" <var_list> ]

<for_stmt>   ::= "for" <var> "in" "range" "(" <expr> "," <expr> "," <expr> ")" ":" <stmt_list>
                 [ "return" <var_list> ]

<yield_stmt> ::= "yield" [ <var_list> ]

<seq_stmts>  ::= <stmt> { ";" <stmt> }

<op_stmts>   ::= <assign_stmt> { ";" <assign_stmt> }

<stmt_list>  ::= <stmt> { <stmt> }

<var_list>   ::= <var> { "," <var> }

<expr>       ::= <var>
               | <const_int>
               | <call>
               | <binary_expr>
               | <unary_expr>

<call>       ::= <op> "(" [ <expr_list> ] ")"

<binary_expr> ::= <expr> <binary_op> <expr>

<unary_expr>  ::= <unary_op> <expr>

<expr_list>   ::= <expr> { "," <expr> }

<binary_op>   ::= "+" | "-" | "*" | "/" | "//" | "%"
                | "==" | "!=" | "<" | "<=" | ">" | ">="
                | "and" | "or" | "xor"
                | "&" | "|" | "^" | "<<" | ">>"
                | "min" | "max" | "**"

<unary_op>    ::= "-" | "abs" | "not" | "~"

<var>         ::= identifier

<const_int>   ::= integer

<op>          ::= identifier

<type>        ::= <scalar_type>
                | <tensor_type>
                | <unknown_type>

<scalar_type> ::= "ScalarType" "(" <data_type> ")"

<tensor_type> ::= "TensorType" "(" <data_type> "," <shape> ")"

<shape>       ::= "[" <expr_list> "]"

<data_type>   ::= "INT32" | "INT64" | "FLOAT32" | "FLOAT64" | ...
```

## IR Node Hierarchy

### IRNode - Base Class

```cpp
class IRNode {
  Span span_;                           // Source location
  virtual std::string TypeName() const; // Returns node type name
};
```

### Expression Hierarchy

#### Expr - Base Expression

```cpp
class Expr : public IRNode {
  TypePtr type_;  // Result type
};
```

#### Var - Variable Reference

```python
from pypto import DataType, ir

x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
```

#### ConstInt - Integer Constant

```python
c = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
```

#### BinaryExpr - Binary Operations

Available operations: `Add`, `Sub`, `Mul`, `FloorDiv`, `FloorMod`, `FloatDiv`, `Min`, `Max`, `Pow`, `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`, `And`, `Or`, `Xor`, `BitAnd`, `BitOr`, `BitXor`, `BitShiftLeft`, `BitShiftRight`

```python
# Build: (x + 5) * 2
x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
five = ir.ConstInt(5, DataType.INT64, ir.Span.unknown())
two = ir.ConstInt(2, DataType.INT64, ir.Span.unknown())
add_expr = ir.Add(x, five, DataType.INT64, ir.Span.unknown())
mul_expr = ir.Mul(add_expr, two, DataType.INT64, ir.Span.unknown())
```

#### UnaryExpr - Unary Operations

Available operations: `Abs`, `Neg`, `Not`, `BitNot`

```python
neg_x = ir.Neg(x, DataType.INT64, ir.Span.unknown())  # -x
```

#### Call - Function Call

```python
op = ir.Op("my_function")
call = ir.Call(op, [x, y], ir.Span.unknown())
```

### Function

#### Function - Function Definition

```python
# def add(x, y) -> int: return x + y
params = [
    ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown()),
    ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())
]
return_types = [ir.ScalarType(DataType.INT64)]

# Define the function body: result = x + y
result = ir.Var("result", ir.ScalarType(DataType.INT64), ir.Span.unknown())
add_expr = ir.Add(params[0], params[1], DataType.INT64, ir.Span.unknown())
body = ir.AssignStmt(result, add_expr, ir.Span.unknown())

func = ir.Function("add", params, return_types, body, ir.Span.unknown())
```

### Statement Hierarchy

#### AssignStmt - Assignment

```python
# x = y
assign = ir.AssignStmt(x, y, ir.Span.unknown())
```

#### IfStmt - Conditional

```python
# if (x > 0) then { y = 1 } else { y = -1 }
condition = ir.Gt(x, zero, DataType.INT64, ir.Span.unknown())
if_stmt = ir.IfStmt(
    condition,
    [then_stmt],
    [else_stmt],
    [y],  # return variables
    ir.Span.unknown()
)
```

#### ForStmt - Loop

```python
# for i in range(0, 10, 1): sum = sum + i
for_stmt = ir.ForStmt(
    i,                # loop variable
    start, stop, step,
    [assign],         # body
    [sum_var],        # return variables
    ir.Span.unknown()
)
```

#### YieldStmt - Yield

```python
yield_stmt = ir.YieldStmt([x, y], ir.Span.unknown())
```

#### SeqStmts - Statement Sequence

```python
# General statement sequence
seq = ir.SeqStmts([stmt1, stmt2, stmt3], ir.Span.unknown())
```

#### OpStmts - Assignment Statement Sequence

```python
# Sequence of assignment statements only
ops = ir.OpStmts([assign1, assign2], ir.Span.unknown())
```

## Type System

### ScalarType

```python
int_type = ir.ScalarType(DataType.INT64)
```

### TensorType

```python
# Tensor with shape [10, 20]
shape = [
    ir.ConstInt(10, DataType.INT64, ir.Span.unknown()),
    ir.ConstInt(20, DataType.INT64, ir.Span.unknown())
]
tensor_type = ir.TensorType(DataType.FLOAT32, shape)
```

### UnknownType

```python
unknown = ir.UnknownType()
```

## Python Usage Examples

### Example 1: Complex Expression

```python
from pypto import DataType, ir

# Build: ((x + 1) * (y - 2)) / (x + y)
span = ir.Span.unknown()
dtype = DataType.INT64

x = ir.Var("x", ir.ScalarType(dtype), span)
y = ir.Var("y", ir.ScalarType(dtype), span)
one = ir.ConstInt(1, dtype, span)
two = ir.ConstInt(2, dtype, span)

x_plus_1 = ir.Add(x, one, dtype, span)
y_minus_2 = ir.Sub(y, two, dtype, span)
numerator = ir.Mul(x_plus_1, y_minus_2, dtype, span)
denominator = ir.Add(x, y, dtype, span)
result = ir.FloatDiv(numerator, denominator, dtype, span)
```

### Example 2: Control Flow

```python
# Absolute value: if (x >= 0) then { result = x } else { result = -x }
x = ir.Var("x", ir.ScalarType(dtype), span)
result = ir.Var("result", ir.ScalarType(dtype), span)
zero = ir.ConstInt(0, dtype, span)
condition = ir.Ge(x, zero, dtype, span)

then_assign = ir.AssignStmt(result, x, span)
neg_x = ir.Neg(x, dtype, span)
else_assign = ir.AssignStmt(result, neg_x, span)

abs_stmt = ir.IfStmt(condition, [then_assign], [else_assign], [result], span)
```

### Example 3: Loop with Accumulation

```python
# sum = 0; for i in range(0, n, 1): sum = sum + i
n = ir.Var("n", ir.ScalarType(dtype), span)
i = ir.Var("i", ir.ScalarType(dtype), span)
sum_var = ir.Var("sum", ir.ScalarType(dtype), span)
zero = ir.ConstInt(0, dtype, span)
one = ir.ConstInt(1, dtype, span)

init = ir.AssignStmt(sum_var, zero, span)
add_expr = ir.Add(sum_var, i, dtype, span)
update = ir.AssignStmt(sum_var, add_expr, span)
loop = ir.ForStmt(i, zero, n, one, [update], [sum_var], span)

program = ir.SeqStmts([init, loop], span)
```

### Example 4: Function Definition

```python
# def sum_range(n) -> int:
#     sum = 0
#     for i in range(0, n, 1):
#         sum = sum + i
#     return sum

# Parameters
n = ir.Var("n", ir.ScalarType(dtype), span)

# Function body (reusing program from Example 3)
body = program  # SeqStmts containing init and loop

# Return types
return_types = [ir.ScalarType(DataType.INT64)]

# Create function
sum_func = ir.Function("sum_range", [n], return_types, body, span)
```

## Summary

The PyPTO IR provides:

- **Immutable tree structure** for safe transformations
- **Comprehensive expression types**: variables, constants, binary/unary operations, function calls
- **Rich statement types**: assignments, conditionals (if), loops (for), yields, sequences
- **Function definitions** with parameters, return types, and bodies
- **Flexible type system** supporting scalars and tensors
- **Reflection-based generic traversal** enabling visitors, mutators, and structural comparison
- **Python-friendly API** for IR construction

For structural comparison and optimization, see [Structural Comparison](01-structural_comparison.md).
