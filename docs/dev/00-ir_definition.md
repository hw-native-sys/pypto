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
<program>    ::= [ identifier ":" ] { <function> }

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

<for_stmt>   ::= "for" <var> [ "," "(" <iter_arg_list> ")" ] "in"
                 ( "range" | "pi.range" ) "(" <expr> "," <expr> "," <expr>
                 [ "," "init_values" "=" "[" <expr_list> "]" ] ")" ":" <stmt_list>
                 [ <return_assignments> ]

<iter_arg_list> ::= <var> { "," <var> }

<return_assignments> ::= <var> "=" <var> { <var> "=" <var> }

<expr_list>  ::= <expr> { "," <expr> }

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

#### IterArg - Iteration Argument

`IterArg` is a special variable type used in for loops to carry values across iterations (SSA-style loop-carried dependencies). It extends `Var` with an initial value.

**Key properties:**
- **Scoping**: IterArg variables are scoped to the loop body only
- **Initial value**: Must provide an `initValue` expression that provides the value for the first iteration
- **Yielding**: Updated via `yield` statement at the end of each iteration
- **Return variables**: To expose final values outside the loop, use `return_vars` which capture the final iteration values

```python
from pypto import DataType, ir

# Create an IterArg with initial value
init_value = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
sum_iter = ir.IterArg(
    "sum",                                # Variable name (scoped to loop)
    ir.ScalarType(DataType.INT64),        # Type
    init_value,                           # Initial value for first iteration
    ir.Span.unknown()
)

# IterArg is used within ForStmt:
# for i, (sum,) in pi.range(0, n, 1, init_values=[0]):
#     sum = pi.yield(sum + i)
# sum_final: pi.Int64 = sum  # Capture final value in return variable
```

**SSA Semantics:**

```
# Python syntax
sum_init = 0
for i, (sum,) in pi.range(0, 10, 1, init_values=[sum_init]):
    sum = pi.yield(sum + i)
sum_final = sum

# Equivalent SSA IR
sum_init = 0
loop (i = 0 to 10):
    sum = phi(sum_init, sum_next)  # First iteration: sum_init, subsequent: sum_next
    sum_next = sum + i
    yield sum_next
sum_final = sum_next
```

**Important Notes:**
- IterArg variables (`sum`) are only accessible within the loop body
- To use the final value outside the loop, capture it in a return variable (`sum_final = sum`)
- The number of IterArgs must match the number of values yielded in the loop body
- The number of return variables must match the number of IterArgs

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

#### Op - Operation/Function Reference

`Op` is the base class for callable operations in the IR.

```python
op = ir.Op("my_function")
call = ir.Call(op, [x, y], ir.Span.unknown())
```

#### GlobalVar - Global Function Reference

`GlobalVar` is a special type of `Op` used to reference functions within a program. It enables intra-program function calls.

**Important**: The GlobalVar name must match the function name and be unique within the program.

```python
# Create a GlobalVar to reference a function
gvar = ir.GlobalVar("my_func")

# Use it in a Call expression to call a function in the same program
call = ir.Call(gvar, [x, y], ir.Span.unknown())
```

#### Call - Function Call

```python
# Call with a generic Op
op = ir.Op("my_function")
call = ir.Call(op, [x, y], ir.Span.unknown())

# Call with a GlobalVar (for intra-program calls)
gvar = ir.GlobalVar("add")
call = ir.Call(gvar, [x, y], ir.Span.unknown())
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

### Program

#### Program - Top-Level Program Container

A `Program` represents a complete program with functions mapped by `GlobalVar` references.

**Key Features:**
- Functions are stored in a sorted map (by GlobalVar name) for deterministic ordering
- Ensures consistent structural equality and hashing
- GlobalVar names must match function names and be unique within the program
- Supports intra-program function calls via GlobalVar references

```cpp
class Program : public IRNode {
  std::string name_;                                               // Program name (IgnoreField)
  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> functions_;  // Map of GlobalVars to Functions
};
```

**Basic Usage:**

```python
# Create a program with multiple functions
func1 = ir.Function("add", params1, return_types1, body1, ir.Span.unknown())
func2 = ir.Function("multiply", params2, return_types2, body2, ir.Span.unknown())

# Program with name (GlobalVars are created automatically from function names)
program = ir.Program([func1, func2], "my_program", ir.Span.unknown())

# Program without name
program = ir.Program([func1, func2], "", ir.Span.unknown())

# Functions are automatically sorted by name: ["add", "multiply"]
```

**Accessing Functions:**

```python
# Get a function by name
add_func = program.get_function("add")

# Get a GlobalVar by name
add_gvar = program.get_global_var("add")

# Functions are stored in a map (dict in Python)
# Access all functions: program.functions (returns dict[GlobalVar, Function])
```

**Using GlobalVar for Intra-Program Calls:**

```python
# Create functions
x = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
y = ir.Var("y", ir.ScalarType(DataType.INT64), ir.Span.unknown())

# Helper function
helper_body = ir.AssignStmt(y, x, ir.Span.unknown())
helper = ir.Function("helper", [x], [ir.ScalarType(DataType.INT64)], helper_body, ir.Span.unknown())

# Main function that calls helper
program = ir.Program([helper], "my_program", ir.Span.unknown())
helper_gvar = program.get_global_var("helper")
call = ir.Call(helper_gvar, [x], ir.Span.unknown())
main_body = ir.AssignStmt(y, call, ir.Span.unknown())
main = ir.Function("main", [x], [ir.ScalarType(DataType.INT64)], main_body, ir.Span.unknown())

# Update program with both functions
program = ir.Program([helper, main], "my_program", ir.Span.unknown())
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

`ForStmt` represents a for loop with optional loop-carried dependencies (SSA-style iteration).

**Parameters:**
- `loop_var`: Loop iteration variable
- `start`, `stop`, `step`: Range expressions
- `iter_args`: List of `IterArg` objects (loop-carried values, scoped to loop body)
- `body`: Loop body statement
- `return_vars`: List of `Var` objects that capture final iteration values (accessible after loop)
- `span`: Source location

**Simple loop without iteration arguments:**

```python
# for i in range(0, 10, 1): x = x + i
i = ir.Var("i", ir.ScalarType(DataType.INT64), ir.Span.unknown())
start = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
stop = ir.ConstInt(10, DataType.INT64, ir.Span.unknown())
step = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())

body = ir.AssignStmt(x, add_expr, ir.Span.unknown())

for_stmt = ir.ForStmt(
    i,                  # loop variable
    start, stop, step,  # range
    [],                 # no iter_args
    body,               # loop body
    [],                 # no return_vars
    ir.Span.unknown()
)
```

**Loop with iteration arguments (loop-carried values):**

```python
# for i, (sum,) in pi.range(0, 10, 1, init_values=[sum_init]):
#     sum = pi.yield(sum + i)
# sum_final = sum

i = ir.Var("i", ir.ScalarType(DataType.INT64), ir.Span.unknown())
start = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
stop = ir.ConstInt(10, DataType.INT64, ir.Span.unknown())
step = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())

# Create IterArg with initial value
sum_init = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
sum_iter = ir.IterArg("sum", ir.ScalarType(DataType.INT64), sum_init, ir.Span.unknown())

# Loop body with yield
add_expr = ir.Add(sum_iter, i, DataType.INT64, ir.Span.unknown())
yield_stmt = ir.YieldStmt([add_expr], ir.Span.unknown())

# Return variable to capture final value
sum_final = ir.Var("sum_final", ir.ScalarType(DataType.INT64), ir.Span.unknown())

for_stmt = ir.ForStmt(
    i,                  # loop variable
    start, stop, step,  # range
    [sum_iter],         # iter_args (scoped to loop)
    yield_stmt,         # body with yield
    [sum_final],        # return_vars (accessible after loop)
    ir.Span.unknown()
)
```

**Key Points:**
- **IterArgs** are only accessible within the loop body
- **Return variables** capture the final iteration values and are accessible after the loop
- The number of yielded values must match the number of IterArgs
- The number of return variables must match the number of IterArgs
- Use assignment after loop (`sum_final = sum`) in Python syntax to expose final values

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

abs_stmt = ir.IfStmt(condition, then_assign, else_assign, [result], span)
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

### Example 5: Complete Program with Multiple Functions

```python
# Create a program containing multiple functions
span = ir.Span.unknown()
dtype = DataType.INT64

# Function 1: add(x, y) -> int
x = ir.Var("x", ir.ScalarType(dtype), span)
y = ir.Var("y", ir.ScalarType(dtype), span)
result = ir.Var("result", ir.ScalarType(dtype), span)
add_expr = ir.Add(x, y, dtype, span)
add_body = ir.AssignStmt(result, add_expr, span)
add_func = ir.Function("add", [x, y], [ir.ScalarType(dtype)], add_body, span)

# Function 2: multiply(x, y) -> int
mul_expr = ir.Mul(x, y, dtype, span)
mul_body = ir.AssignStmt(result, mul_expr, span)
mul_func = ir.Function("multiply", [x, y], [ir.ScalarType(dtype)], mul_body, span)

# Create program with name
program = ir.Program([add_func, mul_func], "math_operations", span)

# Print the program
print(program)  # Uses IRPrinter to format the program
```

## Summary

The PyPTO IR provides:

- **Immutable tree structure** for safe transformations
- **Comprehensive expression types**: variables, constants, binary/unary operations, function calls
- **Rich statement types**: assignments, conditionals (if), loops (for), yields, sequences
- **Function definitions** with parameters, return types, and bodies
- **Program containers** for organizing multiple functions into complete programs
- **Flexible type system** supporting scalars and tensors
- **Reflection-based generic traversal** enabling visitors, mutators, and structural comparison
- **Python-friendly API** for IR construction

For structural comparison and optimization, see [Structural Comparison](01-structural_comparison.md).
