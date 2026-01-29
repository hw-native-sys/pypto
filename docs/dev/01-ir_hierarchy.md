# PyPTO IR Node Hierarchy

This document provides a complete reference of all IR node types, organized by category.

## BNF Grammar

```bnf
<program>    ::= [ identifier ":" ] { <function> }
<function>   ::= "def" identifier "(" [ <param_list> ] ")" [ "->" <type_list> ] ":" <stmt>
<param_list> ::= <var> { "," <var> }
<type_list>  ::= <type> { "," <type> }

<stmt>       ::= <assign_stmt> | <if_stmt> | <for_stmt> | <yield_stmt>
               | <eval_stmt> | <seq_stmts> | <op_stmts>

<assign_stmt> ::= <var> "=" <expr>
<if_stmt>    ::= "if" <expr> ":" <stmt_list> [ "else" ":" <stmt_list> ] [ "return" <var_list> ]
<for_stmt>   ::= "for" <var> [ "," "(" <iter_arg_list> ")" ] "in"
                 ( "range" | "pl.range" ) "(" <expr> "," <expr> "," <expr>
                 [ "," "init_values" "=" "[" <expr_list> "]" ] ")" ":" <stmt_list>
                 [ <return_assignments> ]

<yield_stmt> ::= "yield" [ <var_list> ]
<eval_stmt>  ::= <expr>
<seq_stmts>  ::= <stmt> { ";" <stmt> }
<op_stmts>   ::= <assign_stmt> { ";" <assign_stmt> }

<expr>       ::= <var> | <const_int> | <const_bool> | <const_float> | <call>
               | <binary_op> | <unary_op> | <tuple_get_item>

<call>       ::= <op> "(" [ <expr_list> ] ")"
<op>         ::= identifier | <global_var>

<type>       ::= <scalar_type> | <tensor_type> | <tile_type>
               | <tuple_type> | <pipe_type> | <unknown_type>

<scalar_type> ::= "ScalarType" "(" <data_type> ")"
<tensor_type> ::= "TensorType" "(" <data_type> "," <shape> [ "," <memref> ] ")"
<tile_type>   ::= "TileType" "(" <data_type> "," <shape> [ "," <memref> [ "," <tile_view> ] ] ")"
<tuple_type>  ::= "TupleType" "(" "[" <type_list> "]" ")"
<pipe_type>   ::= "PipeType" "(" <pipe_kind> ")"

<shape>       ::= "[" <expr_list> "]"
<data_type>   ::= "INT32" | "INT64" | "FP16" | "FP32" | "FP64" | "BOOL" | ...
<pipe_kind>   ::= "S" | "V" | "M" | "MTE1" | "MTE2" | "MTE3" | "ALL" | ...
```

## Expression Nodes

| Node Type | Fields | Description |
|-----------|--------|-------------|
| **Var** | `name_`, `type_` | Variable reference |
| **IterArg** | `name_`, `type_`, `initValue_` | Loop iteration argument (extends Var) |
| **ConstInt** | `value_`, `dtype_` | Integer constant |
| **ConstBool** | `value_` | Boolean constant (always BOOL dtype) |
| **ConstFloat** | `value_`, `dtype_` | Floating-point constant |
| **Call** | `op_`, `args_`, `kwargs_` | Function/operator call |
| **TupleGetItemExpr** | `tuple_`, `index_` | Tuple element access |

### Binary Expression Nodes

| Category | Nodes |
|----------|-------|
| **Arithmetic** | Add, Sub, Mul, FloorDiv, FloorMod, FloatDiv |
| **Math** | Min, Max, Pow |
| **Comparison** | Eq, Ne, Lt, Le, Gt, Ge |
| **Logical** | And, Or, Xor |
| **Bitwise** | BitAnd, BitOr, BitXor, BitShiftLeft, BitShiftRight |

All binary expressions have: `lhs_`, `rhs_`, `dtype_`

### Unary Expression Nodes

| Node | Operation |
|------|-----------|
| **Abs** | Absolute value |
| **Neg** | Negation |
| **Not** | Logical NOT |
| **BitNot** | Bitwise NOT |
| **Cast** | Type casting |

All unary expressions have: `operand_`, `dtype_`

### Op and GlobalVar

| Node Type | Purpose | Usage |
|-----------|---------|-------|
| **Op** | Generic operation/function reference | External operators, built-in functions |
| **GlobalVar** | Function reference within a program | Intra-program function calls |

```python
# Op for external functions
op = ir.Op("my_function")
call = ir.Call(op, [x, y], span)

# GlobalVar for program functions
gvar = ir.GlobalVar("helper")  # Must match function name
call = ir.Call(gvar, [x], span)
```

### IterArg - Loop-Carried Values

`IterArg` is a specialized variable for SSA-style loop iterations:

**Key Properties:**
- Extends `Var` with `initValue_` field
- Scoped to loop body only
- Updated via `yield` statement
- Final values captured in `return_vars`

```python
# for i, (sum,) in pl.range(0, n, 1, init_values=[0]):
#     sum = pl.yield(sum + i)
# sum_final = sum

init_val = ir.ConstInt(0, DataType.INT64, span)
sum_iter = ir.IterArg("sum", ir.ScalarType(DataType.INT64), init_val, span)
sum_final = ir.Var("sum_final", ir.ScalarType(DataType.INT64), span)

for_stmt = ir.ForStmt(i, start, stop, step, [sum_iter], body, [sum_final], span)
```

## Statement Nodes

| Node Type | Fields | Description |
|-----------|--------|-------------|
| **AssignStmt** | `var_` (DefField), `value_` (UsualField) | Variable assignment |
| **IfStmt** | `condition_`, `then_stmts_`, `else_stmts_`, `return_vars_` | Conditional branching |
| **ForStmt** | `loop_var_` (DefField), `start_`, `stop_`, `step_`, `iter_args_` (DefField), `body_`, `return_vars_` (DefField) | For loop with optional iteration args |
| **YieldStmt** | `values_` | Yield values in loop iteration |
| **EvalStmt** | `expr_` | Evaluate expression for side effects |
| **SeqStmts** | `stmts_` | General statement sequence |
| **OpStmts** | `stmts_` | Assignment statement sequence |

### ForStmt Details

**Without iteration arguments:**
```python
# for i in range(0, 10, 1): x = x + i
for_stmt = ir.ForStmt(i, start, stop, step, [], body, [], span)
```

**With iteration arguments:**
```python
# for i, (sum,) in pl.range(0, 10, 1, init_values=[0]):
#     sum = pl.yield(sum + i)
# sum_final = sum
for_stmt = ir.ForStmt(i, start, stop, step, [sum_iter], body, [sum_final], span)
```

**Requirements:**
- Number of yielded values = number of IterArgs
- Number of return_vars = number of IterArgs
- IterArgs accessible only within loop body
- Return vars accessible after loop

## Type Nodes

| Node Type | Fields | Description |
|-----------|--------|-------------|
| **ScalarType** | `dtype_` | Scalar type (INT64, FP32, etc.) |
| **TensorType** | `shape_`, `dtype_`, `memref_` (optional) | Multi-dimensional tensor |
| **TileType** | `shape_`, `dtype_`, `memref_` (optional), `tile_view_` (optional) | 2D tile (≤2 dimensions) |
| **TupleType** | `types_` | Tuple of types |
| **PipeType** | `pipe_kind_` | Hardware pipeline/barrier |
| **UnknownType** | - | Unknown or inferred type |

### MemRef - Memory Reference

Describes memory allocation for tensors/tiles:

| Field | Type | Description |
|-------|------|-------------|
| `memory_space_` | MemorySpace enum | DDR, UB, L1, L0A, L0B, L0C |
| `addr_` | ExprPtr | Base address |
| `size_` | size_t | Size in bytes |

```python
memref = ir.MemRef(
    ir.MemorySpace.DDR,
    ir.ConstInt(0x1000, DataType.INT64, span),
    1024  # bytes
)
```

### TileView - Tile Layout

Describes tile layout and access pattern:

| Field | Type | Description |
|-------|------|-------------|
| `valid_shape` | list[ExprPtr] | Valid dimensions |
| `stride` | list[ExprPtr] | Stride per dimension |
| `start_offset` | ExprPtr | Starting offset |

```python
tile_view = ir.TileView()
tile_view.valid_shape = [ir.ConstInt(16, DataType.INT64, span)] * 2
tile_view.stride = [ir.ConstInt(1, DataType.INT64, span), ir.ConstInt(16, DataType.INT64, span)]
tile_view.start_offset = ir.ConstInt(0, DataType.INT64, span)
```

## Function Node

```python
# def add(x, y) -> int: return x + y
params = [
    ir.Var("x", ir.ScalarType(DataType.INT64), span),
    ir.Var("y", ir.ScalarType(DataType.INT64), span)
]
return_types = [ir.ScalarType(DataType.INT64)]
body = ir.AssignStmt(result, ir.Add(params[0], params[1], DataType.INT64, span), span)

func = ir.Function("add", params, return_types, body, span)
```

| Field | Type | Description |
|-------|------|-------------|
| `name_` | string | Function name |
| `params_` | list[VarPtr] | Parameters (DefField) |
| `return_types_` | list[TypePtr] | Return types |
| `body_` | StmtPtr | Function body |

## Program Node

Container for multiple functions with deterministic ordering:

| Field | Type | Description |
|-------|------|-------------|
| `name_` | string | Program name (IgnoreField) |
| `functions_` | map[GlobalVarPtr, FunctionPtr] | Sorted map of functions |

```python
# Create program
func1 = ir.Function("add", params1, return_types1, body1, span)
func2 = ir.Function("multiply", params2, return_types2, body2, span)

program = ir.Program([func1, func2], "my_program", span)

# Access functions
add_func = program.get_function("add")
add_gvar = program.get_global_var("add")

# Functions are automatically sorted by name
```

**Key Features:**
- Functions stored in sorted map for deterministic ordering
- GlobalVar names must match function names
- Enables intra-program calls via GlobalVar
- Ensures consistent structural equality and hashing

## Node Summary by Category

| Category | Count | Nodes |
|----------|-------|-------|
| **Base Classes** | 4 | IRNode, Expr, Stmt, Type |
| **Variables** | 2 | Var, IterArg |
| **Constants** | 3 | ConstInt, ConstFloat, ConstBool |
| **Binary Ops** | 18 | Add, Sub, Mul, FloorDiv, FloorMod, FloatDiv, Min, Max, Pow, Eq, Ne, Lt, Le, Gt, Ge, And, Or, Xor, BitAnd, BitOr, BitXor, BitShiftLeft, BitShiftRight |
| **Unary Ops** | 5 | Abs, Neg, Not, BitNot, Cast |
| **Call/Access** | 2 | Call, TupleGetItemExpr |
| **Operations** | 2 | Op, GlobalVar |
| **Statements** | 7 | AssignStmt, IfStmt, ForStmt, YieldStmt, EvalStmt, SeqStmts, OpStmts |
| **Types** | 6 | ScalarType, TensorType, TileType, TupleType, PipeType, UnknownType |
| **Functions** | 2 | Function, Program |

## Related Documentation

- [IR Overview](00-ir_overview.md) - Core concepts and design principles
- [IR Types and Examples](02-ir_types_examples.md) - Type system details and examples
- [Structural Comparison](03-structural_comparison.md) - Equality and hashing
