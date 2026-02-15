# PyPTO IR Node Hierarchy

This document provides a complete reference of all IR node types, organized by category.

## BNF Grammar

```bnf
<program>    ::= [ identifier ":" ] { <function> }
<function>   ::= "def" identifier "(" [ <param_list> ] ")" [ "->" <type_list> ] ":" <stmt>
<param_list> ::= <var> { "," <var> }
<type_list>  ::= <type> { "," <type> }

<stmt>       ::= <assign_stmt> | <if_stmt> | <for_stmt> | <while_stmt> | <yield_stmt>
               | <eval_stmt> | <seq_stmts> | <op_stmts> | <scope_stmt>
               | <break_stmt> | <continue_stmt>

<assign_stmt> ::= <var> "=" <expr>
<if_stmt>    ::= "if" <expr> ":" <stmt_list> [ "else" ":" <stmt_list> ] [ "return" <var_list> ]
<for_stmt>   ::= "for" <var> [ "," "(" <iter_arg_list> ")" ] "in"
                 ( "range" | "pl.range" ) "(" <expr> "," <expr> "," <expr>
                 [ "," "init_values" "=" "(" <expr_list> ")" ] ")" ":" <stmt_list>
                 [ <return_assignments> ]
<while_stmt> ::= "while" <expr> ":" <stmt_list>
               | "for" "(" <iter_arg_list> ")" "in" "pl.while_"
                 "(" "init_values" "=" "(" <expr_list> ")" ")" ":"
                 "pl.cond" "(" <expr> ")" <stmt_list>
                 [ <return_assignments> ]

<yield_stmt> ::= "yield" [ <var_list> ]
<eval_stmt>  ::= <expr>
<seq_stmts>  ::= <stmt> { ";" <stmt> }
<op_stmts>   ::= <assign_stmt> { ";" <assign_stmt> }
<scope_stmt> ::= "with" "pl.incore" "(" ")" ":" <stmt_list>
<break_stmt> ::= "break"
<continue_stmt> ::= "continue"

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
| --------- | ------ | ----------- |
| **Var** | `name_`, `type_` | Variable reference |
| **IterArg** | `name_`, `type_`, `initValue_` | Loop iteration argument (extends Var) |
| **ConstInt** | `value_`, `dtype_` | Integer constant |
| **ConstBool** | `value_` | Boolean constant (always BOOL dtype) |
| **ConstFloat** | `value_`, `dtype_` | Floating-point constant |
| **Call** | `op_`, `args_`, `kwargs_` | Function/operator call |
| **TupleGetItemExpr** | `tuple_`, `index_` | Tuple element access |

### Binary Expression Nodes

| Category | Nodes |
| -------- | ----- |
| **Arithmetic** | Add, Sub, Mul, FloorDiv, FloorMod, FloatDiv |
| **Math** | Min, Max, Pow |
| **Comparison** | Eq, Ne, Lt, Le, Gt, Ge |
| **Logical** | And, Or, Xor |
| **Bitwise** | BitAnd, BitOr, BitXor, BitShiftLeft, BitShiftRight |

All binary expressions have: `lhs_`, `rhs_`, `dtype_`

### Unary Expression Nodes

| Node | Operation |
| ---- | --------- |
| **Abs** | Absolute value |
| **Neg** | Negation |
| **Not** | Logical NOT |
| **BitNot** | Bitwise NOT |
| **Cast** | Type casting |

All unary expressions have: `operand_`, `dtype_`

### Op and GlobalVar

| Node Type | Purpose | Usage |
| --------- | ------- | ----- |
| **Op** | Generic operation/function reference | External operators, built-in functions |
| **GlobalVar** | Function reference within a program | Intra-program function calls |

```python
op = ir.Op("my_function"); call = ir.Call(op, [x, y], span)  # External
gvar = ir.GlobalVar("helper"); call = ir.Call(gvar, [x], span)  # Internal
```

### IterArg - Loop-Carried Values

`IterArg` extends `Var` with `initValue_` for SSA-style loops. Scoped to loop body, updated via `yield`, final values in `return_vars`.

```python
# for i, (sum,) in pl.range(0, n, 1, init_values=(0,)): sum = pl.yield_(sum + i)
sum_iter = ir.IterArg("sum", ir.ScalarType(DataType.INT64), init_val, span)
for_stmt = ir.ForStmt(i, start, stop, step, [sum_iter], body, [sum_final], span)
```

## Statement Nodes

| Node Type | Fields | Description |
| --------- | ------ | ----------- |
| **AssignStmt** | `var_` (DefField), `value_` (UsualField) | Variable assignment |
| **IfStmt** | `condition_`, `then_stmts_`, `else_stmts_`, `return_vars_` | Conditional branching |
| **ForStmt** | `loop_var_` (DefField), `start_`, `stop_`, `step_`, `iter_args_` (DefField), `body_`, `return_vars_` (DefField), `kind_` | For loop with optional iteration args |
| **WhileStmt** | `condition_`, `iter_args_` (DefField), `body_`, `return_vars_` (DefField) | While loop with condition and iteration args |
| **ScopeStmt** | `scope_kind_`, `body_` | Marks a region with specific execution context (e.g., InCore) |
| **YieldStmt** | `values_` | Yield values in loop iteration |
| **EvalStmt** | `expr_` | Evaluate expression for side effects |
| **SeqStmts** | `stmts_` | General statement sequence |
| **OpStmts** | `stmts_` | Assignment statement sequence |
| **BreakStmt** | *(none)* | Exit loop |
| **ContinueStmt** | *(none)* | Skip to next loop iteration |

### ForStmt Details

```python
# Without iter_args: for i in range(0, 10, 1): x = x + i
for_stmt = ir.ForStmt(i, start, stop, step, [], body, [], span)

# With iter_args: for i, (sum,) in pl.range(0, 10, 1, init_values=(0,)): sum = pl.yield_(sum + i)
for_stmt = ir.ForStmt(i, start, stop, step, [sum_iter], body, [sum_final], span)
```

### WhileStmt Details

```python
# Natural: while x < 10: x = x + 1
while_stmt = ir.WhileStmt(condition, [], body, [], span)

# SSA form: for (x,) in pl.while_(init_values=(0,)): pl.cond(x < 10); x = pl.yield_(x + 1)
while_stmt = ir.WhileStmt(condition, [x_iter], body, [x_final], span)
```

**Properties:** `condition_` evaluated each iteration; supports SSA iter_args/return_vars; DSL uses `pl.cond()` as first statement.

- Natural syntax without iter_args is converted to SSA by ConvertToSSA pass
- Body must end with YieldStmt when iter_args are present

### ScopeStmt Details

Marks a region with specific execution context (e.g., InCore for AICore sub-graphs).

```python
# with pl.incore(): y = pl.add(x, x)
scope_stmt = ir.ScopeStmt(ir.ScopeKind.InCore, body, span)
```

**Properties:**

- `scope_kind_`: Execution context (`ScopeKind.InCore`)
- `body_`: Nested statements
- Transparent to SSA (no iter_args/return_vars)
- Not control flow (executes once, linearly)
- `OutlineIncoreScopes` pass extracts into `Function(InCore)`

**Transformation:**

```python
# Before: with pl.incore(): y = pl.add(x, x); return y
# After: main_incore_0(x) -> y; main(x): y = main_incore_0(x); return y
```

**Parallel for loop (ForKind):**

```python
# for i in pl.parallel(0, 10, 1): ...
for_stmt = ir.ForStmt(i, start, stop, step, [], body, [], span, ir.ForKind.Parallel)
```

The `kind_` field (`ForKind` enum) distinguishes sequential (`ForKind.Sequential`, default) from parallel (`ForKind.Parallel`) loops. In the DSL, `pl.range()` produces sequential and `pl.parallel()` produces parallel loops. The printer emits `pl.parallel(...)` for parallel kind.

**Requirements:**

- Number of yielded values = number of IterArgs
- Number of return_vars = number of IterArgs
- IterArgs accessible only within loop body
- Return vars accessible after loop

## Type Nodes

| Node Type | Fields | Description |
| --------- | ------ | ----------- |
| **ScalarType** | `dtype_` | Scalar type (INT64, FP32, etc.) |
| **TensorType** | `shape_`, `dtype_`, `memref_` (optional) | Multi-dimensional tensor |
| **TileType** | `shape_`, `dtype_`, `memref_` (optional), `tile_view_` (optional) | Tile in unified buffer |
| **TupleType** | `types_` | Tuple of types |
| **PipeType** | `pipe_kind_` | Hardware pipeline/barrier |
| **UnknownType** | - | Unknown or inferred type |

### MemRef - Memory Reference

Describes memory allocation for tensors/tiles:

| Field | Type | Description |
| ----- | ---- | ----------- |
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
| ----- | ---- | ----------- |
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

# With function type
func_orch = ir.Function("orchestrator", params, return_types, body, span, ir.FunctionType.Orchestration)
```

| Field | Type | Description |
| ----- | ---- | ----------- |
| `name_` | string | Function name |
| `func_type_` | FunctionType | Function type (Opaque, Orchestration, or InCore) |
| `params_` | list[VarPtr] | Parameters (DefField) |
| `return_types_` | list[TypePtr] | Return types |
| `body_` | StmtPtr | Function body |

### FunctionType Enum

| Value | Description |
| ----- | ----------- |
| `Opaque` | Unspecified function type (default) |
| `Orchestration` | Runs on host/AICPU for control flow and dependency analysis |
| `InCore` | Sub-graph on specific AICore |

## Program Node

Container for multiple functions with deterministic ordering:

| Field | Type | Description |
| ----- | ---- | ----------- |
| `name_` | string | Program name (IgnoreField) |
| `functions_` | map[GlobalVarPtr, FunctionPtr] | Sorted map of functions |

```python
program = ir.Program([func1, func2], "my_program", span)
add_func = program.get_function("add")  # Access by name
```

Functions stored in sorted map for deterministic ordering. GlobalVar names must match function names.

## Node Summary by Category

| Category | Count | Nodes |
| -------- | ----- | ----- |
| **Base Classes** | 4 | IRNode, Expr, Stmt, Type |
| **Variables** | 2 | Var, IterArg |
| **Constants** | 3 | ConstInt, ConstFloat, ConstBool |
| **Binary Ops** | 18 | Add, Sub, Mul, FloorDiv, FloorMod, FloatDiv, Min, Max, Pow, Eq, Ne, Lt, Le, Gt, Ge, And, Or, Xor, BitAnd, BitOr, BitXor, BitShiftLeft, BitShiftRight |
| **Unary Ops** | 5 | Abs, Neg, Not, BitNot, Cast |
| **Call/Access** | 2 | Call, TupleGetItemExpr |
| **Operations** | 2 | Op, GlobalVar |
| **Statements** | 11 | AssignStmt, IfStmt, ForStmt, WhileStmt, ScopeStmt, YieldStmt, EvalStmt, SeqStmts, OpStmts, BreakStmt, ContinueStmt |
| **Types** | 6 | ScalarType, TensorType, TileType, TupleType, PipeType, UnknownType |
| **Functions** | 2 | Function, Program |

## Related Documentation

- [IR Overview](00-overview.md) - Core concepts and design principles
- [IR Types and Examples](02-types.md) - Type system details and examples
- [Structural Comparison](03-structural_comparison.md) - Equality and hashing
