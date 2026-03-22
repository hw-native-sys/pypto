# PyPTO IR Node Hierarchy

This document provides a complete reference of all IR node types, organized by category.

## BNF Grammar

```bnf
<program>    ::= [ identifier ":" ] { <function> }
<function>   ::= "def" identifier "(" [ <param_list> ] ")" [ "->" <type_list> ] ":" <stmt>
<param_list> ::= <param> { "," <param> }
<param>      ::= <var> | "(" <var> "," <param_direction> ")"
<param_direction> ::= "In" | "Out" | "InOut"
<type_list>  ::= <type> { "," <type> }

<stmt>       ::= <assign_stmt> | <if_stmt> | <for_stmt> | <while_stmt> | <return_stmt> | <yield_stmt>
               | <eval_stmt> | <seq_stmts> | <scope_stmt>
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
<return_stmt> ::= "return" [ <var_list> ]
<eval_stmt>  ::= <expr>
<seq_stmts>  ::= <stmt> { ";" <stmt> }
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
<tile_type>   ::= "TileType" "(" <data_type> "," <shape>
                 [ "," <tile_type_arg> { "," <tile_type_arg> } ]
                 ")"
<tile_type_arg> ::= <memref> | <tile_view> | <memory_space>
<tuple_type>  ::= "TupleType" "(" "[" <type_list> "]" ")"
<pipe_type>   ::= "PipeType" "(" <pipe_kind> ")"

<shape>       ::= "[" <expr_list> "]"
<data_type>   ::= "INT32" | "INT64" | "FP16" | "FP32" | "FP64" | "BOOL" | ...
<memory_space> ::= "DDR" | "Vec" | "Mat" | "Left" | "Right" | "Acc" | "Bias"
<pipe_kind>   ::= "S" | "V" | "M" | "MTE1" | "MTE2" | "MTE3" | "ALL" | ...
```

For `TileType`, each optional argument may appear at most once. If a `MemRef`
is present, `memory_space` must also be present on the `TileType`.

## Expression Nodes

| Node Type | Fields | Description |
| --------- | ------ | ----------- |
| **Var** | `name_hint_`, `type_` | Variable reference (identity by pointer, not by name) |
| **IterArg** | `name_hint_`, `type_`, `initValue_` | Loop iteration argument (extends Var) |
| **ConstInt** | `value_`, `dtype_` | Integer constant |
| **ConstBool** | `value_` | Boolean constant (always BOOL dtype) |
| **ConstFloat** | `value_`, `dtype_` | Floating-point constant |
| **Call** | `op_`, `args_`, `kwargs_` | Function/operator call |
| **TupleGetItemExpr** | `tuple_`, `index_` | Tuple element access |

### Var Identity

Variable identity is determined by **object pointer** (or equivalently `unique_id_`), **not** by `name_hint_`. Two `Var` objects with the same `name_hint_` are distinct variables if they are different objects. The field is named `name_hint_` (rather than `name_`) to make this explicit.

| Field | Purpose |
| ----- | ------- |
| `name_hint_` | Cosmetic label for printing and debugging. `IgnoreField` — excluded from structural comparison and hashing. |
| `unique_id_` | Monotonically increasing ID assigned at construction. Used for deterministic hashing. |
| object pointer | The canonical identity — two references denote the same variable iff they point to the same `Var` object. |

```cpp
// Same name_hint, different variables
auto x1 = std::make_shared<Var>("x", type, span);
auto x2 = std::make_shared<Var>("x", type, span);
// x1 != x2 — they are distinct variables despite sharing the name "x"

// Same variable referenced twice
auto x_ref = x1;
// x1 == x_ref — same pointer, same variable
```

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
# for i, (sum,) in pl.range(n, init_values=(0,)): sum = pl.yield_(sum + i)
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
| **BreakStmt** | *(none)* | Exit loop |
| **ContinueStmt** | *(none)* | Skip to next loop iteration |

### ForStmt Details

```python
# Without iter_args: for i in pl.range(10): x = x + i
for_stmt = ir.ForStmt(i, start, stop, step, [], body, [], span)

# With iter_args: for i, (sum,) in pl.range(10, init_values=(0,)): sum = pl.yield_(sum + i)
for_stmt = ir.ForStmt(i, start, stop, step, [sum_iter], body, [sum_final], span)
```

> **Note:** The DSL accepts concise forms `pl.range(stop)` / `pl.range(start, stop)` as syntactic sugar (like Python's `range()`). The IR always stores all three fields (`start_`, `stop_`, `step_`); the parser fills in defaults (start=0, step=1) and the printer elides them when they match.

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
# for i in pl.parallel(10): ...
for_stmt = ir.ForStmt(i, start, stop, step, [], body, [], span, ir.ForKind.Parallel)
```

The `kind_` field (`ForKind` enum) distinguishes sequential (`ForKind.Sequential`, default), parallel (`ForKind.Parallel`), and unroll (`ForKind.Unroll`) loops. In the DSL, `pl.range()` produces sequential, `pl.parallel()` produces parallel, and `pl.unroll()` produces compile-time unrolled loops. The printer emits `pl.parallel(...)` or `pl.unroll(...)` accordingly.

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
| **TileType** | `shape_`, `dtype_`, `memref_` (optional), `tile_view_` (optional), `memory_space_` (optional) | Tile in unified buffer |
| **TupleType** | `types_` | Tuple of types |
| **PipeType** | `pipe_kind_` | Hardware pipeline/barrier |
| **UnknownType** | - | Unknown or inferred type |

### MemRef - Memory Reference

Describes memory allocation metadata shared by tensors/tiles. The memory space is
stored on `TileType.memory_space_` for tiles; `TensorType` is canonically DDR.

| Field | Type | Description |
| ----- | ---- | ----------- |
| `addr_` | ExprPtr | Base address |
| `size_` | size_t | Size in bytes |
| `id_` | uint64_t | Stable MemRef identifier |

```python
memref = ir.MemRef(
    ir.ConstInt(0x1000, DataType.INT64, span),
    1024,  # bytes
    0     # id
)
```

> **Note:** `ir.Mem` is a short alias for `ir.MemorySpace`.

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
| `func_type_` | FunctionType | Function type (Opaque, Orchestration, InCore, AIC, AIV, or Group) |
| `params_` | list[VarPtr] | Parameter variables (DefField) |
| `param_directions_` | list[ParamDirection] | Parameter directions, same length as params_ |
| `return_types_` | list[TypePtr] | Return types |
| `body_` | StmtPtr | Function body |

### ParamDirection Enum

| Value | Description |
| ----- | ----------- |
| `In` | Read-only input parameter (default) |
| `Out` | Write-only output parameter |
| `InOut` | Read-write input/output parameter |

### FunctionType Enum

| Value | Description |
| ----- | ----------- |
| `Opaque` | Unspecified function type (default) |
| `Orchestration` | Runs on host/AICPU for control flow and dependency analysis |
| `InCore` | AICore sub-graph execution (unspecialized) |
| `AIC` | Cube core kernel (specialized InCore) |
| `AIV` | Vector core kernel (specialized InCore) |
| `Group` | Co-scheduled group of AIC + AIV kernels |

`IsInCoreType(type)` / `ir.is_incore_type(type)` returns `True` for `InCore`, `AIC`, and `AIV`.

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
| **Binary Ops** | 23 | Add, Sub, Mul, FloorDiv, FloorMod, FloatDiv, Min, Max, Pow, Eq, Ne, Lt, Le, Gt, Ge, And, Or, Xor, BitAnd, BitOr, BitXor, BitShiftLeft, BitShiftRight |
| **Unary Ops** | 5 | Abs, Neg, Not, BitNot, Cast |
| **Call/Access** | 2 | Call, TupleGetItemExpr |
| **Operations** | 2 | Op, GlobalVar |
| **Statements** | 11 | AssignStmt, IfStmt, ForStmt, WhileStmt, ReturnStmt, ScopeStmt, YieldStmt, EvalStmt, SeqStmts, BreakStmt, ContinueStmt |
| **Types** | 6 | ScalarType, TensorType, TileType, TupleType, PipeType, UnknownType |
| **Functions** | 2 | Function, Program |

## Related Documentation

- [IR Overview](00-overview.md) - Core concepts and design principles
- [IR Types and Examples](02-types.md) - Type system details and examples
- [Structural Comparison](03-structural_comparison.md) - Equality and hashing
