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
| **Call** | `op_`, `args_`, `kwargs_`, `attrs_` | Function/operator call (see [Call attrs vs kwargs](#call-attrs-vs-kwargs)) |
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

### Call attrs vs kwargs

`Call` carries two ordered string-keyed maps with identical C++ types
(`std::vector<std::pair<std::string, std::any>>`) but different ownership and
intent:

| Field | Purpose | Producer | Reserved keys |
| ----- | ------- | -------- | ------------- |
| `kwargs_` | Language-level keyword arguments supplied by the user at the call site (e.g. `kernel(x, y, axis=2)`). Round-tripped through the parser, printer, and bindings as user-facing data. | Frontend / DSL parser | None — keys come from user code. |
| `attrs_` | Compiler-internal node metadata produced and consumed by passes/verifiers. Not visible as DSL keyword arguments. | Compiler passes (and the deserializer for legacy payloads) | `"arg_directions"` — see below. Future internal attributes should use namespaced keys such as `"hint.*"` or `"profile.*"`. |

**`arg_directions` storage.** Resolved per-argument `ArgDirection` values are
stored under the reserved key `attrs_["arg_directions"]` as
`std::vector<ArgDirection>`. The accessors `Call::HasArgDirections()` and
`Call::GetArgDirections()` (and the Python `Call.arg_directions` property) are
thin wrappers over this attr; `WithArgDirectionsAttr(...)` is the canonical way
to construct an `attrs` vector with the entry set. The existence of this attr
is what `IRProperty::CallDirectionsResolved` verifies after the
`DeriveCallDirections` pass.

### IterArg - Loop-Carried Values

`IterArg` extends `Var` with `initValue_` for SSA-style loops. Scoped to loop body, updated via `yield`, final values in `return_vars`.

```python
# for i, (sum,) in pl.range(n, init_values=(0,)): sum = pl.yield_(sum + i)
sum_iter = ir.IterArg("sum", ir.ScalarType(DataType.INT64), init_val, span)
for_stmt = ir.ForStmt(i, start, stop, step, [sum_iter], body, [sum_final], span)
```

## Statement Nodes

All `Stmt` subclasses inherit a `leading_comments_: vector<string>` metadata
field from the `Stmt` base class. See [Leading comments on statements](#leading-comments-on-statements) below.

| Node Type | Fields | Description |
| --------- | ------ | ----------- |
| **AssignStmt** | `var_` (DefField), `value_` (UsualField) | Variable assignment |
| **IfStmt** | `condition_`, `then_stmts_`, `else_stmts_`, `return_vars_` | Conditional branching |
| **ForStmt** | `loop_var_` (DefField), `start_`, `stop_`, `step_`, `iter_args_` (DefField), `body_`, `return_vars_` (DefField), `kind_` | For loop with optional iteration args |
| **WhileStmt** | `condition_`, `iter_args_` (DefField), `body_`, `return_vars_` (DefField) | While loop with condition and iteration args |
| **InCoreScopeStmt** | `name_hint_`, `body_`, `split_` (optional) | InCore region; outlined to `Function(InCore)` |
| **AutoInCoreScopeStmt** | `name_hint_`, `body_`, `split_` (optional) | Auto-InCore region; consumed by `InterchangeChunkLoops` |
| **ClusterScopeStmt** | `name_hint_`, `body_` | Cluster region; outlined to `Function(Group)` |
| **HierarchyScopeStmt** | `name_hint_`, `body_`, `level_`, `role_` (optional) | Pipeline-stage region for a given Level/Role |
| **SpmdScopeStmt** | `name_hint_`, `body_`, `core_num_` (integer-typed `Expr`), `sync_start_` | SPMD launch region; outlined to `Function(Spmd)` |
| **RuntimeScopeStmt** | `name_hint_`, `body_`, `manual_` | Orchestrator runtime region (`PTO2_SCOPE`); `manual_=true` selects manual dependency mode |
| **YieldStmt** | `values_` | Yield values in loop iteration |
| **EvalStmt** | `expr_` | Evaluate expression for side effects |
| **SeqStmts** | `stmts_` | General statement sequence |
| **BreakStmt** | *(none)* | Exit loop |
| **ContinueStmt** | *(none)* | Skip to next loop iteration |
| **InlineStmt** | `body_`, `language_` (`InlineLanguage`) | Verbatim source body in a target language (e.g. Python). Treated as a leaf by passes; used for HOST SubWorker bodies |

### Leading comments on statements

Each `Stmt` carries an optional `leading_comments_: vector<string>` field that
preserves source-level `#` comments and bare-string docstrings from the Python
DSL. The printer emits each line as `# <text>` directly above the stmt.

- **Constructor arg (symmetric with `span_`).** Every `Stmt` subclass
  constructor takes `leading_comments` as its last parameter (defaulted to
  `{}`). Deserializers read `"leading_comments"` from the fields map and pass
  it alongside `"span"` — the field is initialized at construction time, not
  attached after the fact.
- **Registered as `IgnoreField`.** Comments survive binary serialization
  (`serialize_to_file`), but do NOT participate in `structural_equal` or
  structural hashing. Two stmts that differ only in `leading_comments_`
  compare and hash equal.
- **Read-only from Python.** `stmt.leading_comments` is exposed read-only. The
  sanctioned mutation channel is the free function `ir.attach_leading_comments(stmt, comments)`,
  used by the parser builder and comment-merging passes for late binding.
- **Parser attachment rules.** For simple stmts, comments on lines up to the
  stmt's `end_lineno` are drained as leading — this means same-line trailing
  comments (`y = 1  # note`) attach to the same stmt, not the next one. For
  compound stmts (`for`/`while`/`if`/`with`), draining caps at the header's
  first line so body-inner comments are left for the body stmts. Bare-string
  expressions (docstrings) anywhere in the body become leading comments on
  the next stmt.
- **Tail-of-block comments.** Comments after the last stmt in a block (at the
  block's indentation) have no natural attachment target and are dropped with a
  `UserWarning`. Move them above a stmt or into the outer scope to retain them.
  Column info is used to distinguish genuine tail-of-block comments from
  outer-scope comments that merely appear on intervening lines (e.g. `# fallback`
  between a then-body and `else:`).
- **SeqStmts invariant.** `SeqStmts` is a transparent container and must not
  carry `leading_comments_`; comments always attach to inner (non-Seq) stmts.
- **Pass propagation.** IR passes that rebuild stmts use `MutableCopy(op)` +
  field assignment — the copy auto-preserves `leading_comments_` together with
  every other unchanged field. When a pass splits one stmt into several (e.g.
  `expand_mixed_kernel` expanding an `InCore` call into AIC + AIV), construct
  the split-first stmt via `std::make_shared<NewT>(..., orig->leading_comments_)`
  so the origin's comments attach there. When a pass erases a compound stmt
  (e.g. `unroll_loops` eliminating a `ForStmt`), its comments are forwarded
  onto the first surviving body stmt via `AttachLeadingComments`.

```python
# DSL
"""cache intermediate"""
# reuse later
y = x + 1  # for performance

# Parsed
# AssignStmt.leading_comments == ["cache intermediate", "reuse later", "for performance"]

# Printed
# cache intermediate
# reuse later
# for performance
y: f32 = x + 1
```

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
- Body must end with YieldStmt when iter_args are present, AND no YieldStmt may appear before the trailing position — the yield is the scope's terminator. The same rule applies to ForStmt/IfStmt with non-empty `return_vars_`. Enforced in SSA form by the `SSAVerify` verifier (see `99-verifier.md`, error code `MISPLACED_YIELD`).

### ScopeStmt Details

`ScopeStmt` is an **abstract base class** that marks a region with a specific
execution context. The six concrete subclasses below each carry only the
fields valid for their kind — invalid combinations are unrepresentable at
construction. Use `s.scope_kind` (or `s.GetScopeKind()` in C++) to recover the
kind from a `ScopeStmt`-typed reference, or `isinstance(s, InCoreScopeStmt)`
to dispatch on the concrete type.

All six share the common base fields `name_hint_: str` and `body_: StmtPtr`.
Note that `pl.at(level=Level.CORE_GROUP)` lowers to `InCoreScopeStmt` /
`AutoInCoreScopeStmt`, not `HierarchyScopeStmt` — the parser rejects `role=`
at `CORE_GROUP`. `HierarchyScopeStmt` is reserved for non-`CORE_GROUP` levels
(host, cluster, global) and is not a general replacement for in-core scopes.

```python
# with pl.incore(): y = pl.add(x, x)
in_core = ir.InCoreScopeStmt(name_hint="", body=body, span=span)

# with pl.auto_incore():       (split is optional)
auto = ir.AutoInCoreScopeStmt(name_hint="", body=body, span=span)

# with pl.cluster():
cluster = ir.ClusterScopeStmt(name_hint="", body=body, span=span)

# with pl.at(level=Level.HOST, role=Role.SubWorker):
hier = ir.HierarchyScopeStmt(level=ir.Level.HOST, role=ir.Role.SubWorker,
                             name_hint="", body=body, span=span)

# with pl.spmd(8):
spmd = ir.SpmdScopeStmt(core_num=ir.ConstInt(8, DataType.INDEX, span),
                        sync_start=False, name_hint="", body=body, span=span)

# with pl.manual_scope(): (orchestrator runtime region with manual dep mode)
runtime = ir.RuntimeScopeStmt(manual=True, name_hint="", body=body, span=span)

# for i in pl.spmd(8):                    # loop-style surface syntax
#     offset = i * 64
#     tile = pl.load(a, [offset, 0], [64, 128])
#     ...
# The parser desugars the for-loop to:
#   SpmdScopeStmt(body=InCoreScopeStmt(body=[i = tile.get_block_idx(); ...]))
# so the block index `i` is bound inside the implicit InCore region. The
# OutlineIncoreScopes + OutlineClusterScopes pair then outlines the InCore
# body into a synthetic `Function(InCore)` and the Spmd wrapper into a
# `Function(Spmd)` just like the `with`-form single-kernel case.
```

**Properties:**

- All scope statements are transparent to SSA (no iter_args/return_vars) and
  are not control flow (execute once, linearly).
- Required fields are enforced at construction: `HierarchyScopeStmt.level_`
  is non-optional; `SpmdScopeStmt.core_num_` is a non-null `ExprPtr`. The
  expression can be any integer-typed IR value — `Simplify` folds closure
  arithmetic to `ConstInt`, and codegen resolves `Var` references against
  the enclosing function scope.
- `InCoreScopeStmt` / `AutoInCoreScopeStmt` are scheduled for deprecation;
  prefer `HierarchyScopeStmt` or other surviving kinds in new code.
- Pass behavior:
  - `InterchangeChunkLoops` consumes `AutoInCoreScopeStmt`
  - `OutlineIncoreScopes` extracts `InCoreScopeStmt` into `Function(InCore)`
  - `OutlineClusterScopes` extracts `ClusterScopeStmt` into `Function(Group)`
    and standalone `SpmdScopeStmt` into `Function(Spmd)`
  - `OutlineHierarchyScopes` extracts `HierarchyScopeStmt`
  - `DeriveManualScopeDeps` populates `Call.attrs["manual_dep_edges"]` for
    every kernel call inside `RuntimeScopeStmt(manual=true)` so codegen can
    emit explicit `params.add_dep(task_<m>);` calls.
- `RuntimeScopeStmt` lowers to `PTO2_SCOPE()` for `manual=false` and
  `PTO2_SCOPE(PTO2ScopeMode::MANUAL)` for `manual=true`. It is created by
  `pl.manual_scope()` (manual mode) and by the orchestration codegen path
  (auto mode); it is **not** outlined into a separate function.

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

The `kind_` field (`ForKind` enum) distinguishes sequential (`ForKind.Sequential`, default), parallel (`ForKind.Parallel`), unroll (`ForKind.Unroll`), and pipeline (`ForKind.Pipeline`) loops. In the DSL, `pl.range()` produces sequential, `pl.parallel()` produces parallel, `pl.unroll()` produces compile-time unrolled loops, and `pl.pipeline(N, stage=F)` produces software-pipelined loops. The printer emits `pl.parallel(...)`, `pl.unroll(...)`, or `pl.pipeline(..., stage=F)` accordingly. `ForKind.Pipeline` is a transient marker — `LowerPipelineLoops` replicates the body F times and keeps the kind as a scope marker, then `CanonicalizeIOOrder` reorders the body's IO and demotes the kind back to `Sequential`.

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
| `func_type_` | FunctionType | Function type (Opaque, Orchestration, InCore, AIC, AIV, Group, or Spmd) |
| `params_` | list[VarPtr] | Parameter variables (DefField) |
| `param_directions_` | list[ParamDirection] | Parameter directions, same length as params_ |
| `return_types_` | list[TypePtr] | Return types |
| `body_` | StmtPtr | Function body |
| `level_` | optional[Level] | Hierarchy level (auto-derived from `func_type_` for InCore/AIC/AIV/Group/Orchestration; see below) |
| `role_` | optional[Role] | Hierarchy role (auto-derived from `func_type_` for InCore/AIC/AIV/Group/Orchestration; see below) |
| `attrs_` | list[(str, Any)] | Ordered free-form metadata, exposed as `UsualField` (participates in structural traversal) |

### Auto-derivation of `level_` / `role_`

For `func_type_` in {`InCore`, `AIC`, `AIV`, `Group`, `Orchestration`}, the
`Function` constructor auto-derives `level_` and `role_` when they are not
explicitly provided:

| `func_type_` | Derived `level_` | Derived `role_` |
| ------------ | ---------------- | --------------- |
| `Orchestration` | `CHIP` | `Orchestrator` |
| `InCore` | `CHIP_DIE` | `Worker` |
| `AIC` | `AIC` | `Worker` |
| `AIV` | `AIV` | `Worker` |
| `Group` | `CORE_GROUP` | `Worker` |

If `level_` / `role_` are explicitly supplied, they must match the derived
values — otherwise construction fails with `pypto.ValueError`. For other
function types (`Opaque`, `Spmd`), no derivation is applied and both fields
remain `nullopt` unless set by the caller. The Python printer emits the
`level=` / `role=` keywords on the `@pl.function(...)` decorator whenever they
are present.

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
| `Spmd` | SPMD data-parallel dispatch wrapper |

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
| **Statements** | 16 | AssignStmt, IfStmt, ForStmt, WhileStmt, ReturnStmt, InCoreScopeStmt, AutoInCoreScopeStmt, ClusterScopeStmt, HierarchyScopeStmt, SpmdScopeStmt, YieldStmt, EvalStmt, SeqStmts, BreakStmt, ContinueStmt, InlineStmt |
| **Types** | 6 | ScalarType, TensorType, TileType, TupleType, PipeType, UnknownType |
| **Functions** | 2 | Function, Program |

## Related Documentation

- [IR Overview](00-overview.md) - Core concepts and design principles
- [IR Types and Examples](02-types.md) - Type system details and examples
- [Structural Comparison](03-structural_comparison.md) - Equality and hashing
