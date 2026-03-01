# PyPTO IR 节点层次结构

本文档提供所有 IR 节点类型的完整参考，按类别组织。

## BNF 语法

```bnf
<program>    ::= [ identifier ":" ] { <function> }
<function>   ::= "def" identifier "(" [ <param_list> ] ")" [ "->" <type_list> ] ":" <stmt>
<param_list> ::= <param> { "," <param> }
<param>      ::= <var> | "(" <var> "," <param_direction> ")"
<param_direction> ::= "In" | "Out" | "InOut"
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

## 表达式节点

| 节点类型 | 字段 | 说明 |
| -------- | ---- | ---- |
| **Var** | `name_`, `type_` | 变量引用 |
| **IterArg** | `name_`, `type_`, `initValue_` | 循环迭代参数（扩展自 Var） |
| **ConstInt** | `value_`, `dtype_` | 整数常量 |
| **ConstBool** | `value_` | 布尔常量（始终为 BOOL dtype） |
| **ConstFloat** | `value_`, `dtype_` | 浮点常量 |
| **Call** | `op_`, `args_`, `kwargs_` | 函数/运算符调用 |
| **TupleGetItemExpr** | `tuple_`, `index_` | 元组元素访问 |

### 二元表达式节点

| 类别 | 节点 |
| ---- | ---- |
| **算术运算** | Add, Sub, Mul, FloorDiv, FloorMod, FloatDiv |
| **数学运算** | Min, Max, Pow |
| **比较运算** | Eq, Ne, Lt, Le, Gt, Ge |
| **逻辑运算** | And, Or, Xor |
| **位运算** | BitAnd, BitOr, BitXor, BitShiftLeft, BitShiftRight |

所有二元表达式包含：`lhs_`、`rhs_`、`dtype_`

### 一元表达式节点

| 节点 | 运算 |
| ---- | ---- |
| **Abs** | 绝对值 |
| **Neg** | 取反 |
| **Not** | 逻辑非 |
| **BitNot** | 按位取反 |
| **Cast** | 类型转换 |

所有一元表达式包含：`operand_`、`dtype_`

### Op 和 GlobalVar

| 节点类型 | 用途 | 使用场景 |
| -------- | ---- | -------- |
| **Op** | 通用操作/函数引用 | 外部运算符、内置函数 |
| **GlobalVar** | 程序内的函数引用 | 程序内函数调用 |

```python
op = ir.Op("my_function"); call = ir.Call(op, [x, y], span)  # External
gvar = ir.GlobalVar("helper"); call = ir.Call(gvar, [x], span)  # Internal
```

### IterArg - 循环传递值

`IterArg` 扩展 `Var`，添加 `initValue_` 以支持静态单赋值 (SSA) 风格的循环。作用域限定在循环体内，通过 `yield` 更新，最终值存储在 `return_vars` 中。

```python
# for i, (sum,) in pl.range(0, n, 1, init_values=(0,)): sum = pl.yield_(sum + i)
sum_iter = ir.IterArg("sum", ir.ScalarType(DataType.INT64), init_val, span)
for_stmt = ir.ForStmt(i, start, stop, step, [sum_iter], body, [sum_final], span)
```

## 语句节点

| 节点类型 | 字段 | 说明 |
| -------- | ---- | ---- |
| **AssignStmt** | `var_` (DefField), `value_` (UsualField) | 变量赋值 |
| **IfStmt** | `condition_`, `then_stmts_`, `else_stmts_`, `return_vars_` | 条件分支 |
| **ForStmt** | `loop_var_` (DefField), `start_`, `stop_`, `step_`, `iter_args_` (DefField), `body_`, `return_vars_` (DefField), `kind_` | 带可选迭代参数的 for 循环 |
| **WhileStmt** | `condition_`, `iter_args_` (DefField), `body_`, `return_vars_` (DefField) | 带条件和迭代参数的 while 循环 |
| **ScopeStmt** | `scope_kind_`, `body_` | 标记具有特定执行上下文的区域（如 InCore） |
| **YieldStmt** | `values_` | 在循环迭代中产出值 |
| **EvalStmt** | `expr_` | 为副作用求值表达式 |
| **SeqStmts** | `stmts_` | 通用语句序列 |
| **OpStmts** | `stmts_` | 赋值语句序列 |
| **BreakStmt** | *(无)* | 退出循环 |
| **ContinueStmt** | *(无)* | 跳至下一次循环迭代 |

### ForStmt 详细说明

```python
# Without iter_args: for i in range(0, 10, 1): x = x + i
for_stmt = ir.ForStmt(i, start, stop, step, [], body, [], span)

# With iter_args: for i, (sum,) in pl.range(0, 10, 1, init_values=(0,)): sum = pl.yield_(sum + i)
for_stmt = ir.ForStmt(i, start, stop, step, [sum_iter], body, [sum_final], span)
```

### WhileStmt 详细说明

```python
# Natural: while x < 10: x = x + 1
while_stmt = ir.WhileStmt(condition, [], body, [], span)

# SSA form: for (x,) in pl.while_(init_values=(0,)): pl.cond(x < 10); x = pl.yield_(x + 1)
while_stmt = ir.WhileStmt(condition, [x_iter], body, [x_final], span)
```

**属性：** `condition_` 每次迭代都会求值；支持 SSA iter_args/return_vars；DSL 使用 `pl.cond()` 作为第一条语句。

- 不带 iter_args 的自然语法通过 ConvertToSSA Pass 转换为 SSA
- 存在 iter_args 时，循环体必须以 YieldStmt 结尾

### ScopeStmt 详细说明

标记具有特定执行上下文的区域（如用于 AICore 子图的 InCore）。

```python
# with pl.incore(): y = pl.add(x, x)
scope_stmt = ir.ScopeStmt(ir.ScopeKind.InCore, body, span)
```

**属性：**

- `scope_kind_`：执行上下文（`ScopeKind.InCore`）
- `body_`：嵌套语句
- 对 SSA 透明（无 iter_args/return_vars）
- 不是控制流（执行一次，线性执行）
- `OutlineIncoreScopes` Pass 将其提取为 `Function(InCore)`

**变换示例：**

```python
# Before: with pl.incore(): y = pl.add(x, x); return y
# After: main_incore_0(x) -> y; main(x): y = main_incore_0(x); return y
```

**并行 for 循环 (ForKind)：**

```python
# for i in pl.parallel(0, 10, 1): ...
for_stmt = ir.ForStmt(i, start, stop, step, [], body, [], span, ir.ForKind.Parallel)
```

`kind_` 字段（`ForKind` 枚举）区分顺序执行（`ForKind.Sequential`，默认）和并行执行（`ForKind.Parallel`）的循环。在 DSL 中，`pl.range()` 生成顺序循环，`pl.parallel()` 生成并行循环。打印器对并行类型输出 `pl.parallel(...)`。

**要求：**

- yield 的值数量 = IterArgs 数量
- return_vars 数量 = IterArgs 数量
- IterArgs 仅在循环体内可访问
- return_vars 在循环之后可访问

## 类型节点

| 节点类型 | 字段 | 说明 |
| -------- | ---- | ---- |
| **ScalarType** | `dtype_` | 标量类型（INT64、FP32 等） |
| **TensorType** | `shape_`, `dtype_`, `memref_`（可选） | 多维张量 (Tensor) |
| **TileType** | `shape_`, `dtype_`, `memref_`（可选）, `tile_view_`（可选） | 统一缓冲区中的 Tile |
| **TupleType** | `types_` | 类型元组 |
| **PipeType** | `pipe_kind_` | 硬件管线/屏障 |
| **UnknownType** | - | 未知或推断类型 |

### 内存引用 (MemRef)

描述张量/Tile 的内存分配：

| 字段 | 类型 | 说明 |
| ---- | ---- | ---- |
| `memory_space_` | MemorySpace 枚举 | DDR, Vec, Mat, Left, Right, Acc |
| `addr_` | ExprPtr | 基地址 |
| `size_` | size_t | 大小（字节） |

```python
memref = ir.MemRef(
    ir.MemorySpace.DDR,
    ir.ConstInt(0x1000, DataType.INT64, span),
    1024  # bytes
)
```

### TileView - Tile 布局

描述 Tile 的布局和访问模式：

| 字段 | 类型 | 说明 |
| ---- | ---- | ---- |
| `valid_shape` | list[ExprPtr] | 有效维度 |
| `stride` | list[ExprPtr] | 每维步长 |
| `start_offset` | ExprPtr | 起始偏移量 |

```python
tile_view = ir.TileView()
tile_view.valid_shape = [ir.ConstInt(16, DataType.INT64, span)] * 2
tile_view.stride = [ir.ConstInt(1, DataType.INT64, span), ir.ConstInt(16, DataType.INT64, span)]
tile_view.start_offset = ir.ConstInt(0, DataType.INT64, span)
```

## Function 节点

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

| 字段 | 类型 | 说明 |
| ---- | ---- | ---- |
| `name_` | string | 函数名称 |
| `func_type_` | FunctionType | 函数类型（Opaque、Orchestration 或 InCore） |
| `params_` | list[VarPtr] | 参数变量 (DefField) |
| `param_directions_` | list[ParamDirection] | 参数方向，与 params_ 长度相同 |
| `return_types_` | list[TypePtr] | 返回类型 |
| `body_` | StmtPtr | 函数体 |

### ParamDirection 枚举

| 值 | 说明 |
| -- | ---- |
| `In` | 只读输入参数（默认） |
| `Out` | 只写输出参数 |
| `InOut` | 读写输入/输出参数 |

### FunctionType 枚举

| 值 | 说明 |
| -- | ---- |
| `Opaque` | 未指定的函数类型（默认） |
| `Orchestration` | 运行在主机/AICPU 上，用于控制流和依赖分析 |
| `InCore` | 在特定 AICore 上的子图 |

## Program 节点

包含多个函数的容器，具有确定性排序：

| 字段 | 类型 | 说明 |
| ---- | ---- | ---- |
| `name_` | string | 程序名称 (IgnoreField) |
| `functions_` | map[GlobalVarPtr, FunctionPtr] | 函数的有序映射 |

```python
program = ir.Program([func1, func2], "my_program", span)
add_func = program.get_function("add")  # Access by name
```

函数存储在有序映射中，以确保确定性排序。GlobalVar 名称必须与函数名称匹配。

## 按类别汇总的节点

| 类别 | 数量 | 节点 |
| ---- | ---- | ---- |
| **基类** | 4 | IRNode, Expr, Stmt, Type |
| **变量** | 2 | Var, IterArg |
| **常量** | 3 | ConstInt, ConstFloat, ConstBool |
| **二元运算** | 18 | Add, Sub, Mul, FloorDiv, FloorMod, FloatDiv, Min, Max, Pow, Eq, Ne, Lt, Le, Gt, Ge, And, Or, Xor, BitAnd, BitOr, BitXor, BitShiftLeft, BitShiftRight |
| **一元运算** | 5 | Abs, Neg, Not, BitNot, Cast |
| **调用/访问** | 2 | Call, TupleGetItemExpr |
| **操作** | 2 | Op, GlobalVar |
| **语句** | 11 | AssignStmt, IfStmt, ForStmt, WhileStmt, ScopeStmt, YieldStmt, EvalStmt, SeqStmts, OpStmts, BreakStmt, ContinueStmt |
| **类型** | 6 | ScalarType, TensorType, TileType, TupleType, PipeType, UnknownType |
| **函数** | 2 | Function, Program |

## 相关文档

- [IR 概述](00-overview.md) - 核心概念与设计原则
- [IR 类型与示例](02-types.md) - 类型系统详情与示例
- [结构比较](03-structural_comparison.md) - 相等性和哈希
