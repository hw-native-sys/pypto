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

对于 `TileType`，每个可选参数最多只能出现一次。如果存在 `MemRef`，
则必须在 `TileType` 上同时显式提供 `memory_space`。

## 表达式节点

| 节点类型 | 字段 | 说明 |
| -------- | ---- | ---- |
| **Var** | `name_hint_`, `type_` | 变量引用（以指针为标识，而非名称） |
| **IterArg** | `name_hint_`, `type_`, `initValue_` | 循环迭代参数（扩展自 Var） |
| **ConstInt** | `value_`, `dtype_` | 整数常量 |
| **ConstBool** | `value_` | 布尔常量（始终为 BOOL dtype） |
| **ConstFloat** | `value_`, `dtype_` | 浮点常量 |
| **Call** | `op_`, `args_`, `kwargs_`, `attrs_` | 函数/运算符调用（参见 [Call attrs 与 kwargs 的区别](#call-attrs-与-kwargs-的区别)） |
| **TupleGetItemExpr** | `tuple_`, `index_` | 元组元素访问 |

### Var 的标识（Identity）

变量的标识由**对象指针**（或等价的 `unique_id_`）决定，**而非** `name_hint_`。两个具有相同 `name_hint_` 的 `Var` 对象，如果是不同对象，则是不同的变量。该字段命名为 `name_hint_`（而非 `name_`），正是为了明确这一语义。

| 字段 | 用途 |
| ---- | ---- |
| `name_hint_` | 用于打印和调试的装饰性标签。属于 `IgnoreField` —— 不参与结构比较和哈希。 |
| `unique_id_` | 构造时分配的单调递增 ID，用于确定性哈希。 |
| 对象指针 | 权威标识 —— 两个引用指向同一变量，当且仅当它们指向同一个 `Var` 对象。 |

```cpp
// 相同的 name_hint，但是不同的变量
auto x1 = std::make_shared<Var>("x", type, span);
auto x2 = std::make_shared<Var>("x", type, span);
// x1 != x2 —— 尽管共享名称 "x"，它们是不同的变量

// 同一个变量的两次引用
auto x_ref = x1;
// x1 == x_ref —— 相同的指针，相同的变量
```

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

### Call attrs 与 kwargs 的区别

`Call` 同时持有两个有序的字符串键映射，C++ 类型完全一致
（`std::vector<std::pair<std::string, std::any>>`），但所有权与语义完全不同：

| 字段 | 用途 | 来源 | 保留键 |
| ---- | ---- | ---- | ------ |
| `kwargs_` | 用户在调用点书写的语言层关键字参数（例如 `kernel(x, y, axis=2)`），在 parser、printer 与 Python bindings 之间作为面向用户的数据原样往返。 | 前端 / DSL 解析 | 无 —— 键由用户代码决定。 |
| `attrs_` | 编译器内部的节点元数据，由 pass / verifier 生产与消费，不会作为 DSL 关键字参数暴露给用户。 | 编译器 pass（以及反序列化器在加载旧 payload 时使用） | `"arg_directions"`（见下文）。后续新增的内部属性建议使用 `"hint.*"`、`"profile.*"` 这类带前缀的命名空间。 |

**`arg_directions` 的存放位置。** 解析后的逐参数 `ArgDirection` 序列以
`std::vector<ArgDirection>` 形式存储在保留键 `attrs_["arg_directions"]` 下；
访问器 `Call::HasArgDirections()`、`Call::GetArgDirections()`（以及 Python
端的 `Call.arg_directions` 属性）都是该 attr 的薄封装，
`WithArgDirectionsAttr(...)` 是构造带该 attr 的 `attrs` 向量的标准入口。
`IRProperty::CallDirectionsResolved` 校验的就是该 attr 在 `DeriveCallDirections`
pass 之后是否存在。

### IterArg - 循环携带值

`IterArg` 扩展 `Var`，添加 `initValue_` 以支持静态单赋值 (SSA) 风格的循环。作用域限定在循环体内，通过 `yield` 更新，最终值存储在 `return_vars` 中。

```python
# for i, (sum,) in pl.range(n, init_values=(0,)): sum = pl.yield_(sum + i)
sum_iter = ir.IterArg("sum", ir.ScalarType(DataType.INT64), init_val, span)
for_stmt = ir.ForStmt(i, start, stop, step, [sum_iter], body, [sum_final], span)
```

## 语句节点

所有 `Stmt` 子类都从 `Stmt` 基类继承一个 `leading_comments_: vector<string>` 元数据字段。详见下文 [语句的前导注释](#语句的前导注释)。

| 节点类型 | 字段 | 说明 |
| -------- | ---- | ---- |
| **AssignStmt** | `var_` (DefField), `value_` (UsualField) | 变量赋值 |
| **IfStmt** | `condition_`, `then_stmts_`, `else_stmts_`, `return_vars_` | 条件分支 |
| **ForStmt** | `loop_var_` (DefField), `start_`, `stop_`, `step_`, `iter_args_` (DefField), `body_`, `return_vars_` (DefField), `kind_` | 带可选迭代参数的 for 循环 |
| **WhileStmt** | `condition_`, `iter_args_` (DefField), `body_`, `return_vars_` (DefField) | 带条件和迭代参数的 while 循环 |
| **InCoreScopeStmt** | `name_hint_`, `body_`, `split_`（可选） | InCore 区域；由 `OutlineIncoreScopes` 提取为 `Function(InCore)` |
| **AutoInCoreScopeStmt** | `name_hint_`, `body_`, `split_`（可选） | Auto-InCore 区域；由 `InterchangeChunkLoops` 消费 |
| **ClusterScopeStmt** | `name_hint_`, `body_` | Cluster 区域；由 `OutlineClusterScopes` 提取为 `Function(Group)` |
| **HierarchyScopeStmt** | `name_hint_`, `body_`, `level_`, `role_`（可选） | 给定 Level/Role 的流水线阶段区域 |
| **SpmdScopeStmt** | `name_hint_`, `body_`, `core_num_`（整型 `Expr`）, `sync_start_` | SPMD 启动区域；提取为 `Function(Spmd)` |
| **RuntimeScopeStmt** | `name_hint_`, `body_`, `manual_` | Orchestrator 运行时区域（`PTO2_SCOPE`）；`manual_=true` 选择手工依赖模式 |
| **YieldStmt** | `values_` | 在循环迭代中产出值 |
| **EvalStmt** | `expr_` | 为副作用求值表达式 |
| **SeqStmts** | `stmts_` | 通用语句序列 |
| **BreakStmt** | *(无)* | 退出循环 |
| **ContinueStmt** | *(无)* | 跳至下一次循环迭代 |
| **InlineStmt** | `body_`、`language_`（`InlineLanguage`） | 目标语言（如 Python）的逐字源码片段。各 pass 视其为叶子节点；用于 HOST SubWorker 函数体 |

### 语句的前导注释

每个 `Stmt` 都带有一个可选的 `leading_comments_: vector<string>` 字段，用于保留 Python DSL 中的源码级 `#` 注释和裸字符串文档字符串（docstring）。打印器会将每一行以 `# <text>` 的形式输出在该语句上方。

- **构造函数参数（与 `span_` 对称）。** 每个 `Stmt` 子类的构造函数都在最后增加了 `leading_comments` 形参（默认值为 `{}`）。反序列化器从字段表中读取 `"leading_comments"`，与 `"span"` 一起传入构造函数——该字段在构造时即完成初始化，而非事后附加。
- **注册为 `IgnoreField`。** 注释会在二进制序列化（`serialize_to_file`）中保留，但不参与 `structural_equal` 或结构哈希。两个仅在 `leading_comments_` 上有差异的语句相等且哈希一致。
- **Python 侧只读。** `stmt.leading_comments` 仅暴露为只读。官方的修改通道是自由函数 `ir.attach_leading_comments(stmt, comments)`，供解析器构造器和合并注释的 pass 在晚期绑定时使用。
- **解析器附着规则。** 对于简单语句，不晚于该语句 `end_lineno` 的注释会被作为前导注释收集——这意味着同一行的尾随注释（`y = 1  # note`）附着到当前语句本身，而非下一条语句。对于复合语句（`for`/`while`/`if`/`with`），收集上限为首行行号，以便函数体内部的注释由内部语句自身收集。函数体中任何位置的裸字符串表达式（docstring）都会成为下一条语句的前导注释。
- **块末尾注释。** 出现在块中最后一条语句之后（并与块同级缩进）的注释没有合适的附着目标，将被丢弃并发出 `UserWarning`。将它们移到某条语句之上或外层作用域以保留它们。列信息用于区分真正的块末尾注释和仅仅出现在中间行的外层注释（例如 `else:` 前的 `# fallback`）。
- **SeqStmts 不变式。** `SeqStmts` 是一个透明容器，不应直接持有 `leading_comments_`；注释始终附着到其内部的（非 Seq）语句上。
- **Pass 传递。** 重建语句的 IR pass 采用 `MutableCopy(op)` + 字段赋值——副本会自动保留 `leading_comments_` 以及其他所有未改动的字段。当一个 pass 将一条语句拆分为多条时（例如 `expand_mixed_kernel` 将 `InCore` 调用拆为 AIC + AIV），通过 `std::make_shared<NewT>(..., orig->leading_comments_)` 构造第一条新语句，使原语句的注释附着到第一条发出的语句上。当一个 pass 删除一条复合语句时（例如 `unroll_loops` 消除 `ForStmt`），其注释通过 `AttachLeadingComments` 转移到第一条留存的 body 语句上。

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

### ForStmt 详细说明

```python
# Without iter_args: for i in pl.range(10): x = x + i
for_stmt = ir.ForStmt(i, start, stop, step, [], body, [], span)

# With iter_args: for i, (sum,) in pl.range(10, init_values=(0,)): sum = pl.yield_(sum + i)
for_stmt = ir.ForStmt(i, start, stop, step, [sum_iter], body, [sum_final], span)
```

> **注意:** DSL 接受简写形式 `pl.range(stop)` / `pl.range(start, stop)` 作为语法糖（类似 Python 的 `range()`）。IR 始终存储三个字段（`start_`、`stop_`、`step_`）；解析器填充默认值（start=0, step=1），打印器在匹配时省略它们。

### WhileStmt 详细说明

```python
# Natural: while x < 10: x = x + 1
while_stmt = ir.WhileStmt(condition, [], body, [], span)

# SSA form: for (x,) in pl.while_(init_values=(0,)): pl.cond(x < 10); x = pl.yield_(x + 1)
while_stmt = ir.WhileStmt(condition, [x_iter], body, [x_final], span)
```

**属性：** `condition_` 每次迭代都会求值；支持 SSA iter_args/return_vars；DSL 使用 `pl.cond()` 作为第一条语句。

- 不带 iter_args 的自然语法通过 ConvertToSSA Pass 转换为 SSA
- 存在 iter_args 时，循环体必须以 YieldStmt 结尾，并且尾部之前的位置不允许再出现 YieldStmt——yield 是作用域的终结语句。同一规则同样适用于 `return_vars_` 非空的 ForStmt / IfStmt。在 SSA 形式下由 `SSAVerify` 强制（参见 `99-verifier.md`，错误码 `MISPLACED_YIELD`）。

### ScopeStmt 详细说明

`ScopeStmt` 是一个**抽象基类**，用于标记具有特定执行上下文的区域。下列六个具体子类
各自只携带其类型有效的字段——非法组合在构造时即不可表达。在 `ScopeStmt` 类型的引用上，
可使用 `s.scope_kind`（C++ 中为 `s.GetScopeKind()`）来取回类型，或使用
`isinstance(s, InCoreScopeStmt)` 在具体类型上分派。

六个子类共享公共基类字段 `name_hint_: str` 和 `body_: StmtPtr`。注意：
`pl.at(level=Level.CORE_GROUP)` 实际下沉到 `InCoreScopeStmt` /
`AutoInCoreScopeStmt`，而非 `HierarchyScopeStmt`——解析器会在 `CORE_GROUP`
拒绝 `role=`。`HierarchyScopeStmt` 仅用于非 `CORE_GROUP` 的层级
（host、cluster、global），并不是 in-core 作用域的通用替代。

```python
# with pl.incore(): y = pl.add(x, x)
in_core = ir.InCoreScopeStmt(name_hint="", body=body, span=span)

# with pl.auto_incore():       (split 可选)
auto = ir.AutoInCoreScopeStmt(name_hint="", body=body, span=span)

# with pl.cluster():
cluster = ir.ClusterScopeStmt(name_hint="", body=body, span=span)

# with pl.at(level=Level.HOST, role=Role.SubWorker):
hier = ir.HierarchyScopeStmt(level=ir.Level.HOST, role=ir.Role.SubWorker,
                             name_hint="", body=body, span=span)

# with pl.spmd(8):
spmd = ir.SpmdScopeStmt(core_num=ir.ConstInt(8, DataType.INDEX, span),
                        sync_start=False, name_hint="", body=body, span=span)

# with pl.manual_scope(): (orchestrator 运行时区域，使用手工依赖模式)
runtime = ir.RuntimeScopeStmt(manual=True, name_hint="", body=body, span=span)

# for i in pl.spmd(8):                    # loop-style 语法糖
#     offset = i * 64
#     tile = pl.load(a, [offset, 0], [64, 128])
#     ...
# 解析器会将此 for-loop 脱糖为：
#   SpmdScopeStmt(body=InCoreScopeStmt(body=[i = tile.get_block_idx(); ...]))
# 这样块索引 `i` 就在隐式的 InCore 区域里被绑定。随后
# `OutlineIncoreScopes` + `OutlineClusterScopes` 会把 InCore 体提取为
# 合成的 `Function(InCore)`，并把 Spmd 包装提取为 `Function(Spmd)`，
# 行为与 `with`-form 单内核调用路径一致。
```

**属性：**

- 所有作用域语句对 SSA 透明（无 iter_args/return_vars），且不是控制流
  （执行一次，线性执行）。
- 必填字段在构造时强制校验：`HierarchyScopeStmt.level_` 不可为空；
  `SpmdScopeStmt.core_num_` 为非空 `ExprPtr`。表达式可以是任何整型 IR
  值——`Simplify` 会折叠闭包算术为 `ConstInt`，codegen 则按闭合函数作用
  域解析 `Var` 引用。
- `InCoreScopeStmt` / `AutoInCoreScopeStmt` 已计划弃用；新代码应优先使用
  `HierarchyScopeStmt` 或其它将保留的子类。
- Pass 行为：
  - `InterchangeChunkLoops` 消费 `AutoInCoreScopeStmt`
  - `OutlineIncoreScopes` 将 `InCoreScopeStmt` 提取为 `Function(InCore)`
  - `OutlineClusterScopes` 将 `ClusterScopeStmt` 提取为 `Function(Group)`，
    将独立的 `SpmdScopeStmt` 提取为 `Function(Spmd)`
  - `OutlineHierarchyScopes` 提取 `HierarchyScopeStmt`
  - `DeriveManualScopeDeps` 为 `RuntimeScopeStmt(manual=true)` 内每个
    kernel call 填充 `Call.attrs["manual_dep_edges"]`，让 codegen 能发出
    显式 `params.add_dep(task_<m>);` 调用。
- `RuntimeScopeStmt` 在 `manual=false` 时下沉为 `PTO2_SCOPE()`，在
  `manual=true` 时下沉为 `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`。它由
  `pl.manual_scope()`（manual 模式）和 orchestration codegen 路径
  （auto 模式）创建；**不会**被独立外提为函数。

**变换示例：**

```python
# Before: with pl.incore(): y = pl.add(x, x); return y
# After: main_incore_0(x) -> y; main(x): y = main_incore_0(x); return y
```

**并行 for 循环 (ForKind)：**

```python
# for i in pl.parallel(10): ...
for_stmt = ir.ForStmt(i, start, stop, step, [], body, [], span, ir.ForKind.Parallel)
```

`kind_` 字段（`ForKind` 枚举）区分顺序执行（`ForKind.Sequential`，默认）、并行执行（`ForKind.Parallel`）、编译时展开（`ForKind.Unroll`）和软件流水线（`ForKind.Pipeline`）的循环。在 DSL 中，`pl.range()` 生成顺序循环，`pl.parallel()` 生成并行循环，`pl.unroll()` 生成编译时展开循环，`pl.pipeline(N, stage=F)` 生成软件流水线循环。打印器相应输出 `pl.parallel(...)`、`pl.unroll(...)` 或 `pl.pipeline(..., stage=F)`。`ForKind.Pipeline` 是临时标记：`LowerPipelineLoops` 将循环体复制 F 份并保留该 kind 作为作用域标记，随后 `CanonicalizeIOOrder` 重排循环体 IO 并把 kind 降回 `Sequential`。

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
| **TileType** | `shape_`, `dtype_`, `memref_`（可选）, `tile_view_`（可选）, `memory_space_`（可选） | 统一缓冲区中的 Tile |
| **TupleType** | `types_` | 类型元组 |
| **PipeType** | `pipe_kind_` | 硬件流水线/屏障 |
| **UnknownType** | - | 未知或推断类型 |

### 内存引用 (MemRef)

描述张量/Tile 共享的内存分配元数据。对于 Tile，内存空间保存在
`TileType.memory_space_`；`TensorType` 的规范内存空间固定为 DDR。

| 字段 | 类型 | 说明 |
| ---- | ---- | ---- |
| `addr_` | ExprPtr | 基地址 |
| `size_` | size_t | 大小（字节） |
| `id_` | uint64_t | 稳定的 MemRef 标识符 |

```python
memref = ir.MemRef(
    ir.ConstInt(0x1000, DataType.INT64, span),
    1024,  # bytes
    0     # id
)
```

> **注意：** `ir.Mem` 是 `ir.MemorySpace` 的简写别名。

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
| `func_type_` | FunctionType | 函数类型（Opaque、Orchestration、InCore、AIC、AIV、Group 或 Spmd） |
| `params_` | list[VarPtr] | 参数变量 (DefField) |
| `param_directions_` | list[ParamDirection] | 参数方向，与 params_ 长度相同 |
| `return_types_` | list[TypePtr] | 返回类型 |
| `body_` | StmtPtr | 函数体 |
| `level_` | optional[Level] | 层次级别（对 InCore/AIC/AIV/Group/Orchestration 自动派生，详见下文） |
| `role_` | optional[Role] | 层次角色（对 InCore/AIC/AIV/Group/Orchestration 自动派生，详见下文） |
| `attrs_` | list[(str, Any)] | 有序的自由形式元数据，以 `UsualField` 暴露（参与结构遍历） |

### `level_` / `role_` 自动派生

当 `func_type_` 属于 {`InCore`, `AIC`, `AIV`, `Group`, `Orchestration`} 时，
`Function` 构造函数会在未显式提供 `level_` / `role_` 时自动派生：

| `func_type_` | 派生的 `level_` | 派生的 `role_` |
| ------------ | --------------- | -------------- |
| `Orchestration` | `CHIP` | `Orchestrator` |
| `InCore` | `CHIP_DIE` | `Worker` |
| `AIC` | `AIC` | `Worker` |
| `AIV` | `AIV` | `Worker` |
| `Group` | `CORE_GROUP` | `Worker` |

如果显式提供 `level_` / `role_`，其值必须与派生值一致，否则构造时抛出
`pypto.ValueError`。其他函数类型（`Opaque`、`Spmd`）不进行派生，除非
调用方显式设置，否则两个字段保持 `nullopt`。当 `level_` / `role_` 存在
时，Python 打印器会在 `@pl.function(...)` 装饰器上输出 `level=` / `role=`
关键字。

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
| `InCore` | AICore 子图执行（未特化） |
| `AIC` | Cube 核心内核（特化的 InCore） |
| `AIV` | Vector 核心内核（特化的 InCore） |
| `Group` | AIC + AIV 内核的协调调度组 |
| `Spmd` | SPMD 数据并行调度封装 |

`IsInCoreType(type)` / `ir.is_incore_type(type)` 对 `InCore`、`AIC` 和 `AIV` 返回 `True`。

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
| **二元运算** | 23 | Add, Sub, Mul, FloorDiv, FloorMod, FloatDiv, Min, Max, Pow, Eq, Ne, Lt, Le, Gt, Ge, And, Or, Xor, BitAnd, BitOr, BitXor, BitShiftLeft, BitShiftRight |
| **一元运算** | 5 | Abs, Neg, Not, BitNot, Cast |
| **调用/访问** | 2 | Call, TupleGetItemExpr |
| **操作** | 2 | Op, GlobalVar |
| **语句** | 16 | AssignStmt, IfStmt, ForStmt, WhileStmt, ReturnStmt, InCoreScopeStmt, AutoInCoreScopeStmt, ClusterScopeStmt, HierarchyScopeStmt, SpmdScopeStmt, YieldStmt, EvalStmt, SeqStmts, BreakStmt, ContinueStmt, InlineStmt |
| **类型** | 6 | ScalarType, TensorType, TileType, TupleType, PipeType, UnknownType |
| **函数** | 2 | Function, Program |

## 相关文档

- [IR 概述](00-overview.md) - 核心概念与设计原则
- [IR 类型与示例](02-types.md) - 类型系统详情与示例
- [结构比较](03-structural_comparison.md) - 相等性和哈希
