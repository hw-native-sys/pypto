# Python IR 语法规范

## 概述

PyPTO IR 的 Python 风格语法:

- **完整**: 包含重构 IR 所需的全部信息
- **可解析 (Parser)**: 可解析回 IR (参见 [IR 解析器](../ir/07-parser.md))
- **Pythonic**: 遵循 Python 风格, 通过大部分代码检查工具
- **静态单赋值 (SSA) 风格**: 使用 SSA, 配合 `pl.yield_()` 和 `pl.range()`

## 模块结构

```python
# pypto.program: program_name
import pypto.language as pl
```

对于未命名程序: `# pypto.program`

**注意:** 模块前缀可配置 (默认 `pl`, 旧版 `ir`, 支持自定义)。

## 类型系统

### 标量类型

```python
x: pl.INT64
y: pl.FP32
z: pl.BOOL
```

可用类型:

| 类别 | 类型 |
| ---- | ---- |
| **整数** | `INT4`, `INT8`, `INT16`, `INT32`, `INT64` |
| **无符号整数** | `UINT4`, `UINT8`, `UINT16`, `UINT32`, `UINT64` |
| **浮点数** | `FP4`, `FP8`, `FP16`, `FP32` |
| **Brain Float** | `BF16` |
| **Hisilicon** | `HF4`, `HF8` |
| **布尔值** | `BOOL` |

### 张量 (Tensor) 和 Tile 类型

```python
# Tensor (subscript notation)
a: pl.Tensor[[4, 8], pl.FP32]      # Fixed shape
b: pl.Tensor[[n, m], pl.INT64]     # Symbolic shape

# Tile (block in unified buffer)
t: pl.Tile[[16, 16], pl.FP16]
```

### 内存引用 (MemRef)

```python
# Create MemRef
addr_expr = pl.ConstInt(0x1000, pl.INT64, span)
memref = pl.MemRef(addr_expr, 1024, 0)

# Memory spaces: DDR, Vec, Mat, Left, Right, Acc
# Note: pl.Mem is a short alias for pl.MemorySpace

# Tensor with memref
tensor: pl.Tensor[[64, 128], pl.FP32, pl.MemRef(addr_expr, 8192, 0)]

# Tile 把内存空间保存在 tile 注解上，而不是 MemRef 内部
tile: pl.Tile[[16, 16], pl.FP16, pl.MemRef(addr_expr, 512, 0), pl.Mem.Left]
```

### Tile 视图 (TileView)

```python
# Create TileView
valid_shape = [pl.ConstInt(16, pl.INT64, span)] * 2
stride = [pl.ConstInt(1, pl.INT64, span), pl.ConstInt(16, pl.INT64, span)]
start_offset = pl.ConstInt(0, pl.INT64, span)
tile_view = pl.TileView(valid_shape=valid_shape, stride=stride, start_offset=start_offset)

# Tile with memref and tile_view
tile: pl.Tile[
    [16, 16], pl.FP16,
    pl.MemRef(addr_expr, 512, 0), pl.Mem.Left,
    pl.TileView(valid_shape=..., stride=..., start_offset=...)
]
```

**说明：**

- 省略 `pl.TileView(...)` **不**表示“没有 TileView 语义”。DSL 会根据 tile 的 shape，以及在存在时的
  tile memory space，推导一个隐式 TileView。
- 在这种隐式形式下，`valid_shape` 默认等于 tile shape；布局 / fractal 默认值也会根据
  shape / memory-space 组合推导。
- 显式写出的 `pl.TileView()`（或只是在重复这些隐式默认值的写法）与省略写法在语义上等价。
  parser / printer 的往返过程中，二者可能会被规范化为同一种打印形式。

## 表达式 (Expression)

### 变量和常量

```python
x              # Variable reference
tensor_a       # Tensor variable
42             # Integer literal
3.14           # Float literal
```

**闭包变量:** 在 DSL 作用域中未找到的名称会从外层 Python 作用域解析。支持的类型: `int`, `float`, `bool`, `list`, `tuple` 以及 IR 表达式。

```python
OFFSET = [0, 0]
TILE_SHAPE = [64, 64]

@pl.function
def func(t: pl.Tensor[[128, 128], pl.FP32], out: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
    a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, OFFSET, TILE_SHAPE)  # closure vars as positional args
    ...
```

### 二元操作

| Python 操作符 | PyPTO IR | 类别 |
| ------------- | -------- | ---- |
| `+` | Add | 算术 |
| `-` | Sub | 算术 |
| `*` | Mul | 算术 |
| `//` | FloorDiv | 算术 |
| `%` | FloorMod | 算术 |
| `/` | FloatDiv | 算术 |
| `**` | Pow | 算术 |
| `==`, `!=`, `<`, `<=`, `>`, `>=` | Eq, Ne, Lt, Le, Gt, Ge | 比较 |
| `and`, `or` | And, Or | 逻辑 |
| `^` | Xor | 逻辑 |
| `&` | BitAnd | 位运算 |
| `\|` | BitOr | 位运算 |
| `<<`, `>>` | BitShiftLeft, BitShiftRight | 位运算 |

**注意:** `and`/`or` 从 Python 的 `ast.BoolOp` 语法解析而来。链式表达式如 `a and b and c` 从左到右折叠为 `And(And(a, b), c)`。与 Python 不同，IR 的 `And`/`Or` 节点会求值两个操作数（无短路求值语义）。对应的 IR 工厂函数为 `ir.and_(lhs, rhs)` 和 `ir.or_(lhs, rhs)`。

### 一元操作和函数

```python
-x              # Neg
~x              # BitNot
not x           # Not
abs(x)          # Abs
min(a, b)       # Min
max(a, b)       # Max
```

### 函数/操作调用

```python
# Explicit namespace
pl.tensor.add(a, b)                  # Tensor addition
pl.tile.load(t, [0, 0], [64, 64])      # Tile load

# Unified dispatch (auto-selects tensor/tile based on input type)
pl.add(a, b)                          # Tensor or Tile — dispatched automatically
pl.mul(tile, 2.0)                     # Tile + scalar -> tile.muls
pl.exp(tile)                          # Tile -> tile.exp

# Promoted ops (single-module ops accessible at pl.*)
pl.load(t, [0, 0], [64, 64])            # Promoted from block
pl.create_tensor([64], dtype=pl.FP32)       # Promoted from tensor

# System operations (synchronization primitives)
pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
pl.system.bar_v()                        # Vector barrier
pl.system.bar_m()                        # Matrix barrier
pl.system.bar_all()                      # Global barrier

# Cross-core operations (TPUSH/TPOP protocol)
pl.tpush_to_aic(tile0, split=0, id=0)        # Vector → Cube push on pipe 0
pl.tpush_to_aic(tile1, split=0, id=1)        # Vector → Cube push on pipe 1
tile0 = pl.tpop_from_aiv(split=0, id=0)      # Cube pops from Vector pipe 0
tile1 = pl.tpop_from_aiv(split=0, id=1)      # Cube pops from Vector pipe 1
pl.tfree_to_aiv(tile0, id=0)                 # Release slot to Vector pipe 0
pl.tfree_to_aiv(tile1, id=1)                 # Release slot to Vector pipe 1

# Cross-core pipe initialization and buffer management
buf = pl.reserve_buffer(name="slot_buf", size=4096, base=pl.AUTO)
peer = pl.import_peer_buffer(name="slot_buf", peer_func="other_func")
pl.aic_initialize_pipe(pl.const(0, pl.INT32), buf, dir_mask=2, slot_size=512, id=0)
pl.aiv_initialize_pipe(pl.const(0, pl.INT32), peer, dir_mask=2, slot_size=512, id=0)
```

## 语句 (Statement)

### 赋值

```python
x: pl.INT64 = expr
y: pl.Tensor[[4], pl.FP32] = tensor_op(a)
```

### If 语句 (SSA 风格)

```python
# If with both branches
if condition:
    y1 = pl.yield_(value1)
else:
    y1 = pl.yield_(value2)

# Multiple return values (no inline type annotations)
if condition:
    y1, y2 = pl.yield_(value1, value2)
else:
    y1, y2 = pl.yield_(value3, value4)
```

**要点:**

- `pl.yield_()` 赋值给 SSA phi 节点
- yield 中定义的变量在 if 之后可访问
- 两个分支必须 yield 相同的变量
- 元组解包时不能使用内联类型标注

### For 循环 (带 iter_args 的 SSA 风格)

```python
# 简单循环 (1-3 个位置参数，类似 Python 的 range())
for i in pl.range(stop):                    # start=0, step=1
for i in pl.range(start, stop):             # step=1
for i in pl.range(start, stop, step):       # 完整形式

# 带 iter_args 的循环 (循环携带值)
sum_init: pl.INT64 = 0
for i, (sum,) in pl.range(n, init_values=(sum_init,)):
    sum = pl.yield_(sum + i)
sum_final = sum

# 并行 for 循环 (同样支持 1-3 个参数)
for i in pl.parallel(stop):
for i in pl.parallel(start, stop, step):
    body_statements
```

**要点:** 循环携带值使用 `pl.range()` 或 `pl.parallel()` 的 `init_values`, 元组解包 `(sum,)` 声明 iter_args, `pl.yield_()` 为下一次迭代更新值, 循环结束后 iter_args 包含最终值。`pl.parallel()` 生成 `ForKind.Parallel` 循环, `pl.range()` 生成 `ForKind.Sequential` (默认)。

#### 分块循环 (Chunked Loops)

```python
# 将循环拆分为每块 C 次迭代的嵌套循环
for i in pl.range(10, chunk=5):
    body_statements

for i in pl.parallel(8, chunk=4):
    body_statements

for i in pl.unroll(12, chunk=4):
    body_statements
```

**要点:** `chunk=C` 将循环拆分为外层顺序循环和 `C` 次迭代的内层循环。内层循环保留原始类型 (Sequential/Parallel/Unroll)。`chunk=` 循环支持与 `init_values` 一起使用（iter_args 会贯穿生成的外层/内层/余数循环）。`chunk=` 循环只能出现在 `with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):` 内；在该作用域外，parser 会直接报错。参见 [SplitChunkedLoops Pass](../passes/07-split_chunked_loops.md)。

### While 循环 (带 iter_args 的 SSA 风格)

```python
# 自然 while：条件作为 while 头部表达式
i: pl.Scalar[pl.INT64] = 0
while i < n:
    i = i + 1

# 带 init_values 的 SSA 形式：头部元组 = iter_args，第一条语句是 pl.cond()。
# yield-LHS 名字成为循环外的绑定名（与 pl.range 一致）。
x_init: pl.Scalar[pl.INT64] = 0
for (x,) in pl.while_(init_values=(x_init,)):
    pl.cond(x < n)
    x_next = pl.yield_(x + 1)
# 此处 `x_next` 已由 yield-LHS 绑定；`x` 仅在循环 body 内可见。

# Pre-SSA：body 中完全没有 pl.yield_，由 ConvertToSSA 后续补出。
for (x,) in pl.while_(init_values=(x_init,)):
    pl.cond(x < n)
    x = x + 1

# ❌ init_values 非空时不允许裸 pl.yield_(...)，parser 直接报错：
#    for (x,) in pl.while_(init_values=(x_init,)):
#        pl.cond(x < n)
#        pl.yield_(x + 1)             # ParserSyntaxError: requires assignment-form pl.yield_
```

**要点:** `pl.while_(init_values=(...,))` 复用 `for ... in` 头部，用于 SSA 风格循环；body 的第一条语句必须是 `pl.cond(<bool>)`。循环外的绑定名来自 **yield-LHS**（上面的 `x_next`），而不是头部元组——头部元组中的名字只在循环 body 内可见。这一约定与 `pl.range` **保持一致**：当 `init_values` 非空且 body 中确实出现 `pl.yield_(...)` 调用时，必须使用 assignment 形式。Pre-SSA 形式的循环（body 中完全没有 yield，如最后一种写法）仍然合法。

### 作用域上下文管理器 (Scope Context Managers)

| 形式 | Scope 类型 | 说明 |
| ---- | ---------- | ---- |
| `pl.at(level=pl.Level.CORE_GROUP)` | `InCore` | CORE_GROUP 级固定边界 outline |
| `pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(MODE)])` | `InCore` | InCore + 跨核 split 提示 |
| `pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk])` | `AutoInCore` | 编译器驱动的 chunked 循环 split |
| `pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk, pl.split(MODE)])` | `AutoInCore` | AutoInCore + split 提示（条目独立） |
| `pl.at(level=pl.Level.HOST)`（或任意非 `CORE_GROUP` 级别） | `Hierarchy` | 分布式层级作用域 |
| `pl.cluster()` | `Cluster` | AIC+AIV 协同调度组 |
| `pl.manual_scope()` | `Runtime(manual=true)` | 由用户管理任务排序的 orchestrator 区域——见[手工依赖原语](#手工依赖原语) |
| `pl.incore()` *(已弃用)* | `InCore` | 请改用 `pl.at(level=pl.Level.CORE_GROUP)` |
| `pl.auto_incore(split=...)` *(已弃用)* | `AutoInCore` | 请改用 `pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk, pl.split(...)])` |
| `pl.at(..., optimization=pl.chunked_loop_optimizer[(split=...)])` *(已弃用)* | `AutoInCore` | 请改用 `pl.at(..., optimizations=[pl.auto_chunk, pl.split(...)])` |
| `pl.at(..., split=...)` *(已弃用)* | `InCore` | 请改用 `pl.at(..., optimizations=[pl.split(...)])` |

示例参见 [语言指南](../../user/01-language_guide.md#incore-作用域)。

### 手工依赖原语

默认情况下 runtime 通过缓冲区读写重叠（`OverlapMap`）自动推导任务间依赖。
两个互补的原语让用户可以选择性地退出自动跟踪并显式管理排序：

| 表层语法 | 粒度 | 作用 |
| -------- | ---- | ---- |
| `pl.no_dep(arg)` | per-call 参数 | kernel 调用点上，被包装的参数其 `ArgDirection` 变为 `NoDep`。该次提交对该槽位不进入自动跟踪。**仅 auto scope 生效**——在 `pl.manual_scope` 内没有语义。 |
| `with pl.manual_scope():` | per-region | 下沉为 `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`。区域内 runtime 不做自动跟踪；用户必须通过 `deps=[...]` **显式声明每一条**需要的排序边。 |
| `kernel(..., deps=[var, ...])` | per-call（仅 manual_scope 内） | 声明显式 task-id 边。每个条目必须是：同一 `manual_scope` 内由先前 `self.kernel(...)` 产生的 tensor `Var`，或在循环中承载该 producer 的 iter_arg / return_var，或作为初始种子传入的函数参数。 |

Manual scope 是**只看声明**的：即使 `stage2` 读取了 `stage1` 写过的 buffer，
若用户没写 `deps=[scratch]`，pass 也不会替你补上。之前的数据流自动推导
已被移除，因为当不同 kernel 在同一块 buffer 上原地复用时它会产生大量伪边。

```python
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64], pl.FP32],
         scratch: pl.Out[pl.Tensor[[64], pl.FP32]],
         out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
    with pl.manual_scope():
        scratch = self.stage1(x, scratch)
        out     = self.stage2(scratch, out, deps=[scratch])   # 必需：stage2 读 scratch
    return out
```

`with pl.manual_scope():` 内由
[DeriveManualScopeDeps](../passes/33-derive_manual_scope_deps.md)
pass（底层实现为 `LowerManualDepsToTaskId` lowering，对外保留旧 pass 名）
把每个 `deps=[var, ...]` 解析为 producer 的 TaskId 配套、写入 call 的
`manual_dep_edges`，并为每个外层 `pl.range` / `pl.parallel` 同步追加
TaskId iter_arg / return_var / yield 项。不再对每 call 边数设上限——
codegen 按实际依赖数生成 `ArgWithDeps<N>`。

`pl.no_dep(arg)` 是 auto scope 原语；在 `pl.manual_scope` 内不起作用——
pass 不再以 per-arg direction 做依赖解析。

#### Manual scope 下的 `pl.parallel`：array-carry fence

当 manual-dep 边穿过一个 `pl.parallel` 循环（即循环 iter_arg 承载被依赖的
TaskId）时，orchestration codegen 把对应的 TaskId iter_arg 视作**大小等于
parallel 循环 trip count 的数组**。每次 parallel 迭代写入自己的槽位；
下游消费者对**每个**槽位都生成 `add_dep`（不是只对"最后被发射"的那个
task）。这就保证用户声明的 fence 语义即便在迭代乱序完成时也是正确的。

走 array-carry 路径的前提：

- `pl.parallel` 的 trip count 必须是 Python 字面量（编译期常量）。
  trip count 是动态值的情况下 codegen 会拒绝，提示 "statically-known
  trip count"。

```python
with pl.manual_scope():
    for phase in pl.range(N_PHASES):
        for branch in pl.parallel(N_BRANCHES):           # 编译期常量
            row = (phase * N_BRANCHES + branch) * TILE_M
            out = self.kernel_stripe(data, row, 1.0, out, deps=[out])
```

phase `N+1` 中的每个 task 都会等待 phase `N` 的全部 `N_BRANCHES` 个
task，而非只等最后那个。

### Yield 语句

```python
yield            # No values
yield x          # Single value
yield x, y       # Multiple values
```

### Break 和 Continue

```python
break              # 退出最内层循环
continue           # 跳到下一次迭代
```

**限制:** 仅当**最内层**封闭循环为顺序循环 (`pl.range`) 或 `while` 时有效。当最内层循环为 `pl.parallel()` 或 `pl.unroll()` 时不支持。在外层 `pl.parallel` 循环内嵌套的内层 `pl.range` 循环中使用 `break` 是合法的。**注意:** 代码生成后端对 `break`/`continue` 的支持跟踪在 [#448](https://github.com/hw-native-sys/pypto/issues/448) 中。

### 编译期调试 (Compile-Time Debugging)

`pl.static_print()` 和 `pl.static_assert()` 是仅在解析期执行的构造，用于在解析过程中检查 IR 状态和断言条件。它们**不生成任何 IR**。

```python
@pl.function
def func(x: pl.Tensor[[128, 64], pl.FP16]) -> pl.Tensor[[128, 64], pl.FP16]:
    pl.static_print("input:", x)          # → static_print [file:line]: input: x: pl.Tensor[[128, 64], pl.FP16]
    pl.static_print(f"input: {x}")        # → static_print [file:line]: input: x: pl.Tensor[[128, 64], pl.FP16]
    pl.static_assert(True)                # 静默通过
    pl.static_assert(N > 32, "N too small")  # 在解析期检查闭包变量 N
    return x
```

| 函数 | 用途 | 失败时 |
| ---- | ---- | ------ |
| `pl.static_print(*args)` | 将变量类型/值打印到 stdout | 需要 ≥1 个参数 |
| `pl.static_assert(cond, msg="")` | 断言编译期条件 | 抛出 `ParserError` |

**要点：**

- 两者均为语句级构造（不能用在表达式中）
- `static_print` 接受变量、常量、字符串标签（原样打印）和 f-string 的简单 `{expr}` 占位符（格式化为 IR）。不支持转换标志（`!r`、`!s`、`!a`）和格式说明符（`:...`）。
- `static_assert` 支持闭包变量表达式（如 `N > 32`）和 IR 常量
- `static_assert` 的消息参数必须是字符串字面量
- 即使后续解析失败，输出仍会显示——适用于调试解析错误

### 语句序列

```python
stmt1            # Natural Python sequencing
stmt2
stmt3
```

## 函数

```python
# Single return type
def function_name(param1: pl.INT64, param2: pl.FP32) -> pl.INT64:
    x: pl.INT64 = param1 + 1
    return x

# Multiple return types
def function_name(x: pl.INT64) -> tuple[pl.INT64, pl.INT64]:
    y: pl.INT64 = x + 1
    z: pl.INT64 = x * 2
    return y, z

# No return types
def function_name(x: pl.INT64):
    y: pl.INT64 = x + 1

# With function type
@pl.function(type=pl.FunctionType.Orchestration)
def orchestrator(n: pl.INT64) -> pl.INT64:
    return n + 1

@pl.function(type=pl.FunctionType.InCore)
def aicore_kernel(x: pl.INT64) -> pl.INT64:
    return x * 2
```

### 函数类型

| 类型 | 用途 | 描述 |
| ---- | ---- | ---- |
| `pl.FunctionType.Opaque` | 默认 | 未指定的函数类型 |
| `pl.FunctionType.Orchestration` | Host/AICPU | 控制流和依赖分析 |
| `pl.FunctionType.InCore` | AICore | AICore 子图执行（未特化） |
| `pl.FunctionType.AIC` | Cube 核心 | Cube 核心内核（特化的 InCore） |
| `pl.FunctionType.AIV` | Vector 核心 | Vector 核心内核（特化的 InCore） |
| `pl.FunctionType.Group` | 多核 | AIC + AIV 内核的协调调度组 |

未指定类型时, 函数默认为 `Opaque`。

### 参数方向

参数可以使用包装类型指定 `In` (默认)、`Out` 或 `InOut` 方向:

```python
@pl.function(type=pl.FunctionType.InCore)
def kernel(
    qi: pl.Tensor[[16, 128], pl.BF16],                   # In (default)
    output: pl.InOut[pl.Tensor[[16, 128], pl.FP32]],      # InOut
    result: pl.Out[pl.Tensor[[16, 128], pl.FP32]],        # Out
    scale: pl.Scalar[pl.FP32],                             # In (default)
) -> pl.Tensor[[16, 128], pl.FP32]:
    ...
```

| 方向 | 包装类型 | 描述 |
| ---- | -------- | ---- |
| `In` | 无 (默认) | 只读输入参数 |
| `Out` | `pl.Out[type]` | 只写输出参数 |
| `InOut` | `pl.InOut[type]` | 读写输入/输出参数 |

**约束:** `Scalar` 参数不能使用 `InOut` 方向 (会抛出 `ParserTypeError`)。

## 完整示例

### 张量操作 (带 iter_args 的循环)

```python
# pypto.program: my_program
import pypto.language as pl

def loop_sum(n: pl.INT64) -> pl.INT64:
    sum_init: pl.INT64 = 0
    for i, (sum,) in pl.range(n, init_values=(sum_init,)):
        sum = pl.yield_(sum + i)
    return sum
```

### Tile 操作 (基于 Tile 的计算)

```python
import pypto.language as pl

@pl.program
class BlockExample:
    @pl.function
    def tile_add(
        self,
        input_a: pl.Tensor[[64, 64], pl.FP32],
        input_b: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
        tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(input_b, [0, 0], [64, 64])
        tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)
        result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], output)
        return result
```

## SSA 风格控制流

`pl.yield_()` 为 if/for 语句创建 SSA phi 节点:

```python
# If: phi node at merge point
if condition:
    y1 = pl.yield_(x + 1)
else:
    y1 = pl.yield_(x + 2)
# y1 = phi(x + 1, x + 2)

# For: loop-carried values via iter_args
sum_init: pl.INT64 = 0
for i, (sum,) in pl.range(10, init_values=(sum_init,)):
    sum = pl.yield_(sum + i)
sum_final: pl.INT64 = sum  # captures final value
```

## 打印 IR 节点

对任意 IR 节点调用 `as_python()` 获取其 Python 表示：

```python
print(stmt.as_python())          # "x: pl.Scalar[pl.INT64] = a + b"（默认 "pl" 前缀）
print(stmt.as_python("ir"))      # "x: ir.Scalar[ir.INT64] = a + b"（自定义前缀）
```

### 简洁模式 (Concise Mode)

传入 `concise=True` 可省略中间变量的类型标注。函数签名类型（参数和返回值）始终保留：

```python
print(func.as_python())                  # 详细模式（默认）：每个赋值都包含类型
print(func.as_python(concise=True))      # 简洁模式：省略中间类型标注
```

详细输出：

```python
def main(self, x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP16]:
    y: pl.Tensor[[64, 128], pl.FP32] = pl.some_op(x)
    result: pl.Tensor[[64, 128], pl.FP16] = pl.cast(y, pl.FP16)
    return result
```

简洁输出：

```python
def main(self, x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP16]:
    y = pl.some_op(x)
    result = pl.cast(y, pl.FP16)
    return result
```

自由函数 `ir.python_print(node)` 同样可用，支持相同的参数。

## 参考资料

- [IR 概述](../ir/00-overview.md) - 核心 IR 结构
- [IR 解析器 (Parser)](../ir/07-parser.md) - 将 Python 语法解析回 IR
- [操作符注册](../ir/05-operators.md) - 操作系统和类型推断
