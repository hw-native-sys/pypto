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

### 张量布局 (Layout) 和视图 (TensorView)

下标第三个元素是布局 (layout) 或 `pl.TensorView`，两者均可内联书写或绑定到变量——
绑定一次即可在多个参数间共享同一视图。布局是"无 stride 视图"的简写，因此各种写法
最终都会解析为一个 `TensorView`。`pl.DistributedTensor` 的下标位置与写法相同。

```python
STRIDED = pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)

x: pl.Tensor[[32, 64], pl.FP32, pl.NZ]      # 布局，内联
y: pl.Tensor[[32, 64], pl.FP32, STRIDED]    # 视图，通过变量
```

在 `@pl.jit` 下只支持 **布局 (layout)** 这一种写法。特化会依据记录的
shape/dtype/layout 重新生成注解，而 `pl.TensorView` 在该记录中没有对应字段——
传入时会抛出 `TypeError` 并指明参数名，而不是丢弃 stride。这类 kernel 请改用
`@pl.function`，它直接解析注解本身。

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

### 声明式分配（单参数 MemRef）

单参数形式 `pl.MemRef("name")` 声明一块属于你自己的分配，把它从编译器的机会主义复用中收回。
引用它的 tile 共享这块分配，其他 tile 绝不会被塞进去。当 packer 合并了你希望保持独立的 tile
时使用它——共用存储会引入一条 WAR 依赖，使二者串行。

它与三参数形式是同一个 IR 节点；参数个数区分"描述一块已有分配"还是"声明一块新的"。声明时只
给名字：大小取自绑定到它的最大 tile，地址由分配器决定。

先声明一次，再用变量引用。不带名字的声明会取所绑定变量的名字，这样名字只写一遍：

```python
ping = pl.MemRef()
pong = pl.MemRef()

# 两个 tile 显式共用一块分配；第三个保持独占。
t0: pl.Tile[[64, 64], pl.FP32, ping, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
t1: pl.Tile[[64, 64], pl.FP32, pong, pl.Mem.Vec] = pl.exp(t0)
t2: pl.Tile[[64, 64], pl.FP32, ping, pl.Mem.Vec] = pl.exp(t1)
```

推荐这种写法：引用拼错会直接得到 Python 的 `NameError`，而内联的 `pl.MemRef("pign")` 形式
里字符串拼错只会静默地多声明一块分配。内联形式仍然有效——IR 打印器输出的就是它，这样 dump
出来的程序不依赖外层 Python 作用域即可重新解析；`pl.MemRef("other")` 也可用于显式命名，适用
于变量名不是你想写进 IR 的那个名字时。

既然名字由变量提供，变量与分配就必须一一对应。用两个名字引用同一个声明（`alias = ping`）、
以及两个声明抢占同一个名字，都会被**拒绝**——两者都会静默地合并或拆分分配。

一个 MemRef 是否为声明式分配，由 IR 节点上的显式字段（`MemRef.is_pinned_`）记录，既不靠大小
推断，也不靠"当前跑到哪个 pass"。`InitMemRef` 会消费掉这个声明：此后分配点带上 `pinned=True`，
MemRef 变回普通 MemRef，所以重新解析一份分配后的 dump 不会把编译器分配误当成声明式分配。

#### 槽位

传 `slots=N` 可以把一块分配切成 N 个等大槽位,再用下标选择其中一个。槽位连续且大小一致,
因此在它们之间轮转形成的 ping-pong 是 packer 无法合并的:

```python
l0c = pl.MemRef(slots=2)

ping: pl.Tile[[M, N], pl.FP32, l0c[0], pl.Mem.Acc] = pl.tile.matmul(q, b0)
pong: pl.Tile[[M, N], pl.FP32, l0c[1], pl.Mem.Acc] = pl.tile.matmul(q, b1)
```

**下标可以是运行期的值**,所以轮转不需要展开循环:

```python
for i, (acc,) in pl.range(N, init_values=(out,)):
    a: pl.Tile[[M, N], pl.FP32, l0c[i % 2], pl.Mem.Acc] = pl.tile.matmul(q_l0, b_l0)
```

在 `@pl.jit` 下请用内联命名形式 `pl.MemRef("l0c", slots=2)[i % 2]`,而不是把声明绑到一个
Python 变量上。`@pl.jit` 会在一个新的模块命名空间里重新解析生成的源码,变量里持有的声明在
那里不可见;命名形式是自包含的(也正是 IR printer 输出的形式),两种场景都可用。

`InitMemRef` 按绑定到**任意**槽位的最大 tile 确定单个槽位的大小——槽位是等大的,按槽位分别
定尺寸会让步长不一致——并把下标折算成字节偏移 `index * slot_size`。常量下标在这里就折成常量,
走原有的常量地址路径;运行期下标则留成表达式,在运行时成为 tile 的地址。

共存检查**按槽位**进行,而不是按分配:不同槽位上的两个 tile 本来就应该同时存活,只有落在
**同一**槽位的两个 tile 才会互相破坏。下标是运行期表达式时,没有静态的槽位归属可比,该检查
被跳过——轮转的正确性由作者负责——而与其他分配之间的隔离依然成立。

##### 在 ptoas 内存规划器下

`slots=N` 是 `memory_planner=PTOAS` 唯一接受的声明形式。ptoas 有对应的概念——一块
`pto.alloc_multi_tile` 区域,其 N 个槽位必须落在互不相交的物理段上——因此 codegen 把整个声明
原样交给它:一块区域,每个使用点用 `pto.multi_tile_get` 选中自己的槽位。传给 ptoas 的是槽位
**下标**(而不是它折算出的字节偏移),这正是 ptoas 能证明哪些访问共用槽位、进而为轮转分配
按槽位的 event id 的前提——第 *i* 轮的 load 由此与第 *i-1* 轮的计算重叠。

单槽位的 `pl.MemRef()` 没有对应形式,在该规划器下仍被拒绝:ptoas 可以把你特意分开的缓冲重新
打包到一起。多槽位声明中 ptoas 无法描述的形态同样被拒绝——各槽位上 tile 形状不一致、内存空间
不属于 Vec / Mat / Acc、valid shape 是运行期值,或某个槽位作为 phi 被带出 `if` / 循环。这些都
会报出指明具体形态的错误,而不是静默回退,因为回退就等于把你声明的隔离抹掉。

默认的 PyPTO 规划器不受影响:它烘焙地址,而在 `--pto-level=level3` 下 ptoas 不会折叠逐槽位的
地址展开,区域形式反而会丢掉它赖以存在的槽位分析
（[PTOAS#1106](https://github.com/hw-native-sys/PTOAS/issues/1106)）。

声明的名字自成命名空间——不会解析到恰好同名的 Python 变量。内存空间**必须**写（`TileType`
始终要求 MemRef 与空间成对出现），且绑定到同一块分配的 tile 必须一致。未加注解的 tile 保持
默认的自动复用。

声明式分配不会随流水级复制，因此**当该循环走复制路径下降时**，在 `pl.pipeline(stage=2)` 体内
声明会被**拒绝**：复制出的各级会让同一个 tile 在同一块分配上与自身同时存活。声明槽位与"交给
编译器做多缓冲"是二选一，不能叠加。要自己管理某一层，就用 `pl.range` 驱动它并为每个槽位声明
一块分配；希望编译器管理的层次则不加注解。

在 `memory_planner=PTOAS` 下，编译器用的是**同一套**机制而非另一套：
[`LowerPipelineToSlots`](../passes/27-lower_pipeline_to_slots.md) 会为合格 `pl.pipeline` 循环体中
**顶层**的每个 `tile.load` 合成与上文完全一致的声明——`slots=F`、以 `iv % F` 索引——于是单份循环体
轮转各个槽位而不被复制。你自己绑定过的 tile 不受影响；该 pass 未接手的循环仍走复制路径
（上述拒绝规则对其依然成立）。

```python
l0b_ping, l0b_pong = pl.MemRef(), pl.MemRef()

# 外层交给编译器，内层由作者自己做 ping-pong。
for stack, (out_outer,) in pl.pipeline(STACKS, stage=2, init_values=(out,)):
    b_l1: pl.Tile[[K, N], pl.BF16, pl.Mem.Mat] = pl.load(b, [stack * K, 0], [K, N])
    for col, (out_inner,) in pl.range(0, N, 2 * STEP, init_values=[out_outer]):
        ping: pl.Tile[[K, STEP], pl.BF16, l0b_ping, pl.Mem.Right] = ...
        pong: pl.Tile[[K, STEP], pl.BF16, l0b_pong, pl.Mem.Right] = ...
```

参见 [InitMemRef](../passes/31-init_memref.md#声明式分配) 与
[MemoryReuse](../passes/33-memory_reuse.md#声明式分配)。

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
- `compact=pl.CompactMode.normal` 表示部分 boxed tile 的 PTO 紧凑传输格式。PyPTO 会为
  L0A/L0B 中部分有效的 `tile.extract` 结果自动推导该值，kernel 通常不应直接设置它。

## 表达式 (Expression)

### 变量和常量

```python
x                       # Variable reference
tensor_a                # Tensor variable
42                      # Integer literal — INDEX-typed
3.14                    # Float literal
pl.const(42, pl.INT64)  # Typed integer literal (any non-INDEX dtype)
```

裸整数字面量始终为 `INDEX` 类型。若需携带其他整数 dtype（如 `INT64`），
请使用 `pl.const(value, dtype)`——打印器也以此形式渲染此类常量，
从而保证打印出的 IR 能通过解析器正确往返。
在复合 shape 维度和纯常量算术中（如
`pl.const(32, pl.INDEX) + pl.const(32, pl.INDEX)`），打印器对 `INDEX`
也会输出带类型的叶子，使解析器逐字重建表达式树而不做常量折叠；
化简始终由 Simplify pass 负责。

**闭包变量:** 在 DSL 作用域中未找到的名称会从外层 Python 作用域解析。支持的类型: `int`, `float`, `bool`, `list`, `tuple` 以及 IR 表达式。

```python
OFFSET = [0, 0]
TILE_SHAPE = [64, 64]

@pl.function
def func(t: pl.Tensor[[128, 128], pl.FP32], out: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
    a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, OFFSET, TILE_SHAPE)  # closure vars as positional args
    ...
```

**枚举类算子实参 (Enum op arguments):** 算子包装函数中有一类形参接收 Python
枚举而非 IR 表达式 —— `DataType`、`MemorySpace`、`TensorLayout`、`TileLayout`、
`PadValue`、`ArgDirection`。无论按位置传入还是按关键字传入，也无论写成字面属性
还是闭包名称，它们的解析结果完全一致，因此下面两行构建出相同的调用:

```python
p = pl.fillpad(t, pl.PadValue.min)              # positional
p = pl.fillpad(t, pad_value=pl.PadValue.min)    # keyword
```

算子在此类形参上接受的数值糖也同样如此: `pl.fillpad` 在两种位置上都接受 `0`、
`0.0`、`math.inf` 和 `-math.inf`。

### 下标索引 (Subscript Indexing)

`Tensor` 和 `Tile` 的下标采用 numpy/torch 风格的语义:

- **标量** 索引会移除该维度; **切片 (slice)** 会保留该维度。
- 索引个数少于 `rank` 时, 末尾自动补 `:` —— 4D 张量上的 `C[i]` 等价于 `C[i, :, :, :]`。
- 链式索引可组合 —— `C[i][j]` 是两次降秩视图。
- **全标量且满秩** 的索引读取一个标量 (2D 张量上的 `A[i, j]` → `tensor.read` / `tile.read`)。

```python
C[i, j, k, l]   # all scalar, full rank   -> scalar
C[i, j]         # partial, all scalar      -> 64×64 view (dims 0,1 dropped)
C[i]            # partial                  -> 64×64×64 view (dim 0 dropped)
C[i][j]         # chained                  -> works (C[i] is 3D, then [j])
C[i:i+8, j]     # mixed slice + scalar     -> 8×64×64 view (dim 1 dropped)
C[i:i+8, :, :, :]  # all slices            -> 8×64×64×64 view
```

v1 限制: 不支持切片 `step`、tile 切片的下界必须可静态折叠、不支持 ellipsis / `None` / 负索引 / 高级索引。**Tile 物理上是 2D 的**, 所以自然结果 `< 2D` 的 tile 会被自动提升到 2D (`[N]` → `[1, N]`) 并发出非致命警告 —— 若需要不同的布局, 请显式使用 `pl.tile.reshape`。

实现机制: 非平凡的下标会下降为 `tensor.slice` / `tile.slice`, 其 `shape`/`offset` 保持满秩, 并附带一个 `drop_dims` 列表记录被标量索引的轴 (详见 IR 算子文档)。赋值左侧 (LHS) 遵循相同规则 —— `C[i, j] = rhs` 会在 `tensor.assemble` 之前把 `rhs` reshape 回满秩窗口 (尚不支持链式写入 `C[i][j] = rhs`)。

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
# 可选：显式指定 GM 环形缓冲区槽数量（默认单向 8 / 双向 4），
# 以及（仅 a2/a3）本地槽数量 local_slot_num（必须 <= slot_num）。
# 缓冲区大小需自行设置：a3 -> slot_size * local_slot_num，a5 -> slot_size * slot_num。
pl.aic_initialize_pipe(pl.const(0, pl.INT32), buf, dir_mask=2, slot_size=512, slot_num=16, local_slot_num=4)
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
| `pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.cross_core_slot(slot_num=N)])` | `InCore` | InCore + 跨核 pipe 槽位数 |
| `pl.at(level=pl.Level.HOST)`（或任意非 `CORE_GROUP` 级别） | `Hierarchy` | 分布式层级作用域 |
| `pl.cluster()` | `Cluster` | AIC+AIV 协同调度组 |
| `with pl.spmd(N)` / `for i in pl.spmd(N)` | `Spmd`（for-form 内嵌 `InCore`） | SPMD 多 block 派发——见 [pl.spmd](#plspmd-多-block-派发) |
| `pl.spmd(N, optimizations=[pl.split(MODE)])` | `Spmd(InCore(split=MODE))` | split 提示作用于内层 InCore（两种形式均适用） |
| `pl.spmd(N, optimizations=[pl.cross_core_slot(slot_num=N)])` | `Spmd(InCore(slot_num=N))` | 槽位数作用于内层 InCore（两种形式均适用），可与 `pl.split(MODE)` 组合 |
| `pl.scope(mode=pl.ScopeMode.MANUAL)` / `pl.manual_scope()` | `Runtime(manual=true)` | orchestrator 的 MANUAL scope——由用户管理任务排序。两种 `auto_scope` 模式下都可用（它是依赖语义选择）。见[手工依赖原语](#手工依赖原语) |
| `pl.scope()` | `Runtime(manual=false)` | orchestrator 的 AUTO scope（`PTO2_SCOPE()`）。手写它需要 `@pl.function(auto_scope=False)`（默认 `auto_scope=True` 下由编译器决定 AUTO 放置）。见 [MaterializeRuntimeScopes](../passes/44-materialize_runtime_scopes.md) |

#### `pl.spmd` 多 block 派发

`pl.spmd(N)` 把一个 kernel 派发到 `N` 个 block。形式：

- `with pl.spmd(N): ...` —— body **既可以**是调用已声明 InCore kernel 的*派发型* body（`SpmdScopeStmt(body=<stmts>)`，无内层 InCore 包裹），**也可以**是一段*内联*块，自动外包成一段隐式 InCore 区域（与 for-form 相同，只是不自动绑定索引）。区分依据是语义而非语句数量：body 若读取 `pl.tile.get_block_idx()`，即为内联 body 并被包裹；否则即为派发型 body，无论包含多少条语句都不加包裹。若 body 既不读取索引、也不派发 `self.<kernel>(...)` 调用，则会被拒绝。不捕获 producer TaskId。
  - 当 body 唯一的语句是显式的 `with pl.at(<CORE_GROUP level>, ...):` 时，该嵌套 scope 本身*就是* InCore 载体：它会被当作普通嵌套 scope 解析，而不会被二次包裹（`level` 可用位置或关键字形式，也可带 `as tid` / `name_hint=`）。printer 正是以这种形式输出 `Spmd(InCore(...))`，因此这也是该 IR 能够 round-trip 的原因。当 body 已提供载体时，`optimizations=` 必须写在该 `pl.at(...)` 上；写在 `pl.spmd(...)` 行会被拒绝，无论载体自身是否也带有该项。
  - 派发型 body 只能启动**一个** kernel。它经由 `FindFirstInnerCall` 下降，而后者在第一个调用处即停止，因此第二个派发不会被启动而是被静默丢弃；parser 会直接拒绝这种写法。提升出的临时变量与 tuple 投影不算派发，不计入数量。
- `for i in pl.spmd(N): ...` —— 循环变量绑定到每个 block 的索引（`pl.tile.get_block_idx()`）；body 自动外包成一段隐式 InCore 区域。
- `with pl.spmd(N, deps=[...]) as tid: ...` —— **捕获形式**：与 `with pl.at(...) as tid:` 对称。body 形态与上面的普通形式相同，并额外在 `tid` 中捕获该分发的 grid 级 producer `pl.Scalar[pl.TASK_ID]`（可用作 `deps=` 边、存入 `pl.array.create(N, pl.TASK_ID)`、或跨入 `pl.manual_scope`）。TaskId 捕获与内联 body 正交——这是该形式相比普通形式唯一多出来的能力。lower 成一个 `ir.Submit`，其尾部 tuple 元素即 grid TaskId；`core_num` / `sync_start` 记录在该 `Submit` 自身的字段上（launch spec 属于启动点，而非外包出的被调函数）。参见下文“手动依赖原语”小节。
- `out, tid = pl.spmd_submit(kernel, *args, core_num=N)` —— **submit 形式**：将 kernel 在 `N` 个 block 上分发，同时捕获该分发的 producer `pl.Scalar[pl.TASK_ID]`（针对已声明 kernel 的 `pl.submit` 版本）。参见下文“手动依赖原语”小节。

以上三种形式也都接受 `allow_early_resolve=True`（布尔字面量；与 `pl.submit` / `pl.at` 相同的 early-dispatch 选项）。即使不写 `as tid` 也会强制走 `ir.Submit` 形态，并 lower 为 `Arg::set_allow_early_resolve(true)`。在嵌套于 `pl.cluster()` 内的 `pl.spmd` 上会被拒绝（此类 scope 会被 unwrap 进 Group 函数、永远不会产生 Submit，提示会丢失）。

可选 `optimizations=[...]`。各条目彼此正交，可在同一列表中组合
（例如 `[pl.split(MODE), pl.cross_core_slot(slot_num=4)]`）：

| 条目 | 适用形式 | 作用 |
| ---- | -------- | ---- |
| `pl.split(MODE)` | 两种均适用 | 给内层 InCore 设置 `split_` 字段（跨核数据搬运提示，由 `ExpandMixedKernel` / `MemoryReuse` 消费）。with-form 会在原 call 外多包一层 `InCoreScopeStmt` 来承载该字段。 |
| `pl.cross_core_slot(slot_num=N)` | 两种均适用 | 给内层 InCore 设置 `slot_num` 属性——自动跨核 pipe 的槽位数（环深），由 `ExpandMixedKernel` 消费。它只决定数据通道大小，**不**划分计算，因此可与 `pl.split_aiv` 区域共存（而 `pl.split(...)` 不能）。省略时沿用 PTOAS 默认值（单向 8，双向每方向 4）。 |

> `pl.split(MODE, slot_num=N)` 是该槽位数的已废弃别名，会发出警告——参见
> [ExpandMixedKernel](../passes/21-expand_mixed_kernel.md#覆盖槽位数slot_num)。

示例参见 [作用域与放置](../../user/language/04-scopes.md)。

### 手工依赖原语

默认情况下 runtime 通过缓冲区读写重叠（`OverlapMap`）自动推导任务间依赖。
DSL 暴露**两套正交的机制**，用户可任意组合：

> **两套机制相互独立。** 把某个 buffer / 区域 / arg 从自动跟踪中"摘出来"
> 并**不要求**你同时声明显式边；声明显式边也**不要求**你同时关掉自动
> 跟踪。最终 task 的 fanin 是 **`自动跟踪 deps ∪ 显式 deps`**——它们
> 是相加而非互相替代。

#### 机制 A——退出自动依赖跟踪（3 种粒度）

三种粒度彼此独立。按需选择最小的单位，必要时叠加。

| 表层语法 | 粒度 | 作用 |
| -------- | ---- | ---- |
| `with pl.manual_scope():` | per-region | 下沉为 `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`。区域内 runtime 不做自动跟踪；用户需要的排序边必须通过机制 B 显式声明。 |
| `pl.create_tensor([...], dtype=..., manual_dep=True)` | per-tensor 生命周期 | 任何读 / 写该 tensor 的 task 都**整生命周期**跳过 `OverlapMap` 的 lookup 和 insert，不受 scope 影响。适合那种"完全交给显式边管理"的 scratch buffer。 |
| `pl.no_dep(arg)` | per-call 参数 | kernel 调用点上，被包装的参数其 `ArgDirection` 变为 `NoDep`——**仅本次提交**对该槽位不进入自动跟踪。不论 callee 把该槽位声明为 `In`、`Out` 还是 `InOut` 都合法：用户在带外（out-of-band）承诺该槽位不存在 RaW / WaW / WaR 冲突——例如 paged-attention 那种"写偏移是数据相关、但按分配协议保证不相交"的场景。在 `pl.manual_scope` 内没有意义（scope 已经全员退出）。 |
| `with pl.at(..., no_dep_args=[t1, t2]):` | per-arg, 作用于 `pl.at`-块 | `pl.no_dep(arg)` 在 `pl.at`-块上的对应物。outliner 把列出的 tensor 作为合成 kernel call 的实参；`DeriveCallDirections` 随后把这些实参槽位标为 `NoDep`——和在显式 call 站点用 `pl.no_dep(...)` 等效。每一项必须是外层 scope 可见的张量名。In / Out / InOut 的适用范围与 `pl.no_dep(arg)` 相同：如果 scope 体里用 `pl.assemble` 写过这个 capture，outliner 会把合成 kernel 上该形参推断成 `InOut`，`no_dep_args=` 仍然把它覆盖为 `NoDep`（和覆盖 `In` 一样）。注意：`no_dep_args=` 接收**张量**，`deps=` 接收 **TaskId**——同一个 "dep"，作用在不同层。 |

#### 机制 B——显式声明 task 间的边（`deps=`）

这些表面都会下沉为 `set_dependencies` codegen；按 producer 形态选择：
单个 kernel 调用、outlined `pl.at` 区域，或 dependency-only fan-in。

| 表层语法 | producer 形态 | 备注 |
| -------- | ------------- | ---- |
| `result, tid = pl.submit(kernel, *args, deps=[...], allow_early_resolve=False)` | 单个 kernel 调用 | 尾部 `tid` 是 producer `pl.Scalar[pl.TASK_ID]`。它是 parser construct（类似 `pl.range`），不是 runtime 函数。`allow_early_resolve=True` 将该 task 标记为推测式 early-dispatch producer（让调度器提前预置其 consumer；lower 为 `Arg::set_allow_early_resolve(true)`）。同样接受 `predicate=(t[i] > 0)` —— 调度器在 dispatch 点求值的调度谓词（参见[调度谓词](#调度谓词predicate)）。 |
| `result, tid = pl.spmd_submit(kernel, *args, core_num=N, sync_start=False, deps=[...])` | 单个 SPMD task launch | `pl.submit` 的 SPMD 版本：将 kernel 在 `N` 个 block 上分发（一个 orchestration task → 一个 `tid`）。`core_num` 是必填关键字参数（正整数表达式）；`sync_start=True` 强制所有 block 原子启动。callee 可以是 InCore / AIC / AIV / Group。launch spec 记录在 `Submit.core_num` / `Submit.sync_start` 上。同样接受 `allow_early_resolve=True`（与 `pl.submit` 相同的 early-dispatch 选项）和 `predicate=(t[i] > 0)`（参见[调度谓词](#调度谓词predicate)）。 |
| `with pl.at(level=pl.Level.CORE_GROUP, deps=[...]) as tid:` | outlined `pl.at`-块 | 整块被 outline 成 InCore kernel + `Submit`；`tid` 捕获被合成的 Submit 的 TaskId，可作为后续 `pl.submit` / `pl.at` 的 dep。不写 `as tid` 时 outliner 会合成一个未使用的 TaskId Var——deps 始终走 `Submit::deps_`。同样接受 `allow_early_resolve=True`（与 `pl.submit` 相同的 early-dispatch 选项）；即使不写 `as tid` 也会强制走 `Submit` 形态，并 lower 为 `Arg::set_allow_early_resolve(true)`。 |
| `with pl.spmd(N, deps=[...]) as tid:` | outlined SPMD 分发 | `pl.at ... as tid` 形式的 SPMD 版本。内联 body 自动外包成 InCore kernel 并在 `N` 个 block 上分发；`tid` 捕获 grid 级 producer TaskId。`deps=` 仅在带 `as tid` 时可用。`core_num` / `sync_start` 记录在 lower 出的 `Submit` 自身的 `core_num` / `sync_start` 字段上（launch spec 属于启动点，而非外包出的被调函数）；codegen 直接从那里读取。同样接受 `allow_early_resolve=True`（与 `pl.submit` / `pl.at` 相同的 early-dispatch 选项；`pl.spmd` 三种形式均可用，即使不写 `as tid` 也会强制走 `Submit` 形态）和 `predicate=(t[i] > 0)`（参见[调度谓词](#调度谓词predicate)；同样三种形式均可用，同样强制走 `Submit` 形态）。不能嵌套在 `pl.cluster()` 内。 |
| `barrier = pl.system.task_dummy(deps=[...])` | dependency-only barrier | 不提交 kernel。返回的 TaskId 是一个紧凑的 fan-in 点，可供后续 `deps=[barrier]` 使用。 |
| `None`（Python 字面量） | 种子 / dep 条目 | "暂无 producer" 的哨兵。`prev_tid = None` 用作 TaskId 循环 iter_arg 的种子；`deps=[None]` 中的 `None` 被丢弃（不贡献任何边）。下沉为 `system.task_invalid` → `PTO2TaskId::invalid()`。 |

**这些表面都不依赖机制 A 的状态。** 显式 deps 可用于普通自动跟踪、
`pl.manual_scope()` 内或 `manual_dep=True` tensor 上，并总是在自动跟踪结果
**之上**追加；早期"`deps=` 只在 `pl.manual_scope` 内有效"的限制已经解除。

普通的 `out = self.kernel(...)` 是 **fire-and-forget**：它不返回 task id，
并且在它上面写 `deps=` 会被拒绝（parser 报错，提示 "use `pl.submit`"）。
每个 `deps=[...]` 条目必须是 TaskId 值：先前 `pl.submit(...)` /
`pl.at(..., deps=) as tid` 绑定的 `tid`、`pl.system.task_dummy(deps=[...])`
的返回值、TaskId 循环 iter_arg carry、从 TaskId 数组槽读出的
`Scalar[TASK_ID]`（`prev = tids[k]`）、来自
`pl.array.create(N, pl.TASK_ID)` 的 `Array[N, TASK_ID]`，或字面量 `None`。
`deps=[...]` 不接受 tensor。

```python
# 示例 1——两套机制同用：scope-wide 退出 + 显式边。
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64], pl.FP32],
         scratch: pl.Out[pl.Tensor[[64], pl.FP32]],
         out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
    with pl.manual_scope():                                           # 机制 A: scope-wide
        scratch, stage1_tid = pl.submit(self.stage1, x, scratch)
        out, _ = pl.submit(self.stage2, scratch, out, deps=[stage1_tid])  # 机制 B
    return out
```

```python
# 示例 2——只用机制 B，**不**进 manual_scope。其他 buffer 仍然走自动跟踪；
# 显式边是在自动跟踪结果**之上**追加。注意没有 `with pl.manual_scope():`。
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64], pl.FP32],
         out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
    tmp, prep_tid = pl.submit(self.preprocess, x)
    out, _ = pl.submit(self.consume, tmp, out, deps=[prep_tid])
    return out
```

```python
# 示例 3——以 pl.at-块作为 producer，给下游 pl.at-块加显式边。
# `as tid` 捕获被合成 outlined Call 的 TaskId。
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64], pl.FP32],
         out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP) as tid_a:
        # 块体被 outline 成 InCore kernel
        ...
    with pl.at(level=pl.Level.CORE_GROUP, deps=[tid_a]) as tid_b:
        # 显式边——严格在 tid_a 块之后运行
        ...
    return out
```

```python
# 示例 4——机制 A 的 tensor-lifetime 形态：scratch buffer 整生命周期退出
# 自动跟踪；ordering 完全交给显式边管理。
scratch = pl.create_tensor([N], dtype=pl.FP32, manual_dep=True)
scratch, prod_tid = pl.submit(self.fill, x, scratch)
out, _ = pl.submit(self.consume, scratch, out, deps=[prod_tid])
```

`pl.submit` 脱糖为单个 `ir.Submit`，其返回类型是扁平的增广
`TupleType([*<kernel return types>, ScalarType(TASK_ID)])` ——
元素 `0..N-1` 是 kernel 结果，元素 `N` 是 producer TaskId。parser 把每个
`deps=[...]` 列表直接写入类型化的 `Submit::deps_` 字段（普通 `Call` 永不
携带 `manual_dep_edges`——ManualDepsOnSubmitOnly 不变式）。`pl.at(..., deps=)`
走相同的路径：outliner 读 `ScopeStmt` 上的 `attrs["task_id_var"]` +
`attrs["manual_dep_edges"]`，把它们一起搬到合成的 `Submit` 上（带 deps 但
没写 `as tid` 的 scope 会得到一个合成的未使用 TaskId Var，使派发仍是 Submit）。codegen 填充一个按精确依赖数定长的栈数组，
并对每个 task 发出一次 `params.set_dependencies(arr, count);` 调用。
runtime 的 `Arg::set_dependencies(ptr, count)` 直接接收调用者持有的任意
长度数组，所以单 call 的依赖边数没有硬上限。显式 fan-in 可写成
`barrier = pl.system.task_dummy(deps=[tids])`，再让 consumer `deps=[barrier]`；
它复用同一套 dependency parser，lowering 成 `rt_submit_dummy_task(...)`，
在 dep 全 invalid 时返回 invalid 且跳过 dummy submit，并可与自动
`ExpandManualPhaseFence` barrier 共存。

`pl.no_dep(arg)` 是 auto scope 原语；在 `pl.manual_scope` 内不起作用
（整个 scope 已经退出自动跟踪了）。

#### 调度谓词（`predicate=`）

`pl.submit` / `pl.spmd_submit` 接受可选的
`predicate=(tensor[indices] <op> target)`。调度器在 **dispatch 点**
求值该比较 —— 此时该 task 的依赖已满足，值一定是最新的，
无需 orchestration 阶段 `wait_for_tensor_ready` 的阻塞。当比较结果为 **假** 时，该 task
被 **inline 退休**（根本不下发到 core），但仍结算 fanin/fanout，下游 consumer 照常解锁
—— 它不会从 task 图中消失。为 **真** 时正常下发。

典型用途是 MoE“跳过空专家”：所有专家静态 submit，每个带 `predicate=(row_count[e] > 0)`
并依赖 gather producer —— 调度器只下发非空专家，且无需阻塞 orchestration 去读每个专家的
行数。

> **该比较按普通表达式解析，但永不求值。** `rc[0, 0]` 就是 `pl.read` 的常规语法糖，
> 因此该 kwarg 下降为普通 IR —— `Gt(Cast(tensor.read(rc, [0, 0])), 0)` —— 复用 IR 已有的
> 比较节点，而非任何私有编码。它存放在 `Submit.predicate` 上，**不在语句位置**，所以这个
> `tensor.read` **不会**在 orchestration 中执行：执行它就会阻塞在 `wait_for_tensor_ready`，
> 正是谓词要消灭的事情。orchestration codegen 负责把该 Expr 分解成运行时的
> `operand OP target` 三元组，因此只接受下述形状。

| 组成 | 含义 | 约束 |
| ---- | ---- | ---- |
| `tensor` | dispatch 点读取的操作数张量 | 必须是具名 tensor（函数参数，或绑定到 tensor 的变量），且下标定位到单个元素 |
| `indices` | 定位 `tensor` 中一个元素的下标 | 每个下标是整数标量（`ConstInt` 或 int/index `Var`）；每个维度一个下标 |
| `<op>` | 比较算子 | 取 `==` `!=` `>` `<` `>=` `<=` 之一（单个、非链式比较） |
| `target` | 右侧比较值 | **整数字面量**（可负） |

镜像写法也被接受 —— `0 < rc[e]` 与 `rc[e] > 0` 含义相同。IR 按书写原样保留该比较；
orchestration codegen 会翻转算子，使 tensor 始终是运行时的操作数。

在 orchestration codegen 中 lower 为运行时 `L0TaskPredicate` + `Arg::set_predicate(...)`
（operand → 其 `ext_<name>` 引用，`op` → `PredicateOp::*`，`target` 原样；`elem_size`
由运行时从张量 dtype 推导）。

**契约：** 谓词操作数张量的 producer **必须**是该 submit 的 `deps=` 之一，这样 dispatch
点读到的才是最新值。若遗漏，调度器可能在 producer 尚未写入张量前就求值谓词，从而基于过期
数据做出 dispatch 决策。

parser 只做**尽力而为的抽查**，不是保证:它记录 `pl.submit(...)` 通过元组解包绑定的结果变量，
当谓词操作数是其中之一、而对应 producer TaskId 未出现在 `deps=` 时报错。解析通过应理解为
"未发现明显错误"，而非"已证明正确"。

以下情况它**看不穿**，因而会静默接受:

| 未覆盖 | 原因 |
| ------ | ---- |
| `rc2 = rc` 后使用 `rc2[0, 0]` | 别名是新变量，没有记录的 producer |
| 张量作 `pl.Out` 实参传入、结果绑到新名字 | 只跟踪返回绑定，不跟踪实参别名 |
| `rc3 = self.helper(rc)` | 任何中间调用都会洗掉关联 |
| `res = pl.spmd_submit(...)` 单左值形式 | 该路径根本不记录 |
| `deps=` 中含 `Array[N, TASK_ID]` 条目——包括常见的 `deps=[tids[i]]` 写法 | 数组条目未逐个列出 producer，该 submit 的检查整体跳过 |
| producer 写在源码**后面**，例如循环携带的 `rc` 由上一轮迭代写入 | 查表发生在解析谓词的那一刻，其后的 producer 尚未记录 |

因此 `deps=` 写对仍然是作者的责任。

**表达力**固定为 `tensor[indices] OP const` —— 单个比较，与运行时单比较的
`DispatchPredicate` 对齐。链式比较（`0 < t[i] < 8`）、算术（`t[i] % 8 == 0`）、布尔组合
（`a[0] > 0 and b[0] > 0`）、以及非字面量的右侧（`t[i] > u[i]`）都会在解析期被拒绝；
请在前序 kernel 里把它们归约成一个 gate 值，再对该值做谓词。

```python
with pl.manual_scope():
    rc, g_tid = pl.spmd_submit(self.gate, rc, core_num=1)       # rc 的 producer
    out, _ = pl.spmd_submit(
        self.expert, x, out, core_num=1,
        deps=[g_tid],                                           # producer 是依赖
        predicate=(rc[0, 0] > 0),
    )
```

**范围：** `predicate=` 可用于 `pl.submit` / `pl.spmd_submit`（直接产 `Submit` 的形式），
以及 `with pl.spmd(...)` 作用域形式的全部三种写法（普通 `with`、`with ... as tid`、
`for i in pl.spmd(...)`）。`pl.at(...)` 不接受该参数。

##### 作用域形式

作用域形式使用相同的表达式与相同的校验，区别只在于谓词进入 IR 的路径：它先挂在
`SpmdScopeStmt.attrs` 上，直到该作用域被 outline 时才移动到 `Submit.predicate`。
因此 lowering、codegen 产物与契约都完全一致。

```python
with pl.spmd(1) as g_tid:                                        # rc 的 producer
    rc = self.gate(rc)

with pl.spmd(4, deps=[g_tid], predicate=(rc[0, 0] > 0)) as tid:  # producer 是依赖
    out = self.expert(x, out)
```

由这条路径引出两点：

- **`deps=` 需要 `as tid` 形式。** `deps=` 只在 `with pl.spmd(...) as tid:` 上被接受。
  因此，若谓词读取的张量由同一函数内的其他任务产出，就必须用该形式；普通 `with` 与
  `for` 形式只能对没有函数内 producer 的张量（通常是函数参数）加谓词——这种情况契约
  检查会放行。
- **其余情况不要求 `as tid`。** 与 `allow_early_resolve=True` 一样，谓词会强制该作用域
  lower 为 `Submit`；当作用域没有 `as tid` 时，outliner 会合成一个未被使用的 TaskId Var。

嵌套在 `pl.cluster()` 内的 `pl.spmd` 会被展开进 Group 函数、永远不会产生 `Submit`，
因此 `predicate=`（与 `allow_early_resolve=` 一样）会在解析期被拒绝，而不是被静默丢弃。

契约检查同样覆盖作用域 producer：在 `with pl.spmd(...) as tid:` 体内被赋值的张量会被记录
为该作用域的产物，所以后续 `deps=` 漏写它会被拒绝。上表中列出的 best-effort 限制
（别名、中间调用、`Array[N, TASK_ID]` 依赖）依然适用。

#### Manual scope 下的 `pl.parallel`：array-carry fence

当 manual-dep 边穿过一个 `pl.parallel` 循环（即循环 iter_arg 承载被依赖的
TaskId）时，orchestration codegen 把对应的 TaskId iter_arg 视作**大小等于
parallel 循环 trip count 的数组**。每次 parallel 迭代写入自己的槽位；
下游消费者依赖**每一个**槽位（不是只依赖"最后被发射"的那个
task）。这就保证用户声明的 fence 语义即便在迭代乱序完成时也是正确的。

走 array-carry 路径的前提：

- `pl.parallel` 的 trip count 必须是 Python 字面量（编译期常量）。
  trip count 是动态值的情况下 codegen 会拒绝，提示 "statically-known
  trip count"。

```python
with pl.manual_scope():
    prev_tid = None                                      # 种子：还没有 producer
    for phase in pl.range(N_PHASES):
        for branch in pl.parallel(N_BRANCHES):           # 编译期常量
            row = (phase * N_BRANCHES + branch) * TILE_M
            out, prev_tid = pl.submit(self.kernel_stripe, data, row, 1.0, out, deps=[prev_tid])
```

`prev_tid` 在 `pl.parallel` 内被重新绑定，所以 codegen 把 carry 下沉为
`PTO2TaskId[N_BRANCHES]` 数组。phase `N+1` 中的每个 task 都会等待
phase `N` 的全部 `N_BRANCHES` 个 task，而非只等最后那个。

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
| `pl.dump_tag(tensor)` | 把某个张量标记为运行期选择性 dump 的目标 —— 声明式逐张量标记（在 Orchestration 作用域，或被 orch 内联的 Inline helper 中均可使用 —— 见 [运行期 DFX](../03-runtime-dfx.md#选择性张量-dump)） | 在非 Orchestration / Inline 函数中使用、或参数不是裸变量名时抛出 `ParserSyntaxError` |

**要点：**

- 三者均为语句级构造（不能用在表达式中）
- `static_print` 接受变量、常量、字符串标签（原样打印）和 f-string 的简单 `{expr}` 占位符（格式化为 IR）。不支持转换标志（`!r`、`!s`、`!a`）和格式说明符（`:...`）。
- `static_assert` 支持闭包变量表达式（如 `N > 32`）和 IR 常量
- `static_assert` 的消息参数必须是字符串字面量
- `dump_tag` 接受单个绑定在外层 Orchestration（或 Inline）作用域内的张量变量名，在解析期被消耗，并自始至终以 Var 身份（而非名字）跟踪到 codegen。在显式 `self.kernel(...)` 调用点，它把该张量记录到每个后续消费它的 Call 的 `dump_vars` 上；在 `@pl.jit` / `with pl.at(level=...)` 风格（派发由 outline pass 合成）下，它改为写入所在 scope 的 `dump_vars`，再由 outliner 映射到合成派发的实参上（见 [运行期 DFX](../03-runtime-dfx.md#选择性张量-dump)）。若需要在单次 task 启动处显式列出 dump 目标，请用 `pl.submit(...)` / `pl.at(...)` 上的 `dumps=[...]` kwarg（与 `deps=` 对称）
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

## 跨模块函数复用

在 `@pl.program` 类之外定义的函数可通过两种机制复用。

### 外部 `@pl.function` 调用

在 `@pl.program` 内部可按名称调用外部定义的 `@pl.function`。该函数会自动加入 Program，
并生成 `ir.Call(GlobalVar, args)`。

```python
@pl.function
def softmax(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    ...

@pl.program
class MyModel:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = softmax(x)   # ir.Call(GlobalVar("softmax"), [x])
        return y
```

**规则：**

- 使用函数的 `.name` 作为 GlobalVar（别名透明）
- 外部与内部函数名不得冲突
- 两个不同的外部函数具有相同 `.name` 是错误
- 同一外部函数从多个 method 调用时只加入一次

### `@pl.inline` 装饰器

`@pl.inline` 捕获函数以便在语句级内联。不会向 Program 添加函数——每次调用点展开函数体。

```python
@pl.inline
def normalize(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    return result

@pl.program
class MyModel:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = normalize(x)  # statements inlined in-place
        return y
```

**规则：**

- 实参个数必须与形参列表完全一致
- 内联定义处的闭包变量可用
- 内联函数可多次调用（每次展开相互独立）
- 支持嵌套内联调用

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
