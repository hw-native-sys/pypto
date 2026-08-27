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

**注意:** 模块前缀可配置 (默认 `pl`, 旧版 `ir`, 支持自定义)，但 `pld` 保留给
`pypto.language.distributed` 使用。

本规范拆分为四个页面：

| 页面 | 内容 |
| ---- | ---- |
| 本页 | 模块结构、类型系统、表达式 |
| [语句与控制流](01-statements.md) | 赋值、if/for/while、作用域、yield、编译期指令、SSA phi 节点 |
| [手工依赖原语](02-manual_dependencies.md) | `pl.manual_scope`、`deps=`、调度谓词、array-carry fence |
| [函数与程序结构](03-functions.md) | 函数类型、参数方向、跨模块复用、打印 |

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
[`LowerPipelineToSlots`](../passes/28-lower_pipeline_to_slots.md) 会为合格 `pl.pipeline` 循环体中
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

参见 [InitMemRef](../passes/32-init_memref.md#声明式分配) 与
[MemoryReuse](../passes/34-memory_reuse.md#声明式分配)。

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

#### 统一算子的跨路径参数 (Cross-path arguments)

统一算子 `pl.<op>` 接受两个层级参数的并集，因此只有*另一条*分发路径才能处理的
参数会被**拒绝，而不会被丢弃**——被悄悄丢弃的 `b_trans` 会编译出错误的数学
语义。反方向同样成立：Tensor 输入必须省略的临时操作数
(`pl.row_max(tensor)`)，正是 Tile 输入必须提供的那一个
(`pl.row_max(tile, tmp_tile)`)，因为 tile 缓冲区的生命周期由用户管理。

**两个方向都抛出 `TypeError`**——它们属于"参数与该重载不匹配"这一类错误，
即 Python 自身在遇到意外关键字参数或缺少必需参数时抛出的类型。通过该包装器
到达更深层的校验 (形状、dtype、边界——任何由 C++ `CHECK` 拒绝的情况) 仍然抛出
`ValueError`，因此保护整个调用的代码应当同时捕获两者:

```python
pl.matmul(tile_a, tile_b, b_trans=True)   # TypeError — tile 转置是视图，不是标志位
pl.rsqrt(tile, high_precision=True)       # TypeError — tile 精度通过传入 tmp 选择
pl.div(tile, 2.0, high_precision=True)    # TypeError — high_precision 需要 Tile rhs
pl.row_max(tile)                          # TypeError — Tile 输入必须提供 tmp_tile
pl.slice(tile, [64, 64], [64, 0])         # ValueError — 窗口越出源 tile 边界
```

在 `@pl.function` 函数体内这一区别是不可见的: 解析器会同时捕获两者，并重新
抛出带有源码位置的 `InvalidOperationError`。

## 参考资料

- [IR 概述](../ir/00-overview.md) - 核心 IR 结构
- [IR 解析器 (Parser)](../ir/07-parser.md) - 将 Python 语法解析回 IR
- [操作符注册](../ir/05-operators.md) - 操作系统和类型推断
