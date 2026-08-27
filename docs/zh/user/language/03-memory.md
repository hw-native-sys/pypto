# 内存与数据搬运

片上各个内存空间、在它们之间搬运数据的算子，以及一块未填满的 tile 在边缘处会发生什么。

> **前置**：[编程模型 § 内存层次](../03-programming-model.md#内存层次)。

## Concept

```text
                 off-chip                          AI Core
              ┌────────────┐   ┌──────────────────────────────────────────────┐
              │            │   │                                              │
              │            │   │   Cube unit (AIC)         Vector unit (AIV)  │
              │            │   │  ┌──────────────┐        ┌─────────────────┐ │
              │    DDR     │   │  │ Left  (L0A)  │        │                 │ │
              │            │   │  │ Right (L0B)  │        │   Vec  (UB)     │ │
              │ pl.Tensor  │   │  │ Acc   (L0C)  │        │                 │ │
              │  lives     │   │  │ Bias         │        │                 │ │
              │  here      │   │  └──────▲───────┘        └────────▲────────┘ │
              │            │   │         │ pl.move                 │          │
              │            │   │  ┌──────┴─────────────────────────┴────────┐ │
              │            │   │  │              Mat  (L1)                  │ │
              │            │   │  └──────────────────▲──────────────────────┘ │
              └─────┬──────┘   └─────────────────────┼────────────────────────┘
                    │                                │
                    └────────  pl.load / pl.store  ──┴──►  Vec or Mat only
```

> 此图为占位，待补一张正式的 Ascend 910 平面图。

六个片上空间是**彼此并列的缓冲区，不是嵌套关系**：`Left` 不是 `Mat` 里的一块区域，`Acc` 也不在 `Right` 里面。数据在它们**之间**搬运，而哪些搬运合法是硬件属性，不是编译器策略。

有一条约束带来了大部分后果，图里也直接看得到：**面向 DDR 的 load 只能落到 `Vec` 或 `Mat`。** 矩阵乘的操作数空间（`Left`、`Right`）和累加器（`Acc`）只能通过从 `Mat` 或 `Vec` 出发的 `pl.move` 到达，或者由 `pl.matmul` 写入 `Acc`。这就是为什么矩阵乘通路是明确的两段式形态，而逐元素代码不是。

第二个反复出现的概念是**有效形状（valid shape）**。一块 tile 有它分配出来的形状，以及可选的、真正装着有意义数据的更小区域。读到有效区之外的算子看到的是填充值所决定的内容 —— 这正是对部分有效的 tile 做规约时必须刻意选择填充值的原因。

## Quickstart：两条通路

```python
import pypto.language as pl

@pl.jit.incore
def elementwise(x: pl.Tensor[[128, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[128, 128], pl.FP32]]):
    t = pl.load(x, [0, 0], [128, 128])       # DDR -> Vec (default target)
    y = pl.mul(t, t)                          # compute in Vec
    pl.store(y, [0, 0], out)                  # Vec -> DDR
    return out

@pl.jit.incore
def mm(a: pl.Tensor[[32, 32], pl.FP16],
       b: pl.Tensor[[32, 32], pl.FP16],
       out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]):
    a_l1 = pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat)   # DDR -> Mat
    b_l1 = pl.load(b, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
    a_l0a = pl.move(a_l1, target_memory=pl.Mem.Left)                 # Mat -> Left
    b_l0b = pl.move(b_l1, target_memory=pl.Mem.Right)                # Mat -> Right
    c = pl.matmul(a_l0a, b_l0b)                                      # -> Acc
    pl.store(c, [0, 0], out)                                         # Acc -> DDR
    return out
```

张量级的 `pl.matmul` 会被降级成同样这条链。手写出来，换来的是对分块与常驻的控制。

## Mechanics

### 各个空间

| 空间 | 枚举 | 硬件 | 能否从 DDR 直达？ |
| ---- | ---- | ---- | ----------------- |
| DDR | `pl.Mem.DDR` | 片外全局内存 | 它本身就是 DDR —— `pl.Tensor` 住在这里 |
| Vec | `pl.Mem.Vec` | 统一缓冲区 | **能** —— `pl.load` 的默认目标 |
| Mat | `pl.Mem.Mat` | L1 | **能** —— `target_memory=pl.Mem.Mat` |
| Left | `pl.Mem.Left` | L0A，矩阵乘左操作数 | 不能 —— 只能 `pl.move` |
| Right | `pl.Mem.Right` | L0B，矩阵乘右操作数 | 不能 —— 只能 `pl.move` |
| Acc | `pl.Mem.Acc` | L0C，矩阵乘累加器 | 不能 —— 由 `pl.matmul` 写入 |
| Bias | `pl.Mem.Bias` | AIC 核上的 bias 缓冲区 | 不能 —— 只能 `pl.move` |

`pl.MemorySpace` 与 `pl.Mem` 是同一个枚举的两个名字。

这是数据流而非包含关系 —— 两个矩阵乘操作数**汇合**到 `Acc`，所以这是一张图，不是一棵树：

```text
     pl.load(target_memory=Mat)     pl.move(Left)
DDR ───────────────────────► Mat ─────────────────► Left ┐
                                                         │ pl.matmul
                                                         ├────────► Acc ────► DDR
DDR ───────────────────────► Mat ─────────────────► Right┘            pl.store
     pl.load(target_memory=Mat)     pl.move(Right)

     pl.load()                elementwise ops             pl.store()
DDR ──────────► Vec ────────────────────────────► Vec ──────────────► DDR
     (default)
```

当消费方需要 `Left` / `Right` / `Acc` / `Bias` 时，生产方停在 `Mat`（或 `Vec`），由 [InferTileMemorySpace](../../dev/passes/18-infer_tile_memory_space.md) 插入 `tile.move` —— 你显式写出来，是为了控制它发生在哪里。

### 搬运数据

| 算子 | 方向 | 说明 |
| ---- | ---- | ---- |
| `pl.load(tensor, offsets, shape, target_memory=...)` | DDR → Vec / Mat | 默认 `Vec` |
| `pl.store(tile, offsets, tensor)` | 任意片上空间 → DDR | |
| `pl.move(tile, target_memory=...)` | 片上 → 片上 | 进入 `Left` / `Right` / `Bias` 的唯一途径 |
| `pl.create_tile(shape, dtype, ...)` | — | 分配片上缓冲区 |
| `pl.create_l1(...)` | — | 显式在 L1 上分配 |

一个被放置好的缓冲区在 IR 里由 `pl.MemRef` 描述 —— 空间加上分配器指派的地址。你很少需要自己写它；它出现在读 printed IR 和看 memory map 的时候。

`offsets` 与 `shape` 是被搬运的那块张量区域 —— 偏移是**张量**内的偏移，shape 是结果 tile 的尺寸。

### 标量元素访问

`pl.read` 与 `pl.write` 按下标访问**张量**的单个元素，中间不经过 tile：

```python
n = pl.read(counts, [0])                      # 从 DDR 读一个 INT32
pl.write(plan, [row], pl.cast(v, pl.INT32))   # 往 DDR 写一个 INT32
```

它走的是与 `pl.load` / `pl.store` **不同的通路**，而不是它们的小号版本：

| 对比项 | `pl.load` / `pl.store` | `pl.read` / `pl.write` |
| ------ | ---------------------- | ---------------------- |
| 单位 | 一个 tile | 一个元素 |
| 到 DDR 的通路 | DMA，直达 | 发起核自己的数据 cache |
| 真正抵达 DDR 的粒度 | tile 覆盖的字节 | **整条 64 字节 cache line** |
| 多个核同时写是否安全 | 是 | 仅在下述规则成立时 |

请把它们用于控制值 —— 计数器、偏移、小的描述符表 —— 而不是批量数据。一个 `pl.write` 循环每次只搬一个元素，而一次 `pl.store` 搬的是整个 tile。

### 来自并发任务实例的标量写

一次 `pl.write` 并不会自己抵达 DDR。它落在发起核的数据 cache 里，而该 cache 在 kernel 结束时按**整条 64 字节 line** 写回。不同核的 cache 之间没有任何一致性维护。

于是当两个核写入**恰好落在同一条 64 字节 line 上的不同元素**时，各自都会把自己那份完整的 64 字节写回去 —— 其中只有一个元素是新的，另外 15 个是它从未碰过的陈旧值。最后写回的那个赢下整条 line，另一个核的写就消失了。运行时既不报错也不告警：张量在大多数下标上仍保持旧值，而**哪些**下标幸存每次运行都不一样。

> **致命陷阱：** 两个实例写同一条 64 字节 line 上的*不同*元素，会静默地互相丢失对方的写。下标互不相交**并不够** —— 抵达内存的单位是 line，所以一个实例写回时会连带把它看到的那 15 个相邻元素一并写回，覆盖掉别的实例放在那里的值。

这件事关乎**并发，而不是 `pl.spmd`**。有两种构造会让你的代码以多个实例运行，任意一种都足以踩中：

| 构造 | 实例数 |
| ---- | ------ |
| `pl.spmd(n)`，`n > 1` | `n` 个 block，每核一个 |
| `for g in pl.parallel(n):` | `n` 个可能被运行时重叠执行的任务实例 |

从这两者派发出去的 kernel 会继承这个重数，所以写在 `@pl.function(type=InCore)` 被调函数里的 `pl.write` 同样算数。

**单个实例内部没有这个风险。** 一个实例在一个核上顺序执行自己的函数体，它的写按程序序落进同一个 cache，不存在 line 争用 —— 一个任务内部普通 `pl.range` 循环里的 `pl.write`，无论下标怎么写都是安全的。

**规则：每个实例必须独占完整的 64 字节 line。** 对 `INT32` / `FP32` 是 16 个元素，`FP16` / `BF16` 是 32 个，`INT64` 是 8 个，`INT8` 是 64 个。

```python
N = 64          # INT32 -> 每条 64 字节 line 放 16 个元素

# 错误 —— 网格跨步：block 0..15 各有一个元素落在 out[0:16]，于是 16 个 block
# 共享这条 line（它们同时还会写到后面的 line）
with pl.spmd(24):
    blk = pl.tile.get_block_idx()
    for i in pl.range(pl.cast(blk, pl.INDEX), N, 24):
        pl.write(out, [i], pl.cast(pl.read(src, [i]) + 1, pl.INT32))

# 正确 —— block b 独占 out[16b : 16b+16]，恰好一条 line
with pl.spmd(N // 16):
    blk = pl.tile.get_block_idx()
    base = pl.cast(blk, pl.INDEX) * 16
    for i in pl.range(base, base + 16):
        pl.write(out, [i], pl.cast(pl.read(src, [i]) + 1, pl.INT32))
```

两段代码都对每个下标恰好写一次、且只由一个 block 写。但只有第二段是对的。

| 你的需求 | 做法 |
| -------- | ---- |
| 少量控制值 | `pl.spmd(1)` —— 单个实例在任何布局下都正确 |
| 每个实例写一段连续区间 | 让这段区间的**大小和起始**都对齐到 64 字节 |
| 多个实例真正做 scatter | 往清零的张量做 `pl.store(..., atomic=pl.AtomicType.ADD)` —— 走 DMA 通路，是一致的 |
| 每个实例的部分结果 | 先写到每实例的 scratch 行，之后再用 `pl.spmd(1)` 汇总 |

当编译器无法证明该规则成立时会告警 —— 见下面的 [`ScalarWriteLineShared`](#scalarwritelineshared)。

读同样经过这个 cache，但实践中没有这个风险：只有当另一个实例在**同一个任务内**写过该元素时才可能读到陈旧值，而那本身就已经违反了 `pl.spmd` 与 `pl.parallel` 所断言的独立性。跨任务时该 line 会被 invalidate，下一个任务读到的是新数据。

#### `ScalarWriteLineShared`

对每一个写入"生命周期长于写它的那个实例"的张量的 `pl.write`，编译器会尝试证明每个实例的字节都落在完整的、实例私有的 64 字节 line 上。证明不了的它都会报出来，并区分是哪一种情况。

当下标可分析且布局确实是交错的，它会给出实测的跨步：

```text
[warning] [ScalarWriteLineShared] pl.write into 'out' from 24 concurrent blocks
  ('fill_spmd') in function 'main': consecutive blocks write 4 bytes apart, so 16 of
  them share each 64-byte cache line and their stores overwrite one another. [...]
  Give each one whole 64-byte lines (16 x INT32), or issue the writes from a single
  instance (pl.spmd(1)).
```

当下标根本无法分析 —— 最常见的原因是下标本身是从另一个张量读出来的 —— 它会如实说明，而不去猜：

```text
[warning] [ScalarWriteLineShared] pl.write into 'out' from 24 concurrent blocks
  ('moe_route_gather_spmd') in function 'main': the index is computed at runtime, so
  the compiler cannot tell whether two blocks share a 64-byte cache line. [...]
```

第二种是常见形态，而且它是一个**问题**而非判决：用运行时下标的代码完全可能是对的。请确认你的各个实例落在 64 字节边界上；如果确实如此，这条告警是在告诉你：正确性依赖于一条没有任何东西强制保证的布局不变量，值得在写入处写一句注释。若要在整次构建中关掉这个检查，把 `ScalarWriteLineShared` 放进 pass context 的 `disabled_diagnostics`。

有两种情况它无法精确判定，因为两者都需要此刻还不存在的任务依赖图。两个**不同的**任务写同一个张量：**完全不报** —— 它们是否在时间上重叠无从得知，若要报就会在每一对有序的生产者/消费者上误报。被"把实例下标钉死到单个值"的谓词保护起来的写（`if blk == 0:`）：**会报**，且是保守地报 —— 该谓词其实让它是安全的，但检查不读谓词，于是仍按多实例处理。

### 有效形状与填充

某个算子写入的元素可能少于 tile 分配的量 —— 比如某一维最后一块不能整除的部分。`pl.set_validshape` 记录该区域，`pl.fillpad` 用选定的值填充其余部分。

填充值对规约至关重要，而且失败是静默的：

- 对一块用零填充的 tile 做 `max` 规约，即使所有真实元素都是负数，结果也至少是 0。请用 `pl.PadValue.min`。
- 对 `sum` 而言，填充值只要不是 0，就会被加进结果里。

```python
t = pl.load(x, [off, 0], [128, 128])
t = pl.set_validshape(t, rows_left, 128)       # only `rows_left` rows are real
m = pl.row_max(t)                              # pad value decides what the tail contributes
```

`pl.fillpad_expand` 把填充与广播合成一步。规约与广播家族本身的清单见 [算子](../ops/01-catalog.md)。

### 让数据留在片上

为循环的每一块 tile 重复加载同一个操作数，是最常见的可避免开销。当操作数是循环不变量时把 load 提到循环外；对在 K 轴循环中被反复使用的矩阵乘操作数，优先让它常驻 `Mat`。编译器在这里做与不做什么 —— 缓冲区复用、地址分配 —— 由 [MemoryReuse](../../dev/passes/34-memory_reuse.md) 与 [AllocateMemoryAddr](../../dev/passes/35-allocate_memory_addr.md) 决定；如何驾驭它们见 [内存](../performance/05-memory.md)。

## 边界情况

> **致命陷阱：** 对一块用零填充的部分有效 tile 做 `max` 规约，全负数的行会返回 0。没有报错也没有告警 —— 只是恰好那些不满的行数字错了。请根据规约类型选择填充值：`max` 用 `PadValue.min`，`sum` 用 `0`。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **tile 级 `pl.matmul` 拒绝它的操作数** | 操作数不在 `Left` / `Right` | 先 `pl.load` 到 `Mat`，再 `pl.move` |
| **`pl.load(..., target_memory=pl.Mem.Left)` 被拒绝** | DDR load 只能到 `Vec` / `Mat` | 先 load 到 `Mat`，再 `pl.move` 到 `Left` |
| **只有最后一块 tile 的规约结果不对** | 填充值参与了规约 | 用 `pl.set_validshape`，并选对 `PadValue` |
| **InCore 函数内 `pl.create_tensor` 失败** | 张量分配是控制面的事 | 在控制面分配，或改为接收 `pl.Out[...]` 参数 |
| **大多数 `pl.write` 的写消失了，且每次运行消失的都不一样** | 并发的 `pl.spmd` block 或 `pl.parallel` 实例写进了同一条 64 字节 line | 让每个实例独占完整的 64 字节 line，或改由 `pl.spmd(1)` 来写 |
| **片上缓冲区耗尽** | 同时常驻的东西太多 | 缩小 tile，或用 `pl.cross_core_slot(slot_num=N)` 缩小跨核环 |

## 配套示例

| 示例 | 展示 |
| ---- | ---- |
| `examples/intermediate/05_assemble.py` | 按偏移把 tile 写进目标，不经过 GM 往返 |
| `examples/intermediate/01_fused_linear.py` | 一个中间结果跨 cube 与 vector 操作留在片上 |
| `examples/runtime/multi_program_kv_cache.py` | 跨多个程序共享的设备常驻 buffer |

## See Also

- [类型](00-types.md) —— `Tensor` 与 `Tile` 的区别，以及 dtype 的 `get_byte()` 有什么用。
- [作用域与放置](04-scopes.md) —— `pl.spmd` 的 block，以及你在断言什么样的独立性。
- [作用域与放置](04-scopes.md) —— 代码在哪里执行，以及跨核环深度。
- [算子](../ops/01-catalog.md) —— 搬运、规约与广播家族。
- [InferTileMemorySpace](../../dev/passes/18-infer_tile_memory_space.md) —— 替你插入 move 的那个 pass。
- [MemoryReuse](../../dev/passes/34-memory_reuse.md) —— 缓冲区如何按生命周期共享。
- [Memory Map](../../dev/07-memory-map.md) —— 可视化最终落在片上的东西。
