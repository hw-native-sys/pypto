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

### 有效形状与填充

某个算子写入的元素可能少于 tile 分配的量 —— 比如某一维最后一块不能整除的部分。`pl.set_validshape` 记录该区域，`pl.fillpad` 用选定的值填充其余部分。

填充值对规约至关重要，而且失败是静默的：

- 对一块用零填充的 tile 做 `max` 规约，即使所有真实元素都是负数，结果也至少是 0。请用 `pl.PadValue.min`。
- 对 `sum` 而言，填充值只要不是 0，就会被加进结果里。

```python
t = pl.load(x, [off, 0], [128, 128])
t = pl.set_validshape(t, [rows_left, 128])     # only `rows_left` rows are real
m = pl.row_max(t)                              # pad value decides what the tail contributes
```

`pl.fillpad_expand` 把填充与广播合成一步。规约与广播家族本身的清单见 [算子](../ops/01-catalog.md)。

### 让数据留在片上

为循环的每一块 tile 重复加载同一个操作数，是最常见的可避免开销。当操作数是循环不变量时把 load 提到循环外；对在 K 轴循环中被反复使用的矩阵乘操作数，优先让它常驻 `Mat`。编译器在这里做与不做什么 —— 缓冲区复用、地址分配 —— 由 [MemoryReuse](../../dev/passes/34-memory_reuse.md) 与 [AllocateMemoryAddr](../../dev/passes/35-allocate_memory_addr.md) 决定；讲如何驾驭它们的性能章节尚未编写。

## Edge Cases

> **致命陷阱：** 对一块用零填充的部分有效 tile 做 `max` 规约，全负数的行会返回 0。没有报错也没有告警 —— 只是恰好那些不满的行数字错了。请根据规约类型选择填充值：`max` 用 `PadValue.min`，`sum` 用 `0`。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **tile 级 `pl.matmul` 拒绝它的操作数** | 操作数不在 `Left` / `Right` | 先 `pl.load` 到 `Mat`，再 `pl.move` |
| **`pl.load(..., target_memory=pl.Mem.Left)` 被拒绝** | DDR load 只能到 `Vec` / `Mat` | 先 load 到 `Mat`，再 `pl.move` 到 `Left` |
| **只有最后一块 tile 的规约结果不对** | 填充值参与了规约 | 用 `pl.set_validshape`，并选对 `PadValue` |
| **InCore 函数内 `pl.create_tensor` 失败** | 张量分配是控制面的事 | 在控制面分配，或改为接收 `pl.Out[...]` 参数 |
| **片上缓冲区耗尽** | 同时常驻的东西太多 | 缩小 tile，或用 `pl.cross_core_slot(slot_num=N)` 缩小跨核环 |

## See Also

- [类型](00-types.md) —— `Tensor` 与 `Tile` 的区别，以及 dtype 的 `get_byte()` 有什么用。
- [作用域与放置](04-scopes.md) —— 代码在哪里执行，以及跨核环深度。
- [算子](../ops/01-catalog.md) —— 搬运、规约与广播家族。
- [InferTileMemorySpace](../../dev/passes/18-infer_tile_memory_space.md) —— 替你插入 move 的那个 pass。
- [MemoryReuse](../../dev/passes/34-memory_reuse.md) —— 缓冲区如何按生命周期共享。
- [Memory Map](../../dev/07-memory-map.md) —— 可视化最终落在片上的东西。
