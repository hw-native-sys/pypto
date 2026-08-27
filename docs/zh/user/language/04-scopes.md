# 作用域与放置

代码在哪里执行：把一个区域标记为设备工作、把协同调度的核归组、把一个 kernel 铺开到多个 block。

> **前置**：[函数与程序](01-functions.md) 与 [编程模型 § 执行模型](../03-programming-model.md#执行模型)。

## Concept

放置回答一个问题：**这段代码由哪块硬件执行。**

一共四个构造，都用 `with`（或 `for`）书写，且可以组合：

| 构造 | 把工作放到 |
| ---- | ---------- |
| `pl.at` | 一个核组 —— 把区域标记为设备工作 |
| `pl.cluster` | 同一个物理 cluster —— 让 Cube 与 Vector kernel 协同调度 |
| `pl.spmd` | `n` 个 block —— 同一个 kernel，每个 block 跑一份 |
| `pl.split_aiv` | 两条 AIV lane —— 把一个区域切开分给两条 |

`pl.at` 的替代写法是单独写一个 `@pl.jit.incore` 函数再调用它。两者产生的东西一样：编译时 `pl.at` 正是被 outline 成这样一个函数。区域短、且就该写在原地时用作用域；值得起个名字、或者要被多处调用时写成独立函数。

放置与**定序**不是一回事 —— 定序说的是本任务开始之前什么必须先完成。运行时是从 [类型](00-types.md) 里的参数方向、以及每个任务触碰的缓冲区推导出来的。那套机制和手工干预它的接口在 [任务与定序](../tasks/index.md)；本页只讲代码落在哪里。

## Quickstart：把一个区域标记为设备工作

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

CFG = RunConfig(platform="__PLATFORM__")
torch.manual_seed(0)
X = torch.randn(256, 128, dtype=torch.float32)
Y = torch.randn(256, 128, dtype=torch.float32)
```

<!-- doctest: run -->
```python
@pl.jit
def scale(
    x: pl.Tensor[[256, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        out[:] = pl.mul(x, 2.0)
    return out


out = torch.zeros(256, 128, dtype=torch.float32)
scale(X, out, config=CFG)
torch.testing.assert_close(out, X * 2.0, rtol=1e-4, atol=1e-4)
```

| 元素 | 作用 |
| ---- | ---- |
| `@pl.jit` | 入口 —— 控制面，它自己不能放算子 |
| `with pl.at(level=pl.Level.CORE_GROUP)` | 把该区域标记为设备工作，给算子一个合法的容身之处 |
| `pl.mul(x, 2.0)` | 在核组上执行 |

去掉 `pl.at`，这个 kernel 会编译失败并报 `Misplaced tensor op ... should be inside InCore block` —— 见 [函数与程序](01-functions.md)。

## Mechanics

### `pl.at`

`level=` 选择层级。`pl.Level.CORE_GROUP` 是产生 InCore 作用域的那个，该区域在编译时会变成一个独立的 kernel 函数。

两个可选关键字塑造这个被 outline 的 kernel：

| 关键字 | 含义 |
| ------ | ---- |
| `optimizations=[pl.split(mode)]` | 被 outline kernel 的跨核切分模式 |
| `optimizations=[pl.cross_core_slot(slot_num=N)]` | 自动跨核管道的环深度 |
| `name_hint="..."` | outline 出的函数名 |

`optimizations=` 的条目必须在调用点内联书写 —— 解析器读的是 AST，因此用变量拼出来的列表不被接受。`pl.split` 与 `pl.cross_core_slot` 彼此正交、可自由组合：一个切分工作，一个给通道定尺寸。

```python
with pl.at(level=pl.Level.CORE_GROUP,
           optimizations=[pl.split(pl.SplitMode.UP_DOWN),
                          pl.cross_core_slot(slot_num=4)]):
    ...
```

省略 `cross_core_slot` 就保持默认环深度：每个活跃方向 2 个 slot——刚好够对这次交接做双缓冲，
同时给 tile 本身留出片上空间。若希望生产侧核能跑得更超前，再调高它。

### SPMD

`pl.spmd(n)` 让同一个 kernel 在 `n` 个 block 上执行。两种形式，区别在于函数体是否读 block 索引：

```python
# Dispatch form — the body launches a kernel defined elsewhere.
with pl.spmd(4):
    out = self.kernel(a, b, out)

# Loop form — the body is auto-outlined and `i` binds the block index.
for i in pl.spmd(4):
    off = i * 128
    out = pl.store(pl.add(pl.load(a, [off, 0], [128, 128]),
                          pl.load(b, [off, 0], [128, 128])), [off, 0], out)
```

`@pl.jit` 入口能直接写的是循环形式：

<!-- doctest: run -->
```python
@pl.jit
def spmd_add(
    a: pl.Tensor[[256, 128], pl.FP32],
    b: pl.Tensor[[256, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
):
    for i in pl.spmd(2):                    # `i` binds the block index
        off = i * 128
        out = pl.store(
            pl.add(pl.load(a, [off, 0], [128, 128]), pl.load(b, [off, 0], [128, 128])),
            [off, 0],
            out,
        )
    return out


out = torch.zeros(256, 128, dtype=torch.float32)
spmd_add(X, Y, out, config=CFG)
torch.testing.assert_close(out, X + Y, rtol=1e-4, atol=1e-4)
```

一个既不读 block 索引、也不派发 kernel 的 `with pl.spmd(n):` 体会被拒绝 —— 那样每个 block 都在做完全相同的工作。

当涉及硬 `pl.system.syncall` 时，请按设备实际规模而非字面量来定启动规模：传 `pl.system.available_cluster_count()`（混合或纯 cube kernel）或 `pl.system.available_aiv_count()`（纯 vector kernel），并在调用点内联书写。

### Cluster 与 AIV lane

`with pl.cluster():` 把 AIC 与 AIV kernel 归组，让它们在同一个物理 cluster 上协同调度，产生一个 `Group` 函数。

`for aiv_id in pl.split_aiv(2, mode=...):` 把一个区域切开分给两条 AIV lane。它属于混合 kernel 编程（AIC 与 AIV 在同一个函数里协作），教程章会端到端地讲。

`mode=` 决定两条 lane 如何分担工作：

| `mode=` | 每条 lane 拿到 |
| ------- | -------------- |
| `pl.SplitMode.UP_DOWN` / `LEFT_RIGHT` | 每块 tile 的一半（按行 / 按列）—— 数据并行 |
| `pl.SplitMode.NONE` | **完整**函数体；由你通过 `aiv_id` 分派互不相交的工作 —— 任务并行 |

**只要开了一个区域，整个函数的规则就变了。** 此后区域拥有向量计算的全部放置决定权，因此向量计算必须写在区域里：

```python
with pl.at(level=pl.Level.CORE_GROUP):
    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
        ...                                    # 阶段 1 —— 通过 aiv_id 分派每 lane 的工作
    pl.system.syncall(core_type=pl.KernelType.MIX)  # 屏障：写在外面，两核都执行
    mm = pl.matmul(q, k)                       # cube 计算：写在外面，跑在 AIC 上
    for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):
        out = pl.add(pl.aiv_shard(mm), bias)   # 阶段 2 —— 全宽向量计算
```

每个向量阶段一个区域。`mode=NONE` 正是给**不希望被折半**的阶段用的包装：两条 lane 都运行完整函数体，而这本来就是没写区域时向量代码的行为，所以包起来只改变写法、不改变执行。cube 算子与屏障留在区域外。

`mm` 由 cube 产出、却在向量 lane 上被读取，因此它跨越了 AIC/AIV 边界 —— `pl.aiv_shard`
就是把这件事写出来。下一节解释为什么这是必须的。

完全**没有** `pl.split_aiv` 的函数不受影响 —— 按原来的写法写即可。

**同一函数内的所有跨越必须在 split / no-split 上取得一致。** 函数里所有 `pl.aiv_shard` /
`pl.aic_gather` 都跑在同一条跨核 pipe 上，而硬件在这条 pipe 的整个生命周期内把它固定为
split 或 no-split 之一。因此，一个会跨越边界的 `mode=NONE` 区域，不能与同样会跨越边界的
`UP_DOWN` 或 `LEFT_RIGHT` 区域并存：

```python
for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):
    a = pl.exp(pl.aiv_shard(mm0))         # 跨越，不切分
for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
    b = pl.exp(pl.aiv_shard(mm1))         # 跨越，切分       -> 被拒绝
```

两种**不同**的切分轴则没有问题：轴是逐次传输选择的，只有 split / no-split 属于 pipe：

```python
for r in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
    a = pl.exp(pl.aiv_shard(mm0))
for c in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
    b = pl.exp(pl.aiv_shard(mm1))         # 接受
```

**不含**任何跨越的区域可以使用任意模式 —— 只用来把 `pld.system.notify` 钉在向量 lane 上的
`mode=NONE` 区域根本不碰这条 pipe。当两个阶段确实需要不同的传输方式时，把它们放进各自的
`pl.at(level=pl.Level.CORE_GROUP)` scope：每个 scope 会成为独立的函数，从而各得一条 pipe。

### 每一个跨区域边界的 tile 都要写明

**只要函数开了区域，跨越区域边界的 tile 就必须写明。** 手动模式下两核之间的边界由你放置，
编译器不再替你选：

| 方向 | 写在哪里 | 算子 |
| ---- | -------- | ---- |
| cube 产出的值在向量 lane 上被读（C->V） | 区域开头 | `pl.aiv_shard(x)` |
| 向量产出的值在 cube lane 上被读（V->C） | 区域内、被读之前 | `pl.aic_gather(x)` |

```python
mm = pl.matmul(q, k)                        # cube，写在所有区域之外
for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
    v = pl.exp(pl.aiv_shard(mm))            # C->V：已写明
    kv = pl.aic_gather(v)                   # V->C：已写明
out = pl.matmul(kv, w)                      # 又回到 cube，写在区域之外
```

去掉其中任何一个调用，这次跨越**依然能工作** —— 编译器两种写法都会发出同样的传输 ——
但那样一来边界就成了没人选择的边界，而这恰恰与手动模式的意义相反。因此它会被拒绝；
诊断会指出是哪个值、被哪个算子读走。

**在 `mode=NONE` 区域里，两个算子都只跨越、不切分。** 没有可切分的轴，形状原样透传 ——
`[128, 128]` 的 tile 经 `pl.aiv_shard` 之后仍是 `[128, 128]`。只有在 `UP_DOWN` /
`LEFT_RIGHT` 下它们才同时折半（shard）与重新拼合（gather）。

**只能 gather 两条 AIV lane 取值一致的值。** 硬件要求两条子 lane 都参与 no-split 握手，
而它们共用同一个目标槽位、没有每 lane 偏移。两者之间没有任何仲裁：两条 lane 都会push，
因此当它们持有**不同**的值时，cube 收到的是二者之一，且**不确定是哪一个** —— 不是
lane 0 的，而是不确定的。

这里没有任何办法指定 lane。把该值的*产出*限定到某条 lane 也没有用：lane 1 依然会执行到
push，依然会把它自己 tile 中的内容发出去。因此从 `mode=NONE` 区域向外做 `pl.aic_gather`，
只有在该值是 lane-uniform（两条 lane 计算结果相同，或在 gather 之前已被统一）时才是良定义
的。若两条 lane 必须向 cube 提供不同的数据，这个构造无法表达：请改为经由 GM 传递并自行
定序，或使用数据并行（`UP_DOWN` / `LEFT_RIGHT`）区域 —— 那里每条 lane 拥有一个被声明的
半块，gather 会把它们重新拼合。

编译器不会检查上述任何一点。

**GM 上的数据流不在此规则覆盖范围内。** 上述规则针对的是跨区域边界的 *tile* 值。GM
tensor 不属于任何一条 lane，因此没有任何边界算子能表达经由它的跨越 —— `pld.tensor.put`
的签名本身就收 GM tensor。AIC 与 AIV 异步运行，所以为 cube lane 的写与 vector lane 对同一
块 GM 缓冲区的读定序，仍然由你负责。`ExpandMixedKernel` 只会自动处理一种窄特例：唯一的 cube
`tile.store` producer 与同 origin 的 vector `tile.load` 位于同一 body 或嵌套 body。其余 GM 交接——包括
V->C、通信算子和 sibling body——都需要你处理。`syncall` 本身只对齐到达：在 barrier 前发布 producer
的 cache line 并执行 GM fence，然后在 consumer 读之前使其 cache 失效。对于可能跨多条 cache line
的 buffer，请使用保守的全 GM `pl.system.cacheinvalid()` 形式；tensor-region overload 当前只覆盖
view 基地址所在的那一条 cache line。

### 把跨 rank 通信算子写进区域

区域也会为「自身没有 lane 归属」的算子决定放置。`pld.system.notify` 与核无关 ——
硬件上 TNOTIFY 两种核都能跑 —— 因此在 cube 与 vector 混合的 kernel 中，编译器会把它
同时发到 **AIC 与 AIV 两条** lane 上。把通信阶段包进区域，它就会被钉在向量 lane 上：

```python
for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):
    pld.tensor.put(dst=win, peer=peer, src=out,
                   dst_offsets=[0, 0], src_offsets=[0, 0], shape=[16, 256])
    pld.system.notify(target=sig, peer=peer, offsets=[0, 0], value=1,
                      op=pld.NotifyOp.AtomicAdd)
```

危险的是 cube 侧那份副本：AIC lane 可能在 AIV lane 的 put 尚未把数据落盘之前就执行到
notify，为尚不存在的数据发布信号。区域会把这份副本去掉。

### 把只应发生一次的副作用在两条 AIV lane 之间分片

**`mode=NONE` 区域的函数体会在两条 AIV 子 lane 上都运行。** 这正是该模式的用意 ——
区域不是「一条 lane」，而是两条 lane 跑同一段代码，由你用循环变量 `aiv_id` 分派互不
相交的工作。因此上面那段代码仍不完整：它每条 lane 各发一次 notify，即对同一个 peer
发了**两次**。

只应发生一次的副作用算子 —— 尤其是 `pld.system.notify` —— 必须**按 `aiv_id` 分片**，
或**限定到一条 lane**：

```python
# 分片：每条 lane 负责不同的 peer 集合
for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
    for owner in pl.range(aiv_id, NUM_PEERS, 2):
        pld.system.notify(target=sig, peer=owner, offsets=[0, 0], value=1,
                          op=pld.NotifyOp.AtomicAdd)

# 限定：lane 0 执行，lane 1 跳过
for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
    if aiv_id == 0:
        pld.system.notify(target=sig, peer=peer, offsets=[0, 0], value=1,
                          op=pld.NotifyOp.AtomicAdd)
```

**「限定到 lane 0」这种写法带有分片写法所没有的定序义务。** 两条 AIV lane 异步运行，
没有任何机制为 lane 0 与 lane 1 定序。因此被限定在 lane 0 的 notify 可能在 lane 1 的
写入落盘*之前*就发布信号，对端于是读到仍在传输途中的数据。只有当该信号所释放的数据
全部由 lane 0 自己写入时，这种写法才是安全的。若其中有任何一部分来自 lane 1，就必须
在 notify 之前显式地为两条 lane 定序，或者改用分片写法 —— 分片写法中每条 lane 只释放
自己写入的数据，因此不存在这个问题。

**不这样做会发生什么。** `NotifyOp.AtomicAdd` 会累加到对端的槽位上。两条 lane 通知
*同一个* peer，会让该 rank 的计数器在只有一个 rank 到达时就读到 `2`。原本要等两次
到达的 `pld.system.wait` 被一次就放行 —— 于是该 rank 抢跑，读到数据尚未落盘的缓冲区。
症状是多 rank 运行时某个 rank 间歇性算错；单 rank 复现不了，看上去也不像同步问题。

### 编译器在这里做什么、不做什么

| 行为 | 含义 |
| ---- | ---- |
| **会做** | 让区域内的通信算子远离 **cube** lane。写在区域外时，它们同样会被复制到 AIC lane 上。 |
| **不做** | 检查 lane 分片。两条 AIV lane 对同一个 peer 发 notify 能正常编译，且**不会有任何诊断**。 |

编译器无法诊断这一点：正确写法与错误写法在 AIV 函数中生成的是同一条语句，唯一的差别
是 `aiv_id` 有没有进入该调用的实参。写对这件事是作者的职责。

### 跨 lane 的定序同样由你负责

AIC 与 AIV **异步**运行。边界算子只为它所搬运的那一个值定序 —— 这正是这次传输的含义 ——
但没有任何东西能为 cube lane 的写与 vector lane 对同一块 **GM 缓冲区**的读定序。先发布 producer
的 cache line 并执行 GM fence，再在阶段之间放置跨核 barrier，最后在 consumer 读之前使其 cache
失效；barrier 本身只同步到达。区域只决定工作放在哪条 lane 上，并不为两条 lane 之间定序。

下面的保守序列使用全 GM cache 维护和 soft barrier，因此对跨多条 cache line 的 buffer 与部分占用都
安全。`sync_ws` 是独占且已清零的 16 元素 `INT32` GM tensor；`participant_count` 是参与同步的 AIC 与
AIV 核总数。

```python
pl.system.cacheinvalid()  # 发布 producer 的全部 cache line
pl.system.fence()         # 等待它们对 GM 可见
pl.system.syncall(
    mode=pl.SyncAllMode.SOFT,
    core_type=pl.KernelType.MIX,
    gm_workspace=sync_ws,
    used_cores=participant_count,
)                         # 只同步到达
pl.system.cacheinvalid()  # consumer 读之前使 cache 失效
```

如果不需要整体会合这么粗的粒度，`pl.system.sync_set` / `pl.system.sync_wait` 可以发起并等待单个跨核事件。
在**混合** InCore kernel 中，用 `core_type=pl.KernelType.AIC` 或 `core_type=pl.KernelType.AIV`
把每个事件操作钉在应当执行它的核通道上；在显式指定类型的 AIC 或 AIV kernel 中通道已经确定，省略该参数即可。

```python
pl.system.sync_set(0, pipe=pl.PipeType.MTE3, core_type=pl.KernelType.AIV)   # 在 AIV 上发起
pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)    # 在 AIC 上等待
```

## 边界情况

> **致命陷阱：** `pl.spmd` 是一个断言，不是请求。你是在告诉编译器这些 block 彼此独立。如果它们其实不独立，结果是竞态，而不是一条诊断。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **`Misplaced tensor op ... should be inside InCore block`** | 算子直接写在 `@pl.jit` 体内 | 包进 `with pl.at(level=pl.Level.CORE_GROUP):` |
| **`with pl.spmd(n):` 体被拒绝** | 它既不读 block 索引也不派发 kernel | 读 `pl.tile.get_block_idx()`，或调用一个 kernel |
| **大多数 `pl.write` 的写消失了，且每次运行消失的都不一样** | 并发实例写了同一条 64 字节 cache line 的不同元素 —— 抵达 DDR 的单位是 line，不是元素 | 让每个实例独占完整的 64 字节 line，或改由 `pl.spmd(1)` 来写；见 [内存](03-memory.md#来自并发任务实例的标量写) |
| **`optimizations=` 被拒绝** | 用变量拼出来的 —— 解析器读的是 AST | 在调用点内联书写该列表 |
| **printed IR 无法被重新解析** | 设备规模查询在使用前被绑定到了名字上 | 在使用处内联书写该调用 |
| **`vector op '...' sits outside every pl.split_aiv region`** | 函数开了区域，区域即拥有向量放置决定权 | 把该阶段包进 `for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):` |
| **`cube op '...' inside a pl.split_aiv region`** | 区域体是 AIV 的工作 | 把 `pl.matmul` 移出区域 |
| **`'x' is produced on the CUBE lane ... reads it on the VECTOR lane inside one`** | 未写明的 C->V 跨越（进入区域） | 在区域开头以 `pl.aiv_shard(x)` 读取 |
| **`'x' is defined inside a pl.split_aiv region but ... reads it on the CUBE lane outside`** | 未写明的 V->C 跨越（离开区域） | 在区域内 gather：`x = pl.aic_gather(x)` |
| **`'pl.aiv_shard' crosses the AIC/AIV boundary under ... but ... earlier in this function crosses it under ...`** | 同一函数的跨越混用了 `mode=NONE` 与切分模式，而它们共用一条跨核 pipe | 让所有跨越在 split / no-split 上一致，或去掉其中一个区域的跨越，或把两个阶段拆进各自的 `pl.at(level=pl.Level.CORE_GROUP)` scope |
| **cube 随机读到其中一条 lane 的值** | 从 `mode=NONE` 区域向外的 V->C 跨越 —— 两条 lane 都 push、共用一个槽位、无仲裁，**不会有诊断** | 只 gather lane-uniform 的值；若两条 lane 持有不同的半块，请改用数据并行区域 |
| **对端的信号计数器读到的值是应有值的两倍** | 两条 AIV lane 都执行了同一条 `pld.system.notify` —— **不会有诊断** | 按 `aiv_id` 分片该 notify，或用 `if aiv_id == 0:` 限定 |
| **某个 rank 的 `pld.system.wait` 返回后读到过期数据** | 要么是上面的重复 notify，要么是 cube 与 vector 阶段之间的 cache 发布 / fence / barrier / 失效序列不完整 | 分片该 notify；补全 GM 交接序列 |

## See Also

- [函数与程序](01-functions.md) —— `pl.at` 的替代写法：独立的 `@pl.jit.incore` 函数。
- [控制流](02-control-flow.md) —— 包含这些作用域的循环。
- [内存与数据搬运](03-memory.md) —— 被放置的代码拿缓冲区做什么。
- [任务与定序](../tasks/index.md) —— 被放置的工作相对其他任务什么时候跑。
- [OutlineIncoreScopes](../../dev/passes/08-outline_incore_scopes.md) —— `pl.at` 如何变成函数。
- [ExpandMixedKernel](../../dev/passes/22-expand_mixed_kernel.md) —— `pl.split` 驱动的是什么。
