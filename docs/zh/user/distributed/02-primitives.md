# 原语

大多数用户应直接调用 `pld.tensor.*` 集合通信——只有在构建自定义协议时
才需要这些更底层的原语。

> **说明：** 下面 notify/wait、put/get 以及 remote-load/store 的代码块均为
> 示意性片段——省略了 `nranks`/`my_rank` 推导和 buffer 设置，并非可直接
> 运行的程序。可运行版本见下方"可运行示例"一节。

## 类型与枚举

| 名称 | 取值 | 描述 |
| ---- | ---- | ---- |
| `NotifyOp` | `AtomicAdd`, `Set` | 信号投递模式。`AtomicAdd`：原子递增对端信号槽（多 rank 屏障）。`Set`：覆盖对端信号槽（1:1 握手）。 |
| `WaitCmp` | `Eq`, `Ge` | 等待谓词。`Eq`：等于时解除阻塞。`Ge`：大于等于时解除阻塞。 |
| `ReduceOp` | `Sum`, `Max`, `Min`, `Prod` | 集合通信的规约算子。支持情况按操作而定：`allreduce` 支持全部四种；`reduce_scatter` 仅支持 `Sum`，其余在 deducer 阶段被拒绝。 |
| `AtomicType` | `None_`, `Add` | 远程存储合并模式。`None_`：普通存储。`Add`：原子累加。 |
| `DistributedTensor` | — | 绑定到通信域 window buffer 的 tensor 视图。 |
| `CommCtx` | — | 通信上下文句柄。 |

## 系统基础设施 (`pld.system.*`)

| 名称 | 签名 | 描述 |
| ---- | ---- | ---- |
| `world_size` | `() -> Scalar` | **仅限 host。** 分布式执行中的 rank 数量。 |
| `get_comm_ctx` | `(dist_tensor: DT) -> Ctx` | 提升为 `CommCtx` 句柄。host 编排器和 InCore 代码均可用。 |
| `rank` | `(ctx: Ctx) -> Scalar` | **仅限 InCore。** 本地 rank 索引。 |
| `nranks` | `(ctx: Ctx) -> Scalar` | **仅限 InCore。** 通信组中 rank 数量。 |
| `notify` | `(target, peer, offsets, value, *, op) -> Call` | **仅限 InCore。** 跨 rank 信号投递。仅副作用。 |
| `wait` | `(signal, offsets, expected, *, cmp) -> Call` | **仅限 InCore。** 跨 rank 等待。仅副作用。 |

## Window Buffer 管理 (`pld.tensor.*`)

`window` 和 `alloc_window_buffer` 属于 `pld.tensor.*`，而非 `pld.system.*`，
尽管它们和上面的基础设施同样底层。

| 名称 | 签名 | 描述 |
| ---- | ---- | ---- |
| `window` | `(buf: Ptr, shape, *, dtype) -> DT` | 物化为 `DistributedTensor` 视图，`buf` 来自 `alloc_window_buffer`。 |
| `alloc_window_buffer` | `(size, *, name="") -> Ptr` | 分配每 rank 一份的 HCCL window buffer。**size 以字节为单位。** `name` 由解析器从赋值左侧注入——不要手动传入。 |
| `alloc_window_buffer` | `(shape, *, dtype, name="") -> Ptr` | 便捷重载：自动计算 `size = prod(shape) x dtype.get_byte()`。 |

## Notify & Wait：信号握手

最底层的同步原语。每个 rank 写入对端信号槽，然后阻塞直到自己的槽被写入。

```python
@pl.jit.incore
def handshake_step(
    out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
    signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
    peer: pl.Scalar[pl.INT32],
    tag: pl.Scalar[pl.INT32],
) -> pl.Tensor[[1, 1], pl.INT32]:
    pld.system.notify(
        signal, peer=peer, offsets=[0, 0],
        value=tag, op=pld.NotifyOp.Set,
    )
    pld.system.wait(
        signal=signal, offsets=[0, 0],
        expected=1, cmp=pld.WaitCmp.Ge,
    )
    received = pl.read(signal, [0, 0])
    pl.write(out, [0, 0], received)
    return out
```

> `wait` 使用 `Ge` 且 `expected=1`，对端的 `tag` **必须 >= 1**。传入 `tag=0`
> 会导致永久挂起。

### 选择 NotifyOp 和 WaitCmp

| 场景 | NotifyOp | WaitCmp | 原因 |
| ---- | -------- | ------- | ---- |
| 1:1 交换（每个槽一个写者） | `Set` | `Eq` 或 `Ge` | 不需要原子递增 |
| N-to-1 屏障（多个写者一个槽） | `AtomicAdd` | `Ge` | 每个写者原子累加，等待总量 |
| 多轮协议 | `AtomicAdd` | `Ge` | 计数跨轮推进 |

**2 个 rank 的预期输出：** rank 0 写入 tag=2，等待来自 rank 1 的 tag 1：
`outputs[0] == 1`。rank 1 写入 tag=1，等待来自 rank 0 的 tag 2：
`outputs[1] == 2`。结果：`outputs == [[1], [2]]`。

> **Buffer 重用安全：** Signal 使用单调计数器且不会自重置。不要在背靠背集合通信中
> 重用同一 signal buffer。每次调用分配新 buffer。

## Tile 级 RMA (`pld.tile.*`)

| 名称 | 签名 | 描述 |
| ---- | ---- | ---- |
| `remote_load` | `(target, peer, offsets, shape, valid_shape=None) -> Tile` | 加载对端区域到本地 tile。`shape` 定义 tile 维度。`valid_shape` 可在物理 tile 保持固定大小的同时，让参差不齐的尾部只读取真实数据。Offsets 必须与对端写入时使用的一致——偏移 1 个元素就会导致静默数据损坏。 |
| `remote_store` | `(src_tile, target, peer, offsets) -> Call` | 写入本地 tile 到对端。 |

## Put 和 Get (`pld.tensor.*`)

单边批量传输——rank A 写入或读取 rank B 的 window，rank B 无需参与传输
（除了 signal 屏障）。

### Put（写入对端）

| 名称 | 签名 | 变更 | 描述 |
| ---- | ---- | ---- | ---- |
| `put` | `(dst: DT, peer: IntLike, src: DT \| Tensor, dst_offsets=None, src_offsets=None, shape=None, *, atomic=AtomicType.None_, chunk_rows=0, chunk_cols=0, pipeline=False) -> Call` | `dst: InOut`，`src: In` | 将本地 `src` 写入对端 rank 的 `dst`。`dst` **必须**为 window-bound；`src` 可以是普通 `Tensor`。未指定 offsets/shape 时，写入完整的本地分片。`atomic=Add` 时累加而非覆盖。 |

```python
# dst 必须为 window-bound
pld.tensor.put(dst, peer=1, src=local_chunk, atomic=pld.AtomicType.Add)
```

### Get（从对端读取）

| 名称 | 签名 | 变更 | 描述 |
| ---- | ---- | ---- | ---- |
| `get` | `(dst: DT \| Tensor, peer: IntLike, src: DT, dst_offsets=None, src_offsets=None, shape=None, *, chunk_rows=0, chunk_cols=0, pipeline=False) -> Call` | `dst: Out`，`src: In` | 读取对端 rank 的 `src` 到本地 `dst`。`src` **必须**为 window-bound；`dst` 可以是普通 `Tensor`。 |

```python
# src 必须为 window-bound
pld.tensor.get(dst, peer=1, src=peer_data)
```

### 分块与流水线约束

`chunk_rows`/`chunk_cols`（`0` 表示完整范围）会缩小 staging tile，让超过
片上 staging 预算的传输仍能一次调用完成，自动滑动通过较小的 stage。

> **致命陷阱：** `pipeline=True` **要求 `chunk_rows > 0` 且 `chunk_cols > 0`
> 同时成立**——双缓冲的收益只有在传输确实被分块时才存在。若 `pipeline=True`
> 而任一 chunk 维度仍为 `0`，会在派发前抛出 `ValueError`。

**动态**传输范围（运行时确定的 `shape`，或 `dst`/`src` 自身维度为动态的
整片传输）必须由匹配的静态 chunk 限定：动态的最内层维度要求设置
`chunk_cols`，动态的最外层维度要求设置 `chunk_rows`——staging tile 是静态
分配的，无法按运行时值确定大小。

## 编写自己的集合通信

每个内置集合通信都是底层原语的组合。mesh allreduce 的模式为：
stage-in → barrier → remote-accumulate → stage-out。

### 隔离的 Barrier

```python
for peer in pl.range(nranks):
    if peer != my_rank:
        pld.system.notify(
            signal, peer=peer, offsets=[my_rank, 0],
            value=1, op=pld.NotifyOp.AtomicAdd,
        )
for src in pl.range(nranks):
    if src != my_rank:
        pld.system.wait(
            signal, offsets=[src, 0],
            expected=1, cmp=pld.WaitCmp.Ge,
        )
```

使用 `offsets=[my_rank, 0]` 时，每个 rank 拥有专属的一行——每个对端
window 中的 `[r, 0]` 格子只有一个写者，即 rank `r` 自己，因此这里用
`Set` 效果完全相同。之所以展示 `AtomicAdd`，是因为本文档中每个 barrier
使用的都是同一个 notify 调用；真正需要 `AtomicAdd` 的"多写者、一个槽位"
场景是*共享格子*的 barrier（见上表）——只有在需要区分*哪些*对端已经到达，
而不仅仅是*是否*所有对端都已到达时，才像这里一样给每个 rank 分配不同的
offset。

### 远程累加

```python
acc = pl.load(data, [0, 0], [1, SIZE])
for peer in pl.range(nranks):
    if peer != my_rank:
        peer_tile = pld.tile.remote_load(
            data, peer=peer, offsets=[0, 0], shape=[1, SIZE]
        )
        acc = pl.add(acc, peer_tile)
```

## 2 段 vs 3 段命名空间

| 短格式 (`pld.*`) | 完整路径 |
| ---------------- | -------- |
| `pld.world_size()` | `pld.system.world_size()` |
| `pld.rank(ctx)` | `pld.system.rank(ctx)` |
| `pld.nranks(ctx)` | `pld.system.nranks(ctx)` |
| `pld.get_comm_ctx(dt)` | `pld.system.get_comm_ctx(dt)` |
| `pld.alloc_window_buffer(...)` | `pld.tensor.alloc_window_buffer(...)` |
| `pld.window(...)` | `pld.tensor.window(...)` |
| `pld.remote_load(...)` | `pld.tile.remote_load(...)` |
| `pld.remote_store(...)` | `pld.tile.remote_store(...)` |

**无短格式：** `pld.notify(...)`、`pld.wait(...)`、`pld.put(...)`、
`pld.get(...)`、`pld.allreduce(...)` 等——这些需要完整的 3 段命名空间。

## 可运行示例

[教程](05-tutorials.md)在揭示任何内置原语之前，先手工教授每个原语
（步骤 03–16 全部已交付）：

| 原语 | 教程步骤 |
| ---- | -------- |
| window buffer | [08-window_buffer](08-window_buffer.md)（步骤 03） |
| notify / wait | [09-barrier](09-barrier.md)（步骤 04） |
| remote_load / remote_store | [10-remote_load_store](10-remote_load_store.md)（步骤 05） |
| put / get | [11-put_get](11-put_get.md)（步骤 06） |

| 原语 | 测试 |
| ---- | ---- |
| notify / wait | `test_l3_notify_wait.py` |
| put / get | `test_l3_put.py` / `test_l3_get.py` |
| remote_store | `test_l3_remote_store.py` |

（路径均相对于 `tests/st/distributed/`）

## 相关链接

- [01-collectives](01-collectives.md) — 基于这些原语构建的集合通信
- [03-execution](03-execution.md) — DistributedWorker 生命周期
- [04-debugging](04-debugging.md) — 常见故障模式
