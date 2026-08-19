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
| `AtomicType` | `None_`, `Add` | 远程存储合并模式。`None_`：普通存储。`Add`：原子累加——要求目标为 `fp32`/`bf16`/`fp16`/`int32`/`int16`/`int8`，其中 `bf16` 目标仅支持 Ascend910B（A2/A3）。 |
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
| `defer_wait` | `(signal, offsets, expected, *, cmp) -> Call` | **仅限专用的顶层 `pl.at(CORE_GROUP)` waiter。** 注册 `signal[offsets] >= expected` 完成条件并返回，不让 AIV 自旋；条件满足前，所在任务的 TaskId 保持未完成。 |

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

### 混合 cube+vector kernel 中的 `notify`

上面的内容都默认一条 notify 只执行一次。在把 `pl.matmul` 与通信算子混写的 kernel
中，这一点并不会自动成立，而且**不会有任何诊断**：

- **写在 `pl.split_aiv` 区域之外**时，notify 没有声明所属的核，编译器会把它同时发射到
  cube 通路*和* vector 通路上。把通信阶段放进区域，编译器就会让它们远离 cube 通路。
- **写在 `mode=NONE` 区域之内**时，区域体会在**两条 AIV sub-lane 上都运行**，因此除非
  你按 `aiv_id` 对它分片、或把它限定到某一条 lane，notify 仍然会触发两次。

这两条规则、它们所避免的错误，以及「限定到 lane 0」这种写法所附带的定序义务，都写在
[作用域 → pl.split_aiv](../language/04-scopes.md)。在把 notify 写到 cube 计算旁边之前，
请先阅读该文档。

### 选择 NotifyOp 和 WaitCmp

| 场景 | NotifyOp | WaitCmp | 原因 |
| ---- | -------- | ------- | ---- |
| 1:1 交换（每个槽一个写者） | `Set` | `Eq` 或 `Ge` | 不需要原子递增 |
| N-to-1 屏障（多个写者一个槽） | `AtomicAdd` | `Ge` | 每个写者原子累加，等待总量 |
| 多轮协议 | `AtomicAdd` | `Ge` | 计数跨轮推进 |

**2 个 rank 的预期输出：** rank 0 写入 tag=2，等待来自 rank 1 的 tag 1：
`outputs[0] == 1`。rank 1 写入 tag=1，等待来自 rank 0 的 tag 2：
`outputs[1] == 2`。结果：`outputs == [[1], [2]]`。

> **Buffer 重用安全：** Signal 使用单调计数器且不会自重置。这些 tile 级
> `notify`/`wait` 原语每次调用需分配新 buffer；`pld.tensor.*` 集合通信除外，
> 其 signal buffer 自清理，可在连续调用间复用。

## 延迟完成：释放物理核，保留逻辑任务

`pld.system.wait` 是阻塞式 `TWAIT`：AIV 留在 kernel 内，等待之后的语句继续在同一
AIV 上执行。`pld.system.defer_wait` 的契约不同。它向 runtime 注册计数器条件后即
返回；专用 waiter kernel 结束时，**物理 AIV 已释放**，但 waiter 的**逻辑 TaskId
仍未完成**。只有全部注册条件都满足后，scheduler 才会解析该 TaskId。原 kernel
不会恢复执行，因此 continuation 必须放在独立任务中。

```python
# 每个 rank 的 publisher 独立运行：先发布 payload，再发布 signal。
with pl.at(level=pl.Level.CORE_GROUP, name_hint="publish"):
    pld.tensor.remote_store(payload_value, peer_payload, peer, [0, 0])
    pld.system.notify(
        signal, peer=peer, offsets=[my_rank, 0],
        value=epoch, op=pld.NotifyOp.Set,
    )

# Receiver：观察 peer publisher。这里有意不添加本地 publisher -> waiter 依赖；
# 只有存在真实的本地顺序要求时才添加 deps。
with pl.at(
    level=pl.Level.CORE_GROUP,
    name_hint="payload_wait",
    allow_early_resolve=False,
) as wait_tid:
    pld.system.defer_wait(
        signal, offsets=[peer, 0], expected=epoch,
        cmp=pld.WaitCmp.Ge,
    )

# wait_tid 在逻辑上完成之后，consumer 才会被派发。
with pl.at(
    level=pl.Level.CORE_GROUP,
    name_hint="consume_payload",
    deps=[wait_tid],
) as consume_tid:
    payload_tile = pl.load(peer_payload, [0, 0], [1, WIDTH])
    # ... consume payload_tile ...
```

内联 SPMD consumer 同样使用捕获形式：

```python
with pl.spmd(
    NUM_BLOCKS,
    name_hint="consume_payload_spmd",
    deps=[wait_tid],
) as consume_tid:
    block = pl.get_block_idx()
    # ... each AIV block reads its payload partition ...
```

延迟完成不引入第二套依赖命名空间。`deps` 仍表示普通的严格 TaskId 依赖：waiter 的
AIV 结束后，Simpler 动态延迟这个普通 TaskId 的完成，因此已存在的依赖边要等注册的
counter 达标后才会释放 consumer。Simpler 的标准 AICore executor 会在取得每个任务后
立即失效整个 data cache；该操作位于可选的 speculative gate 之前，也位于任务 kernel
读取输入之前。waiter 不允许 early resolve，因此其直接 consumer 不会被预放到这个 gate，
而只会在 counter-backed TaskId 完成后通过普通路径被取得；该 consumer 的任务起始
invalidation 因而发生在 readiness 之后。producer 端仍必须在发布 notify
**之前**让所有 payload 写入可见；任务开始时的 cache invalidation 无法修复先 notify、
后写数据的错误。

### 延迟等待契约

- 把 `defer_wait` 放在专用的顶层 task
  `with pl.at(level=pl.Level.CORE_GROUP) as wait_tid:` 作用域内。该 task-level launch
  让 PyPTO 能验证 single-block 执行并提供 runtime `AsyncCtx`。未标记而直接从
  `@pl.jit.incore` / AIV 使用会因绕过这些契约而被拒绝；以编程方式构造的内部 IR 只有在
  PyPTO 重新验证完整 waiter body 与 orchestration call site 后才会被接受。
- Waiter 必须是 pure AIV，不能带 dispatch predicate，也不能使用
  `allow_early_resolve=True`。registration 之间可以执行纯标量 bookkeeping 与控制流，但
  registration 开始后不能继续做 `tensor.read`、payload/cache 操作或其他通信。这正是 Simpler 自身的 early-dispatch
  契约：waiter 保持 `False` 会使其直接 consumer 不具备在 counter-backed TaskId 完成前
  预派发的资格。普通 consumer 可以按需设置自己的 `allow_early_resolve`；该值控制的是
  consumer 的下游任务能否预派发，而不是让当前 consumer 绕过 waiter。
- `signal` 必须是直接作为参数传入、绑定 window 的 INT32 `DistributedTensor`；
  v1 不支持 slice、view 或其他 alias，且仅接受 `cmp=pld.WaitCmp.Ge`。
- 条件在 INT32 signal storage 上按单调 uint32 轮询（`counter >= expected`）。
  `expected` 和每一个发布的 counter 值都必须保持非负并位于 `[0, INT32_MAX]`；例如写入
  `-1` 会被观察成 `UINT32_MAX`，从而错误满足所有合法 threshold。动态 expected 值在
  运行时检查。旧 generation 仍可能 pending 时，不要重置 counter 或让其后退，也不要
  依赖 uint32 回绕。
- 单个 waiter task 最多注册 64 个条件。另有一个独立限制：每个 runtime scheduler
  最多同时跟踪 64 个延迟任务。这两个 64 含义不同，且都不是物理核数。
- continuation 使用与任意 TaskId 相同的普通 `deps=[..., wait_tid]` 连接。延迟完成不要求
  另一种 consumer kernel 类型或依赖表示。没有 continuation 的末端 waiter 可以在同一
  task 作用域中 fire-and-forget，无需捕获 TaskId。
- 如果 producer 永远没有把 counter 推进到 `expected`，AIV 虽不再自旋，TaskId 及
  所有依赖它的 consumer 仍会一直 pending。协议仍必须保证 notify 最终到达。

该机制不是异步预取：`pl.prefetch.*` 管理的是 SDMA 数据搬运 session/event，而延迟
完成机制是在远程 counter 上门控 scheduler TaskId。它也不是 host 异步执行——没有
Python future、host callback 或等待信号的 host thread。需要在同一个 kernel 中从
等待点继续执行的代码仍使用原有 `pld.system.wait`；其行为不变，并继续支持 `Eq` 和
`Ge`。

## Tile 级 RMA (`pld.tile.*`)

| 名称 | 签名 | 描述 |
| ---- | ---- | ---- |
| `remote_load` | `(target, peer, offsets, shape, valid_shape=None) -> Tile` | 加载对端区域到本地 tile。`shape` 定义 tile 维度。`valid_shape` 可在物理 tile 保持固定大小的同时，让参差不齐的尾部只读取真实数据。Offsets 必须与对端写入时使用的一致——偏移 1 个元素就会导致静默数据损坏。 |
| `remote_store` | `(src_tile, target, peer, offsets, *, atomic=AtomicType.None_) -> Call` | 写入本地 tile 到对端。`atomic=Add` 表示累加到对端区域而非覆写。被写入的区域必须在 `offsets` 处落在 `target` 内部。 |

`remote_store` 还有一个上移一层的形式 **`pld.tensor.remote_store`**
`(src: Tensor, target: DT, peer: IntLike, offsets, *, atomic=...)`，用于在 tensor 级
`@pl.jit` kernel 中推送*计算值*（那里没有 tile 可命名）。它 1:1 下降为 tile 形式，
因此该值以单次远程写抵达对端，中间不经过全局内存往返。短形式 `pld.remote_store`
会按传入的操作数在两者之间分派。

## Put 和 Get (`pld.tensor.*`)

单边批量传输——rank A 写入或读取 rank B 的 window，rank B 无需参与传输
（除了 signal 屏障）。

### Put（写入对端）

| 名称 | 签名 | 变更 | 描述 |
| ---- | ---- | ---- | ---- |
| `put` | `(dst: DT, peer: IntLike, src: DT \| Tensor, dst_offsets=None, src_offsets=None, shape=None, *, atomic=AtomicType.None_, chunk_rows=0, chunk_cols=0, pipeline=False) -> Call` | `dst: InOut`，`src: In` | 将本地 `src` 写入对端 rank 的 `dst`。`dst` **必须**为 window-bound；`src` 可以是普通 `Tensor`。未指定 offsets/shape 时，写入完整的本地分片。`atomic=Add` 时累加而非覆盖（仅限硬件原子加 dtype：`fp32`/`bf16`/`fp16`/`int32`/`int16`/`int8`；`bf16` 仅支持 Ascend910B）。 |

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
| `pld.remote_store(...)` | `pld.tile.remote_store(...)` / `pld.tensor.remote_store(...)`（按 `src` 分派） |

**无短格式：** `pld.notify(...)`、`pld.wait(...)`、`pld.defer_wait(...)`、`pld.put(...)`、
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
