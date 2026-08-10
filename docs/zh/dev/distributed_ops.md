# 分布式算子（Distributed Operators，N6）

## 概述

N6 分布式算子族为 Python DSL 提供了对硬件跨 rank（cross-rank）通信原语的直接、带类型的访问。族内每个算子都作用于一个**窗口绑定的（window-bound）**
[`DistributedTensorType`](ir/02-types.md) —— 其存储是 `pld.alloc_window_buffer`
分配的对称、按 rank 划分的通信窗口的一个切片。族内 verifier 通常会拒绝普通
`TensorType`（严格的 kind-trait 匹配 —— `As<DistributedTensorType>` 不匹配普通
`TensorType`），以保证非窗口绑定的 tensor 永远不会被误传入跨 rank 槽位。
**两个明确的例外：** `pld.tensor.put`（以及它下降出的 `pld.tile.put`）的
`src` 参数通过 `AsTensorTypeLike` 接受普通 `Tensor` —— TPUT 在源端只需要一段
可读的本地 GM 区域,因此 kernel 可以直接从 host 输入推送,不必先经过窗口缓冲
中转；`dst` 仍然必须是窗口绑定的 `DistributedTensor`。
`pld.tensor.get`（以及它下降出的 `pld.tile.get`）的 `dst` 参数通过
`AsTensorTypeLike` 接受普通 `Tensor` —— TGET 在目标端只需要一段可写的本地 GM
区域来接收数据,因此 kernel 可以将 TGET 结果直接写入 host 输出 tensor；
`src` 仍然必须是窗口绑定的 `DistributedTensor`。

共有**十三个算子**和**四个 ABI 枚举**：

| 算子 | 方向 | 结果 | 硬件 |
| ---- | ---- | ---- | ---- |
| `pld.tile.remote_load` | pull（读 peer → 本地 tile） | `TileType` | TLOAD |
| `pld.tile.remote_store` | push（写本地 tile → peer） | `Unknown`（副作用） | TSTORE |
| `pld.tensor.get` | pull（读 peer → 本地 GM） | `Unknown`（副作用） | TGET |
| `pld.tensor.put` | push（写本地 → peer） | `Unknown`（副作用） | TPUT |
| `pld.tensor.allreduce` | collective reduce over window slices | `DistributedTensorType`（同 src） | builtin collective |
| `pld.tensor.barrier` | 跨 rank 同步窗口数据可见性 | `DistributedTensorType`（同 src） | builtin collective |
| `pld.tensor.broadcast` | 将 root rank 的数据复制到所有 rank | `DistributedTensorType`（同 src） | builtin collective |
| `pld.tensor.reduce_scatter` | 跨 rank 规约并分散 | `DistributedTensorType`（同 src） | builtin collective |
| `pld.tensor.allgather` | 从所有 rank 收集数据到窗口 | `DistributedTensorType`（同 src） | builtin collective |
| `pld.tensor.all_to_all` | 基于推送的对称个性化交换——每个 rank 通过 `pld.tensor.put`（TPUT）将自己的各目标 block 推送到每个对等方的窗口中，返回窗口作为结果 | `DistributedTensorType`（同 src） | composite / HOST builtin |
| `pld.tensor.all_to_all_v` | 变长 all-to-all（MPI_Alltoallv）——按每个目标推送完整的 MAX_RECV 行容量块，写入平面 2D 暂存窗口（传输大小是每个目标完整的容量块），同时通过 `pld.system.notify`（Set）把 `min(send_counts[dest], MAX_RECV)` 发布到对端 `recv_counts[my_rank, 0]`，使接收方能跳过超出其计数的行；返回窗口作为结果（与对称 `all_to_all` 相同的窗口即结果模式） | `DistributedTensorType`（与 target 相同） | composite / HOST builtin |
| `pld.system.notify` | 给 peer 的槽位发信号 | `Unknown`（副作用） | TNOTIFY |
| `pld.system.wait` | 在自身槽位上阻塞 | `Unknown`（副作用） | TWAIT |

五个仅有副作用（side-effect-only）的算子产生
[`UnknownType`](ir/02-types.md)：它们因跨 rank 副作用而存在，而非为消费者读取的
SSA 值而存在。

## 命名空间：为何区分 `tile.*` / `tensor.*` / `system.*`

命名空间编码的是算子所在的 IR 层级，而非随意分组：

- **`pld.tile.remote_load`** 产生一个 *tile*（片上 SRAM 区域），因此是 `tile.load`
  的兄弟,归入 `pld.tile`。
- **`pld.tile.remote_store`** 消费一个 *tile*（`remote_load` 的对称写伴生算子），
  因此是 `tile.store` 的兄弟,同样归入 `pld.tile`。
- **`pld.tensor.get`** 读写 *tensor*（GM）操作数 —— `dst` 可以是窗口绑定的
  `DistributedTensor` 视图,**也可以**是普通 `Tensor`（TGET 在目标端只需要一段
  可写的本地 GM 区域来接收数据）；`src` 必须是窗口绑定的 `DistributedTensor`
  视图（peer 需要窗口槽位用于读取）。TGET 中转用的 VEC staging tile 由
  `ConvertTensorToTileOps` 物化为内部 `pld.tile.get`,不出现在 DSL 表面。
  因此它是 `pld.tensor.alloc_window_buffer` / `pld.tensor.window` 的兄弟,
  而**不是**产出 tile 的 `remote_load` 的兄弟。
- **`pld.tensor.put`** 读写 *tensor*（GM）操作数 —— `dst` 必须是窗口绑定的
  `DistributedTensor` 视图（peer 需要窗口槽位用于接收）；`src` 可以是窗口绑定的
  `DistributedTensor` 视图,**也可以**是普通 `Tensor`（TPUT 在源端只需要一段可
  读的本地 GM 区域）。TPUT 中转用的 VEC staging tile 由
  `ConvertTensorToTileOps` 物化为内部 `pld.tile.put`,不出现在 DSL 表面。
  因此它是 `pld.tensor.alloc_window_buffer` / `pld.tensor.window` 的兄弟,
  而**不是**产出 tile 的 `remote_load` 的兄弟。
- **`pld.system.notify` / `pld.system.wait`** 驱动按 rank 的信号槽位 —— 纯控制面
  同步,无数据操作数 —— 因此归入 `pld.system`。

## ABI 枚举（`include/pypto/ir/comm.h`）

四个枚举是**仅追加（append-only）的 ABI**。它们的底层 `int` 值被序列化为算子的
kwarg 负载（notify 的 `op`、wait 的 `cmp`、put 的 `atomic`）,并在 codegen 时转
回枚举。新变体只能加在**末尾**,以保证已有 IR 和缓存程序的语义不变。

```cpp
enum class NotifyOp : int { kAtomicAdd = 0, kSet = 1 };   // pld.system.notify
enum class WaitCmp  : int { kEq = 0,        kGe = 1 };     // pld.system.wait
enum class AtomicType : int { kNone = 0,    kAdd = 1 };    // pld.tensor.put
enum class ReduceOp : int { kSum = 0, kMax = 1, kMin = 2, kProd = 3 };  // pld.tensor.allreduce
```

| 枚举 | 变体 | 含义 |
| ---- | ---- | ---- |
| `NotifyOp` | `kAtomicAdd` | 原子地把 `value` 加到 peer 的信号槽位 |
| `NotifyOp` | `kSet` | 非原子地把 `value` 存入 peer 的信号槽位 |
| `WaitCmp` | `kEq` | 阻塞直到 `*signal_slot == expected` |
| `WaitCmp` | `kGe` | 阻塞直到 `*signal_slot >= expected` |
| `AtomicType` | `kNone` | 普通远程写 —— 覆盖 peer 的 dst 切片 |
| `AtomicType` | `kAdd` | 原子地把源数据加到 peer 的 dst 切片 |
| `ReduceOp` | `kSum` | 对所有参与 rank 的窗口切片做求和规约 |
| `ReduceOp` | `kMax` | 对所有参与 rank 的窗口切片做最大值规约 |
| `ReduceOp` | `kMin` | 对所有参与 rank 的窗口切片做最小值规约 |
| `ReduceOp` | `kProd` | 对所有参与 rank 的窗口切片做乘法规约 |

每个枚举跨三层保持一致（C++ `enum class` → bindings 中的 `nb::enum_` → `.pyi`
存根）,并以 `pld.NotifyOp` / `pld.WaitCmp` / `pld.AtomicType` / `pld.ReduceOp` 暴露给 DSL。
deducer 会校验打包的 `int` 落在枚举范围内,使 codegen 无需二次保护即可转回。

## 屏障-信号协议

每个 `pld.tensor.*` 集合通信算子（`allreduce`、`barrier`、`broadcast`、
`reduce_scatter`、`allgather`、`all_to_all`）都使用同一个**自清理信用屏障**
（self-clearing credit barrier）进行同步，该屏障由 `pld.system.notify` /
`pld.system.wait` 构建：

```text
Body:      barrier(1); barrier(2); ...; barrier(N)   # g 在本次调用内计数
                                                      # （仅本次调用）
  barrier(g):
    for peer != my_rank: notify(signal, peer, <my cell>, 1, op=AtomicAdd)
    for src  != my_rank: wait  (signal, <src cell>, g,   cmp=Ge)

Epilogue:  for src != my_rank:
               notify(signal, my_rank, <src cell>, -N, op=AtomicAdd)
```

`AtomicAdd` 把每个 cell 变成一个信用计数器：每次 notify 是生产者的 `+1`，尾声
（epilogue）是唯一消费者的 `-N`。由于加法与减法是原子的且可交换，一旦所有 rank
都完成本次调用的尾声，signal 可证明地恢复为全零 —— **signal 不携带任何超出
单次调用生命周期的状态**，因此每次调用的 generation `g` 都从 1 重新开始，无需
跨调用记账。慢 rank 在完成当前调用的过程中最多会让快 rank 自己的下次调用
信用膨胀 1（有界 skew），因此计数器不会溢出，快 rank 也永远不会观察到虚假
通过。

`Ge`（而非 `Eq`）是关键负载：快 peer 可能在慢 rank 轮询前就把 cell 推到期望值
之上，因此相等等待会永久阻塞。同理，`Set` 绝对不能与 `AtomicAdd` 混用在同一个
cell 上 —— set 可能会覆盖已经被推高的计数器。

`N`（尾声要减去的信用总数）可以是**运行时常量** —— `pld.system.notify` 的
`value` 只需要 `ScalarType` —— 因此 mesh allreduce 的每块信用计数不需要在编译期
已知。

**约束：**

| 约束 | 原因 |
| ---- | ---- |
| 同一个 signal 不能在 mesh（`[NR, 1]`）和 ring（`[2*(NR-1), NR]`）allreduce 之间共享 | mesh 寻址 `[rank, 0]`；ring 寻址 `[row, rank]` —— 形状不匹配，在降级时检查 |
| 调用在中途被中止（错误/超时）会留下 signal 为非零 | 信用泄漏；在下一次 dispatch 之前通过 host 端重置（`reset_persistent_windows`）恢复 |

由于协议是调用局部的，且 signal 在每次调用开始时始终为零，集合通信在 `for` /
`while` / `if` 内都是合法的 —— 每次调用都是封闭循环，因此相同的编译期
`expected` 值在每次迭代中复用。唯一的剩余要求是 rank 均匀执行（任何屏障的固有
要求）：rank 分叉的控制流会死锁，由 `TWAIT` 的自旋计数断言暴露。

## 算子参考

### `pld.tile.remote_load`（TLOAD）

```text
pld.tile.remote_load(target, peer, offsets, shape[, valid_shape])
    -> TileType(shape, target.dtype)
```

把 `peer` rank 的窗口绑定 `DistributedTensor` 切片中的一个区域读入本地 tile。
在 IR 层面镜像 `tile.load`（位置参数 `offsets` / `shape` 元组、`TileType` 结果）,
但源是*远程*切片 —— 地址转换在 codegen 时由
`CommRemoteOffset(ctx, peer) + addptr + make_tensor_view` 实现。

`valid_shape` 可选。无论是否传入，类型推导都会将请求窗口与源 tensor 的实际有效
区域取交集，并检查可证明的物理边界。传入时，`shape` 仍决定 UB tile 的物理分配
大小，`valid_shape` 进一步限制远程 partition 和 tile 的有效范围。分块集合通信
用这种形式表达固定宽度的非对齐尾块。

任何在推导后仍为符号表达式的源有效范围或请求有效范围，都必须在 kernel 中通过
标量参数、循环变量或物理 Tensor shape 参数获得运行时绑定；仅出现在类型元数据
中的符号会在 PTO codegen 阶段被拒绝。

Verifier：`target` 必须是 `DistributedTensorType`；`peer` 必须是 `ScalarType`
rank 索引；`offsets` / `shape` / 可选 `valid_shape` 必须各为 `MakeTuple`,
其 rank 等于 `target.shape.size()`。

DSL（`python/pypto/language/distributed/op/tile_ops.py`）接受位置或关键字参数；
IR 算子保持位置参数，与 `tile.load` 一致。

### `pld.tile.remote_store`（TSTORE）

```text
pld.tile.remote_store(src_tile, target, peer, offsets) -> Unknown
```

把本地 tile 写入 `peer` rank 的窗口绑定 `DistributedTensor` 切片中的一个区域。
在 IR 层面镜像 `tile.store`（位置参数 `offsets` 元组、仅副作用返回值），但目的是
*远程*切片 —— 地址转换在 codegen 时由
`CommRemoteOffset(ctx, peer) + addptr + make_tensor_view` 实现。

Verifier：`src_tile` 必须是 `TileType`；`target` 必须是 `DistributedTensorType`；
`peer` 必须是 `ScalarType` rank 索引；`offsets` 必须是 `MakeTuple`,其 rank 等于
`target.shape.size()`；`src_tile.dtype` 必须等于 `target.dtype`。

Codegen：经过标准 tile pipeline 之后 tile 是 2-D（height × width）；发出的
`pto.partition_view` 与 `target` 同 rank，前 `(target.rank - 2)` 维都填 1（与
`notify` 的 `one_dims(rank, "1")` 模式一致）。这样无论 target 是几维（N ≥ 2），
2-D 的 tile push 都能落到 peer 切片的内两维上，调用方无需自行 reshape —— 这也
是用来抓住之前 codegen 对任意 rank 都按 2-D 发 `partition_view` 的隐藏 bug
的回归保护。

DSL（`python/pypto/language/distributed/op/tile_ops.py`）把 `target` / `peer` /
`offsets` 暴露为仅关键字（keyword-only）参数以提升可读性；IR 算子保持位置参数,
与 `tile.store` 一致。

### `pld.tensor.put`（TPUT）

```text
pld.tensor.put(dst, peer, src, *, atomic: int,
               chunk_rows: int = 0, chunk_cols: int = 0, pipeline: bool = False) -> Unknown
pld.tensor.put(dst, peer, src, dst_offsets, src_offsets, shape,
               *, atomic: int, chunk_rows: int = 0, chunk_cols: int = 0, pipeline: bool = False) -> Unknown
```

同步地把本地 `src` 数据写入 `peer` rank 的窗口绑定 `dst` 切片。`dst` 是 GM
层级的 `DistributedTensor` 视图；`src` 可以是 `DistributedTensor` 视图,**也
可以**是普通 `Tensor` —— TPUT 在源端只需要一段可读的本地 GM 区域,因此 kernel
可以直接从 host 输入推送,不必先经过窗口缓冲中转。VEC staging tile 由
`ConvertTensorToTileOps` 物化为内部 `tile.create + pld.tile.put`,因此会经过
PyPTO 的内存分配器,但不出现在 DSL 表面。

不提供 offsets/shape 时,该操作把完整的本地 `src` 切片写入完整的 peer `dst`
切片。提供 `dst_offsets`、`src_offsets` 和 `shape` 时,传输会缩小到匹配的
subregion；三者必须一起提供。

**staging tile 分块。** 默认 staging tile 覆盖整个展平后的传输 `[rows, cols]`
范围（`rows` = 前导维之积,`cols` = 最内维），因此一次传输必须放得进 UB。可选的
`chunk_rows` / `chunk_cols`（`0` = 全量）把 staging tile 缩成该范围的子块；codegen
仍让 `pto.comm.tput` 的 partition view 保持**完整**传输范围,由 pto-isa TPUT 在
更小的 stage 上做 2D 滑窗。这样单个 `put` 即可搬运大于 UB 的数据,无需调用方手写
分块循环。超出范围的 chunk 值会被钳到传输范围内。

**双缓冲（`pipeline`）。** 设置 `pipeline=True` 时,
`ConvertTensorToTileOps` 会物化**两个**完全相同的 VEC staging tile
（`tput_stage_ping` / `tput_stage_pong`）并作为第二个 `stage` 操作数一起传给
`pld.tile.put`。codegen 随后发出 ping-pong 形式
`pto.comm.tput(dst_pv, src_pv, buf(%ping, %pong) : …)`,PTOAS 将其路由到 pto-isa
的双缓冲 `TPUT` 重载 —— 它跨两个 tile 把下一个 chunk 的 TLOAD 与上一个 chunk 的
TSTORE 重叠流水。由于只有传输被切成多个 chunk 时双缓冲才有收益,`pipeline`
**要求同时设置 `chunk_rows` 与 `chunk_cols`**（deducer 与 DSL 都会拒绝缺少完整
chunk 的 `pipeline`）。两个 tile 是各自独立的 `tile.create` 分配,内存分配器会给
它们不重叠的 UB 地址（满足 pto-isa 对 ping/pong 的要求）。

**动态传输范围。** 传输范围可以是**动态**的 —— 既可以是 subregion 的 `shape`
（窗口内一段运行时子范围）,也可以是 full-slice 时 `dst` / `src` 窗口
（`DistributedTensorType`）本身的维度。pto-isa 在运行时从 partition view 读取
范围,因此 codegen 发出动态 partition view（`<?x…>`）并对其分块。动态的展平维
必须由对应的静态 chunk 约束,因为 VEC staging tile 是静态分配的:动态最内维需要
`chunk_cols`,动态前导维需要 `chunk_rows`。full-slice 时 `dst` 与 `src` 的维度
必须一致 —— 静态维按值比较,动态维按结构（structural）比较。

Verifier：`dst` 必须是 `DistributedTensorType`；`src` 必须是 `TensorType` 或
`DistributedTensorType`（通过 `AsTensorTypeLike` 匹配）；`peer` 必须是
`ScalarType`；`dst` 与 `src` 必须 element type 相同、rank 相同,且各维都是
**正（positive）**维度（正性仅对静态维校验；动态维允许,由 chunk 约束）。
full-slice `put` 要求 `dst` / `src` 形状一致；subregion `put` 允许完整切片尺寸
不同,只要显式传输区域不越界（仅校验静态维）；任何动态传输维都需配套静态 chunk
（见上）。`atomic` 选择覆盖还是原子加（见 `AtomicType`）。下降出的
`pld.tile.put` verifier 要求 staging tile 在两个
**静态**维度上都**不超过**展平后的传输范围（可以更小 —— 即一个 chunk —— 但不能
更大；动态维由 chunk 在运行时约束）。

### `pld.tensor.get`（TGET）

```text
pld.tensor.get(dst, peer, src, *, chunk_rows: int = 0, chunk_cols: int = 0, pipeline: bool = False) -> Unknown
pld.tensor.get(dst, peer, src, dst_offsets, src_offsets, shape,
               *, chunk_rows: int = 0, chunk_cols: int = 0, pipeline: bool = False) -> Unknown
```

同步地把 `peer` rank 的窗口绑定 `src` 切片读入本地 `dst`。`dst` 可以是窗口绑
定的 `DistributedTensor` 或普通 `Tensor`；`src` 必须是窗口绑定的
`DistributedTensor`。VEC staging tile 由 `ConvertTensorToTileOps` 物化为内部
`tile.create + pld.tile.get`,因此会经过 PyPTO 的内存分配器,但不出现在 DSL
表面。

不提供 offsets/shape 时,该操作把完整的 peer `src` 切片读入完整的本地 `dst`
切片。提供 `dst_offsets`、`src_offsets` 和 `shape` 时,传输会缩小到匹配的
subregion；三者必须一起提供。可选的 `chunk_rows` / `chunk_cols`（`0` = 全量）把
staging tile 缩成展平后传输范围的子块,由 pto-isa TGET 自动分块搬运 —— 与上面
`put` 的契约一致,**包括动态传输**（subregion 的 `shape`,或 full-slice 时
`dst` / `src` 窗口维度）,需配套静态 chunk（动态最内维需 `chunk_cols`,动态前导维
需 `chunk_rows`）。设置 `pipeline=True` 时,会通过两个 staging
tile（`tget_stage_ping` / `tget_stage_pong`）对分块读做双缓冲,发出
`pto.comm.tget(…, buf(%ping, %pong) : …)` 以使用 pto-isa 的 ping-pong `TGET`
重载 —— 契约与 `put` 一致,同样**要求同时设置 `chunk_rows` 与 `chunk_cols`**。

Verifier：`dst` 可以是 `DistributedTensorType` 或普通 `TensorType`（通过
`AsTensorTypeLike` 匹配）；`src` 必须是 `DistributedTensorType`；`peer` 必须是
`ScalarType`；`dst` 与 `src` 必须 element type 相同、rank 相同,且各维都是
**正（positive）**维度（正性仅对静态维校验；动态维允许,由 chunk 约束）。
full-slice `get` 要求 `dst` / `src` 形状一致；subregion `get` 允许完整切片尺寸
不同,只要显式传输区域不越界（仅校验静态维）；任何动态传输维都需配套静态 chunk。
除 `chunk_rows` / `chunk_cols` 外,`get` 不接受 keyword attributes。

### `pld.tensor.all_to_all_v`

```text
pld.tensor.all_to_all_v(
    input, target, signal, send_counts, recv_counts
) -> DistributedTensorType(target)
```

变长 all-to-all（MPI_Alltoallv）。平面 2D 布局：

- `input` — Tensor 或 DistributedTensor `[NR*MAX_RECV, SIZE]`
- `target` — DistributedTensor `[NR*MAX_RECV, SIZE]`（窗口即结果）
- `signal` — DistributedTensor INT32 `[NR, 1]`（单次使用的 Set(1)/wait≥1 屏障）
- `send_counts` — Tensor-like INT32 `[NR]` 或 `[NR, 1]`（运行时每目标行数）
- `recv_counts` — DistributedTensor INT32 `[NR, 1]`（InOut recvcounts）

`MAX_RECV = target.shape[0] // NR`。降级在运行时读取 `send_counts[dest]`、钳制到
`MAX_RECV`，并把**钳制后**的计数通过 `pld.system.notify`（Set）写入对端
`recv_counts[my_rank, 0]`。推送本身总是传输每个目标完整的 `MAX_RECV` 行容量
块——与运行时计数无关——因此超出发送方实际计数的行也会经过链路传输；屏障之后
接收方用 `recv_counts[src, 0]` 跳过这些行（MPI_Alltoallv 语义适用于逻辑结果，
而非链路传输本身）。InCore 路径的传输是编译期定长的 `pld.tile.put`（PTOAS 要求
静态 partition-view 维度）；HOST 路径的内核在入口根据运行时 rank 数推导
`MAX_RECV`（`target.shape[0] / nranks`），因此始终与实际运行的设备数一致。

**InCore composite**（`LowerCompositeOps`）：上述原语在芯片内核中被分解为
`pld.tile.put` + `pld.system.notify`/`wait`。

**HOST builtin**（`LowerHostTensorCollectives`）：同样的 5 参数调用，在
`host_orch` 函数中发起时，会按设备下降为 `builtin.tensor.all_to_all_v`——
一个内核内 TPUT 的 AIV builtin，遵循与 `builtin.tensor.all_to_all` 相同的模式。
在这一层，`input` 与 `send_counts` 都必须是窗口绑定的 `DistributedTensor`（比
composite 的 `AsTensorTypeLike` 更严格，这是 HOST 派发代码生成强制要求的——
它只支持窗口绑定或 tile 参数）——五个操作数（`input`、`target`、`signal`、
`send_counts`、`recv_counts`）必须两两解析到不同的窗口分配（任意一对发生别名
都是跨进程竞争：data 与 data 之间是 TPUT 覆盖，data 与 control 之间是
notify/count 写入与内核读取竞争，control 与 control 之间是 notify 与 count
发布竞争）。内核在入口把 `MAX_RECV` 推导为 `target.shape[0] / nranks`（运行时
通信域大小），因此块布局始终与实际运行的设备数一致——不再需要
`signal.shape[0]` 与设备数**精确相等**的要求，也不再需要按 `MAX_RECV` 进行
variant 混入。不支持在 `host_orch` 的 `for`/`while` 循环内调用（单次使用
的信号协议）——与 `LowerCompositeOps` 在 InCore 路径上强制的限制相同。

### `pld.tensor.allreduce`

```text
pld.tensor.allreduce(src, *, op: ReduceOp = ReduceOp.Sum, mode: str = "mesh", core_num: int = 1) -> DistributedTensorType(src)
pld.tensor.allreduce(src, signal, *, op: ReduceOp = ReduceOp.Sum, mode: str = "mesh", core_num: int = 1) -> DistributedTensorType(src)
```

完全有效的 packed mesh 目标会被视为一个逻辑 `[1, N]` 线性流，并按最大
16 KiB 的 UB 块处理。若静态已知的 `N` 小于该预算，物理块宽度会收缩到能够
覆盖 `N` 的最小 32-byte 对齐宽度；更大或动态的输入仍使用最大块宽。最后一块保持所选物理宽度不变，同时携带
`valid_shape=[1, min(chunk, N-offset)]`，因此任意元素数量都不会越界读写。

对于 mesh 降级，如果 packed ND 目标的 partial `TensorView.valid_shape` 能通过折叠
leading dimensions 表示为单个 2D 矩形，且静态有界的物理 tile 可放入一个 16-KiB
chunk，Pass 会保留该元数据，并沿用单矩形路径只归约这个矩形。符号型有效范围会在
源 tensor 的物理矩形能放入预算时回退使用该物理矩形；过大的 partial 矩形、strided 目标、DN partial view
和无法按该方式表示的 partial 区域会被明确拒绝。

任何在降级后仍为符号表达式的目标范围或 partial-valid 范围，都必须在 kernel 中
通过标量参数、循环变量或物理 Tensor shape 参数获得运行时绑定；仅出现在类型元数据
中的符号会在 PTO codegen 阶段被拒绝。完全动态的物理目标维度由该 Tensor 参数绑定。

对所有参与 rank 的窗口绑定 `src` 切片做原地 all-reduce，并返回与 `src`
相同的类型。`mode` 关键字选择降级算法：

- **`"mesh"`（默认）** — 全对全直接交换，O(P) 个 HCCL 窗口。信号 shape
  `[NR, 1]`（每 rank 一个槽位）。ready 屏障（generation 1）之后进入 chunk
  循环；每个 chunk 执行 `remote_load+accumulate`，再对本调用局部的
  generation 做屏障，最后才 store-back，从而避免写后读 (WAR) 竞态。自清理
  尾声随后把本次调用的总信用数减回每个 cell（参见
  [屏障-信号协议](#屏障-信号协议)），因此调用完成后 signal 恢复为全零。
- **`"ring"`** — NCCL 风格的分块 reduce-scatter + allgather 调度，
  O(1) 个 HCCL 窗口。信号 shape `[2 * (NR − 1), NR]`（每轮 ring 一行，
  每 rank 一个槽位）。packed ND 目标会被视为逻辑 `[1, SIZE]` 线性流；
  partial valid box 必须是连续的 row-major 前缀。降级会保留完整的物理
  `[1, product(target.shape)]` 视图，并把逻辑前缀记录为
  `TensorView.valid_shape=[1, product(target.valid_shape)]`。
  FP32 使用均衡的 `floor(i * SIZE / NR)` 边界；FP16 会把每个内部边界向上
  对齐到 16 个元素（32 字节）并限制在 `SIZE` 内，因此每个非空 segment 都从
  MTE 安全地址开始，同时不改变用户可见的 packed 布局。很短的输入仍允许空
  segment。每个 segment 再按最大 16 KiB 的物理 subchunk 处理；FP16 尾块只把
  remote load 的物理读取范围向上对齐到 32 字节，并在归约或写回前恢复逻辑
  `valid_shape`。每个 subchunk 在 store-back 前都使用本调用局部的 ready
  和 read-complete generation 做屏障，从而避免写后读 (WAR) 竞态。自清理
  尾声随后把本次调用的总信用数从 signal 的每一行中减去。

host-orchestrator 用户代码可以省略 `signal`，包括在 `for` / `while`
循环内；
[`SynthesizeAllReduceSignals`](passes/41-synthesize_allreduce_signals.md) 阶段会为该 call 插入 private INT32 signal window，
语义 shape 为 `[world_size, core_num]`（仅 mesh 模式 — `mode="ring"` 必须显式传入
signal）。该阶段会先插入 standalone `world_size = pld.world_size()` binding，
再用该变量构造 buffer size 和 window shape。自清理协议（参见
[屏障-信号协议](#屏障-信号协议)）使每次调用都是无状态循环，
因此 `for` / `while` 循环内的调用与其他集合通信一样受支持。显式 `signal`
仍然是 InCore lowering 和内部测试使用的形态。通信域物化会把该 signal buffer
保留在与 `src` 相同的 comm-domain 中，即使它没有传给用户自定义
chip kernel。mesh、ring 和
host builtin 路径均支持 FP16、FP32，以及任意正元素数量下的
`ReduceOp.Sum`、`Max`、`Min` 和 `Prod`。InCore lowering 使用受 UB 上限约束的
分块，host builtin 使用 256 元素分块。InCore mesh 和 ring 只把 FP16 remote
尾块的物理范围向上对齐到 32 字节；host builtin 会把 FP16 和 FP32 的 ragged load
范围都对齐到 32 字节。两者都保留逻辑 valid shape。host builtin 接受 rank-1
`[world_size]` 或 rank-2 `[world_size, signal_stride]` signal。Ring 模式
（`mode="ring"`）在 host orchestrator 中降级为 `builtin.tensor.allreduce_ring`，
要求显式 rank-2 `[2 * (NR - 1) + 1, NR]` INT32 signal（额外增加一行用于返回屏障）。

#### HOST 多核 AllReduce（`core_num`）

`core_num` 表示**每个 rank** 上一次 HOST `pld.tensor.allreduce` 分发使用多少个
AIV block。它不改变任务层级：`device=r` 仍然选择卡，调用仍然为每个 rank 降级为
一个 builtin orchestration task，只是该 task 现在启动一个包含 `core_num` 个
block 的同步 SPMD grid。

```python
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, core_num=4)
```

| 约束 | 规则 |
| ---- | ---- |
| 取值 | 编译期正整数，默认 `1`（与既有行为一致） |
| 调度 | 仅 mesh —— `mode="ring"` 要求 `core_num == 1` |
| 容量 | 不超过 backend 的 AIV 核数（经 `rt_submit_aiv_task` 提交，一个 block 对应一个 AIV 核） |
| InCore | 必须保持 `1`，多核应使用外层 `pl.spmd(...)` |

**Signal 布局。** signal 是 peer-major、lane 连续的
`[world_size, signal_stride]` 矩阵，且 `signal_stride >= core_num`。block `b` 在
`signal_base + peer * signal_stride + b` 上等待，并在
`signal_base + my_rank * signal_stride + b` 上通知 peer `p`，因此每个
`(peer, block)` 组合拥有一个独立计数器。rank-1 `[world_size]` signal（stride 为
1）仅在 `core_num == 1` 时有效。自动合成的 signal 恰好是
`[world_size, core_num]`；显式 signal 可以更宽。

**Kernel 切分。** block 以 block-cyclic 方式拥有 256 元素 tile：block `b` 处理
tile `b, b + C, b + 2C, ...`（`C` 为启动的 block 数），因此任意两个 block 不会
触碰同一个 chunk。每个 block 执行一次 ready barrier，然后每个 chunk 执行一次
read-done barrier。该 per-chunk barrier 必须保持在 store **之前**：否则某个 rank
可能在另一个 rank 上对应的 block 完成 remote load 之前就覆盖了自己的源 chunk。
没有数据的 block 仍会执行 ready barrier，从而保持跨 rank 对称，也允许
`core_num` 超过 chunk 数量。索引达到或超过 `signal_stride` 的 block 没有可用
lane，会直接退出而不参与 barrier；由于各 rank 的 `signal_stride` 一致，所有 rank
退出的是同一批 block，协议依然对称。

**为什么用一个 SPMD grid 而不是 `pl.parallel`。** `pl.parallel(N)` 会产生 `N`
个独立 task，每个都有自己的 TaskId 和调度生命周期，对这种原地集合通信并不安全：
不同 rank 可能以不同顺序调度 chunk task，因此等待另一 rank 对应 chunk 的 task
可能死锁；而且共享 InOut window 上保守的依赖分析往往会把它们串行化。单个 SPMD
grid 避免了这两个问题 —— `require_sync_start` 让所有 block 一起准入，
`block_idx` 在每个 rank 上给出确定且互相匹配的划分。它是单卡准入保证而非跨 rank
的全局同时启动；跨 rank 的启动偏差由 ready barrier 吸收。

### `pld.system.notify`（TNOTIFY）

```text
pld.system.notify(target, peer, offsets, value, *, op: int) -> Unknown
```

把 `value` 写入 `peer` rank 的 `target` 信号槽位（一个窗口绑定 `DistributedTensor`,
通常是一维 INT32 "信号矩阵"）。`op` 选择原子加还是 set（见 `NotifyOp`）。

Verifier：`target` 必须是 `DistributedTensorType`；`peer` 与 `value` 必须是
`ScalarType`；`offsets` 必须是 rank 等于 target rank 的 `MakeTuple`。

### `pld.system.wait`（TWAIT）

```text
pld.system.wait(signal, offsets, expected, *, cmp: int) -> Unknown
```

阻塞直到本 rank 自身的 `signal` 信号槽位相对 `expected` 满足 `cmp` 谓词
（见 `WaitCmp`）。

Verifier：`signal` 必须是 `DistributedTensorType`；`expected` 必须是
`ScalarType`；`offsets` 必须是 rank 等于 signal rank 的 `MakeTuple`。

## 共享 codegen 基础设施

六个算子全部经由 `src/backend/common/pto_ops_distributed.cpp` 和
`src/codegen/pto/pto_codegen.cpp` 中的 PTO codegen 辅助函数下降。共享的可复用部件
—— 使每个算子的下降都不携带专门的 peer 算术 —— 如下：

| 辅助函数 | 作用 |
| -------- | ---- |
| `CommRemoteOffset_<dtype>` | 按 dtype 的 MLIR 辅助函数（由 `PTOCodegen::EmitCommRemoteOffsetHelpers` 一次性发出）,把 `(ctx, peer)` 转为 peer 窗口切片的字节偏移 |
| `EmitCommRemoteView` | 在调用点发出 `CommRemoteOffset + addptr + make_tensor_view`,得到 peer 寻址的视图（被 `remote_load`、`get` 的 `src` 和 `put` 的 `dst` 使用） |
| `EmitPartitionViewPTO` | 用给定 offsets/sizes 把 tensor view 包成全切片 `partition_view`（被每个算子的本地与 peer 操作数使用） |
| `ResolveDistTensorBinding` | 把 `DistributedTensor` 实参解析为其 codegen 绑定（类型 + 窗口变量） |
| `AsTensorTypeLike` | kind-trait 向下转换,在统一读取视图 element/shape 信息处同时接受 `TensorType` 与 `DistributedTensorType` |

本地与远程的拆分是有意的：*本地*操作数（如 `get` 的 `dst`、`put` 的 `src`、`wait` 的 `signal`）
复用 `EmitMakeTensorViews` 已创建的 tensor view,无 peer 算术；而*远程*操作数
（如 `remote_load` 的 `target`、`get` 的 `src`、`put` 的 `dst`）则经由
`EmitCommRemoteView`。

## 流水线集成

通信域与其槽位分配由
[`MaterializeCommDomainScopes`](passes/42-materialize_comm_domain_scopes.md) pass 完成。该 pass 将每个
host_orch 函数体包裹进嵌套的 `CommDomainScopeStmt` 节点（按推断出的通信域逐层嵌套），并产生运行时据以
绑定物理缓冲的按窗口 `WindowBuffer` 记录。
随后 [`LowerHostTensorCollectives`](passes/43-lower_host_tensor_collectives.md) 会在最终
`Simplify` 之前把 host-level tensor collectives 降为内部 builtin chip dispatch。

## 测试

- **IR / parser**：`tests/ut/ir/parser/test_remote_load.py`、
  `tests/ut/ir/parser/test_remote_store.py`、`test_system_ops.py`、
  `test_get_op.py`、`test_put_op.py`,以及
  `tests/ut/ir/test_distributed_ops.py` 中的 negative verifier 覆盖。
- **Codegen**：`tests/ut/codegen/distributed/test_distributed_pto_codegen.py`。
- **端到端（ST）**：`tests/st/distributed/test_l3_allreduce.py`（mesh allreduce；
  动态秩维 ``NR = pl.dynamic("NR")``；默认 **P=2**，任意四卡跑 **P=4**，例如
  ``--device=0,1,2,3`` 或 ``--device=0-3``）、`test_l3_allgather.py`、
  `test_l3_reduce_scatter.py`、`test_l3_broadcast.py`（三者同样采用动态 NR，
  P=2/P=4）、`test_l3_tensor_allreduce_intrinsic.py`、
  `test_l3_tensor_allreduce_ring_intrinsic.py`、
  `test_l3_allreduce_ring.py`（手写 ring RS+AG）、
  `test_l3_host_tensor_allreduce.py`、`test_l3_host_tensor_allreduce_ring.py`、
  `test_l3_ep_dispatch_combine.py`、`test_l3_notify_wait.py`、
  `test_l3_tensor_all_to_all_v_intrinsic.py`（InCore composite）、
  `test_l3_host_tensor_all_to_all_v.py`（HOST builtin），以及
  `tests/st/distributed/` 下其他 L3 ST。**Put/Get 端到端权威契约** 已启用：
  `test_l3_put.py`（环形覆写、行偏移 put、原子加 put、分块/流水 transfer ✅）、
  `test_l3_get.py`（环形读、行偏移 get ✅）、以及 `test_l3_remote_store.py`
  （tile 级子视图 push ✅）。所有测试均采用由 notify/wait 和集体 ST 建立的
  `pld.system.notify` / `pld.system.wait` 握手模式。
