# 分布式算子（Distributed Operators，N6）

## 概述

N6 分布式算子族为 Python DSL 提供了对硬件跨 rank（cross-rank）通信原语的直接、带类型的访问。族内每个算子都作用于一个**窗口绑定的（window-bound）**
[`DistributedTensorType`](ir/02-types.md) —— 其存储是 `pld.alloc_window_buffer`
分配的对称、按 rank 划分的通信窗口的一个切片。族内 verifier 通常会拒绝普通
`TensorType`（严格的 kind-trait 匹配 —— `As<DistributedTensorType>` 不匹配普通
`TensorType`），以保证非窗口绑定的 tensor 永远不会被误传入跨 rank 槽位。
**唯一明确的例外：** `pld.tensor.put`（以及它下降出的 `pld.tile.put`）的
`src` 参数通过 `AsTensorTypeLike` 接受普通 `Tensor` —— TPUT 在源端只需要一段
可读的本地 GM 区域,因此 kernel 可以直接从 host 输入推送,不必先经过窗口缓冲
中转；`dst` 仍然必须是窗口绑定的 `DistributedTensor`。

共有**七个算子**和**四个 ABI 枚举**：

| 算子 | 方向 | 结果 | 硬件 |
| ---- | ---- | ---- | ---- |
| `pld.tile.remote_load` | pull（读 peer → 本地 tile） | `TileType` | TLOAD |
| `pld.tile.remote_store` | push（写本地 tile → peer） | `Unknown`（副作用） | TSTORE |
| `pld.tensor.get` | pull（读 peer → 本地 GM） | `Unknown`（副作用） | TGET |
| `pld.tensor.put` | push（写本地 → peer） | `Unknown`（副作用） | TPUT |
| `pld.tensor.allreduce` | collective reduce over window slices | `DistributedTensorType`（同 src） | builtin collective |
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
- **`pld.tensor.get`** 读写 *tensor*（GM）操作数 —— `dst` 和 `src` 都是窗口绑定的
  `DistributedTensor` 视图。TGET 中转用的 VEC staging tile 由
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
| `ReduceOp` | `kMax` | 预留的最大值规约变体；lowering 待实现 |
| `ReduceOp` | `kMin` | 预留的最小值规约变体；lowering 待实现 |
| `ReduceOp` | `kProd` | 预留的乘法规约变体；lowering 待实现 |

每个枚举跨三层保持一致（C++ `enum class` → bindings 中的 `nb::enum_` → `.pyi`
存根）,并以 `pld.NotifyOp` / `pld.WaitCmp` / `pld.AtomicType` / `pld.ReduceOp` 暴露给 DSL。
deducer 会校验打包的 `int` 落在枚举范围内,使 codegen 无需二次保护即可转回。

## 算子参考

### `pld.tile.remote_load`（TLOAD）

```text
pld.tile.remote_load(target, peer, offsets, shape) -> TileType(shape, target.dtype)
```

把 `peer` rank 的窗口绑定 `DistributedTensor` 切片中的一个区域读入本地 tile。
在 IR 层面镜像 `tile.load`（位置参数 `offsets` / `shape` 元组、`TileType` 结果）,
但源是*远程*切片 —— 地址转换在 codegen 时由
`CommRemoteOffset(ctx, peer) + addptr + make_tensor_view` 实现。

Verifier：`target` 必须是 `DistributedTensorType`；`peer` 必须是 `ScalarType`
rank 索引；`offsets` / `shape` 必须各为 `MakeTuple`,其 rank 等于
`target.shape.size()`。

DSL（`python/pypto/language/distributed/op/tile_ops.py`）把 `peer` / `offsets` /
`shape` 暴露为仅关键字（keyword-only）参数以提升可读性；IR 算子保持位置参数,
与 `tile.load` 一致。

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
pld.tensor.put(dst, peer, src, *, atomic: int) -> Unknown
pld.tensor.put(dst, peer, src, dst_offsets, src_offsets, shape, *, atomic: int) -> Unknown
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

Verifier：`dst` 必须是 `DistributedTensorType`；`src` 必须是 `TensorType` 或
`DistributedTensorType`（通过 `AsTensorTypeLike` 匹配）；`peer` 必须是
`ScalarType`；`dst` 与 `src` 必须 element type 相同、rank 相同,且各维都是
**正的静态（positive static）**维度。full-slice `put` 要求形状完全相同；
subregion `put` 允许完整切片尺寸不同,只要显式传输区域不越界。`atomic` 选择覆盖
还是原子加（见 `AtomicType`）。

### `pld.tensor.get`（TGET）

```text
pld.tensor.get(dst, peer, src) -> Unknown
pld.tensor.get(dst, peer, src, dst_offsets, src_offsets, shape) -> Unknown
```

同步地把 `peer` rank 的窗口绑定 `src` 切片读入本地窗口绑定 `dst`。两个操作数
都是 GM 层级的 `DistributedTensor` 视图；VEC staging tile 由
`ConvertTensorToTileOps` 物化为内部 `tile.create + pld.tile.get`,因此会经过
PyPTO 的内存分配器,但不出现在 DSL 表面。

不提供 offsets/shape 时,该操作把完整的 peer `src` 切片读入完整的本地 `dst`
切片。提供 `dst_offsets`、`src_offsets` 和 `shape` 时,传输会缩小到匹配的
subregion；三者必须一起提供。

Verifier：`dst` / `src` 必须都是 `DistributedTensorType`；`peer` 必须是
`ScalarType`；`dst` 与 `src` 必须 element type 相同、rank 相同,且各维都是
**正的静态（positive static）**维度。full-slice `get` 要求形状完全相同；
subregion `get` 允许完整切片尺寸不同,只要显式传输区域不越界。`get` 不接受
keyword attributes。

### `pld.tensor.allreduce`

```text
pld.tensor.allreduce(src, signal, *, op: ReduceOp = ReduceOp.Sum) -> DistributedTensorType(src)
```

对所有参与 rank 的窗口绑定 `src` 切片做原地 all-reduce，并返回与 `src`
相同的类型。用户显式传入窗口绑定的 INT32 `signal` tensor，并保证它为参与 rank
提供足够的信号槽位；通信域物化会把该 signal buffer 保留在与 `src` 相同的
comm-domain 中，即使它没有传给用户自定义 chip kernel。public op 当前接受
`ReduceOp.Sum`，并会拒绝预留的 `Max` / `Min` / `Prod` 变体，直到这些
lowering 落地。host builtin lowering 路径当前仅支持 `Sum` + FP32 变体，并要求
signal tensor 为 rank-1。

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

六个算子全部经由 `src/backend/common/pto_ops_common.cpp` 和
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
[`MaterializeCommDomainScopes`](passes/37-materialize_comm_domain_scopes.md) pass 完成。该 pass 将每个
host_orch 函数体包裹进嵌套的 `CommDomainScopeStmt` 节点（按推断出的通信域逐层嵌套），并产生运行时据以
绑定物理缓冲的按窗口 `WindowBuffer` 记录。
随后 [`LowerHostTensorCollectives`](passes/38-lower_host_tensor_collectives.md) 会在最终
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
  `test_l3_reduce_scatter.py`、`test_l3_broadcast.py`、`test_l3_gemm.py`、
  `test_l3_ep_dispatch_combine.py`、`test_l3_notify_wait.py`，以及
  `tests/st/distributed/` 下其他 L3 ST。`test_l3_put.py` 与 `test_l3_get.py` 目前**被
  skip**，等待 N7 host codegen（每个 `DistributedTensor` 的 `add_scalar(ctx)`、
  `ContinuousTensor.make(..., child_memory=True)`）与 N8 driver glue（在 codegen 发出的
  `orch.allocate_domain(...)` 块上接线 `HostBufferStaging` 窗口）。其中内嵌的程序与 golden
  校验是端到端的权威契约 —— 待上述 host 侧工作落地后即可移除 skip 标记。
