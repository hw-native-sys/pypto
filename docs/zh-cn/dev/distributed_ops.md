# 分布式算子（Distributed Operators，N6）

## 概述

N6 分布式算子族为 Python DSL 提供了对硬件跨 rank（cross-rank）通信原语的直接、带类型的访问。族内每个算子都作用于一个**窗口绑定的（window-bound）**
[`DistributedTensorType`](ir/02-types.md) —— 其存储是 `pld.alloc_window_buffer`
分配的对称、按 rank 划分的通信窗口的一个切片。普通 `TensorType` 会被族内每个
verifier *拒绝*（严格的 kind-trait 匹配 —— `As<DistributedTensorType>` 不匹配普通
`TensorType`），因此非窗口绑定的 tensor 永远不会被误传入跨 rank 操作。

共有**四个算子**和**三个 ABI 枚举**：

| 算子 | 方向 | 结果 | 硬件 |
| ---- | ---- | ---- | ---- |
| `pld.tile.remote_load` | pull（读 peer → 本地 tile） | `TileType` | TLOAD |
| `pld.tensor.put` | push（写本地 → peer） | `Unknown`（副作用） | TPUT |
| `pld.system.notify` | 给 peer 的槽位发信号 | `Unknown`（副作用） | TNOTIFY |
| `pld.system.wait` | 在自身槽位上阻塞 | `Unknown`（副作用） | TWAIT |

三个仅有副作用（side-effect-only）的算子产生
[`UnknownType`](ir/02-types.md)：它们因跨 rank 副作用而存在，而非为消费者读取的
SSA 值而存在。

## 命名空间：为何区分 `tile.*` / `tensor.*` / `system.*`

命名空间编码的是算子所在的 IR 层级，而非随意分组：

- **`pld.tile.remote_load`** 产生一个 *tile*（片上 SRAM 区域），因此是 `tile.load`
  的兄弟,归入 `pld.tile`。
- **`pld.tensor.put`** 读写 *tensor*（GM）操作数 —— `dst` 和 `src` 都是窗口绑定的
  `DistributedTensor` 视图,TPUT 中转用的 VEC staging tile 在 codegen 阶段合成,
  不出现在 DSL 表面。因此它是 `pld.tensor.alloc_window_buffer` /
  `pld.tensor.window` 的兄弟,而**不是**产出 tile 的 `remote_load` 的兄弟。
- **`pld.system.notify` / `pld.system.wait`** 驱动按 rank 的信号槽位 —— 纯控制面
  同步,无数据操作数 —— 因此归入 `pld.system`。

## ABI 枚举（`include/pypto/ir/comm.h`）

三个枚举是**仅追加（append-only）的 ABI**。它们的底层 `int` 值被序列化为算子的
kwarg 负载（notify 的 `op`、wait 的 `cmp`、put 的 `atomic`）,并在 codegen 时转
回枚举。新变体只能加在**末尾**,以保证已有 IR 和缓存程序的语义不变。

```cpp
enum class NotifyOp : int { kAtomicAdd = 0, kSet = 1 };   // pld.system.notify
enum class WaitCmp  : int { kEq = 0,        kGe = 1 };     // pld.system.wait
enum class AtomicType : int { kNone = 0,    kAdd = 1 };    // pld.tensor.put
```

| 枚举 | 变体 | 含义 |
| ---- | ---- | ---- |
| `NotifyOp` | `kAtomicAdd` | 原子地把 `value` 加到 peer 的信号槽位 |
| `NotifyOp` | `kSet` | 非原子地把 `value` 存入 peer 的信号槽位 |
| `WaitCmp` | `kEq` | 阻塞直到 `*signal_slot == expected` |
| `WaitCmp` | `kGe` | 阻塞直到 `*signal_slot >= expected` |
| `AtomicType` | `kNone` | 普通远程写 —— 覆盖 peer 的 dst 切片 |
| `AtomicType` | `kAdd` | 原子地把源数据加到 peer 的 dst 切片 |

每个枚举跨三层保持一致（C++ `enum class` → bindings 中的 `nb::enum_` → `.pyi`
存根）,并以 `pld.NotifyOp` / `pld.WaitCmp` / `pld.AtomicType` 暴露给 DSL。
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

### `pld.tensor.put`（TPUT）

```text
pld.tensor.put(dst, peer, src, *, atomic: int) -> Unknown
```

同步地把本地窗口绑定的 `src` 写入 `peer` rank 的窗口绑定 `dst` 切片。两个操作数
都是 GM 层级的 `DistributedTensor` 视图；VEC staging tile 在 codegen 时合成
（`src/backend/common/pto_ops_common.cpp` 中的 `MakePutCodegenPTO`）,不出现在
DSL 表面。

Verifier：`dst` / `src` 必须都是 `DistributedTensorType`；`peer` 必须是
`ScalarType`；`dst` 与 `src` 必须 element type 相同且形状为相同的**正的静态
（positive static）**形状（staging VEC 缓冲需要编译期范围）。`atomic` 选择覆盖
还是原子加（见 `AtomicType`）。

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

四个算子全部经由 `src/backend/common/pto_ops_common.cpp` 和
`src/codegen/pto/pto_codegen.cpp` 中的 PTO codegen 辅助函数下降。共享的可复用部件
—— 使每个算子的下降都不携带专门的 peer 算术 —— 如下：

| 辅助函数 | 作用 |
| -------- | ---- |
| `CommRemoteOffset_<dtype>` | 按 dtype 的 MLIR 辅助函数（由 `PTOCodegen::EmitCommRemoteOffsetHelpers` 一次性发出）,把 `(ctx, peer)` 转为 peer 窗口切片的字节偏移 |
| `EmitCommRemoteView` | 在调用点发出 `CommRemoteOffset + addptr + make_tensor_view`,得到 peer 寻址的视图（被 `remote_load` 和 `put` 的 `dst` 使用） |
| `EmitPartitionViewPTO` | 用给定 offsets/sizes 把 tensor view 包成全切片 `partition_view`（被每个算子的本地与 peer 操作数使用） |
| `ResolveDistTensorBinding` | 把 `DistributedTensor` 实参解析为其 codegen 绑定（类型 + 窗口变量） |
| `AsTensorTypeLike` | kind-trait 向下转换,在统一读取视图 element/shape 信息处同时接受 `TensorType` 与 `DistributedTensorType` |

本地与远程的拆分是有意的：*本地*操作数（如 `put` 的 `src`、`wait` 的 `signal`）
复用 `EmitMakeTensorViews` 已创建的 tensor view,无 peer 算术；而*远程*操作数
（如 `remote_load` 的 `target`、`put` 的 `dst`）则经由 `EmitCommRemoteView`。

## 流水线集成

窗口缓冲和 comm group 由
[`CollectCommGroups`](passes/34-collect_comm_groups.md) pass 收集,该 pass 填充
`Program.comm_groups_` 以及运行时据以绑定物理缓冲的按窗口 `WindowBuffer` 记录。

## 测试

- **IR / parser**：`tests/ut/ir/parser/test_remote_load.py`、
  `test_system_ops.py`、`test_put_op.py`,以及
  `tests/ut/ir/test_distributed_ops.py` 中的 negative verifier 覆盖。
- **Codegen**：`tests/ut/codegen/distributed/test_distributed_pto_codegen.py`。
- **端到端（ST）**：`tests/st/distributed/test_l3_remote_load.py`、
  `test_l3_notify_wait.py`、`test_l3_put.py`。它们目前**被 skip**,等待 N7 host
  codegen（每个 `DistributedTensor` 的 `add_scalar(ctx)`、
  `ContinuousTensor.make(..., child_memory=True)`）与 N8 driver glue
  （`HostBufferStaging` / `ChipBootstrapConfig` 窗口接线）。其中内嵌的程序与
  golden 校验是端到端的权威契约 —— 待上述 host 侧工作落地后即可移除 skip 标记。
