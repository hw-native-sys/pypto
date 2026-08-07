# LowerHostTensorCollectives Pass

## 概览

`LowerHostTensorCollectives` 将 host orchestrator 中的
`pld.tensor.allreduce`、`pld.tensor.barrier`、`pld.tensor.broadcast`、
`pld.tensor.reduce_scatter`、`pld.tensor.allgather`、
`pld.tensor.all_to_all` 和 `pld.tensor.all_to_all_v` 调用改写为编译器内部的
builtin chip dispatch。它在 [`MaterializeCommDomainScopes`](42-materialize_comm_domain_scopes.md) 之后运行，
因此 window 绑定的 data tensor 和用户显式传入或编译器合成的 signal tensor 已经带有
`WindowBuffer` 反向引用，并属于推断出的通信域。

该 pass 不修改非 host 函数。InCore allreduce 仍然走
[`LowerCompositeOps`](13-lower_composite_ops.md)。

## Pipeline 位置

```text
... -> SynthesizeAllReduceSignals -> MaterializeCommDomainScopes -> LowerHostTensorCollectives -> MaterializeDistTensorCtx -> Simplify（最终） -> MaterializeRuntimeScopes
```

最终的 `Simplify` 位于本 pass 之后，用于继续折叠生成的循环边界或常量表达式，
随后再插入 runtime scopes。

## 行为

对于 host orchestrator 中的调用：

```python
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, core_num=4)
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")
signal = pld.tensor.barrier(signal)
data = pld.tensor.broadcast(data, signal, root=0)
data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
data = pld.tensor.allgather(stage, data, signal)
data = pld.tensor.all_to_all(stage, data, signal)
data = pld.tensor.all_to_all_v(input, target, signal, send_counts, recv_counts)
```

`pld.tensor.allreduce` 根据 `mode` kwarg 进行分发：默认 `mode="mesh"` 会 lower 到
`builtin.tensor.allreduce`，而 `mode="ring"` 会 lower 到
`builtin.tensor.allreduce_ring`。其他取值将作为用户错误被拒绝。

对于 `allgather` / `all_to_all` / `all_to_all_v`，`stage`/`input`（TPUT 源）
与 `data`/`target`（结果）必须是两个不同的 window。`allgather` 的 `stage`
只保存本 rank 的单个分片，形状为 `[1, SIZE]`；`all_to_all` 的 `stage` 每行
携带一个按目的地划分的分片，形状为 `[NR, SIZE]`；`all_to_all_v` 的 `input`
每个目的地携带一个 `MAX_RECV` 行容量块，形状为 `[NR*MAX_RECV, SIZE]`。
`all_to_all` / `all_to_all_v` 两种情况下 `data`/`target` 都是 peer 推入的
结果窗口。`all_to_all_v` 还额外要求 `send_counts`（在这一层是窗口绑定的，
仅本地使用）和 `recv_counts`（窗口绑定，通过 `pld.system.notify` 跨 rank
发布）——五个窗口参数都必须位于同一个 `CommDomainScopeStmt` 中，并且必须
两两互不相同（任意一对发生别名都是跨进程竞争，无论是 data 与 data、
data 与 control，还是 control 与 control 之间）。

本 pass 会为每个参与设备生成对应的 `builtin.tensor.*` 调用（如
`builtin.tensor.allreduce`、`builtin.tensor.allreduce_ring`、
`builtin.tensor.barrier`、`builtin.tensor.broadcast`、
`builtin.tensor.reduce_scatter`、`builtin.tensor.allgather`、
`builtin.tensor.all_to_all`、`builtin.tensor.all_to_all_v`）。若外层
comm-domain scope 带有显式 device 列表，则生成 `SeqStmts`；否则生成顺序
`for r in pld.system.world_size()` 循环。

每个生成的 builtin call 携带来源 `pld.tensor.*` 调用中 collective 特定的
参数和 kwarg 属性。窗口绑定的 INOUT tensor 原样传递；标量 kwarg 值
（`op`、`root`、`dtype`，以及 mesh AllReduce 的 `core_num`）转发给 builtin。
`all_to_all_v` 的 `MAX_RECV` 不是 lowering 时的属性：HOST 内核在入口把它推导为
`target.shape[0] / nranks`（运行时通信域大小），因此不再需要按 `MAX_RECV`
进行代码生成的 variant 混入，块布局也始终与实际运行的设备数一致。

若用户代码使用赋值形式，pass 会在生成的 builtin 调用之后追加
`<result> = <original expr>`，保留 public API 的 rebind 语义。

## 检查

该 pass 要求两个参数都是已经 materialize 的 `DistributedTensorType` view，并且位于同一个
`CommDomainScopeStmt` 中。host allreduce builtin 支持 FP16、FP32 上的
`ReduceOp.Sum`、`Max`、`Min` 和 `Prod`，并支持任意正元素数量。它按 256 个
元素分块，并把 FP16 和 FP32 的 ragged load 范围都对齐到 32 字节，不改变逻辑 tensor shape。
signal 必须是 INT32 tensor，形状可以是 rank-1 `[world_size]` 或 rank-2
`[world_size, signal_stride]`；当参与设备数静态可知时，signal 的静态容量必须足够。
由于 signal 由 `pld.window` 产生，它天然是 packed 的，builtin 按扁平 row-major
数组索引它。

mesh allreduce 为每个启动的 AIV block 分配一条 signal lane：rank-1 signal 仅在
`core_num == 1` 时有效；rank-2 signal 要求第二维是常量且
`signal_stride >= core_num`（允许更宽的 stride，因此显式 signal 可以带有多余
lane）。`core_num` 还必须不超过所配置 backend 的 AIV 核数——该 builtin 以
standalone AIV kernel 提交并设置了 `require_sync_start`，超额的 launch 永远无法
被准入，表现为挂死而非报错。未配置 backend 时（纯 IR 测试）跳过该检查。
多核仅支持 mesh：`mode="ring"` 要求 `core_num == 1`。

Ring allreduce（`mode="ring"`）
的 signal 为 rank-2，形状为 `[2 * (NR - 1) + 1, NR]`，其
`shape[0]` 在 signal 两个维度均为编译期常量时必须恰好等于 `2 * (NR - 1) + 1`；仅
`shape[0]` 静态可知时则至少为 `2 * (NR - 1) + 1`（两个维度均为动态时无静态检查）。
当参与设备数静态可知时，signal 的静态容量必须足够。ring allreduce 还要求 `numel(src) % NR == 0`（ring schedule 将 src 划分为 NR 个连续 chunk；余数非零会留下内核无法处理的尾部部分 chunk）。host-ring 的 `src` 形状必须静态已知——动态 extent 会被拒绝，否则运行时 `numel` 不被 `NR` 整除时内核会静默返回未归约的数据。

Ring allreduce 目前仅支持 `ReduceOp.Sum` 和 `dtype=FP32`。
`ReduceOp.Max`、`ReduceOp.Min`、`ReduceOp.Prod` 以及 `FP16` 在
`mode="ring"` 下尚未支持。Ring allreduce 最多支持 16 个参与设备
（`world_size <= 16`）。

`all_to_all_v` 的单次使用 Set(1)/wait≥1 信号无法在 `host_orch` 的
`for`/`while` 循环中复用——本 pass 之前紧邻运行的
[`MaterializeCommDomainScopes`](42-materialize_comm_domain_scopes.md) 会提前
拒绝这种情况（与 `LowerCompositeOps` 在 InCore 路径上强制的限制相同）。在显式
静态 device 子集上，`all_to_all_v` 的 signal `shape[0]` 必须与子集大小
**精确相等**（而非其他 collective 所要求的 `>=`），因为 `MAX_RECV` 是由
`target.shape[0] / signal.shape[0]` 推导得出的，signal 过度分配会导致
静默的错误降级。

## Pass 属性

| 字段 | 取值 |
| ---- | ---- |
| `required` | `{IRProperty::CommDomainScopesMaterialized}` |
| `produced` | `{IRProperty::CommDomainScopesMaterialized}` |
| `invalidated` | `{}` |

## 参考

- 实现：[src/ir/transforms/lower_host_tensor_collectives_pass.cpp](../../../../src/ir/transforms/lower_host_tensor_collectives_pass.cpp)
- 头文件：[include/pypto/ir/transforms/passes.h](../../../../include/pypto/ir/transforms/passes.h)
- 测试：[tests/ut/ir/transforms/test_lower_host_tensor_collectives.py](../../../../tests/ut/ir/transforms/test_lower_host_tensor_collectives.py)
