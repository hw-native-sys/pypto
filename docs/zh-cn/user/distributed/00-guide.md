# 分布式（多卡）指南

> **状态：** 草稿骨架。使用 `pypto.language.distributed`（`pld.*`）家族编写跨**多卡**
> 的 kernel。片上多 block 并行（`pl.spmd`/`pl.cluster`）请改看
> [性能 › 切分与并行](../handbook/perf/02-split-parallel.md)。

## 分布式类型

_TODO：_

- `pld.DistributedTensor` —— 跨卡分片/复制的张量。
- `pld.CommCtx` / `CommCtxType` —— 通信上下文。

## 张量级 Collective

_TODO：_

- `pld.tensor.put` / `pld.tensor.get` —— 单边传输。
- `pld.tensor.allreduce`（`ReduceOp`）—— 跨卡归约。

## 片级远程搬运

_TODO：_

- `pld.tile.remote_load` / `pld.tile.remote_store`。

## 信号与同步

_TODO：_

- `pld.system.notify` / `pld.system.wait`（`NotifyOp` / `WaitCmp` / `AtomicType`）。

## Peer Buffer

_TODO：_

- `reserve_buffer` / `import_peer_buffer`。

## 组合示例

_TODO —— 一个小的端到端多卡示例（host 编排 + CommCtx 传递 + 一个 collective）。_

## 参见

- 开发者参考：[`dev/distributed_ops.md`](../../dev/distributed_ops.md)
