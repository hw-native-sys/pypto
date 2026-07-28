# Distributed (Multi-Card) Guide

> **Status:** DRAFT skeleton. Authoring kernels that span **multiple cards** with
> the `pypto.language.distributed` (`pld.*`) family. For single-chip multi-block
> parallelism (`pl.spmd`/`pl.cluster`), see
> [Perf › Split & Parallel](../handbook/perf/02-split-parallel.md) instead.

## Distributed Types

_TODO:_

- `pld.DistributedTensor` — a tensor sharded/replicated across cards.
- `pld.CommCtx` / `CommCtxType` — the communication context.

## Tensor-Level Collectives

_TODO:_

- `pld.tensor.put` / `pld.tensor.get` — one-sided transfers.
- `pld.tensor.allreduce` (`ReduceOp`) — reductions across cards.

## Tile-Level Remote Movement

_TODO:_

- `pld.tile.remote_load` / `pld.tile.remote_store`.

## Signals & Synchronization

_TODO:_

- `pld.system.notify` / `pld.system.wait` (`NotifyOp` / `WaitCmp` / `AtomicType`).

## Peer Buffers

_TODO:_

- `reserve_buffer` / `import_peer_buffer`.

## Putting It Together

_TODO — a small end-to-end multi-card example (host orchestration + CommCtx
passing + a collective)._

## See Also

- Developer reference: [`dev/distributed_ops.md`](../../dev/distributed_ops.md)
