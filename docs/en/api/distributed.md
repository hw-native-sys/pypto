# `pld`

The cross-rank namespace. It mirrors `pl`'s shape — a 3-segment `pld.<category>.<op>`
surface and a 2-segment short form that dispatches by name. `world_size` takes no
argument; `rank`/`nranks` take a `CommCtx`, not a tensor; `get_comm_ctx` takes a
window-bound `DistributedTensor` to look up its `CommCtx`; `alloc_window_buffer`
creates the per-rank window and `window` views it as a `DistributedTensor`. Every
other communication and data-transfer operation addresses a **window-bound**
`DistributedTensor` directly: a slice of the symmetric, per-rank communication
window `alloc_window_buffer` allocated. See
[Choosing a namespace](../user/ops/00-dispatch.md) and the
[Distributed Programming chapter](../user/distributed/index.md).

::: pypto.language.distributed
    options:
      show_root_heading: false
      filters: ["!^_"]
      members:
        - AtomicType
        - CommCtx
        - CommEngine
        - DistributedTensor
        - NotifyOp
        - ReduceOp
        - WaitCmp
        - alloc_window_buffer
        - get_comm_ctx
        - nranks
        - rank
        - remote_load
        - remote_store
        - window
        - world_size

## `pld.tensor`

Collectives, window allocation and whole-tensor cross-rank transfers.

::: pypto.language.distributed.op.tensor_ops
    options:
      show_root_heading: false
      filters: ["!^_"]

## `pld.tile`

Tile-level cross-rank transfers, inside an InCore function.

::: pypto.language.distributed.op.tile_ops
    options:
      show_root_heading: false
      filters: ["!^_"]

## `pld.system`

Rank/world queries, `CommCtx` access, and the notify/wait signal protocol.

::: pypto.language.distributed.op.system_ops
    options:
      show_root_heading: false
      filters: ["!^_"]
