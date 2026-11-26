# `pld`

跨 rank 命名空间。它的形状与 `pl` 一致 —— 一个 3 段式的 `pld.<category>.<op>` 表面，
加上按名字分派的 2 段式简写。`world_size` 不接受任何参数；`rank`/`nranks` 接受的是
`CommCtx`，而非张量；`get_comm_ctx` 接受一个窗口绑定的 `DistributedTensor` 来查找其
`CommCtx`；`alloc_window_buffer` 创建 per-rank 窗口，`window` 则把它视图化为
`DistributedTensor`。除此之外，其余每个通信与数据传输操作都直接作用于**窗口绑定**的
`DistributedTensor`：即 `alloc_window_buffer` 所分配的、对称的 per-rank 通信窗口中的
一段切片。见
[选择命名空间](../user/ops/00-dispatch.md)与
[分布式编程章节](../user/distributed/index.md)。

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

集合通信、窗口分配，以及整张张量级别的跨 rank 传输。

::: pypto.language.distributed.op.tensor_ops
    options:
      show_root_heading: false
      filters: ["!^_"]

## `pld.tile`

Tile 级别的跨 rank 传输，用于 InCore 函数内部。

::: pypto.language.distributed.op.tile_ops
    options:
      show_root_heading: false
      filters: ["!^_"]

## `pld.system`

Rank/world 查询、`CommCtx` 访问，以及 notify/wait 信号协议。

::: pypto.language.distributed.op.system_ops
    options:
      show_root_heading: false
      filters: ["!^_"]
