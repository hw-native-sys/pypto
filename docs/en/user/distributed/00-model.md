# Distributed Programming Model

> **Prerequisites:** [Getting Started](../00-getting_started.md) — basic PyPTO
> tensor/tile model. This guide uses the `pld` namespace
> (`import pypto.language.distributed as pld`).
>
> **DSL form:** this chapter authors programs with `@pl.jit` (plain Python
> functions). `@pl.program`/`@pl.function` is the equivalent class-based form
> used by the older tests under `tests/st/distributed/` — see the
> [Compiling](../execution/00-compile.md) for the full `@pl.jit` family.

## Quickstart: 2-Rank AllReduce

The simplest distributed program — two ranks sum their data, both see the same result.

> This is the **mesh all-reduce** pattern — stage in, barrier, read every
> peer's slice and sum — which the [mesh allreduce walkthrough](13-allreduce_mesh.md)
> builds step-by-step in the [tutorial ladder](05-tutorials.md). At two ranks
> every algorithm collapses to this one exchange; the
> [ring allreduce walkthrough](15-allreduce_ring.md) is the last manual step
> before the builtin is revealed.

```python
import pypto.language as pl
import pypto.language.distributed as pld

NR = pl.dynamic("NR")
SIZE = 256

@pl.jit.incore
def reduce_step(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
) -> pl.Tensor[[1, SIZE], pl.FP32]:
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)

    # 1. Stage-in: copy local input into this rank's window slice.
    local = pl.load(inp, [0, 0], [1, SIZE])
    data = pl.store(local, [0, 0], data)

    # 2. Barrier: notify every peer, then wait on every peer.
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

    # 3. Compute: accumulate every peer's slice.
    acc = pl.load(data, [0, 0], [1, SIZE])
    for peer in pl.range(nranks):
        if peer != my_rank:
            peer_tile = pld.tile.remote_load(
                data, peer=peer, offsets=[0, 0], shape=[1, SIZE]
            )
            acc = pl.add(acc, peer_tile)

    # 4. Stage-out: store the accumulator to local output.
    return pl.store(acc, [0, 0], out)

@pl.jit
def chip_orch(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
) -> pl.Tensor[[1, SIZE], pl.FP32]:
    # Per-device orchestration wrapper — HOST dispatches this, not the
    # InCore kernel directly.
    return reduce_step(inp, out, data, signal)

@pl.jit.host
def orchestrator(
    inputs: pl.Tensor[[NR, 1, SIZE], pl.FP32],
    outputs: pl.Out[pl.Tensor[[NR, 1, SIZE], pl.FP32]],
) -> pl.Tensor[[NR, 1, SIZE], pl.FP32]:
    data_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
    signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())

    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
        chip_orch(inputs[r], outputs[r], data, signal, device=r)
    return outputs
```

### Running It

Save the functions above and the driver below into the same file (`script.py`):

```python
import torch
from pypto.runtime import RunConfig
from pypto.ir.distributed_compiled_program import DistributedConfig

dc = DistributedConfig(device_ids=[0, 1])
cfg = RunConfig(platform="a2a3", distributed_config=dc)

inputs = torch.randn(2, 1, SIZE)
outputs = torch.zeros_like(inputs)
orchestrator(inputs, outputs, config=cfg)   # blocks until both ranks finish
```

This is the "one-shot" dispatch pattern. See [03-execution](03-execution.md)
for `DistributedWorker`, persistent workers, and multi-program dispatch.

### Expected Output

`outputs[r] == sum(inputs[*])` for every rank `r`. For 2 ranks with inputs
`[[1, 2, 3]]` and `[[10, 20, 30]]`, both ranks see `[[11, 22, 33]]`.

### Launch Command

```bash
python script.py
```

No multi-process launcher is involved — the runtime forks one worker
process per device (per `device_ids`) from this single Python process.

## What Is Distributed Programming in PyPTO?

PyPTO's distributed model is **symmetric-memory + signals**. Each rank has a
per-rank **window buffer** with symmetric address spaces across peers.
Communication happens through one-sided `put`/`get`/`remote_load` plus
**signal synchronisation** (`notify`/`wait`). A **comm domain** is a subset
of ranks sharing a symmetric window pool; the full world is the default
domain.

Every allreduce, broadcast, and barrier the compiler lowers is a composition
of these same primitives — the `pld.tensor.*` collectives (`allreduce`,
`barrier`, etc.) are syntactic sugar over them, not a separate library. The
full API is at [01-collectives](01-collectives.md).

## The Model

### HOST Orchestrator

The HOST function allocates window buffers, dispatches kernels, and manages
the control plane:

- Declared with `@pl.jit.host`
- Calls `alloc_window_buffer`, `window()`, and per-rank dispatch via `device=r`
- Runs once per process — not on the NPU
- Dispatches a per-device `@pl.jit` wrapper (not the `InCore` kernel directly)
  — see Per-Rank Dispatch below

### InCore Kernel

The InCore function runs on the NPU device:

- Declared with `@pl.jit.incore`
- Receives window-bound `DistributedTensor` arguments
- Uses `notify`/`wait` for cross-rank sync, `remote_load`/`remote_store` for RMA
- Never calls `alloc_window_buffer` or `world_size()`

### Per-Rank Dispatch

HOST never dispatches an `InCore` function directly. It dispatches a per-device
`@pl.jit` wrapper by setting `device=r` in the function call; that wrapper then
calls the `InCore` kernel with no
`device=` argument. Each rank sees its own view of the symmetric window
buffers through `CommContext`.

### Window Buffer Lifetime

`alloc_window_buffer(size)` creates per-rank buffers. `window(buf, shape, dtype)`
creates typed views. Buffers live for the duration of the host orchestrator call;
there is no persistent IPC between orchestrator invocations **by default** —
see [03-execution](03-execution.md) § `persistent=True` to retain windows
across dispatches.

### Control Plane vs Execution Plane

```text
HOST orchestrator (@pl.jit.host)
  ├── alloc_window_buffer(...)   ← control plane: declare layout
  ├── window(buf, shape, dtype)  ← control plane: create typed view
  └── for r in ranks:            ← dispatch loop
        chip_orch(..., device=r) ← bridges to the per-device wrapper

Orchestration wrapper (@pl.jit)
  └── reduce_step(...)           ← calls the InCore kernel, no device=

InCore kernel (@pl.jit.incore)
  ├── notify / wait               ← execution plane: cross-rank sync
  ├── remote_load                 ← execution plane: read peer data
  └── store                       ← execution plane: write local output
```

## Line-by-Line Walkthrough

| What | Why |
| ---- | --- |
| `NR = pl.dynamic("NR")` | The world size is not known at build time. `pl.dynamic` defers the dimension to runtime dispatch — the host binds it from `len(device_ids)`. |
| `pl.InOut[pld.DistributedTensor[...]]` | `data` and `signal` are window-bound: every rank shares the same address space layout. `InOut` means the kernel both reads and writes them. |
| `pld.get_comm_ctx(data)` | Lifts the window-bound tensor into a comm-domain handle. Every rank gets its own `ctx`, from which `rank()` and `nranks()` read per-rank values. |
| `pld.system.notify(..., op=AtomicAdd)` | Each rank atomically adds 1 to every peer's signal slot. With `offsets=[my_rank, 0]`, each cell has exactly one writer, so `Set` would work identically here — `AtomicAdd` is shown because it's the same notify call every barrier in this doc uses. It's only required for a *shared*-cell barrier where multiple ranks write the same slot; see [02-primitives](02-primitives.md) § "The Barrier in Isolation" for the distinction. |
| `pld.system.wait(..., cmp=Ge, expected=1)` | Blocks until the local signal slot reaches at least 1 — meaning all peer notifies have landed. |
| `pld.tile.remote_load(...)` | Reads a **remote** slice of a `DistributedTensor` into a local tile. This is the tile-level cross-rank equivalent of `pl.tile.load`. |
| `pl.add(acc, peer_tile)` | The local add loop sums all peer contributions. After the loop, `acc` holds `sum(inputs[*])`. |
| `chip_orch` (`@pl.jit`) | HOST dispatches this per-device wrapper via `device=r`, not the `InCore` kernel directly. It then calls `reduce_step` with no `device=` argument. |
| `inputs[r]` / `outputs[r]` | Indexing drops the leading rank dimension, giving `reduce_step` the rank-2 `[1, SIZE]` shape it declares — `pl.slice` with an explicit shape would keep the dimension and produce a rank mismatch. |

## See Also

- [05-tutorials](05-tutorials.md) — The step-by-step distributed tutorial ladder
- [01-collectives](01-collectives.md) — Built-in collectives and their semantics
- [02-primitives](02-primitives.md) — The substrate beneath the collectives
- [03-execution](03-execution.md) — DistributedWorker lifecycle and production patterns
- [04-debugging](04-debugging.md) — Common failure patterns and diagnostic flags
