# Execution

A one-off compile-and-dispatch call is enough for a quick test. Production
code instead amortizes setup — forking chip processes, assembling kernels —
across many dispatches on a reusable `DistributedWorker`.

## DistributedWorker

Obtained via `compiled.prepare()`. Setup (fork, comm bootstrap, kernel assembly)
happens once; dispatch happens many times.

```python
with compiled.prepare() as rt:
    rt(host_x, host_out)
    # ... more dispatches ...
# rt.close() runs on exit — releases buffers and shuts down workers.
```

`compiled.prepare()` returns a `DistributedWorker` (also constructible
directly via `DistributedWorker(compiled)`, importable from
`pypto.runtime`, but `prepare()` is the documented entry point).

### Methods

| Method | Description |
| ------ | ----------- |
| `compiled.prepare(config=None, *, extra_compiled=(), persistent=False, reset_persistent_windows=None, callbacks=None, sub_worker_overrides=None, startup_timeout_s=None)` | Create worker, fork chip processes, return `DistributedWorker`. Use as context manager. |
| `rt(x, y, z)` | Single dispatch — coerces args, calls host_orch. |
| `rt.run(compiled, x, y, z)` | Multi-program dispatch — selects the target program. |
| `rt.submit(compiled, x, y, z)` | Bounded asynchronous dispatch — returns a `DistributedRunHandle`. |
| `rt.alloc_tensor(shape, dtype, *, init=None)` | Allocate a worker-resident `DeviceTensor`. `init` copies from host (one-time H2D). |
| `rt.free_tensor(tensor)` | Release a `DeviceTensor`. |
| `rt.copy_to(dst_dev_ptr, src_host_ptr, nbytes, *, worker_id=0)` | Explicit staged H2D copy. A host `torch.Tensor` source only needs to be CPU-contiguous and may be created after `prepare()`. |
| `rt.copy_from(dst_host_ptr, src_dev_ptr, nbytes, *, worker_id=0)` | Explicit staged D2H copy. A host `torch.Tensor` destination only needs to be CPU-contiguous and may be created after `prepare()`. |
| `rt.alloc_stacked_tensor(host_w)` | Shard host_w along dim 0 — shard `i` uploaded to card `i`. Returns `StackedDeviceTensor`. |
| `rt.free_stacked_tensor(stacked)` | Release all shards of a `StackedDeviceTensor`. |
| `rt.copy_stacked_from(stacked, host_out)` | Staged D2H read-back of every shard into a CPU-contiguous `host_out`; it may be allocated after `prepare()`. |
| `rt.committed_device_memory(worker_id=0)` | Device HBM (bytes) this worker's own allocator has committed on card `worker_id` — tensors, pooled arenas, runtime buffers. Sum across ids for a multi-chip total. |
| `rt.device_memory_info(worker_id=0)` | `(free_bytes, total_bytes)` for the whole card `worker_id` runs on, as the driver sees it. Use this to size an allocation; anything else on the card moves `free_bytes` without moving the committed total. Raises on simulator backends rather than reporting zeros. |
| `rt.release_inherited_host_tensor_refs()` | Drop compatibility lifetime references retained in the parent process. |
| `rt.close()` | Release buffers, shut down chip workers. Called automatically as context manager. |

### `prepare()` Parameters Worth Knowing

- **`config`** — an optional `RunConfig` used *only* to pre-warm the
  runtime arena cache for a given ring sizing, so the first dispatch
  skips its ~800 ms cold build. It is **not retained**: every dispatch
  still needs its own `config=`. The prewarm only pays off when the
  *first* dispatch's sizing matches the pre-warmed one — see
  `docs/en/dev/05-runtime-ring-sizing.md` § arena prewarm.
- **`persistent=True`** — retains CommDomain windows for the worker's
  entire lifetime instead of allocating/releasing them on every dispatch.
  Pairs with **`reset_persistent_windows`**, which controls whether
  retained windows are zeroed between requests (a correctness-vs-overhead
  trade-off). See `docs/en/dev/06-persistent-l3.md`.
- **`extra_compiled`** — see "Several Programs on One Worker" below.
- **`startup_timeout_s`** — optionally overrides Simpler's positive finite
  startup-readiness deadline for the forked worker hierarchy. Leave it as
  `None` to keep Simpler's default; increase it for legitimately slow cold
  starts rather than disabling the bound.

### Bounded Asynchronous Dispatch

`DistributedWorker.submit(compiled, *args)` returns a
`DistributedRunHandle` after Simpler accepts the dispatch. When the backend
supports asynchronous execution, the caller can prepare host work for the next
request while the current request is still running. `run()` and `rt(...)`
remain blocking compatibility wrappers.

The worker owns exactly two reusable dispatch metadata frames. The first two
submissions may be in flight together; a third `submit()` waits for the oldest
handle before it constructs or publishes another dispatch. Each handle
snapshots its runtime configuration and retains its arguments and generated
task metadata until completion.

Use `handle.result(timeout)` or its alias `handle.wait(timeout)` to wait and
raise the cached dispatch error. `handle.done` reports terminal completion
without blocking.

```python
with compiled.prepare() as rt:
    first = rt.submit(compiled, input_a, weight, output_a)
    second = rt.submit(compiled, input_b, weight, output_b)
    first.result()
    second.result()
```

Overlapping dispatches require distinct mutable input and output buffers. Do
not modify or release those buffers before the corresponding `result()`
returns. Read-only resident weights may be shared. Closing the worker drains
all accepted handles in FIFO order. Diagnostic two-pass swimlane capture stays
synchronous and returns an already-completed handle.

### Resident Tensor Ownership

Resident arguments are supported only on a prepared worker. A `DeviceTensor`
must be returned by that same `DistributedWorker`'s `alloc_tensor`, and a
`StackedDeviceTensor` must be returned by its `alloc_stacked_tensor`. These
allocation APIs retain the Simpler owner `Buffer` on every device tensor or
shard so the address-free wire ABI can derive a valid Tensor descriptor.
Manually wrapping a raw pointer, or reusing a resident tensor with another
worker, is rejected.

## DeviceTensor

A worker-resident buffer that lives on the device across dispatches.
When a `DeviceTensor` is passed as an argument to a compiled program,
the runtime skips H2D/D2H copies — the device already has the data.

```python
import torch

with compiled.prepare() as rt:
    weight = rt.alloc_tensor((1024, 4096), torch.float16, init=host_weight)
    rt(x, weight, out)   # dispatch via worker — no H2D/D2H
```

## StackedDeviceTensor

Sharded across devices — obtained via `rt.alloc_stacked_tensor()`:

```python
# Host tensor sharded along dim 0 — shard[i] lives on card i.
with compiled.prepare() as rt:
    host_weights = torch.randn(4, 1024, 4096).contiguous()  # post-prepare host tensor
    stacked = rt.alloc_stacked_tensor(host_weights)
    rt(x, stacked, out)
```

Explicit resident upload and copy-back through `rt.alloc_tensor(init=...)`,
`rt.alloc_stacked_tensor(...)`, `rt.copy_to(...)`, `rt.copy_from(...)`, and
`rt.copy_stacked_from(...)` stage through runtime-owned POSIX shared memory.
When a host endpoint is a `torch.Tensor`, it only needs to be CPU-contiguous;
it may be an ordinary tensor allocated after `prepare()`. It does not need
`.share_memory_()`, pre-fork allocation, or `inherited_host_tensors`.

### Skipping the staging copy

Staging costs one full host-side copy of the payload, which for a large resident weight
is worth avoiding. A range registered through `inherited_host_tensors` is named in place
instead — no staging buffer, no memcpy — because it predates the fork and every child
holds it at the same address.

**Listing a tensor is a guarantee you make.** By passing it in
`inherited_host_tensors` you assert two things:

- its backing is **visible across processes** — a `MAP_SHARED` mapping, whether torch's
  own shared memory or an external file mapping; and
- that mapping stays **valid for the worker's lifetime**.

Passing a `MAP_PRIVATE` backing is **unsupported**. Copy-on-write leaves the child reading
its pre-fork snapshot, so an upload may carry stale or incorrect data. Nothing detects
this for you.

PyPTO cannot verify the guarantee, and does not try. `torch.is_shared()` answers a
different question — whether the storage is a torch shared-memory allocation — and a
read-only `MAP_SHARED` file mapping wrapped with `mmap` + `numpy.frombuffer` +
`torch.from_numpy` is genuinely shared while reporting `False`. Reading
`/proc/self/maps` would answer correctly but only on Linux, and the simulator also runs
on macOS. So `is_shared()` is kept as a one-way signal: `True` confirms a torch-managed
shared backing, `False` is inconclusive, and for the tensors it cannot confirm PyPTO
emits one `RuntimeWarning` at `prepare()` time and proceeds. It never rejects the tensor
and never falls back to staging — falling back would silently reinstate the copy you
asked it to skip.

```python
weights = map_readonly_shared(path)          # mmap-backed MAP_SHARED, is_shared() == False
with compiled.prepare(
    inherited_host_tensors=[weights],        # "visible in every child, for my lifetime"
) as rt:
    resident = rt.alloc_stacked_tensor(weights)
```

The guarantee is about visibility, so it holds in both directions: a listed range may be
an upload source or a read-back destination. Writability is left to the hardware — a range
mapped `MAP_SHARED` from a read-only file descriptor faults on write rather than corrupting
anything. Anything not listed, or allocated after `prepare()`, keeps staging exactly as
before.

A named upload source is granted `READ`; only a destination is granted `READWRITE`. That
matters for a read-only mapping: describing it as writable would tell the runtime a
consumer may write memory that faults on a write.

## One-Shot vs Persistent Worker

### One-Shot

```python
import torch
from pypto.ir.distributed_compiled_program import DistributedConfig
from pypto.runtime import RunConfig

dc = DistributedConfig(device_ids=[0, 1, 2, 3])
cfg = RunConfig(platform="a2a3", distributed_config=dc)
compiled = orchestrator.compile(config=cfg)   # reads shapes from orchestrator's own annotations

inputs = torch.randn(4, 1, 256)
outputs = torch.zeros_like(inputs)
compiled(inputs, outputs)   # blocks until all ranks finish
```

One-shot execution accepts host `torch.Tensor` parameters only. It rejects
`DeviceTensor` and `StackedDeviceTensor`; use a prepared worker for either
resident type.

### Persistent Worker (Repeated Dispatch)

Reusing the same worker object across many dispatches — the default
lifecycle for any `DistributedWorker`. (Not to be confused with the
`persistent=True` CommDomain-window-retention flag above — that's an
opt-in that skips per-dispatch window alloc/release; amortizing the
fork/comm-bootstrap cost shown here happens regardless of `persistent=`.)

```python
host_x = torch.zeros((4, 1, 256), dtype=torch.float32).share_memory_()
host_out = torch.zeros_like(host_x).share_memory_()

with compiled.prepare() as rt:
    for step in steps:
        host_x.copy_(next_input(step))
        rt(host_x, host_out)
        consume(host_out)
```

> **Fatal pitfall:** host `torch.Tensor` arguments passed directly to
> `rt(...)` or `rt.run(...)` must call `.share_memory_()` before `prepare()`.
> If you forget, the runtime rejects the buffer at dispatch time — the child
> processes cannot access the parent's private memory. This rule does not
> apply to the explicit staged upload/copy-back APIs listed above.

## Several Programs on One Worker

A single `DistributedWorker` can dispatch multiple compiled programs:

```python
compiled_a = ir.compile(ProgramA, platform="a2a3", distributed_config=dc)
compiled_b = ir.compile(ProgramB, platform="a2a3", distributed_config=dc)

with compiled_a.prepare(extra_compiled=[compiled_b]) as rt:
    rt.run(compiled_a, host_x, host_out)  # dispatch ProgramA
    rt.run(compiled_b, host_x, host_out)  # dispatch ProgramB
```

The worker reuses its chip processes and comm setup — no fork penalty.
`compiled_b` must be passed via `extra_compiled=` for `rt.run(compiled_b, ...)`
to find it; passing an unregistered program raises `ValueError`. Preparing
more than one program also puts the worker in multi-program mode, where the
`rt(*args)` shortcut is ambiguous and raises `TypeError` — dispatch every
program explicitly through `rt.run(...)`, including the primary one.

## CLI Launch

Launching a distributed program is identical to launching a single-device
one — see [00-model § Launch Command](00-model.md#launch-command): plain
`python script.py`, no separate multi-process launcher.

## Environment Variables

### Compile-Time Macros

These are C preprocessor `#define` macros in `profiling_config.h`, **not environment variables**.
They default to `1` (enabled) and are set at build time via CMake flags. Setting them as shell
env vars has no effect.

| Macro | Default | Effect |
| ----- | ------- | ------ |
| `SIMPLER_HOST_STRACE` | `1` (on) | Required at build time for `benchmark()` timing markers. Without it, `benchmark()` raises `RuntimeError`. |
| `SIMPLER_DFX` | `1` (on) | Umbrella gate for device-side profiling (orchestrator/scheduler metrics, PMU counters, scope stats, swimlane trace). Sub-tier flags require this to be `1`. |

### Runtime Environment Variable

| Variable | Default | Effect |
| -------- | ------- | ------ |
| `SIMPLER_DEVICE_STRACE_ENABLE` | on (unset or non-`"0"`) | Runtime toggle for device-domain `[STRACE]` markers. Set to `0` to suppress device markers while keeping host markers. |

### Benchmark Env Vars

The `pypto-lib` golden benchmark harness reads `PYPTO_BENCH` /
`PYPTO_BENCH_ROUNDS` / `PYPTO_BENCH_WARMUP` / `PYPTO_BENCH_RAW` — these are
not defined or consumed anywhere in this repository. See `pypto-lib`'s own
documentation for current defaults. `pypto.runtime.benchmark()` (this
repo's own harness) is documented separately in the performance guide.

## Worked example

`examples/runtime/distributed_callback.py` — a HOST-level `SubWorker` whose body is `...`,
so it runs as a pure-Python callback in the forked orchestrator process. That is the shape
to reach for when the logic cannot be written at compile time: a sampling closure over live
model state, a host-side metric collector, a result inspector.

## See Also

- [00-model](00-model.md) — Quickstart and model vocabulary
- [04-debugging](04-debugging.md) — Common failure patterns
- [Getting Started](../00-getting_started.md) — Runtime setup
