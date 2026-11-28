# Execution

> **Legacy Python API examples:** `platform=` calls on this page are retained for
> compatibility. They emit `DeprecationWarning`; new code should use
> `backend_type` and `execution_mode`. See the [target mapping](../../dev/08-entry-points.md#legacy-platform-examples).

A one-off compile-and-dispatch call is enough for a quick test. Production
code instead amortizes setup — forking chip processes, assembling kernels —
across many dispatches on a reusable `DistributedWorker`.

Use `worker=` to select the logical child for a dispatch. In a HOST
orchestrator it indexes the configured devices, and in a POD orchestrator
it indexes HOST workers. `pld.num_workers()` returns the number of immediate
children. HOST code can still use `device=` and `pld.world_size()`.

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
| `compiled.prepare(config=None, *, extra_compiled=(), persistent=False, reset_persistent_windows=None, window_sources=None, callbacks=None, sub_worker_overrides=None, startup_timeout_s=None)` | Create worker, fork chip processes, return `DistributedWorker`. Use as context manager. |
| `rt(x, y, z)` | Single dispatch — coerces args, calls host_orch. |
| `rt.run(compiled, x, y, z)` | Multi-program dispatch — selects the target program. |
| `rt.submit(compiled, x, y, z)` | Bounded asynchronous dispatch — returns a `DistributedRunHandle`. |
| `rt.alloc_tensor(shape, dtype, *, init=None)` | Allocate a worker-resident `DeviceTensor`. `init` copies from host (one-time H2D). |
| `rt.free_tensor(tensor)` | Release a `DeviceTensor`. |
| `rt.copy_to(dst_dev_ptr, src_host_ptr, nbytes, *, dst_offset=0, src_offset=0, worker_id=0)` | Explicit staged H2D copy. `dst_offset` and `src_offset` address sub-ranges: copy from `src_host_ptr + src_offset` into `dst_dev_ptr + dst_offset`. `dst_offset + nbytes` must fit the device allocation. A host `torch.Tensor` source only needs to be CPU-contiguous and may be created after `prepare()`. |
| `rt.copy_from(dst_host_ptr, src_dev_ptr, nbytes, *, dst_offset=0, src_offset=0, worker_id=0)` | Explicit staged D2H copy. `dst_offset` and `src_offset` address sub-ranges: copy from `src_dev_ptr + src_offset` into `dst_host_ptr + dst_offset`. `src_offset + nbytes` must fit the device allocation. A host `torch.Tensor` destination only needs to be CPU-contiguous and may be created after `prepare()`. |
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
`StackedDeviceTensor` may also be assembled from that worker's local allocations
and remote domain views. These allocation APIs retain the Simpler owner `Buffer` on every device tensor or
shard so the address-free wire ABI can derive a valid Tensor descriptor.
Manually wrapping a raw pointer, or reusing a resident tensor with another
worker, is rejected.

## Remote Placement

Pass a `DistributedDeployment` to `compiled.prepare(deployment=...)` to place
logical ranks on local cards and daemon-hosted remote cards. Its ordered
`local_device_ids` come first, followed by `RemoteChipTarget(endpoint, device_id)`
entries. `rt.world_size` includes both. Endpoints are runtime configuration and
are not written to compiled artifact metadata.

The first remote path supports A3 Device-source Fabric domains (`a2a3`,
`comm_profile="a3-fabric-v1"`), plus the corresponding simulator profile
(`a2a3sim`, `comm_profile="sim"`). Each remote card has its own Simpler session;
logical ranks are translated to the actual worker IDs returned by Simpler.

### Device push/notify groups

Inside a `@pl.jit.host` orchestrator, use an explicit `pl.parallel` loop for
cards whose CHIP tasks exchange data and wait for each other's notifications:

```python
for rank in pl.parallel(pld.world_size()):
    chip_step(window, device=rank)
```

Each iteration contributes one direct call to the same CHIP entry and names a
distinct logical device. Scalar expressions and tensor views prepare per-card
arguments. Allocate communication domains outside the loop. The compiler emits
a group scope that collects every member's arguments and then calls Simpler's
`submit_next_level_group` once. Argument construction and validation finish
before the group is published. An empty iteration range submits no work.

Within each CHIP task, `remote_store`, `notify`, and a registration-only
`defer_wait` express device stages. Consumers depend on the waiter's deferred
completion token. The existing communication-fence pass supplies publishing
markers, while the A3 executor provides task-entry cache handling. The
application defines receive slots, signal epochs, capacity and the peer set
expected by every participant. Group admission gives all participating cards
an execution opportunity; device notifications express payload readiness.

Keep mutually waiting device stages inside one CHIP task per card. Each card
retains its single in-flight L2 task limit; later groups using that card are
scheduled after its current task completes. Independent receive slots and
signals distinguish chunks within the task.

The initial group form uses independent iterations and a single unconditional
CHIP call site. Nested loops, branches, loop-carried values and per-member
TaskId submissions receive a compiler diagnostic. Ordinary `pl.range` keeps
its existing submission behavior. Remote artifacts use entry binding version 2;
recompile artifacts for this group contract.

| API | Contract |
| --- | --- |
| `rt.allocate_domain(name, window_size=..., buffers=..., members=None)` | Bootstrap a retained Device domain before the first request. Members are logical ranks; the default is all ranks. |
| `rt.bind_domain(compiled, name, domain)` | Bind a generated scope name (for example `comm_d0`) to the allocation. The full ordered buffer layout and membership must match. |
| `with rt.request(domain): ...` | Hold runtime exclusivity across copies, HOST decisions, multiple dispatches and readback. Accepted dispatch handles finish before scope exit. |
| `rt.remote_tensor(domain, rank, buffer, shape, dtype, byte_offset=0)` | Borrow a view of a remote member's named Device buffer. Returns `RemoteDeviceTensor`, which has no `data_ptr`. |
| `rt.copy_remote_to(view, host)` / `rt.copy_remote_from(view, host)` | Synchronous, bounded-chunk copies of a CPU-contiguous tensor inside the request scope. Shape and dtype must match. |
| `rt.copy_domain_to(domain, rank, buffer, host, byte_offset=0)` / `rt.copy_domain_from(...)` | Upload or read a named local or remote member buffer inside the request scope. Rank is the logical deployment rank. |
| `rt.release_remote_tensor(view)` | Retire the view without freeing its domain-owned storage. |

The compiled object exposes `domain_layouts(world_size=..., **scalars)` before
worker startup. Each plan describes ordered logical members, named buffers,
logical and allocated bytes, byte offsets, memory source and workspace engines.
`rt.prepare_domains(compiled, **scalars)` allocates and binds Device plans without
workspace engines. Its payload window excludes the runtime-owned ordering slot.
Concrete scalar bindings resolve dynamic sizes.

`rt.share_domain(domain, [(artifact_a, scope_a), (artifact_b, scope_b)], **scalars)`
explicitly shares an allocation across prepared artifacts. It validates all
bindings before publication: ordered members, buffer names/sizes, and the full
physical window must match. Explicit bindings apply to local and remote ranks.

Use a separate Device domain for reusable parameters. `rt.domain_tensor(...)`
borrows local or remote storage; `rt.stack_domain_tensor(domain, buffer, shape,
dtype, shard_axis=..., worker_ids=...)` creates uninitialized output shards.
`rt.alloc_stacked_tensor(host, domain=..., buffer=..., shard_axis=...)` uploads
shards inside `request(domain)`. For `[round, rank, ...]`, use `shard_axis=1`;
HOST indexing `x[round, rank]` selects that rank's bounded view. Positive-step
views preserve strides; explicit copies require dense device views.
`rt.copy_stacked_from(shards, host)` restores the declared HOST axis order.
Hold all communication and parameter domains in the application request.

With remote placement, synchronous `rt.run(compiled, *cpu_args)` stages ordinary
contiguous CPU tensors through a separate Device domain. Upload happens once
per underlying storage per participating rank and call. Downloads all succeed
before modified bytes are published to the caller. Conflicting cross-rank
read/write aliases require explicit communication buffers; span checks are
conservative for strided views. Use resident shards for large reusable weights.
Asynchronous `submit()` uses resident arguments. Scalar HOST parameters use
Python scalar values.

New distributed manifests retain generated-source hashes, layout plans and the
compiler's CommContext ABI. `compiled.validate_artifacts()` checks source hashes;
`prepare()` also compares the ABI with Simpler's native `COMM_CONTEXT_ABI` query.
Rebuild matching Simpler and PyPTO extensions when that query is unavailable or
the layouts differ. Cold reload restores these contracts.

Supply `CommBufferSpec` values from `simpler.task_interface` with physical sizes
aligned to 64 bytes. `window_size` covers the application buffers; PyPTO appends
a reserved 64-byte `__pypto_order` buffer and adds it to the physical window.
This member-local buffer provides stable DAG ordering without controller HOST
memory. Every stage bound to one allocation must declare the same ordered
layout, including padding. Generated communication contexts carry domain identity;
the target L2 binds its own device pointer.

Remote placement enables retained-domain execution and disables automatic
window resets. Initialize application data and status explicitly inside
`request()`. Conflicting requests fail through Simpler's exclusive domain scope;
the scope is not another TCP connection. On uncertain communication teardown,
the runtime retains resources rather than releasing a window still used by peers.

Remote tensor storage comes from retained Device-domain buffers. Upload data
through the domain copy APIs and pass a `RemoteDeviceTensor` view. The raw-pointer
allocation and memory-query helpers cover local cards. Simpler's generic remote
allocator supplies HOST staging; domain views carry the device-resident identity.
Mixed
`StackedDeviceTensor` values may contain locally allocated `DeviceTensor` shards
and these remote views, with logical rank IDs matching the dispatch placement.

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
an upload source or a read-back destination. Anything not listed, or allocated after
`prepare()`, keeps staging exactly as before.

**One allocation, one Buffer.** A listed tensor names its whole *storage*, not the extent
of the view you passed. Every view of one storage therefore collapses to a single runtime
Buffer and reaches its own bytes by offset — list `w`, `w[:2]` and `w[2:]` and you still
get one. This is what keeps the number of Buffers a property of the memory rather than of
which views you happened to list; a per-view Buffer would name the same bytes twice, and
one identity may name only one backing.

### Read-only backings

One Buffer carries one access mode, fixed when it is created, so a listed allocation is
declared writable by default — that is what a read-back destination needs. Wrap an entry
in `ReadOnlyHostTensor` when the backing is shared but **not** writable, typically a
`MAP_SHARED` mapping of a read-only file descriptor:

```python
from pypto.runtime import ReadOnlyHostTensor

with compiled.prepare(
    inherited_host_tensors=[kv_cache, ReadOnlyHostTensor(weights)],
) as rt:
    ...
```

That allocation's Buffer is then declared `READ`, and a `copy_from` into it raises a
`ValueError` instead of leaving the forked child to fault on the store. Writability cannot
be inferred any more than visibility can — torch erases it, a write probe faults rather
than raising, and `/proc/self/maps` is Linux-only — so marking is a caller guarantee, just
like listing the tensor in the first place.

Listing one allocation both read-only and writable raises `ValueError`: one allocation has
one access mode, and neither narrowing nor widening it silently is recoverable.

The marker governs the named-copy descriptor only. Dispatch arguments are named by the
runtime's own path, which this does not reach, so marking a dispatch IO buffer read-only
does not stop the child from writing it.

## One-Shot vs Persistent Worker

### One-Shot

```python
import torch
from pypto.ir import DistributedConfig
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

### Host window file sources

Pass `window_sources={"engram_l1": shard_files}` to `compiled.prepare()` or
`DistributedWorker` for a window declared with `memory_source="host", init="engram_l1"`.
Open the files **after compilation**, in domain rank order. Initialization reads each
file from offset zero, restores its position, and copies the logical window size.
The caller owns the files. With `persistent=True`, initialization runs once; Host
and Device windows are both zeroed before reuse by default. Set
`reset_persistent_windows=False` to preserve the loaded table and manage the state
of all reused communication buffers yourself. Resetting a domain does not rerun
its initializer.
See [Host DDR windows](../../dev/passes/50-materialize_comm_domain_scopes.md)
for the full lifecycle and error contract.

The rank list may also contain contiguous CPU `torch.Tensor` objects, for example
`window_sources={"engram_l1": [rank0_tensor, rank1_tensor]}`, or mix files and tensors.
Tensor sources are copied as raw bytes at domain creation, with no dtype conversion.
They must contain at least the logical window size. Keep their storage and contents
stable until initialization completes. Runtime capability checks run during worker
setup rather than on each submit.

## Sharing an external table between programs

Use external window arguments with an all-local deployment. For mixed local and remote
placement, bind compiled domain layouts with `share_domain`.

Prepare the ordinary inputs and outputs before constructing the worker. Both HOST entries declare a static `pld.DistributedTensor[[1, 65], pl.UINT8]` table argument. Rank sources follow the declared member order. Here each source supplies 65 bytes and each physical slot occupies 128 bytes.

```python
from pypto import DataType
from pypto.runtime import DistributedWorker, DomainBufferSpec

with DistributedWorker([prefill, decode]) as worker:
    with worker.create_comm_domain(
        members=[0, 1],
        buffers=[DomainBufferSpec("weights", (1, 65), DataType.UINT8)],
        memory_source="host",
        window_sources={"weights": [rank0_file, rank1_file]},
    ) as domain:
        weights = domain.tensor("weights")
        worker.submit(prefill, weights, prefill_output).result()
        worker.submit(decode, weights, decode_output).result()
```

`members` lists local worker indices in domain-rank order. Bind the resulting
windows on their creating worker. For the default `memory_source="device"`,
omit `window_sources` and populate each window with a CHIP task. A5 Host
asynchronous peer GET requires `workspace_engines=("urma",)` and the native
runtime's dense member order starting at worker zero. Leaving the domain's
`with` block waits for accepted work and releases its allocations.
