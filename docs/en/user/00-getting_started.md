# Running on Device

Keeping data resident on a worker, dispatching explicitly, measuring, and running
distributed programs.

> **This page is in transition.** Its introductory material moved to
> [Quickstart](02-quickstart.md). What remains is device-execution and runtime
> material, which will move again once the `execution/`, `performance/`, and
> `distributed/` chapters land:
>
> | Section | Destination |
> | ------- | ----------- |
> | Resident device tensors, explicit dispatch, compiling from a signature | `execution/01-run.md` |
> | Per-launch timing, `benchmark` | `performance/06-host.md` |
> | Distributed (L3+) execution | `distributed/03-execution.md` |
>
> Nothing here is deprecated — only the address is temporary.
>
> **Prerequisites:** [Quickstart](02-quickstart.md), and a machine with a device or a
> simulator platform. Unlike the quickstart, the examples below dispatch to hardware.

## Reusing weights on the worker (DeviceTensor)

When the same large tensor is consumed by many kernel invocations — e.g. a
weight matrix used across batches of a forward pass — uploading it on every
call wastes bandwidth. `ChipWorker.alloc_tensor` allocates persistent device
memory and returns a `DeviceTensor` handle that `CompiledProgram` accepts in
place of a `torch.Tensor`. The runtime treats the buffer as already resident
and skips both H2D and D2H copies for that argument.

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto import ir
from pypto.runtime import ChipWorker, RunConfig

ROWS, COLS = 128, 128
PLATFORM = "__PLATFORM__"


@pl.jit
def add_kernel(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        ta = pl.load(a, [0, 0], [ROWS, COLS])
        tb = pl.load(b, [0, 0], [ROWS, COLS])
        pl.store(pl.add(ta, tb), [0, 0], out)
    return out


torch.manual_seed(0)
A = torch.randn(ROWS, COLS, dtype=torch.float32)
B = torch.randn(ROWS, COLS, dtype=torch.float32)

# A DeviceTensor carries no shape/dtype for the @pl.jit specializer to read, so
# the resident-weight pattern below runs a *compiled* program rather than the
# jit entry directly.
compiled = ir.compile(add_kernel.lower(A, B, torch.zeros(ROWS, COLS)), platform=PLATFORM)
```

<!-- doctest: run -->
```python
cfg = RunConfig(platform=PLATFORM)

with ChipWorker(config=cfg) as w:
    resident = w.alloc_tensor((ROWS, COLS), torch.float32, init=B)  # stays on device
    for _ in range(3):                                              # three "batches"
        out = torch.zeros(ROWS, COLS, dtype=torch.float32)
        w.run(compiled, A, resident, out)
        torch.testing.assert_close(out, A + B, rtol=1e-4, atol=1e-4)
    w.free_tensor(resident)
```

### Caveats

- A `DeviceTensor` is never copied back to the host. If a kernel writes to
  one, call `w.copy_from(host_ptr, t.data_ptr, t.nbytes)` on the same
  ChipWorker instance to read the result.
- Free the handle with `w.free_tensor(t)` before the ChipWorker is closed,
  otherwise the memory leaks for the lifetime of the ChipWorker.
- Only the ChipWorker instance that allocated the buffer can use it.

### Explicit dispatch (`worker.run`, `worker.register`)

The implicit `with ChipWorker(): compiled(...)` pattern shown above relies on
`ContextVar` discovery: any `compiled(...)` call inside the block finds the
active worker and reuses it. That's convenient for scripts but leaves the
worker hidden — library code that needs to pass the worker around, or a
serving runtime that wants to pre-register many kernels, should drive
dispatch explicitly:

<!-- doctest: run -->
```python
worker = ChipWorker(config=RunConfig(platform=PLATFORM))
try:
    handle = worker.register(compiled)               # eager registration
    out = torch.zeros(ROWS, COLS, dtype=torch.float32)
    for _ in range(3):                               # hot loop, no cid lookup
        handle(A, B, out)
    torch.testing.assert_close(out, A + B, rtol=1e-4, atol=1e-4)
finally:
    worker.close()                                   # cids + DeviceTensors released
```

`worker.register(compiled)` triggers `compile_and_assemble` + simpler
`register` immediately, so configuration errors surface here rather than on
first dispatch. The returned `RegistrationHandle` is callable, supports
`with handle:` for scoped cleanup, and exposes `handle.unregister()` for
explicit early release. Multiple `register` calls for the same
`compiled.chip_callable` return aliases of the same cid; the underlying
simpler unregister runs once, in `worker.close()`.

For `@pl.jit` kernels, the same flow works via `JITFunction.compile()`:

```python
@pl.jit
def add_kernel(a, b, out): ...

compiled = add_kernel.compile(sample_a, sample_b, sample_out)
handle = worker.register(compiled)
for batch in stream:
    handle(batch.a, batch.b, batch.out)
```

`compile()` only reads each tensor argument's shape/dtype — contents are never
touched — so the sample tensors are pure metadata carriers.

### Compiling from the signature (no sample tensors)

When every tensor parameter is **fully annotated** with its shape and dtype,
`compile()` can read the whole shape contract straight from the signature — call
it with **no positional arguments** and skip the sample tensors entirely:

```python
HIDDEN, VOCAB = 4096, 152064
M = pl.dynamic("M")          # runtime-dynamic dim

@pl.jit
def prefill_fwd(
    hidden: pl.Tensor[[M, HIDDEN], pl.BF16],
    lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[M, VOCAB], pl.FP32]],
): ...

# No torch.empty(...) dummies — shapes come from the annotations.
compiled = prefill_fwd.compile()
```

This is the ergonomic path for kernels with large signatures: the shape contract
lives in one place (the signature) instead of being re-declared as a list of
throwaway `torch.empty(...)` buffers. Details:

- **Static dims** (`HIDDEN`, `VOCAB`, …) come from the annotation constants.
- **Dynamic dims** (`pl.dynamic` / `bind_dynamic`) need no value — the compiled
  artifact is extent-independent, and `compile()` shares one cache entry with an
  equivalent `compile(sample_tensors)` call.
- **Scalar parameters** carry no value in the signature — pass them as keyword
  args. A literal **specializes** the value into the artifact, e.g.
  `kernel.compile(num_tokens=128)` compiles a kernel that only ever sees 128.
  Pass `pl.RUNTIME` instead — `kernel.compile(num_tokens=pl.RUNTIME)` — to leave
  the parameter **unspecialized**: it stays a real `pl.Scalar` parameter whose
  value arrives at dispatch and, like a dynamic dim, drops out of the cache key,
  so one artifact serves every value. Supply that value through the compiled
  artifact — `compiled(...)` or a `worker.register(compiled)` handle — not by
  calling the kernel eagerly: `kernel(x, out, 128)` re-specializes on 128 and
  compiles a separate artifact. `pl.RUNTIME` also works as the signature default
  (`num_tokens: pl.Scalar[pl.INT32] = pl.RUNTIME`), which makes the keyword
  unnecessary at every `compile()` call site.
- A **bare `pl.Tensor`** parameter (no shape) has nothing to read and raises a
  clear error; give it a full `pl.Tensor[[...], dtype]` annotation, or fall back
  to `compile(*sample_tensors)`.

See `examples/runtime/explicit_dispatch.py` for three end-to-end patterns
(inference service, training loop, register/dispatch overhead check).

### Reading per-launch timing

`worker.run` / `handle(...)` return tensor outputs only and no longer surface
a per-launch timing object. The runtime emits per-run host/device timing as
`[STRACE]` log markers (simpler PR #1177, on by default under
`SIMPLER_DFX`); parse them with simpler's `strace_timing` /
`device_log_timing` tools rather than reading a return value. For per-task
device timing, enable the L2 swimlane DFX (`RunConfig(enable_chip_swimlane=True)`)
and read `chip_swimlane_records.json`.

### Benchmarking (`benchmark`)

For the register-once + rounds pattern, `pypto.runtime.benchmark` owns the loop
and aggregation: it registers *compiled* once and dispatches `rounds` cheap
launches (no per-round register/load), reads each launch's `[STRACE]` markers,
and returns a `BenchmarkStats`:

```python
from pypto.runtime import benchmark

stats = benchmark(compiled, [a, b, c], rounds=100, warmup=3,
                  platform="a2a3", device_id=0)
print(stats.device_wall_us_median, stats.device_wall_us_min, len(stats.samples))
```

Pass `platform=` / `device_id=` for the common case, or a full `RunConfig` via
`config=` for `aicpu_thread_num` control (not both). Aggregates are
exposed under both `device_wall_us_*` and shorter `device_us_*` names, with
`samples` aliasing the raw `device_wall_us` list.

`compiled` does not have to come from a fresh compile: `CompiledProgram.from_dir(work_dir)`
rebuilds a benchmarkable program from an existing build directory, so you can
hand-edit the generated orchestration cpp / `.pto` and re-measure without
recompiling — pair it with the `.pto` rebuild and binary-cache invalidation
shown in
[Benchmarking a replayed single-chip build](../dev/03-runtime-replay.md#benchmarking-a-replayed-single-chip-build),
or your edit may not be compiled.

`benchmark` reads timing from the `[STRACE]` markers (simpler PR #1177): it
sets the runtime log level to `timing` for the worker's lifetime and captures
`stderr` at the fd level around the measured loop, so stderr emitted during the
loop is diverted into a temp file rather than shown live. `device_wall_us` is a
real on-NPU wall for L2 single-chip runs (see the L3 note below for distributed
programs); it is `0` on runtimes built without `SIMPLER_HOST_STRACE` or on `*sim`
platforms (check `stats.all_zero_device`).

Beyond the aggregates, each measured launch keeps its full `[STRACE]` span tree
on `stats.invocations` (a list of `TraceInvocation`; warmup excluded). Render it
with branch connectors — one launch, or averaged across all launches with a
per-node spread (`spread` is `"stdev"` (default), `"minmax"`, `"both"`, or
`"none"`):

```python
stats.print_tree(launch=0)            # one launch's nested span tree
stats.print_mean_tree(spread="both")  # mean per node, with ±stdev and [min..max]
```

```text
mean of 20 launches (warmup 5 excluded); each node: mean ±stdev [min..max]:
chip.run                   71784.1us  ±6797.5  [66482.4..89832.6]
|- bind                    27943.6us  ±4163.7  [24836.7..37713.3]
|- runner_run               3030.8us   ±184.4    [2822.3..3694.7]
|  `- device_wall [dev]     2005.2us    ±74.6    [1875.1..2173.2]
|     `- graph_build [dev]  1634.8us    ±64.6    [1490.2..1777.6]
`- validate                40697.7us  ±3063.5  [38606.3..48200.6]
```

Nesting is reconstructed from the dotted span names, so device-domain spans
(`...device_wall.*`, tagged `[dev]`) nest under their host parent. Each node is a
wall-clock window, *not* a partition: children may overlap (e.g. `orch`/`sched`
run concurrently) or sit in a different clock domain (`runner_run` host wall vs
`device_wall` on-NPU), so child durations need not sum to the parent. Drill into
raw spans via `stats.invocations[i].by_name()[<name>].dur_us`.

`benchmark` also accepts an L3 `DistributedCompiledProgram` and opens its own
prepared worker. Pass shared-memory host tensors; an externally allocated
resident tensor belongs to a different worker and is not accepted. Omit
`platform=` / `device_id=` (the device set is fixed at compile time via
`distributed_config`). L3 has no single DAG-level device wall, so timing is
folded from the per-rank chip-child markers into per-round samples — the headline
`device_wall_us[k]` is the max across ranks of each rank's summed dispatch device
walls. Query the four metrics uniformly:

```python
stats.per_round("device" | "host" | "effective" | "union")  # -> [one value per round]
stats.per_rank("device" | "host" | "effective")             # -> {pid: [one per round]}
stats.per_dispatch("device" | "host" | "effective")         # -> {(pid, slot): [one per round]}
```

`per_round` / `per_rank` aggregate **per rank per round**: each entry sums that
rank's dispatches within the round (a card runs its dispatches serially), so they
are per-round-per-rank busy figures, **not** per-dispatch.

`per_dispatch` is the un-fused view — it sums nothing. It keys on `(pid, slot)`,
where `slot` is the dispatch's position within its rank's round. A rank that
issues several dispatches per round therefore keeps one series per dispatch
instead of a single summed number.

A slot only identifies a dispatch if the rank issues the same callables in the
same order every round. A constant dispatch count does not guarantee that, so
the parse checks it: if any slot carried more than one task across the rounds,
`stats.unstable_dispatch_slots` is set and the per-dispatch views report empty
rather than averaging distinct kernels under the first round's label. Round
boundaries are unaffected, so `per_rank` / `per_round` stay valid.
`stats.dispatch_tasks()` labels each slot with the orchestration function it
runs, and `stats.dispatch_groups()` returns the underlying `TraceInvocation` per
round.

```python
stats.per_dispatch("device")   # {(4242, 0): [4.1, 3.8, ...], (4242, 1): [6.3, 6.5, ...]}
stats.dispatch_tasks()         # {(4242, 0): "prefill_orch", (4242, 1): "decode_orch"}
```

The markers themselves carry no name — only `hid`, the ELF Build-ID of the
callable's orchestration `.so` (still available as `TraceInvocation.task`). pypto
recovers the name by recomputing that Build-ID over the same `.so` bytes it hands
the runtime, and pairing it with that orchestration's generated name at assemble time
(`TraceInvocation.task_name`). The label falls back to the raw hash when the
pairing is unavailable: on `*sim` platforms (whose host seeds `hid` with the
runtime `callable_id` instead of a Build-ID), or for a callable assembled in a
different process.

The mean-tree views are dispatch-aware too: on an L3 run
`print_mean_tree()` renders **one tree per `(pid, slot)`** rather than one tree
averaging a rank's different kernels together, and `pid=` / `slot=` narrow it to a
single dispatch. `format_tree()`'s launch headers carry `round=` / `slot=` for the
same reason. `mean_invocation()` returns a single tree, so it raises unless
`pid=` / `slot=` select one dispatch.

`effective` is the orch∪sched on-device window (per-card L2 Effective); `union`
is the cross-rank host-timeline window (captures start skew — host-domain, so it
includes dispatch overhead). The navigable `round -> rank -> [dispatch]` grid is
`stats.rounds_dispatches`, where each `TraceInvocation` exposes `.task` (callable
id), `.device_wall_us`, `.host_wall_us`, `.effective_us`. A pure-device
cross-rank end-to-end wall is not recoverable from the markers today. If the
dispatch shape is non-deterministic, `stats.fallback_flattened` is set and the
per-rank / `union` views are empty.

### Distributed (L3+) programs

The complete distributed programming model — from `alloc_window_buffer` through
collectives like `allreduce`, `barrier`, and `broadcast` — is covered in the
[Distributed Programming](distributed/00-model.md). Here is the mesh allreduce
Hello World, shown as the InCore kernel (execution plane); a fully runnable
program also needs the host orchestrator, `ir.compile`, and distributed worker
setup — see the guide above:

```python
import pypto.language as pl
import pypto.language.distributed as pld

NR = pl.dynamic("NR")

@pl.program
class HelloAllReduce:
    @pl.function(type=pl.FunctionType.InCore)
    def reduce_step(
        self,
        inp: pl.Tensor[[1, 256], pl.FP32],
        out: pl.Out[pl.Tensor[[1, 256], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[1, 256], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
    ) -> pl.Tensor[[1, 256], pl.FP32]:
        ctx = pld.get_comm_ctx(data)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)

        # 1. Stage-in: copy local input into this rank's window slice.
        data = pl.store(pl.load(inp, [0, 0], [1, 256]), [0, 0], data)

        # 2. Barrier: notify every peer, then wait on every peer.
        for peer in pl.range(nranks):
            if peer != my_rank:
                pld.system.notify(signal, peer=peer, offsets=[my_rank, 0],
                                  value=1, op=pld.NotifyOp.AtomicAdd)
        for src in pl.range(nranks):
            if src != my_rank:
                pld.system.wait(signal, offsets=[src, 0],
                                expected=1, cmp=pld.WaitCmp.Ge)

        # 3. Compute: load own slice, remote-load every peer, accumulate.
        acc = pl.load(data, [0, 0], [1, 256])
        for peer in pl.range(nranks):
            if peer != my_rank:
                peer_tile = pld.tile.remote_load(
                    data, peer=peer, offsets=[0, 0], shape=[1, 256])
                acc = pl.add(acc, peer_tile)

        # 4. Stage-out: store accumulator to output.
        out = pl.store(acc, [0, 0], out)
        return out
```

The guide includes a **line-by-line walkthrough, ring allreduce trade-offs,
notify/wait handshake patterns, and a debugging table**. The full chapter is
at [distributed/index.md](distributed/index.md).

L3+ resident tensors must be allocated by the same prepared
`DistributedWorker` that dispatches them: use its `alloc_tensor` for a
`DeviceTensor` or its `alloc_stacked_tensor` for a `StackedDeviceTensor`.
These APIs retain the Simpler owner `Buffer` on every tensor or shard, as
required by the address-free wire ABI; a manually built raw-pointer tensor
cannot safely cross that boundary. One-shot `compiled(...)` calls accept host
`torch.Tensor` parameters only and reject both resident types.

```python
import torch
compiled = ir.compile(MyDistributedProgram)   # returns DistributedCompiledProgram
with compiled.prepare() as rt:
    weight = rt.alloc_tensor((1024, 4096), torch.float16, init=host_weight)
    rt(x, weight, out)                         # weight: no per-dispatch H2D/D2H
```

#### Reusing setup across dispatches (`prepare()`)

`compiled(*args)` runs the full distributed setup (per-chip assembly, simpler
Worker construction + fork) on every call. For a resident service that
dispatches the same program many times (e.g. a generate loop), call
`compiled.prepare()` once to get a `DistributedWorker` handle that runs setup
once and dispatches many times on the same worker.

Per-call dispatch IO buffers are **shared-memory host tensors allocated before
`prepare()`** and reused in place, so child writes are visible to the parent.
Explicit resident uploads and copy-backs through `rt.alloc_tensor(init=...)`,
`rt.alloc_stacked_tensor(...)`, `rt.copy_to(...)`, `rt.copy_from(...)`, and
`rt.copy_stacked_from(...)` stage through runtime-owned shared memory. Their
host `torch.Tensor` endpoints only need to be CPU-contiguous and may be created
after `prepare()`; they do not need `.share_memory_()`, pre-fork allocation, or
`inherited_host_tensors`. The low-level `copy_to`/`copy_from` methods take the
host tensor's `.data_ptr()` plus a byte count.

```python
from pypto.runtime import DistributedWorker

compiled = ir.compile(MyDistributedProgram)

# shared-memory host buffers — allocated BEFORE prepare()
host_x = torch.zeros((seq, 4096), dtype=torch.float16).share_memory_()
host_out = torch.zeros((seq, 4096), dtype=torch.float16).share_memory_()

with DistributedWorker(compiled) as rt:
    # Explicit upload source may be an ordinary tensor created after prepare().
    host_weight = load_weight().contiguous()
    weight = rt.alloc_tensor(host_weight.shape, host_weight.dtype, init=host_weight)
    for step in generate_steps:
        host_x.copy_(next_input(step))          # refresh input in place
        rt(host_x, weight, host_out)            # host shm IO + resident weight
        consume(host_out)                       # read output directly
    rt.free_tensor(weight)
# rt.close() runs on exit
```

#### Sharding a weight across cards (`alloc_stacked_tensor`)

When a HOST orchestrator slices a `[B, N, M]` weight along its leading dimension
and dispatches a per-rank child — the canonical
`for r in range(world_size): child(x[r], device=r)` — passing the whole host
tensor re-uploads each `x[r]` slice to its card on **every** dispatch. To upload
each shard **once** and keep it resident on its card, build a
`StackedDeviceTensor` with `rt.alloc_stacked_tensor`:

```python
host_a = torch.zeros((B, N, M), dtype=...).share_memory_()
host_out = torch.zeros((B, N, M), dtype=...).share_memory_()

with DistributedWorker(compiled) as rt:
    host_w = load_weight().contiguous()          # post-prepare ordinary CPU tensor
    w = rt.alloc_stacked_tensor(host_w)          # shard i uploaded to card i, once
    for step in steps:
        host_a.copy_(next_input(step))
        rt(host_a, w, host_out)                  # x[r] resolves to the resident shard r
        consume(host_out)
    rt.free_stacked_tensor(w)
```

Internally each shard `host_w[i]` becomes a worker-resident `DeviceTensor`, so the
generated `x[r]` indexing derives a wire Tensor from the retained Buffer and
skips the H2D upload. Shards are
auto-freed on `close()` if not released earlier via `free_stacked_tensor`.

Like a single `DeviceTensor`, a `StackedDeviceTensor` is never copied back
automatically. To read the current device contents of every shard back to the
host in one call — e.g. a resident KV cache at the end of a step — use
`rt.copy_stacked_from(w, host_out)`, the read-back symmetric of
`alloc_stacked_tensor`. `host_out` is filled in place (`host_out[i]` receives
shard `i`) and must be a CPU-contiguous `[B, *tail]` tensor matching the
stack's shape and dtype. It may be allocated after `prepare()` because the D2H
operation stages through a runtime-owned shared Buffer; `.share_memory_()` and
`inherited_host_tensors` are not required.

The leading dimension is the shard dimension and `B` must equal the number of
cards the program dispatches to. By default shard `i` lands on worker `i`
(matching `device=r`). If the program uses a **non-identity** placement — a
permutation or a subset of cards (e.g. `device=2*r`, or literal `device=1` /
`device=0`) — pass the matching `worker_ids`, where `worker_ids[i]` is the worker
the program submits `x[i]`'s task to:

```python
# orchestrator dispatches x[0] to card 1 and x[1] to card 0
w = rt.alloc_stacked_tensor(host_w, worker_ids=[1, 0])
```

`worker_ids` must be distinct and within `[0, world_size)`; a mismatch with the
program's `device=` would leave a shard on the wrong card and read garbage.

`rt.alloc_tensor(..., worker_id=r)` similarly accepts a non-default `worker_id`
to place a single resident `DeviceTensor` on any card (pass the same `worker_id`
to `free_tensor`).

#### Dispatching several programs on one worker (multi-program)

Serving needs prefill and decode as separate HOST programs that share one L3
worker and one device-resident KV cache. Pass a list of compatible
`DistributedCompiledProgram` objects to `DistributedWorker`, or equivalently
`prefill.prepare(extra_compiled=[decode])` — they are prepared once on the same
worker, and `rt.run(compiled, *args)` selects which one to dispatch. Programs
must agree on platform, runtime, and device ids. In multi-program mode the
`rt(*args)` shortcut is disabled (the target is ambiguous) — always dispatch via
`rt.run(...)`. A worker-resident `DeviceTensor` (e.g. the KV cache) stays valid
across dispatches from either program.

A runnable end-to-end skeleton is in
[`examples/runtime/multi_program_kv_cache.py`](../../../examples/runtime/multi_program_kv_cache.py).

```python
from pypto.runtime import DistributedWorker, RunConfig

cfg = RunConfig(platform="a2a3", distributed_config=dc)
prefill_c = prefill.compile(host_prompt, kv_sample, config=cfg)   # @pl.jit.host kernels:
decode_c = decode.compile(host_token, kv_sample, host_logits, config=cfg)  # compile, no dispatch

with DistributedWorker([prefill_c, decode_c]) as rt:    # one worker, one fork
    kv_cache = rt.alloc_tensor(kv_shape, torch.float16)  # resident across both
    rt.run(prefill_c, host_prompt, kv_cache)             # writes the KV cache
    for _ in range(max_new_tokens):
        rt.run(decode_c, host_token, kv_cache, host_logits)  # reads/updates it
```

## See Also

- [Quickstart](02-quickstart.md) — writing and compiling the kernels dispatched here.
- [Programming Model](03-programming-model.md) — why the runtime, not statement order, decides execution order.
- [Runtime DFX](../dev/03-runtime-dfx.md) — the diagnostic flags behind the timing and profiling shown here.
- [Per-Task Ring Sizing](../dev/05-runtime-ring-sizing.md) — tuning the runtime's per-task rings.
- [Persistent L3 execution](../dev/06-persistent-l3.md) — reusing one worker across prepared distributed programs.
- [Runtime documentation](https://hw-native-sys.github.io/simpler/) — the runtime's own internals.
