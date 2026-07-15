# Getting Started with PyPTO

## What is PyPTO?

PyPTO is a Python-based kernel programming framework for Ascend NPUs. You write compute kernels in Python using the `pypto.language` module, and PyPTO compiles them into optimized device code.

```python
import pypto.language as pl
from pypto import ir
```

All kernel code uses the `pl` namespace. The `ir` module provides compilation and IR utilities.

## Hello World: Vector Add (Tensor Level)

The simplest kernel operates on **Tensors** — high-level arrays in DDR memory. PyPTO automatically handles data movement and memory allocation.

```python
import pypto.language as pl
from pypto import ir

@pl.function
def vector_add(
    a: pl.Tensor[[64], pl.FP32],
    b: pl.Tensor[[64], pl.FP32],
) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
    return result
```

**Line by line:**

| Line | What it does |
| ---- | ------------ |
| `@pl.function` | Parses the Python function body into PyPTO IR |
| `a: pl.Tensor[[64], pl.FP32]` | Input: 1D tensor, 64 elements, 32-bit float |
| `pl.add(a, b)` | Element-wise addition (dispatches to tensor add) |
| `return result` | The function returns a tensor |

After decoration, `vector_add` is an `ir.Function` object — not a Python callable. Print the IR:

```python
print(vector_add.as_python())
```

## Tile Kernel: Load-Compute-Store

For hardware-level control, use **Tiles** — on-chip memory buffers. You explicitly load data from DDR, compute on-chip, and store results back.

```python
@pl.function
def vector_add_tile(
    a: pl.Tensor[[64], pl.FP32],
    b: pl.Tensor[[64], pl.FP32],
    output: pl.Out[pl.Tensor[[64], pl.FP32]],
) -> pl.Tensor[[64], pl.FP32]:
    # Load from DDR → on-chip (Vec memory)
    a_tile: pl.Tile[[64], pl.FP32] = pl.load(a, [0], [64])
    b_tile: pl.Tile[[64], pl.FP32] = pl.load(b, [0], [64])

    # Compute on-chip
    result: pl.Tile[[64], pl.FP32] = pl.add(a_tile, b_tile)

    # Store back to DDR
    out: pl.Tensor[[64], pl.FP32] = pl.store(result, [0], output)
    return out
```

**Key differences from the Tensor version:**

| Concept | Tensor level | Tile level |
| ------- | ------------ | ---------- |
| Data location | DDR (automatic) | Explicit load/store |
| Type | `pl.Tensor` | `pl.Tile` (on-chip) |
| Output parameter | Return value | `pl.Out[pl.Tensor[...]]` |
| Memory control | Compiler decides | You decide |

**`pl.load(tensor, offsets, shapes)`** copies a region from a DDR Tensor into an on-chip Tile.

**`pl.store(tile, offsets, output_tensor)`** copies a Tile back to DDR.

## Loops and Accumulation

Use `pl.range()` for loops. With `init_values`, you get loop-carried values (accumulators):

```python
@pl.function
def sum_elements(
    a: pl.Tensor[[64], pl.FP32],
) -> pl.Tensor[[1], pl.FP32]:
    zero: pl.Tensor[[1], pl.FP32] = pl.create_tensor([1], dtype=pl.FP32)

    for i, (acc,) in pl.range(64, init_values=(zero,)):
        elem: pl.Tensor[[1], pl.FP32] = pl.slice(a, [1], [i])
        new_acc: pl.Tensor[[1], pl.FP32] = pl.add(acc, elem)
        acc_out: pl.Tensor[[1], pl.FP32] = pl.yield_(new_acc)

    return acc_out
```

**How `init_values` works:**

1. `init_values=(zero,)` — initial value for the accumulator
2. `for i, (acc,)` — `i` is the loop variable, `acc` is the current accumulator
3. `pl.yield_(new_acc)` — passes `new_acc` as the accumulator to the next iteration
4. After the loop, `acc_out` holds the final value

Simple loops without accumulators:

```python
for i in pl.range(10):
    # i goes from 0 to 9
    ...

for i in pl.range(0, 100, 2):
    # i goes from 0 to 98, step 2
    ...
```

## Multi-Function Programs

Use `@pl.program` to group multiple functions that call each other:

```python
@pl.program
class VectorAddProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_add(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        a_tile: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
        b_tile: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [0, 0], [128, 128])
        result: pl.Tile[[128, 128], pl.FP32] = pl.add(a_tile, b_tile)
        out: pl.Tensor[[128, 128], pl.FP32] = pl.store(
            result, [0, 0], output
        )
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        c: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor(
            [128, 128], dtype=pl.FP32
        )
        c = self.kernel_add(a, b, c)
        return c
```

**Key concepts:**

| Concept | Description |
| ------- | ----------- |
| `@pl.program` | Decorates a class → becomes an `ir.Program` |
| `self` | Required first parameter; stripped from IR |
| `self.kernel_add(...)` | Cross-function call within the program |
| `FunctionType.InCore` | Runs on AICore (compute kernel) |
| `FunctionType.Orchestration` | Runs on host (task graph coordinator) |

**Function types:**

- **`Opaque`** (default) — no specific execution context
- **`InCore`** — AICore compute kernel; uses load/store for data movement
- **`Orchestration`** — host-side function that creates tensors and dispatches InCore tasks

## Compiling

Compile a program to generate device code:

```python
from pypto.backend import BackendType

output_dir = ir.compile(
    VectorAddProgram,
    strategy=ir.OptimizationStrategy.Default,
    dump_passes=True,
    backend_type=BackendType.Ascend910B,
)
print(f"Generated code in: {output_dir}")
```

**`ir.compile()` parameters:**

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `program` | (required) | The `ir.Program` to compile |
| `output_dir` | `None` → `<base>/<name>_<timestamp>` | Directory for codegen, reports, and (when dumping) pass IR. `<base>` is the `PYPTO_PROG_BUILD_DIR` env var, or `build_output` if unset |
| `strategy` | `OptimizationStrategy.Default` | Pass pipeline preset (`Default` or `DebugTileOptimization`) |
| `dump_passes` | `True` | When `True`, write IR snapshots under `output_dir/passes_dump/` after each pass |
| `backend_type` | `BackendType.Ascend910B` | Target hardware for passes and codegen (`Ascend910B` or `Ascend950`) |
| `skip_ptoas` | `False` | If `True`, skip the ptoas step and emit raw `.pto` (MLIR) instead of compiled C++ wrappers |
| `verification_level` | `None` | Optional `ir.VerificationLevel` override; `None` uses defaults (or `PYPTO_VERIFY_LEVEL`) |

`DebugTileOptimization` is a debug-only shortcut for inspecting the PTO tile
pipeline. Prefer `Default` unless you are explicitly debugging strategy
selection or pass ordering.

**Inspect IR without compiling:**

```python
# Print a single function
print(vector_add.as_python())

# Print an entire program
print(VectorAddProgram.as_python())

# Print without intermediate type annotations (concise mode)
print(vector_add.as_python(concise=True))
```

## Reusing weights on the worker (DeviceTensor)

When the same large tensor is consumed by many kernel invocations — e.g. a
weight matrix used across batches of a forward pass — uploading it on every
call wastes bandwidth. `ChipWorker.alloc_tensor` allocates persistent device
memory and returns a `DeviceTensor` handle that `CompiledProgram` accepts in
place of a `torch.Tensor`. The runtime treats the buffer as already resident
and skips both H2D and D2H copies for that argument.

```python
import torch
from pypto import ir
from pypto.runtime import ChipWorker, RunConfig

compiled = ir.compile(MyKernel)

with ChipWorker(config=RunConfig(platform="a2a3sim")) as w:
    weight = w.alloc_tensor((1024, 4096), torch.float16, init=host_weight)
    for batch in batches:
        out = torch.empty(batch.shape[0], 4096, dtype=torch.float16)
        compiled(batch, weight, out)
    w.free_tensor(weight)
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

```python
worker = ChipWorker(config=RunConfig(platform="a2a3sim"))
try:
    out = worker.run(compiled, a, b)                 # one-shot
    handle = worker.register(compiled)               # eager registration
    for _ in range(1000):                            # hot loop, no cid lookup
        handle(a, b, out)
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
  args, e.g. `kernel.compile(num_tokens=128)`.
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
device timing, enable the L2 swimlane DFX (`RunConfig(enable_l2_swimlane=True)`)
and read `l2_swimlane_records.json`.

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
`config=` for `block_dim` / `aicpu_thread_num` control (not both). Aggregates are
exposed under both `device_wall_us_*` and shorter `device_us_*` names, with
`samples` aliasing the raw `device_wall_us` list.

`benchmark` reads timing from the `[STRACE]` markers (simpler PR #1177): it
raises the runtime log level to `v9` for the worker's lifetime and captures
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
simpler_run                71784.1us  ±6797.5  [66482.4..89832.6]
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

`benchmark` also accepts an L3 `DistributedCompiledProgram` (opened via
`compiled.prepare()`): pass shared-memory host tensors (or `DeviceTensor`s) and
omit `platform=` / `device_id=` (the device set is fixed at compile time via
`distributed_config`). L3 has no single DAG-level device wall, so timing is
folded from the per-rank chip-child markers into per-round samples — the headline
`device_wall_us[k]` is the max across ranks of each rank's summed dispatch device
walls. Query the four metrics uniformly:

```python
stats.per_round("device" | "host" | "effective" | "union")  # -> [one value per round]
stats.per_rank("device" | "host" | "effective")             # -> {pid: [one per round]}
```

Both views aggregate **per rank per round**: each entry sums that rank's
dispatches within the round (a card runs its dispatches serially), so they are
per-round-per-rank figures, **not** per-dispatch. When a rank runs exactly one
dispatch per round the sum is that single dispatch's value; for the individual
dispatches in any case, read `stats.rounds_dispatches[k][pid]` (see below).

`effective` is the orch∪sched on-device window (per-card L2 Effective); `union`
is the cross-rank host-timeline window (captures start skew — host-domain, so it
includes dispatch overhead). The navigable `round -> rank -> [dispatch]` grid is
`stats.rounds_dispatches`, where each `TraceInvocation` exposes `.task` (callable
id), `.device_wall_us`, `.host_wall_us`, `.effective_us`. A pure-device
cross-rank end-to-end wall is not recoverable from the markers today. If the
dispatch shape is non-deterministic, `stats.fallback_flattened` is set and the
per-rank / `union` views are empty.

### Distributed (L3+) programs

L3+ distributed programs returned by `ir.compile` (a `DistributedCompiledProgram`)
accept `DeviceTensor` arguments the same way as `CompiledProgram`: pass a
worker-resident buffer in place of a `torch.Tensor` and the runtime skips H2D/D2H
for that argument. This is the recommended way to keep large static weights
resident across the many dispatches of a generate loop.

```python
import torch
from pypto.runtime import DeviceTensor

compiled = ir.compile(MyDistributedProgram)   # returns DistributedCompiledProgram
weight = DeviceTensor(dev_ptr, (1024, 4096), torch.float16)   # caller-managed buffer
compiled(x, weight, out)                       # weight: no H2D/D2H copy
```

#### Reusing setup across dispatches (`prepare()`)

`compiled(*args)` runs the full distributed setup (per-chip assembly, simpler
Worker construction + fork) on every call. For a resident service that
dispatches the same program many times (e.g. a generate loop), call
`compiled.prepare()` once to get a `DistributedWorker` handle that runs setup
once and dispatches many times on the same worker.

Mutable per-call IO buffers are **shared-memory host tensors allocated before `prepare()`** and reused
in place, so child writes are visible to the parent. Immutable inputs may remain ordinary contiguous
CPU tensors when the same objects are passed to `compiled.prepare(inherited_host_tensors=[...])` before
fork; they are retained and read through inherited copy-on-write mappings. This exception is
input-only: outputs, in-place tensors, and unregistered non-shared tensors are rejected. Large static
weights may be uploaded once via `rt.alloc_tensor`; its `init` source must still be pre-`prepare()` shared.

```python
compiled = ir.compile(MyDistributedProgram)

# shared-memory host buffers — allocated BEFORE prepare()
host_x = torch.zeros((seq, 4096), dtype=torch.float16).share_memory_()
host_out = torch.zeros((seq, 4096), dtype=torch.float16).share_memory_()
host_weight = load_weight().share_memory_()

with compiled.prepare() as rt:                  # setup runs once
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
host_w = load_weight().share_memory_()           # [B, N, M], B == world_size
host_a = torch.zeros((B, N, M), dtype=...).share_memory_()
host_out = torch.zeros((B, N, M), dtype=...).share_memory_()

with compiled.prepare() as rt:
    w = rt.alloc_stacked_tensor(host_w)          # shard i uploaded to card i, once
    for step in steps:
        host_a.copy_(next_input(step))
        rt(host_a, w, host_out)                  # x[r] resolves to the resident shard r
        consume(host_out)
    rt.free_stacked_tensor(w)
```

Internally each shard `host_w[i]` becomes a worker-resident `DeviceTensor`, so the
generated `x[r]` indexing skips the H2D upload (`child_memory`). Shards are
auto-freed on `close()` if not released earlier via `free_stacked_tensor`.

Like a single `DeviceTensor`, a `StackedDeviceTensor` is never copied back
automatically. To read the current device contents of every shard back to the
host in one call — e.g. a resident KV cache at the end of a step — use
`rt.copy_stacked_from(w, host_out)`, the read-back symmetric of
`alloc_stacked_tensor`. `host_out` is filled in place (`host_out[i]` receives
shard `i`) and, like the upload source, must be a CPU, contiguous, **shared-memory**
`[B, *tail]` tensor matching the stack's shape and dtype, allocated before
`prepare()` (call `.share_memory_()`): the D2H copy runs in the forked chip worker,
which can only write host memory it inherited at fork.

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

## What's Next

- **[Language Guide](01-language_guide.md)** — complete reference for types, operations, control flow, memory, and compilation
- **[Operation Reference](02-operation_reference.md)** — lookup tables for every `pl.*` operation
