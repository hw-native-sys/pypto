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
| `compiled.prepare(config=None, *, extra_compiled=(), persistent=False, reset_persistent_windows=None, callbacks=None, sub_worker_overrides=None)` | Create worker, fork chip processes, return `DistributedWorker`. Use as context manager. |
| `rt(x, y, z)` | Single dispatch — coerces args, calls host_orch. |
| `rt.run(compiled, x, y, z)` | Multi-program dispatch — selects the target program. |
| `rt.alloc_tensor(shape, dtype, *, init=None)` | Allocate a worker-resident `DeviceTensor`. `init` copies from host (one-time H2D). |
| `rt.free_tensor(tensor)` | Release a `DeviceTensor`. |
| `rt.alloc_stacked_tensor(host_w)` | Shard host_w along dim 0 — shard `i` uploaded to card `i`. Returns `StackedDeviceTensor`. |
| `rt.free_stacked_tensor(stacked)` | Release all shards of a `StackedDeviceTensor`. |
| `rt.copy_stacked_from(stacked, host_out)` | D2H read-back of every shard into `host_out` (shared-memory, allocated before `prepare()`). |
| `rt.release_inherited_host_tensor_refs()` | Drop runtime-held host references in the parent process after fork. |
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

## DeviceTensor

A worker-resident buffer that lives on the device across dispatches.
When a `DeviceTensor` is passed as an argument to a compiled program,
the runtime skips H2D/D2H copies — the device already has the data.

```python
import torch
from pypto.runtime import DeviceTensor

with compiled.prepare() as rt:
    weight = rt.alloc_tensor((1024, 4096), torch.float16, init=host_weight)
    rt(x, weight, out)   # dispatch via worker — no H2D/D2H
```

## StackedDeviceTensor

Sharded across devices — obtained via `rt.alloc_stacked_tensor()`:

```python
# Host tensor sharded along dim 0 — shard[i] lives on card i.
host_weights = torch.randn(4, 1024, 4096).share_memory_()  # 4 shards
with compiled.prepare() as rt:
    stacked = rt.alloc_stacked_tensor(host_weights)
    rt(x, stacked, out)
```

> **Fatal pitfall:** `host_weights` must call `.share_memory_()` *before*
> `prepare()`. The upload runs inside the already-forked chip worker, which
> can only read host memory it inherited at fork — a plain `torch.Tensor`
> raises `ValueError` at `alloc_stacked_tensor()`.

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

> **Fatal pitfall:** IO buffers passed to `DistributedWorker` must call
> `.share_memory_()` before `prepare()`. If you forget, the runtime rejects
> the buffer at dispatch time — the child processes cannot access the
> parent's private memory.

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

## See Also

- [00-model](00-model.md) — Quickstart and model vocabulary
- [04-debugging](04-debugging.md) — Common failure patterns
- [Getting Started](../00-getting_started.md) — Runtime setup
