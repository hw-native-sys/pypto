# Running

> **Legacy Python API examples:** `platform=` calls on this page are retained for
> compatibility. They emit `DeprecationWarning`; new code should use
> `backend_type` and `execution_mode`. See the [target mapping](../../dev/08-entry-points.md#legacy-platform-examples).

Dispatching a `CompiledProgram`, and keeping resident data resident.

## Concept

A `CompiledProgram` is a handle to compiled artifacts plus the metadata the runtime needs
to launch them. `ChipWorker` owns the device connection and the registrations. For
program execution, obtain the handle with `compiled = kernel.compile(*args, config=cfg)`
and dispatch through `compiled(*args, config=cfg)`, or use `worker.run(compiled, *args)`
when library code needs an explicit worker or a serving runtime pre-registers kernels.

Program execution transfers host tensors according to their parameter directions.
A `DeviceTensor` skips those transfers, which makes resident weights and KV caches viable.
Direct `kernel(*args)` calls use a separate eager path: they require `pypto.torch.init()`
and real NPU tensors, borrow their storage on the current Torch stream, and use the
KERNEL contract/cache instead of the PROGRAM contract/cache used by `.compile()`.
They do not provide host-tensor or simulator program dispatch. See
[kernel mode](../../dev/runtime/kernel-mode.md) for the eager prerequisites.

## Quickstart: keep a weight on the device

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

`alloc_tensor` returns a `DeviceTensor` that compiled programs accept anywhere a
`torch.Tensor` is accepted. The runtime treats the buffer as already resident and skips
both H2D and D2H for that argument.

## Mechanics

### The `CompiledProgram` contract

| Member | Gives you |
| ------ | --------- |
| `output_dir` | Where the artifacts are |
| `platform` / `backend_type` | What it was built for; the worker checks the first |
| `param_names` / `output_indices` / `has_return` | The call shape, for a harness binding arguments itself |
| `program` | The `Program` that was handed to `compile` — usually pre-pass, and `None` after `from_dir` |
| `chip_callable` / `runtime_name` / `runtime_config` | The runtime-side handles |
| `validate_ir` | Per-pass semantic comparison ([Precision](../precision/00-workflow.md)) |
| `from_dir` / `load` | Rebuild a handle from a saved artifact directory |

> **`compiled.program` is not the IR that produced the artifacts.** It is whatever
> `Program` you handed to `compile`, stored as-is; the transformed program codegen ran on is
> not retained, and a handle rebuilt with `from_dir` has no program at all.
>
> On the usual `ir.compile(MyProgram)` path that input is pre-pass IR. It need not be — the
> setup above compiles `add_kernel.lower(...)`, which is already lowered, so *there*
> `compiled.program` is post-pass. The property makes no promise either way; it hands back
> what it was given. For the lowered IR specifically, use `kernel.lower(*args)` or read a
> pass dump.

### Explicit dispatch

`worker.run(compiled, *args)` is a one-shot. `worker.register(compiled)` returns a handle
that skips the per-call lookup, which is what a hot loop wants:

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

`register` triggers the assembly and load once; the returned handle is what you call per
launch. `close()` releases the registrations and any `DeviceTensor` the caller forgot.

### `DeviceTensor`

| Rule | Detail |
| ---- | ------ |
| **Allocated by the worker** | `w.alloc_tensor(shape, dtype, init=...)` |
| **Not copied back automatically** | Read it with `w.copy_from(host_ptr, t.data_ptr, t.nbytes)` |
| **Freed explicitly** | `w.free_tensor(t)`; `close()` auto-frees as a backstop, not as the plan |
| **Bound to its worker** | It is not portable to another `ChipWorker` |

### `RunConfig` fields that affect dispatch

`RunConfig` carries both compile-side and runtime-side settings; the compile-side ones are
[the previous page](00-compile.md). What matters at dispatch:

| Field | Effect |
| ----- | ------ |
| `platform` / `device_id` | Which device, and which artifact the worker will accept |
| `enable_chip_swimlane` / `enable_dep_gen` / `enable_pmu` / `enable_dump_args` / `enable_scope_stats` | DFX capture ([Performance](../performance/00-swimlane.md)) |
| `ring_task_window` / `ring_heap` / `ring_dep_pool` | Runtime ring sizing ([Memory](../performance/05-memory.md)) |
| `aicpu_thread_num` | AICPU thread count override |

**Some `RunConfig` fields belong to the harness, not to dispatch.** `rtol` / `atol`,
`golden_data_dir`, `save_kernels` and `codegen_only` are read by the system-test harness,
which compiles, generates a golden and compares. Going through `compiled(...)`,
`worker.run(...)` or a registration handle, they do nothing — in particular
**`codegen_only=True` does not stop a dispatch on this path**, so do not rely on it to
avoid a launch. (`save_kernels_dir` is the exception: `RunConfig.compile_kwargs()`
forwards it as `ir.compile`'s `output_dir`.)

## Edge Cases

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| **Worker rejects the program before the first dispatch** | Artifact's `platform` differs from the worker's | Compile for the platform you dispatch on |
| **`DeviceTensor` rejected at a JIT entry** | Eager calls require real Torch NPU tensors; `.compile()` cannot infer shape/dtype from a `DeviceTensor` | Compile using host metadata samples or full annotations, then dispatch the compiled program |
| **Device memory grows across launches** | `DeviceTensor` never freed | `free_tensor`, or scope the worker with `with` |
| **`run()` gives no host/device split** | `execution_time` is total wall clock | Use `pypto.runtime.benchmark` for `device_wall_us` / `host_wall_us` |

## See Also

- [Compiling](00-compile.md) — producing the artifact this page dispatches.
- [Getting started](../00-getting_started.md) — the same ground, in the shortest form.
- [Host](../performance/06-host.md) — when the host span is the one to shrink.
- [Distributed execution](../distributed/03-execution.md) — the multi-rank worker.
