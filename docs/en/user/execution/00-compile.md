# Compiling

Turning a program or a `@pl.jit` kernel into device kernels and host orchestration.

## Concept

Compilation runs the pass pipeline over your IR, generates PTO for each InCore function,
hands that to **ptoas** for the device binary, and emits the host orchestration that
launches them. What you get back is a `CompiledProgram` — a handle to a directory of
artifacts plus the metadata the runtime needs to dispatch them.

There are two entry points and they differ only in where the IR comes from.
`ir.compile(program)` takes a `@pl.program` class. `kernel.compile(*args)` takes a
`@pl.jit` function plus sample arguments, specializes it against their shapes, and then
does the same thing.

## Quickstart: compile and keep the artifacts

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

CFG = RunConfig(platform="__PLATFORM__")
torch.manual_seed(0)
A = torch.randn(128, 128, dtype=torch.float32)
B = torch.randn(128, 128, dtype=torch.float32)
```

<!-- doctest: run -->
```python
@pl.jit
def add(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        out[:] = pl.add(a, b)
    return out


out = torch.zeros(128, 128, dtype=torch.float32)
compiled = add.compile(A, B, out, config=CFG)

print("artifacts in:", compiled.output_dir)
assert compiled.platform == CFG.platform
assert compiled.param_names == ["a", "b", "out"]
```

`compile()`'s positional arguments are **the kernel's own arguments**, not compile options.
`add.compile(skip_ptoas=True)` is bound against the kernel signature and raises
`TypeError: got an unexpected keyword argument`; compile options travel in
`config=RunConfig(...)`.

## Mechanics

### `ir.compile` parameters

Eighteen, of which four carry most decisions. The rest have defaults you rarely
move. All are keyword-only — only `program` is positional.

| Parameter | Default | What it decides |
| --------- | ------- | --------------- |
| `program` | — | The `ir.Program` to compile |
| `output_dir` | `None` | Where artifacts land; `None` means `<base>/<name>_<timestamp>` under `PYPTO_PROG_BUILD_DIR` or `build_output` |
| `strategy` | `Default` | The pass pipeline. `Default` is the only strategy |
| `dump_passes` | `True` | IR snapshot after every pass — see below |
| `backend_type` | `Ascend910B` | Target for passes and codegen (`Ascend910B` / `Ascend950`) |
| `platform` | `None` | Execution platform the artifact is built for; must match the worker that dispatches it |
| `skip_ptoas` | `False` | Stop at `.pto` (MLIR) instead of building the device binary |
| `verification_level` | `None` | `NONE` / `BASIC` / `ROUNDTRIP`; `None` defers to `PYPTO_VERIFY_LEVEL`, else `BASIC` |
| `memory_planner` | `None` → `DSA_RP` | `PYPTO` / `DSA_RP` / `PTOAS` — who plans on-chip buffers ([Memory](../performance/05-memory.md)); an active `PassContext` overrides the fallback |
| `diagnostic_phase` | `None` | Which phase warnings and perf hints are gated at |
| `disabled_diagnostics` | `None` | Silence specific checks rather than all of them |
| `profiling` | `False` | Per-stage compile timing into `report/pipeline_profile.{txt,json}` |
| `distributed_config` | `None` | Compile a HOST-level program per rank ([Distributed](../distributed/index.md)) |
| `analyze_auto_scopes_for_deps` | `False` | Compiler-derived dependencies for AUTO scopes |
| `enable_pypto_l0c_double_buffer` | `None` | L0C double buffering |
| `emit_source_loc` | `None` | Carry DSL source locations into the emitted `.pto` |
| `dump_ptoas_passes` | `False` | Also dump ptoas's own pass IR |
| `runtime` | `None` | Simpler runtime ABI to target — `TENSORMAP_AND_RINGBUFFER` or `HOST_BUILD_GRAPH` (what `@pl.jit.graph` needs — [Functions](../language/01-functions.md)); `None` inherits the active `PassContext`. The worker must match it |

### Pass dumps

`dump_passes` accepts a bool or a `PassDumpLevel`:

| Value | Effect |
| ----- | ------ |
| `False` / `PassDumpLevel.NONE` | No snapshots |
| `True` / `PassDumpLevel.CONCISE` | One snapshot per pass |
| `PassDumpLevel.EXPLICIT` | Same, with implicit tile layouts and distributed window buffers resolved |

They land in `<output_dir>/passes_dump/NN_after_<PassName>.py`, numbered in execution
order. `EXPLICIT` is what the [memory map](../tools/02-memory-map.md) reads, and what you
want when the question is "which pass changed this".

### What ends up in the output directory

| Path | Holds |
| ---- | ----- |
| `passes_dump/` | Per-pass IR snapshots, when `dump_passes` asks for them |
| `ptoas/` | Generated `.pto` per InCore function, beside the `.cpp` ptoas produced |
| `kernels/` | The compiled device kernels |
| `report/perf_hints.log` | Compile-time performance hints ([Performance](../performance/index.md)) |
| `report/pipeline_profile.*` | Per-stage compile timing, when `profiling=True` |
| `dfx_outputs/` | Written at *run* time by the DFX flags, not by compilation |

### `JITFunction.compile`

`@pl.jit` normally fuses specialize + compile + dispatch into one `kernel(*args)` call.
`compile(*sample_args)` stops after compilation and returns the `CompiledProgram` the JIT
cache holds — so a later call with the same specialization key gets the identical object.

It exposes the whole extraction surface, which is what a harness driving the runtime
directly needs: `chip_callable`, `runtime_name`, `runtime_config`, `output_dir`,
`platform`, `output_indices`, `param_names`, `orchestration_names`, `has_return`.
Argument marshalling is not part of that surface — dispatch through
[`ChipWorker`](01-run.md#explicit-dispatch), which does it for you.

`lower(*args)` goes one step less far: it runs the passes and returns the `Program`,
writing no artifacts — which also means no `passes_dump/`. It is the right form for
[torch codegen](../tools/01-torch-codegen.md), which wants the IR itself; the
[memory map](../tools/02-memory-map.md) reads a dump on disk and therefore needs
`compile()`.

## Edge Cases

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| **`TypeError: got an unexpected keyword argument`** | Compile option passed positionally to `compile()` | Put it in `config=RunConfig(...)` |
| **Worker rejects the artifact** | `platform` at compile time differs from the worker's | Compile with the platform you will dispatch on |
| **No `passes_dump/`** | `lower()` writes no artifacts; `dump_passes=False` | Use `compile()`, or pass `dump_passes=PassDumpLevel.EXPLICIT` |
| **Memory map has nothing to draw** | `memory_planner=PTOAS` skips `AllocateMemoryAddr`, so the dump carries no offsets | Compare end to end instead |

> **`verification_level` is a debugging lever, not a default to raise.** `BASIC` is the
> default — unless `PYPTO_VERIFY_LEVEL` says otherwise, since the parameter defers to it
> when left `None`. `ROUNDTRIP` additionally reparses each dump and is markedly slower.
> Raise it when you suspect malformed IR, and put it back.

## See Also

- [Running](01-run.md) — dispatching what this page produced.
- [Pass Manager](../../dev/passes/00-pass_manager.md) — the pipeline `strategy` selects.
- [Debugging](../tools/00-debugging.md) — reading the IR this page dumps.
- [Precision](../precision/index.md) — when compilation succeeds but the numbers are wrong.
