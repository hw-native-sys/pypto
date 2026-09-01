# Compile and Execution Entry Points

Which entry point does what, and which layer it belongs to.

PyPTO has more compile and execution entry points than it has concepts, and
several share a name across abstraction layers: six different functions are
called `compile`, and `run` is a method on both the runtime worker handles and
`PassPipeline`. This page is the map — what each entry point takes, what it
produces, and when to reach for it.

## The four layers

Every entry point belongs to exactly one of four layers. Confusion arises when
a name does not say which:

```text
  define              compile                 artifact                execute
  ────────────────────────────────────────────────────────────────────────────
  @pl.program  ──┐                        CompiledProgram      ChipWorker.run()
  @pl.function   ├─>  ir.compile()   ──>   Distributed-        DistributedWorker.run()
  @pl.jit      ──┘    kernel.compile()     CompiledProgram     compiled(*args)
                      kernel.lower()  ──>  ir.Program
```

Below the artifact layer sits an **assembly layer** — turning `.pto` and `.cpp`
into loadable binaries. It is internal and has no supported entry point.

## Choosing an entry point

| You have | You want | Use |
| -------- | -------- | --- |
| A `@pl.jit` kernel and torch tensors | The result | `kernel(*args, config=...)` |
| A `@pl.jit` kernel | The artifact, to dispatch repeatedly | `kernel.compile(*args, config=...)` |
| A `@pl.jit` kernel | The IR, without codegen | `kernel.lower(*args)` |
| A `@pl.program` class | The artifact | `ir.compile(program, ...)` |
| A `CompiledProgram` | One dispatch | `compiled(*args, config=...)` |
| A `CompiledProgram` and device-resident data | Many dispatches | `ChipWorker.run(compiled, *args)` |
| A `DistributedCompiledProgram` | Many dispatches | `compiled.prepare()` → `DistributedWorker.run(...)` |
| A build directory on disk | One dispatch, no recompile | `execute_compiled(work_dir, args, ...)` |
| A `CompiledProgram` | Timed dispatches | `benchmark(compiled, args, ...)` |

## Compile

| Entry | Layer | Location |
| ----- | ----- | -------- |
| `ir.compile` | compile driver | [`ir/compile.py`](../../../python/pypto/ir/compile.py) |
| `JITFunction.compile` | specialize + driver | [`jit/decorator.py`](../../../python/pypto/jit/decorator.py) |
| `JITFunction.lower` | specialize only, stops at `ir.Program` | same |
| `device_runner.compile_and_assemble` | assembly | [`runtime/device_runner.py`](../../../python/pypto/runtime/device_runner.py) |
| `device_runner.compile_single_kernel` / `compile_single_orchestration` | assembly | same |
| `KernelCompiler.compile_incore` | ptoas invocation | [`runtime/kernel_compiler.py`](../../../python/pypto/runtime/kernel_compiler.py) |

`ir.compile` is the only supported one; the rest of the table is here so a name
that turns up in a traceback can be placed on a layer. It also shadows the
Python builtin `compile`, so prefer `from pypto import ir` and call
`ir.compile` over importing the name directly.

Its parameters are documented in [Compiling](../user/execution/00-compile.md).

Both artifact types it can return, and the config that selects the distributed
one, are exported from `pypto.ir`:

```python
from pypto.ir import CompiledProgram, DistributedCompiledProgram, DistributedConfig
```

The defining module, `pypto.ir.distributed_compiled_program`, stays importable —
`pypto.runtime` reaches for it directly to avoid an import cycle — but user code
and tests should take the three names from `pypto.ir`.

## Execute

| Entry | Layer | Takes | Location |
| ----- | ----- | ----- | -------- |
| `CompiledProgram.__call__` | artifact | artifact handle | [`ir/compiled_program.py`](../../../python/pypto/ir/compiled_program.py) |
| `Worker.run` / `ChipWorker.run` / `DistributedWorker.run` | execute | artifact + worker | [`runtime/worker.py`](../../../python/pypto/runtime/worker.py) |
| `runtime.execute_compiled` | execute | output directory | [`runtime/runner.py`](../../../python/pypto/runtime/runner.py) |
| `execute_distributed_compiled` | execute | output directory | [`runtime/distributed_runner.py`](../../../python/pypto/runtime/distributed_runner.py) |
| `device_runner.execute_on_device` | assembly | assembled binaries | [`runtime/device_runner.py`](../../../python/pypto/runtime/device_runner.py) |
| `execute_artifact_dir` / `execute_batch_manifest` | CLI | output directory | [`runtime/execute_artifact.py`](../../../python/pypto/runtime/execute_artifact.py) |

`run` does not say which layer you are on; the receiver does. `ChipWorker.run`
and `DistributedWorker.run` dispatch an artifact, both implementing
`Worker.run`. `PassPipeline.run` is unrelated — it transforms a `Program` — as
is `PassManager.run_passes`.

`execute_compiled` and `execute_distributed_compiled` are the L2 and L3
directory-driven paths. They skip the PyPTO compile entirely, which is what
makes [replay](03-runtime-replay.md) possible.

## Driving both phases from one config

`RunConfig` carries compile-time and dispatch-time settings together.
`compile_kwargs()` extracts the compile-time half as `ir.compile` keyword
arguments, so one config object drives both phases:

```python
config = RunConfig(platform="a2a3")
compiled = ir.compile(program, **config.compile_kwargs())
compiled(*tensors, config=config)
```

Its fields split three ways:

| Read by | Fields |
| ------- | ------ |
| `ir.compile`, via `compile_kwargs()` | `strategy`, `backend_type`, `platform`, `memory_planner`, `dump_passes`, `dump_ptoas_passes`, `compile_profiling`, `diagnostic_phase`, `disabled_diagnostics`, `analyze_auto_scopes_for_deps`, `save_kernels_dir` (as `output_dir`), `distributed_config` |
| Dispatch | `device_id`, `aicpu_thread_num`, the `ring_*` overrides ([Ring sizing](05-runtime-ring-sizing.md)), the DFX toggles ([DFX](03-runtime-dfx.md)) |
| The system-test harness only | `rtol`, `atol`, `golden_data_dir`, `save_kernels`, `codegen_only` |

The `@pl.jit` path keeps its own mapping in
`jit.decorator._run_config_compile_kwargs`, which omits `platform` and
`backend_type` because that path forwards them separately.

## What is internal

These have no stability guarantee. They are not in any `__all__`, and code
outside PyPTO should not import them:

- `pypto.ir.compile` — `_ensure_orchestration_headers`
- `pypto.ir.compiled_program` — `CompiledProgram._build_orch_args`,
  `CompiledProgram._build_call_config`
- `pypto.runtime.device_runner` — `compile_and_assemble`, `compile_single_kernel`,
  `compile_single_orchestration`, `execute_on_device`
- `pypto.runtime.kernel_compiler` — `KernelCompiler`, `compile_incore`
- `pypto.runtime.runner` — `_build_call_config`, `_coerced_to_orch_args`,
  `_DfxOpts`
- `pypto.runtime.tensor_arg`, `pypto.runtime.elf_parser`, `pypto.runtime._binary_cache`

The two argument builders marshal user arguments into simpler's `TaskArgs` and
`CallConfig`. `ChipWorker.run` and `ChipWorker.register` are the supported way
to reach that path — they call the builders for you, and a caller driving a
hand-constructed `simpler.worker.Worker` still gets `chip_callable`,
`runtime_name` and `runtime_config` from the public surface.

## Command-line entry points

| Command | Purpose |
| ------- | ------- |
| `pypto-ir-trace` | Render a pass dump as an interactive lowering trace ([IR lowering trace](07-ir-lower-trace.md)) |
| `python -m pypto.runtime.execute_artifact` | Execute an artifact directory or a batch manifest |
| `python -m pypto.runtime.debug.replay` | Re-run a build directory after hand-editing generated code ([Replay](03-runtime-replay.md)) |
| `python build_output/<dir>/debug/run.py` | The self-contained reproducer emitted with every build |
| `python -m pypto.tools.memory_map` | Render on-chip memory as HTML ([Memory Map](07-memory-map.md)) |
| `python -m pypto.tools.clean_sim_trace` | Convert simulator dumps to readable traces ([Trace cleaning](04-simulator-trace-cleaning.md)) |

## See Also

- [Compiling](../user/execution/00-compile.md) — `ir.compile`'s parameters and the artifact directory.
- [Running](../user/execution/01-run.md) — `CompiledProgram`, `ChipWorker`, `DeviceTensor`.
- [Runtime DFX Flags](03-runtime-dfx.md) — the five diagnostic sub-features.
- [Per-Task Ring Sizing](05-runtime-ring-sizing.md) — the three per-dispatch ring overrides.
