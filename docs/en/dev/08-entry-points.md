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
| A build directory on disk | One dispatch, no recompile | `CompiledProgram.from_dir(work_dir)(*args, config=...)` |
| A `CompiledProgram` | Timed dispatches | `benchmark(compiled, args, ...)` |

## Compile

| Entry | Layer | Location |
| ----- | ----- | -------- |
| `ir.compile` | compile driver | [`ir/compile.py`](../../../python/pypto/ir/compile.py) |
| `JITFunction.compile` | specialize + driver | [`jit/decorator.py`](../../../python/pypto/jit/decorator.py) |
| `JITFunction.lower` | specialize only, stops at `ir.Program` | same |
| `device_runner._compile_and_assemble` | assembly | [`runtime/device_runner.py`](../../../python/pypto/runtime/device_runner.py) |
| `device_runner._compile_single_kernel` / `_compile_single_orchestration` | assembly | same |
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
| `CompiledProgram.from_dir` / `DistributedCompiledProgram.from_dir` | artifact | output directory | [`ir/compiled_program.py`](../../../python/pypto/ir/compiled_program.py) |
| `runtime.execute_compiled` *(deprecated)* | execute | output directory | [`runtime/runner.py`](../../../python/pypto/runtime/runner.py) |
| `execute_distributed_compiled` *(deprecated)* | execute | output directory | [`runtime/distributed_runner.py`](../../../python/pypto/runtime/distributed_runner.py) |
| `device_runner._execute_on_device` | assembly | assembled binaries | [`runtime/device_runner.py`](../../../python/pypto/runtime/device_runner.py) |
| `execute_artifact_dir` / `execute_batch_manifest` | CLI | output directory | [`runtime/execute_artifact.py`](../../../python/pypto/runtime/execute_artifact.py) |

`run` does not say which layer you are on; the receiver does. `ChipWorker.run`
and `DistributedWorker.run` dispatch an artifact, both implementing
`Worker.run`. `PassPipeline.run` is unrelated — it transforms a `Program` — as
is `PassManager.run_passes`.

Dispatching a directory rather than a live handle is what makes
[replay](03-runtime-replay.md) possible: the PyPTO compile is skipped
entirely. `from_dir` is how you get there — it rebuilds the artifact handle
from the persisted sidecar, and calling that handle takes the same path as a
handle that never left memory.

`execute_compiled` and `execute_distributed_compiled` were the L2 and L3
spellings of that, and are **deprecated**. Each still forwards to the same
implementation, and each emits a `DeprecationWarning`.

The L3 one is a plain rename — it already was `from_dir` plus a call:

```python
# before
execute_distributed_compiled(work_dir, args, config=cfg, platform="a2a3")

# after
ir.DistributedCompiledProgram.from_dir(work_dir, platform="a2a3")(*args, config=cfg)
```

**The L2 one is not**, because the two paths disagree on precedence:

| Setting | `execute_compiled` | `CompiledProgram.__call__` |
| ------- | ------------------ | -------------------------- |
| `platform` | the explicit argument | `config.platform` when a config is passed, else the artifact's |
| `device_id` / `dfx` / `aicpu_thread_num` | the explicit arguments | always from `config` |
| ring overrides | from `config` | from `config` |

So a `config` passed for ring sizing alone silently takes over the rest, and
`RunConfig.platform` defaults to `a2a3sim` — dropping the explicit arguments
would move the run to the simulator. Fold them into the config:

```python
# before -- explicit args win; cfg was read for ring sizing only
execute_compiled(work_dir, args, platform="a2a3", device_id=0, config=cfg)

# after -- cfg carries every execution setting
cfg = dataclasses.replace(cfg, platform="a2a3", device_id=0)
ir.CompiledProgram.from_dir(work_dir)(*args, config=cfg)
```

`dfx` and `aicpu_thread_num` need no translation: the DFX toggles and
`aicpu_thread_num` are already `RunConfig` fields. `from_dir(platform=...)`
still decides the platform for a call made *without* a config.

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
- `pypto.runtime.runner` — `_execute_compiled`, `_execute_golden_case`,
  `_build_call_config`, `_coerced_to_orch_args`, `_DfxOpts`
- `pypto.runtime.distributed_runner` — `_execute_distributed`
- `pypto.runtime.device_runner` — the whole assembly layer:
  `_compile_and_assemble`, `_compile_single_kernel`,
  `_compile_single_orchestration`, `_execute_on_device`
- `pypto.runtime.kernel_compiler` — `KernelCompiler`, `compile_incore`
- `pypto.runtime.tensor_arg`, `pypto.runtime.elf_parser`, `pypto.runtime._binary_cache`

The assembly layer carries the underscore so a traceback says which side of the
boundary it came from: an import that has to spell `_` is an import that had to
decide to. `KernelCompiler` / `compile_incore` and the three modules on the last
line are the exceptions — they are internal by module rather than by name, and
carry the same absence of a stability guarantee.

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
