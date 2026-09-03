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
`ir.compile` over importing the name directly. Every option it takes is
keyword-only; `program` is the one positional parameter.

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

Its fields split three ways, and each way is a type:

| Read by | Type | Fields |
| ------- | ---- | ------ |
| `ir.compile` | `CompileOptions` | `platform`, `strategy`, `dump_passes`, `dump_ptoas_passes`, `profiling` (`compile_profiling`), `diagnostic_phase`, `disabled_diagnostics`, `analyze_auto_scopes_for_deps`, `output_dir` (`save_kernels_dir`), `memory_planner`, `distributed_config` |
| Dispatch | `RunOptions` | `platform`, `device_id`, `aicpu_thread_num`, the `ring_*` overrides ([Ring sizing](05-runtime-ring-sizing.md)), and a nested `DfxOptions` |
| A dispatch's diagnostics | `DfxOptions` | `enable_chip_swimlane`, `enable_dump_args`, `enable_pmu`, `enable_dep_gen`, `enable_scope_stats` ([DFX](03-runtime-dfx.md)) |
| The system-test harness only | — | `rtol`, `atol`, `golden_data_dir`, `save_kernels`, `codegen_only` |
| Nobody — derived | — | `backend_type`, set from `platform` during construction |

**`platform` names the target once.** `RunConfig` derives `backend_type` from it
during construction, and `ir.compile` lets `platform` win whenever one is given,
so a `backend_type` that disagrees has never taken effect — passing one to
`RunConfig` now warns and is discarded. `CompileOptions` therefore does not carry
it at all: the object always passes a platform, so a second spelling of the same
decision could only ever be redundant or wrong. `ir.compile` keeps its
`backend_type` parameter for the lower-level callers that pass no platform.

`RunConfig.compile_options()` / `run_options()` / `dfx_options()` are views onto
the aggregate, and `compile_kwargs()` is `compile_options().as_compile_kwargs()`.
`CompileOptions` names its fields the way `ir.compile` does — `output_dir`, not
`save_kernels_dir` — because it exists to say the compile side in the compiler's
own vocabulary, and it stands alone:

```python
from pypto.runtime import CompileOptions

compiled = ir.compile(program, **CompileOptions(platform="a2a3").as_compile_kwargs())
```

**Only two of the three are exported**, because only two are things a caller can
hand somewhere. `CompileOptions` unpacks into `ir.compile`, above.
`DfxOptions` is the `dfx=` parameter of `execute_compiled`, `execute_artifact_dir`
and `execute_batch_manifest`. `RunOptions` is neither: every dispatch entry point
— `CompiledProgram.__call__`, `ChipWorker.run`, and the distributed pair — takes
a `RunConfig` and reaches the dispatch half through `run_options()` itself.
Handing one in raises `AttributeError`, so it stays internal
(`pypto.runtime.runner.RunOptions`) until those signatures widen. Widening them
is a migration of its own: it means a union type on the `config=` parameter of
the primary dispatch API, and doing it to some entry points and not others would
be worse than not doing it.

`RunConfig` keeps every field and every caller; the three types are the
vocabulary underneath it. A unit test pins the split as *total* — every
`RunConfig` field is claimed by one of the views or is one of the five
harness-only fields — so a field added later cannot quietly belong to neither.
Moving those five out to the harness is the step this does not take: they reach
`pypto-lib`'s constructor calls too.

`compile_kwargs()` is the only mapping onto `ir.compile`'s parameters. The
`@pl.jit` path used to carry a second copy that omitted `platform` and
`backend_type` and forwarded the platform separately; it now calls
`compile_kwargs()` like every other caller, so a knob added to one is not
silently missing from the other. `lower()` still has its own narrower mapping —
it stops before codegen and targets the pass pipeline, not `ir.compile`.

## Which way the layers point

`ir` produces the artifact; `runtime` dispatches it. The arrow between them
should point one way, and mostly does — but `CompiledProgram` is both the
compilation artifact *and* the execution handle, so `pypto.ir.compiled_program`
reaches forward into `pypto.runtime` through ten function-local imports.

Those deferrals are usually described as breaking an import cycle. Measured, one
at a time, that was true of exactly one of them:

| Deferred import | Hoists cleanly? | Actual reason |
| --------------- | --------------- | ------------- |
| `runtime.runner` (6 sites) | Yes | Layering choice — keeps the dispatch layer out of `pypto.ir`'s import graph |
| `runtime.distributed_runner` (2 sites) | Yes | Same |
| `runtime.debug.run_script_writer` | Yes, *now* | Was a real cycle: it read `ParamInfo` back out of `compiled_program` |
| `runtime.device_runner` | No | Needs the optional `simpler` package at import time |

The one real cycle is gone. The parameter metadata moved to
[`ir/param_info.py`](../../../python/pypto/ir/param_info.py), a leaf that imports
nothing from `pypto.runtime`, and the replay-script writer reads it there;
`compiled_program` re-exports the names, so nothing else moved. A unit test pins
the leaf, because pulling one runtime import into it restores the cycle.

`pypto.runtime` importing `pypto.ir.distributed_compiled_program` directly — the
one place the arrow points back — *is* cycle-avoidance, and stays.

Fully inverting the dependency is a larger question than the import statements.
It means `CompiledProgram` stops being callable and `ir.compile` returns a
descriptor that `runtime` wraps, which changes the return type of the API every
example and both downstream repositories use. The import list is a symptom of
that double role, not the cause, and hoisting the eight that hoist cleanly would
make the coupling *stronger*, not weaker — it would put `pypto.runtime.runner`
on `import pypto.ir`'s critical path.

## What is internal

These have no stability guarantee. They are not in any `__all__`, and code
outside PyPTO should not import them:

- `pypto.ir.compile` — `_ensure_orchestration_headers`
- `pypto.ir.compiled_program` — `CompiledProgram._build_orch_args`,
  `CompiledProgram._build_call_config`
- `pypto.runtime.runner` — `_execute_compiled`, `_execute_golden_case`,
  `_build_call_config`, `_coerced_to_orch_args`, `RunOptions`
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
