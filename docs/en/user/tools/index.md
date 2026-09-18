# DFX Tools

Everything PyPTO can tell you about a program it compiled or ran — and which question each
answer belongs to.

## Concept

DFX is the observability surface: what the compiler noticed, what it decided, and what the
hardware actually did. It splits along one axis that decides which half of the chapter you
are in:

| Half | Runs | Costs | Answers |
| ---- | ---- | ----- | ------- |
| **Compile-time** | Without a device | Seconds | What the compiler *decided* |
| **Run-time** | On a device or the simulator | A run, sometimes two | What the hardware *did* |

Almost every question is cheaper on the compile-time side, and the compiler has usually
answered it before you ask. Start there.

## The cheap checks first

Two things cost nothing to read, and every compile has already produced both:

- **`report/perf_hints.log`** — what the compiler noticed but did not refuse: transfers
  below the hardware granularity, a matmul it could not tile, a pipeline depth that did not
  fit. One summary line also goes to stderr.
- **The error message**, when there is one. PyPTO distinguishes a user error from an
  internal one, and the distinction tells you whether to fix your code or file a bug — see
  [Debugging](00-debugging.md).

## Contents

| Page | Covers | Side |
| ---- | ------ | ---- |
| [Debugging](00-debugging.md) | Error types, log levels, pass dumps, reading the lowered IR | Compile |
| [Torch codegen](01-torch-codegen.md) | Running the IR's semantics on the host, to separate an IR bug from a device bug | Compile |
| [Memory map](02-memory-map.md) | What is on chip, where, and for how long | Compile |
| [IR trace](03-ir-trace.md) | Which pass changed what, as a navigable diff over the whole pipeline | Compile |
| [Runtime DFX](04-runtime-dfx.md) | The five collection flags — swimlane, args dump, PMU, dependency graph, scope stats | Run |
| [Replaying a build](05-replay.md) | Re-running, hand-editing, and re-measuring an existing output directory | Run |
| [In-core trace](06-incore-trace.md) | Cycle-accurate per-pipe instruction traces for one kernel | Run |

## Which one

Four questions cover most of it, and they are ordered by cost:

| Question | Tool | Page |
| -------- | ---- | ---- |
| What went wrong, and where? | Error types, log levels, IR dumps | [Debugging](00-debugging.md) |
| Which pass changed my IR? | `pypto-ir-trace` over `passes_dump/` | [IR trace](03-ir-trace.md) |
| Is the **IR** wrong, or the device? | `pypto.debug.torch_codegen` | [Torch codegen](01-torch-codegen.md) |
| What is on chip, and for how long? | `pypto.tools.memory_map` | [Memory map](02-memory-map.md) |
| Where did the time go? | The chip swimlane | [Runtime DFX](04-runtime-dfx.md) |
| Which pipe is the kernel spending on? | The in-core instruction trace | [In-core trace](06-incore-trace.md) |

## Where the artifacts live

Compilation writes under the output directory; the runtime writes under `dfx_outputs/`
inside it. Nothing here is produced unless you ask for it, except `report/`.

```text
build_output/<program>_<timestamp>_<random>/
├── passes_dump/            # dump_passes=          -> memory map, IR trace
├── ptoas/                  # the .pto per InCore function, and ptoas's .cpp
├── ptoas_passes/           # dump_ptoas_passes=
├── kernels/                # the compiled device kernels
├── report/                 # always written
│   ├── perf_hints.log
│   └── pipeline_profile.*  # profiling=
├── debug/run.py            # the auto-emitted re-runner
└── dfx_outputs/            # written at RUN time, by the flags in 04-runtime-dfx
```

`compile()` prints none of this, so ask it where it went:

```python
compiled = kernel.compile(*args, config=RunConfig(dump_passes=PassDumpLevel.EXPLICIT))
print(compiled.output_dir)
```

> **`lower()` writes nothing.** It runs the passes and hands back the `Program`, which is
> what [torch codegen](01-torch-codegen.md) wants — but the three tools that read
> `passes_dump/` need `compile()`.

## See Also

- [Precision](../precision/index.md) — the workflow these tools serve when numbers are wrong.
- [Performance](../performance/index.md) — the same, when the numbers are right but slow.
- [Execution](../execution/index.md) — the compile and dispatch surface being observed.
- [Runtime DFX Flags](../../dev/03-runtime-dfx.md) — the `CallConfig` contract behind the run-time half.
