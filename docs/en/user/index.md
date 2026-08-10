# User Manual

How to write, compile, run, and debug PyPTO programs.

## Reading paths

Pick the path that matches what you are trying to do. All four assume
[Installation](01-installation.md) is done.

### I want to write my first kernel

[Quickstart](02-quickstart.md) → [Programming Model](03-programming-model.md) →
[Language Guide](language/index.md)

Start by getting something to compile, then learn what it was doing, then fill in the
rest of the surface. Keep the [operator catalog](ops/01-catalog.md) open alongside — you
will be looking up operators constantly at first.

### I have a kernel and the numbers are wrong

[Torch Codegen Debug Guide](03-torch_codegen_debug.md) →
[Programming Model § execution model](03-programming-model.md#the-execution-model)

Lower the IR to a PyTorch script and compare tensor by tensor. If the result changes
between runs rather than being consistently wrong, the problem is ordering, not
arithmetic — read the execution-model section, because statement order does not
constrain execution order.

### I have a kernel and it is slow

[Programming Model § memory hierarchy](03-programming-model.md#memory-hierarchy) →
[Diagnostics](../dev/passes/92-diagnostics.md) →
[Runtime DFX](../dev/03-runtime-dfx.md)

Check `report/perf_hints.log` from your compile output before measuring anything — the
compiler may already have told you. The dedicated performance chapter is not written yet;
see the table below for where its material currently lives.

### I want to run across multiple devices

[Distributed Programming](distributed/index.md)

Get a single-device kernel running first — distributed programs compose the same
`pl.*` kernels behind `pld.*` collectives and a HOST orchestrator. Once it runs
correctly, the distributed chapter covers ring vs. mesh trade-offs and
cross-rank overlap.

## Contents

| Page | What it covers |
| ---- | -------------- |
| [Installation](01-installation.md) | Prerequisites, install from source, build options, verification, a tour of `examples/` |
| [Quickstart](02-quickstart.md) | Tensor-level kernels with `@pl.jit` — no manual data movement — plus loops, splitting work across functions, compiling and reading the IR |
| [Programming Model](03-programming-model.md) | Tensor / Tile / Block levels, control vs. execution plane, the pass pipeline, memory hierarchy, execution model |
| [Language Guide](language/index.md) | The full language, one topic per page: types, functions, control flow, memory, scopes and tasks, directives |
| [Operations](ops/index.md) | Choosing between the `pl.*`, `pl.tensor.*`, and `pl.tile.*` namespaces, plus the operator catalog |
| [Compiling a Program](01-language_guide.md) | `ir.compile()` and `JITFunction.compile()`, and inspecting the result |
| [Running on Device](00-getting_started.md) | Resident device tensors, explicit dispatch, benchmarking, distributed execution |
| [Torch Codegen Debug Guide](03-torch_codegen_debug.md) | Generating a PyTorch reference implementation from the IR to isolate accuracy problems |
| [Distributed Programming](distributed/index.md) | Symmetric-memory model, collectives, primitives, execution, and debugging for cross-rank programs |

## What PyPTO gives you

| Capability | Where it is documented |
| ---------- | ---------------------- |
| Kernel authoring with `@pl.jit` (and the `@pl.function` / `@pl.program` form it specializes into) | [Quickstart](02-quickstart.md), [Functions and Programs](language/01-functions.md) |
| Explicit on-chip memory placement (Vec / Mat / L0A / L0B / L0C) | [Programming Model](03-programming-model.md#memory-hierarchy) |
| Control flow: loops, carried values, conditionals, while | [Control Flow](language/02-control-flow.md) |
| Multi-function programs and cross-function calls | [Quickstart](02-quickstart.md) |
| The full `@pl.jit` family (`.incore`, `.inline`, `.opaque`, `.host`) | [Quickstart](02-quickstart.md), [Functions and Programs](language/01-functions.md) |
| Hand-written C++ kernel integration | [External Kernels](../dev/language/01-external-kernels.md) |
| Device-resident tensors, explicit dispatch, benchmarking | [Running on Device](00-getting_started.md) |
| Distributed (multi-card) programs and collectives | [Distributed Programming](distributed/index.md) |
| Accuracy debugging against a PyTorch reference | [Torch Codegen Debug Guide](03-torch_codegen_debug.md) |
| Compile-time diagnostics and performance hints | [Diagnostics](../dev/passes/92-diagnostics.md) |
| Runtime DFX: swimlane, PMU, dependency graph, scope stats | [Runtime DFX](../dev/03-runtime-dfx.md) |
| On-chip memory map visualization | [Memory Map](../dev/07-memory-map.md) |

## What is not here yet

This manual is being expanded into a full chaptered structure — tutorials,
performance optimization, and accuracy debugging each get their own chapter.
Until those land, the
corresponding material lives in the [developer documentation](../dev/index.md):

| Topic | Current location |
| ----- | ---------------- |
| Mixed kernels (AIC + AIV in one function) | [LowerAutoVectorSplit](../dev/passes/21-lower_auto_vector_split.md), [ExpandMixedKernel](../dev/passes/22-expand_mixed_kernel.md), [TPUSH/TPOP](../reference/pto-isa/01-tpush_tpop.md) |
| Performance hints and diagnostics | [Diagnostics](../dev/passes/92-diagnostics.md), [Compile Profiling](../dev/01-compile-profiling.md) |
| Runtime DFX flags, ring sizing, memory map | [Runtime DFX](../dev/03-runtime-dfx.md), [Per-Task Ring Sizing](../dev/05-runtime-ring-sizing.md), [Memory Map](../dev/07-memory-map.md) |
| External C++ kernels | [Integrating Hand-Written C++ Kernels](../dev/language/01-external-kernels.md) |

## See Also

- [Developer documentation](../dev/index.md) — how the compiler lowers what you write.
- [PTO ISA reference](../reference/index.md) — the instruction semantics behind the generated code.
- [Runtime documentation](https://hw-native-sys.github.io/simpler/) — the scheduler that executes compiled programs.
