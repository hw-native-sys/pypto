# User Manual

How to write, compile, run, and debug PyPTO programs.

## Contents

| Page | What it covers |
| ---- | -------------- |
| [Getting Started](00-getting_started.md) | Installation, a first tensor program, tile kernels, loops, multi-function programs, compiling and running, `DeviceTensor` and explicit dispatch |
| [Language Guide](01-language_guide.md) | The type system, programs and functions, operations, SSA and control flow, memory and data movement, InCore scopes, compilation |
| [Operation Reference](02-operation_reference.md) | The operator surface across the `pl.*`, `pl.tensor.*`, and `pl.tile.*` namespaces |
| [Torch Codegen Debug Guide](03-torch_codegen_debug.md) | Generating a PyTorch reference implementation from the IR to isolate accuracy problems |

## Reading path

1. **[Getting Started](00-getting_started.md)** — get something running end to end
   before reading anything else.
2. **[Language Guide](01-language_guide.md)** — the syntax and semantics behind what
   you just ran.
3. **[Operation Reference](02-operation_reference.md)** — consult as needed while
   writing kernels.
4. **[Torch Codegen Debug Guide](03-torch_codegen_debug.md)** — when the output does
   not match your reference.

## What is not here yet

This manual is being expanded into a full chaptered structure — tutorials,
distributed programming, performance optimization, and accuracy debugging each get
their own chapter. Until those land, the corresponding material lives in the
[developer documentation](../dev/index.md):

| Topic | Current location |
| ----- | ---------------- |
| Tasks, dependencies, `manual_scope` / `submit` | [Python IR Syntax Specification](../dev/language/00-python_syntax.md), [AutoDeriveTaskDependencies](../dev/passes/36-auto_derive_task_dependencies.md) |
| Distributed DSL and collectives | [Distributed Operators](../dev/distributed_ops.md) |
| Performance hints and diagnostics | [Diagnostics](../dev/passes/92-diagnostics.md), [Compile Profiling](../dev/01-compile-profiling.md) |
| Runtime DFX flags, ring sizing, memory map | [Runtime DFX](../dev/03-runtime-dfx.md), [Per-Task Ring Sizing](../dev/05-runtime-ring-sizing.md), [Memory Map](../dev/07-memory-map.md) |
| External C++ kernels | [Integrating Hand-Written C++ Kernels](../dev/language/01-external-kernels.md) |

## See Also

- [Developer documentation](../dev/index.md) — how the compiler lowers what you write.
- [PTO ISA reference](../reference/index.md) — the instruction semantics behind the generated code.
- [Runtime documentation](https://hw-native-sys.github.io/simpler/) — the scheduler that executes compiled programs.
