# Execution

Turning a program into artifacts, and running those artifacts on a device.

Everything up to this chapter describes what you write. This one describes what happens to
it: `ir.compile` and `JITFunction.compile` produce a `CompiledProgram`, and `ChipWorker`
dispatches it. Two knobs decide most of what you will care about — where the artifacts go,
and which of them the runtime is allowed to reuse between launches.

| Page | Covers |
| ---- | ------ |
| [Compiling](00-compile.md) | `ir.compile` and its parameters, `JITFunction.compile`, the artifact directory, pass dumps |
| [Running](01-run.md) | `CompiledProgram`'s contract, `ChipWorker`, `DeviceTensor`, and the `RunConfig` fields that affect dispatch |

## Where PyPTO stops

PyPTO produces the artifacts and hands them to the **simpler** runtime, which owns
scheduling, the task rings, and the device lifecycle. This chapter documents the PyPTO
side of that boundary only: the API you call, what it produces, and which of its fields
the runtime reads.

For the mechanisms behind the boundary — how tasks are scheduled, how dependencies are
resolved, what the rings do — see the [runtime documentation](https://hw-native-sys.github.io/simpler/).
[Memory](../performance/05-memory.md) covers the part of that machinery you can tune from
here.

## See Also

- [Getting started](../00-getting_started.md) — the shortest path from a kernel to a result.
- [Tools](../tools/index.md) — what to reach for when the result is not what you expected.
- [Performance](../performance/index.md) — measuring and tuning what this chapter launches.
