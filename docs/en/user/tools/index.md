# Tools

What to reach for when the result is not what you expected.

Four questions, four tools. The order matters — each one is cheaper than the next, and the
compiler has usually answered the first before you ask it.

| Question | Tool | Page |
| -------- | ---- | ---- |
| What went wrong, and where? | Error types, log levels, IR dumps | [Debugging](00-debugging.md) |
| Is the **IR** wrong, or the device? | `pypto.debug.torch_codegen` | [Torch codegen](01-torch-codegen.md) |
| What is on chip, and for how long? | `pypto.tools.memory_map` | [Memory map](02-memory-map.md) |
| Where did the time go? | The L2 swimlane | [Performance](../performance/00-swimlane.md) |

## The cheap checks first

Two artifacts are written on every compile and cost nothing to read:

- **`report/perf_hints.log`** — what the compiler noticed but did not refuse: transfers
  below the hardware granularity, a matmul it could not tile, a pipeline depth that did not
  fit. One summary line also goes to stderr.
- **The error message itself.** PyPTO distinguishes a user error from an internal one, and
  the distinction tells you whether to fix your code or file a bug — see
  [Debugging](00-debugging.md).

## See Also

- [Precision](../precision/index.md) — the workflow these tools serve when numbers are wrong.
- [Performance](../performance/index.md) — the same, when the numbers are right but slow.
- [Execution](../execution/index.md) — the compile and dispatch surface being debugged.
