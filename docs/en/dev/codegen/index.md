# Code Generation

Lowering PyPTO IR to the two artifacts a compiled program consists of: device
kernels and the host-side orchestration that launches them.

Both generators follow the same design principle — a **strict 1-to-1 mapping** from
IR to generated code. Anything that needs a decision was already decided by a pass.

| Page | What it covers |
| ---- | -------------- |
| [PTO Codegen](00-pto_codegen.md) | Generating PTO-ISA dialect MLIR from PyPTO IR |
| [Orchestration Code Generation](01-orchestration_codegen.md) | Generating the host-side C++ that submits tasks to the runtime |

## See Also

- [Passes](../passes/index.md) — everything that must hold before codegen runs.
- [PTO ISA reference](../../reference/index.md) — the instruction semantics the generated code uses.
- [PTOAS Op Status Matrix](../ptoas-op-status.md) — which PTOAS ops are available to emit.
