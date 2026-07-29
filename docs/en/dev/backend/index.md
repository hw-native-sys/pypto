# Backend

Per-architecture behaviour, kept out of the passes.

Passes never branch on `BackendType`. Everything architecture-specific — codegen
target, runtime API names, hazard workarounds, cross-core layout rules — is answered
by a `BackendHandler` obtained from the active `PassContext`. Adding an architecture
means adding a handler, not editing passes.

| Page | What it covers |
| ---- | -------------- |
| [BackendHandler: principled backend dispatch](00-backend_handler.md) | The virtual interface, how passes query it, and what adding a new backend requires |

## See Also

- [Pass, PassContext, PassPipeline, and PassManager](../passes/00-pass_manager.md) — where the handler comes from.
- [PTO ISA reference](../../reference/index.md) — the hardware differences the handlers abstract over.
