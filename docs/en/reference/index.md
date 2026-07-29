# Reference

Hardware and instruction-set material behind the code PyPTO generates.

Read this when you are looking at generated PTO code, tuning a cross-core pipeline,
or reasoning about why a pass lowers something the way it does. For the language you
write, see the [User Manual](../user/index.md); for how the compiler transforms it,
see the [developer documentation](../dev/index.md).

## PTO ISA

| Page | What it covers |
| ---- | -------------- |
| [Cluster Architecture](pto-isa/00-cluster_architecture.md) | The 1 Cube + 2 buddy Vector core cluster and its flag-based synchronization |
| [TPUSH/TPOP Instructions](pto-isa/01-tpush_tpop.md) | Moving tiles between InCore kernels co-scheduled on Cube and Vector cores |
| [Buffer Management](pto-isa/02-buffer_management.md) | Where the TPUSH/TPOP ring buffer lives per platform — GM on A2/A3, consumer on-chip memory on A5 |

## See Also

- [PTO Project Ecosystem](../dev/00-ecosystem.md) — how PyPTO, PTOAS, pto-isa, and the runtime fit together.
- [PTO Codegen](../dev/codegen/00-pto_codegen.md) — how PyPTO IR becomes PTO-ISA dialect MLIR.
- [PTOAS Op Status Matrix](../dev/ptoas-op-status.md) — which PTOAS ops the compiler currently emits.
