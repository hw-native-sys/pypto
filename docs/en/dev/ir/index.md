# IR

PyPTO's intermediate representation — the contract every other layer depends on.

The IR definition is the source of truth for the whole compiler: passes are
replaceable, IR node definitions are not. Read [Overview](00-overview.md) first, then
the page covering what you are changing.

| Page | What it covers |
| ---- | -------------- |
| [Overview](00-overview.md) | The compilation pipeline from Python DSL to generated kernels, and where the IR sits in it |
| [Node Hierarchy](01-hierarchy.md) | Complete reference of every IR node type, organized by category |
| [Types and Examples](02-types.md) | The type system with practical usage examples |
| [Structural Comparison](03-structural_comparison.md) | Comparing IR nodes by structure rather than pointer identity |
| [Serialization](04-serialization.md) | MessagePack-based `.pto` serialization |
| [Operator System](05-operators.md) | Type-safe operator definitions with automatic type deduction |
| [IR Builder](06-builder.md) | Constructing IR incrementally — context managers in Python, Begin/End in C++ |
| [IR Parser](07-parser.md) | Converting Python DSL to IR via `@pl.function` / `@pl.program`, and the SSA properties it enforces |
| [Parameter Directions](08-param-directions.md) | How `In`/`Out`/`InOut` is inferred — the registry declaration every stage reads, and the four passes that build on it |

## See Also

- [Passes](../passes/index.md) — the transformations that run over this IR.
- [Python IR Syntax Specification](../language/00-python_syntax.md) — the surface syntax the parser accepts.
- [IR Verifier](../passes/99-verifier.md) — the property verifiers that check IR legality between passes.
