# Language

The Python DSL as the compiler sees it.

| Page | What it covers |
| ---- | -------------- |
| [Python IR Syntax Specification](00-python_syntax.md) | Module structure, type system, expressions — the entry point to the spec |
| [Statements and Control Flow](01-statements.md) | Assignment, if/for/while, scope context managers, yield, compile-time directives, SSA phi nodes |
| [Manual Dependency Primitives](02-manual_dependencies.md) | `pl.manual_scope`, explicit `deps=` edges, dispatch predicates, array-carry fences |
| [Functions and Program Structure](03-functions.md) | Function types, parameter directions, cross-module reuse, printing IR |
| [Integrating Hand-Written C++ Kernels](04-external-kernels.md) | Calling an existing hand-written C++ InCore kernel from a PyPTO program |
| [GM Cache-Access Policy and L2 Hints](05-cache-policy.md) | Scope declarations and per-transfer `cache=` controls: unified `CachePolicy` for A2/A3 runtime aliases and A5 modes |

## See Also

- [IR Parser](../ir/07-parser.md) — how this syntax becomes IR.
- [Language Guide](../../user/language/index.md) — the same language from a user's perspective.

- [Register operations](06-register-operations.md): P0 schemas, implicit captures,
  region ownership and Builder construction (VF backend emission requires PR3).
