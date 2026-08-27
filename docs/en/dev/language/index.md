# Language

The Python DSL as the compiler sees it.

| Page | What it covers |
| ---- | -------------- |
| [Python IR Syntax Specification](00-python_syntax.md) | Module structure, type system, expressions — the entry point to the spec |
| [Statements and Control Flow](01-statements.md) | Assignment, if/for/while, scope context managers, yield, compile-time directives, SSA phi nodes |
| [Manual Dependency Primitives](02-manual_dependencies.md) | `pl.manual_scope`, explicit `deps=` edges, dispatch predicates, array-carry fences |
| [Functions and Program Structure](03-functions.md) | Function types, parameter directions, cross-module reuse, printing IR |
| [Integrating Hand-Written C++ Kernels](04-external-kernels.md) | Calling an existing hand-written C++ InCore kernel from a PyPTO program |
| [GM Cache-Access Policy](05-cache-policy.md) | `pl.set_cache_policy` / `pl.load(cache=...)` — the declared GM cache policy, its coherency contract, and how it reaches codegen |

## See Also

- [IR Parser](../ir/07-parser.md) — how this syntax becomes IR.
- [Language Guide](../../user/language/index.md) — the same language from a user's perspective.
