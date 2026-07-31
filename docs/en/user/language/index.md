# Language Guide

The `pypto.language` surface, one topic per page: what you can write, what each
construct means, and where it fails.

> **Prerequisites:** [Quickstart](../02-quickstart.md) and
> [Programming Model](../03-programming-model.md). This chapter assumes you have compiled
> something and know the difference between the control plane and the execution plane.

## What this chapter is

A **guide**, organized by capability — each page covers one part of the language in full,
including the edge cases. It is not a tutorial: no page builds a complete kernel from
start to finish. It is also not an API reference: signatures live in
[Operations](../ops/index.md) and in the docstrings.

Conventionally, `import pypto.language as pl` — every name on these pages is reached
through that alias.

## Contents

| Page | What it covers |
| ---- | -------------- |
| [Types](00-types.md) | dtypes, `Tensor` / `Tile` / `Scalar` / `Array` / `Tuple`, layouts, dynamic shapes, parameter directions |
| [Functions and Programs](01-functions.md) | `@pl.jit` family, `@pl.function`, `@pl.program`, `@pl.inline`, cross-function calls, external kernels |
| [Control Flow](02-control-flow.md) | `pl.range` / `parallel` / `unroll` / `pipeline` / `while_`, carried values, `yield_`, `cond`, SSA |
| [Memory and Data Movement](03-memory.md) | the memory spaces, `load` / `store` / `move`, `valid_shape` and `fillpad`, L1 residency |
| [Scopes and Tasks](04-scopes-and-tasks.md) | `at` / `cluster` / `spmd` / `split_aiv`, runtime scopes, `submit` and `deps=`, `predicate=`, `no_dep` |
| [Directives](05-directives.md) | `static_print` / `static_assert`, `dump_tag` / `dumps=`, `const`, `pl.array`, subscript sugar, closure capture |

## Reading order

Read [Types](00-types.md) and [Functions and Programs](01-functions.md) first — every
other page assumes both. After that the pages are independent:

```text
00-types ──► 01-functions ──┬─► 02-control-flow
                            ├─► 03-memory
                            ├─► 04-scopes-and-tasks   ← the widest gap if you are
                            └─► 05-directives           coming from single-kernel code
```

[Scopes and Tasks](04-scopes-and-tasks.md) is the page most readers have not seen
equivalent material for elsewhere: it covers how work is placed on cores and how the
task graph the runtime executes is shaped.

## See Also

- [Operations](../ops/index.md) — which namespace an operator lives in, and the full catalog.
- [Programming Model](../03-programming-model.md) — the abstractions this chapter is the surface of.
- [Python IR Syntax Specification](../../dev/language/00-python_syntax.md) — the parser's own reference, including forms this guide does not recommend.
- [Passes](../../dev/passes/index.md) — what the compiler does with each construct.
