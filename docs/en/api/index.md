# API Reference

Generated from the source docstrings, so it cannot drift from the code.

| Page | Covers |
| ---- | ------ |
| [`pl`](language.md) | Decorators, types, control flow, and the type-dispatched operator wrappers |
| [`pl.tile`](tile.md) | Tile-level operators — inside an InCore function |
| [`pl.tensor`](tensor.md) | Tensor-level operators — orchestration, whole tensors |
| [`pl.system`](system.md) | Synchronization, cache and cross-core primitives |
| [`pl.array`](array.md) | Fixed-length arrays, mainly for `pl.TASK_ID` fan-in |
| [`pl.prefetch`](prefetch.md) | Asynchronous GM to L2 prefetch |
| [`pl.optimizations`](optimizations.md) | The entries that `pl.at(..., optimizations=[...])` accepts |

## How to use it

**Start from the [catalog](../user/ops/01-catalog.md), not from here.** It groups every operator
by family with a one-line description, and each name links into the page above that carries
its signature. These pages answer "what are the arguments"; the catalog answers "which
operator do I want".

**Names are canonical, not as you spell them.** `pl.create_tensor` is an alias for
`tensor.create`, so it appears here as `create` on the `pl.tensor` page. The catalog links
resolve that for you.

**A missing symbol is a bug in the docstring, not in the page.** These pages render whatever
the source carries; if something reads thin, the fix is in the docstring.

## What the build checks

`mkdocs build --strict` fails on a docstring whose `Args:` names a parameter the signature
does not have, and on a catalog link naming a symbol that is not rendered. Both are real
defects, and both were found the first time these pages were built.

## See Also

- [Catalog](../user/ops/01-catalog.md) — the classified index into these pages.
- [Choosing a namespace](../user/ops/00-dispatch.md) — `pl.` vs `pl.tile.` vs `pl.tensor.`.
- [Language Guide](../user/language/index.md) — the prose behind the types these signatures use.
