# `pl.*` API Reference

> **Status:** DRAFT skeleton / placeholder. This reference will be
> **auto-generated** from the frontend docstrings (Issue #2120), covering every
> symbol in `pypto.language.__all__` (223 symbols). Do not hand-write entries
> here — they will be produced by the docs generator.

## Generation Plan

_TODO — decide and document:_

- **Generator:** Sphinx (`autodoc` + `autosummary` + `napoleon`) **or**
  MkDocs-Material (`mkdocstrings[python]`). Both consume Google-style docstrings
  (mandated by `.claude/rules/python-style.md`).
- **CI gate:** build with `--fail-on-warning` / a coverage check so any new or
  renamed public symbol without a proper docstring fails the build — keeping
  docs and code in lockstep.
- **Translation policy:** decide EN-authoritative + zh-cn mirror vs EN-only, and
  record the choice (per `.claude/rules/documentation.md`).

## Category Grouping (as generated)

The generated reference groups `pl.__all__` by category:

- Types & dtypes
- Unified ops
- Tile ops
- Control flow
- `@pl.function` / `@pl.program` / `@pl.jit` / `@pl.inline`
- Memory & data movement
- On-chip parallelism (`spmd`, `cluster`, `at`, `spmd_submit`)
- Task / manual-scope (`manual_scope`, `scope`, `submit`, `no_dep`, `TaskId`, …)
- Cross-core split (`split_aiv`, `split`, `aiv_shard`, `aic_gather`, `tpush*`,
  `tpop*`, `tfree*`, `*_initialize_pipe`)
- Peer buffers (`reserve_buffer`, `import_peer_buffer`)
- Distributed (`pld.*`)

## Prose vs Reference

The hand-written prose guides ([Language Guide](../02-language_guide.md),
[Operation Reference](../03-operation_reference.md)) stay as conceptual/tutorial
material and cross-link here. The manually-maintained op tables that drift will
be removed once this reference is live.
