# DFX Features: Diagnostics Rendering

> **Status:** DRAFT skeleton. Turn raw diagnostic artifacts into browsable HTML.

## Dependency Graph (`deps.json` → HTML)

_TODO — how to render the task dependency graph and read it (nodes = tasks,
edges = deps, how to spot serialization)._

## Scope Statistics (`scope_stats.jsonl` → HTML)

_TODO — per-scope timing/occupancy view; how to read it._

## See Also

- [Flag Matrix](00-flag-matrix.md) — which flags emit these files
- [Perf › Dependency & Dispatch](../perf/03-dependency-dispatch.md) — act on what you find
- Developer reference: [`dev/03-runtime-dfx.md`](../../../dev/03-runtime-dfx.md)
