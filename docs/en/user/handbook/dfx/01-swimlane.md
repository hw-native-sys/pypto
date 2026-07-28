# DFX Features: Swimlane

> **Status:** skeleton — outline only; prose/examples TBD.
> Per-task timeline captured on hardware. Focus of this doc: **how to read it**.

## Enable & outputs (brief)

- Enable: `enable_l2_swimlane` (`RunConfig`) / `--enable-l2-swimlane` (pytest).
- Onboard runs the kernel twice (untimed graph + timed); `*sim` single-pass.
- `l2_swimlane_records.json` → `swimlane_converter` → `merged_swimlane_*.json`
  (`--func-names` for readable task names).

## How to read the swimlane (focus)

- What a lane is (core) and what a block is (task). A mixed kernel is **one**
  task — its internal cube/vector overlap is *not* here (use in-core profiling).
- What to look for: gaps (stalls), serialized chains, imbalance across cores.
- Where to act: serialized tasks → [Dependency & Dispatch](../perf/03-dependency-dispatch.md);
  across-core imbalance → SPMD in [Split & Parallel](../perf/02-split-parallel.md).
- TBD: viewer / how to open `merged_swimlane_*.json`.

## Task field reference (focus)

- TBD: enumerate every field shown on a task block and its meaning
  (name / id, start / end / duration, engine / pipe, dependencies, …).

## L0–L4 swimlane capability (focus)

- TBD: what each level (L0 / L1 / L2 / L3 / L4) captures and its granularity —
  what you can and cannot see at each.

## See Also

- [Flag Matrix](00-flag-matrix.md)
- [Perf › Measuring Impact](../perf/05-measuring-impact.md)
- Developer reference: [`dev/03-runtime-dfx.md`](../../../dev/03-runtime-dfx.md)
